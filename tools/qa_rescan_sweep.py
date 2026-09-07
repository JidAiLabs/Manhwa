#!/usr/bin/env python3
"""
qa_rescan_sweep.py — give every chapter parked by the OLD autopilot policy the
one event that un-parks it: a qa_scan.

    python tools/qa_rescan_sweep.py                      # dry-run: classify + print
    python tools/qa_rescan_sweep.py --apply --limit 1    # smoke one
    python tools/qa_rescan_sweep.py --apply              # the rest

Run it on the host that holds studio.db + ongoing/ (the Mini), AFTER the
worker that carries the 2026-09-07 policy is running (launchctl kickstart).

Why it exists. Until 2026-09-07 studio/worker._autopilot_clean parked a chapter
on ANY ERROR code (visible_text alone held 112 of 152 prepared chapters behind
a green badge). The policy now keys on `blocking`, like every other gate — but
nothing re-evaluates a parked chapter passively: only the end of a full
prepare and a qa_scan call _advance_after_prepare. So the policy change moves
nothing already parked. This enqueues a qa_scan per candidate (cheap: no
render, no model beyond the grounding judge's content-addressed cache); the
worker's own gate then decides, exactly as it would at the end of a prepare.

Candidates: series.autopilot = 1; chapter.status in scripted / planned /
voiced; newest qa_scan stage_run ok = 1; no live job on the chapter. Each is
classified the way the worker will read it: ERROR codes on disk ∩
_effective_blocking() − _qa_arbitrated(ep) empty -> WILL ADVANCE, otherwise
WILL BLOCK (skipped unless --include-blocked — those need a human, not a
re-scan). jobs.enqueue dedupes on (type, chapter_id), so --apply is idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PARKED_STATUSES = ("scripted", "planned", "voiced")


def classify(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    from studio import worker
    from studio.dashboard import gates
    rows = con.execute(
        "SELECT c.id, c.label, c.status, c.ep_dir, s.slug FROM chapter c "
        "JOIN series s ON s.id = c.series_id WHERE s.autopilot = 1 AND "
        f"c.status IN ({','.join('?' * len(PARKED_STATUSES))}) "
        "ORDER BY s.id, c.number", PARKED_STATUSES).fetchall()
    out: List[Dict[str, Any]] = []
    for cid, label, status, ep_dir, slug in rows:
        if not gates.latest_qa_ok(con, cid):
            continue                                  # never scanned green
        live = con.execute(
            "SELECT 1 FROM job WHERE chapter_id=? AND state IN "
            "('queued','running','cancelling') LIMIT 1", (cid,)).fetchone()
        if live:
            continue                                  # already moving
        ep = Path(ep_dir or "")
        codes = worker._qa_error_codes(ep)
        blocking = (codes & worker._effective_blocking()) - worker._qa_arbitrated(ep)
        judge = False
        try:
            rep = json.loads((ep / "prep_qa.json").read_text())
            judge = any(f.get("code") == "narration_mismatch"
                        for f in (rep.get("flags") or []) if isinstance(f, dict))
        except Exception:
            pass
        out.append({"id": cid, "label": label, "status": status, "series": slug,
                    "errors": sorted(codes), "blocking": sorted(blocking),
                    "judge_warn": judge,
                    "verdict": "WILL BLOCK" if blocking else "WILL ADVANCE"})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=os.path.join(REPO, "studio.db"))
    ap.add_argument("--apply", action="store_true",
                    help="enqueue a qa_scan per WILL ADVANCE chapter (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="enqueue at most N")
    ap.add_argument("--include-blocked", action="store_true",
                    help="enqueue the WILL BLOCK ones too (they will fail loudly)")
    ap.add_argument("--priority", type=int, default=1,
                    help="job priority (1 = ahead of the ordinary prepares)")
    args = ap.parse_args(argv)

    from studio.catalog.db import connect
    from studio.dashboard import jobs
    con = connect(args.db)
    rows = classify(con)
    if not rows:
        print("no parked autopilot chapters with a green qa_scan and no live job")
        return 0
    print(f"{'id':>5} {'series':22s} {'chapter':14s} {'status':9s} "
          f"{'verdict':13s} judge  errors")
    for r in rows:
        print(f"{r['id']:5d} {r['series'][:22]:22s} {r['label'][:14]:14s} "
              f"{r['status']:9s} {r['verdict']:13s} "
              f"{'WARN' if r['judge_warn'] else '-':5s}  "
              f"{','.join(r['errors']) or '-'}")
    adv = [r for r in rows if r["verdict"] == "WILL ADVANCE"]
    blk = [r for r in rows if r["verdict"] == "WILL BLOCK"]
    print(f"\n{len(adv)} will advance, {len(blk)} will block, "
          f"{sum(r['judge_warn'] for r in rows)} carry a narration_mismatch WARN")
    if not args.apply:
        print("dry-run — pass --apply to enqueue a qa_scan per WILL ADVANCE chapter")
        return 0
    todo = adv + (blk if args.include_blocked else [])
    if args.limit:
        todo = todo[:args.limit]
    n = 0
    for r in todo:
        jid = jobs.enqueue(con, "qa_scan", chapter_id=r["id"],
                           priority=args.priority)
        n += 1
        print(f"enqueued qa_scan job {jid} for chapter {r['id']} ({r['label']})")
    print(f"{n} qa_scan job(s) enqueued (dedupe: re-running adds none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
