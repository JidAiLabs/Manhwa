#!/usr/bin/env python3
"""qa_rules_sweep.py — which FINISHED chapters break a rule of docs/qa/RULES.md,
and (with --apply) queue the cheapest repair for each.

    python tools/qa_rules_sweep.py                                # every rendered chapter
    python tools/qa_rules_sweep.py --series boundless-necromancer
    python tools/qa_rules_sweep.py --apply --chapter-ids 904,912  # repair these
    python tools/qa_rules_sweep.py --apply                        # repair every flagged one
    python tools/qa_rules_sweep.py --apply --rows                 # + re-render without rows

Reads only what SHIPPED — render.plan.clean.json (what is shown and spoken), the
vision items of the shown cuts, manifest.cast.json, prep_qa.json — and judges it
with the SAME functions the pipeline uses (scene_chrome, prep_qa, render_prep,
the worker's blocking set): never a second implementation of a rule. Run it
before and after every deploy that touches narration or QA (RULES.md, "the
contract"); the after-run on the chapters made since the deploy must be clean.

Repairs, heaviest first (one per chapter; the heaviest needed wins):
  redo       rewind to 'detected' + prepare {auto_to: video, narrate: true}: page
             text, labels, cast and narration are all rebuilt (the licensed-title
             backfill recipe, verified 2026-10-05). Rewinds delete the chapter's
             video link and any bundle containing it (studio.catalog.reset).
  reprepare  prepare {auto_to: video}: the heal re-narrates only flagged groups.
  rerender   voiceover {auto_to: video}: clips reused, plan + rows + QA + render
             rebuilt. Only with --rows (the owner decides the rows backfill).
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import scene_chrome as sc  # noqa: E402


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


class Ctx:
    """One finished chapter, loaded once for every check."""

    def __init__(self, ep: Path, title: str):
        self.ep, self.title = ep, title
        self.plan = _json(ep / "render.plan.clean.json") or {}
        self.timeline = self.plan.get("timeline") or []
        self.vitems = {str(it.get("scene_file") or ""): it for it in
                       ((_json(ep / "manifest.vision.json") or {}).get("items") or [])}
        self.cast = _json(ep / "manifest.cast.json") or {}

    def shown(self) -> List[Dict[str, Any]]:
        """(segment_id, vision item) for every distinct panel on screen."""
        import prep_qa
        seen, out = set(), []
        for c in prep_qa.iter_shown_cuts(self.plan):
            f = prep_qa.parent_scene(c["file"])
            if f in seen or c.get("branding"):
                continue
            seen.add(f)
            it = self.vitems.get(f)
            if it:
                out.append({"segment_id": c["segment_id"], "file": f, "item": it})
        return out


def check_licensed_title_shown(ctx: Ctx) -> List[str]:
    return [f"{s['segment_id']} {s['file']}" for s in ctx.shown()
            if ctx.title and sc.is_licensed_title_scene(s["item"], ctx.title)]


def check_publication_shown(ctx: Ctx) -> List[str]:
    out = []
    for s in ctx.shown():
        why = sc.furniture_reason(s["item"], midtone_frac=sc.item_midtone(s["item"]))
        if why:
            out.append(f"{s['segment_id']} {s['file']} ({why})")
    return out


def check_publication_voiced(ctx: Ctx) -> List[str]:
    out = []
    for it in ctx.timeline:
        why = sc.voiced_furniture(str(it.get("tts_text") or ""))
        if why:
            out.append(f"{it.get('segment_id')} ({why}): {str(it.get('tts_text'))[:90]}")
    return out


def check_watermark_name(ctx: Ctx) -> List[str]:
    import prep_qa
    return [f["detail"][:120] for f in
            prep_qa.watermark_name_flags(ctx.plan, ctx.cast, ctx.vitems)]


def check_rows_missing(ctx: Ctx) -> List[str]:
    """No side-by-side row in the video although render_prep would make some."""
    if not ctx.timeline or any(it.get("row") for it in ctx.timeline):
        return []
    import render_prep
    from studio.config import load
    cfg = load()
    if not cfg.row_min_fit:
        return []
    new, logs = render_prep.assign_rows(copy.deepcopy(ctx.plan), min_fit=cfg.row_min_fit,
                                        min_gap_items=cfg.row_min_gap_items)
    n = sum(1 for it in new.get("timeline") or [] if it.get("row"))
    return [f"{n} row(s) would be made"] if n else []


def check_blocking_now(ctx: Ctx) -> List[str]:
    """ERROR codes in the shipped QA report that block today (a code promoted to
    blocking after the chapter shipped)."""
    from studio import worker
    codes = (worker._qa_error_codes(ctx.ep) & worker._effective_blocking()
             - worker._qa_arbitrated(ctx.ep))
    return sorted(codes)


# check name -> (function, repair). RULES.md names these in its "Sweep" column.
CHECKS: Dict[str, tuple] = {
    "licensed_title_shown": (check_licensed_title_shown, "redo"),
    "publication_shown": (check_publication_shown, "redo"),
    "publication_voiced": (check_publication_voiced, "redo"),
    "watermark_name": (check_watermark_name, "redo"),
    "blocking_now": (check_blocking_now, "reprepare"),
    "rows_missing": (check_rows_missing, "rerender"),
}
_WEIGHT = {"redo": 3, "reprepare": 2, "rerender": 1}


def sweep(con: sqlite3.Connection, *, series: Optional[str] = None,
          chapter_ids: Optional[List[int]] = None,
          checks: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    q = ("SELECT c.id, s.slug, s.title, c.number, c.label, c.ep_dir FROM chapter c "
         "JOIN series s ON s.id = c.series_id WHERE c.status = 'rendered'")
    args: List[Any] = []
    if series:
        q += " AND s.slug = ?"
        args.append(series)
    if chapter_ids:
        q += f" AND c.id IN ({','.join('?' * len(chapter_ids))})"
        args += chapter_ids
    out = []
    for cid, slug, title, num, label, ep_dir in con.execute(q + " ORDER BY s.id, c.number", args):
        ep = Path(ep_dir or "")
        if not (ep / "render.plan.clean.json").exists():
            continue
        ctx = Ctx(ep, title or slug.replace("-", " "))
        found = {}
        for name, (fn, _) in CHECKS.items():
            if checks and name not in checks:
                continue
            hits = fn(ctx)
            if hits:
                found[name] = hits
        repair = max((CHECKS[n][1] for n in found), key=_WEIGHT.get, default=None)
        out.append({"chapter_id": cid, "series": slug, "number": num, "label": label,
                    "findings": found, "repair": repair})
    return out


def apply(con: sqlite3.Connection, rows: List[Dict[str, Any]], *, rows_backfill: bool,
          priority: int, log: Callable[[str], None] = print) -> Dict[str, int]:
    from studio.catalog import reset
    from studio.dashboard import gates, jobs
    done = {"redo": 0, "reprepare": 0, "rerender": 0, "skipped_live_job": 0}
    for r in rows:
        rep, cid = r["repair"], r["chapter_id"]
        if not rep or (rep == "rerender" and not rows_backfill):
            continue
        if con.execute("SELECT 1 FROM job WHERE chapter_id=? AND state IN "
                       "('queued','running','cancelling') LIMIT 1", (cid,)).fetchone():
            done["skipped_live_job"] += 1
            continue
        if rep == "redo":
            reset.rewind_chapter(con, cid, "detected")
            jobs.enqueue(con, "prepare", chapter_id=cid, priority=priority,
                         payload={"auto_to": "video", "narrate": True})
        elif rep == "reprepare":
            jobs.enqueue(con, "prepare", chapter_id=cid, priority=priority,
                         payload={"auto_to": "video"})
        else:
            gates.ensure_approval(con, "voice", chapter_id=cid,
                                  ep_dir=gates.chapter_ep_dir(con, cid), note="rows backfill")
            jobs.enqueue(con, "voiceover", chapter_id=cid, priority=priority,
                         payload={"auto_to": "video"}, dedupe=False)
        con.commit()
        done[rep] += 1
        log(f"[apply] {r['series']} {r['label']}: {rep}")
    return done


def report(rows: List[Dict[str, Any]]) -> str:
    lines = []
    per = {}
    for r in rows:
        for name in r["findings"]:
            per.setdefault(name, {}).setdefault(r["series"], 0)
            per[name][r["series"]] += 1
    lines.append(f"[sweep] {len(rows)} rendered chapters checked")
    for name in CHECKS:
        by_series = per.get(name, {})
        total = sum(by_series.values())
        lines.append(f"  {name:22} {total:5} chapters"
                     + ("  " + ", ".join(f"{s} {n}" for s, n in sorted(by_series.items()))
                        if total else ""))
    reps = {}
    for r in rows:
        if r["repair"]:
            reps[r["repair"]] = reps.get(r["repair"], 0) + 1
    lines.append("  repairs: " + (", ".join(f"{k} {v}" for k, v in reps.items()) or "none"))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=str(REPO / "studio.db"))
    ap.add_argument("--series")
    ap.add_argument("--chapter-ids", help="comma-separated chapter ids")
    ap.add_argument("--check", action="append", choices=sorted(CHECKS),
                    help="run only these checks (repeatable)")
    ap.add_argument("--out", help="JSON report path (default dist/qa_rules/<time>.json)")
    ap.add_argument("--apply", action="store_true", help="queue the repairs (default: report)")
    ap.add_argument("--rows", action="store_true", help="with --apply: also re-render "
                    "chapters that only lack rows (the owner's call)")
    ap.add_argument("--priority", type=int, default=100)
    args = ap.parse_args(argv)
    con = sqlite3.connect(args.db)
    ids = [int(x) for x in re.split(r"[,\s]+", args.chapter_ids or "") if x]
    rows = sweep(con, series=args.series, chapter_ids=ids or None, checks=args.check)
    out = Path(args.out) if args.out else (
        REPO / "dist" / "qa_rules" / f"{_dt.datetime.now():%Y-%m-%d_%H%M}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r for r in rows if r["findings"]], indent=1,
                              ensure_ascii=False))
    print(report(rows))
    print(f"[sweep] report: {out}")
    if args.apply:
        print(f"[apply] {apply(con, rows, rows_backfill=args.rows, priority=args.priority)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
