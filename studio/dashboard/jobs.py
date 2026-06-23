"""Serial job queue over studio.db.

UI handlers only INSERT rows here; `studio worker` claims and executes.
claim_next is the serial-GPU policy: it returns nothing while any job is
running, so exactly one pipeline stage owns the machine at a time.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from studio.dashboard import eta

_COLS = ("id, type, series_id, chapter_id, bundle_id, payload_json, state, "
         "priority, created_at, started_at, finished_at, log_path, error")

# assembly-line lanes: stages occupy different resources, so one job may run
# PER LANE simultaneously (prepare ch N+1 on gpu while render ch N-1 on cpu)
LANES = {
    # gemma (ollama LLM) work
    "prepare": "gpu", "qa_scan": "gpu", "chain": "gpu",
    # qwen (TTS) is a DIFFERENT model — its own lane so a voiceover overlaps a
    # prepare instead of waiting behind it (64GB fits gemma + qwen together)
    "voiceover": "tts",
    "render_segment": "cpu", "branding_segments": "cpu", "concat": "cpu",
    "refresh": "api", "discovery_scan": "api", "add_series": "api",
    # metadata + thumbnail: short local-Gemma / Nano-Banana calls — keep them
    # off the gpu lane so they never block a chapter's prepare/voiceover.
    # EVERY claimable type MUST be listed here: lane loops only ever claim
    # types in their own lane (the serial lane=None path is unused by the
    # worker), so a handler in HANDLERS but absent from LANES queues forever.
    "publish_meta": "api", "series_thumbnail": "api",
}

# parallel width per lane (64GB mini): two gpu jobs overlap one chapter's
# Gemma minutes with another's OCR/CPU minutes — ollama serializes its own
# requests so the GPU never thrashes. Renders stay exclusive on cpu.
LANE_WIDTH = {
    "gpu": int(os.environ.get("STUDIO_GPU_WIDTH", "2")),
    "tts": int(os.environ.get("STUDIO_TTS_WIDTH", "1")),   # one qwen voiceover at a time
    "cpu": int(os.environ.get("STUDIO_CPU_WIDTH", "1")),
    "api": int(os.environ.get("STUDIO_API_WIDTH", "2")),
}


def _lane_types(lane: str):
    return [t for t, l in LANES.items() if l == lane]


def _row(r) -> Dict[str, Any]:
    d = dict(zip([c.strip() for c in _COLS.split(",")], r))
    d["payload"] = json.loads(d.get("payload_json") or "{}")
    return d


def enqueue(con: sqlite3.Connection, type: str, *, series_id: Optional[int] = None,
            chapter_id: Optional[int] = None, bundle_id: Optional[int] = None,
            payload: Optional[Dict[str, Any]] = None, priority: int = 100) -> int:
    cur = con.execute(
        "INSERT INTO job (type, series_id, chapter_id, bundle_id, payload_json,"
        " priority) VALUES (?,?,?,?,?,?)",
        (type, series_id, chapter_id, bundle_id,
         json.dumps(payload or {}), priority))
    con.commit()
    return int(cur.lastrowid)


def claim_next(con: sqlite3.Connection,
               lane: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """No lane: fully serial (legacy). With a lane: one running job per lane
    — the serial-GPU guarantee holds inside each resource."""
    if lane is None:
        running = con.execute(
            "SELECT COUNT(*) FROM job WHERE state='running' AND "
            "type!='heartbeat'").fetchone()[0]
        if running:
            return None
        r = con.execute(
            f"SELECT {_COLS} FROM job WHERE state='queued' "
            "ORDER BY priority, id LIMIT 1").fetchone()
    else:
        types = _lane_types(lane)
        qs = ",".join("?" for _ in types)
        running = con.execute(
            f"SELECT COUNT(*) FROM job WHERE state='running' AND type IN "
            f"({qs})", types).fetchone()[0]
        if running >= LANE_WIDTH.get(lane, 1):
            return None
        r = con.execute(
            f"SELECT {_COLS} FROM job WHERE state='queued' AND type IN "
            f"({qs}) ORDER BY priority, id LIMIT 1", types).fetchone()
    if not r:
        return None
    cur = con.execute(
        "UPDATE job SET state='running', started_at=datetime('now') "
        "WHERE id=? AND state='queued'", (r[0],))
    con.commit()
    if cur.rowcount == 0:
        return None     # a sibling lane thread won the claim race
    r2 = con.execute(f"SELECT {_COLS} FROM job WHERE id=?", (r[0],)).fetchone()
    return _row(r2)


def finish(con: sqlite3.Connection, job_id: int, *, ok: bool,
           error: str = "") -> None:
    con.execute("UPDATE job SET state=?, finished_at=datetime('now'), error=? "
                "WHERE id=?",
                ("done" if ok else "failed", error or None, job_id))
    con.commit()


def cancel(con: sqlite3.Connection, job_id: int) -> bool:
    cur = con.execute(
        "UPDATE job SET state='cancelled', finished_at=datetime('now') "
        "WHERE id=? AND state='queued'", (job_id,))
    con.commit()
    return cur.rowcount > 0


def bump(con: sqlite3.Connection, job_id: int) -> None:
    con.execute("UPDATE job SET priority = priority - 1 "
                "WHERE id=? AND state='queued'", (job_id,))
    con.commit()


def set_log(con: sqlite3.Connection, job_id: int, log_path: str) -> None:
    con.execute("UPDATE job SET log_path=? WHERE id=?", (log_path, job_id))
    con.commit()


# Stages a job of each type runs, for a rough running-job countdown. Real
# stage_run medians refine these as history accumulates (eta.stage_eta).
_TYPE_STAGES = {
    # the stages the worker actually records via record_stage for a prepare job —
    # so the ETA self-corrects from real run medians instead of stale per-stage seeds
    "prepare": ["chain:scripted", "planned", "prepped", "qa_scan"],
    "voiceover": ["voiced"],
    "render_segment": ["render_segment"],
    "concat": ["concat"],
}


def _with_timing(con: sqlite3.Connection, job: Dict[str, Any]) -> Dict[str, Any]:
    """Annotate a job with wall-clock timing so the queue shows progress:
    running jobs get ``elapsed_sec`` (live) + ``est_total_sec`` (rough, ~);
    finished jobs get ``duration_sec``. All computed in SQL so UTC stored
    times never collide with the local clock."""
    st = job.get("state")
    if st == "running":
        job["elapsed_sec"] = 0
        if job.get("started_at"):
            row = con.execute(
                "SELECT CAST((julianday('now') - julianday(?)) * 86400 AS INT)",
                (job["started_at"],)).fetchone()
            if row and row[0] is not None:
                job["elapsed_sec"] = max(0, int(row[0]))
        stages = _TYPE_STAGES.get(job.get("type"))
        if stages:
            job["est_total_sec"] = sum(
                eta.stage_eta(con, s, job.get("series_id")) for s in stages)
    elif st in ("done", "failed", "cancelled") and job.get("started_at") \
            and job.get("finished_at"):
        row = con.execute(
            "SELECT CAST((julianday(?) - julianday(?)) * 86400 AS INT)",
            (job["finished_at"], job["started_at"])).fetchone()
        if row and row[0] is not None:
            job["duration_sec"] = max(0, int(row[0]))
    # readable scope label — "Manhwa · Chapter" instead of a bare ch#id
    cid, sid = job.get("chapter_id"), job.get("series_id")
    if cid:
        r = con.execute("SELECT s.title, c.label FROM chapter c "
                        "JOIN series s ON s.id = c.series_id WHERE c.id = ?",
                        (cid,)).fetchone()
        if r:
            job["scope_name"] = f"{r[0]} · {r[1]}"
    elif sid:
        r = con.execute("SELECT title FROM series WHERE id = ?", (sid,)).fetchone()
        if r:
            job["scope_name"] = str(r[0])
    return job


def queue_view(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Running first, then the queue, then the most recent finished jobs —
    a job must never silently vanish the moment it completes."""
    active = con.execute(
        f"SELECT {_COLS} FROM job WHERE type!='heartbeat' AND "
        "state IN ('running','queued') "
        "ORDER BY CASE state WHEN 'running' THEN 0 ELSE 1 END, priority, id"
    ).fetchall()
    recent = con.execute(
        f"SELECT {_COLS} FROM job WHERE type!='heartbeat' AND "
        "state IN ('done','failed','cancelled') "
        "ORDER BY finished_at DESC, id DESC LIMIT 12").fetchall()
    return ([_with_timing(con, _row(r)) for r in active]
            + [_with_timing(con, _row(r)) for r in recent])


# stage_run.stage -> short label, in display order. A stage appears multiple
# times per chapter (prepare + voiceover both run prep/QA, heal loops re-run
# them) so we SUM per stage — one row shows the chapter's TOTAL time per activity.
_CHAPTER_STAGE_LABELS = [
    ("chain:scripted", "prep"),
    ("voiced", "voice"),
    ("prepped", "render-prep"),
    ("qa_scan", "QA"),
    ("render_segment", "render"),
]


def chapter_history(con: sqlite3.Connection,
                    limit: int = 15) -> List[Dict[str, Any]]:
    """One row per chapter for the dashboard: the per-stage time breakdown
    (summed across prepare/voiceover/heal re-runs) + total, most-recently-active
    first. Replaces the per-JOB spam — a chapter's whole cost on a single line."""
    rows = con.execute(
        "SELECT chapter_id, MAX(id) AS last_id FROM stage_run "
        "WHERE chapter_id IS NOT NULL GROUP BY chapter_id "
        "ORDER BY last_id DESC LIMIT ?", (limit,)).fetchall()
    out: List[Dict[str, Any]] = []
    for cid, _last in rows:
        agg: Dict[str, Any] = {}
        for stage, dur, ok in con.execute(
                "SELECT stage, COALESCE(SUM(duration_sec), 0.0), MIN(ok) "
                "FROM stage_run WHERE chapter_id=? GROUP BY stage", (cid,)):
            agg[str(stage)] = (float(dur or 0.0), ok)
        breakdown: List[Dict[str, Any]] = []
        total = 0.0
        for stage, label in _CHAPTER_STAGE_LABELS:
            if stage in agg:
                dur, ok = agg[stage]
                breakdown.append({"label": label, "sec": dur, "ok": ok})
                total += dur
        nm = con.execute(
            "SELECT s.title, c.label FROM chapter c JOIN series s "
            "ON s.id = c.series_id WHERE c.id = ?", (cid,)).fetchone()
        out.append({
            "chapter_id": cid,
            "scope_name": f"{nm[0]} · {nm[1]}" if nm else f"ch#{cid}",
            "breakdown": breakdown,
            "total_sec": total,
        })
    return out


def failed_chapters(con: sqlite3.Connection,
                    series_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Chapters whose MOST RECENT job FAILED (auto-retry exhausted, nothing
    queued/running) — they need a manual reload. Surfaced on the Series tab so a
    dead chapter never silently vanishes from a long run. A chapter that later
    succeeded, or has a pending retry queued, is NOT listed — its latest job
    isn't 'failed'. (SQLite returns the bare columns from the MAX(id) row.)"""
    q = ("SELECT c.id, c.label, c.number, c.status, j.error "
         "FROM chapter c JOIN (SELECT chapter_id, state, error, MAX(id) AS last "
         "  FROM job WHERE chapter_id IS NOT NULL GROUP BY chapter_id) j "
         "  ON j.chapter_id = c.id "
         "WHERE j.state = 'failed' ")
    args: List[Any] = []
    if series_id is not None:
        q += "AND c.series_id = ? "
        args.append(series_id)
    q += "ORDER BY c.number"
    return [dict(zip(("chapter_id", "label", "number", "status", "error"), r))
            for r in con.execute(q, args).fetchall()]
