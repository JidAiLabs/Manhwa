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
    # arc teaser: a model select + a full render -> the render-like cpu lane
    # (like concat/render_segment). MUST be listed or the job queues forever.
    "plan_teaser": "cpu",
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
            payload: Optional[Dict[str, Any]] = None, priority: int = 100,
            dedupe: bool = True) -> int:
    """Insert a job row; returns its id — or, when deduped, the id of the
    already-QUEUED duplicate instead of inserting a new row.

    dedupe=True (default) folds a new job into an existing 'queued' row with
    the same (type, chapter_id, bundle_id) key. The payload is NOT part of the
    key: a queued duplicate with a different payload still dedups onto the
    existing row (pass dedupe=False to force a fresh row regardless — the
    escape hatch for callers that need one). series_id is deliberately NOT
    part of the key when chapter_id or bundle_id is set (they already pin the
    series); for series-scoped jobs (chapter_id AND bundle_id both NULL — e.g.
    refresh/discovery_scan/add_series) series_id IS part of the key so two
    different series' jobs never dedupe against each other. Only 'queued' rows
    match, so a retry enqueued while the original job is still 'running' is
    always a fresh row (this is how the worker's auto-retry stays unaffected).
    A dedupe hit adopts the more urgent (lower) priority so expedite call sites
    keep working.

    Two properties this depends on, both fixed 2026-07-21:

    * It matches every PENDING-OR-LIVE state, not just 'queued'. Matching only
      'queued' meant that once the worker claimed the job, the guard stopped
      working: a second browser tab (or an impatient double-click) sailed past
      it and enqueued a duplicate of a ~40 GPU-minute prepare.
    * The check and the insert are ONE atomic step (BEGIN IMMEDIATE). The
      dashboard opens a fresh connection per request, so concurrent requests
      could both run the SELECT, both miss, and both INSERT.
    """
    def _insert() -> int:
        cur = con.execute(
            "INSERT INTO job (type, series_id, chapter_id, bundle_id, "
            "payload_json, priority) VALUES (?,?,?,?,?,?)",
            (type, series_id, chapter_id, bundle_id,
             json.dumps(payload or {}), priority))
        return int(cur.lastrowid)

    if not dedupe:
        rid = _insert()
        con.commit()
        return rid

    if con.in_transaction:
        con.commit()
    try:
        con.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        pass            # already inside a transaction — still serialized
    try:
        if chapter_id is not None or bundle_id is not None:
            existing = con.execute(
                f"SELECT id, priority FROM job WHERE state IN {_PENDING_SQL} "
                "AND type=? AND chapter_id IS ? AND bundle_id IS ? LIMIT 1",
                (type, chapter_id, bundle_id)).fetchone()
        else:
            existing = con.execute(
                f"SELECT id, priority FROM job WHERE state IN {_PENDING_SQL} "
                "AND type=? AND chapter_id IS NULL AND bundle_id IS NULL AND "
                "series_id IS ? LIMIT 1", (type, series_id)).fetchone()
        if existing:
            existing_id, existing_priority = existing
            # adopt lower (more urgent) priority if the new call is more urgent
            if priority < existing_priority:
                con.execute(
                    "UPDATE job SET priority=? WHERE id=? AND state='queued'",
                    (priority, existing_id))
            rid = int(existing_id)
        else:
            rid = _insert()
        con.commit()
        return rid
    except BaseException:
        con.rollback()
        raise


# 'cancelling' is a LIVE state, not a finished one: the row's child process is
# still running until the worker's cancel-monitor reaps it. Every query that
# asks "is this job occupying a resource?" must therefore count it — the lease,
# the lane width, the claim re-check, and the delete-series guard. Treating it
# as finished let a second job claim the same chapter/lane while the first
# job's child was still writing that chapter's manifests.
LIVE_STATES = ("running", "cancelling")
_LIVE_SQL = "('running','cancelling')"
# live + not-yet-started: "this work is already on the books" — the dedupe key
_PENDING_SQL = "('queued','running','cancelling')"

# chapter lease: a chapter with any LIVE job (any type/lane) is not
# eligible — closes the multi-writer hole where e.g. prepare(gpu) and
# voiceover(tts) for the SAME chapter could otherwise run at once and both
# write the same manifests. NULL chapter_id (series/bundle-scoped jobs) is
# never leased. Appended to a queued-job SELECT's WHERE clause.
_CHAPTER_LEASE_FILTER = (
    " AND (chapter_id IS NULL OR chapter_id NOT IN "
    f"(SELECT chapter_id FROM job WHERE state IN {_LIVE_SQL} AND "
    "chapter_id IS NOT NULL))")


def claim_next(con: sqlite3.Connection,
               lane: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """No lane: fully serial (legacy). With a lane: one running job per lane
    — the serial-GPU guarantee holds inside each resource. A chapter-scoped
    job is additionally skipped (not blocked — the next eligible queued job,
    e.g. a different chapter, is still claimable) while ANY other job for the
    same chapter is already running, in any lane: the chapter lease. There is
    no bundle-level lease (deliberate): concat/plan_teaser already share the
    cpu lane at width 1, so bundle jobs are serialized by the lane alone."""
    if lane is None:
        running = con.execute(
            f"SELECT COUNT(*) FROM job WHERE state IN {_LIVE_SQL} AND "
            "type!='heartbeat'").fetchone()[0]
        if running:
            return None
        r = con.execute(
            f"SELECT {_COLS} FROM job WHERE state='queued'"
            f"{_CHAPTER_LEASE_FILTER} "
            "ORDER BY priority, id LIMIT 1").fetchone()
    else:
        types = _lane_types(lane)
        qs = ",".join("?" for _ in types)
        running = con.execute(
            f"SELECT COUNT(*) FROM job WHERE state IN {_LIVE_SQL} AND type IN "
            f"({qs})", types).fetchone()[0]
        if running >= LANE_WIDTH.get(lane, 1):
            return None
        r = con.execute(
            f"SELECT {_COLS} FROM job WHERE state='queued' AND type IN "
            f"({qs}){_CHAPTER_LEASE_FILTER} "
            "ORDER BY priority, id LIMIT 1", types).fetchone()
    if not r:
        return None
    cur = con.execute(
        "UPDATE job SET state='running', started_at=datetime('now') "
        "WHERE id=? AND state='queued' AND (chapter_id IS NULL OR NOT EXISTS "
        f"(SELECT 1 FROM job j2 WHERE j2.state IN {_LIVE_SQL} AND "
        "j2.chapter_id = job.chapter_id))", (r[0],))
    con.commit()
    if cur.rowcount == 0:
        return None     # a sibling lane thread won the claim race, or leased
    r2 = con.execute(f"SELECT {_COLS} FROM job WHERE id=?", (r[0],)).fetchone()
    return _row(r2)


def finish(con: sqlite3.Connection, job_id: int, *, ok: bool,
           error: str = "") -> bool:
    """Record a terminal outcome. Only a 'running' job may be finished, so a
    handler that returns normally can never overwrite an operator's cancel
    ('cancelling') with 'done' — the race that made Cancel look like it did
    nothing. Returns whether the row was actually moved."""
    cur = con.execute(
        "UPDATE job SET state=?, finished_at=datetime('now'), error=?, "
        "pgid=NULL WHERE id=? AND state='running'",
        ("done" if ok else "failed", error or None, job_id))
    con.commit()
    return cur.rowcount > 0


def cancel(con: sqlite3.Connection, job_id: int) -> bool:
    """Cancel a job. A QUEUED job is cancelled immediately. A RUNNING job is
    marked 'cancelling' — the worker's cancel-monitor kills its subprocess tree
    and records it 'cancelled' (an operator cancel does NOT auto-retry)."""
    cur = con.execute(
        "UPDATE job SET state='cancelled', finished_at=datetime('now') "
        "WHERE id=? AND state='queued'", (job_id,))
    if cur.rowcount:
        con.commit()
        return True
    cur = con.execute(
        "UPDATE job SET state='cancelling' WHERE id=? AND state='running'",
        (job_id,))
    con.commit()
    return cur.rowcount > 0


def requeue(con: sqlite3.Connection, job_id: int) -> bool:
    """Put a stuck LIVE job back on the queue after reaping whatever is left
    of its child. This is the manual recovery for a job whose process died
    without the worker noticing — before this, the only cure was restarting
    the whole worker, because the row held its lane and chapter lease forever.

    Reaps FIRST and only then requeues: requeuing while an old child is still
    alive would put two processes on the same chapter's manifests, which is
    the exact corruption the chapter lease exists to prevent.
    """
    row = con.execute(f"SELECT state, pgid FROM job WHERE id=? AND state IN "
                      f"{_LIVE_SQL}", (job_id,)).fetchone()
    if not row:
        return False
    if row[1]:
        from studio.worker import _reap_pgid     # identity-checked kill
        try:
            _reap_pgid(int(row[1]))
        except Exception:
            pass                # never block recovery on a reap failure
    cur = con.execute(
        f"UPDATE job SET state='queued', started_at=NULL, pgid=NULL "
        f"WHERE id=? AND state IN {_LIVE_SQL}", (job_id,))
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
    if st in LIVE_STATES:
        job["elapsed_sec"] = 0
        if job.get("started_at"):
            row = con.execute(
                "SELECT CAST((julianday('now') - julianday(?)) * 86400 AS INT)",
                (job["started_at"],)).fetchone()
            if row and row[0] is not None:
                job["elapsed_sec"] = max(0, int(row[0]))
        jt, sid = job.get("type"), job.get("series_id")
        # honest estimate = median of REAL finished-job wall-clock for this type
        # (includes heal cycles + the prep/QA inside a voiceover job). Falls back
        # to the per-stage seed sum until that type has run-history.
        est = eta.job_eta(con, jt, sid)
        if est is None:
            stages = _TYPE_STAGES.get(jt)
            est = (sum(eta.stage_eta(con, s, sid) for s in stages)
                   if stages else None)
        if est:
            job["est_total_sec"] = est
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
    # 'cancelling' MUST be listed: it was in no query at all, so pressing
    # Cancel made a job vanish from the UI while its child kept running.
    active = con.execute(
        f"SELECT {_COLS} FROM job WHERE type!='heartbeat' AND "
        "state IN ('running','cancelling','queued') "
        "ORDER BY CASE state WHEN 'queued' THEN 1 ELSE 0 END, priority, id"
    ).fetchall()
    recent = con.execute(
        f"SELECT {_COLS} FROM job WHERE type!='heartbeat' AND "
        "state IN ('done','failed','cancelled') "
        "ORDER BY finished_at DESC, id DESC LIMIT 12").fetchall()
    return ([_with_timing(con, _row(r)) for r in active]
            + [_with_timing(con, _row(r)) for r in recent])


def runs_view(con: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    """Full run history — every finished job, newest first, with durations.
    The job table is never pruned, so this IS the durable per-run record."""
    rows = con.execute(
        f"SELECT {_COLS} FROM job WHERE type!='heartbeat' AND "
        "state IN ('done','failed','cancelled') "
        "ORDER BY finished_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
    return [_with_timing(con, _row(r)) for r in rows]


def _pgid_alive(pgid: Optional[int]) -> bool:
    """Is a process group still alive? signal 0 delivers nothing — it only
    performs the permission/existence check. Used to distinguish 'the job is
    slow' from 'the job's process is GONE', which the dashboard could not
    tell apart before (both looked like a healthy running row forever)."""
    if not pgid:
        return False
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True     # exists, owned by someone else — do not call it dead
    except Exception:
        return True     # cannot prove it is dead -> fail closed, stay silent


def health(con: sqlite3.Connection, *, stale_after_sec: int = 300,
           worker_down_after_sec: int = 90) -> Dict[str, Any]:
    """The one place three DIFFERENT 'looks stuck' signals are computed, so
    the queue page, the chapter page and the health strip can never disagree:

      worker_down — the worker's heartbeat row has not ticked recently. The
                    worker process itself is gone or wedged.
      overdue     — a live job has run past 2x its estimate. Slow, not dead;
                    informational only.
      stale       — a live job whose lease is old AND whose process group is
                    confirmed GONE. This is the real "stuck child": nothing
                    will ever finish it, and it holds its lane and chapter
                    lease forever. Requeue is safe here and nowhere else.

    Every query excludes type='heartbeat' — that row is 'running' by design
    and would otherwise report itself permanently stale.
    """
    hb = con.execute(
        "SELECT CAST((julianday('now') - julianday(started_at)) * 86400 AS "
        "INT) FROM job WHERE type='heartbeat' ORDER BY id LIMIT 1").fetchone()
    hb_age = int(hb[0]) if hb and hb[0] is not None else None

    live = con.execute(
        f"SELECT id, type, chapter_id, series_id, pgid, started_at, "
        "CAST((julianday('now') - julianday(started_at)) * 86400 AS INT) "
        f"FROM job WHERE type!='heartbeat' AND state IN {_LIVE_SQL}"
    ).fetchall()

    stale: List[Dict[str, Any]] = []
    overdue: List[Dict[str, Any]] = []
    for jid, jtype, cid, sid, pgid, started, elapsed in live:
        elapsed = int(elapsed or 0)
        est = eta.job_eta(con, jtype, sid)
        if est and elapsed > 2 * est:
            overdue.append({"id": jid, "type": jtype, "chapter_id": cid,
                            "elapsed_sec": elapsed, "est_total_sec": est})
        # A job with NO pgid is between shell-outs (or has not started one
        # yet) — that is normal, not stale. Only a job whose recorded child
        # is confirmed dead counts.
        if elapsed > stale_after_sec and pgid and not _pgid_alive(pgid):
            stale.append({"id": jid, "type": jtype, "chapter_id": cid,
                          "elapsed_sec": elapsed, "pgid": pgid})

    queued = con.execute(
        "SELECT COUNT(*), CAST((julianday('now') - julianday(MIN(created_at)))"
        " * 86400 AS INT) FROM job WHERE state='queued'").fetchone()
    failed = con.execute(
        "SELECT COUNT(*) FROM job WHERE state='failed' AND "
        "finished_at > datetime('now','-1 day')").fetchone()[0]

    lanes = {}
    for lane in set(LANES.values()):
        types = _lane_types(lane)
        qs = ",".join("?" for _ in types)
        n = con.execute(f"SELECT COUNT(*) FROM job WHERE state IN {_LIVE_SQL} "
                        f"AND type IN ({qs})", types).fetchone()[0]
        lanes[lane] = {"busy": int(n), "width": LANE_WIDTH.get(lane, 1)}

    return {
        "heartbeat_age_sec": hb_age,
        "worker_down": hb_age is None or hb_age > worker_down_after_sec,
        "live": len(live),
        "stale": stale,
        "overdue": overdue,
        "lanes": lanes,
        "queued": int(queued[0] or 0),
        "queued_oldest_sec": int(queued[1] or 0) if queued[0] else 0,
        "failed_recent": int(failed),
    }


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
