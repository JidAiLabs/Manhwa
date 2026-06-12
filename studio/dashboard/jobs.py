"""Serial job queue over studio.db.

UI handlers only INSERT rows here; `studio worker` claims and executes.
claim_next is the serial-GPU policy: it returns nothing while any job is
running, so exactly one pipeline stage owns the machine at a time.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

_COLS = ("id, type, series_id, chapter_id, bundle_id, payload_json, state, "
         "priority, created_at, started_at, finished_at, log_path, error")

# assembly-line lanes: stages occupy different resources, so one job may run
# PER LANE simultaneously (prepare ch N+1 on gpu while render ch N-1 on cpu)
LANES = {
    "prepare": "gpu", "voiceover": "gpu", "qa_scan": "gpu", "chain": "gpu",
    "render_segment": "cpu", "branding_segments": "cpu", "concat": "cpu",
    "refresh": "api", "discovery_scan": "api", "add_series": "api",
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
        if running:
            return None
        r = con.execute(
            f"SELECT {_COLS} FROM job WHERE state='queued' AND type IN "
            f"({qs}) ORDER BY priority, id LIMIT 1", types).fetchone()
    if not r:
        return None
    con.execute("UPDATE job SET state='running', started_at=datetime('now') "
                "WHERE id=? AND state='queued'", (r[0],))
    con.commit()
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
    return [_row(r) for r in active] + [_row(r) for r in recent]
