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


def claim_next(con: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    running = con.execute(
        "SELECT COUNT(*) FROM job WHERE state='running' AND type!='heartbeat'"
    ).fetchone()[0]
    if running:
        return None
    r = con.execute(
        f"SELECT {_COLS} FROM job WHERE state='queued' "
        "ORDER BY priority, id LIMIT 1").fetchone()
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
    rows = con.execute(
        f"SELECT {_COLS} FROM job WHERE type!='heartbeat' AND "
        "state IN ('running','queued','failed') "
        "ORDER BY CASE state WHEN 'running' THEN 0 WHEN 'queued' THEN 1 "
        "ELSE 2 END, priority, id").fetchall()
    return [_row(r) for r in rows]
