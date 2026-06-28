"""Approval + QA gates. Enforced by the WORKER only — the UI just inserts
approval rows; nothing renders, concatenates, or uploads without (a) the
latest prep-QA scan passing and (b) an explicit user approval."""

from __future__ import annotations

import sqlite3
from typing import Optional, Tuple


def approve(con: sqlite3.Connection, gate: str, *,
            series_id: Optional[int] = None, chapter_id: Optional[int] = None,
            bundle_id: Optional[int] = None, note: str = "") -> int:
    cur = con.execute(
        "INSERT INTO approval (gate, series_id, chapter_id, bundle_id, note) "
        "VALUES (?,?,?,?,?)", (gate, series_id, chapter_id, bundle_id, note))
    con.commit()
    return int(cur.lastrowid)


def _has_approval(con: sqlite3.Connection, gate: str, *,
                  series_id: Optional[int] = None,
                  chapter_id: Optional[int] = None,
                  bundle_id: Optional[int] = None) -> bool:
    if chapter_id is not None:
        q = con.execute("SELECT 1 FROM approval WHERE gate=? AND chapter_id=? "
                        "LIMIT 1", (gate, chapter_id))
    elif bundle_id is not None:
        q = con.execute("SELECT 1 FROM approval WHERE gate=? AND bundle_id=? "
                        "LIMIT 1", (gate, bundle_id))
    else:
        q = con.execute("SELECT 1 FROM approval WHERE gate=? AND series_id=? "
                        "LIMIT 1", (gate, series_id))
    return q.fetchone() is not None


def thumbnail_approved(con: sqlite3.Connection, series_id: int) -> bool:
    """One thumbnail per manhwa — approved at the SERIES level. Regenerating
    the thumbnail clears this (the worker deletes the row), so an APPROVED
    badge always refers to the image currently on disk."""
    return _has_approval(con, "thumbnail", series_id=series_id)


def latest_qa_ok(con: sqlite3.Connection, chapter_id: int) -> bool:
    r = con.execute(
        "SELECT ok FROM stage_run WHERE chapter_id=? AND stage='qa_scan' "
        "ORDER BY id DESC LIMIT 1", (chapter_id,)).fetchone()
    return bool(r and r[0])


def voice_allowed(con: sqlite3.Connection, chapter_id: int) -> Tuple[bool, str]:
    """Confirm-upstream-before-expensive-downstream: the narration must be
    read and approved before ~20 GPU-minutes of voiceover are spent on it."""
    if not _has_approval(con, "voice", chapter_id=chapter_id):
        return False, "needs narration approval (read the script first)"
    return True, ""


def render_allowed(con: sqlite3.Connection, chapter_id: int) -> Tuple[bool, str]:
    if not latest_qa_ok(con, chapter_id):
        return False, "needs a passing QA scan (latest scan missing or failed)"
    if not _has_approval(con, "render", chapter_id=chapter_id):
        return False, "needs render approval"
    return True, ""


def concat_allowed(con: sqlite3.Connection, bundle_id: int) -> Tuple[bool, str]:
    # A teaser that's PLANNED but not yet reviewed blocks the bundle — never
    # ship a teaser nobody approved. 'approved'/'declined'/'none' all proceed.
    # None-safe: fetchone() is None when the bundle row doesn't exist (the
    # legacy concat-gate test calls this with no bundle row).
    row = con.execute("SELECT teaser_state FROM bundle WHERE id=?",
                      (bundle_id,)).fetchone()
    if row and row[0] == "planned":
        return False, "teaser planned but not reviewed"
    if not _has_approval(con, "concat", bundle_id=bundle_id):
        return False, "needs concat approval"
    return True, ""


def teaser_allowed(con: sqlite3.Connection, bundle_id: int) -> Tuple[bool, str]:
    """The teaser is the bundle's cold open — confirm-upstream-before-render:
    the user must read+approve the selected hook before it's prepended."""
    if not _has_approval(con, "teaser", bundle_id=bundle_id):
        return False, "needs teaser approval"
    return True, ""
