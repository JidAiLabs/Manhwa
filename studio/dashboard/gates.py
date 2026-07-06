"""Approval + QA gates. Enforced by the WORKER only — the UI just inserts
approval rows; nothing renders, concatenates, or uploads without (a) the
latest prep-QA scan passing and (b) an explicit user approval."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Union


def gate_sha(gate: str, ep_dir: Optional[Union[str, Path]]) -> Optional[str]:
    """Hash of the content a gate's approval is FOR. Comparing this against an
    approval row's stored content_sha is what binds an approval to CONTENT
    instead of a checkbox: a healed script or a regenerated plan is different
    content, so the old approval must stop covering it.

    'voice' hashes manifest.script.json BYTES (deliberately no parse — _meta
    churn inside the file is intended to invalidate; a rewritten script is a
    different approval subject). 'render' hashes render.plan.clean.json +
    tts/tts_index.json bytes together. Any referenced file missing, or no
    ep_dir at all -> None (caller stores/compares as legacy semantics). Other
    gates (thumbnail, teaser, concat) -> None, out of scope for now."""
    if not ep_dir:
        return None
    ep = Path(ep_dir)
    if gate == "voice":
        return _sha_file(ep / "manifest.script.json")
    if gate == "render":
        plan_sha = _sha_file(ep / "render.plan.clean.json")
        tts_sha = _sha_file(ep / "tts" / "tts_index.json")
        return f"{plan_sha}:{tts_sha}" if plan_sha and tts_sha else None
    return None


def _sha_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def chapter_ep_dir(con: sqlite3.Connection, chapter_id: int) -> Optional[str]:
    """ep_dir for a chapter id, None-safe. Shared by call sites that need
    ep_dir before deciding whether a full chapter row is even required — a
    gate check must tolerate a bad/missing chapter_id, not raise."""
    row = con.execute("SELECT ep_dir FROM chapter WHERE id=?",
                      (chapter_id,)).fetchone()
    return row[0] if row else None


def approve(con: sqlite3.Connection, gate: str, *,
            series_id: Optional[int] = None, chapter_id: Optional[int] = None,
            bundle_id: Optional[int] = None, note: str = "",
            content_sha: Optional[str] = None) -> int:
    cur = con.execute(
        "INSERT INTO approval (gate, series_id, chapter_id, bundle_id, note, "
        "content_sha) VALUES (?,?,?,?,?,?)",
        (gate, series_id, chapter_id, bundle_id, note, content_sha))
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


def _approval_valid(con: sqlite3.Connection, gate: str, *,
                    series_id: Optional[int] = None,
                    chapter_id: Optional[int] = None,
                    bundle_id: Optional[int] = None,
                    current_sha: Optional[str]) -> bool:
    """Like _has_approval, but binds to CONTENT: the newest matching row must
    exist; a NULL content_sha is a legacy row (pre-dates this column, or was
    approved when the subject file didn't exist) and is grandfathered valid;
    otherwise the stored sha must match what the content hashes to NOW."""
    if chapter_id is not None:
        q = con.execute(
            "SELECT content_sha FROM approval WHERE gate=? AND chapter_id=? "
            "ORDER BY id DESC LIMIT 1", (gate, chapter_id))
    elif bundle_id is not None:
        q = con.execute(
            "SELECT content_sha FROM approval WHERE gate=? AND bundle_id=? "
            "ORDER BY id DESC LIMIT 1", (gate, bundle_id))
    else:
        q = con.execute(
            "SELECT content_sha FROM approval WHERE gate=? AND series_id=? "
            "ORDER BY id DESC LIMIT 1", (gate, series_id))
    row = q.fetchone()
    if row is None:
        return False
    stored = row[0]
    return True if stored is None else stored == current_sha


def ensure_approval(con: sqlite3.Connection, gate: str, *,
                    series_id: Optional[int] = None,
                    chapter_id: Optional[int] = None,
                    bundle_id: Optional[int] = None,
                    ep_dir: Optional[Union[str, Path]] = None,
                    note: str = "") -> None:
    """For auto-advance flows only (bulk 'run to X', autopilot, re-voice):
    treat a STALE approval — row exists but its content_sha no longer
    matches current content — the same as an ABSENT one. Those flows used to
    guard on existence-only `_has_approval`, which a healed script or
    regenerated plan defeats: the stale row still "exists", so the guard
    skips re-approving AND re-enqueueing, silently wedging the chain even
    though the flow's own comments declare it auto-approves on drift.

    No-op when the newest row is already valid for the current content;
    otherwise inserts a fresh row stamped with the current sha, which
    supersedes the stale one (newest-wins in `_approval_valid`). Never touches
    the manual /approve endpoint or any human-review gate."""
    current_sha = gate_sha(gate, ep_dir)
    if _approval_valid(con, gate, series_id=series_id, chapter_id=chapter_id,
                       bundle_id=bundle_id, current_sha=current_sha):
        return
    approve(con, gate, series_id=series_id, chapter_id=chapter_id,
           bundle_id=bundle_id, note=note, content_sha=current_sha)


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


def voice_allowed(con: sqlite3.Connection, chapter_id: int,
                  ep_dir: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
    """Confirm-upstream-before-expensive-downstream: the narration must be
    read and approved before ~20 GPU-minutes of voiceover are spent on it.
    Bound to CONTENT: a script healed/rewritten after approval invalidates
    the old approval (gate_sha comparison), not just its existence."""
    current_sha = gate_sha("voice", ep_dir)
    if not _approval_valid(con, "voice", chapter_id=chapter_id,
                           current_sha=current_sha):
        return False, "needs narration approval (read the script first)"
    return True, ""


def render_allowed(con: sqlite3.Connection, chapter_id: int,
                   ep_dir: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
    if not latest_qa_ok(con, chapter_id):
        return False, "needs a passing QA scan (latest scan missing or failed)"
    current_sha = gate_sha("render", ep_dir)
    if not _approval_valid(con, "render", chapter_id=chapter_id,
                           current_sha=current_sha):
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
