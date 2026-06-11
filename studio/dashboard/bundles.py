"""Bundles: a video = intro + ordered chapter segments + outro.

Chapters render once as standalone segments (prep --branding per position);
the bundle is an ffmpeg stream-copy concat — re-bundleable without
re-rendering anything.
"""

from __future__ import annotations

import sqlite3
from typing import Callable, List, Optional, Tuple

from studio.dashboard import eta

INTRO_OUTRO_SEC = 26.0   # measured intro 7s+pad + outro 12s+pad


def create_bundle(con: sqlite3.Connection, series_id: int, kind: str, *,
                  season_no: Optional[int] = None,
                  chapter_range: Optional[Tuple[float, float]] = None,
                  title: str = "") -> int:
    if kind == "season":
        rows = con.execute(
            "SELECT id FROM chapter WHERE series_id=? AND season=? "
            "ORDER BY number", (series_id, season_no)).fetchall()
    elif kind == "full":
        rows = con.execute(
            "SELECT id FROM chapter WHERE series_id=? ORDER BY number",
            (series_id,)).fetchall()
    elif kind == "manual":
        lo, hi = chapter_range or (0, 0)
        rows = con.execute(
            "SELECT id FROM chapter WHERE series_id=? AND number BETWEEN ? "
            "AND ? ORDER BY number", (series_id, lo, hi)).fetchall()
    else:
        raise ValueError(f"bundle kind {kind!r}")
    cur = con.execute(
        "INSERT INTO bundle (series_id, title, kind, season_no) "
        "VALUES (?,?,?,?)", (series_id, title, kind, season_no))
    bid = int(cur.lastrowid)
    for pos, (cid,) in enumerate(rows):
        con.execute("INSERT INTO bundle_chapter (bundle_id, chapter_id, "
                    "position) VALUES (?,?,?)", (bid, cid, pos))
    con.commit()
    return bid


def bundle_chapters(con: sqlite3.Connection, bundle_id: int) -> List[int]:
    return [r[0] for r in con.execute(
        "SELECT chapter_id FROM bundle_chapter WHERE bundle_id=? "
        "ORDER BY position", (bundle_id,))]


def branding_for_position(i: int, n: int) -> str:
    if n == 1:
        return "both"
    if i == 0:
        return "intro"
    if i == n - 1:
        return "outro"
    return "none"


def projected_runtime_sec(con: sqlite3.Connection, bundle_id: int,
                          plan_loader: Callable[[int], Optional[float]]) -> float:
    total = INTRO_OUTRO_SEC
    chapter_audio_eta = eta.SEED_SEC["voiced"] / 2  # ~10min target per chapter
    for cid in bundle_chapters(con, bundle_id):
        dur = plan_loader(cid)
        total += float(dur) if dur else chapter_audio_eta
    return total


def concat_cmd(segments: List[str], out_path: str) -> Tuple[List[str], str]:
    """ffmpeg stream-copy concat argv + the listfile body (caller writes the
    listfile next to out_path and substitutes its path for LISTFILE)."""
    listfile = "\n".join(f"file '{s}'" for s in segments) + "\n"
    argv = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "LISTFILE",
            "-c", "copy", out_path]
    return argv, listfile


def segments_ready(con: sqlite3.Connection, bundle_id: int,
                   probe: Callable[[int], bool]) -> Tuple[int, int]:
    cids = bundle_chapters(con, bundle_id)
    return sum(1 for c in cids if probe(c)), len(cids)
