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
                  chapter_ids: Optional[List[int]] = None,
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
        if chapter_ids is not None:
            # an EXPLICIT set — used by the debut-bundle auto-proposer, whose
            # "first N prepared" chapters are not guaranteed to be a gapless
            # number range (a failed chapter mid-run leaves a hole, and a
            # BETWEEN range would then silently pull the unprepared chapter in).
            qs = ",".join("?" for _ in chapter_ids)
            rows = con.execute(
                f"SELECT id FROM chapter WHERE series_id=? AND id IN ({qs}) "
                "ORDER BY number", (series_id, *chapter_ids)).fetchall()
        else:
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


def create_debut_bundle_once(con: sqlite3.Connection, series_id: int,
                             chapter_ids: List[int], *,
                             title: str = "") -> Optional[int]:
    """Atomically create a 'manual' debut bundle of *chapter_ids* IFF *series_id*
    has NO bundle yet. Returns the new bundle id, or None if one already existed.

    The auto-proposer runs from _h_prepare, which the worker executes on TWO gpu
    lane threads concurrently (STUDIO_GPU_WIDTH=2), each with its own connection.
    A plain 'SELECT COUNT(*) FROM bundle' then 'INSERT' is a check-and-create
    race: under WAL both threads read 0 before either commits, and both insert —
    two debut bundles, two teaser proposals. This serializes the check and the
    insert under one BEGIN IMMEDIATE write lock (the same idiom jobs.enqueue
    uses), so the loser observes the winner's committed row and does nothing.
    """
    if not chapter_ids:
        return None
    if con.in_transaction:
        con.commit()
    con.execute("BEGIN IMMEDIATE")
    try:
        if con.execute("SELECT COUNT(*) FROM bundle WHERE series_id=?",
                       (series_id,)).fetchone()[0]:
            con.commit()
            return None                       # someone already owns the layout
        qs = ",".join("?" for _ in chapter_ids)
        rows = con.execute(
            f"SELECT id FROM chapter WHERE series_id=? AND id IN ({qs}) "
            "ORDER BY number", (series_id, *chapter_ids)).fetchall()
        cur = con.execute(
            "INSERT INTO bundle (series_id, title, kind, season_no) "
            "VALUES (?,?,'manual',NULL)", (series_id, title))
        bid = int(cur.lastrowid)
        for pos, (cid,) in enumerate(rows):
            con.execute("INSERT INTO bundle_chapter (bundle_id, chapter_id, "
                        "position) VALUES (?,?,?)", (bid, cid, pos))
        con.commit()
        return bid
    except BaseException:
        con.rollback()
        raise


def unbundled_chapters(con: sqlite3.Connection,
                       series_id: int) -> List[dict]:
    """Chapters of *series_id* that are RENDERED (a finished episode video) and
    NOT already in ANY bundle, in reading order — the candidates for a NEW
    video. A video is a concat of rendered segments, so a chapter that hasn't
    been rendered yet is not a candidate; and already-produced (bundled)
    chapters are excluded so a continuation batch can never re-select them.
    status='rendered' is the reconciled truth (reconcile sets it iff the
    segment .mp4 exists), so the caller should reconcile the series first."""
    rows = con.execute(
        "SELECT id, number, label, status, ep_dir FROM chapter c "
        "WHERE series_id=? AND status='rendered' AND NOT EXISTS "
        "(SELECT 1 FROM bundle_chapter bc JOIN bundle b ON b.id=bc.bundle_id "
        " WHERE b.series_id=? AND bc.chapter_id=c.id) "
        "ORDER BY number", (series_id, series_id)).fetchall()
    return [dict(zip(("id", "number", "label", "status", "ep_dir"), r))
            for r in rows]


def rendered_chapters(con: sqlite3.Connection,
                      series_id: int) -> List[dict]:
    """EVERY rendered chapter of *series_id*, in reading order, each flagged
    with whether it is already in a video (`bundled`).

    unbundled_chapters() REMOVES produced chapters so a continuation cannot
    re-pick them. That also made them unpickable on purpose — the new-video
    form simply started after the last produced episode, so a 12-episode debut
    left the range beginning at Episode 12 with no way to include anything
    earlier. This keeps every rendered episode selectable and instead makes the
    state visible, so re-using one is a deliberate choice rather than a hidden
    option. Chapters that are NOT rendered are still excluded: a video is a
    concat of finished segments, and an unrendered chapter has no .mp4.
    """
    rows = con.execute(
        "SELECT c.id, c.number, c.label, c.status, c.ep_dir, "
        "       EXISTS(SELECT 1 FROM bundle_chapter bc "
        "              JOIN bundle b ON b.id=bc.bundle_id "
        "              WHERE b.series_id=? AND bc.chapter_id=c.id) AS bundled "
        "FROM chapter c WHERE c.series_id=? AND c.status='rendered' "
        "ORDER BY c.number", (series_id, series_id)).fetchall()
    out = []
    for r in rows:
        d = dict(zip(("id", "number", "label", "status", "ep_dir", "bundled"), r))
        d["bundled"] = bool(d["bundled"])
        out.append(d)
    return out


def ids_in_range(con: sqlite3.Connection, series_id: int, *,
                 num_from: Optional[float] = None,
                 num_to: Optional[float] = None) -> List[int]:
    """Rendered chapter ids inside [num_from, num_to] — INCLUDING ones already
    in a video. The free-selection counterpart of unbundled_ids_in_range; the
    form and the POST must agree, or the dropdown would offer an episode the
    POST then silently drops. Bounds are sentinel-free (None -> the series'
    min/max) so chapter 0 is a real bound."""
    lo, hi = num_from, num_to
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return [ch["id"] for ch in rendered_chapters(con, series_id)
            if (lo is None or ch["number"] >= lo)
            and (hi is None or ch["number"] <= hi)]


def unbundled_ids_in_range(con: sqlite3.Connection, series_id: int, *,
                           num_from: Optional[float] = None,
                           num_to: Optional[float] = None) -> List[int]:
    """Chapter ids the new-video range would actually bundle: unbundled AND
    inside [num_from, num_to]. Sentinel-free bounds (None -> the series'
    min/max) so a 0-based series' chapter 0 is a real, selectable bound."""
    lo, hi = num_from, num_to
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    out = []
    for ch in unbundled_chapters(con, series_id):
        n = ch["number"]
        if (lo is None or n >= lo) and (hi is None or n <= hi):
            out.append(ch["id"])
    return out


def bundle_with_exact_chapters(con: sqlite3.Connection, series_id: int,
                               chapter_ids: List[int]) -> Optional[int]:
    """The id of an existing bundle of *series_id* holding EXACTLY this set of
    chapters, else None.

    Episodes may now be re-used across videos deliberately, so "already
    bundled" can no longer be the thing that stops a double-submit. This keeps
    that protection precisely: an identical video is refused, while any
    genuinely different range — overlapping or not — is allowed.
    """
    want = set(chapter_ids)
    if not want:
        return None
    for (bid,) in con.execute("SELECT id FROM bundle WHERE series_id=?",
                              (series_id,)).fetchall():
        if set(bundle_chapters(con, bid)) == want:
            return bid
    return None


def series_bundle_count(con: sqlite3.Connection, series_id: int) -> int:
    return con.execute("SELECT COUNT(*) FROM bundle WHERE series_id=?",
                       (series_id,)).fetchone()[0]


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


def wrap_with_branding(segments: List[str], intro: str, outro: str,
                       *, exists=None) -> List[str]:
    """Every published video = chapter segments + [series outro]. The INTRO was
    dropped (channel decision 2026-06-15): videos open straight on the story so
    viewers don't bounce on a cold open. *intro* is accepted but ignored. The
    outro is a per-series standalone mp4 (singles, season packs, ladder steps and
    FULL compilations are all just concats of the same segments)."""
    import os as _os
    exists = exists or _os.path.exists
    out = list(segments)
    if outro and exists(outro):
        out = out + [outro]
    return out


def branding_intro_plan(thumb_file: str, w: int, h: int, *,
                        intro_dur: float, pad: float = 1.0) -> dict:
    """Minimal render plan for the standalone series intro segment: the
    channel intro overlay held over the series thumbnail."""
    d = round(intro_dur + pad, 4)
    return {
        "timeline": [{
            "segment_id": "branding_intro", "branding": "intro",
            "start_sec": 0.0, "duration_sec": d, "end_sec": d,
            "cuts": [{"file": thumb_file, "start": 0.0, "dur": d}],
        }],
        "scenes_subdir": ".",
        "scene_dims": {thumb_file: {"w": int(w), "h": int(h), "doc": False}},
        "total_duration_sec": d,
    }


def branding_outro_plan(*, outro_dur: float, pad: float = 3.0) -> dict:
    """The outro end-card is drawn by the renderer itself (cuts=[])."""
    d = round(outro_dur + pad, 4)
    return {
        "timeline": [{
            "segment_id": "branding_outro", "branding": "outro",
            "start_sec": 0.0, "duration_sec": d, "end_sec": d, "cuts": [],
        }],
        "scenes_subdir": ".", "scene_dims": {}, "total_duration_sec": d,
    }
