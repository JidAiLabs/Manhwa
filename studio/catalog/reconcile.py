"""reconcile_chapter — make the DB agree with the artifacts on disk.

Where ``rewind_chapter`` (reset.py) CHANGES a chapter (deletes files + rewinds
the DB together), reconcile only REPAIRS drift. A chapter's state lives in ~6
unsynced stores (chapter.status, stage_run, approval + manifests/tts/render on
disk) and every op writes a subset, so the dashboard can show a state the files
contradict — "voiced" after ``tts/`` was deleted, readiness counters claiming
work whose artifacts are gone, or a stale ``render`` approval that silently
skips the re-render (2026-07-03). reconcile derives the TRUE status from what
artifacts actually exist and repairs status + prunes lying stage_run rows +
clears a render approval that outlived its video, so the UI can never show a
state the files contradict.

Read-only-first + idempotent — safe to run on every dashboard load. Cheap by
construction: every call stats the marker (never avoidable — that is how
existence/size/mtime are known), but a ``.json`` marker's content is only
``json.load``ed once per distinct (mtime, size); repeat validation of an
unchanged multi-MB manifest (e.g. ``vision.json``) across chapters/series on
the same load costs a stat, not a re-parse (see ``_VALID_CACHE``). It never
deletes an artifact and never touches a chapter the worker is actively
running (that state is legitimately transitional) or one left in a
``*_failed`` status (preserve the error for the operator).
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Mapping, Optional, Tuple

from studio import paths

# status -> the marker file (relative to ep_dir) whose existence proves the stage
# finished, HIGHEST first. Mirrors pipeline._STAGE_TABLE (test_reconcile asserts
# they stay in sync) plus the render step (`render_segment`, which sets
# chapter.status='rendered' but is NOT in STATUS_ORDER) as the top rung.
_STATUS_MARKERS = [
    ("rendered", paths.SEGMENT_MP4),
    ("planned",  "render.plan.json"),
    ("voiced",   "tts/tts_index.json"),
    ("scripted", "manifest.script.json"),
    ("beated",   "manifest.beats.json"),
    ("grouped",  "manifest.groups.json"),
    ("visioned", "manifest.vision.json"),
    ("scened",   "manifest.scenes.json"),
    ("detected", "manifest.panels.expanded.json"),
    ("stitched", "manifest.stitch.json"),
]

# stage_run.stage -> the artifact whose ABSENCE makes that row a lie. Readiness
# and the run-timeline are computed from stage_run, so a surviving row claims
# work the files no longer back ("prep 2 · voice 1" after the artifacts were
# deleted). Only stages with a stable single marker are pruned; scene/vision
# meta-stages are left alone.
_STAGE_ARTIFACT = {
    "chain:scripted": "manifest.script.json",
    "voiced":         "tts/tts_index.json",
    "planned":        "render.plan.json",
    "prepped":        "render.plan.clean.json",
    "qa_scan":        "prep_qa.json",
    "render_segment": paths.SEGMENT_MP4,
}

_PAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# path -> (mtime, size, verdict) the last time _valid actually parsed it. The
# working set is every marker path of every chapter ever reconciled in this
# process — bounded by chapter count (thousands, not unbounded), so no
# eviction policy is needed (ponytail: don't build an LRU for a dict this small).
_VALID_CACHE: Dict[str, Tuple[float, int, bool]] = {}


def _valid(ep: str, rel: str) -> bool:
    """A marker proves its stage only if it is USABLE: .json markers must
    exist AND parse (a torn/0-byte manifest must never promote a status);
    non-JSON markers (mp4, dirs) keep exists-only semantics.

    Stats the file on every call (unavoidable — that's how mtime/size are
    known), but only json.load's a .json marker when its (mtime, size) has
    changed since the last check; an unchanged file hits _VALID_CACHE instead
    of being re-parsed."""
    path = os.path.join(ep, rel)
    try:
        st = os.stat(path)
    except OSError:
        return False
    if not rel.endswith(".json"):
        return True
    if st.st_size == 0:
        return False  # short-circuit: a 0-byte file is never valid, no parse needed
    cached = _VALID_CACHE.get(path)
    if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
        return cached[2]
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        verdict = True
    except (ValueError, OSError):
        verdict = False
    _VALID_CACHE[path] = (st.st_mtime, st.st_size, verdict)
    return verdict


def _has_active_job(con: sqlite3.Connection, chapter_id: int) -> bool:
    return con.execute(
        "SELECT 1 FROM job WHERE chapter_id=? AND state IN "
        "('pending','queued','running') LIMIT 1", (chapter_id,)).fetchone() is not None


def derive_status(ep: str) -> Optional[str]:
    """TRUE status = the highest stage whose marker is valid on disk. Returns None
    when nothing is derivable (leave the DB status untouched rather than guess —
    e.g. a chapter with neither manifests nor source pages)."""
    if not ep or not os.path.isdir(ep):
        return None
    for status, marker in _STATUS_MARKERS:
        if not _valid(ep, marker):
            continue
        # render.plan.json is ALSO written as an ESTIMATED preview during the
        # 'prepare' stage (before voicing — the estimate_plan QA flags), so it must
        # not outrank 'voiced': only count 'planned' once the real voiced audio
        # exists, else a scripted-awaiting-review chapter would be advanced.
        if status == "planned" and not _valid(ep, "tts/tts_index.json"):
            continue
        return status
    # no derived manifest, but raw source pages present -> downloaded
    try:
        for f in os.listdir(ep):
            if f[:1].isdigit() and f.lower().endswith(_PAGE_EXTS):
                return "downloaded"
    except OSError:
        pass
    return None


def _render_is_stale(con: sqlite3.Connection, cid: int, ep: str) -> bool:
    """A render approval outlived its rendered content: no mp4, or the
    approval's content_sha no longer matches what render.plan.clean.json +
    tts/tts_index.json hash to NOW. sha comparison (not mtime) — mtime races
    under concurrent writes."""
    mp4 = os.path.join(ep, paths.SEGMENT_MP4)
    if not os.path.exists(mp4):
        return True
    from studio.dashboard import gates  # lazy: dashboard imports catalog
    return not gates._approval_valid(
        con, "render", chapter_id=cid, current_sha=gates.gate_sha("render", ep))


def reconcile_chapter(con: sqlite3.Connection,
                      ch: Mapping[str, Any]) -> Dict[str, Any]:
    """Repair the DB to match the artifacts on disk for one chapter. Returns a
    summary (all falsey when already consistent). ``ch`` is any mapping with
    ``id``, ``status`` and ``ep_dir`` (a sqlite3.Row or a dict)."""
    out: Dict[str, Any] = {"status_from": None, "status_to": None,
                           "stage_runs_pruned": 0, "render_approval_cleared": 0}
    cid = int(ch["id"])
    status = str(ch["status"] or "")
    ep = str(ch["ep_dir"] or "")
    # never fight the worker's transitional state, never mask a real failure
    if status.endswith("_failed") or not ep or not os.path.isdir(ep):
        return out
    if _has_active_job(con, cid):
        return out

    changed = False
    # 1. status: repair to what the artifacts actually say
    true_status = derive_status(ep)
    if true_status and true_status != status:
        con.execute("UPDATE chapter SET status=? WHERE id=?", (true_status, cid))
        out["status_from"], out["status_to"] = status, true_status
        changed = True

    # 2. stage_run: drop rows whose backing artifact is gone (the optimistic read)
    for (stage,) in con.execute(
            "SELECT DISTINCT stage FROM stage_run WHERE chapter_id=?",
            (cid,)).fetchall():
        marker = _STAGE_ARTIFACT.get(stage)
        if marker and not _valid(ep, marker):
            n = con.execute("DELETE FROM stage_run WHERE chapter_id=? AND stage=?",
                            (cid, stage)).rowcount
            out["stage_runs_pruned"] += n
            changed = changed or bool(n)

    # 3. render approval must match the CONTENT it was approved for, not just
    #    exist. A legacy NULL-sha row is grandfathered by stamping it with
    #    TODAY's sha, one time, so future drift is actually detectable; a row
    #    that already carries a sha and no longer matches (or the video is
    #    gone) -> clear it, else auto_to=video silently skips the re-render
    #    (the worker gate would otherwise read "already approved").
    from studio.dashboard import gates  # lazy: dashboard imports catalog
    approval_row = con.execute(
        "SELECT id, content_sha FROM approval WHERE gate='render' AND "
        "chapter_id=? ORDER BY id DESC LIMIT 1", (cid,)).fetchone()
    if approval_row is not None:
        approval_id, stored_sha = approval_row
        if stored_sha is None:
            backfill_sha = gates.gate_sha("render", ep)
            if backfill_sha is not None:
                con.execute("UPDATE approval SET content_sha=? WHERE id=?",
                            (backfill_sha, approval_id))
                changed = True
        if _render_is_stale(con, cid, ep):
            n = con.execute(
                "DELETE FROM approval WHERE gate='render' AND chapter_id=?",
                (cid,)).rowcount
            out["render_approval_cleared"] = n
            changed = changed or bool(n)

    if changed:
        con.commit()
    return out


def reconcile_series(con: sqlite3.Connection, series_id: int) -> Dict[str, int]:
    """reconcile every chapter of a series — the series-list readiness counts are
    computed from stage_run and stay wrong until the dead rows are pruned."""
    totals = {"chapters_repaired": 0, "stage_runs_pruned": 0}
    rows = con.execute(
        "SELECT id, status, ep_dir FROM chapter WHERE series_id=?",
        (series_id,)).fetchall()
    for r in rows:
        res = reconcile_chapter(con, {"id": r[0], "status": r[1], "ep_dir": r[2]})
        if (res["status_to"] or res["stage_runs_pruned"]
                or res["render_approval_cleared"]):
            totals["chapters_repaired"] += 1
        totals["stage_runs_pruned"] += res["stage_runs_pruned"]
    return totals
