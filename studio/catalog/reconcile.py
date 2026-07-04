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

Read-only-first + cheap (stat calls) + idempotent — safe to run on every
dashboard load. It never deletes an artifact and never touches a chapter the
worker is actively running (that state is legitimately transitional) or one left
in a ``*_failed`` status (preserve the error for the operator).
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, Mapping, Optional

# status -> the marker file (relative to ep_dir) whose existence proves the stage
# finished, HIGHEST first. Mirrors pipeline._STAGE_TABLE (test_reconcile asserts
# they stay in sync) plus the render step (`render_segment`, which sets
# chapter.status='rendered' but is NOT in STATUS_ORDER) as the top rung.
_STATUS_MARKERS = [
    ("rendered", "render/segment_both.mp4"),
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
    "render_segment": "render/segment_both.mp4",
}

_PAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _exists(ep: str, rel: str) -> bool:
    return os.path.exists(os.path.join(ep, rel))


def _has_active_job(con: sqlite3.Connection, chapter_id: int) -> bool:
    return con.execute(
        "SELECT 1 FROM job WHERE chapter_id=? AND state IN "
        "('pending','queued','running') LIMIT 1", (chapter_id,)).fetchone() is not None


def derive_status(ep: str) -> Optional[str]:
    """TRUE status = the highest stage whose marker exists on disk. Returns None
    when nothing is derivable (leave the DB status untouched rather than guess —
    e.g. a chapter with neither manifests nor source pages)."""
    if not ep or not os.path.isdir(ep):
        return None
    for status, marker in _STATUS_MARKERS:
        if _exists(ep, marker):
            return status
    # no derived manifest, but raw source pages present -> downloaded
    try:
        for f in os.listdir(ep):
            if f[:1].isdigit() and f.lower().endswith(_PAGE_EXTS):
                return "downloaded"
    except OSError:
        pass
    return None


def _render_is_stale(ep: str) -> bool:
    """A render approval outlived its video: no mp4, or the mp4 is older than the
    clean plan it should have been rendered from."""
    mp4 = os.path.join(ep, "render", "segment_both.mp4")
    plan = os.path.join(ep, "render.plan.clean.json")
    if not os.path.exists(mp4):
        return True
    return os.path.exists(plan) and os.path.getmtime(mp4) < os.path.getmtime(plan)


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
        if marker and not _exists(ep, marker):
            n = con.execute("DELETE FROM stage_run WHERE chapter_id=? AND stage=?",
                            (cid, stage)).rowcount
            out["stage_runs_pruned"] += n
            changed = changed or bool(n)

    # 3. a render approval that outlived its video -> clear it, else auto_to=video
    #    silently skips the re-render (the worker gate reads "not approved")
    from studio.dashboard import gates  # lazy: dashboard imports catalog
    if gates._has_approval(con, "render", chapter_id=cid) and _render_is_stale(ep):
        n = con.execute("DELETE FROM approval WHERE gate='render' AND chapter_id=?",
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
