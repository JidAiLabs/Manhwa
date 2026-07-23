"""rewind_chapter — the ONE atomic chapter rewind (files + DB together).

Every chapter rewind used to hand-pick its own subset (the dashboard's
rebuild/revoice buttons, ad-hoc SQL resets) and each left a different sliver
of stale state behind — most visibly the append-only ``stage_run`` rows that
kept the dashboard's readiness counters claiming voiced/rendered after the
artifacts were deleted (2026-07-03). All rewinds go through here now:
derived files, stage_run history, approval gates and the status column move
together, scoped strictly to the one chapter.
"""
from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

# Chapter-scoped stage_run rows that claim work BEYOND each rewind target —
# cleared so readiness/timeline (computed from stage_run) matches reality.
# Series/bundle stages (series_thumbnail, plan_teaser, concat) are untouched.
_NARRATION_STAGES = ("chain:scripted", "prepped", "qa_scan", "planned",
                     "voiced", "render_segment")
# "scripted" deliberately omits planned/prepped/qa_scan below: the rewind
# deletes their plan FILES (render.plan*.json) but keeps those stage_run
# rows; reconcile owns DB<->FS drift and prunes them on next load — do not
# widen this list (whole-branch review 2026-07-06).
_STAGES_BEYOND: Dict[str, tuple] = {
    "scripted": ("voiced", "render_segment"),
    "grouped": _NARRATION_STAGES,
    "detected": _NARRATION_STAGES,
    "downloaded": _NARRATION_STAGES,
}

# Per-target delete lists are DERIVED from studio/deps.py (the one dependency
# table): every pipeline manifest produced by a stage strictly beyond the
# rewind target dies with the rewind, so no stale survivor can poison the
# re-run (the old literals left understood/groups/story/cast behind on a
# 'detected' rewind, describing scenes about to be re-materialized).
# Sources (page jpgs) and manifest.series.json are never touched.

# Kept despite being beyond the target: their inputs SURVIVE the rewind, so
# they are still fresh. cast (groups+vision survive a 'grouped' rewind) is a
# paid Gemini call the beated retry must not re-buy; _stage_beated's
# deps-staleness guard owns the rebuild if that ever stops holding.
_KEEP: Dict[str, tuple] = {"grouped": ("manifest.cast.json",)}

# Non-manifest extras (dirs, QA/heal byproducts, caches) that the deps table
# does not track — mirrors the old literals exactly; nothing dropped out.
# (.grounding_cache.json is deliberately NOT here: it is content-addressed,
# so it stays valid across rewinds by design.)
_VOICE_EXTRAS = ("tts", "render")
_NARRATION_EXTRAS = ("prep_qa.json", "prep_qa.html", "heal_corrections.json",
                     "manifest.beats.preheal.json")
_VISUAL_EXTRAS = ("stitch_chunks", "scenes", "scenes_clean",
                  ".cut_judge_cache.json", "manual_drops.json", "auto_drops.json")


def artifacts_for(target: str) -> tuple:
    """Everything a rewind to *target* deletes (relative to ep_dir), derived
    from deps.artifacts_beyond(target) + the non-manifest extras. keep_base
    and the .narration_keepbase marker are handled by rewind_chapter itself."""
    from studio import deps
    rels = [a for a in deps.artifacts_beyond(target)
            if a not in _KEEP.get(target, ())]
    rels += _VOICE_EXTRAS
    if target != "scripted":
        rels += _NARRATION_EXTRAS
    if target == "downloaded":
        rels += _VISUAL_EXTRAS
    elif target == "detected":
        # scenes re-materialize -> per-panel visual verdicts go stale
        rels += (".cut_judge_cache.json",)
    return tuple(dict.fromkeys(rels))


def _rm(ep_dir: Path, rel: str, deleted: list) -> None:
    p = ep_dir / rel
    try:
        if p.is_dir():
            shutil.rmtree(p)
            deleted.append(rel + "/")
        elif p.exists():
            p.unlink()
            deleted.append(rel)
    except OSError:
        pass


def delete_bundle(con: sqlite3.Connection, bundle_id: int) -> list:
    """Delete one video (bundle): its rows (bundle, bundle_chapter, its jobs +
    approvals) AND its dist/bundle_<id> artifacts (teaser, publish_meta). The
    CHAPTERS it referenced are untouched — only the grouping is removed, so
    their episodes return to the pool. Returns the file paths deleted.

    Shared by the dashboard's delete-video button and rewind_chapter's
    clear-on-reset, so the two can never drift.
    """
    from studio.config import REPO_ROOT
    deleted: list = []
    d = REPO_ROOT / "dist" / f"bundle_{bundle_id}"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        deleted.append(str(d))
    con.execute("DELETE FROM bundle_chapter WHERE bundle_id=?", (bundle_id,))
    con.execute("DELETE FROM job WHERE bundle_id=?", (bundle_id,))
    con.execute("DELETE FROM approval WHERE bundle_id=?", (bundle_id,))
    con.execute("DELETE FROM bundle WHERE id=?", (bundle_id,))
    return deleted


def _clear_bundles_for_chapter(con: sqlite3.Connection, chapter_id: int,
                               deleted: list) -> list:
    """Delete every video that contains *chapter_id*. A rewound chapter's
    rendered segment is gone, so any video built from it can no longer
    concat — the whole video is stale and must be rebuilt. The video's OTHER
    chapters are untouched and go back into the pool."""
    bids = [r[0] for r in con.execute(
        "SELECT DISTINCT bundle_id FROM bundle_chapter WHERE chapter_id=?",
        (chapter_id,)).fetchall()]
    for bid in bids:
        deleted.extend(delete_bundle(con, bid))
    return bids


def rewind_chapter(
    con: sqlite3.Connection,
    chapter_id: int,
    to: str,
    *,
    clear_approvals: Optional[Sequence[str]] = None,
    keep_base: bool = False,
) -> Dict[str, Any]:
    """Atomically rewind one chapter to status *to* — files and DB together.

    to: 'scripted' (re-voice; narration kept) | 'grouped' (re-narrate;
    understanding kept) | 'detected' (re-materialize scenes onward) |
    'downloaded' (full re-process; only the page downloads survive).

    clear_approvals: gates to drop (default: ('render',) for 'scripted',
    ('voice', 'render') otherwise). keep_base: leave the .narration_keepbase
    marker in place (a 'grouped' rewind normally removes it — it would
    silently defeat the re-narration being asked for).

    Backs up manifest.beats.json before any narration-destroying target.
    Returns a summary dict of everything it did.
    """
    if to not in _STAGES_BEYOND:
        raise ValueError(f"unknown rewind target {to!r} "
                         f"(expected one of {sorted(_STAGES_BEYOND)})")
    row = con.execute("SELECT ep_dir FROM chapter WHERE id=?",
                      (chapter_id,)).fetchone()
    if row is None:
        raise ValueError(f"chapter {chapter_id} not found")
    ep_dir = Path(row[0]) if row[0] else None

    deleted: list = []
    backup: Optional[str] = None
    if ep_dir and ep_dir.exists():
        beats = ep_dir / "manifest.beats.json"
        if to != "scripted" and beats.exists():
            backup = f"manifest.beats.rewind-{time.strftime('%Y%m%d-%H%M%S')}-BAK.json"
            shutil.copy2(beats, ep_dir / backup)
        for rel in artifacts_for(to):
            _rm(ep_dir, rel, deleted)
        if to != "scripted" and not keep_base:
            _rm(ep_dir, ".narration_keepbase", deleted)

    stages = _STAGES_BEYOND[to]
    cur = con.execute(
        "DELETE FROM stage_run WHERE chapter_id=? AND stage IN (%s)"
        % ",".join("?" * len(stages)), (chapter_id, *stages))
    stage_rows = cur.rowcount

    gates: Iterable[str] = (clear_approvals if clear_approvals is not None
                            else (("render",) if to == "scripted"
                                  else ("voice", "render")))
    gates = tuple(gates)
    approvals = 0
    if gates:
        cur = con.execute(
            "DELETE FROM approval WHERE chapter_id=? AND gate IN (%s)"
            % ",".join("?" * len(gates)), (chapter_id, *gates))
        approvals = cur.rowcount

    # A rewound chapter invalidates any VIDEO built from it (its rendered
    # segment is gone), so those bundles are cleared — otherwise a stale video
    # lingers on the Videos tab and locks the chapter out of the video creator
    # (it counts as "already produced"). Owner directive 2026-07-22.
    bundles_cleared = _clear_bundles_for_chapter(con, chapter_id, deleted)

    con.execute("UPDATE chapter SET status=?, error=NULL, "
                "updated_at=datetime('now') WHERE id=?", (to, chapter_id))
    con.commit()
    return {"chapter_id": chapter_id, "status": to, "deleted": deleted,
            "backup": backup, "stage_run_cleared": stage_rows,
            "approvals_cleared": approvals,
            "bundles_cleared": bundles_cleared}
