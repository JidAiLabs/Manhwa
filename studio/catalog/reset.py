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
_STAGES_BEYOND: Dict[str, tuple] = {
    "scripted": ("voiced", "render_segment"),
    "grouped": _NARRATION_STAGES,
    "detected": _NARRATION_STAGES,
    "downloaded": _NARRATION_STAGES,
}

# Derived artifacts per target, relative to ep_dir. Sources (page jpgs) are
# never touched; scenes + understanding survive anything above 'downloaded'.
_VOICE_ARTIFACTS = ("tts", "render")
_NARRATION_ARTIFACTS = (
    "manifest.beats.json", "manifest.script.json", "manifest.sanitize.json",
    "render.plan.json", "render.plan.clean.json",
    "prep_qa.json", "prep_qa.html", "heal_corrections.json",
    "manifest.beats.preheal.json",
)
_VISUAL_ARTIFACTS = (
    "manifest.stitch.json", "manifest.panels.json",
    "manifest.panels.expanded.json", "manifest.scenes.json",
    "manifest.vision.json", "manifest.panels.understood.json",
    "manifest.groups.json", "manifest.story.json", "manifest.cast.json",
    "stitch_chunks", "scenes", "scenes_clean",
    ".cut_judge_cache.json", "manual_drops.json",
)


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
        for rel in _VOICE_ARTIFACTS:
            _rm(ep_dir, rel, deleted)
        if to != "scripted":
            for rel in _NARRATION_ARTIFACTS:
                _rm(ep_dir, rel, deleted)
            if not keep_base:
                _rm(ep_dir, ".narration_keepbase", deleted)
        if to == "downloaded":
            for rel in _VISUAL_ARTIFACTS:
                _rm(ep_dir, rel, deleted)
        elif to == "detected":
            # scenes re-materialize -> per-panel visual verdicts go stale
            _rm(ep_dir, ".cut_judge_cache.json", deleted)

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

    con.execute("UPDATE chapter SET status=?, error=NULL, "
                "updated_at=datetime('now') WHERE id=?", (to, chapter_id))
    con.commit()
    return {"chapter_id": chapter_id, "status": to, "deleted": deleted,
            "backup": backup, "stage_run_cleared": stage_rows,
            "approvals_cleared": approvals}
