"""
tools/scene_selection.py

Shared pure logic for the Gemini scene-understanding pass (SP2 #2 real fix +
semantic dedup). Kept tool-agnostic and dependency-free so both producer and
consumer share one contract:

  - gemini_narrative_pass.py calls normalize_scene_selection() to sanitize the
    model's per-scene judgments before writing them into manifest.beats.json.
  - timeline_planner.py calls choose_kept_scenes() to decide which panels to show,
    dropping "redundant" panels FIRST (so kept panels get their >=min_cut seconds)
    instead of the old arbitrary truncation.

A per-scene selection entry is::

    {"scene_file": str,
     "role": "keep" | "redundant",
     "bubble_mode": "spoken" | "inner_thought" | "narration" | "shout" | "none" | "unknown",
     "intensity": "calm" | "tense" | "intense" | "explosive" | "unknown",
     "reason": str}

Design rule: **default to keep.** A panel is only dropped when the model
explicitly marks it redundant — never as a side effect of a parse gap.
"""

from __future__ import annotations

from typing import Any, Dict, List

VALID_ROLES = ("keep", "redundant")
VALID_BUBBLE_MODES = ("spoken", "inner_thought", "narration", "shout", "none")
VALID_INTENSITIES = ("calm", "tense", "intense", "explosive")


def _sanitize_entry(raw: Dict[str, Any], scene_file: str) -> Dict[str, Any]:
    role = str((raw or {}).get("role") or "").strip().lower()
    if role not in VALID_ROLES:
        role = "keep"
    bubble = str((raw or {}).get("bubble_mode") or "").strip().lower()
    if bubble not in VALID_BUBBLE_MODES:
        bubble = "unknown"
    intensity = str((raw or {}).get("intensity") or "").strip().lower()
    if intensity not in VALID_INTENSITIES:
        intensity = "unknown"
    reason = str((raw or {}).get("reason") or "")[:200]
    return {
        "scene_file": scene_file,
        "role": role,
        "bubble_mode": bubble,
        "intensity": intensity,
        "reason": reason,
    }


def normalize_scene_selection(
    raw_selection: List[Dict[str, Any]] | None,
    scene_files: List[str],
) -> List[Dict[str, Any]]:
    """Return exactly one sanitized entry per *scene_file*, in order.

    Unknown/invalid values fall back to safe defaults (role=keep). Model entries
    for scenes not in *scene_files* are ignored; scenes the model omitted get a
    default keep entry.
    """
    by_file: Dict[str, Dict[str, Any]] = {}
    for r in raw_selection or []:
        sf = (r or {}).get("scene_file")
        if sf:
            by_file[str(sf)] = r
    return [_sanitize_entry(by_file.get(sf, {}), sf) for sf in scene_files]


def choose_kept_scenes(
    scene_files: List[str],
    selection: List[Dict[str, Any]] | None,
    max_keep: int,
    *,
    protected: "set[str] | None" = None,
) -> List[str]:
    """Pick which panels to show, DROPPING ``redundant`` panels entirely.

    A recap shows only the panels worth showing: ``redundant`` panels are not
    displayed at all (so the kept panels get the freed time as longer holds and
    same-moment duplicates never appear on screen), rather than being padded back
    in to fill the time budget. Shows up to *max_keep* ``keep`` panels in original
    order. If a shot has NO keepers (all redundant), it falls back to the first
    *max_keep* panels so the shot is never empty.

    *protected* files (title/system/status cards) are story beats that
    must NEVER be dropped, even when the LLM scene-selection marks them redundant
    (its verdict is non-deterministic — the same card is kept on one run, dropped
    on another). They are always kept, ahead of the *max_keep* budget.
    """
    if not scene_files:
        return []
    prot = protected or set()
    role_by_file: Dict[str, str] = {
        str(e.get("scene_file")): str(e.get("role") or "keep")
        for e in (selection or [])
    }
    keepers = [sf for sf in scene_files
               if sf in prot or role_by_file.get(sf, "keep") == "keep"]

    n = max(1, int(max_keep))
    # mandatory cards are kept ALL; other keepers fill the remaining budget,
    # preserving original order across the union
    must = [sf for sf in scene_files if sf in prot]
    chosen = [sf for sf in (keepers or scene_files)
              if sf in must][:len(must)]
    for sf in (keepers or scene_files):
        if len(chosen) >= max(n, len(must)):
            break
        if sf not in chosen:
            chosen.append(sf)
    chosen = [sf for sf in scene_files if sf in chosen]   # restore order
    if not chosen:
        chosen = scene_files[:1]
    return chosen
