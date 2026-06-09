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
) -> List[str]:
    """Pick which panels to show, dropping ``redundant`` ones first.

    Keeps ``keep``-role panels before ``redundant`` ones, but returns the chosen
    set in the ORIGINAL scene order (never reorders the montage). Always returns
    at least one panel for a non-empty shot, even if ``max_keep`` < 1.
    """
    if not scene_files:
        return []
    role_by_file: Dict[str, str] = {
        str(e.get("scene_file")): str(e.get("role") or "keep")
        for e in (selection or [])
    }
    keepers = [sf for sf in scene_files if role_by_file.get(sf, "keep") == "keep"]
    redundant = [sf for sf in scene_files if role_by_file.get(sf, "keep") != "keep"]

    n = max(1, int(max_keep))
    chosen = set(keepers[:n])
    # fill remaining slots with redundant panels (in order) if room remains
    for sf in redundant:
        if len(chosen) >= n:
            break
        chosen.add(sf)
    # guarantee at least one panel
    if not chosen:
        chosen.add(scene_files[0])
    return [sf for sf in scene_files if sf in chosen]
