#!/usr/bin/env python3
"""teaser_planner.py — bundle-level "arc teaser" planner.

Selects a single high-stakes window from the chapters in a bundle and (Stage 2,
later chunks) writes spoiler-safe narration + a synthetic episode dir so the
existing render/TTS tools can turn it into a short cold-open `teaser.mp4`
prepended to the bundle concat.

This module is split in two stages:

  Stage 1 (this chunk) — DETERMINISTIC, $0, pure functions. No LLM, no I/O.
    eligible_panels  — drop chrome/empty/error panels, keep story|caption|system.
    score_window     — score one contiguous window by keyword/intensity signals.
    score_windows    — spoiler guard (exclude the payoff tail + the single global
                       peak panel), enumerate min..max windows, greedily shortlist
                       the top non-overlapping windows.

  Stage 2 (later chunks) — one injectable model call to pick a window + write
    narration, then materialize the synthetic teaser dir.

The keyword sets and weights below are the ONLY calibration knobs and are
intentionally agnostic (no per-series config).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# tools/ is not a package; make sibling modules importable by name (the same
# idiom gemini_narrative_pass.py uses to reach recap_style). recap_style is
# lightweight (stdlib only) so a module-level import is safe; the heavier
# gemini_narrative_pass / google-genai imports stay lazy inside main()'s
# model-call builder so unit tests never pull them in.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
import recap_style  # noqa: E402

# Panel kinds that can carry the story forward (chrome/empty/etc. are dropped).
_ELIGIBLE_KINDS = {"story", "caption", "system"}

# Ordinal rank of an understood panel's intensity tag.
INTENSITY_RANK = {"calm": 0, "tense": 1, "intense": 2, "explosive": 3}

# --- Stage-1 scoring signals (agnostic; the ONLY calibration knobs) ---------- #
# Keyword families that mark a teaser-worthy moment. Word-boundary + greedy
# suffix (\w*) so "humiliat" catches humiliate/humiliated/humiliation, etc.
_STAKES_RE = re.compile(
    r"\b(exam|test|trial|rank|survival|expel|expuls|execut|contract|tournament|duel)\w*", re.I)
_SOCIAL_RE = re.compile(
    r"\b(humiliat|mock|laugh|badge|token|reject|outcast|disgrace|shame|peasant)\w*", re.I)
_POWER_RE = re.compile(
    r"\b(system|status window|skill|rank up|awaken|hidden|impossible|level|power|technique)\w*", re.I)
_ENEMY_RE = re.compile(
    r"\b(elder|heir|clan|authority|enemy|assassin|master|commander|villain)\w*", re.I)

# Per-signal keyword hits are capped so one keyword-stuffed window can't dominate.
_SIGNAL_CAP = 4
# How many distinct subjects a window may carry before it reads as cluttered.
_CLARITY_SUBJECT_LIMIT = 6

# Weights — the single calibration table. Keyword families, the intensity peak,
# visual variety, and a clarity penalty for too many distinct subjects.
_W_STAKES = 1.0
_W_SOCIAL = 1.0
_W_POWER = 0.8
_W_ENEMY = 0.6
_W_INTENSITY = 1.5
_W_VARIETY = 0.5
_W_CLARITY_PENALTY = 0.75


# --------------------------------------------------------------------------- #
# Task 3: panel eligibility + flattening
# --------------------------------------------------------------------------- #
def eligible_panels(panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only panels usable in a teaser, in reading order.

    A panel is eligible when its ``panel_kind`` is one of story|caption|system
    AND it has no ``error`` key (a parse/understanding failure). Chrome, empty
    fields, watermarks, etc. are dropped.
    """
    out: List[Dict[str, Any]] = []
    for p in panels:
        if "error" in p:
            continue
        if p.get("panel_kind") not in _ELIGIBLE_KINDS:
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Task 4: signal scoring of one window
# --------------------------------------------------------------------------- #
def _panel_text(panel: Dict[str, Any]) -> str:
    """The searchable text of one panel: description + action + dialogue."""
    return " ".join(
        str(panel.get(k, "") or "") for k in ("description", "action", "dialogue")
    )


def score_window(panels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one contiguous window of panels for teaser-worthiness.

    The score is a weighted sum of:
      - stakes/social/power/enemy keyword hits (each capped at ``_SIGNAL_CAP``),
      - the intensity PEAK (max ``INTENSITY_RANK`` across the window),
      - visual variety (distinct panel_kinds + distinct intensities),
    minus a clarity penalty when the union of distinct ``subjects`` exceeds
    ``_CLARITY_SUBJECT_LIMIT``.

    Returns ``{"score": float, "signals": dict}``; an empty window scores 0.
    """
    if not panels:
        return {"score": 0.0, "signals": {}}

    text = " ".join(_panel_text(p) for p in panels)
    stakes = min(len(_STAKES_RE.findall(text)), _SIGNAL_CAP)
    social = min(len(_SOCIAL_RE.findall(text)), _SIGNAL_CAP)
    power = min(len(_POWER_RE.findall(text)), _SIGNAL_CAP)
    enemy = min(len(_ENEMY_RE.findall(text)), _SIGNAL_CAP)

    intensity_peak = max(
        INTENSITY_RANK.get(p.get("intensity"), 0) for p in panels
    )

    distinct_kinds = {p.get("panel_kind") for p in panels}
    distinct_intensities = {p.get("intensity") for p in panels}
    variety = len(distinct_kinds) + len(distinct_intensities)

    subjects: set = set()
    for p in panels:
        for s in (p.get("subjects") or []):
            subjects.add(s)
    clarity_overflow = max(0, len(subjects) - _CLARITY_SUBJECT_LIMIT)

    score = (
        _W_STAKES * stakes
        + _W_SOCIAL * social
        + _W_POWER * power
        + _W_ENEMY * enemy
        + _W_INTENSITY * intensity_peak
        + _W_VARIETY * variety
        - _W_CLARITY_PENALTY * clarity_overflow
    )

    signals = {
        "stakes": stakes,
        "social": social,
        "power": power,
        "enemy": enemy,
        "intensity_peak": intensity_peak,
        "variety": variety,
        "distinct_subjects": len(subjects),
        "clarity_overflow": clarity_overflow,
    }
    return {"score": float(score), "signals": signals}


# --------------------------------------------------------------------------- #
# Task 5: spoiler guard + window enumeration + non-overlapping shortlist
# --------------------------------------------------------------------------- #
def score_windows(
    panels: List[Dict[str, Any]],
    *,
    min_panels: int,
    max_panels: int,
    payoff_tail_frac: float,
    shortlist_n: int,
) -> List[Dict[str, Any]]:
    """Enumerate, score, and shortlist teaser windows over the eligible pool.

    Steps:
      1. ``pool = eligible_panels(panels)``; bail (``[]``) if shorter than
         ``min_panels``.
      2. SPOILER GUARD — exclude indices in the last ``payoff_tail_frac`` of the
         pool plus the single global max-intensity panel (the likely payoff).
      3. Enumerate every contiguous window of size ``min_panels..max_panels``
         that contains no excluded index and score each with ``score_window``.
      4. Greedily pick the top-scoring NON-OVERLAPPING windows (score desc) until
         ``shortlist_n`` are chosen or candidates are exhausted.

    Window dicts use half-open spans: ``{"start": i, "end": j, "panels": [...],
    "score": float, "signals": dict}`` where the slice is ``pool[i:j]``.
    """
    pool = eligible_panels(panels)
    n = len(pool)
    if n < min_panels:
        return []

    # --- spoiler guard: exclude the payoff tail + the single global peak panel
    tail_count = int(math.ceil(n * payoff_tail_frac))
    excluded = set(range(n - tail_count, n))
    peak_idx = max(
        range(n), key=lambda i: INTENSITY_RANK.get(pool[i].get("intensity"), 0)
    )
    excluded.add(peak_idx)

    # --- enumerate candidate windows that dodge every excluded index
    candidates: List[Dict[str, Any]] = []
    for start in range(n):
        for size in range(min_panels, max_panels + 1):
            end = start + size
            if end > n:
                break
            if any(i in excluded for i in range(start, end)):
                continue
            slc = pool[start:end]
            scored = score_window(slc)
            candidates.append({
                "start": start,
                "end": end,
                "panels": slc,
                "score": scored["score"],
                "signals": scored["signals"],
            })

    # --- greedy non-overlapping shortlist (score desc, stable on ties)
    candidates.sort(key=lambda w: w["score"], reverse=True)
    picked: List[Dict[str, Any]] = []
    for w in candidates:
        if len(picked) >= shortlist_n:
            break
        if any(w["start"] < p["end"] and p["start"] < w["end"] for p in picked):
            continue
        picked.append(w)
    return picked


# --------------------------------------------------------------------------- #
# Task 6: Stage-2 select + write narration (one injected model call)
# --------------------------------------------------------------------------- #
def _window_payload(window: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Understood text per panel (scene_file as a BASENAME) for the model call."""
    rows: List[Dict[str, Any]] = []
    for p in window.get("panels") or []:
        rows.append({
            "scene_file": os.path.basename(str(p.get("scene_file") or "")),
            "chapter_number": p.get("chapter_number"),
            "description": str(p.get("description", "") or ""),
            "action": str(p.get("action", "") or ""),
            "dialogue": str(p.get("dialogue", "") or ""),
            "panel_kind": p.get("panel_kind"),
            "intensity": p.get("intensity"),
            "subjects": list(p.get("subjects") or []),
        })
    return rows


def select_and_write(
    windows: List[Dict[str, Any]],
    *,
    loglines: List[str],
    model_call: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    cast_obj: Optional[Dict[str, Any]] = None,
    vision_by_file: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Ask the (injected) model to pick the strongest window + write narration.

    ``model_call(payload) -> dict|None`` follows the ``story_group`` call_fn
    pattern so tests never hit a real LLM. The payload carries the shortlisted
    ``windows`` (understood text per panel, basenames) plus the bundle
    ``loglines`` for context. The model returns ``chosen_index`` +
    spoiler-safe per-panel narration; we assemble the ``manifest.teaser.json``
    dict and run the shared spoiler/fragment post-pass on it.

    Returns the teaser dict, or ``None`` when the model abstains.
    """
    if not windows:
        return None

    payload = {
        "windows": [_window_payload(w) for w in windows],
        "loglines": list(loglines or []),
    }
    resp = model_call(payload)
    if not isinstance(resp, dict) or "chosen_index" not in resp:
        return None

    try:
        idx = int(resp["chosen_index"])
        chosen = windows[idx]
    except (ValueError, TypeError, IndexError):
        return None

    scene_files = [os.path.basename(str(p.get("scene_file") or ""))
                   for p in chosen.get("panels") or []]
    source_chapters = sorted({
        p.get("chapter_number") for p in chosen.get("panels") or []
        if p.get("chapter_number") is not None
    })
    # per-panel narration from the model, scene_file normalized to a basename
    panel_narration: List[Dict[str, Any]] = []
    for row in resp.get("panel_narration") or []:
        panel_narration.append({
            "scene_file": os.path.basename(str(row.get("scene_file") or "")),
            "line": str(row.get("line") or "").strip(),
        })

    teaser = {
        "source_chapters": source_chapters,
        "scene_files": scene_files,
        "panel_narration": panel_narration,
        "reason": str(resp.get("reason") or ""),
        "rewind_line": str(resp.get("rewind_line") or ""),
        "spoiler_boundary": str(resp.get("spoiler_boundary") or ""),
        "scores": {
            "chosen": float(chosen.get("score", 0.0)),
            "shortlist": [float(w.get("score", 0.0)) for w in windows],
        },
    }

    # Spoiler/fragment post-pass — reuse the channel's shared writers. Wrap the
    # teaser as a single beat so the recap_style mutators (which operate on
    # {"beats":[{"panel_narration":[...]}]}) apply in-place; default cast/vision
    # to empty keeps it a safe no-op when no cast/OCR context is supplied.
    beats_obj = {"beats": [{"panel_narration": panel_narration}]}
    recap_style.neutralize_identity_reveal_leaks(
        beats_obj, cast_obj or {}, vision_by_file or {})
    recap_style.repair_spoken_fragments(beats_obj)
    teaser["panel_narration"] = beats_obj["beats"][0].get("panel_narration") or []
    return teaser


# --------------------------------------------------------------------------- #
# Task 7: synthetic-dir builder (manifests + scene symlinks)
# --------------------------------------------------------------------------- #
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def _scenes_entry(src_ep: str, scene_file: str,
                  _cache: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the source ``manifest.scenes.json`` entry whose out_file == basename."""
    man_path = os.path.join(src_ep, "manifest.scenes.json")
    if man_path not in _cache:
        try:
            _cache[man_path] = json.loads(Path(man_path).read_text())
        except Exception:
            _cache[man_path] = {}
    for entry in (_cache[man_path].get("scenes") or []):
        if entry.get("out_file") == scene_file:
            return entry
    return None


def materialize_teaser_dir(
    teaser: Dict[str, Any],
    src_of: Dict[str, str],
    out_dir: Any,
    cast: Dict[str, Any],
) -> Path:
    """Build the synthetic episode dir the render/TTS tools consume.

    Symlinks (fallback copy) each window scene into ``out_dir/scenes/`` and
    writes the four manifests the downstream tools expect plus the teaser dict.
    A scene missing from its source ``manifest.scenes.json`` (or whose source
    dir is unknown) is dropped from the teaser — logged, not fatal — rather than
    emit a broken scenes manifest.
    """
    out_dir = Path(out_dir)
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    cache: Dict[str, Dict[str, Any]] = {}
    kept: List[str] = []
    scene_entries: List[Dict[str, Any]] = []
    for scene_file in teaser.get("scene_files") or []:
        src_ep = src_of.get(scene_file)
        if not src_ep:
            print(f"[teaser] WARN no source dir for {scene_file}; dropping")
            continue
        entry = _scenes_entry(src_ep, scene_file, cache)
        if entry is None:
            print(f"[teaser] WARN {scene_file} not in "
                  f"{src_ep}/manifest.scenes.json; dropping")
            continue
        src_img = Path(src_ep) / "scenes" / scene_file
        dst_img = scenes_dir / scene_file
        if not dst_img.exists():
            try:
                os.symlink(src_img, dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)
        kept.append(scene_file)
        scene_entries.append(entry)

    kept_set = set(kept)
    panel_narration = [
        p for p in (teaser.get("panel_narration") or [])
        if os.path.basename(str(p.get("scene_file") or "")) in kept_set
    ]
    narration = " ".join(
        str(p.get("line") or "").strip() for p in panel_narration
        if str(p.get("line") or "").strip())

    _write_json(out_dir / "manifest.beats.json", {"beats": [{
        "group_id": 1,
        "scene_files": kept,
        "panel_narration": panel_narration,
        "narration": narration,
    }]})
    _write_json(out_dir / "manifest.groups.json", {"shots": [{
        "shot_id": 1,
        "scene_files": kept,
        "segment": "present",
        "arc_label": "teaser",
    }]})
    _write_json(out_dir / "manifest.scenes.json", {"scenes": scene_entries})
    _write_json(out_dir / "manifest.cast.json", cast)
    _write_json(out_dir / "manifest.teaser.json", teaser)
    return out_dir
