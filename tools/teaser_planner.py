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

import math
import re
from typing import Any, Dict, List

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
