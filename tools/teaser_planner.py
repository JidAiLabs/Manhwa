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

import re
from typing import Any, Dict, List

# Panel kinds that can carry the story forward (chrome/empty/etc. are dropped).
_ELIGIBLE_KINDS = {"story", "caption", "system"}

# Ordinal rank of an understood panel's intensity tag.
INTENSITY_RANK = {"calm": 0, "tense": 1, "intense": 2, "explosive": 3}


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
