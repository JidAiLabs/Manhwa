#!/usr/bin/env python3
"""
narration_consistency.py — deterministic audio↔narration drift detection.

The voiced clips are produced once from the narration of the moment. When the
beats/script are regenerated, the spoken audio no longer matches the plan's
narration — yet the old clips keep shipping (the voiced stage used to cache on
file existence alone). This module gives a $0, LLM-free fingerprint so the
mismatch is caught deterministically: same text → keep the clip; changed text →
re-voice it (use the new one).

Comparison is on the SPOKEN content: a leading mood/delivery tag (``[excited]``)
is stripped (TTS strips it before synthesis and the planner prefixes it to
``tts_text``), whitespace is collapsed, and case is folded — so the clip's
source text and the plan's ``tts_text`` compare apples-to-apples.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

# leading run of [mood]/[delivery] bracket tags, e.g. "[excited] [fast] "
_LEADING_TAGS = re.compile(r"^\s*(?:\[[^\]]*\]\s*)+")


def normalize_narration(text: Optional[str]) -> str:
    """Canonical spoken-content form: leading bracket tags removed, whitespace
    collapsed, casefolded. Two lines that would be SPOKEN identically normalize
    to the same string."""
    t = _LEADING_TAGS.sub("", text or "")
    return re.sub(r"\s+", " ", t).strip().casefold()


def narration_sha(text: Optional[str]) -> str:
    """Stable fingerprint of the spoken content of *text*."""
    return hashlib.sha256(normalize_narration(text).encode("utf-8")).hexdigest()


def _clip_sha(clip: Dict[str, Any]) -> Optional[str]:
    """The fingerprint a clip was voiced from. Prefers the stored ``text_sha``;
    falls back to hashing the stored source text for pre-upgrade indexes; None
    when the clip records no source text at all (must be treated as stale)."""
    sha = clip.get("text_sha")
    if sha:
        return str(sha)
    for key in ("source_text", "sent_text"):
        if clip.get(key) is not None:
            return narration_sha(str(clip.get(key)))
    return None


def audio_consistency(plan_obj: Dict[str, Any],
                      index_obj: Dict[str, Any]) -> Dict[str, List[str]]:
    """Compare a render plan's per-segment narration against the voiced clips.

    Returns segment_id lists: ``fresh`` (audio matches narration), ``stale``
    (audio was voiced from different text), ``missing`` (narration with no clip).
    Branding and silent/hold segments (no ``tts_text``) are ignored — holds
    carry no narration of their own.
    """
    clips = {c.get("segment_id"): c for c in (index_obj.get("clips") or [])
             if isinstance(c, dict)}
    fresh: List[str] = []
    stale: List[str] = []
    missing: List[str] = []
    for it in (plan_obj.get("timeline") or []):
        if it.get("branding"):
            continue
        seg = it.get("segment_id")
        text = it.get("tts_text") or ""
        if not text.strip():
            continue                       # silent/hold — nothing to voice
        clip = clips.get(seg)
        if clip is None:
            missing.append(str(seg))
            continue
        (fresh if _clip_sha(clip) == narration_sha(text) else stale).append(str(seg))
    return {"fresh": fresh, "stale": stale, "missing": missing}


def is_voiced_current(plan_obj: Dict[str, Any],
                      index_obj: Dict[str, Any]) -> bool:
    """True when every narrated segment has audio voiced from its current text."""
    r = audio_consistency(plan_obj, index_obj)
    return not r["stale"] and not r["missing"]
