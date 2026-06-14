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

# ALL [mood]/[delivery] bracket tags, anywhere in the line — TTS strips every
# one of them before synthesis (strip_bracket_tags), so the spoken content must
# ignore them wherever they sit, not just at the start.
_BRACKET_TAGS = re.compile(r"\[[^\]]*\]")


def normalize_narration(text: Optional[str]) -> str:
    """Canonical spoken-content form: every bracket tag removed, whitespace
    collapsed, casefolded. Two lines that would be SPOKEN identically normalize
    to the same string."""
    t = _BRACKET_TAGS.sub(" ", text or "")
    return re.sub(r"\s+", " ", t).strip().casefold()


def narration_sha(text: Optional[str]) -> str:
    """Stable fingerprint of the spoken content of *text*."""
    return hashlib.sha256(normalize_narration(text).encode("utf-8")).hexdigest()


# --- series-intro / title-card "chrome" scrub -------------------------------
# The beats writer invents opening chrome ("Welcome to the world of <Title>.",
# "The chapter begins with a title card for <Title>.") — AI slop AND the one
# place the licensed series title leaks into VOICED narration. We drop any
# SENTENCE that reads as this meta framing, anywhere in the line. Title-AGNOSTIC
# (keys on the framing, not the title) so legitimate story nouns survive (e.g.
# "Nano Machine" the in-story device). Applied at the BEATS SOURCE so script,
# plan and audio all inherit the same clean narration — no cross-stage desync.
_CHROME_SENTENCE_RE = re.compile(
    r"(?:\bwelcome to\b|\bstep into\b|\benter the\b|\bdive into\b|\bventure into\b|"
    r"\bprepare to (?:enter|dive|witness)\b|\bthis is (?:the )?(?:story|tale|world|saga) of\b|"
    r"\bget ready for\b|\blet me (?:introduce|tell you about)\b|\bjoin us (?:in|as)\b|"
    r"\bin the world of\b|\bthe (?:chapter|episode|story|series|tale) (?:begins|opens|starts|kicks off)\b|"
    r"\btitle card\b|\bopening (?:panel|shot|scene|card)\b|"
    # meta-narration ABOUT the format instead of the story (AI slop): the
    # narrator commenting on the recap/presentation rather than narrating it.
    r"\bwe(?:'re| are)? (?:presented|shown|introduced|treated)\b|"
    r"\bbefore the (?:story|tale|episode|chapter|recap) (?:unfolds|begins|starts|opens|kicks off)\b|"
    r"\bmeta[- ]commentary\b|"
    r"\bour (?:true )?(?:adventure|journey|tale|story|recap) (?:is about to|begins|opens|starts|unfolds|awaits|kicks off)\b)",
    re.IGNORECASE)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Licensed series titles leak as a trailing clause inside an OTHERWISE-good
# sentence (e.g. "...a distant light, under the title 'Omniscient Reader'.") —
# strip just the clause, keep the rest. Title-AGNOSTIC: keys on the framing
# ("(under) the title 'X'" / "titled 'X'"), so any quoted title is removed.
_TITLE_CLAUSE_RE = re.compile(
    r"[,;:]?\s*(?:under|bearing|beneath|below|with)?\s*(?:the\s+title|titled|entitled)"
    r"\s*[:,]?\s*['\"‘“][^'\"’”]+['\"’”]",
    re.IGNORECASE)
_DANGLE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")


def strip_chrome_opener(text: Optional[str]) -> str:
    """Drop sentences that read as series-intro / title-card / meta chrome and
    strip embedded 'the title \"X\"' clauses (the licensed-name leak); keep the
    rest of the line intact. Series-title-agnostic (spares story nouns)."""
    t = _TITLE_CLAUSE_RE.sub("", (text or "").strip())
    sents = _SENTENCE_SPLIT_RE.split(t)
    kept = [s for s in sents if s.strip() and not _CHROME_SENTENCE_RE.search(s)]
    out = " ".join(kept).strip()
    return _DANGLE_PUNCT_RE.sub(r"\1", out)        # tidy " ." left by clause cut


def _clip_sha(clip: Dict[str, Any]) -> Optional[str]:
    """The fingerprint a clip was actually voiced from — ONLY the stored
    ``text_sha`` is trustworthy. We deliberately do NOT fall back to hashing a
    stored ``source_text``/``sent_text``: a producer that rewrites that field
    without re-synthesizing (e.g. a file-existence cache) would then look fresh
    while shipping stale audio. No text_sha → unknown → caller treats as stale,
    forcing a one-time re-voice that backfills the sha (self-healing migration)."""
    sha = clip.get("text_sha")
    return str(sha) if sha else None


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
