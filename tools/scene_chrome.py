#!/usr/bin/env python3
"""
scene_chrome.py — deterministic detection of chapter CHROME scenes.

Chrome = pages that belong to the publication, not the story: publisher and
studio logo pages, series cover/title pages, chapter-number cards, app UI
(view counters), author/translator credits. Chrome must never be grouped,
narrated, or shown — the ORV/IE openings narrated "presented by Redice
Studio" and "VIEWS: 1" because nothing modeled this concept.

Used by scene_group_builder (exclude before any LLM sees them) and
render_prep (belt-and-suspenders cut gate for already-processed chapters).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# UI counters / app chrome — "VIEWS: 1", "LIKES 203", subscriber counts.
# The value class includes I/l/| /O: Vision routinely OCRs digits as letters
# ("VIEWS: I" is the real ORV panel that slipped the first filter pass).
_COUNTER_RE = re.compile(r"\b(views?|likes?|subscribers?|comments?)\s*[:.]?\s*[\dIlO|,.]+", re.I)

# Publication credits and platform names.
_CREDITS_RE = re.compile(
    r"\b(studio|publisher|published\s+by|presented\s+by|author|art\s+by|story\s+by|"
    r"adaptation|translat\w*|proofread\w*|typeset\w*|scanlat\w*|copyright|"
    r"all\s+rights|webtoon|naver|kakao|redice)\b|©",
    re.I,
)

# Site plugs and scanlation-team credits (aggregator stamps on covers).
# Domains and team-credit tags are chrome no matter how wordy the page
# (the IE cover OCRs ~58 words around ELFTOON.COM). OCR often splits the
# domain ("ELFTOON .com" on the IE end card) — tolerate spaces around the
# dot. Bare plug PHRASES are chrome only on short banners: story dialogue
# legitimately says "read this" (the ORV novel-app panel).
_SITE_HARD_RE = re.compile(
    r"\b\w[\w-]*\s?\.\s?(com|net|org|io|to|gg)\b|\b(ed|tl|pr|qc|clrd|rd)\s*:",
    re.I,
)
_SITE_PLUG_RE = re.compile(
    r"please\s+read|read\s+(this|free)|thanks\s+for\s+reading|"
    r"join\s+our\s+discord|our\s+(web\s?site|discord)|discord\s+server",
    re.I,
)
_SITE_PLUG_MAX_WORDS = 12

_MARKER_RE = re.compile(
    r"^\s*(?:chapter|episode|ep\.?|prologue|final(?:e)?|season\s*\d+)?[\s:#-]*\d{0,4}\s*$",
    re.I,
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm_words(s: str) -> list:
    return _WORD_RE.findall((s or "").lower())


def needs_image_stats(ocr: str) -> bool:
    """Callers should compute midtone_frac for these OCR signatures: empty
    OCR (stylized number cards) or a site hit that may be a mere watermark."""
    ocr = (ocr or "").strip()
    return not ocr or bool(_SITE_HARD_RE.search(ocr))


def is_chrome_scene(
    item: Dict[str, Any],
    *,
    series_title: Optional[str] = None,
    midtone_frac: Optional[float] = None,
) -> bool:
    """True when a vision item's scene is publication chrome, not story.

    Rules (any hit = chrome):
      1. UI counters dominate the text (VIEWS/LIKES + numbers, including
         OCR digit-confusions like 'VIEWS: I').
      2. Credits/publisher/platform vocabulary present.
      3. The OCR is nothing but a chapter/episode/number marker.
      4. The OCR is dominated by the series title (cover/title page).
      5. OCR-BLIND chrome (stylized number cards): empty OCR + a near-binary
         pixel profile (*midtone_frac*, supplied by callers with image
         access) — real art always has midtones.
    """
    ocr = str(item.get("ocr_clean") or "").strip()
    if not ocr:
        # pure art — chrome only when image stats prove a binary card
        return midtone_frac is not None and midtone_frac < 0.08

    if _COUNTER_RE.search(ocr):
        # counters are chrome when they ARE the content, not one mention in
        # a long dialogue line
        counter_hits = len(_COUNTER_RE.findall(ocr))
        words = _norm_words(ocr)
        if counter_hits * 2 >= max(1, len(words)) / 2 or len(words) <= 6:
            return True

    if _CREDITS_RE.search(ocr):
        return True

    hard_hits = sum(1 for _ in _SITE_HARD_RE.finditer(ocr))
    plug = bool(_SITE_PLUG_RE.search(ocr))
    if hard_hits >= 2:
        return True  # domains + team-credit tags pile up on real covers
    if hard_hits == 1:
        if plug:
            return True  # domain + "thanks for reading"/"join our" = end card
        # ONE domain amid real dialogue is an aggregator watermark stamped ON
        # story art (IE p000039) — chrome only when the panel is otherwise
        # text-sparse AND image stats don't prove real art
        if (len(_norm_words(ocr)) <= _SITE_PLUG_MAX_WORDS
                and not (midtone_frac is not None and midtone_frac >= 0.15)):
            return True

    if plug and len(_norm_words(ocr)) <= _SITE_PLUG_MAX_WORDS:
        return True

    if _MARKER_RE.match(ocr):
        return True

    if series_title:
        title_words = set(_norm_words(series_title))
        words = _norm_words(ocr)
        if title_words and words:
            in_title = sum(1 for w in words if w in title_words)
            if in_title / len(words) >= 0.6 and len(words) <= len(title_words) + 3:
                return True
            # vertical/stylized covers OCR as garbage tokens around the one
            # distinctive title word ("SR OMNISCIENT CE IA ED NEO TR")
            big = [w for w in title_words if len(w) >= 8]
            if big and len(words) <= 10 and any(w in words for w in big):
                return True

    return False
