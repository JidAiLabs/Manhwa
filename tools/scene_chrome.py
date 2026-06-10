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
_COUNTER_RE = re.compile(r"\b(views?|likes?|subscribers?|comments?)\s*[:.]?\s*[\d,.]+", re.I)

# Publication credits and platform names.
_CREDITS_RE = re.compile(
    r"\b(studio|publisher|published\s+by|presented\s+by|author|art\s+by|story\s+by|"
    r"adaptation|translat\w*|proofread\w*|typeset\w*|scanlat\w*|copyright|"
    r"all\s+rights|webtoon|naver|kakao|redice)\b|©",
    re.I,
)

# A bare chapter/episode marker (the whole text is just the marker).
_MARKER_RE = re.compile(
    r"^\s*(?:chapter|episode|ep\.?|prologue|final(?:e)?|season\s*\d+)?[\s:#-]*\d{0,4}\s*$",
    re.I,
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm_words(s: str) -> list:
    return _WORD_RE.findall((s or "").lower())


def is_chrome_scene(
    item: Dict[str, Any],
    *,
    series_title: Optional[str] = None,
) -> bool:
    """True when a vision item's scene is publication chrome, not story.

    Rules (any hit = chrome):
      1. UI counters dominate the text (VIEWS/LIKES + numbers).
      2. Credits/publisher/platform vocabulary present.
      3. The OCR is nothing but a chapter/episode/number marker.
      4. The OCR is dominated by the series title (cover/title page) —
         title words must make up most of the text, so dialogue that merely
         reuses title words stays story.
    """
    ocr = str(item.get("ocr_clean") or "").strip()
    if not ocr:
        return False  # pure art — never chrome by text rules

    if _COUNTER_RE.search(ocr):
        # counters are chrome when they ARE the content, not one mention in
        # a long dialogue line
        counter_hits = len(_COUNTER_RE.findall(ocr))
        words = _norm_words(ocr)
        if counter_hits * 2 >= max(1, len(words)) / 2 or len(words) <= 6:
            return True

    if _CREDITS_RE.search(ocr):
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

    return False
