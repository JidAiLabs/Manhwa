#!/usr/bin/env python3
"""
tools/ocr_chrome.py

Conservative webtoon / manhwa reader UI-chrome stripper.

The core function ``strip_ui_chrome`` removes high-confidence UI overlay text
(view/comment/like counters, episode navigation, watermarks) from OCR output
while deliberately preserving normal dialogue and narration.

Strategy
--------
* Work line-by-line, then token-by-token inside each surviving line.
* A line is dropped wholesale only when it consists *entirely* of UI tokens
  after we blank those tokens out (i.e. nothing narrative remains).
* UI patterns are designed with word-boundary / context anchors so that
  words like "view" inside a sentence ("a beautiful view") are never touched.
* An optional ``extra_patterns`` argument lets callers add site/scanlator
  watermark strings without modifying this module.

CLI re-cleaner
--------------
    python tools/ocr_chrome.py --manifest <manifest.vision.json>

Loads the manifest, applies ``strip_ui_chrome`` to every ``items[*].ocr_clean``
in-place (no API calls), writes the manifest back, prints how many items changed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# UI-chrome patterns
# ---------------------------------------------------------------------------
# Each pattern is applied case-insensitively to individual *tokens* (words /
# short runs) extracted from a line.  They are deliberately narrow so that
# ordinary sentences are not damaged.
#
# Rule of thumb: a pattern is safe only if ALL strings it could match are
# unambiguously reader-UI, not story text.

# --- Counter patterns -------------------------------------------------------
# Match things like: "VIEWS: 1", "1 VIEW", "1,234 VIEWS", "1 COMMENT", "LIKES 5"
# Require a digit component or bare keyword at a line-level anchor.
_COUNTER_PATTERNS: list[str] = [
    # "VIEWS: 1234"  /  "VIEWS 1,234"  (keyword then optional colon then digits)
    r"VIEWS?\s*:?\s*[\d,]+",
    # "1 VIEW" / "12,345 VIEWS"  (digits then keyword)
    r"[\d,]+\s+VIEWS?",
    # "1 COMMENT" / "12 COMMENTS"
    r"[\d,]+\s+COMMENTS?",
    # "COMMENTS: 1"
    r"COMMENTS?\s*:?\s*[\d,]+",
    # "1 LIKE" / "LIKES: 5" — MUST have adjacent digit so bare "I like you" is safe
    r"[\d,]+\s+LIKES?",
    r"LIKES?\s*:?\s*[\d,]+",
    # "1 SUBSCRIBER" / "1,200 SUBSCRIBERS"
    r"[\d,]+\s+SUBSCRIBERS?",
    r"SUBSCRIBERS?\s*:?\s*[\d,]+",
]

# --- Navigation / episode keywords ----------------------------------------
# These are matched only when the *whole token* (or token + adjacent digit) is
# UI-specific.  They include a word-boundary so "CREATOR" in "my creator" is
# safe.  However note: we apply these as whole-line anchored patterns below.
_NAV_PATTERNS: list[str] = [
    r"EPISODE\s*\d+",          # "EPISODE 12", "EPISODE1"
    r"\bEP\.?\s*\d+\b",        # "EP.12", "EP 5"
    r"CHAPTER\s*\d+",          # "CHAPTER 3"
    r"\bCH\.?\s*\d+\b",        # "CH.3", "CH 3"
    r"\bNEXT\s+EPISODE\b",     # "NEXT EPISODE"
    r"\bUP\s+NEXT\b",          # "UP NEXT"
    r"\bPREV(?:IOUS)?\s+EPISODE\b",  # "PREVIOUS EPISODE"
    # Bare navigation words that are unambiguously UI at line-level
    # (applied only when the whole cleaned line IS just this word)
    r"^NEXT$",
    r"^PREV(?:IOUS)?$",
    r"^CREATOR$",
    r"^AUTHOR$",
    r"^GRADE$",
    r"^SUBSCRIBE$",
    r"^SUBSCRIBED?$",
    r"^SHARE$",
    r"^RATE\s+THIS$",
    r"^RATE$",
    r"^DOWNLOAD$",
    r"AGES\s+\d+\+?",          # "AGES 13+" / "AGES 18"
]

# Compiled once at module load
_ALL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in _COUNTER_PATTERNS + _NAV_PATTERNS
]


def _compile_extra(extra: list[str]) -> list[re.Pattern[str]]:
    """Compile caller-supplied extra patterns (literal strings or regex)."""
    out: list[re.Pattern[str]] = []
    for p in extra:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            # Fall back to literal match if pattern is invalid regex
            out.append(re.compile(re.escape(p), re.IGNORECASE))
    return out


def _scrub_line(line: str, patterns: list[re.Pattern[str]]) -> str:
    """
    Remove UI-chrome tokens from a single line.

    Applies each pattern as a substitution, replacing matches with a single
    space.  Returns the collapsed result (may be empty if the whole line was
    chrome).
    """
    s = line
    for pat in patterns:
        s = pat.sub(" ", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


# Tokens that are pure noise if they survive after chrome stripping:
# lone digits/punctuation left behind by counter removal.
_RESIDUAL_NOISE_RE = re.compile(r"^[\d,.:;|/\\!\-\+\*#@&^%$~`]+$")


def _is_residual_noise(token: str) -> bool:
    """True for lone digit/punctuation tokens that are leftover counter fragments."""
    return bool(_RESIDUAL_NOISE_RE.match(token))


def strip_ui_chrome(
    text: str,
    extra_patterns: Optional[List[str]] = None,
) -> str:
    """
    Remove webtoon reader UI chrome from OCR text conservatively.

    Parameters
    ----------
    text:
        Raw OCR text, possibly multi-line.
    extra_patterns:
        Optional list of additional regex strings (or literal strings) to also
        strip — useful for site/scanlator watermarks like ``"NIGHTSUP.NET"``.

    Returns
    -------
    Cleaned text with UI chrome removed and whitespace collapsed.  Returns the
    original text unchanged when no chrome is detected.

    Notes
    -----
    The function is intentionally conservative:
    - It operates on each line independently.
    - Within a line it erases only the matched spans, keeping whatever narrative
      text remains.
    - A line is dropped entirely only when *nothing narrative* remains after
      chrome removal (empty or pure residual punctuation/digits).
    - The word "view" inside a sentence is NOT matched — counter patterns
      require an adjacent digit or the plural "views" next to a digit.
    """
    if not text:
        return text

    extra_compiled = _compile_extra(extra_patterns) if extra_patterns else []
    all_patterns = _ALL_PATTERNS + extra_compiled

    surviving_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        scrubbed = _scrub_line(line, all_patterns)

        if not scrubbed:
            # Entire line was chrome — drop it
            continue

        # Only clean up residual noise tokens when a pattern actually fired
        # (i.e. the line changed).  If nothing matched, return the line as-is
        # so that lone digits in normal dialogue ("3 survivors") are untouched.
        if scrubbed != line.strip():
            tokens = scrubbed.split()
            narrative_tokens = [t for t in tokens if not _is_residual_noise(t)]
            if not narrative_tokens:
                continue
            scrubbed = " ".join(narrative_tokens)

        surviving_lines.append(scrubbed)

    result = " ".join(surviving_lines)
    result = re.sub(r"\s+", " ", result).strip()
    return result


# ---------------------------------------------------------------------------
# CLI re-cleaner
# ---------------------------------------------------------------------------
def _recleaner_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-apply UI-chrome stripping to an existing manifest.vision.json in-place."
    )
    ap.add_argument(
        "--manifest", required=True, help="Path to manifest.vision.json"
    )
    ap.add_argument(
        "--extra-patterns",
        nargs="*",
        default=[],
        help="Additional regex/literal patterns to strip (e.g. watermark strings)",
    )
    args = ap.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    items: list[dict] = manifest.get("items", [])
    changed = 0

    for item in items:
        original = item.get("ocr_clean", "") or ""
        cleaned = strip_ui_chrome(original, extra_patterns=args.extra_patterns or None)
        if cleaned != original:
            item["ocr_clean"] = cleaned
            changed += 1

    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    total = len(items)
    print(f"[ok] manifest={args.manifest} items={total} changed={changed}")
    return 0


if __name__ == "__main__":
    sys.exit(_recleaner_main())
