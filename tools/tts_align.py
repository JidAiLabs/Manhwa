"""
tools/tts_align.py

Forced-align per-panel narration lines to a single group audio clip,
returning each panel's [start_sec, end_sec] slice within that clip.

Use case: TTS now produces one clip per GROUP (faster, more natural).
This module derives the per-panel visual cut points so the video renderer
can switch to each panel image at the right moment of the continuous audio.

Public API
----------
align_panels(panel_lines, clip_dur_sec, *, transcribe_fn=None,
             clip_path=None, match_threshold=0.6)
    -> list[{"start_sec": float, "end_sec": float, "method": "asr"|"proportional"}]

No heavy deps at import time — faster-whisper is lazy-loaded only when the
default transcriber is actually called without an injected transcribe_fn.
"""

from __future__ import annotations

import difflib
import re
import string
from typing import Callable, Optional

# Module-level singleton for the lazy-loaded Whisper model.
_whisper_model = None

_MIN_PANEL_DUR = 0.3  # seconds — minimum duration for any panel slice


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> list[str]:
    """Lowercase + strip punctuation → list of non-empty word tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w]


# ---------------------------------------------------------------------------
# Default transcriber (lazy faster-whisper, graceful if absent)
# ---------------------------------------------------------------------------

def _default_word_transcribe(clip_path: str) -> list[tuple[str, float, float]]:
    """Transcribe audio to word-level timestamps using faster-whisper.

    Returns a list of (word, start_sec, end_sec) tuples.
    Returns [] if faster-whisper is not installed or transcription fails.
    The WhisperModel is cached in _whisper_model for reuse.
    """
    global _whisper_model

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        return []

    try:
        if _whisper_model is None:
            _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

        segments, _ = _whisper_model.transcribe(
            clip_path,
            word_timestamps=True,
            beam_size=1,
            language="en",
        )
        words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    word_text = w.word.strip()
                    if word_text:
                        words.append((word_text, w.start, w.end))
        return words
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Proportional fallback
# ---------------------------------------------------------------------------

def _proportional_split(
    panel_lines: list[str], clip_dur_sec: float
) -> list[dict]:
    """Split clip_dur_sec proportional to each panel's word count (min 1 word).

    Cumulative boundaries; forces min panel duration; last panel ends at clip_dur_sec.
    """
    word_counts = [max(1, len(_normalise(line))) for line in panel_lines]
    total_words = sum(word_counts)

    boundaries = []
    cumulative = 0.0
    for i, count in enumerate(word_counts):
        fraction = count / total_words
        cumulative += fraction * clip_dur_sec
        boundaries.append(min(cumulative, clip_dur_sec))

    # Force last boundary to exactly clip_dur_sec
    boundaries[-1] = clip_dur_sec

    # Enforce minimum duration by nudging forward
    result = []
    prev_end = 0.0
    for i, end in enumerate(boundaries):
        start = prev_end
        end = max(end, start + _MIN_PANEL_DUR)
        # Don't overshoot clip_dur_sec for non-final panels
        if i < len(boundaries) - 1:
            end = min(end, clip_dur_sec - _MIN_PANEL_DUR * (len(boundaries) - 1 - i))
        else:
            end = clip_dur_sec
        result.append({"start_sec": start, "end_sec": end, "method": "proportional"})
        prev_end = end

    return result


# ---------------------------------------------------------------------------
# ASR alignment
# ---------------------------------------------------------------------------

def _asr_align(
    panel_lines: list[str],
    words: list[tuple[str, float, float]],
    clip_dur_sec: float,
) -> list[dict]:
    """Align panel lines to recognised words via difflib SequenceMatcher.

    Returns per-panel dicts with start_sec, end_sec, method="asr".
    Carries forward the last known end-time for unmatched tail words.
    """
    # Build expected word stream: list of (normalised_word, panel_index)
    expected: list[tuple[str, int]] = []
    for panel_idx, line in enumerate(panel_lines):
        for w in _normalise(line):
            expected.append((w, panel_idx))

    # Build recognised word list (normalised)
    recog_words_norm = [_normalise(w)[0] if _normalise(w) else "" for w, s, e in words]
    expected_words_norm = [w for w, _ in expected]

    # Use SequenceMatcher to align expected ↔ recognised
    sm = difflib.SequenceMatcher(None, expected_words_norm, recog_words_norm, autojunk=False)
    opcodes = sm.get_opcodes()

    # Map each expected word index → recognised word index (or None)
    exp_to_recog: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for offset in range(i2 - i1):
                exp_to_recog[i1 + offset] = j1 + offset

    # For each panel, find the end-time of its last matched expected word
    # panel_end_times[p] = end_sec of last recognised word for panel p
    panel_end_times: dict[int, float] = {}
    last_known_end = 0.0

    for exp_idx, (_, panel_idx) in enumerate(expected):
        if exp_idx in exp_to_recog:
            recog_idx = exp_to_recog[exp_idx]
            _, _, end_t = words[recog_idx]
            last_known_end = end_t
            panel_end_times[panel_idx] = end_t

    # Fill in panels that got no matches using carry-forward
    # Scan forward: if a panel has no matched word, use the next panel's start
    # (determined by first matched word of later panels, or clip_dur)
    # We do a forward pass with carry-forward from the last known time.
    carry = 0.0
    for panel_idx in range(len(panel_lines)):
        if panel_idx in panel_end_times:
            carry = panel_end_times[panel_idx]
        else:
            # Try to find a matched word in a later panel to set a ceiling
            # For now, use carry (will be clamped later)
            panel_end_times[panel_idx] = carry

    # Ensure last panel ends at clip_dur
    panel_end_times[len(panel_lines) - 1] = clip_dur_sec

    # Build result with monotonic enforcement
    result = []
    prev_end = 0.0
    for i in range(len(panel_lines)):
        start = prev_end
        end = max(panel_end_times[i], start + _MIN_PANEL_DUR)
        if i < len(panel_lines) - 1:
            end = min(end, clip_dur_sec - _MIN_PANEL_DUR * (len(panel_lines) - 1 - i))
        else:
            end = clip_dur_sec
        result.append({"start_sec": start, "end_sec": end, "method": "asr"})
        prev_end = end

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_panels(
    panel_lines: list[str],
    clip_dur_sec: float,
    *,
    transcribe_fn: Optional[Callable] = None,
    clip_path: Optional[str] = None,
    match_threshold: float = 0.6,
) -> list[dict]:
    """Return per-panel time slices within a group audio clip.

    Parameters
    ----------
    panel_lines:
        Ordered list of the spoken narration text for each panel in the group.
    clip_dur_sec:
        Total duration of the group audio clip in seconds.
    transcribe_fn:
        Optional injectable word-timestamp transcriber:
            transcribe_fn(clip_path) -> list[(word:str, start:float, end:float)]
        If None, uses _default_word_transcribe (lazy faster-whisper).
        If it raises or returns falsy → proportional fallback.
    clip_path:
        Path to the audio clip — passed to transcribe_fn. May be None if
        transcribe_fn is also None (proportional will be used).
    match_threshold:
        Minimum SequenceMatcher ratio for the ASR path to be trusted.
        If the ratio is below this value, fall back to proportional.
        Default 0.6.

    Returns
    -------
    list of dicts, one per panel line, each with:
        "start_sec": float
        "end_sec":   float
        "method":    "asr" | "proportional"

    Guarantees
    ----------
    - len(result) == len(panel_lines)
    - Boundaries are monotonic non-decreasing and non-overlapping
    - result[0]["start_sec"] == 0.0
    - result[-1]["end_sec"] == clip_dur_sec
    - Every panel has duration >= 0.3s
    """
    if not panel_lines:
        return []

    # Single-panel shortcut
    if len(panel_lines) == 1:
        return [{"start_sec": 0.0, "end_sec": clip_dur_sec, "method": "proportional"}]

    # Resolve transcribe_fn
    if transcribe_fn is None:
        if clip_path is None:
            return _proportional_split(panel_lines, clip_dur_sec)
        transcribe_fn = _default_word_transcribe

    # Attempt ASR transcription
    try:
        words = transcribe_fn(clip_path)
    except Exception:
        return _proportional_split(panel_lines, clip_dur_sec)

    if not words:
        return _proportional_split(panel_lines, clip_dur_sec)

    # Check match quality
    expected_words = []
    for line in panel_lines:
        expected_words.extend(_normalise(line))
    recog_words = [_normalise(w)[0] if _normalise(w) else "" for w, s, e in words]

    sm = difflib.SequenceMatcher(None, expected_words, recog_words, autojunk=False)
    ratio = sm.ratio()

    if ratio < match_threshold:
        return _proportional_split(panel_lines, clip_dur_sec)

    # ASR alignment
    result = _asr_align(panel_lines, words, clip_dur_sec)

    # Safety: if ASR produced non-monotonic result (shouldn't happen but guard it),
    # fall back to proportional
    for i in range(1, len(result)):
        if result[i]["start_sec"] < result[i - 1]["end_sec"] - 1e-9:
            return _proportional_split(panel_lines, clip_dur_sec)

    return result
