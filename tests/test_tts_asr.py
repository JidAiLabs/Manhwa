"""
tests/test_tts_asr.py

TDD for ASR-verified take selection in tools/local_tts_from_manifest.py.

Covers:
  - asr_mismatch_score: pure scorer, no model needed
  - pick_best_take: selection logic with stubbed transcribe_fn + flatness_fn
  - fallback when transcribe_fn raises or returns None → flatness-only, no crash
  - graceful import absence (faster_whisper not installed)

No heavy model deps required; everything is stubbed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, List, Optional

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "local_tts",
    Path(__file__).resolve().parent.parent / "tools" / "local_tts_from_manifest.py",
)
lt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lt)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# asr_mismatch_score — pure scorer
# ---------------------------------------------------------------------------

class TestAsrMismatchScore:
    """Unit tests for the pure scoring function — no model, no IO."""

    def test_identical_returns_zero(self):
        score = lt.asr_mismatch_score("it was a novel", "it was a novel")
        assert score == 0.0

    def test_identical_after_case_normalisation(self):
        score = lt.asr_mismatch_score("The Hero Rises", "the hero rises")
        assert score == 0.0

    def test_stutter_gives_high_penalty(self):
        """Repeated token run in heard not in intended → high score."""
        score = lt.asr_mismatch_score(
            intended="it was a novel",
            heard="it it it it was a novel",
        )
        assert score > 0.4, f"Expected stutter penalty > 0.4, got {score:.3f}"

    def test_stutter_higher_than_substitution(self):
        """Stutter on 3 repeats should score worse than a single word swap."""
        stutter = lt.asr_mismatch_score("it was a novel", "it it it was a novel")
        sub = lt.asr_mismatch_score("it was a novel", "it is a novel")
        assert stutter > sub, (
            f"Stutter ({stutter:.3f}) should outscore substitution ({sub:.3f})"
        )

    def test_filler_gives_penalty(self):
        """Short non-word filler inserted into heard → non-zero score."""
        score = lt.asr_mismatch_score(
            intended="the man",
            heard="the ah man",
        )
        assert score > 0.0, f"Expected filler penalty > 0, got {score:.3f}"

    def test_filler_variants(self):
        """Common filler vocalisations all trigger a penalty."""
        base = "she stood still"
        fillers = ["uh", "um", "er", "hmm", "mm", "oh", "ah", "aouh"]
        for f in fillers:
            heard = f"she {f} stood still"
            s = lt.asr_mismatch_score(base, heard)
            assert s > 0.0, f"Filler '{f}' should give penalty > 0, got {s:.3f}"

    def test_single_substitution_moderate(self):
        """One-word swap → moderate WER, not zero, not sky-high."""
        score = lt.asr_mismatch_score("the blade falls", "the blade lands")
        assert 0.0 < score < 0.8, f"One-word substitution should be moderate, got {score:.3f}"

    def test_empty_intended_does_not_crash(self):
        score = lt.asr_mismatch_score("", "aouh")
        assert isinstance(score, float)

    def test_empty_both_returns_zero(self):
        assert lt.asr_mismatch_score("", "") == 0.0

    def test_punctuation_stripped_before_comparison(self):
        """Punctuation differences alone should not score as errors."""
        score = lt.asr_mismatch_score(
            "He runs, fast.",
            "he runs fast",
        )
        assert score == 0.0, f"Punctuation-only diff should score 0, got {score:.3f}"

    def test_aouh_filler_penalty(self):
        """The specific reported artifact 'aouh' is penalised."""
        score = lt.asr_mismatch_score("blood runs cold", "aouh blood runs cold")
        assert score > 0.0


# ---------------------------------------------------------------------------
# pick_best_take — selection logic (everything stubbed)
# ---------------------------------------------------------------------------

def _make_takes(wavs_srs: List[tuple]) -> list:
    """Build fake take tuples (wav, sr) as pick_best_take expects."""
    return list(wavs_srs)


class TestPickBestTake:
    """Tests for the refactored take-selection helper — stubbed transcribe + flatness."""

    def _run(self, transcripts, flatnesses, intended="it was a novel",
             mismatch_threshold=None, flatness_threshold=None):
        """Helper: build fake takes and invoke pick_best_take with stubs."""
        takes = [(f"wav_{i}", 24000) for i in range(len(transcripts))]
        transcript_iter = iter(transcripts)

        def stub_transcribe(wav, sr):
            return next(transcript_iter)

        def stub_flatness(wav):
            idx = takes.index((wav, sr) if False else (wav, sr))
            return flatnesses[idx]

        # flatness_fn receives the wav (first element of tuple)
        flat_map = {f"wav_{i}": flatnesses[i] for i in range(len(flatnesses))}

        def stub_flat(wav):
            return flat_map[wav]

        kwargs = {}
        if mismatch_threshold is not None:
            kwargs["mismatch_threshold"] = mismatch_threshold
        if flatness_threshold is not None:
            kwargs["flatness_threshold"] = flatness_threshold

        return lt.pick_best_take(
            takes=takes,
            intended=intended,
            transcribe_fn=stub_transcribe,
            flatness_fn=stub_flat,
            **kwargs,
        )

    def test_picks_clean_take_over_stutter(self):
        """3 takes: stutter, clean, filler → picks the clean one (index 1)."""
        transcripts = [
            "it it it it was a novel",   # stutter
            "it was a novel",            # clean
            "it uh was a novel",         # filler
        ]
        flatnesses = [0.30, 0.28, 0.31]
        result = self._run(transcripts, flatnesses)
        wav, sr = result
        assert wav == "wav_1", f"Expected clean take (wav_1), got {wav}"

    def test_picks_clean_take_by_mismatch_not_flatness(self):
        """Even if clean take has higher flatness, mismatch is PRIMARY."""
        transcripts = [
            "it it it was a novel",  # stutter → bad mismatch
            "it was a novel",        # clean → zero mismatch
        ]
        flatnesses = [0.20, 0.45]   # stutter has LOWER (better) flatness
        result = self._run(transcripts, flatnesses)
        wav, sr = result
        assert wav == "wav_1", "Mismatch must dominate over flatness"

    def test_if_all_bad_picks_least_bad(self):
        """When all takes stutter, still returns something (never nothing)."""
        transcripts = [
            "it it it was a novel",
            "it it was a novel",
            "it it it it was a novel",
        ]
        flatnesses = [0.35, 0.33, 0.38]
        result = self._run(transcripts, flatnesses)
        assert result is not None
        wav, _ = result
        # least bad stutter is index 1 (shorter run)
        assert wav == "wav_1"

    def test_accepts_early_on_clean_take(self):
        """If first take is clean and flat, stops without calling transcribe again."""
        calls = []

        def counting_transcribe(wav, sr):
            calls.append(wav)
            return "it was a novel"  # always clean

        takes = [("wav_0", 24000), ("wav_1", 24000), ("wav_2", 24000)]
        flat_map = {"wav_0": 0.25, "wav_1": 0.25, "wav_2": 0.25}
        lt.pick_best_take(
            takes=takes,
            intended="it was a novel",
            transcribe_fn=counting_transcribe,
            flatness_fn=lambda w: flat_map[w],
        )
        assert len(calls) == 1, (
            f"Should accept-early after first clean take; got {len(calls)} calls"
        )

    def test_fallback_when_transcribe_raises(self):
        """If transcribe_fn raises for every take, falls back to flatness-only."""
        def crashing_transcribe(wav, sr):
            raise RuntimeError("no asr model")

        takes = [("wav_0", 24000), ("wav_1", 24000)]
        flat_map = {"wav_0": 0.45, "wav_1": 0.28}
        result = lt.pick_best_take(
            takes=takes,
            intended="it was a novel",
            transcribe_fn=crashing_transcribe,
            flatness_fn=lambda w: flat_map[w],
        )
        wav, _ = result
        assert wav == "wav_1", "Flatness fallback should pick lowest flatness"

    def test_fallback_when_transcribe_returns_none(self):
        """If transcribe_fn returns None, falls back to flatness-only gracefully."""
        takes = [("wav_0", 24000), ("wav_1", 24000)]
        flat_map = {"wav_0": 0.50, "wav_1": 0.22}
        result = lt.pick_best_take(
            takes=takes,
            intended="any text",
            transcribe_fn=lambda w, sr: None,
            flatness_fn=lambda w: flat_map[w],
        )
        wav, _ = result
        assert wav == "wav_1"

    def test_single_take_always_returned(self):
        """Even a single bad take must be returned (never write nothing)."""
        result = lt.pick_best_take(
            takes=[("wav_0", 24000)],
            intended="hello world",
            transcribe_fn=lambda w, sr: "ah uh hello world",
            flatness_fn=lambda w: 0.55,
        )
        assert result is not None
        wav, _ = result
        assert wav == "wav_0"


# ---------------------------------------------------------------------------
# _get_asr — lazy loader + graceful absence
# ---------------------------------------------------------------------------

class TestGetAsr:
    def test_get_asr_returns_callable_or_none_without_crash(self, monkeypatch):
        """_get_asr must not crash even if faster_whisper is absent.

        We monkeypatch import to simulate absence, then call _get_asr.
        It should return None (no model) and log a warning instead of raising.
        """
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("no module faster_whisper")
            return real_import(name, *args, **kwargs)

        # Reset cached ASR singleton so the lazy loader re-runs
        lt._asr_model = None

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = lt._get_asr()
        assert result is None, f"Expected None when faster_whisper absent, got {result!r}"

    def test_get_asr_is_cached(self, monkeypatch):
        """Second call to _get_asr returns the same object (singleton)."""
        sentinel = object()
        lt._asr_model = sentinel
        result = lt._get_asr()
        assert result is sentinel
        lt._asr_model = None  # cleanup
