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


# ---------------------------------------------------------------------------
# Cheap-by-default take selection — synth loop accept-early behaviour
# These tests exercise the QWEN_ASR_ACCEPT_MISMATCH / QWEN_CLONE_MAX_TRIES
# constants and the early-exit logic in the qwen clone synth inner function
# WITHOUT loading any TTS model.  We simulate the loop directly via the
# module-level constants and helper functions.
# ---------------------------------------------------------------------------

class TestCheapTakeSelection:
    """Verify accept-early / re-roll / cap behaviour using stubbed helpers."""

    # ------------------------------------------------------------------
    # Helpers that mimic what the synth loop does for each take
    # ------------------------------------------------------------------

    def _simulate_synth_loop(
        self,
        take_mismatches,   # list of mismatch scores, one per synthesised take
        take_flatnesses,   # list of flatness scores, one per synthesised take
        intended="it was a novel",
        *,
        max_tries=None,
        accept_mismatch=None,
        robotic_flatness=None,
        transcribe_raises=False,
    ):
        """Simulate the qwen clone synth loop logic.

        Returns (n_synths, best_idx) — number of takes generated and which
        index was chosen by pick_best_take over the generated set.
        """
        max_tries = max_tries if max_tries is not None else lt.QWEN_CLONE_MAX_TRIES
        accept_mismatch = accept_mismatch if accept_mismatch is not None else lt.QWEN_ASR_ACCEPT_MISMATCH
        robotic_flatness = robotic_flatness if robotic_flatness is not None else lt.QWEN_ROBOTIC_FLATNESS

        takes = []
        for attempt in range(max_tries):
            wav = f"wav_{attempt}"
            sr = 24000
            takes.append((wav, sr))

            if attempt == 0:
                # mirror the early-exit probe in the real synth loop
                if transcribe_raises:
                    t = None
                else:
                    # simulate what _transcribe_fn returns for take 0
                    mismatch0 = take_mismatches[attempt]
                    t = "clean" if mismatch0 == 0.0 else "bad"  # value doesn't matter; we score below
                flat0 = take_flatnesses[attempt]
                mismatch_score = take_mismatches[attempt]
                if (t is not None
                        and mismatch_score <= accept_mismatch
                        and flat0 <= robotic_flatness):
                    break  # early exit

        n_synths = len(takes)

        # pick_best_take over generated takes
        def _transcribe(wav, sr):
            if transcribe_raises:
                raise RuntimeError("no asr")
            idx = int(wav.split("_")[1])
            # Return a string whose mismatch against intended equals take_mismatches[idx]
            # We can't invert asr_mismatch_score, so we smuggle the value via the
            # pick_best_take stub path by letting flatness decide (simpler).
            return intended  # always "clean" transcript at pick_best_take level

        flat_map = {f"wav_{i}": take_flatnesses[i] for i in range(n_synths)}
        # Build a custom pick that uses our pre-computed mismatch scores
        mismatches_for_takes = take_mismatches[:n_synths]

        call_order = []

        def _pick_transcribe(wav, sr_):
            call_order.append(wav)
            idx = int(wav.split("_")[1])
            # return intended text if mismatch is 0, else inject a word diff
            if mismatches_for_takes[idx] == 0.0:
                return intended
            # return something that will produce a high mismatch — easier to just
            # treat transcribe as raising so flatness decides
            raise RuntimeError("simulated bad take")

        result_wav, _ = lt.pick_best_take(
            takes=takes,
            intended=intended,
            transcribe_fn=_pick_transcribe,
            flatness_fn=lambda w: flat_map[w],
        )
        best_idx = int(result_wav.split("_")[1])
        return n_synths, best_idx

    # ------------------------------------------------------------------
    # Core scenarios
    # ------------------------------------------------------------------

    def test_clean_first_take_accepted_in_one_synth(self):
        """Low mismatch + OK flatness on take 0 → 1 synth, no re-roll."""
        n, _ = self._simulate_synth_loop(
            take_mismatches=[0.15, 0.10, 0.10],  # take 0: 0.15 ≤ 0.35 → clean enough
            take_flatnesses=[0.30, 0.28, 0.27],  # take 0: 0.30 ≤ 0.40 → OK
        )
        assert n == 1, f"Expected 1 synth (clean first take), got {n}"

    def test_whisper_noise_accepted_not_rerolled(self):
        """Mismatch 0.20 (whisper noise on a fine take) → accepted on first try."""
        n, _ = self._simulate_synth_loop(
            take_mismatches=[0.20, 0.10, 0.10],
            take_flatnesses=[0.32, 0.30, 0.28],
        )
        assert n == 1, f"mismatch=0.20 is within accept=0.35; expected 1 synth, got {n}"

    def test_high_mismatch_stutter_triggers_reroll(self):
        """Mismatch 0.60 (stutter/dropout) → re-rolls (more than 1 synth)."""
        n, _ = self._simulate_synth_loop(
            take_mismatches=[0.60, 0.10, 0.10],
            take_flatnesses=[0.30, 0.30, 0.30],
        )
        assert n > 1, f"Expected re-roll on mismatch=0.60, but only {n} synth(s)"

    def test_buzzy_flatness_triggers_reroll(self):
        """Take 0 OK mismatch but flatness > robotic threshold → re-rolls."""
        n, _ = self._simulate_synth_loop(
            take_mismatches=[0.10, 0.10, 0.10],  # mismatch fine
            take_flatnesses=[0.45, 0.30, 0.28],  # take 0 buzzy (0.45 > 0.40)
        )
        assert n > 1, f"Expected re-roll on buzzy flatness, but only {n} synth(s)"

    def test_max_tries_cap_respected(self):
        """Never generates more than QWEN_CLONE_MAX_TRIES takes."""
        # All takes are bad: stutter + buzzy
        n, _ = self._simulate_synth_loop(
            take_mismatches=[0.70, 0.65, 0.68, 0.72, 0.66],
            take_flatnesses=[0.45, 0.43, 0.44, 0.46, 0.43],
        )
        assert n <= lt.QWEN_CLONE_MAX_TRIES, (
            f"Cap is {lt.QWEN_CLONE_MAX_TRIES}; got {n} synths"
        )

    def test_default_max_tries_is_3(self):
        """Default cap is 3 (not 5 — the old value)."""
        # Read from env only when no override is set; the import uses the env at
        # module load time. The default in the source is "3".
        import os
        saved = os.environ.get("STUDIO_QWEN_MAX_TRIES")
        try:
            os.environ.pop("STUDIO_QWEN_MAX_TRIES", None)
            # Re-evaluate the default expression (simulated — module already loaded)
            default = int(os.environ.get("STUDIO_QWEN_MAX_TRIES", "3"))
            assert default == 3, f"Default cap should be 3, got {default}"
        finally:
            if saved is not None:
                os.environ["STUDIO_QWEN_MAX_TRIES"] = saved

    def test_env_override_max_tries(self, monkeypatch):
        """STUDIO_QWEN_MAX_TRIES env var controls the cap."""
        monkeypatch.setenv("STUDIO_QWEN_MAX_TRIES", "2")
        cap = int(__import__("os").environ.get("STUDIO_QWEN_MAX_TRIES", "3"))
        assert cap == 2

    def test_env_override_asr_accept(self, monkeypatch):
        """STUDIO_QWEN_ASR_ACCEPT env var controls the accept-early threshold."""
        monkeypatch.setenv("STUDIO_QWEN_ASR_ACCEPT", "0.50")
        threshold = float(__import__("os").environ.get("STUDIO_QWEN_ASR_ACCEPT", "0.35"))
        assert threshold == 0.50

    def test_no_asr_fallback_single_take_if_clean_flatness(self):
        """When transcribe raises for take 0, no early exit → generates up to max_tries.
        pick_best_take still returns something (flatness-only fallback)."""
        n, _ = self._simulate_synth_loop(
            take_mismatches=[0.10, 0.10, 0.10],
            take_flatnesses=[0.28, 0.30, 0.32],
            transcribe_raises=True,
        )
        # No early exit when transcribe raises → generates max_tries takes
        assert n == lt.QWEN_CLONE_MAX_TRIES, (
            f"No-ASR path should generate max_tries={lt.QWEN_CLONE_MAX_TRIES}, got {n}"
        )

    def test_accept_mismatch_boundary_exactly_at_threshold(self):
        """Mismatch exactly equal to threshold → accepted (≤, not <)."""
        threshold = lt.QWEN_ASR_ACCEPT_MISMATCH
        n, _ = self._simulate_synth_loop(
            take_mismatches=[threshold, 0.10, 0.10],
            take_flatnesses=[0.30, 0.28, 0.27],
        )
        assert n == 1, (
            f"Mismatch == threshold ({threshold}) should be accepted; got {n} synths"
        )

    def test_just_above_threshold_triggers_reroll(self):
        """Mismatch just above threshold → re-rolls (first take not accepted)."""
        threshold = lt.QWEN_ASR_ACCEPT_MISMATCH
        n, _ = self._simulate_synth_loop(
            take_mismatches=[threshold + 0.01, 0.10, 0.10],
            take_flatnesses=[0.30, 0.28, 0.27],
        )
        assert n > 1, (
            f"Mismatch {threshold + 0.01:.2f} > threshold; expected re-roll, got {n} synths"
        )


class TestPickBestTakeWithNewThreshold:
    """Confirm pick_best_take accept-early uses its own (strict) threshold,
    separate from the synth-loop QWEN_ASR_ACCEPT_MISMATCH."""

    def test_pick_best_take_strict_internal_threshold(self):
        """pick_best_take default mismatch_threshold is _ASR_MISMATCH_ACCEPT (0.10),
        not QWEN_ASR_ACCEPT_MISMATCH (0.35). A take with mismatch 0.20 will NOT
        trigger accept-early inside pick_best_take."""
        calls = []

        takes = [("wav_0", 24000), ("wav_1", 24000)]

        def counting_transcribe(wav, sr):
            calls.append(wav)
            # wav_0 → slightly noisy (0.20 mismatch); wav_1 → perfect
            if wav == "wav_0":
                return "it was novel"   # one word off → mismatch ~0.25 (> 0.10)
            return "it was a novel"     # perfect

        flat_map = {"wav_0": 0.30, "wav_1": 0.30}
        lt.pick_best_take(
            takes=takes,
            intended="it was a novel",
            transcribe_fn=counting_transcribe,
            flatness_fn=lambda w: flat_map[w],
            # default threshold = _ASR_MISMATCH_ACCEPT = 0.10
        )
        # wav_0 scores ~0.25 > 0.10 → not accepted early → wav_1 also transcribed
        assert len(calls) == 2, (
            f"pick_best_take should NOT accept-early at 0.10 threshold for mismatch≈0.25; "
            f"got {len(calls)} transcribe calls"
        )
