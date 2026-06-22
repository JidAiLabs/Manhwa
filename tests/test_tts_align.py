"""
tests/test_tts_align.py

TDD for tools/tts_align.py — forced-align panel cuts within a group audio clip.

Given a group's single audio clip + ordered per-panel narration lines,
return each panel's [start_sec, end_sec] slice so the video can cut
to each panel at the right moment.

All tests use a STUBBED transcribe_fn — no real model is ever loaded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module-level singleton (standard pattern for this repo)
# ---------------------------------------------------------------------------
_SPEC = importlib.util.spec_from_file_location(
    "tts_align",
    Path(__file__).resolve().parent.parent / "tools" / "tts_align.py",
)
ta = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ta)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _words(*tokens: tuple) -> list:
    """Build stub word list: each tuple is (word, start, end)."""
    return [(w, s, e) for w, s, e in tokens]


def _check_invariants(result, panel_lines, clip_dur_sec, min_dur=0.3):
    """Assert all required guarantees hold on a result."""
    assert len(result) == len(panel_lines), (
        f"len mismatch: got {len(result)}, expected {len(panel_lines)}"
    )
    assert result[0]["start_sec"] == pytest.approx(0.0), "first start must be 0.0"
    assert result[-1]["end_sec"] == pytest.approx(clip_dur_sec), (
        f"last end must be clip_dur={clip_dur_sec}, got {result[-1]['end_sec']}"
    )
    for i, seg in enumerate(result):
        assert seg["start_sec"] >= 0.0, f"[{i}] start_sec < 0"
        assert seg["end_sec"] <= clip_dur_sec + 1e-9, f"[{i}] end_sec > clip_dur"
        assert seg["end_sec"] - seg["start_sec"] >= min_dur - 1e-9, (
            f"[{i}] duration {seg['end_sec'] - seg['start_sec']:.3f} < min_dur {min_dur}"
        )
        assert "method" in seg, f"[{i}] missing 'method' key"
        assert seg["method"] in ("asr", "proportional"), (
            f"[{i}] unknown method {seg['method']!r}"
        )
    # monotonic non-decreasing
    for i in range(len(result) - 1):
        assert result[i]["end_sec"] <= result[i + 1]["start_sec"] + 1e-9, (
            f"[{i}] end={result[i]['end_sec']:.3f} > [{i+1}] start={result[i+1]['start_sec']:.3f}"
        )
    # non-overlapping: each start == previous end
    for i in range(1, len(result)):
        assert result[i]["start_sec"] == pytest.approx(result[i - 1]["end_sec"]), (
            f"[{i}] gap or overlap: start={result[i]['start_sec']:.3f} != "
            f"prev_end={result[i - 1]['end_sec']:.3f}"
        )


# ---------------------------------------------------------------------------
# Clean ASR case
# ---------------------------------------------------------------------------

class TestCleanAsr:
    """Two panels, stub returns 6 words with known times → split on word boundary."""

    def _stub(self, clip_path):
        # "the cat sat" at 0.0–1.5, "the dog ran" at 1.5–3.0
        return _words(
            ("the",  0.0, 0.5),
            ("cat",  0.5, 1.0),
            ("sat",  1.0, 1.5),
            ("the",  1.5, 2.0),
            ("dog",  2.0, 2.5),
            ("ran",  2.5, 3.0),
        )

    def test_split_between_panels(self):
        panel_lines = ["the cat sat", "the dog ran"]
        result = ta.align_panels(panel_lines, 3.0, transcribe_fn=self._stub, clip_path="fake.wav")
        assert len(result) == 2
        assert result[0]["method"] == "asr"
        assert result[1]["method"] == "asr"
        # Panel 0 ends at word 3's end (1.5), panel 1 starts there
        assert result[0]["end_sec"] == pytest.approx(1.5)
        assert result[1]["start_sec"] == pytest.approx(1.5)

    def test_starts_at_zero(self):
        result = ta.align_panels(["the cat sat", "the dog ran"], 3.0,
                                 transcribe_fn=self._stub, clip_path="fake.wav")
        assert result[0]["start_sec"] == pytest.approx(0.0)

    def test_ends_at_clip_dur(self):
        result = ta.align_panels(["the cat sat", "the dog ran"], 3.0,
                                 transcribe_fn=self._stub, clip_path="fake.wav")
        assert result[-1]["end_sec"] == pytest.approx(3.0)

    def test_monotonic_non_overlapping(self):
        result = ta.align_panels(["the cat sat", "the dog ran"], 3.0,
                                 transcribe_fn=self._stub, clip_path="fake.wav")
        _check_invariants(result, ["the cat sat", "the dog ran"], 3.0)

    def test_method_is_asr(self):
        result = ta.align_panels(["the cat sat", "the dog ran"], 3.0,
                                 transcribe_fn=self._stub, clip_path="fake.wav")
        for seg in result:
            assert seg["method"] == "asr"


# ---------------------------------------------------------------------------
# Low-match → proportional fallback
# ---------------------------------------------------------------------------

class TestLowMatchFallback:
    """Stub returns garbage words with no overlap → proportional fallback."""

    def _garbage_stub(self, clip_path):
        return _words(
            ("zzz", 0.0, 1.0),
            ("qqq", 1.0, 2.0),
            ("xkcd", 2.0, 3.0),
        )

    def test_method_is_proportional(self):
        result = ta.align_panels(["the cat sat", "the dog ran"], 3.0,
                                 transcribe_fn=self._garbage_stub, clip_path="fake.wav")
        for seg in result:
            assert seg["method"] == "proportional"

    def test_invariants_hold(self):
        result = ta.align_panels(["the cat sat", "the dog ran"], 3.0,
                                 transcribe_fn=self._garbage_stub, clip_path="fake.wav")
        _check_invariants(result, ["the cat sat", "the dog ran"], 3.0)


# ---------------------------------------------------------------------------
# Empty transcription → proportional
# ---------------------------------------------------------------------------

class TestEmptyTranscription:
    """transcribe_fn returns [] → proportional fallback."""

    def test_empty_returns_proportional(self):
        result = ta.align_panels(["hello world", "foo bar baz"], 6.0,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        for seg in result:
            assert seg["method"] == "proportional"

    def test_invariants_hold(self):
        result = ta.align_panels(["hello world", "foo bar baz"], 6.0,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        _check_invariants(result, ["hello world", "foo bar baz"], 6.0)


# ---------------------------------------------------------------------------
# Transcribe raises → proportional, no crash
# ---------------------------------------------------------------------------

class TestTranscribeRaises:
    def _crashing(self, clip_path):
        raise RuntimeError("no faster-whisper installed")

    def test_no_crash(self):
        result = ta.align_panels(["line one", "line two"], 4.0,
                                 transcribe_fn=self._crashing, clip_path="fake.wav")
        assert result is not None
        assert len(result) == 2

    def test_method_is_proportional(self):
        result = ta.align_panels(["line one", "line two"], 4.0,
                                 transcribe_fn=self._crashing, clip_path="fake.wav")
        for seg in result:
            assert seg["method"] == "proportional"

    def test_invariants_hold(self):
        result = ta.align_panels(["line one", "line two"], 4.0,
                                 transcribe_fn=self._crashing, clip_path="fake.wav")
        _check_invariants(result, ["line one", "line two"], 4.0)


# ---------------------------------------------------------------------------
# Proportional correctness
# ---------------------------------------------------------------------------

class TestProportional:
    """3 panels word counts [2, 4, 2], clip_dur=8 → slices ~[2, 4, 2]s."""

    def test_proportional_split_word_weighted(self):
        panel_lines = ["aa bb", "cc dd ee ff", "gg hh"]  # words: 2, 4, 2 → total 8
        clip_dur = 8.0
        result = ta.align_panels(panel_lines, clip_dur,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        # 2/8 * 8 = 2.0, 4/8 * 8 = 4.0, 2/8 * 8 = 2.0
        assert result[0]["end_sec"] == pytest.approx(2.0, abs=0.01)
        assert result[1]["end_sec"] == pytest.approx(6.0, abs=0.01)
        assert result[2]["end_sec"] == pytest.approx(8.0, abs=0.01)

    def test_proportional_cumulative_starts(self):
        panel_lines = ["aa bb", "cc dd ee ff", "gg hh"]
        result = ta.align_panels(panel_lines, 8.0,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        assert result[0]["start_sec"] == pytest.approx(0.0)
        assert result[1]["start_sec"] == pytest.approx(2.0, abs=0.01)
        assert result[2]["start_sec"] == pytest.approx(6.0, abs=0.01)

    def test_proportional_end_equals_clip_dur(self):
        panel_lines = ["aa bb", "cc dd ee ff", "gg hh"]
        result = ta.align_panels(panel_lines, 8.0,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        assert result[-1]["end_sec"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Invariants always hold across multiple scenarios
# ---------------------------------------------------------------------------

class TestInvariantsAlwaysHold:
    """Parametrized invariant checks across varied inputs."""

    @pytest.mark.parametrize("panel_lines,clip_dur,stub_words", [
        # ASR case
        (
            ["the cat sat", "the dog ran"],
            3.0,
            [("the", 0.0, 0.5), ("cat", 0.5, 1.0), ("sat", 1.0, 1.5),
             ("the", 1.5, 2.0), ("dog", 2.0, 2.5), ("ran", 2.5, 3.0)],
        ),
        # Proportional (garbage words)
        (
            ["line one goes here", "line two short"],
            5.0,
            [("zzz", 0.0, 2.5), ("qqq", 2.5, 5.0)],
        ),
        # Empty transcription
        (["first panel", "second panel", "third"], 9.0, []),
    ])
    def test_invariants(self, panel_lines, clip_dur, stub_words):
        result = ta.align_panels(panel_lines, clip_dur,
                                 transcribe_fn=lambda p: stub_words,
                                 clip_path="fake.wav")
        _check_invariants(result, panel_lines, clip_dur)

    def test_len_equals_panel_lines(self):
        panel_lines = ["a b c", "d e", "f g h i", "j"]
        result = ta.align_panels(panel_lines, 10.0,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        assert len(result) == len(panel_lines)

    def test_no_zero_length_panels(self):
        """Even with many panels and a short clip, every panel >= min_dur."""
        # 10 panels, only 4 seconds — min_dur=0.3 → 10×0.3=3.0 < 4.0, fits
        panel_lines = [f"word{i} extra" for i in range(10)]
        result = ta.align_panels(panel_lines, 4.0,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        for i, seg in enumerate(result):
            dur = seg["end_sec"] - seg["start_sec"]
            assert dur >= 0.3 - 1e-9, f"panel {i} duration {dur:.3f} < 0.3"

    def test_all_panels_timed(self):
        """Every panel has a start_sec and end_sec, never None or missing."""
        panel_lines = ["one two", "three four five", "six"]
        result = ta.align_panels(panel_lines, 6.0,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        for i, seg in enumerate(result):
            assert seg.get("start_sec") is not None, f"[{i}] start_sec is None"
            assert seg.get("end_sec") is not None, f"[{i}] end_sec is None"


# ---------------------------------------------------------------------------
# Single panel
# ---------------------------------------------------------------------------

class TestSinglePanel:
    def test_single_panel_full_clip(self):
        """One panel → one slice [0, clip_dur]."""
        result = ta.align_panels(["the whole story"], 5.0,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        assert len(result) == 1
        assert result[0]["start_sec"] == pytest.approx(0.0)
        assert result[0]["end_sec"] == pytest.approx(5.0)

    def test_single_panel_invariants(self):
        result = ta.align_panels(["hello world"], 7.5,
                                 transcribe_fn=lambda p: [], clip_path="fake.wav")
        _check_invariants(result, ["hello world"], 7.5)


# ---------------------------------------------------------------------------
# Default transcriber: module imports without faster-whisper
# ---------------------------------------------------------------------------

class TestDefaultTranscriber:
    """Verify the module is importable and _default_word_transcribe is graceful
    when faster-whisper is absent — simulated via monkeypatching."""

    def test_module_imports_without_faster_whisper(self):
        """No ImportError at module import time even without faster-whisper."""
        # Already imported as `ta` above — success proves this
        assert hasattr(ta, "align_panels")

    def test_align_panels_without_clip_path_and_no_transcribe_fn(self):
        """With transcribe_fn=None and clip_path=None, must not crash.
        Should fall back to proportional since there's nothing to transcribe."""
        result = ta.align_panels(["hello world", "foo bar"], 4.0,
                                 transcribe_fn=None, clip_path=None)
        assert len(result) == 2
        for seg in result:
            assert seg["method"] == "proportional"

    def test_default_transcriber_returns_list_or_empty(self, monkeypatch):
        """_default_word_transcribe returns [] when faster-whisper absent."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        # Reset cached model so lazy loader re-runs
        ta._whisper_model = None
        result = ta._default_word_transcribe("fake.wav")
        assert result == [], f"Expected [] when faster-whisper absent, got {result!r}"


# ---------------------------------------------------------------------------
# match_threshold parameter
# ---------------------------------------------------------------------------

class TestMatchThreshold:
    """Verify match_threshold controls when ASR vs proportional is used."""

    def _partial_stub(self, clip_path):
        """Returns words for only the first panel — second panel gets nothing."""
        return _words(
            ("the", 0.0, 0.5),
            ("cat", 0.5, 1.0),
            ("sat", 1.0, 1.5),
        )

    def test_high_threshold_forces_proportional(self):
        """With threshold=0.99, partial match forces proportional."""
        result = ta.align_panels(
            ["the cat sat", "the dog ran"], 3.0,
            transcribe_fn=self._partial_stub, clip_path="fake.wav",
            match_threshold=0.99,
        )
        for seg in result:
            assert seg["method"] == "proportional"

    def test_low_threshold_allows_asr(self):
        """With threshold=0.0, any match allows ASR path."""
        result = ta.align_panels(
            ["the cat sat", "the dog ran"], 3.0,
            transcribe_fn=self._partial_stub, clip_path="fake.wav",
            match_threshold=0.0,
        )
        # At least some panels should use asr
        methods = {seg["method"] for seg in result}
        assert "asr" in methods
