"""
tests/test_local_tts_timeout.py

TDD for the per-clip watchdog in tools/local_tts_from_manifest.py.

The bug (measured): one normal 26s clip (g0005) hung for 48 MINUTES mid-
generation, stalling the whole chapter. There was no timeout/retry around the
per-clip backend `model.generate` call. This suite locks in:

  - a fast synth runs on the normal path (no watchdog interference)
  - a synth that hangs longer than the timeout is ABANDONED (the run does not
    hang) and retried up to RETRIES times
  - a synth that fails N-1 times then succeeds recovers via retry
  - on final timeout/failure a SILENCE placeholder of the expected duration is
    written so timeline alignment is preserved, and the clip is marked failed
  - the timeout formula scales with the expected audio length but is floored

All stubs are pure-Python (sleep / raise) — NO real TTS model is loaded.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
import wave
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "local_tts",
    Path(__file__).resolve().parent.parent / "tools" / "local_tts_from_manifest.py",
)
lt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lt)  # type: ignore[union-attr]


def _wav_seconds(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


# ---- timeout formula -----------------------------------------------------

def test_clip_timeout_formula_floored_and_scaling():
    # short line -> floored at the minimum
    assert lt.clip_timeout_sec(0.0) == lt.CLIP_TIMEOUT_MIN_SEC
    assert lt.clip_timeout_sec(1.0) == lt.CLIP_TIMEOUT_MIN_SEC
    # long line -> scales with expected audio (8x), above the floor
    big = lt.clip_timeout_sec(60.0)
    assert big == pytest.approx(60.0 * lt.CLIP_TIMEOUT_FACTOR)
    assert big > lt.CLIP_TIMEOUT_MIN_SEC


def test_expected_audio_sec_grows_with_text():
    short = lt.expected_audio_sec("Hi.")
    long = lt.expected_audio_sec("word " * 200)
    assert long > short > 0.0


# ---- watchdog: fast path -------------------------------------------------

def test_guarded_synth_fast_path_runs_once(tmp_path):
    calls = []

    def fast(text, out_path, exaggeration):
        calls.append(text)
        Path(out_path).write_bytes(b"RIFFok")

    out = tmp_path / "g0001_p00.wav"
    res = lt.run_guarded_synth(
        fast, "hello", str(out), 0.5,
        timeout_sec=0.2, retries=2, expected_sec=1.0)
    assert res["ok"] is True
    assert res["timed_out"] is False
    assert calls == ["hello"]
    assert out.exists()


# ---- watchdog: hang is abandoned + retried -------------------------------

def test_guarded_synth_times_out_and_gives_up(tmp_path):
    attempts = []

    def hang(text, out_path, exaggeration):
        attempts.append(1)
        time.sleep(5.0)              # >> the 0.2s timeout — simulates the 48min hang
        Path(out_path).write_bytes(b"too-late")

    out = tmp_path / "g0005_p00.wav"
    t0 = time.time()
    res = lt.run_guarded_synth(
        hang, "a hung line", str(out), 0.5,
        timeout_sec=0.2, retries=2, expected_sec=3.0)
    elapsed = time.time() - t0

    # the run did NOT hang for 5s*3 — it abandoned each attempt at ~0.2s
    assert elapsed < 3.0
    assert res["ok"] is False
    assert res["timed_out"] is True
    assert res["attempts"] == 3            # 1 initial + 2 retries
    assert len(attempts) == 3
    # a silence placeholder of the EXPECTED duration was written (alignment kept)
    assert out.exists()
    assert _wav_seconds(str(out)) == pytest.approx(3.0, abs=0.05)


def test_guarded_synth_retry_recovers_after_failures(tmp_path):
    state = {"n": 0}

    def flaky(text, out_path, exaggeration):
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("boom")   # fail twice
        Path(out_path).write_bytes(b"RIFFgood")

    out = tmp_path / "g0002_p00.wav"
    res = lt.run_guarded_synth(
        flaky, "recovers", str(out), 0.5,
        timeout_sec=0.2, retries=2, expected_sec=1.0)
    assert res["ok"] is True
    assert res["attempts"] == 3
    assert out.read_bytes() == b"RIFFgood"


def test_guarded_synth_timeout_then_success_leaves_only_daemon_worker(tmp_path):
    state = {"n": 0}

    def first_hangs_then_succeeds(text, out_path, exaggeration):
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(5.0)
            Path(out_path).write_bytes(b"late")
            return
        Path(out_path).write_bytes(b"RIFFgood")

    out = tmp_path / "g0004_p00.wav"
    res = lt.run_guarded_synth(
        first_hangs_then_succeeds, "recovers", str(out), 0.5,
        timeout_sec=0.2, retries=1, expected_sec=1.0, segment_id="g0004_p00")

    assert res["ok"] is True
    assert res["attempts"] == 2
    assert out.read_bytes() == b"RIFFgood"
    leaked = [t for t in threading.enumerate()
              if t.name.startswith("tts-synth-g0004_p00") and t.is_alive()]
    assert leaked
    assert all(t.daemon for t in leaked)


def test_guarded_synth_exhausts_retries_on_persistent_error(tmp_path):
    def always_fail(text, out_path, exaggeration):
        raise RuntimeError("nope")

    out = tmp_path / "g0003_p00.wav"
    res = lt.run_guarded_synth(
        always_fail, "bad", str(out), 0.5,
        timeout_sec=0.2, retries=2, expected_sec=2.5)
    assert res["ok"] is False
    assert res["attempts"] == 3
    # placeholder still written so the timeline aligns
    assert _wav_seconds(str(out)) == pytest.approx(2.5, abs=0.05)


# ---- silence placeholder writer ------------------------------------------

def test_write_silence_wav_has_requested_duration(tmp_path):
    out = tmp_path / "sil.wav"
    lt.write_silence_wav(str(out), 1.5)
    assert _wav_seconds(str(out)) == pytest.approx(1.5, abs=0.02)
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2


# ---- end-to-end through synthesize_manifest: ONE hung clip never stalls --

def _script():
    return {
        "sections": [
            {
                "section_index": 0,
                "tts_paragraphs_v3": ["[tense] First line here.", "[calm] Second line here."],
                "shots": [
                    {"group_id": 1, "beat_id": 1},
                    {"group_id": 2, "beat_id": 2},
                ],
            }
        ]
    }


def test_synthesize_manifest_continues_past_a_hung_clip(tmp_path):
    seen = []

    def synth(text, out_path, exaggeration):
        seen.append(text)
        if "First" in text:
            time.sleep(5.0)                     # this clip "hangs"
            return
        Path(out_path).write_bytes(b"RIFFok")

    t0 = time.time()
    index = lt.synthesize_manifest(
        _script(), str(tmp_path),
        backend="kokoro", synth_fn=synth,
        duration_fn=lt.wav_duration_sec,
        clip_timeout_sec=0.2, clip_retries=1,
        group_mode=False,   # this test pins the PER-PANEL watchdog path explicitly
    )
    elapsed = time.time() - t0

    # the hung clip was abandoned; the whole chapter did not stall
    assert elapsed < 4.0
    clips = {c["segment_id"]: c for c in index["clips"]}
    assert set(clips) == {"g0001_p00", "g0002_p01"}
    # hung clip: marked failed but present with a silence placeholder duration
    assert clips["g0001_p00"]["tts_failed"] is True
    assert clips["g0001_p00"]["duration_sec"] > 0.0
    # good clip voiced normally
    assert clips["g0002_p01"].get("tts_failed", False) is False
    # both clip files exist on disk (alignment preserved)
    assert (tmp_path / "clips" / "g0001_p00.wav").exists()
    assert (tmp_path / "clips" / "g0002_p01.wav").exists()


# ---- null-ish lines + duration sanity (ORV Ep33 / Ep38) -------------------
# Ep33 g0030_p16 and Ep38 g0007_p01 both carried the literal narration line
# "None." between two real lines. The failure was NOT consistent: Ep33's synth
# raised (placeholder, flagged, chapter blocked) while Ep38's RETURNED NORMALLY
# with 0.06s of audio, so run_guarded_synth reported ok and the chapter shipped
# ~2.4s of dead air inside a 2.5s slot. `ok` only means the call did not raise
# or time out -- it never inspects the audio.

def test_is_nullish_line_matches_stringified_nulls():
    for s in ("None", "None.", "none", " NULL ", "nan", "N/A", "n/a", "undefined", "nil."):
        assert lt.is_nullish_line(s), s


def test_is_nullish_line_keeps_real_narration():
    for s in ("No one moves.", "Nothing stirs in the dark.", "None of them survive.",
              "Nan grips the blade.", "He says nothing."):
        assert not lt.is_nullish_line(s), s


def _script_with(lines):
    return {"sections": [{
        "section_index": 0,
        "tts_paragraphs_v3": list(lines),
        "shots": [{"group_id": i + 1, "beat_id": i + 1} for i in range(len(lines))],
    }]}


def test_nullish_line_is_never_sent_to_the_backend(tmp_path):
    seen = []

    def synth(text, out_path, exaggeration):
        seen.append(text)
        Path(out_path).write_bytes(b"RIFFok")

    index = lt.synthesize_manifest(
        _script_with(["[excited] None.", "[calm] A real line that should be voiced."]),
        str(tmp_path), backend="kokoro", synth_fn=synth,
        duration_fn=lt.wav_duration_sec, clip_timeout_sec=0.5, clip_retries=0,
        group_mode=False)

    clips = {c["segment_id"]: c for c in index["clips"]}
    assert clips["g0001_p00"]["tts_failed"] is True
    assert not any("None" in t for t in seen), "null-ish text reached the backend"
    # the real line still voiced, and the segment is kept for alignment
    assert clips["g0002_p01"].get("tts_failed", False) is False
    assert (tmp_path / "clips" / "g0001_p00.wav").exists()


def test_successful_synth_with_near_empty_audio_is_failed(tmp_path):
    # THE Ep38 CASE: the backend returns normally but writes ~nothing.
    long_line = "[calm] " + ("a fairly long narration line that takes real time to speak. " * 2)

    def synth(text, out_path, exaggeration):
        lt.write_silence_wav(out_path, 0.05)     # "succeeds", emits 0.05s

    index = lt.synthesize_manifest(
        _script_with([long_line]), str(tmp_path),
        backend="kokoro", synth_fn=synth, duration_fn=lt.wav_duration_sec,
        clip_timeout_sec=0.5, clip_retries=0, group_mode=False)

    clip = index["clips"][0]
    assert clip["tts_failed"] is True, "a near-empty 'successful' take must fail"


def test_short_but_valid_take_is_not_failed(tmp_path):
    # guard against over-firing: a genuinely short line voiced proportionally
    def synth(text, out_path, exaggeration):
        lt.write_silence_wav(out_path, lt.expected_audio_sec(text) * 0.9)

    index = lt.synthesize_manifest(
        _script_with(["[calm] A perfectly ordinary narration line goes here."]),
        str(tmp_path), backend="kokoro", synth_fn=synth,
        duration_fn=lt.wav_duration_sec, clip_timeout_sec=0.5, clip_retries=0,
        group_mode=False)

    assert index["clips"][0].get("tts_failed", False) is False


# ---- one-word lines + ellipsis pauses (Wimp Ch5 "Tada!", ORV Ep120 "So...",
# Wimp Ch1 "...wondering if this is...") --------------------------------------
# Qwen-MLX samples (no seed), and on a one-word line it sometimes stops after a
# blip: 2 of the corpus' 6 one-word clips (of 17,799) came out at 0.06s/0.08s.
# The dead-take check flagged "Tada!" but never re-rolled it (only raises and
# timeouts were retried) and skipped "So..." entirely (estimate under 0.30s),
# so one chapter blocked on audio_failed and one shipped a blip.

def _take_of(seconds):
    return lambda text, out_path, exaggeration: lt.write_silence_wav(out_path, seconds)


def test_a_near_empty_take_is_re_rolled_and_a_good_retake_kept(tmp_path):
    calls = []

    def synth(text, out_path, exaggeration):
        calls.append(text)
        lt.write_silence_wav(out_path, 0.06 if len(calls) == 1 else 0.6)

    index = lt.synthesize_manifest(
        _script_with(["[tense] Tada!"]), str(tmp_path), backend="kokoro",
        synth_fn=synth, duration_fn=lt.wav_duration_sec, clip_timeout_sec=0.5,
        clip_retries=2, group_mode=False)

    clip = index["clips"][0]
    assert clip.get("tts_failed", False) is False
    assert len(calls) == 2
    assert clip["duration_sec"] == pytest.approx(0.6, abs=0.02)


def test_a_take_that_stays_near_empty_fails_after_every_retry(tmp_path):
    calls = []

    def synth(text, out_path, exaggeration):
        calls.append(text)
        lt.write_silence_wav(out_path, 0.06)

    index = lt.synthesize_manifest(
        _script_with(["[tense] Tada!"]), str(tmp_path), backend="kokoro",
        synth_fn=synth, duration_fn=lt.wav_duration_sec, clip_timeout_sec=0.5,
        clip_retries=2, group_mode=False)

    assert index["clips"][0]["tts_failed"] is True
    assert len(calls) == 3                     # 1 take + 2 re-rolls, then give up


def test_a_blip_on_the_shortest_line_is_caught(tmp_path):
    # "So." is estimated at ~0.21s, under the old 0.30s gate that skipped it
    index = lt.synthesize_manifest(
        _script_with(["[whisper] So..."]), str(tmp_path), backend="kokoro",
        synth_fn=_take_of(0.08), duration_fn=lt.wav_duration_sec,
        clip_timeout_sec=0.5, clip_retries=0, group_mode=False)
    assert index["clips"][0]["tts_failed"] is True


def test_a_real_one_word_take_is_not_failed(tmp_path):
    # the shortest real clips measured in the corpus are ~0.46-0.65s
    index = lt.synthesize_manifest(
        _script_with(["[calm] Hi."]), str(tmp_path), backend="kokoro",
        synth_fn=_take_of(0.45), duration_fn=lt.wav_duration_sec,
        clip_timeout_sec=0.5, clip_retries=0, group_mode=False)
    assert index["clips"][0].get("tts_failed", False) is False


def test_a_line_ending_in_an_ellipsis_keeps_its_pause(tmp_path):
    seen = []

    def synth(text, out_path, exaggeration):
        seen.append(text)
        lt.write_silence_wav(out_path, 1.0)

    index = lt.synthesize_manifest(
        _script_with(["[tense] Leaving him wondering if this is...",
                      "[tense] Leaving him wondering if this is.",
                      "[calm] So... he waits for the door to open.",
                      "[calm] And then…"]),
        str(tmp_path), backend="kokoro", synth_fn=synth,
        duration_fn=lt.wav_duration_sec, clip_timeout_sec=0.5, clip_retries=0,
        group_mode=False)

    dur = [c["duration_sec"] for c in index["clips"]]
    pause = lt.PAUSE_AFTER_ELLIPSIS_SEC
    assert pause >= 0.4
    assert dur[0] == pytest.approx(1.0 + pause, abs=0.02)   # trails off: waits
    assert dur[1] == pytest.approx(1.0, abs=0.02)           # a full stop: no extra
    assert dur[2] == pytest.approx(1.0, abs=0.02)           # mid-line ellipsis: no extra
    assert dur[3] == pytest.approx(1.0 + pause, abs=0.02)   # unicode ellipsis
    # the backend still never sees "..." (it triggered filler vocalization)
    assert not any("..." in t or "…" in t for t in seen)


def test_a_clip_voiced_before_the_pause_existed_gets_it_once(tmp_path, monkeypatch):
    # Wimp Ch1's "...wondering if this is..." is already voiced and cached: a
    # re-run must add the pause without re-synthesizing, and never add it twice
    script = _script_with(["[tense] Leaving him wondering if this is..."])
    calls = []

    def synth(text, out_path, exaggeration):
        calls.append(text)
        lt.write_silence_wav(out_path, 1.0)

    def run():
        idx = lt.synthesize_manifest(
            script, str(tmp_path), backend="kokoro", synth_fn=synth,
            duration_fn=lt.wav_duration_sec, clip_timeout_sec=0.5,
            clip_retries=0, group_mode=False)
        (tmp_path / "tts_index.json").write_text(json.dumps(idx))   # as main() does
        return idx["clips"][0]

    pause = lt.PAUSE_AFTER_ELLIPSIS_SEC
    monkeypatch.setattr(lt, "PAUSE_AFTER_ELLIPSIS_SEC", 0.0)   # the old code
    assert run()["duration_sec"] == pytest.approx(1.0, abs=0.02)
    monkeypatch.setattr(lt, "PAUSE_AFTER_ELLIPSIS_SEC", pause)

    topped = run()
    assert topped["cached"] is True and len(calls) == 1         # no re-synthesis
    assert topped["duration_sec"] == pytest.approx(1.0 + pause, abs=0.02)
    again = run()
    assert again["duration_sec"] == pytest.approx(1.0 + pause, abs=0.02)
    assert len(calls) == 1
