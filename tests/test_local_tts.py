"""
tests/test_local_tts.py

TDD for tools/local_tts_from_manifest.py — the free local-TTS adapter. Covers
the pure logic and the synth-injected orchestrator (no model loaded), so the
contract with timeline_planner is verified without heavy deps.
"""

from __future__ import annotations

import importlib.util
import json
import os
import wave
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "local_tts",
    Path(__file__).resolve().parent.parent / "tools" / "local_tts_from_manifest.py",
)
lt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lt)  # type: ignore[union-attr]


# ---- tag + mood helpers --------------------------------------------------

def test_leading_tag_and_strip():
    assert lt.leading_tag("[tense] He runs.") == "tense"
    assert lt.leading_tag("No tag here") is None
    assert lt.strip_bracket_tags("[tense] He runs [beat] now.") == "He runs now."


def test_exaggeration_to_instruction_scales():
    calm = lt.exaggeration_to_instruction(0.30)
    intense = lt.exaggeration_to_instruction(0.78)
    boom = lt.exaggeration_to_instruction(0.92)
    assert "calm" in calm.lower()
    assert "intense" in intense.lower() or "dramatic" in intense.lower()
    assert "explosive" in boom.lower() or "forcefully" in boom.lower()
    # every bucket returns a non-empty instruction
    for e in (0.1, 0.4, 0.6, 0.8, 0.95):
        assert lt.exaggeration_to_instruction(e).strip()


def test_exaggeration_to_speed_scales():
    # calmer -> slower; more intense -> faster
    assert lt.exaggeration_to_speed(0.2) < lt.exaggeration_to_speed(0.5) < lt.exaggeration_to_speed(0.95)
    assert lt.exaggeration_to_speed(0.2) < 1.0       # somber slows down
    assert lt.exaggeration_to_speed(0.95) > 1.0      # explosive speeds up


def test_mood_to_exaggeration_scale():
    calm = lt.mood_to_exaggeration("calm")
    tense = lt.mood_to_exaggeration("tense")
    boom = lt.mood_to_exaggeration("explosive")
    assert calm < tense < boom
    assert lt.mood_to_exaggeration(None) == lt._DEFAULT_EXAGGERATION
    assert lt.mood_to_exaggeration("gibberish") == lt._DEFAULT_EXAGGERATION


# ---- item extraction (segment_id contract) -------------------------------

def _script():
    return {
        "sections": [
            {
                "section_index": 0,
                "tts_paragraphs_v3": ["[tense] The blade falls.", "[calm] Silence settles."],
                "script_paragraphs": ["The blade falls.", "Silence settles."],
                "shots": [
                    {"group_id": 1, "beat_id": 1},
                    {"group_id": 2, "beat_id": 2},
                ],
            }
        ]
    }


def test_extract_items_canonical_segment_ids():
    items = lt.extract_items_from_manifest(_script(), "tts_v3")
    assert [it["segment_id"] for it in items] == ["g0001_p00", "g0002_p01"]
    assert items[0]["text"].startswith("[tense]")


# ---- wav duration --------------------------------------------------------

def test_wav_duration_sec(tmp_path):
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(1000)
        w.writeframes(b"\x00\x00" * 2000)   # 2000 frames @ 1000 Hz = 2.0s
    assert lt.wav_duration_sec(str(p)) == pytest.approx(2.0)


# ---- orchestrator (synth injected) ---------------------------------------

def test_synthesize_manifest_builds_aligned_index(tmp_path):
    calls = []

    def fake_synth(text, out_path, exaggeration):
        calls.append((text, exaggeration))
        Path(out_path).write_bytes(b"FAKEWAV")   # just create the file

    index = lt.synthesize_manifest(
        _script(), str(tmp_path),
        backend="chatterbox", synth_fn=fake_synth,
        duration_fn=lambda p: 3.0,   # stub duration
        group_mode=False,            # per-panel path (what this test verifies)
    )
    clips = index["clips"]
    assert [c["segment_id"] for c in clips] == ["g0001_p00", "g0002_p01"]
    # tags stripped before synthesis; mood drives exaggeration
    assert calls[0][0] == "The blade falls."
    assert calls[0][1] > calls[1][1]            # tense > calm
    # contract fields timeline needs
    assert clips[0]["audio_file"] == "clips/g0001_p00.wav"
    assert clips[0]["duration_sec"] == 3.0
    assert index["total_duration_sec"] == 6.0
    assert (tmp_path / "clips" / "g0001_p00.wav").exists()


def _synth_write(calls):
    def _fn(text, out_path, exaggeration):
        calls.append(out_path)
        with open(out_path, "wb") as f:
            f.write(b"RIFFXXXXWAVE")
    return _fn


def test_synthesize_manifest_caches_unchanged_text(tmp_path):
    # first run synthesizes both and writes the index
    calls1 = []
    idx = lt.synthesize_manifest(
        _script(), str(tmp_path), backend="kokoro",
        synth_fn=_synth_write(calls1), duration_fn=lambda p: 1.0,
        group_mode=False)            # per-panel path: one clip per segment
    (tmp_path / "tts_index.json").write_text(json.dumps(idx))
    assert len(calls1) == 2
    assert all(c.get("text_sha") for c in idx["clips"])      # fingerprint stored
    # second run, identical script: both cached (text unchanged) -> no synthesis
    calls2 = []
    lt.synthesize_manifest(
        _script(), str(tmp_path), backend="kokoro",
        synth_fn=_synth_write(calls2), duration_fn=lambda p: 1.0,
        group_mode=False)
    assert calls2 == []


def test_synthesize_manifest_revoices_only_changed_segments(tmp_path):
    # establish a baseline index
    idx = lt.synthesize_manifest(
        _script(), str(tmp_path), backend="kokoro",
        synth_fn=_synth_write([]), duration_fn=lambda p: 1.0,
        group_mode=False)            # per-panel path: cache keyed by segment_id
    (tmp_path / "tts_index.json").write_text(json.dumps(idx))
    # edit ONLY the second paragraph's narration
    changed = _script()
    changed["sections"][0]["tts_paragraphs_v3"][1] = "[calm] A new, different line."
    calls = []
    lt.synthesize_manifest(
        changed, str(tmp_path), backend="kokoro",
        synth_fn=_synth_write(calls), duration_fn=lambda p: 1.0,
        group_mode=False)
    # only g0002_p01 re-voiced; g0001_p00 kept (deterministic gate, incremental)
    assert len(calls) == 1
    assert os.path.basename(calls[0]).startswith("g0002_p01.attempt")
    assert (tmp_path / "clips" / "g0002_p01.wav").exists()


def test_synthesize_manifest_prunes_orphan_clips(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "g0099_p09.wav").write_bytes(b"orphan")   # not in script
    lt.synthesize_manifest(
        _script(), str(tmp_path), backend="kokoro",
        synth_fn=_synth_write([]), duration_fn=lambda p: 1.0,
        group_mode=False)            # per-panel path: orphan g####_p## pruning
    assert not (tmp_path / "clips" / "g0099_p09.wav").exists()      # pruned


# ---- voice-clone ref sidecar (locked narrator g0021_p02) -------------------

def test_ref_text_for_reads_sidecar_transcript(tmp_path):
    ref = tmp_path / "narrator_ref.wav"
    ref.write_bytes(b"RIFF")
    (tmp_path / "narrator_ref.txt").write_text("Three cloaked figures appear.\n")
    assert lt.ref_text_for(str(ref)) == "Three cloaked figures appear."


def test_ref_text_for_empty_when_no_sidecar(tmp_path):
    ref = tmp_path / "narrator_ref.wav"
    ref.write_bytes(b"RIFF")
    assert lt.ref_text_for(str(ref)) == ""


# ---- clip conditioning: lead/tail trim + soft-attack lift ------------------
# Root cause (measured on the Modal ch1 run): some clips open at 10-22% of
# body loudness for 300ms+ — the first word is perceptually swallowed.

import numpy as np


def _tone(sr=24000, sec=2.0, amp=0.5):
    t = np.arange(int(sr * sec)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_condition_trims_long_lead_to_pad():
    sr = 24000
    x = np.concatenate([np.zeros(sr, np.float32), _tone(sr, 2.0)])  # 1.0s dead lead
    y, info = lt.condition_wav(x, sr)
    lead = np.argmax(np.abs(y) > 0.01) / sr
    assert lead <= lt.PAD_LEAD_SEC + 0.02
    assert info["lead_trim_sec"] >= 0.8


def test_condition_keeps_tight_clip_intact():
    sr = 24000
    x = _tone(sr, 2.0)
    y, info = lt.condition_wav(x, sr)
    assert len(y) <= len(x) + int(lt.PAD_LEAD_SEC * sr)
    assert info["soft_attack"] is False
    assert info["attack_gain"] == 1.0
    # body untouched
    assert np.allclose(y[-sr:], x[len(x) - sr:], atol=1e-6) or len(y) <= len(x)


def test_condition_lifts_soft_attack_bounded():
    sr = 24000
    head = _tone(sr, 0.4, amp=0.05)          # 10% of body level
    body = _tone(sr, 2.0, amp=0.5)
    x = np.concatenate([head, body])
    y, info = lt.condition_wav(x, sr)
    assert info["soft_attack"] is True
    assert 1.0 < info["attack_gain"] <= lt.ATTACK_MAX_GAIN
    aw = int(lt.ATTACK_WINDOW_SEC * sr)
    head_rms = float(np.sqrt((y[:aw] ** 2).mean()))
    body_rms = float(np.sqrt((y[-sr:] ** 2).mean()))
    assert head_rms / body_rms >= 0.35       # audibly present now (was 0.10)
    assert np.abs(y).max() <= 1.0            # never clips


def test_condition_silence_only_is_noop():
    sr = 24000
    x = np.zeros(sr, np.float32)
    y, info = lt.condition_wav(x, sr)
    assert len(y) == len(x)
    assert info["soft_attack"] is False


def test_condition_wav_file_fails_soft_on_unreadable_file(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"FAKEWAV")
    info = lt.condition_wav_file(str(p))
    assert "condition_error" in info          # visible in the index, not silent
    assert p.read_bytes() == b"FAKEWAV"       # original file left untouched


# ---- Fix A: normalize_tts_text -------------------------------------------

def test_normalize_ellipsis_collapsed_to_period():
    """Unicode ellipsis and dot-runs → single period; no ... in result."""
    result = lt.normalize_tts_text("'Damn it all...' he hissed")
    assert "..." not in result
    assert "…" not in result
    # dot should still be present (sentence close), just not repeated
    assert result == "'Damn it all.' he hissed"


def test_normalize_unicode_ellipsis_collapsed():
    result = lt.normalize_tts_text("Wait… I see it now.")
    assert "…" not in result
    assert result == "Wait. I see it now."


def test_normalize_em_dash_becomes_comma_space():
    """Em-dash between phrases → ', ' (natural pause, no filler trigger)."""
    result = lt.normalize_tts_text("the branch—the assassins")
    assert "—" not in result
    assert result == "the branch, the assassins"


def test_normalize_en_dash_becomes_comma_space():
    result = lt.normalize_tts_text("victory–defeat, two sides.")
    assert "–" not in result
    assert ", " in result


def test_normalize_double_hyphen_becomes_comma_space():
    result = lt.normalize_tts_text("He turned -- then stopped.")
    assert "--" not in result
    assert ", " in result


def test_normalize_leading_ellipsis_stripped():
    """A line that starts with ellipsis/dot must not begin with punctuation."""
    result = lt.normalize_tts_text("…serves you all right")
    assert not result.startswith(".")
    assert not result.startswith("…")
    assert "serves" in result


def test_normalize_leading_ellipsis_after_opening_quote():
    """An opening quote followed by ellipsis: quote stays, dot stripped."""
    result = lt.normalize_tts_text('"…serves you all right"')
    assert result.startswith('"')
    # The dot from collapsed ellipsis immediately after the opening quote is stripped
    assert not result.startswith('".')
    assert "serves" in result


def test_normalize_repeated_exclamation_collapsed():
    """Three or more ! → single !"""
    assert lt.normalize_tts_text("wide!!!") == "wide!"
    assert lt.normalize_tts_text("Run!!") == "Run!"


def test_normalize_repeated_question_collapsed():
    assert lt.normalize_tts_text("Really??") == "Really?"


def test_normalize_interrobang_becomes_question():
    assert lt.normalize_tts_text("What?!") == "What?"
    assert lt.normalize_tts_text("No!?") == "No?"


def test_normalize_clean_line_unchanged():
    """A normal line with no problematic punctuation comes through verbatim."""
    line = "He steps forward into the light."
    assert lt.normalize_tts_text(line) == line


def test_normalize_quoted_dialogue_unchanged():
    """Quotation marks must not be stripped — dialogue stays intact."""
    line = 'She said, "You have to go now."'
    assert lt.normalize_tts_text(line) == line


def test_synth_site_uses_normalized_text(tmp_path):
    """The text passed to the synthesizer must have no ellipsis or em-dash.

    We inject a stubbed synth_fn that captures the text it receives, then
    call synthesize_clips with a manifest item containing both markers.
    The stub writes a minimal valid WAV so duration_fn doesn't crash.
    """
    import struct
    import wave as _wave

    captured: list = []

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        captured.append(text)
        # Write a minimal silent WAV so the pipeline can measure duration
        with _wave.open(out_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00\x00" * 2400)

    script_obj = {
        "narration_source": "gemini_verbatim",
        "sections": [{
            "section_index": 0,
            "script_paragraphs": ["[tense] Silence… then—the blade falls."],
            "tts_paragraphs_v3": ["[tense] Silence… then—the blade falls."],
            "shots": [{"segment_id": "g0001_p00", "group_id": 1,
                       "beat_id": 1, "section_index": 0, "paragraph_index": 0}],
            "tts_meta": [{"segment_id": "g0001_p00", "group_id": 1,
                          "beat_id": 1, "section_index": 0, "paragraph_index": 0,
                          "text": "[tense] Silence… then—the blade falls."}],
        }],
    }

    lt.synthesize_manifest(
        script_obj=script_obj,
        out_dir=str(tmp_path),
        synth_fn=stub_synth,
        backend="stub",
        text_source="tts_v3",
        group_mode=False,            # per-panel path: verify text normalization
    )

    assert captured, "synth_fn was never called"
    synth_text = captured[0]
    assert "…" not in synth_text, f"ellipsis reached synth: {synth_text!r}"
    assert "—" not in synth_text, f"em-dash reached synth: {synth_text!r}"
