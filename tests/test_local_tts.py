"""
tests/test_local_tts.py

TDD for tools/local_tts_from_manifest.py — the free local-TTS adapter. Covers
the pure logic and the synth-injected orchestrator (no model loaded), so the
contract with timeline_planner is verified without heavy deps.
"""

from __future__ import annotations

import importlib.util
import json
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


def test_synthesize_manifest_caches_existing(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "g0001_p00.wav").write_bytes(b"X")
    (tmp_path / "clips" / "g0002_p01.wav").write_bytes(b"X")
    synth_calls = []
    lt.synthesize_manifest(
        _script(), str(tmp_path), backend="kokoro",
        synth_fn=lambda t, o, e: synth_calls.append(o),
        duration_fn=lambda p: 1.0, overwrite=False,
    )
    assert synth_calls == []                      # both cached, no synthesis
