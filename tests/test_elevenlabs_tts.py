"""
tests/test_elevenlabs_tts.py

TDD for tools/elevenlabs_tts_from_manifest.py's run-level provenance guard:
load_prior_tts_reuse must bulk-invalidate the segment_id->text_sha reuse map
when voice_id/model_id/output_format changed since the prior run, even when
a clip's own text_sha still matches (mirrors local_tts's backend/voice_ref
guard — see tests/test_local_tts.py).

No prior test file covered this module's logic (only its manifest-graph
wiring, in tests/test_pipeline.py); this file is new.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "elevenlabs_tts",
    Path(__file__).resolve().parent.parent / "tools" / "elevenlabs_tts_from_manifest.py",
)
et = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(et)  # type: ignore[union-attr]


def _write_index(path: Path) -> None:
    idx = {
        "voice_id": "voice-A",
        "model_id": "eleven_v3",
        "output_format": "mp3_44100_128",
        "clips": [
            {"segment_id": "g0001_p00", "text_sha": "sha-1"},
            {"segment_id": "g0002_p01", "text_sha": "sha-2"},
        ],
    }
    path.write_text(json.dumps(idx))


def test_voice_id_change_discards_reuse(tmp_path):
    idx_path = tmp_path / "tts_index.json"
    _write_index(idx_path)
    prior_sha = et.load_prior_tts_reuse(
        str(idx_path), False, "voice-B", "eleven_v3", "mp3_44100_128")
    assert prior_sha == {}


def test_model_id_change_discards_reuse(tmp_path):
    idx_path = tmp_path / "tts_index.json"
    _write_index(idx_path)
    prior_sha = et.load_prior_tts_reuse(
        str(idx_path), False, "voice-A", "eleven_multilingual_v2", "mp3_44100_128")
    assert prior_sha == {}


def test_output_format_change_discards_reuse(tmp_path):
    idx_path = tmp_path / "tts_index.json"
    _write_index(idx_path)
    prior_sha = et.load_prior_tts_reuse(
        str(idx_path), False, "voice-A", "eleven_v3", "mp3_22050_32")
    assert prior_sha == {}


def test_unchanged_provenance_keeps_reuse_map(tmp_path):
    # regression guard: identical voice_id/model_id/output_format -> the
    # per-clip text_sha map survives untouched.
    idx_path = tmp_path / "tts_index.json"
    _write_index(idx_path)
    prior_sha = et.load_prior_tts_reuse(
        str(idx_path), False, "voice-A", "eleven_v3", "mp3_44100_128")
    assert prior_sha == {"g0001_p00": "sha-1", "g0002_p01": "sha-2"}


def test_missing_index_returns_empty(tmp_path):
    prior_sha = et.load_prior_tts_reuse(
        str(tmp_path / "tts_index.json"), False,
        "voice-A", "eleven_v3", "mp3_44100_128")
    assert prior_sha == {}
