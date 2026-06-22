"""
tests/test_tts_group_synth.py

TDD for per-group TTS synthesis mode in tools/local_tts_from_manifest.py.

Per-group mode: synthesize ONE clip per GROUP (g####.wav) from that group's
panel-lines joined, then forced-align to get each panel's time-slice in
manifest.align.json.

All tests use stubbed synth + alignment — no model is ever loaded.
"""

from __future__ import annotations

import importlib.util
import json
import os
import wave
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module-level spec-loader (standard pattern for this repo)
# ---------------------------------------------------------------------------
_SPEC = importlib.util.spec_from_file_location(
    "local_tts",
    Path(__file__).resolve().parent.parent / "tools" / "local_tts_from_manifest.py",
)
lt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lt)  # type: ignore[union-attr]

# The audio<->narration freshness gate (prep-QA's audio_stale check) lives here.
_NC_SPEC = importlib.util.spec_from_file_location(
    "narration_consistency",
    Path(__file__).resolve().parent.parent / "tools" / "narration_consistency.py",
)
nc = importlib.util.module_from_spec(_NC_SPEC)
_NC_SPEC.loader.exec_module(nc)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_script(groups: list[dict]) -> dict:
    """Build a minimal manifest.script.json structure.

    groups: list of dicts with keys:
        group_id (int), panel_texts (list[str])
    Each panel in a group gets a sequential paragraph_index within its section.
    """
    sections = []
    para_idx = 0
    for g in groups:
        gid = g["group_id"]
        texts = g["panel_texts"]
        shots = [{"group_id": gid, "beat_id": gid} for _ in texts]
        paras = [{"text": t} for t in texts]
        sections.append({
            "section_index": para_idx,
            "shots": shots,
            "tts_paragraphs_v3": paras,
        })
        para_idx += 1
    return {"sections": sections}


def _write_silence_wav(path: str, duration_sec: float = 2.0, sr: int = 24000) -> None:
    """Write a minimal silent PCM16 wav for duration testing."""
    n = int(duration_sec * sr)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)


def _stub_align(panel_lines, clip_dur_sec, *, transcribe_fn=None, clip_path=None):
    """Stub alignment: splits proportionally, method='proportional'."""
    n = len(panel_lines)
    if n == 0:
        return []
    dur = clip_dur_sec / n
    result = []
    for i in range(n):
        result.append({
            "start_sec": round(i * dur, 4),
            "end_sec": round((i + 1) * dur, 4),
            "method": "proportional",
        })
    result[-1]["end_sec"] = clip_dur_sec
    return result


def _stub_align_asr(offsets: list[tuple[float, float]]):
    """Stub alignment that returns known ASR offsets with method='asr'."""
    def _align(panel_lines, clip_dur_sec, *, transcribe_fn=None, clip_path=None):
        result = []
        for i, (start, end) in enumerate(offsets):
            result.append({"start_sec": start, "end_sec": end, "method": "asr"})
        return result
    return _align


# ---------------------------------------------------------------------------
# Test 1: Grouping — segments grouped by group_id, panel order preserved
# ---------------------------------------------------------------------------

def test_grouping_two_groups_three_plus_one(tmp_path):
    """g0001_p00/p01/p02 (group 1) + g0002_p03 (group 2) → 2 groups.

    Joined texts per group must be correct; panel order within each group
    is preserved; the second group gets one joined text.
    """
    script = _make_script([
        {"group_id": 1, "panel_texts": [
            "[tense] The gate opens.",
            "[calm] Silence follows.",
            "[dramatic] The blade falls.",
        ]},
        {"group_id": 2, "panel_texts": [
            "[explosive] Everything ends.",
        ]},
    ])

    group_calls: list[dict] = []

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        group_calls.append({"text": text, "out_path": out_path, "exag": exaggeration})
        _write_silence_wav(out_path, duration_sec=3.0)

    index = lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=lambda p: 3.0,
        group_mode=True,
        align_fn=_stub_align,
    )

    # Exactly 2 synth calls (one per group)
    assert len(group_calls) == 2, f"expected 2 group synth calls, got {len(group_calls)}"

    # Group 1: three panel lines joined with space (tags stripped + normalized)
    joined_g1 = group_calls[0]["text"]
    assert "The gate opens." in joined_g1
    assert "Silence follows." in joined_g1
    assert "The blade falls." in joined_g1

    # Group 2: single panel
    joined_g2 = group_calls[1]["text"]
    assert "Everything ends." in joined_g2

    # Clips written: g0001.wav and g0002.wav (not per-panel names)
    clips_dir = tmp_path / "clips"
    assert (clips_dir / "g0001.wav").exists(), "missing g0001.wav group clip"
    assert (clips_dir / "g0002.wav").exists(), "missing g0002.wav group clip"
    # No per-panel clips written in group mode
    assert not (clips_dir / "g0001_p00.wav").exists(), "per-panel clip must NOT exist in group mode"


# ---------------------------------------------------------------------------
# Test 2: manifest.align.json written with one entry per panel
# ---------------------------------------------------------------------------

def test_group_mode_writes_align_manifest(tmp_path):
    """Group mode writes manifest.align.json next to tts_index.json.

    Each panel (g####_p##) appears exactly once, with group_clip, start_sec,
    end_sec, method. Panel order is preserved across all groups.
    """
    script = _make_script([
        {"group_id": 1, "panel_texts": ["[tense] Line one.", "[calm] Line two."]},
        {"group_id": 2, "panel_texts": ["[intense] Line three."]},
    ])

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        _write_silence_wav(out_path, duration_sec=4.0)

    lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=lambda p: 4.0,
        group_mode=True,
        align_fn=_stub_align,
    )

    align_path = tmp_path / "manifest.align.json"
    assert align_path.exists(), "manifest.align.json must be written in group mode"

    align = json.loads(align_path.read_text())
    assert "segments" in align, "manifest.align.json must have 'segments' key"

    segs = align["segments"]
    # 3 panels total: g0001_p00, g0001_p01, g0002_p00
    assert len(segs) == 3, f"expected 3 panel entries, got {len(segs)}: {[s['segment_id'] for s in segs]}"

    # Every entry has required fields
    for seg in segs:
        assert "segment_id" in seg
        assert "group_clip" in seg
        assert "start_sec" in seg
        assert "end_sec" in seg
        assert "method" in seg

    # group_clip paths point to group clips (not per-panel)
    for seg in segs:
        gid_str = seg["segment_id"].split("_")[0]   # e.g. "g0001"
        assert seg["group_clip"].endswith(f"{gid_str}.wav"), (
            f"group_clip for {seg['segment_id']} must be {gid_str}.wav, "
            f"got {seg['group_clip']}"
        )

    # Boundaries monotonic within each group
    g1_segs = [s for s in segs if s["segment_id"].startswith("g0001")]
    assert g1_segs[0]["start_sec"] == pytest.approx(0.0)
    assert g1_segs[0]["end_sec"] <= g1_segs[1]["start_sec"] + 1e-9
    assert g1_segs[1]["end_sec"] == pytest.approx(4.0)  # last panel ends at clip_dur


# ---------------------------------------------------------------------------
# Test 3: flag OFF → per-panel path unchanged; no manifest.align.json
# ---------------------------------------------------------------------------

def test_flag_off_uses_per_panel_path(tmp_path):
    """group_mode=False → exact per-panel behavior; manifest.align.json absent."""
    script = _make_script([
        {"group_id": 7, "panel_texts": ["[tense] Alpha.", "[calm] Beta."]},
    ])

    per_panel_calls: list[str] = []

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        per_panel_calls.append(os.path.basename(out_path))
        _write_silence_wav(out_path)

    index = lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=lambda p: 2.0,
        group_mode=False,
    )

    # Per-panel clips: g0007_p00.wav and g0007_p01.wav (via attempt files then rename)
    clips_dir = tmp_path / "clips"
    assert (clips_dir / "g0007_p00.wav").exists(), "per-panel clip g0007_p00 must exist"
    assert (clips_dir / "g0007_p01.wav").exists(), "per-panel clip g0007_p01 must exist"

    # No group clip written
    assert not (clips_dir / "g0007.wav").exists(), "group clip must NOT exist in per-panel mode"

    # No manifest.align.json
    assert not (tmp_path / "manifest.align.json").exists(), (
        "manifest.align.json must NOT be written in per-panel mode"
    )

    # tts_index has per-panel audio_file entries
    clips = index["clips"]
    assert len(clips) == 2
    assert clips[0]["audio_file"] == "clips/g0007_p00.wav"
    assert clips[1]["audio_file"] == "clips/g0007_p01.wav"


# ---------------------------------------------------------------------------
# Test 4: align integration — stub returns known offsets → reflected in manifest
# ---------------------------------------------------------------------------

def test_group_mode_align_integration_known_offsets(tmp_path):
    """Stub align returns known offsets; manifest.align.json reflects them exactly."""
    known_offsets_g1 = [(0.0, 3.5), (3.5, 7.0)]
    known_offsets_g2 = [(0.0, 5.0)]

    call_order: list[str] = []

    def stub_align_ordered(panel_lines, clip_dur_sec, *, transcribe_fn=None, clip_path=None):
        if len(panel_lines) == 2:
            call_order.append("g1")
            return [
                {"start_sec": 0.0,  "end_sec": 3.5, "method": "asr"},
                {"start_sec": 3.5, "end_sec": 7.0, "method": "asr"},
            ]
        else:
            call_order.append("g2")
            return [{"start_sec": 0.0, "end_sec": 5.0, "method": "asr"}]

    script = _make_script([
        {"group_id": 1, "panel_texts": ["[tense] First.", "[calm] Second."]},
        {"group_id": 2, "panel_texts": ["[intense] Third."]},
    ])

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        dur = 7.0 if "First" in text else 5.0
        _write_silence_wav(out_path, duration_sec=dur)

    def stub_dur(path: str) -> float:
        name = os.path.basename(path)
        return 7.0 if name == "g0001.wav" else 5.0

    lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=stub_dur,
        group_mode=True,
        align_fn=stub_align_ordered,
    )

    align = json.loads((tmp_path / "manifest.align.json").read_text())
    segs = {s["segment_id"]: s for s in align["segments"]}

    # g0001 panel 0
    assert segs["g0001_p00"]["start_sec"] == pytest.approx(0.0)
    assert segs["g0001_p00"]["end_sec"] == pytest.approx(3.5)
    assert segs["g0001_p00"]["method"] == "asr"

    # g0001 panel 1
    assert segs["g0001_p01"]["start_sec"] == pytest.approx(3.5)
    assert segs["g0001_p01"]["end_sec"] == pytest.approx(7.0)
    assert segs["g0001_p01"]["method"] == "asr"

    # g0002 panel 0
    assert segs["g0002_p00"]["start_sec"] == pytest.approx(0.0)
    assert segs["g0002_p00"]["end_sec"] == pytest.approx(5.0)
    assert segs["g0002_p00"]["method"] == "asr"


# ---------------------------------------------------------------------------
# Test 5: align fallback to proportional → method="proportional" recorded
# ---------------------------------------------------------------------------

def test_group_mode_align_proportional_fallback_recorded(tmp_path):
    """When align_fn returns proportional splits, method='proportional' in manifest."""
    script = _make_script([
        {"group_id": 3, "panel_texts": ["[tense] A.", "[calm] B.", "[dramatic] C."]},
    ])

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        _write_silence_wav(out_path, duration_sec=9.0)

    lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=lambda p: 9.0,
        group_mode=True,
        align_fn=_stub_align,   # always returns proportional
    )

    align = json.loads((tmp_path / "manifest.align.json").read_text())
    segs = align["segments"]
    assert len(segs) == 3
    for seg in segs:
        assert seg["method"] == "proportional", (
            f"expected proportional, got {seg['method']} for {seg['segment_id']}"
        )


# ---------------------------------------------------------------------------
# Test 6: invariant — every segment_id appears exactly once in manifest.align.json
# ---------------------------------------------------------------------------

def test_group_mode_every_segment_id_appears_once(tmp_path):
    """Every segment_id in the script appears exactly once in manifest.align.json."""
    script = _make_script([
        {"group_id": 1, "panel_texts": ["[tense] A.", "[calm] B."]},
        {"group_id": 2, "panel_texts": ["[intense] C."]},
        {"group_id": 3, "panel_texts": ["[dramatic] D.", "[explosive] E.", "[somber] F."]},
    ])

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        _write_silence_wav(out_path, duration_sec=6.0)

    lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=lambda p: 6.0,
        group_mode=True,
        align_fn=_stub_align,
    )

    align = json.loads((tmp_path / "manifest.align.json").read_text())
    seg_ids = [s["segment_id"] for s in align["segments"]]

    # 6 panels total across 3 groups
    assert len(seg_ids) == 6, f"expected 6 segment entries, got {len(seg_ids)}: {seg_ids}"

    # No duplicates
    assert len(seg_ids) == len(set(seg_ids)), f"duplicate segment_ids: {seg_ids}"

    # All expected IDs present (note: paragraph index is within-section, not global)
    for sid in seg_ids:
        assert sid.startswith("g00"), f"malformed segment_id: {sid}"


# ---------------------------------------------------------------------------
# Test 7: env default — STUDIO_TTS_GROUP_SYNTH=1 enables group mode
# ---------------------------------------------------------------------------

def test_env_default_group_synth_on(tmp_path, monkeypatch):
    """STUDIO_TTS_GROUP_SYNTH='1' → group mode active (explicit opt-in)."""
    monkeypatch.setenv("STUDIO_TTS_GROUP_SYNTH", "1")

    script = _make_script([
        {"group_id": 5, "panel_texts": ["[calm] Solo panel."]},
    ])

    group_clips_written: list[str] = []

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        group_clips_written.append(os.path.basename(out_path))
        _write_silence_wav(out_path)

    # Don't pass group_mode explicitly — rely on env
    lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=lambda p: 2.0,
        align_fn=_stub_align,
    )

    clips_dir = tmp_path / "clips"
    assert (clips_dir / "g0005.wav").exists(), (
        "STUDIO_TTS_GROUP_SYNTH=1 must activate group mode → g0005.wav"
    )
    assert not (clips_dir / "g0005_p00.wav").exists()


def test_env_off_group_synth_disabled(tmp_path, monkeypatch):
    """STUDIO_TTS_GROUP_SYNTH='0' → per-panel mode (old behavior)."""
    monkeypatch.setenv("STUDIO_TTS_GROUP_SYNTH", "0")

    script = _make_script([
        {"group_id": 6, "panel_texts": ["[tense] Panel one.", "[calm] Panel two."]},
    ])

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        _write_silence_wav(out_path)

    lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=lambda p: 2.0,
    )

    clips_dir = tmp_path / "clips"
    assert (clips_dir / "g0006_p00.wav").exists()
    assert (clips_dir / "g0006_p01.wav").exists()
    assert not (clips_dir / "g0006.wav").exists()
    assert not (tmp_path / "manifest.align.json").exists()


# ---------------------------------------------------------------------------
# Test 9: group clips carry text_sha so the audio<->narration gate stays fresh
# ---------------------------------------------------------------------------

def test_group_clip_text_sha_passes_audio_consistency_gate(tmp_path):
    """Regression for the first group-mode Ch1 render failure (BLOCKING
    ``audio_stale``).

    A group clip's ``segment_id`` is the group label (``g####``); prep-QA's
    ``audio_consistency`` compares its stored ``text_sha`` against
    ``narration_sha(plan_item["tts_text"])``. The planner derives that tts_text
    by joining the group's RAW script paragraphs, so the clip MUST store
    ``narration_sha`` of the same raw join. Earlier the group clip had no
    ``text_sha`` → ``_clip_sha`` returned None → every group read as stale →
    render blocked.

    Asserts both halves of the contract:
      1. stored text_sha == narration_sha(raw joined panel source), and
      2. audio_consistency() reports the group segment FRESH for a plan whose
         group tts_text is that same raw join (and correctly STALE on drift, so
         the gate is a real fingerprint — not a constant that always matches).
    """
    raw_texts = [
        "[tense] The gate groans open.",
        "[calm] Nothing stirs in the dark.",
        "[dramatic] Then the blade descends.",
    ]
    script = _make_script([{"group_id": 1, "panel_texts": raw_texts}])

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        _write_silence_wav(out_path, duration_sec=6.0)

    index = lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=lambda p: 6.0,
        group_mode=True,
        align_fn=_stub_align,
    )

    clips = {c["segment_id"]: c for c in index["clips"]}
    assert "g0001" in clips, f"expected group clip g0001, got {list(clips)}"
    group_clip = clips["g0001"]

    # 1. text_sha hashes the RAW joined panel source (same as the planner)
    expected_sha = nc.narration_sha(" ".join(raw_texts))
    assert group_clip.get("text_sha") == expected_sha, (
        "group clip text_sha must hash the RAW joined panel source "
        f"(expected {expected_sha!r}, got {group_clip.get('text_sha')!r})"
    )

    # 2. the gate sees a matching group plan as FRESH (the render would proceed)
    plan = {"timeline": [{"segment_id": "g0001", "tts_text": " ".join(raw_texts)}]}
    result = nc.audio_consistency(plan, index)
    assert result["fresh"] == ["g0001"], f"group seg must be fresh: {result}"
    assert result["stale"] == []
    assert result["missing"] == []

    # ...and drifted narration is correctly STALE (proves it's a real sha)
    drifted = {"timeline": [{"segment_id": "g0001",
                             "tts_text": "Wholly different narration words."}]}
    assert nc.audio_consistency(drifted, index)["stale"] == ["g0001"]


# ---------------------------------------------------------------------------
# Test 10: DEFAULT is per-panel — group mode must be explicit opt-in
# ---------------------------------------------------------------------------

def test_no_env_defaults_to_per_panel(tmp_path, monkeypatch):
    """With NO STUDIO_TTS_GROUP_SYNTH set, the default is PER-PANEL.

    Locks the 2026-06-22 revert: per-group joined-synth degraded voice and
    restructured the dashboard display, so it must never be the silent default
    again. No env + no explicit group_mode → per-panel clips, no align manifest.
    """
    monkeypatch.delenv("STUDIO_TTS_GROUP_SYNTH", raising=False)
    assert lt._group_mode_default() is False, "per-group must be OFF by default"

    script = _make_script([
        {"group_id": 9, "panel_texts": ["[tense] Alpha.", "[calm] Beta."]},
    ])

    def stub_synth(text: str, out_path: str, exaggeration: float) -> None:
        _write_silence_wav(out_path)

    # no group_mode arg, no env → must take the per-panel path
    lt.synthesize_manifest(
        script, str(tmp_path),
        backend="qwen", synth_fn=stub_synth,
        duration_fn=lambda p: 2.0,
    )

    clips_dir = tmp_path / "clips"
    assert (clips_dir / "g0009_p00.wav").exists(), "per-panel clip must exist by default"
    assert (clips_dir / "g0009_p01.wav").exists()
    assert not (clips_dir / "g0009.wav").exists(), "group clip must NOT exist by default"
    assert not (tmp_path / "manifest.align.json").exists(), "no align manifest in per-panel default"
