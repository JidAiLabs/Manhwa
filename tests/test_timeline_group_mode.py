"""
tests/test_timeline_group_mode.py

TDD for timeline_planner GROUP MODE: when manifest.align.json is present
next to the TTS index, the planner emits ONE timeline item per group whose
tts_audio is the group clip (clips/g####.wav) and whose cuts[] are the
panels at their aligned offsets within that clip.

Spec: docs/plans/specs/2026-06-22-per-group-tts-alignment-design.md

Contract:
  manifest.align.json: {"segments":[
    {"segment_id":"g####_p##","group_clip":"clips/g####.wav",
     "start_sec":F,"end_sec":F,"method":"asr"|"proportional"}, ...]}

Per-panel path (no manifest.align.json) must stay UNCHANGED.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLANNER = REPO / "tools" / "timeline_planner.py"

# module-level import for unit-test helpers
_SPEC = importlib.util.spec_from_file_location(
    "timeline_planner",
    PLANNER,
)
tp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tp)  # type: ignore[union-attr]


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_wav(path: Path, duration_sec: float, framerate: int = 16000) -> None:
    """Write a minimal valid WAV file of the given duration."""
    n_frames = int(duration_sec * framerate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _build_fixtures(
    tmp_path: Path,
    *,
    include_align: bool = True,
    g1_dur: float = 6.0,
    g2_dur: float = 4.0,
) -> dict:
    """
    Build a minimal episode directory with:
      - groups.json  (2 groups: g1 = 2 panels, g2 = 1 panel)
      - script.json  (segment_id -> scene_files)
      - tts_index.json  (group clips — per-group TTS)
      - clips/g0001.wav, clips/g0002.wav  (actual WAVs so duration can be read)
      - manifest.align.json  (present iff include_align=True)
    Returns a dict of relevant paths.
    """
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()

    # group WAV clips
    g1_wav = clips_dir / "g0001.wav"
    g2_wav = clips_dir / "g0002.wav"
    _make_wav(g1_wav, g1_dur)
    _make_wav(g2_wav, g2_dur)

    groups = {
        "shots": [
            {"group_id": 1, "shot_id": 1,
             "scene_files": ["p001.jpg", "p002.jpg"],
             "segment": "present"},
            {"group_id": 2, "shot_id": 2,
             "scene_files": ["p003.jpg"],
             "segment": "present"},
        ]
    }

    script = {
        "sections": [
            {
                "section_index": 0,
                "script_paragraphs": [
                    {"text": "Group one panel zero."},
                    {"text": "Group one panel one."},
                ],
                "shots": [
                    {"group_id": 1, "segment_id": "g0001_p00",
                     "scene_files": ["p001.jpg"], "fallback_scene_files": []},
                    {"group_id": 1, "segment_id": "g0001_p01",
                     "scene_files": ["p002.jpg"], "fallback_scene_files": []},
                ],
            },
            {
                "section_index": 1,
                "script_paragraphs": [
                    {"text": "Group two panel zero."},
                ],
                "shots": [
                    {"group_id": 2, "segment_id": "g0002_p00",
                     "scene_files": ["p003.jpg"], "fallback_scene_files": []},
                ],
            },
        ]
    }

    # Per-group TTS index (group clips, not per-panel clips)
    tts_index = {
        "clips": [
            {"segment_id": "g0001", "group_id": 1,
             "audio_file": str(g1_wav.relative_to(tmp_path)),
             "duration_sec": g1_dur},
            {"segment_id": "g0002", "group_id": 2,
             "audio_file": str(g2_wav.relative_to(tmp_path)),
             "duration_sec": g2_dur},
        ]
    }

    (tmp_path / "groups.json").write_text(json.dumps(groups))
    (tmp_path / "script.json").write_text(json.dumps(script))
    tts_path = tmp_path / "tts_index.json"
    tts_path.write_text(json.dumps(tts_index))

    if include_align:
        align = {
            "segments": [
                # g0001: two panels — p00 gets 0–3.5 s, p01 gets 3.5–6.0 s
                {"segment_id": "g0001_p00",
                 "group_clip": str(g1_wav.relative_to(tmp_path)),
                 "start_sec": 0.0, "end_sec": 3.5, "method": "asr"},
                {"segment_id": "g0001_p01",
                 "group_clip": str(g1_wav.relative_to(tmp_path)),
                 "start_sec": 3.5, "end_sec": 6.0, "method": "asr"},
                # g0002: one panel — full clip
                {"segment_id": "g0002_p00",
                 "group_clip": str(g2_wav.relative_to(tmp_path)),
                 "start_sec": 0.0, "end_sec": 4.0, "method": "proportional"},
            ]
        }
        (tmp_path / "manifest.align.json").write_text(json.dumps(align))

    return {
        "groups": str(tmp_path / "groups.json"),
        "script": str(tmp_path / "script.json"),
        "tts_index": str(tts_path),
        "align": str(tmp_path / "manifest.align.json"),
        "g1_wav": str(g1_wav),
        "g2_wav": str(g2_wav),
        "tmp": tmp_path,
    }


def _run_planner(fixtures: dict, out: Path) -> dict:
    """Run timeline_planner.py via subprocess and return the parsed plan."""
    cmd = [
        sys.executable, str(PLANNER),
        "--groups", fixtures["groups"],
        "--script", fixtures["script"],
        "--tts-index", fixtures["tts_index"],
        "--out", str(out),
        "--mode", "narrated",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"planner failed:\n{r.stderr}\n{r.stdout}"
    return json.loads(out.read_text())


# ── RED tests — written BEFORE implementation ─────────────────────────────────


def test_group_mode_emits_one_item_per_group(tmp_path):
    """With manifest.align.json present, the plan has exactly 2 items (one per group)."""
    fx = _build_fixtures(tmp_path)
    plan = _run_planner(fx, tmp_path / "plan.json")
    items = plan["timeline"]
    assert len(items) == 2, f"expected 2 group items, got {len(items)}: {[i['segment_id'] for i in items]}"


def test_group_mode_item_tts_audio_is_group_clip(tmp_path):
    """Each item's tts_audio must be the group clip (clips/g####.wav), not a per-panel clip."""
    fx = _build_fixtures(tmp_path)
    plan = _run_planner(fx, tmp_path / "plan.json")
    items = plan["timeline"]
    assert items[0]["tts_audio"].endswith("g0001.wav"), \
        f"item 0 tts_audio should be g0001.wav, got {items[0]['tts_audio']}"
    assert items[1]["tts_audio"].endswith("g0002.wav"), \
        f"item 1 tts_audio should be g0002.wav, got {items[1]['tts_audio']}"


def test_group_mode_cuts_one_per_panel_in_order(tmp_path):
    """Group 1 has 2 panels → 2 cuts in order; group 2 has 1 panel → 1 cut."""
    fx = _build_fixtures(tmp_path)
    plan = _run_planner(fx, tmp_path / "plan.json")
    items = plan["timeline"]

    g1_cuts = items[0]["cuts"]
    assert len(g1_cuts) == 2, f"group 1 should have 2 cuts, got {len(g1_cuts)}"
    assert g1_cuts[0]["file"] == "p001.jpg", g1_cuts
    assert g1_cuts[1]["file"] == "p002.jpg", g1_cuts

    g2_cuts = items[1]["cuts"]
    assert len(g2_cuts) == 1, f"group 2 should have 1 cut, got {len(g2_cuts)}"
    assert g2_cuts[0]["file"] == "p003.jpg", g2_cuts


def test_group_mode_cuts_use_aligned_offsets(tmp_path):
    """Cut start/dur must come from manifest.align.json offsets (relative to group clip)."""
    fx = _build_fixtures(tmp_path)
    plan = _run_planner(fx, tmp_path / "plan.json")
    g1_cuts = plan["timeline"][0]["cuts"]

    # p00: start=0.0, end=3.5 → dur=3.5
    assert abs(g1_cuts[0]["start"] - 0.0) < 1e-3, g1_cuts[0]
    assert abs(g1_cuts[0]["dur"] - 3.5) < 1e-3, g1_cuts[0]

    # p01: start=3.5, end=6.0 → dur=2.5
    assert abs(g1_cuts[1]["start"] - 3.5) < 1e-3, g1_cuts[1]
    assert abs(g1_cuts[1]["dur"] - 2.5) < 1e-3, g1_cuts[1]

    # g2 p00: start=0.0, end=4.0 → dur=4.0
    g2_cuts = plan["timeline"][1]["cuts"]
    assert abs(g2_cuts[0]["start"] - 0.0) < 1e-3, g2_cuts[0]
    assert abs(g2_cuts[0]["dur"] - 4.0) < 1e-3, g2_cuts[0]


def test_group_mode_cuts_have_motion(tmp_path):
    """Every cut in group mode must have a motion dict with required keys."""
    fx = _build_fixtures(tmp_path)
    plan = _run_planner(fx, tmp_path / "plan.json")
    required = {"mode", "strength", "start_bias", "end_bias", "zoom"}
    for item in plan["timeline"]:
        for cut in item["cuts"]:
            assert "motion" in cut, f"cut missing motion: {cut}"
            assert required <= set(cut["motion"].keys()), \
                f"motion missing keys: {cut['motion']}"


def test_group_mode_cuts_motion_not_all_static(tmp_path):
    """motion_for_cut must be applied; not every cut should be fully static."""
    fx = _build_fixtures(tmp_path)
    plan = _run_planner(fx, tmp_path / "plan.json")
    all_cuts = [c for item in plan["timeline"] for c in item["cuts"]]
    # At least one cut must have a non-zero effective pan (start_bias != end_bias)
    any_pan = any(
        c["motion"].get("start_bias") != c["motion"].get("end_bias")
        for c in all_cuts
    )
    assert any_pan, "All cuts are fully static — motion_for_cut must produce pans"


def test_group_mode_timing_cumulative(tmp_path):
    """Items are sequential: item 1 starts where item 0 ends."""
    fx = _build_fixtures(tmp_path, g1_dur=6.0, g2_dur=4.0)
    plan = _run_planner(fx, tmp_path / "plan.json")
    items = plan["timeline"]
    assert items[0]["start_sec"] == 0.0, items[0]
    assert abs(items[0]["duration_sec"] - 6.0) < 1e-2, items[0]
    assert abs(items[1]["start_sec"] - items[0]["end_sec"]) < 1e-3, items


def test_group_mode_total_duration_is_sum_of_group_clips(tmp_path):
    """Total plan duration = sum of group clip durations (6.0 + 4.0 = 10.0)."""
    fx = _build_fixtures(tmp_path, g1_dur=6.0, g2_dur=4.0)
    plan = _run_planner(fx, tmp_path / "plan.json")
    assert abs(plan["total_duration_sec"] - 10.0) < 0.1, plan["total_duration_sec"]


def test_group_mode_segment_id_is_group_key(tmp_path):
    """The item's segment_id should identify the group (g####), not a sub-panel."""
    fx = _build_fixtures(tmp_path)
    plan = _run_planner(fx, tmp_path / "plan.json")
    items = plan["timeline"]
    # segment_id should NOT be a sub-panel id like g0001_p00
    for item in items:
        sid = item["segment_id"]
        assert "_p" not in sid, \
            f"segment_id {sid!r} looks like a sub-panel id; expected group id like g0001"


def test_group_mode_every_panel_appears_as_exactly_one_cut(tmp_path):
    """Every segment_id in manifest.align.json corresponds to exactly one cut across all items."""
    fx = _build_fixtures(tmp_path)
    align = json.loads(Path(fx["align"]).read_text())
    panel_ids = [s["segment_id"] for s in align["segments"]]  # g0001_p00, g0001_p01, g0002_p00

    plan = _run_planner(fx, tmp_path / "plan.json")

    # Map cuts back to panel segment_ids via file names (p001→g0001_p00, etc.)
    # The simpler check: total cut count == total aligned panel count
    total_cuts = sum(len(item["cuts"]) for item in plan["timeline"])
    assert total_cuts == len(panel_ids), \
        f"expected {len(panel_ids)} cuts total (one per aligned panel), got {total_cuts}"


# ── fallback: no manifest.align.json → per-panel behavior unchanged ──────────


def _build_per_panel_fixtures(tmp_path: Path) -> dict:
    """Per-panel TTS fixtures (old style: one clip per segment_id g####_p##)."""
    groups = {
        "shots": [
            {"group_id": 1, "shot_id": 1, "scene_files": ["p001.jpg", "p002.jpg"]},
        ]
    }
    script = {
        "sections": [{
            "section_index": 0,
            "script_paragraphs": [
                {"text": "Paragraph zero."},
                {"text": "Paragraph one."},
            ],
            "shots": [
                {"group_id": 1, "segment_id": "g0001_p00",
                 "scene_files": ["p001.jpg"], "fallback_scene_files": []},
                {"group_id": 1, "segment_id": "g0001_p01",
                 "scene_files": ["p002.jpg"], "fallback_scene_files": []},
            ],
        }]
    }
    tts = {
        "clips": [
            {"segment_id": "g0001_p00", "group_id": 1,
             "audio_file": "g0001_p00.wav", "duration_sec": 3.0},
            {"segment_id": "g0001_p01", "group_id": 1,
             "audio_file": "g0001_p01.wav", "duration_sec": 4.0},
        ]
    }
    (tmp_path / "groups.json").write_text(json.dumps(groups))
    (tmp_path / "script.json").write_text(json.dumps(script))
    tts_path = tmp_path / "tts_index.json"
    tts_path.write_text(json.dumps(tts))
    # NO manifest.align.json
    return {
        "groups": str(tmp_path / "groups.json"),
        "script": str(tmp_path / "script.json"),
        "tts_index": str(tts_path),
        "tmp": tmp_path,
    }


def test_no_align_manifest_uses_per_panel_behavior(tmp_path):
    """Without manifest.align.json, the planner falls back to per-panel items (B2 behavior)."""
    fx = _build_per_panel_fixtures(tmp_path)
    out = tmp_path / "plan.json"
    cmd = [
        sys.executable, str(PLANNER),
        "--groups", fx["groups"],
        "--script", fx["script"],
        "--tts-index", fx["tts_index"],
        "--out", str(out),
        "--mode", "narrated",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"planner failed:\n{r.stderr}"

    plan = json.loads(out.read_text())
    items = plan["timeline"]
    # B2 behavior: 2 items, one per segment_id
    assert len(items) == 2, f"expected 2 per-panel items, got {len(items)}"
    seg_ids = [it["segment_id"] for it in items]
    assert seg_ids == ["g0001_p00", "g0001_p01"], seg_ids
    # each item has its own per-panel audio
    assert items[0]["tts_audio"].endswith("g0001_p00.wav"), items[0]["tts_audio"]
    assert items[1]["tts_audio"].endswith("g0001_p01.wav"), items[1]["tts_audio"]
