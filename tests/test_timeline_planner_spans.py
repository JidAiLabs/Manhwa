"""
tests/test_timeline_planner_spans.py

Chunk 2 (adaptive flow narration): the planner routes a flow-SPAN shot
(script shot with several scene_files) through the EXISTING `multi_cut`
pacing — the clip's real duration is spread across the span's panels, every
cut >= the 2.0s floor, no gap, no drift. Single-file shots keep today's path
byte-for-byte, protected system cards inside a span survive the redundant
verdict, and a span file dropped upstream (caption filter) reallocates the
full clip duration across the survivors.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLANNER = REPO / "tools" / "timeline_planner.py"

_RP_SPEC = importlib.util.spec_from_file_location(
    "render_prep", REPO / "tools" / "render_prep.py")
rp = importlib.util.module_from_spec(_RP_SPEC)
_RP_SPEC.loader.exec_module(rp)  # type: ignore[union-attr]

SPAN = ["p001.jpg", "p002.jpg", "p003.jpg"]
SOLO = "p004.jpg"
FLOW_LINE = ("He drops through the canopy, bounces off two branches, and "
             "lands in the one spot the assassins forgot to watch.")
SOLO_LINE = "The system window blinks awake."


def _make_wav(path: Path, duration_sec: float, framerate: int = 16000) -> None:
    n_frames = int(duration_sec * framerate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _build_fixtures(tmp_path: Path, *, span_dur: float = 10.5,
                    solo_dur: float = 4.0, beats: dict | None = None,
                    vision: dict | None = None) -> dict:
    """One group, 4 panels; script shot g0001_p00 spans 3 panels, g0001_p01 is
    a solo — per-SEGMENT TTS clips (no manifest.align.json: per-segment path)."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    p00_wav = clips_dir / "g0001_p00.wav"
    p01_wav = clips_dir / "g0001_p01.wav"
    _make_wav(p00_wav, span_dur)
    _make_wav(p01_wav, solo_dur)

    groups = {"shots": [{"group_id": 1, "shot_id": 1,
                         "scene_files": SPAN + [SOLO],
                         "segment": "present"}]}
    script = {"sections": [{
        "section_index": 0,
        "script_paragraphs": [{"text": FLOW_LINE}, {"text": SOLO_LINE}],
        "shots": [
            {"group_id": 1, "segment_id": "g0001_p00",
             "scene_files": list(SPAN), "fallback_scene_files": []},
            {"group_id": 1, "segment_id": "g0001_p01",
             "scene_files": [SOLO], "fallback_scene_files": []},
        ],
    }]}
    tts_index = {"clips": [
        {"segment_id": "g0001_p00", "group_id": 1,
         "audio_file": str(p00_wav.relative_to(tmp_path)),
         "duration_sec": span_dur},
        {"segment_id": "g0001_p01", "group_id": 1,
         "audio_file": str(p01_wav.relative_to(tmp_path)),
         "duration_sec": solo_dur},
    ]}

    (tmp_path / "groups.json").write_text(json.dumps(groups))
    (tmp_path / "script.json").write_text(json.dumps(script))
    (tmp_path / "tts_index.json").write_text(json.dumps(tts_index))
    fx = {
        "groups": str(tmp_path / "groups.json"),
        "script": str(tmp_path / "script.json"),
        "tts_index": str(tmp_path / "tts_index.json"),
        "tmp": tmp_path,
    }
    if beats is not None:
        (tmp_path / "beats.json").write_text(json.dumps(beats))
        fx["beats"] = str(tmp_path / "beats.json")
    if vision is not None:
        (tmp_path / "vision.json").write_text(json.dumps(vision))
        fx["vision"] = str(tmp_path / "vision.json")
    return fx


def _run_planner(fx: dict, out: Path, *extra: str) -> dict:
    cmd = [sys.executable, str(PLANNER),
           "--groups", fx["groups"],
           "--script", fx["script"],
           "--tts-index", fx["tts_index"],
           "--out", str(out),
           "--mode", "narrated",
           "--audio-pad-sec", "0"]
    if fx.get("beats"):
        cmd += ["--beats", fx["beats"]]
    if fx.get("vision"):
        cmd += ["--vision", fx["vision"]]
    cmd += list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"planner failed:\n{r.stderr}\n{r.stdout}"
    return json.loads(out.read_text())


def _story_items(plan: dict) -> list:
    return [it for it in plan["timeline"] if not it.get("branding")]


# ---- span > 1 → multi_cut over the span --------------------------------------

def test_span_shot_multi_cut_three_cuts_floor_and_full_duration(tmp_path):
    plan = _run_planner(_build_fixtures(tmp_path), tmp_path / "plan.json")
    span_item = _story_items(plan)[0]
    assert span_item["segment_id"] == "g0001_p00"
    assert span_item["display_strategy"] == "multi_cut"
    cuts = span_item["cuts"]
    assert [c["file"] for c in cuts] == SPAN          # all 3, span order
    assert all(c["dur"] >= 2.0 for c in cuts)         # per-panel floor
    total = sum(c["dur"] for c in cuts)
    assert abs(total - 10.5) < 1e-3                   # sum == clip duration
    assert abs(span_item["duration_sec"] - total) < 1e-3


def test_single_file_shot_unchanged(tmp_path):
    plan = _run_planner(_build_fixtures(tmp_path), tmp_path / "plan.json")
    solo_item = _story_items(plan)[1]
    assert solo_item["segment_id"] == "g0001_p01"
    cuts = solo_item["cuts"]
    assert len(cuts) == 1
    assert cuts[0]["file"] == SOLO
    assert abs(cuts[0]["dur"] - 4.0) < 1e-3
    assert abs(solo_item["duration_sec"] - 4.0) < 1e-3


def test_span_forces_multi_cut_even_under_single_hold_default(tmp_path):
    """--default-display single_hold must not collapse a span to its head —
    that would silently drop narrated panels (panel_uncovered)."""
    plan = _run_planner(_build_fixtures(tmp_path), tmp_path / "plan.json",
                        "--default-display", "single_hold")
    span_item, solo_item = _story_items(plan)[:2]
    assert span_item["display_strategy"] == "multi_cut"
    assert [c["file"] for c in span_item["cuts"]] == SPAN
    # the 1-file shot honors the operator's single_hold exactly as today
    assert solo_item["display_strategy"] == "single_hold"
    assert [c["file"] for c in solo_item["cuts"]] == [SOLO]


def test_span_head_is_primary_and_in_protection_set(tmp_path):
    """primary_scene_file = span head → narrated_files_from_plan protects it
    against the dedup passes (the 2.1 span-head contract, end to end)."""
    plan = _run_planner(_build_fixtures(tmp_path), tmp_path / "plan.json")
    span_item = _story_items(plan)[0]
    assert span_item["primary_scene_file"] == SPAN[0]
    assert SPAN[0] in rp.narrated_files_from_plan(plan)


# ---- protected system card inside a span ------------------------------------

def test_protected_system_card_in_span_survives_redundant_verdict(tmp_path):
    """A span panel stamped panel_kind='story' (system/info card) is protected:
    the beats LLM's 'redundant' role must not drop it from the span's cuts."""
    beats = {"beats": [{"group_id": 1, "scene_selection": [
        {"scene_file": "p001.jpg", "role": "keep", "intensity": "calm"},
        {"scene_file": "p002.jpg", "role": "redundant", "intensity": "calm"},
        {"scene_file": "p003.jpg", "role": "keep", "intensity": "calm"},
        {"scene_file": SOLO, "role": "keep", "intensity": "calm"},
    ]}]}
    vision = {"items": [{"scene_file": "p002.jpg", "panel_kind": "story"}]}
    fx = _build_fixtures(tmp_path, beats=beats, vision=vision)
    plan = _run_planner(fx, tmp_path / "plan.json")
    span_item = _story_items(plan)[0]
    files = [c["file"] for c in span_item["cuts"]]
    assert files == SPAN, f"protected p002.jpg must stay in the span: {files}"
    assert all(c["dur"] >= 2.0 for c in span_item["cuts"])


def test_unprotected_redundant_span_panel_dropped_duration_reallocated(tmp_path):
    """Without protection, the existing redundant-drop applies inside a span:
    the surviving panels absorb the FULL clip duration (no gap)."""
    beats = {"beats": [{"group_id": 1, "scene_selection": [
        {"scene_file": "p001.jpg", "role": "keep", "intensity": "calm"},
        {"scene_file": "p002.jpg", "role": "redundant", "intensity": "calm"},
        {"scene_file": "p003.jpg", "role": "keep", "intensity": "calm"},
        {"scene_file": SOLO, "role": "keep", "intensity": "calm"},
    ]}]}
    fx = _build_fixtures(tmp_path, beats=beats)
    plan = _run_planner(fx, tmp_path / "plan.json")
    span_item = _story_items(plan)[0]
    files = [c["file"] for c in span_item["cuts"]]
    assert files == ["p001.jpg", "p003.jpg"]
    total = sum(c["dur"] for c in span_item["cuts"])
    assert abs(total - 10.5) < 1e-3
    assert all(c["dur"] >= 2.0 for c in span_item["cuts"])


# ---- span file dropped upstream (caption filter) -----------------------------

def test_dropped_span_file_reallocates_full_clip_across_survivors(tmp_path):
    """spec 3.5: visual drops inside a span shrink the CUT list — narration
    untouched. A span panel filtered by the existing caption path (panel_kind
    'caption' in vision) yields 2 cuts, full clip duration, no sub-floor cut,
    no gap."""
    vision = {"items": [{"scene_file": "p002.jpg", "panel_kind": "caption"}]}
    fx = _build_fixtures(tmp_path, vision=vision)
    plan = _run_planner(fx, tmp_path / "plan.json")
    span_item = _story_items(plan)[0]
    assert span_item["segment_id"] == "g0001_p00"
    files = [c["file"] for c in span_item["cuts"]]
    assert files == ["p001.jpg", "p003.jpg"], files
    cuts = span_item["cuts"]
    assert all(c["dur"] >= 2.0 for c in cuts)          # no sub-floor cut
    total = sum(c["dur"] for c in cuts)
    assert abs(total - 10.5) < 1e-3                    # full duration, no gap
    assert abs(span_item["duration_sec"] - total) < 1e-3
    # contiguous tiling: each cut starts where the previous ended
    t = 0.0
    for c in cuts:
        assert abs(c["start"] - t) < 1e-3
        t += c["dur"]
