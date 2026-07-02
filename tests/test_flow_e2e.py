"""
tests/test_flow_e2e.py

Adaptive flow narration, end to end on a tiny synthetic episode:
one group / 5 panels (one stamped system card); a stubbed writer returns
1 flow-span(3) + 2 solos. The REAL stages then run in sequence —
gemini_narrative_pass.main() (stubbed model) → script_expander subprocess
(gemini_verbatim) → timeline_planner subprocess (real wav clips) — and the
chain must hold: segments validate, one paragraph+shot per segment with the
span as scene_files, every panel paced >= the 2.0s floor under its segment's
clip, prep_qa's span cover check green, and the system card solo + shown.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import tools.gemini_narrative_pass as gnp

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"

_PQ_SPEC = importlib.util.spec_from_file_location(
    "prep_qa", TOOLS / "prep_qa.py")
pq = importlib.util.module_from_spec(_PQ_SPEC)
_PQ_SPEC.loader.exec_module(pq)  # type: ignore[union-attr]


FILES = ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg", "sys5.jpg"]
SYSTEM_FILE = "sys5.jpg"
SPAN = FILES[:3]

FLOW_LINE = ("He drops through the canopy and the ravine keeps taking him, "
             "branch after branch, stone after stone, until the dark floor "
             "finally catches his broken body and holds it still.")
SOLO_LINE = "The stranger's eyes snap open in the dark."
SYS_LINE = "A cold blue window declares the activation complete."

MODEL_BEAT = {
    "beat_title": "The Fall", "what_happens": "He falls; the system wakes.",
    "narration": "model join placeholder",
    "segments": [
        {"span": list(SPAN), "line": FLOW_LINE},
        {"span": ["p4.jpg"], "line": SOLO_LINE},
        {"span": [SYSTEM_FILE], "line": SYS_LINE},
    ],
    "scene_selection": [],
}


def _make_wav(path: Path, duration_sec: float, framerate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * int(duration_sec * framerate))


def _write_manifests(tmp_path: Path):
    groups = {"shots": [{"shot_id": 1, "group_id": 1,
                         "scene_files": list(FILES),
                         "segment": "present",
                         "arc_label": "opening", "intensity": "tense"}]}
    vision = {"items": [{"scene_file": f, "ocr_clean": "", "vision": {}}
                        for f in FILES]}
    understood = {"panels": [
        {"scene_file": f,
         "description": f"A figure moves near {f}.",
         "action": f"He crosses toward {f}.",
         "panel_kind": "system" if f == SYSTEM_FILE else "story",
         "intensity": "tense", "subjects": ["the prince"]} for f in FILES]}
    (tmp_path / "groups.json").write_text(json.dumps(groups))
    (tmp_path / "vision.json").write_text(json.dumps(vision))
    (tmp_path / "understood.json").write_text(json.dumps(understood))


def _run_beats(tmp_path: Path, monkeypatch) -> dict:
    out = tmp_path / "beats.json"

    def stub(**kw):
        return dict(MODEL_BEAT), "raw", {"input": 1, "output": 1, "cached": 0}

    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    monkeypatch.delenv("STUDIO_NARR_SEGMENTATION", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "gemini_narrative_pass.py",
        "--groups-manifest", str(tmp_path / "groups.json"),
        "--vision-manifest", str(tmp_path / "vision.json"),
        "--understood", str(tmp_path / "understood.json"),
        "--out", str(out), "--backend", "ollama", "--min-sleep", "0"])
    assert gnp.main() == 0
    return json.loads(out.read_text())


def _run_script(tmp_path: Path) -> dict:
    out = tmp_path / "script.json"
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    r = subprocess.run(
        [sys.executable, str(TOOLS / "script_expander.py"),
         "--beats", str(tmp_path / "beats.json"), "--out", str(out),
         "--narration-source", "gemini_verbatim"],
        capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"expander failed:\n{r.stdout}\n{r.stderr}"
    return json.loads(out.read_text())


def _run_planner(tmp_path: Path, script: dict) -> dict:
    clips = tmp_path / "clips"
    clips.mkdir()
    durs = {"g0001_p00": 9.0, "g0001_p01": 4.0, "g0001_p02": 3.0}
    entries = []
    for sh in script["sections"][0]["shots"]:
        seg = sh["segment_id"]
        wav = clips / f"{seg}.wav"
        _make_wav(wav, durs[seg])
        entries.append({"segment_id": seg, "group_id": 1,
                        "audio_file": str(wav.relative_to(tmp_path)),
                        "duration_sec": durs[seg]})
    (tmp_path / "tts_index.json").write_text(json.dumps({"clips": entries}))
    out = tmp_path / "plan.json"
    r = subprocess.run(
        [sys.executable, str(TOOLS / "timeline_planner.py"),
         "--groups", str(tmp_path / "groups.json"),
         "--script", str(tmp_path / "script.json"),
         "--tts-index", str(tmp_path / "tts_index.json"),
         "--beats", str(tmp_path / "beats.json"),
         "--vision", str(tmp_path / "vision.json"),
         "--out", str(out), "--mode", "narrated", "--audio-pad-sec", "0"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"planner failed:\n{r.stdout}\n{r.stderr}"
    return json.loads(out.read_text())


def test_flow_narration_end_to_end(tmp_path, monkeypatch):
    _write_manifests(tmp_path)

    # 1. beats: the writer's segments validate and land unchanged
    beats_obj = _run_beats(tmp_path, monkeypatch)
    beat = beats_obj["beats"][0]
    kinds = {f: ("system" if f == SYSTEM_FILE else "story") for f in FILES}
    assert gnp.validate_segments(beat["segments"], FILES, kinds) == []
    assert [s["span"] for s in beat["segments"]] == [
        SPAN, ["p4.jpg"], [SYSTEM_FILE]]
    assert beat["narration"] == " ".join(
        s["line"] for s in beat["segments"])           # load-bearing join

    # 2. script: ONE paragraph + ONE shot per segment, scene_files = span
    script = _run_script(tmp_path)
    sec = script["sections"][0]
    assert len(sec["script_paragraphs"]) == 3
    assert len(sec["tts_paragraphs_v3"]) == 3
    shots = sec["shots"]
    assert [sh["segment_id"] for sh in shots] == [
        "g0001_p00", "g0001_p01", "g0001_p02"]
    assert [sh["scene_files"] for sh in shots] == [
        SPAN, ["p4.jpg"], [SYSTEM_FILE]]

    # 3. planner: the flow span is multi_cut; EVERY panel >= the 2.0s floor,
    #    each span's cuts tile its clip exactly
    plan = _run_planner(tmp_path, script)
    story = [it for it in plan["timeline"] if not it.get("branding")]
    assert [it["segment_id"] for it in story] == [
        "g0001_p00", "g0001_p01", "g0001_p02"]
    span_item = story[0]
    assert span_item["display_strategy"] == "multi_cut"
    assert [c["file"] for c in span_item["cuts"]] == SPAN
    for it in story:
        for c in it["cuts"]:
            assert float(c["dur"]) >= 2.0, (it["segment_id"], c)
        tiled = sum(float(c["dur"]) for c in it["cuts"])
        assert abs(tiled - float(it["duration_sec"])) < 1e-3

    # 4. prep_qa cover check: spans partition the shown panels
    assert pq.span_cover_flags(plan, beats_obj) == []

    # 5. the system card is a SOLO segment and actually on screen
    sys_item = story[2]
    assert [c["file"] for c in sys_item["cuts"]] == [SYSTEM_FILE]
    shown = {c["file"] for it in story for c in it["cuts"]}
    assert shown == set(FILES)                         # all 5 panels shown
