"""tests/test_provenance_e2e.py

Cross-stage provenance invalidation chains for the dependency-authority
refactor (studio/deps.py) — driven through the REAL producer/consumer
functions of each stage, not mocks of the thing under test. Each test proves
one link: a content change upstream must be visible to (and rejected /
recomputed / invalidated by) the next stage down, end to end.

Some of these chains already carry dedicated unit coverage in their own
stage's test file (tests/dashboard/test_gates.py, tests/test_pipeline.py,
tests/test_local_tts.py); they are repeated here, driven through real file
writes, as the one place that documents the full cross-stage story together.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# 1a. Scene bytes change -> panel_understand resume rejects the stale cache
# ---------------------------------------------------------------------------

def test_scene_bytes_change_rejects_panel_understand_resume(tmp_path):
    """A genuine --resume round trip: understand a real scene file, persist the
    result to manifest.panels.understood.json (stamped scene_sha), reload it as
    main()'s --resume would, then mutate the scene's bytes underneath the same
    filename. The content-keyed acceptance check must re-run that panel instead
    of reusing the old description."""
    pu = _load_tool("panel_understand")
    scene = tmp_path / "p000001.jpg"
    scene.write_bytes(b"ORIGINAL PIXELS")
    items = [{"scene_file": "p000001.jpg", "scene_path": str(scene)}]
    calls = []

    def call_fn(payload, image_path):
        calls.append(payload["scene_file"])
        return {"description": "a hero stands", "action": "stands",
                "intensity": "calm", "panel_kind": "story"}

    first = pu.understand_panels(items, call_fn)
    assert calls == ["p000001.jpg"]

    understood_path = tmp_path / "manifest.panels.understood.json"
    understood_path.write_text(json.dumps({"panels": first}))

    # mirrors main()'s own --resume prior-loading (it lives inside argparse
    # gated main(), so the dict comprehension is reproduced here verbatim).
    prior = {p.get("scene_file"): p for p in
             json.loads(understood_path.read_text()).get("panels") or []
             if p.get("scene_file")}
    assert prior["p000001.jpg"]["scene_sha"] == pu._scene_sha(str(scene))

    # sanity: unchanged bytes -> resume ACCEPTS the cached record
    calls.clear()
    again = pu.understand_panels(items, call_fn, prior=prior)
    assert calls == []
    assert again[0]["description"] == "a hero stands"

    # the art changes underneath the SAME filename
    scene.write_bytes(b"COMPLETELY DIFFERENT PIXELS, NEW ART")
    calls.clear()
    out = pu.understand_panels(items, call_fn, prior=prior)
    assert calls == ["p000001.jpg"]            # resume REJECTED the stale sha
    assert out[0]["scene_sha"] != prior["p000001.jpg"]["scene_sha"]


# ---------------------------------------------------------------------------
# 1b. Script edit after voice approval -> gates.voice_allowed goes False
# ---------------------------------------------------------------------------

def test_script_edit_after_voice_approval_invalidates_voice_gate(tmp_path):
    """content_sha binds a voice approval to the SCRIPT BYTES: healing or
    regenerating the narration after approval must invalidate it — the
    confirm-upstream-before-expensive-downstream contract (never voice a
    script nobody actually read)."""
    from studio.catalog.db import connect
    from studio.dashboard import gates

    con = connect(tmp_path / "s.db")
    ep = tmp_path / "ep"
    ep.mkdir()
    script = ep / "manifest.script.json"
    script.write_text(json.dumps({"paragraphs": ["original line"]}))

    gates.approve(con, "voice", chapter_id=1,
                  content_sha=gates.gate_sha("voice", ep))
    assert gates.voice_allowed(con, 1, ep) == (True, "")

    script.write_text(json.dumps({"paragraphs": ["healed line"]}))  # re-narrated
    allowed, why = gates.voice_allowed(con, 1, ep)
    assert allowed is False and "approval" in why


# ---------------------------------------------------------------------------
# 1c. Sanitize marker older than the script -> _stage_voiced recomputes it
# ---------------------------------------------------------------------------

def _pipeline_cfg(tmp_path):
    from studio.config import Config
    return replace(Config(sites={}, yolo_weights=tmp_path / "fake.pt",
                          detect_backend="yolo", 
                          beats_backend="ollama"),
                  tts_backend="chatterbox")  # local backend: no ElevenLabs cred wall


def test_stale_sanitize_marker_triggers_recompute_before_voicing(tmp_path, monkeypatch):
    import os
    import studio.pipeline as pipeline_mod

    ep = tmp_path / "ep"
    ep.mkdir()
    script = ep / "manifest.script.json"
    script.write_text("{}")
    marker = ep / "manifest.sanitize.json"
    marker.write_text(json.dumps({"unresolved_blocks": []}))
    old = script.stat().st_mtime - 100
    os.utime(marker, (old, old))          # marker now predates the script -> stale

    reruns = []

    def fake_rerun(ep_dir, cfg, p):
        reruns.append(p["script"])
        marker.write_text(json.dumps({"unresolved_blocks": []}))  # rerun -> clean

    monkeypatch.setattr(pipeline_mod, "_run_sanitize_pass", fake_rerun)
    monkeypatch.setattr(pipeline_mod, "_run_tool", lambda *a, **k: None)  # TTS dispatch, not under test

    pipeline_mod._stage_voiced(ep, _pipeline_cfg(tmp_path))   # must not raise
    assert reruns == [script]


# ---------------------------------------------------------------------------
# 1d. voice_ref content change -> local_tts bulk-invalidates every cached clip
# ---------------------------------------------------------------------------

def _tts_script():
    return {"sections": [{
        "section_index": 0,
        "tts_paragraphs_v3": ["[tense] The blade falls.", "[calm] Silence settles."],
        "script_paragraphs": ["The blade falls.", "Silence settles."],
        "shots": [{"group_id": 1, "beat_id": 1}, {"group_id": 2, "beat_id": 2}],
    }]}


def _tts_synth_write(calls):
    def _fn(text, out_path, exaggeration):
        calls.append(out_path)
        with open(out_path, "wb") as f:
            f.write(b"RIFFXXXXWAVE")
    return _fn


def test_voice_ref_content_change_bulk_invalidates_tts_cache(tmp_path):
    lt = _load_tool("local_tts_from_manifest")
    ref = tmp_path / "narrator_ref.wav"
    ref.write_bytes(b"ORIGINAL_VOICE_BYTES")

    idx = lt.synthesize_manifest(
        _tts_script(), str(tmp_path), backend="qwen-mlx", voice_ref=str(ref),
        synth_fn=_tts_synth_write([]), duration_fn=lambda p: 1.0, group_mode=False)
    (tmp_path / "tts_index.json").write_text(json.dumps(idx))
    assert idx.get("voice_ref_sha")             # stamped on write

    ref.write_bytes(b"DIFFERENT_VOICE_BYTES_ENTIRELY")   # re-recorded narrator ref
    calls = []
    out = lt.synthesize_manifest(
        _tts_script(), str(tmp_path), backend="qwen-mlx", voice_ref=str(ref),
        synth_fn=_tts_synth_write(calls), duration_fn=lambda p: 1.0, group_mode=False)
    assert len(calls) == 2                       # BOTH clips re-synthesized
    assert all(c["cached"] is False for c in out["clips"])


# ---------------------------------------------------------------------------
# 1e. Corrupt beats/plan manifests hard-error instead of silently degrading
# ---------------------------------------------------------------------------

def test_corrupt_beats_manifest_hard_errors_timeline_planner(tmp_path, monkeypatch):
    tp = _load_tool("timeline_planner")
    mio = sys.modules["manifest_io"]   # the exact module tp's own import bound

    (tmp_path / "manifest.groups.json").write_text(json.dumps({"groups": []}))
    beats = tmp_path / "manifest.beats.json"
    beats.write_text("{not valid json")           # torn write

    monkeypatch.setattr(sys, "argv", [
        "timeline_planner.py",
        "--groups", str(tmp_path / "manifest.groups.json"),
        "--beats", str(beats),
        "--out", str(tmp_path / "render.plan.json")])
    with pytest.raises(mio.ManifestError):
        tp.main()


def test_corrupt_plan_hard_errors_prep_qa(tmp_path, monkeypatch):
    pq = _load_tool("prep_qa")
    mio = sys.modules["manifest_io"]

    ep = tmp_path / "ep"
    ep.mkdir()                                     # no sentinel manifests at all
    bad_plan = tmp_path / "render.plan.clean.json"
    bad_plan.write_text("{not valid json")         # torn write

    monkeypatch.setattr(sys, "argv", [
        "prep_qa.py", "--episode-dir", str(ep), "--plan", str(bad_plan)])
    with pytest.raises(mio.ManifestError):
        pq.main()
