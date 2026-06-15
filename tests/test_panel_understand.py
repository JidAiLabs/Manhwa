"""panel_understand (Pass 1): per-panel understanding = full coverage by
construction. Tests the pure payload/record logic + the ordered loop with a
stubbed model call (no Gemma needed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "panel_understand",
    Path(__file__).resolve().parent.parent / "tools" / "panel_understand.py")
pu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pu)  # type: ignore[union-attr]


def test_build_payload_pulls_ocr_signals_and_rolling_context():
    panel = {"scene_file": "p5.jpg", "ocr_clean": "WHO ARE YOU?",
             "vision": {"labels": [{"desc": "sword"}], "objects": [{"name": "Person"}]}}
    p = pu.build_payload(panel, ["he draws his blade", "the train shakes"])
    assert p["scene_file"] == "p5.jpg"
    assert p["ocr"] == "WHO ARE YOU?"
    assert p["labels"] == ["sword"] and p["objects"] == ["Person"]
    assert p["previous_panels"] == ["he draws his blade", "the train shakes"]


def test_assemble_record_normalizes_and_flags_parse_failure():
    good = pu.assemble_record("p1.jpg", {
        "description": " A monster looms. ", "subjects": ["monster"],
        "action": "it roars", "dialogue": "ROAR", "setting": "train",
        "intensity": "EXPLOSIVE"})
    assert good["description"] == "A monster looms." and good["intensity"] == "explosive"
    assert "error" not in good
    bad = pu.assemble_record("p2.jpg", None)
    assert bad["error"] == "parse_failed" and bad["intensity"] == "unknown"
    # invalid intensity -> 'unknown', never crash
    assert pu.assemble_record("p3.jpg", {"intensity": "epic"})["intensity"] == "unknown"


def test_understand_panels_is_ordered_threads_context_and_covers_all():
    items = [{"scene_file": f"p{i}.jpg", "scene_path": f"/s/p{i}.jpg"} for i in range(3)]
    seen = []

    def stub(payload, image_path):
        seen.append((payload["scene_file"], list(payload["previous_panels"]), image_path))
        return {"description": f"desc {payload['scene_file']}", "action": "x",
                "intensity": "calm"}

    out = pu.understand_panels(items, stub)
    assert [r["scene_file"] for r in out] == ["p0.jpg", "p1.jpg", "p2.jpg"]  # full coverage
    # rolling context threads the prior descriptions, image path passed through
    assert seen[0][1] == [] and seen[1][1] == ["desc p0.jpg"]
    assert seen[2][1] == ["desc p0.jpg", "desc p1.jpg"] and seen[2][2] == "/s/p2.jpg"


def test_resume_skips_already_understood_panels():
    items = [{"scene_file": "p0.jpg", "scene_path": "a"},
             {"scene_file": "p1.jpg", "scene_path": "b"}]
    prior = {"p0.jpg": {"scene_file": "p0.jpg", "description": "kept", "intensity": "calm"}}
    calls = []

    def stub(payload, image_path):
        calls.append(payload["scene_file"])
        return {"description": "new", "action": "y", "intensity": "tense"}

    out = pu.understand_panels(items, stub, prior=prior)
    assert calls == ["p1.jpg"]                       # p0 reused, only p1 called
    assert out[0]["description"] == "kept" and out[1]["description"] == "new"
