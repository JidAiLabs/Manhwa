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
        "intensity": "EXPLOSIVE", "panel_kind": "story"})
    assert good["description"] == "A monster looms." and good["intensity"] == "explosive"
    assert good["panel_kind"] == "story" and "error" not in good
    bad = pu.assemble_record("p2.jpg", None)
    assert bad["error"] == "parse_failed" and bad["intensity"] == "unknown"
    assert bad["panel_kind"] == "empty"          # unparsed -> filtered out of grouping
    # invalid intensity -> 'unknown', never crash
    assert pu.assemble_record("p3.jpg", {"intensity": "epic"})["intensity"] == "unknown"
    # chrome/empty/caption pass through; missing/invalid kind defaults to 'story'
    assert pu.assemble_record("p4.jpg", {"panel_kind": "chrome"})["panel_kind"] == "chrome"
    assert pu.assemble_record("p6.jpg", {"panel_kind": "caption"})["panel_kind"] == "caption"
    assert pu.assemble_record("p5.jpg", {})["panel_kind"] == "story"


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


# --- in-world screen rescue (chrome -> story via a real speech balloon) ------

def test_inworld_balloon_promotes_confident_compact():
    # ORV ep1 p000003: the masterpiece comment balloon (conf 0.96, ~0.14 area)
    dets = [(56, 768, 499, 1054, 0.96)]
    assert pu._is_inworld_balloon(dets, 736, 1169) is True


def test_inworld_balloon_rejects_screen_sized_false_positive():
    # ORV ep1 p000004 (stats card): low-conf, ~0.6-area boxes = whole panel
    dets = [(162, 41, 753, 423, 0.47), (216, 38, 781, 423, 0.36)]
    assert pu._is_inworld_balloon(dets, 800, 480) is False


def test_inworld_balloon_rejects_no_detection():
    # ORV ep1 p000033 (publisher credit): no balloon at all
    assert pu._is_inworld_balloon([], 800, 600) is False


def test_inworld_balloon_needs_both_confidence_and_compactness():
    # confident but huge -> rejected; compact but low-conf -> rejected
    assert pu._is_inworld_balloon([(0, 0, 760, 560, 0.95)], 800, 600) is False
    assert pu._is_inworld_balloon([(50, 50, 200, 180, 0.45)], 800, 600) is False
