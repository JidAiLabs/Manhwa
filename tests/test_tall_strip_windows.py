"""Extreme-tall strips are understood in windows; any chrome window makes the
whole strip publication chrome (the ORV Ep0 title/credits strip class).
Normal panels stay single-pass, and only tall records re-run on upgrade."""

import numpy as np
from PIL import Image

import tools.panel_understand as pu


def _tall_img(tmp_path, w=400, h=4800):
    p = tmp_path / "strip.jpg"
    arr = np.random.default_rng(1).integers(40, 220, size=(h, w, 3),
                                            dtype=np.uint8)
    Image.fromarray(arr).save(p, quality=88)
    return str(p)


def _normal_img(tmp_path, w=800, h=1000):
    p = tmp_path / "panel.jpg"
    Image.new("RGB", (w, h), (128, 128, 128)).save(p, quality=88)
    return str(p)


def _item(path):
    return {"scene_file": path.rsplit("/", 1)[-1], "scene_path": path,
            "ocr_clean": "", "vision": {}}


def test_tall_dims_gate(tmp_path):
    assert pu._tall_dims(_tall_img(tmp_path)) == (400, 4800)
    assert pu._tall_dims(_normal_img(tmp_path)) is None      # ratio too low
    assert pu._tall_dims(None) is None
    assert pu._tall_dims(str(tmp_path / "missing.jpg")) is None


def test_chrome_window_makes_whole_strip_chrome(tmp_path):
    path = _tall_img(tmp_path)
    calls = []

    def call_fn(payload, image_path):
        calls.append(image_path)
        if len(calls) == 3:
            return {"description": "series logo and credits",
                    "panel_kind": "chrome"}
        return {"description": f"art window {len(calls)}",
                "subjects": ["hero"], "panel_kind": "story",
                "intensity": "calm"}

    recs = pu.understand_panels([_item(path)], call_fn)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["panel_kind"] == "chrome"
    assert rec["prompt_version"] == pu.TALL_WINDOWS_VERSION
    assert len(rec["tall_windows"]) == len(calls) >= 3
    # windows never see the original full-strip file
    assert all(c != path for c in calls)


def test_no_chrome_merges_to_one_story_panel(tmp_path):
    path = _tall_img(tmp_path)

    def call_fn(payload, image_path):
        return {"description": "falling through the sky",
                "subjects": ["hero", "sky"], "panel_kind": "story",
                "intensity": "tense", "dialogue": "", "setting": "sky"}

    recs = pu.understand_panels([_item(path)], call_fn)
    rec = recs[0]
    assert rec["panel_kind"] == "story"
    assert "falling" in rec["description"]
    assert rec["subjects"] == ["hero", "sky"]        # union, deduped
    assert rec["intensity"] == "tense"


def test_all_windows_failing_records_error_for_resume(tmp_path):
    path = _tall_img(tmp_path)
    recs = pu.understand_panels([_item(path)], lambda p, ip: None)
    assert recs[0].get("error") == "parse_failed"


def test_normal_panel_single_pass(tmp_path):
    path = _normal_img(tmp_path)
    calls = []

    def call_fn(payload, image_path):
        calls.append(image_path)
        return {"description": "a room", "panel_kind": "story"}

    recs = pu.understand_panels([_item(path)], call_fn)
    assert calls == [path]                            # exactly one, real file
    assert recs[0]["prompt_version"] == pu.PROMPT_VERSION


def test_resume_invalidates_only_tall_records(tmp_path):
    tall, normal = _tall_img(tmp_path), _normal_img(tmp_path)
    items = [_item(normal), _item(tall)]
    calls = []

    def call_fn(payload, image_path):
        calls.append(image_path)
        return {"description": "d", "panel_kind": "story"}

    # prior records stamped pu_v3 (pre-windowing) for BOTH scenes
    prior = {}
    for it in items:
        prior[it["scene_file"]] = {
            "scene_file": it["scene_file"], "description": "cached",
            "scene_sha": pu._scene_sha(it["scene_path"]),
            "prompt_version": pu.PROMPT_VERSION}
    recs = pu.understand_panels(items, call_fn, prior=prior)
    by = {r["scene_file"]: r for r in recs}
    assert by[items[0]["scene_file"]]["description"] == "cached"   # accepted
    assert by[items[1]["scene_file"]]["description"] != "cached"   # re-ran
    assert all(c != normal for c in calls)
