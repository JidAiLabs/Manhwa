"""
tests/test_timeline_face_pan.py

Camera-pan-to-face fix. On a dialogue panel the speech bubble is inpainted BLANK
(its words ride the narration), so the Ken Burns move must NOT end on a text_block
target — it should end CENTERED ON A FACE (the emotional payoff). These tests pin
the per-cut motion the planner emits:

  - a panel with a face -> end_bias points TOWARD that face (so the move lands the
    face in frame at t=1), per remotion/src/Cut.tsx's bias->translate convention;
  - a panel with ONLY a text_block -> the pan is NOT redirected onto it (stays the
    shot's generic default);
  - the largest / most-central face wins when several are present.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "timeline_planner",
    Path(__file__).resolve().parent.parent / "tools" / "timeline_planner.py",
)
tp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tp)  # type: ignore[union-attr]


def _face(bbox):
    return {"id": "face_1", "type": "face", "bbox": bbox}


def _text(bbox):
    return {"id": "text_1", "type": "text_block", "bbox": bbox}


def _base_motion(mode="kenburns"):
    # mirrors _motion_params_for_mode output shape (only the fields we assert on)
    return {
        "mode": mode,
        "strength": 0.8,
        "start_bias": {"x": 0.35, "y": 0.20},
        "end_bias": {"x": -0.35, "y": -0.20},
        "zoom": {"start": 1.05, "end": 1.12},
        "bg_fill": {"enabled": True, "amount": 35, "dim": 0.18},
        "fg_fit": {"mode": "contain", "safe_inset_pct": 0.06},
    }


# ---- face_end_bias: the sign convention that lands the face centered ----------

def test_face_end_bias_face_right_pulls_image_left():
    # face center cx=0.8 (right of frame) -> negative x bias (image moves LEFT so
    # the right-side face comes to center). cy=0.3 (above) -> negative y bias.
    b = tp.face_end_bias([0.7, 0.2, 0.9, 0.4])  # center (0.8, 0.3)
    assert b["x"] < 0.0
    assert b["y"] < 0.0


def test_face_end_bias_face_lower_left_signs_flip():
    b = tp.face_end_bias([0.1, 0.6, 0.3, 0.8])  # center (0.2, 0.7)
    assert b["x"] > 0.0     # face left -> image moves RIGHT
    assert b["y"] > 0.0     # face below -> positive (Cut.tsx negates Y => image UP)


def test_face_end_bias_centered_face_is_neutral():
    assert tp.face_end_bias([0.4, 0.4, 0.6, 0.6]) == {"x": 0.0, "y": 0.0}


def test_face_end_bias_is_clamped_to_pan_budget():
    b = tp.face_end_bias([0.95, 0.95, 1.0, 1.0])  # far corner
    assert -1.0 <= b["x"] <= 1.0
    assert -1.0 <= b["y"] <= 1.0


# ---- pick_face_target: never text, largest/most-central face wins ------------

def test_pick_face_target_ignores_text_blocks():
    targets = [
        {"id": "wide", "type": "frame", "bbox": [0, 0, 1, 1]},
        _text([0.1, 0.1, 0.9, 0.3]),
    ]
    assert tp.pick_face_target(targets) is None


def test_pick_face_target_prefers_largest_face():
    small = _face([0.10, 0.10, 0.18, 0.18])
    big = {"id": "face_2", "type": "face", "bbox": [0.50, 0.50, 0.85, 0.85]}
    chosen = tp.pick_face_target([small, big])
    assert chosen is big


# ---- face_aware_motion: the per-cut motion the planner attaches --------------

def test_face_aware_motion_ends_pan_on_the_face():
    # A panel with a face to the upper-right: the produced cut motion must END
    # biased TOWARD that face (negative x, negative y here), travelling FROM a
    # neutral-ish start, with zoom untouched.
    base = _base_motion()
    targets = [
        {"id": "wide", "type": "frame", "bbox": [0, 0, 1, 1]},
        _text([0.0, 0.85, 1.0, 1.0]),          # bubble at the bottom (inpainted)
        _face([0.70, 0.18, 0.92, 0.42]),       # face upper-right, center ~(0.81,0.30)
    ]
    m = tp.face_aware_motion(base, targets)
    assert m is not base                        # a per-cut override was produced
    # ends on the face:
    assert m["end_bias"]["x"] < 0.0
    assert m["end_bias"]["y"] < 0.0
    # start travels toward it (opposite sign, smaller magnitude) -> not static:
    assert m["start_bias"]["x"] > 0.0
    assert abs(m["start_bias"]["x"]) < abs(m["end_bias"]["x"])
    # zoom is preserved exactly (fix is pan-only):
    assert m["zoom"] == base["zoom"]
    assert m["strength"] == base["strength"]
    # breadcrumb for QA
    assert m.get("focus") == "face"


def test_face_aware_motion_never_redirects_onto_text_block():
    # A dialogue panel with ONLY a bubble (text_block) and NO face: the pan must
    # NOT be redirected onto the bubble — it keeps the shot's generic motion
    # (the bubble is rendered blank, so ending there is the bug we're fixing).
    base = _base_motion()
    targets = [
        {"id": "wide", "type": "frame", "bbox": [0, 0, 1, 1]},
        _text([0.15, 0.10, 0.85, 0.45]),       # the speech bubble, upper area
    ]
    m = tp.face_aware_motion(base, targets)
    assert m is base                            # unchanged: no face -> no override
    assert m["end_bias"] == base["end_bias"]    # NOT pulled toward the text block


def test_face_aware_motion_static_shot_stays_static():
    base = _base_motion(mode="static")
    targets = [_face([0.7, 0.2, 0.9, 0.4])]
    assert tp.face_aware_motion(base, targets) is base


def test_face_aware_motion_no_targets_is_passthrough():
    base = _base_motion()
    assert tp.face_aware_motion(base, None) is base
    assert tp.face_aware_motion(base, []) is base


# ---- index_targets_by_file: basename keying from the vision manifest ---------

def test_index_targets_by_file_keys_on_basename(tmp_path):
    vision = {"items": [
        {"scene_file": "scenes/p1.jpg",
         "targets": [_face([0.6, 0.2, 0.8, 0.4])]},
        {"scene_file": "p2.jpg", "targets": [_text([0.1, 0.1, 0.9, 0.3])]},
        {"scene_file": "p3.jpg"},                 # no targets key
    ]}
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    idx = tp.index_targets_by_file(str(vp))
    assert set(idx) == {"p1.jpg", "p2.jpg"}       # basename keyed; p3 has none
    assert idx["p1.jpg"][0]["type"] == "face"
    assert tp.index_targets_by_file("") == {}     # missing -> empty
