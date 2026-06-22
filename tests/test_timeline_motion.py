"""
tests/test_timeline_motion.py

Tests for perceptible per-cut motion: every cut must have a visible pan, and
short cuts must receive a larger motion boost so the move reads in the short window.

Covers:
  - motion_for_cut: duration-aware strength/bias scaling helper
  - _MOTION_VARIANTS: no variant has zero pan (start_bias == end_bias == {0,0})
  - face-aware pans are preserved, not clobbered by the duration scale
  - output stays within sane bounds (strength [0,1], biases [-1,1])
  - schema integrity: all required keys present
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "timeline_planner",
    Path(__file__).resolve().parent.parent / "tools" / "timeline_planner.py",
)
tp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tp)  # type: ignore[union-attr]


def _base_kenburns(strength=0.75):
    """Minimal motion dict that mirrors _motion_params_for_mode's output shape."""
    return {
        "mode": "kenburns",
        "strength": strength,
        "ease": "ease_in_out",
        "start_bias": {"x": 0.30, "y": 0.18},
        "end_bias": {"x": -0.30, "y": -0.18},
        "zoom": {"start": 1.05, "end": 1.12},
        "bg_fill": {"mode": "blur", "enabled": True, "amount": 35, "dim": 0.18},
        "fg_fit": {"mode": "contain", "safe_inset_pct": 0.06},
    }


def _zero_pan_base(strength=0.75):
    """Simulates the 'push in' / 'pull out' variants that have no lateral pan."""
    return {
        "mode": "kenburns",
        "strength": strength,
        "ease": "ease_in_out",
        "start_bias": {"x": 0.0, "y": 0.0},
        "end_bias": {"x": 0.0, "y": 0.0},
        "zoom": {"start": 1.04, "end": 1.18},
        "bg_fill": {"mode": "blur", "enabled": True, "amount": 35, "dim": 0.18},
        "fg_fit": {"mode": "contain", "safe_inset_pct": 0.06},
    }


def _static_base():
    return {
        "mode": "static",
        "strength": 0.0,
        "ease": "ease_in_out",
        "start_bias": {"x": 0.0, "y": 0.0},
        "end_bias": {"x": 0.0, "y": 0.0},
        "zoom": {"start": 1.0, "end": 1.0},
        "bg_fill": {"mode": "blur", "enabled": True, "amount": 35, "dim": 0.18},
        "fg_fit": {"mode": "contain", "safe_inset_pct": 0.06},
    }


def _face_target(bbox):
    return [{"id": "face_1", "type": "face", "bbox": bbox}]


# ── _MOTION_VARIANTS: no variant may have zero pan ──────────────────────────────

def test_motion_variants_no_zero_pan():
    """Every variant in the rotation table must have start_bias != end_bias.
    Pure push-in/pull-out (start=end=(0,0)) are imperceptible on short cuts."""
    for i, v in enumerate(tp._MOTION_VARIANTS):
        start = v["start"]
        end = v["end"]
        assert start != end or (start[0] != 0.0 or start[1] != 0.0), (
            f"Variant {i} has start_bias == end_bias == (0,0): no pan — "
            "viewer won't see movement on a 2–3 s cut"
        )


# ── motion_for_cut: the duration-aware helper ───────────────────────────────────

def test_motion_for_cut_zero_pan_base_gets_drift():
    """A base motion with start_bias == end_bias (zero pan) must receive a
    non-zero drift so the cut is never fully static."""
    base = _zero_pan_base()
    result = tp.motion_for_cut(dur=3.0, base_motion=base)
    assert result["start_bias"] != result["end_bias"], (
        "motion_for_cut must ensure start_bias != end_bias on a zero-pan base"
    )


def test_motion_for_cut_short_cut_gets_more_motion_than_long():
    """A 2.5 s cut should have a higher effective strength than a 12 s cut
    given the same base motion (short window needs visible movement)."""
    base = _base_kenburns(strength=0.75)
    short = tp.motion_for_cut(dur=2.5, base_motion=base)
    long_ = tp.motion_for_cut(dur=12.0, base_motion=base)

    def _effective(m):
        # Effective motion = strength * magnitude of pan travel
        dx = m["end_bias"]["x"] - m["start_bias"]["x"]
        dy = m["end_bias"]["y"] - m["start_bias"]["y"]
        travel = (dx ** 2 + dy ** 2) ** 0.5
        return m["strength"] * travel

    assert _effective(short) > _effective(long_), (
        "short cut (2.5 s) must have higher effective motion than long cut (12 s)"
    )


def test_motion_for_cut_meaningful_pan_preserved():
    """If the base already has a meaningful pan (face-aware, or strong variant),
    motion_for_cut must not zero it out or reduce the travel to nothing."""
    base = _base_kenburns()
    result = tp.motion_for_cut(dur=5.0, base_motion=base)
    dx = result["end_bias"]["x"] - result["start_bias"]["x"]
    dy = result["end_bias"]["y"] - result["start_bias"]["y"]
    travel = (dx ** 2 + dy ** 2) ** 0.5
    assert travel > 0.10, "meaningful pan must be preserved (travel > 0.10)"


def test_motion_for_cut_face_aware_pan_not_clobbered():
    """When base_motion already has a face-aware pan (focus='face'), the
    direction must be preserved — duration scaling should only strengthen it,
    not redirect it."""
    face_motion = {
        "mode": "kenburns",
        "strength": 0.75,
        "ease": "ease_in_out",
        "start_bias": {"x": 0.15, "y": -0.10},   # start: slight opposite nudge
        "end_bias": {"x": -0.60, "y": 0.40},      # end: ON the face (upper-right)
        "zoom": {"start": 1.05, "end": 1.12},
        "bg_fill": {"mode": "blur", "enabled": True, "amount": 35, "dim": 0.18},
        "fg_fit": {"mode": "contain", "safe_inset_pct": 0.06},
        "focus": "face",
    }
    result = tp.motion_for_cut(dur=2.5, base_motion=face_motion)
    # The direction of travel must be preserved: end_bias must still point
    # toward the face (negative x, positive y in this example)
    assert result["end_bias"]["x"] < 0.0, (
        "face-aware end_bias x sign must be preserved after duration scaling"
    )
    assert result["end_bias"]["y"] > 0.0, (
        "face-aware end_bias y sign must be preserved after duration scaling"
    )


def test_motion_for_cut_static_stays_static():
    """Static shots must not be modified — they are intentionally motionless."""
    base = _static_base()
    result = tp.motion_for_cut(dur=2.0, base_motion=base)
    assert result["mode"] == "static"
    assert result["strength"] == 0.0
    assert result["start_bias"] == result["end_bias"]


def test_motion_for_cut_strength_bounded():
    """Strength must stay within [0.0, 1.0] even on very short cuts."""
    base = _base_kenburns(strength=0.95)
    result = tp.motion_for_cut(dur=1.5, base_motion=base)
    assert 0.0 <= result["strength"] <= 1.0, (
        f"strength {result['strength']} out of bounds"
    )


def test_motion_for_cut_bias_bounded():
    """Bias values must stay within [-1.0, 1.0] (the renderer's pan budget)."""
    base = _zero_pan_base()
    result = tp.motion_for_cut(dur=2.0, base_motion=base)
    for key in ("start_bias", "end_bias"):
        for axis in ("x", "y"):
            v = result[key][axis]
            assert -1.0 <= v <= 1.0, (
                f"{key}[{axis}] = {v} exceeds [-1, 1] pan budget"
            )


def test_motion_for_cut_schema_intact():
    """Output must include all keys the renderer expects."""
    required = {"mode", "strength", "ease", "start_bias", "end_bias", "zoom",
                "bg_fill", "fg_fit"}
    base = _base_kenburns()
    result = tp.motion_for_cut(dur=4.0, base_motion=base)
    missing = required - set(result)
    assert not missing, f"motion_for_cut dropped required keys: {missing}"


def test_motion_for_cut_returns_copy_not_mutated():
    """motion_for_cut must not mutate the base_motion dict passed to it."""
    base = _base_kenburns()
    original_start = dict(base["start_bias"])
    original_strength = base["strength"]
    tp.motion_for_cut(dur=2.5, base_motion=base)
    assert base["start_bias"] == original_start
    assert base["strength"] == original_strength


def test_motion_for_cut_none_base_gets_default():
    """Passing None as base_motion must return a valid motion dict (not crash)."""
    result = tp.motion_for_cut(dur=3.0, base_motion=None)
    assert isinstance(result, dict)
    assert result.get("mode") not in (None, "")
    assert result["start_bias"] != result["end_bias"]
