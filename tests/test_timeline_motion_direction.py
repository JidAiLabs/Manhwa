"""
tests/test_timeline_motion_direction.py

Pins the DIRECTIONAL-slide motion policy (2026-06-22).

The old policy made `kenburns` (a diagonal drift) the fallback and mapped "calm"
beats to `static`. On a real chapter that produced 66/81 kenburns cuts and only 2
directional slides — a muddy, repetitive diagonal on nearly every panel.

New policy:
  - the FALLBACK motion is a clean directional slide (slide_left/right/tilt_up/down),
    NOT kenburns;
  - "calm" panels get a gentle slide, NOT static;
  - `static` only when the beat is EXPLICITLY hinted to hold;
  - a sequence of plain beats yields mostly directional slides whose direction
    rotates so adjacent beats differ;
  - the per-cut variant table (`_MOTION_VARIANTS`) is biased toward pure
    lateral/vertical moves so adjacent face-less cuts differ in direction and
    diagonals are rare;
  - explicit `rendering_hints.camera_motion` is still honoured.
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


_DIRECTIONAL = {"slide_left", "slide_right", "tilt_up", "tilt_down"}


def _beat(**kw):
    b = {"mood_words": [], "emotional_turn": "", "rendering_hints": {}}
    b.update(kw)
    return b


# ── fallback is a directional slide, NOT kenburns / static ──────────────────────

def test_fallback_mode_is_directional_not_kenburns():
    """A plain beat (no hint, no mood) must fall back to a directional slide."""
    mode = tp._choose_motion_mode(_beat())
    assert mode in _DIRECTIONAL, (
        f"fallback should be a directional slide, got {mode!r}"
    )
    assert mode != "kenburns", "kenburns must no longer be the fallback"
    assert mode != "static", "static must no longer be the fallback"


def test_calm_beat_gets_gentle_slide_not_static():
    """'calm' must map to a gentle directional slide, never a frozen static frame."""
    mode = tp._choose_motion_mode(
        _beat(mood_words=["calm", "reflection"], emotional_turn="calm")
    )
    assert mode in _DIRECTIONAL, f"calm should slide, got {mode!r}"
    assert mode != "static"


# ── static only when explicitly hinted ──────────────────────────────────────────

def test_static_only_when_explicitly_hinted():
    """`static` must appear only when rendering_hints explicitly asks to hold."""
    # explicit static hint -> static
    held = tp._choose_motion_mode(
        _beat(rendering_hints={"camera_motion": "static"})
    )
    assert held == "static"

    # explicit 'hold' hint -> static
    hold = tp._choose_motion_mode(
        _beat(rendering_hints={"camera_motion": "hold"})
    )
    assert hold == "static"

    # nothing else (even calm) yields static
    for b in (
        _beat(),
        _beat(mood_words=["calm", "peace"], emotional_turn="calm"),
        _beat(mood_words=["sad", "regret"]),
        _beat(mood_words=["reveal", "hero"]),
    ):
        assert tp._choose_motion_mode(b) != "static", (
            f"static leaked for non-hinted beat {b}"
        )


# ── explicit camera_motion hints still respected ────────────────────────────────

def test_explicit_hints_still_respected():
    assert tp._choose_motion_mode(
        _beat(rendering_hints={"camera_motion": "push_in"})) == "zoom_in"
    assert tp._choose_motion_mode(
        _beat(rendering_hints={"camera_motion": "pull_out"})) == "zoom_out"
    assert tp._choose_motion_mode(
        _beat(rendering_hints={"camera_motion": "slide_right"})) == "slide_right"
    assert tp._choose_motion_mode(
        _beat(rendering_hints={"camera_motion": "tilt_up"})) == "tilt_up"
    # an explicit kenburns hint is still honoured (deliberate accent)
    assert tp._choose_motion_mode(
        _beat(rendering_hints={"camera_motion": "kenburns"})) == "kenburns"


# ── a sequence of plain beats: mostly directional, neighbours differ ─────────────

def test_plain_sequence_is_mostly_directional():
    """A run of plain beats must yield MOSTLY directional slides (not kenburns)."""
    n = 12
    modes = [tp._choose_motion_mode(_beat(), ordinal=i) for i in range(n)]
    directional = sum(1 for m in modes if m in _DIRECTIONAL)
    kb = sum(1 for m in modes if m == "kenburns")
    assert directional > n // 2, (
        f"majority of plain beats must be directional slides, got {modes}"
    )
    assert kb == 0, f"kenburns must not appear in a plain sequence, got {modes}"


def test_plain_sequence_neighbours_differ():
    """Adjacent plain beats must not repeat the same slide direction."""
    modes = [tp._choose_motion_mode(_beat(), ordinal=i) for i in range(12)]
    for a, b in zip(modes, modes[1:]):
        assert a != b, f"adjacent beats share a direction: {modes}"


def test_sequence_covers_multiple_directions():
    """Across a run the fallback should reach several of L/R/U/D, not one axis."""
    modes = {tp._choose_motion_mode(_beat(), ordinal=i) for i in range(8)}
    assert len(modes & _DIRECTIONAL) >= 3, (
        f"fallback rotation should span >=3 directions, got {modes}"
    )


def test_choose_motion_mode_ordinal_is_optional():
    """Back-compat: callers that omit `ordinal` must still work (defaults applied)."""
    # no TypeError; returns a directional default
    assert tp._choose_motion_mode(_beat()) in _DIRECTIONAL


# ── _MOTION_VARIANTS: biased toward pure lateral/vertical, diagonals rare ────────

def _is_pure_axis(v) -> bool:
    sx, sy = v["start"]
    ex, ey = v["end"]
    # pure lateral: no vertical component; pure vertical: no lateral component
    horizontal = (sy == 0.0 and ey == 0.0) and (sx != 0.0 or ex != 0.0)
    vertical = (sx == 0.0 and ex == 0.0) and (sy != 0.0 or ey != 0.0)
    return horizontal or vertical


def test_motion_variants_mostly_pure_axis():
    """Most variants must be pure lateral/vertical slides; diagonals are the rare
    minority (the diagonal-on-every-panel look was the complaint)."""
    pure = sum(1 for v in tp._MOTION_VARIANTS if _is_pure_axis(v))
    total = len(tp._MOTION_VARIANTS)
    assert pure > total / 2, (
        f"variant table must be MOSTLY pure-axis slides, got {pure}/{total} pure"
    )


def test_motion_variants_cover_all_four_directions():
    """The rotation must include left, right, up and down pure moves so adjacent
    face-less cuts can differ in direction."""
    have_left = have_right = have_up = have_down = False
    for v in tp._MOTION_VARIANTS:
        sx, sy = v["start"]
        ex, ey = v["end"]
        if sy == 0.0 and ey == 0.0:
            if ex < sx:
                have_left = True   # image travels left
            elif ex > sx:
                have_right = True
        if sx == 0.0 and ex == 0.0:
            if ey > sy:
                have_down = True
            elif ey < sy:
                have_up = True
    assert have_left and have_right, "need both lateral directions"
    assert have_up and have_down, "need both vertical directions"


def test_motion_variants_adjacent_differ_in_axis_or_direction():
    """Adjacent variants must differ in travel direction so neighbouring cuts that
    rotate through the table never read as the same move."""
    def travel(v):
        return (round(v["end"][0] - v["start"][0], 3),
                round(v["end"][1] - v["start"][1], 3))
    n = len(tp._MOTION_VARIANTS)
    for i in range(n):
        a = travel(tp._MOTION_VARIANTS[i])
        b = travel(tp._MOTION_VARIANTS[(i + 1) % n])
        assert a != b, f"variants {i} and {(i+1)%n} share travel vector {a}"
