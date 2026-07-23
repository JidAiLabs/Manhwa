"""impact_mismatch: the narration-vs-art heal-then-block QA gate.

A narrated segment whose span contains a panel the UNDERSTANDING marks as a
strike in progress (strikes_or_weapons == 'in_use' in
manifest.panels.understood.json) must carry at least one impact-class lexeme —
a stab panel narrated as a peaceful stroll is the verified root cause this
attacks. The trigger is gemma's OWN semantic read (it sees the art and reads
the SFX), NOT the old CV impact-lettering detector, which fired on any big red
glyph and could not tell a boom from a blow (retired from this gate
2026-07-23). The lexicon is deliberately crude (word forms, case-insensitive):
over-matching can only SUPPRESS a flag, never create one.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "prep_qa",
    Path(__file__).resolve().parent.parent / "tools" / "prep_qa.py",
)
pq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pq)  # type: ignore[union-attr]


CALM_LINE = "He strolls through the quiet market, savoring the morning air."


def _understood(*impact_files, extra=()):
    # impact_files: the model marked a blow in progress (strikes_or_weapons=
    # 'in_use'); extra: ordinary panels with no strike delivered.
    panels = [{"scene_file": f, "description": "x", "action": "y",
               "panel_kind": "story", "intensity": "tense",
               "strikes_or_weapons": "in_use"}
              for f in impact_files]
    panels += [{"scene_file": f, "description": "x", "action": "y",
                "panel_kind": "story", "intensity": "calm",
                "strikes_or_weapons": "none"}
               for f in extra]
    return {"panels": panels}


def _beat(gid, segments):
    segs = [{"span": list(span), "line": line} for span, line in segments]
    return {"group_id": gid, "segments": segs,
            "scene_files": [f for span, _ in segments for f in span],
            "narration": " ".join(line for _, line in segments)}


def test_fires_on_impact_panel_with_calm_line():
    beats = {"beats": [_beat(3, [(["p1.jpg"], CALM_LINE)])]}
    fl = pq.impact_mismatch_flags(beats, _understood("p1.jpg"))
    assert len(fl) == 1
    f = fl[0]
    assert f["code"] == "impact_mismatch" and f["severity"] == pq.ERROR
    assert f["segment_id"] == "g0003"          # heal keys on the group id
    assert f["scene"] == "p1.jpg"


@pytest.mark.parametrize("line", [
    "The blade stabs deep into his side.",
    "A single strike shatters the silence.",
    "Steel pierces cloth and flesh.",
    "He slashes across the assassin's chest.",
    "Blood sprays across the snow.",
    "The impact hurls him backward.",
    "It hits harder than any fist should.",
    "He drives the blade home.",
])
def test_suppressed_when_line_carries_an_impact_lexeme(line):
    beats = {"beats": [_beat(1, [(["p1.jpg"], line)])]}
    assert pq.impact_mismatch_flags(beats, _understood("p1.jpg")) == []


def test_suppressed_when_no_impact_panel_in_span():
    beats = {"beats": [_beat(1, [(["p2.jpg"], CALM_LINE)])]}
    assert pq.impact_mismatch_flags(
        beats, _understood("p1.jpg", extra=("p2.jpg",))) == []


def test_fires_when_any_span_panel_is_impact():
    # one impact panel inside a multi-panel span is enough
    beats = {"beats": [_beat(2, [(["p1.jpg", "p2.jpg", "p3.jpg"], CALM_LINE)])]}
    fl = pq.impact_mismatch_flags(
        beats, _understood("p2.jpg", extra=("p1.jpg", "p3.jpg")))
    assert [f["scene"] for f in fl] == ["p2.jpg"]


def test_split_halves_trace_back_to_the_impact_parent():
    # a span may reference render-split halves (p1_a.jpg) of the understood
    # panel (p1.jpg) — same _base_scene normalization as span_cover_flags
    beats = {"beats": [_beat(1, [(["p1_a.jpg"], CALM_LINE)])]}
    assert len(pq.impact_mismatch_flags(beats, _understood("p1.jpg"))) == 1


def test_visible_weapon_not_in_use_does_not_fire():
    # The precision gain: a weapon merely DRAWN ('visible') is not a blow being
    # delivered, so a tense stand-off narrated without impact wording is fine.
    # This is the class the CV detector could not distinguish (a throne-boom or
    # a drawn-but-unused blade both showed big red SFX).
    understood = {"panels": [{"scene_file": "p1.jpg", "description": "x",
                              "action": "y", "panel_kind": "story",
                              "intensity": "tense",
                              "strikes_or_weapons": "visible"}]}
    beats = {"beats": [_beat(1, [(["p1.jpg"], CALM_LINE)])]}
    assert pq.impact_mismatch_flags(beats, understood) == []


def test_legacy_understood_without_strike_field_is_silent():
    legacy = {"panels": [{"scene_file": "p1.jpg", "description": "x",
                          "action": "y", "panel_kind": "story"}]}
    beats = {"beats": [_beat(1, [(["p1.jpg"], CALM_LINE)])]}
    assert pq.impact_mismatch_flags(beats, legacy) == []


@pytest.mark.parametrize("kind", ["system", "chrome", "caption"])
def test_exempt_panel_kinds_never_fire_the_gate(kind):
    # Belt-and-suspenders: even if the model marks a strike ('in_use') on a
    # system/chrome/caption panel (a stat card, a UI banner, an author
    # caption), this QA gate must never block on it regardless of the (calm)
    # narration line.
    understood = {"panels": [{"scene_file": "p1.jpg", "description": "x",
                              "action": "y", "panel_kind": kind,
                              "intensity": "tense",
                              "strikes_or_weapons": "in_use"}]}
    beats = {"beats": [_beat(1, [(["p1.jpg"], CALM_LINE)])]}
    assert pq.impact_mismatch_flags(beats, understood) == []


def test_missing_inputs_are_silent():
    assert pq.impact_mismatch_flags({}, {}) == []
    assert pq.impact_mismatch_flags(None, None) == []
    assert pq.impact_mismatch_flags(
        {"beats": []}, _understood("p1.jpg")) == []


def test_lexeme_matching_is_word_start_anchored():
    assert pq.has_impact_lexeme("he hits the wall")
    assert pq.has_impact_lexeme("A STRIKING counterattack")
    assert pq.has_impact_lexeme("the wound bleeds")
    # 'hit' must NOT match inside 'white'; 'blow' not inside 'billow'
    assert not pq.has_impact_lexeme("a white robe in the wind")
    assert not pq.has_impact_lexeme("sails billow over calm water")
    assert not pq.has_impact_lexeme("")


@pytest.mark.parametrize("line", [
    "The kingdom remains stable despite the chaos.",   # 'stab' swallowed 'stable'
    "The knight polished the cutlery before dinner.",  # 'cut' swallowed 'cutlery'
    "They ran through the empty market square.",       # loose phrase collision
])
def test_lexeme_matching_requires_trailing_boundary_too(line):
    # A leading-boundary-ONLY match let a stem swallow a longer unrelated
    # word ('stab' in 'stable', 'cut' in 'cutlery'); a bare phrase entry
    # ('ran through') collided with ordinary foot-travel. All three must be
    # false-suppression-free, i.e. carry NO impact lexeme.
    assert not pq.has_impact_lexeme(line), line


@pytest.mark.parametrize("line", [
    "Steel flashes in a lethal strike.",
    "He stabbed him without hesitation.",
    "Blood sprays across the stone floor.",
])
def test_lexeme_matching_still_catches_genuine_impact_lines(line):
    assert pq.has_impact_lexeme(line), line
