"""
tests/test_timeline_montage_narrated.py

ORV Ep18/Ep50 regression (`empty_item`): drop_caption_cards returns an EMPTY
montage entry for a beat it believes is caption-only, on the assumption that
such beats were folded into a neighbour upstream and own no art. When that
assumption fails, the planner kept the shot's narration and duration but built
no cuts -- a hole in the video that prep_qa blocks as `empty_item`, and that no
heal can repair (re-narrating cannot put a panel back).

montage_files_for_beat restores the beat's OWN art in exactly that case, and
only then -- never a neighbour's, so the stand-in hold drop_caption_cards warns
about (the p097x3 panel-collapse symptom) cannot come back.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "timeline_planner", REPO / "tools" / "timeline_planner.py")
tp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tp)  # type: ignore[union-attr]

OWN = ["p000099.jpg"]


def _beat(line: str) -> dict:
    return {"group_id": 25, "segments": [{"span": OWN, "line": line}]}


def test_non_empty_montage_is_passed_through_untouched():
    # the common path must not change: caption cards stay dropped
    out = tp.montage_files_for_beat(["p000100.jpg"], OWN, _beat("A real line."))
    assert out == ["p000100.jpg"]


def test_emptied_narrated_beat_keeps_its_own_art():
    # THE BUG: montage emptied the group, the beat still has a line to voice
    out = tp.montage_files_for_beat([], OWN, _beat(
        "Huiwon Jeong stares in wide-eyed horror, trembling."))
    assert out == OWN, "a narrated beat must never end up with zero panels"


def test_emptied_beat_without_narration_still_shows_nothing():
    # a genuinely caption-only beat keeps today's behaviour -- no stand-in hold
    assert tp.montage_files_for_beat([], OWN, _beat("")) == []


def test_filler_narration_does_not_resurrect_a_caption_beat():
    filler = "The scene continues."
    assert tp.is_filler_narration(filler), "fixture must be real filler"
    assert tp.montage_files_for_beat([], OWN, _beat(filler)) == []


def test_beat_with_no_own_files_stays_empty():
    assert tp.montage_files_for_beat([], [], _beat("A real line.")) == []


def test_missing_beat_is_tolerated():
    assert tp.montage_files_for_beat([], OWN, None) == []


def test_falls_back_to_beat_level_narration_when_no_segments():
    beat = {"group_id": 25, "narration": "A real narrated line."}
    assert tp.montage_files_for_beat([], OWN, beat) == OWN
