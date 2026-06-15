"""
tests/test_timeline_selection.py

TDD for timeline_planner.build_cuts honoring scene_selection: when a shot has
more panels than fit at >=min_cut_sec, drop the 'redundant' panels FIRST
(instead of the old arbitrary files[:k] truncation).
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


def _sel(roles):
    return [{"scene_file": k, "role": v} for k, v in roles.items()]


def test_build_cuts_without_selection_unchanged():
    # 7s @ 3.5 => kmax 2 => first two panels (legacy behavior preserved)
    cuts = tp.build_cuts(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], 7.0, min_cut_sec=3.5)
    assert [c["file"] for c in cuts] == ["a.jpg", "b.jpg"]


def test_build_cuts_drops_redundant_first():
    sel = _sel({"a.jpg": "keep", "b.jpg": "redundant", "c.jpg": "keep", "d.jpg": "redundant"})
    cuts = tp.build_cuts(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], 7.0,
                         min_cut_sec=3.5, selection=sel)
    # room for 2 -> the two keepers, in order, redundant dropped
    assert [c["file"] for c in cuts] == ["a.jpg", "c.jpg"]


def test_build_cuts_durations_split_evenly_over_kept():
    sel = _sel({"a.jpg": "keep", "b.jpg": "redundant", "c.jpg": "keep"})
    cuts = tp.build_cuts(["a.jpg", "b.jpg", "c.jpg"], 8.0, min_cut_sec=3.5, selection=sel)
    assert [c["file"] for c in cuts] == ["a.jpg", "c.jpg"]
    assert sum(c["dur"] for c in cuts) == 8.0          # full shot covered
    assert all(c["dur"] >= 3.5 for c in cuts)          # kept panels meet the floor


def test_build_cuts_all_redundant_still_shows_one():
    sel = _sel({"a.jpg": "redundant", "b.jpg": "redundant"})
    cuts = tp.build_cuts(["a.jpg", "b.jpg"], 2.0, min_cut_sec=3.5, selection=sel)
    assert len(cuts) == 1                                # never an empty shot


# ---- coverage invariant: stretch to fit panels, never truncate --------------

def test_coverage_duration_stretches_to_fit_every_keeper():
    # 4 panels, no redundancy, short narration -> stretch to 4*3.5 so NONE drop
    # (the old kmax cap showed only floor(7/3.5)=2 and silently dropped 2)
    assert tp.coverage_duration(["a", "b", "c", "d"], None, 7.0, 3.5) == 14.0


def test_coverage_duration_counts_only_keepers_not_redundant():
    sel = _sel({"a": "keep", "b": "redundant", "c": "keep", "d": "redundant"})
    # only 2 keepers -> 2*3.5 = 7; the narration is already 7 -> they all fit
    assert tp.coverage_duration(["a", "b", "c", "d"], sel, 7.0, 3.5) == 7.0


def test_coverage_duration_keeps_a_long_narration_unchanged():
    # narration longer than the coverage need -> unchanged (panels pace under it)
    assert tp.coverage_duration(["a", "b"], None, 20.0, 3.5) == 20.0
    assert tp.coverage_duration([], None, 5.0, 3.5) == 5.0


# ---- filler-beat drop (build #3) -------------------------------------------

def test_is_filler_narration():
    assert tp.is_filler_narration("")
    assert tp.is_filler_narration("   ")
    assert tp.is_filler_narration("The scene continues.")
    assert tp.is_filler_narration("the story continues")
    assert tp.is_filler_narration("To be continued")
    assert not tp.is_filler_narration("Prince Cheon flees the dark forest.")
    assert not tp.is_filler_narration("The reason she's special is because...")
