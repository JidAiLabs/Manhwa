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


def test_build_cuts_no_selection_shows_every_panel():
    # NEW coverage rule: no distinct panel is dropped to fit a short shot — all 4
    # are paced WITHIN it (the old kmax cap truncated to 2; with no music we pace
    # under the narration instead of dropping or stretching into silence).
    cuts = tp.build_cuts(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], 7.0, min_cut_sec=3.5)
    assert [c["file"] for c in cuts] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert abs(sum(c["dur"] for c in cuts) - 7.0) < 1e-6   # all fit inside the shot


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


def test_build_cuts_protected_story_panel_survives_redundant_verdict():
    # THE premise-panel bug: the beats LLM tagged p14 (the phone showing the
    # novel title — the whole premise) 'redundant'; the understanding calls it
    # 'story', so it's PROTECTED and must still be shown. An unprotected redundant
    # caption (p16) still drops — captions ride the narration, not the montage.
    sel = _sel({"p14.jpg": "redundant", "p15.jpg": "keep", "p16.jpg": "redundant"})
    cuts = tp.build_cuts(["p14.jpg", "p15.jpg", "p16.jpg"], 9.0,
                         min_cut_sec=3.0, selection=sel, protected={"p14.jpg"})
    files = [c["file"] for c in cuts]
    assert "p14.jpg" in files            # protected story panel survives the drop
    assert "p15.jpg" in files            # the keeper is still there
    assert "p16.jpg" not in files        # unprotected redundant caption still drops


def test_drop_caption_cards_keeps_scenes_and_holds_for_caption_only_beats():
    caps = {"cap1.jpg", "cap2.jpg"}
    order = [
        (1, ["cap1.jpg", "scene1.jpg", "cap2.jpg"]),  # mixed -> scene only, caps out
        (2, ["cap1.jpg"]),                              # caption-only -> hold a scene
        (3, ["scene3.jpg"]),
    ]
    m = tp.drop_caption_cards(order, caps)
    assert m[1] == ["scene1.jpg"]      # the bare cards drop; the scene stays
    assert m[2] == ["scene1.jpg"]      # held the previous real scene (never blank)
    assert m[3] == ["scene3.jpg"]
    # nothing flagged -> unchanged
    assert tp.drop_caption_cards([(1, ["a.jpg"])], set()) == {1: ["a.jpg"]}


def test_caption_files_flags_only_captions_not_in_world_screens(tmp_path):
    import json
    v = {"items": [
        {"scene_file": "c.jpg", "panel_kind": "caption"},    # bare monologue card
        {"scene_file": "s.jpg", "panel_kind": "story"},      # scene
        {"scene_file": "scr.jpg", "panel_kind": "story"},    # in-world screen = story
    ]}
    vp = tmp_path / "v.json"; vp.write_text(json.dumps(v))
    assert tp.caption_files(str(vp)) == {"c.jpg"}
    assert tp.caption_files("") == set()


def test_protected_story_files_reads_stamped_panel_kind(tmp_path):
    import json
    vision = {"items": [
        {"scene_file": "scenes/p1.jpg", "panel_kind": "story"},   # basename kept
        {"scene_file": "p2.jpg", "panel_kind": "caption"},        # not story
        {"scene_file": "p3.jpg", "panel_kind": "empty"},          # not story
        {"scene_file": "p4.jpg", "panel_kind": "story"},
        {"scene_file": "p5.jpg"},                                 # unstamped
    ]}
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    assert tp.protected_story_files(str(vp)) == {"p1.jpg", "p4.jpg"}
    assert tp.protected_story_files("") == set()                  # missing -> empty


# ---- filler-beat drop (build #3) -------------------------------------------

def test_is_filler_narration():
    assert tp.is_filler_narration("")
    assert tp.is_filler_narration("   ")
    assert tp.is_filler_narration("The scene continues.")
    assert tp.is_filler_narration("the story continues")
    assert tp.is_filler_narration("To be continued")
    assert not tp.is_filler_narration("Prince Cheon flees the dark forest.")
    assert not tp.is_filler_narration("The reason she's special is because...")
