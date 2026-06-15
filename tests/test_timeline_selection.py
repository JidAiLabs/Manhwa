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


def test_has_actor_detects_person_or_creature():
    assert tp._has_actor(["a man in a suit"]) is True
    assert tp._has_actor(["a large snarling beast"]) is True
    assert tp._has_actor(["a smartphone displaying a list of episodes"]) is False
    assert tp._has_actor([], "A crowd of people stands in the dark.") is True
    assert tp._has_actor([], "A black screen of stat numbers.") is False


def test_prefer_scenes_in_beat_scene_wins_and_text_only_keeps_one():
    ts = {"ui1.jpg", "ui2.jpg"}
    # mixed beat -> the scene wins, both UI screens drop (their info is narrated)
    assert tp.prefer_scenes_in_beat(["scene.jpg", "ui1.jpg", "ui2.jpg"], ts) == ["scene.jpg"]
    # a text-ONLY beat keeps just the first screen (never empty, no UI parade)
    assert tp.prefer_scenes_in_beat(["ui1.jpg", "ui2.jpg"], ts) == ["ui1.jpg"]
    # nothing flagged -> unchanged
    assert tp.prefer_scenes_in_beat(["a.jpg", "b.jpg"], set()) == ["a.jpg", "b.jpg"]


def test_text_screen_files_flags_ui_not_scenes_or_cards(tmp_path):
    import json
    understood = {"panels": [
        {"scene_file": "feed.jpg",
         "subjects": ["a smartphone screen showing an episode list"],
         "description": "A phone app lists episodes with view counts."},
        {"scene_file": "train.jpg",
         "subjects": ["a man in a suit", "smartphone"],
         "description": "A man reads his phone on a train."},
        {"scene_file": "card.jpg", "subjects": [], "description": "A title card."},
    ]}
    vision = {"items": [
        {"scene_file": "feed.jpg", "panel_kind": "story", "text_coverage": 0.4,
         "ocr_clean": "READ EPISODE 1389 COMMENTS 1 VIEWS 1 READ EPISODE 1388 "
                      "COMMENTS 1 VIEWS 1 READ EPISODE 1387 COMMENTS 1 VIEWS 1"},
        {"scene_file": "train.jpg", "panel_kind": "story", "text_coverage": 0.2,
         "ocr_clean": "WHO WOULD READ A WEB NOVEL THAT HAS OVER 3000 EPISODES "
                      "I AM THE ONLY READER AND IT TOOK TEN YEARS"},
        {"scene_file": "card.jpg", "panel_kind": "story", "text_coverage": 0.1,
         "ocr_clean": "SKY CORPORATION"},
    ]}
    up = tmp_path / "u.json"; up.write_text(json.dumps(understood))
    vp = tmp_path / "v.json"; vp.write_text(json.dumps(vision))
    ts = tp.text_screen_files(str(up), str(vp))
    assert ts == {"feed.jpg"}           # the UI feed is information, not a shot
    assert "train.jpg" not in ts        # a man is present -> scene, even with text
    assert "card.jpg" not in ts         # short card -> kept (title/system card)
    assert tp.text_screen_files("", "") == set()    # no understanding -> safe no-op


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
