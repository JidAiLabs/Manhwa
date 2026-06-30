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


def test_drop_caption_cards_folds_caption_only_beats_without_held_duplicate():
    """REGRESSION (panel-collapse / p097x3): a caption-only beat must NOT hold a
    stand-in copy of an adjacent real panel — that manufactured the repeated
    static cut. It folds (its words ride the adjacent narration upstream) and
    shows NOTHING of its own; the bare caption card never becomes a shot."""
    caps = {"cap1.jpg", "cap2.jpg"}
    order = [
        (1, ["cap1.jpg", "scene1.jpg", "cap2.jpg"]),  # mixed -> scene only, caps out
        (2, ["cap1.jpg"]),                              # caption-only -> NO held copy
        (3, ["scene3.jpg"]),
    ]
    m = tp.drop_caption_cards(order, caps)
    assert m[1] == ["scene1.jpg"]      # the bare cards drop; the scene stays
    assert m[2] == []                  # caption-only: no held neighbor duplicate
    assert m[2] != ["scene1.jpg"]      # not the previous real panel
    assert m[2] != ["scene3.jpg"]      # not the next real panel either
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


def test_text_context_only_files_flags_story_bubble_without_subject(tmp_path):
    import json
    v = {"items": [
        {
            "scene_file": "bubble.jpg",
            "panel_kind": "story",
            "subjects": ["speech bubble"],
            "ocr_clean": "AS I THOUGHT, THIS GUY IS A GENIUS!",
            "text_coverage": 0.1552,
        },
        {
            "scene_file": "thought_hair.jpg",
            "panel_kind": "story",
            "subjects": ["speech bubble", "character's hair"],
            "ocr_clean": "HE'LL HAVE NO PROBLEM WITH OPERATING FORMATION.",
            "text_coverage": 0.0984,
        },
        {
            "scene_file": "dialogue_scene.jpg",
            "panel_kind": "story",
            "subjects": ["young man", "speech bubble"],
            "ocr_clean": "You really are smart!",
            "text_coverage": 0.08,
        },
        {
            "scene_file": "system.jpg",
            "panel_kind": "story",
            "subjects": ["text"],
            "ocr_clean": "SYSTEM ACTIVATION",
            "text_coverage": 0.05,
        },
        {
            "scene_file": "chapter_title.jpg",
            "panel_kind": "chrome",
            "subjects": ["title logo"],
            "ocr_clean": "Nano Machine CHAPTER 7 그림 각색 원작",
            "text_coverage": 0.12,
        },
    ]}
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(v))
    assert tp.text_context_only_files(str(vp)) == {
        "bubble.jpg", "thought_hair.jpg", "chapter_title.jpg"}


def test_publication_chrome_files_does_not_catch_story_screens(tmp_path):
    import json
    v = {"items": [
        {
            "scene_file": "chapter_title.jpg",
            "panel_kind": "chrome",
            "ocr_clean": "Nano Machine CHAPTER 7 그림 각색 원작",
        },
        {
            "scene_file": "status.jpg",
            "panel_kind": "story",
            "subjects": ["in-world screen"],
            "ocr_clean": "STRENGTH 12 AGILITY 9",
        },
    ]}
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(v))
    assert tp.publication_chrome_files(str(vp)) == {"chapter_title.jpg"}


def test_protected_card_files_skips_context_bubbles(tmp_path):
    import cv2
    import json
    import numpy as np

    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    img = np.full((240, 320, 3), 248, np.uint8)
    cv2.imwrite(str(scene_dir / "system.jpg"), img)
    cv2.imwrite(str(scene_dir / "bubble.jpg"), img)
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps({"items": [
        {"scene_file": "system.jpg", "panel_kind": "story",
         "subjects": ["text"], "ocr_clean": "SYSTEM ACTIVATION",
         "text_coverage": 0.05},
        {"scene_file": "bubble.jpg", "panel_kind": "empty",
         "subjects": ["speech bubble"], "ocr_clean": "YES, PRINCE.",
         "text_coverage": 0.03},
    ]}))
    assert tp.protected_card_files(str(vp), [str(scene_dir)]) == {"system.jpg"}


def test_protected_story_files_reads_stamped_panel_kind(tmp_path):
    import json
    vision = {"items": [
        {"scene_file": "scenes/p1.jpg", "panel_kind": "story"},   # basename kept
        {"scene_file": "p2.jpg", "panel_kind": "caption"},        # not story
        {"scene_file": "p3.jpg", "panel_kind": "empty"},          # not story
        {"scene_file": "p4.jpg", "panel_kind": "story"},
        {"scene_file": "p6.jpg", "panel_kind": "story",
         "subjects": ["speech bubble"], "ocr_clean": "As I thought, this guy is a genius!",
         "text_coverage": 0.12},                                  # context only
        {"scene_file": "p5.jpg"},                                 # unstamped
    ]}
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    assert tp.protected_story_files(str(vp)) == {"p1.jpg", "p4.jpg"}
    assert tp.protected_story_files("") == set()                  # missing -> empty


# ---- group-protection -> per-segment propagation ---------------------------
# THE in-world card bug: protected_story_files()/protected_card_files() add an
# in-world STORY/system card (e.g. "SKY CORPORATION.") to the group's protected
# set, but the planner emits ONE item per SCRIPT segment whose panels come from
# the script's per-shot list — which EXCLUDES the card (the LLM tagged it
# 'redundant'). So the card lands in no segment and never renders. The fix
# guarantees every protected file in the group's scene_files is shown in at
# least one segment.

def test_inject_missing_protected_adds_card_to_a_segment():
    # group scene_files include the protected card; neither segment's per-shot
    # pick chose it -> it must be injected into one segment.
    picks = [["a.jpg", "b.jpg"], ["c.jpg"]]
    out = tp.inject_missing_protected(
        picks, ["a.jpg", "b.jpg", "c.jpg", "card.jpg"], {"card.jpg"})
    shown = {f for p in out for f in p}
    assert "card.jpg" in shown                 # protected card now rendered
    # injected into the SMALLEST pick list (the closing 1-panel segment)
    assert out[1] == ["c.jpg", "card.jpg"]
    assert out[0] == ["a.jpg", "b.jpg"]        # other segment untouched


def test_inject_missing_protected_noop_when_already_shown():
    picks = [["card.jpg"], ["d.jpg"]]
    out = tp.inject_missing_protected(
        picks, ["card.jpg", "d.jpg"], {"card.jpg"})
    assert out == [["card.jpg"], ["d.jpg"]]     # already shown -> unchanged


def test_inject_missing_protected_ignores_non_protected_drops():
    # a non-protected redundant panel the per-shot selection dropped STAYS
    # dropped (it never enters the protected set).
    picks = [["a.jpg"], ["b.jpg"]]
    out = tp.inject_missing_protected(
        picks, ["a.jpg", "b.jpg", "redundant.jpg"], {"card.jpg"})
    shown = {f for p in out for f in p}
    assert "redundant.jpg" not in shown         # non-protected drop stays dropped
    assert "card.jpg" not in shown              # not in group -> nothing to inject
    assert out == [["a.jpg"], ["b.jpg"]]        # panel-rich group unaffected


def test_inject_missing_protected_only_injects_files_in_group_scene_files():
    # a protected file NOT in this group's scene_files is never injected here
    # (it belongs to another group).
    picks = [["a.jpg"]]
    out = tp.inject_missing_protected(
        picks, ["a.jpg"], {"other_group_card.jpg"})
    assert out == [["a.jpg"]]


def test_pick_protected_inject_segment_prefers_smallest_then_latest():
    assert tp.pick_protected_inject_segment([["a", "b"], ["c"]]) == 1   # smallest
    # tie on size -> the LATER segment (closing hold)
    assert tp.pick_protected_inject_segment([["a"], ["b"]]) == 1
    assert tp.pick_protected_inject_segment([]) == -1


# ---- filler-beat drop (build #3) -------------------------------------------

def test_is_filler_narration():
    assert tp.is_filler_narration("")
    assert tp.is_filler_narration("   ")
    assert tp.is_filler_narration("The scene continues.")
    assert tp.is_filler_narration("the story continues")
    assert tp.is_filler_narration("To be continued")
    assert not tp.is_filler_narration("Prince Cheon flees the dark forest.")
    assert not tp.is_filler_narration("The reason she's special is because...")


# --- compute_duration_sec: a narrated panel must COVER its voiceover ----------
# Regression for the cut-off bug: max_sec=25 was clamping the narrated duration,
# so any line whose audio ran past 25s got clipped mid-sentence in the video
# (e.g. "...had absolutely no neigong at all"). max_sec must cap only SILENT
# holds, never a panel that is playing narration.

def test_compute_duration_never_truncates_narration_audio():
    d = tp.compute_duration_sec(mode="narrated", tts_text="x", overlays=[],
                                base_min=2.5, max_sec=25.0, chars_per_sec=18.0,
                                audio_duration_sec=30.0, audio_pad_sec=0.2)
    assert abs(d - 30.2) < 1e-6            # full voiceover (+pad), NOT 25.0


def test_compute_duration_floors_short_audio_at_base_min():
    d = tp.compute_duration_sec(mode="narrated", tts_text="x", overlays=[],
                                base_min=2.5, max_sec=25.0, chars_per_sec=18.0,
                                audio_duration_sec=1.0, audio_pad_sec=0.2)
    assert abs(d - 2.5) < 1e-6             # 1.2 -> floored to base_min


def test_compute_duration_still_caps_silent_holds_at_max_sec():
    # narrated mode but NO audio -> reading-time estimate, still capped (silent
    # linger guard unchanged by the fix).
    d = tp.compute_duration_sec(mode="narrated", tts_text="y" * 1000, overlays=[],
                                base_min=2.5, max_sec=25.0, chars_per_sec=18.0,
                                audio_duration_sec=0.0, audio_pad_sec=0.2)
    assert abs(d - 25.0) < 1e-6


# ---- per-panel (one-panel-per-segment) identity ---------------------------
# Regression guard: with per-panel narration each script segment carries exactly
# one panel. build_cuts must return that one panel as the single cut — not zero
# (dropped) and not >1 (phantom extra). The _pick_for_segment closure inside
# main() also returns scene_files[:1] for a one-element list, but the observable
# contract is build_cuts.

def test_one_panel_segment_yields_exactly_one_cut():
    cuts = tp.build_cuts(["only.jpg"], 6.0, min_cut_sec=3.0)
    assert len(cuts) == 1
    assert cuts[0]["file"] == "only.jpg"
    assert abs(cuts[0]["dur"] - 6.0) < 1e-6


def test_one_panel_segment_with_selection_still_one_cut():
    # Even when the panel is tagged 'redundant' by the beats LLM, a one-element
    # segment has no alternative — it must still produce exactly one cut.
    sel = [{"scene_file": "only.jpg", "role": "redundant"}]
    cuts = tp.build_cuts(["only.jpg"], 5.0, min_cut_sec=3.0, selection=sel)
    assert len(cuts) == 1
    assert cuts[0]["file"] == "only.jpg"


# ---- C2: image-aware duration --------------------------------------------

def test_compute_duration_image_min_floors_short_audio():
    # A visually heavy panel under a SHORT line: image_min raises the dwell above
    # the audio+pad / base_min floor.
    d = tp.compute_duration_sec(mode="narrated", tts_text="x", overlays=[],
                                base_min=2.5, max_sec=25.0, chars_per_sec=18.0,
                                audio_duration_sec=1.0, audio_pad_sec=0.2,
                                image_min=3.4)
    assert abs(d - 3.4) < 1e-6


def test_compute_duration_image_min_never_truncates_long_audio():
    # A long line still governs — image_min is a FLOOR, never a cap.
    d = tp.compute_duration_sec(mode="narrated", tts_text="x", overlays=[],
                                base_min=2.5, max_sec=25.0, chars_per_sec=18.0,
                                audio_duration_sec=30.0, audio_pad_sec=0.2,
                                image_min=3.4)
    assert abs(d - 30.2) < 1e-6


def test_compute_duration_image_min_default_is_noop():
    # No image_min supplied -> identical to before (backward-compatible).
    d = tp.compute_duration_sec(mode="narrated", tts_text="x", overlays=[],
                                base_min=2.5, max_sec=25.0, chars_per_sec=18.0,
                                audio_duration_sec=1.0, audio_pad_sec=0.2)
    assert abs(d - 2.5) < 1e-6


def test_image_min_unknown_geometry_and_intensity_is_floor():
    assert tp.compute_image_min(0, 0, "") == tp.PANEL_FLOOR_SEC


def test_image_min_large_tall_panel_exceeds_small_crop():
    small = tp.compute_image_min(400, 300, "calm")
    big = tp.compute_image_min(1200, 2000, "calm")
    assert big > small >= tp.PANEL_FLOOR_SEC


def test_image_min_intensity_adds_bump():
    calm = tp.compute_image_min(800, 800, "calm")
    explosive = tp.compute_image_min(800, 800, "explosive")
    assert explosive > calm


def test_image_min_is_bounded_by_cap():
    assert tp.compute_image_min(99999, 99999, "explosive") <= tp.IMAGE_DWELL_CAP


def test_image_min_is_deterministic():
    assert tp.compute_image_min(1200, 1600, "intense") == tp.compute_image_min(1200, 1600, "intense")


def test_panel_floor_is_two_seconds():
    # C4: the per-panel cut floor backstop is 2.0s (coupled to prep_qa flash_cut).
    assert tp.PANEL_FLOOR_SEC == 2.0
