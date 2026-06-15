"""story_group (Pass 2): group by understanding into a CONSECUTIVE, fully-covering
partition. The repair logic is the coverage invariant — every panel lands in
exactly one shot, in order, no matter how the model mis-orders/omits."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "story_group",
    Path(__file__).resolve().parent.parent / "tools" / "story_group.py")
sg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sg)  # type: ignore[union-attr]

ORDER = ["p0", "p1", "p2", "p3", "p4"]


def _files(shots):
    return [s["scene_files"] for s in shots]


def test_basic_grouping_assigns_contiguous_shot_ids():
    shots = sg.repair_to_shots(ORDER[:3], [
        {"scene_files": ["p0", "p1"], "segment": "present", "arc_label": "intro"},
        {"scene_files": ["p2"], "segment": "present", "arc_label": "next"}])
    assert _files(shots) == [["p0", "p1"], ["p2"]]
    assert [s["shot_id"] for s in shots] == [1, 2]


def test_coverage_invariant_unassigned_panel_is_never_dropped():
    # model forgot p2 entirely -> it continues the current beat, still shown
    shots = sg.repair_to_shots(ORDER[:3], [
        {"scene_files": ["p0", "p1"]}])
    flat = [f for s in shots for f in s["scene_files"]]
    assert flat == ["p0", "p1", "p2"]                 # all 3 covered, in order


def test_model_misordering_becomes_consecutive_runs():
    # model groups p0+p2 together (non-consecutive); reading order is preserved
    shots = sg.repair_to_shots(ORDER[:3], [
        {"scene_files": ["p0", "p2"], "arc_label": "A"},
        {"scene_files": ["p1"], "arc_label": "B"}])
    assert _files(shots) == [["p0"], ["p1"], ["p2"]]  # consecutive runs only


def test_max_beat_len_splits_long_runs():
    shots = sg.repair_to_shots(ORDER, [
        {"scene_files": ORDER, "segment": "present", "arc_label": "battle"}],
        max_beat_len=2)
    assert _files(shots) == [["p0", "p1"], ["p2", "p3"], ["p4"]]
    assert all(s["arc_label"] == "battle" for s in shots)   # tag preserved


def test_flashback_segment_is_carried():
    shots = sg.repair_to_shots(ORDER[:3], [
        {"scene_files": ["p0"], "segment": "present"},
        {"scene_files": ["p1", "p2"], "segment": "flashback", "arc_label": "ten years ago"}])
    assert shots[1]["segment"] == "flashback" and shots[1]["arc_label"] == "ten years ago"
    assert shots[0]["segment"] == "present"
    # invalid segment normalizes to present
    assert sg.repair_to_shots(["x"], [{"scene_files": ["x"], "segment": "weird"}])[0]["segment"] == "present"


def test_group_panels_full_pipeline_with_stub_covers_everything():
    panels = [{"scene_file": f} for f in ORDER]
    captured = {}

    def stub(payload):
        captured["payload"] = payload
        return {"chapter": {"logline": "A lonely reader's novel becomes real.",
                            "premise": "He alone knows how the world ends."},
                "beats": [{"scene_files": ["p0", "p1"], "segment": "present"},
                          {"scene_files": ["p2", "p3", "p4"], "segment": "flashback"}]}

    shots, chapter = sg.group_panels(panels, stub, max_beat_len=4)
    flat = [f for s in shots for f in s["scene_files"]]
    assert flat == ORDER                              # full coverage
    assert captured["payload"]["panels"][0]["n"] == 0  # numbered, ordered input
    assert chapter["logline"].startswith("A lonely reader")   # spine captured
    assert sg.group_panels([], stub) == ([], {})


def test_nonstory_files_drops_chrome_empty_and_parse_failures():
    panels = [
        {"scene_file": "p0", "panel_kind": "story"},
        {"scene_file": "p1", "panel_kind": "chrome"},       # logo / end-card
        {"scene_file": "p2", "panel_kind": "empty"},        # blank / empty bubble
        {"scene_file": "p3", "panel_kind": "story", "error": "parse_failed"},  # unparsed
        {"scene_file": "p4"},                                # missing kind -> kept
    ]
    dropped = sg.nonstory_files(panels)
    assert dropped == {"p1", "p2", "p3"}                 # chrome + empty + error
    assert "p0" not in dropped and "p4" not in dropped   # real story stays
    # captions are NOT dropped — they're kept (their words ride the narration)
    assert sg.nonstory_files([{"scene_file": "c", "panel_kind": "caption"}]) == set()


def test_caption_solo_beat_folds_into_previous_same_segment_beat():
    panels = [{"scene_file": "p0", "panel_kind": "story"},
              {"scene_file": "c1", "panel_kind": "caption"},
              {"scene_file": "p2", "panel_kind": "story"}]
    assert sg.caption_files(panels) == {"c1"}
    shots = [
        {"shot_id": 1, "scene_files": ["p0"], "segment": "present", "arc_label": "a"},
        {"shot_id": 2, "scene_files": ["c1"], "segment": "present", "arc_label": "cap"},
        {"shot_id": 3, "scene_files": ["p2"], "segment": "present", "arc_label": "b"}]
    merged = sg.merge_caption_solos(shots, {"c1"})
    assert [s["scene_files"] for s in merged] == [["p0", "c1"], ["p2"]]
    assert [s["shot_id"] for s in merged] == [1, 2]      # renumbered contiguous


def test_title_card_files_protects_story_system_cards_not_chrome():
    # flat_frac pre-set so the detector skips image I/O; reuses prep_qa._is_title_card
    items = [
        {"scene_file": "c", "ocr_clean": "SKY CORPORATION.", "flat_frac": 0.9,
         "text_coverage": 0.05, "text_only": False},                       # story org card
        {"scene_file": "a", "ocr_clean": "LIN ZICHEN - AGE: 5 MONTHS", "flat_frac": 0.9,
         "text_coverage": 0.05, "text_only": False},                       # story time card
        {"scene_file": "d", "ocr_clean": "thanks for reading join our discord and subscribe now please",
         "flat_frac": 0.9, "text_coverage": 0.05, "text_only": False},     # promo chrome
    ]
    cards = sg.title_card_files(items)
    assert "c" in cards and "a" in cards     # in-world system cards protected
    assert "d" not in cards                  # long promo is chrome, not protected


def test_caption_closer_with_no_same_segment_neighbour_stays():
    shots = [
        {"shot_id": 1, "scene_files": ["p0"], "segment": "flashback", "arc_label": "x"},
        {"shot_id": 2, "scene_files": ["c9"], "segment": "present", "arc_label": "end"}]
    merged = sg.merge_caption_solos(shots, {"c9"})
    assert [s["scene_files"] for s in merged] == [["p0"], ["c9"]]   # segment differs
