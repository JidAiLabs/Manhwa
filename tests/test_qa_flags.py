"""
tests/test_qa_flags.py

TDD tests for studio.qa_flags — the automated QA "confidence instrument".

These are pure-data functions (no image I/O): they consume the same manifest
dicts the pipeline already produces and emit per-scene / per-group flags plus a
summary scorecard scored against the SP2 acceptance criteria.

Manifest field shapes used (verified against real Nano Machine ch1 manifests):
  manifest.scenes.json  → {"scenes": [{"out_file": str, "dhash64": int,
                                        "blank_score": float, "edge_density": float}]}
  manifest.vision.json  → {"items": [{"scene_file": str, "ocr_clean": str,
                                       "text_only": bool,
                                       "targets": [{"type": "text_block", "bbox": [y0,x0,y1,x1]}]}]}
  manifest.groups.json  → {"shots": [{"shot_id": int, "scene_files": [str, ...]}]}
  manifest.script.json  → {"sections": [{"script_paragraphs": [str, ...],
                                          "shots": [{"group_id": int, "duration_s": float,
                                                     "scene_files": [str, ...]}]}]}
"""

from __future__ import annotations

import pytest

from studio import qa_flags


# ---------------------------------------------------------------------------
# hamming64
# ---------------------------------------------------------------------------

def test_hamming64_identical_is_zero():
    assert qa_flags.hamming64(0xDEADBEEF, 0xDEADBEEF) == 0


def test_hamming64_counts_differing_bits():
    # 0b1010 vs 0b0001 differ in 3 bits
    assert qa_flags.hamming64(0b1010, 0b0001) == 3


# ---------------------------------------------------------------------------
# near_duplicate_pairs
# ---------------------------------------------------------------------------

def test_near_duplicate_pairs_finds_identical_dhash():
    scenes = [
        {"out_file": "a.jpg", "dhash64": 100},
        {"out_file": "b.jpg", "dhash64": 100},   # identical → dup
        {"out_file": "c.jpg", "dhash64": 0xFFFF_FFFF_FFFF_FFFF},  # far away
    ]
    pairs = qa_flags.near_duplicate_pairs(scenes, max_hamming=8)
    assert len(pairs) == 1
    p = pairs[0]
    assert {p["a"], p["b"]} == {"a.jpg", "b.jpg"}
    assert p["hamming"] == 0


def test_near_duplicate_pairs_respects_threshold():
    scenes = [
        {"out_file": "a.jpg", "dhash64": 0b0000},
        {"out_file": "b.jpg", "dhash64": 0b1111},  # hamming 4
    ]
    assert qa_flags.near_duplicate_pairs(scenes, max_hamming=3) == []
    assert len(qa_flags.near_duplicate_pairs(scenes, max_hamming=4)) == 1


def test_near_duplicate_pairs_ignores_missing_dhash():
    scenes = [
        {"out_file": "a.jpg", "dhash64": 5},
        {"out_file": "b.jpg"},               # no dhash → not compared
        {"out_file": "c.jpg", "dhash64": None},
    ]
    assert qa_flags.near_duplicate_pairs(scenes) == []


# ---------------------------------------------------------------------------
# text_block_area_frac
# ---------------------------------------------------------------------------

def test_text_block_area_frac_sums_text_blocks_only():
    item = {
        "targets": [
            {"type": "frame", "bbox": [0.0, 0.0, 1.0, 1.0]},          # ignored
            {"type": "text_block", "bbox": [0.0, 0.0, 0.5, 0.5]},     # area 0.25
            {"type": "object", "bbox": [0.0, 0.0, 1.0, 1.0]},        # ignored
        ]
    }
    assert qa_flags.text_block_area_frac(item) == pytest.approx(0.25)


def test_text_block_area_frac_clamps_to_one():
    item = {
        "targets": [
            {"type": "text_block", "bbox": [0.0, 0.0, 1.0, 1.0]},
            {"type": "text_block", "bbox": [0.0, 0.0, 1.0, 1.0]},
        ]
    }
    assert qa_flags.text_block_area_frac(item) == pytest.approx(1.0)


def test_text_block_area_frac_no_targets_is_zero():
    assert qa_flags.text_block_area_frac({}) == 0.0


# ---------------------------------------------------------------------------
# longest_common_run (OCR-echo detection)
# ---------------------------------------------------------------------------

def test_longest_common_run_detects_verbatim_echo():
    narration = "He stares ahead and whispers it is not over yet to himself."
    ocr = "IT IS NOT OVER YET"
    run = qa_flags.longest_common_run(narration, ocr, min_words=4)
    assert run.lower() == "it is not over yet"


def test_longest_common_run_below_min_returns_empty():
    narration = "A quiet night falls over the city."
    ocr = "night falls"  # only 2 words shared
    assert qa_flags.longest_common_run(narration, ocr, min_words=4) == ""


def test_longest_common_run_case_and_punctuation_insensitive():
    assert qa_flags.longest_common_run(
        "well, THE world will burn today!", "the world will burn", min_words=4
    ).lower() == "the world will burn"


# ---------------------------------------------------------------------------
# seconds_per_scene
# ---------------------------------------------------------------------------

def test_seconds_per_scene_divides_duration():
    shot = {"duration_s": 12.0, "scene_files": ["a.jpg", "b.jpg", "c.jpg"]}
    assert qa_flags.seconds_per_scene(shot) == pytest.approx(4.0)


def test_seconds_per_scene_no_scenes_is_zero():
    assert qa_flags.seconds_per_scene({"duration_s": 5.0, "scene_files": []}) == 0.0


# ---------------------------------------------------------------------------
# compute_flags — the scorecard
# ---------------------------------------------------------------------------

def _fixture_manifests():
    scenes = {
        "scenes": [
            {"out_file": "p0.jpg", "dhash64": 10},
            {"out_file": "p1.jpg", "dhash64": 10},   # dup of p0
            {"out_file": "p2.jpg", "dhash64": 0xABCDEF},
            {"out_file": "p3.jpg", "dhash64": 0x123456},
        ]
    }
    vision = {
        "items": [
            {"scene_file": "p0.jpg", "ocr_clean": "RUN NOW OR DIE HERE", "targets": []},
            {"scene_file": "p1.jpg", "ocr_clean": "", "targets": []},
            {"scene_file": "p2.jpg", "ocr_clean": "PEASANT BLOOD STAINS THE SNOW", "text_only": True,
             "targets": [{"type": "text_block", "bbox": [0.0, 0.0, 0.6, 0.9]}]},  # text-dominated
            {"scene_file": "p3.jpg", "ocr_clean": "", "targets": []},
        ]
    }
    groups = {
        "shots": [
            {"shot_id": 1, "scene_files": ["p0.jpg", "p1.jpg"]},
            {"shot_id": 2, "scene_files": ["p2.jpg", "p3.jpg"]},
        ]
    }
    script = {
        "sections": [
            {
                "script_paragraphs": [
                    "He screams run now or die here as the blade falls.",  # echoes p0 OCR
                    "The snow drinks the fallen.",
                ],
                "shots": [
                    {"group_id": 1, "duration_s": 4.0, "scene_files": ["p0.jpg", "p1.jpg"]},  # 2s/pic → short
                    {"group_id": 2, "duration_s": 10.0, "scene_files": ["p2.jpg", "p3.jpg"]},
                ],
            }
        ]
    }
    return scenes, vision, groups, script


def test_compute_flags_scorecard_counts():
    scenes, vision, groups, script = _fixture_manifests()
    out = qa_flags.compute_flags(
        scenes=scenes, vision_items=vision, groups=groups, script=script,
        source_page_count=2, min_sec_per_pic=3.5, text_frac=0.30,
    )
    sc = out["scorecard"]
    assert sc["total_scenes"] == 4
    assert sc["source_pages"] == 2
    assert sc["scenes_per_page"] == pytest.approx(2.0)
    assert sc["near_dup_pairs"] == 1          # p0/p1
    assert sc["text_dominated"] == 1          # p2
    assert sc["short_pictures"] == 2          # p0,p1 @ 2s each
    assert sc["ocr_echo"] == 1                # group 1 paragraph echoes p0 OCR


def test_compute_flags_marks_scene_and_group_flags():
    scenes, vision, groups, script = _fixture_manifests()
    out = qa_flags.compute_flags(
        scenes=scenes, vision_items=vision, groups=groups, script=script,
        source_page_count=2, min_sec_per_pic=3.5, text_frac=0.30,
    )
    # p0 is flagged as a near-duplicate and as too-short
    p0 = out["scene_flags"].get("p0.jpg", [])
    assert any(f["kind"] == "near_duplicate" for f in p0)
    assert any(f["kind"] == "short_on_screen" for f in p0)
    # p2 flagged text-dominated
    p2 = out["scene_flags"].get("p2.jpg", [])
    assert any(f["kind"] == "text_dominated" for f in p2)
    # group 1 flagged OCR-echo
    g1 = out["group_flags"].get(1, [])
    assert any(f["kind"] == "ocr_echo" for f in g1)


def test_compute_flags_detects_scene_set_drift():
    scenes, vision, groups, script = _fixture_manifests()
    # groups references a scene file that is NOT in the scenes manifest
    groups["shots"][0]["scene_files"] = ["p0.jpg", "GHOST.jpg"]
    out = qa_flags.compute_flags(
        scenes=scenes, vision_items=vision, groups=groups, script=script,
        source_page_count=2,
    )
    assert out["scorecard"]["scene_set_drift"] is True
    assert "GHOST.jpg" in out["scorecard"]["drift_missing_files"]


def test_compute_flags_missing_narration_group():
    scenes, vision, groups, script = _fixture_manifests()
    # remove narration for group 2
    script["sections"][0]["script_paragraphs"] = [
        "He screams run now or die here as the blade falls."
    ]
    script["sections"][0]["shots"] = [
        {"group_id": 1, "duration_s": 4.0, "scene_files": ["p0.jpg", "p1.jpg"]},
    ]
    out = qa_flags.compute_flags(
        scenes=scenes, vision_items=vision, groups=groups, script=script,
        source_page_count=2,
    )
    assert out["scorecard"]["missing_narration_groups"] == 1
    assert any(f["kind"] == "no_narration" for f in out["group_flags"].get(2, []))
