"""
tests/test_protected_cards.py

TDD for the dropped-story-card bug: an in-world STYLED TEXT / SYSTEM / INFO card
(panel_kind=story, short mostly-caps phrase, low text_coverage) was excluded by
text_context_only_panel — because the detector mis-boxed the styled card as a
"speech bubble" subject — so protected_story_files dropped it and build_cuts
removed it from the video. Concrete cases (Nano Ch1): "SKY CORPORATION." and
"7TH GENERATION NANO MACHINE, STARTING ACTIVATION." are PLOT and must be SHOWN.

These cards must end up protected. A pure speech bubble (lowercase conversational
text over little art) must STILL be excludable — only the styled-card case is
rescued, never every text panel.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "timeline_planner",
    Path(__file__).resolve().parent.parent / "tools" / "timeline_planner.py",
)
tp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tp)  # type: ignore[union-attr]

_PQ_SPEC = importlib.util.spec_from_file_location(
    "prep_qa",
    Path(__file__).resolve().parent.parent / "tools" / "prep_qa.py",
)
pq = importlib.util.module_from_spec(_PQ_SPEC)
_PQ_SPEC.loader.exec_module(pq)  # type: ignore[union-attr]


# ---- looks_like_system_card: the manifest-level title/system-card signal ------

def test_looks_like_system_card_sky_corporation():
    it = {"scene_file": "p000113.jpg", "panel_kind": "story",
          "subjects": ["speech bubble"], "ocr_clean": "SKY CORPORATION.",
          "text_coverage": 0.04}
    assert tp.looks_like_system_card(it) is True


def test_looks_like_system_card_starting_activation():
    it = {"scene_file": "p.jpg", "panel_kind": "story",
          "subjects": ["text"],
          "ocr_clean": "7TH GENERATION NANO MACHINE, STARTING ACTIVATION.",
          "text_coverage": 0.06}
    assert tp.looks_like_system_card(it) is True


def test_looks_like_system_card_rejects_conversational_bubble():
    # lowercase conversational dialogue is NOT a styled card
    it = {"scene_file": "b.jpg", "panel_kind": "story",
          "subjects": ["speech bubble"], "ocr_clean": "what is this place?",
          "text_coverage": 0.05}
    assert tp.looks_like_system_card(it) is False


def test_looks_like_system_card_rejects_caps_dialogue_high_coverage():
    # caps SHOUT in a big bubble (high text_coverage) is dialogue, not a card
    it = {"scene_file": "b.jpg", "panel_kind": "story",
          "subjects": ["speech bubble"],
          "ocr_clean": "AS I THOUGHT, THIS GUY IS A GENIUS!",
          "text_coverage": 0.1552}
    assert tp.looks_like_system_card(it) is False


def test_looks_like_system_card_rejects_chrome():
    it = {"scene_file": "t.jpg", "panel_kind": "chrome",
          "subjects": ["title logo"],
          "ocr_clean": "Nano Machine CHAPTER 7", "text_coverage": 0.05}
    assert tp.looks_like_system_card(it) is False


def test_looks_like_system_card_rejects_no_ocr():
    it = {"scene_file": "a.jpg", "panel_kind": "story",
          "subjects": ["young man"], "ocr_clean": "", "text_coverage": 0.0}
    assert tp.looks_like_system_card(it) is False


# ---- protected_story_files: the styled card is rescued from the redundant drop -

def test_protected_story_rescues_sky_corporation_card(tmp_path):
    vision = {"items": [
        # styled in-world card the detector mis-boxed as a speech bubble:
        # text_context_only_panel would drop it, but it is PLOT.
        {"scene_file": "scenes/p000113.jpg", "panel_kind": "story",
         "subjects": ["speech bubble"], "ocr_clean": "SKY CORPORATION.",
         "text_coverage": 0.04},
    ]}
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    assert "p000113.jpg" in tp.protected_story_files(str(vp))


def test_protected_story_rescues_nano_activation_card(tmp_path):
    vision = {"items": [
        {"scene_file": "p_activate.jpg", "panel_kind": "story",
         "subjects": ["text"],
         "ocr_clean": "7TH GENERATION NANO MACHINE, STARTING ACTIVATION.",
         "text_coverage": 0.06},
    ]}
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    assert "p_activate.jpg" in tp.protected_story_files(str(vp))


def test_protected_story_still_excludes_speech_bubble(tmp_path):
    # the conservative invariant: a plain conversational speech-bubble panel is
    # NOT force-protected by the new card path (existing text_context_only kept).
    vision = {"items": [
        {"scene_file": "bubble.jpg", "panel_kind": "story",
         "subjects": ["speech bubble"], "ocr_clean": "what is this place?",
         "text_coverage": 0.05},
    ]}
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    assert "bubble.jpg" not in tp.protected_story_files(str(vp))


def test_protected_story_keeps_normal_art_panel(tmp_path):
    # a normal art panel (little text) is protected as before
    vision = {"items": [
        {"scene_file": "art.jpg", "panel_kind": "story",
         "subjects": ["a swordsman", "a mountain"], "ocr_clean": "",
         "text_coverage": 0.0},
    ]}
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps(vision))
    assert "art.jpg" in tp.protected_story_files(str(vp))


# ---- END-TO-END: the group-protected card propagates into a rendered segment --
# The half-working protection: protected_story_files() now includes the in-world
# card AND the group keeps it in scene_files, but the planner emits ONE item per
# SCRIPT segment whose panels come from the script's per-shot list — which the
# narration LLM trimmed of the card ('redundant'). Drive the real main() loop and
# assert the card actually lands in a rendered cut. Control: a non-protected
# redundant panel the per-shot list excluded stays dropped.

def _write_planner_inputs(tmp_path, *, group_scene_files, shot_scene_files):
    vision = {"items": [
        # the in-world system card -> protected_story_files protects it
        {"scene_file": "card.jpg", "panel_kind": "story",
         "subjects": ["text"], "ocr_clean": "SKY CORPORATION.",
         "text_coverage": 0.04},
        {"scene_file": "art1.jpg", "panel_kind": "story",
         "subjects": ["a swordsman"], "ocr_clean": "", "text_coverage": 0.0},
        {"scene_file": "art2.jpg", "panel_kind": "story",
         "subjects": ["a mountain"], "ocr_clean": "", "text_coverage": 0.0},
        # a non-protected caption card the per-shot list excluded: its words ride
        # the narration, so it must NOT be re-injected (only story/system cards are)
        {"scene_file": "dup.jpg", "panel_kind": "caption",
         "subjects": ["text"], "ocr_clean": "Meanwhile...", "text_coverage": 0.2},
    ]}
    groups = {"groups": [
        {"group_id": 1, "shot_id": 1, "segment": "present",
         "scene_files": group_scene_files},
    ]}
    script = {"sections": [
        {"section_index": 0,
         "script_paragraphs": [
             {"text": "A swordsman climbs the mountain toward the corporation."}],
         "shots": [
             {"group_id": 1, "segment_id": "g0001_p00",
              "scene_files": shot_scene_files}],
         }]}
    vp = tmp_path / "manifest.vision.json"; vp.write_text(json.dumps(vision))
    gp = tmp_path / "manifest.groups.json"; gp.write_text(json.dumps(groups))
    sp = tmp_path / "manifest.script.json"; sp.write_text(json.dumps(script))
    outp = tmp_path / "render.plan.json"
    return str(vp), str(gp), str(sp), str(outp)


def _run_planner(tmp_path, *, group_scene_files, shot_scene_files):
    import sys
    vp, gp, sp, outp = _write_planner_inputs(
        tmp_path, group_scene_files=group_scene_files,
        shot_scene_files=shot_scene_files)
    argv = ["timeline_planner", "--groups", gp, "--script", sp,
            "--vision", vp, "--out", outp, "--mode", "narrated"]
    old = sys.argv
    try:
        sys.argv = argv
        tp.main()
    finally:
        sys.argv = old
    return json.loads(Path(outp).read_text())


def test_e2e_protected_card_excluded_by_shot_still_renders(tmp_path):
    # the card IS in the group's scene_files but the script's per-shot list
    # EXCLUDED it -> it must still appear in some segment's cuts.
    plan = _run_planner(
        tmp_path,
        group_scene_files=["card.jpg", "art1.jpg", "art2.jpg"],
        shot_scene_files=["art1.jpg", "art2.jpg"])           # card omitted by LLM
    rendered = {c["file"] for item in plan["timeline"] for c in item.get("cuts", [])}
    assert "card.jpg" in rendered            # protected card propagated + rendered


def test_e2e_non_protected_excluded_panel_stays_dropped(tmp_path):
    # 'dup.jpg' is a non-protected near-duplicate the per-shot list excluded ->
    # it must NOT be re-injected (only protected cards are).
    plan = _run_planner(
        tmp_path,
        group_scene_files=["card.jpg", "art1.jpg", "dup.jpg"],
        shot_scene_files=["art1.jpg"])                       # both card+dup omitted
    rendered = {c["file"] for item in plan["timeline"] for c in item.get("cuts", [])}
    assert "card.jpg" in rendered            # protected card injected
    assert "dup.jpg" not in rendered         # non-protected drop stays dropped


def test_e2e_panel_rich_group_unaffected(tmp_path):
    # no protected card missing -> the per-shot selection is untouched (the LLM's
    # two-panel pick renders exactly, no injection).
    plan = _run_planner(
        tmp_path,
        group_scene_files=["art1.jpg", "art2.jpg", "dup.jpg"],
        shot_scene_files=["art1.jpg", "art2.jpg"])
    rendered = {c["file"] for item in plan["timeline"] for c in item.get("cuts", [])}
    assert rendered == {"art1.jpg", "art2.jpg"}   # exactly the picks, nothing extra


# =============================================================================
# FIX 3 — a panel stamped panel_kind=='system' is NEVER dropped from the cuts.
# protected_system_files defers ENTIRELY to the stamped kind (the SAME predicate
# render_prep.system_files + prep_qa.system_coverage_flags use), so the planner's
# drop filters, the fit-trim and the QA gate agree by construction — a stamped
# system card that MISSES the OCR title-card heuristic is still shown.
# =============================================================================

def test_protected_system_files_protects_card_failing_ocr_heuristic(tmp_path):
    # a stamped-system card whose OCR is mixed-case with an ellipsis: the caps/
    # flat title-card heuristic REJECTS it, but the stamped kind protects it.
    sys_item = {"scene_file": "scenes/sys.jpg", "panel_kind": "system",
                "subjects": ["speech bubble"], "ocr_clean": "Loading the world...",
                "text_coverage": 0.05}
    assert tp.looks_like_system_card(sys_item) is False     # heuristic misses it
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps({"items": [sys_item]}))
    assert "sys.jpg" in tp.protected_system_files(str(vp))  # stamp protects it


def test_protected_system_files_ignores_caption_panel(tmp_path):
    # a caption panel (words ride the narration, folded) is NOT a system card
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps({"items": [
        {"scene_file": "cap.jpg", "panel_kind": "caption",
         "subjects": ["text"], "ocr_clean": "Meanwhile...", "text_coverage": 0.2},
        {"scene_file": "sys.jpg", "panel_kind": "system",
         "ocr_clean": "SYSTEM ALERT", "text_coverage": 0.05},
    ]}))
    out = tp.protected_system_files(str(vp))
    assert out == {"sys.jpg"}                    # only the system card
    assert "cap.jpg" not in out


def test_protected_system_files_missing_manifest_is_empty():
    assert tp.protected_system_files("") == set()          # degrade, never raises


def test_build_cuts_must_show_keeps_system_card_under_aggressive_trim():
    # a fit-trim that affords only ONE slot (2.0s @ 2.0 floor) still pins the
    # system card via must_show, even though the BROAD protected set (every panel)
    # would otherwise let the leading story panels take the slot.
    trim = tp.build_cuts(["a.jpg", "b.jpg", "sys.jpg"], 2.0,
                         min_cut_sec=2.0, floor=2.0, trim_to_fit=True,
                         protected={"a.jpg", "b.jpg", "sys.jpg"},
                         must_show={"sys.jpg"})
    files = [c["file"] for c in trim]
    assert "sys.jpg" in files                    # system card pinned, shown
    assert all(c["dur"] >= 2.0 for c in trim)    # never a flash


def test_filter_scene_files_keeps_system_card_drops_blank(tmp_path):
    # is_bad_panel reads a flat notification card as 'blank'. keep_files (the
    # stamped system set) exempts it from the husk drop; a genuine non-system
    # blank with the SAME flat pixels still drops.
    import cv2
    import numpy as np
    clean = tmp_path / "clean"; clean.mkdir()
    flat = np.full((240, 320, 3), 255, np.uint8)           # flat white -> "blank"
    cv2.imwrite(str(clean / "sys.jpg"), flat)
    cv2.imwrite(str(clean / "blank.jpg"), flat.copy())
    kept, dropped = tp.filter_scene_files(
        files=["sys.jpg", "blank.jpg"], clean_dir=str(clean), raw_dir="",
        prefer_clean=True,
        blank_dom_ratio_thr=0.975, blank_std_thr=6.0,
        strip_white_ratio_thr=0.82, strip_bbox_h_frac_thr=0.25,
        bubble_dom_ratio_thr=0.88, bubble_edge_mean_thr=0.055,
        bubble_std_thr=14.0,
        keep_files={"sys.jpg"})
    assert "sys.jpg" in kept                     # exempt system card kept
    assert "blank.jpg" not in kept               # real blank still dropped
    assert any(d["file"] == "blank.jpg" for d in dropped)


def test_system_card_survives_planner_and_qa_clean(tmp_path):
    # CROSS-STAGE INVARIANT: a stamped-system card the script's per-shot list
    # EXCLUDED, mis-boxed as a 'speech bubble' subject (so the montage caption-drop
    # WOULD fold it) AND failing the OCR title-card heuristic (mixed case, ellipsis)
    # is pinned back by the planner protection at every stage, so the QA gate that
    # keys on the SAME panel_kind=='system' predicate reports NO system_card_unshown.
    import sys
    vision = {"items": [
        {"scene_file": "sys.jpg", "panel_kind": "system",
         "subjects": ["speech bubble"], "ocr_clean": "Loading the world...",
         "text_coverage": 0.05},
        {"scene_file": "art1.jpg", "panel_kind": "story",
         "subjects": ["a swordsman"], "ocr_clean": "", "text_coverage": 0.0},
    ]}
    groups = {"groups": [
        {"group_id": 1, "shot_id": 1, "segment": "present",
         "scene_files": ["sys.jpg", "art1.jpg"]}]}
    script = {"sections": [
        {"section_index": 0,
         "script_paragraphs": [{"text": "A swordsman steps into the loading world."}],
         "shots": [{"group_id": 1, "segment_id": "g0001_p00",
                    "scene_files": ["art1.jpg"]}]}]}          # sys.jpg omitted by LLM
    vp = tmp_path / "manifest.vision.json"; vp.write_text(json.dumps(vision))
    gp = tmp_path / "manifest.groups.json"; gp.write_text(json.dumps(groups))
    sp = tmp_path / "manifest.script.json"; sp.write_text(json.dumps(script))
    outp = tmp_path / "render.plan.json"
    argv = ["timeline_planner", "--groups", str(gp), "--script", str(sp),
            "--vision", str(vp), "--out", str(outp), "--mode", "narrated"]
    old = sys.argv
    try:
        sys.argv = argv
        tp.main()
    finally:
        sys.argv = old
    plan = json.loads(outp.read_text())
    rendered = {c["file"] for it in plan["timeline"] for c in it.get("cuts", [])}
    assert "sys.jpg" in rendered                  # shown despite the LLM omission
    # the QA gate (same stamped predicate) is therefore clean
    beats = {"beats": [{"group_id": 1, "scene_files": ["sys.jpg", "art1.jpg"]}]}
    vitems = {"sys.jpg": {"panel_kind": "system"},
              "art1.jpg": {"panel_kind": "story"}}
    assert pq.system_coverage_flags(beats, plan, vitems) == []
