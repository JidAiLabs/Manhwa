"""narration_heal: prep-QA ERROR flags -> per-group corrections for a targeted
regeneration (auto-heal re-narrates the failing group, never drops it)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "narration_heal",
    Path(__file__).resolve().parent.parent / "tools" / "narration_heal.py")
nh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nh)  # type: ignore[union-attr]


def _flag(code, sev, detail, seg):
    return {"code": code, "severity": sev, "detail": detail, "segment_id": seg}


def test_caption_unvoiced_carries_the_caption_into_the_note():
    rep = {"flags": [_flag("caption_unvoiced", "ERROR",
                           "caption text missing from narration (0% word coverage): "
                           "'BACK THEN, I HAD NO IDEA.'", "g0005_p04")]}
    corr = nh.corrections_from_qa(rep)
    assert set(corr) == {5}
    assert "BACK THEN, I HAD NO IDEA." in corr[5]


def test_only_error_and_healable_codes_map_to_groups():
    rep = {"flags": [
        _flag("chrome_narration", "ERROR", "mentions view count", "g0003_p02"),
        _flag("caption_paraphrased", "WARN", "ok by judge", "g0002_p01"),   # WARN -> skip
        _flag("panel_substituted", "ERROR", "held stand-in", "g0009_p02"),  # not healable
        _flag("fragment_dangle", "ERROR", "half a sentence", "g0013_p00"),
    ]}
    corr = nh.corrections_from_qa(rep)
    assert set(corr) == {3, 13}                      # only healable ERRORs
    assert "view count" in corr[3] or "format" in corr[3]


def test_multiple_errors_on_one_group_combine():
    rep = {"flags": [
        _flag("caption_unvoiced", "ERROR", "... : 'HELLO'", "g0007_p00"),
        _flag("chrome_narration", "ERROR", "screenshot", "g0007_p00"),
    ]}
    corr = nh.corrections_from_qa(rep)
    assert set(corr) == {7}
    assert "HELLO" in corr[7] and ("view count" in corr[7] or "format" in corr[7])


def test_chrome_narration_heals_even_as_a_warning():
    # the channel voices NO interface chatter, so a chrome/meta leak is healed at
    # ANY severity (other codes only heal as ERRORs)
    rep = {"flags": [
        _flag("chrome_narration", "WARN", "mentions 'view count'", "g0003_p02"),
        _flag("flash_cut", "WARN", "too short", "g0004_p03"),   # other WARN -> skip
    ]}
    corr = nh.corrections_from_qa(rep)
    assert set(corr) == {3}
    assert "interface" in corr[3].lower() or "view count" in corr[3]


def test_empty_report_is_empty():
    assert nh.corrections_from_qa({}) == {}
    assert nh.corrections_from_qa({"flags": []}) == {}


# --- grounding_weak: QA-eyes report by default, opt-in quality heal ----------

def test_grounding_weak_warn_is_report_only_by_default():
    rep = {"flags": [_flag("grounding_weak", "WARN",
                           "weak/mis-grounded narration: beasts called 'dogs'",
                           "g0004_p03")]}
    corr = nh.corrections_from_qa(rep)
    assert corr == {}


def test_grounding_weak_warn_heals_when_enabled():
    rep = {"flags": [_flag("grounding_weak", "WARN",
                           "weak/mis-grounded narration: beasts called 'dogs'",
                           "g0004_p03")]}
    corr = nh.corrections_from_qa(rep, include_grounding_warn=True)
    assert 4 in corr
    note = corr[4].lower()
    assert "mis-grounded" in note or "name exactly what the panel shows" in note
    assert "dogs" in note          # the specific issue is threaded into the note


def test_grounding_weak_error_still_heals():
    rep = {"flags": [_flag("grounding_weak", "ERROR",
                           "weak/mis-grounded narration: beasts called 'dogs'",
                           "g0004_p03")]}
    assert set(nh.corrections_from_qa(rep)) == {4}


def test_grounding_weak_in_healable_set():
    assert "grounding_weak" in nh.HEALABLE


def test_grounding_weak_without_segment_is_skipped():
    rep = {"flags": [_flag("grounding_weak", "WARN", "weak: x", None)]}
    assert nh.corrections_from_qa(rep) == {}


# ---- panel_narration coverage: heal path preserves group granularity ---------
# narration_heal.corrections_from_qa produces a {group_id: note} dict keyed
# on INTEGER group ids. gemini_narrative_pass re-narrates each flagged group
# from its source panels and calls align_panel_narration to rebuild
# panel_narration — so the 1:1 alignment invariant is satisfied by the
# subprocess, not by narration_heal itself (which has no manifest access).
#
# What we can guard here: the corrections dict is keyed on group_id integers
# (not segment ids), so the re-narration subprocess receives the right group
# context to rebuild panel_narration for all panels in that group.

def test_heal_corrections_keyed_on_group_id_int_for_realignment():
    """A flagged segment g0003_p01 maps to group 3 — the integer group_id
    that gemini_narrative_pass uses to look up the group's scene_files and
    rebuild panel_narration.  If the key were a string or segment-id the
    re-narration would silently get no panels and produce no panel_narration."""
    rep = {"flags": [_flag("caption_unvoiced", "ERROR",
                           "caption missing: 'I WILL BECOME THE STRONGEST'",
                           "g0003_p01")]}
    corr = nh.corrections_from_qa(rep)
    # key must be a plain int (group 3), not "g0003_p01" or "3"
    assert 3 in corr
    assert isinstance(list(corr.keys())[0], int)


def test_heal_multi_panel_group_single_correction_entry():
    """A group with three panels that generated two different QA flags
    (on two different segment ids) still produces exactly ONE correction
    entry — so gemini_narrative_pass re-narrates the whole group once and
    align_panel_narration covers all three panels."""
    rep = {"flags": [
        _flag("caption_unvoiced", "ERROR", "... : 'PANEL 1'", "g0007_p00"),
        _flag("chrome_narration", "ERROR", "screenshot", "g0007_p02"),
    ]}
    corr = nh.corrections_from_qa(rep)
    assert list(corr.keys()) == [7]          # one entry, not two
    assert "PANEL 1" in corr[7]             # both notes merged
