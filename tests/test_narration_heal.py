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


def test_empty_report_is_empty():
    assert nh.corrections_from_qa({}) == {}
    assert nh.corrections_from_qa({"flags": []}) == {}
