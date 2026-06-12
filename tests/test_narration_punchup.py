"""Punch-up pass: persona rewrite with a hard grounding contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "narration_punchup",
    Path(__file__).resolve().parent.parent / "tools" / "narration_punchup.py")
npu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(npu)  # type: ignore[union-attr]

ORIG = ("Prince Cheon flees through the fog as the Assassins close in, "
        "his robes torn and his breath ragged.")


def test_validate_accepts_styled_same_facts():
    punched = ("Our guy Prince Cheon is speedrunning a mountain escape — "
               "robes shredded, lungs on fire, and the Assassins are "
               "closing the gap like it's a ranked match.")
    assert npu.validate_line(ORIG, punched, ["Prince Cheon"]) is True


def test_validate_rejects_dropped_cast_name():
    assert npu.validate_line(ORIG, "Some guy runs from some people, "
                             "robes torn, breath ragged, vibes bad.",
                             ["Prince Cheon"]) is False


def test_validate_rejects_blowup_and_chrome():
    assert npu.validate_line(ORIG, "He runs. " * 30, ["Prince Cheon"]) is False
    assert npu.validate_line(
        ORIG, "Prince Cheon flees — go read chapter 2 on elftoon.com!",
        ["Prince Cheon"]) is False


def test_validate_preserves_mood_tags():
    o = "[panicked] He runs for the treeline as arrows fall."
    assert npu.validate_line(o, "[panicked] Our guy books it for the "
                             "treeline, arrows raining like patch-day "
                             "complaints.", []) is True
    assert npu.validate_line(o, "Our guy books it for the treeline.",
                             []) is False     # tag dropped


def test_merge_applies_valid_keeps_original_otherwise():
    beats = {"beats": [
        {"group_id": 1, "narration": ORIG},
        {"group_id": 2, "narration": "[tense] The Assassins surround him."}]}
    punched = [
        {"group_id": 1, "narration":
         "Our guy Prince Cheon is speedrunning a mountain escape — robes "
         "shredded, breath ragged, Assassins closing in."},
        {"group_id": 2, "narration": "go read it on elftoon.com"}]  # invalid
    out = npu.merge(beats, punched, ["Prince Cheon"])
    assert "speedrunning" in out["beats"][0]["narration"]
    assert out["beats"][0]["narration_plain"] == ORIG       # original kept
    assert out["beats"][1]["narration"] == "[tense] The Assassins surround him."
    assert out["stats"]["punchup_applied"] == 1


def test_prompt_contains_persona_and_rules():
    p = npu.build_prompt([{"group_id": 1, "narration": ORIG}],
                         ["Prince Cheon"], "full")
    for needle in ("zip code", "NEVER invent", "Prince Cheon",
                   "JSON array", "HUMOR=full"):
        assert needle in p
