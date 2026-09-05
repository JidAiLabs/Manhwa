"""The strictly-better safeguard: auto-heal may only keep a regenerated line
when a judge says it is strictly better; every other verdict reverts to the
original. Tests the pure decision core with a stub judge (no model needed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "narration_accept_better",
    Path(__file__).resolve().parent.parent / "tools" / "narration_accept_better.py")
ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ab)  # type: ignore[union-attr]


def test_accept_new_only_on_strictly_better():
    assert ab.accept_new("B_better") is True
    for v in ("equivalent", "A_better", "", "unknown", "b_better", None):
        assert ab.accept_new(v) is False   # conservative: anything but B_better keeps A


def test_changed_groups_detects_rewrites_only():
    old = [{"group_id": 1, "narration": "a beast lunges"},
           {"group_id": 2, "narration": "he runs"},
           {"group_id": 3, "narration": "silence falls"}]
    new = [{"group_id": 1, "narration": "two dogs snarl"},      # changed
           {"group_id": 2, "narration": "he runs"},             # same
           {"group_id": 3, "narration": "  silence   falls "}]  # whitespace-only -> same
    assert ab.changed_groups(old, new) == [1]


def test_gate_keeps_new_when_strictly_better():
    old = [{"group_id": 1, "narration": "two dogs"}]
    new = [{"group_id": 1, "narration": "two snarling beasts"}]
    accepted, decisions = ab.gate_beats(old, new, judge=lambda o, n: "B_better")
    assert accepted[0]["narration"] == "two snarling beasts"
    assert decisions[0]["kept"] == "new"


def test_gate_reverts_when_not_strictly_better():
    old = [{"group_id": 1, "narration": "two snarling beasts"}]
    new = [{"group_id": 1, "narration": "some animals appear"}]
    for verdict in ("equivalent", "A_better"):
        accepted, decisions = ab.gate_beats(old, new, judge=lambda o, n: verdict)
        assert accepted[0]["narration"] == "two snarling beasts"   # reverted
        assert decisions[0]["kept"] == "old"


def test_gate_passes_unchanged_beats_without_judging():
    old = [{"group_id": 1, "narration": "kept line"},
           {"group_id": 2, "narration": "healed away"}]
    new = [{"group_id": 1, "narration": "kept line"},
           {"group_id": 2, "narration": "regenerated"}]
    judged = []

    def judge(o, n):
        judged.append(n["group_id"])
        return "equivalent"

    accepted, decisions = ab.gate_beats(old, new, judge=judge)
    assert judged == [2]                       # only the changed group is judged
    assert accepted[0]["narration"] == "kept line"
    assert accepted[1]["narration"] == "healed away"   # reverted (equivalent)


def test_gate_is_a_noop_when_nothing_changed():
    beats = [{"group_id": 1, "narration": "x"}, {"group_id": 2, "narration": "y"}]
    accepted, decisions = ab.gate_beats(beats, [dict(b) for b in beats],
                                        judge=lambda o, n: "B_better")
    assert decisions == []
    assert [b["narration"] for b in accepted] == ["x", "y"]


def test_gate_never_restores_an_unshippable_old_line():
    # nano ch1 g0026: the incumbent NAMED an image file, the judge kept
    # returning A_better, and the heal could never land. Validity outranks
    # taste — the judge is not even asked.
    old = [{"group_id": 26,
            "narration": "The sequence begins with p000110.jpg.",
            "segments": [{"span": ["p000110.jpg"],
                          "line": "The sequence begins with p000110.jpg."}]}]
    new = [{"group_id": 26,
            "narration": "He kneels beside a body that is not moving.",
            "segments": [{"span": ["p000110.jpg"],
                          "line": "He kneels beside a body that is not moving."}]}]
    judged = []

    def judge(o, n):
        judged.append(n["group_id"])
        return "A_better"

    accepted, decisions = ab.gate_beats(old, new, judge=judge)
    assert judged == []                                   # never judged
    assert accepted[0]["narration"].startswith("He kneels")
    assert decisions[0]["kept"] == "new"
    assert decisions[0]["verdict"] == "old_unshippable"


def test_gate_still_reverts_when_the_old_line_is_shippable():
    # the safety rule itself is untouched: a real incumbent still wins ties
    old = [{"group_id": 3, "narration": "He kneels beside the body.",
            "segments": [{"span": ["p1.jpg"], "line": "He kneels beside the body."}]}]
    new = [{"group_id": 3, "narration": "Someone is on the ground.",
            "segments": [{"span": ["p1.jpg"], "line": "Someone is on the ground."}]}]
    accepted, decisions = ab.gate_beats(old, new, judge=lambda o, n: "A_better")
    assert accepted[0]["narration"] == "He kneels beside the body."
    assert decisions[0]["kept"] == "old"


def _seg(words, span=("p1.jpg",)):
    line = " ".join(["word"] * words)
    return {"span": list(span), "line": line}


def test_gate_keeps_a_shorter_rewrite_of_an_over_cap_line():
    # nano ch1 g0023: the heal was fired to SHORTEN a 61-word line, came back
    # with 47, and the judge reverted it as "equivalent" — so line_overlong
    # never cleared. Measurable progress outranks taste.
    old = [{"group_id": 23, "narration": "x", "segments": [_seg(61)]}]
    new = [{"group_id": 23, "narration": "y", "segments": [_seg(47)]}]
    judged = []

    def judge(o, n):
        judged.append(n["group_id"])
        return "equivalent"

    accepted, decisions = ab.gate_beats(old, new, judge=judge)
    assert judged == []
    assert accepted[0]["segments"][0]["line"].split().__len__() == 47
    assert decisions[0]["kept"] == "new"
    assert "shorter" in decisions[0]["verdict"]


def test_gate_does_not_take_a_longer_rewrite_of_an_over_cap_line():
    # the floor is one-directional: a heal must not smuggle in a LONGER line
    old = [{"group_id": 23, "narration": "x", "segments": [_seg(47)]}]
    new = [{"group_id": 23, "narration": "y", "segments": [_seg(61)]}]
    accepted, decisions = ab.gate_beats(old, new, judge=lambda o, n: "equivalent")
    assert len(accepted[0]["segments"][0]["line"].split()) == 47   # reverted
    assert decisions[0]["kept"] == "old"


def test_gate_leaves_within_cap_rewrites_to_the_judge():
    # both fit the cap -> nothing measurable to compare, taste decides
    old = [{"group_id": 23, "narration": "x", "segments": [_seg(20)]}]
    new = [{"group_id": 23, "narration": "y", "segments": [_seg(12)]}]
    accepted, decisions = ab.gate_beats(old, new, judge=lambda o, n: "A_better")
    assert len(accepted[0]["segments"][0]["line"].split()) == 20   # reverted
    assert decisions[0]["verdict"] == "A_better"


def test_gate_keeps_a_rewrite_that_voices_more_of_the_caption():
    # ORV Ep1 g0022: the heal was fired to voice a skipped on-panel caption,
    # did it, and the judge reverted it ("g 22 A_better") — so caption_unvoiced
    # re-fired on every run. Deterministic coverage outranks taste.
    old = [{"group_id": 22, "narration": "He looks at his phone."}]
    new = [{"group_id": 22, "narration": "He has no idea what is about to happen."}]
    gaps = {"He looks at his phone.": 6, "He has no idea what is about to happen.": 1}
    judged = []

    def judge(o, n):
        judged.append(n["group_id"])
        return "A_better"

    accepted, decisions = ab.gate_beats(
        old, new, judge=judge, caption_gap=lambda b: gaps[b["narration"]])
    assert judged == []
    assert accepted[0]["narration"].startswith("He has no idea")
    assert "covers_caption" in decisions[0]["verdict"]


def test_gate_does_not_take_a_rewrite_that_voices_LESS_of_the_caption():
    old = [{"group_id": 22, "narration": "He has no idea what is about to happen."}]
    new = [{"group_id": 22, "narration": "He looks at his phone."}]
    gaps = {"He looks at his phone.": 6, "He has no idea what is about to happen.": 1}
    accepted, decisions = ab.gate_beats(
        old, new, judge=lambda o, n: "equivalent",
        caption_gap=lambda b: gaps[b["narration"]])
    assert accepted[0]["narration"].startswith("He has no idea")   # reverted
    assert decisions[0]["kept"] == "old"


def test_caption_floor_is_silent_when_the_old_line_already_covers_it():
    # Real sentences, not "a"/"b": a one-letter line is unvoiceable, so the
    # VALIDITY floor now fires before the judge is ever asked and the verdict
    # would be old_unshippable. The floor order is deliberate; this test is
    # about the caption floor staying silent, so give it shippable prose.
    old = [{"group_id": 22, "narration": "He looks at his phone."}]
    new = [{"group_id": 22, "narration": "He glances down at the screen."}]
    accepted, decisions = ab.gate_beats(
        old, new, judge=lambda o, n: "A_better", caption_gap=lambda b: 0)
    assert decisions[0]["verdict"] == "A_better"                   # judge ruled
