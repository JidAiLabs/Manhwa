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


# ---- validity floor: a dead character cannot act ----------------------------

_CAST = {"cast": [
    {"id": "beast_lord", "canonical_name": "Beast Lord", "role": "antagonist",
     "aliases": ["the beast"], "visual_description": "a woman with long dark hair"},
    {"id": "captain", "canonical_name": "the Captain", "role": "ally",
     "aliases": [], "visual_description": "a man in a heavy coat"},
]}


def _dead_beats(line_old, line_new, gid=10):
    old = [{"group_id": gid, "narration": line_old,
            "segments": [{"span": ["p000043.jpg"], "line": line_old}]}]
    new = [{"group_id": gid, "narration": line_new,
            "segments": [{"span": ["p000043.jpg"], "line": line_new}]}]
    return old, new


def test_the_judge_can_never_restore_a_line_the_ledger_says_is_impossible():
    """ORV Ep107's shape: the heal removes the dead Beast Lord from a line,
    the judge calls the incumbent equivalent and puts her back, prep_qa flags
    dead_actor again, the corrections repeat and the chapter parks."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from cast_identity import actor_noun_map
    old, new = _dead_beats("The beast lunges at the Captain again.",
                           "The Captain turns to the empty ridge.")
    noun_map = actor_noun_map(_CAST)
    accepted, decisions = ab.gate_beats(
        old, new, judge=lambda o, n: "A_better",
        dead_for=lambda b: {"Beast Lord"}, noun_map=noun_map)
    assert accepted[0]["narration"].startswith("The Captain turns")
    assert decisions[0]["verdict"] == "old_dead_actor"
    assert decisions[0]["kept"] == "new"
    # ...and with nobody dead the incumbent still wins on taste, as before
    accepted2, decisions2 = ab.gate_beats(
        old, new, judge=lambda o, n: "A_better",
        dead_for=lambda b: set(), noun_map=noun_map)
    assert accepted2[0]["narration"].startswith("The beast lunges")
    assert decisions2[0]["kept"] == "old"


def test_the_floor_is_silent_without_a_ledger():
    old, new = _dead_beats("The beast lunges at the Captain again.",
                           "The Captain turns to the empty ridge.")
    accepted, decisions = ab.gate_beats(old, new, judge=lambda o, n: "A_better")
    assert accepted[0]["narration"].startswith("The beast lunges")
    assert decisions[0]["kept"] == "old"
    assert ab.make_dead_for("") == (None, None)
    assert ab.make_dead_for("/nope/manifest.vision.json") == (None, None)


def test_make_dead_for_reads_the_ledger_beside_the_vision_manifest(tmp_path):
    import json
    (tmp_path / "manifest.vision.json").write_text("{}")
    (tmp_path / "manifest.cast.json").write_text(json.dumps(_CAST))
    (tmp_path / "manifest.ledger.json").write_text(json.dumps(
        {"beat_facts": {"g0010": {"dead_by_now": ["Beast Lord"]},
                        "g0011": {"dead_by_now": []}}}))
    dead_for, noun_map = ab.make_dead_for(str(tmp_path / "manifest.vision.json"))
    assert dead_for({"group_id": 10}) == {"Beast Lord"}
    assert dead_for({"group_id": 11}) == set()
    assert noun_map                      # the same map prep_qa's gate fires on
