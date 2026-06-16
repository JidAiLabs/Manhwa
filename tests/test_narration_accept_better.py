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
