"""Transitions wave (2026-07-16): cold-open detection is ONE authority
(recap_style.is_cold_opener) consumed by prep_qa's cold_open WARN, the heal
note (which carries the previous line = the (prev, this) bridge rewrite),
and narration_punchup's bridge preservation."""
import importlib.util
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load("recap_style")
pq = _load("prep_qa")
nh = _load("narration_heal")
npu = _load("narration_punchup")


def test_is_cold_opener_precision():
    cold = [
        "The scene shows a dark ravine below.",
        "In a dark ravine, a figure stirs.",
        "A warrior stands at the cliff edge.",
        "We see the assassins regroup.",
        "An old man appears from the mist.",
    ]
    flowing = [
        "But the quiet doesn't last.",
        "That focus shatters when the branch snaps.",
        "He hits the rocks hard.",
        "Before he can react, the blade is already falling.",
        "Down in the ravine, he finally stops rolling.",
        "The blade catches him mid-turn.",       # 'The <noun>' is not a reset
    ]
    for s in cold:
        assert rs.is_cold_opener(s), s
    for s in flowing:
        assert not rs.is_cold_opener(s), s


def _beats(*first_lines):
    return {"beats": [
        {"group_id": i + 1,
         "segments": [{"span": [f"p{i:06d}.jpg"], "line": line}]}
        for i, line in enumerate(first_lines)]}


def test_cold_open_flags_skip_first_beat_and_carry_prev_line():
    beats = _beats("The scene shows a mountain range.",       # first: exempt
                   "In a dark ravine, a figure stirs.")       # flagged
    flags = pq.cold_open_flags(beats)
    assert [f["code"] for f in flags] == ["cold_open"]
    assert flags[0]["severity"] == pq.WARN
    assert "mountain range" in flags[0]["detail"]             # prev line rides
    assert flags[0]["segment_id"] == "g0002"
    # flowing chapter: silent
    assert pq.cold_open_flags(_beats(
        "The scene shows a mountain range.",
        "But the quiet doesn't last.")) == []


def test_cold_open_heals_only_under_semantic_heal():
    report = {"flags": [{"code": "cold_open", "severity": "WARN",
                         "segment_id": "g0002",
                         "detail": "beat opens cold ('In a dark ravine...') "
                                   "instead of bridging from ('the peaks "
                                   "swallow the moon.')"}]}
    assert nh.corrections_from_qa(report) == {}
    corr = nh.corrections_from_qa(report, include_grounding_warn=True)
    assert 2 in corr and "bridge" in corr[2].lower()


def test_punchup_rejects_cold_reset_of_a_flowing_line():
    orig = "But the quiet doesn't last."
    cold = "In a dark ravine, a figure stirs."
    assert not npu.validate_line(orig, cold, [])
    # a flowing punch of a flowing line passes
    assert npu.validate_line(orig, "And then the quiet breaks apart.", [])
    # an already-cold original may stay cold (no new regression introduced)
    assert npu.validate_line("The scene shows a ravine.",
                             "The scene shows a deep ravine.", [])
