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
                         ["Prince Cheon"], "full", genre="murim")
    for needle in ("zip code", "NEVER invent", "Prince Cheon",
                   "JSON array", "HUMOR=full"):
        assert needle in p


def test_genre_addons_change_the_comedy_axis():
    murim = npu.build_prompt([{"group_id": 1, "narration": "x"}], [],
                             "full", genre="murim")
    modern = npu.build_prompt([{"group_id": 1, "narration": "x"}], [],
                              "full", genre="modern apocalypse thriller")
    system = npu.build_prompt([{"group_id": 1, "narration": "x"}], [],
                              "full", genre="reincarnation system fantasy")
    # murim keeps the ancient-world anachronism engine
    assert "ancient setting" in murim and "sect" in murim
    # modern settings must NOT be told to use ancient-world anachronisms —
    # the world IS modern; contrast axis is mundane vs apocalypse
    assert "ancient setting" not in modern
    assert "mundane" in modern
    # system/regression: tutorial/newbie framing
    assert "tutorial" in system
    # unknown genre falls back to the neutral persona only
    generic = npu.build_prompt([{"group_id": 1, "narration": "x"}], [],
                               "full", genre="")
    assert "ancient setting" not in generic


def test_genre_key_mapping():
    assert npu.genre_key("Murim martial arts action") == "murim"
    assert npu.genre_key("wuxia cultivation") == "murim"
    assert npu.genre_key("modern apocalypse regression") == "modern"
    assert npu.genre_key("Sci-Fi, Reincarnation, System") == "system"
    assert npu.genre_key("shoujo romance") == "generic"


# ---- parser resilience + caption protection (wired into beated stage) -------

def test_extract_json_array_fenced_and_truncated():
    raw = ('```json\n[{"group_id": 1, "narration": "a"},\n'
           '{"group_id": 2, "narration": "b"},\n{"group_id": 3, "narr')
    arr = npu._extract_json_array(raw)
    assert [d["group_id"] for d in arr] == [1, 2]
    raw2 = 'noise ```json [ {"group_id": 3, "narration": "c"} ] ``` tail'
    assert npu._extract_json_array(raw2)[0]["group_id"] == 3


def test_validate_line_protects_caption_words():
    req = {"only", "person", "knew", "world", "going", "end"}
    orig = ("He became the only person who knew how the world was "
            "going to end.")
    assert npu.validate_line(
        orig, "Spoiler: he basically pre-read the apocalypse patch notes.",
        [], required=req) is False
    assert npu.validate_line(
        orig, "And just like that, he became the only person who knew how "
        "the world was going to end.", [], required=req) is True


def test_merge_keeps_grounded_original_for_caption_groups():
    beats = {"beats": [{"group_id": 9, "narration":
                        "He became the only person who knew how the world "
                        "was going to end."}]}
    punched = [{"group_id": 9, "narration":
                "Spoiler: he pre-read the apocalypse patch notes."}]
    out = npu.merge(beats, punched, [], caption_words={
        9: {"only", "person", "knew", "world", "going", "end"}})
    assert out["beats"][0]["narration"].startswith("He became the only")
    assert out["stats"]["punchup_applied"] == 0


def test_config_wires_punchup_default_full(tmp_path):
    from studio.config import load
    toml = tmp_path / "studio.toml"
    toml.write_text("")
    assert load(toml).punchup == "full"
    toml.write_text("[models]\npunchup = \"off\"\n")
    assert load(toml).punchup == "off"


def test_caption_guard_is_per_scene_not_union():
    """A group with TWO captions must keep BOTH — union coverage is not
    enough (job 23 regression: 'ON THE DAY I FINISHED...' dropped while
    'BACK THEN...' survived)."""
    req = [{"back", "then", "i", "had", "no", "idea"},
           {"on", "the", "day", "i", "finished", "web", "novel"}]
    orig = ("Back then, I had no idea... on the day I finished the web "
            "novel, everything changed.")
    keeps_one = "Back then, I had no idea what was coming for me at all."
    assert npu.validate_line(orig, keeps_one, [], required=req) is False
    keeps_both = ("Back then, I had no idea that the day I finished the "
                  "web novel would flip everything.")
    assert npu.validate_line(orig, keeps_both, [], required=req) is True


def test_merge_is_idempotent_on_punched_files():
    """No valid candidate -> BOTH fields return to the grounded line; a
    stale punch must never survive a failed re-validation."""
    beats = {"beats": [{"group_id": 1,
                        "narration": "PUNCHED: zero active players.",
                        "narration_plain": "A web novel no one read."}]}
    out = npu.merge(beats, [], [])
    assert out["beats"][0]["narration_plain"] == "A web novel no one read."
    assert out["beats"][0]["narration"] == "A web novel no one read."
