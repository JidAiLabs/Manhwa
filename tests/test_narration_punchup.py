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


# ---- alternate-by-beat (cinematic) mode ----------------------------------

def test_classify_beats_by_scene_intensity():
    beats = {"beats": [
        {"group_id": 1, "scene_selection": [{"intensity": "calm"},
                                            {"intensity": "tense"}]},
        {"group_id": 2, "scene_selection": [{"intensity": "intense"}]},
        {"group_id": 3, "scene_selection": [{"intensity": "explosive"},
                                            {"intensity": "calm"}]},
        {"group_id": 4, "scene_selection": []},          # no signal
    ]}
    assert npu.classify_beats(beats) == {
        1: "CONNECTIVE", 2: "DRAMATIC", 3: "DRAMATIC", 4: "CONNECTIVE"}


def test_cinematic_prompt_tags_lines_with_class_and_both_rules():
    lines = [{"group_id": 1, "narration": "a"}, {"group_id": 2, "narration": "b"}]
    classes = {1: "DRAMATIC", 2: "CONNECTIVE"}
    p = npu.build_prompt(lines, [], "cinematic", genre="murim", classes=classes)
    assert "DRAMATIC" in p and "CONNECTIVE" in p
    assert "cinematic" in p.lower()
    assert '"style"' in p                      # per-line class threaded
    # full mode is unchanged — no per-line style tags
    assert '"style"' not in npu.build_prompt(lines, [], "full", genre="murim")


def test_validate_allows_longer_dramatic_line():
    orig = "He runs through the dark mountain forest."     # 7 words
    cine = ("Under a pale blood moon he tears through the fog drowned forest, "
            "every breath ragged, every step a desperate gamble against the "
            "closing dark.")                                # ~24 words
    assert npu.validate_line(orig, cine, [], max_ratio=2.6) is True
    assert npu.validate_line(orig, cine, []) is False       # default 1.5 rejects


def test_merge_uses_dramatic_budget_for_dramatic_beats():
    long_cine = ("Under a pale blood moon Prince Cheon tears through the fog "
                 "drowned forest, robes shredded, every breath a ragged prayer, "
                 "every step a desperate gamble against the closing dark.")
    beats = {"beats": [{"group_id": 1, "narration":
                        "Prince Cheon runs through the dark mountain forest."}]}
    punched = [{"group_id": 1, "narration": long_cine}]
    # as CONNECTIVE the long line is rejected; as DRAMATIC it's kept
    assert npu.merge(beats, punched, ["Prince Cheon"],
                     classes={1: "CONNECTIVE"})["stats"]["punchup_applied"] == 0
    assert npu.merge(beats, punched, ["Prince Cheon"],
                     classes={1: "DRAMATIC"})["stats"]["punchup_applied"] == 1


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


def test_gate_rejects_invented_server_on_real_world():
    # "the nightmare's server just went live" — there is no server, it's real life
    orig = "Then, a sudden crash, and the beast tears through the world."
    bad = "Then, a sudden crash, and the nightmare's server just went live."
    assert npu.invented_term(orig, bad) == "server"
    assert npu.validate_line(orig, bad, []) is False


def test_gate_rejects_invented_collective_count():
    # two beasts must never become "a wolf pack" / "a swarm"
    orig = "A man in a white coat faces two snarling beasts."
    assert npu.validate_line(orig, "A man faces a snarling wolf pack.", []) is False
    assert npu.invented_term(orig, "the swarm descends on him") == "swarm"
    assert npu.invented_term(orig, "two beasts close in") is None      # count kept


def test_gate_allows_persona_metaphor_but_not_literal_invention():
    orig = "He sprints down the mountain as the killers close in."
    # metaphorical gamer flavour is style, not a fabricated fact -> allowed
    assert npu.invented_term(
        orig, "Our guy is speedrunning the descent, a boss fight on his heels") is None
    # a literal game word is legit in a SYSTEM-genre world, invented in a modern one
    assert npu.invented_term("A status window blinks.",
                             "The respawn timer ticks down.", genre="system") is None
    assert npu.invented_term("A status window blinks.",
                             "The respawn timer ticks down.", genre="modern") == "respawn"


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


def test_config_tts_python_is_host_agnostic(tmp_path, monkeypatch):
    from studio.config import load, REPO_ROOT
    toml = tmp_path / "studio.toml"
    # a RELATIVE path resolves against the repo root (works on any host)
    toml.write_text("[tts]\npython = \".qwen_venv/bin/python\"\n")
    assert load(toml).tts_python == str(REPO_ROOT / ".qwen_venv/bin/python")
    # env override wins (per-host)
    monkeypatch.setenv("STUDIO_TTS_PYTHON", "/custom/py")
    assert load(toml).tts_python == "/custom/py"


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


def test_config_env_override_for_punchup(tmp_path, monkeypatch):
    from studio.config import load
    toml = tmp_path / "studio.toml"
    toml.write_text("[models]\npunchup = \"full\"\n")
    monkeypatch.setenv("STUDIO_PUNCHUP", "light")
    assert load(toml).punchup == "light"
    monkeypatch.delenv("STUDIO_PUNCHUP")
    assert load(toml).punchup == "full"
