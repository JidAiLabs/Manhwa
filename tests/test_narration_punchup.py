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


def test_classify_visual_gag_overrides_intense_as_comic():
    beats = {"beats": [{
        "group_id": 1,
        "mood_words": ["manic", "mocking", "humiliated", "intense"],
        "narration": ("He bursts into uncontrollable laughter, pointing at "
                      "Heo Bong and shouting, 'Where did all your hair "
                      "disappear to overnight?!'"),
        "scene_selection": [{"intensity": "intense"}],
    }]}
    assert npu.classify_beats(beats) == {1: "COMIC"}


def test_classify_panel_lines_does_not_make_whole_group_dramatic():
    beats = {"beats": [{
        "group_id": 1,
        "panel_narration": [
            {"scene_file": "a.jpg", "line": "He lands the decisive strike."},
            {"scene_file": "b.jpg", "line": "The others process what happened."},
        ],
        "scene_selection": [
            {"scene_file": "a.jpg", "intensity": "explosive"},
            {"scene_file": "b.jpg", "intensity": "calm"},
        ],
    }]}
    assert npu.classify_panel_lines(beats) == {
        (1, 0): "DRAMATIC",
        (1, 1): "CONNECTIVE",
    }


def test_cinematic_prompt_tags_lines_with_class_and_both_rules():
    lines = [{"group_id": 1, "narration": "a"},
             {"group_id": 2, "narration": "b"},
             {"group_id": 3, "narration": "c"}]
    classes = {1: "DRAMATIC", 2: "CONNECTIVE", 3: "COMIC"}
    p = npu.build_prompt(lines, [], "cinematic", genre="murim", classes=classes)
    assert "DRAMATIC" in p and "CONNECTIVE" in p and "COMIC" in p
    assert "cinematic" in p.lower()
    assert "visual gag" in p and "recap-channel punch" in p
    assert '"style"' in p                      # per-line class threaded
    # full mode is unchanged — no per-line style tags
    assert '"style"' not in npu.build_prompt(lines, [], "full", genre="murim")


def test_validate_allows_longer_dramatic_line():
    orig = "He runs through the dark mountain forest."     # 7 words
    cine = ("Under a pale blood moon he tears through the fog drowned forest, "
            "every breath ragged, every step a desperate gamble against the "
            "closing dark.")                                # ~24 words
    assert npu.validate_line(orig, cine, [], max_ratio=2.6) is True
    assert npu.validate_line(orig, cine, []) is True        # pace is panel-led


def test_merge_does_not_treat_length_as_a_style_gate():
    long_cine = ("Under a pale blood moon Prince Cheon tears through the fog "
                 "drowned forest, robes shredded, every breath a ragged prayer, "
                 "every step a desperate gamble against the closing dark.")
    beats = {"beats": [{"group_id": 1, "narration":
                        "Prince Cheon runs through the dark mountain forest."}]}
    punched = [{"group_id": 1, "narration": long_cine}]
    # Length is decided by the panel's job, not by class-specific word budgets.
    assert npu.merge(beats, punched, ["Prince Cheon"],
                     classes={1: "CONNECTIVE"})["stats"]["punchup_applied"] == 1
    assert npu.merge(beats, punched, ["Prince Cheon"],
                     classes={1: "DRAMATIC"})["stats"]["punchup_applied"] == 1


def test_merge_keeps_punchline_length_when_grounded():
    original = ("Heo Bong stands there bald and humiliated while the man laughs "
                "at his missing hair.")
    punched = ("Heo Bong stands there bald and humiliated while the man loses "
               "it laughing at his missing hair, because apparently the real "
               "casualty tonight was his entire dignity taking a public sect "
               "beating.")
    beats = {"beats": [{"group_id": 1, "narration": original}]}
    cand = [{"group_id": 1, "narration": punched}]
    assert npu.merge(beats, cand, [],
                     classes={1: "CONNECTIVE"})["stats"]["punchup_applied"] == 1
    assert npu.merge(beats, cand, [],
                     classes={1: "COMIC"})["stats"]["punchup_applied"] == 1


def test_validate_accepts_styled_same_facts():
    punched = ("Our guy Prince Cheon is speedrunning a mountain escape — "
               "robes shredded, lungs on fire, and the Assassins are "
               "closing the gap like it's a ranked match.")
    assert npu.validate_line(ORIG, punched, ["Prince Cheon"]) is True


def test_validate_allows_aggressive_per_panel_compression():
    original = ("Prince Cheon stares at the attackers in complete shock as "
                "they close in around him from every side.")
    compressed = "The assassins close in. Prince Cheon is trapped."
    assert npu.validate_line(original, compressed, ["Prince Cheon"]) is True


def test_validate_rejects_dropped_cast_name():
    assert npu.validate_line(ORIG, "Some guy runs from some people, "
                             "robes torn, breath ragged, vibes bad.",
                             ["Prince Cheon"]) is False


def test_validate_rejects_blowup_and_chrome():
    assert npu.validate_line(ORIG, "He runs. " * 30, ["Prince Cheon"]) is False
    assert npu.validate_line(
        ORIG, "Prince Cheon flees — go read chapter 2 on elftoon.com!",
        ["Prince Cheon"]) is False


def test_infer_genre_from_content_reads_the_manhwa_type():
    sys_b = {"beats": [{"narration_plain": "A status window blinks LEVEL UP; "
                        "he opens his skill tree and accepts the quest."}]}
    assert npu.infer_genre_from_content(sys_b) == "system"
    murim_b = {"beats": [{"narration_plain": "The sect elder channels his qi as "
                          "the murim clans gather; cultivation peaks tonight."}]}
    assert npu.infer_genre_from_content(murim_b) == "murim"
    modern_b = {"beats": [{"narration_plain": "He scrolls his phone on the subway "
                           "as monsters tear through the city."}]}
    assert npu.infer_genre_from_content(modern_b) == "modern"


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


def test_prompt_includes_recap_rules_and_story_spine():
    base = npu.build_prompt(
        [{"group_id": 1, "panel_index": 0, "narration": ORIG}],
        ["Prince Cheon"], "cinematic", genre="murim")
    assert "NO SCREEN READING" in base
    assert "REVEAL PACING" in base
    with_story = npu.build_prompt(
        [{"group_id": 1, "narration": ORIG}], ["Prince Cheon"], "full",
        story_context="A prince receives a nano machine.")
    assert "WHOLE-CHAPTER STORY SPINE" in with_story
    assert "A prince receives a nano machine." in with_story


def test_quoted_source_line_is_marked_for_indirect_paraphrase():
    prompt = npu.build_prompt(
        [{"group_id": 1, "panel_index": 0,
          "narration": "'Who are you!' the assassin shouts."}],
        [], "cinematic")
    assert '"must_paraphrase_dialogue": true' in prompt
    assert "use NO quotation marks" in prompt
    assert not npu.validate_line(
        "'Who are you!' the assassin shouts.",
        "'Who are you?' the assassin demands.",
        [], forbid_quotes=True)
    assert npu.validate_line(
        "'Who are you!' the assassin shouts.",
        "The assassin demands to know who the stranger is.",
        [], forbid_quotes=True)
    assert not npu.validate_line(
        "The strike lands.",
        "A heavy impact tears through the clearing,",
        [], forbid_fragments=True)


def test_punchup_rejects_newly_invented_quoted_thought():
    beats = {"beats": [{"group_id": 1, "narration":
                        "The assassin realizes the stranger is behind him."}]}
    punched = [{"group_id": 1, "narration":
                "The assassin's face says, 'Oh, he is behind me now.'"}]
    out = npu.merge(beats, punched, [])
    assert out["beats"][0]["narration"] == (
        "The assassin realizes the stranger is behind him.")
    assert out["stats"]["punchup_applied"] == 0


def test_per_panel_prompt_is_pace_contract_not_word_budget():
    lines = [{"group_id": 1, "panel_index": i,
              "narration": "A long descriptive panel line."}
             for i in range(10)]
    prompt = npu.build_prompt(lines, [], "cinematic")
    assert "PACING CONTRACT" in prompt
    assert "chapter average" in prompt
    assert "TOTAL WORD BUDGET" not in prompt
    assert "words/panel" not in prompt


def test_panel_lines_are_batched_without_reordering():
    lines = [{"panel_index": i} for i in range(7)]
    batches = npu._batch_lines(lines, 3)
    assert [[x["panel_index"] for x in b] for b in batches] == [
        [0, 1, 2], [3, 4, 5], [6]]


def test_panel_lines_auto_batch_by_payload_not_magic_panel_count():
    lines = [{"panel_index": i, "narration": "x" * 5000}
             for i in range(5)]
    batches = npu._batch_lines(lines, 0, max_payload_chars=260)
    assert [x["panel_index"] for b in batches for x in b] == list(range(5))
    assert len(batches) > 1
    assert all(len(b) != 24 for b in batches)


def test_story_context_reads_logline_and_premise(tmp_path):
    path = tmp_path / "manifest.story.json"
    path.write_text('{"logline":"A hunted prince survives.",'
                    '"premise":"Future technology changes his fate."}')
    context = npu._story_context(str(path))
    assert "A hunted prince survives." in context
    assert "Future technology changes his fate." in context


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


# ---- per-panel punchup (Chunk 4) ------------------------------------------

def test_punchup_payload_is_per_panel_line():
    beats = {"beats": [{"group_id": 1, "panel_narration": [
        {"scene_file": "a.jpg", "line": "He stands."},
        {"scene_file": "b.jpg", "line": "The system speaks."}]}]}
    payload = npu.build_panel_payload(beats)
    assert payload == [
        {"group_id": 1, "panel_index": 0, "narration": "He stands."},
        {"group_id": 1, "panel_index": 1, "narration": "The system speaks."}]


def test_apply_punchup_preserves_alignment_and_plain():
    beat = {"group_id": 1, "panel_narration": [
        {"scene_file": "a.jpg", "line": "He stands."},
        {"scene_file": "b.jpg", "line": "The system speaks."}]}
    rewrites = {(1, 0): "He rises, blade ready.", (1, 1): "The System hums to life."}
    accepted = npu.apply_panel_punchup(beat, rewrites)
    pn = beat["panel_narration"]
    assert [p["line"] for p in pn] == ["He rises, blade ready.", "The System hums to life."]
    assert pn[0]["line_plain"] == "He stands."
    assert beat["narration"] == "He rises, blade ready. The System hums to life."
    assert accepted == 2   # both panels accepted


def test_post_punchup_backstop_reneutralizes_concealed_figure():
    """The beats pass neutralizes a concealed figure's identity, but the persona
    punchup runs AFTER it and can re-attach the protagonist handle ('our guy') to
    that same still-unresolved figure. The post-punchup backstop must neutralize
    it back to a neutral handle on a panel whose understanding carries a
    concealment/power cue and does NOT match the established protagonist."""
    cast = {"cast": [{"canonical_name": "Kim Dokja", "is_protagonist": True,
                      "visual_description": "ordinary office worker glasses"}]}
    # post-punch beats: panel 0 OPENS the unresolved window (concealment cue);
    # panel 1 was rewritten by persona to 'Our guy ...' on the SAME masked figure.
    out = {"beats": [{"group_id": 1, "panel_narration": [
        {"scene_file": "p0.jpg",
         "line": "A hooded figure appears in the smoke.",
         "line_plain": "A hooded figure appears in the smoke."},
        {"scene_file": "p1.jpg",
         "line": "Our guy raises a hand wreathed in lightning.",
         "line_plain": "The stranger raises a hand wreathed in lightning."}]}]}
    understood = {
        "p0.jpg": {"subjects": ["hooded stranger"],
                   "description": "a masked, hooded figure emerges from smoke"},
        "p1.jpg": {"subjects": ["glowing figure"],
                   "description": "the masked figure crackling with lightning"}}
    vision = {"p0.jpg": {"ocr_clean": ""}, "p1.jpg": {"ocr_clean": ""}}

    counts = npu.apply_post_punchup_backstop(out, cast, vision, understood)

    line = out["beats"][0]["panel_narration"][1]["line"].lower()
    assert "our guy" not in line            # protagonist handle removed
    assert "stranger" in line               # neutral handle restored
    assert counts["identity_reveals_neutralized"] >= 1
    # the joined narration is rebuilt from the neutralized panel line
    assert "our guy" not in out["beats"][0]["narration"].lower()


def test_apply_punchup_rejects_chrome_and_keeps_plain():
    """The grounding gate must fire per line: chrome-leaking or overlong
    rewrites are rejected and the grounded plain line is restored."""
    beat = {"group_id": 2, "panel_narration": [
        {"scene_file": "c.jpg", "line": "Prince Cheon flees through the fog."},
        {"scene_file": "d.jpg", "line": "He survives."}]}
    rewrites = {
        # chrome injection — must be rejected
        (2, 0): "Prince Cheon flees — go read chapter 2 on elftoon.com!",
        # wildly overlong (30× the original) — must be rejected
        (2, 1): "He survives. " * 30,
    }
    accepted = npu.apply_panel_punchup(beat, rewrites, cast_names=["Prince Cheon"])
    pn = beat["panel_narration"]
    # both rewrites rejected; lines stay as grounded originals
    assert pn[0]["line"] == "Prince Cheon flees through the fog."
    assert pn[1]["line"] == "He survives."
    assert pn[0]["line_plain"] == "Prince Cheon flees through the fog."
    assert pn[1]["line_plain"] == "He survives."
    # joined narration reflects the (unchanged) lines
    assert beat["narration"] == "Prince Cheon flees through the fog. He survives."
    assert accepted == 0   # both rewrites rejected


# ---- adaptive flow segments (Chunk 2): punchup operates on segments ---------

FLOW_ORIG = ("He drops through the canopy, bounces off two branches, and "
             "lands in the one spot the assassins forgot to watch.")


def test_punchup_payload_flattens_segments():
    """build_panel_payload enumerates SEGMENTS (a 1-4 panel flow span is one
    entry), keyed by segment index — same contract the rewrite lookup uses."""
    beats = {"beats": [{"group_id": 1, "segments": [
        {"span": ["a.jpg", "b.jpg"],
         "line": "He runs the rooftops and drops into the alley."},
        {"span": ["c.jpg"], "line": "The system pings."}]}]}
    payload = npu.build_panel_payload(beats)
    assert payload == [
        {"group_id": 1, "panel_index": 0,
         "narration": "He runs the rooftops and drops into the alley."},
        {"group_id": 1, "panel_index": 1, "narration": "The system pings."}]


def test_classify_panel_lines_span_uses_max_intensity():
    """A flow span takes the MAX intensity across its panels (peaks preserved):
    one intense panel inside the span makes the whole segment DRAMATIC."""
    beats = {"beats": [{
        "group_id": 1,
        "scene_selection": [
            {"scene_file": "a.jpg", "intensity": "calm"},
            {"scene_file": "b.jpg", "intensity": "intense"},
            {"scene_file": "c.jpg", "intensity": "calm"},
        ],
        "segments": [
            {"span": ["a.jpg", "b.jpg"],
             "line": "He crosses the ridge as the horde closes in."},
            {"span": ["c.jpg"], "line": "The dust settles over the pass."},
        ]}]}
    assert npu.classify_panel_lines(beats) == {
        (1, 0): "DRAMATIC", (1, 1): "CONNECTIVE"}


def test_apply_punchup_segments_accept_and_plain():
    """An accepted rewrite lands in segments[].line; the grounded original is
    stamped as line_plain; spans stay untouched; the narration join rebuilds."""
    beat = {"group_id": 1, "segments": [
        {"span": ["a.jpg", "b.jpg", "c.jpg"], "line": FLOW_ORIG},
        {"span": ["d.jpg"], "line": "The system window blinks awake."}]}
    rewrites = {
        (1, 0): ("Our guy faceplants through the canopy, bounces off two "
                 "branches, and sticks the landing in the assassins' only "
                 "blind spot."),
        (1, 1): "The system window blinks awake, unimpressed."}
    accepted = npu.apply_panel_punchup(beat, rewrites)
    segs = beat["segments"]
    assert segs[0]["line"].startswith("Our guy faceplants")
    assert segs[0]["line_plain"] == FLOW_ORIG
    assert segs[0]["span"] == ["a.jpg", "b.jpg", "c.jpg"]   # spans never change
    assert beat["narration"] == segs[0]["line"] + " " + segs[1]["line"]
    assert beat["narration_plain"].startswith("He drops through")
    assert accepted == 2


def test_apply_punchup_rejects_span_budget_violation():
    """spec 3.1: the span word budget survives punchup. A 5-word rewrite for a
    3-panel span (~2.2s of voice for 3 panels needing >=6s) is REJECTED — the
    grounded original survives, exactly like the caption-preservation
    fallback."""
    beat = {"group_id": 3, "segments": [
        {"span": ["a.jpg", "b.jpg", "c.jpg"], "line": FLOW_ORIG}]}
    accepted = npu.apply_panel_punchup(beat, {(3, 0): "He drops and lands."})
    assert accepted == 0
    assert beat["segments"][0]["line"] == FLOW_ORIG
    assert beat["narration"] == FLOW_ORIG


def test_apply_punchup_rejects_over_fat_span_rewrite():
    """The budget gate rejects TOO-FAT rewrites as well: one solo panel must
    never carry ~27s of voice."""
    beat = {"group_id": 3, "segments": [
        {"span": ["a.jpg"], "line": "The system window blinks awake."}]}
    fat = " ".join(["The window keeps talking and talking"] * 10) + "."
    accepted = npu.apply_panel_punchup(beat, {(3, 0): fat})
    assert accepted == 0
    assert beat["segments"][0]["line"] == "The system window blinks awake."


def test_apply_punchup_legacy_lines_keep_aggressive_compression():
    """Legacy panel_narration manifests keep today's behavior: NO span budget
    gate (pace belongs to the panel), so an aggressive 3-word compression of a
    singleton line is still accepted."""
    beat = {"group_id": 4, "panel_narration": [
        {"scene_file": "a.jpg",
         "line": "He rises to his feet slowly, breath ragged from the fall."}]}
    accepted = npu.apply_panel_punchup(beat, {(4, 0): "He gets up."})
    assert accepted == 1
    assert beat["panel_narration"][0]["line"] == "He gets up."
