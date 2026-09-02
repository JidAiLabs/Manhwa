"""publish_concept: coherent title/hook/style/description/pinned assembly."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "publish_concept",
    Path(__file__).resolve().parent.parent / "tools" / "publish_concept.py")
pc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pc)  # type: ignore[union-attr]


def test_pinned_comment_is_only_place_with_real_name():
    p = pc.pinned_comment("Infinite Evolution From Zero", "https://x.com/book/1")
    assert "Infinite Evolution From Zero" in p and "official" in p
    assert pc.pinned_comment("") .startswith("Manhwa:")


def test_pick_hook_matches_style():
    assert pc.pick_hook(["GENIUS", "LEVEL 9999", "HE WINS"], "stat_callout") == "LEVEL 9999"
    assert pc.pick_hook(["WEAK|GOD", "GENIUS"], "before_after") == "WEAK|GOD"
    assert pc.pick_hook(["GENIUS", "SSS"], "power_reveal") == "GENIUS"
    assert pc.pick_hook([], "power_reveal") == ""


def test_description_has_synopsis_tags_boilerplate_but_no_real_name():
    d = pc.build_description("A nobody awakens a hidden class! 🔥",
                            ["#manhwa", "necromancer"])
    assert "hidden class" in d and "#manhwa" in d and "#necromancer" in d
    assert "Patreon" in d and "Tags:" in d


def test_assemble_concept_is_coherent_and_copyright_safe():
    # the beats must actually SAY the rank the hook claims: pick_hook now
    # rejects a stat hook whose number/rank is absent from the story (the fixture
    # previously said "rank S" while the hook claimed "SSS" -- the same inflation
    # that put "LEVEL 999" on ORV's thumbnail when the real maximum is 11).
    beats = {"beats": [{"group_id": 1, "what_happens": "he checks his status "
                        "window; level and rank SSS skill appear"}]}
    llm = {"title": "When a Nobody Awakens the Rarest Class!",
           "hooks": ["GENIUS", "RANK SSS", "HE WINS"],
           "synopsis": "A mocked boy awakens a hidden class. 🔥",
           "hashtags": ["#manhwa", "#system"]}
    c = pc.assemble_concept(beats, llm, series_title="Solo Necromancer",
                            official_link="http://x")
    assert c["style"] == "stat_callout"            # from the UI/level signal
    assert c["hook"] == "RANK SSS"                  # stat hook for stat style
    assert "Solo Necromancer" not in c["title"]
    assert "Solo Necromancer" not in c["description"]
    assert "Solo Necromancer" in c["pinned_comment"]   # only here
    assert c["style_overlay"]["label_pos"]            # overlay wired


# ---- bundle (per-video) level --------------------------------------------

def test_parts_timestamps_start_at_zero_and_accumulate():
    p = pc.parts_timestamps([3925.0, 3923.0, 3800.0])
    assert p[0].startswith("0:00 ")                 # YouTube rule: first = 0:00
    assert p[1].startswith("1:05:25 ")              # 3925s -> 1:05:25
    assert "Part 3" in p[2]


def test_select_bundle_climax_picks_highest_intensity_chapter():
    ch1 = {"beats": [{"group_id": 1, "scene_selection": [{"intensity": "calm",
            "scene_file": "a.jpg"}]}]}
    ch2 = {"beats": [{"group_id": 1, "scene_selection": [{"intensity": "explosive",
            "scene_file": "boom.jpg"}]}]}
    ci, refs = pc.select_bundle_climax([ch1, ch2])
    assert ci == 1 and refs == ["boom.jpg"]         # climax is in chapter 2


def test_bundle_digest_spans_chapters():
    chs = [{"beats": [{"group_id": 1, "hook": f"hook {i}"}]} for i in range(3)]
    d = pc.bundle_digest(chs)
    # labels carry the arc POSITION ("1 of 3"), so a sampled digest still
    # tells the model where each excerpt sits in the series
    assert "[Chapter 1 of 3]" in d and "[Chapter 3 of 3]" in d
    assert "hook 0" in d and "hook 2" in d


def test_build_bundle_concept_arc_title_climax_refs_and_parts():
    weak = {"beats": [{"group_id": 1, "what_happens": "a mocked weakling",
            "scene_selection": [{"intensity": "calm", "scene_file": "w.jpg"}]}]}
    payoff = {"beats": [{"group_id": 1, "what_happens": "he breaks every record",
              "scene_selection": [{"intensity": "explosive", "scene_file": "win.jpg"}]}]}
    llm = {"title": "From Mocked Weakling to Record Breaker!",
           "hooks": ["GENIUS"], "synopsis": "Setup to payoff. 🔥",
           "hashtags": ["#manhwa"]}
    c = pc.build_bundle_concept([weak, payoff], llm, durations=[3600.0, 3600.0],
                                series_title="Hidden Series")
    assert c["climax_chapter_index"] == 1 and c["refs"] == ["win.jpg"]
    assert c["parts"][0].startswith("0:00") and "1:00:00" in c["parts"][1]
    assert "0:00" in c["description"]               # parts appended to desc
    assert "Hidden Series" not in c["description"]  # still copyright-safe


# --- digest bounding: the prompt must not grow with the series -------------

def _beats_n(n):
    return [{"beats": [{"hook": f"chapter {i} hook " + "x" * 800}]}
            for i in range(n)]


def test_digest_is_bounded_regardless_of_series_length():
    """It used to describe EVERY chapter: ~713 chars each, so 300 chapters
    produced ~213,000 chars (~53,000 tokens). MLX ignores num_ctx and simply
    processes that — ~3 minutes of prefill to write a title."""
    small = pc.bundle_digest(_beats_n(10), max_chapters=24)
    huge = pc.bundle_digest(_beats_n(300), max_chapters=24)
    assert huge.count("[Chapter ") == 24
    assert small.count("[Chapter ") == 10          # under the cap: untouched
    assert len(huge) < 30_000


def test_digest_keeps_opening_climax_and_ending():
    """Sampled, not truncated — the arc shape is what a title needs."""
    d = pc.bundle_digest(_beats_n(300), max_chapters=24, climax_index=176)
    assert "[Chapter 1 of 300]" in d
    assert "[Chapter 300 of 300]" in d
    assert "[Chapter 177 of 300 (CLIMAX)]" in d


def test_sampling_is_spread_not_front_loaded():
    idxs = pc.sample_arc_indices(300, max_chapters=24)
    assert idxs[0] == 0 and idxs[-1] == 299
    assert len(idxs) <= 24
    # the middle of the arc must be represented, not just the first chapters
    assert any(100 <= i <= 200 for i in idxs)
    assert idxs == sorted(set(idxs))


def test_climax_scan_stays_exhaustive():
    """Bounding the DIGEST must not change which moment the thumbnail shows:
    the climax is found in pure Python over every chapter."""
    beats = [{"beats": [{"scene_selection": [
        {"scene_file": f"p{i}.jpg", "intensity": "calm"}]}]} for i in range(300)]
    beats[287] = {"beats": [{"scene_selection": [
        {"scene_file": "boom.jpg", "intensity": "explosive"}]}]}
    ci, refs = pc.select_bundle_climax(beats)
    assert ci == 287 and refs == ["boom.jpg"]


# --- before_after needs an actual "before" --------------------------------

def _arc_beats():
    """A 5-chapter arc: weak opening, transformation climax in chapter 4."""
    weak = {"beats": [{"what_happens": "a mocked weakling is humiliated",
                       "scene_selection": [
                           {"scene_file": "w1.jpg", "intensity": "calm"},
                           {"scene_file": "w2.jpg", "intensity": "tense"}]}]}
    filler = {"beats": [{"what_happens": "training",
                         "scene_selection": [
                             {"scene_file": "f.jpg", "intensity": "tense"}]}]}
    payoff = {"beats": [{"what_happens": "he transforms, from zero, and "
                                        "breaks every record",
                         "scene_selection": [
                             {"scene_file": "boom.jpg",
                              "intensity": "explosive"}]}]}
    return [weak, filler, filler, payoff, filler]


def test_before_after_refs_include_a_weak_panel_from_before_the_climax():
    """The composition promises the same character weak AND transformed. Refs
    used to come only from the climax beat, so both halves were painted from
    the SAME moment — there was no 'before' at all."""
    beats = _arc_beats()
    eps = [f"/x/ongoing/s/Chapter_{i + 1}" for i in range(len(beats))]
    c = pc.build_bundle_concept(
        beats, {"title": "t", "hooks": ["WEAK|GOD"], "synopsis": "s",
                "hashtags": ["#m"]},
        durations=[10.0] * len(beats), series_title="S", ep_dirs=eps)
    assert c["style"] == "before_after"
    assert c["climax_chapter_index"] == 3
    # the climax panel is still there...
    assert "boom.jpg" in c["refs"]
    # ...and now a weak panel from an EARLIER chapter leads
    assert c["refs"][0] == "/x/ongoing/s/Chapter_1/scenes/w1.jpg"


def test_before_ref_is_absolute_because_it_crosses_chapters():
    """Refs resolve against the CLIMAX chapter's scenes/ dir, so a panel from
    another chapter can only be expressed as an absolute path."""
    beats = _arc_beats()
    eps = [f"/x/ongoing/s/Chapter_{i + 1}" for i in range(len(beats))]
    before = pc.select_before_ref(beats, eps, climax_ci=3)
    import os as _os
    assert _os.path.isabs(before) and before.endswith("Chapter_1/scenes/w1.jpg")


def test_non_before_after_styles_keep_climax_only_refs():
    """Only the split composition needs a contrasting panel; don't spend a
    reference slot elsewhere."""
    beats = [{"beats": [{"what_happens": "a giant dragon boss attacks",
                         "scene_selection": [
                             {"scene_file": "d.jpg", "intensity": "explosive"}]}]}]
    c = pc.build_bundle_concept(
        beats, {"title": "t", "hooks": ["RUN"], "synopsis": "s",
                "hashtags": ["#m"]},
        durations=[10.0], series_title="S", ep_dirs=["/x/Chapter_1"])
    assert c["style"] == "vs_monster"
    assert c["refs"] == ["d.jpg"]


def test_before_ref_absent_when_climax_is_the_first_chapter():
    beats = _arc_beats()[3:4]
    c = pc.build_bundle_concept(
        beats, {"title": "t", "hooks": ["A|B"], "synopsis": "s",
                "hashtags": ["#m"]},
        durations=[10.0], series_title="S", ep_dirs=["/x/Chapter_1"])
    assert all(not r.startswith("/") for r in c["refs"])


def test_hook_prompt_asks_for_the_shape_each_style_needs():
    """pick_hook has style branches (a piped pair for before_after, a number
    for stat_callout) but the prompt only said 'punchy', so those branches
    were unreachable and before_after always fell back to a literal
    BEFORE / AFTER."""
    p_ba = pc.build_concept_prompt("d", "Banned", "before_after")
    assert "|" in p_ba and "PAIR" in p_ba
    p_stat = pc.build_concept_prompt("d", "Banned", "stat_callout")
    assert "NUMBER" in p_stat or "RANK" in p_stat
    p_def = pc.build_concept_prompt("d", "Banned", "power_reveal")
    # the default spec used to just say "punchy", which produced atmospheric
    # captions ("Story Bleeds In"). Every thumbnail label that works is a
    # NAMETAG -- a role/title/rank/status you could point an arrow at.
    assert "NAMETAG" in p_def and "role" in p_def
    assert "arrow" in p_def
    for p in (p_ba, p_stat, p_def):
        assert "Banned" in p          # ban list survives in every variant


# --- thumbnail + teaser agree on the arc peak -----------------------------

def test_scored_climax_prefers_transformation_over_earlier_combat(tmp_path):
    """The old beats picker was argmax over a 4-value enum: many beats tie at
    'explosive' and the strict '>' keeps the FIRST. The scored picker ranks by
    the same weighted model the teaser uses, so a late transformation reveal
    beats an earlier, more violent combat frame."""
    import json as _json
    eps = []
    for i, panels in enumerate([
            # chapter 1: violent combat, explosive intensity, NO transform cue
            [{"scene_file": "c.jpg", "panel_kind": "story", "intensity":
              "explosive", "description": "he swings his blade with brutal force",
              "action": "a savage strike"}],
            # chapter 2: the genre-defining transformation reveal
            [{"scene_file": "reveal.jpg", "panel_kind": "story", "intensity":
              "intense", "description": "the nano core activates and his power "
              "awakens", "action": "system window: awakening unlocked"}]]):
        d = tmp_path / f"ch{i}"
        d.mkdir()
        (d / "manifest.panels.understood.json").write_text(
            _json.dumps({"panels": panels}))
        eps.append(str(d))
    ci, refs = pc.select_bundle_climax_scored(eps)
    assert ci == 1                                   # the transformation chapter
    assert refs == ["reveal.jpg"]


def test_scored_climax_none_without_understood_manifests(tmp_path):
    d = tmp_path / "ch0"
    d.mkdir()
    assert pc.select_bundle_climax_scored([str(d)]) is None


def test_bundle_concept_uses_scored_climax_when_available(tmp_path):
    """build_bundle_concept must prefer the understood-panel scorer so the
    thumbnail agrees with the teaser."""
    import json as _json
    d0, d1 = tmp_path / "c0", tmp_path / "c1"
    for d, panels in [
        (d0, [{"scene_file": "a.jpg", "panel_kind": "story",
               "intensity": "explosive", "description": "loud fight"}]),
        (d1, [{"scene_file": "boom.jpg", "panel_kind": "story",
               "intensity": "intense",
               "description": "he awakens a hidden power, transformed"}])]:
        d.mkdir()
        (d / "manifest.panels.understood.json").write_text(
            _json.dumps({"panels": panels}))
    beats = [{"beats": [{"scene_selection": [
                {"scene_file": "a.jpg", "intensity": "explosive"}]}]},
             {"beats": [{"scene_selection": [
                {"scene_file": "boom.jpg", "intensity": "intense"}]}]}]
    c = pc.build_bundle_concept(
        beats, {"title": "t", "hooks": ["X"], "synopsis": "s",
                "hashtags": ["#m"]},
        durations=[10.0, 10.0], series_title="S", ep_dirs=[str(d0), str(d1)])
    # scored picker chose the transformation (chapter 2); beats argmax would
    # have chosen chapter 1 (explosive, first)
    assert c["climax_chapter_index"] == 1


# ---- hook grounding (ORV "LEVEL 999 PROPHET") -----------------------------
# The stat_callout shape spec used to END with 'e.g. "LEVEL 999", "RANK SSS"'.
# The model returned BOTH examples verbatim as hooks, and pick_hook's stat
# branch returned the FIRST hook containing any digit -- actively preferring the
# invented stat over the one hook actually derived from the story. ORV's series
# thumbnail shipped "LEVEL 999 PROPHET"; the highest number anywhere in 54
# chapters of narration is 11.

def _beats(*lines):
    return {"beats": [{"group_id": i, "segments": [{"line": ln}]}
                      for i, ln in enumerate(lines)]}


def test_hook_claims_extracts_numbers_and_ranks():
    assert pc.hook_claims("LEVEL 999 PROPHET") == ["999"]
    assert pc.hook_claims("RANK SSS KNOWLEDGE") == ["SSS"]
    assert pc.hook_claims("THE SCRIPT IS BROKEN") == []


def test_hook_grounding_rejects_an_invented_number():
    corpus = "He reaches Level 3. Later the floor 11 gate opens."
    assert pc.hook_is_grounded("LEVEL 3 PROPHET", corpus)
    assert not pc.hook_is_grounded("LEVEL 999 PROPHET", corpus)
    # word-bounded: 11 must not ground 999, and 3 must not ground 33
    assert not pc.hook_is_grounded("LEVEL 33 HERO", corpus)


def test_hook_grounding_is_permissive_without_a_corpus():
    # no evidence != proof of fabrication; never reject everything silently
    assert pc.hook_is_grounded("LEVEL 999", "")


def test_pick_hook_prefers_a_grounded_label_over_an_invented_stat():
    hooks = ["LEVEL 999 PROPHET", "RANK SSS KNOWLEDGE", "THE SCRIPT IS BROKEN"]
    corpus = "He is called a prophet. The script is broken. Level 3 clears."
    # both stat hooks invent their number -> the grounded plain label wins
    assert pc.pick_hook(hooks, "stat_callout", corpus=corpus) == "THE SCRIPT IS BROKEN"


def test_pick_hook_keeps_a_grounded_stat():
    hooks = ["THE SCRIPT IS BROKEN", "LEVEL 3 PROPHET"]
    corpus = "He claws his way to Level 3 before the scenario ends."
    assert pc.pick_hook(hooks, "stat_callout", corpus=corpus) == "LEVEL 3 PROPHET"


def test_beats_corpus_never_grounds_on_geometry():
    # a normalized bbox (0.9995) contains "999"; serializing the manifest would
    # have "verified" the exact fabrication this guards against.
    beats = _beats("He reaches Level 3.")
    beats["beats"][0]["box_norm"] = [0.876, 0.9995, 0.1, 0.2]
    corpus = pc.beats_text_corpus(beats)
    assert "999" not in corpus
    assert not pc.hook_is_grounded("LEVEL 999", corpus)


def test_stat_callout_prompt_carries_no_copyable_number():
    spec = pc.build_concept_prompt("digest", "", "stat_callout")
    assert "LEVEL 999" not in spec and "RANK SSS" not in spec
    assert "STORY DIGEST" in spec


# ---- forced style (variant generation) ------------------------------------
# The hook SHAPE differs per style -- before_after wants an "A|B" pair -- so a
# forced style must reach BOTH build_concept_prompt and assemble_concept, or the
# model writes hooks for one style while pick_hook selects for another.

def test_forced_style_overrides_the_auto_selection():
    beats = {"beats": [{"group_id": 1, "what_happens": "he checks his status "
                        "window; level and rank SSS skill appear"}]}
    llm = {"title": "T", "hooks": ["WEAK|GOD", "GENIUS"], "synopsis": "S",
           "hashtags": ["#m"]}
    auto = pc.assemble_concept(beats, llm, series_title="X")
    forced = pc.assemble_concept(beats, llm, series_title="X",
                                 style="before_after")
    assert auto["style"] == "stat_callout"        # what the story implies
    assert forced["style"] == "before_after"      # what we asked for
    assert forced["hook"] == "WEAK|GOD"           # picked for the FORCED style
    assert forced["style_overlay"] == pc.style_for("before_after")["overlay"]


def test_empty_style_keeps_the_production_auto_path():
    beats = {"beats": [{"group_id": 1, "what_happens": "a quiet conversation"}]}
    llm = {"title": "T", "hooks": ["GENIUS"], "synopsis": "S", "hashtags": ["#m"]}
    a = pc.assemble_concept(beats, llm, series_title="X")
    b = pc.assemble_concept(beats, llm, series_title="X", style="")
    assert a["style"] == b["style"]


# ---- subject tags come from ENUMERATED data, not a corpus search ----------
# Checking a tag against the narration blob does not work: at ~208k words nearly
# every common English word appears somewhere, so "WEAK -> GOD" and "DEMON KING"
# both passed. Frequency fails the other way -- it ranks 'king' (95) and 'god'
# (48) above 'script' (10), admitting the trope and rejecting this story's most
# central idea. So the vocabulary is enumerated: cast names the extractor found
# on the pages, plus words printed on stamped in-world SYSTEM screens.

VOCAB = {"dokkaebi", "scenario", "main", "system", "constellation", "coins",
         "skill", "sangah", "bihyeong"}


def test_tag_grounding_requires_every_word_in_the_vocabulary():
    assert pc.tag_is_grounded("THE DOKKAEBI", VOCAB)
    assert pc.tag_is_grounded("MAIN SCENARIO", VOCAB)
    assert not pc.tag_is_grounded("DEMON KING", VOCAB)


def test_generic_power_fantasy_tropes_are_rejected():
    """The exact strings the old corpus check let through."""
    for trope in ("WEAK -> GOD", "TRASH -> LEGEND", "DEMON KING", "SSS RANK"):
        assert not pc.tag_is_grounded(trope, VOCAB), trope


def test_tag_with_no_checkable_word_is_rejected():
    assert not pc.tag_is_grounded("THE ONE", VOCAB)


def test_empty_vocabulary_rejects_rather_than_waves_through():
    # an unverifiable tag is not a safe default; the badge carries the layout
    assert not pc.tag_is_grounded("MAIN SCENARIO", set())


def test_pick_tags_orders_story_specific_first_then_caps():
    """vocab ORDERS, it does not reject: an enumerated word list rejected THE
    SCRIPT and THE PROPHET (narration prose, never on a system screen), so the
    model's reading decides WHAT a tag says and the vocabulary only decides
    which of its tags lead. DEMON KING is not rejected here -- it simply sorts
    behind the two story-specific tags and falls outside the 2-tag limit."""
    tags = pc.pick_tags(["DEMON KING", "THE DOKKAEBI", "MAIN SCENARIO"], VOCAB)
    assert [t["text"] for t in tags] == ["THE DOKKAEBI", "MAIN SCENARIO"]
    # with room for it, the ungrounded tag is KEPT (ordered last), not dropped
    three = pc.pick_tags(["DEMON KING", "THE DOKKAEBI"], VOCAB, limit=2)
    assert [t["text"] for t in three] == ["THE DOKKAEBI", "DEMON KING"]
    assert tags[0]["pos"] == "lower_left" and tags[0]["arrow"] is True
    assert tags[1]["pos"] == "mid_left"


def test_pick_tags_still_rejects_an_invented_number():
    assert pc.pick_tags(["SCENARIO 999"], VOCAB, corpus="he clears scenario 1") == []
    got = pc.pick_tags(["SCENARIO 1"], VOCAB, corpus="he clears scenario 1")
    assert [t["text"] for t in got] == ["SCENARIO 1"]


def test_bundle_badge_states_a_fact_about_the_upload():
    chs = [{"beats": [{"what_happens": "a dokkaebi opens the scenario"}]}
           for _ in range(7)]
    llm = {"title": "T", "hooks": ["GENIUS"], "synopsis": "S", "hashtags": ["#m"]}
    c = pc.build_bundle_concept(chs, llm, durations=[60.0] * 7,
                                series_title="X")
    assert c["badge"] == "7 CHAPTERS"      # true of the upload, not the story


def test_story_vocabulary_is_empty_without_manifests(tmp_path):
    assert pc.story_vocabulary([str(tmp_path)]) == set()
    assert pc.story_vocabulary([]) == set()


# ---- a model reply must not be discarded on a trailing syntax slip --------
# qwen3.6:27b wrote a good title and three good hooks, then part-way through the
# LAST array emitted `#litRPG,` instead of `"#litRPG",`. raw_decode failed on the
# whole object, _gemma returned {} SILENTLY, and the run wrote an empty concept
# (hook='' title='') while reporting [ok]. Two defects: the parse, and the silence.

def test_first_json_recovers_a_reply_with_bare_array_tokens():
    import importlib.util as _u
    from pathlib import Path as _P
    s = _u.spec_from_file_location(
        "oc", _P(__file__).resolve().parent.parent / "tools" / "ollama_compat.py")
    oc = _u.module_from_spec(s); s.loader.exec_module(oc)
    bad = ('{"title": "T", "hooks": ["A", "B"], '
           '"hashtags": ["#manhwa", #litRPG, #webtoonrecap]}')
    got = oc.first_json(bad)
    assert got and got["title"] == "T"
    assert got["hooks"] == ["A", "B"]
    assert got["hashtags"] == ["#manhwa", "#litRPG", "#webtoonrecap"]
    # already-valid JSON is untouched, and true garbage still fails
    assert oc.first_json('{"a": [1, 2]}') == {"a": [1, 2]}
    assert oc.first_json("no json here") is None
    # a repaired parse recovers content, it never invents any
    assert oc.first_json('{"x": [1,2,]}') == {"x": [1, 2]}


def test_triptych_hook_needs_three_parts():
    """A two-part hook would leave the third panel unlabelled."""
    hooks = ["READER|PROPHET", "ORDINARY|AWAKENING|PROPHET", "GENIUS"]
    assert pc.pick_hook(hooks, "triptych") == "ORDINARY|AWAKENING|PROPHET"
    # before_after is happy with two
    assert pc.pick_hook(hooks, "before_after") == "READER|PROPHET"


def test_triptych_style_is_registered_with_split3():
    ov = pc.style_for("triptych")["overlay"]
    assert ov.get("split3") is True


def test_style_cli_choices_track_the_registry():
    """The choices list was hand-written and went stale the moment `triptych`
    was added: the style worked, but argparse rejected --style triptych with
    exit 2. Deriving them means a new style module can never desync again."""
    import re as _re
    src = (Path(__file__).resolve().parent.parent
           / "tools" / "publish_concept.py").read_text()
    m = _re.search(r'--style"[^)]*choices=([^,\n]+)', src)
    assert m, "the --style argument moved or changed shape"
    assert "STYLE_MODULES" in m.group(1), (
        "choices must derive from the style registry, not a hand-written list")
    assert "triptych" in pc.STYLE_MODULES
