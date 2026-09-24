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


def test_before_after_hook_needs_a_pair():
    hooks = ["GENIUS", "READER|PROPHET"]
    assert pc.pick_hook(hooks, "before_after") == "READER|PROPHET"


def test_the_only_split_layout_is_before_after():
    """Owner, 2026-09-17: no 3-panel layout, ever. A split is before/after or
    nothing. The triptych was added as "the most common layout in the reference
    thumbnails" -- counted, 0 of 18 refs and 0 of the owner's 9 new examples."""
    assert "triptych" not in pc.STYLE_MODULES
    assert not any(m["overlay"].get("split3") for m in pc.STYLE_MODULES.values())
    assert [n for n, m in pc.STYLE_MODULES.items()
            if m["overlay"].get("split")] == ["before_after"]


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


# ---- two-stage: understand the story, THEN write the copy -----------------
# The digest is 17k chars of moment-to-moment narration stitched across 24
# chapters -- not a story, its debris. Asked to write a title straight from
# that, the model returned fragment-shaped copy ("THE STORY IS REAL",
# "TARGET"). Validating the OUTPUT could never fix a bad INPUT.

def test_brief_prompt_asks_what_the_story_is_not_for_copy():
    p = pc.build_brief_prompt("some narration", "Banned")
    for field in ("premise", "protagonist", "engine", "arc", "distinctive"):
        assert field in p, field
    assert "Do not write any marketing copy yet" in p
    assert "Banned" in p          # ban list survives into stage 1


def test_package_prompt_writes_from_the_brief_not_the_narration():
    brief = {"premise": "P", "protagonist": "H", "engine": "E"}
    p = pc.build_package_prompt(brief, "Banned")
    assert "STORY UNDERSTANDING" in p and '"premise": "P"' in p
    # the MODEL picks the layout -- a keyword heuristic kept choosing
    # stat_callout for a series whose largest number is 11
    assert "thumbnail_style" in p
    # the AUTO path only ever offers single-scene layouts: before_after is
    # always built as its own option and the owner picks between them
    assert "before_after" not in p and "triptych" not in p
    assert "3-panel" not in p
    assert "could ONLY belong to this story" in p


def test_assemble_package_honours_the_models_layout_choice():
    beats = {"beats": [{"segments": [{"line": "he reaches level 3"}]}]}
    pkg = {"title": "T", "description": "D", "thumbnail_style": "vs_monster",
           "style_reason": "a giant foe",
           "labels": ["F-RANK PORTER"], "hashtags": ["#m"]}
    c = pc.assemble_package(beats, {"premise": "P"}, pkg, series_title="X",
                            styles=["power_reveal", "vs_monster"])
    assert c["style"] == "vs_monster"
    assert c["hook"] == "F-RANK PORTER"
    assert c["brief"] == {"premise": "P"}


def test_assemble_package_falls_back_on_an_unknown_style():
    pkg = {"title": "T", "thumbnail_style": "not_a_real_style",
           "labels": ["HERO"], "hashtags": []}
    c = pc.assemble_package({}, {}, pkg, series_title="X")
    assert c["style"] == pc.DEFAULT_STYLE        # never crashes the run


def test_assemble_package_still_rejects_an_invented_number():
    beats = {"beats": [{"segments": [{"line": "he reaches level 3"}]}]}
    pkg = {"title": "T", "thumbnail_style": "power_reveal",
           "labels": ["LEVEL 999", "THE READER"], "hashtags": []}
    c = pc.assemble_package(beats, {}, pkg, series_title="X")
    assert c["hook"] == "THE READER"   # the false claim is skipped


def test_labels_returned_as_a_bare_string_are_not_split_into_characters():
    """A JSON schema in a prompt is a request, not a guarantee. qwen3.6 returned
    labels as the STRING "READER|SURVIVOR|AUTHOR"; iterating it yielded one
    CHARACTER per label and the hook became "R"."""
    pkg = {"title": "T", "thumbnail_style": "before_after",
           "labels": "READER|AUTHOR", "hashtags": []}
    c = pc.assemble_package({}, {}, pkg, series_title="X", styles=["power_reveal"])
    assert c["hooks"] == ["READER -> AUTHOR"]
    assert c["hook"] == "READER -> AUTHOR"


def test_single_label_layouts_are_untouched():
    pkg = {"title": "T", "thumbnail_style": "power_reveal",
           "labels": ["THE READER", "SPARE"], "hashtags": []}
    assert pc.assemble_package({}, {}, pkg, series_title="X")["hook"] == "THE READER"


# ---- reference panels must SHOW the character ------------------------------
# The ORV thumbnail was generated from two refs containing no person: a system
# notification window (highest dramatic score in a system-driven story) and a
# panel with no recorded subjects. The art prompt's "use the EXACT character
# designs from the reference images: same face, hairstyle, clothing" was
# therefore unfollowable, and the model invented a military uniform.

def test_panel_shows_a_character_rejects_ui_and_empty_panels():
    assert not pc.panel_shows_a_character(
        {"subjects": ["a blue system notification window stating something"]})
    assert not pc.panel_shows_a_character({"panel_kind": "system",
                                           "subjects": ["a man in a suit"]})
    assert not pc.panel_shows_a_character({"subjects": []})   # no evidence
    assert not pc.panel_shows_a_character({})


def test_panel_shows_a_character_accepts_people_however_described():
    assert pc.panel_shows_a_character(
        {"subjects": ["a man with black hair in a dark grey suit"]})
    # an exclusion test, not a person-word list: unusual descriptions pass
    assert pc.panel_shows_a_character(
        {"subjects": ["a large glowing red eye on a pale, distorted face"]})
    # mixed panel: a person alongside UI still shows a person
    assert pc.panel_shows_a_character(
        {"subjects": ["a status window", "a woman in a tan coat"]})


def test_protagonist_name_finds_the_lead_by_its_own_id():
    cast = {"cast": [{"id": "junghyeok"}, {"id": "our_protagonist"},
                     {"id": "jihye_lee"}]}
    assert pc.protagonist_name(cast) == "our_protagonist"
    # name field wins when present
    assert pc.protagonist_name(
        {"cast": [{"id": "mc", "name": "the lead"}]}) == "the lead"
    # no cast at all -> empty, and the portrait filter then returns nothing
    assert pc.protagonist_name({}) == ""


def test_protagonist_portrait_files_is_empty_without_manifests(tmp_path):
    # a missing/unreadable chapter must not crash the picker, it just yields
    # no portraits and the caller falls back a tier
    assert pc.protagonist_portrait_files(str(tmp_path)) == set()


def test_climax_refs_send_several_portraits_not_one():
    """One reference is weak constraint: against a genre prompt the model fills
    the rest from its priors and imports costumes and props that appear nowhere
    in this story. Several panels of the same lead pin appearance by evidence."""
    import importlib.util as _u
    from pathlib import Path as _P
    s = _u.spec_from_file_location(
        "tg", _P(__file__).resolve().parent.parent / "tools" / "thumbnail_gen.py")
    tg = _u.module_from_spec(s); s.loader.exec_module(tg)
    p = tg.build_art_prompt("COMPOSITION HERE")
    # the fidelity block must forbid invented props and borrowed designs
    assert "ADD NOTHING the references do not show" in p
    assert "swords" in p or "weapons" in p
    assert "any other manhwa" in p


# ---- the title follows the format that performs (owner examples 2026-09-17) --
# 27 of 27 example titles end in the Manhwa Recap suffix, 0 carry an emoji, and
# 7 of the owner's 9 capitalise every word, FULL CAPS kept on status words. Ours
# shipped "He Read THE ENDING, Now He's TRAPPED in a REAL NIGHTMARE 😱".

def test_title_gets_the_channel_suffix_and_loses_emoji():
    t = pc.normalize_title("He Read THE ENDING, Now He's TRAPPED in a REAL NIGHTMARE 😱")
    assert t == "He Read THE ENDING, Now He's TRAPPED In A REAL NIGHTMARE - Manhwa Recap"


def test_title_suffix_is_never_doubled():
    for raw in ("BETRAYED Porter Awakens OP Power - Manhwa Recap",
                "BETRAYED Porter Awakens OP Power - Manhwa Recaps",
                "BETRAYED Porter Awakens OP Power | Manhwa Recap",
                "BETRAYED Porter Awakens OP Power"):
        assert pc.normalize_title(raw) == \
            "BETRAYED Porter Awakens OP Power - Manhwa Recap", raw


def test_title_case_keeps_numbers_caps_and_contractions():
    assert pc.normalize_title("poor loser multiplies every dollar 200x! he's #1") == \
        "Poor Loser Multiplies Every Dollar 200x! He's #1 - Manhwa Recap"


def test_empty_title_stays_empty():
    assert pc.normalize_title("") == ""
    assert pc.normalize_title("  📖🔥 ") == ""


def test_assemble_package_normalizes_the_title():
    pkg = {"title": "He used the SCRIPT 📖", "thumbnail_style": "power_reveal",
           "labels": ["READER"], "hashtags": []}
    c = pc.assemble_package({}, {}, pkg, series_title="X")
    assert c["title"] == "He Used The SCRIPT - Manhwa Recap"


def test_package_prompt_title_has_no_worked_example_and_no_emoji_ask():
    p = pc.build_package_prompt({"premise": "P"}, "Banned")
    assert "No emoji" in p
    assert "Manhwa Recap" not in p      # the suffix is added in code, not asked for
    assert "e.g." not in p              # a worked example ships verbatim (LEVEL 999)


def test_package_prompt_reuses_the_picked_thumbnail_labels():
    p = pc.build_package_prompt({"premise": "P"}, "Banned",
                                thumb_labels="F-RANK SUMMONER")
    assert "F-RANK SUMMONER" in p
    assert "F-RANK SUMMONER" not in pc.build_package_prompt({"premise": "P"}, "B")


# ---- before_after refs: the BEFORE half comes from before the climax -------
# The two-stage path (production default) never called select_before_ref, so
# series_9's before_after got three refs from its climax chapter: both halves
# painted from one moment -- the bug select_before_ref was written to fix.

def test_refs_for_before_after_prepend_the_before_panel(monkeypatch):
    monkeypatch.setattr(pc, "select_before_ref",
                        lambda beats, eps, climax_ci: "/abs/ch0/scenes/p1.jpg")
    assert pc.refs_for_style("before_after", [{}], ["ch0"], climax_ci=3,
                             climax_refs=["p9.jpg", "p8.jpg"]) == \
        ["/abs/ch0/scenes/p1.jpg", "p9.jpg", "p8.jpg"]


def test_refs_for_a_scene_style_are_the_climax_refs(monkeypatch):
    monkeypatch.setattr(pc, "select_before_ref",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert pc.refs_for_style("power_reveal", [{}], ["ch0"], climax_ci=3,
                             climax_refs=["p9.jpg"]) == ["p9.jpg"]


def test_before_ref_prefers_the_lead_over_any_person(tmp_path, monkeypatch):
    """The before half must show the LEAD. 'Shows a person' alone handed the
    model a side character before (see protagonist_portrait_files)."""
    import json as _j
    ep = tmp_path / "ch0"; ep.mkdir()
    (ep / "manifest.panels.understood.json").write_text(_j.dumps({"panels": [
        {"scene_file": "p1.jpg", "subjects": ["a guard in armour"]},
        {"scene_file": "p2.jpg", "subjects": ["a thin boy in rags"]}]}))
    beats = [{"beats": [{"scene_selection": [
        {"scene_file": "p1.jpg", "intensity": "calm"},
        {"scene_file": "p2.jpg", "intensity": "calm"}]}]}, {}]
    monkeypatch.setattr(pc, "protagonist_portrait_files", lambda d: {"p2.jpg"})
    assert pc.select_before_ref(beats, [str(ep), str(tmp_path / "ch1")],
                                climax_ci=1) == str(ep / "scenes" / "p2.jpg")


# ---- verifier findings 2026-09-17 ------------------------------------------

def test_the_scene_option_can_never_come_back_as_a_split():
    """A prompt is a request. gemma returning before_after (or stat_callout)
    for the scene option must not turn it into a split."""
    for rogue in ("before_after", "stat_callout", "triptych", ""):
        pkg = {"title": "T", "thumbnail_style": rogue,
               "labels": ["DEAD|KING"], "hashtags": []}
        c = pc.assemble_package({}, {}, pkg, series_title="X")
        assert c["style"] in pc.SCENE_STYLES, rogue
        assert not c["style_overlay"].get("split")
        # the pair becomes ONE transformation label, never a literal pipe
        assert c["hook"] == "DEAD -> KING"


def test_the_before_after_option_can_never_come_back_as_a_scene():
    pkg = {"title": "T", "thumbnail_style": "power_reveal",
           "labels": ["WEAK|KING"], "hashtags": []}
    c = pc.assemble_package({}, {}, pkg, series_title="X",
                            styles=["before_after"])
    assert c["style"] == "before_after" and c["hook"] == "BEFORE|AFTER"

def test_title_suffix_in_any_trailing_form_is_not_doubled():
    for raw in ("He Wins Manhwa Recap", "He wins - MANHWA RECAPS!",
                "He wins (Manhwa Recap)", "He wins: Manhwa Recap"):
        assert pc.normalize_title(raw) == "He Wins - Manhwa Recap", raw


def test_before_ref_never_falls_back_to_a_panel_known_to_show_nobody(tmp_path):
    import json as _j
    ep = tmp_path / "ch0"; ep.mkdir()
    (ep / "manifest.panels.understood.json").write_text(_j.dumps({"panels": [
        {"scene_file": "p1.jpg", "panel_kind": "system",
         "subjects": ["a status window"]}]}))
    beats = [{"beats": [{"scene_selection": [
        {"scene_file": "p1.jpg", "intensity": "calm"}]}]}, {}]
    assert pc.select_before_ref(beats, [str(ep), str(tmp_path / "ch1")],
                                climax_ci=1) == ""


def test_before_after_with_the_climax_in_chapter_one_gets_no_before_ref(monkeypatch):
    """select_before_ref would search the climax chapter itself and hand back
    the climax panel twice (verifier probe, 2026-09-17)."""
    monkeypatch.setattr(pc, "select_before_ref",
                        lambda *a, **k: "/abs/ch0/scenes/p1.jpg")
    assert pc.refs_for_style("before_after", [{}], ["ch0"], climax_ci=0,
                             climax_refs=["p1.jpg"]) == ["p1.jpg"]


def test_before_after_run_refuses_to_paint_without_a_before_panel(tmp_path,
                                                                  monkeypatch):
    """The two-stage run exits 2 before any image is paid for."""
    import json as _j
    import sys as _s
    ep = tmp_path / "ch1"; ep.mkdir()
    (ep / "manifest.beats.json").write_text(_j.dumps({"beats": [
        {"scene_selection": [{"scene_file": "p1.jpg", "intensity": "calm"}]}]}))
    replies = iter([{"premise": "P"},
                    {"title": "T", "thumbnail_style": "before_after",
                     "labels": ["WEAK|KING"], "hashtags": []}])
    monkeypatch.setattr(pc, "_gemma", lambda prompt, model: next(replies))
    out = tmp_path / "c.json"
    monkeypatch.setattr(_s, "argv", ["publish_concept.py", "--episode-dirs",
                                     str(ep), "--style", "before_after",
                                     "--out", str(out)])
    assert pc.main() == 2
    assert not out.exists()


# ---- a lead ref must not contradict the registry look (ORV long hair) --------
# 390 of 2925 ORV lead portraits describe LONG hair; the registry says Dokja has
# short dark hair. cast_identity compares hair COLOUR only, so p000001 (a
# long-haired close-up, speech bubble "JUNGHYEOK...?") passed as the lead and
# the before_after "after" half was painted with long hair.

def test_hair_length_class():
    assert pc.hair_length("a young man with short dark hair") == "short"
    assert pc.hair_length("a character with long, dark, messy hair") == "long"
    assert pc.hair_length("a young man with messy black hair") is None
    assert pc.hair_length("a long white coat over dark hair") is None


def test_portraits_drop_panels_whose_hair_length_contradicts_the_lead(tmp_path,
                                                                      monkeypatch):
    import json as _j
    import types as _t
    ep = tmp_path / "ch"; ep.mkdir()
    (ep / "manifest.cast.json").write_text(_j.dumps({"cast": [
        {"id": "our_protagonist", "canonical_name": "our protagonist",
         "visual_description": "A young man with short dark hair."}]}))
    (ep / "manifest.panels.understood.json").write_text(_j.dumps({"panels": [
        {"scene_file": "p1.jpg", "subjects": ["a young man with messy black hair"]},
        {"scene_file": "p2.jpg", "subjects": ["a character with long, dark, messy hair"]}]}))
    fake = _t.ModuleType("cast_identity")
    fake.__dict__.update(__import__("cast_identity").__dict__)
    fake.resolve_figures_by_file = lambda u, c: {
        "p1.jpg": [{"name": "our protagonist"}], "p2.jpg": [{"name": "our protagonist"}]}
    monkeypatch.setitem(__import__("sys").modules, "cast_identity", fake)
    assert pc.protagonist_portrait_files(str(ep)) == {"p1.jpg"}


# ---- ORV regenerate (job 1694), 2026-09-17 ------------------------------------

def test_auto_scene_offers_only_styles_whose_refs_code_can_fill():
    """feat_object asks for a giant weapon/hammer and no code finds an object
    ref: ORV got an invented stone hammer. vs_monster / humiliation are the same
    lead-only gap. Offered again only when their counterpart finders exist."""
    assert pc.SCENE_STYLES == ["power_reveal"]


def test_climax_refs_prefer_panels_that_match_the_registry_look(tmp_path,
                                                                monkeypatch):
    """Ep306 refs are drawn with longer hair than the registry's Dokja; Ep251
    p000038 says "short dark hair wearing a white coat". A lead panel with
    POSITIVE evidence of the registry look beats a more dramatic one without."""
    import json as _j
    eps = []
    for name, subj in (("ch1", "a young man with short dark hair"),
                       ("ch2", "a young man with messy dark hair screaming")):
        d = tmp_path / name; d.mkdir()
        (d / "manifest.panels.understood.json").write_text(_j.dumps({"panels": [
            {"scene_file": "p1.jpg", "subjects": [subj], "panel_kind": "story",
             "intensity": "explosive" if name == "ch2" else "calm"}]}))
        eps.append(str(d))
    monkeypatch.setattr(pc, "_lead_panels", lambda d, max_figures=2: (
        {"p1.jpg"}, {"p1.jpg"} if d.endswith("ch1") else set()))
    ci, refs = pc.select_bundle_climax_scored(eps)
    assert ci == 0 and refs == ["p1.jpg"]


def test_look_evidence_must_not_belong_to_a_figure_of_the_other_gender():
    """Ep196: "a woman with short dark hair" beside Dokja counted as HIS short
    hair; subjects aren't tied to figures, so gender words must not contradict."""
    assert pc.gender_word("a woman with short dark hair") == "f"
    assert pc.gender_word("A young man with short dark hair") == "m"
    assert pc.gender_word("a person with short dark hair") is None
    assert pc.gender_word("a human-like woman-shaped statue") == "f"
    assert pc.gender_word("a manhole cover") is None


# ---- suggested reference panels, the owner picks (2026-09-17) ----------------
# Owner: "instead of handpick you should propose few options so i can select
# ref image". Automatic refs gave ORV long hair (Ep306) and bystanders (Ep96).

def _cand_arc(tmp_path, monkeypatch, n=6, panels=None):
    """n chapters, each with the given panels [(file, subjects, extra, vision)]."""
    import json as _j
    eps = []
    panels = panels or [("p1.jpg", ["a young man with short dark hair"], {},
                         {"text_coverage": 0.0, "width": 800, "height": 900})]
    for i in range(n):
        d = tmp_path / f"ch{i}"; d.mkdir()
        (d / "manifest.panels.understood.json").write_text(_j.dumps({"panels": [
            dict({"scene_file": f, "subjects": subj, "panel_kind": "story",
                  "dialogue": "", "description": ""}, **extra)
            for f, subj, extra, _v in panels]}))
        (d / "manifest.vision.json").write_text(_j.dumps({"items": [
            dict({"scene_file": f}, **vis) for f, _s, _e, vis in panels]}))
        eps.append(str(d))
    files = {f for f, *_ in panels}
    monkeypatch.setattr(pc, "_lead_panels",
                        lambda d, max_figures=2: (set(files), set(files)))
    return eps


def test_ref_candidates_are_clear_solo_shots_of_the_lead(tmp_path, monkeypatch):
    """Owner: the suggested refs were "wierd", not clear images of the MC.
    They were ranked by DRAMA (crowds, effects, bubbles). A tile is now a solo
    shot: one subject, no dialogue, almost no text, not a tall strip."""
    ok = {"text_coverage": 0.0, "width": 800, "height": 900}
    eps = _cand_arc(tmp_path, monkeypatch, n=1, panels=[
        ("solo.jpg", ["a young man with short dark hair"], {}, ok),
        ("two.jpg", ["a young man", "a woman"], {}, ok),
        ("talk.jpg", ["a young man"], {"dialogue": "RUN!"}, ok),
        ("text.jpg", ["a young man"], {}, dict(ok, text_coverage=0.2)),
        ("strip.jpg", ["a young man"], {}, dict(ok, width=800, height=3000)),
        ("tiny.jpg", ["a young man"], {}, dict(ok, width=300, height=300)),
    ])
    assert [x["file"] for x in pc.ref_candidates(eps)["refs"]] == ["solo.jpg"]


def test_ref_candidates_rank_look_then_close_up_one_per_chapter(tmp_path,
                                                               monkeypatch):
    ok = {"text_coverage": 0.0, "width": 800, "height": 900}
    eps = _cand_arc(tmp_path, monkeypatch, n=3, panels=[
        ("wide.jpg", ["a young man"], {}, ok),
        ("close.jpg", ["a young man"], {"description": "A close-up of his face"}, ok),
    ])
    c = pc.ref_candidates(eps, n=8)["refs"]
    assert [x["chapter"] for x in c] == [0, 1, 2]           # one per chapter
    assert all(x["file"] == "close.jpg" for x in c)
    assert all(Path(x["path"]).is_absolute() for x in c)
    assert set(c[0]) >= {"path", "chapter", "label", "file", "subjects"}


def test_picked_refs_are_used_as_is_for_both_layouts(monkeypatch):
    """Owner: no separate BEFORE refs; the prompt makes before and after."""
    monkeypatch.setattr(pc, "select_before_ref",
                        lambda *a, **k: "/auto/before.jpg")
    picked = ["/p/ep96/p20.jpg", "/p/ep251/p38.jpg"]
    for style in ("power_reveal", "before_after"):
        assert pc.choose_refs(style, [{}], ["e"], climax_ci=3,
                              auto_refs=["a.jpg"], picked=picked) == picked
    assert pc.choose_refs("before_after", [{}], ["e"], climax_ci=3,
                          auto_refs=["a.jpg"]) == ["/auto/before.jpg", "a.jpg"]


def test_before_after_is_always_labelled_before_and_after():
    """Owner: "i told you to use before / after not invent words such as
    Observer / protagonist". The split never carries model-written words."""
    pkg = {"title": "T", "thumbnail_style": "before_after",
           "labels": ["OBSERVER|PROTAGONIST"], "hashtags": []}
    c = pc.assemble_package({}, {}, pkg, series_title="X",
                            styles=["before_after"])
    assert c["hook"] == "BEFORE|AFTER" and c["hooks"] == ["BEFORE|AFTER"]
    assert "labels" not in pc.build_package_prompt({"premise": "P"}, "B",
                                                   ["before_after"])


def test_scene_label_prompt_asks_for_an_extreme_and_five_candidates():
    """Owner: "THE ONLY READER ... doesnt tell anything ... i wouldnt get
    excited". The 9 examples each put the MC at an extreme on a ladder."""
    p = pc.build_package_prompt({"premise": "P"}, "B")
    assert "EXTREME" in p and "5 candidate labels" in p
    assert "e.g." not in p


def test_only_grounded_label_candidates_are_offered():
    beats = {"beats": [{"segments": [{"line": "he reaches level 11"}]}]}
    pkg = {"title": "T", "thumbnail_style": "power_reveal",
           "labels": ["LEVEL 999 GOD", "F-RANK -> KING", "LEVEL 11 READER"],
           "hashtags": []}
    c = pc.assemble_package(beats, {}, pkg, series_title="X")
    assert c["hooks"] == ["F-RANK -> KING", "LEVEL 11 READER"]
    assert c["hook"] == "F-RANK -> KING"


def test_look_evidence_must_not_contradict_the_registry_hair_colour(tmp_path,
                                                                   monkeypatch):
    """ORV suggestions offered "a man with short light blonde hair" (Ep154) and
    "short grey hair" (Ep302) as Dokja: length and gender matched, colour not."""
    import json as _j
    import types as _t
    ep = tmp_path / "ch"; ep.mkdir()
    (ep / "manifest.cast.json").write_text(_j.dumps({"cast": [
        {"id": "our_protagonist", "canonical_name": "our protagonist",
         "visual_description": "A young man with short dark hair."}]}))
    (ep / "manifest.panels.understood.json").write_text(_j.dumps({"panels": [
        {"scene_file": "p1.jpg", "subjects": ["a young man with short black hair"]},
        {"scene_file": "p2.jpg", "subjects": ["a man with short light blonde hair"]},
        {"scene_file": "p3.jpg", "subjects": ["a man with short grey hair"]}]}))
    import sys as _s
    real = _s.modules.get("cast_identity") or __import__("cast_identity")
    fake = _t.ModuleType("cast_identity")
    fake.__dict__.update(real.__dict__)
    fake.resolve_figures_by_file = lambda u, c: {
        f: [{"name": "our protagonist"}] for f in ("p1.jpg", "p2.jpg", "p3.jpg")}
    monkeypatch.setitem(_s.modules, "cast_identity", fake)
    portraits, look = pc._lead_panels(str(ep))
    assert look == {"p1.jpg"}
    assert portraits == {"p1.jpg"}        # a colour clash is not the lead at all


def test_lead_panels_trust_the_image_pass_when_it_exists(tmp_path, monkeypatch):
    """Thumbnail refs come from the same authority as the narration. 5 of the 8
    tiles the keyword path suggested for ORV were other men; the image pass got
    8 of 8 in the spike."""
    import json as _j
    import types as _t
    ep = tmp_path / "ch"; ep.mkdir()
    (ep / "manifest.cast.json").write_text(_j.dumps({"cast": [
        {"id": "our_protagonist", "canonical_name": "our protagonist",
         "visual_description": "A young man with short dark hair."}]}))
    (ep / "manifest.panels.understood.json").write_text(_j.dumps({"panels": [
        {"scene_file": "p1.jpg", "subjects": ["a young man with short dark hair"]},
        {"scene_file": "p2.jpg", "subjects": ["a young man with short dark hair"]},
        {"scene_file": "p3.jpg", "subjects": ["a young man with short dark hair"]}]}))
    (ep / "manifest.identity.json").write_text(_j.dumps({"panels": {
        "p1.jpg": {"names": ["our protagonist"], "others": 0},   # confirmed
        "p2.jpg": {"names": [], "others": 1},                    # someone else
        "p3.jpg": {"names": ["our protagonist"], "others": 3}}}))  # a crowd
    fake = _t.ModuleType("cast_identity")
    fake.__dict__.update(__import__("cast_identity").__dict__)
    fake.resolve_figures_by_file = lambda u, c, **kw: (_ for _ in ()).throw(
        AssertionError("keyword matching must not run when the image pass exists"))
    monkeypatch.setitem(__import__("sys").modules, "cast_identity", fake)
    portraits, look = pc._lead_panels(str(ep))
    assert portraits == {"p1.jpg"} and look == {"p1.jpg"}


# --- the teaser window is the thumbnail's universe --------------------------

def _understood_chapters(tmp_path, chapters):
    """Chapter dirs named so load_bundle_panels parses their number."""
    import json as _json
    eps = []
    for i, panels in enumerate(chapters, start=1):
        d = tmp_path / f"Chapter_{i}"
        d.mkdir()
        # `printed` is a FIXTURE key: it is written as the vision manifest's OCR
        # and stripped from the understood copy, exactly as production has it
        (d / "manifest.panels.understood.json").write_text(_json.dumps(
            {"panels": [{k: v for k, v in p.items() if k != "printed"}
                        for p in panels]}))
        (d / "manifest.vision.json").write_text(_json.dumps({"items": [
            {"scene_file": p["scene_file"], "ocr_clean": p.get("printed", "")}
            for p in panels]}))
        eps.append(str(d))
    return eps


def _story(name, desc, intensity="tense"):
    return {"scene_file": name, "panel_kind": "story", "intensity": intensity,
            "description": desc, "action": desc}


def test_teaser_montage_computed_stays_inside_the_scan_window(tmp_path):
    """No teaser manifest: the montage is computed with the teaser's own
    selector over the teaser's own window. A bigger reveal in a chapter past
    the window must not be reachable -- that is how the ORV thumbnail landed
    on Episode 124 while its teaser lives in chapters 3-9."""
    eps = _understood_chapters(tmp_path, [
        [_story("a1.jpg", "he is mocked as the weakest hunter"),
         _story("a2.jpg", "a countdown appears above the city")],
        [_story("b1.jpg", "his hidden power awakens, he is transformed",
                "intense"),
         _story("b2.jpg", "onlookers stare in shock")],
        [_story("z1.jpg", "he awakens as a god, reborn and transformed, "
                "ultimate power unlocked", "explosive")],
    ])
    montage, source = pc.teaser_montage(
        eps, "", scan=2, min_panels=2, max_panels=4, tail_frac=0.0)
    assert source == "montage:computed"
    files = [p["scene_file"] for p in montage]
    assert not any(f.endswith("z1.jpg") for f in files)   # outside the window
    assert files[-1].endswith("b1.jpg")                   # climax LAST


def _teaser_manifest(tmp_path, eps, picks, lines):
    """A manifest.teaser.json in the shape teaser_planner.select_and_write
    emits: namespaced ids, absolute panel_sources, climax LAST."""
    import json as _json
    import os as _os
    scene_files, sources, narration = [], {}, []
    for (ci, name), line in zip(picks, lines):
        ns = f"ch{ci + 1}__{name}"
        scenes = Path(eps[ci]) / "scenes"
        scenes.mkdir(exist_ok=True)
        (scenes / name).write_bytes(b"")
        scene_files.append(ns)
        sources[ns] = _os.path.join(eps[ci], "scenes", name)
        narration.append({"scene_file": ns, "line": line})
    out = tmp_path / "teaser"
    out.mkdir()
    man = out / "manifest.teaser.json"
    man.write_text(_json.dumps({
        "source_chapters": sorted({ci + 1 for ci, _ in picks}),
        "scene_files": scene_files, "panel_sources": sources,
        "panel_narration": narration, "reason": "the countdown hooks"}))
    return str(man)


def test_teaser_montage_too_few_panels_says_digest(tmp_path):
    """Never a silent empty set: the caller must be able to record that the
    teaser window could not supply a montage."""
    eps = _understood_chapters(tmp_path, [[_story("a1.jpg", "a quiet street")]])
    montage, source = pc.teaser_montage(
        eps, "", scan=12, min_panels=4, max_panels=10, tail_frac=0.0)
    assert montage is None and source == "digest"


def test_teaser_montage_prefers_the_planned_teaser(tmp_path):
    """A planned/approved teaser is what the owner reviewed: its panels, in its
    order, with its lines -- not a fresh re-selection that may differ."""
    eps = _understood_chapters(tmp_path, [
        [_story("a1.jpg", "he is mocked as the weakest hunter"),
         _story("a2.jpg", "a countdown appears above the city")],
        [_story("b1.jpg", "his hidden power awakens, he is transformed",
                "intense"),
         _story("b2.jpg", "onlookers stare in shock")],
    ])
    man = _teaser_manifest(tmp_path, eps, [(0, "a1.jpg"), (0, "a2.jpg")],
                           ["Nobody believed in him.", "The countdown began."])
    montage, source = pc.teaser_montage(
        eps, man, scan=12, min_panels=2, max_panels=4, tail_frac=0.0)
    assert source == "teaser:manifest"
    assert [Path(p["scene_file"]).name for p in montage] == ["a1.jpg", "a2.jpg"]
    assert montage[-1]["line"] == "The countdown began."
    assert "countdown" in montage[-1]["description"]      # understood fields ride


def test_teaser_montage_ignores_a_manifest_with_dangling_panels(tmp_path):
    """panel_sources are absolute paths from whichever run planned the teaser;
    a panel that no longer exists cannot be a thumbnail reference."""
    import os as _os
    eps = _understood_chapters(tmp_path, [
        [_story("a1.jpg", "he is mocked as the weakest hunter"),
         _story("a2.jpg", "a countdown appears above the city")],
        [_story("b1.jpg", "his hidden power awakens, he is transformed",
                "intense"),
         _story("b2.jpg", "onlookers stare in shock")],
    ])
    man = _teaser_manifest(tmp_path, eps, [(0, "a1.jpg"), (0, "a2.jpg")],
                           ["x", "y"])
    _os.remove(_os.path.join(eps[0], "scenes", "a2.jpg"))
    montage, source = pc.teaser_montage(
        eps, man, scan=12, min_panels=2, max_panels=4, tail_frac=0.0)
    assert source == "montage:computed"


# --- the hook DESIGN is ranked from the teaser, never chosen by the model ---

def _system(name, text):
    """Measured on ORV's teaser (2026-09-21): all six system panels carry an
    EMPTY dialogue; the window's words are OCR in manifest.vision.json."""
    return {"scene_file": name, "panel_kind": "system", "intensity": "tense",
            "description": "a glowing blue system window", "dialogue": "",
            "printed": text}


def test_rank_designs_system_teaser_leads_with_the_system_window():
    montage = [_system("s1.jpg", "The free service has ended."),
               _story("p1.jpg", "a man stares at the window in shock"),
               _system("s2.jpg", "Main scenario has begun."),
               _story("p2.jpg", "he grips a blade as the crowd panics")]
    names, reason = pc.rank_designs(montage)
    assert names[0] == "system_window" and len(names) == 2
    assert reason["system_panels"] == 2 and reason["card_lines"] == 2
    assert reason["system_share"] == 0.5


def test_rank_designs_transformation_climax_prefers_the_headline():
    montage = [_story("p1.jpg", "he is mocked as the weakest hunter"),
               _story("p2.jpg", "onlookers laugh at him"),
               _story("p3.jpg", "his hidden power awakens, he is transformed",
                      "intense")]                       # climax is LAST
    names, reason = pc.rank_designs(montage)
    assert names == ["nametag_headline", "nametag"]
    assert reason["climax_transform_hits"] >= 1


def test_rank_designs_quiet_montage_still_offers_two_known_designs():
    from thumbnail_styles import HOOK_DESIGNS
    montage = [_story("p1.jpg", "a quiet street at dusk"),
               _story("p2.jpg", "he walks home alone")]
    names, _ = pc.rank_designs(montage)
    assert names == ["nametag", "nametag_headline"]
    assert all(n in HOOK_DESIGNS for n in names)


def test_rank_designs_null_system_text_does_not_earn_a_window():
    """'None.' is a non-empty string that passes every `if x:` -- a card built
    from it would print a null value on the thumbnail."""
    montage = [_system("s1.jpg", "None."), _system("s2.jpg", ""),
               _story("p1.jpg", "he walks home alone")]
    names, reason = pc.rank_designs(montage)
    assert "system_window" not in names
    assert reason["system_panels"] == 2 and reason["card_lines"] == 0


# --- a design reaches the concept: overlay, second label, quoted card -------

def _pkg(**extra):
    return {"title": "T", "description": "D", "thumbnail_style": "power_reveal",
            "labels": ["WEAKEST HUNTER"], "hashtags": ["#m"], **extra}


def test_assemble_package_design_sets_the_overlay_and_is_recorded():
    from thumbnail_styles import HOOK_DESIGNS
    c = pc.assemble_package({}, {}, _pkg(), series_title="X", design="nametag")
    assert c["design"] == "nametag"
    assert c["style"] == "power_reveal"              # the ART stays a scene style
    assert c["style_overlay"] == HOOK_DESIGNS["nametag"]["overlay"]


def test_assemble_package_headline_becomes_a_tag_without_an_arrow():
    """The second label was unreachable: the two-stage path never wrote tags."""
    from thumbnail_styles import HOOK_DESIGNS
    c = pc.assemble_package(
        {}, {}, _pkg(headlines=["BACK FROM THE DEAD", "NOBODY -> KING"]),
        series_title="X", design="nametag_headline")
    pos = HOOK_DESIGNS["nametag_headline"]["overlay"]["headline_pos"]
    assert c["tags"] == [{"text": "BACK FROM THE DEAD", "pos": pos,
                          "arrow": False}]
    assert c["headlines"] == ["BACK FROM THE DEAD", "NOBODY -> KING"]


def test_assemble_package_invented_number_headline_is_not_offered():
    beats = {"beats": [{"segments": [{"line": "he reaches level 3"}]}]}
    c = pc.assemble_package(
        beats, {}, _pkg(headlines=["LEVEL 999 GOD", "BACK FROM THE DEAD"]),
        series_title="X", design="nametag_headline")
    assert c["headlines"] == ["BACK FROM THE DEAD"]
    assert c["tags"][0]["text"] == "BACK FROM THE DEAD"


def test_package_prompt_asks_for_headlines_only_for_that_design():
    plain = pc.build_package_prompt({"premise": "P"}, "B")
    head = pc.build_package_prompt({"premise": "P"}, "B",
                                   design="nametag_headline")
    assert '"headlines"' in head and '"headlines"' not in plain


def test_system_card_lines_quote_the_story_verbatim():
    """The card QUOTES printed system text: usable lines only, no repeats,
    at most three -- never a word the story did not print."""
    montage = [_system("s1.jpg", "The free service has ended."),
               _system("s2.jpg", "None."),
               _story("p1.jpg", "a man stares in shock"),
               _system("s3.jpg", "The free service has ended."),
               _system("s4.jpg", "Main scenario has begun."),
               _system("s5.jpg", "Kill one or more living things."),
               _system("s6.jpg", "Time limit: 30 minutes.")]
    assert pc.system_card_lines(montage) == [
        "The free service has ended.", "Main scenario has begun.",
        "Kill one or more living things."]


def test_assemble_package_system_window_carries_the_quoted_card():
    c = pc.assemble_package({}, {}, _pkg(), series_title="X",
                            design="system_window",
                            card_lines=["Main scenario has begun."])
    assert c["card"] == ["Main scenario has begun."]


# --- main(): the teaser window drives the run -------------------------------

def test_brief_prompt_puts_the_hook_block_above_the_narration():
    plain = pc.build_brief_prompt("NARRATION", "B")
    hooked = pc.build_brief_prompt("NARRATION", "B", hook_block="THE HOOK\nx")
    assert hooked.index("THE HOOK") < hooked.index("STORY NARRATION")
    assert "THE HOOK" not in plain


def test_teaser_hook_block_lists_the_montage_and_marks_the_climax():
    montage = [{**_story("p1.jpg", "he is mocked as the weakest hunter"),
                "line": "Nobody believed in him."},
               {**_story("p2.jpg", "his hidden power awakens"),
                "line": "Until the day it woke."}]
    block = pc.teaser_hook_block(montage, "the awakening hooks")
    assert "the awakening hooks" in block
    assert "Nobody believed in him." in block
    assert block.index("weakest hunter") < block.index("power awakens")
    assert "CLIMAX" in block.splitlines()[-1]          # the LAST panel
    assert pc.teaser_hook_block([], "") == ""


def _series(tmp_path, chapters):
    """Chapter dirs carrying both manifests main() and the teaser loader read."""
    import json as _j
    eps = _understood_chapters(tmp_path, chapters)
    for e, panels in zip(eps, chapters):
        (Path(e) / "manifest.beats.json").write_text(_j.dumps({"beats": [{
            "segments": [{"line": p.get("description", "")} for p in panels],
            "scene_selection": [{"scene_file": p["scene_file"],
                                 "intensity": p["intensity"]} for p in panels]}]}))
    return eps


_ARC = [
    [_story("a1.jpg", "he is mocked as the weakest hunter"),
     _system("a2.jpg", "Main scenario has begun.")],
    [_story("b1.jpg", "his hidden power awakens, he is transformed", "intense"),
     _system("b2.jpg", "[YOU HAVE EXCEEDED THE TIME LIMIT.]")],
    [_story("z1.jpg", "he awakens as a god, reborn and transformed, ultimate "
            "power unlocked", "explosive")],
]


def _run_main(monkeypatch, argv, replies=()):
    import sys as _s
    prompts, it = [], iter(replies)

    def fake(prompt, model):
        prompts.append(prompt)
        return next(it)
    monkeypatch.setattr(pc, "_gemma", fake)
    monkeypatch.setattr(_s, "argv", ["publish_concept.py", *argv])
    return pc.main(), prompts


def test_rank_designs_run_is_free_and_writes_the_ranking(tmp_path, monkeypatch):
    import json as _j
    eps = _series(tmp_path, _ARC)
    out = tmp_path / "rank.json"
    rc, prompts = _run_main(monkeypatch, [
        "--episode-dirs", ",".join(eps), "--rank-designs",
        "--teaser-scan-chapters", "2", "--teaser-min-panels", "2",
        "--out", str(out)])
    assert rc == 0 and prompts == []                   # no model call, nothing paid
    got = _j.loads(out.read_text())
    assert got["designs"][0] == "system_window" and len(got["designs"]) == 2
    assert got["claim_source"] == "montage:computed"
    assert got["reason"]["system_panels"] == 2
    assert got["reason"]["card_lines"] == 2      # two windows offered
    # the WINDOW, not just a count: "text is reviewed before art is bought"
    # cannot be done from a number (verifier, 2026-09-21)
    assert got["card"][0]["line"] == "Main scenario has begun."


def test_rank_designs_run_without_a_montage_fails_loud(tmp_path, monkeypatch):
    eps = _series(tmp_path, [[_story("a1.jpg", "a quiet street")]])
    out = tmp_path / "rank.json"
    rc, _ = _run_main(monkeypatch, [
        "--episode-dirs", ",".join(eps), "--rank-designs",
        "--teaser-scan-chapters", "12", "--teaser-min-panels", "4",
        "--out", str(out)])
    assert rc == 2 and not out.exists()


def test_design_run_takes_its_claim_and_climax_from_the_teaser_window(
        tmp_path, monkeypatch):
    import json as _j
    eps = _series(tmp_path, _ARC)
    out = tmp_path / "c.json"
    rc, prompts = _run_main(monkeypatch, [
        "--episode-dirs", ",".join(eps), "--design", "nametag_headline",
        "--teaser-scan-chapters", "2", "--teaser-min-panels", "2",
        "--out", str(out)],
        replies=[{"premise": "P"},
                 {"title": "T", "thumbnail_style": "power_reveal",
                  "labels": ["WEAKEST HUNTER"], "headlines": ["NOBODY -> KING"],
                  "hashtags": []}])
    assert rc == 0
    c = _j.loads(out.read_text())
    assert c["design"] == "nametag_headline"
    assert c["claim_source"] == "montage:computed"
    assert c["climax_chapter_index"] == 1             # NOT chapter 3's bigger reveal
    assert c["tags"][0]["text"] == "NOBODY -> KING"
    assert c["design_reason"]["climax_transform_hits"] >= 1
    assert all(not p.endswith("z1.jpg") for p in c["teaser_panels"])
    assert "THE HOOK" in prompts[0] and "power awakens" in prompts[0]
    assert "reborn" not in prompts[0]                  # chapter 3 never reaches the brief
    assert c["badge"] == "3 CHAPTERS"                  # the upload's facts stay whole


def test_system_window_run_quotes_the_card_from_the_teaser(tmp_path, monkeypatch):
    import json as _j
    eps = _series(tmp_path, _ARC)
    out = tmp_path / "c.json"
    rc, _ = _run_main(monkeypatch, [
        "--episode-dirs", ",".join(eps), "--design", "system_window",
        "--teaser-scan-chapters", "2", "--teaser-min-panels", "2",
        "--out", str(out)],
        replies=[{"premise": "P"},
                 {"title": "T", "thumbnail_style": "power_reveal",
                  "labels": ["WEAKEST HUNTER"], "hashtags": []}])
    assert rc == 0
    assert _j.loads(out.read_text())["card"][0]["line"] == "Main scenario has begun."


def test_title_reuses_the_picked_thumbnails_headline_too():
    """The title repeats the thumbnail's status words so both make one claim.
    With a hook design the headline lives in `tags`, not in `hook`."""
    picked = {"hook": "WEAKEST HUNTER",
              "tags": [{"text": "NOBODY -> KING", "pos": "lower_left",
                        "arrow": False}, {"text": "  "}]}
    assert pc.thumb_label_words(picked) == "WEAKEST HUNTER / NOBODY -> KING"
    assert pc.thumb_label_words({"hook": "WEAK|KING"}) == "WEAK / KING"
    assert pc.thumb_label_words({}) == ""


# --- the card quotes OCR, and raw OCR is not printable ----------------------
# Measured over 8 series' teaser windows: ORV 66 system panels -> 16 printable
# lines. The rest is markup, mirrored text read as Cyrillic, multi-window blobs.

def test_printable_card_line_keeps_a_clean_line_without_its_brackets():
    assert pc.printable_card_line("[THE MAIN SCENARIO HAS ARRIVED.]") == \
        "THE MAIN SCENARIO HAS ARRIVED."


def test_printable_card_line_repairs_brackets_ocr_read_as_letters():
    """'[' and ']' are read as 'I' and 'J' glued to the first and last word."""
    assert pc.printable_card_line(
        "INO ONE MAY ENTER OR LEAVE THE CABIN.J") == \
        "NO ONE MAY ENTER OR LEAVE THE CABIN."


def test_printable_card_line_refuses_what_must_never_be_printed():
    for junk in ("?! DING \u0418\u041790 \u04178 \u0422\u041e\u0419\u041c\u0410\u0417",  # mirrored text -> Cyrillic
                 "<SKILL> KNIFE FIGHTING LV. 1 E0x",         # crop debris
                 "IS** CLEAR REWARD: YOU HAVE BEEN AWARDED A BADGE.",  # markup debris
                 "DING YOU HAVE RECEIVED AN EXCLUSIVE ATTRIBUTE AND A SKILL SLOT "
                 "HAS BEEN UNLOCKED AND EACH OF THE SURVIVORS RECEIVES ONE",  # a blob
                 "None.", "", "OK"):
        assert pc.printable_card_line(junk) == "", junk


def test_teaser_montage_attaches_the_printed_words_from_the_vision_manifest(tmp_path):
    eps = _understood_chapters(tmp_path, [
        [_system("s1.jpg", "[THE MAIN SCENARIO HAS ARRIVED.]"),
         _story("a1.jpg", "he is mocked as the weakest hunter")],
        [_story("b1.jpg", "his hidden power awakens, he is transformed",
                "intense"),
         _story("b2.jpg", "onlookers stare in shock")]])
    montage, _ = pc.teaser_montage(eps, "", scan=12, min_panels=2,
                                   max_panels=4, tail_frac=0.0)
    by_name = {Path(p["scene_file"]).name: p for p in montage}
    assert by_name["s1.jpg"]["printed"] == "[THE MAIN SCENARIO HAS ARRIVED.]"
    assert by_name["b1.jpg"]["printed"] == ""


def test_printable_card_line_refuses_two_things_glued_into_one():
    """Measured on the tower series: a system line, a bracket misread as '1',
    then a character's speech -- every token a real word, the line garbage."""
    for glued in ("TRAITS AND COMBAT SKILLS ARE NOW AVAILABLE.1 LET'S MAKE THIS WORK...!!",
                  "THE SCENARIO HAS ENDED. WHAT IS GOING ON HERE",
                  "YOU HAVE LEVELED UP...!!"):
        assert pc.printable_card_line(glued) == "", glued
    # one sentence with its own full stop is still fine
    assert pc.printable_card_line("[THE SCENARIO HAS ENDED.]") == \
        "THE SCENARIO HAS ENDED."


def test_system_card_never_prints_the_licensed_title():
    """Integrity rule 1, the one with legal weight: the series name is never
    rendered. ORV's only printable montage line names the series' own skill."""
    montage = [_system("s1.jpg", "[YOU HAVE UNLOCKED YOUR PERSONAL SKILL, "
                       "OMNISCIENT READER'S VIEWPOINT.]"),
               _system("s2.jpg", "[THE MAIN SCENARIO HAS ARRIVED.]")]
    assert pc.system_card_lines(montage, banned="Omniscient Reader") == [
        "THE MAIN SCENARIO HAS ARRIVED."]
    # a one-word overlap is the story's own vocabulary, not the title
    reader = [_system("s3.jpg", "[THE READER HAS ENTERED THE SCENARIO.]")]
    assert pc.system_card_lines(reader, banned="Omniscient Reader") == [
        "THE READER HAS ENTERED THE SCENARIO."]


def test_rank_designs_does_not_count_a_line_the_ban_will_refuse():
    """Ranking and building must agree: a window whose only line names the
    series is not a window the build can fill."""
    montage = [_system("s1.jpg", "[YOU HAVE UNLOCKED YOUR PERSONAL SKILL, "
                       "OMNISCIENT READER'S VIEWPOINT.]"),
               _system("s2.jpg", "?! DING E0x"),
               _story("p1.jpg", "he walks home alone")]
    assert "system_window" in pc.rank_designs(montage)[0]
    assert "system_window" not in pc.rank_designs(
        montage, banned="Omniscient Reader")[0]


def test_system_window_run_without_a_printable_line_fails_loud(tmp_path, monkeypatch):
    """Never ship a system-window design with an empty left third."""
    eps = _series(tmp_path, [
        [_story("a1.jpg", "he is mocked as the weakest hunter"),
         _system("a2.jpg", "?! DING E0x")],
        [_story("b1.jpg", "his hidden power awakens, he is transformed", "intense"),
         _story("b2.jpg", "onlookers stare in shock")]])
    out = tmp_path / "c.json"
    rc, _ = _run_main(monkeypatch, [
        "--episode-dirs", ",".join(eps), "--design", "system_window",
        "--teaser-scan-chapters", "2", "--teaser-min-panels", "2",
        "--out", str(out)],
        replies=[{"premise": "P"},
                 {"title": "T", "thumbnail_style": "power_reveal",
                  "labels": ["WEAKEST HUNTER"], "hashtags": []}])
    assert rc == 2 and not out.exists()


# --- the card quotes the teaser WINDOW; the montage decides eligibility -----
# ORV, measured: 6 of 10 montage panels are system windows, but their own OCR
# yields one line (and it names the series). The window holds 16 printable ones.

_WINDOW = [
    [_system("s1.jpg", "?! DING E0x"),                       # montage, junk OCR
     _story("a1.jpg", "he is mocked as the weakest hunter"),
     _system("s2.jpg", "[THE MAIN SCENARIO HAS ARRIVED.]")],
    [_story("b1.jpg", "his hidden power awakens, he is transformed", "intense"),
     _system("s3.jpg", "[YOU HAVE EXCEEDED THE TIME LIMIT.]")],
]


def test_window_card_lines_take_the_montage_first_then_the_window(tmp_path):
    eps = _understood_chapters(tmp_path, _WINDOW)
    montage = pc._attach_printed([
        {**_system("s3.jpg", ""), "scene_file": eps[1] + "/scenes/s3.jpg"},
        {**_story("b1.jpg", "his hidden power awakens"),
         "scene_file": eps[1] + "/scenes/b1.jpg"}])
    assert pc.window_card_lines(eps, montage) == [
        "YOU HAVE EXCEEDED THE TIME LIMIT.",          # the montage's own line
        "THE MAIN SCENARIO HAS ARRIVED."]             # then the window, in order
    assert pc.window_card_lines(eps, montage, max_lines=1) == [
        "YOU HAVE EXCEEDED THE TIME LIMIT."]


def test_rank_designs_offers_the_window_when_the_window_can_fill_it():
    montage = [_system("s1.jpg", "?! DING E0x"), _system("s2.jpg", ""),
               _story("p1.jpg", "he walks home alone"),
               _story("p2.jpg", "a quiet street at dusk")]
    names, reason = pc.rank_designs(
        montage, card_lines=["THE MAIN SCENARIO HAS ARRIVED."])
    assert names[0] == "system_window"
    assert reason["system_panels"] == 2 and reason["system_share"] == 0.5
    assert reason["card_lines"] == 1


def test_rank_designs_one_incidental_system_panel_is_not_a_system_story():
    """Measured on the Mini (2026-09-21), system panels per 10-panel montage:
    ORV 6, the tower series 2, Nano Machine 1, Infinite Evolution 1 -- and that
    last one's "window" is a TV news broadcast. One panel is incidental."""
    one = [_system("s1.jpg", "[WE HAVE SOME BREAKING NEWS.]"),
           _story("p1.jpg", "he walks home alone"),
           _story("p2.jpg", "a quiet street at dusk")]
    assert "system_window" not in pc.rank_designs(one)[0]
    two = one + [_system("s2.jpg", "[THE MAIN SCENARIO HAS ARRIVED.]")]
    assert pc.rank_designs(two)[0][0] == "system_window"


def test_rank_designs_a_story_without_system_panels_gets_no_window():
    """One title card somewhere in chapter 2 does not make a system story."""
    montage = [_story("p1.jpg", "a quiet street at dusk"),
               _story("p2.jpg", "he walks home alone")]
    names, _ = pc.rank_designs(
        montage, card_lines=["THE MAIN SCENARIO HAS ARRIVED."])
    assert "system_window" not in names


def test_ref_candidates_prefer_the_teaser_window(tmp_path, monkeypatch):
    """The claim and climax come from the first N chapters, so the LOOK the art
    copies should too: ORV's late chapters draw the lead with longer hair, and a
    better-ranked tile from chapter 300 would bring that look back."""
    import json as _j
    ok = {"text_coverage": 0.0, "width": 800, "height": 900}
    eps = _cand_arc(tmp_path, monkeypatch, n=6,
                    panels=[("p.jpg", ["a young man"], {}, ok)])
    for late in eps[2:]:                 # late chapters rank BETTER on the old rules
        man = Path(late) / "manifest.panels.understood.json"
        u = _j.loads(man.read_text())
        u["panels"][0]["description"] = "A close-up of his face"
        man.write_text(_j.dumps(u))
    assert [x["chapter"] for x in pc.ref_candidates(eps, n=2)["refs"]] == [2, 3]
    assert [x["chapter"] for x in
            pc.ref_candidates(eps, n=2, prefer_first=2)["refs"]] == [0, 1]
    # ONLY the window, never a top-up. Measured on ORV 2026-09-21: the window has
    # 2 clean tiles, and "top up to 8" refilled the page with Episodes 100-262 --
    # other characters, the exact tiles the owner had just rejected. Two right
    # tiles beat two right and six wrong.
    assert [x["chapter"] for x in
            pc.ref_candidates(eps, n=8, prefer_first=2)["refs"]] == [0, 1]


def test_design_run_copies_the_lead_from_clean_solo_shots_in_the_window(
        tmp_path, monkeypatch):
    """The automatic refs used to be whatever 'lead' panels sat in the CLIMAX
    chapter. Looked at on ORV (2026-09-21): the back of a head, a half profile,
    and a close-up of a different character. The clean-solo-shot rule the
    suggestion tiles use already finds the right panel (Episode_5 p000022), so
    the automatic path uses it too, and keeps the old choice only when the
    window has no clean shot."""
    import json as _j
    eps = _series(tmp_path, _ARC)

    def run(out):
        return _run_main(monkeypatch, [
            "--episode-dirs", ",".join(eps), "--design", "nametag",
            "--teaser-scan-chapters", "2", "--teaser-min-panels", "2",
            "--out", str(out)],
            replies=[{"premise": "P"},
                     {"title": "T", "thumbnail_style": "power_reveal",
                      "labels": ["WEAKEST HUNTER"], "hashtags": []}])

    # no clean solo shot anywhere: the old climax refs are kept
    monkeypatch.setattr(pc, "_lead_panels", lambda d, max_figures=2: (set(), set()))
    rc, _ = run(tmp_path / "a.json")
    assert rc == 0
    assert _j.loads((tmp_path / "a.json").read_text())["refs"] == ["b1.jpg"]

    # chapter 1 holds a clean solo shot of the lead; chapter 3 (outside the
    # window) holds a better-looking one that must never be used
    for ep, name in ((eps[0], "a1.jpg"), (eps[2], "z1.jpg")):
        (Path(ep) / "manifest.vision.json").write_text(_j.dumps({"items": [
            {"scene_file": name, "ocr_clean": "", "text_coverage": 0.0,
             "width": 800, "height": 900}]}))
        u = _j.loads((Path(ep) / "manifest.panels.understood.json").read_text())
        for pnl in u["panels"]:
            pnl["subjects"] = ["a young man with short black hair"]
        (Path(ep) / "manifest.panels.understood.json").write_text(_j.dumps(u))
    monkeypatch.setattr(pc, "_lead_panels",
                        lambda d, max_figures=2: ({"a1.jpg", "z1.jpg"},
                                                  {"a1.jpg", "z1.jpg"}))
    rc, _ = run(tmp_path / "b.json")
    assert rc == 0
    refs = _j.loads((tmp_path / "b.json").read_text())["refs"]
    assert refs == [str(Path(eps[0]) / "scenes" / "a1.jpg")]


# --- ONE claim per series: every card, the painted scene and the titles read it ---
# Owner, 2026-09-22, on the first two ORV cards: "why left and right label
# selections are different? ... we dont see a story on the thumbnail, the video
# title, teaser and thumbnail must in complementary to eachother". Each option
# had run its OWN two model calls: two summaries, two label lists, two titles;
# and the summary never reached the image model, which got a fixed "hero with an
# aura and shocked onlookers" for every series.

def _a_claim(**extra):
    return {"title": "T - Manhwa Recap", "style": "power_reveal",
            "brief": {"premise": "P"},
            "scene": "A crowded subway car; a blank glowing window hangs in the "
                     "air; one calm man reads his phone while passengers panic.",
            "labels": ["THE ONLY READER", "SCENARIO CLEARER"],
            "headlines": ["READER -> LEGEND", "SPOILERS AS WEAPONS"],
            "card": ["THE MAIN SCENARIO HAS ARRIVED.",
                     "YOU HAVE EXCEEDED THE TIME LIMIT."],
            "designs": ["system_window", "nametag_headline"],
            "design_reason": {"system_panels": 6}, "claim_source": "teaser:manifest",
            "teaser_panels": ["/abs/ch3/scenes/p1.jpg"], "refs": ["/abs/ep5/p22.jpg"],
            "climax_chapter_index": 1, "badge": "3 CHAPTERS", **extra}


def test_cards_built_from_one_claim_share_labels_scene_and_refs():
    a = pc.concept_from_claim(_a_claim(), "system_window")
    b = pc.concept_from_claim(_a_claim(), "nametag_headline")
    for key in ("hooks", "hook", "scene", "refs", "title", "claim_source",
                "climax_chapter_index", "badge"):
        assert a[key] == b[key], key
    assert a["hook"] == "THE ONLY READER"
    assert a["design"] == "system_window" and b["design"] == "nametag_headline"


def test_a_system_window_carries_ONE_quoted_line():
    """The examples' windows hold one short line in huge type (the hook). Three
    sentences of fine print in a mostly empty box read as nothing."""
    assert pc.concept_from_claim(_a_claim(), "system_window")["card"] == [
        "THE MAIN SCENARIO HAS ARRIVED."]
    assert "card" not in pc.concept_from_claim(_a_claim(), "nametag_headline")


def test_only_the_headline_design_gets_the_headline_tag():
    from thumbnail_styles import HOOK_DESIGNS
    b = pc.concept_from_claim(_a_claim(), "nametag_headline")
    assert b["tags"] == [{"text": "READER -> LEGEND", "arrow": False,
                          "pos": HOOK_DESIGNS["nametag_headline"]["overlay"]["headline_pos"]}]
    assert not pc.concept_from_claim(_a_claim(), "system_window").get("tags")


def test_concept_from_claim_refuses_a_design_the_claim_cannot_fill():
    import pytest
    with pytest.raises(ValueError, match="system_window"):
        pc.concept_from_claim(_a_claim(card=[]), "system_window")
    with pytest.raises(ValueError, match="unknown"):
        pc.concept_from_claim(_a_claim(), "triptych")


def test_claim_prompt_asks_for_one_paintable_scene():
    p = pc.build_claim_prompt({"premise": "P"}, "Banned Title")
    assert '"scene"' in p and '"labels"' in p and '"headlines"' in p
    low = p.lower()
    assert "blank" in low                   # any window or screen is painted BLANK
    assert "no character names" in low or "no real character names" in low
    assert "Banned Title" in p


_CLAIM_REPLIES = [
    {"premise": "A commuter's finished web novel becomes reality."},
    {"title": "lonely reader becomes the only one who knows the script",
     "description": "D", "hashtags": ["#m"],
     "scene": "A crowded subway car; a blank glowing window hangs in the air; "
              "one calm man reads his phone while passengers panic.",
     "labels": ["WEAKEST HUNTER", "LEVEL 999 GOD"],
     "headlines": ["NOBODY -> KING"]}]


def _write_claim(tmp_path, monkeypatch, arc=_ARC, extra=()):
    eps = _series(tmp_path, arc)
    out = tmp_path / "claim.json"
    rc, prompts = _run_main(monkeypatch, [
        "--episode-dirs", ",".join(eps), "--write-claim", str(out),
        "--teaser-scan-chapters", "2", "--teaser-min-panels", "2", *extra],
        replies=list(_CLAIM_REPLIES))
    return eps, out, rc, prompts


def test_write_claim_understands_once_and_saves_everything_a_card_needs(
        tmp_path, monkeypatch):
    import json as _j
    eps, out, rc, prompts = _write_claim(tmp_path, monkeypatch)
    assert rc == 0 and len(prompts) == 2          # understand, then claim: ONCE
    assert "THE HOOK" in prompts[0] and '"scene"' in prompts[1]
    c = _j.loads(out.read_text())
    assert c["scene"].startswith("A crowded subway car")
    assert c["labels"] == ["WEAKEST HUNTER"]      # the invented number is not offered
    assert c["headlines"] == ["NOBODY -> KING"]
    assert c["designs"][0] == "system_window" and len(c["designs"]) == 2
    assert c["card"] and c["claim_source"] == "montage:computed"
    assert c["climax_chapter_index"] == 1 and c["badge"] == "3 CHAPTERS"
    assert c["title"].endswith(" - Manhwa Recap") and c["refs"]


def test_cards_are_built_from_the_claim_without_a_model_call(tmp_path, monkeypatch):
    import json as _j
    eps, claim, rc, _ = _write_claim(tmp_path, monkeypatch)
    assert rc == 0
    got = {}
    for design in ("system_window", "nametag_headline"):
        out = tmp_path / (design + ".json")
        rc, prompts = _run_main(monkeypatch, [
            "--episode-dirs", ",".join(eps), "--claim", str(claim),
            "--design", design, "--out", str(out)])
        assert rc == 0 and prompts == [], design   # NO model call per card
        got[design] = _j.loads(out.read_text())
    a, b = got["system_window"], got["nametag_headline"]
    assert a["hooks"] == b["hooks"] and a["scene"] == b["scene"]
    assert a["title"] == b["title"]
    assert a["card"]["line"] == "Main scenario has begun."   # ONE window
    assert b["tags"][0]["text"] == "NOBODY -> KING"


def test_write_claim_without_a_teaser_montage_fails_loud(tmp_path, monkeypatch):
    eps = _series(tmp_path, [[_story("a1.jpg", "a quiet street")]])
    out = tmp_path / "claim.json"
    rc, prompts = _run_main(monkeypatch, [
        "--episode-dirs", ",".join(eps), "--write-claim", str(out),
        "--teaser-scan-chapters", "12", "--teaser-min-panels", "4"])
    assert rc == 2 and not out.exists() and prompts == []


# --- tone reaches the claim; the owner picks the window line ---------------

def test_claim_prompt_carries_the_tone_and_defaults_to_absurd():
    import pytest
    from thumbnail_styles import CLAIM_TONES
    p = pc.build_claim_prompt({"premise": "P"}, "B")
    assert CLAIM_TONES["absurd"] in p
    assert CLAIM_TONES["erotic"] in pc.build_claim_prompt({"premise": "P"}, "B",
                                                          tone="erotic")
    with pytest.raises(ValueError, match="tone"):
        pc.build_claim_prompt({"premise": "P"}, "B", tone="gritty")


def test_write_claim_records_the_tone_and_every_printable_window_line(
        tmp_path, monkeypatch):
    import json as _j
    eps, out, rc, prompts = _write_claim(tmp_path, monkeypatch,
                                         extra=["--tone", "erotic"])
    assert rc == 0
    c = _j.loads(out.read_text())
    assert c["tone"] == "erotic"
    from thumbnail_styles import CLAIM_TONES
    assert CLAIM_TONES["erotic"] in prompts[1]
    # every printable window is offered; the card is the first
    assert [w["line"] for w in c["card_options"]] == [
        "Main scenario has begun.", "YOU HAVE EXCEEDED THE TIME LIMIT."]
    assert c["card"] == c["card_options"][:1]


def test_concept_from_claim_paints_the_line_the_owner_chose():
    claim = _a_claim(card=["THE MAIN SCENARIO HAS ARRIVED."],
                     card_options=["THE MAIN SCENARIO HAS ARRIVED.",
                                   "100 COINS HAVE BEEN DEDUCTED."])
    assert pc.concept_from_claim(claim, "system_window")["card"] == [
        "THE MAIN SCENARIO HAS ARRIVED."]
    claim["card"] = ["100 COINS HAVE BEEN DEDUCTED."]
    assert pc.concept_from_claim(claim, "system_window")["card"] == [
        "100 COINS HAVE BEEN DEDUCTED."]


# --- a system WINDOW, not a line: header / big line / stat footer -----------
# ORV's first scenario is printed as one window: "MAIN SCENARIO #1 [PROVE YOUR
# VALUE] KILL ONE OR MORE LIVING ORGANISMS. CATEGORY: MAIN DIFFICULTY: F TIME
# LIMIT: 30 MINUTES REWARD: 300 COINS PENALTY FOR FAILURE: DEATH". The
# one-clean-sentence filter threw the whole thing away; the punchiest text in
# the opening was invisible (owner, 2026-09-22: "do this properly").

_ORV_WINDOW = ("MAIN SCENARIO #1 [PROVE YOUR VALUE] KILL ONE OR MORE LIVING "
               "ORGANISMS. CATEGORY: MAIN DIFFICULTY: F TIME LIMIT: 30 MINUTES "
               "REWARD: 300 COINS PENALTY FOR FAILURE: DEATH")


def test_window_from_ocr_splits_header_big_line_and_footer():
    w = pc.window_from_ocr(_ORV_WINDOW)
    assert w == {"header": "MAIN SCENARIO #1",
                 "line": "KILL ONE OR MORE LIVING ORGANISMS.",
                 "footer": ["TIME LIMIT: 30 MINUTES", "PENALTY FOR FAILURE: DEATH"]}


def test_window_from_ocr_keeps_a_plain_sentence_as_the_line():
    assert pc.window_from_ocr("[THE MAIN SCENARIO HAS ARRIVED.]") == {
        "header": "", "line": "THE MAIN SCENARIO HAS ARRIVED.", "footer": []}
    assert pc.window_from_ocr("?! DING E0x") is None
    assert pc.window_from_ocr("") is None


def test_window_from_ocr_never_carries_the_licensed_title():
    assert pc.window_from_ocr(
        "[YOU HAVE UNLOCKED YOUR PERSONAL SKILL, OMNISCIENT READER'S VIEWPOINT.]",
        banned="Omniscient Reader") is None


def test_window_options_offer_structured_windows_first(tmp_path):
    eps = _understood_chapters(tmp_path, [
        [_system("s1.jpg", "[THE MAIN SCENARIO HAS ARRIVED.]"),
         _system("s2.jpg", _ORV_WINDOW),
         _story("a1.jpg", "he is mocked as the weakest hunter")],
        [_story("b1.jpg", "his hidden power awakens, he is transformed", "intense")]])
    montage = pc._attach_printed([
        {**_story("b1.jpg", "his hidden power awakens"),
         "scene_file": eps[1] + "/scenes/b1.jpg"}])
    opts = pc.window_options(eps, montage)
    assert opts[0]["line"] == "KILL ONE OR MORE LIVING ORGANISMS."   # richest first
    assert opts[0]["footer"] == ["TIME LIMIT: 30 MINUTES", "PENALTY FOR FAILURE: DEATH"]
    assert {o["line"] for o in opts} == {"KILL ONE OR MORE LIVING ORGANISMS.",
                                         "THE MAIN SCENARIO HAS ARRIVED."}


def test_window_from_ocr_drops_the_bell_icon_misread():
    """Measured on ORV's opening: the window's bell/box icon is read as DING,
    TING, FOX, HOX, ONIO, INC, VING glued in front of the text."""
    for junk in ("FOX ", "HOX ", "TING ", "DING ", "ONIO ", "VING "):
        w = pc.window_from_ocr(junk + "MAIN SCENARIO #1 [PROVE YOUR VALUE] KILL ONE "
                               "OR MORE LIVING ORGANISMS. PENALTY FOR FAILURE: DEATH")
        assert w and w["line"] == "KILL ONE OR MORE LIVING ORGANISMS.", junk
        assert w["header"] == "MAIN SCENARIO #1", junk
    assert pc.window_from_ocr("TING THE DIFFICULTY HAS BEEN ADJUSTED.")["line"] == \
        "THE DIFFICULTY HAS BEEN ADJUSTED."


def test_claim_title_shape_is_he_plus_one_twist_and_short():
    """Counted on 7 ORV titles other channels ship: the ones that perform start
    with "He" and carry ONE twist ("He Read the Ending of the World... Then It
    Became REAL!"). Ours asked for four parts in 80 characters and got 113."""
    p = pc.build_claim_prompt({"premise": "P"}, "B")
    assert "45-70 characters" in p and "starting with He or She" in p and "ONE twist" in p
    assert "45-80 characters" not in p
