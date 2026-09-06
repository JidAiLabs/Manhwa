"""Round-2 identity misattribution fix: deterministic cast-grounded FIGURE
resolution (tools/cast_identity.py) + the prep_qa actor_mismatch gate + the
writer payload `figures` lines.

The dominant round-2 residual (~6 findings) was the writer naming actors from
vibes: "the assassin draws his steel" over Prince Cheon's counter-draw
(g0008_p06), the dying prince's eye narrated as "an assassin's eye"
(g0019_p00), a departed assassin given the descendant's inner thoughts
(g0020_p01). Resolution is DETERMINISTIC keyword evidence against
manifest.cast.json — the failure mode being killed is model misattribution,
so no model is asked. Fixtures mirror the REAL Nano Machine ch1 cast."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import tools.cast_identity as ci
import tools.gemini_narrative_pass as g

_SPEC = importlib.util.spec_from_file_location(
    "prep_qa",
    Path(__file__).resolve().parent.parent / "tools" / "prep_qa.py",
)
pq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pq)  # type: ignore[union-attr]


# The real Nano ch1 cast_builder output, verbatim shapes (gemma4:26b run).
CAST = {"cast": [
    {"id": "our_protagonist", "canonical_name": "our protagonist",
     "role": "protagonist",
     "visual_description": ("Young man with dark/purple hair, often seen "
                            "wounded or bleeding from the mouth and face. "
                            "Wears light-colored, possibly white or grey "
                            "clothing that becomes bloodstained."),
     "is_protagonist": True,
     "aliases": ["Prince Cheon", "the kid", "descendant"]},
    {"id": "assassin_leader", "canonical_name": "unnamed assassin",
     "role": "antagonist",
     "visual_description": ("A figure wearing a dark brown or reddish hooded "
                            "cloak and a black face mask that only reveals "
                            "the eyes. Often seen wielding a sword."),
     "is_protagonist": False, "aliases": ["the assassin"]},
    {"id": "assassin_group", "canonical_name": "the assassins",
     "role": "antagonist",
     "visual_description": ("A group of figures wearing matching dark brown "
                            "hooded cloaks and black face masks, carrying "
                            "swords."),
     "is_protagonist": False, "aliases": ["the members", "the bastards"]},
    {"id": "mysterious_stranger", "canonical_name": "unnamed stranger",
     "role": "minor",
     "visual_description": ("A mysterious figure appearing suddenly with a "
                            "beam of white light. Wears modern-looking blue "
                            "and white clothing (a hoodie)."),
     "is_protagonist": False,
     "aliases": ["the strange guy", "that strange guy"]},
    {"id": "dying_ancestor", "canonical_name": "Ancestor", "role": "mentor",
     "visual_description": ("An elderly or dying figure mentioned in "
                            "dialogue as being on a deathbed."),
     "is_protagonist": False, "aliases": ["the ancestor"]},
]}

PROFILES = ci.cast_profiles(CAST)


def _names(figs):
    return [f["name"] for f in figs]


# --- resolution ------------------------------------------------------------

def test_g0008_counter_draw_subject_resolves_to_cheon():
    # the real misattribution: CHEON's counter-draw narrated as the assassin's
    u = {"subjects": ["a young man in a light robe with a blue sash, "
                      "bleeding from the mouth"],
         "action": "draws a hidden blade"}
    assert _names(ci.resolve_figures(u, PROFILES)) == ["our protagonist"]


def test_masked_hooded_cloak_resolves_to_the_assassin_faction():
    # leader vs group tie on near-identical appearance is ONE narrative
    # identity (both carry the 'assassin' name token) — never 'unknown'
    u = {"subjects": ["a masked figure in a dark hooded cloak drawing a "
                      "sword"]}
    names = _names(ci.resolve_figures(u, PROFILES))
    assert names and names[0] in ("unnamed assassin", "the assassins")


def test_hoodie_is_the_stranger_not_the_hooded_assassins():
    # 'armored figure' round-1/2 miss: hoodie is stranger-exclusive and must
    # not collapse into the assassins' hood/hooded
    u = {"subjects": ["a man in a blue and white hoodie with goggles"]}
    assert _names(ci.resolve_figures(u, PROFILES)) == ["unnamed stranger"]


# --- look-alikes (ORV Ep128, 2026-09-06) ------------------------------------
# The real cast_builder output, verbatim: two men, one description.
ORV_LOOKALIKE_CAST = {"cast": [
    {"id": "protagonist", "canonical_name": "our protagonist",
     "role": "protagonist", "is_protagonist": True, "aliases": ["Dokja Kim"],
     "visual_description": ("A young man with dark, messy hair and glasses, "
                            "wearing a black shirt over a white t-shirt and a "
                            "dark jacket.")},
    {"id": "michio_shoji", "canonical_name": "Michio Shoji", "role": "ally",
     "is_protagonist": False, "aliases": ["Michio"],
     "visual_description": ("A young man with dark hair and glasses, wearing "
                            "a black button-down shirt over a white t-shirt "
                            "and a dark jacket.")},
]}

# What the owner's series registry locks in (looks written POSITIVELY;
# exclusions go in `not`).
ORV_REGISTRY_CAST = {"cast": [
    {"id": "protagonist", "canonical_name": "our protagonist",
     "role": "protagonist", "is_protagonist": True, "aliases": ["Dokja Kim"],
     "visual_description": ("A young man with dark hair, wearing a long white "
                            "coat over a black shirt."),
     "not": ["glasses"]},
    {"id": "michio_shoji", "canonical_name": "Michio Shoji", "role": "ally",
     "is_protagonist": False, "aliases": ["Michio"],
     "visual_description": ("A young man with dark hair and glasses, wearing "
                            "a black button-down shirt over a white t-shirt "
                            "and a dark jacket.")},
]}


def test_two_identical_descriptions_never_resolve_on_one_incidental_word():
    # the chapter's actual verdicts: p000025 -> Dokja purely on "messy",
    # p000026 -> Michio purely on "button". One look = unknown, not a coin flip.
    prof = ci.cast_profiles(ORV_LOOKALIKE_CAST)
    name, ev = ci.resolve_name("a young man with messy dark hair and glasses", prof)
    assert name == "unknown" and "lookalike" in ev, (name, ev)
    name, _ = ci.resolve_name("a young man in the background with dark hair, "
                              "glasses, and a dark blue button-down shirt over "
                              "a white graphic t-shirt", prof)
    assert name == "unknown"


def test_registry_looks_and_not_traits_separate_the_pair():
    prof = ci.cast_profiles(ORV_REGISTRY_CAST)
    # p000024: the white coat IS Dokja (was "a stranger")
    name, _ = ci.resolve_name("a young man with dark hair wearing a long white "
                              "coat over a black shirt", prof)
    assert name == "our protagonist"
    # p000079: glasses + suit -> Michio; Dokja's `not: glasses` keeps him out
    name, ev = ci.resolve_name("a young man with dark hair, wearing glasses, a "
                               "black suit jacket, and a white shirt", prof)
    assert name == "Michio Shoji", (name, ev)
    # an explicit NAME still beats a forbidden trait (a penalty, not a veto)
    name, _ = ci.resolve_name("Dokja Kim, in glasses for once", prof)
    assert name == "our protagonist"


def test_ambiguous_subject_resolves_to_unknown_never_a_guess():
    # 'light' pairs with garments for BOTH protagonist and stranger — a tie
    # without a shared faction token must yield unknown (evidence preserved)
    u = {"subjects": ["a person in light robes"]}
    figs = ci.resolve_figures(u, PROFILES)
    assert _names(figs) == ["unknown"]
    assert "light robes" in figs[0]["evidence"]


def test_non_person_subjects_yield_no_figures():
    assert ci.resolve_figures({"subjects": ["a beast in the forest"]},
                              PROFILES) == []


def test_name_token_in_description_resolves_directly():
    u = {"subjects": [], "description": "Cheon staggers back, clutching the "
                                        "wound.", "action": ""}
    assert "our protagonist" in _names(ci.resolve_figures(u, PROFILES))


def test_resolution_is_deterministic():
    u = {"subjects": ["a masked figure in a dark hooded cloak",
                      "a wounded young man in white clothing"]}
    a = ci.resolve_figures(u, PROFILES)
    b = ci.resolve_figures(u, PROFILES)
    assert a == b


# --- round-2 review, class C: id-derived tokens are not identity evidence --

def test_group_of_villagers_never_hard_claims_the_assassins():
    # assassin_group's id ("assassin_group") must never leak "group" in as a
    # name token: an unrelated group of villagers must never hard-claim the
    # assassins (score 10 on the old bug).
    assert ci.resolve_figures({"subjects": ["a group of villagers"]},
                              PROFILES) == []


def test_masked_assassin_in_dark_tunic_still_resolves_after_id_fix():
    # real-shaped cast still resolves correctly once id-word noise is gone —
    # canonical_name ("unnamed assassin"/"the assassins") + aliases alone
    # carry enough evidence, same-faction tie included.
    names = _names(ci.resolve_figures(
        {"subjects": ["a masked assassin in a dark tunic"]}, PROFILES))
    assert names and names[0] in ("unnamed assassin", "the assassins")


# --- round-2 review, class C: generic words carry no appearance evidence ---

def test_generic_words_alone_never_hard_claim_a_profile():
    # _GENERIC_PERSON/_GENERIC_DESCRIPTOR are documented as NOT evidence
    # (module comment) but used to leak into the appearance/subject token
    # sets: 'a young man' (man+young=2.0) used to clear the resolution bar
    # and hard-claim the protagonist. Now scores 0 -> falls through to the
    # unknown _looks_person stamp, never a guess.
    assert _names(ci.resolve_figures({"subjects": ["a young man"]},
                                     PROFILES)) == ["unknown"]
    assert _names(ci.resolve_figures({"subjects": ["a bleeding man"]},
                                     PROFILES)) == ["unknown"]


def test_specific_appearance_words_still_clear_the_bar():
    # Isolated cast (protagonist + one unrelated, non-color-coded member) so
    # this checks SPECIFIC evidence clearing the bar in isolation; the
    # full-cast cross-character color disambiguation is already covered by
    # test_hoodie_is_the_stranger_not_the_hooded_assassins.
    cast = {"cast": [CAST["cast"][0], CAST["cast"][4]]}  # protagonist + Ancestor
    profiles = ci.cast_profiles(cast)
    u = {"subjects": ["a young man in a white robe with a blue sash"]}
    assert _names(ci.resolve_figures(u, profiles)) == ["our protagonist"]


# --- noun map + subject-position filter ------------------------------------

def test_noun_map_derives_from_cast_manifest_only():
    nm = ci.actor_noun_map(CAST)
    assert nm["assassin"] == {"unnamed assassin", "the assassins"}
    assert nm["prince"] == {"our protagonist"}
    assert nm["descendant"] == {"our protagonist"}
    assert nm["stranger"] == {"unnamed stranger"}
    # generic person-words are sanctioned neutral handles, never noun keys
    # ('our guy' is the protagonist's stand-in; 'guy' rides a stranger alias)
    assert "guy" not in nm
    # adjectives that ride cast names never become matchable nouns
    assert "mysterious" not in nm and "mysteriou" not in nm
    # id-slug structural words (assassin_group, assassin_leader) are not
    # identity nouns; "member"/"bastard" (real aliases) still are
    assert "group" not in nm and "leader" not in nm
    assert nm["member"] == {"the assassins"}


def test_subject_position_filter_skips_late_object_mentions():
    nm = ci.actor_noun_map(CAST)
    assert ci.subject_actor_nouns("The assassin draws his steel.", nm)
    assert ci.subject_actor_nouns(
        "An assassin's eye goes wide with dread.", nm)
    # off-panel/object mention deep in the sentence — deliberately not an
    # actor claim (the precision lever that keeps the gate a heal-target)
    assert not ci.subject_actor_nouns(
        "Their blades were meant for the ancestor all along.", nm)
    # second sentence of a segment line gets its own subject window
    assert ci.subject_actor_nouns(
        "The dust settles at last. The assassin lunges again.", nm)


def test_quoted_dialogue_does_not_leak_its_names_into_the_speakers_window():
    # a name inside what a character SAYS is who they talk about, not the
    # narrator's claim about this line's actor
    nm = ci.actor_noun_map(CAST)
    hits = ci.subject_actor_nouns(
        "The stranger sneers 'the prince dies tonight.'", nm)
    assert [n for n, _ in hits] == ["stranger"]


# --- prep_qa actor_mismatch ------------------------------------------------

_UNDERSTOOD = {"panels": [
    {"scene_file": "p000010.jpg",
     # same real g0008 phrasing as test_g0008_counter_draw_subject_resolves_
     # to_cheon below: "bleeding from the mouth" is the protagonist-exclusive
     # cue. Without it, "light robe" + "blue sash" alone ties the stranger's
     # "blue and white hoodie" on raw color overlap (correctly, post round-2
     # generic-token fix: neither profile still gets a free young+man bump) —
     # this fixture must carry the same real evidence the live panel did.
     "subjects": ["a young man in a light robe with a blue sash, bleeding "
                 "from the mouth"],
     "action": "draws a hidden blade", "description": ""},
    {"scene_file": "p000011.jpg",
     "subjects": ["a masked figure in a dark hooded cloak with a sword"],
     "action": "", "description": ""},
]}


def _beats(*lines_spans):
    return {"beats": [{"group_id": 8, "segments": [
        {"span": list(span), "line": line} for line, span in lines_spans]}]}


def test_actor_mismatch_fires_on_assassin_line_over_cheon_span():
    beats = _beats(("The assassin draws his steel.", ["p000010.jpg"]))
    flags = pq.actor_mismatch_flags(beats, _UNDERSTOOD, CAST)
    assert [f["code"] for f in flags] == ["actor_mismatch"]
    assert flags[0]["severity"] == pq.ERROR
    assert "assassin" in flags[0]["detail"]


def test_actor_mismatch_silent_when_the_actor_is_correct():
    beats = _beats(("The assassin closes in.", ["p000011.jpg"]),
                   ("The prince rips his hidden knife free.",
                    ["p000010.jpg"]))
    assert pq.actor_mismatch_flags(beats, _UNDERSTOOD, CAST) == []


def test_actor_mismatch_silent_on_correct_line_with_id_noun_collision():
    # round-2 review, class C: assassin_group's id used to leak "group" into
    # the noun map, so this CORRECT line over the protagonist's span used to
    # fire a false actor_mismatch (and the heal note would then rewrite a
    # correct line).
    beats = _beats(("A group of guards floods the hall.", ["p000010.jpg"]))
    assert pq.actor_mismatch_flags(beats, _UNDERSTOOD, CAST) == []


def test_actor_mismatch_silent_on_unresolved_spans_and_missing_cast():
    # zero resolved figures = no ground truth = no flag
    u = {"panels": [{"scene_file": "p000010.jpg",
                     "subjects": ["a person in light robes"]}]}
    beats = _beats(("The assassin draws his steel.", ["p000010.jpg"]))
    assert pq.actor_mismatch_flags(beats, u, CAST) == []
    assert pq.actor_mismatch_flags(beats, _UNDERSTOOD, {}) == []


def test_actor_mismatch_is_a_healable_code():
    import tools.narration_heal as nh
    assert "actor_mismatch" in nh.HEALABLE
    corr = nh.corrections_from_qa({"flags": [
        {"code": "actor_mismatch", "severity": "ERROR",
         "segment_id": "g0008", "detail": "line names 'assassin'"}]})
    assert 8 in corr and "figures" in corr[8]


# --- writer payload --------------------------------------------------------

def test_pack_group_payload_carries_figures_lines():
    figures = ci.resolve_figures_by_file(_UNDERSTOOD, CAST)
    u_by_file = {p["scene_file"]: p for p in _UNDERSTOOD["panels"]}
    group = {"shot_id": 8,
             "scene_files": ["p000010.jpg", "p000011.jpg"]}
    payload = g._pack_group_payload(group, {}, u_by_file,
                                    figures_by_file=figures)
    sig = {s["scene_file"]: s for s in payload["scenes_signals"]}
    assert sig["p000010.jpg"]["figures"] == ["our protagonist"]
    assert sig["p000011.jpg"]["figures"][0] in ("unnamed assassin",
                                                "the assassins")
    # no cast -> no figures key (byte-compatible payload)
    p2 = g._pack_group_payload(group, {}, u_by_file)
    assert all("figures" not in s for s in p2["scenes_signals"])


def test_writer_system_prompt_carries_the_figures_hard_rule():
    src = Path("tools/gemini_narrative_pass.py").read_text(encoding="utf-8")
    assert "FIGURES ARE GROUND TRUTH" in src
    assert "SYSTEM CARDS SPEAK THEIR TEXT" in src


# ---- 2026-07-16 wave: plurality bit ------------------------------------------

def test_subject_actor_nouns_ex_plurality_bit():
    nm = {"assassin": {"unnamed assassin"}, "prince": {"our protagonist"}}
    hits = ci.subject_actor_nouns_ex("His assassins close in fast.", nm)
    assert hits == [("assassin", {"unnamed assassin"}, True)]
    hits = ci.subject_actor_nouns_ex("The assassin closes in.", nm)
    assert hits == [("assassin", {"unnamed assassin"}, False)]
    # possessive is never plural
    hits = ci.subject_actor_nouns_ex("The assassin's blade gleams.", nm)
    assert hits == [("assassin", {"unnamed assassin"}, False)]
    # legacy wrapper keeps its two-tuple shape
    assert ci.subject_actor_nouns("His assassins close in.", nm) == [
        ("assassin", {"unnamed assassin"})]


def test_multitoken_name_collapses_to_one_member_in_the_wrapper():
    # A multi-token name — "Prince Cheon" here, "Cheon Yoo Jong" in nano ch6 —
    # is ONE person whose every word maps to the same member. The wrapper
    # collapses it to a single hit so the identity gates (actor_mismatch,
    # dead_actor) flag the actor once, not once per name word (ch6 reported a
    # single mismatch three times). _ex stays per-token for actor_count.
    nm = {"prince": {"our protagonist"}, "cheon": {"our protagonist"},
          "assassin": {"unnamed assassin"}}
    assert ci.subject_actor_nouns("Prince Cheon draws his steel.", nm) == [
        ("prince", {"our protagonist"})]
    # _ex is intentionally NOT collapsed — both name words survive
    assert [t for t, _m, _p in
            ci.subject_actor_nouns_ex("Prince Cheon draws his steel.", nm)] == [
        "prince", "cheon"]
    # 2026-08-18: an actor named in OBJECT position is NOT the line's actor
    # (this assertion used to expect the object hit — it encoded the defect
    # that flagged "The news hits the students" as the students acting).
    assert ci.subject_actor_nouns("Prince Cheon fights the assassin.", nm) == [
        ("prince", {"our protagonist"})]
    # two coordinated SUBJECTS are both actors
    assert ci.subject_actor_nouns("The prince and the assassin trade blows.", nm) == [
        ("prince", {"our protagonist"}), ("assassin", {"unnamed assassin"})]


# ---- 2026-07-16 wave: deterministic identity gate (writer-side) --------------

_NM = {"assassin": {"unnamed assassin"}, "prince": {"our protagonist"},
       "cheon": {"our protagonist"}, "protagonist": {"our protagonist"}}
_PROT = {"our protagonist"}


def _beat(line, span=("p1.jpg",)):
    return {"group_id": 1, "segments": [{"span": list(span), "line": line}],
            "narration": line}


def test_handle_gate_rewrites_protagonist_handle_over_helper_span():
    figs = {"p1.jpg": [{"name": "unnamed assassin", "evidence": "a masked figure"}]}
    b = _beat("Blue sparks crackle around our protagonist.")
    rw = g.enforce_actor_handles(b, figs, _NM, _PROT)
    assert rw and "protagonist handle" in rw[0]
    assert b["segments"][0]["line"] == (
        "Blue sparks crackle around the assassin.")


def test_handle_gate_keeps_handle_when_protagonist_resolved():
    figs = {"p1.jpg": [{"name": "our protagonist", "evidence": "purple hair"}]}
    b = _beat("Our guy staggers upright.")
    assert g.enforce_actor_handles(b, figs, _NM, _PROT) == []
    assert b["segments"][0]["line"] == "Our guy staggers upright."


def test_handle_gate_neutral_for_unknown_only_span():
    figs = {"p1.jpg": [{"name": "unknown",
                        "evidence": "a masked figure in a dark hooded cloak"}]}
    b = _beat("Our protagonist lunges forward.")
    rw = g.enforce_actor_handles(b, figs, _NM, _PROT)
    assert rw
    assert b["segments"][0]["line"].startswith("The masked figure lunges") or \
        b["segments"][0]["line"].startswith("the masked figure lunges")


def test_handle_gate_rewrites_wrong_actor_noun_and_keeps_possessive():
    figs = {"p1.jpg": [{"name": "our protagonist", "evidence": "purple hair"}]}
    b = _beat("The assassin's eyes burn with resolve.")
    rw = g.enforce_actor_handles(b, figs, _NM, _PROT)
    assert rw and "'assassin'" in rw[0]
    assert b["segments"][0]["line"] == "our protagonist's eyes burn with resolve."


def test_handle_gate_hands_off_ambiguous_and_ungrounded_spans():
    # multi-figure span: ambiguous -> untouched
    figs = {"p1.jpg": [{"name": "our protagonist", "evidence": "e"},
                       {"name": "unnamed assassin", "evidence": "e"}]}
    b = _beat("The stranger watches our protagonist bleed.")
    assert g.enforce_actor_handles(b, figs, _NM, _PROT) == []
    # zero figures: no ground truth -> untouched
    b2 = _beat("Our guy tumbles into the dark.")
    assert g.enforce_actor_handles(b2, {}, _NM, _PROT) == []
    # plural mismatch left for the actor_count heal net
    figs3 = {"p1.jpg": [{"name": "our protagonist", "evidence": "e"}]}
    b3 = _beat("His assassins tumble with him.")
    assert g.enforce_actor_handles(b3, figs3, _NM, _PROT) == []


# ---- 2026-07-16 wave: actor_count_mismatch (plural over single-figure span) --

def test_actor_count_fires_on_plural_over_single_figure_span():
    beats = _beats(("His assassins go tumbling down with him.",
                    ["p000010.jpg"]))
    flags = pq.actor_count_flags(beats, _UNDERSTOOD, CAST)
    assert [f["code"] for f in flags] == ["actor_count_mismatch"]
    assert flags[0]["severity"] == pq.ERROR
    assert "assassin" in flags[0]["detail"]


def test_actor_count_silent_on_singular_multi_figure_and_uncertain():
    # singular actor: fine
    beats = _beats(("The assassin closes in.", ["p000011.jpg"]))
    assert pq.actor_count_flags(beats, _UNDERSTOOD, CAST) == []
    # two-person span: plural is legal
    u2 = {"panels": [
        {"scene_file": "p000012.jpg",
         "subjects": ["a masked figure in a dark hooded cloak",
                      "a second masked figure in dark robes with a blade"],
         "action": "", "description": ""}]}
    beats = _beats(("The assassins close in.", ["p000012.jpg"]))
    assert pq.actor_count_flags(beats, u2, CAST) == []
    # uncertain panel contributes no ground truth (pu_v4)
    u3 = {"panels": [
        {"scene_file": "p000013.jpg", "uncertain": True,
         "subjects": ["a masked figure in a dark hooded cloak"],
         "action": "", "description": ""}]}
    beats = _beats(("The assassins close in.", ["p000013.jpg"]))
    assert pq.actor_count_flags(beats, u3, CAST) == []


def test_actor_count_is_a_healable_code_with_note():
    import tools.narration_heal as nh
    assert "actor_count_mismatch" in nh.HEALABLE
    note = nh._note_for("actor_count_mismatch", "line pluralizes 'assassin'")
    assert "ONE figure" in note and "companions" in note


# ---- 2026-07-16 job-48 review fixes ------------------------------------------

def test_appearance_words_are_not_identity_nouns():
    # 'the hooded leader' identifies by LEADER — any hooded figure must not
    # name-hit him (+10). Same for masked/cloaked/colors.
    cast = {"cast": [
        {"canonical_name": "the hooded leader", "aliases": [],
         "visual_description": "A man wearing a dark hooded cloak and a "
                               "black face mask"},
    ]}
    profs = ci.cast_profiles(cast)
    assert profs[0]["name_tokens"] == {"leader"}
    nm = ci.actor_noun_map(cast)
    assert "hooded" not in nm and "masked" not in nm and "leader" in nm


def test_color_clash_blocks_cross_color_garment_resolution():
    # job-48 g0014/15: the light-blue-hooded arrival resolved to the
    # dark-cloaked leader and the identity gate rewrote the protagonist's
    # transformation reveal to the villain.
    cast = {"cast": [
        {"canonical_name": "the hooded leader", "aliases": [],
         "visual_description": "A man wearing a dark hooded cloak and a "
                               "black face mask covering his nose"},
        {"canonical_name": "our protagonist", "aliases": ["Prince Cheon"],
         "visual_description": "A young man with messy dark hair in a "
                               "white martial arts tunic"},
    ]}
    u = {"subjects": ["a person wearing a light blue hooded jacket with "
                      "dark trim, surrounded by glowing blue electrical "
                      "energy"]}
    figs = ci.resolve_figures(u, ci.cast_profiles(cast))
    assert all(f["name"] == "unknown" for f in figs)   # neutral, never claimed
    # the REAL assassin-class subject still resolves via appearance evidence
    u2 = {"subjects": ["a masked figure in a dark hooded cloak with a "
                       "black face mask"]}
    figs2 = ci.resolve_figures(u2, ci.cast_profiles(cast))
    assert [f["name"] for f in figs2] == ["the hooded leader"]


def test_neutral_handle_never_ends_in_a_gerund():
    figs = [{"name": "unknown",
             "evidence": "a person wearing a light blue hooded jacket"}]
    assert g._neutral_from_evidence(figs) == "the person"
    figs = [{"name": "unknown",
             "evidence": "a masked figure in a dark hooded cloak"}]
    assert g._neutral_from_evidence(figs) == "the masked figure"
    assert g._neutral_from_evidence([]) == "the figure"


def test_echo_belt_ignores_non_adjacent_pairs():
    surviving = ["a.jpg", "b.jpg", "c.jpg"]
    segs = [{"span": ["a.jpg"], "line": "The eye narrows on the ridge."},
            {"span": ["b.jpg"], "line": "Steel whispers out of its sheath."},
            {"span": ["c.jpg"], "line": "The same eye snaps wide open."}]
    # non-adjacent echo (c echoes a) — glue skips it, the belt must too
    errs = g.validate_segments(segs, surviving, {},
                               echo_of={"c.jpg": "a.jpg"})
    assert not any("echo pair split" in e for e in errs)
    # adjacent split pair still trips the belt
    errs = g.validate_segments(segs, surviving, {},
                               echo_of={"b.jpg": "a.jpg"})
    assert any("echo pair split" in e for e in errs)


# ---- nano ch1 job 149 (2026-08-18): the PRINCE resolved to the master ---------
# cast_builder wrote "light-colored tunic" for the prince while every subject
# says "white robe"; hair color was a mere +1 token. 22 panels resolved to the
# late-appearing blue-haired master and the identity gate rewrote the prince's
# lines to "the mysterious master" (8 wrong-actor lines, one stutter).
_CH1_CAST = {"cast": [
    {"canonical_name": "our protagonist", "aliases": ["Prince Cheon", "the kid"],
     "is_protagonist": True,
     "visual_description": "A young man with messy, dark purple hair and intense eyes. "
                           "He wears a light-colored, long-sleeved tunic and trousers with "
                           "a purple sash. He is often seen wounded, covered in blood or dirt."},
    {"canonical_name": "the mysterious master", "aliases": [],
     "visual_description": "A character appearing suddenly with glowing light blue hair and "
                           "skin. He wears a blue and white hooded sweatshirt/tunic."},
    {"canonical_name": "an assassin member", "aliases": [],
     "visual_description": "A masked figure in a dark red hooded cloak and grey clothing."},
]}


def _resolve1(subject):
    figs = ci.resolve_figures({"subjects": [subject]}, ci.cast_profiles(_CH1_CAST))
    return figs[0]["name"] if figs else None


def test_ch1_prince_in_white_robe_with_purple_hair_is_the_protagonist():
    assert _resolve1("a young man with messy, long dark purple hair wearing a tattered "
                     "white robe with a blue collar and dark stains") == "our protagonist"
    assert _resolve1("a young man with dark, messy hair wearing a white and blue garment "
                     "with dark stains") == "our protagonist"          # white ~ light-colored
    assert _resolve1("a man with long, messy dark hair and a white robe with purple trim") == "our protagonist"


def test_ch1_master_still_resolves_and_bare_white_robe_stays_unknown():
    # ("glowing light blue hair and skin" is a TINT — see the tint test below)
    assert _resolve1("a young man in a blue hooded sweatshirt") == "the mysterious master"
    assert _resolve1("a young man in a white robe") == "unknown"        # ambiguous: no rewrite
    assert _resolve1("a masked figure in a dark red hooded cloak and grey clothing") == "an assassin member"


def test_hair_color_is_first_class_evidence_and_shades_modify_colors():
    # hair: 'dark hair' meets 'dark purple hair' (+3); 'light blue' is a shade
    # of blue for garments (no light/white class match), but hair keeps shade
    # words so "dark hair" still matches "dark purple hair"
    assert ci._hair_colors(ci._informative(ci._tokens("messy, long dark purple hair"))) == {"dark", "purple"}
    assert ci._color_garment_pairs(ci._informative(ci._tokens("a light blue hooded jacket"))) == {("blue", "garment")}
    assert ci._color_garment_pairs(ci._informative(ci._tokens("a white martial arts tunic"))) == {("light", "garment")}


def test_tinted_panel_withholds_color_evidence():
    # nano ch1 p000019: the purple-haired prince under a blue flashback tint,
    # described "glowing light blue hair and skin" -> must NOT become the
    # light-blue-haired master (skin is never blue: that's the light)
    assert _resolve1("a young person with glowing light blue hair and skin") == "unknown"
    assert _resolve1("a silhouetted figure with white hair") == "unknown"
    # an untinted description of the master still resolves
    assert _resolve1("a young man in a blue hooded sweatshirt") == "the mysterious master"
    # names still hit under a tint
    assert _resolve1("a glowing figure, Prince Cheon, with blue skin") == "our protagonist"



# ---- 2026-08-18: grammatical role decides the actor, not word position ------
# Audited false positives (nano ch1 g0018, ch6 g0002/g0003/g0013): a cast noun
# inside a partitive of-PP, a copular predicate, or a direct object landed in
# the first five words and was asserted to be the panel's actor.

_NM2 = {"assassin": {"an assassin member"}, "student": {"the students"},
        "bastard": {"Seob Meng"}, "prince": {"our protagonist"},
        "cheon": {"our protagonist"}}


def test_partitive_of_phrase_is_not_the_actor():
    # "One of the assassins ..." asserts ONE actor; the of-PP is not the subject
    assert ci.subject_actor_nouns_ex(
        "One of the assassins watches the chaos unfold, screaming out.", _NM2) == []
    assert ci.subject_actor_nouns_ex(
        "A pair of the assassins step back.", _NM2) == []


def test_predicate_nominal_after_copula_is_not_the_actor():
    assert ci.subject_actor_nouns_ex(
        "They were the bastards that tried to kill me whenever they had the chance.",
        _NM2) == []


def test_direct_object_inside_the_first_five_words_is_not_the_actor():
    # the two lines make the SAME claim; today only the shorter one flagged
    assert ci.subject_actor_nouns_ex("The news hits the students like a blow.", _NM2) == []
    assert ci.subject_actor_nouns_ex("The bad news hits the students like a blow.", _NM2) == []


def test_true_subject_actors_still_flag():
    hits = ci.subject_actor_nouns_ex("The assassins lunge forward together.", _NM2)
    assert [(t, p) for t, _m, p in hits] == [("assassin", True)]
    hits2 = ci.subject_actor_nouns_ex("Prince Cheon draws his blade.", _NM2)
    assert [t for t, _m, _p in hits2] == ["prince", "cheon"]
    # possessive subject inside the window still counts
    assert [t for t, _m, _p in ci.subject_actor_nouns_ex(
        "The assassin's eye narrows.", _NM2)] == ["assassin"]


def test_fronted_adjunct_and_subordinate_clause_subjects_survive():
    for line in ("Even now, the assassins press the attack.",
                 "Suddenly the assassins scatter.",
                 "As the assassins close in, he steadies himself."):
        assert [t for t, _m, _p in ci.subject_actor_nouns_ex(line, _NM2)] == ["assassin"], line


def test_group_member_names_and_collective_subject_counting():
    cast = {"cast": [{"canonical_name": "the students", "role": "group",
                      "visual_description": "a crowd of headbanded students"},
                     {"canonical_name": "our protagonist", "role": "protagonist",
                      "visual_description": "a young man with dark hair"}]}
    assert ci.group_member_names(cast) == {"the students"}
    # a crowd written as ONE subject string denotes many, not one
    assert ci.subject_person_count("a group of people with dark hair") > 1
    assert ci.subject_person_count("several students in headbands") > 1
    assert ci.subject_person_count("a young man with messy dark hair") == 1
    assert ci.subject_person_count("a bright moon over the mountains") == 0


def test_actor_mismatch_silent_when_the_name_is_printed_on_the_panel():
    # ORV Ep1 p000087: the protagonist reads his phone and the narration names
    # the commenter ('TLS123') off the screen. The panel draws one body, so the
    # gate fired — but healing it would force the WRONG actor onto the line. A
    # name the panel SPELLS OUT is grounded, not misattributed.
    beats = _beats(("The assassin draws his steel.", ["p000010.jpg"]))
    assert pq.actor_mismatch_flags(
        beats, _UNDERSTOOD, CAST,
        {"p000010.jpg": {"ocr_clean": "WANTED: THE ASSASSIN OF THE SOUTH"}}) == []
    # a stylized handle still resolves — the remainder is non-alphabetic
    assert pq.actor_mismatch_flags(
        beats, _UNDERSTOOD, CAST,
        {"p000010.jpg": {"ocr_clean": "assassin99: thanks for the chapter"}}) == []
    # control: a longer word that merely STARTS with the noun is not the name
    assert [f["code"] for f in pq.actor_mismatch_flags(
        beats, _UNDERSTOOD, CAST,
        {"p000010.jpg": {"ocr_clean": "the assassination attempt failed"}})] \
        == ["actor_mismatch"]
    # and with no OCR at all the gate is exactly as armed as before
    assert [f["code"] for f in pq.actor_mismatch_flags(
        beats, _UNDERSTOOD, CAST, {})] == ["actor_mismatch"]


def test_actor_mismatch_evidence_window_covers_the_folded_panels():
    # ORV Ep1 g0021 spans p000087+p000089 and voices the chat handle printed on
    # p000088 — a caption panel dropped from the shown frames whose words
    # folded into this segment. An evidence window built from the span alone is
    # blind to exactly the text the line is speaking.
    beats = _beats(("The assassin draws his steel.",
                    ["p000010.jpg", "p000012.jpg"]))
    assert pq.actor_mismatch_flags(beats, _UNDERSTOOD, CAST, {
        "p000010.jpg": {"ocr_clean": ""},
        "p000011.jpg": {"ocr_clean": "ASSASSIN99: THANK YOU"},   # folded
        "p000012.jpg": {"ocr_clean": ""}}) == []
    # A LONE-panel span has no range to widen, so the window has to reach into
    # the neighbours no span claims — those are the folded panels (ORV Ep1
    # g0021's third segment spans p000089 alone and voices p000090's chat text).
    lone = _beats(("The assassin draws his steel.", ["p000010.jpg"]))
    assert pq.actor_mismatch_flags(lone, _UNDERSTOOD, CAST, {
        "p000009.jpg": {"ocr_clean": ""},
        "p000010.jpg": {"ocr_clean": ""},
        "p000011.jpg": {"ocr_clean": "ASSASSIN99: THANK YOU"}}) == []
    # ...but the window still STOPS: a panel another segment OWNS is that
    # segment's evidence, never this one's.
    two = {"beats": [{"group_id": 8, "segments": [
        {"span": ["p000010.jpg"], "line": "The assassin draws his steel."},
        {"span": ["p000011.jpg"], "line": "The rain keeps falling."}]}]}
    assert [f["code"] for f in pq.actor_mismatch_flags(two, _UNDERSTOOD, CAST, {
        "p000010.jpg": {"ocr_clean": ""},
        "p000011.jpg": {"ocr_clean": "ASSASSIN99: THANK YOU"}})] \
        == ["actor_mismatch"]
    # ...and it is BOUNDED, so a chapter with a big dropped stretch cannot hand
    # one segment the rest of the chapter as evidence (_FOLD_REACH, measured).
    far = {f"p0000{n}.jpg": {"ocr_clean": ""} for n in range(10, 16)}
    far[f"p0000{10 + pq._FOLD_REACH + 1}.jpg"] = {
        "ocr_clean": "ASSASSIN99: THANK YOU"}
    assert [f["code"] for f in pq.actor_mismatch_flags(
        lone, _UNDERSTOOD, CAST, far)] == ["actor_mismatch"]
