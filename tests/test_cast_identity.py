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
