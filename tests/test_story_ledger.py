"""Story-state ledger wave (2026-07-20): the root-cause fix for the nano ch1
narration-logic errors — inverted kill (visual misread vs dialogue truth),
dead leader re-labeled "the leader", two characters both "our guy", answered
questions re-asked.

Covers: pu_v5 structured actions (panel_understand), cast_identity faction-tie
+ excluded sets, cast_builder recurring figures, tools/story_ledger.py (the
keystone), identity_gate's ledger-aware cases, prep_qa dead_actor/role_stale,
narration_heal notes, and the punchup backstop's gate re-run. No model calls —
arbitration is injected."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import tools.cast_builder as cb
import tools.cast_identity as ci
import tools.identity_gate as ig
import tools.narration_heal as nh
import tools.narration_punchup as np_
import tools.panel_understand as pu
import tools.story_ledger as sl

_SPEC = importlib.util.spec_from_file_location(
    "prep_qa",
    Path(__file__).resolve().parent.parent / "tools" / "prep_qa.py",
)
pq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pq)  # type: ignore[union-attr]


# Mirrors the REAL Nano ch1 cast shapes (cast_builder labels the faction
# member 'antagonist', not 'group' — the plural canonical name is the signal).
CAST = {"cast": [
    {"id": "protagonist", "canonical_name": "our protagonist",
     "role": "protagonist", "is_protagonist": True,
     "aliases": ["Prince Cheon"],
     "visual_description": "A young man in a light robe with a blue sash"},
    {"id": "assassin_leader", "canonical_name": "unnamed assassin",
     "role": "antagonist", "is_protagonist": False,
     "aliases": ["the leader"],
     "visual_description": "A figure wearing a dark brown or reddish hooded "
                           "cloak and a black face mask that only reveals "
                           "the eyes. Often seen wielding a sword."},
    {"id": "assassin_group", "canonical_name": "the assassins",
     "role": "antagonist", "is_protagonist": False,
     "aliases": ["the members"],
     "visual_description": "A group of figures wearing matching dark brown "
                           "hooded cloaks and black face masks, carrying "
                           "swords"},
]}

UNDERSTOOD = {"panels": [
    {"scene_file": "p1.jpg", "panel_kind": "story",
     "subjects": ["a young man in a light robe with a blue sash",
                  "a masked figure in a dark hooded cloak"],
     "action": "a strike lands", "dialogue": "", "description": "",
     "actions": [{"actor": "a masked figure in a dark hooded cloak",
                  "verb": "strikes",
                  "target": "a young man in a light robe with a blue sash"}]},
    {"scene_file": "p2.jpg", "panel_kind": "story",
     "subjects": ["three masked figures in dark hooded cloaks"],
     "action": "the assassins recoil in shock", "description": "",
     "dialogue": "HOW DID A KID KILL ONE OF OUR MEMBERS?", "actions": []},
]}

GROUPS = {"shots": [{"shot_id": 8, "scene_files": ["p1.jpg"]},
                    {"shot_id": 9, "scene_files": ["p2.jpg"]}]}


def _arb_kill(digest):
    assert "HOW DID A KID" in digest          # dialogue reaches the arbiter
    return {
        "events": [
            {"type": "death", "scene_file": "p1.jpg",
             "subject": "unnamed assassin", "detail": "killed by the kid",
             "evidence_quote": "HOW DID A KID KILL ONE OF OUR MEMBERS?"},
            {"type": "question_answered", "scene_file": "p2.jpg",
             "subject": "our protagonist",
             "detail": "how did the kid kill? -> the hidden device",
             "evidence_quote": "HOW DID A KID KILL ONE OF OUR MEMBERS?"},
        ],
        "overrides": [
            {"scene_file": "p1.jpg", "actor": "our protagonist",
             "target": "unnamed assassin",
             "reason": "the assassins' own dialogue proves the kid killed"},
        ],
    }


def _ledger():
    return sl.build_ledger(UNDERSTOOD, GROUPS, CAST, arbitrate_fn=_arb_kill)


# ---- pu_v5: structured actions ----------------------------------------------

def test_norm_actions_coerces_and_caps():
    raw = [{"actor": "a man", "verb": "strikes", "target": "a beast"},
           {"actor": "", "verb": "collapses", "target": ""},   # dropped: no one
           "garbage",
           {"actor": "x", "verb": "", "target": "y"},          # dropped: no verb
           {"actor": "a", "verb": "v1", "target": ""},
           {"actor": "b", "verb": "v2", "target": ""},
           {"actor": "c", "verb": "v3", "target": ""},
           {"actor": "d", "verb": "v4", "target": ""}]
    out = pu._norm_actions(raw)
    assert [a["verb"] for a in out] == ["strikes", "v1", "v2", "v3"]  # cap 4
    assert pu._norm_actions(None) == [] and pu._norm_actions("x") == []


def test_assemble_record_carries_actions():
    rec = pu.assemble_record("p.jpg", {
        "description": "d", "action": "a", "intensity": "tense",
        "panel_kind": "story",
        "actions": [{"actor": "a man", "verb": "strikes",
                     "target": "unclear"}]})
    assert rec["actions"] == [{"actor": "a man", "verb": "strikes",
                               "target": "unclear"}]
    assert pu.assemble_record("p.jpg", None)["actions"] == []


def test_unclear_strike_trigger():
    base = {"strikes_or_weapons": "in_use",
            "actions": [{"actor": "unclear", "verb": "strikes",
                         "target": "a man"}]}
    assert pu.unclear_strike(base) is True
    assert pu.unclear_strike({**base, "strikes_or_weapons": "visible"}) is False
    assert pu.unclear_strike({"strikes_or_weapons": "in_use",
                              "actions": [{"actor": "a man", "verb": "hits",
                                           "target": "a wall"}]}) is False


def test_forced_choice_reask_fires_on_unclear_strike():
    first = {"description": "a strike", "action": "strike",
             "intensity": "intense", "panel_kind": "story",
             "strikes_or_weapons": "in_use",
             "actions": [{"actor": "unclear", "verb": "strikes",
                          "target": "a young man"}]}
    second = {**first,
              "actions": [{"actor": "a masked figure", "verb": "strikes",
                           "target": "a young man"}]}
    calls = []

    def call_fn(payload, image_path):
        calls.append(payload)
        return second if "forced_choice_notice" in payload else first

    out = pu.understand_panels([{"scene_file": "p.jpg"}], call_fn)
    assert len(calls) == 2 and "WHO strikes" in calls[1]["forced_choice_notice"]
    assert out[0]["actions"][0]["actor"] == "a masked figure"
    assert out[0].get("reask") is True


def test_reask_keeps_first_read_when_second_still_unclear():
    first = {"description": "a strike", "action": "strike",
             "intensity": "intense", "panel_kind": "story",
             "strikes_or_weapons": "in_use",
             "actions": [{"actor": "unclear", "verb": "strikes",
                          "target": "a young man"}]}

    def call_fn(payload, image_path):
        return dict(first)

    out = pu.understand_panels([{"scene_file": "p.jpg"}], call_fn)
    assert out[0]["actions"][0]["actor"] == "unclear"
    assert "reask" not in out[0]


# ---- cast_identity: faction tie + excluded ----------------------------------

def test_faction_tie_resolves_to_group_member_not_leader():
    # generic assassin subject: leader and group tie on appearance; the GROUP
    # (plural canonical name, whatever its role label) wins the tie
    u = {"subjects": ["a masked figure in a dark hooded cloak with a sword"]}
    figs = ci.resolve_figures(u, ci.cast_profiles(CAST))
    named = [f["name"] for f in figs if f["name"] != "unknown"]
    assert named and named[0] == "the assassins"


def test_excluded_entity_never_resolves():
    u = {"subjects": ["a masked figure in a dark hooded cloak with a sword"]}
    figs = ci.resolve_figures(
        u, ci.cast_profiles(CAST),
        excluded={"unnamed assassin", "the assassins"})
    assert all(f["name"] == "unknown" for f in figs)


def test_resolve_figures_by_file_threads_excluded():
    out = ci.resolve_figures_by_file(
        UNDERSTOOD, CAST,
        excluded_by_file={"p2.jpg": {"the assassins", "unnamed assassin"}})
    assert any(f["name"] == "our protagonist" for f in out["p1.jpg"])
    assert all(f["name"] == "unknown" for f in out["p2.jpg"])


# ---- cast_builder: recurring figures ----------------------------------------

def test_recurring_figures_clusters_across_panels():
    u = {"panels": [
        {"scene_file": f"p{i}.jpg",
         "subjects": ["a man in a blue sash robe", "a plain wall"]}
        for i in range(4)]}
    figs = cb.recurring_figures(u)
    assert len(figs) == 1 and "blue sash" in figs[0] and "4 panels" in figs[0]
    # below the panel floor -> nothing
    u2 = {"panels": u["panels"][:2]}
    assert cb.recurring_figures(u2) == []


# ---- story_ledger: the keystone ---------------------------------------------

def test_ledger_flips_inverted_action_via_dialogue():
    led = _ledger()
    pa = [a for a in led["panel_actions"] if a["scene_file"] == "p1.jpg"][0]
    assert pa["actor"] == "our protagonist"
    assert pa["target"] == "unnamed assassin"
    assert pa["evidence"] == "dialogue_arbitrated"
    assert led["stats"]["overrides_applied"] == 1


def test_ledger_death_propagates_forward_only():
    led = _ledger()
    g8, g9 = led["beat_facts"]["g0008"], led["beat_facts"]["g0009"]
    assert g8["dead_by_now"] == [] and g8["banned_handles"] == []
    assert g9["dead_by_now"] == ["unnamed assassin"]
    assert "the leader" in g9["banned_handles"]
    # 'the assassins' survive, so faction nouns are NOT banned
    assert not any("assassin" in h for h in g9["banned_handles"])
    assert g9["answered"] == []          # answered in g9 itself, not before


def test_ledger_rejects_unanchored_events():
    def arb(_):
        return {"events": [
            {"type": "death", "scene_file": "p1.jpg", "subject": "a dragon",
             "detail": "", "evidence_quote": ""},           # unknown entity
            {"type": "death", "scene_file": "zz.jpg",
             "subject": "unnamed assassin", "detail": "",
             "evidence_quote": ""},                          # unknown panel
        ], "overrides": []}
    led = sl.build_ledger(UNDERSTOOD, GROUPS, CAST, arbitrate_fn=arb)
    assert led["events"] == []


def test_ledger_fail_soft_on_arbitration_crash():
    def arb(_):
        raise RuntimeError("model exploded")
    logged = []
    led = sl.build_ledger(UNDERSTOOD, GROUPS, CAST, arbitrate_fn=arb,
                          log=logged.append)
    assert led["events"] == [] and led["stats"]["overrides_applied"] == 0
    assert led["panel_actions"]                  # visual ledger still there
    assert any("ARBITRATION FAILED" in m for m in logged)


def test_ledger_derived_entity_for_uncast_recurring_figure():
    u = {"panels": [
        {"scene_file": f"p{i}.jpg",
         "subjects": ["a helper in a green cloak with a scar"],
         "dialogue": "", "action": "", "actions": []}
        for i in range(3)]}
    led = sl.build_ledger(u, {"shots": []}, CAST, arbitrate_fn=None)
    derived = [e for e in led["entities"] if e["source"] == "derived"]
    assert len(derived) == 1
    assert derived[0]["canonical_name"].startswith("the ")


def test_dead_sets_by_file_strictly_after_death_panel():
    led = _ledger()
    dead = sl.dead_sets_by_file(led, ["p1.jpg", "p2.jpg"])
    assert "p1.jpg" not in dead                  # dies AT p1, not before
    assert dead["p2.jpg"] == {"unnamed assassin"}


# ---- identity_gate: ledger-aware cases --------------------------------------

_NM = ci.actor_noun_map(CAST)
_PROT = ig.protagonist_names(CAST)


def _beat(line, span=("p1.jpg",), gid=8):
    return {"group_id": gid,
            "segments": [{"span": list(span), "line": line}],
            "narration": line}


def test_gate_ledger_breaks_multi_figure_tie():
    # hero + faction both resolved on the span (multi-figure -> the old gate
    # bailed with ""); the noun 'leader' is disjoint from the span's figures
    # and the ledger's single arbitrated actor breaks the tie
    figs = {"p1.jpg": [{"name": "our protagonist", "evidence": "blue sash"},
                       {"name": "the assassins", "evidence": "cloaks"}]}
    led = _ledger()
    b = _beat("The leader finishes the job, leaving our guy broken.")
    rw = ig.enforce_actor_handles(b, figs, _NM, _PROT, ledger=led)
    assert rw and "'leader'" in rw[0]
    assert b["segments"][0]["line"].startswith("our protagonist finishes")


def test_gate_without_ledger_is_byte_identical_hands_off():
    figs = {"p1.jpg": [{"name": "our protagonist", "evidence": "e"},
                       {"name": "the assassins", "evidence": "e"}]}
    b = _beat("The leader finishes the job.")
    assert ig.enforce_actor_handles(b, figs, _NM, _PROT) == []
    assert b["segments"][0]["line"] == "The leader finishes the job."


def test_gate_rewrites_dead_actor_even_when_figures_stale():
    # stale figure resolution still lists the (dead) leader on a g9 span;
    # dead_by_now overrules the members&named skip
    led = _ledger()
    figs = {"p2.jpg": [{"name": "unnamed assassin", "evidence": "masked"},
                       {"name": "the assassins", "evidence": "cloaks"}]}
    led2 = dict(led)
    led2["panel_actions"] = led["panel_actions"] + [
        {"scene_file": "p2.jpg", "actor": "the assassins", "verb": "recoil",
         "target": "", "evidence": "visual", "raw": {}}]
    b = _beat("The leader looms over the boy.", span=("p2.jpg",), gid=9)
    rw = ig.enforce_actor_handles(b, figs, _NM, _PROT, ledger=led2)
    assert rw and "dead actor" in rw[0]
    assert "the assassins" in b["segments"][0]["line"].lower()


def test_gate_zero_figure_span_repoints_protagonist_handle():
    led = _ledger()
    led2 = dict(led)
    # a beat whose facts place ONLY the assassins present, one clear actor
    led2["beat_facts"] = {"g0009": {
        "present": ["the assassins"], "actions": [], "dead_by_now": [],
        "banned_handles": [], "answered": []}}
    led2["panel_actions"] = [
        {"scene_file": "p2.jpg", "actor": "the assassins", "verb": "recoil",
         "target": "", "evidence": "visual", "raw": {}}]
    b = _beat("Our guy recoils in disbelief.", span=("p2.jpg",), gid=9)
    rw = ig.enforce_actor_handles(b, {}, _NM, _PROT, ledger=led2)
    assert rw and "(ledger)" in rw[0]
    assert b["segments"][0]["line"] == "the assassins recoils in disbelief."


# ---- prep_qa: dead_actor / role_stale ---------------------------------------

def _beats(*lines_spans_gids):
    return {"beats": [
        {"group_id": gid, "segments": [{"span": list(span), "line": line}],
         "narration": line}
        for line, span, gid in lines_spans_gids]}


def test_dead_actor_flag_fires_with_evidence_quote():
    led = _ledger()
    beats = _beats(("The leader lunges again.", ["p2.jpg"], 9))
    flags = pq.ledger_contradiction_flags(beats, led, CAST)
    assert [f["code"] for f in flags] == ["dead_actor"]
    assert flags[0]["severity"] == pq.ERROR
    assert "HOW DID A KID" in flags[0]["detail"]


def test_role_stale_fires_on_banned_handle_not_double_flagged():
    led = _ledger()
    # 'the leader' as a LATE (object-position) mention: dead_actor's
    # subject-position net skips it, role_stale still catches the handle
    beats = _beats(
        ("They all stare at what is left of the leader now.", ["p2.jpg"], 9))
    flags = pq.ledger_contradiction_flags(beats, led, CAST)
    assert [f["code"] for f in flags] == ["role_stale"]
    # subject-position 'the leader' -> dead_actor ONLY (no double flag)
    beats2 = _beats(("The leader stands back up.", ["p2.jpg"], 9))
    codes = [f["code"] for f in
             pq.ledger_contradiction_flags(beats2, led, CAST)]
    assert codes == ["dead_actor"]


def test_ledger_flags_silent_without_ledger_or_before_death():
    led = _ledger()
    assert pq.ledger_contradiction_flags(
        _beats(("The leader lunges.", ["p1.jpg"], 8)), led, CAST) == []
    assert pq.ledger_contradiction_flags(
        _beats(("The leader lunges.", ["p2.jpg"], 9)), {}, CAST) == []
    assert pq.ledger_contradiction_flags(
        _beats(("The leader lunges.", ["p2.jpg"], 9)), None, CAST) == []


# ---- narration_heal + worker posture ----------------------------------------

def test_new_codes_are_healable_with_fact_notes():
    assert "dead_actor" in nh.HEALABLE and "role_stale" in nh.HEALABLE
    note = nh._note_for(
        "dead_actor",
        "line has 'leader' acting but ['the hooded leader'] are dead by this "
        "beat (killed at p2.jpg: \"HOW DID A KID KILL ONE OF OUR MEMBERS?\")")
    assert "ALREADY DEAD" in note and "HOW DID A KID" in note
    assert "title" in nh._note_for("role_stale", "line says 'the leader'")


def test_worker_blocks_dead_actor_but_not_role_stale():
    import studio.worker as w
    assert "dead_actor" in w._CRITICAL_QA_CODES
    assert "role_stale" not in w._CRITICAL_QA_CODES
    assert "dead_actor" not in w._WRITER_ARBITRATED_CODES


# ---- punchup backstop: gate re-runs after the persona pass ------------------

def test_punchup_backstop_reverts_reintroduced_our_guy():
    # persona pass re-attached 'our guy' to a helper-only span; the backstop's
    # gate re-run (the extraction's whole point) rewrites it back
    understood_by_file = {
        "p1.jpg": {"scene_file": "p1.jpg",
                   "subjects": ["a masked figure in a dark hooded cloak "
                                "with a sword"]}}
    out = {"beats": [
        {"group_id": 8,
         "segments": [{"span": ["p1.jpg"],
                       "line": "Our guy slips through the smoke."}],
         "narration": "Our guy slips through the smoke."}]}
    stats = np_.apply_post_punchup_backstop(out, CAST, {}, understood_by_file)
    assert stats["actor_handles_rewritten"] == 1
    assert "our guy" not in out["beats"][0]["segments"][0]["line"].lower()


def test_punchup_backstop_ledger_threads_dead_exclusion():
    led = _ledger()
    understood_by_file = {p["scene_file"]: p for p in UNDERSTOOD["panels"]}
    out = {"beats": [
        {"group_id": 9,
         "segments": [{"span": ["p2.jpg"],
                       "line": "The leader stands back up."}],
         "narration": "The leader stands back up."}]}
    stats = np_.apply_post_punchup_backstop(
        out, CAST, {}, understood_by_file, ledger=led)
    assert stats["actor_handles_rewritten"] >= 1
    assert "leader" not in out["beats"][0]["segments"][0]["line"].lower()
