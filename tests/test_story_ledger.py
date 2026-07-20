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

import tools.story_pass as sp
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
    if "HOW DID A KID" not in digest:         # window without the evidence
        return {"events": [], "overrides": []}
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


# ---- pu_v6: direction verification on a CONFIDENT two-person strike ---------

def test_contested_strike_targets_two_person_violence_only():
    two = {"strikes_or_weapons": "in_use", "subjects": [
        "a masked figure in a dark hooded cloak",
        "a young man in a light robe with a blue sash"]}
    assert pu.contested_strike(two) is True
    # one person: no direction to get backwards
    assert pu.contested_strike({**two, "subjects": two["subjects"][:1]}) is False
    # no strike landing: not the risky class
    assert pu.contested_strike({**two, "strikes_or_weapons": "visible"}) is False
    # props are not people
    assert pu.contested_strike({"strikes_or_weapons": "in_use", "subjects": [
        "a sword lying on the ground", "a stone wall"]}) is False


def test_direction_reask_fires_on_confident_inversion_and_flips_it():
    """The nano ch1 regression in miniature: pu_v5's re-ask never fired
    because the model was CONFIDENT, so the inverted actor shipped."""
    inverted = {"actor": "a masked figure in a dark hooded cloak",
                "verb": "lunges at",
                "target": "a young man in a light robe with a blue sash"}
    corrected = {"actor": "a young man in a light robe with a blue sash",
                 "verb": "stabs", "target":
                 "a masked figure in a dark hooded cloak"}
    first = {"description": "two figures collide", "action": "a strike",
             "intensity": "intense", "panel_kind": "story",
             "strikes_or_weapons": "in_use",
             "subjects": [inverted["actor"], inverted["target"]],
             "actions": [inverted]}
    seen = []

    def call_fn(payload, image_path):
        seen.append(payload)
        if "forced_choice_notice" in payload:
            return {**first, "actions": [corrected]}
        return first

    items = [{"scene_file": "p33.jpg", "ocr_clean": "?!"},
             {"scene_file": "p37.jpg",
              "ocr_clean": "HOW DID A KID KILL ONE OF OUR MEMBERS?"}]
    out = pu.understand_panels(items, call_fn)
    rec = out[0]
    assert rec["actions"] == [corrected]        # direction flipped
    assert rec.get("direction_reask") is True
    # the neighbouring dialogue — the evidence that settles it — was attached
    notice_payload = [p for p in seen if "forced_choice_notice" in p][0]
    assert any("HOW DID A KID" in d
               for d in notice_payload.get("nearby_dialogue") or [])
    # description/subjects are NOT taken from the second read (blast radius)
    assert rec["description"] == "two figures collide"


def test_direction_reask_keeps_first_actions_when_second_wont_commit():
    first = {"description": "d", "action": "a", "intensity": "intense",
             "panel_kind": "story", "strikes_or_weapons": "in_use",
             "subjects": ["a masked figure in a dark hooded cloak",
                          "a young man in a light robe with a blue sash"],
             "actions": [{"actor": "a masked figure in a dark hooded cloak",
                          "verb": "lunges at", "target": "a young man in a "
                          "light robe with a blue sash"}]}

    def call_fn(payload, image_path):
        if "forced_choice_notice" in payload:
            return {**first, "actions": [
                {"actor": "unclear", "verb": "strikes", "target": "unclear"}]}
        return first

    out = pu.understand_panels([{"scene_file": "p.jpg"}], call_fn)
    assert out[0]["actions"] == first["actions"]
    assert "direction_reask" not in out[0]


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


def test_generic_faction_subject_resolves_to_the_least_specific_member():
    """THE bug behind 'unclear lunges at our protagonist': a generic hooded
    subject ties between 'unnamed assassin' and 'the assassins', and the old
    ledger copy erased it to 'unclear'. Evidence that fits a plain member and
    a titled one equally must never claim the titled one."""
    profs = ci.cast_profiles(CAST)
    name, _ev = ci.resolve_name(
        "a masked figure in a dark hooded cloak with a sword", profs)
    assert name == "the assassins"
    # and the ledger's wrapper agrees — one oracle, no drift
    assert sl._resolve_name(
        "a masked figure in a dark hooded cloak with a sword", profs) == name
    # a two-member cast with NO plural entry still avoids the titled one
    cast2 = {"cast": [
        {"canonical_name": "the masked assassin", "role": "antagonist",
         "aliases": [], "visual_description": "a figure in a dark hooded "
                                              "cloak with a black mask"},
        {"canonical_name": "the assassin leader", "role": "antagonist",
         "aliases": [], "visual_description": "a figure in a dark hooded "
                                              "cloak with a black mask"}]}
    got, _ = ci.resolve_name("a figure in a dark hooded cloak with a black "
                             "mask", ci.cast_profiles(cast2))
    assert got == "the masked assassin"


def test_ledger_windows_the_arbitration_and_dedupes():
    """One whole-chapter call returned three trivial events and missed a kill
    stated outright in the dialogue; windows keep each call anchorable."""
    panels = [{"scene_file": f"p{i:03d}.jpg", "panel_kind": "story",
               "subjects": [], "action": "", "dialogue": "", "actions": []}
              for i in range(40)]
    seen = []

    def arb(digest):
        seen.append(digest)
        return {"events": [{"type": "death", "scene_file": "p005.jpg",
                            "subject": "our protagonist", "detail": "d",
                            "evidence_quote": "q"}], "overrides": []}

    led = sl.build_ledger({"panels": panels}, {"shots": []}, CAST,
                          arbitrate_fn=arb)
    assert len(seen) >= 3                    # windowed, not one giant call
    assert all(len(d) < 6000 for d in seen)  # each call stays small
    assert len(led["events"]) == 1           # every window claimed it: deduped


def test_ledger_survives_one_bad_window():
    panels = [{"scene_file": f"p{i:03d}.jpg", "panel_kind": "story",
               "subjects": [], "action": "", "dialogue": "", "actions": []}
              for i in range(40)]
    calls = {"n": 0}

    def arb(digest):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("window 2 exploded")
        return {"events": [{"type": "reveal", "scene_file": "p001.jpg",
                            "subject": "our protagonist", "detail": "d",
                            "evidence_quote": "q"}], "overrides": []}

    logged = []
    led = sl.build_ledger({"panels": panels}, {"shots": []}, CAST,
                          arbitrate_fn=arb, log=logged.append)
    assert led["events"]                     # other windows' facts survive
    assert any("window 2" in m or "arbitration FAILED" in m for m in logged)


def test_event_subject_resolves_through_the_oracle_and_rejects_are_logged():
    logged = []

    def arb(digest):
        return {"events": [
            # natural phrasing, not a verbatim cast name -> oracle resolves it
            {"type": "death", "scene_file": "p1.jpg",
             "subject": "a masked figure in a dark hooded cloak with a sword",
             "detail": "killed by the kid", "evidence_quote": "q"},
            # genuinely unknown -> rejected, and SAID SO (never silent)
            {"type": "death", "scene_file": "p1.jpg", "subject": "a dragon",
             "detail": "", "evidence_quote": ""},
        ], "overrides": []}

    led = sl.build_ledger(UNDERSTOOD, GROUPS, CAST, arbitrate_fn=arb,
                          log=logged.append)
    assert [e["subject"] for e in led["events"]] == ["the assassins"]
    assert any("dragon" in m and "matches no entity" in m for m in logged)


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


# ---- story_pass: the whole-chapter read -------------------------------------

def test_transcript_keeps_reading_order_and_drops_only_chrome():
    vision = {"items": [
        {"scene_file": "p1.jpg", "ocr_clean": "FIRST LINE"},
        {"scene_file": "p2.jpg", "ocr_clean": ""},          # wordless: KEPT
        {"scene_file": "p3.jpg", "ocr_clean": "subscribe!"},  # chrome: dropped
        {"scene_file": "p4.jpg", "ocr_clean": "LAST LINE"},
    ]}
    understood = {"panels": [
        {"scene_file": "p1.jpg", "panel_kind": "story"},
        {"scene_file": "p2.jpg", "panel_kind": "story"},
        {"scene_file": "p3.jpg", "panel_kind": "chrome"},
        {"scene_file": "p4.jpg", "panel_kind": "story"},
    ]}
    t = sp.build_transcript(vision, understood)
    lines = t.splitlines()
    assert [line.split()[0] for line in lines] == ["p1.jpg", "p2.jpg", "p4.jpg"]
    assert "(no text)" in lines[1]          # wordless panels hold their place
    assert "subscribe" not in t
    # without an understanding the transcript still works (nothing filtered)
    assert len(sp.build_transcript(vision).splitlines()) == 4


def test_build_story_normalizes_and_refuses_an_empty_answer():
    out = sp.build_story("t", lambda _p: {
        "synopsis": " The prince kills an assassin. ",
        "cast": [{"name": " Prince Cheon ", "role": "protagonist",
                  "fate": "wounded"}, {"name": "", "role": "x", "fate": "y"}],
        "events": [{"panels": "p1-p2", "actor": "Prince Cheon",
                    "does": "kills an assassin", "target": "the assassins",
                    "evidence": "q"},
                   {"panels": "p9", "actor": "x", "does": "", "target": ""}]})
    assert out["synopsis"] == "The prince kills an assassin."
    assert [c["name"] for c in out["cast"]] == ["Prince Cheon"]   # blank dropped
    assert len(out["events"]) == 1                                # no-verb dropped
    assert out["prompt_version"] == sp.PROMPT_VERSION
    for bad in ({"synopsis": "", "cast": [], "events": []}, "not a dict"):
        try:
            sp.build_story("t", lambda _p: bad)
            assert False, "should have raised"
        except (ValueError, Exception):
            pass


# ---- story -> ledger derivation (no model call) ------------------------------

_ORDERED = [f"p{i:06d}.jpg" for i in range(30, 42)]
_U12 = {"panels": [{"scene_file": f, "subjects": [], "actions": [],
                    "dialogue": ""} for f in _ORDERED]}


def test_panel_range_parses_ids_with_or_without_extension():
    assert sp is not None
    assert sl._panel_range("p000036-p000037", _ORDERED) == [
        "p000036.jpg", "p000037.jpg"]
    assert sl._panel_range("p000036.jpg", _ORDERED) == ["p000036.jpg"]
    assert sl._panel_range("p000030-p000032", _ORDERED) == [
        "p000030.jpg", "p000031.jpg", "p000032.jpg"]
    assert sl._panel_range("nonsense", _ORDERED) == []
    assert sl._panel_range("p999999", _ORDERED) == []


def test_story_events_become_deaths_and_direction_without_a_model_call():
    ents = sl.build_entities(_U12, CAST)
    profs = sl.entity_profiles(ents)
    story = {"cast": [{"name": "the assassins", "role": "antagonist",
                       "fate": "one member killed by Prince Cheon"}],
             "events": [{"panels": "p000036-p000037",
                         "actor": "Prince Cheon",
                         "does": "kills an assassin with a hidden blade",
                         "target": "the assassins",
                         "evidence": "HOW DID A KID KILL ONE OF OUR MEMBERS?"}]}
    ev, ov = sl.facts_from_chapter_story(story, ents, _U12, profs, log=lambda _m: None)
    assert len(ev) == 1 and ev[0]["type"] == "death"
    assert ev[0]["subject"] == "the assassins"
    assert ev[0]["scene_file"] == "p000037.jpg"     # anchored at span END
    assert len(ov) == 2                              # both panels attributed
    assert ov[0]["actor"] == "our protagonist"
    assert ov[0]["target"] == "the assassins"


def test_non_fatal_story_event_sets_direction_but_no_death():
    ents = sl.build_entities(_U12, CAST)
    profs = sl.entity_profiles(ents)
    story = {"cast": [], "events": [
        {"panels": "p000033", "actor": "the assassins", "does": "lunges at",
         "target": "Prince Cheon", "evidence": "?!"}]}
    ev, ov = sl.facts_from_chapter_story(story, ents, _U12, profs, log=lambda _m: None)
    assert ev == []
    assert len(ov) == 1 and ov[0]["actor"] == "the assassins"


def test_unanchorable_event_and_unpropagated_death_are_logged_not_silent():
    ents = sl.build_entities(_U12, CAST)
    profs = sl.entity_profiles(ents)
    logs = []
    story = {"cast": [{"name": "the assassins", "role": "antagonist",
                       "fate": "killed"}],
             "events": [{"panels": "chapter 4", "actor": "Prince Cheon",
                         "does": "kills someone", "target": "the assassins",
                         "evidence": "q"}]}
    ev, ov = sl.facts_from_chapter_story(story, ents, _U12, profs, log=logs.append)
    assert ev == [] and ov == []
    assert any("not anchorable" in m for m in logs)
    assert any("no event anchors that death" in m for m in logs)


def test_build_ledger_prefers_the_chapter_story_and_skips_arbitration():
    calls = []

    def arb(_digest):
        calls.append(1)
        return {"events": [], "overrides": []}

    story = {"cast": [], "events": [
        {"panels": "p000036-p000037", "actor": "Prince Cheon",
         "does": "kills an assassin", "target": "the assassins",
         "evidence": "HOW DID A KID KILL ONE OF OUR MEMBERS?"}]}
    led = sl.build_ledger(_U12, {"shots": [
        {"shot_id": 9, "scene_files": ["p000036.jpg", "p000037.jpg"]},
        {"shot_id": 10, "scene_files": ["p000038.jpg"]}]},
        CAST, arbitrate_fn=arb, chapter_story=story, log=lambda _m: None)
    assert calls == []                       # ZERO model calls
    assert [e["subject"] for e in led["events"]] == ["the assassins"]
    # and the death propagates to the LATER beat, not its own
    assert led["beat_facts"]["g0009"]["dead_by_now"] == []
    assert led["beat_facts"]["g0010"]["dead_by_now"] == ["the assassins"]


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
