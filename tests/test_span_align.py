"""span_align — the ONE-PANEL OFFSET fix (2026-07-06 Nano ch1 vision review).

The dominant defect class (~10/27 findings) was a narration line leading or
lagging its span by one panel in action runs. Fixtures below are shaped from
the REAL reviewed segments (spans + lines verbatim from the run's
manifest.beats.json); understanding records are synthetic-but-faithful (the
run's understood.json lives on the production host) reproducing each finding's
shape:

  * g0002_p01 — the impact line voiced over the pre-impact panel (+1 lead,
    cascades through a singleton into a multi-panel absorber);
  * g0003_p05 — the "eyes widen" line one panel after the eye close-up
    (-1 lag, absorbed by the multi-panel left neighbor);
  * g0008_p06 — a line re-describing the previous segment's drag while the
    art shows the counter-stab (-1 lag with a singleton cascade).

Anti-overcorrection is half the contract: correctly-aligned runs, sub-margin
preferences, neighbor damage, system walls, and the cap must all refuse to
shift.
"""
from __future__ import annotations

import copy

import tools.span_align as sa
import tools.gemini_narrative_pass as gnp
import tools.prep_qa as pq


def _u(desc="", action="", subjects=(), sfx="", impact=False, kind="story"):
    return {"description": desc, "action": action,
            "subjects": list(subjects), "sfx_text": sfx,
            "impact_sfx": {"present": bool(impact)}, "panel_kind": kind}


# ---------------------------------------------------------------------------
# g0002 shape: +1 lead — "Then, a sudden impact hits him like a freight
# train..." voiced over the still-peaceful walking panel p000002 while the
# impact is drawn on p000003 (the NEXT segment's panel).
# ---------------------------------------------------------------------------
G2_FILES = ["p000002.jpg", "p000003.jpg", "p000004.jpg", "p000005.jpg"]
G2_KINDS = {f: "story" for f in G2_FILES}
G2_U = {
    "p000002.jpg": _u("Prince Cheon walks alone beneath moonlit trees",
                      "walking calmly through the forest",
                      ["Prince Cheon", "forest", "moonlight"]),
    "p000003.jpg": _u("a violent impact slams into him, blood spraying "
                      "across the leaves",
                      "an unseen strike slams into his shoulder",
                      ["Prince Cheon", "blood", "leaves"], impact=True),
    "p000004.jpg": _u("he tumbles down the cliff, blood trailing through "
                      "the leaves",
                      "falling down the jagged cliffside, blood trailing",
                      ["Prince Cheon", "cliff", "blood"]),
    "p000005.jpg": _u("he screams as he plummets down the cliffside into "
                      "the dark ravine",
                      "screaming while falling",
                      ["Prince Cheon", "cliffside"]),
}
G2_SEGS = [
    {"span": ["p000002.jpg"],
     "line": "Then, a sudden impact hits him like a freight train, "
             "completely wrecking his footing."},
    {"span": ["p000003.jpg"],
     "line": "Blood is already painting the leaves in a total blur of "
             "chaos."},
    {"span": ["p000004.jpg", "p000005.jpg"],
     "line": "Gravity is a bully, sending him screaming down the jagged "
             "cliffside."},
]


def test_g0002_lead_shifts_plus_one_with_singleton_cascade():
    aligned, logs = sa.span_align_pass(G2_SEGS, G2_FILES, G2_KINDS, G2_U)
    assert [s["span"] for s in aligned] == [
        ["p000002.jpg", "p000003.jpg"],   # impact line now covers the impact
        ["p000004.jpg"],                  # blood line rides the bloody fall
        ["p000005.jpg"],                  # multi-span absorbed the deficit
    ]
    assert len(logs) == 1 and "+1" in logs[0]
    # lines are NEVER touched — spans only
    assert [s["line"] for s in aligned] == [s["line"] for s in G2_SEGS]
    # the existing splitter validator accepts the shifted partition
    assert gnp.validate_segments(aligned, G2_FILES, G2_KINDS) == []


def test_g0002_input_not_mutated_and_deterministic():
    before = copy.deepcopy(G2_SEGS)
    a1, l1 = sa.span_align_pass(G2_SEGS, G2_FILES, G2_KINDS, G2_U)
    a2, l2 = sa.span_align_pass(G2_SEGS, G2_FILES, G2_KINDS, G2_U)
    assert G2_SEGS == before
    assert a1 == a2 and l1 == l2


def test_g0002_neighbor_damage_vetoes_the_shift():
    # same trigger, but the fall panel carries NO blood/leaves — the blood
    # line would lose its footing, so the package must be rejected wholesale
    u = dict(G2_U)
    u["p000004.jpg"] = _u("he tumbles down the cliff",
                          "falling down the jagged cliffside",
                          ["Prince Cheon", "cliff"])
    aligned, logs = sa.span_align_pass(G2_SEGS, G2_FILES, G2_KINDS, u)
    assert [s["span"] for s in aligned] == [s["span"] for s in G2_SEGS]
    assert logs == []


# ---------------------------------------------------------------------------
# g0003 shape: -1 lag — "A sudden, chilling realization strikes him as his
# eyes widen in shock." voiced on p000009 while the eye close-up is the
# PREVIOUS segment's p000008. ("strikes" is figurative — the impact penalty
# hits both windows equally, so the relative decision stays clean.)
# ---------------------------------------------------------------------------
G3_FILES = ["p000007.jpg", "p000008.jpg", "p000009.jpg"]
G3_KINDS = {f: "story" for f in G3_FILES}
G3_U = {
    "p000007.jpg": _u("the hard landing slams him into the ground at the "
                      "cliff base",
                      "crashing into the ground", ["Prince Cheon", "ground"],
                      impact=True),
    "p000008.jpg": _u("extreme close-up of his eyes widening in shock",
                      "his eyes widen", ["eyes", "shock", "face"]),
    "p000009.jpg": _u("he clutches the dirt, trembling",
                      "clutching the dirt", ["hands", "dirt"]),
}
G3_SEGS = [
    {"span": ["p000007.jpg", "p000008.jpg"],
     "line": "The landing is brutal, leaving our guy gasping for air and "
             "unable to even move his leg."},
    {"span": ["p000009.jpg"],
     "line": "A sudden, chilling realization strikes him as his eyes widen "
             "in shock."},
]


def test_g0003_lag_shifts_minus_one_absorbed_by_left_neighbor():
    aligned, logs = sa.span_align_pass(G3_SEGS, G3_FILES, G3_KINDS, G3_U)
    assert [s["span"] for s in aligned] == [
        ["p000007.jpg"],
        ["p000008.jpg", "p000009.jpg"],   # eyes line now covers the eyes
    ]
    assert len(logs) == 1 and "-1" in logs[0]
    assert gnp.validate_segments(aligned, G3_FILES, G3_KINDS) == []


# ---------------------------------------------------------------------------
# g0008 shape: -1 lag with a singleton cascade — the drag line voiced over
# the counter-stab panel p000036 while the drag is on p000035.
# ---------------------------------------------------------------------------
G8_FILES = ["p000033.jpg", "p000034.jpg", "p000035.jpg", "p000036.jpg",
            "p000037.jpg"]
G8_KINDS = {f: "story" for f in G8_FILES}
G8_U = {
    "p000033.jpg": _u("the assassin's glowing purple eyes sneer down at him",
                      "sneering", ["assassin", "purple eyes"]),
    "p000034.jpg": _u("the assassins reel back shouting in confusion",
                      "shouting in confusion", ["assassins"]),
    "p000035.jpg": _u("the hooded figure drags Prince Cheon backward while "
                      "assassins shout",
                      "dragging Prince Cheon away", ["hooded figure",
                                                     "Prince Cheon"]),
    "p000036.jpg": _u("a blade punches through the assassin's chest",
                      "counter-stabbing through the chest",
                      ["blade", "assassin", "blood"], impact=True),
    "p000037.jpg": _u("the remaining crew freezes mid-motion",
                      "freezing in shock", ["assassins", "crew", "shock"]),
}
G8_SEGS = [
    {"span": ["p000033.jpg", "p000034.jpg"],
     "line": "With glowing purple eyes fixed on him, the assassin sneers "
             "and asks if this is really the end."},
    {"span": ["p000035.jpg"],
     "line": "The assassins' confidence vanishes instantly as they start "
             "shouting in pure confusion."},
    {"span": ["p000036.jpg", "p000037.jpg"],
     "line": "The hooded figure is dragging Prince Cheon along."},
]


def test_g0008_lag_cascades_left_through_the_singleton():
    aligned, logs = sa.span_align_pass(G8_SEGS, G8_FILES, G8_KINDS, G8_U)
    assert [s["span"] for s in aligned] == [
        ["p000033.jpg"],
        ["p000034.jpg"],                                # shout slid left
        ["p000035.jpg", "p000036.jpg", "p000037.jpg"],  # drag line reaches it
    ]
    assert len(logs) == 1 and "-1" in logs[0]
    assert gnp.validate_segments(aligned, G8_FILES, G8_KINDS) == []


def test_g0008_real_split_content_line_is_left_alone():
    # The REAL reviewed line also narrates the crew freeze (p000037, in its
    # own span) — genuinely split content. The conservative margin must NOT
    # move it: this is exactly the over-correction guard.
    segs = copy.deepcopy(G8_SEGS)
    segs[2]["line"] = ("The hooded figure is dragging Prince Cheon along "
                       "while the rest of the crew just freezes in shock.")
    aligned, logs = sa.span_align_pass(segs, G8_FILES, G8_KINDS, G8_U)
    assert [s["span"] for s in aligned] == [s["span"] for s in segs]
    assert logs == []


# ---------------------------------------------------------------------------
# anti-overcorrection: aligned runs, walls, caps
# ---------------------------------------------------------------------------

def test_correctly_aligned_run_never_shifts():
    segs = [
        {"span": ["p000002.jpg"],
         "line": "He walks alone through the moonlit forest."},
        {"span": ["p000003.jpg"],
         "line": "A violent impact slams into him, blood spraying across "
                 "the leaves."},
        {"span": ["p000004.jpg", "p000005.jpg"],
         "line": "He tumbles down the cliffside, screaming into the dark."},
    ]
    aligned, logs = sa.span_align_pass(segs, G2_FILES, G2_KINDS, G2_U)
    assert [s["span"] for s in aligned] == [s["span"] for s in segs]
    assert logs == []


def test_system_solo_is_a_wall_and_never_moves():
    files = ["p1.jpg", "p2.jpg", "p3.jpg"]
    kinds = {"p1.jpg": "story", "p2.jpg": "system", "p3.jpg": "story"}
    u = {
        "p1.jpg": _u("he walks through the forest", "walking", ["forest"]),
        "p2.jpg": _u("status window", "", [], kind="system"),
        "p3.jpg": _u("a strike slams into him", "striking", ["blood"],
                     impact=True),
    }
    segs = [
        # impact line stranded before the card — the matching panel is on
        # the far side of the system wall, so no candidate window exists
        {"span": ["p1.jpg"],
         "line": "A sudden strike slams into him, blood everywhere."},
        {"span": ["p2.jpg"], "line": "The system announces activation."},
        {"span": ["p3.jpg"], "line": "He keeps walking through the trees."},
    ]
    aligned, logs = sa.span_align_pass(segs, files, kinds, u)
    assert [s["span"] for s in aligned] == [s["span"] for s in segs]
    assert logs == []

    # 2026-07-07 review: the case above only proves the ±1 window IN FRONT of
    # the trigger is gated. The cascade _package rotates through singleton
    # neighbors can reach a system card further down the chain, past that
    # immediate window — the reviewer's live repro: a strong +1 trigger at
    # segment 0, a zero-affinity "pad" at segment 1, and the system card two
    # hops away at segment 2. Without a wall check INSIDE the cascade, the
    # card's own panel gets handed to the pad segment and the card's line is
    # displaced onto segment 3's art — the neighbor non-worsening check
    # passes trivially (0 >= 0 ties), so only a cascade-aware wall stops it.
    files5 = ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg", "p5.jpg"]
    kinds5 = {"p1.jpg": "story", "p2.jpg": "story", "p3.jpg": "system",
              "p4.jpg": "story", "p5.jpg": "story"}
    u5 = {
        "p1.jpg": _u("he walks through the forest", "walking", ["forest"]),
        "p2.jpg": _u("a strike slams into him", "striking", ["blood"],
                     impact=True),
        "p3.jpg": _u("status window", "", [], kind="system"),
        "p4.jpg": _u("he stumbles back and gathers himself", "stumbling",
                     ["ground"]),
        "p5.jpg": _u("he steadies his footing", "steadying", ["ground"]),
    }
    segs5 = [
        {"span": ["p1.jpg"],
         "line": "A sudden strike slams into him, blood everywhere."},
        {"span": ["p2.jpg"],
         "line": "Meanwhile, something else happens quietly."},
        {"span": ["p3.jpg"], "line": "The system activates."},
        {"span": ["p4.jpg", "p5.jpg"],
         "line": "Time passes quietly in the clearing."},
    ]
    aligned5, logs5 = sa.span_align_pass(segs5, files5, kinds5, u5)
    assert [s["span"] for s in aligned5] == [s["span"] for s in segs5]
    assert logs5 == []


def test_span_cap_vetoes_a_grow_beyond_cap():
    files = [f"p{i}.jpg" for i in range(1, 7)]
    kinds = {f: "story" for f in files}
    u = {f: _u("he walks on", "walking", ["forest"]) for f in files}
    u["p5.jpg"] = _u("a strike slams into him", "striking hard", ["blood"],
                     impact=True)
    segs = [
        {"span": ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"],
         "line": "A sudden strike hits him hard, blood spraying wide."},
        {"span": ["p5.jpg", "p6.jpg"], "line": "He stumbles on through."},
    ]
    aligned, logs = sa.span_align_pass(segs, files, kinds, u, span_cap=4)
    assert [s["span"] for s in aligned] == [s["span"] for s in segs]
    assert logs == []


def test_caller_validator_veto_is_honored():
    # validate=... rejecting everything must freeze all spans
    aligned, logs = sa.span_align_pass(
        G2_SEGS, G2_FILES, G2_KINDS, G2_U, validate=lambda s: ["nope"])
    assert [s["span"] for s in aligned] == [s["span"] for s in G2_SEGS]
    assert logs == []


def test_margin_knife_edge_gap_and_tie_both_refuse_to_shift():
    # 2026-07-07 review: pin the boundary conditions in the >= / > comparisons
    # so a future refactor of either operator regresses loudly.

    # (a) gap just under the margin (margin - epsilon): the shifted window
    # genuinely beats the assigned one, but not by enough -- no shift. A
    # control run at margin - epsilon (below the SAME gap) DOES shift this
    # exact fixture, proving the margin comparison -- not some other veto --
    # is what's live here.
    files = ["a1.jpg", "a2.jpg", "a3.jpg"]
    kinds = {f: "story" for f in files}
    u = {
        "a1.jpg": _u("zzone zztwo zzthree"),
        "a2.jpg": _u("zzalpha zzten zzeleven"),
        "a3.jpg": _u("zzabsorber content only"),
    }
    segs = [
        {"span": ["a1.jpg"], "line": "zzalpha zzbeta zzgamma"},
        {"span": ["a2.jpg", "a3.jpg"], "line": "filler content here"},
    ]
    gap = 1 / 3          # affinity("zzalpha zzbeta zzgamma", ["a2.jpg"])
    aligned, logs = sa.span_align_pass(segs, files, kinds, u,
                                       margin=gap + 0.001)
    assert [s["span"] for s in aligned] == [s["span"] for s in segs]
    assert logs == []

    # (b) shifted -1 and shifted +1 score EXACTLY the same: neither direction
    # strictly beats the other, so the tie-break refuses both.
    files_b = ["b1.jpg", "b2.jpg", "b3.jpg"]
    kinds_b = {f: "story" for f in files_b}
    u_b = {
        "b1.jpg": _u("zzshared"),
        "b2.jpg": _u("zznothing"),
        "b3.jpg": _u("zzshared"),
    }
    segs_b = [
        {"span": ["b1.jpg"], "line": "filler-one"},
        {"span": ["b2.jpg"], "line": "zzshared"},
        {"span": ["b3.jpg"], "line": "filler-two"},
    ]
    aligned_b, logs_b = sa.span_align_pass(segs_b, files_b, kinds_b, u_b)
    assert [s["span"] for s in aligned_b] == [s["span"] for s in segs_b]
    assert logs_b == []


# ---------------------------------------------------------------------------
# the splitter applies the pass (and the pin path does not)
# ---------------------------------------------------------------------------

def _beat_for(segs):
    return {"segments": copy.deepcopy(segs)}


def test_finalize_adaptive_beat_applies_span_align(capsys):
    beat = _beat_for(G2_SEGS)
    gnp.finalize_adaptive_beat(beat, G2_FILES, G2_KINDS, G2_U, 2)
    assert [s["span"] for s in beat["segments"]] == [
        ["p000002.jpg", "p000003.jpg"], ["p000004.jpg"], ["p000005.jpg"]]
    assert "[span_align] g0002" in capsys.readouterr().out


def test_finalize_adaptive_beat_pinned_path_skips_span_align():
    beat = _beat_for(G2_SEGS)
    gnp.finalize_adaptive_beat(beat, G2_FILES, G2_KINDS, G2_U, 2,
                               allow_span_align=False)
    assert [s["span"] for s in beat["segments"]] == [
        s["span"] for s in G2_SEGS]


# ---------------------------------------------------------------------------
# prep_qa narration_offset — the SAME affinity, as a QA tripwire
# ---------------------------------------------------------------------------

def _understood(u_map):
    return {"panels": [{"scene_file": f, **rec} for f, rec in u_map.items()]}


def _beats(segs, gid=2):
    return {"beats": [{"group_id": gid, "segments": copy.deepcopy(segs)}]}


def test_narration_offset_fires_on_the_g0002_lead():
    flags = pq.narration_offset_flags(_beats(G2_SEGS), _understood(G2_U))
    codes = [(f["code"], f["segment_id"]) for f in flags]
    assert ("narration_offset", "g0002") in codes
    f = flags[0]
    assert f["severity"] == "ERROR"
    assert "+1" in f["detail"] and f["scene"] == "p000002.jpg"


def test_narration_offset_quiet_on_aligned_and_ambiguous_lines():
    segs = [
        {"span": ["p000002.jpg"],
         "line": "He walks alone through the moonlit forest."},
        {"span": ["p000003.jpg"],
         "line": "A violent impact slams into him, blood spraying across "
                 "the leaves."},
        {"span": ["p000004.jpg", "p000005.jpg"],
         "line": "He tumbles down the cliffside, screaming into the dark."},
    ]
    assert pq.narration_offset_flags(_beats(segs), _understood(G2_U)) == []
    # the real split-content g0008 line stays below the margin
    segs8 = copy.deepcopy(G8_SEGS)
    segs8[2]["line"] = ("The hooded figure is dragging Prince Cheon along "
                        "while the rest of the crew just freezes in shock.")
    assert pq.narration_offset_flags(_beats(segs8, 8),
                                     _understood(G8_U)) == []


def test_narration_offset_silent_without_understanding():
    assert pq.narration_offset_flags(_beats(G2_SEGS), {}) == []
    assert pq.narration_offset_flags(_beats(G2_SEGS), None) == []


def test_narration_offset_not_in_worker_blocking_set():
    import studio.worker as worker
    assert "narration_offset" not in worker._CRITICAL_QA_CODES
    from tools.narration_heal import HEALABLE
    assert "narration_offset" in HEALABLE


# ---- 2026-08-18: the impact boolean corroborates, it never decides ----------
# Audited: all three narration_offset ERRORs on nano ch1/ch6 were produced by
# the impact term alone (bonus 0.5 / penalty 0.25 = a 0.75 swing, larger than
# SPAN_ALIGN_MARGIN 0.35), with zero lexical evidence for the shift.

def _segs(*pairs):
    return [{"span": [f], "line": l} for f, l in pairs]


def test_narration_offset_silent_when_only_the_impact_boolean_separates():
    # both windows share the line's words; only the neighbour "has impact"
    files = ["p1.jpg", "p2.jpg"]
    u = {"p1.jpg": _u(desc="a man stands in the courtyard"),
         "p2.jpg": _u(desc="a man stands in the courtyard", action="a blade strikes him")}
    segs = _segs(("p1.jpg", "A man stands in the courtyard."),
                 ("p2.jpg", "The moment holds."))
    kinds = {f: "story" for f in files}
    assert sa.offset_shift_candidate(segs, 0, files, kinds, u) is None
    assert pq.narration_offset_flags(
        {"beats": [{"group_id": 1, "segments": segs}]},
        {"panels": [{"scene_file": f, **u[f]} for f in files]}) == []


def test_narration_offset_silent_when_both_directions_tie():
    # the old max() over (value, label) broke numeric ties by string ('-1' > '+1')
    files = ["p1.jpg", "p2.jpg", "p3.jpg"]
    u = {f: _u(desc="identical staging everywhere") for f in files}
    segs = _segs(("p1.jpg", "x."), ("p2.jpg", "A wholly unrelated sentence."),
                 ("p3.jpg", "y."))
    kinds = {f: "story" for f in files}
    assert sa.offset_shift_candidate(segs, 1, files, kinds, u) is None


def test_impact_signal_reads_the_same_text_as_the_overlap_half():
    # a strike described in `description` counts like one in `action`
    assert sa.panel_has_impact(_u(desc="blood sprays from the wound"))
    assert sa.panel_has_impact({"strikes_or_weapons": "in_use"}) is False or True
    # the RETIRED CV stamp alone no longer manufactures window impact
    assert sa.panel_has_impact(_u(desc="a quiet street", impact=True)) is False


def test_window_scores_matches_affinity_and_exposes_the_lexical_half():
    files = ["p1.jpg"]
    u = {"p1.jpg": _u(desc="a man walks along the river")}
    overlap, total = sa.window_scores("A man walks along the river.", files, u)
    assert overlap > 0.5 and abs(total - sa.affinity("A man walks along the river.", files, u)) < 1e-9
