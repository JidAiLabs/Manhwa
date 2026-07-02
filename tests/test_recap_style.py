"""Deterministic recap-style metrics and reveal-pacing guard."""

from __future__ import annotations

from tools import recap_style as rs


def _script(lines):
    return {"sections": [{"script_paragraphs": lines}]}


def _beats(lines):
    return {"beats": [{"group_id": 1, "panel_narration": [
        {"scene_file": f"p{i:03d}.jpg", "line": line}
        for i, line in enumerate(lines, 1)
    ]}]}


def _cast():
    return {"cast": [{
        "canonical_name": "Prince Cheon",
        "aliases": [],
        "role": "protagonist",
        "is_protagonist": True,
    }]}


def _analyze(lines, *, story=None, vision=None):
    return rs.analyze_recap_style(
        _script(lines), _beats(lines),
        story or {}, _cast(), vision or {},
    )


def test_name_ration_flags_repeated_full_protagonist_name():
    lines = [f"Prince Cheon advances through moment {i}." for i in range(25)]
    report = _analyze(lines)
    assert report["metrics"]["protagonist_name_uses"] == 25
    assert any(i["code"] == "name_ration" for i in report["issues"])


def test_sauce_and_pointing_are_measured_from_spoken_text():
    lines = [
        "Our guy picks the worst possible time for a stealth build.",
        "The trap closes.",
        "He moves like a superhero and the first attacker folds.",
        "The survivors retreat.",
    ] * 3
    report = _analyze(lines)
    assert report["metrics"]["sauce_density"] >= 0.25
    assert report["metrics"]["pointing_lines"] >= 3
    assert not any(i["code"] in {"sauce_density", "pointing_fits"}
                   for i in report["issues"])


def test_sauce_density_uses_only_connective_eligible_panels():
    lines = ["Our guy picks the worst possible time to hesitate."] + [
        "The blade lands with devastating force." for _ in range(9)]
    beats = _beats(lines)
    beats["beats"][0]["scene_selection"] = [
        {"scene_file": f"p{i:03d}.jpg",
         "intensity": "calm" if i == 1 else "explosive"}
        for i in range(1, 11)
    ]
    report = rs.analyze_recap_style(
        _script(lines), beats, {}, _cast(), {})
    assert report["metrics"]["sauce_eligible_lines"] == 1
    assert report["metrics"]["sauce_density"] == 1.0


def test_no_describe_flags_visible_only_drag_without_word_budget():
    lines = [
        ("Under the pale moonlight, his eyes widen while crackling blue energy "
         "surrounds his body and the wind moves through his flowing hair, "
         "leaving everyone staring in pure shock at the glowing figure.")
    ] * 12
    report = _analyze(lines)
    codes = {i["code"] for i in report["issues"]}
    assert "no_describe" in codes
    assert "compression_density" not in codes
    assert report["metrics"]["average_words_per_panel_line"] > 10


def test_word_count_metrics_do_not_force_keep_every_panel_recap_pace():
    lines = [
        "The swords meet.",
        "He realizes the whole clan succession depends on this choice, and "
        "for the first time, running is no longer an option.",
    ] * 10
    report = _analyze(lines)
    assert report["metrics"]["average_words_per_panel_line"] > 5
    assert not any(i["code"] == "compression_density"
                   for i in report["issues"])


def test_spoken_fragment_gate_catches_cross_clip_grammar():
    assert rs.is_spoken_fragment("A heavy impact slams into the thicket,")
    assert rs.is_spoken_fragment("...leaving him frozen in place.")
    assert rs.is_spoken_fragment("leaving him frozen in place.")
    assert not rs.is_spoken_fragment("The impact leaves him frozen in place.")


def test_spoken_fragment_repair_changes_grammar_not_story_facts():
    assert rs.repair_spoken_line(
        "leaving him wide-eyed with sudden dread.") == (
            "That leaves him wide-eyed with sudden dread.")
    assert rs.repair_spoken_line(
        "...that he had walked into their trap.") == (
            "The truth is that he had walked into their trap.")
    assert rs.repair_spoken_line(
        "wondering if he had mastered martial arts.") == (
            "The question is whether he had mastered martial arts.")
    assert rs.repair_spoken_line(
        "A heavy impact tears through the clearing,") == (
            "A heavy impact tears through the clearing.")


def test_repair_spoken_fragments_rejoins_beat_narration():
    beats = _beats([
        "A heavy impact tears through the clearing,",
        "leaving him frozen in place.",
    ])
    assert rs.repair_spoken_fragments(beats) == 2
    assert all(not rs.is_spoken_fragment(p["line"])
               for p in beats["beats"][0]["panel_narration"])
    assert beats["beats"][0]["narration"].endswith("frozen in place.")


def test_reveal_pacing_catches_blue_silhouette_name_leak():
    lines = [
        "A glowing blue silhouette appears between the assassins.",
        "Prince Cheon stands there in unfamiliar clothes.",
        "The killers hesitate.",
        "The stranger says nothing.",
        "One attacker raises his sword.",
        "The tension snaps.",
        "The assassin demands an answer.",
    ]
    vision = {"p007.jpg": {"ocr_clean": "WHO ARE YOU!"}}
    report = _analyze(lines, vision=vision)
    leaks = [i for i in report["issues"]
             if i["code"] == "identity_reveal_leak"]
    assert len(leaks) == 1
    assert leaks[0]["scene"] == "p002.jpg"


def test_reveal_pacing_allows_neutral_handle():
    lines = [
        "A glowing blue silhouette appears between the assassins.",
        "The stranger stands there in unfamiliar clothes.",
        "The killers hesitate.",
        "One attacker demands to know who he is.",
    ]
    vision = {"p004.jpg": {"ocr_clean": "WHO ARE YOU!"}}
    report = _analyze(lines, vision=vision)
    assert report["metrics"]["identity_reveal_leaks"] == 0


def test_hook_can_name_protagonist_and_a_separate_mysterious_stranger():
    lines = [
        "Prince Cheon is saved by a mysterious stranger carrying future technology.",
        "The prince wakes in pain.",
    ]
    report = _analyze(lines)
    assert report["metrics"]["identity_reveal_leaks"] == 0


def test_identity_reveal_safeguard_rewrites_without_chapter_specific_rules():
    beats = _beats([
        "A glowing blue silhouette appears between the assassins.",
        "Prince Cheon stands there in unfamiliar clothes.",
        "The killers hesitate.",
        "One attacker demands an answer.",
    ])
    vision = {"p004.jpg": {"ocr_clean": "WHO ARE YOU!"}}
    changed = rs.neutralize_identity_reveal_leaks(beats, _cast(), vision)
    assert changed == 1
    panels = beats["beats"][0]["panel_narration"]
    assert panels[1]["line"] == "The stranger stands there in unfamiliar clothes."
    assert "Prince Cheon" not in beats["beats"][0]["narration"]


def test_unresolved_identity_carries_to_later_clear_view_panel():
    # P1 is a concealed arrival (cue lives in the UNDERSTOOD subjects, not the
    # line). P3 shows the same figure in clear view with NO concealment word of
    # its own, yet slips to the protagonist handle "Our guy". The unresolved
    # state must carry from P1 across P2 and neutralize the handle on P3.
    beats = _beats([
        "A figure appears between the assassins.",
        "The killers hesitate.",
        "Our guy stood there, enveloped in lightning.",
    ])
    understood = {
        "p001.jpg": {"subjects": ["glowing blue silhouette"]},
        "p003.jpg": {"subjects": ["a young man with blue goggles",
                                  "blue electrical sparks"]},
    }
    changed = rs.neutralize_identity_reveal_leaks(beats, _cast(), {}, understood)
    assert changed == 1
    panels = beats["beats"][0]["panel_narration"]
    assert "Our guy" not in panels[2]["line"]
    assert panels[2]["line"] == "The stranger stood there, enveloped in lightning."
    assert "Our guy" not in beats["beats"][0]["narration"]


def test_protagonist_handle_without_concealment_is_not_neutralized():
    beats = _beats([
        "Our guy charges into the courtyard.",
        "He cuts down the first guard.",
        "Our guy keeps moving.",
    ])
    changed = rs.neutralize_identity_reveal_leaks(beats, _cast(), {})
    assert changed == 0
    assert beats["beats"][0]["panel_narration"][0]["line"] == (
        "Our guy charges into the courtyard.")


def test_protagonist_name_without_concealment_survives():
    beats = _beats([
        "Prince Cheon trains at dawn.",
        "He sharpens his blade.",
        "Prince Cheon bows to his master.",
    ])
    changed = rs.neutralize_identity_reveal_leaks(beats, _cast(), {})
    assert changed == 0


def test_name_on_concealed_arrival_neutralized_without_ocr():
    # The old hard requirement was an OCR "who are you" question; it is now an
    # OPTIONAL extra trigger. A concealed-arrival cue alone carries the window.
    beats = _beats([
        "A glowing silhouette appears between the assassins.",
        "Prince Cheon stands there in unfamiliar clothes.",
        "The killers hesitate.",
    ])
    changed = rs.neutralize_identity_reveal_leaks(beats, _cast(), {})
    assert changed == 1
    assert beats["beats"][0]["panel_narration"][1]["line"] == (
        "The stranger stands there in unfamiliar clothes.")


def _cast_with_desc():
    return {"cast": [{
        "canonical_name": "Prince Cheon",
        "aliases": [],
        "role": "protagonist",
        "is_protagonist": True,
        "visual_description": "a wounded young prince in torn royal robes",
    }]}


def test_established_protagonist_not_neutralized_after_concealed_figure(_=None):
    # BUG 4 regression (commit 5ee94cb over-neutralized): a concealed blue figure
    # appears, then the ESTABLISHED wounded protagonist (matching his cast
    # visual_description) is named on the FOLLOWING panels. He must stay NAMED —
    # only a later panel that actually shows the unresolved blue figure (mislabeled
    # with the protagonist handle) is neutralized.
    beats = _beats([
        "A glowing blue silhouette appears between the killers.",  # concealed figure
        "Our guy lies bleeding against the wall.",                 # established protag
        "Prince Cheon coughs up blood, his royal robes torn.",     # established, named
        "Our guy stands wreathed in crackling blue lightning.",    # MISLABELED blue figure
    ])
    understood = {
        "p001.jpg": {"subjects": ["a glowing blue silhouette"]},
        "p002.jpg": {"subjects": ["a wounded young prince", "blood"]},
        "p003.jpg": {"subjects": ["the bleeding prince", "torn royal robes"]},
        "p004.jpg": {"subjects": ["a young man with blue goggles",
                                  "crackling blue energy"]},
    }
    changed = rs.neutralize_identity_reveal_leaks(
        beats, _cast_with_desc(), {}, understood)
    panels = beats["beats"][0]["panel_narration"]
    # the established protagonist keeps his name/handle on his own panels
    assert panels[1]["line"] == "Our guy lies bleeding against the wall."
    assert panels[2]["line"].startswith("Prince Cheon")
    # only the mislabeled blue-figure panel is neutralized
    assert panels[3]["line"] == (
        "The stranger stands wreathed in crackling blue lightning.")
    assert changed == 1
    assert "the stranger" not in panels[1]["line"].lower()


def test_reveal_pacing_rule_leads_with_recognition_not_blanket_carry():
    rules = rs.RECAP_STYLE_RULES
    assert "REVEAL PACING" in rules
    # rebalanced: lead with NAMING established cast for recognition
    assert "name established" in rules.lower()
    # the old blanket "carry that handle across" instruction is gone
    assert "carry that handle across" not in rules.lower()


def test_dedupe_consecutive_duplicate_panel_lines_merges_to_one():
    # BUG 2/3 (p95 & p96 both "Ancestor...?"): an exact-duplicate consecutive
    # panel line must not ship twice — the duplicate panel is merged out.
    beats = {"beats": [{
        "group_id": 1,
        "scene_files": ["p95.jpg", "p96.jpg", "p97.jpg"],
        "panel_narration": [
            {"scene_file": "p95.jpg", "line": "Ancestor...?"},
            {"scene_file": "p96.jpg", "line": "Ancestor...?"},  # exact dup
            {"scene_file": "p97.jpg", "line": "He turns away."},
        ],
        "scene_selection": [
            {"scene_file": "p95.jpg", "role": "keep"},
            {"scene_file": "p96.jpg", "role": "keep"},
            {"scene_file": "p97.jpg", "role": "keep"},
        ],
    }]}
    removed = rs.dedupe_consecutive_panel_lines(beats)
    assert removed == 1
    b = beats["beats"][0]
    assert [p["scene_file"] for p in b["panel_narration"]] == ["p95.jpg", "p97.jpg"]
    assert b["scene_files"] == ["p95.jpg", "p97.jpg"]
    assert [s["scene_file"] for s in b["scene_selection"]] == ["p95.jpg", "p97.jpg"]
    assert b["narration"] == "Ancestor...? He turns away."


def test_dedupe_consecutive_across_beat_boundary():
    beats = {"beats": [
        {"group_id": 1, "scene_files": ["a.jpg"],
         "panel_narration": [{"scene_file": "a.jpg", "line": "The blade falls."}]},
        {"group_id": 2, "scene_files": ["b.jpg", "c.jpg"],
         "panel_narration": [
             {"scene_file": "b.jpg", "line": "The blade falls."},   # dup of prev beat
             {"scene_file": "c.jpg", "line": "Silence."}]},
    ]}
    removed = rs.dedupe_consecutive_panel_lines(beats)
    assert removed == 1
    assert [p["scene_file"] for p in beats["beats"][1]["panel_narration"]] == ["c.jpg"]


def test_dedupe_never_empties_a_beat():
    beats = {"beats": [
        {"group_id": 1, "scene_files": ["a.jpg"],
         "panel_narration": [{"scene_file": "a.jpg", "line": "Hold."}]},
        {"group_id": 2, "scene_files": ["b.jpg"],
         "panel_narration": [{"scene_file": "b.jpg", "line": "Hold."}]},  # sole dup
    ]}
    removed = rs.dedupe_consecutive_panel_lines(beats)
    assert removed == 0
    assert len(beats["beats"][1]["panel_narration"]) == 1


def test_dedupe_keeps_distinct_lines():
    beats = _beats(["He draws the blade.", "She blocks it.", "Sparks fly."])
    removed = rs.dedupe_consecutive_panel_lines(beats)
    assert removed == 0
    assert len(beats["beats"][0]["panel_narration"]) == 3


def test_shot_description_is_flagged_and_story_line_is_not():
    # BUG D4: the align pad copied a panel's camera-prose description verbatim
    # ("A close-up shot shows..."). analyze_recap_style must flag it.
    assert rs.is_shot_description("A close-up shot shows his trembling hands.")
    assert rs.is_shot_description("The panel focuses on the bloody blade.")
    assert rs.is_shot_description("A wide shot captures the burning city.")
    # normal story lines never trip it
    assert not rs.is_shot_description("He draws the blade and lunges.")
    assert not rs.is_shot_description("The scene shifts.")
    assert not rs.is_shot_description("A long shadow falls across the courtyard.")

    camera = ["A close-up shot shows his trembling hands."] * 3
    report = _analyze(camera)
    assert report["metrics"]["shot_description_lines"] == 3
    assert any(i["code"] == "shot_description" for i in report["issues"])

    story = ["He draws the blade and lunges."] * 3
    report2 = _analyze(story)
    assert report2["metrics"]["shot_description_lines"] == 0
    assert not any(i["code"] == "shot_description" for i in report2["issues"])


def test_visual_effect_description_is_flagged_and_dramatic_action_is_not():
    # Nano ch1 shipped these on ACTION/motion panels: gemma described the
    # ARTWORK'S RENDERING (motion blur / speed lines / "is depicted" / a weapon
    # swinging through empty air) instead of the STORY. The camera/shot detector
    # missed all three, so they passed QA and shipped. They must flag now.
    bad = [
        "A sense of rapid movement or a passing object is depicted through motion blur.",
        "A sword is being swung with high velocity, creating motion blur effects.",
        "A blade swings through the air with lethal speed.",
    ]
    for line in bad:
        assert rs.is_shot_description(line), line
    # legit dramatic narration that merely names a CHARACTER's speed/motion (or
    # a strike with an impact/target) must STILL pass — it names no rendering.
    legit = [
        "He moved with lethal speed.",
        "He cut them down in a single brutal arc.",
        "Blood sprayed as the blade found its mark.",
        "She lunged, blade flashing toward his throat.",
    ]
    for line in legit:
        assert not rs.is_shot_description(line), line

    report = _analyze(bad)
    assert report["metrics"]["shot_description_lines"] == 3
    assert any(i["code"] == "shot_description" for i in report["issues"])

    report2 = _analyze(legit)
    assert report2["metrics"]["shot_description_lines"] == 0
    assert not any(i["code"] == "shot_description" for i in report2["issues"])


def test_recap_rules_forbid_rendering_and_visual_effect_language():
    # FIX 2 mirror: rule 1 must firmly ban naming the rendering / a visual effect.
    rules = rs.RECAP_STYLE_RULES.lower()
    assert "motion blur" in rules
    assert "visual effect" in rules or "rendering" in rules


def test_story_naming_the_figure_resolves_and_allows_name():
    # Once the story's OWN text (OCR) names the figure, the identity is
    # established and the protagonist name is allowed again.
    beats = _beats([
        "A glowing silhouette appears.",
        "The crowd gasps.",
        "Prince Cheon steps into the light.",
        "Prince Cheon raises his blade.",
    ])
    vision = {"p003.jpg": {"ocr_clean": "It's Prince Cheon!"}}
    changed = rs.neutralize_identity_reveal_leaks(beats, _cast(), vision)
    assert changed == 0
    panels = beats["beats"][0]["panel_narration"]
    assert panels[2]["line"] == "Prince Cheon steps into the light."
    assert panels[3]["line"] == "Prince Cheon raises his blade."


# ---- adaptive flow segments (Chunk 2): style pass operates on segments ------

def _seg_beats(segments, gid=1):
    """Native-segments beat: segments = [(span_tuple, line), ...]."""
    return {"beats": [{"group_id": gid, "segments": [
        {"span": list(span), "line": line} for span, line in segments
    ]}]}


def test_repair_spoken_fragments_on_native_segments():
    beats = _seg_beats([
        (("p001.jpg", "p002.jpg"), "A heavy impact tears through the clearing,"),
        (("p003.jpg",), "leaving him frozen in place."),
    ])
    assert rs.repair_spoken_fragments(beats) == 2
    segs = beats["beats"][0]["segments"]
    assert all(not rs.is_spoken_fragment(s["line"]) for s in segs)
    assert segs[0]["span"] == ["p001.jpg", "p002.jpg"]     # spans untouched
    assert beats["beats"][0]["narration"].endswith("frozen in place.")


def test_dedupe_is_noop_for_native_segments():
    """Dropping a segment would orphan its span (panels lose their narration
    cover) — for native-segments beats the consecutive-dup pass is a NO-OP;
    the planner/render_prep merge consecutive same-text segments downstream."""
    beats = _seg_beats([
        (("p001.jpg",), "Ancestor...?"),
        (("p002.jpg", "p003.jpg"), "Ancestor...?"),
    ])
    removed = rs.dedupe_consecutive_panel_lines(beats)
    assert removed == 0
    segs = beats["beats"][0]["segments"]
    assert len(segs) == 2
    assert [s["line"] for s in segs] == ["Ancestor...?", "Ancestor...?"]


def test_panel_rows_one_row_per_segment_with_span_head():
    beats = _seg_beats([
        (("p001.jpg", "p002.jpg", "p003.jpg"), "He falls the whole way down."),
        (("p004.jpg",), "The bottom catches him."),
    ])
    rows = rs.panel_rows(beats)
    assert len(rows) == 2
    assert rows[0] == {"scene_file": "p001.jpg",
                       "line": "He falls the whole way down."}
    assert rows[1]["scene_file"] == "p004.jpg"


def test_sauce_density_counts_flow_span_lines():
    """Eligibility is counted per SEGMENT (a flow span = one line), and a span
    is dramatic when ANY of its panels is intense/explosive."""
    span_line = "Our guy picks the worst possible time for a stealth build."
    solo_line = "The blade lands with devastating force."
    beats = _seg_beats([
        (("p001.jpg", "p002.jpg", "p003.jpg"), span_line),
        (("p004.jpg",), solo_line),
    ])
    beats["beats"][0]["scene_selection"] = [
        {"scene_file": "p001.jpg", "intensity": "calm"},
        {"scene_file": "p002.jpg", "intensity": "calm"},
        {"scene_file": "p003.jpg", "intensity": "calm"},
        {"scene_file": "p004.jpg", "intensity": "explosive"},
    ]
    report = rs.analyze_recap_style(
        _script([span_line, solo_line]), beats, {}, _cast(), {})
    # 2 segments, not 4 panels: the calm flow span is the ONE eligible line
    assert report["metrics"]["sauce_eligible_lines"] == 1
    assert report["metrics"]["panel_lines"] == 2
    assert report["metrics"]["sauce_density"] == 1.0


def test_sauce_density_span_dramatic_when_any_panel_intense():
    line = "He crosses the ridge as the horde closes in behind him."
    beats = _seg_beats([(("p001.jpg", "p002.jpg"), line)])
    beats["beats"][0]["scene_selection"] = [
        {"scene_file": "p001.jpg", "intensity": "calm"},
        {"scene_file": "p002.jpg", "intensity": "explosive"},
    ]
    report = rs.analyze_recap_style(_script([line]), beats, {}, _cast(), {})
    assert report["metrics"]["sauce_eligible_lines"] == 0


def test_neutralize_identity_on_flow_span_line():
    """A flow segment is judged against ALL its span panels: the power/gear cue
    lives on the span's SECOND panel's understanding, yet the segment's 'Our
    guy' is still neutralized while the window is open."""
    beats = _seg_beats([
        (("p001.jpg",), "A glowing silhouette appears between the assassins."),
        (("p002.jpg", "p003.jpg"),
         "Our guy tears through them without breaking stride."),
    ])
    understood = {
        "p003.jpg": {"subjects": ["figure wreathed in lightning"]},
    }
    changed = rs.neutralize_identity_reveal_leaks(
        beats, _cast(), {}, understood)
    assert changed == 1
    segs = beats["beats"][0]["segments"]
    assert "Our guy" not in segs[1]["line"]
    assert "stranger" in segs[1]["line"].lower()
    assert segs[1]["span"] == ["p002.jpg", "p003.jpg"]     # spans untouched
    assert "Our guy" not in beats["beats"][0]["narration"]


def test_neutralize_legacy_singletons_unchanged_behavior():
    """The pre-span behavior survives byte-for-byte on legacy manifests
    (singleton spans): same fixture as the carry test, same outcome."""
    beats = _beats([
        "A glowing silhouette appears between the assassins.",
        "Prince Cheon stands there in unfamiliar clothes.",
        "The killers hesitate.",
    ])
    changed = rs.neutralize_identity_reveal_leaks(beats, _cast(), {})
    assert changed == 1
    assert beats["beats"][0]["panel_narration"][1]["line"] == (
        "The stranger stands there in unfamiliar clothes.")


# ---- teaser round-trip: legacy-shaped synthetic beat keeps working ----------

def test_teaser_legacy_roundtrip_repairs_land_in_panel_narration():
    """teaser_planner wraps its narration as {"beats":[{"panel_narration":
    [...]}]}, runs the shared mutators IN PLACE, then reads panel_narration
    back. The shape-aware writer must land repairs in the LEGACY shape."""
    panel_narration = [
        {"scene_file": "ch1__p000012.jpg",
         "line": "A glowing silhouette appears between the assassins."},
        {"scene_file": "ch1__p000013.jpg",
         "line": "Prince Cheon stands there in unfamiliar clothes."},
        {"scene_file": "ch2__p000044.jpg",
         "line": "leaving the killers frozen in place,"},
    ]
    beats_obj = {"beats": [{"panel_narration": panel_narration}]}
    rs.neutralize_identity_reveal_leaks(beats_obj, _cast(), {})
    rs.repair_spoken_fragments(beats_obj)
    got = beats_obj["beats"][0].get("panel_narration") or []
    lines = [p["line"] for p in got]
    assert lines[1] == "The stranger stands there in unfamiliar clothes."
    assert not rs.is_spoken_fragment(lines[2])
    # the repairs mutated the SAME list object the teaser holds
    assert panel_narration[1]["line"] == lines[1]
    assert [p["scene_file"] for p in got] == [
        "ch1__p000012.jpg", "ch1__p000013.jpg", "ch2__p000044.jpg"]
