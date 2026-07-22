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


def test_mentions_image_file_catches_filename_leaks():
    # prose-first hands the writer scene_file names as tags — a tag leaking
    # into a voiced line ("…to conclude at p000032.jpg") must be detectable
    assert rs.mentions_image_file(
        "It progresses through the series to conclude at p000032.jpg.")
    assert rs.mentions_image_file("He stares at chunk_003.PNG in silence.")
    # the SAME leak without an extension — the bare p-number scene id read aloud
    # ("The sequence begins with frame p000098.", real Nano ch1 group 22)
    assert rs.mentions_image_file("The sequence begins with frame p000098.")
    assert rs.mentions_image_file("It ends on p000032 as the dust settles.")
    assert not rs.mentions_image_file(
        "He tears through the underbrush, blade out, and the hunt turns.")
    # bare numbers that are NOT the p-scheme must stay clean (no false positives)
    assert not rs.mentions_image_file("He was shot 12 times before he fell.")
    assert not rs.mentions_image_file("")
    assert not rs.mentions_image_file(None)


def test_mentions_impact_marker_catches_the_bracket_echo():
    # the impact-SFX detector's writer-input marker ("[IMPACT SFX on panel]")
    # is a bracket TAG fed to the model as context — the SAME leak channel as
    # the scene_file tags above.
    assert rs.mentions_impact_marker(
        "[IMPACT SFX on panel] as the blade comes down.")
    assert rs.mentions_impact_marker(
        "He staggers back, [Impact SFX on panel], and falls.")  # case-insensitive
    assert not rs.mentions_impact_marker("The blade strikes true.")
    assert not rs.mentions_impact_marker("")
    assert not rs.mentions_impact_marker(None)


def test_is_shot_description_catches_montage_meta_filler():
    # the small-model punt on a dense action run: narrate the MEDIUM (camera,
    # shots, sequence) instead of the story. Real Nano ch1 group 7 (2026-07-04)
    # collapsed 6 richly-understood attack panels into these two lines, and they
    # slipped every other guard (no "camera"/"unfolds" in _SHOT_DESC_RE, and a
    # 9-word line clears the 1.0s/panel budget floor over a 3-panel span).
    for bad in [
        "The scene unfolds through a series of connected shots.",
        "The action continues as the camera follows the subject.",
        "The sequence begins as connected panels reveal the fight.",
        "The montage shows the battle from every angle.",
    ]:
        assert rs.is_shot_description(bad), bad
    # real story prose that SHARES the risky vocabulary must stay clean — the
    # medium-noun+verb pairing and lone "camera" are the only triggers.
    for good in [
        "The scene shifts to the courtyard.",
        "He follows the trail into the woods.",
        "The battle continues past dawn.",
        "A cold panel of light falls across the floor.",
        "The panel begins to glow with system light.",   # story sense of "panel"
        "The action moves to the rooftop as he gives chase.",  # story "action"
        "The ambush goes full zero-star experience as the assassins swarm him.",
        "In a blur of steel, our guy lunges forward with a desperate scream.",
    ]:
        assert not rs.is_shot_description(good), good


# ---------------------------------------------------------------------------
# 2026-07-06 Nano ch1 vision-review classes B (display-format meta), C
# (writer-truncated line) and D (camera-POV prose). Real lines verbatim.
# ---------------------------------------------------------------------------

def test_display_meta_lines_are_shot_descriptions():
    # class B — the two REAL voiced lines (g0024_p13 / g0024_p15): the old
    # montage-meta patterns missed the phrasing, so NO check fired
    for bad in [
        "The text is displayed as a standalone caption.",
        "The text is displayed as a title or organizational name card.",
        "A caption is shown over the artwork.",
        "The chapter title appears as a banner.",
        "Bold lettering is presented across the frame.",
    ]:
        assert rs.is_shot_description(bad), bad
    # in-story senses of the same nouns must stay clean
    for good in [
        "He pulled a card from his sleeve and smiled.",
        "The royal title passes to his brother.",
        "She presented the medal to the victor.",
        "His name appears on the assassination list.",
        "A mechanical voice drones on, announcing that the 7th generation "
        "nano machine activation has officially begun.",   # g0025_p00, correct
    ]:
        assert not rs.is_shot_description(good), good


def test_camera_pov_lines_are_shot_descriptions():
    # class D — the two REAL voiced lines (g0018_p22 / g0004_p08)
    for bad in [
        "An electrified hand reaches out toward the viewer.",
        "The hooded figure in the foreground holds a sword ready for combat "
        "while delivering a threat.",
        "Dark shapes gather in the background.",
    ]:
        assert rs.is_shot_description(bad), bad
    for good in [
        "He reaches out toward the fallen prince.",
        "The hooded figure holds a sword ready for combat while delivering "
        "a threat.",
        "She fades into the crowd without a sound.",
    ]:
        assert not rs.is_shot_description(good), good


def test_truncated_real_line_is_fragment_and_amputates_cleanly():
    # class C — the REAL g0011_p16 line: ends on a bare word, so the old
    # fragment test (,;: / ellipsis / lowercase) never fired
    line = "But there is no mercy to be found, only the"
    assert not rs.ends_terminal(line)
    assert rs.is_spoken_fragment(line)
    # deterministic repair: amputate the dangling function-word stub after
    # the last clause separator — every complete clause survives
    assert rs.repair_spoken_line(line) == "But there is no mercy to be found."


def test_ends_terminal_accepts_normal_line_shapes():
    for ok in [
        "The clearing goes silent.",
        "Seriously, what even is that light?!",
        "He can only mutter, 'Ancestor...?'",
        "A pause hangs in the air…",
        'He whispers, "It ends tonight."',
    ]:
        assert rs.ends_terminal(ok), ok
        assert not rs.is_spoken_fragment(ok), ok


def test_repair_closes_complete_clause_and_leaves_bare_stub():
    # complete clause that just lost its period -> close it
    assert (rs.repair_spoken_line("He grabs the rope, hauling himself up")
            == "He grabs the rope, hauling himself up.")
    # dangling function-word tail with NO separator to cut at -> unchanged
    # (the truncated_line QA flag heals it with a real re-write)
    assert (rs.repair_spoken_line("He turns toward the sound of")
            == "He turns toward the sound of")


def test_repair_does_not_amputate_a_complete_clause_ending_on_a_function_word():
    # 2026-07-07 review: the old {0,4}-word tail window amputated up to 5
    # words after the last clause separator whenever the sentence ended on a
    # function word -- silently deleting a complete, meaningful clause just
    # because it happened to end on "is" ("In the end, he knows what this
    # is" -> "In the end."). Tightening the window to <=2 trailing words
    # means only a genuinely SHORT stub gets cut; a real clause now survives
    # unchanged (it still fails ends_terminal, so truncated_line/heal catches
    # it with a real re-write instead of silently mangling it).
    line = "In the end, he knows what this is"
    assert not rs.ends_terminal(line)
    assert rs.repair_spoken_line(line) == line

    # the real stub (a 2-word tail) must still repair -- the tightened window
    # must not regress the case it was built for
    assert (rs.repair_spoken_line(
        "But there is no mercy to be found, only the")
        == "But there is no mercy to be found.")


def test_repair_spoken_line_leaves_short_complete_lines_untouched():
    assert rs.repair_spoken_line("No.") == "No."
    assert rs.repair_spoken_line("Run!") == "Run!"


def test_repair_spoken_fragments_amputates_truncated_segment_line():
    beats = {"beats": [{"group_id": 11, "segments": [
        {"span": ["p000052.jpg"],
         "line": "But there is no mercy to be found, only the"},
    ]}]}
    assert rs.repair_spoken_fragments(beats) == 1
    seg = beats["beats"][0]["segments"][0]
    assert seg["line"] == "But there is no mercy to be found."
    assert beats["beats"][0]["narration"] == "But there is no mercy to be found."


# ---------------------------------------------------------------------------
# Round-2 (2026-07-07) classes E3 (truncation MUTATION: possessive/function
# word + bare period slips the terminal-char test) and E4 (filler mutation:
# panel-as-UI narration slips the display-meta arms). Real lines verbatim.
# ---------------------------------------------------------------------------

def test_possessive_period_ending_is_a_truncation_mutation():
    # the REAL round-2 line: ends "…an assassin's." — possessive + period,
    # which the old guard read as a finished sentence
    line = "He whips the blade around and sends blood splattering across " \
           "an assassin's."
    assert not rs.ends_terminal(line)
    assert rs.is_spoken_fragment(line)
    # plural possessive mutation too
    assert not rs.ends_terminal("The plan was always the assassins'.")


def test_possessive_followed_by_noun_is_terminal():
    # careful negative from the spec: possessive + NOUN + period IS a
    # complete thought — only possessive-period-END fails
    for ok in [
        "Blood splatters across the assassin's blade.",
        "He tears the mask from the assassin's face.",
        "The ancestor's last wish hangs in the air.",
    ]:
        assert rs.ends_terminal(ok), ok
        assert not rs.is_spoken_fragment(ok), ok


def test_article_preposition_conjunction_period_is_not_terminal():
    for bad in [
        "But there is no mercy to be found, only the.",
        "He reaches for the hilt and.",
        "The strike was meant for.",
        "She vanishes into the mist with.",
    ]:
        assert not rs.ends_terminal(bad), bad
    # the same words INSIDE a finished sentence stay terminal
    for ok in [
        "He fights for the ones he lost.",
        "That is what the blade is for!",
        "And with that, the hall goes silent.",
    ]:
        assert rs.ends_terminal(ok), ok


def test_mutation_endings_keep_exclamation_and_ellipsis_behavior():
    # scoped to bare '.' — deliberate trails/exclamations are untouched
    assert rs.ends_terminal("A pause hangs in the air…")
    assert rs.ends_terminal("Seriously, what even is that light?!")


def test_repair_never_double_punctuates_a_mutation_ending():
    # already ends '.', not terminal: repair must NOT append another period
    # (no separator to amputate at -> left for the truncated_line heal)
    line = "He sends blood splattering across an assassin's."
    assert rs.repair_spoken_line(line) == line


def test_panel_as_ui_filler_mutation_is_shot_description():
    # class E4 — the REAL voiced line evaded the display-meta arms
    for bad in [
        "A white panel appears with the text: serial number.",
        "A panel appears with the text announcing his awakening.",
        "The panel displays a string of numbers.",
        "The white panel shows the activation message.",
        "The card reads, with the text: nano machine online.",
    ]:
        assert rs.is_shot_description(bad), bad
    # in-story senses: a named in-world screen or story 'panel' stays clean
    for good in [
        "The control panel displays his vitals in red.",
        "The status panel shows his strength climbing.",
        "The panel begins to glow beneath his palm.",
        "He hammers on the door panel until it gives.",
        "A mechanical voice announces the seventh activation.",
    ]:
        assert not rs.is_shot_description(good), good


def test_mentions_mood_tag_leak_catches_the_real_round3_shapes():
    # round-3 Nano ch1 regression: 18 segments (15 "Dramatic:", 3 "Comic:")
    # voiced a bare mood/tone label instead of the sanctioned bracketed tag.
    # ALL 18 real leaks were colon-form; the bare/comma form only counts when
    # a fresh capitalized PRONOUN sentence follows the dangling label.
    for bad in [
        "Dramatic: He’s tumbling down a massive cliff, screaming his "
        "lungs out while plummeting into the abyss.",
        "Dramatic: Suddenly, a blinding flash of light erupts out of nowhere.",
        "Comic: The masked guy grabs him by the throat and asks if that "
        "was his big attempt at revenge.",
        "DRAMATIC: He's tumbling down a massive cliff.",   # raw writer caps
        "Dramatic He's free-falling down a rocky cliff, screaming.",  # no colon
        "calm, He lowers his blade and breathes.",  # comma + pronoun restart
    ]:
        assert rs.mentions_mood_tag_leak(bad), bad


def test_mentions_mood_tag_leak_silent_on_story_and_bracket_forms():
    for good in [
        "He draws the blade and lunges.",
        "[dramatic] He's tumbling down a massive cliff.",   # sanctioned form
        # a real sentence that starts with a mood word as an adjective,
        # continuing in lowercase, is not a leaked label
        "Dramatic reveals stay restrained even in the quiet panels.",
        "Serious injuries mount as the battle rages on.",
        "Dramatic tension fills the hall.",
        # PRECISION (2026-07-07): mood word + comma + PROPER NOUN is real
        # narration, not a dangling label — the old any-capital gate flagged
        # these and the TTS-side stripper (same pattern) silently deleted the
        # mood word from the voiced text.
        "Tense, Mira grips the railing.",
        "Calm, Prince Cheon steadies his blade.",
        "calm, Jang lowers his blade and breathes.",
        "",
    ]:
        assert not rs.mentions_mood_tag_leak(good), good
    assert not rs.mentions_mood_tag_leak(None)


# --- protagonist handle VARIETY (reduce "our guy" repetition) ---------------

def _prot_cast(name="Prince Cheon"):
    return {"cast": [{"canonical_name": "our protagonist", "role": "protagonist",
                      "is_protagonist": True, "aliases": [name]}]}


def _beat(g, *lines):
    return {"group_id": g, "segments": [
        {"span": [f"p{g}_{i}.jpg"], "line": l} for i, l in enumerate(lines)]}


def _lines(B):
    return [s["line"] for b in B["beats"] for s in b["segments"]]


def test_handle_pool_derives_a_title_epithet():
    import tools.recap_style as rs
    assert rs._protagonist_handles("the prince")[0] == "the prince"
    assert rs._protagonist_handles("") == ["the protagonist", "our MC"]


def test_title_epithet_only_when_unambiguous():
    import tools.recap_style as rs
    prot = {"is_protagonist": True, "aliases": ["Prince Cheon"]}
    # sole prince -> epithet allowed
    assert rs._title_epithet("Prince Cheon", [prot], prot) == "the prince"
    # a SECOND prince in the cast -> ambiguous, no epithet
    other = {"aliases": ["Prince Baek"]}
    assert rs._title_epithet("Prince Cheon", [prot, other], prot) == ""
    # untitled name -> no epithet
    assert rs._title_epithet("Kim Dokja", [prot], prot) == ""


def test_vary_reduces_our_guy_repetition():
    """The point: a tail that was ALL 'our guy' becomes a mix — 'our guy' no
    longer dominates."""
    import tools.recap_style as rs
    # name introduced 3x, then 6 more protagonist refs the writer wrote as
    # 'our guy'
    B = {"beats": [
        _beat(1, "Prince Cheon appears.", "Prince Cheon fights.",
              "Prince Cheon wins."),
        _beat(2, "our guy runs.", "our guy jumps.", "our guy falls.",
              "our guy stands.", "our guy roars.", "our guy leaves.")]}
    rs.cap_protagonist_name(B, _prot_cast(), keep=3, vary=True)
    tail = " ".join(_lines(B)[3:]).lower()
    # every reference still points at the protagonist (noun phrase), but
    # 'our guy' is now a minority, not all 6
    assert tail.count("our guy") < 6
    assert "the prince" in tail                       # the title epithet appears
    # intro preserved
    assert _lines(B)[:3] == ["Prince Cheon appears.", "Prince Cheon fights.",
                             "Prince Cheon wins."]


def test_vary_preserves_sentence_case_and_possessive():
    import tools.recap_style as rs
    B = {"beats": [_beat(1, "Prince Cheon starts.", "Prince Cheon acts.",
                         "Prince Cheon ends.",
                         "Our guy raises his fist.",         # sentence-start
                         "It was our guy's blade.")]}       # possessive
    rs.cap_protagonist_name(B, _prot_cast(), keep=3, vary=True)
    ln = _lines(B)
    # whatever handle each rotated to, capitalisation + possessive are intact
    assert ln[3][0].isupper() and ln[3].endswith("his fist.")
    assert "'s blade." in ln[4]
    # no lowercase mid-sentence artifact like 'the prince' at a sentence start
    assert not ln[3].startswith("the ") and not ln[3].startswith("our ")  # capitalised


def test_vary_is_deterministic_on_the_same_input():
    import tools.recap_style as rs, copy
    base = {"beats": [_beat(1, "Prince Cheon a.", "Prince Cheon b.",
                            "Prince Cheon c.", "our guy d.", "our guy e.",
                            "our guy f.")]}
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    rs.cap_protagonist_name(a, _prot_cast(), keep=3, vary=True)
    rs.cap_protagonist_name(b, _prot_cast(), keep=3, vary=True)
    assert _lines(a) == _lines(b)                     # reproducible


def test_vary_never_touches_a_non_protagonist():
    """The rotation only rewrites protagonist references — a name that isn't
    the protagonist is left alone."""
    import tools.recap_style as rs
    B = {"beats": [_beat(1, "Prince Cheon strikes.", "Prince Cheon parries.",
                         "Prince Cheon lunges.", "our guy retreats.",
                         "The assassin leader watches from the ridge.")]}
    rs.cap_protagonist_name(B, _prot_cast(), keep=3, vary=True)
    assert "The assassin leader watches from the ridge." in _lines(B)


def test_collapse_name_stutter():
    """A degenerate repeated proper noun collapses; real grammar is untouched."""
    import tools.recap_style as rs
    B = {"beats": [_beat(1,
        "Jang Jang Jang snaps his head up.",       # triple name -> collapse
        "Prince Prince stares him down.",          # double name  -> collapse
        "He had had enough of the noise.",         # legit 'had had' (lowercase)
        "Jang jang echoes through the hall.",      # mixed case  -> NOT a name echo
        "Prince Cheon drew his blade.")]}          # distinct words -> untouched
    n = rs.collapse_name_stutter(B)
    ln = _lines(B)
    assert ln[0] == "Jang snaps his head up."
    assert ln[1] == "Prince stares him down."
    assert ln[2] == "He had had enough of the noise."   # NOT collapsed
    assert ln[3] == "Jang jang echoes through the hall."  # NOT collapsed
    assert ln[4] == "Prince Cheon drew his blade."       # NOT collapsed
    assert n == 2


def test_vary_never_doubles_a_determiner():
    """A 'the …' handle swapped in after 'the <participle>' must not double the
    article ('the wounded the protagonist') — the real nano ch1 p000050 line."""
    import tools.recap_style as rs
    B = {"beats": [_beat(1, "Prince Cheon a.", "Prince Cheon b.", "Prince Cheon c.",
                         "The assassin stands over the wounded our guy.",
                         "A blade grazes an injured our guy.")]}
    rs.cap_protagonist_name(B, _prot_cast(), keep=3, vary=True)
    joined = " ".join(_lines(B)).lower()
    assert "the wounded the " not in joined
    assert "an injured the " not in joined
    # the reference still resolves to a protagonist noun phrase, not deleted
    assert "wounded" in joined and "injured" in joined
    # a determiner that governs a DIFFERENT noun is NOT stripped
    B2 = {"beats": [_beat(2, "Prince Cheon a.", "Prince Cheon b.", "Prince Cheon c.",
                          "The assassin stalks our guy.")]}
    rs.cap_protagonist_name(B2, _prot_cast(), keep=3, vary=True)
    assert _lines(B2)[3].lower().startswith("the assassin stalks ")
