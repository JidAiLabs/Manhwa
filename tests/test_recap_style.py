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
