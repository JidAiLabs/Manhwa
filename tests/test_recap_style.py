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
