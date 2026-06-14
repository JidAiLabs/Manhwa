"""strip_chrome_opener: remove series-intro chrome (the licensed-title leak in
spoken narration) without touching legitimate story nouns. Title-agnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "script_expander",
    Path(__file__).resolve().parent.parent / "tools" / "script_expander.py")
se = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(se)  # type: ignore[union-attr]
f = se.strip_chrome_opener


def test_strips_series_intro_chrome_opener():
    assert f("Welcome to the world of Infinite Evolution From Zero.") == ""
    assert f("Welcome to the grind of Infinite Evolution From Zero.") == ""
    assert f("This is the story of the Strongest Newbie.") == ""


def test_strips_title_card_and_chapter_begins_chrome():
    # the leak that slipped through the first fix (what_happens fallback)
    assert f("The chapter begins with a title card for Infinite Evolution From Zero.") == ""
    assert f("We open on a title card for The Strongest Newbie.") == ""
    assert f("The episode opens with our hero asleep.") == ""


def test_keeps_real_sentence_after_chrome_opener():
    assert f("Welcome to the world of X. He wakes as a baby.") == "He wakes as a baby."


def test_spares_mid_sentence_story_nouns():
    # "Nano Machine" is the in-story device, not a chrome opener -> untouched
    s = "Suddenly, the 7th Generation Nano Machine announces the system start."
    assert f(s) == s
    drama = "Under a pale moon, Prince Cheon runs for his life."
    assert f(drama) == drama


def test_idempotent_and_blank_safe():
    assert f("") == ""
    assert f("He fights.") == "He fights."
    assert f(f("Welcome to the world of X. Real line.")) == "Real line."
