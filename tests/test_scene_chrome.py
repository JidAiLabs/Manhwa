"""
tests/test_scene_chrome.py

TDD for tools/scene_chrome.py — deterministic detection of chapter CHROME:
publisher/studio logo pages, chapter-number cards, app-UI panels (view
counters), credits. Chrome must never be grouped, narrated, or shown
(user: ORV opened with 'Chapter 1' + 'VIEWS: 1' + a Redice Studio logo
narrated as story).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "scene_chrome",
    Path(__file__).resolve().parent.parent / "tools" / "scene_chrome.py",
)
sc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sc)  # type: ignore[union-attr]


def _is(ocr, **kw):
    item = {"ocr_clean": ocr, "text_only": kw.get("text_only", False),
            "text_coverage": kw.get("text_coverage", 0.1)}
    return sc.is_chrome_scene(item)


def test_view_counters_are_chrome():
    assert _is("VIEWS: 1\nVIEWS: 1\nVIEWS: 1") is True
    assert _is("views: 1525") is True


def test_publisher_studio_credits_are_chrome():
    assert _is("REDICE STUDIO\nPublished by Webtoon") is True
    assert _is("Author: Sing Shong\nAdaptation") is True
    assert _is("Translated by LINE Webtoon") is True


def test_bare_chapter_number_cards_are_chrome():
    assert _is("1", text_only=True) is True
    assert _is("Chapter 1") is True
    assert _is("EPISODE 12") is True
    assert _is("PROLOGUE") is True


def test_story_panels_are_not_chrome():
    assert _is("YOU MUST'VE HAD QUITE A LOT OF TROUBLE RUNNING") is False
    assert _is("") is False                       # pure-art panel
    # a story line that merely CONTAINS a number
    assert _is("HE SURVIVED FOR 10 YEARS IN THE TOWER") is False
    # dialogue mentioning reading/views in a sentence is story, not chrome
    assert _is("I'VE BEEN READING THIS FOR TEN YEARS, MY VIEWS NEVER CHANGED") is False


def test_cover_page_with_series_title_is_chrome():
    item = {"ocr_clean": "INFINITE EVOLUTION FROM ZERO", "text_only": False,
            "text_coverage": 0.2}
    assert sc.is_chrome_scene(item, series_title="Infinite Evolution From Zero") is True
    # partial-but-dominant title match still counts
    item2 = {"ocr_clean": "OMNISCIENT READER Webtoon", "text_only": False,
             "text_coverage": 0.15}
    assert sc.is_chrome_scene(item2, series_title="Omniscient Reader") is True


def test_title_words_in_dialogue_not_chrome():
    # a story line that happens to reuse title words inside a sentence
    item = {"ocr_clean": "THE READER KNEW THIS STORY WOULD END TODAY SOMEHOW, EVERY PATH CLOSED",
            "text_only": False, "text_coverage": 0.2}
    assert sc.is_chrome_scene(item, series_title="Omniscient Reader") is False


def test_in_story_app_panel_with_counters_is_NOT_chrome():
    # ORV's premise panel: the web-novel app list — counters present but
    # embedded in rich story content (title + episode rows). Must survive.
    ocr = ("THREE WAYS TO SURVIVE THE APOCALYPSE "
           "READ EPISODE 1383 COMMENTS: 1 VIEWS: 1 "
           "READ EPISODE 1384 COMMENTS: 1 VIEWS: 1 "
           "READ EPISODE 1385 COMMENTS: 1 VIEWS: 1")
    assert _is(ocr) is False


def test_bare_counter_walls_still_chrome():
    assert _is("VIEWS: 1 VIEWS: 1 VIEWS: 1 VIEWS: 1") is True


def test_counter_with_ocr_digit_confusion_is_chrome():
    # Vision reads "VIEWS: 1" as "VIEWS: I" (digit/letter confusion) — the
    # real ORV panel that survived the first filter pass.
    assert _is("[VIEWS: I") is True
    assert _is("VIEWS: I VIEWS: I VIEWS: l") is True


def test_empty_ocr_binary_card_is_chrome_with_image_stats():
    # the giant stylized "1" card: OCR-blind, but near-binary pixels betray it
    item = {"ocr_clean": "", "text_only": False, "text_coverage": 0.0}
    assert sc.is_chrome_scene(item, midtone_frac=0.04) is True     # binary card
    assert sc.is_chrome_scene(item, midtone_frac=0.45) is False    # real art
    # without image stats, empty OCR stays non-chrome (pure-art panels)
    assert sc.is_chrome_scene(item) is False
