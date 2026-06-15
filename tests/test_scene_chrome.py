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


def test_panel_kind_understanding_is_the_single_source_of_truth():
    # The multimodal understanding (stamped onto the vision item by
    # panel_understand) is authoritative — story_group/render_prep/prep_qa all
    # reach chrome through here, so this is where the recursion must stop.
    # 'story' is NEVER chrome, even with a watermark/counter OCR (the beast panel
    # whose only OCR was '1' that kept failing chrome_leak).
    assert sc.is_chrome_scene({"ocr_clean": "ELFTOON.com VIEWS: 1", "panel_kind": "story"}) is False
    assert sc.is_chrome_scene({"ocr_clean": "1", "panel_kind": "story"}) is False
    # 'chrome' is chrome regardless of OCR; 'caption' is content (not chrome)
    assert sc.is_chrome_scene({"ocr_clean": "a swordsman draws his blade", "panel_kind": "chrome"}) is True
    assert sc.is_chrome_scene({"ocr_clean": "BACK THEN, I HAD NO IDEA.", "panel_kind": "caption"}) is False
    # no panel_kind -> the OCR heuristic still runs (a clean end-card domain + plug)
    assert sc.is_chrome_scene({"ocr_clean": "thanks for reading, visit elftoon.com"}) is True


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


def test_site_plug_and_scanlation_credits_are_chrome():
    # the real IE cover: title + site plug + editor credit
    assert _is("INFINITE EVOLUTION FROM ZERO PLEASE READ THIS CHAPTER ON ELFTOON.COM ED: HAL") is True
    assert _is("READ FREE AT MANHWASITE.NET TL: JOE PR: AMY") is True


def test_read_this_dialogue_in_document_panel_not_chrome():
    # the real ORV novel-app screen: 30+ words of in-story reader UI; the
    # bubble line "WHY DOESN'T ANYONE READ THIS? IT'S A MASTERPIECE!" must not
    # trip the site-plug rule — plug phrases are chrome only on short banners
    item = {"ocr_clean": "THREE WAYS TO SURVIVE THE APOCALYPSE "
                         + "READ EPISODE COMMENTS VIEWS " * 6
                         + "WHY DOESN'T ANYONE READ THIS? IT'S A MASTERPIECE!",
            "text_only": True, "text_coverage": 0.233}
    assert sc.is_chrome_scene(item, series_title="Omniscient Reader") is False


def test_short_read_plug_banner_still_chrome():
    assert _is("PLEASE READ THIS CHAPTER ON OUR SITE") is True


def test_domain_plug_chrome_even_when_wordy():
    # the IE cover OCRs ~58 words (CJK credits etc.) but contains a domain —
    # domains/team-credit tags are chrome regardless of length
    ocr = ("INFINITE EVOLUTION FROM ZERO PLEASE READ THIS CHAPTER ON "
           "ELFTOON.COM ED: HAL PR: TL: HAL " + "WORD " * 40)
    assert _is(ocr) is True


def test_endcard_with_spaced_domain_and_plug_phrases_is_chrome():
    # the real IE p000096 end card: OCR splits the domain ("ELFTOON .com")
    # and is wordy — but "thanks for reading" / "join our discord" are
    # unmistakable publication plugs
    ocr = ("f on THANKS FOR READING THIS CHAPTER ON OUR WEBSITE ELFTOON .com "
           "DON'T HESITATE TO JOIN OUR DISCORD SERVER AND 1FT US")
    assert _is(ocr) is True


def test_single_watermark_domain_on_story_panel_not_chrome():
    # the IE p000039 case: aggregators stamp ELFTOON.COM ON story art — one
    # domain hit amid real dialogue is a watermark, not a chrome page
    assert _is("HOLY CRAP! I GET IT! HONEY, IS IT POSSIBLE THAT OUR SON IS "
               "A ONE-IN-A-MILLION PRODIGY? ELFTOON.COM") is False


def test_watermark_only_panel_midtone_decides():
    item = {"ocr_clean": "ELFTOON.COM", "text_only": False, "text_coverage": 0.01}
    assert sc.is_chrome_scene(item, midtone_frac=0.40) is False   # real art
    assert sc.is_chrome_scene(item) is True                       # no stats: banner


def test_needs_image_stats_for_empty_or_single_site_hit():
    assert sc.needs_image_stats("") is True
    assert sc.needs_image_stats("ELFTOON.COM") is True
    assert sc.needs_image_stats("A normal dialogue line.") is False


def test_vertical_title_ocr_garbage_is_chrome():
    # real ORV cover: vertical title letters OCR as fake words around the
    # one distinctive title word
    item = {"ocr_clean": "SR OMNISCIENT CE IA ED NEO TR", "text_only": False,
            "text_coverage": 0.1}
    assert sc.is_chrome_scene(item, series_title="Omniscient Reader") is True


def test_long_dialogue_using_title_word_not_chrome():
    item = {"ocr_clean": "HE FELT TRULY OMNISCIENT FOR ONE MOMENT TODAY AFTER READING EVERY SINGLE PAGE",
            "text_only": False, "text_coverage": 0.2}
    assert sc.is_chrome_scene(item, series_title="Omniscient Reader") is False


def test_korean_staff_credits_are_chrome():
    """The 나노마신 title card (Nano Machine, user report): hangul credits
    escaped every Latin-pattern rule."""
    assert _is("나노마신 喇勞魔神 그림 : 금강불괴 | 각색 : 현철무 | 원작 : 한중월야") is True


def test_single_korean_role_word_is_not_chrome():
    assert _is("그림 속의 검은 그림자가 움직인다") is False   # story prose using 그림
