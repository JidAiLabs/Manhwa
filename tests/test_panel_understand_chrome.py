"""panel_understand: the in-world rescue must NOT promote publication chrome
(scanlator credit / recruitment / Discord / "thanks for reading" cards) to
story, even when such a card carries dialogue-like text. A genuine in-world
chat / game-UI screen with dialogue MUST still be promoted (the real rescue).

The bug: Ch141 p000068 — "A promotional recruitment card for Korean
translators (join our Discord to apply)" was classified story w/ dialogue and
narrated as a story beat. The drop mechanism (story_group.nonstory_files) only
fires on panel_kind chrome/empty, so an ad mislabeled story is never a drop
candidate. The fix keeps it chrome.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "panel_understand",
    Path(__file__).resolve().parent.parent / "tools" / "panel_understand.py")
pu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pu)  # type: ignore[union-attr]


# --- the chrome-signal text gate -------------------------------------------

def test_looks_like_chrome_furniture_matches_recruitment_and_promo():
    f = pu._looks_like_chrome_furniture
    assert f("A promotional recruitment card for Korean translators.")
    assert f("Join our Discord to apply as a typesetter!")
    assert f("Support us on Patreon for early chapters.")
    assert f("Thanks for reading! See you next chapter.")
    assert f("We are recruiting proofreaders and redrawers.")
    assert f("Read the rest on our website, AsuraToon.")
    assert f("Translated by ElfToon scanlation team.")


def test_looks_like_chrome_furniture_ignores_inworld_story_text():
    f = pu._looks_like_chrome_furniture
    # genuine in-world dialogue / UI must NOT trip the chrome gate
    assert not f("WHY DOESN'T ANYONE READ THIS? IT'S A MASTERPIECE!")
    assert not f("The hero draws his blade as the beast lunges.")
    assert not f("STATUS WINDOW — LEVEL 5, HP 200, MANA 80.")
    assert not f("I never thought it would end like this.")


# --- the rescue must defer to the chrome gate -------------------------------

def _det_balloon():
    """A confident, compact balloon detection (the in-world signal)."""
    return [(56, 768, 499, 1054, 0.96)]


def test_rescue_keeps_recruitment_card_as_chrome_even_with_balloon():
    # p000068: a chrome-classified recruitment card that ALSO has a balloon +
    # dialogue. The balloon would normally promote chrome->story; the chrome
    # text signal must veto that promotion -> stays chrome -> story_group drops it.
    panels = [{
        "scene_file": "p000068.jpg",
        "panel_kind": "chrome",
        "dialogue": "JOIN OUR DISCORD TO APPLY",
        "description": "A promotional recruitment card for Korean translators.",
        "action": "the card invites readers to join the team",
        "subjects": [],
    }]
    items = [{"scene_file": "p000068.jpg", "scene_path": "/s/p000068.jpg"}]
    n = pu.apply_inworld_screen_overrides(
        panels, items,
        detect_fn=lambda sp: (736, 1169, _det_balloon()),
        log=lambda _m: None)
    assert n == 0
    assert panels[0]["panel_kind"] == "chrome"


def test_rescue_still_promotes_genuine_inworld_screen_with_dialogue():
    # ORV p000003: an in-world reader-app screen with a real comment balloon and
    # NO chrome signal -> the rescue must STILL promote chrome->story.
    panels = [{
        "scene_file": "p000003.jpg",
        "panel_kind": "chrome",
        "dialogue": "WHY DOESN'T ANYONE READ THIS? IT'S A MASTERPIECE!",
        "description": "A phone screen showing an episode list for a web novel.",
        "action": "a reader scrolls the novel's episode list",
        "subjects": [],
    }]
    items = [{"scene_file": "p000003.jpg", "scene_path": "/s/p000003.jpg"}]
    n = pu.apply_inworld_screen_overrides(
        panels, items,
        detect_fn=lambda sp: (736, 1169, _det_balloon()),
        log=lambda _m: None)
    assert n == 1
    assert panels[0]["panel_kind"] == "story"
    assert any("in-world screen" in str(s).lower()
               for s in panels[0]["subjects"])


# --- structural demotion: a credits/cover card mislabeled 'story' -> chrome --

def test_furniture_gate_matches_creator_credits():
    f = pu._looks_like_chrome_furniture
    assert f("Nano machine AUTOR HAN JOONG WUEOL YA  ARTISTA GUEM GANG BUL GAE")
    assert f("Story by Kim · Art by Lee")
    assert f("Illustrated by Studio Redice")
    # in-world status / skill text must STILL be invisible to the gate
    assert not f("STATUS WINDOW LEVEL 5 HP 200 QUEST NOTIFICATION")
    assert not f("7TH GENERATION NANO MACHINE, STARTING ACTIVATION")


def test_demotes_credits_card_story_to_chrome():
    # The Nano-Machine end-card: Gemma read the stylized art as 'story', but the
    # OCR carries the creator credits -> demote to chrome so the grouper drops it.
    panels = [{
        "scene_file": "p000021.jpg", "panel_kind": "story", "dialogue": "",
        "description": "A title card with stylized lettering and a silhouette.",
        "action": "the chapter title screen", "subjects": [],
    }]
    items = [{"scene_file": "p000021.jpg", "scene_path": "/s/p000021.jpg",
              "ocr_clean": "Nano machine AUTOR HAN JOONG WUEOL YA ARTISTA GUEM GANG"}]
    pu.apply_inworld_screen_overrides(
        panels, items, detect_fn=lambda sp: None, log=lambda _m: None)
    assert panels[0]["panel_kind"] == "chrome"


def test_demotion_never_touches_a_system_panel():
    # THE SAFETY GUARANTEE: a plot-critical in-world system window survives, even
    # though it is UI-like — its OCR carries no creator-credit vocabulary.
    panels = [{
        "scene_file": "p000007.jpg", "panel_kind": "system",
        "dialogue": "QUEST: DEFEAT THE STEEL-FANGED LYCAN",
        "description": "A glowing in-world status window.",
        "action": "a system notification appears", "subjects": [],
    }]
    items = [{"scene_file": "p000007.jpg", "scene_path": "/s/p000007.jpg",
              "ocr_clean": "STATUS LEVEL 5 HP 200 NOTIFICATION QUEST DIRECTIONS"}]
    pu.apply_inworld_screen_overrides(
        panels, items, detect_fn=lambda sp: None, log=lambda _m: None)
    assert panels[0]["panel_kind"] == "system"      # untouched


def test_demotion_never_touches_a_normal_story_panel():
    panels = [{
        "scene_file": "p000010.jpg", "panel_kind": "story",
        "dialogue": "I'll protect you, no matter what.",
        "description": "The hero shields his ally from the blast.",
        "action": "the hero raises his blade", "subjects": [],
    }]
    items = [{"scene_file": "p000010.jpg", "scene_path": "/s/p000010.jpg",
              "ocr_clean": "I'LL PROTECT YOU NO MATTER WHAT"}]
    pu.apply_inworld_screen_overrides(
        panels, items, detect_fn=lambda sp: None, log=lambda _m: None)
    assert panels[0]["panel_kind"] == "story"        # untouched
