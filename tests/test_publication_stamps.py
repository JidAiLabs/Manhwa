"""Publication stamps: aggregator/scanlator text printed ON the page — a watermark
over story art, a release plug, a recruitment card, a credits roster, a "To Be
Continued" end card (owner 2026-10-13: the narrator read "Read at Asurascans.com
for the fastest releases" and called the hero "Asura" after the watermark).
Every string below is real OCR from the 2026-10-13 census of 91,346 panels."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import scene_chrome as sc  # noqa: E402


@pytest.mark.parametrize("ocr, keep", [
    ("KIM SHI HYUN! ASURASCANS.COM", "KIM SHI HYUN!"),
    ("THE SUCCESSORS DO NOT. ASURASCANS.COM EACH OF THEM HAS AN INDIVIDUAL GOAL.",
     "THE SUCCESSORS DO NOT. EACH OF THEM HAS AN INDIVIDUAL GOAL."),
    ("THIS IS MY SOUL DOMAIN. THE ENTIRE NAIYE KINGDOM IS MY LIFE FORCE. "
     "YOU CAN READ THE CHAPTER ON: EN-THUNDERSCANS.COM",
     "THIS IS MY SOUL DOMAIN. THE ENTIRE NAIYE KINGDOM IS MY LIFE FORCE."),
    ("AN READ THE CHAPTER ONI HUNDERSCANS.COM AND THE STANDARDS FOR TALENT ARE: COURAGE",
     "AND THE STANDARDS FOR TALENT ARE: COURAGE"),
    ("TO KILL EVERYONE?! alla READ ON DFlowScans", "TO KILL EVERYONE?! alla"),
    ("WILL YOU USE THE ORIGIN OF AK CHEON ON THE CHEONMA BLADE? [V/N] ASURASCANS.COM",
     "WILL YOU USE THE ORIGIN OF AK CHEON ON THE CHEONMA BLADE? [V/N]"),
    ("YOU'RE A FUCKING BOTHER RIGHT NOW. SE ASURAGG", "YOU'RE A FUCKING BOTHER RIGHT NOW. SE"),
    ("...HE'S COMING. GET READY. ASURA 6G", "...HE'S COMING. GET READY."),
    ("STOP IT...! ASURAIGG", "STOP IT...!"),
    ("DAMN IT...! > ASURA. GO", "DAMN IT...!"),          # BN ch2: the stamp read as GO
    ("U ALSO WANT THE ACHIEVEMENT, DON'T YOU? ASURASCAL", "U ALSO WANT THE ACHIEVEMENT, DON'T YOU?"),
    ("AsuraS cans @ discord. gg/asuran", ""),
    ("ASURA S GAUS", ""),
    ("WH-WHAT ARE YOU GOING TO DO...? ASURASGANS.COM WE JUST NEED TO WAIT A LITTLE.",
     "WH-WHAT ARE YOU GOING TO DO...? WE JUST NEED TO WAIT A LITTLE."),
    ("IT DID NOTHING TO ME. Kk lor ASURASCANS COM", "IT DID NOTHING TO ME. Kk lor"),
    ("I WON'T BE IN NING HAI CITY FOR A WHILE. ELFTOON com",
     "I WON'T BE IN NING HAI CITY FOR A WHILE."),
    ("MANGA YY.ORG FIRST, HE USED THE ASTRO-ELEMENT MAGE SKILL",
     "FIRST, HE USED THE ASTRO-ELEMENT MAGE SKILL"),
    ("HE'S HERE. DISCORD. GG/ASURAN", "HE'S HERE."),
    ("RUN! MEREAD. COM", "RUN!"),
    ("WHAT!? EN-THUNDERSCANS. COM", "WHAT!? EN"),
    ("AND MAKE HIM SUFFER JUST AS MUCH AS I DID. To Be Continued",
     "AND MAKE HIM SUFFER JUST AS MUCH AS I DID."),
    ("THIS DREADFUL SCENARIO BEGINS AT LAST. REDICE STUDIO Published by RVE RIVERSE Inc.",
     "THIS DREADFUL SCENARIO BEGINS AT LAST."),
])
def test_a_stamp_is_scrubbed_and_the_story_text_kept(ocr, keep):
    clean, stamps = sc.scrub_stamps(ocr)
    assert clean == keep
    assert stamps


@pytest.mark.parametrize("ocr", [
    "QC/N - IT SAYS 'EVERYONE, COME THIS WAY - VENERABLE ASURA IS OMNISCIENT AND LLL-KNOWING",
    "I'LL USE THE ASURA NULLIFYING FIST'S PRINCIPLE",
    "[THE DEMON KING, THE ARCHITECT EDIX OF DISCORD, HAS TAKEN NOTE OF YOUR PRESENCE]",
    "[YOUR ATTRIBUTE HAS BEEN ACTIVATED. YOU ARE NOW ABLE TO READ AT A FASTER SPEED.]",
    "50X [ALL PARTICIPANTS HAVE FINISHED RECRUITING THEIR HELPERS.] GLINT",
    "WOULD YOU LIKE TO FORM AN ALLIANCE WITH US? ...TO ACT AS A RECRUITER. WHY SHOULD WE?",
    "SENDER: TLS <TLS123@NAVER.COM> (ATTACHED FILE: 1)",
    "PIRATE IT...? PAY FOR IT IF YOU WANT TO READ IT, YOU BASTARDS. WEBTOONS TOO...",
    "IT'S A SKILL THAT LETS YOU FIND THE WAY USING A THREAD. TO BE HONEST, I TIED IT",
    "HE SCANS THE AREA. NOTHING.",
    "ASURA",
    "YOU SHOULD LEAVE. COM",          # a split watermark's fragment beside dialogue
    "THERE'S ONE PARTICIPANT. ORG",
])
def test_in_world_text_is_never_a_stamp(ocr):
    clean, stamps = sc.scrub_stamps(ocr)
    assert clean == ocr and stamps == []


def _item(ocr, kind="story"):
    return {"scene_file": "p1.jpg", "ocr_clean": ocr, "panel_kind": kind}


@pytest.mark.parametrize("ocr, kind, reason", [
    ("READ AT ASURASCANS.COM FOR THE FASTEST RELEASES", "system", "release_plug"),
    ("READ AT ASURA.GG FOR THE FASTEST RELEASES", "system", "release_plug"),
    ("Read at... for the fastest releases", "chrome", "release_plug"),
    ("We're now recruiting Korean Translators! Join our Discord to apply - Asura.gg/discord",
     "system", "recruitment"),
    ("PODOCO LECLO We're now recruiting Korean Translators! Join our Discord to apply - "
     "Asura.gg/discord", "system", "recruitment"),
    ("To Be Continue 원작 방구석 김씨 · 그림 블루 각색 콘티 채색 배경 편집 STUDIO", "story", "credits"),
    ("ASURA SCANS TRANSLATOR PROOFREADER CLEANER REDRAWER TYPESETTER QUALITY CHECKER "
     "THE TUTORIAL TOWER'S ADVANCED PLAYER 1SOCK SHAN BRAHIM", "empty", "credits"),
    ("STRONGEST ABANDONED SON PLEASE READ THIS CHAPTER ON ELFTOON.COM ED: ZEUS TL: SHARK "
     "PR: : ELFTDON.COM HTTPS:J/DISCORD.GG/ELFTDON", "empty", "credits"),
    ("READ ONLY AT SCANS SUPPORT THE TEAM BEHIND THE TRANSLATION READ ON OUR WEBSITE! "
     "JOIN US! WEBSITE DUSKSCANS.COM DISCORD https://discord.gg/9ankSaprHr", "system",
     "release_plug"),
    ("To be continued... Edited by Euwen", "caption", "end_card"),
    ("To Be Continued", "caption", "end_card"),
    ("This chapter is provided by: Saku READ ON DFlowScans", "story", "release_plug"),
])
def test_a_panel_that_IS_the_message_is_furniture_whatever_its_label(ocr, kind, reason):
    assert sc.furniture_reason(_item(ocr, kind)) == reason
    assert sc.is_chrome_scene(_item(ocr, kind)) is True


@pytest.mark.parametrize("ocr", [
    "KIM SHI HYUN! ASURASCANS.COM",
    "YOU CAN READ THE CHAPTER ON: EN-THUNDERSCANS.COM",     # FTA's stamp on bare art
    "@ ASURASCANS.COM",
    "You... 1144 ASURASCANS.COM",
    "AND MAKE HIM SUFFER JUST AS MUCH AS I DID. To Be Continued",
    "WILL YOU USE THE ORIGIN OF AK CHEON ON THE CHEONMA BLADE? [V/N] ASURASCANS.COM",
    "50X [ALL PARTICIPANTS HAVE FINISHED RECRUITING THEIR HELPERS.] GLINT",
    "GO, CHILD!! READ AT ASURASCANS.COM FOR THE FASTEST RELEASES",   # DDM ch11
    "PRINCESS?! !!! READ AT ASURASCANS.COM FOR THE FASTEST RELEASES",
    "READ ONLY AT MADARASCANS.COM",                     # Clan: stamped on wordless art
    "HASHIMOTO. I'M HERE. READ ONLY AT MADARASCANS.COM",
    "GEUMHO REDICE STUDIO Published by RV RIVERSE Inc.",  # ORV: credit over the last panel
    "Published by RIVE RIVERSE Inc.",
    "REMAINING TIME: 8 DAYS REPOOR X 3B2S REDICE STUDIO Published by RV RIVERSE Inc.",
])
def test_a_stamp_on_a_story_panel_never_makes_it_furniture(ocr):
    assert sc.furniture_reason(_item(ocr)) is None


def test_a_stamp_alone_never_drops_a_panel_even_a_dark_one():
    # Clan ch11 p000057: a dark panel of a door creaking open whose only OCR is the
    # stamp (midtone 0.05) — a near-binary bar would have dropped real art
    assert sc.furniture_reason(_item("READ ONLY AT MADRRASCANS.COM"), midtone_frac=0.05) is None
    assert sc.furniture_reason(_item("MANGAYY.ORG READ ON DFlowScans"), midtone_frac=0.03) is None


def test_a_to_be_continued_stamp_over_real_art_is_not_an_end_card():
    # the art proves itself with midtones; a black end card has none
    assert sc.furniture_reason(_item("To Be Continued"), midtone_frac=0.4) is None
    assert sc.furniture_reason(_item("To Be Continued"), midtone_frac=0.02) == "end_card"


def test_a_scrubbed_item_is_judged_from_its_recorded_watermarks():
    # vision_extract stores the scrubbed text + the stamps; the decision is the same
    raw = "READ AT ASURASCANS.COM FOR THE FASTEST RELEASES"
    clean, stamps = sc.scrub_stamps(raw)
    item = {"scene_file": "p1.jpg", "ocr_clean": clean, "watermarks": stamps,
            "panel_kind": "system"}
    assert sc.furniture_reason(item) == "release_plug"


@pytest.mark.parametrize("line, hit", [
    ("Read at. Asurascans.com for the fastest releases.", True),
    ("The story will continue.", True),
    ("To be continued...", True),
    ("The team is now recruiting Korean translators, so join the Discord at "
     "Asura.gg/discord to apply!", True),
    ("Successfully advanced. To o-class nucled warrior. You can read the chapter on. "
     "En-thunderscans.com.", True),
    ("Asuracomic.Net.", True),
    ("A stranger sneers that you also want the achievement, Asurascals, and tells him off.",
     True),
    ("As a rocky golem slams its arm down, Asura screams, 'No!'", False),
    ("Asura grits his teeth, muttering, 'D-damn it...'", False),
    ("He decides to check the attached file first, allowing him to read at a faster speed.",
     False),
    ("The system announces that the Demon King, the Architect of Discord, has taken note "
     "of him.", False),
    ("He's wondering if recruiting someone this powerful could change the selection.", False),
    ("Venerable Asura is omniscient, the old man insists.", False),
    ("The story of his family continues to haunt him.", False),
])
def test_a_voiced_line_with_publication_text_is_caught(line, hit):
    assert bool(sc.voiced_furniture(line)) is hit


def test_a_name_printed_only_in_watermarks_is_found():
    # Boundless Necromancer: "Asura" is on its pages ONLY as the aggregator stamp
    items = [{"scene_file": "p1.jpg", "panel_kind": "story", "ocr_clean": "STOP IT...! ASURA.GG"},
             {"scene_file": "p2.jpg", "panel_kind": "story", "ocr_clean": "DAMN IT! ASURA 6G"},
             {"scene_file": "p3.jpg", "panel_kind": "story", "ocr_clean": "WELL DONE, PRINCE CHEON."},
             {"scene_file": "p4.jpg", "panel_kind": "chrome", "ocr_clean": "TL CHEON-NIM ASURA"}]
    assert sc.watermark_names(["Asura", "Prince Cheon", "our protagonist", "Kim"], items) == {"Asura"}
    # TT: 'Venerable Asura' is printed in the story — a real name
    items.append({"scene_file": "p5.jpg", "panel_kind": "story",
                  "ocr_clean": "VENERABLE ASURA IS OMNISCIENT"})
    assert sc.watermark_names(["Asura"], items) == set()
