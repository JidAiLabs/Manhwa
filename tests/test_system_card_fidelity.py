"""System cards are READ ALOUD (2026-09-14).

Measured over the processed corpus: of 201 solo system-card segments with at
least 4 card words, 92 (46%) voiced under half of the card's printed words and
50 (25%) under a quarter — typically a 12-word paraphrase standing in for a
32-word card. The writer kept the model's line whenever it shared ONE word with
the card, and no QA gate checked voicing at all (system_coverage_flags only
checks that the panel is SHOWN). ORV Ep207 g17 after the word-cap fix: two
cards printing 38 and 50 words, voiced at 21% and 53%.
"""
import sys

sys.path.insert(0, "tools")
import gemini_narrative_pass as gnp  # noqa: E402
import narration_heal  # noqa: E402

CARD = ("[THE CONSTELLATION 'DEMON-LIKE JUDGE OF FIRE' IS WATCHING YOU WITH "
        "INTEREST. THE SUB SCENARIO HAS BEEN COMPLETED AND YOUR REWARD WILL "
        "BE PAID SHORTLY.]")


def _u(card):
    return {"p1.jpg": {"scene_file": "p1.jpg", "panel_kind": "system",
                       "dialogue": "", "ocr_clean": card}}


def _no_dictionary(monkeypatch, tmp_path):
    # ocr_looks_clipped fails open without a dictionary: the card is clean
    monkeypatch.setattr(gnp, "_DICT_PATH", str(tmp_path / "missing"))


def test_a_paraphrase_voicing_under_half_the_card_is_replaced_by_the_card(
        monkeypatch, tmp_path):
    _no_dictionary(monkeypatch, tmp_path)
    line = "A watchful constellation takes an interest as the scenario ends."
    out = gnp.system_card_line("p1.jpg", _u(CARD), line)
    assert out != line
    assert "judge of fire" in out.lower() and "reward" in out.lower()


def test_a_voicing_of_at_least_half_the_card_is_kept(monkeypatch, tmp_path):
    _no_dictionary(monkeypatch, tmp_path)
    line = ("The Demon-like Judge of Fire is watching you with interest; the "
            "sub scenario has been completed and your reward comes soon.")
    assert gnp.system_card_line("p1.jpg", _u(CARD), line) == line


def test_the_half_card_boundary_is_inclusive(monkeypatch, tmp_path):
    _no_dictionary(monkeypatch, tmp_path)
    card = "[SKILL ACQUIRED: IRON BODY]"               # 4 card words
    half = "He gains the iron body."                   # iron, body -> 2/4
    assert gnp.system_card_line("p1.jpg", _u(card), half) == half
    quarter = "He gains an iron will."                 # iron -> 1/4
    out = gnp.system_card_line("p1.jpg", _u(card), quarter)
    assert out != quarter and "iron body" in out.lower()


def test_a_short_card_keeps_the_one_shared_word_rule(monkeypatch, tmp_path):
    # too few words to measure a share: one shared word still counts as voicing
    _no_dictionary(monkeypatch, tmp_path)
    line = "His level climbs as the window flashes."
    assert gnp.system_card_line("p1.jpg", _u("[LEVEL UP]"), line) == line


def test_card_voiced_share_measures_the_cards_words():
    assert gnp.card_voiced_share("He gains the iron body.",
                                 "SKILL ACQUIRED: IRON BODY") == 0.5
    assert gnp.card_voiced_share("anything", "LEVEL UP") is None   # < 4 words


def test_the_heal_turns_an_unvoiced_card_into_a_correction():
    report = {"flags": [{
        "code": "system_card_unvoiced", "severity": "WARN",
        "segment_id": "g0017", "scene": "p000079.jpg",
        "detail": "the line voices 21% of the card's printed words (6/28)"}]}
    notes = narration_heal.corrections_from_qa(report)
    assert list(notes) == [17]
    assert "READ THE CARD" in notes[17]
