"""A system card is read aloud only when its text can be SPOKEN (2026-09-15).

Two chapters stopped on a card rule that forced printed text into the
narration without asking whether it is language or a finished thought:
- Wimp Ch21 p000028: an hourglass ringed by Roman numerals. Gemma recorded no
  printed text (correct: it is a picture); Apple's text recognition read the
  numeral ring as "+ IIX", and the rule replaced the writer's line with it.
  The voice cannot say "Iix." (three dead takes), so voicing blocked.
- Wimp Ch19 p000020: a heading split across two panels, "[PHYSICAL" |
  "[TRAITS AND". The rule forced "Traits and.", QA blocked it as cut
  mid-sentence, and heal could not change it because the rule put it back.
Across 1,852 solo cards the decoration case is 4 lines ("+ IIX", "#BI-90594
XOH", "VING", "9,995"), and the writer's own line was the better one in all 4.
"""
import sys

sys.path.insert(0, "tools")
import gemini_narrative_pass as gnp  # noqa: E402

_WORDS = ["an", "the", "on", "hourglass", "timer", "appear", "interface",
          "level", "up", "he", "grow", "strong", "notice", "carry", "across",
          "next", "screen", "you", "have", "fail", "to", "clear", "scenario",
          "reveal", "land", "flash", "red"]


def _dict(monkeypatch, tmp_path, words=_WORDS):
    p = tmp_path / "words"
    p.write_text("\n".join(words))
    gnp._dict_words.cache_clear()
    monkeypatch.setattr(gnp, "_DICT_PATH", str(p))


def _u(dialogue="", ocr=""):
    return {"p1.jpg": {"scene_file": "p1.jpg", "panel_kind": "system",
                       "dialogue": dialogue, "ocr_clean": ocr}}


def test_a_numeral_ring_is_never_read_over_the_writers_line(monkeypatch,
                                                           tmp_path):
    _dict(monkeypatch, tmp_path)
    line = "An hourglass timer appears on the interface."
    assert gnp.system_card_line("p1.jpg", _u(ocr="+ IIX"), line) == line


def test_codes_and_bare_numbers_are_not_card_text(monkeypatch, tmp_path):
    _dict(monkeypatch, tmp_path)
    line = "The next screen flashes red."
    for ocr in ("#BI-90594 XOH", "9,995"):
        assert gnp.system_card_line("p1.jpg", _u(ocr=ocr), line) == line, ocr


def test_a_writer_line_that_only_copies_the_marks_gets_the_grounded_line(
        monkeypatch, tmp_path):
    _dict(monkeypatch, tmp_path)
    u = _u(ocr="+ IIX")
    out = gnp.system_card_line("p1.jpg", u, "Iix.")
    assert "iix" not in out.lower()
    assert out == gnp._grounded_pad_line("p1.jpg", u)


def test_without_a_word_list_nothing_is_called_decoration(monkeypatch,
                                                         tmp_path):
    # fail open, like ocr_looks_clipped: a missing list must not silence every
    # OCR-only card in the corpus
    monkeypatch.setattr(gnp, "_DICT_PATH", str(tmp_path / "missing"))
    out = gnp.system_card_line("p1.jpg", _u(ocr="+ IIX"),
                               "An hourglass timer appears.")
    assert "iix" in out.lower()


def test_a_card_that_stops_mid_phrase_is_never_forced(monkeypatch, tmp_path):
    _dict(monkeypatch, tmp_path)
    line = "The notice carries on across the next screen."
    assert gnp.system_card_line("p1.jpg", _u(dialogue="[TRAITS AND"),
                                line) == line


def test_a_complete_card_is_still_read_aloud(monkeypatch, tmp_path):
    _dict(monkeypatch, tmp_path)
    out = gnp.system_card_line(
        "p1.jpg", _u(dialogue="[YOU HAVE FAILED TO CLEAR THE SCENARIO.]"),
        "The screen flashes red.")
    assert out.startswith("You have failed to clear the scenario")


def test_an_ocr_only_card_made_of_words_is_still_read_aloud(monkeypatch,
                                                           tmp_path):
    _dict(monkeypatch, tmp_path)
    out = gnp.system_card_line("p1.jpg", _u(ocr="[LEVEL UP]"),
                               "He grows stronger.")
    assert out.lower().startswith("level up")


def test_a_card_gemma_transcribed_is_trusted_without_the_word_list(
        monkeypatch, tmp_path):
    # BOTH signals are needed to call a card decoration: here Gemma read words
    _dict(monkeypatch, tmp_path)
    out = gnp.system_card_line("p1.jpg", _u(dialogue="TADA~"),
                               "The reveal lands.")
    assert out.lower().startswith("tada")
