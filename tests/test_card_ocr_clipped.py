"""ocr_looks_clipped must not mistake ordinary English for a cropped card
(2026-09-14).

The heuristic calls a token a crop fragment when it is NOT in the system word
list but IS a prefix of a word there ("scenar" -> scenario). That word list
lacks inflected forms and common irregulars, so "has" (-> hasan), "coins",
"monsters", "completed" (-> completedness) and "paid" (-> paideutic) all read
as fragments. Measured over the corpus: 396 of 1,329 OCR-only solo system
cards were judged clipped, and the writer deliberately keeps its own
paraphrase over a clipped card — so a clean card was never read aloud.
De-inflecting against the same word list brings that to 84, and the real
cropped card (Ch130: "Main scenar abandoned w ... to extin") is still caught.
"""
import sys

sys.path.insert(0, "tools")
import gemini_narrative_pass as gnp  # noqa: E402

# base forms ONLY: inflected, irregular and contracted forms are what the host
# word list is missing, so the test must not supply them either
_BASE_WORDS = ["abandon", "activate", "area", "been", "body", "category",
               "clear", "coin", "complete", "condition", "constellation",
               "create", "day", "difficulty", "drive", "effect", "extinction",
               "failure", "hasan", "heroine", "increase", "limit", "main",
               "monster", "paideutic", "penalty", "planet", "point", "quest",
               "reward", "scenario", "the", "time", "university", "use",
               "weapon", "your", "completedness"]


def _dict(monkeypatch, tmp_path, words=_BASE_WORDS):
    p = tmp_path / "words"
    p.write_text("\n".join(words))
    gnp._dict_words.cache_clear()
    monkeypatch.setattr(gnp, "_DICT_PATH", str(p))


def test_inflected_and_irregular_words_are_not_crop_fragments(monkeypatch,
                                                            tmp_path):
    _dict(monkeypatch, tmp_path)
    clean = ("YOUR REWARD HAS BEEN PAID. THE QUEST IS COMPLETED, COINS ARE "
             "CREATED AND YOUR POINTS HAVE INCREASED USING THE WEAPONS OF "
             "THE MONSTERS.")
    assert not gnp.ocr_looks_clipped(clean)


def test_the_word_test_knows_inflections_contractions_and_irregulars(
        monkeypatch, tmp_path):
    # asserted on the word test itself: whether a host list happens to carry a
    # longer word starting "doesn"/"women" must not decide this test
    _dict(monkeypatch, tmp_path)
    for word in ("coins", "monsters", "completed", "created", "increased",
                 "using", "points", "has", "paid", "held", "doesn", "wasn",
                 "women"):
        assert gnp._is_card_word(word), word
    for fragment in ("scenar", "extin", "conste", "heroi", "univ"):
        assert not gnp._is_card_word(fragment), fragment


def test_a_real_crop_is_still_caught(monkeypatch, tmp_path):
    _dict(monkeypatch, tmp_path)
    # the Ch130 card the heuristic exists for: words cut at the crop edge
    assert gnp.ocr_looks_clipped(
        "Main scenar abandoned w category: main difficulty: s clear "
        "conditions: drive th planet p 97815t pl/ to extin time limit: 40 "
        "days rewards: 200,000 coins,??? penalty for failure: -")
    # fragments that de-inflect to nothing stay fragments
    assert gnp.ocr_looks_clipped("THE CONSTE HEROI OF THE UNIV AREA")
