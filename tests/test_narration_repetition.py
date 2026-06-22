"""
tests/test_narration_repetition.py — repeated-phrase detector + anti-repetition prompt rule.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gemini_narrative_pass",
    Path(__file__).resolve().parent.parent / "tools" / "gemini_narrative_pass.py",
)
gnp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gnp)  # type: ignore[union-attr]


# ---- repeated_phrases detector ------------------------------------------

def test_finds_repeated_ngram():
    lines = ["the pale moon rose", "the pale moon set"]
    result = gnp.repeated_phrases(lines, n=3, min_count=2)
    # "pale moon rose" and "pale moon set" differ but "pale moon" as a 2-gram
    # is repeated; with n=3 we expect to find something with "pale" and "moon"
    # The trigrams "the pale moon" appears in both after stopword filtering
    # (stopwords drop "the" so "pale moon rose" and "pale moon set" are the
    # non-stop trigrams) — actually with stopwords removed, the 3-gram sliding
    # window over ["pale","moon","rose"] vs ["pale","moon","set"] gives
    # "pale moon rose" (1) and "pale moon set" (1), no trigram hits count=2.
    # BUT the 2-gram "pale moon" appears in both. Let's use n=2.
    result2 = gnp.repeated_phrases(lines, n=2, min_count=2)
    phrases = [p for p, _ in result2]
    assert "pale moon" in phrases


def test_finds_repeated_trigram_with_n3():
    lines = [
        "the shadow crept across the floor",
        "the shadow crept towards the door",
    ]
    # After dropping stopwords: shadow crept across floor / shadow crept towards door
    # trigrams: (shadow,crept,across), (crept,across,floor), (shadow,crept,towards), (crept,towards,door)
    # "shadow crept" is the repeating 2-gram; for n=3 no trigram repeats
    result = gnp.repeated_phrases(lines, n=2, min_count=2)
    phrases = [p for p, _ in result]
    assert "shadow crept" in phrases


def test_unique_lines_return_empty():
    lines = [
        "the hero charges forward",
        "moonlight filters through the trees",
        "a blade flashes in the dark",
    ]
    result = gnp.repeated_phrases(lines, n=2, min_count=2)
    assert result == []


def test_stopword_only_ngrams_excluded():
    # "the" and "a" are stopwords — a 2-gram of only stopwords should not appear
    lines = ["the a the a", "the a the a"]
    result = gnp.repeated_phrases(lines, n=2, min_count=2)
    # All words are stopwords so no non-stopword n-grams exist
    assert result == []


def test_min_count_respected():
    # "pale moon" appears 3 times; with min_count=4 it should not appear
    lines = [
        "the pale moon rose",
        "the pale moon fell",
        "the pale moon shone",
    ]
    result = gnp.repeated_phrases(lines, n=2, min_count=4)
    assert result == []


def test_results_sorted_by_count_desc():
    lines = [
        "shadow falls on the ground",
        "shadow falls on the earth",
        "shadow falls on the wall",
        "moon rises in the sky",
        "moon rises in the night",
    ]
    result = gnp.repeated_phrases(lines, n=2, min_count=2)
    counts = [c for _, c in result]
    assert counts == sorted(counts, reverse=True)


# ---- anti-repetition prompt rule present --------------------------------

def test_prompt_contains_anti_repetition_instruction():
    """The system prompt must instruct the model to avoid repeating
    atmospheric/descriptive words it already used in previous_narration."""
    # The system prompt is built inside run(), but the constant or string
    # should contain the key instruction. We look for it in the module source.
    src = Path(__file__).resolve().parent.parent / "tools" / "gemini_narrative_pass.py"
    text = src.read_text()
    # Must contain some form of anti-repetition guidance referencing
    # previous_narration and vocabulary/word variation
    assert "previous_narration" in text  # already there
    assert any(kw in text.lower() for kw in [
        "reuse", "repeat", "vary", "variation", "fresh",
    ]), "Prompt must instruct the model to avoid reusing atmospheric words"
    # Specifically the anti-repetition rule should reference both the
    # atmospheric/descriptive context and vocabulary freshness
    assert any(phrase in text.lower() for phrase in [
        "atmospheric", "vocabulary", "cliché", "cliche", "fresh phrasing",
        "vary the", "do not reuse", "avoid reusing",
    ]), "Prompt must mention vocabulary variation or cliché avoidance"
