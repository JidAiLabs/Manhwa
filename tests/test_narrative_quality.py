"""
tests/test_narrative_quality.py

Unit tests for the narrative-quality overhaul of tools/script_expander.py.

Covers PURE helpers and prompt assembly only — NO live LLM calls:
  - R2: no-OCR-echo / no-repetition rules present; old "MAY quote" line gone.
  - R3: build_story_so_far() compact running synopsis + continuity block.
  - R4: manhwa jargon lexicon injected; BANNED_PHRASES persona reconciliation.
  - Flashback / time-shift rule present.
  - Genre flavor still injects for the hunter genre.

The module imports `openai` at import time but only constructs the client inside
main(), so importing it here is safe and makes no network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

# tools/ lives at repo root; add it to sys.path so the import works without
# installing the package.
_TOOLS_DIR = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR))

import script_expander as se  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: assemble the system prompt exactly like main() does (minus the LLM).
# ---------------------------------------------------------------------------
def _assemble_system(genre_mode: str = "hunter", word_target: int = 600, tol: float = 0.10) -> str:
    trope_lines = se._trope_lines_for_genre(genre_mode)
    genre_flavor = "\n".join(f"- {t}" for t in trope_lines)
    return (
        se.ENHANCED_SYSTEM_TEMPLATE
        .replace("{JARGON_LEXICON}", se._manhwa_jargon_block())
        .replace("{WORD_TARGET}", str(word_target))
        .replace("{TOL_PCT}", str(int(tol * 100)))
        .replace(
            "{GENRE_FLAVOR}",
            "=== GENRE FLAVOR (ONLY WHEN SUPPORTED BY VISUALS/OCR) ===\n" + genre_flavor,
        )
    )


# ---------------------------------------------------------------------------
# R3 — build_story_so_far
# ---------------------------------------------------------------------------
def test_build_story_so_far_empty_returns_empty():
    assert se.build_story_so_far([]) == ""
    assert se.build_story_so_far(None) == ""  # type: ignore[arg-type]


def test_build_story_so_far_compact_synopsis():
    prior = [
        {
            "section_summary": "The MC awakens as an F-rank hunter inside a collapsing gate.",
            "cliffhanger_line": "Then the gate sealed behind him.",
        },
        {
            "section_summary": "Years earlier, he had buried his mentor after the first dungeon break.",
            "cliffhanger_line": "A voice called his name from the dark.",
        },
    ]
    out = se.build_story_so_far(prior, max_chars=600)
    assert out  # non-empty
    assert len(out) <= 600
    # Pulls from summaries and cliffhangers.
    assert "F-rank hunter" in out
    assert "gate sealed" in out
    assert "voice called his name" in out


def test_build_story_so_far_falls_back_to_paragraphs():
    prior = [
        {
            "section_summary": "",
            "script_paragraphs": ["The antagonist sneered at the awakener.", "Mana surged."],
            "cliffhanger_line": "",
        }
    ]
    out = se.build_story_so_far(prior)
    assert "antagonist sneered" in out
    assert "Mana surged" in out


def test_build_story_so_far_truncates_to_max_chars():
    prior = [{"section_summary": "word " * 500, "cliffhanger_line": ""}]
    out = se.build_story_so_far(prior, max_chars=120)
    assert len(out) <= 120
    assert out.endswith("…")


# ---------------------------------------------------------------------------
# R2 / R3 / R4 / flashback — prompt assembly content checks
# ---------------------------------------------------------------------------
def test_prompt_contains_jargon_lexicon():
    sys_prompt = _assemble_system()
    # A representative spread of the curated vocabulary.
    for term in ("protagonist", "antagonist", "aura farming", "face-slap", "regressor", "dantian"):
        assert term in sys_prompt, f"missing jargon term: {term}"


def test_prompt_contains_continuity_block():
    sys_prompt = _assemble_system()
    assert "=== CONTINUITY ===" in sys_prompt
    assert "STORY SO FAR" in sys_prompt
    assert "do NOT" in sys_prompt and "re-introduce" in sys_prompt


def test_prompt_contains_no_ocr_quote_rule():
    sys_prompt = _assemble_system()
    assert "NEVER quote UI/interface text" in sys_prompt
    assert "view counts" in sys_prompt
    assert "PARAPHRASE dialogue" in sys_prompt
    assert "NEVER repeat the same phrase" in sys_prompt


def test_prompt_contains_flashback_rule():
    sys_prompt = _assemble_system()
    assert "=== FLASHBACK / TIME-SHIFT ===" in sys_prompt
    assert "FLASHBACK" in sys_prompt
    assert "Years earlier" in sys_prompt


def test_prompt_contains_noisy_ocr_rule():
    sys_prompt = _assemble_system()
    assert "NOISY hint" in sys_prompt
    assert "'I' for '1'" in sys_prompt


def test_prompt_drops_old_may_quote_line():
    sys_prompt = _assemble_system()
    assert "you MAY quote short fragments" not in sys_prompt
    assert "MAY quote short fragments" not in sys_prompt
    # The OCR-quote anchor option must be gone from VISUAL ANCHORING.
    assert "An OCR quote (1–6 words) in quotes" not in sys_prompt


# ---------------------------------------------------------------------------
# R4 — BANNED_PHRASES / validator reconciliation
# ---------------------------------------------------------------------------
def test_persona_terms_no_longer_banned():
    for allowed in ("our protagonist", "our hero", "the character", "the characters"):
        assert allowed not in se.BANNED_PHRASES
    # Generic camera-speak we/our/us forms are still banned.
    for still_banned in ("we see", "we witness", "we watch"):
        assert still_banned in se.BANNED_PHRASES


def test_mc_phrase_passes_quality_validator():
    issues = se.validate_paragraph_quality(
        ["The MC stood tall as the antagonist faltered, blood dripping from his clenched fist."]
    )
    assert issues == [], f"unexpected quality issues: {issues}"


def test_protagonist_phrase_passes_quality_validator():
    issues = se.validate_paragraph_quality(
        ["The protagonist gripped his blade while the antagonist sneered from the gate."]
    )
    assert issues == [], f"unexpected quality issues: {issues}"


def test_camera_phrase_still_fails_validator():
    issues = se.validate_paragraph_quality(["The camera pans across the silent battlefield slowly."])
    assert any("camera" in i for i in issues), f"camera phrase not flagged: {issues}"


# ---------------------------------------------------------------------------
# Genre flavor still injects for hunter.
# ---------------------------------------------------------------------------
def test_genre_flavor_injects_for_hunter():
    sys_prompt = _assemble_system("hunter")
    assert "Genre: hunter/system fantasy" in sys_prompt
    assert "Aura Farming" in sys_prompt
    assert "Face-Slapping trope" in sys_prompt
