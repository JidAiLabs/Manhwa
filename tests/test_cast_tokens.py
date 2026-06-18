"""cast-token resolution (gemini_narrative_pass): the narration model sometimes
COPIES a bracketed cast token like [protagonist]/[antagonist] straight from the
cast block into the final narration, and the TTS then voices the literal token.

These tests pin:
  1. _build_cast_block no longer emits a bracketed [role] (uses (role) instead),
  2. _resolve_cast_tokens replaces a bracketed cast token with that member's
     reference (a proper-name alias if one exists, else canonical_name),
  3. stray/unknown bracket tokens are stripped to readable text (never blanked).
Stubbed model — no Gemma/network — in the importlib style of the sibling test."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gemini_narrative_pass",
    Path(__file__).resolve().parent.parent / "tools" / "gemini_narrative_pass.py")
gnp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gnp)  # type: ignore[union-attr]


def _cast():
    return [
        {"id": "our_protagonist", "role": "protagonist",
         "canonical_name": "our protagonist", "aliases": ["Cheon Mu Geum"]},
        {"id": "antagonist", "role": "antagonist",
         "canonical_name": "the antagonist", "aliases": ["this bastard"]},
    ]


def test_resolve_bracket_tokens_to_name_or_canonical():
    # protagonist -> proper-name alias; antagonist -> canonical (alias is a slur,
    # not a proper name)
    out = gnp._resolve_cast_tokens(
        "[protagonist] strikes [antagonist] hard", _cast())
    assert out == "Cheon Mu Geum strikes the antagonist hard"


def test_resolve_matches_id_and_canonical_tokens():
    # a token can also be the member's id (our_protagonist) or canonical phrase
    out = gnp._resolve_cast_tokens("[our_protagonist] and [our protagonist]", _cast())
    assert out == "Cheon Mu Geum and Cheon Mu Geum"
    assert "[" not in out and "]" not in out


def test_resolve_is_case_insensitive():
    out = gnp._resolve_cast_tokens("[Protagonist] and [ANTAGONIST]", _cast())
    assert out == "Cheon Mu Geum and the antagonist"


def test_resolve_strips_unknown_token_to_inner_text():
    # an unknown bracket token is cleaned (no brackets remain); never blanked
    out = gnp._resolve_cast_tokens("[someone] runs", _cast())
    assert "[" not in out and "]" not in out
    assert "runs" in out
    assert out.strip() != ""


def test_resolve_handles_possessive_token():
    # the live bug had "He recalls [protagonist]'s lack of internal energy."
    out = gnp._resolve_cast_tokens(
        "He recalls [protagonist]'s lack of internal energy.", _cast())
    assert out == "He recalls Cheon Mu Geum's lack of internal energy."
    assert "[" not in out


def test_resolve_no_brackets_is_noop():
    text = "Cheon Mu Geum strikes the antagonist hard."
    assert gnp._resolve_cast_tokens(text, _cast()) == text


def test_resolve_empty_cast_still_strips_brackets():
    # no cast at all: stray tokens still get cleaned to inner text, never blanked
    out = gnp._resolve_cast_tokens("[protagonist] runs", [])
    assert "[" not in out and "]" not in out
    assert "runs" in out


def test_resolve_never_blanks_line():
    out = gnp._resolve_cast_tokens("[protagonist]", _cast())
    assert out.strip() != ""
    assert out == "Cheon Mu Geum"


def test_proper_name_alias_selection():
    # the alias-picker accepts a capitalized 1-4 token proper name, rejects
    # phrases with generic/role words.
    assert gnp._proper_name_alias(["Cheon Mu Geum"]) == "Cheon Mu Geum"
    assert gnp._proper_name_alias(["this bastard"]) is None
    assert gnp._proper_name_alias(["the old man"]) is None
    assert gnp._proper_name_alias(["a young guy"]) is None
    assert gnp._proper_name_alias([]) is None
    # first qualifying alias wins
    assert gnp._proper_name_alias(["that guy", "Jin Woo"]) == "Jin Woo"


def test_cast_block_has_no_bracket_role():
    # the cast block renders the role as (protagonist), NOT [protagonist] — so the
    # model has no bracket token to copy into the narration.
    import tempfile
    import os
    cast_json = {"cast": [
        {"id": "our_protagonist", "role": "protagonist",
         "canonical_name": "our protagonist", "aliases": ["Cheon Mu Geum"],
         "visual_description": "a young swordsman"},
    ]}
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cast_json, f)
        block = gnp._build_cast_block(path)
    finally:
        os.unlink(path)
    assert "(protagonist)" in block
    # the rendered cast LINE uses (role), never a [role] token. (The header
    # instruction names [protagonist] as a forbidden EXAMPLE, so we check the
    # member line specifically rather than the whole block.)
    cast_line = next(ln for ln in block.splitlines()
                     if ln.strip().startswith("- our protagonist"))
    assert "(protagonist)" in cast_line
    assert "[protagonist]" not in cast_line
    # the header instructs the model NEVER to emit a bracketed token
    assert "NEVER output a bracketed token" in block


def test_load_cast_list_reads_cast_array():
    import tempfile
    import os
    cast_json = {"cast": [{"id": "a", "role": "protagonist",
                           "canonical_name": "our protagonist", "aliases": ["Bob"]}]}
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cast_json, f)
        cast = gnp._load_cast_list(path)
    finally:
        os.unlink(path)
    assert isinstance(cast, list) and cast[0]["id"] == "a"
    # missing / empty path -> empty list (never raises)
    assert gnp._load_cast_list("") == []
    assert gnp._load_cast_list("/no/such/file.json") == []
