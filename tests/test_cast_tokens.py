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


# ---- invented names never reach the cast (2026-08-18, nano ch1 "Alden") -----
_CB_SPEC = importlib.util.spec_from_file_location(
    "cast_builder", Path(__file__).resolve().parent.parent / "tools" / "cast_builder.py")
_cb = importlib.util.module_from_spec(_CB_SPEC)
_CB_SPEC.loader.exec_module(_cb)  # type: ignore[union-attr]
sanitize_cast_names = _cb.sanitize_cast_names


def _mk_cast(**over):
    base = {"id": "hooded_leader", "canonical_name": "Alden", "role": "antagonist",
            "spoken_name": "the masked leader", "aliases": [],
            "visual_description": "masked figure in a hooded cloak", "is_protagonist": False}
    base.update(over)
    return {"cast": [base]}


_OCR_A = "WELL DONE PRINCE CHEON. GIVE IT UP, THERE'S NO OTHER WAY OUT. HEY, ANCESTOR NIM?"


def test_invented_proper_name_falls_back_to_spoken_handle():
    out, fixes = sanitize_cast_names(_mk_cast(), _OCR_A)
    assert out["cast"][0]["canonical_name"] == "the masked leader"
    assert fixes == [("hooded_leader", "Alden", "the masked leader")]


def test_invented_name_without_spoken_name_uses_the_id_handle():
    out, _ = sanitize_cast_names(_mk_cast(spoken_name=""), _OCR_A)
    assert out["cast"][0]["canonical_name"] == "the hooded leader"
    # id carrying the invented name itself (real ch1: id 'alden') -> description head
    out2, _ = sanitize_cast_names(_mk_cast(spoken_name="", id="alden"), _OCR_A)
    assert out2["cast"][0]["canonical_name"] == "the masked figure"
    out3, _ = sanitize_cast_names(_mk_cast(spoken_name="", id="alden", visual_description=""), _OCR_A)
    assert out3["cast"][0]["canonical_name"] == "the antagonist"


def test_real_name_from_ocr_is_kept_case_insensitively():
    out, fixes = sanitize_cast_names(_mk_cast(canonical_name="Prince Cheon"), _OCR_A)
    assert out["cast"][0]["canonical_name"] == "Prince Cheon" and fixes == []
    out2, _ = sanitize_cast_names(_mk_cast(canonical_name="Ancestor-nim"), _OCR_A)
    assert out2["cast"][0]["canonical_name"] == "Ancestor-nim"


def test_descriptive_handles_and_protagonist_are_left_alone():
    out, fixes = sanitize_cast_names(_mk_cast(canonical_name="the dying old master"), _OCR_A)
    assert out["cast"][0]["canonical_name"] == "the dying old master" and fixes == []
    out2, fixes2 = sanitize_cast_names(_mk_cast(canonical_name="our protagonist", is_protagonist=True), _OCR_A)
    assert out2["cast"][0]["canonical_name"] == "our protagonist" and fixes2 == []


def test_invented_aliases_are_dropped_but_ocr_ones_kept():
    out, _ = sanitize_cast_names(_mk_cast(aliases=["Prince Cheon", "Aldenius", "the kid"]), _OCR_A)
    assert out["cast"][0]["aliases"] == ["Prince Cheon", "the kid"]


def test_page_words_ignore_glyph_sized_ocr_regions():
    # nano ch1 p000019: Apple Vision read a 65%-wide off-image SFX glyph as the
    # word "Alden"; the cast builder took it for a name. Only sane-geometry words
    # count as "on the page".
    items = [{"scene_file": "p000019.jpg", "ocr_clean": "EVEN IF YOU HAVE PEASANT BLOOD Alden",
              "vision": {"ocr_words": [
                  {"t": "PEASANT", "bbox": [0.10, 0.05, 0.30, 0.09]},
                  {"t": "Alden", "bbox": [-0.088, 0.3457, 0.6515, 1.0587]}]}}]
    words = _cb._page_words(items)
    assert "PEASANT" in words and "Alden" not in words
    cast = {"cast": [{"id": "hooded_leader", "canonical_name": "Alden", "role": "antagonist",
                      "spoken_name": "the masked leader", "aliases": [], "is_protagonist": False}]}
    out, fixes = _cb.sanitize_cast_names(cast, words)
    assert out["cast"][0]["canonical_name"] == "the masked leader" and len(fixes) == 1


def test_sane_text_region_rules():
    import importlib.util as _iu
    spec = _iu.spec_from_file_location("apple_vision", Path(__file__).resolve().parent.parent / "tools" / "apple_vision.py")
    av = _iu.module_from_spec(spec); spec.loader.exec_module(av)  # type: ignore[union-attr]
    assert av.sane_text_region(0.1, 0.05, 0.9, 0.12)            # a title line
    assert not av.sane_text_region(-0.088, 0.3457, 0.6515, 1.0587)  # off-image glyph
    assert not av.sane_text_region(0.0, 0.0, 0.6, 0.6)          # 36% of the panel
    assert not av.sane_text_region(0.4, 0.1, 0.5, 0.9)          # 80% tall sliver
