"""tests/test_ocr_garbage_scrub.py

Garbage-OCR scrub in vision_extract: a long run of repeated non-word chars
(underscores/dots/dashes from a bad SFX/speedline scan) must be removed before
it reaches the understanding/narration stages — that corruption is what made
the Ch20 g0014 narration model emit JSON meta-commentary.

TDD: these tests drive the scrubber's behavior. Real text is never harmed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# vision_extract imports sibling tool modules (ocr_chrome) by bare name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

_SPEC = importlib.util.spec_from_file_location(
    "vision_extract",
    Path(__file__).resolve().parent.parent / "tools" / "vision_extract.py")
ve = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ve)  # type: ignore[union-attr]


def _cfg():
    return ve.VisionConfig()


# ---------------------------------------------------------------------------
# clean_ocr_text — garbage run removed, real text kept
# ---------------------------------------------------------------------------
def test_long_underscore_run_stripped_keeps_real_text():
    out = ve.clean_ocr_text("____________________ WHAT?!", _cfg())
    assert out == "WHAT?!"


def test_underscore_run_inside_line_collapsed():
    out = ve.clean_ocr_text("He froze __________________ then ran.", _cfg())
    assert "_" not in out
    assert "He froze" in out and "then ran" in out


def test_long_dot_run_stripped():
    out = ve.clean_ocr_text(".................... silence", _cfg())
    assert "...................." not in out
    assert "silence" in out


def test_long_dash_run_stripped():
    out = ve.clean_ocr_text("-------------------- BOOM", _cfg())
    assert "--------------------" not in out
    assert "BOOM" in out


def test_normal_line_unchanged():
    text = "Cheon Mu Geum unleashes the Butterfly Dance and the hall falls silent."
    assert ve.clean_ocr_text(text, _cfg()) == text


def test_short_punct_run_preserved():
    # ellipsis and ?! are legitimate — a SHORT run (<6) must survive
    text = "Wait... what?!"
    assert ve.clean_ocr_text(text, _cfg()) == text


def test_all_underscore_line_becomes_empty():
    assert ve.clean_ocr_text("__________________________________________________", _cfg()) == ""


# ---------------------------------------------------------------------------
# ocr_words — an all-underscore/punctuation token is dropped
# ---------------------------------------------------------------------------
class _Vert:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Poly:
    def __init__(self, verts):
        self.vertices = verts


class _Ann:
    def __init__(self, desc, box):
        self.description = desc
        x0, y0, x1, y1 = box
        self.bounding_poly = _Poly([_Vert(x0, y0), _Vert(x1, y0), _Vert(x1, y1), _Vert(x0, y1)])


class _Resp:
    def __init__(self, anns):
        # [0] is the full-text annotation; [1:] are words
        self.text_annotations = anns


def test_garbage_underscore_token_dropped_from_ocr_words():
    resp = _Resp([
        _Ann("__________________ WHAT", (0, 0, 200, 50)),   # full text (ignored)
        _Ann("__________________", (0, 0, 120, 20)),         # garbage word -> dropped
        _Ann("WHAT", (130, 0, 200, 20)),                     # real word -> kept
    ])
    words = ve.extract_ocr_words(resp, 200, 50)
    toks = [w["t"] for w in words]
    assert "WHAT" in toks
    assert all("_" not in t for t in toks)
    assert "__________________" not in toks


def test_real_words_survive_ocr_words():
    resp = _Resp([
        _Ann("Cheon Mu", (0, 0, 200, 50)),
        _Ann("Cheon", (0, 0, 80, 20)),
        _Ann("Mu", (90, 0, 130, 20)),
    ])
    toks = [w["t"] for w in ve.extract_ocr_words(resp, 200, 50)]
    assert toks == ["Cheon", "Mu"]
