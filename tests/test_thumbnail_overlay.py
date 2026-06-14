"""Deterministic thumbnail text overlay (label/arrow/marks/speech)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "thumbnail_overlay",
    Path(__file__).resolve().parent.parent / "tools" / "thumbnail_overlay.py")
ov = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ov)  # type: ignore[union-attr]


def _stub(tmp_path, color=(20, 30, 40)):
    p = tmp_path / "art.jpg"
    Image.new("RGB", (1280, 720), color).save(p)
    return str(p)


def _yellow_pixels(path):
    im = Image.open(path).convert("RGB")
    return sum(1 for r, g, b in im.getdata() if r > 200 and g > 170 and b < 90)


def test_overlay_draws_label_and_outputs_720p(tmp_path):
    out = str(tmp_path / "thumb.jpg")
    base = _stub(tmp_path)
    ov.render_overlay(base, out, hook="GENIUS",
                      style_overlay={"label_pos": "upper_right",
                                     "arrow": "to_hero", "marks": ["!", "?"],
                                     "speech_slots": 1},
                      speech=["HOW?!"])
    im = Image.open(out)
    assert im.size == (1280, 720)
    # the yellow label + arrow must have painted a meaningful number of pixels
    assert _yellow_pixels(out) > 500


def test_split_style_renders_two_labels(tmp_path):
    out = str(tmp_path / "thumb.jpg")
    ov.render_overlay(_stub(tmp_path), out, hook="WEAK|GODLIKE",
                      style_overlay={"label_pos": "split", "split": True,
                                     "arrow": "none", "marks": [], "speech_slots": 0})
    assert Image.open(out).size == (1280, 720)
    assert _yellow_pixels(out) > 500


def test_empty_hook_is_safe(tmp_path):
    out = str(tmp_path / "thumb.jpg")
    ov.render_overlay(_stub(tmp_path), out, hook="",
                      style_overlay={"label_pos": "upper_right", "arrow": "none",
                                     "marks": [], "speech_slots": 0})
    assert Image.open(out).size == (1280, 720)
