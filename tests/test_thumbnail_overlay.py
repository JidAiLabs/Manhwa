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


# ---- multi-label layout (badge + subject tags + transformation) ------------
# One centred phrase reads as a caption on a picture. The layouts that work read
# as TAGS STUCK ONTO THINGS: a small status badge, one or two short arrowed
# labels, and a transformation label with the arrow BETWEEN the two states.

def test_single_hook_callers_are_byte_identical(tmp_path):
    """The whole extension is opt-in: no badge, no tags -> unchanged output."""
    base = _stub(tmp_path)
    style = {"label_pos": "upper_right", "arrow": "to_hero", "marks": ["!"],
             "speech_slots": 1}
    a = str(tmp_path / "a.jpg"); b = str(tmp_path / "b.jpg")
    ov.render_overlay(base, a, hook="GENIUS", style_overlay=style, speech=["HOW?!"])
    ov.render_overlay(base, b, hook="GENIUS", style_overlay=style, speech=["HOW?!"],
                      badge="", tags=None)
    assert Path(a).read_bytes() == Path(b).read_bytes()


def test_badge_paints_and_sits_opposite_the_main_label(tmp_path):
    base = _stub(tmp_path)
    plain = str(tmp_path / "plain.jpg"); badged = str(tmp_path / "badged.jpg")
    style = {"label_pos": "upper_right", "arrow": "none", "marks": []}
    ov.render_overlay(base, plain, hook="GENIUS", style_overlay=style)
    ov.render_overlay(base, badged, hook="GENIUS", style_overlay=style,
                      badge="FULL RECAP")
    assert _yellow_pixels(badged) > _yellow_pixels(plain)
    # main label is upper-RIGHT, so the badge must land on the LEFT half
    im = Image.open(badged).convert("RGB")
    left = im.crop((0, 0, 400, 120))
    assert sum(1 for r, g, b in left.getdata() if r > 200 and g > 170 and b < 90) > 200


def test_tags_render_at_their_positions(tmp_path):
    base = _stub(tmp_path)
    out = str(tmp_path / "tagged.jpg"); plain = str(tmp_path / "plain.jpg")
    style = {"label_pos": "upper_right", "arrow": "none", "marks": []}
    ov.render_overlay(base, plain, hook="HOOK", style_overlay=style)
    ov.render_overlay(base, out, hook="HOOK", style_overlay=style,
                      tags=[{"text": "THE DOKKAEBI", "pos": "lower_left",
                             "arrow": True}])
    assert _yellow_pixels(out) > _yellow_pixels(plain)
    im = Image.open(out).convert("RGB")
    lower_left = im.crop((0, 520, 520, 700))
    assert sum(1 for r, g, b in lower_left.getdata()
               if r > 200 and g > 170 and b < 90) > 200


def test_transformation_label_splits_on_an_arrow(tmp_path):
    assert ov._split_transform("TRASH -> GOD") == ("TRASH", "GOD")
    assert ov._split_transform("READER → PROPHET") == ("READER", "PROPHET")
    assert ov._split_transform("JUST ONE LABEL") is None
    base = _stub(tmp_path)
    out = str(tmp_path / "x.jpg"); flat = str(tmp_path / "flat.jpg")
    style = {"label_pos": "lower_left", "arrow": "none", "marks": []}
    ov.render_overlay(base, flat, hook="READER", style_overlay=style)
    ov.render_overlay(base, out, hook="READER -> PROPHET", style_overlay=style)
    # two states + a drawn arrow paint more than the single word
    assert _yellow_pixels(out) > _yellow_pixels(flat)


def test_empty_tags_are_skipped_not_drawn_blank(tmp_path):
    base = _stub(tmp_path)
    a = str(tmp_path / "a.jpg"); b = str(tmp_path / "b.jpg")
    style = {"label_pos": "upper_right", "arrow": "none", "marks": []}
    ov.render_overlay(base, a, hook="H", style_overlay=style)
    ov.render_overlay(base, b, hook="H", style_overlay=style,
                      tags=[{"text": "  ", "pos": "lower_left"}, {}])
    assert Path(a).read_bytes() == Path(b).read_bytes()


def test_split_styles_do_not_stack_extra_tags_on_one_half(tmp_path):
    """A split composition spends both halves on the before/after pair, so the
    two labels ARE the tags. Adding more put badge + hook + two tags + a
    diagonal arrow all on the LEFT half against one label on the right."""
    base = _stub(tmp_path)
    style = {"label_pos": "split", "split": True, "arrow": "none", "marks": []}
    a = str(tmp_path / "a.jpg"); b = str(tmp_path / "b.jpg")
    ov.render_overlay(base, a, hook="READER|TARGET", style_overlay=style)
    ov.render_overlay(base, b, hook="READER|TARGET", style_overlay=style,
                      tags=[{"text": "CONSTELLATION", "pos": "lower_left",
                             "arrow": True},
                            {"text": "NIGHTMARE -> REALITY", "pos": "mid_left"}])
    assert Path(a).read_bytes() == Path(b).read_bytes()


def test_non_split_styles_still_draw_tags(tmp_path):
    base = _stub(tmp_path)
    style = {"label_pos": "upper_right", "arrow": "none", "marks": []}
    a = str(tmp_path / "a.jpg"); b = str(tmp_path / "b.jpg")
    ov.render_overlay(base, a, hook="READER", style_overlay=style)
    ov.render_overlay(base, b, hook="READER", style_overlay=style,
                      tags=[{"text": "DOKKAEBI", "pos": "lower_left"}])
    assert Path(a).read_bytes() != Path(b).read_bytes()


def test_triptych_draws_three_labels_one_under_each_panel(tmp_path):
    """The most common performing layout: three vertical panels, one label at
    the BOTTOM of each. A two-part hook must not leave a panel unlabelled."""
    base = _stub(tmp_path)
    out = str(tmp_path / "t.jpg")
    style = {"label_pos": "split", "split3": True, "arrow": "none", "marks": []}
    ov.render_overlay(base, out, hook="ORDINARY|AWAKENING|PROPHET",
                      style_overlay=style)
    im = Image.open(out).convert("RGB")
    # one label low in each vertical third
    for x0, x1 in ((0, 426), (426, 853), (853, 1280)):
        band = im.crop((x0, 520, x1, 700))
        assert sum(1 for r, g, b in band.getdata()
                   if r > 200 and g > 170 and b < 90) > 150, (x0, x1)


def test_triptych_suppresses_extra_tags_like_split(tmp_path):
    base = _stub(tmp_path)
    style = {"label_pos": "split", "split3": True, "arrow": "none", "marks": []}
    a = str(tmp_path / "a.jpg"); b = str(tmp_path / "b.jpg")
    ov.render_overlay(base, a, hook="A|B|C", style_overlay=style)
    ov.render_overlay(base, b, hook="A|B|C", style_overlay=style,
                      tags=[{"text": "DOKKAEBI", "pos": "lower_left",
                             "arrow": True}])
    assert Path(a).read_bytes() == Path(b).read_bytes()
