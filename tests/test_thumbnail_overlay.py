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


# ---- split labels must fit their half (ORV before_after, 2026-09-17) --------
# "UNIMPORTANT SPECTATOR" at a fixed H*0.13 ran off the LEFT edge and
# "THE SCRIPT MASTER" off the RIGHT; the "309 CHAPTERS" badge sat on the top
# label because both used the upper-left corner.

def _yellow_cols(im, y0, y1):
    px = im.load()
    return [x for x in range(im.width)
            if any(px[x, y][0] > 200 and px[x, y][1] > 170 and px[x, y][2] < 90
                   for y in range(y0, y1, 3))]


def test_long_split_labels_stay_inside_their_half(tmp_path):
    base = _stub(tmp_path)
    out = str(tmp_path / "s.jpg")
    style = {"label_pos": "split", "split": True, "arrow": "none", "marks": []}
    ov.render_overlay(base, out, hook="UNIMPORTANT SPECTATOR|THE SCRIPT MASTER",
                      style_overlay=style)
    im = Image.open(out).convert("RGB")
    top = _yellow_cols(im, 0, 200)
    bottom = _yellow_cols(im, 560, 720)
    assert top and min(top) > 4 and max(top) < 640        # inside the left half
    assert bottom and min(bottom) > 640 and max(bottom) < 1276


def test_split_badge_does_not_sit_on_the_before_label(tmp_path):
    base = _stub(tmp_path)
    style = {"label_pos": "split", "split": True, "arrow": "none", "marks": []}
    a = str(tmp_path / "a.jpg"); b = str(tmp_path / "b.jpg")
    ov.render_overlay(base, a, hook="WEAK|KING", style_overlay=style)
    ov.render_overlay(base, b, hook="WEAK|KING", style_overlay=style,
                      badge="309 CHAPTERS")
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    # the badge lands in the right half's top strip, the left half is untouched
    assert list(ia.crop((0, 0, 640, 60)).getdata()) == \
        list(ib.crop((0, 0, 640, 60)).getdata())


# ---- a label never runs across the centre, where the hero is (ORV scene) -----
# "THE ONLY SURVIVOR" at a fixed H*0.16 from the upper-right corner spanned
# ~890 of 1280px and covered the protagonist's face (power_reveal centres him).

def test_a_long_corner_label_stays_out_of_the_centre(tmp_path):
    base = _stub(tmp_path)
    out = str(tmp_path / "c.jpg")
    style = {"label_pos": "upper_right", "arrow": "none", "marks": []}
    ov.render_overlay(base, out, hook="THE ONLY SURVIVOR", style_overlay=style)
    cols = _yellow_cols(Image.open(out).convert("RGB"), 0, 300)
    assert cols and min(cols) > int(1280 * 0.55)          # right of the hero


def test_a_label_that_would_shrink_stacks_on_two_lines_instead(tmp_path):
    """The owner's examples stack a two-word tag (NEW / SLAVE, #1 / HUNTER)
    rather than shrinking it to a thin strip."""
    base = _stub(tmp_path)
    out = str(tmp_path / "c.jpg")
    style = {"label_pos": "upper_right", "arrow": "none", "marks": []}
    ov.render_overlay(base, out, hook="THE ONLY SURVIVOR", style_overlay=style)
    im = Image.open(out).convert("RGB")
    px = im.load()
    rows = [y for y in range(0, 400, 2)
            if any(px[x, y][0] > 200 and px[x, y][1] > 170 and px[x, y][2] < 90
                   for x in range(700, 1280, 4))]
    assert rows and max(rows) - min(rows) > 150           # two lines tall


def test_an_arrow_never_crosses_a_bottom_label(tmp_path):
    """on_object puts the label low; the arrow ran up THROUGH "THE ONLY READER"."""
    base = _stub(tmp_path)
    a = str(tmp_path / "a.jpg"); b = str(tmp_path / "b.jpg")
    for out, arrow in ((a, "none"), (b, "to_object")):
        ov.render_overlay(base, out, hook="THE ONLY READER",
                          style_overlay={"label_pos": "on_object",
                                         "arrow": arrow, "marks": []})
    band = (0, 560, 1280, 720)                        # the label's own rows
    ia = Image.open(a).convert("RGB").crop(band)
    ib = Image.open(b).convert("RGB").crop(band)
    diff = sum(1 for p, q in zip(ia.getdata(), ib.getdata())
               if abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2]) > 60)
    assert diff < 50


# ---- arrows land on their subject BY CONSTRUCTION; the quoted system card ---
# The design fixes where the lead stands (art_clause) and tells the overlay
# where to aim (arrow_to). About 13 of the owner's 20 examples land an arrow on
# its own subject; ours all aimed at frame centre.

def _box_pixels(path, box, pred):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H),
                      int(box[2] * W), int(box[3] * H))
    return sum(1 for y in range(y0, y1) for x in range(x0, x1)
               if pred(im.getpixel((x, y))))


def _is_yellow(px):
    r, g, b = px
    return r > 200 and g > 170 and b < 90


_FAR_BOX = (0.78, 0.70, 0.92, 0.86)      # nowhere near frame centre or the label


def test_arrow_to_moves_the_style_arrow_onto_its_target(tmp_path):
    spec = {"label_pos": "upper_left", "arrow": "to_hero", "marks": [],
            "speech_slots": 0}
    plain, aimed = str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")
    ov.render_overlay(_stub(tmp_path), plain, hook="OUTCAST", style_overlay=spec)
    ov.render_overlay(_stub(tmp_path), aimed, hook="OUTCAST",
                      style_overlay={**spec, "arrow_to": [0.85, 0.78]})
    assert _box_pixels(plain, _FAR_BOX, _is_yellow) == 0
    assert _box_pixels(aimed, _FAR_BOX, _is_yellow) > 50


def test_tag_arrow_to_lands_on_the_tags_own_subject(tmp_path):
    spec = {"label_pos": "upper_left", "arrow": "none", "marks": [],
            "speech_slots": 0}
    out = str(tmp_path / "t.jpg")
    ov.render_overlay(_stub(tmp_path), out, hook="", style_overlay=spec,
                      tags=[{"text": "RIVAL", "pos": "mid_left", "arrow": True,
                             "arrow_to": [0.85, 0.78]}])
    assert _box_pixels(out, _FAR_BOX, _is_yellow) > 50


_CARD_SPEC = {"label_pos": "upper_right", "arrow": "none", "marks": [],
              "speech_slots": 0,
              "card": {"pos": [0.04, 0.22], "size": [0.36, 0.46]}}
_CARD_BOX = (0.05, 0.24, 0.39, 0.66)


def _not_base(px, base=(20, 30, 40)):
    return any(abs(c - b) > 40 for c, b in zip(px, base))


def test_card_draws_the_quoted_lines_in_its_slot(tmp_path):
    out = str(tmp_path / "c.jpg")
    ov.render_overlay(_stub(tmp_path), out, hook="", style_overlay=_CARD_SPEC,
                      card=["Main scenario has begun.", "Time limit: 30 minutes."])
    assert _box_pixels(out, _CARD_BOX, _not_base) > 2000


def test_no_card_lines_draws_no_empty_window(tmp_path):
    out = str(tmp_path / "c.jpg")
    ov.render_overlay(_stub(tmp_path), out, hook="", style_overlay=_CARD_SPEC,
                      card=[])
    assert _box_pixels(out, _CARD_BOX, _not_base) == 0


def test_one_card_line_is_drawn_big(tmp_path):
    """The examples' system windows hold ONE short line in huge type. Measured
    2026-09-22: the first card drew this line at 5,235 white pixels -- fine
    print in a mostly empty box, unreadable at thumbnail size."""
    out = str(tmp_path / "c.jpg")
    spec = {"label_pos": "upper_left", "arrow": "none", "marks": [],
            "speech_slots": 0, "card": {"pos": [0.04, 0.52], "size": [0.42, 0.30]}}
    ov.render_overlay(_stub(tmp_path), out, hook="", style_overlay=spec,
                      card=["THE MAIN SCENARIO HAS ARRIVED."])
    white = _box_pixels(out, (0.0, 0.45, 0.5, 0.9), lambda px: min(px) > 200)
    assert white > 12000, white
