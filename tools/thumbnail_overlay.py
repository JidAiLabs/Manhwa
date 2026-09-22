#!/usr/bin/env python3
"""
thumbnail_overlay.py — deterministic, branded text layer for thumbnails.

The competitor thumbnails share ONE consistent text style (heavy yellow caps +
thick black outline + a bold arrow + floating !/? marks + short speech callouts).
That consistency means the text is a deterministic OVERLAY, not model-drawn:
- always legible (model text garbles),
- always copyright-safe (we control every glyph — no licensed name can leak),
- re-textable without paying to regenerate the art.

Nano Banana renders the ART (no text); this draws the words on top. Pure/PIL,
no model — unit-tested by compositing onto a stub image.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Impact is the canonical thumbnail face; fall back through bold system fonts.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]
_YELLOW = (255, 214, 10)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _fitted(draw: ImageDraw.ImageDraw, text: str, size: int,
            max_w: int) -> ImageFont.FreeTypeFont:
    """The largest font up to *size* whose *text* fits *max_w* pixels. A split
    label at a fixed size ran off the frame ("UNIMPORTANT SPECTATOR")."""
    f = _font(size)
    while size > 12 and draw.textlength(text, font=f) > max_w:
        size = int(size * 0.92)
        f = _font(size)
    return f


def _label_lines(draw: ImageDraw.ImageDraw, text: str, size: int,
                 max_w: int):
    """(lines, font) for a label that must fit *max_w*. A two-word-or-more
    label that would have to shrink is STACKED on two lines instead -- the
    owner's examples stack their tags (NEW / SLAVE, #1 / HUNTER) rather than
    shrinking them to a thin strip."""
    one = _fitted(draw, text, size, max_w)
    words = text.split()
    if one.size >= size or len(words) < 2:
        return [text], one
    cut = min(range(1, len(words)), key=lambda i: abs(
        len(" ".join(words[:i])) - len(" ".join(words[i:]))))
    lines = [" ".join(words[:cut]), " ".join(words[cut:])]
    two = _fitted(draw, max(lines, key=lambda l: draw.textlength(l, font=one)),
                  size, max_w)
    return (lines, two) if two.size > one.size else ([text], one)


def _outlined(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str,
              font: ImageFont.FreeTypeFont, *, fill=_YELLOW, anchor="la") -> None:
    stroke = max(3, font.size // 12)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor,
              stroke_width=stroke, stroke_fill=_BLACK)


def _anchor_xy(pos: str, W: int, H: int) -> Tuple[Tuple[int, int], str]:
    """Return (xy, PIL anchor) for a named label position."""
    return {
        "upper_right": ((int(W * 0.97), int(H * 0.08)), "ra"),
        "upper_left": ((int(W * 0.03), int(H * 0.08)), "la"),
        "lower_right": ((int(W * 0.97), int(H * 0.80)), "ra"),
        "lower_left": ((int(W * 0.03), int(H * 0.80)), "la"),
        "mid_left": ((int(W * 0.03), int(H * 0.44)), "la"),
        "mid_right": ((int(W * 0.97), int(H * 0.44)), "ra"),
        "on_object": ((int(W * 0.50), int(H * 0.78)), "ma"),
        "split": ((int(W * 0.25), int(H * 0.06)), "ma"),
        "center": ((int(W * 0.50), int(H * 0.10)), "ma"),
    }.get(pos, ((int(W * 0.97), int(H * 0.08)), "ra"))


# A transformation tag ("A -> B") is one label, not two: the arrow glyph sits
# BETWEEN the states so the eye reads the change in one hop. Distinct from the
# `split` style, which puts two labels at opposite corners of a split image.
_TRANSFORM_SEP_RE = re.compile(r"\s*(?:->|=>|→)\s*")


def _split_transform(text: str) -> Optional[Tuple[str, str]]:
    """('TRASH', 'GOD') for a transformation tag, else None."""
    parts = [p.strip() for p in _TRANSFORM_SEP_RE.split(str(text or "")) if p.strip()]
    return (parts[0], parts[1]) if len(parts) == 2 else None


def _draw_transform(draw: ImageDraw.ImageDraw, xy: Tuple[int, int],
                    left: str, right: str, font: ImageFont.FreeTypeFont,
                    anchor: str, W: int) -> None:
    """Draw 'LEFT → RIGHT' with a real arrow between the two states."""
    gap = int(font.size * 0.9)
    lw = int(draw.textlength(left, font=font))
    rw = int(draw.textlength(right, font=font))
    total = lw + gap + rw
    x, y = xy
    # resolve the anchor to a left edge so the composite stays inside the frame
    if anchor.startswith("r"):
        x0 = x - total
    elif anchor.startswith("m"):
        x0 = x - total // 2
    else:
        x0 = x
    x0 = max(int(W * 0.02), min(x0, int(W * 0.98) - total))
    _outlined(draw, (x0, y), left, font, anchor="la")
    ay = y + int(font.size * 0.42)
    _arrow(draw, (x0 + lw + int(gap * 0.15), ay),
           (x0 + lw + int(gap * 0.85), ay), max(5, font.size // 9))
    _outlined(draw, (x0 + lw + gap, y), right, font, anchor="la")


def _arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int],
           end: Tuple[int, int], width: int) -> None:
    import math
    draw.line([start, end], fill=_YELLOW, width=width)
    # arrowhead
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    L = width * 4
    for da in (math.radians(150), math.radians(-150)):
        draw.line([end, (int(end[0] + L * math.cos(ang + da)),
                         int(end[1] + L * math.sin(ang + da)))],
                  fill=_YELLOW, width=width)


def _target(spec: Dict[str, Any], W: int, H: int,
            default: Tuple[float, float]) -> Tuple[int, int]:
    """Where an arrow lands. A hook design fixes where its subject stands and
    says so in ``arrow_to`` (frame fractions); without one the arrow keeps the
    old frame-centre aim, so existing callers render byte-identically."""
    fx, fy = spec.get("arrow_to") or default
    return int(W * fx), int(H * fy)


# The standard manhwa SYSTEM PANEL, as the example thumbnails draw it (MARRY
# HER, LEVEL UP!, SQUATS COMPLETED 47/10,000): a saturated blue translucent
# panel, a glowing cyan edge, a small header line, ONE huge line, small stat
# lines under it. Owner, 2026-09-22, on the first dark-navy box: "it is not
# blue panel from standard manhwa approach".
_CARD_FILL = (24, 96, 214, 215)
_CARD_TOP = (70, 150, 250, 230)            # a lighter band along the top
_CARD_EDGE = (120, 225, 255, 255)
_CARD_GLOW = (90, 200, 255, 70)
_CARD_DIM = (205, 232, 255)                # header / footer text


def _window_parts(card: Any) -> Dict[str, Any]:
    """A card is a dict {header, line, footer[]} or, from an older claim, a
    list of lines whose first is the big one."""
    if isinstance(card, dict):
        return {"header": str(card.get("header") or "").strip(),
                "line": str(card.get("line") or "").strip(),
                "footer": [str(x).strip() for x in card.get("footer") or []
                           if str(x).strip()]}
    lines = [str(x).strip() for x in (card or []) if str(x).strip()]
    return {"header": "", "line": lines[0] if lines else "",
            "footer": lines[1:3]}


def _draw_card(img: Image.Image, slot: Dict[str, Any], card: Any,
               W: int, H: int) -> None:
    """A system window holding the story's OWN printed words (quoted by code
    upstream, never written by a model). Drawn here, not in the art: every word
    on a thumbnail is an overlay."""
    parts = _window_parts(card)
    if not parts["line"]:
        return
    (fx, fy), (fw, fh) = slot["pos"], slot["size"]
    x0, y0, x1, y1 = int(W * fx), int(H * fy), int(W * (fx + fw)), int(H * (fy + fh))
    d = ImageDraw.Draw(img, "RGBA")           # RGBA draw blends the fill
    r = int(H * 0.03)
    # outer glow: the same rounded rect, larger and fainter, twice
    for grow in (int(H * 0.018), int(H * 0.009)):
        d.rounded_rectangle((x0 - grow, y0 - grow, x1 + grow, y1 + grow),
                            radius=r + grow, fill=_CARD_GLOW)
    d.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=_CARD_FILL,
                        outline=_CARD_EDGE, width=max(4, H // 160))
    band = int((y1 - y0) * 0.16)
    d.rounded_rectangle((x0, y0, x1, y0 + band + r), radius=r, fill=_CARD_TOP)
    d.rectangle((x0, y0 + band, x1, y0 + band + r), fill=_CARD_FILL)
    d.line((x0, y0 + band, x1, y0 + band), fill=_CARD_EDGE, width=2)

    pad = int((x1 - x0) * 0.06)
    inner_w = x1 - x0 - 2 * pad
    f_small = _fitted(d, parts["header"] or "SYSTEM", int(H * 0.045), inner_w)
    y = y0 + (band - f_small.size) // 2
    d.text((x0 + pad, y), "[ %s ]" % (parts["header"] or "SYSTEM"), font=f_small,
           fill=_WHITE)
    y = y0 + band + pad // 2
    footer = parts["footer"][:2]
    foot_h = int(H * 0.042 * 1.25) * len(footer) if footer else 0
    room = y1 - pad // 2 - foot_h - y
    # the big line: stacked on two balanced rows rather than shrunk
    rows, f = _label_lines(d, parts["line"], int(H * 0.105), inner_w)
    row_h = int(f.size * 1.1)
    while row_h * len(rows) > room and f.size > 14:
        f = _font(int(f.size * 0.92))
        row_h = int(f.size * 1.1)
    top = y + max(0, (room - row_h * len(rows)) // 2)
    for i, row in enumerate(rows):
        d.text((x0 + pad + 3, top + i * row_h + 3), row, font=f, fill=(0, 30, 90, 200))
        d.text((x0 + pad, top + i * row_h), row, font=f, fill=_WHITE)
    fy_ = y1 - pad // 2 - foot_h
    for i, line in enumerate(footer):
        ff = _fitted(d, line, int(H * 0.042), inner_w)
        d.text((x0 + pad, fy_ + i * int(H * 0.042 * 1.25)), line, font=ff,
               fill=_CARD_DIM)


def render_overlay(base_image: str, out_path: str, *, hook: str,
                   style_overlay: Dict[str, Any],
                   speech: Optional[List[str]] = None,
                   size: Tuple[int, int] = (1280, 720),
                   badge: str = "",
                   tags: Optional[List[Dict[str, Any]]] = None,
                   card: Any = None) -> str:
    """Composite the branded text layer onto *base_image*. Returns *out_path*.

    A single centred phrase reads as a caption on a picture; the thumbnails that
    actually work read as TAGS STUCK ONTO THINGS — a small status badge, one or
    two short labels with arrows onto their subject, and a transformation label.
    So beyond the main *hook* this draws:

    *badge* — a small corner tag for TRUE video metadata ("FULL RECAP",
      "CH 1-55"). Deliberately separate from the hook: a badge states a fact
      about the upload, never a claim about the story, so it can never invent
      anything.
    *tags*  — [{"text", "pos", "arrow"}] short labels (1-2 words) at named
      positions, each optionally arrowed toward the subject. "A -> B" in any
      label renders as a transformation with the arrow between the states.

    *card*  — the quoted system-window lines, drawn only when the style's
      overlay declares a ``card`` slot AND there are lines: never an empty box.

    All are optional and default to nothing, so existing single-hook callers
    render byte-identically."""
    W, H = size
    img = Image.open(base_image).convert("RGB").resize((W, H))
    if card and style_overlay.get("card"):
        _draw_card(img, style_overlay["card"], card, W, H)   # under the labels
    draw = ImageDraw.Draw(img)
    hook = (hook or "").strip().upper()

    label_pos = style_overlay.get("label_pos", "upper_right")
    if style_overlay.get("split"):
        parts = (hook.split("|", 1) + [""])[:2] if "|" in hook else ("BEFORE", "AFTER")
        for (cx, cy), part in zip(((0.25, 0.08), (0.75, 0.82)), parts):
            part = part.strip()
            _outlined(draw, (int(W * cx), int(H * cy)), part,
                      _fitted(draw, part, int(H * 0.13), int(W * 0.44)),
                      anchor="ma")
    elif hook:
        (lx, ly), anc = _anchor_xy(label_pos, W, H)
        xform = _split_transform(hook)
        if xform:
            f = _fitted(draw, "%s    %s" % xform, int(H * 0.16), int(W * 0.9))
            _draw_transform(draw, (lx, ly), xform[0], xform[1], f, anc, W)
        else:
            # A corner label stays in its own side: power_reveal CENTRES the
            # hero, and "THE ONLY SURVIVOR" at a fixed size ran across his face.
            # ponytail: a fixed 40% column, not face detection -- Apple's face
            # model misses anime faces and its body fallback found 1 figure of 3.
            max_w = int(W * (0.8 if anc == "ma" else 0.40))
            lines, f = _label_lines(draw, hook, int(H * 0.16), max_w)
            step = int(f.size * 1.05)
            for i, line in enumerate(lines):
                _outlined(draw, (lx, ly + i * step), line, f, anchor=anc)
        if style_overlay.get("arrow", "none") != "none" and not xform:
            # arrow from the label's edge NEAREST the subject toward frame
            # centre: a low label ("on_object") had its arrow run up through
            # its own text ("THE ONLY READER")
            sx = lx if anc == "ma" else lx - (
                int(W * 0.10) if anc == "ra" else -int(W * 0.10))
            sy = (ly + len(lines) * step + int(H * 0.02) if ly < H * 0.5
                  else ly - int(H * 0.03))
            _arrow(draw, (sx, sy), _target(style_overlay, W, H, (0.52, 0.46)),
                   max(6, H // 90))

    # status badge: a FACT about the upload (chapter range, full recap), never a
    # claim about the story. Sits opposite the main label so the two never stack.
    if badge:
        # a split's BEFORE label owns the top-left, so its badge goes right
        bpos = ("upper_right" if label_pos in ("upper_left", "split")
                else "upper_left")
        (bx, by), banc = _anchor_xy(bpos, W, H)
        _outlined(draw, (bx, int(H * 0.03)), str(badge).strip().upper(),
                  _font(int(H * 0.062)), anchor=banc)

    # SPLIT compositions already spend both halves on the before/after pair --
    # those two labels ARE the subject tags. Adding more piles every extra
    # element onto the left half (measured: badge + hook + a mid-left tag + a
    # lower-left tag + a diagonal arrow across the "before" character, against
    # one lonely label on the right) and the arrow, aimed at frame centre, cuts
    # straight over the artwork. Tags are for single-composition styles.
    f_tag = _font(int(H * 0.095))
    for t in ([] if style_overlay.get("split") else (tags or [])):
        text = str((t or {}).get("text") or "").strip().upper()
        if not text:
            continue
        (tx, ty), tanc = _anchor_xy(str(t.get("pos") or "lower_left"), W, H)
        tx2 = _split_transform(text)
        if tx2:
            _draw_transform(draw, (tx, ty), tx2[0], tx2[1], f_tag, tanc, W)
        else:
            _outlined(draw, (tx, ty), text, f_tag, anchor=tanc)
            if t.get("arrow"):
                sx = tx - (int(W * 0.06) if tanc == "ra" else -int(W * 0.06))
                _arrow(draw, (sx, ty + int(H * 0.09)),
                       _target(t, W, H, (0.50, 0.50)), max(5, H // 110))

    # floating reaction marks
    f_mark = _font(int(H * 0.14))
    for i, m in enumerate(style_overlay.get("marks", []) or []):
        _outlined(draw, (int(W * (0.10 + 0.10 * i)), int(H * 0.10)), m, f_mark,
                  fill=_WHITE, anchor="ma")

    # short speech callouts (colored caps), bottom-left stack
    slots = int(style_overlay.get("speech_slots", 0) or 0)
    f_sp = _font(int(H * 0.075))
    for i, line in enumerate((speech or [])[:slots]):
        _outlined(draw, (int(W * 0.04), int(H * (0.60 + 0.12 * i))),
                  str(line).strip().upper(), f_sp, fill=_WHITE, anchor="la")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path, quality=90)
    return out_path
