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


def render_overlay(base_image: str, out_path: str, *, hook: str,
                   style_overlay: Dict[str, Any],
                   speech: Optional[List[str]] = None,
                   size: Tuple[int, int] = (1280, 720),
                   badge: str = "",
                   tags: Optional[List[Dict[str, Any]]] = None) -> str:
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

    Both are optional and default to nothing, so existing single-hook callers
    render byte-identically."""
    W, H = size
    img = Image.open(base_image).convert("RGB").resize((W, H))
    draw = ImageDraw.Draw(img)
    hook = (hook or "").strip().upper()

    label_pos = style_overlay.get("label_pos", "upper_right")
    # split style: two labels (left weak / right strong) from a "A|B" hook
    if style_overlay.get("split"):
        parts = (hook.split("|", 1) + [""])[:2] if "|" in hook else ("BEFORE", "AFTER")
        f = _font(int(H * 0.13))
        _outlined(draw, (int(W * 0.25), int(H * 0.08)), parts[0].strip(), f, anchor="ma")
        _outlined(draw, (int(W * 0.75), int(H * 0.82)), parts[1].strip(), f, anchor="ma")
    elif hook:
        f = _font(int(H * 0.16))
        (lx, ly), anc = _anchor_xy(label_pos, W, H)
        xform = _split_transform(hook)
        if xform:
            _draw_transform(draw, (lx, ly), xform[0], xform[1], f, anc, W)
        else:
            _outlined(draw, (lx, ly), hook, f, anchor=anc)
        if style_overlay.get("arrow", "none") != "none" and not xform:
            # arrow from just under the label toward frame center (the subject)
            sx = lx - (int(W * 0.10) if anc == "ra" else -int(W * 0.10))
            _arrow(draw, (sx, ly + int(H * 0.14)),
                   (int(W * 0.52), int(H * 0.46)), max(6, H // 90))

    # status badge: a FACT about the upload (chapter range, full recap), never a
    # claim about the story. Sits opposite the main label so the two never stack.
    if badge:
        bpos = "upper_right" if label_pos == "upper_left" else "upper_left"
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
                       (int(W * 0.50), int(H * 0.50)), max(5, H // 110))

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
