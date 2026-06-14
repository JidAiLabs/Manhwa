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
        "on_object": ((int(W * 0.50), int(H * 0.78)), "ma"),
        "split": ((int(W * 0.25), int(H * 0.06)), "ma"),
        "center": ((int(W * 0.50), int(H * 0.10)), "ma"),
    }.get(pos, ((int(W * 0.97), int(H * 0.08)), "ra"))


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
                   size: Tuple[int, int] = (1280, 720)) -> str:
    """Composite the branded text layer (label + arrow + marks + speech) for one
    hook onto *base_image* (the Nano Banana art). Returns *out_path*."""
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
        _outlined(draw, (lx, ly), hook, f, anchor=anc)
        if style_overlay.get("arrow", "none") != "none":
            # arrow from just under the label toward frame center (the subject)
            sx = lx - (int(W * 0.10) if anc == "ra" else -int(W * 0.10))
            _arrow(draw, (sx, ly + int(H * 0.14)),
                   (int(W * 0.52), int(H * 0.46)), max(6, H // 90))

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
