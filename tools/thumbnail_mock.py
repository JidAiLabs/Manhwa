#!/usr/bin/env python3
"""thumbnail_mock.py -- FREE thumbnail mocks: the hook's layout and labels drawn
on the story's REAL panels. No paid painter.

Owner, 2026-09-29: "we keep spending money and tokens but we are cycling". Every
correction used to be a repaint. The owner now sees the composition (which
moment, which label, split or hero, the arrow) on a mock first, and pays for
art only after picking one.

A mock is a claim option (publish_concept.choose_layout):
  {"name", "layout": "hero"|"split", "labels": [...], "panels": [path, ...],
   "arrow": "hero"|"object"|None}
  hero  -- one panel framed to 16:9, the label upper-left; an arrow ONLY when a
           face or body is detected on the composed image (a fixed arrow pointed
           at a chin or at air: owner, 2026-09-27).
  split -- two halves, LOW on the left and HIGH on the right, each labelled at
           the bottom (the examples' F- | S-RANK, S+ | SSS+).
The video title is shown as HTML under the mock, never baked in, so changing
the title never needs a re-render.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from thumbnail_overlay import _arrow, _fitted, _label_lines, _outlined  # noqa: E402

W, H = 1280, 720


def _faces(path: str) -> List[List[float]]:
    """Face (else human-body) boxes [x0,y0,x1,y1] in fractions, largest first.
    [] when on-device Vision is unavailable (not macOS) or finds nothing."""
    try:
        from apple_vision import faces
        got = faces(path)
    except Exception:
        return []
    boxes = [list(map(float, f["bbox"])) for f in got if f.get("bbox")]
    return sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


def focus_of(path: str) -> Tuple[float, float]:
    """Where to keep the panel's subject: the largest face's centre, else a
    point in the upper-middle (manhwa panels put faces high)."""
    b = _faces(path)
    if b:
        x0, y0, x1, y1 = b[0]
        return ((x0 + x1) / 2, (y0 + y1) / 2)
    return (0.5, 0.4)


def _cover(im: Image.Image, tw: int, th: int,
           focus: Tuple[float, float]) -> Image.Image:
    """Scale to COVER tw x th and crop around *focus* (never stretch)."""
    aw, ah = im.size
    s = max(tw / aw, th / ah)
    im = im.resize((max(tw, int(aw * s + 0.5)), max(th, int(ah * s + 0.5))),
                   Image.LANCZOS)
    aw, ah = im.size
    x0 = max(0, min(aw - tw, int(aw * focus[0] - tw / 2)))
    y0 = max(0, min(ah - th, int(ah * focus[1] - th * 0.42)))
    return im.crop((x0, y0, x0 + tw, y0 + th))


def frame_panel(path: str, tw: int = W, th: int = H, *,
                focus: Optional[Tuple[float, float]] = None,
                x_center: float = 0.66, cover: bool = False) -> Image.Image:
    """A webtoon panel in a tw x th frame without black bars or stretching.

    Wide panels (aspect >= 1.2 x the frame's height ratio) are cover-cropped
    around the focus. Tall ones (most webtoon panels: ORV Ep6 has 61 portrait of
    105) sit at full height on a blurred, darkened copy of themselves, centred
    at x_center so the label side stays free; a very tall strip is first cut
    to a 3:4 window around the focus."""
    im = Image.open(path).convert("RGB")
    focus = focus or focus_of(path)
    aw, ah = im.size
    if cover or aw / ah >= min(1.2, tw / th):
        return _cover(im, tw, th, focus)
    if aw / ah < 0.42:                       # a strip: keep a 3:4 window of it
        ch = int(aw / 0.75)
        y0 = max(0, min(ah - ch, int(ah * focus[1] - ch * 0.42)))
        im = im.crop((0, y0, aw, y0 + ch))
        aw, ah = im.size
    bg = _cover(im, tw, th, focus).filter(ImageFilter.GaussianBlur(24))
    bg = ImageEnhance.Brightness(bg).enhance(0.55)
    fg = im.resize((max(1, int(aw * th / ah)), th), Image.LANCZOS)
    x = max(0, min(tw - fg.width, int(tw * x_center - fg.width / 2)))
    bg.paste(fg, (x, 0))
    return bg


def render_mock(spec: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """Draw one option. Returns {"image", "arrow_box"} (arrow_box None when no
    arrow was drawn, and why is in "arrow_note")."""
    panels = [p for p in (spec.get("panels") or []) if p and os.path.exists(p)]
    labels = [str(x).strip().upper() for x in (spec.get("labels") or []) if str(x).strip()]
    out = {"image": out_path, "arrow_box": None, "arrow_note": ""}
    if not panels:
        out["arrow_note"] = "no panel on disk"
        return dict(out, image="")
    if spec.get("layout") == "split" and len(panels) >= 2:
        img = Image.new("RGB", (W, H))
        for i, p in enumerate(panels[:2]):
            img.paste(frame_panel(p, W // 2, H, cover=True), (i * W // 2, 0))
        draw = ImageDraw.Draw(img)
        draw.line([(W // 2, 0), (W // 2, H)], fill=(255, 255, 255), width=6)
        for i, text in enumerate(labels[:2]):
            f = _fitted(draw, text, int(H * 0.15), int(W * 0.42))
            _outlined(draw, (int(W * (0.25 + 0.5 * i)), int(H * 0.80)), text, f,
                      anchor="ma")
    else:
        img = frame_panel(panels[0])
        draw = ImageDraw.Draw(img)
        if labels:
            text = labels[0]
            lines, f = _label_lines(draw, text, int(H * 0.16), int(W * 0.40))
            step = int(f.size * 1.05)
            for i, line in enumerate(lines):
                _outlined(draw, (int(W * 0.04), int(H * 0.06) + i * step), line,
                          f, anchor="la")
            if spec.get("arrow") == "hero":
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
                    img.save(t.name, quality=90)
                    boxes = _faces(t.name)
                os.unlink(t.name)
                box = next((b for b in boxes if (b[0] + b[2]) / 2 > 0.42), None)
                if box:
                    # label edge -> just outside the subject's left side, at its top
                    tip = (int(W * max(0.0, box[0] - 0.01)),
                           int(H * (box[1] + (box[3] - box[1]) * 0.35)))
                    lw = max(int(draw.textlength(ln, font=f)) for ln in lines)
                    start = (int(W * 0.04) + min(lw, int(W * 0.40)) // 2,
                             int(H * 0.06) + len(lines) * step + int(H * 0.02))
                    if tip[0] - start[0] > int(W * 0.05):
                        _arrow(draw, start, tip, max(6, H // 90))
                        out["arrow_box"] = box
                    else:
                        out["arrow_note"] = "subject too close to the label"
                else:
                    out["arrow_note"] = "no face or body found on the image"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path, quality=90)
    return out


def render_claim_mocks(claim: Dict[str, Any], out_dir: str) -> List[Dict[str, Any]]:
    """Render every option of *claim* into out_dir/<name>.jpg; the results are
    written back onto the claim's mocks (image, arrow_box, arrow_note)."""
    done: List[Dict[str, Any]] = []
    for m in claim.get("mocks") or []:
        r = render_mock(m, os.path.join(out_dir, "%s.jpg" % m.get("name", "mock")))
        m.update(r)
        done.append(m)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--claim", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    claim = json.load(open(args.claim, encoding="utf-8"))
    for m in render_claim_mocks(claim, args.out_dir):
        print("[ok] %s %s %s arrow=%s %s" % (m["name"], m["layout"], m["labels"],
                                            bool(m.get("arrow_box")),
                                            m.get("arrow_note") or ""))
    with open(args.claim, "w", encoding="utf-8") as f:
        json.dump(claim, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
