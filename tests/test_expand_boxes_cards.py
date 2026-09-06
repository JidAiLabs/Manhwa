"""System cards are refined to their own fill colour (2026-09-06, ORV Ep128).

YOLO boxed a 187px sliver of a ~1000px blue scenario card; the gutter walk
recovered the height (clean white above/below) but not the width — the
card's decorated frame breaks the pure-white column run before the page
edge — so the scene crop kept half of every text line and Apple OCR read
"Main scenar… abandoned w… to extin…". A card is a flat-fill rectangle, so
its own colour is the boundary."""
from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image

import tools.expand_boxes_to_gutters as ebg

W, H = 800, 1500
CARD = (45, 300, 760, 1290)          # x0, y0, x1, y1 of the blue fill
SLIVER = [956 / H, 89 / W, 1143 / H, 464 / W]   # ymin, xmin, ymax, xmax (norm)
ART = [0.0, 0.0, 250 / H, 1.0]


def _page(path):
    rng = np.random.default_rng(3)
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    img[0:250] = rng.integers(40, 215, (250, W, 3), dtype=np.uint8)   # art
    x0, y0, x1, y1 = CARD
    img[y0:y1, x0:x1] = (40, 90, 220)                                # card
    for y in range(340, 1250, 60):                                   # white "text"
        img[y:y + 12, 100:700] = 255
    img[y0:y1, x1:W] = (20, 20, 20)     # dark frame to the page edge: no white run
    Image.fromarray(img).save(path, "JPEG", quality=92)


def _run(tmp_path, cards_norm):
    chunk = tmp_path / "chunk_0000.jpg"
    _page(chunk)
    stitch = {"chunks": [{"chunk_file": "chunk_0000.jpg", "chunk_path": str(chunk)}]}
    ch = {"chunk_file": "chunk_0000.jpg", "panels_norm": [ART, SLIVER]}
    if cards_norm is not None:
        ch["cards_norm"] = cards_norm
    sp, pp, op = tmp_path / "s.json", tmp_path / "p.json", tmp_path / "o.json"
    sp.write_text(json.dumps(stitch))
    pp.write_text(json.dumps({"chunks": [ch]}))
    old = sys.argv
    sys.argv = ["expand_boxes_to_gutters.py", "--stitch-manifest", str(sp),
                "--panels-manifest", str(pp), "--out-panels-manifest", str(op)]
    try:
        ebg.main()
    finally:
        sys.argv = old
    out = json.loads(op.read_text())["chunks"][0]
    px = [ebg.norm_to_px(b, W, H) for b in out["panels_norm"]]
    return out, px


def test_card_box_refines_to_the_fill_and_plain_boxes_do_not(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with_cards, px_c = _run(tmp_path / "a", [{"box": SLIVER, "classes": ["system_box"]}])
    without, px_n = _run(tmp_path / "b", None)

    art_c, card_c = px_c
    art_n, card_n = px_n
    # the card: width recovered from the sliver to the fill (±3px of JPEG edge)
    assert card_c[0] <= CARD[0] + 3 and card_c[2] >= CARD[2] - 3, card_c
    assert card_c[1] <= CARD[1] + 3 and card_c[3] >= CARD[3] - 3, card_c
    assert with_cards["expanded"]["cards_refined"] == 1
    # no card class -> the gutter walk alone, which cannot widen past the frame
    assert card_n[0] == 89 and card_n[2] == 464, card_n
    assert without["expanded"]["cards_refined"] == 0
    # an ordinary art box is untouched by the card pass
    assert art_c == art_n


def test_refine_leaves_gutter_coloured_interiors_alone():
    arr = np.full((200, 200, 3), 255, dtype=np.uint8)      # a white "card"
    assert ebg.refine_card_box(arr, [50, 50, 100, 100]) == [50, 50, 100, 100]
