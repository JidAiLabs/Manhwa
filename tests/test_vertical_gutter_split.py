"""Unit tests for the vertical (side-by-side) gutter split in panels_to_scenes.

The defect class: two bordered panels sitting side by side (ORV Ep2 p000006 —
two phone screens) detected as ONE box. The split requires a thin full-height
blank column run flanked on BOTH sides by contiguous dark border lines; a
blank run without borders (plain white background art) must NOT split.
"""

import os

import numpy as np
import pytest
from PIL import Image

from tools.panels_to_scenes import (
    GutterParams,
    _find_best_internal_vertical_gutter_run,
    split_crop_on_gutters,
)

P000006_CHUNK = ("/Users/anka/webtoon-ai/dataset_v2/corpus/omniscient-reader/"
                 "Episode_2/stitch_chunks/chunk_0001.jpg")


def _gp() -> GutterParams:
    # mirrors main(): blank_thr/edge_max defaults, prod min_run/margin
    return GutterParams(min_run_px=90, margin_px=24, max_splits=4)


def _noise(rng, w, h):
    return rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)


def _side_by_side(borders: bool = True, bridge: bool = False) -> Image.Image:
    """Synthetic 800x960 crop: two panels split by a ~12px white gutter at
    x=388..400. borders=False removes the panel border lines (keeps content).
    bridge=True draws a dark band across the gutter (sfx overlapping both)."""
    rng = np.random.default_rng(7)
    arr = np.full((960, 800, 3), 255, dtype=np.uint8)
    # panel interiors (busy art, not blank)
    arr[80:900, 20:385] = _noise(rng, 365, 820)
    arr[80:900, 403:780] = _noise(rng, 377, 820)
    if borders:
        # 3px dark border rectangles around each panel
        for x0, x1 in ((17, 388), (400, 783)):
            arr[77:900, x0:x0 + 3] = 10   # left edge
            arr[77:900, x1 - 3:x1] = 10   # right edge
            arr[77:80, x0:x1] = 10        # top
            arr[897:900, x0:x1] = 10      # bottom
    if bridge:
        arr[400:520, 300:520] = 10  # dark band crossing the gutter
    return Image.fromarray(arr)


def test_side_by_side_bordered_splits():
    im = _side_by_side(borders=True)
    parts = split_crop_on_gutters(im, [0, 0, 800, 960], [], _gp(),
                                  min_h_px=180, min_w_px=240)
    assert len(parts) == 2
    (left_im, left_box, _), (right_im, right_box, _) = parts
    # split lands inside the gutter band
    assert 385 <= left_box[2] <= 403
    assert left_box[2] == right_box[0]
    assert left_im.width >= 240 and right_im.width >= 240
    # full height preserved on both parts
    assert left_box[1] == 0 and left_box[3] == 960
    assert right_box[1] == 0 and right_box[3] == 960


def test_no_borders_no_split():
    # same blank gap, but no flanking border lines -> weak evidence, keep whole
    im = _side_by_side(borders=False)
    parts = split_crop_on_gutters(im, [0, 0, 800, 960], [], _gp(),
                                  min_h_px=180, min_w_px=240)
    assert len(parts) == 1


def test_bridged_gutter_no_split():
    # dark sfx band crossing the gutter kills the full-height blank run
    im = _side_by_side(borders=True, bridge=True)
    parts = split_crop_on_gutters(im, [0, 0, 800, 960], [], _gp(),
                                  min_h_px=180, min_w_px=240)
    assert len(parts) == 1


def test_protected_spans_disable_vertical_split():
    im = _side_by_side(borders=True)
    parts = split_crop_on_gutters(im, [0, 0, 800, 960], [[430, 470]], _gp(),
                                  min_h_px=180, min_w_px=240)
    assert len(parts) == 1


def test_narrow_part_no_split():
    # gutter too close to the edge: left part would be < min_w_px
    rng = np.random.default_rng(7)
    arr = np.full((960, 800, 3), 255, dtype=np.uint8)
    arr[80:900, 20:145] = _noise(rng, 125, 820)
    arr[80:900, 163:780] = _noise(rng, 617, 820)
    for x0, x1 in ((17, 148), (160, 783)):
        arr[77:900, x0:x0 + 3] = 10
        arr[77:900, x1 - 3:x1] = 10
    parts = split_crop_on_gutters(Image.fromarray(arr), [0, 0, 800, 960], [],
                                  _gp(), min_h_px=180, min_w_px=240)
    assert len(parts) == 1


def test_stacked_row_split_still_wins():
    # two stacked panels with a 120px horizontal gutter: the existing row
    # split fires; the vertical pass must not preempt or break it
    rng = np.random.default_rng(7)
    arr = np.full((1200, 800, 3), 255, dtype=np.uint8)
    arr[40:520, 30:770] = _noise(rng, 740, 480)
    arr[640:1160, 30:770] = _noise(rng, 740, 520)
    parts = split_crop_on_gutters(Image.fromarray(arr), [0, 0, 800, 1200], [],
                                  _gp(), min_h_px=180, min_w_px=240)
    assert len(parts) == 2
    (_, top_box, _), (_, bot_box, _) = parts
    assert 520 <= top_box[3] <= 640
    assert top_box[3] == bot_box[1]


@pytest.mark.skipif(not os.path.exists(P000006_CHUNK),
                    reason="corpus chunk not on this machine")
def test_real_p000006_side_by_side_splits():
    with Image.open(P000006_CHUNK) as chunk:
        crop = chunk.convert("RGB").crop((0, 3887, 800, 4847))
    found = _find_best_internal_vertical_gutter_run(crop, _gp(), 240)
    assert found is not None
    x0, x1 = found
    # measured gutter: borders at ~301-303 and ~314-316, white run between
    assert 295 <= x0 <= x1 <= 320

    parts = split_crop_on_gutters(crop, [0, 3887, 800, 4847], [], _gp(),
                                  min_h_px=180, min_w_px=240)
    assert len(parts) == 2
    (_, left_box, _), (_, right_box, _) = parts
    assert left_box[2] == right_box[0]
    assert 295 <= left_box[2] <= 320
