"""Unit tests for geometric overlap/containment dedup in panels_to_scenes.

Boxes are in pixel xyxy format: [x0, y0, x1, y1] (matching box_xyxy in the
cropper), which is what the dedup pass operates on.
"""

import pytest

from tools.panels_to_scenes import (
    box_iou,
    box_containment,
    dedupe_overlapping_boxes,
)


# -----------------------------
# box_iou
# -----------------------------
def test_box_iou_identical():
    a = [0, 0, 100, 100]
    assert box_iou(a, list(a)) == pytest.approx(1.0)


def test_box_iou_disjoint():
    a = [0, 0, 10, 10]
    b = [100, 100, 110, 110]
    assert box_iou(a, b) == 0.0


def test_box_iou_disjoint_touching_edge():
    # share an edge but zero overlap area
    a = [0, 0, 10, 10]
    b = [10, 0, 20, 10]
    assert box_iou(a, b) == 0.0


def test_box_iou_known_partial_overlap():
    # a = [0,0,10,10] area 100; b = [5,0,15,10] area 100
    # intersection = x in [5,10], y in [0,10] -> 5*10 = 50
    # union = 100 + 100 - 50 = 150 ; IoU = 50/150 = 1/3
    a = [0, 0, 10, 10]
    b = [5, 0, 15, 10]
    assert box_iou(a, b) == pytest.approx(1.0 / 3.0)


# -----------------------------
# box_containment
# -----------------------------
def test_box_containment_inner_fully_inside_outer():
    outer = [0, 0, 100, 100]
    inner = [10, 10, 50, 50]
    # area(inner ∩ outer) / area(inner) == 1.0
    assert box_containment(inner, outer) == pytest.approx(1.0)


def test_box_containment_disjoint():
    inner = [0, 0, 10, 10]
    outer = [100, 100, 110, 110]
    assert box_containment(inner, outer) == 0.0


def test_box_containment_partial():
    # inner = [0,0,10,10] area 100 ; outer = [5,0,15,10]
    # intersection = 5*10 = 50 ; containment = 50/100 = 0.5
    inner = [0, 0, 10, 10]
    outer = [5, 0, 15, 10]
    assert box_containment(inner, outer) == pytest.approx(0.5)


# -----------------------------
# dedupe_overlapping_boxes
# -----------------------------
def test_dedupe_drops_contained_smaller_box():
    bigA = [0, 0, 100, 200]          # area 20000
    smallB = [10, 10, 90, 100]       # inside A, area 7200
    separateC = [500, 500, 600, 700]  # disjoint
    kept = dedupe_overlapping_boxes(
        [bigA, smallB, separateC], iou_thr=0.6, contain_thr=0.8
    )
    assert sorted(kept) == [0, 2]


def test_dedupe_high_iou_keeps_larger():
    # construct two boxes with IoU ~0.7, different sizes -> keep larger
    # A = [0,0,100,100] area 10000
    # B = [0,0,100,118] -> intersection 10000, union 11800, IoU ~0.847
    # Use a pair with IoU >= 0.6 where B is larger so larger (B) survives.
    A = [0, 0, 100, 100]
    B = [0, 0, 100, 130]  # area 13000, inter 10000, union 13000, IoU ~0.769
    assert box_iou(A, B) >= 0.6
    kept = dedupe_overlapping_boxes([A, B], iou_thr=0.6, contain_thr=0.8)
    assert kept == [1]  # the larger box B


def test_dedupe_iou_exactly_above_threshold_drops_one():
    a = [0, 0, 100, 100]
    b = [5, 0, 105, 100]  # inter 95*100=9500, union 10500, IoU ~0.905
    assert box_iou(a, b) >= 0.6
    kept = dedupe_overlapping_boxes([a, b], iou_thr=0.6, contain_thr=0.8)
    assert len(kept) == 1


def test_dedupe_non_overlapping_keeps_all():
    boxes = [
        [0, 0, 10, 10],
        [100, 100, 110, 110],
        [200, 200, 210, 210],
    ]
    kept = dedupe_overlapping_boxes(boxes, iou_thr=0.6, contain_thr=0.8)
    assert sorted(kept) == [0, 1, 2]


def test_dedupe_low_overlap_keeps_all():
    # IoU 1/3 < 0.6 and containment 0.5 < 0.8 -> keep both
    a = [0, 0, 10, 10]
    b = [5, 0, 15, 10]
    kept = dedupe_overlapping_boxes([a, b], iou_thr=0.6, contain_thr=0.8)
    assert sorted(kept) == [0, 1]


def test_dedupe_empty():
    assert dedupe_overlapping_boxes([], iou_thr=0.6, contain_thr=0.8) == []


def test_dedupe_preserves_order_of_kept_indices():
    boxes = [
        [0, 0, 10, 10],
        [100, 100, 110, 110],
        [105, 105, 108, 108],  # contained in box 1
    ]
    kept = dedupe_overlapping_boxes(boxes, iou_thr=0.6, contain_thr=0.8)
    assert kept == [0, 1]
