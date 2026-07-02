"""Unit tests for geometric overlap/containment dedup in panels_to_scenes.

Boxes are in pixel xyxy format: [x0, y0, x1, y1] (matching box_xyxy in the
cropper), which is what the dedup pass operates on.
"""

import pytest

from tools.panels_to_scenes import (
    box_iou,
    box_containment,
    dedupe_overlapping_boxes,
    same_strip_overlap,
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


# -----------------------------
# same_strip_overlap (cross-chunk overlap-band dedupe)
# -----------------------------
def _strip(seq, gy0, y0, y1, overlap=700):
    """Build a dedupe-memory entry from REAL manifest fields: chunk stitch
    index, naive global offset, and the crop's chunk-local y-range."""
    base = gy0 - seq * overlap
    return {"seq": seq, "ty0": base + y0, "ty1": base + y1}


def test_same_strip_real_ch1_overlap_dup():
    # Nano ch1 p000063/p000064: the SAME panel captured by chunks 7 and 8 via
    # the 700px stitch overlap (dhash Hamming 5). It survived dedupe because
    # per-chunk trim variance broke the ratio guard (2.025 vs 2.133, tol 0.08).
    # True-global ranges nearly coincide: 73975-74370 vs 74001-74370.
    a = _strip(6, 68786, 9389, 9784)   # chunk_0007
    b = _strip(7, 78889, 12, 381)      # chunk_0008
    assert same_strip_overlap(a, b, 700)


def test_same_strip_real_ch1_second_dup():
    # p000084/p000085 (chunks 9->10, dhash Hamming 7)
    a = _strip(8, 90817, 11135, 11495)
    b = _strip(9, 102614, 20, 398)
    assert same_strip_overlap(a, b, 700)


def test_same_strip_rejects_same_chunk():
    a = _strip(6, 68786, 1000, 1400)
    b = _strip(6, 68786, 5000, 5400)
    assert not same_strip_overlap(a, b, 700)


def test_same_strip_rejects_non_adjacent_chunks():
    a = _strip(4, 43983, 9389, 9784)
    b = _strip(7, 78889, 12, 381)
    assert not same_strip_overlap(a, b, 700)


def test_same_strip_rejects_disjoint_ranges():
    # adjacent chunks but panels nowhere near the shared band
    a = _strip(6, 68786, 1000, 1400)   # far above chunk_0007's bottom
    b = _strip(7, 78889, 5000, 5400)   # far below chunk_0008's top
    assert not same_strip_overlap(a, b, 700)


def test_same_strip_rejects_thin_partial_overlap():
    # adjacent + touching ranges, but shared rows < 50% of the smaller crop
    a = _strip(6, 68786, 9389, 9784)   # true 73975-74370 (h=395)
    b = _strip(7, 78889, 200, 900)     # true 74189-74889 -> 181px shared
    assert not same_strip_overlap(a, b, 700)


def test_same_strip_disabled_without_overlap_px():
    a = _strip(6, 68786, 9389, 9784, overlap=0)
    b = _strip(7, 78889, 12, 381, overlap=0)
    assert not same_strip_overlap(a, b, 0)
