"""
tests/test_oversegmentation.py

TDD tests for the SP2 #2 over-segmentation fix in tools/panels_to_scenes.py.

The fix folds short "sliver" panel bands (tiny reaction/text strips) into their
contiguous neighbor. The threshold is **series-agnostic**: it is expressed as a
fraction of the chapter's own median source-page height (derived from the stitch
manifest), NOT a hardcoded pixel value — so it adapts to any manhwa's resolution
and panel density automatically.

Pure functions under test:
  median_page_height(stitch_manifest) -> float
  merge_small_bands(boxes_norm, chunk_h, min_px) -> list[[ymin,xmin,ymax,xmax]]
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load tools/panels_to_scenes.py as a module (tools/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "panels_to_scenes",
    Path(__file__).resolve().parent.parent / "tools" / "panels_to_scenes.py",
)
pts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pts)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# median_page_height — the agnostic normalizer
# ---------------------------------------------------------------------------

def test_median_page_height_from_sources():
    stitch = {
        "chunks": [
            {"sources": [{"resized_h": 5000}, {"resized_h": 5500}]},
            {"sources": [{"resized_h": 6000}]},
        ]
    }
    # heights = [5000, 5500, 6000] -> median 5500
    assert pts.median_page_height(stitch) == 5500


def test_median_page_height_falls_back_to_orig_h():
    stitch = {"chunks": [{"sources": [{"orig_h": 4000}, {"orig_h": 4200}]}]}
    assert pts.median_page_height(stitch) == 4100


def test_median_page_height_empty_returns_zero():
    assert pts.median_page_height({"chunks": []}) == 0.0
    assert pts.median_page_height({}) == 0.0


# ---------------------------------------------------------------------------
# merge_small_bands — the agnostic merge
# ---------------------------------------------------------------------------

def _h(box):
    return box[2] - box[0]


def test_merge_small_middle_band_into_previous():
    # chunk_h=1000; bands at y: [0,0.4] big, [0.4,0.45] sliver (50px), [0.45,0.9] big
    boxes = [
        [0.0, 0.0, 0.40, 1.0],
        [0.40, 0.0, 0.45, 1.0],   # 50px sliver
        [0.45, 0.0, 0.90, 1.0],
    ]
    out = pts.merge_small_bands(boxes, chunk_h=1000, min_px=120)
    # sliver folds into its previous neighbor -> 2 bands
    assert len(out) == 2
    # first band now spans 0.0..0.45 (absorbed the sliver)
    assert out[0][0] == pytest.approx(0.0)
    assert out[0][2] == pytest.approx(0.45)
    assert out[1][0] == pytest.approx(0.45)


def test_merge_leading_sliver_folds_into_next():
    boxes = [
        [0.0, 0.0, 0.05, 1.0],    # 50px leading sliver
        [0.05, 0.0, 0.6, 1.0],
    ]
    out = pts.merge_small_bands(boxes, chunk_h=1000, min_px=120)
    assert len(out) == 1
    assert out[0][0] == pytest.approx(0.0)
    assert out[0][2] == pytest.approx(0.6)


def test_merge_no_slivers_unchanged():
    boxes = [
        [0.0, 0.0, 0.5, 1.0],
        [0.5, 0.0, 1.0, 1.0],
    ]
    out = pts.merge_small_bands(boxes, chunk_h=1000, min_px=120)
    assert len(out) == 2


def test_merge_disabled_when_min_px_zero():
    boxes = [
        [0.0, 0.0, 0.40, 1.0],
        [0.40, 0.0, 0.42, 1.0],   # tiny
        [0.42, 0.0, 0.9, 1.0],
    ]
    out = pts.merge_small_bands(boxes, chunk_h=1000, min_px=0)
    assert len(out) == 3  # no merging when threshold disabled


def test_merge_union_preserves_x_extent():
    # x extents differ; union should take the widest span
    boxes = [
        [0.0, 0.1, 0.4, 0.8],
        [0.4, 0.0, 0.45, 1.0],   # sliver wider in x
    ]
    out = pts.merge_small_bands(boxes, chunk_h=1000, min_px=120)
    assert len(out) == 1
    assert out[0][1] == pytest.approx(0.0)   # min xmin
    assert out[0][3] == pytest.approx(1.0)   # max xmax


def test_merge_is_agnostic_dense_vs_sparse():
    # Same RULE, different densities: with min_px set from a page fraction,
    # a sparse layout (all bands large) is untouched; a dense one folds slivers.
    sparse = [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]   # 500px bands
    dense = [[i * 0.1, 0.0, i * 0.1 + 0.08, 1.0] for i in range(10)]  # 80px bands
    assert len(pts.merge_small_bands(sparse, 1000, min_px=120)) == 2  # untouched
    assert len(pts.merge_small_bands(dense, 1000, min_px=120)) < 10   # folded
