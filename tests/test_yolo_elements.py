"""
tests/test_yolo_elements.py

Step 1 of the detector plan: use the non-panel classes the webtoon model
already knows (speech_bubble / system_box / sfx).

(b) snap_panels_to_elements — a panel edge that slices through a detected
    bubble grows to swallow it (kills boundary bubble-remnants at the source).
(a) chunk_box_to_scene_local — map chunk-space element boxes into scene-crop
    coordinates for pixel-accurate inpaint masks (remnant slivers included).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from studio.detect.yolo_panels import snap_panels_to_elements

_SPEC = importlib.util.spec_from_file_location(
    "clean_panels_inpaint",
    Path(__file__).resolve().parent.parent / "tools" / "clean_panels_inpaint.py",
)
cpi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpi)  # type: ignore[union-attr]


# ---- (b) panel-box refinement ----------------------------------------------
# Boxes are normalized [ymin, xmin, ymax, xmax] (the panels manifest format).

def test_snap_swallows_bubble_sliced_at_panel_edge():
    panels = [[0.1, 0.1, 0.5, 0.5]]
    bubble = [0.2, 0.4, 0.3, 0.55]          # 2/3 inside, sliced at xmax
    out = snap_panels_to_elements(panels, [bubble])
    assert out == [[0.1, 0.1, 0.5, 0.55]]   # grew to include the whole bubble


def test_snap_assigns_bubble_to_larger_overlap_panel_only():
    a = [0.0, 0.0, 0.5, 1.0]
    b = [0.5, 0.0, 1.0, 1.0]
    bubble = [0.45, 0.2, 0.6, 0.4]          # 1/3 in A, 2/3 in B
    out = snap_panels_to_elements([a, b], [bubble])
    assert out[0] == [0.0, 0.0, 0.5, 1.0]   # A untouched
    assert out[1] == [0.45, 0.0, 1.0, 1.0]  # B grew upward to swallow it


def test_snap_leaves_inside_and_outside_bubbles_alone():
    panels = [[0.1, 0.1, 0.5, 0.5]]
    fully_inside = [0.2, 0.2, 0.3, 0.3]
    mostly_outside = [0.45, 0.45, 0.9, 0.9]  # only a corner overlaps
    out = snap_panels_to_elements(panels, [fully_inside, mostly_outside])
    assert out == [[0.1, 0.1, 0.5, 0.5]]


def test_snap_keeps_panels_sorted_by_ymin():
    a = [0.6, 0.0, 0.9, 1.0]
    b = [0.1, 0.0, 0.4, 1.0]
    out = snap_panels_to_elements([a, b], [])
    assert out == [b, a]


# ---- (a) chunk-space element box -> scene-local inpaint mask ----------------
# Pixel xyxy in both spaces; scene crop given by its box_px_xyxy in the chunk.

def test_chunk_box_maps_into_scene_local_coords():
    scene = (100, 200, 500, 800)
    bubble = (450, 300, 600, 400)            # right part lies outside the crop
    out = cpi.chunk_box_to_scene_local(bubble, scene)
    assert out == (350, 100, 400, 200)       # clipped + origin-shifted


def test_chunk_box_remnant_sliver_is_kept():
    scene = (100, 200, 500, 800)
    bubble = (490, 300, 700, 400)            # only a 10px arc pokes into the crop
    out = cpi.chunk_box_to_scene_local(bubble, scene)
    assert out == (390, 100, 400, 200)       # the remnant IS the mask target


def test_chunk_box_outside_returns_none():
    scene = (100, 200, 500, 800)
    assert cpi.chunk_box_to_scene_local((600, 300, 700, 400), scene) is None
    assert cpi.chunk_box_to_scene_local((498, 300, 502, 400), scene) is None  # <3px
