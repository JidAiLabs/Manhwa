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
    assert out[1] == [0.5, 0.0, 1.0, 1.0]   # B grows only to the shared edge,
    #                                         not up into A (no overlapping crop)


def test_snap_does_not_cross_into_neighbor_panel():
    # Two stacked panels with a real gutter (top ends 0.45, bottom starts 0.50).
    top = [0.10, 0.0, 0.45, 1.0]
    bot = [0.50, 0.0, 0.90, 1.0]
    # A bubble mostly inside TOP but slicing DOWN across the gutter into BOTTOM.
    bubble = [0.20, 0.30, 0.58, 0.60]
    out = snap_panels_to_elements([top, bot], [bubble])
    grown_top, kept_bot = out[0], out[1]
    # Top grows down only to the gutter midpoint (0.475), NOT into BOTTOM's band.
    assert grown_top[2] <= kept_bot[0], (
        f"snapped top ymax {grown_top[2]} crosses into bottom ymin {kept_bot[0]}")
    assert grown_top == [0.1, 0.0, 0.475, 1.0]
    assert kept_bot == [0.5, 0.0, 0.9, 1.0]


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


# ---- (c) attach gutter dialogue to its panel (art-only detector, 2026-08-17) --
# v4 boxes hug the art, so a bubble hanging off a panel edge (or floating in
# the gutter right next to it) is no longer inside the box. attach_gap (norm.)
# pulls such VERTICALLY-ADJACENT elements in when they sit over the panel's
# x-span; corner touches / far floaters stay out. Default 0.0 = legacy behavior.

def test_attach_pulls_in_bubble_hanging_off_the_top_edge():
    panel = [0.30, 0.0, 0.60, 1.0]
    bubble = [0.24, 0.10, 0.31, 0.50]        # ~14% inside (below the 0.55 slice rule)
    assert snap_panels_to_elements([panel], [bubble]) == [panel]          # legacy: untouched
    out = snap_panels_to_elements([panel], [bubble], attach_gap=0.03)
    assert out == [[0.24, 0.0, 0.60, 1.0]]


def test_attach_pulls_in_floating_gutter_bubble_within_gap():
    panel = [0.30, 0.0, 0.60, 1.0]
    floating = [0.26, 0.20, 0.29, 0.60]      # 0.01 above the panel, no overlap
    out = snap_panels_to_elements([panel], [floating], attach_gap=0.03)
    assert out == [[0.26, 0.0, 0.60, 1.0]]
    far = [0.10, 0.20, 0.20, 0.60]           # 0.10 above — beyond the gap
    assert snap_panels_to_elements([panel], [far], attach_gap=0.03) == [panel]


def test_attach_ignores_corner_touch_and_side_floaters():
    panel = [0.1, 0.1, 0.5, 0.5]
    corner = [0.45, 0.45, 0.9, 0.9]          # legacy 'mostly_outside' case
    beside = [0.2, 0.52, 0.3, 0.9]           # to the RIGHT of the panel, not above/below
    out = snap_panels_to_elements([panel], [corner, beside], attach_gap=0.05)
    assert out == [panel]


def test_attach_goes_to_nearer_panel_and_never_crosses_neighbor():
    top = [0.10, 0.0, 0.40, 1.0]
    bot = [0.50, 0.0, 0.90, 1.0]
    bubble = [0.44, 0.2, 0.49, 0.6]          # in the gutter, 0.01 from bot, 0.04 from top
    out = snap_panels_to_elements([top, bot], [bubble], attach_gap=0.05)
    assert out[0] == top                     # top untouched
    assert out[1] == [0.44, 0.0, 0.90, 1.0]  # bot grew up to the bubble, still below top
    assert out[1][0] >= out[0][2]


def test_attach_prefers_the_panel_the_bubble_overlaps_and_never_bisects():
    # nano ch1 chunk_0001 (px/10000): hand panel upper-left, eye panel lower-right
    # (x-overlapping), "!!!" bubble in the gutter overlapping the EYE panel's
    # top-left corner (26% inside). It belongs to the eye panel — attaching it to
    # the hand panel above (gap rule) and clamping at the eye panel's top would
    # bisect it and drag the expander into the eye panel (duplicate scene bug).
    hand = [0.5465, 0.000, 0.5838, 0.630]
    eye = [0.6118, 0.302, 0.6518, 0.800]
    bubble = [0.6025, 0.116, 0.6277, 0.429]
    out = snap_panels_to_elements([hand, eye], [bubble], attach_gap=0.03)
    assert out[0] == hand                                    # untouched
    assert out[1] == [0.6025, 0.116, 0.6518, 0.800]          # eye grew to the whole bubble


def test_attach_skips_when_the_only_candidate_would_bisect():
    # bubble hangs below TOP over its x-span but a side panel intersects the growth
    top = [0.10, 0.0, 0.40, 0.60]
    side = [0.42, 0.65, 0.90, 1.00]          # lower-right, clear of top's x-span
    bubble = [0.41, 0.05, 0.47, 0.25]        # 0.01 below top
    out = snap_panels_to_elements([top, side], [bubble], attach_gap=0.05)
    assert out[0] == [0.10, 0.0, 0.47, 0.60] # grown box misses side -> whole bubble taken
    side2 = [0.42, 0.30, 0.90, 1.00]         # x-overlaps top: growth would intersect it
    out2 = snap_panels_to_elements([top, side2], [bubble], attach_gap=0.05)
    assert out2[0] == top                    # would bisect/overlap -> left alone entirely


# ---- (d) floating cards become panels (2026-08-18) ----------------------------
# v4 (correctly) does not box text cards as panels, so a chapter's ending
# "SKY CORPORATION." / "7TH GENERATION NANO MACHINE" system cards only survived
# as one 3395px recovered strip mixed with the logo/credits, which the visual
# judge called garbage — orphaning the narrated ending into a 15s hold. The
# tiled element pass DOES localize them (radio/caption/system boxes floating in
# the gap): promote element boxes that overlap NO panel to panels of their own.
from studio.detect.yolo_panels import promote_floating_cards


def test_floating_cards_become_panels_and_overlapping_pieces_merge():
    panels = [[0.10, 0.0, 0.40, 1.0]]
    els = {"radio": [[0.60, 0.20, 0.65, 0.60], [0.70, 0.15, 0.75, 0.50]],
           "caption_box": [[0.74, 0.35, 0.79, 0.70]],       # overlaps the 2nd radio -> merge
           "speech_bubble": [[0.30, 0.10, 0.35, 0.30]],      # inside the panel -> not promoted
           "sfx": [[0.85, 0.1, 0.9, 0.5]]}                    # sfx never
    out = promote_floating_cards(panels, els)
    assert out == [[0.10, 0.0, 0.40, 1.0], [0.60, 0.20, 0.65, 0.60], [0.70, 0.15, 0.79, 0.70]]


def test_floating_card_touching_a_panel_is_left_alone():
    panels = [[0.10, 0.0, 0.40, 1.0]]
    els = {"caption_box": [[0.38, 0.2, 0.45, 0.6]]}          # partly over the panel -> not a card
    assert promote_floating_cards(panels, els) == panels


def test_floating_cards_report_their_classes():
    from studio.detect.yolo_panels import floating_cards
    panels = [[0.10, 0.0, 0.40, 1.0]]
    els = {"radio": [[0.60, 0.20, 0.65, 0.60]], "caption_box": [[0.62, 0.30, 0.66, 0.70]],
           "speech_bubble": [[0.30, 0.10, 0.35, 0.30]]}
    cards = floating_cards(panels, els)
    assert cards == [{"box": [0.60, 0.20, 0.66, 0.70], "classes": ["caption_box", "radio"]}]
