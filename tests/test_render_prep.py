"""
tests/test_render_prep.py

TDD for tools/render_prep.py — the render-prep stage between `planned` and the
renderer. Three user-reported defects from the first watch-through of ch1:

1. cross-chunk seam duplicates still rendered (p000015 full panel + p000016,
   the same artwork re-detected at the next chunk's top) -> drop contained
   fragment cuts using GLOBAL page coordinates, redistribute the freed time;
2. bubble dialogue text still visible -> ogkalu bubble boxes + oval-aware
   inpaint into scenes_clean/;
3. baked page margins shown (white border around the art) -> trim uniform
   light borders; emit scene dims so the renderer can go full-bleed on wide
   panels.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "render_prep",
    Path(__file__).resolve().parent.parent / "tools" / "render_prep.py",
)
rp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rp)  # type: ignore[union-attr]


# ---- 1. cross-chunk contained-fragment filter -------------------------------

def _geom(global_y0, box):
    x1, y1, x2, y2 = box
    return {"x1": x1, "y1": global_y0 + y1, "x2": x2, "y2": global_y0 + y2}


def test_contained_fragment_cut_is_dropped_and_time_redistributed():
    # mirrors the real p000015/p000016 pair: chunk2 bottom vs chunk3 top
    cuts = [
        {"file": "p000015.jpg", "start": 0.0, "dur": 5.0},
        {"file": "p000016.jpg", "start": 5.0, "dur": 5.0},
    ]
    geom = {
        "p000015.jpg": _geom(10000, (2, 1104, 793, 2514)),   # global y 11104-12514
        "p000016.jpg": _geom(11826, (2, 0, 784, 688)),       # global y 11826-12514
    }
    out, dropped = rp.drop_contained_duplicate_cuts(cuts, geom, contain_frac=0.8)
    assert dropped == ["p000016.jpg"]
    assert [c["file"] for c in out] == ["p000015.jpg"]
    # the freed 5s went back to the surviving cut; shot stays fully covered
    assert out[0]["start"] == 0.0
    assert abs(out[0]["dur"] - 10.0) < 1e-6


def test_distinct_panels_are_both_kept():
    cuts = [
        {"file": "a.jpg", "start": 0.0, "dur": 4.0},
        {"file": "b.jpg", "start": 4.0, "dur": 4.0},
    ]
    geom = {
        "a.jpg": _geom(0, (0, 0, 800, 1000)),
        "b.jpg": _geom(0, (0, 1200, 800, 2200)),   # below, no overlap
    }
    out, dropped = rp.drop_contained_duplicate_cuts(cuts, geom, contain_frac=0.8)
    assert dropped == []
    assert [c["file"] for c in out] == ["a.jpg", "b.jpg"]
    assert out[1]["dur"] == 4.0


# ---- 1b. VISUAL containment (chunk_global_y0 lies across stitch overlaps:
# the real p15/p16 pair is "adjacent" in global coords yet pixel-identical,
# NCC 0.9954 — so the filter must also match pixels, not just geometry) ------

def _pattern(h, w, phase=0.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    g = (np.sin(xx / 7.0 + phase) + np.cos(yy / 11.0 + phase)) * 60 + 128
    return cv2_3c(g.clip(0, 255).astype(np.uint8))


def cv2_3c(gray):
    return np.dstack([gray, gray, gray])


def test_visual_duplicate_fragment_dropped():
    big = _pattern(400, 300)
    small = big[250:400, 0:300].copy()          # bottom crop = seam fragment
    cuts = [{"file": "big.jpg", "start": 0.0, "dur": 4.0},
            {"file": "small.jpg", "start": 4.0, "dur": 4.0}]
    out, dropped = rp.drop_visual_duplicate_cuts(
        cuts, {"big.jpg": big, "small.jpg": small})
    assert dropped == ["small.jpg"]
    assert [c["file"] for c in out] == ["big.jpg"]
    assert abs(out[0]["dur"] - 8.0) < 1e-6      # freed time redistributed


def test_visual_distinct_images_kept():
    a = _pattern(400, 300)
    b = _pattern(150, 300, phase=2.2)           # different artwork
    cuts = [{"file": "a.jpg", "start": 0.0, "dur": 4.0},
            {"file": "b.jpg", "start": 4.0, "dur": 4.0}]
    out, dropped = rp.drop_visual_duplicate_cuts(cuts, {"a.jpg": a, "b.jpg": b})
    assert dropped == []
    assert len(out) == 2


# ---- 3. uniform light border trim -------------------------------------------

def test_content_bbox_trims_light_page_margin():
    img = np.full((200, 100, 3), 235, dtype=np.uint8)   # light page margin
    img[40:160, 20:80] = 60                              # dark art block
    x1, y1, x2, y2 = rp.content_bbox(img)
    assert 15 <= x1 <= 20 and 80 <= x2 <= 85
    assert 35 <= y1 <= 40 and 160 <= y2 <= 165


def test_content_bbox_caps_trim_and_keeps_dark_art():
    img = np.full((200, 100, 3), 30, dtype=np.uint8)     # dark art everywhere
    x1, y1, x2, y2 = rp.content_bbox(img)
    assert (x1, y1) == (0, 0) and (x2, y2) == (100, 200)  # nothing trimmed


# ---- 2. oval-aware bubble mask ----------------------------------------------

def _bubble_scene(bg=90, fill=250, ring=10):
    """Synthetic panel: dark art, white oval bubble with a dark outline + text."""
    img = np.full((200, 200, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[0:200, 0:200]
    oval = ((xx - 100) / 60.0) ** 2 + ((yy - 100) / 40.0) ** 2
    img[oval <= 1.0] = ring          # outline disc
    img[oval <= 0.85] = fill         # white interior
    img[95:105, 70:130] = 20         # "text" strokes inside
    return img


def test_bubble_text_mask_targets_text_only():
    """User direction: keep the bubble (shape + outline), blank ONLY the text."""
    img = _bubble_scene()
    mask = rp.bubble_text_mask(img, (30, 50, 170, 150))
    assert mask[100, 100] > 0          # text stroke masked
    assert mask[80, 100] == 0          # plain white interior NOT masked
    assert mask[100, 41] == 0          # outline ring untouched
    assert mask[10, 10] == 0           # far art untouched


def test_clean_scene_image_blanks_text_keeps_bubble():
    img = _bubble_scene()
    out = rp.clean_scene_image(img, [(30, 50, 170, 150)])
    assert out[95:105, 72:128].mean() > 230        # text now bubble-white
    assert out[80, 100].mean() > 230               # interior still white
    assert out[100, 41].mean() < 60                # outline ring still dark
    assert abs(int(out[10:30, 10:30].mean()) - 90) <= 2   # art untouched


def test_clean_scene_image_black_shout_bubble():
    img = _bubble_scene(bg=160, fill=15, ring=240)   # black bubble, light ring
    img[95:105, 70:130] = 235                        # light text inside
    out = rp.clean_scene_image(img, [(30, 50, 170, 150)])
    assert out[95:105, 72:128].mean() < 35           # text now bubble-black


# ---- plan rewrite ------------------------------------------------------------

def test_rewrite_plan_sets_subdir_dims_and_filtered_cuts():
    plan = {"timeline": [{"segment_id": "g0001_p00", "start_sec": 0.0,
                          "duration_sec": 10.0, "end_sec": 10.0,
                          "cuts": [{"file": "a.jpg", "start": 0.0, "dur": 10.0}]}],
            "total_duration_sec": 10.0}
    out = rp.rewrite_plan(plan, scenes_subdir="scenes_clean",
                          scene_dims={"a.jpg": {"w": 800, "h": 600}},
                          cuts_by_segment={"g0001_p00": plan["timeline"][0]["cuts"]})
    assert out["scenes_subdir"] == "scenes_clean"
    assert out["scene_dims"]["a.jpg"] == {"w": 800, "h": 600}
    assert out["timeline"][0]["cuts"][0]["file"] == "a.jpg"
    # original object untouched
    assert "scenes_subdir" not in plan


# ---- branding insertion (intro after first beat, end-card outro) ------------

def _mini_plan():
    return {"timeline": [
        {"segment_id": "g0001_p00", "start_sec": 0.0, "duration_sec": 10.0,
         "end_sec": 10.0, "cuts": [{"file": "a.jpg", "start": 0.0, "dur": 10.0}]},
        {"segment_id": "g0002_p01", "start_sec": 10.0, "duration_sec": 8.0,
         "end_sec": 18.0, "cuts": [{"file": "b.jpg", "start": 0.0, "dur": 8.0}]},
    ], "total_duration_sec": 18.0}


def test_branding_inserts_intro_after_first_item_and_appends_outro():
    plan = _mini_plan()
    out = rp.insert_branding_items(plan, intro_dur=6.0, outro_dur=12.1)
    tl = out["timeline"]
    assert [t["segment_id"] for t in tl] == [
        "g0001_p00", "branding_intro", "g0002_p01", "branding_outro"]
    intro = tl[1]
    assert intro["branding"] == "intro"
    assert intro["start_sec"] == 10.0
    assert intro["duration_sec"] >= 6.0            # audio + breathing pad
    # intro shows the SAME panel the story paused on (last cut of item 1)
    assert intro["cuts"][0]["file"] == "a.jpg"
    # story resumes shifted by the intro length
    assert abs(tl[2]["start_sec"] - (10.0 + intro["duration_sec"])) < 1e-6
    outro = tl[3]
    assert outro["branding"] == "outro"
    assert outro["start_sec"] == tl[2]["end_sec"]
    assert outro["duration_sec"] >= 12.1
    assert out["total_duration_sec"] == outro["end_sec"]


def test_branding_untouched_when_durations_zero():
    plan = _mini_plan()
    out = rp.insert_branding_items(plan, intro_dur=0.0, outro_dur=0.0)
    assert [t["segment_id"] for t in out["timeline"]] == ["g0001_p00", "g0002_p01"]
    assert out["total_duration_sec"] == 18.0


# ---- bubble-dominated panels must not be scenes (user: spiky shout bubble
# rendered full-screen as its own cut) ----------------------------------------

def test_bubble_coverage_fraction():
    boxes = [(0, 0, 50, 100)]                  # half of a 100x100 panel
    cov = rp.bubble_coverage((100, 100), boxes)
    assert 0.45 <= cov <= 0.55


def test_bubble_dominated_cut_dropped_time_redistributed():
    cuts = [{"file": "art.jpg", "start": 0.0, "dur": 5.0},
            {"file": "bubble.jpg", "start": 5.0, "dur": 5.0}]
    cov = {"art.jpg": 0.08, "bubble.jpg": 0.72}
    out, dropped = rp.drop_bubble_dominated_cuts(cuts, cov)
    assert dropped == ["bubble.jpg"]
    assert [c["file"] for c in out] == ["art.jpg"]
    assert abs(out[0]["dur"] - 10.0) < 1e-6


def test_bubble_filter_never_empties_a_shot():
    cuts = [{"file": "a.jpg", "start": 0.0, "dur": 4.0},
            {"file": "b.jpg", "start": 4.0, "dur": 4.0}]
    cov = {"a.jpg": 0.9, "b.jpg": 0.6}
    out, dropped = rp.drop_bubble_dominated_cuts(cuts, cov)
    assert [c["file"] for c in out] == ["b.jpg"]   # least bubbly survives
    assert dropped == ["a.jpg"]


# ---- system-message protection (the Sky Corporation cliffhanger was dropped:
# ogkalu false-positives system windows as bubbles; our model's system_box
# class works on crops and must veto both the gate and the text blanking) -----

def test_system_panels_exempt_from_dominance_gate():
    cuts = [{"file": "art.jpg", "start": 0.0, "dur": 5.0},
            {"file": "sys.jpg", "start": 5.0, "dur": 5.0}]
    cov = {"art.jpg": 0.1, "sys.jpg": 0.9}
    out, dropped = rp.drop_bubble_dominated_cuts(cuts, cov, exempt={"sys.jpg"})
    assert dropped == []
    assert len(out) == 2


def test_bubble_boxes_overlapping_system_boxes_filtered():
    bubbles = [(0, 0, 100, 100), (200, 200, 300, 300)]
    system = [(10, 10, 90, 90)]
    out = rp.filter_protected_boxes(bubbles, system)
    assert out == [(200, 200, 300, 300)]


# ---- over-merged scene splitter (two stacked panels + a big white void in
# ONE crop -> split parts, render side by side) --------------------------------

def test_split_on_white_bands_two_panels():
    img = np.full((300, 100, 3), 250, dtype=np.uint8)
    img[10:100] = 60                                  # top panel
    img[200:290] = 80                                 # bottom panel
    parts = rp.split_on_white_bands(img, min_band_h=40)
    assert len(parts) == 2
    (y1a, y2a), (y1b, y2b) = parts
    assert y1a <= 10 and 100 <= y2a <= 140
    assert 160 <= y1b <= 200 and y2b >= 290


def test_split_on_white_bands_no_band_single_part():
    img = np.full((300, 100, 3), 70, dtype=np.uint8)  # solid art
    assert len(rp.split_on_white_bands(img, min_band_h=40)) == 1


# ---- split-part content filter: spiky/SFX bubble parts are near-binary
# (black ring + white core, no midtones) and must not survive as "content" ----

def test_filter_content_parts_discards_binary_bubble_part():
    img = np.full((400, 200, 3), 250, dtype=np.uint8)
    # part 1: real art (midtones)
    img[20:140] = 120
    # part 2: spiky scream bubble look — black ring, white core, no midtones
    img[260:380] = 255
    img[270:370, 30:170] = 5
    img[300:340, 70:130] = 255
    parts = [(10, 150), (250, 390)]
    out = rp.filter_content_parts(img, parts, boxes=[])
    assert out == [(10, 150)]


def test_filter_content_parts_keeps_two_art_parts():
    img = np.full((400, 200, 3), 250, dtype=np.uint8)
    img[20:140] = 120
    img[260:380] = 90
    parts = [(10, 150), (250, 390)]
    out = rp.filter_content_parts(img, parts, boxes=[])
    assert out == [(10, 150), (250, 390)]
