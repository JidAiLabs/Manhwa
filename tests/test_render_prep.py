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


# ---- 2b. near-identical same-size dedup (the "? face" pair) ------------------

def test_near_identical_same_size_pair_deduped():
    # two SEPARATE panels, same framing, tiny per-pixel differences — the
    # "reaction face with ?" case (p000013 / p000016). area_ratio ~1.0 so the
    # containment filter never fires; this filter must catch it.
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, (400, 300, 3), dtype=np.uint8)
    noisy = base.astype(np.int16) + rng.integers(-3, 4, base.shape)
    noisy = noisy.clip(0, 255).astype(np.uint8)
    cuts = [{"file": "p13.jpg", "start": 0.0, "dur": 4.0},
            {"file": "p16.jpg", "start": 4.0, "dur": 4.0}]
    out, dropped = rp.drop_near_identical_cuts(
        cuts, {"p13.jpg": base, "p16.jpg": noisy})
    assert dropped == ["p16.jpg"]               # LATER cut dropped, earlier kept
    assert [c["file"] for c in out] == ["p13.jpg"]
    assert abs(out[0]["dur"] - 8.0) < 1e-6       # freed time redistributed


def test_near_identical_distinct_images_both_kept():
    # mostly-black vs mostly-white — clearly different, must both survive.
    black = np.full((400, 300, 3), 8, np.uint8)
    white = np.full((400, 300, 3), 247, np.uint8)
    cuts = [{"file": "blk.jpg", "start": 0.0, "dur": 4.0},
            {"file": "wht.jpg", "start": 4.0, "dur": 4.0}]
    out, dropped = rp.drop_near_identical_cuts(
        cuts, {"blk.jpg": black, "wht.jpg": white})
    assert dropped == []
    assert len(out) == 2


def test_near_identical_distinct_random_images_both_kept():
    # two independent random panels (different characters/scenes) — kept.
    rng = np.random.default_rng(11)
    a = rng.integers(0, 256, (400, 300, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (380, 300, 3), dtype=np.uint8)
    cuts = [{"file": "a.jpg", "start": 0.0, "dur": 5.0},
            {"file": "b.jpg", "start": 5.0, "dur": 3.0}]
    out, dropped = rp.drop_near_identical_cuts(cuts, {"a.jpg": a, "b.jpg": b})
    assert dropped == []
    assert len(out) == 2


def test_near_identical_redistributes_total_duration_preserved():
    base = _pattern(400, 300)
    near = base.astype(np.int16) + 2             # uniform tiny shift
    near = near.clip(0, 255).astype(np.uint8)
    cuts = [{"file": "x.jpg", "start": 0.0, "dur": 3.0},
            {"file": "y.jpg", "start": 3.0, "dur": 5.0}]
    total_before = sum(c["dur"] for c in cuts)
    out, dropped = rp.drop_near_identical_cuts(
        cuts, {"x.jpg": base, "y.jpg": near})
    assert dropped == ["y.jpg"]
    assert abs(sum(c["dur"] for c in out) - total_before) < 1e-6
    assert out[0]["file"] == "x.jpg"


def test_near_identical_different_size_seam_pair_not_handled_here():
    # a small contained-in-big seam pair has area_ratio well below 0.7, so
    # this filter leaves it for drop_visual_duplicate_cuts (no double-handling).
    big = _pattern(400, 300)
    small = big[300:400, 0:300].copy()           # area ratio 0.25
    cuts = [{"file": "big.jpg", "start": 0.0, "dur": 4.0},
            {"file": "small.jpg", "start": 4.0, "dur": 4.0}]
    out, dropped = rp.drop_near_identical_cuts(
        cuts, {"big.jpg": big, "small.jpg": small})
    assert dropped == []
    assert len(out) == 2


def test_near_identical_single_cut_noop():
    out, dropped = rp.drop_near_identical_cuts(
        [{"file": "a.jpg", "start": 0.0, "dur": 4.0}], {"a.jpg": _pattern(400, 300)})
    assert dropped == []
    assert len(out) == 1


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


def test_clean_scene_image_flattens_faint_blank_bubble_residue():
    img = _bubble_scene()
    img[92:108, 72:128] = 175                       # faint ghost, not ink
    out = rp.clean_scene_image(img, [(30, 50, 170, 150)])
    assert out[92:108, 72:128].mean() > 235
    assert out[100, 41].mean() < 60                 # outline still preserved


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


def test_branding_drops_intro_appends_outro_only():
    # channel decision (2026-06-15): NO intro on any video — even with intro_dur
    # passed, only the outro is appended; the video opens straight on the story.
    plan = _mini_plan()
    out = rp.insert_branding_items(plan, intro_dur=6.0, outro_dur=12.1)
    tl = out["timeline"]
    assert [t["segment_id"] for t in tl] == [
        "g0001_p00", "g0002_p01", "branding_outro"]
    assert all(t.get("branding") != "intro" for t in tl)   # no intro anywhere
    outro = tl[-1]
    assert outro["branding"] == "outro" and outro["duration_sec"] >= 12.1
    assert out["total_duration_sec"] == outro["end_sec"]


def test_branding_which_intro_only_and_outro_only():
    # bundle segments: FIRST chapter renders intro only, LAST outro only,
    # middles none — so a concatenated season has exactly one intro/outro
    plan = {"timeline": [
        {"segment_id": "g0001_p00", "start_sec": 0.0, "end_sec": 10.0,
         "duration_sec": 10.0, "cuts": [{"file": "a.jpg", "start": 0, "dur": 10}]}],
        "total_duration_sec": 10.0}
    # "intro" (a bundle's first segment) now carries NO branding at all
    intro = rp.insert_branding_items(plan, intro_dur=6.0, outro_dur=12.0,
                                     which="intro")
    assert [i.get("branding") for i in intro["timeline"]] == [None]
    outro = rp.insert_branding_items(plan, intro_dur=6.0, outro_dur=12.0,
                                     which="outro")
    segs = [i.get("branding") for i in outro["timeline"]]
    assert "outro" in segs and "intro" not in segs
    none = rp.insert_branding_items(plan, intro_dur=6.0, outro_dur=12.0,
                                    which="none")
    assert [i.get("branding") for i in none["timeline"]] == [None]


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


# ---- system-message protection (a flat in-world system card was dropped:
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


# ---- OCR-word-box cleaning (user's original method): remove the exact text
# rects (invisible small-region inpaint), gated to bubble interiors -----------

def test_clean_with_word_boxes_removes_text_cleanly():
    img = _bubble_scene()
    out = rp.clean_scene_image(img, [(30, 50, 170, 150)],
                               text_boxes=[(68, 93, 132, 107)])
    assert out[95:105, 72:128].mean() > 230        # text gone
    assert out[80, 100].mean() > 230               # interior intact
    assert out[100, 41].mean() < 60                # ring intact
    assert abs(int(out[10:30, 10:30].mean()) - 90) <= 2   # art untouched


def test_clean_word_boxes_outside_bubbles_ignored():
    img = _bubble_scene()
    img[20:30, 20:60] = 250                        # light text ON ART (embedded)
    out = rp.clean_scene_image(img, [(30, 50, 170, 150)],
                               text_boxes=[(20, 20, 60, 30), (68, 93, 132, 107)])
    assert out[20:30, 20:60].mean() > 230          # embedded art text SURVIVES
    assert out[95:105, 72:128].mean() > 230        # bubble text removed


# ---- speech-mode arbiter: false system_box detections on DIALOGUE panels
# must not shield speech bubbles from cleaning; real system windows live on
# panels Gemini classified as non-speech (bubble_mode none/narration) --------

def test_speech_mode_files_from_beats():
    beats = {"beats": [
        {"scene_selection": [
            {"scene_file": "talk.jpg", "bubble_mode": "spoken"},
            {"scene_file": "think.jpg", "bubble_mode": "inner_thought"},
            {"scene_file": "sys.jpg", "bubble_mode": "none"},
            {"scene_file": "cap.jpg", "bubble_mode": "narration"},
        ]},
    ]}
    out = rp.speech_mode_files(beats)
    assert out == {"talk.jpg", "think.jpg"}


# ---- post-clean blankness: a panel that is only empty bubbles + gradient
# after cleaning (user's IE husk) must not be shown -----------------------------

def _husk(w=300, h=400):
    """Gradient background + two empty bubble outlines + caption rect."""
    img = np.zeros((h, w, 3), np.uint8)
    ramp = np.linspace(190, 230, h).astype(np.uint8)
    img[:] = ramp[:, None, None]
    cv2.circle(img, (100, 200), 70, (20, 20, 20), 3)
    cv2.circle(img, (220, 330), 60, (20, 20, 20), 3)
    cv2.rectangle(img, (60, 40), (240, 110), (20, 20, 20), 3)
    cv2.circle(img, (100, 200), 67, (250, 250, 250), -1)
    cv2.circle(img, (220, 330), 57, (250, 250, 250), -1)
    return img


def test_art_score_low_for_cleaned_husk():
    img = _husk()
    boxes = [(20, 120, 180, 280), (150, 260, 290, 400), (50, 30, 250, 120)]
    assert rp.art_content_score(img, boxes) < 0.012


def test_art_score_high_for_real_art():
    # hard-edged line art: posterized pattern has ink-like boundaries
    img = (_pattern(400, 300) > 128).astype(np.uint8) * 255
    assert rp.art_content_score(img, []) > 0.03


import cv2  # noqa: E402  (used by _husk)


# ---- unify drop-vs-recrop: gradient husk parts (midtones but no edges) are
# not content; a panel is only DROPPED when no part survives ------------------

def test_filter_content_parts_rejects_gradient_husk_part():
    img = np.zeros((400, 300, 3), np.uint8)
    ramp = np.linspace(170, 220, 400).astype(np.uint8)   # gradient = midtones
    img[:] = ramp[:, None, None]
    # top part: real line art (hard edges AND midtones, like actual manhwa)
    pat = _pattern(150, 300)[..., 0]
    art = np.select([pat > 170, pat > 85], [255, 128], default=30).astype(np.uint8)
    img[10:160] = np.dstack([art, art, art])[:150]
    # bottom part: gradient only (the husk look) — stays as initialized
    parts = [(0, 170), (220, 400)]
    out = rp.filter_content_parts(img, parts, boxes=[])
    assert out == [(0, 170)]                     # husk part rejected


def test_panel_recoverable_when_any_part_has_art():
    img = np.zeros((400, 300, 3), np.uint8)
    ramp = np.linspace(170, 220, 400).astype(np.uint8)
    img[:] = ramp[:, None, None]
    art = (_pattern(150, 300) > 128).astype(np.uint8) * 255
    img[10:160] = art[:150]
    assert rp.panel_recoverable(img, boxes=[]) is True     # recrop, don't drop
    img2 = np.zeros((400, 300, 3), np.uint8)
    img2[:] = ramp[:, None, None]                          # all gradient
    assert rp.panel_recoverable(img2, boxes=[]) is False   # nothing to save


# ---- document-like panels (the ORV app-list) must be judged WHOLE: the
# splitter would shred their rows into sub-min_h fragments and discard all ----

def _app_list_panel():
    img = np.full((600, 400, 3), 252, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (390, 70), (90, 90, 90), -1)     # title bar
    for i in range(6):                                            # episode rows
        y = 120 + i * 80
        cv2.line(img, (20, y), (380, y), (60, 60, 60), 2)
        cv2.putText(img, f"READ EPISODE {1380+i} VIEWS: 1", (24, y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2)
    return img


def test_document_panel_recoverable_and_not_shredded():
    img = _app_list_panel()
    assert rp.panel_recoverable(img, boxes=[], text_rich=True) is True
    # text_rich forces whole-panel judgment — exactly one span, the panel
    spans = rp.split_spans_for_panel(img, text_rich=True)
    assert spans == [(0, img.shape[0])]


def test_empty_bubble_panel_metadata_is_hard_junk():
    assert rp.empty_bubble_panel({
        "panel_kind": "empty",
        "subjects": ["speech bubble"],
        "ocr_clean": "DAMN IT,",
        "text_coverage": 0.0299,
    })
    assert rp.empty_bubble_panel({
        "panel_kind": "empty",
        "ocr_clean": "Hah, geez. YOU'RE DOING ALL SORTS OF THINGS.",
        "text_coverage": 0.0735,
    })
    assert rp.empty_bubble_panel({
        "panel_kind": "story",
        "subjects": ["speech bubble"],
        "ocr_clean": "AS I THOUGHT, THIS GUY IS A GENIUS!",
        "text_coverage": 0.1552,
    })
    assert not rp.empty_bubble_panel({
        "panel_kind": "story",
        "subjects": ["speech bubble", "three men"],
        "ocr_clean": "SHIT... WE THOUGHT EVERYTHING WAS FINE...",
        "text_coverage": 0.0537,
    })
    assert rp.empty_bubble_panel({
        "panel_kind": "story",
        "subjects": ["speech bubble", "character's hair"],
        "ocr_clean": "HE'LL HAVE NO PROBLEM WITH OPERATING FORMATION.",
        "text_coverage": 0.0984,
    })


def test_story_visual_panel_keeps_text_plus_characters_not_hair_only():
    assert rp.story_visual_panel({
        "panel_kind": "story",
        "subjects": ["dark-haired character", "character with ponytail"],
        "ocr_clean": "CAN DOCTOR BAEK USE MARTIAL ARTS TOO?",
        "text_coverage": 0.0702,
    })
    assert not rp.story_visual_panel({
        "panel_kind": "story",
        "subjects": ["speech bubble", "character's hair"],
        "ocr_clean": "HE'LL HAVE NO PROBLEM WITH OPERATING FORMATION.",
        "text_coverage": 0.0984,
    })


def test_non_text_rich_panel_still_splits():
    img = np.full((300, 100, 3), 250, dtype=np.uint8)
    img[10:100] = 60
    img[200:290] = 80
    assert len(rp.split_spans_for_panel(img, text_rich=False)) == 2


# ---- dead-box recrop (user's #22): large blanked caption boxes must not
# dominate the frame — crop to the art region outside them, or flag husk -----

def _tinted_art(h, w):
    """Webtoon-like art block: midtones AND chroma (the recoverability
    fallback rejects chroma-zero panels as spike/blob garbage)."""
    art = (_pattern(h, w)[..., 0])
    art3 = np.select([art > 170, art > 85], [255, 128], default=40).astype(np.uint8)
    g = np.minimum(255, art3.astype(int) + 50).astype(np.uint8)
    return np.dstack([art3, g, art3])


def _feet_and_captions():
    """Art strip on top, two big blanked caption boxes below (ghost remnants)."""
    img = np.full((500, 800, 3), 250, dtype=np.uint8)
    img[0:150] = _tinted_art(150, 800)
    cv2.rectangle(img, (40, 180), (380, 480), (15, 15, 15), 4)      # box borders
    cv2.rectangle(img, (420, 160), (780, 470), (15, 15, 15), 4)
    img[300, 100:300] = 235                                          # ghost line
    boxes = [(40, 180, 380, 480), (420, 160, 780, 470)]
    return img, boxes


def test_dead_box_recrop_crops_to_art():
    img, boxes = _feet_and_captions()
    out, info = rp.dead_box_recrop(img, boxes)
    assert info["blank_box_frac"] > 0.35       # boxes dominate the panel
    assert info["recropped"] is True
    assert out.shape[0] <= 200                  # cropped to the art strip
    assert rp.art_content_score(out, []) > 0.012


def test_dead_box_recrop_noop_on_normal_panel():
    img = (_pattern(400, 300) > 128).astype(np.uint8) * 255
    out, info = rp.dead_box_recrop(img, [(10, 10, 60, 60)])   # small box
    assert info["recropped"] is False
    assert out.shape == img.shape


# ---- dead-box WIRING (#22): the gate must rescue blank-box-dominated panels
# whose art band survives, and the writer must emit the recropped strip -------

def test_dead_box_recrop_reports_band():
    img, boxes = _feet_and_captions()
    out, info = rp.dead_box_recrop(img, boxes)
    a, b = info["band"]
    assert b - a == out.shape[0] >= 120            # band == emitted crop rows


def test_panel_recoverable_rescues_deadbox_dominated_strip():
    # boxes cover ~0.53 of the panel: whole-span bubble coverage fails the
    # part filter, yet the feet strip is real art — must NOT be dropped
    img, boxes = _feet_and_captions()
    assert rp.panel_recoverable(img, boxes) is True


def test_select_panel_crops_deadbox_panel_yields_art_strip():
    img, boxes = _feet_and_captions()
    parts, info = rp.select_panel_crops(img, boxes, text_rich=False)
    assert info["recropped"] is True
    assert len(parts) == 1 and parts[0].shape[0] <= 200


def test_select_panel_crops_still_splits_merged_panels():
    img = np.full((400, 200, 3), 250, dtype=np.uint8)
    img[20:140] = 120                              # two real-art panels
    img[260:380] = 90                              # split by a white void
    parts, info = rp.select_panel_crops(img, [], text_rich=False)
    assert len(parts) == 2


def test_select_panel_crops_document_panel_never_recropped():
    img, boxes = _feet_and_captions()
    parts, info = rp.select_panel_crops(img, boxes, text_rich=True)
    assert len(parts) == 1 and parts[0].shape[0] == img.shape[0]
    assert info["recropped"] is False


def test_dead_box_recrop_rejects_binary_band():
    # Nano p000020: an EMPTY spiky scream bubble (radiating black/white lines)
    # has plenty of Canny edges but zero midtones — it is NOT an art band and
    # must not be rescued; the panel then fails recoverability and drops.
    img = np.full((500, 800, 3), 250, dtype=np.uint8)
    spiky = np.zeros((160, 800), np.uint8)
    spiky[:, ::3] = 255                                  # binary line burst
    img[20:180] = np.dstack([spiky, spiky, spiky])
    boxes = [(40, 200, 380, 480), (420, 200, 780, 480)]  # blank caption voids
    out, info = rp.dead_box_recrop(img, boxes)
    assert info["blank_box_frac"] >= 0.35
    assert info["recropped"] is False
    assert rp.panel_recoverable(img, boxes) is False


# ---- residue sweep: word-box fill must leave NO remnants in the interior ----

def test_clean_scene_residue_swept_after_word_fill():
    # OCR word boxes covered only part of the text: the missed strokes and
    # faint ghosts inside the bubble must be flattened to the fill color too
    img = _bubble_scene()
    img[120:128, 80:120] = 200                     # faint ghost line (missed)
    out = rp.clean_scene_image(img, [(30, 50, 170, 150)],
                               text_boxes=[(68, 93, 132, 107)])
    assert out[121:127, 85:115].mean() > 230       # ghost flattened
    assert out[100, 41].mean() < 60                # outline ring still dark
    assert abs(int(out[10:30, 10:30].mean()) - 90) <= 2   # art untouched


# ---- orphan word boxes: bubbles the detector missed entirely ----------------

def test_orphan_word_boxes_on_white_surround_blanked():
    # p000069/p000101: spiky/oval bubble evaded the detector, so its words got
    # no interior gating — words sitting on a uniform near-white surround are
    # bubble text and must be blanked anyway
    img = np.full((300, 300, 3), 90, dtype=np.uint8)
    img[40:160, 60:240] = 252                      # white bubble, NO box
    for y in range(70, 130, 16):
        for x in range(80, 210, 24):
            img[y:y + 8, x:x + 14] = 25            # crisp glyph blobs
    out = rp.clean_scene_image(img, [],
                               text_boxes=[(80, 70, 224, 126)])
    assert out[70:126, 90:210].mean() > 230        # text gone
    assert abs(int(out[20, 20].mean()) - 90) <= 2  # art untouched


def test_orphan_word_boxes_on_art_kept():
    img = np.full((300, 300, 3), 90, dtype=np.uint8)
    img[100:120, 80:220] = 250                     # light text embedded ON art
    out = rp.clean_scene_image(img, [],
                               text_boxes=[(80, 100, 220, 120)])
    assert out[100:120, 80:220].mean() > 230       # embedded art text SURVIVES


# ---- garbage sole-cut substitution: never ship chrome/husk as a segment's
# only visual — show the nearest kept story panel instead ---------------------

def test_substitute_garbage_sole_cut_with_neighbor():
    cbs = {"g0001_p00": [{"file": "p000000.jpg", "start": 0.0, "dur": 8.0}],
           "g0002_p01": [{"file": "p000003.jpg", "start": 0.0, "dur": 6.0}]}
    cov = {"p000000.jpg": 1.0, "p000003.jpg": 0.1}
    out, subs = rp.substitute_garbage_sole_cuts(
        cbs, cov, durations={"g0001_p00": 8.0, "g0002_p01": 6.0},
        order=["g0001_p00", "g0002_p01"])
    # garbage at the chapter head holds the NEXT good panel (story-adjacent),
    # marked held so QA exempts it
    assert out["g0001_p00"] == [{"file": "p000003.jpg", "start": 0.0,
                                 "dur": 8.0, "held": True}]
    assert out["g0002_p01"] == cbs["g0002_p01"]
    assert [(s[0], s[1], s[2]) for s in subs] == [
        ("g0001_p00", "p000000.jpg", "p000003.jpg")]


def test_caption_run_alternates_holds_to_avoid_freeze():
    # A run of narration-only caption boxes between scene A and scene D must
    # not all freeze on A (IE ch1: 3 captions held p93 -> 4-in-a-row, 33s).
    # Alternate the held image between the scene BEFORE and the scene AFTER so
    # no on-screen image holds more than twice consecutively. Manhwa-agnostic.
    from itertools import groupby
    cbs = {
        "s0": [{"file": "A.jpg", "start": 0.0, "dur": 5.0}],     # real scene A
        "c1": [{"file": "cap1.jpg", "start": 0.0, "dur": 5.0}],  # caption (garbage)
        "c2": [{"file": "cap2.jpg", "start": 0.0, "dur": 5.0}],
        "c3": [{"file": "cap3.jpg", "start": 0.0, "dur": 5.0}],
        "s4": [{"file": "D.jpg", "start": 0.0, "dur": 5.0}],     # real scene D
    }
    cov = {"A.jpg": 0.1, "cap1.jpg": 1.0, "cap2.jpg": 1.0,
           "cap3.jpg": 1.0, "D.jpg": 0.1}
    out, subs = rp.substitute_garbage_sole_cuts(
        cbs, cov, durations={k: 5.0 for k in cbs},
        order=["s0", "c1", "c2", "c3", "s4"])
    shown = [out[s][0]["file"] for s in ("s0", "c1", "c2", "c3", "s4")]
    runs = [len(list(grp)) for _, grp in groupby(shown)]
    assert max(runs) <= 2, shown                       # no freeze
    for c in ("c1", "c2", "c3"):                        # captions hold real art
        assert out[c][0]["file"] in ("A.jpg", "D.jpg")
        assert out[c][0]["held"] is True


def test_caption_run_at_chapter_end_cycles_recent_scenes():
    # Cliffhanger: the chapter ENDS on a run of narration-only caption boxes
    # with NO scene after (IE ch1 p94/p95/p96). With no forward scene to bridge
    # to, cycle the recent real scenes instead of freezing on the last one.
    from itertools import groupby
    cbs = {
        "s0": [{"file": "A.jpg", "start": 0.0, "dur": 5.0}],     # real
        "s1": [{"file": "B.jpg", "start": 0.0, "dur": 5.0}],     # real
        "c1": [{"file": "cap1.jpg", "start": 0.0, "dur": 5.0}],  # caption (garbage)
        "c2": [{"file": "cap2.jpg", "start": 0.0, "dur": 5.0}],
        "c3": [{"file": "cap3.jpg", "start": 0.0, "dur": 5.0}],
    }
    cov = {"A.jpg": 0.1, "B.jpg": 0.1, "cap1.jpg": 1.0,
           "cap2.jpg": 1.0, "cap3.jpg": 1.0}
    out, subs = rp.substitute_garbage_sole_cuts(
        cbs, cov, durations={k: 5.0 for k in cbs}, order=list(cbs))
    shown = [out[s][0]["file"] for s in cbs]
    assert max(len(list(g)) for _, g in groupby(shown)) <= 2, shown
    for c in ("c1", "c2", "c3"):
        assert out[c][0]["file"] in ("A.jpg", "B.jpg")     # recent real scenes
        assert out[c][0]["held"] is True


def test_speech_shaped_boxes_excludes_ui_rows():
    # the ORV app screen: the detector boxes its full-width list ROWS as
    # "bubbles" — only oval-ish, sub-full-width boxes are speech bubbles
    boxes = [(54, 165, 752, 357),     # full-width flat app row
             (62, 15, 755, 158),      # another row
             (83, 784, 525, 1071)]    # the actual thought bubble
    assert rp.speech_shaped_boxes(boxes, 800) == [(83, 784, 525, 1071)]


def test_panel_recoverable_whole_panel_fallback_for_bright_art():
    # every split part can fail individually (bubble-dominated span, bright
    # glow span) while the WHOLE cleaned panel is real art — the writer keeps
    # the whole image in that case, so the gate must judge that same image
    # (IE p000039: dad + lightbulb glow + two big bubbles)
    img = _tinted_art(500, 400)
    box = (0, 0, 400, 400)                     # bubble "covers" 80% of height
    assert rp.panel_recoverable(img, [box]) is True


def test_panel_recoverable_whole_fallback_accepts_bright_glow_art():
    # the real IE p000039 profile: bright glow, ~11% midtones, but COLORFUL
    # (chroma_p90≈63) — recoverable art
    img = np.full((400, 400, 3), 250, dtype=np.uint8)
    img[330:400] = _tinted_art(70, 400)
    assert rp.panel_recoverable(img, []) is True
    # while a NEAR-BINARY panel (midtone < 0.08) stays unrecoverable
    binimg = np.full((400, 400, 3), 250, dtype=np.uint8)
    binimg[330:400, ::3] = 0                    # spiky binary burst rows
    assert rp.panel_recoverable(binimg, []) is False


def test_panel_recoverable_rejects_monochrome_aa_spiky():
    # the real Nano p000020: full-bleed anti-aliased spike burst (midtones
    # from AA but chroma EXACTLY 0) + a big blanked caption box — neither
    # the whole-panel fallback nor the dead-box band may rescue it
    img = np.full((400, 400, 3), 250, dtype=np.uint8)
    img[:, ::3] = 0                                # spikes everywhere
    img[:, 1::3] = 128                             # anti-aliased gray between
    boxes = [(0, 0, 400, 210)]                     # blanked caption, 52% cover
    assert rp.panel_recoverable(img, boxes) is False


def test_panel_recoverable_rim_edges_outside_boxes_dont_count():
    # the real IE p000008: edge-dead gradient curtain + empty bubbles whose
    # OUTLINE RIMS sit just outside the detector boxes — padding the
    # exclusion kills the fake art score; not recoverable
    img = np.zeros((600, 400, 3), np.uint8)
    for y in range(600):
        img[y, :] = 120 + int(60 * y / 600)        # smooth gradient (no edges)
    cv2.ellipse(img, (200, 300), (120, 90), 0, 0, 360, (10, 10, 10), 3)
    cv2.ellipse(img, (200, 300), (114, 84), 0, 0, 360, (250, 250, 250), -1)
    boxes = [(80, 210, 320, 390)]                  # detector box INSIDE rim
    assert rp.panel_recoverable(img, boxes) is False


# ---- cross-segment duplicates: consecutive shown cuts must differ -----------

def _stamp(img, seed):
    rng = np.random.default_rng(seed)
    for _ in range(40):
        x, y = rng.integers(10, img.shape[1] - 30), rng.integers(10, img.shape[0] - 30)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 18, int(y) + 12),
                      (int(rng.integers(0, 255)),) * 3, -1)
    return img


def test_multi_scale_contained_catches_zoom_pair():
    big = _stamp(np.full((600, 400, 3), 200, np.uint8), 7)
    zoom = cv2.resize(big[380:560, 100:340], (400, 300))   # blow-up of a region
    other = _stamp(np.full((600, 400, 3), 200, np.uint8), 99)
    assert rp.multi_scale_contained(zoom, big) is True
    assert rp.multi_scale_contained(other, big) is False


def test_cross_segment_duplicate_dropped_from_multicut_segment():
    big = _stamp(np.full((600, 400, 3), 200, np.uint8), 7)
    zoom = cv2.resize(big[380:560, 100:340], (400, 300))
    other = _stamp(np.full((600, 400, 3), 200, np.uint8), 99)
    cbs = {"g1": [{"file": "a.jpg", "start": 0.0, "dur": 4.0}],
           "g2": [{"file": "b.jpg", "start": 0.0, "dur": 3.0},
                  {"file": "c.jpg", "start": 3.0, "dur": 3.0}]}
    imgs = {"a.jpg": big, "b.jpg": zoom, "c.jpg": other}
    order = ["g1", "g2"]
    out, dropped = rp.drop_cross_segment_duplicate_cuts(
        cbs, order, lambda f: imgs.get(f))
    assert dropped == [("g2", "b.jpg")]
    assert [c["file"] for c in out["g2"]] == ["c.jpg"]
    assert abs(sum(c["dur"] for c in out["g2"]) - 6.0) < 0.01  # time kept


def test_cross_segment_duplicate_sole_cut_marked_not_emptied():
    big = _stamp(np.full((600, 400, 3), 200, np.uint8), 7)
    near = big.copy()
    cbs = {"g1": [{"file": "a.jpg", "start": 0.0, "dur": 4.0}],
           "g2": [{"file": "b.jpg", "start": 0.0, "dur": 5.0}]}
    imgs = {"a.jpg": big, "b.jpg": near}
    out, dropped = rp.drop_cross_segment_duplicate_cuts(
        cbs, ["g1", "g2"], lambda f: imgs.get(f))
    # sole-cut segments are never emptied here — flagged for substitution
    assert out["g2"] == cbs["g2"]
    assert dropped == [("g2", "b.jpg")]


def test_caption_box_is_not_a_dedup_reference():
    # A real-art panel must NOT be dropped just because it embeds a generic
    # caption/blank box that the (garbage) PREVIOUS panel also was. After
    # bubble-inpainting every caption box collapses to a near-blank rectangle,
    # so a caption panel "contains" any other panel's caption region — that is
    # shared BLANK SPACE, not shared art, and must never drive visual dedup.
    # Manhwa-agnostic: keys only on coverage (geometry), not on any art style.
    art_a = _stamp(np.full((600, 400, 3), 200, np.uint8), 7)     # unique art
    caption = np.full((150, 360, 3), 255, np.uint8)              # blank box
    cv2.putText(caption, "NARRATION", (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
    art_b = _stamp(np.full((600, 400, 3), 200, np.uint8), 13)    # different art
    art_b[40:190, 20:380] = caption                             # …that embeds a caption
    cbs = {"g1": [{"file": "A.jpg", "start": 0.0, "dur": 4.0}],
           "g2": [{"file": "C.jpg", "start": 0.0, "dur": 4.0}],  # caption = garbage
           "g3": [{"file": "B.jpg", "start": 0.0, "dur": 4.0}]}  # real art w/ caption
    imgs = {"A.jpg": art_a, "C.jpg": caption, "B.jpg": art_b}
    cov = {"A.jpg": 0.1, "C.jpg": 1.0, "B.jpg": 0.2}             # only C is garbage
    out, dropped = rp.drop_cross_segment_duplicate_cuts(
        cbs, ["g1", "g2", "g3"], lambda f: imgs.get(f),
        coverage_by_file=cov, exempt=set())
    # the caption box must not be used as a reference, so B survives
    assert ("g3", "B.jpg") not in dropped
    assert out["g3"] == cbs["g3"]


def test_doc_like_separates_documents_from_dialogue():
    # app/stats screen: many words, few inside any bubble -> DOCUMENT
    words = [(10 * i, 10, 10 * i + 8, 18) for i in range(30)]
    assert rp.doc_like(0.3, 30, words, [(0, 100, 50, 150)]) is True
    # wordy DIALOGUE panel: words clustered inside bubbles -> NOT a document
    # (the IE p000059 case: 15+ OCR words misread as a document, so its
    # dialogue stayed on screen while narration spoke the same lines)
    inwords = [(10 * i, 8, 10 * i + 8, 30) for i in range(18)]
    assert rp.doc_like(0.25, 18, inwords, [(0, 0, 320, 40)]) is False
    # wordy with no detected bubbles -> document (stats page)
    assert rp.doc_like(0.25, 18, inwords, []) is True
    # sparse text -> never a document
    assert rp.doc_like(0.05, 5, inwords[:5], []) is False


def test_doc_like_mixed_panel_with_substantial_ui_text_is_document():
    # ORV p000025: a speech bubble (majority of words) ABOVE app-list rows —
    # the outside-bubble words are substantial, so it is still a document
    # (losing doc status shredded the app UI in the splitter)
    bubble = (0, 0, 320, 60)
    inwords = [(10 * i, 10, 10 * i + 8, 30) for i in range(20)]      # in bubble
    outwords = [(10 * i, 100, 10 * i + 8, 112) for i in range(10)]   # app rows
    assert rp.doc_like(0.25, 30, inwords + outwords, [bubble]) is True
    # but a couple of stray outside words do NOT make dialogue a document
    assert rp.doc_like(0.25, 18, inwords[:16] + outwords[:2], [bubble]) is False


def test_substitute_holds_previous_good_panel():
    # a garbage sole cut HOLDS the story-adjacent panel just before it, rather
    # than swapping in a numerically-nearest unrelated one (IE Bai Xue): the
    # narration runs over a held image, marked held so QA exempts it
    cbs = {"g1": [{"file": "p000007.jpg", "start": 0.0, "dur": 5.0}],
           "g2": [{"file": "p000008.jpg", "start": 0.0, "dur": 5.0}],
           "g3": [{"file": "p000009.jpg", "start": 0.0, "dur": 5.0}]}
    cov = {"p000007.jpg": 0.0, "p000008.jpg": 1.0, "p000009.jpg": 0.0}
    out, subs = rp.substitute_garbage_sole_cuts(
        cbs, cov, durations={k: 5.0 for k in cbs}, order=["g1", "g2", "g3"])
    assert out["g2"] == [{"file": "p000007.jpg", "start": 0.0,
                          "dur": 5.0, "held": True}]   # holds the prior good panel
    assert subs == [("g2", "p000008.jpg", "p000007.jpg")]


def test_substitute_garbage_run_holds_last_good():
    # a RUN of garbage segments holds story-adjacent scenes, staying on the
    # last good panel until it would freeze (>2 in a row), then cycling to the
    # previous recent scene rather than freezing (IE tail). All held.
    cbs = {"g0": [{"file": "p000093.jpg", "start": 0.0, "dur": 4.0}],   # good
           "g1": [{"file": "p000050.jpg", "start": 0.0, "dur": 4.0}],   # good
           "g2": [{"file": "p000094.jpg", "start": 0.0, "dur": 4.0}],   # garbage
           "g3": [{"file": "p000095.jpg", "start": 0.0, "dur": 4.0}]}   # garbage
    cov = {"p000093.jpg": 0.0, "p000050.jpg": 0.0,
           "p000094.jpg": 1.0, "p000095.jpg": 1.0}
    out, subs = rp.substitute_garbage_sole_cuts(
        cbs, cov, durations={k: 4.0 for k in cbs},
        order=["g0", "g1", "g2", "g3"])
    # g2 stays on the just-narrated scene p050 (real p050 + 1 hold = 2 frames)
    assert out["g2"][0] == {"file": "p000050.jpg", "start": 0.0,
                            "dur": 4.0, "held": True}
    # g3 would make p050 freeze 3-in-a-row -> cycles to the prior recent scene
    assert out["g3"][0] == {"file": "p000093.jpg", "start": 0.0,
                            "dur": 4.0, "held": True}
    assert [s[0] for s in subs] == ["g2", "g3"]


def test_substitute_skips_exempt_borderline_and_multicut():
    cbs = {
        "g1": [{"file": "p000113.jpg", "start": 0.0, "dur": 5.0}],   # exempt sys
        "g2": [{"file": "p000060.jpg", "start": 0.0, "dur": 5.0}],   # borderline
        "g3": [{"file": "p000070.jpg", "start": 0.0, "dur": 2.0},    # multi-cut
               {"file": "p000071.jpg", "start": 2.0, "dur": 3.0}],
        "g4": [{"file": "p000050.jpg", "start": 0.0, "dur": 5.0}],   # good
    }
    cov = {"p000113.jpg": 1.0, "p000060.jpg": 0.6, "p000070.jpg": 1.0,
           "p000071.jpg": 0.0, "p000050.jpg": 0.0}
    out, subs = rp.substitute_garbage_sole_cuts(
        cbs, cov, durations={k: 5.0 for k in cbs},
        exempt={"p000113.jpg"})
    assert out == cbs and subs == []


def test_substitute_garbage_holds_not_wrong_swaps():
    """IE ch1 tail: a run of garbage segments between two good panels holds
    only STORY-ADJACENT art — the scene just before or just after the run —
    never numerically-nearest/unrelated art, and never freezes on one image
    for more than two consecutive frames (the run alternates before/after)."""
    from itertools import groupby
    cuts = {
        "g0001_p00": [{"file": "p000001.jpg", "start": 0.0, "dur": 4.0}],   # good
        "g0002_p00": [{"file": "p000090.jpg", "start": 0.0, "dur": 4.0}],   # garbage
        "g0003_p00": [{"file": "p000091.jpg", "start": 0.0, "dur": 4.0}],   # garbage
        "g0004_p00": [{"file": "p000092.jpg", "start": 0.0, "dur": 4.0}],   # garbage
        "g0005_p00": [{"file": "p000002.jpg", "start": 0.0, "dur": 4.0}],   # good
    }
    cov = {"p000090.jpg": 1.0, "p000091.jpg": 1.0, "p000092.jpg": 1.0,
           "p000001.jpg": 0.0, "p000002.jpg": 0.0}
    order = list(cuts)
    out, subs = rp.substitute_garbage_sole_cuts(
        cuts, cov, durations={s: 4.0 for s in cuts}, order=order)
    assert len(subs) == 3
    # holds are only the two adjacent good panels (never unrelated art), held
    for seg in ("g0002_p00", "g0003_p00", "g0004_p00"):
        assert out[seg][0]["file"] in ("p000001.jpg", "p000002.jpg")
        assert out[seg][0]["held"] is True
    # and the stretch never freezes: no image runs more than twice in a row
    shown = [out[s][0]["file"] for s in order]
    assert max(len(list(g)) for _, g in groupby(shown)) <= 2, shown


def test_cap_repeats_holds_previous_panel():
    """The 'so what?' fix: a 3rd show is dropped and the previous panel
    HOLDS through the segment — narration continues, no loop, no dead-end
    human review."""
    cuts = {
        "g1": [{"file": "A.jpg", "start": 0.0, "dur": 4.0}],
        "g2": [{"file": "B.jpg", "start": 0.0, "dur": 4.0}],
        "g3": [{"file": "A.jpg", "start": 0.0, "dur": 4.0}],
        "g4": [{"file": "B.jpg", "start": 0.0, "dur": 4.0}],
        "g5": [{"file": "A.jpg", "start": 0.0, "dur": 5.0}],   # 3rd A
    }
    order = ["g1", "g2", "g3", "g4", "g5"]
    out, holds = rp.cap_repeats_with_holds(
        cuts, durations={s: 4.0 for s in order} | {"g5": 5.0}, order=order)
    # nearby-repeat rule: A,B,A,B,A collapses the mid alternation into a
    # held stretch, then ends on a FRESH panel — better than spacing loops
    assert holds == [("g3", "B.jpg"), ("g4", "B.jpg")]
    assert out["g3"][0]["held"] is True
    g5 = out["g5"][0]
    assert not g5.get("held") and g5["file"] == "A.jpg"
    counts = {}
    for s_ in order:
        for c in out[s_]:
            if not c.get("held"):
                counts[c["file"]] = counts.get(c["file"], 0) + 1
    assert max(counts.values()) <= 2


def test_cap_repeats_exempts_sys_panels():
    """Consecutive identical system cards keep the card ON SCREEN every segment
    (exemption bypasses the global cap), but in-window recurrence renders as a
    HOLD — the single allocation invariant: no panel, not even sys/doc, is
    re-emitted as a fresh cut inside the degenerate window."""
    cuts = {f"g{i}": [{"file": "sys.jpg", "start": 0.0, "dur": 3.0}]
            for i in range(1, 5)}
    order = sorted(cuts)
    out, holds = rp.cap_repeats_with_holds(
        cuts, durations={s: 3.0 for s in order}, order=order,
        exempt={"sys.jpg"})
    assert all(out[s][0]["file"] == "sys.jpg" for s in order)  # card always shown
    assert [s for s, _ in holds] == ["g2", "g3", "g4"]         # recurrence held


def test_cap_repeats_holds_exempt_panel_on_nearby_repeat():
    """The IE ABA-dup fix. A sys/doc-exempt panel must NOT visibly repeat inside
    the radius-3 window; it holds the previous panel instead. Exemption only
    relaxes the GLOBAL cap (far-apart recurrence), never the in-window hold."""
    cuts = {
        "g1": [{"file": "art1.jpg", "start": 0.0, "dur": 4.0}],
        "g2": [{"file": "doc.jpg", "start": 0.0, "dur": 4.0}],   # exempt, 1st show
        "g3": [{"file": "art2.jpg", "start": 0.0, "dur": 4.0}],
        "g4": [{"file": "doc.jpg", "start": 0.0, "dur": 4.0}],   # exempt, repeats in-window
    }
    order = ["g1", "g2", "g3", "g4"]
    out, holds = rp.cap_repeats_with_holds(
        cuts, durations={s: 4.0 for s in order}, order=order, exempt={"doc.jpg"})
    assert out["g2"][0]["file"] == "doc.jpg" and not out["g2"][0].get("held")
    assert out["g4"][0]["held"] and out["g4"][0]["file"] == "art2.jpg"
    assert ("g4", "art2.jpg") in holds


def test_cap_repeats_exempt_recurs_far_apart():
    """A true system card MAY reappear far apart (outside the radius-3 window) —
    exemption bypasses the global cap so both showings are fresh, not held."""
    order = [f"g{i}" for i in range(1, 7)]
    cuts = {s: [{"file": f"art{i}.jpg", "start": 0.0, "dur": 4.0}]
            for i, s in enumerate(order)}
    cuts["g1"] = [{"file": "sys.jpg", "start": 0.0, "dur": 4.0}]
    cuts["g6"] = [{"file": "sys.jpg", "start": 0.0, "dur": 4.0}]   # 5 later, far apart
    out, holds = rp.cap_repeats_with_holds(
        cuts, durations={s: 4.0 for s in order}, order=order, exempt={"sys.jpg"})
    assert out["g1"][0]["file"] == "sys.jpg" and not out["g1"][0].get("held")
    assert out["g6"][0]["file"] == "sys.jpg" and not out["g6"][0].get("held")


def test_judge_cut_visuals_drops_junk_keeps_good(tmp_path, monkeypatch):
    import sys, types, json as _json
    verdicts = {"bad.jpg": {"keep": False, "reason": "empty blanked bubbles"},
                "good.jpg": {"keep": True, "reason": "character face"}}
    calls = []

    def fake_chat(**kw):
        path = kw["messages"][0]["images"][0]
        name = path.rsplit("/", 1)[-1]
        calls.append(name)
        return {"message": {"content": _json.dumps(verdicts[name])}}
    fake = types.ModuleType("ollama_compat")
    fake.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama_compat", fake)
    for n in ("bad.jpg", "good.jpg", "sysy.jpg"):
        (tmp_path / n).write_bytes(b"jpg")
    junk = rp.judge_cut_visuals(["bad.jpg", "good.jpg", "sysy.jpg"],
                                str(tmp_path), exempt={"sysy.jpg"})
    assert set(junk) == {"bad.jpg"} and "bubbles" in junk["bad.jpg"]
    assert "sysy.jpg" not in calls          # exempt never even judged


def test_static_on_consecutive_repeats():
    # a panel repeated in consecutive cuts must NOT re-play its pan -> static.
    plan = {"timeline": [
        {"segment_id": "g1", "cuts": [{"file": "a.jpg", "motion": {"mode": "pan"}}]},
        {"segment_id": "g2", "cuts": [{"file": "a.jpg", "motion": {"mode": "pan"}}]},
        {"segment_id": "g3", "cuts": [{"file": "b.jpg", "motion": {"mode": "pan"}}]},
        {"segment_id": "g4", "cuts": [{"file": "b.jpg", "motion": {"mode": "kenburns"}},
                                      {"file": "b.jpg", "motion": {"mode": "pan"}}]},
    ]}
    tl = rp.static_on_consecutive_repeats(plan)["timeline"]
    assert tl[0]["cuts"][0]["motion"]["mode"] == "pan"       # first stays
    assert tl[1]["cuts"][0]["motion"]["mode"] == "static"    # cross-segment repeat
    assert tl[2]["cuts"][0]["motion"]["mode"] == "pan"       # new file (first b) stays
    assert tl[3]["cuts"][0]["motion"]["mode"] == "static"    # b again after g3 -> static
    assert tl[3]["cuts"][1]["motion"]["mode"] == "static"    # within-segment repeat too


def test_merge_consecutive_duplicate_narration_holds_static():
    # p95/p96 both "Ancestor...?" -> the duplicate segment holds the FIRST
    # segment's image as ONE static cut (no second animated panel, no re-voiced
    # loop of the same line).
    plan = {"timeline": [
        {"segment_id": "g1", "tts_text": "Ancestor...?", "duration_sec": 3.0,
         "primary_scene_file": "p95.jpg",
         "cuts": [{"file": "p95.jpg", "motion": {"mode": "pan"}}]},
        {"segment_id": "g2", "tts_text": "[panicked] Ancestor...?", "duration_sec": 3.0,
         "primary_scene_file": "p96.jpg",
         "cuts": [{"file": "p96.jpg", "motion": {"mode": "kenburns"}}]},
        {"segment_id": "g3", "tts_text": "He turns away.", "duration_sec": 2.0,
         "primary_scene_file": "p97.jpg",
         "cuts": [{"file": "p97.jpg", "motion": {"mode": "pan"}}]},
    ]}
    tl = rp.merge_consecutive_duplicate_narration(plan)["timeline"]
    # first segment is untouched
    assert tl[0]["cuts"][0]["file"] == "p95.jpg"
    assert tl[0]["cuts"][0]["motion"]["mode"] == "pan"
    # duplicate (normalized match across mood tag) collapses to one held static cut
    assert len(tl[1]["cuts"]) == 1
    assert tl[1]["cuts"][0]["file"] == "p95.jpg"
    assert tl[1]["cuts"][0]["held"] is True
    assert tl[1]["cuts"][0]["motion"]["mode"] == "static"
    assert tl[1]["cuts"][0]["dur"] == 3.0
    # distinct narration is untouched
    assert tl[2]["cuts"][0]["file"] == "p97.jpg"
    assert tl[2]["cuts"][0]["motion"]["mode"] == "pan"


def test_merge_consecutive_duplicate_narration_three_in_a_row():
    plan = {"timeline": [
        {"segment_id": "g1", "tts_text": "Silence.", "duration_sec": 2.0,
         "cuts": [{"file": "a.jpg", "motion": {"mode": "pan"}}]},
        {"segment_id": "g2", "tts_text": "Silence.", "duration_sec": 2.0,
         "cuts": [{"file": "b.jpg", "motion": {"mode": "pan"}}]},
        {"segment_id": "g3", "tts_text": "Silence.", "duration_sec": 2.0,
         "cuts": [{"file": "c.jpg", "motion": {"mode": "pan"}}]},
    ]}
    tl = rp.merge_consecutive_duplicate_narration(plan)["timeline"]
    assert tl[0]["cuts"][0]["file"] == "a.jpg"
    assert tl[1]["cuts"][0]["file"] == "a.jpg" and tl[1]["cuts"][0]["held"]
    assert tl[2]["cuts"][0]["file"] == "a.jpg" and tl[2]["cuts"][0]["held"]


def test_merge_consecutive_duplicate_narration_empty_text_not_merged():
    # empty/whitespace tts_text must NOT count as a duplicate of empty
    plan = {"timeline": [
        {"segment_id": "g1", "tts_text": "", "duration_sec": 2.0,
         "cuts": [{"file": "a.jpg", "motion": {"mode": "pan"}}]},
        {"segment_id": "g2", "tts_text": "  ", "duration_sec": 2.0,
         "cuts": [{"file": "b.jpg", "motion": {"mode": "pan"}}]},
    ]}
    tl = rp.merge_consecutive_duplicate_narration(plan)["timeline"]
    assert tl[1]["cuts"][0]["file"] == "b.jpg"
    assert not tl[1]["cuts"][0].get("held")
