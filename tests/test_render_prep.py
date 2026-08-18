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
import json
from pathlib import Path

import cv2
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


# ---- 2c. cross-segment near-identical (the source-repeated eye p090/p095) ----
# Non-saturating ramps give a guaranteed 8x8-dhash split: two horizontal ramps
# (any brightness) hash identical (hamming 0 -> near); a vertical ramp hashes to
# the opposite (hamming 64 -> distinct).

def _hramp(w=300, h=400, base=0):
    row = np.linspace(0, 200, w).astype(np.uint8)
    img = np.stack([np.tile(row, (h, 1))] * 3, axis=-1)
    return np.clip(img.astype(int) + base, 0, 255).astype(np.uint8)


def _vramp(w=300, h=400):
    col = np.linspace(0, 200, h).astype(np.uint8)
    return np.stack([np.tile(col.reshape(-1, 1), (1, w))] * 3, axis=-1).astype(np.uint8)


def test_cross_segment_near_identical_dropped_when_segment_keeps_a_cut():
    # the eye p090 (g0017 tail) repeats as p095 (g0018 head); g0018 keeps p096/p097
    imgs = {"eye0.jpg": _hramp(base=0), "eye1.jpg": _hramp(base=20),
            "other.jpg": _vramp()}
    cbs = {"g17": [{"file": "eye0.jpg", "start": 0.0, "dur": 6.0}],
           "g18": [{"file": "eye1.jpg", "start": 0.0, "dur": 3.0},
                   {"file": "other.jpg", "start": 3.0, "dur": 3.0}]}
    out, dropped, canon = rp.drop_cross_segment_near_identical_cuts(
        cbs, ["g17", "g18"], lambda f: imgs.get(f))
    assert dropped == [("g18", "eye1.jpg")]              # LATER twin dropped
    assert canon == []                                   # dropped, not canonicalized
    assert [c["file"] for c in out["g18"]] == ["other.jpg"]
    assert [c["file"] for c in out["g17"]] == ["eye0.jpg"]
    assert abs(out["g18"][0]["dur"] - 6.0) < 1e-6        # freed time redistributed


def test_cross_segment_near_identical_sole_cut_canonicalized_to_twin():
    # if g0018's ONLY cut is the repeat, dropping it would empty the segment and
    # its narration would hold a neighbour 12-16s (the held-image regression).
    # Instead CANONICALIZE it to the earlier twin (eye0): the sole cut keeps its
    # own dur/audio but now shows eye0, so a follow-on merge folds the two
    # consecutive same-image sole cuts into ONE continuous Ken-Burns pan.
    imgs = {"eye0.jpg": _hramp(base=0), "eye1.jpg": _hramp(base=20)}
    cbs = {"g17": [{"file": "eye0.jpg", "start": 0.0, "dur": 6.0}],
           "g18": [{"file": "eye1.jpg", "start": 0.0, "dur": 6.0}]}
    out, dropped, canon = rp.drop_cross_segment_near_identical_cuts(
        cbs, ["g17", "g18"], lambda f: imgs.get(f))
    assert dropped == []                                 # never emptied
    assert canon == [("g18", "eye1.jpg", "eye0.jpg")]    # rewritten to the twin
    assert [c["file"] for c in out["g18"]] == ["eye0.jpg"]  # now the earlier twin
    assert len(out["g18"]) == 1                          # still exactly one cut
    assert abs(out["g18"][0]["dur"] - 6.0) < 1e-6        # its own dur untouched

    # merge-composition follow-on: the rewritten plan's two consecutive same-image
    # sole cuts fold into a single continuous-motion run (no held-image freeze).
    plan = {"timeline": [
        {"segment_id": "g17", "duration_sec": 6.0, "cuts": out["g17"]},
        {"segment_id": "g18", "duration_sec": 6.0, "cuts": out["g18"]},
    ]}
    merged = rp.merge_consecutive_same_image_cuts(plan)
    m17 = merged["timeline"][0]["cuts"]
    m18 = merged["timeline"][1]["cuts"]
    assert [c["file"] for c in m17] == ["eye0.jpg"]      # both items still show eye0
    assert [c["file"] for c in m18] == ["eye0.jpg"]
    assert len(m17) == 1 and len(m18) == 1               # neither segment emptied
    # each item carries a slice of ONE continuous slow move (not a static hold)
    assert m17[0].get("motion", {}).get("mode") == "kenburns"
    assert m18[0].get("motion", {}).get("mode") == "kenburns"


def test_cross_segment_distinct_panels_both_kept():
    imgs = {"a.jpg": _hramp(), "b.jpg": _vramp()}
    cbs = {"g1": [{"file": "a.jpg", "start": 0.0, "dur": 4.0}],
           "g2": [{"file": "b.jpg", "start": 0.0, "dur": 4.0}]}
    out, dropped, canon = rp.drop_cross_segment_near_identical_cuts(
        cbs, ["g1", "g2"], lambda f: imgs.get(f))
    assert dropped == []
    assert canon == []                                   # distinct: no canonicalize


def test_cross_segment_exempt_is_not_a_dedup_reference():
    # an exempt (system/blank) panel is never dropped AND never a comparison
    # reference, so a following near-identical panel has no prev to match.
    imgs = {"sys.jpg": _hramp(base=0), "eye.jpg": _hramp(base=20), "x.jpg": _vramp()}
    cbs = {"g1": [{"file": "sys.jpg", "start": 0.0, "dur": 4.0}],
           "g2": [{"file": "eye.jpg", "start": 0.0, "dur": 4.0},
                  {"file": "x.jpg", "start": 4.0, "dur": 4.0}]}
    out, dropped, canon = rp.drop_cross_segment_near_identical_cuts(
        cbs, ["g1", "g2"], lambda f: imgs.get(f), exempt={"sys.jpg"})
    assert dropped == []
    assert canon == []                                   # exempt prev: no canonicalize


# ---- FIX 1 Root A: bubble-masked hashing (identical art, different dialogue) --
# Text cleaning removes only what is INSIDE a bubble — the outline stays — so two
# identical drawings under different dialogue split the hash and slip the dedup
# ladder (the assassin p054/p055, crying p104/p105 pairs). _mask_bubbles_for_hash
# neutralizes each bubble region (flat-surround fill, else Telea inpaint) so the
# art collapses to a near-dup — HASH-ONLY, never written to disk.

def _bubble_base(grad="h"):
    """Flat-white top band (where the bubbles live, so a flat-surround fill
    restores them EXACTLY) over a low-frequency gradient (gives NCC variance that
    survives the 64x64 downscale, so distinct art stays distinct)."""
    img = np.full((400, 300, 3), 255, np.uint8)
    if grad == "h":
        row = np.linspace(0, 220, 300).astype(np.uint8)
        img[200:400] = np.stack([np.tile(row, (200, 1))] * 3, axis=-1)
    else:
        col = np.linspace(0, 220, 200).astype(np.uint8)
        img[200:400] = np.stack([np.tile(col.reshape(-1, 1), (1, 300))] * 3, axis=-1)
    return img.astype(np.uint8)


def _with_bubble(img, x1):
    """Paint a cleaned bubble (dark OUTLINE only — text already removed) on the
    white band; return (image, box) with a small pad around the outline."""
    out = img.copy()
    cv2.rectangle(out, (x1, 40), (x1 + 120, 160), (0, 0, 0), 4)
    return out, (x1 - 2, 38, x1 + 122, 162)


def test_mask_bubbles_makes_identical_art_a_near_dup_within_segment():
    # identical art, the bubble drawn at a DIFFERENT spot per copy
    a, boxA = _with_bubble(_bubble_base("h"), 30)
    b, boxB = _with_bubble(_bubble_base("h"), 150)
    cuts = [{"file": "p1.jpg", "start": 0.0, "dur": 4.0},
            {"file": "p2.jpg", "start": 4.0, "dur": 4.0}]
    imgs = {"p1.jpg": a, "p2.jpg": b}
    # raw: the differing bubbles hold the NCC below the 0.96 gate -> both kept
    assert rp._near_identical_similarity(a, b) < 0.96
    _, raw = rp.drop_near_identical_cuts(cuts, imgs)
    assert raw == []
    # masked: bubbles neutralized -> identical art scores ~1.0 -> later dropped
    assert rp._near_identical_similarity(
        a, b, boxes_a=[boxA], boxes_b=[boxB]) >= 0.96
    _, masked = rp.drop_near_identical_cuts(
        cuts, imgs, boxes_by_file={"p1.jpg": [boxA], "p2.jpg": [boxB]})
    assert masked == ["p2.jpg"]


def test_mask_bubbles_keeps_distinct_art_distinct():
    # different art (horizontal vs vertical gradient), each with a bubble: masking
    # must NOT collapse them into a false-positive drop.
    a, boxA = _with_bubble(_bubble_base("h"), 30)
    c, boxC = _with_bubble(_bubble_base("v"), 30)
    cuts = [{"file": "a.jpg", "start": 0.0, "dur": 4.0},
            {"file": "c.jpg", "start": 4.0, "dur": 4.0}]
    assert rp._near_identical_similarity(
        a, c, boxes_a=[boxA], boxes_b=[boxC]) < 0.96
    _, masked = rp.drop_near_identical_cuts(
        cuts, {"a.jpg": a, "c.jpg": c},
        boxes_by_file={"a.jpg": [boxA], "c.jpg": [boxC]})
    assert masked == []


def test_mask_bubbles_threads_into_cross_segment_dhash_drop():
    # Root A also masks the cross-segment dhash pass. Identical art with a FILLED
    # bubble at different spots (art surround -> Telea inpaint branch): raw dhash
    # is far apart (kept); masked collapses so the later twin drops when its
    # segment keeps another cut.
    eyeL = _hramp(); boxL = (30, 30, 170, 170)
    cv2.rectangle(eyeL, boxL[:2], boxL[2:], (255, 255, 255), -1)
    eyeR = _hramp(); boxR = (130, 230, 270, 370)
    cv2.rectangle(eyeR, boxR[:2], boxR[2:], (255, 255, 255), -1)
    imgs = {"eyeL.jpg": eyeL, "eyeR.jpg": eyeR, "other.jpg": _vramp()}
    cbs = {"g1": [{"file": "eyeL.jpg", "start": 0.0, "dur": 4.0}],
           "g2": [{"file": "eyeR.jpg", "start": 0.0, "dur": 3.0},
                  {"file": "other.jpg", "start": 3.0, "dur": 3.0}]}
    boxes = {"eyeL.jpg": [boxL], "eyeR.jpg": [boxR], "other.jpg": []}
    # unmasked: the bubbles hold the twins apart -> nothing dropped
    _, d0, c0 = rp.drop_cross_segment_near_identical_cuts(
        {k: list(v) for k, v in cbs.items()}, ["g1", "g2"], lambda f: imgs.get(f))
    assert d0 == [] and c0 == []
    # masked (get_boxes): the twins collapse -> later dropped (g2 keeps 'other')
    out, d1, canon = rp.drop_cross_segment_near_identical_cuts(
        {k: list(v) for k, v in cbs.items()}, ["g1", "g2"],
        lambda f: imgs.get(f), get_boxes=lambda f: boxes.get(f, []))
    assert d1 == [("g2", "eyeR.jpg")] and canon == []
    assert [c["file"] for c in out["g2"]] == ["other.jpg"]


def test_mask_bubbles_empty_boxes_is_noop():
    img = _hramp(base=10)
    out = rp._mask_bubbles_for_hash(img, [])
    assert out is img                                    # untouched, no copy


# ---- FIX 4: narration-bearing panels are protected from the dedup ------------
# REGRESSION (panel-collapse): a panel that owns its own narration segment must
# never be dropped as a "duplicate". Only a TRUE same-file consecutive run folds
# (merge_consecutive_same_image_cuts); distinct files are always kept.

def test_narrated_files_from_plan_collects_owning_panels():
    plan = {"timeline": [
        {"segment_id": "g0001_p00", "primary_scene_file": "p1.jpg",
         "tts_text": "He draws his blade.", "cuts": [{"file": "p1.jpg"}]},
        {"segment_id": "g0001_p01", "primary_scene_file": "p2.jpg",
         "tts_text": "", "cuts": [{"file": "p2.jpg"}]},          # no narration -> not owned
        {"segment_id": "branding_intro", "primary_scene_file": "logo.jpg",
         "tts_text": "x", "branding": True, "cuts": [{"file": "logo.jpg"}]},  # skipped
        {"segment_id": "g0002_p00", "primary_scene_file": "scenes/p3.jpg",
         "tts_text": "[tense] The countdown begins.", "cuts": [{"file": "p3.jpg"}]},
    ]}
    got = rp.narrated_files_from_plan(plan)
    assert got == {"p1.jpg", "p3.jpg"}          # basename; narration-bearing only


def test_narrated_files_from_plan_protects_whole_span():
    # a multi-panel flow span (smirk + transformation-flash) shows BOTH — every
    # panel in a narrated span is protected, not just the primary, else the seam
    # dedup drops the later panel (p074 flash) and the span renders only its head
    plan = {"timeline": [
        {"segment_id": "g0014_p02", "primary_scene_file": "p073.jpg",
         "tts_text": "A blinding flash erupts.",
         "scene_files": ["p073.jpg", "p074.jpg"],
         "cuts": [{"file": "p073.jpg"}, {"file": "p074.jpg"}]},
    ]}
    assert rp.narrated_files_from_plan(plan) == {"p073.jpg", "p074.jpg"}


def test_narration_bearing_panel_survives_near_identical_drop():
    """Two near-identical DISTINCT files each own a narration line. Without
    protection the later would be dropped as a dup; once it is in `protect`
    (because it owns a narration segment) BOTH survive — every distinct narrated
    panel is shown."""
    base = _pattern(400, 300)
    near = (base.astype(np.int16) + 2).clip(0, 255).astype(np.uint8)
    cuts = [{"file": "p1.jpg", "start": 0.0, "dur": 4.0},
            {"file": "p2.jpg", "start": 4.0, "dur": 4.0}]
    imgs = {"p1.jpg": base, "p2.jpg": near}
    # baseline: unprotected -> the later near-identical panel drops
    _, dropped = rp.drop_near_identical_cuts(cuts, imgs)
    assert dropped == ["p2.jpg"]
    # protected (owns a narration segment) -> nothing dropped, both shown
    out, dropped = rp.drop_near_identical_cuts(cuts, imgs, protect={"p2.jpg"})
    assert dropped == []
    assert [c["file"] for c in out] == ["p1.jpg", "p2.jpg"]


def test_true_same_file_consecutive_merges_to_one_kenburns():
    """A TRUE exact same-FILE consecutive run folds into ONE continuous Ken Burns
    (the legit collapse we keep) — distinct files are never merged."""
    plan = {"timeline": [
        {"segment_id": "g0001_p00", "cuts": [
            {"file": "p1.jpg", "start": 0.0, "dur": 2.0},
            {"file": "p1.jpg", "start": 2.0, "dur": 3.0}]},      # same file x2
        {"segment_id": "g0001_p01", "cuts": [
            {"file": "p2.jpg", "start": 0.0, "dur": 2.0},
            {"file": "p3.jpg", "start": 2.0, "dur": 2.0}]},      # distinct files
    ]}
    out = rp.merge_consecutive_same_image_cuts(plan)
    first = out["timeline"][0]["cuts"]
    assert len(first) == 1                         # same-file run collapsed to one
    assert first[0]["file"] == "p1.jpg"
    assert abs(first[0]["dur"] - 5.0) < 1e-6       # summed duration, one Ken Burns
    second = out["timeline"][1]["cuts"]
    assert [c["file"] for c in second] == ["p2.jpg", "p3.jpg"]   # distinct never merged


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


def test_branding_drops_intro_and_outro_ends_on_last_panel():
    # channel decision (2026-06-15): NO intro. (2026-06-29): NO outro either —
    # every video now ends on the last STORY panel; the only branding left is the
    # corner watermark overlay drawn by remotion/src/Branding.tsx (not a timeline
    # item). insert_branding_items must therefore never emit a branding_outro item,
    # even when an outro_dur is passed in.
    plan = _mini_plan()
    out = rp.insert_branding_items(plan, intro_dur=6.0, outro_dur=12.1)
    tl = out["timeline"]
    assert [t["segment_id"] for t in tl] == ["g0001_p00", "g0002_p01"]
    assert all(t.get("branding") not in ("intro", "outro") for t in tl)
    assert all(t.get("segment_id") != "branding_outro" for t in tl)
    # the video ends on the last story panel, unchanged duration
    assert out["total_duration_sec"] == 18.0


def test_branding_no_outro_for_any_which():
    # bundle segments used to render intro-on-first / outro-on-last; now NO
    # position carries any branding item — a concatenated season ends on its
    # last story panel.
    plan = {"timeline": [
        {"segment_id": "g0001_p00", "start_sec": 0.0, "end_sec": 10.0,
         "duration_sec": 10.0, "cuts": [{"file": "a.jpg", "start": 0, "dur": 10}]}],
        "total_duration_sec": 10.0}
    for which in ("both", "intro", "outro", "none"):
        out = rp.insert_branding_items(plan, intro_dur=6.0, outro_dur=12.0,
                                       which=which)
        brandings = [i.get("branding") for i in out["timeline"]]
        assert brandings == [None], which
        assert all(i.get("segment_id") != "branding_outro"
                   for i in out["timeline"]), which


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


def test_junk_sole_cut_substituted_via_coverage_map():
    # the manual_drops/heal drop path: junk files get coverage 1.0 so the SAME
    # substitute pass that covers chrome/husk garbage also replaces operator-
    # dropped sole cuts (per-panel 1:1 makes every segment sole-cut). The
    # dropped panel's narration keeps playing while the story-adjacent good
    # panel holds; exempt (system/protected) files are never held away.
    cbs = {"g1_p00": [{"file": "good.jpg", "start": 0.0, "dur": 4.0}],
           "g1_p01": [{"file": "dup.jpg", "start": 0.0, "dur": 3.0}],
           "g1_p02": [{"file": "sys.jpg", "start": 0.0, "dur": 2.0}]}
    junk = {"dup.jpg", "sys.jpg"}
    out, subs = rp.substitute_garbage_sole_cuts(
        cbs, {f: 1.0 for f in junk},
        durations={"g1_p00": 4.0, "g1_p01": 3.0, "g1_p02": 2.0},
        exempt={"sys.jpg"}, order=["g1_p00", "g1_p01", "g1_p02"])
    assert out["g1_p01"] == [{"file": "good.jpg", "start": 0.0, "dur": 3.0,
                              "held": True}]
    assert out["g1_p02"] == cbs["g1_p02"]      # exempt card never held away
    assert [(s[0], s[1], s[2]) for s in subs] == [
        ("g1_p01", "dup.jpg", "good.jpg")]


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
    """GLOBAL cap: a non-exempt panel shows at most ONCE chapter-wide; any
    re-show is dropped and the previous distinct panel HOLDS — narration
    continues, the same image never re-animates. A,B,A,B,A → A and B each show
    ONCE and the rest hold (merge_consecutive later animates the held same-image
    run as ONE continuous pan); no fresh re-show of A."""
    cuts = {
        "g1": [{"file": "A.jpg", "start": 0.0, "dur": 4.0}],
        "g2": [{"file": "B.jpg", "start": 0.0, "dur": 4.0}],
        "g3": [{"file": "A.jpg", "start": 0.0, "dur": 4.0}],
        "g4": [{"file": "B.jpg", "start": 0.0, "dur": 4.0}],
        "g5": [{"file": "A.jpg", "start": 0.0, "dur": 5.0}],   # 3rd A — dropped, holds B
    }
    order = ["g1", "g2", "g3", "g4", "g5"]
    out, holds = rp.cap_repeats_with_holds(
        cuts, durations={s: 4.0 for s in order} | {"g5": 5.0}, order=order)
    # A,B,A,B,A: after the first A,B every reuse holds the previous distinct
    # panel (B) — no same-image re-animation anywhere.
    assert holds == [("g3", "B.jpg"), ("g4", "B.jpg"), ("g5", "B.jpg")]
    assert out["g3"][0]["held"] is True
    g5 = out["g5"][0]
    assert g5.get("held") is True and g5["file"] == "B.jpg"   # 3rd A dropped → holds B
    counts = {}
    for s_ in order:
        for c in out[s_]:
            if not c.get("held"):
                counts[c["file"]] = counts.get(c["file"], 0) + 1
    assert max(counts.values()) <= 1                          # each panel emitted ONCE


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


def test_merge_same_image_collapses_within_item_to_one_kenburns_cut():
    # 3 consecutive same-image cuts in ONE item -> 1 merged cut spanning the total
    # duration, with a single non-static slow Ken Burns (not N frozen holds).
    plan = {"timeline": [
        {"segment_id": "g1", "duration_sec": 14.0, "cuts": [
            {"file": "a.jpg", "start": 0.0, "dur": 4.0, "motion": {"mode": "pan"}},
            {"file": "a.jpg", "start": 4.0, "dur": 5.0, "motion": {"mode": "static"}},
            {"file": "a.jpg", "start": 9.0, "dur": 5.0, "motion": {"mode": "static"}},
        ]},
    ]}
    cuts = rp.merge_consecutive_same_image_cuts(plan)["timeline"][0]["cuts"]
    assert len(cuts) == 1 and cuts[0]["file"] == "a.jpg"
    assert round(cuts[0]["dur"], 4) == 14.0                  # spans the full duration
    m = cuts[0]["motion"]
    assert m["mode"] != "static" and m["strength"] > 0       # NOT static
    assert m["zoom"]["start"] != m["zoom"]["end"]            # a real Ken Burns


def test_merge_same_image_one_continuous_slow_kenburns_across_segments():
    # production case: p92 held over 3 per-panel narration SEGMENTS (g0024_p20,
    # g0025_p00, g0025_p01) by cap_repeats_with_holds. Show the image ONCE with one
    # slow Ken Burns spanning the whole run; per-segment audio + timing untouched.
    plan = {"timeline": [
        {"segment_id": "g0024_p20", "duration_sec": 4.0, "tts_audio": "a20.wav",
         "cuts": [{"file": "p92.jpg", "start": 0.0, "dur": 4.0, "motion": {"mode": "pan"}}]},
        {"segment_id": "g0025_p00", "duration_sec": 5.0, "tts_audio": "b00.wav",
         "cuts": [{"file": "p92.jpg", "start": 0.0, "dur": 5.0, "held": True,
                   "motion": {"mode": "static", "zoom": {"start": 1.0, "end": 1.0},
                              "strength": 0.0}}]},
        {"segment_id": "g0025_p01", "duration_sec": 5.0, "tts_audio": "b01.wav",
         "cuts": [{"file": "p92.jpg", "start": 0.0, "dur": 5.0, "held": True,
                   "motion": {"mode": "static", "zoom": {"start": 1.0, "end": 1.0},
                              "strength": 0.0}}]},
        {"segment_id": "g0026_p00", "duration_sec": 3.0, "tts_audio": "c.wav",
         "cuts": [{"file": "p93.jpg", "start": 0.0, "dur": 3.0, "motion": {"mode": "pan"}}]},
    ]}
    tl = rp.merge_consecutive_same_image_cuts(plan)["timeline"]
    # audio + per-segment timing intact: same items, same audio, same durations
    assert [it["segment_id"] for it in tl] == \
        ["g0024_p20", "g0025_p00", "g0025_p01", "g0026_p00"]
    assert [it["tts_audio"] for it in tl] == ["a20.wav", "b00.wav", "b01.wav", "c.wav"]
    assert [it["cuts"][0]["dur"] for it in tl] == [4.0, 5.0, 5.0, 3.0]
    run = [tl[0]["cuts"][0], tl[1]["cuts"][0], tl[2]["cuts"][0]]
    assert all(c["file"] == "p92.jpg" for c in run)              # one image, shown once
    # NOT static / not frozen holds: every cut in the run animates
    assert all(c["motion"]["mode"] != "static" and c["motion"]["strength"] > 0
               for c in run)
    # ONE continuous slow Ken Burns: head starts the pan, tail ends it, each slice
    # continuous at the boundary (no restart, spread over the merged span)
    assert run[0]["motion"]["zoom"]["start"] == rp._MERGE_ZOOM_START
    assert run[2]["motion"]["zoom"]["end"] == rp._MERGE_ZOOM_END
    assert run[0]["motion"]["zoom"]["start"] != run[2]["motion"]["zoom"]["end"]
    assert run[0]["motion"]["zoom"]["end"] == run[1]["motion"]["zoom"]["start"]
    assert run[1]["motion"]["zoom"]["end"] == run[2]["motion"]["zoom"]["start"]
    # easing reads as ONE move: ease IN on the head slice, LINEAR through the
    # middle, ease OUT on the tail — per-slice ease_in_out made the pan
    # decelerate + re-accelerate at every segment boundary (the "restarting
    # pan" pulse the user saw on held runs).
    assert [c["motion"]["ease"] for c in run] == \
        ["ease_in", "linear", "ease_out"]
    # the distinct image after the run is untouched
    assert tl[3]["cuts"][0]["motion"] == {"mode": "pan"}


def test_merge_same_image_two_item_run_eases_in_then_out():
    plan = {"timeline": [
        {"segment_id": "g1", "duration_sec": 4.0, "tts_audio": "a.wav",
         "cuts": [{"file": "x.jpg", "start": 0.0, "dur": 4.0, "motion": {"mode": "s"}}]},
        {"segment_id": "g2", "duration_sec": 4.0, "tts_audio": "b.wav",
         "cuts": [{"file": "x.jpg", "start": 0.0, "dur": 4.0, "motion": {"mode": "s"}}]},
    ]}
    tl = rp.merge_consecutive_same_image_cuts(plan)["timeline"]
    assert [it["cuts"][0]["motion"]["ease"] for it in tl] == \
        ["ease_in", "ease_out"]


def test_merge_same_image_distinct_images_untouched():
    plan = {"timeline": [
        {"segment_id": "g1", "duration_sec": 4.0,
         "cuts": [{"file": "a.jpg", "start": 0.0, "dur": 4.0, "motion": {"mode": "pan"}}]},
        {"segment_id": "g2", "duration_sec": 4.0,
         "cuts": [{"file": "b.jpg", "start": 0.0, "dur": 4.0, "motion": {"mode": "kb"}}]},
    ]}
    tl = rp.merge_consecutive_same_image_cuts(plan)["timeline"]
    assert tl[0]["cuts"][0]["motion"] == {"mode": "pan"}
    assert tl[1]["cuts"][0]["motion"] == {"mode": "kb"}


def test_merge_same_image_branding_breaks_the_run():
    # the same image either side of a branding item is NOT one run
    plan = {"timeline": [
        {"segment_id": "g1", "duration_sec": 4.0,
         "cuts": [{"file": "a.jpg", "start": 0.0, "dur": 4.0, "motion": {"mode": "pan"}}]},
        {"segment_id": "branding_outro", "branding": "outro", "cuts": []},
        {"segment_id": "g2", "duration_sec": 4.0,
         "cuts": [{"file": "a.jpg", "start": 0.0, "dur": 4.0, "motion": {"mode": "pan"}}]},
    ]}
    tl = rp.merge_consecutive_same_image_cuts(plan)["timeline"]
    assert tl[0]["cuts"][0]["motion"] == {"mode": "pan"}
    assert tl[2]["cuts"][0]["motion"] == {"mode": "pan"}


def test_merge_same_image_longer_run_is_slower():
    # the per-second zoom rate = total_zoom_delta / run_duration, so a longer merge
    # moves slower (the user's "slowed to span the full duration").
    def _rate(durs):
        plan = {"timeline": [
            {"segment_id": f"g{i}", "duration_sec": d,
             "cuts": [{"file": "x.jpg", "start": 0.0, "dur": d, "motion": {"mode": "s"}}]}
            for i, d in enumerate(durs)]}
        tl = rp.merge_consecutive_same_image_cuts(plan)["timeline"]
        total = sum(durs)
        delta = tl[-1]["cuts"][0]["motion"]["zoom"]["end"] - \
            tl[0]["cuts"][0]["motion"]["zoom"]["start"]
        return delta / total
    assert _rate([4.0, 5.0, 5.0]) < _rate([2.0, 2.0])       # longer span -> slower


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


# ---- system-card guarantee vs empty-bubble exclusion (the ping-pong fix) ------
# These two tests are the BOTH-SIDES guard so the fix can't ping-pong: a system
# card is ALWAYS kept (its text IS the beat), while the empty dialogue bubble is
# still excluded (its text becomes narration). They must hold simultaneously.

def test_system_card_unconditionally_exempt_even_when_blanked_to_husk():
    # The regression (9118b8b) gated the system/title-card exemption on
    # `recoverable`, so a text-on-plain system card (Nano ch1 p000114
    # "7TH GENERATION NANO MACHINE") went non-recoverable after its text was
    # blanked and was DROPPED -> system_card_unshown. A system card's TEXT is the
    # on-screen story beat: it must ALWAYS be exempt, regardless of recoverability.
    assert rp.exempt_from_drop(
        recoverable=False, sys_box=True, title_card=False, rich=False,
        visual_story=False, panel_kind="system", has_ocr=True) is True
    # even with NO sys_box detection (weights missing / undetected), the
    # understanding's panel_kind="system" alone keeps it shown
    assert rp.exempt_from_drop(
        recoverable=False, sys_box=False, title_card=False, rich=False,
        visual_story=False, panel_kind="system", has_ocr=True) is True
    # a styled title card is likewise unconditionally kept
    assert rp.exempt_from_drop(
        recoverable=False, sys_box=False, title_card=True, rich=False,
        visual_story=False, panel_kind="story", has_ocr=True) is True


def test_empty_dialogue_caption_husk_is_not_exempt_and_drops():
    # The OTHER half of the ping-pong: an empty DIALOGUE bubble (Nano ch1 p000020)
    # whose text becomes narration is marked panel_kind=caption upstream and folded
    # away (panel_understand + story_group). Here, defense-in-depth, a caption husk
    # that is NON-recoverable after cleaning is NOT shielded — the system-card
    # guarantee above does not keep it (no sys_box, no title_card, kind != system).
    assert rp.exempt_from_drop(
        recoverable=False, sys_box=False, title_card=False, rich=False,
        visual_story=False, panel_kind="caption", has_ocr=True) is False
    # a recoverable caption (real art survives behind the blanked bubble) stays
    assert rp.exempt_from_drop(
        recoverable=True, sys_box=False, title_card=False, rich=False,
        visual_story=False, panel_kind="caption", has_ocr=True) is True


def test_near_identical_dedup_never_drops_protected_system_card():
    # Two near-identical cards back-to-back: normally the LATER drops. When the
    # later is a system card (its text IS the beat) it must SURVIVE — protect it.
    a = _stamp(np.full((400, 400, 3), 200, np.uint8), 5)
    cuts = [{"file": "art.jpg", "start": 0.0, "dur": 3.0},
            {"file": "sys.jpg", "start": 3.0, "dur": 3.0}]
    imgs = {"art.jpg": a, "sys.jpg": a.copy()}
    # baseline: without protection the later near-identical cut drops
    _, d0 = rp.drop_near_identical_cuts(cuts, imgs)
    assert d0 == ["sys.jpg"]
    # protected: the system card is kept and stays shown
    out, dropped = rp.drop_near_identical_cuts(cuts, imgs, protect={"sys.jpg"})
    assert dropped == []
    assert [c["file"] for c in out] == ["art.jpg", "sys.jpg"]


def test_cross_segment_dedup_never_drops_protected_system_card():
    # A system card visually matching the prior segment's panel would normally be
    # flagged a cross-segment duplicate; protect keeps it shown.
    big = _stamp(np.full((600, 400, 3), 200, np.uint8), 7)
    other = _stamp(np.full((600, 400, 3), 200, np.uint8), 99)
    cbs = {"g1": [{"file": "a.jpg", "start": 0.0, "dur": 4.0}],
           "g2": [{"file": "sys.jpg", "start": 0.0, "dur": 3.0},
                  {"file": "c.jpg", "start": 3.0, "dur": 3.0}]}
    imgs = {"a.jpg": big, "sys.jpg": big.copy(), "c.jpg": other}
    # baseline: sys.jpg matches a.jpg across the seam -> flagged dropped
    _, d0 = rp.drop_cross_segment_duplicate_cuts(
        cbs, ["g1", "g2"], lambda f: imgs.get(f))
    assert ("g2", "sys.jpg") in d0
    # protected: the system card is never flagged and stays in the shown cuts
    out, dropped = rp.drop_cross_segment_duplicate_cuts(
        cbs, ["g1", "g2"], lambda f: imgs.get(f), protect={"sys.jpg"})
    assert ("g2", "sys.jpg") not in dropped
    assert "sys.jpg" in [c["file"] for c in out["g2"]]


def test_content_bearing_system_card_stays_exempt():
    # a REAL in-world system/stat card survives cleaning (its styled text IS the
    # content) -> recoverable -> stays protected and shown.
    assert rp.exempt_from_drop(
        recoverable=True, sys_box=True, title_card=False, rich=False,
        visual_story=False, panel_kind="system", has_ocr=True) is True
    assert rp.exempt_from_drop(
        recoverable=True, sys_box=False, title_card=True, rich=False,
        visual_story=False, panel_kind="story", has_ocr=True) is True


def test_exempt_from_drop_doc_and_visual_story_unconditional():
    # document panels (text IS the content) and real story visuals are exempt
    # regardless — they always carry content.
    assert rp.exempt_from_drop(
        recoverable=False, sys_box=False, title_card=False, rich=True,
        visual_story=False, panel_kind="story", has_ocr=False) is True
    assert rp.exempt_from_drop(
        recoverable=False, sys_box=False, title_card=False, rich=False,
        visual_story=True, panel_kind="story", has_ocr=False) is True


def test_exempt_from_drop_broad_ocr_only_when_recoverable():
    # story/caption/system with OCR is exempt ONLY if it survived cleaning; an
    # emptied husk (not recoverable) is NOT shielded by carrying OCR.
    assert rp.exempt_from_drop(
        recoverable=True, sys_box=False, title_card=False, rich=False,
        visual_story=False, panel_kind="caption", has_ocr=True) is True
    assert rp.exempt_from_drop(
        recoverable=False, sys_box=False, title_card=False, rich=False,
        visual_story=False, panel_kind="caption", has_ocr=True) is False


# ---- D1: partial-drop reflow (no black time-holes) --------------------------

def test_cap_repeats_partial_drop_reflows_survivor_no_gap():
    """D1: a multi-cut segment that loses SOME-but-not-all cuts must reflow its
    survivors across the FULL window — otherwise the no-cut span renders as the
    #000 background (black). Mirrors g0003_p06 / g0018_p37 / g0022_p16 gaps.

    Segment cuts [A@0-2, B@2-4] (window 4s); B repeats a recently-shown panel
    so it is dropped; survivor A must reflow to [0,4]."""
    cuts = {
        "g1_p00": [{"file": "B.jpg", "start": 0.0, "dur": 2.0}],     # B shown @idx0
        "g1_p01": [{"file": "C.jpg", "start": 0.0, "dur": 4.0}],
        "g1_p02": [{"file": "A.jpg", "start": 0.0, "dur": 2.0},      # window 4s
                   {"file": "B.jpg", "start": 2.0, "dur": 2.0}],     # B near -> dropped
    }
    order = ["g1_p00", "g1_p01", "g1_p02"]
    durations = {"g1_p00": 2.0, "g1_p01": 4.0, "g1_p02": 4.0}
    out, _holds = rp.cap_repeats_with_holds(
        cuts, durations=durations, order=order)
    seg = out["g1_p02"]
    assert [c["file"] for c in seg] == ["A.jpg"]                 # B dropped
    assert seg[0]["start"] == 0.0                                # reflowed to start
    assert abs(seg[0]["start"] + seg[0]["dur"] - 4.0) < 1e-6     # covers to window end


def test_cap_repeats_partial_drop_survivors_tile_window():
    """D1: with TWO survivors and a dropped middle cut, the survivors must tile
    the window contiguously — next.start == prev.end, last.end == window end,
    no overlap, no hole."""
    cuts = {
        "g2_p00": [{"file": "D.jpg", "start": 0.0, "dur": 2.0}],    # D shown @idx0
        "g2_p01": [{"file": "X.jpg", "start": 0.0, "dur": 1.0},     # window 3s
                   {"file": "D.jpg", "start": 1.0, "dur": 1.0},     # D near -> dropped
                   {"file": "Y.jpg", "start": 2.0, "dur": 1.0}],
    }
    order = ["g2_p00", "g2_p01"]
    durations = {"g2_p00": 2.0, "g2_p01": 3.0}
    out, _holds = rp.cap_repeats_with_holds(
        cuts, durations=durations, order=order)
    seg = out["g2_p01"]
    assert [c["file"] for c in seg] == ["X.jpg", "Y.jpg"]        # middle D dropped
    assert seg[0]["start"] == 0.0
    for prev, nxt in zip(seg, seg[1:]):
        assert abs((prev["start"] + prev["dur"]) - nxt["start"]) < 1e-6  # no gap/overlap
    assert abs((seg[-1]["start"] + seg[-1]["dur"]) - 3.0) < 1e-6         # window end


# ---- D2: no non-adjacent re-emit of the same panel within a group -----------

def test_cap_repeats_no_nonadjacent_reemit_within_group():
    """D2: a non-exempt panel reused NON-adjacently in the SAME group (gap beyond
    the radius-3 near window) must show ONCE; the later slot HOLDS the adjacent
    distinct panel, with no black gap (IE p000091 idx89&93, p000109 idx106&110)."""
    order = [f"g0001_p0{i}" for i in range(5)]
    cuts = {
        "g0001_p00": [{"file": "P.jpg", "start": 0.0, "dur": 4.0}],   # P 1st show
        "g0001_p01": [{"file": "Q.jpg", "start": 0.0, "dur": 4.0}],
        "g0001_p02": [{"file": "R.jpg", "start": 0.0, "dur": 4.0}],
        "g0001_p03": [{"file": "S.jpg", "start": 0.0, "dur": 4.0}],
        "g0001_p04": [{"file": "P.jpg", "start": 0.0, "dur": 5.0}],   # P again, gap 4
    }
    durations = {s: 4.0 for s in order} | {"g0001_p04": 5.0}
    out, holds = rp.cap_repeats_with_holds(
        cuts, durations=durations, order=order)
    p04 = out["g0001_p04"][0]
    assert p04.get("held") is True and p04["file"] == "S.jpg"     # holds adjacent panel
    assert p04["start"] == 0.0 and abs(p04["dur"] - 5.0) < 1e-6   # full window, no gap
    assert ("g0001_p04", "S.jpg") in holds
    counts = {}
    for s in order:
        for c in out[s]:
            if not c.get("held"):
                counts[c["file"]] = counts.get(c["file"], 0) + 1
    assert counts["P.jpg"] == 1                                   # P emitted ONCE


def test_cap_repeats_no_reemit_across_groups_global():
    """D2 is GLOBAL, not group-scoped: a non-exempt panel shown in one group must
    NOT replay in a LATER group either (group N shows it, group N+1 must not show
    it again with the same animation — the cross-group repeat). It shows ONCE
    chapter-wide; the later slot HOLDS the adjacent distinct panel. (Exempt cards
    MAY still recur far apart — see test_cap_repeats_exempt_recurs_far_apart.)"""
    order = ["g0001_p00", "g0001_p01", "g0001_p02", "g0001_p03", "g0002_p00"]
    cuts = {
        "g0001_p00": [{"file": "P.jpg", "start": 0.0, "dur": 4.0}],
        "g0001_p01": [{"file": "Q.jpg", "start": 0.0, "dur": 4.0}],
        "g0001_p02": [{"file": "R.jpg", "start": 0.0, "dur": 4.0}],
        "g0001_p03": [{"file": "S.jpg", "start": 0.0, "dur": 4.0}],
        "g0002_p00": [{"file": "P.jpg", "start": 0.0, "dur": 4.0}],   # new group, gap 4
    }
    durations = {s: 4.0 for s in order}
    out, holds = rp.cap_repeats_with_holds(
        cuts, durations=durations, order=order)
    g2 = out["g0002_p00"][0]
    assert g2.get("held") is True and g2["file"] == "S.jpg"      # holds adjacent, NOT P
    assert ("g0002_p00", "S.jpg") in holds
    counts = {}
    for s in order:
        for c in out[s]:
            if not c.get("held"):
                counts[c["file"]] = counts.get(c["file"], 0) + 1
    assert counts["P.jpg"] == 1                                  # P emitted ONCE chapter-wide




def test_protect_narrated_from_junk_removes_narrated_keeps_rest():
    # a narrated panel the auto-judge called junk (an action/flash climax it
    # read as "abstract glow") must survive — dropping it holds a neighbour
    # while the narrator describes an unshown shot. Non-narrated junk stays.
    junk = {"p044.jpg": "abstract glow", "p090.jpg": "blank husk",
            "p074.jpg": "flat gradient"}
    narrated = {"p044.jpg", "p074.jpg"}          # these own spoken lines
    out = rp.protect_narrated_from_junk(junk, narrated)
    assert set(out) == {"p090.jpg"}              # only non-narrated junk remains
    assert junk is out                           # mutates in place


def test_protect_narrated_from_junk_empty_narrated_is_noop():
    junk = {"p001.jpg": "sliver"}
    assert rp.protect_narrated_from_junk(dict(junk), set()) == junk
    assert rp.protect_narrated_from_junk(dict(junk), None) == junk


def test_protect_narrated_from_junk_also_protects_system_files():
    # FIX 3: a stamped system card the visual judge called junk must SURVIVE — its
    # on-screen text IS the beat, and dropping it here surfaces a blocking
    # system_card_unshown. also_protect spares it even when it owns no narration.
    junk = {"sys.jpg": "flat glow", "cap.jpg": "blank husk", "art.jpg": "abstract"}
    narrated = {"art.jpg"}                        # only art owns a spoken line
    out = rp.protect_narrated_from_junk(junk, narrated, also_protect={"sys.jpg"})
    assert set(out) == {"cap.jpg"}               # sys (system) + art (narrated) spared
    assert junk is out                           # mutates in place


# ---- Tracks B+C: shown-panel twin INVARIANT + manufactured-twin guard --------
# Regression fixtures: 4 REAL Nano ch1 panels (downscaled, tests/fixtures/dedup)
# pinning the two forensically verified production escapes (2026-07-06):
#   D3 p000054/p000055 — consecutive artist "echo" panels; the second's dialogue
#     is a tail-substring of the first's. Both own narration -> protect_files
#     shielded them from every per-segment dedup, and the one narration-overriding
#     pass bailed at its min_area_ratio gate (0.560 < 0.7) before hashing.
#   D4 p000090/p000095 — DISTINCT panels (raw masked ham ~22) whose CLEANED crops
#     became hash-twins (eye band, ham <= 8) -> the sole-cut canonicalize swapped
#     p095 to p000090 and the merge held ONE image ~24s; p095's art never shown.
# Bubble boxes below were measured with the production ogkalu detector ON the
# downscaled fixtures (hardcoded so the tests stay hermetic — no model download).

_FIXDIR = Path(__file__).resolve().parent / "fixtures" / "dedup"
_FIX_BOXES = {
    # p000054/p000055 measured at max-width 600 (q65) — re-margined off the
    # ham==14 boundary (was max-width 400/q60); masked ham now 12, matching
    # the full-resolution originals (see D3 note above).
    "p000054.jpg": [(2, 1, 245, 181), (80, 150, 344, 362)],
    "p000055.jpg": [(83, 0, 328, 106)],
    "p000090.jpg": [],
    "p000095.jpg": [(155, 276, 389, 447)],
}
_FIX_OCR = {
    "p000054.jpg": ("DOING SOMETHING LIKE THAT ISN'T MY STYLE AT ALL BUT... "
                    "SINCE OUR COMRADE DIED, I GUESS THERE ISN'T A CHOICE..?"),
    "p000055.jpg": "GUESS THERE ISN'T A CHOICE..?",
}


def _fiximg(f):
    img = cv2.imread(str(_FIXDIR / f))
    assert img is not None, f"fixture missing: {_FIXDIR / f}"
    return img


def test_dedup_invariant_folds_echo_pair():
    # D3 through the new FINAL invariant pass: masked-raw hashing on the FULL
    # panels (crop geometry can't bypass) + OCR dialogue containment catches the
    # echo pair; resolution is MERGE into the richer panel (2 bubbles > 1), so
    # both narration lines still land on a shown image and no line is dropped.
    plan = {"timeline": [
        {"segment_id": "g0011_p00", "tts_text": "The assassin wavers.",
         "scene_files": ["p000054.jpg"], "duration_sec": 4.0,
         "cuts": [{"file": "p000054.jpg", "start": 0.0, "dur": 4.0}]},
        {"segment_id": "g0011_p01", "tts_text": "But a comrade died.",
         "scene_files": ["p000055.jpg"], "duration_sec": 3.0,
         "cuts": [{"file": "p000055.jpg", "start": 0.0, "dur": 3.0}]},
    ], "scene_dims": {}}
    imgs = {f: _fiximg(f) for f in ("p000054.jpg", "p000055.jpg")}
    out, folds = rp.enforce_shown_twin_invariant(
        plan, lambda f: imgs.get(f),
        get_raw_boxes=lambda f: _FIX_BOXES.get(f, []),
        get_ocr=lambda f: _FIX_OCR.get(f, ""))
    assert len(folds) == 1                       # exactly the echo pair folded
    shown = [c["file"] for it in out["timeline"] for c in it.get("cuts") or []]
    assert shown == ["p000054.jpg", "p000054.jpg"]   # ONE shown file (the richer)
    for it in out["timeline"]:                   # every narration line kept, 1:1
        assert str(it.get("tts_text") or "").strip()
        assert len(it.get("cuts") or []) == 1
        assert abs(float(it["cuts"][0]["dur"]) -
                   float(it["duration_sec"])) < 1e-6   # audio timing untouched
    # writer intent (scene_files) preserved — QA can still see the original span
    assert out["timeline"][1]["scene_files"] == ["p000055.jpg"]
    # the EXISTING merge pass then animates the run as one continuous slow pan
    merged = rp.merge_consecutive_same_image_cuts(out)
    for it in merged["timeline"]:
        assert it["cuts"][0].get("motion", {}).get("mode") == "kenburns"


def test_dedup_invariant_skips_system_cards():
    # two DISTINCT system notifications share a UI frame -> hash-twins; the
    # invariant must never merge them (their on-screen text IS the story beat).
    imgs = {"sysA.jpg": _hramp(base=0), "sysB.jpg": _hramp(base=15)}
    plan = {"timeline": [
        {"segment_id": "s1", "tts_text": "The system speaks.",
         "cuts": [{"file": "sysA.jpg", "start": 0.0, "dur": 3.0}]},
        {"segment_id": "s2", "tts_text": "It speaks again.",
         "cuts": [{"file": "sysB.jpg", "start": 0.0, "dur": 3.0}]},
    ], "scene_dims": {}}
    out, folds = rp.enforce_shown_twin_invariant(
        {"timeline": [dict(i, cuts=[dict(c) for c in i["cuts"]])
                      for i in plan["timeline"]], "scene_dims": {}},
        lambda f: imgs.get(f), skip_files={"sysA.jpg", "sysB.jpg"})
    assert folds == []
    assert [c["file"] for it in out["timeline"] for c in it["cuts"]] == \
        ["sysA.jpg", "sysB.jpg"]
    # sanity contrast: WITHOUT the system exemption the same pair DOES fold
    out2, folds2 = rp.enforce_shown_twin_invariant(plan, lambda f: imgs.get(f))
    assert len(folds2) == 1


def test_dedup_invariant_leaves_distinct_panels_alone():
    # distinct art (horizontal vs vertical ramp, ham ~64) is never folded — and
    # the D4 raws (masked ham ~22, no dialogue containment) are NOT twins either:
    # the invariant must not paper over two genuinely different drawings.
    imgs = {"a.jpg": _hramp(), "b.jpg": _vramp(),
            "p000090.jpg": _fiximg("p000090.jpg"),
            "p000095.jpg": _fiximg("p000095.jpg")}
    plan = {"timeline": [
        {"segment_id": "g1", "tts_text": "x",
         "cuts": [{"file": "a.jpg", "start": 0.0, "dur": 3.0}]},
        {"segment_id": "g2", "tts_text": "y",
         "cuts": [{"file": "b.jpg", "start": 0.0, "dur": 3.0}]},
        {"segment_id": "g3", "tts_text": "z",
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 3.0}]},
        {"segment_id": "g4", "tts_text": "w",
         "cuts": [{"file": "p000095.jpg", "start": 0.0, "dur": 3.0}]},
    ], "scene_dims": {}}
    out, folds = rp.enforce_shown_twin_invariant(
        plan, lambda f: imgs.get(f),
        get_raw_boxes=lambda f: _FIX_BOXES.get(f, []),
        get_ocr=lambda f: _FIX_OCR.get(f, ""))
    assert folds == []
    assert [c["file"] for it in out["timeline"] for c in it["cuts"]] == \
        ["a.jpg", "b.jpg", "p000090.jpg", "p000095.jpg"]


def test_dedup_invariant_transitive_three_in_a_row():
    # A/B/C are three CONSECUTIVE shown panels, each a twin of its neighbour,
    # richness strictly increasing A < B < C (by raw area) with window=1 so
    # each comparison only ever sees its IMMEDIATE predecessor: B folds into A
    # first (alias a->b), then C's comparison hits b.jpg -- already aliased --
    # so it folds ONTO that alias (alias b->c), producing a genuine two-hop
    # chain a->b->c. This pins _resolve's while-loop: a naive single-lookup
    # alias would leave A's cut pointing at "b.jpg" instead of walking on to
    # the final survivor, so not everything would fold to ONE file.
    imgs = {"a.jpg": _hramp(w=300, h=400), "b.jpg": _hramp(w=320, h=400),
            "c.jpg": _hramp(w=340, h=400)}
    plan = {"timeline": [
        {"segment_id": "g1", "tts_text": "x",
         "cuts": [{"file": "a.jpg", "start": 0.0, "dur": 3.0}]},
        {"segment_id": "g2", "tts_text": "y",
         "cuts": [{"file": "b.jpg", "start": 0.0, "dur": 3.0}]},
        {"segment_id": "g3", "tts_text": "z",
         "cuts": [{"file": "c.jpg", "start": 0.0, "dur": 3.0}]},
    ], "scene_dims": {}}
    out, folds = rp.enforce_shown_twin_invariant(
        plan, lambda f: imgs.get(f), window=1)
    assert {(lo, hi) for _, lo, hi, *_ in folds} == \
        {("a.jpg", "b.jpg"), ("b.jpg", "c.jpg")}
    # every cut collapses to ONE survivor -- the alias chain fully resolved,
    # not just the one-hop "b.jpg" an unresolved chain would leave behind
    shown = [c["file"] for it in out["timeline"] for c in it["cuts"]]
    assert shown == ["c.jpg", "c.jpg", "c.jpg"]


def test_canonicalize_requires_raw_twins():
    # D4: the cleaner cropped p000095 to its eye band -> the SHOWN crops are
    # hash-twins of p000090 (measured ham <= 8) though the RAW panels are
    # distinct (masked ham ~22). Canonicalizing swaps p095's art away and holds
    # ONE image ~24s. The guard must require the RAWS to also be twins; when
    # they are not, take the re-crop path (show p095 whole) instead.
    raw90, raw95 = _fiximg("p000090.jpg"), _fiximg("p000095.jpg")
    crop95 = raw95[10:260]        # the manufactured eye-band crop
    shown = {"p000090.jpg": raw90, "p000095.jpg": crop95}
    raws = {"p000090.jpg": raw90, "p000095.jpg": raw95}
    # preconditions pinned from production forensics
    assert (rp._dhash8_bgr(raw90) ^ rp._dhash8_bgr(crop95)).bit_count() <= 8
    assert (rp._dhash8_bgr(raw90) ^ rp._dhash8_bgr(raw95)).bit_count() > 8
    cbs = {"g0019_p00": [{"file": "p000090.jpg", "start": 0.0, "dur": 12.0}],
           "g0020_p01": [{"file": "p000095.jpg", "start": 0.0, "dur": 12.0}]}
    recropped = []
    out, dropped, canon = rp.drop_cross_segment_near_identical_cuts(
        cbs, ["g0019_p00", "g0020_p01"], lambda f: shown.get(f),
        get_raw_img=lambda f: raws.get(f),
        get_raw_boxes=lambda f: _FIX_BOXES.get(f, []),
        on_recrop=lambda seg, f: recropped.append((seg, f)))
    assert canon == []                            # NEVER canonicalize raw-distinct
    assert dropped == []                          # sole cut: nothing dropped
    assert [c["file"] for c in out["g0020_p01"]] == ["p000095.jpg"]  # own art kept
    assert recropped == [("g0020_p01", "p000095.jpg")]   # re-crop path taken


def test_canonicalize_still_folds_true_raw_twins():
    # counterpart: when the raws REALLY are twins (a source-repeated panel), the
    # sole-cut canonicalize must keep working exactly as before the guard.
    imgs = {"eye0.jpg": _hramp(base=0), "eye1.jpg": _hramp(base=20)}
    cbs = {"g17": [{"file": "eye0.jpg", "start": 0.0, "dur": 6.0}],
           "g18": [{"file": "eye1.jpg", "start": 0.0, "dur": 6.0}]}
    recropped = []
    out, dropped, canon = rp.drop_cross_segment_near_identical_cuts(
        cbs, ["g17", "g18"], lambda f: imgs.get(f),
        get_raw_img=lambda f: imgs.get(f),
        on_recrop=lambda seg, f: recropped.append((seg, f)))
    assert canon == [("g18", "eye1.jpg", "eye0.jpg")]
    assert recropped == [] and dropped == []
    assert [c["file"] for c in out["g18"]] == ["eye0.jpg"]


# ---- V1/V2/V3 (2026-07 73-segment visual review): ken variety + husk policy --
# V1: unwatchable STATIC holds on a panel that legitimately OWNS its narration
# (22.8s g0020_p01 eye + cleaned-empty bubble) -> split_long_hold_cuts breaks
# the display into 2-3 sub-cuts of the SAME file with DIFFERENT ken regions.
# V2: perceptual echoes (shown-crop twins whose RAW panels are distinct — the
# p000090/p000095 eye-husk pair) -> ken_differentiate_echo_pairs re-aims the
# pair at different regions. V3: husk_recrop_decision re-crops blank-bubble-
# dominated crops that hold the screen long, unless the re-crop would twin a
# neighbour within the 3-cut window.

def _art_noise(h, w, seed=7, lo=40, hi=210):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, (h, w, 3), dtype=np.uint8)


def test_focal_point_prefers_face_then_art_and_avoids_dead_regions():
    # art (noise) in the lower-left quadrant, flat white elsewhere
    img = np.full((400, 400, 3), 255, np.uint8)
    img[200:400, 0:200] = _art_noise(200, 200)
    fx, fy, src = rp.focal_point_for_crop(img)
    assert src == "art" and fx < 0.5 and fy > 0.5
    # a known face center wins outright
    fx, fy, src = rp.focal_point_for_crop(img, face_center=(0.9, 0.1))
    assert (fx, fy, src) == (0.9, 0.1, "face")
    # art-looking texture INSIDE a dead (blanked-bubble/word) box is ignored
    img2 = img.copy()
    img2[0:150, 200:400] = _art_noise(150, 200, seed=9)
    fx2, fy2, src2 = rp.focal_point_for_crop(
        img2, dead_boxes=[(200, 0, 400, 150)])
    assert src2 == "art" and fx2 < 0.5 and fy2 > 0.5
    # unreadable / fully flat crop falls back to upper-middle
    assert rp.focal_point_for_crop(None) == (0.5, 0.4, "default")
    flat = np.full((300, 300, 3), 255, np.uint8)
    assert rp.focal_point_for_crop(flat) == (0.5, 0.4, "default")


def test_ken_variety_motions_valid_schema_and_distinct_regions():
    # exactly the schema Cut.tsx consumes; zooms inside [1.0, 1.35]; the 2-3
    # sub-moves must differ from each other even at a dead-center focal
    for n in (2, 3):
        for fx, fy in ((0.7, 0.3), (0.5, 0.5)):
            motions = rp.ken_variety_motions(n, fx, fy)
            assert len(motions) == n
            sigs = set()
            for m in motions:
                assert m["mode"] == "kenburns"
                assert 0.0 < m["strength"] <= 1.0
                assert m["ease"] in ("ease_in", "ease_out", "linear",
                                     "ease_in_out")
                for k in ("start_bias", "end_bias"):
                    assert -1.0 <= m[k]["x"] <= 1.0
                    assert -1.0 <= m[k]["y"] <= 1.0
                assert 1.0 <= m["zoom"]["start"] <= 1.35
                assert 1.0 <= m["zoom"]["end"] <= 1.35
                assert 0.0 <= m["focus_y"] <= 1.0     # tall-branch variety
                sigs.add((m["zoom"]["start"], m["zoom"]["end"],
                          m["start_bias"]["x"], m["start_bias"]["y"],
                          m["end_bias"]["x"], m["end_bias"]["y"]))
            assert len(sigs) == n                     # pairwise distinct


def test_split_long_hold_20s_single_cut_into_three_ken_varied_subcuts():
    plan = {"scene_dims": {"a.jpg": {"w": 795, "h": 832, "doc": False,
                                     "sys": True}},   # pixel-sys must NOT skip
            "timeline": [{"segment_id": "g0020_p01", "duration_sec": 20.0,
                          "cuts": [{"file": "a.jpg", "start": 0.0,
                                    "dur": 20.0,
                                    "motion": {"mode": "tilt_down"}}]}]}
    out, logs = rp.split_long_hold_cuts(
        plan, max_hold_sec=10.0, focal_for_file=lambda f: (0.7, 0.3, "art"))
    cuts = out["timeline"][0]["cuts"]
    assert logs == [("g0020_p01", "a.jpg", 20.0, 3, "art")]
    assert len(cuts) == 3 and all(c["file"] == "a.jpg" for c in cuts)
    assert all(c.get("ken_variety") is True for c in cuts)
    # durations sum EXACTLY to the original; starts contiguous; no flash cuts
    assert abs(sum(c["dur"] for c in cuts) - 20.0) < 1e-9
    assert [c["start"] for c in cuts] == [0.0, 7.0, 14.0]
    assert all(c["dur"] >= 2.0 for c in cuts)
    # valid, DISTINCT motion dicts (the whole point)
    regions = [c["motion"]["ken_region"] for c in cuts]
    assert regions == ["wide", "tight", "pull"]
    assert len({json.dumps(c["motion"], sort_keys=True) for c in cuts}) == 3


def _effective_zoom_cap(camera):
    """zoomCap reimplemented from remotion/src/plan.ts constants
    (MAX_ZOOM_CAP=1.35, TEXT_ZOOM_CAP=1.06): hard = min(MAX_ZOOM_CAP,
    camera.max_zoom ?? MAX_ZOOM_CAP); avoid_text_zoom -> min(hard, TEXT)."""
    MAX_ZOOM_CAP, TEXT_ZOOM_CAP = 1.35, 1.06
    cam = camera or {}
    mz = cam.get("max_zoom")
    hard = min(MAX_ZOOM_CAP, MAX_ZOOM_CAP if mz is None else float(mz))
    return min(hard, TEXT_ZOOM_CAP) if cam.get("avoid_text_zoom") else hard


def test_ken_split_rewrites_item_camera_so_zooms_survive_effective_cap():
    # PRODUCTION-shaped item: the planner mirrors camera.max_zoom off the
    # item's ORIGINAL motion end (1.08 here) and avoid_text_zoom defaults
    # True — Cut.tsx's zoomCap (item-level camera, Shot.tsx) would clamp the
    # 1.18->1.32 tight push to a STATIC 1.06 hold and the 22.8s defect would
    # ship invisibly. The split must rewrite item['camera'] too.
    def _plan(blanked):
        return {"scene_dims": {"a.jpg": {"w": 795, "h": 832, "doc": False,
                                         "sys": True,      # pixel-level YOLO
                                         "blanked": blanked}},
                "timeline": [{"segment_id": "g0020_p01", "duration_sec": 20.0,
                              "camera": {"avoid_text_zoom": True,
                                         "max_zoom": 1.08},
                              "cuts": [{"file": "a.jpg", "start": 0.0,
                                        "dur": 20.0}]}]}

    out, logs = rp.split_long_hold_cuts(
        _plan(True), max_hold_sec=10.0,
        focal_for_file=lambda f: (0.7, 0.3, "art"))
    item = out["timeline"][0]
    assert len(item["cuts"]) == 3 and len(logs) == 1
    cam = item["camera"]
    # blanked non-doc panel: nothing readable left to protect -> cleared;
    # the pixel-level sys:True must NOT keep the 1.06 text clamp
    assert cam["avoid_text_zoom"] is False
    assert cam["max_zoom"] >= 1.32
    cap = _effective_zoom_cap(cam)
    assert cap >= 1.32
    for c in item["cuts"]:
        z = c["motion"]["zoom"]
        # every sub-cut zoom range survives the EFFECTIVE cap un-flattened:
        # both endpoints inside [1.0, cap], so clamp() is the identity
        assert 1.0 <= z["start"] <= cap and 1.0 <= z["end"] <= cap
        assert abs(z["end"] - z["start"]) > 0.01
    # pre-fix camera for contrast: effective cap 1.06 flattened the tight
    # push (start 1.18 and end 1.32 both clamp to 1.06 -> zero travel)
    pre = _effective_zoom_cap({"avoid_text_zoom": True, "max_zoom": 1.08})
    tight = item["cuts"][1]["motion"]["zoom"]
    assert pre == 1.06
    assert min(tight["start"], pre) == min(tight["end"], pre) == pre

    # text NOT blanked: max_zoom still raised, but the text clamp is KEPT
    out2, _ = rp.split_long_hold_cuts(
        _plan(False), max_hold_sec=10.0,
        focal_for_file=lambda f: (0.7, 0.3, "art"))
    cam2 = out2["timeline"][0]["camera"]
    assert cam2["avoid_text_zoom"] is True and cam2["max_zoom"] >= 1.32

    # no camera on the item (nothing clamps below MAX_ZOOM_CAP): untouched
    plan3 = _plan(True)
    del plan3["timeline"][0]["camera"]
    out3, _ = rp.split_long_hold_cuts(
        plan3, max_hold_sec=10.0, focal_for_file=lambda f: (0.7, 0.3, "art"))
    assert "camera" not in out3["timeline"][0]


def test_split_long_hold_two_way_and_untouched_cases():
    dims = {"a.jpg": {"w": 795, "h": 832},
            "wide.jpg": {"w": 1300, "h": 900},        # w/h>=1.3: cover drift
            "tall.jpg": {"w": 400, "h": 900},         # h/w>=2.0: scroll
            "doc.jpg": {"w": 795, "h": 832, "doc": True},
            "sys.jpg": {"w": 795, "h": 832}}

    def _one(f, dur):
        return {"scene_dims": dims,
                "timeline": [{"segment_id": "s", "duration_sec": dur,
                              "cuts": [{"file": f, "start": 0.0,
                                        "dur": dur}]}]}
    foc = lambda f: (0.5, 0.5, "default")
    # 12s -> 2 sub-cuts (<= 1.5x cap)
    out, logs = rp.split_long_hold_cuts(_one("a.jpg", 12.0),
                                        max_hold_sec=10.0,
                                        focal_for_file=foc)
    assert len(out["timeline"][0]["cuts"]) == 2 and logs[0][3] == 2
    assert abs(sum(c["dur"] for c in out["timeline"][0]["cuts"]) - 12.0) < 1e-9
    # under the cap: untouched
    out, logs = rp.split_long_hold_cuts(_one("a.jpg", 9.0),
                                        max_hold_sec=10.0,
                                        focal_for_file=foc)
    assert len(out["timeline"][0]["cuts"]) == 1 and logs == []
    # wide / tall (renderer ignores the motion dict — built-in drift) and doc
    # (text needs stillness): untouched
    for f in ("wide.jpg", "tall.jpg", "doc.jpg"):
        out, logs = rp.split_long_hold_cuts(_one(f, 20.0),
                                            max_hold_sec=10.0,
                                            focal_for_file=foc)
        assert len(out["timeline"][0]["cuts"]) == 1 and logs == []
    # stamped system cards: exempt from everything as usual
    out, logs = rp.split_long_hold_cuts(_one("sys.jpg", 20.0),
                                        max_hold_sec=10.0,
                                        focal_for_file=foc,
                                        skip_files={"sys.jpg"})
    assert len(out["timeline"][0]["cuts"]) == 1 and logs == []
    # a tiny cap can never manufacture a flash cut: 3.5s over a 3.0 cap would
    # split into <2s sub-cuts -> stays whole
    out, logs = rp.split_long_hold_cuts(_one("a.jpg", 3.5),
                                        max_hold_sec=3.0,
                                        focal_for_file=foc)
    assert len(out["timeline"][0]["cuts"]) == 1 and logs == []


def test_merge_passes_never_collapse_ken_variety_subcuts():
    # the same-image merge would fold the split straight back into one cut —
    # the guards on _collapse/_item_sole_image must leave V1 sub-cuts alone
    plan = {"scene_dims": {"a.jpg": {"w": 795, "h": 832}},
            "timeline": [{"segment_id": "s", "duration_sec": 20.0,
                          "cuts": [{"file": "a.jpg", "start": 0.0,
                                    "dur": 20.0}]}]}
    out, _ = rp.split_long_hold_cuts(
        plan, max_hold_sec=10.0, focal_for_file=lambda f: (0.5, 0.4, "art"))
    before = [json.dumps(c, sort_keys=True)
              for c in out["timeline"][0]["cuts"]]
    merged = rp.merge_consecutive_same_image_cuts(out)
    after = [json.dumps(c, sort_keys=True)
             for c in merged["timeline"][0]["cuts"]]
    assert after == before


def test_echo_pair_p90_p95_gets_distinct_ken_regions():
    # the REAL eye-husk pair: p000095's SHIPPED crop hash-twinned p000090
    # (production dhash 3) while the raws are distinct (masked ham 22). The
    # fixture JPGs are downscaled q65 re-encodes, so the shipped crop is
    # reconstructed from p000090's art region; the RAW predicate runs on the
    # real fixture panels.
    p90 = _fiximg("p000090.jpg")
    p95 = _fiximg("p000095.jpg")
    crop95 = p90[5:225, 5:395]
    plan = {"scene_dims": {"p000090.jpg": {"w": 795, "h": 832,
                                           "blanked": True},
                           "p000095.jpg": {"w": 795, "h": 832}},
            "timeline": [
                {"segment_id": "g0019_p00", "tts_text": "a",
                 "camera": {"avoid_text_zoom": True, "max_zoom": 1.0},
                 "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 5.0}]},
                {"segment_id": "g0020_p01", "tts_text": "b",
                 "camera": {"avoid_text_zoom": True, "max_zoom": 1.0},
                 "cuts": [{"file": "p000095.jpg", "start": 0.0, "dur": 6.0}]},
            ]}
    clean = {"p000090.jpg": p90, "p000095.jpg": crop95}
    raw = {"p000090.jpg": p90, "p000095.jpg": p95}
    out, logs = rp.ken_differentiate_echo_pairs(
        plan, lambda f: clean.get(f), lambda f: [],
        lambda f: raw.get(f), lambda f: _FIX_BOXES.get(f, []),
        focal_for_file=lambda f: (0.5, 0.4, "art"))
    assert len(logs) == 1
    seg_i, fi, seg_j, fj, sham, rham = logs[0]
    assert (fi, fj) == ("p000090.jpg", "p000095.jpg")
    assert sham <= 8 < rham
    c0 = out["timeline"][0]["cuts"][0]
    c1 = out["timeline"][1]["cuts"][0]
    # both re-aimed, visibly different regions, files/durs untouched
    assert c0["echo_differentiated"] and c1["echo_differentiated"]
    assert c0["motion"]["ken_region"] == "wide"
    assert c1["motion"]["ken_region"] == "tight"
    assert c0["motion"] != c1["motion"]
    assert (c0["file"], c0["dur"]) == ("p000090.jpg", 5.0)
    assert (c1["file"], c1["dur"]) == ("p000095.jpg", 6.0)
    # ITEM cameras rewritten so the re-aimed zooms survive Cut.tsx's zoomCap:
    # each member's cap covers its own motion; the text clamp clears only on
    # the blanked panel (p000090), stays on the un-blanked one (p000095)
    cam0 = out["timeline"][0]["camera"]
    cam1 = out["timeline"][1]["camera"]
    assert cam0["max_zoom"] >= max(c0["motion"]["zoom"].values())
    assert cam1["max_zoom"] >= 1.32
    assert cam0["avoid_text_zoom"] is False
    assert cam1["avoid_text_zoom"] is True


def test_echo_differentiation_ignores_distinct_panels_and_raw_twins():
    p90 = _fiximg("p000090.jpg")
    p54 = _fiximg("p000054.jpg")
    # genuinely different panels: never an echo
    plan = {"scene_dims": {}, "timeline": [
        {"segment_id": "a", "cuts": [{"file": "p000054.jpg",
                                      "start": 0.0, "dur": 4.0}]},
        {"segment_id": "b", "cuts": [{"file": "p000090.jpg",
                                      "start": 0.0, "dur": 4.0}]}]}
    imgs = {"p000054.jpg": p54, "p000090.jpg": p90}
    out, logs = rp.ken_differentiate_echo_pairs(
        plan, lambda f: imgs.get(f), lambda f: [],
        lambda f: imgs.get(f), lambda f: [],
        focal_for_file=lambda f: (0.5, 0.4, "art"))
    assert logs == []
    # crop twins whose RAWS are twins too: the dedup invariant's domain (a
    # genuine duplicate), NOT an echo — leave it to dup_shown/fold
    crop_a, crop_b = p90, p90[5:225, 5:395]
    plan2 = {"scene_dims": {}, "timeline": [
        {"segment_id": "a", "cuts": [{"file": "x.jpg",
                                      "start": 0.0, "dur": 4.0}]},
        {"segment_id": "b", "cuts": [{"file": "y.jpg",
                                      "start": 0.0, "dur": 4.0}]}]}
    clean2 = {"x.jpg": crop_a, "y.jpg": crop_b}
    raw2 = {"x.jpg": p90, "y.jpg": p90}
    out2, logs2 = rp.ken_differentiate_echo_pairs(
        plan2, lambda f: clean2.get(f), lambda f: [],
        lambda f: raw2.get(f), lambda f: [],
        focal_for_file=lambda f: (0.5, 0.4, "art"))
    assert logs2 == []


def _husk_panel(h=800, w=600, bubble_frac=0.45):
    """Flat-white blanked-bubble band on top (bubble_frac of the area), real
    art below — the post-clean husk shape (p000095's lower-bubble mirrored)."""
    img = np.full((h, w, 3), 255, np.uint8)
    cut = int(h * bubble_frac)
    img[cut:h] = _art_noise(h - cut, w, seed=3, lo=30, hi=220)
    return img, (0, 0, w, cut)


def test_husk_recrop_drops_dead_bubble_band_for_long_display():
    img, box = _husk_panel()
    out, info = rp.husk_recrop_decision(
        img, [box], display_sec=6.0, max_hold_sec=10.0, neighbor_crops=[])
    assert info["blank_frac"] > 0.40
    assert info["husk_recropped"] and info["band"] is not None
    a, b = info["band"]
    # the surviving crop is the art band: strictly smaller, excludes the
    # bubble region (band starts at/after the bubble's bottom, minus pad)
    assert out.shape[0] < img.shape[0]
    assert a >= box[3] - 40
    # log contract: caller prints one line per husk re-crop off this info


def test_husk_recrop_refuses_when_recrop_would_twin_a_neighbor():
    img, box = _husk_panel()
    band_img, info0 = rp.husk_recrop_decision(
        img, [box], display_sec=6.0, max_hold_sec=10.0, neighbor_crops=[])
    assert info0["husk_recropped"]
    # neighbour within the 3-cut window already shows (essentially) that band
    out, info = rp.husk_recrop_decision(
        img, [box], display_sec=6.0, max_hold_sec=10.0,
        neighbor_crops=[("p000090.jpg", band_img)])
    assert not info["husk_recropped"]
    assert info["refused_twin"] is not None
    nf, ham = info["refused_twin"]
    assert nf == "p000090.jpg" and ham <= 8
    assert out.shape == img.shape                     # full crop kept


def test_husk_recrop_duration_and_coverage_gates():
    img, box = _husk_panel()
    # short display: watchable as-is, keep the full crop
    out, info = rp.husk_recrop_decision(
        img, [box], display_sec=3.0, max_hold_sec=10.0, neighbor_crops=[])
    assert not info["husk_recropped"] and out.shape == img.shape
    # bubbles under the 40% coverage gate: keep the full crop
    img2, box2 = _husk_panel(bubble_frac=0.25)
    out2, info2 = rp.husk_recrop_decision(
        img2, [box2], display_sec=6.0, max_hold_sec=10.0, neighbor_crops=[])
    assert not info2["husk_recropped"] and out2.shape == img2.shape
    # no boxes at all: no-op
    out3, info3 = rp.husk_recrop_decision(
        img, [], display_sec=6.0, max_hold_sec=10.0, neighbor_crops=[])
    assert not info3["husk_recropped"] and info3["blank_frac"] == 0.0


def test_cross_segment_canonicalize_is_hold_cap_aware():
    # nano ch2 job 53 (2026-07-17): canonicalizing a run of late-chapter echo
    # panels re-seated the SAME earlier file across segments for 11.3s — past
    # the 10s hold cap, a BLOCKING long_hold/panel_substituted. Past the cap
    # the cut keeps its OWN art (raw twins: the repeat reads as an artist
    # echo, styled by the ken echo pass — WARN, never a block).
    imgs = {"eye0.jpg": _hramp(base=0), "eye1.jpg": _hramp(base=20),
            "eye2.jpg": _hramp(base=10)}
    cbs = {"g17": [{"file": "eye0.jpg", "start": 0.0, "dur": 6.0}],
           "g18": [{"file": "eye1.jpg", "start": 0.0, "dur": 3.0}],
           "g19": [{"file": "eye2.jpg", "start": 0.0, "dur": 3.0}]}
    out, dropped, canon = rp.drop_cross_segment_near_identical_cuts(
        cbs, ["g17", "g18", "g19"], lambda f: imgs.get(f), max_hold_sec=10.0)
    assert dropped == []
    # g18 canonicalizes (6+3 = 9s <= 10s); g19 would make it 12s -> keeps own
    assert canon == [("g18", "eye1.jpg", "eye0.jpg")]
    assert [c["file"] for c in out["g18"]] == ["eye0.jpg"]
    assert [c["file"] for c in out["g19"]] == ["eye2.jpg"]   # own art kept
    # without a cap the whole run still canonicalizes (legacy behavior)
    cbs2 = {"g17": [{"file": "eye0.jpg", "start": 0.0, "dur": 6.0}],
            "g18": [{"file": "eye1.jpg", "start": 0.0, "dur": 3.0}],
            "g19": [{"file": "eye2.jpg", "start": 0.0, "dur": 3.0}]}
    out2, _d2, canon2 = rp.drop_cross_segment_near_identical_cuts(
        cbs2, ["g17", "g18", "g19"], lambda f: imgs.get(f))
    assert len(canon2) == 2


def test_twin_invariant_fold_yields_to_the_hold_cap():
    # nano ch2 jobs 50-56: the final twin fold re-seated ONE survivor across
    # an 11.3s run (blocking long_hold + panel_substituted, rebuilt
    # deterministically on every re-prep). When fold and cap conflict, the
    # CAP wins: the cut keeps its own art (far-apart twins are styleable).
    imgs = {"a.jpg": _hramp(base=0), "b.jpg": _hramp(base=20)}
    plan = {"timeline": [
        {"segment_id": "g18", "cuts": [{"file": "a.jpg", "dur": 5.1},
                                       {"file": "a.jpg", "dur": 6.2}]},
        {"segment_id": "g19", "cuts": [{"file": "b.jpg", "dur": 11.3}]},
    ]}
    out, folds = rp.enforce_shown_twin_invariant(
        plan, lambda f: imgs.get(f), max_hold_sec=10.0)
    assert folds and folds[0][1] == "b.jpg"     # twin detected (fold intended)
    g19 = out["timeline"][1]["cuts"]
    assert [c["file"] for c in g19] == ["b.jpg"]   # ...but own art kept (cap)
    # without the cap, legacy behavior folds it
    plan2 = {"timeline": [
        {"segment_id": "g18", "cuts": [{"file": "a.jpg", "dur": 5.1},
                                       {"file": "a.jpg", "dur": 6.2}]},
        {"segment_id": "g19", "cuts": [{"file": "b.jpg", "dur": 11.3}]},
    ]}
    out2, _f2 = rp.enforce_shown_twin_invariant(plan2, lambda f: imgs.get(f))
    assert [c["file"] for c in out2["timeline"][1]["cuts"]] == ["a.jpg"]
    # a SHORT fold under the cap still folds (the invariant holds when safe)
    plan3 = {"timeline": [
        {"segment_id": "g18", "cuts": [{"file": "a.jpg", "dur": 4.0}]},
        {"segment_id": "g19", "cuts": [{"file": "b.jpg", "dur": 3.0}]},
    ]}
    out3, _f3 = rp.enforce_shown_twin_invariant(
        plan3, lambda f: imgs.get(f), max_hold_sec=10.0)
    assert [c["file"] for c in out3["timeline"][1]["cuts"]] == ["a.jpg"]


def test_twin_fold_cap_blocks_single_overcap_cut_and_nonadjacent_prior():
    # job 57: the folded cut was 11.3s ALONE (prior twin cuts were panels
    # away, so no run was being "extended") — the cap must gate on the
    # projected run including the single-cut case.
    imgs = {"a.jpg": _hramp(base=0), "b.jpg": _hramp(base=20),
            "x.jpg": _vramp()}
    plan = {"timeline": [
        {"segment_id": "g18", "cuts": [{"file": "a.jpg", "dur": 5.0}]},
        {"segment_id": "gx", "cuts": [{"file": "x.jpg", "dur": 4.0}]},
        {"segment_id": "g19", "cuts": [{"file": "b.jpg", "dur": 11.3}]},
    ]}
    out, folds = rp.enforce_shown_twin_invariant(
        plan, lambda f: imgs.get(f), max_hold_sec=10.0)
    assert folds and folds[0][1] == "b.jpg"
    assert [c["file"] for c in out["timeline"][2]["cuts"]] == ["b.jpg"]


def test_cap_kept_twin_pair_is_stamped_and_dup_shown_honors_it():
    imgs = {"a.jpg": _hramp(base=0), "b.jpg": _hramp(base=20),
            "x.jpg": _vramp()}
    plan = {"timeline": [
        {"segment_id": "g18", "cuts": [{"file": "a.jpg", "dur": 5.0}]},
        {"segment_id": "gx", "cuts": [{"file": "x.jpg", "dur": 4.0}]},
        {"segment_id": "g19", "cuts": [{"file": "b.jpg", "dur": 11.3}]},
    ]}
    out, _folds = rp.enforce_shown_twin_invariant(
        plan, lambda f: imgs.get(f), max_hold_sec=10.0)
    assert out.get("twin_cap_kept") == [["b.jpg", "a.jpg"]]

    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "prep_qa_capkept", Path(__file__).resolve().parents[1] / "tools" / "prep_qa.py")
    pq = importlib.util.module_from_spec(spec)
    sys.modules["prep_qa_capkept"] = pq
    spec.loader.exec_module(pq)
    cuts = [{"file": "a.jpg", "segment_id": "g18"},
            {"file": "x.jpg", "segment_id": "gx"},
            {"file": "b.jpg", "segment_id": "g19"}]
    raw = lambda f: imgs.get(f)
    boxes = lambda f: ()
    ocr = lambda f: ""
    # unstamped: the twin pair blocks (the tripwire's whole job)
    assert [f["code"] for f in pq.dup_shown_flags(
        cuts, raw, boxes, ocr)] == ["dup_shown"]
    # stamped: the documented cap exception is legal
    assert pq.dup_shown_flags(
        cuts, raw, boxes, ocr,
        cap_kept_pairs=out["twin_cap_kept"]) == []


def test_dedup_invariant_cap_projects_the_whole_run_not_just_the_folded_cut():
    # nano ch1 job 147: p000094 (g0020_p02, 9.5s) is a hash-twin of p000099
    # (g0021_p03, 11.0s). Folding p000094 -> p000099 passed the old cap check
    # (9.5s <= 10s counts only the folded cut) but the survivor's OWN cut follows
    # in the same run -> p000099 stands in for 20.5s -> long_hold. The cap must
    # see the whole projected run; when it breaches, keep own art.
    imgs = {"a.jpg": _hramp(base=0), "b.jpg": _hramp(base=15)}
    plan = {"timeline": [
        {"segment_id": "g0020_p02", "tts_text": "An eye peers through.",
         "cuts": [{"file": "a.jpg", "start": 0.0, "dur": 9.5}]},
        {"segment_id": "g0021_p03", "tts_text": "The confusion deepens.",
         "cuts": [{"file": "b.jpg", "start": 0.0, "dur": 11.0}]},
    ], "scene_dims": {}}
    out, folds = rp.enforce_shown_twin_invariant(
        {"timeline": [dict(i, cuts=[dict(c) for c in i["cuts"]])
                      for i in plan["timeline"]], "scene_dims": {}},
        lambda f: imgs.get(f), max_hold_sec=10.0)
    shown = [c["file"] for it in out["timeline"] for c in it["cuts"]]
    assert shown == ["a.jpg", "b.jpg"]           # fold refused: 20.5s run > cap
    assert len(folds) == 1                       # twin still DETECTED (contract)
    kept = out.get("twin_cap_kept", [])
    assert any(sorted(pair) == ["a.jpg", "b.jpg"] for pair in kept)
    # under a generous cap the same pair DOES fold (both cuts on the survivor)
    out2, folds2 = rp.enforce_shown_twin_invariant(plan, lambda f: imgs.get(f), max_hold_sec=30.0)
    assert len(folds2) == 1
    assert len({c["file"] for it in out2["timeline"] for c in it["cuts"]}) == 1


# ---- consecutive identical frames are never shipped (2026-08-18) ------------
# nano ch1 g0022_p05 ended on the eye and g0023_p06 showed the SAME drawing
# again (the artist copy-pasted the panel; everything between folded away).
# The twin fold wanted to merge them but the hold cap refused (29.6s), and
# ken_differentiate_echo_pairs skips raw twins by design -> the viewer saw the
# identical frame twice. Final backstop: re-frame the LATER cut as a push-in.

def _twin_plan():
    return {"timeline": [
        {"segment_id": "g22", "cuts": [{"file": "a.jpg", "dur": 7.5},
                                       {"file": "b.jpg", "dur": 7.5}]},
        {"segment_id": "g23", "cuts": [{"file": "c.jpg", "dur": 22.2}]},
    ], "scene_dims": {"b.jpg": {"w": 800, "h": 400}, "c.jpg": {"w": 782, "h": 439}},
        "twin_cap_kept": [["b.jpg", "c.jpg"]]}


def test_reframe_capped_twins_pushes_in_on_the_repeated_frame():
    imgs = {"a.jpg": _vramp(800, 600), "b.jpg": _hramp(800, 600, base=0),
            "c.jpg": _hramp(800, 600, base=1)}                       # b ~ c twins
    written = {}

    def _write(f, box):
        written[f] = box
        return f                                   # in place: identity unchanged

    out, logs = rp.reframe_capped_twins(
        _twin_plan(), lambda f: imgs.get(f), lambda f: (),
        focal_for_file=lambda f: (0.5, 0.4, "art"), write_variant=_write)
    shown = [c["file"] for it in out["timeline"] for c in it["cuts"]]
    assert shown == ["a.jpg", "b.jpg", "c.jpg"]            # SAME names: panel identity intact
    assert len(logs) == 1 and logs[0][1] == "c.jpg"
    assert out["scene_dims"]["c.jpg"]["w"] < 800           # dims follow the crop
    x0, y0, x1, y1 = written["c.jpg"]                       # a real crop, inside bounds
    assert 0 <= x0 < x1 <= imgs["c.jpg"].shape[1] and 0 <= y0 < y1 <= imgs["c.jpg"].shape[0]
    assert (x1 - x0) < imgs["c.jpg"].shape[1]               # tighter than the original


def test_reframe_leaves_distinct_neighbours_and_same_file_ken_splits_alone():
    imgs = {"a.jpg": _vramp(800, 600), "b.jpg": _hramp(800, 600, base=0),
            "c.jpg": _hramp(800, 600, base=1)}
    # distinct neighbours -> untouched
    plan = {"timeline": [{"segment_id": "g1", "cuts": [{"file": "a.jpg", "dur": 5.0},
                                                       {"file": "b.jpg", "dur": 5.0}]}],
            "scene_dims": {}}
    out, logs = rp.reframe_capped_twins(plan, lambda f: imgs.get(f), lambda f: (),
                                        focal_for_file=lambda f: (0.5, 0.5, "art"),
                                        write_variant=lambda f, b: "NEVER.jpg")
    assert logs == [] and [c["file"] for c in out["timeline"][0]["cuts"]] == ["a.jpg", "b.jpg"]
    # same file twice = a deliberate ken-variety split -> untouched
    plan2 = {"timeline": [{"segment_id": "g2", "cuts": [{"file": "b.jpg", "dur": 5.0},
                                                        {"file": "b.jpg", "dur": 5.0}]}],
             "scene_dims": {}}
    out2, logs2 = rp.reframe_capped_twins(plan2, lambda f: imgs.get(f), lambda f: (),
                                          focal_for_file=lambda f: (0.5, 0.5, "art"),
                                          write_variant=lambda f, b: "NEVER.jpg")
    assert logs2 == [] and [c["file"] for c in out2["timeline"][0]["cuts"]] == ["b.jpg", "b.jpg"]


def test_reframe_skips_system_cards_and_tiny_frames():
    imgs = {"b.jpg": _hramp(800, 600, base=0), "c.jpg": _hramp(800, 600, base=1)}
    p = _twin_plan(); p["timeline"][0]["cuts"] = [{"file": "b.jpg", "dur": 5.0}]
    out, logs = rp.reframe_capped_twins(p, lambda f: imgs.get(f), lambda f: (),
                                        focal_for_file=lambda f: (0.5, 0.5, "art"),
                                        write_variant=lambda f, b: f + "X",
                                        skip_files={"c.jpg"})
    assert logs == []                                        # system card exempt
    small = {"b.jpg": _hramp(w=200, h=200, base=0), "c.jpg": _hramp(w=200, h=200, base=1)}
    out2, logs2 = rp.reframe_capped_twins(p, lambda f: small.get(f), lambda f: (),
                                          focal_for_file=lambda f: (0.5, 0.5, "art"),
                                          write_variant=lambda f, b: f)
    assert logs2 == []                                       # too small to crop further


def test_reframe_skips_a_file_shown_more_than_once():
    # in-place re-framing would retro-change the file's OTHER appearance
    imgs = {"b.jpg": _hramp(800, 600, base=0), "c.jpg": _hramp(800, 600, base=1)}
    plan = {"timeline": [
        {"segment_id": "g1", "cuts": [{"file": "b.jpg", "dur": 4.0}]},
        {"segment_id": "g2", "cuts": [{"file": "c.jpg", "dur": 4.0}]},   # twin of b...
        {"segment_id": "g3", "cuts": [{"file": "c.jpg", "dur": 4.0}]}],  # ...but shown twice
        "scene_dims": {}}
    out, logs = rp.reframe_capped_twins(plan, lambda f: imgs.get(f), lambda f: (),
                                        focal_for_file=lambda f: (0.5, 0.5, "art"),
                                        write_variant=lambda f, b: f)
    assert logs == []


# ---- 2026-08-18: verdict caches must key on PIXELS, not on the filename -----
# scenes_clean/<f> is a derived rendition (art_only crop, and since 75218d4 an
# in-place push-in re-frame), so the same name can hold a different picture
# between runs — a filename-keyed cache then reuses a judgment of an image that
# no longer exists.

def test_frame_digest_changes_with_the_pixels_not_the_name(tmp_path):
    import numpy as np, cv2
    a = tmp_path / "p1.jpg"
    cv2.imwrite(str(a), _hramp(400, 300, base=0))
    d1 = rp.frame_digest(str(a))
    cv2.imwrite(str(a), _hramp(400, 300, base=60))      # same name, new picture
    d2 = rp.frame_digest(str(a))
    assert d1 and d2 and d1 != d2
    assert rp.frame_digest(str(tmp_path / "missing.jpg")) == ""


def test_cut_judge_cache_is_invalidated_when_the_frame_changes(tmp_path):
    import json, cv2
    clean = tmp_path / "scenes_clean"; clean.mkdir()
    f = "p1.jpg"
    cv2.imwrite(str(clean / f), _hramp(400, 300, base=0))
    cache_path = tmp_path / ".cut_judge_cache.json"
    # a cached "junk" verdict recorded against the OLD pixels
    (cache_path).write_text(json.dumps({
        f: {"keep": False, "reason": "blank", "digest": "stale-digest"}}))
    junk = rp.judge_cut_visuals([f], str(clean), reuse=True,
                                cache_path=str(cache_path))
    assert junk == {}          # digest mismatch -> the stale verdict is ignored
    # a cached verdict recorded against the CURRENT pixels is reused
    (cache_path).write_text(json.dumps({
        f: {"keep": False, "reason": "blank",
            "digest": rp.frame_digest(str(clean / f))}}))
    junk2 = rp.judge_cut_visuals([f], str(clean), reuse=True,
                                 cache_path=str(cache_path))
    assert junk2 == {f: "blank"}
