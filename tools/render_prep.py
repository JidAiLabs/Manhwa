#!/usr/bin/env python3
"""
render_prep.py — prepare a chapter's plan + scene images for the renderer.

Sits between `planned` (render.plan.json) and the renderer (Remotion/Blender),
fixing the three defects reported on the first ch1 watch-through:

1. CROSS-CHUNK SEAM DUPLICATES: a panel spanning a chunk boundary gets
   detected twice (full panel at chunk N's bottom + fragment at chunk N+1's
   top — the p000015/p000016 pair). Same-chunk dedupe can't see across the
   seam; here we compare cuts in GLOBAL page coordinates
   (chunk_global_y0 + box_px_xyxy from manifest.scenes.json) and drop the
   contained fragment, redistributing its time across the shot.
2. BUBBLE TEXT: the narration voices the dialogue, so the printed bubbles are
   removed from the SHOWN scenes only — ogkalu speech-bubble boxes -> an
   oval-aware mask (white AND black bubbles; flood from the box centre, the
   outline ring is dilated in) -> cv2.inpaint -> scenes_clean/.
3. BAKED PAGE MARGINS: uniform light borders around the art are trimmed when
   writing the clean copies, and per-scene dims are recorded so the renderer
   can show wide panels full-bleed instead of contained-with-margins.

Outputs: <episode>/scenes_clean/*.jpg + render.plan.clean.json
(originals are never touched — vision/Gemini/resume still see the real art).

Run:
  .eval_venv/bin/python tools/render_prep.py \
      --plan ongoing/<series>/<ch>/render.plan.json \
      --scenes-manifest ongoing/<series>/<ch>/manifest.scenes.json \
      --episode-dir ongoing/<series>/<ch>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. cross-chunk contained-fragment filter (pure)
# ---------------------------------------------------------------------------

def drop_contained_duplicate_cuts(
    cuts: Sequence[Dict[str, Any]],
    geom_by_file: Dict[str, Dict[str, float]],
    *,
    contain_frac: float = 0.8,
    protect: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Drop cuts whose GLOBAL box is >= contain_frac inside another cut's box.

    geom_by_file: {file: {x1,y1,x2,y2}} in global page pixels. The smaller box
    is the fragment; the complete panel survives. Freed time is redistributed
    proportionally so the shot window stays fully covered. *protect* files (a
    system card whose text IS the on-screen beat) are never dropped.
    """
    prot = protect or set()

    def area(g: Dict[str, float]) -> float:
        return max(0.0, g["x2"] - g["x1"]) * max(0.0, g["y2"] - g["y1"])

    dropped: List[str] = []
    keep = list(cuts)
    for i, ci in enumerate(cuts):
        gi = geom_by_file.get(str(ci.get("file")))
        if not gi:
            continue
        for j, cj in enumerate(cuts):
            if i == j or cj["file"] in dropped or ci["file"] in dropped:
                continue
            gj = geom_by_file.get(str(cj.get("file")))
            if not gj:
                continue
            small, big = (gi, gj) if area(gi) <= area(gj) else (gj, gi)
            small_file = ci["file"] if small is gi else cj["file"]
            ix = max(0.0, min(small["x2"], big["x2"]) - max(small["x1"], big["x1"]))
            iy = max(0.0, min(small["y2"], big["y2"]) - max(small["y1"], big["y1"]))
            a = area(small)
            if a > 0 and (ix * iy) / a >= contain_frac:
                if small_file not in dropped and small_file not in prot:
                    dropped.append(small_file)

    return _redistribute(cuts, dropped), dropped


def _redistribute(
    cuts: Sequence[Dict[str, Any]],
    dropped: Sequence[str],
) -> List[Dict[str, Any]]:
    """Survivors keep their order; the dropped cuts' time is spread
    proportionally so the shot window stays fully covered."""
    survivors = [c for c in cuts if c["file"] not in dropped]
    if not survivors or not dropped:
        return list(cuts) if not dropped else survivors

    total = sum(float(c.get("dur") or 0.0) for c in cuts)
    surv_total = sum(float(c.get("dur") or 0.0) for c in survivors)
    scale = (total / surv_total) if surv_total > 0 else 1.0
    out: List[Dict[str, Any]] = []
    t = min(float(survivors[0].get("start") or 0.0),
            float(cuts[0].get("start") or 0.0))
    for c in survivors:
        d = round(float(c.get("dur") or 0.0) * scale, 4)
        out.append({**c, "start": round(t, 4), "dur": d})
        t += d
    return out


def multi_scale_contained(
    small_img: np.ndarray,
    big_img: np.ndarray,
    *,
    thresh: float = 0.86,
    max_dim: int = 400,
) -> bool:
    """True when *small_img* is (a possibly ZOOMED) region of *big_img*.

    Artists repeat a beat as a blow-up detail panel (the chibi-run +
    foot-zoom pair); same-scale template matching cannot see that — try a
    ladder of scales."""
    def gray(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im

    g1, g2 = gray(small_img), gray(big_img)
    if float(g1.std()) < 4 or float(g2.std()) < 4:
        return False  # featureless panel: zero-variance NCC is meaningless
    sb = min(1.0, max_dim / max(g2.shape[:2]))
    big = cv2.resize(g2, (max(1, int(g2.shape[1] * sb)),
                          max(1, int(g2.shape[0] * sb))))
    for s in (1.0, 0.85, 0.72, 0.6, 0.5, 0.42, 0.35):
        w = int(g1.shape[1] * sb * s)
        h = int(g1.shape[0] * sb * s)
        if w < 24 or h < 24 or h > big.shape[0] or w > big.shape[1]:
            continue
        t = cv2.resize(g1, (w, h))
        res = np.nan_to_num(cv2.matchTemplate(big, t, cv2.TM_CCOEFF_NORMED))
        if float(res.max()) >= thresh:
            return True
    return False


def drop_cross_segment_duplicate_cuts(
    cuts_by_segment: Dict[str, List[Dict[str, Any]]],
    order: Sequence[str],
    get_img,
    *,
    thresh: float = 0.86,
    coverage_by_file: Optional[Dict[str, float]] = None,
    exempt: Optional[set] = None,
    min_cov: float = 0.99,
    protect: Optional[set] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str]]]:
    """Consecutive SHOWN cuts must differ — across segment boundaries too.

    The per-segment dedup never compares neighbors from different segments,
    so eye-closeup/keyboard/foot-zoom pairs reached the screen back-to-back.
    Duplicates in multi-cut segments are dropped (time redistributed);
    sole-cut duplicates are only REPORTED — the caller forces them through
    garbage substitution instead of emptying the segment.

    A near-blank caption/system box (coverage >= *min_cov*, not *exempt*)
    carries NO unique art: after bubble-inpainting it collapses to a generic
    blank rectangle that template-matches every other panel's caption region.
    Letting one stand as a comparison reference made REAL art panels look like
    duplicates of blank space (IE ch1: the transfer-student reveal p93 was
    killed because it embeds a caption box like its blank neighbour p92). So
    such panels are skipped entirely here — neither flagged nor used as the
    `prev_file` reference; the garbage-substitution pass handles them. This is
    art-style agnostic: it keys on coverage geometry, never on pixels. *protect*
    files (a system card whose text IS the on-screen beat) are kept verbatim:
    never flagged a duplicate and never used as a comparison reference, so a
    system card always survives to be shown."""
    cov = coverage_by_file or {}
    ex = exempt or set()
    prot = protect or set()

    def _blank_ref(f: str) -> bool:
        return bool(cov) and f not in ex and cov.get(f, 0.0) >= min_cov

    out = {k: list(v) for k, v in cuts_by_segment.items()}
    dropped: List[Tuple[str, str]] = []
    prev_file: Optional[str] = None
    for seg in order:
        kept: List[Dict[str, Any]] = []
        cuts = out.get(seg) or []
        for c in cuts:
            f = str(c.get("file"))
            if _blank_ref(f) or f in prot:
                kept.append(c)        # caption/blank/system card: never a visual
                continue              # dup or a reference — leave prev_file intact
            dup = False
            if prev_file and prev_file != f:
                ia, ib = get_img(prev_file), get_img(f)
                if ia is not None and ib is not None and (
                        multi_scale_contained(ib, ia, thresh=thresh)
                        or multi_scale_contained(ia, ib, thresh=thresh)):
                    dup = True
                    dropped.append((seg, f))
            if dup and len(cuts) > 1:
                continue                      # drop; prev_file unchanged
            kept.append(c)
            prev_file = f
        if len(kept) != len(cuts) and kept:
            removed = [str(c.get("file")) for c in cuts
                       if c not in kept]
            out[seg] = _redistribute(cuts, removed)
    return out, dropped


def visually_contained(
    small_img: np.ndarray,
    big_img: np.ndarray,
    *,
    thresh: float = 0.92,
    max_dim: int = 400,
) -> bool:
    """True when *small_img* appears as a region of *big_img* (template match).

    Needed because chunk_global_y0 does NOT account for stitch overlap bands:
    a seam-duplicated panel pair can be 'adjacent' in global coordinates while
    being pixel-identical (the real p15/p16 pair matches at NCC 0.9954).
    Both images share pixel density, so one common downscale preserves match.
    """
    def gray(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im

    sm, bg = gray(small_img), gray(big_img)
    scale = min(1.0, max_dim / max(sm.shape[:2]))
    if scale < 1.0:
        sm = cv2.resize(sm, None, fx=scale, fy=scale)
        bg = cv2.resize(bg, None, fx=scale, fy=scale)
    if sm.shape[0] > bg.shape[0] or sm.shape[1] > bg.shape[1]:
        return False
    res = cv2.matchTemplate(bg, sm, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) >= thresh


def drop_visual_duplicate_cuts(
    cuts: Sequence[Dict[str, Any]],
    images_by_file: Dict[str, np.ndarray],
    *,
    thresh: float = 0.92,
    area_ratio_max: float = 0.9,
    protect: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Drop the smaller cut of any pair whose pixels match inside the larger.
    *protect* files (system cards) are never dropped."""
    prot = protect or set()
    dropped: List[str] = []
    n = len(cuts)
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = str(cuts[i]["file"]), str(cuts[j]["file"])
            if fi in dropped or fj in dropped or fi == fj:
                continue
            a, b = images_by_file.get(fi), images_by_file.get(fj)
            if a is None or b is None:
                continue
            (small_f, small), (big_f, big) = sorted(
                [(fi, a), (fj, b)], key=lambda kv: kv[1].shape[0] * kv[1].shape[1])
            ratio = (small.shape[0] * small.shape[1]) / max(1, big.shape[0] * big.shape[1])
            if (ratio <= area_ratio_max and small_f not in prot
                    and visually_contained(small, big, thresh=thresh)):
                dropped.append(small_f)
    return _redistribute(cuts, dropped), dropped


def _near_identical_similarity(a: np.ndarray, b: np.ndarray, *, size: int = 64) -> float:
    """Full-image similarity in [0,1] for two SIMILAR-SIZED panels.

    Both images are downscaled to a fixed *size*x*size* grayscale grid (so a
    few px of size mismatch don't matter) and compared with normalized
    cross-correlation. NCC keys on STRUCTURE, not absolute brightness, so a
    global tone shift between two genuinely-different panels never reads as a
    match; only the same drawing, barely changed, scores near 1.0. Returns 0.0
    when either image is featureless (flat) — a zero-variance NCC is undefined
    and would spuriously match every other flat panel.
    """
    def gray64(im: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)

    ga, gb = gray64(a), gray64(b)
    sa, sb = float(ga.std()), float(gb.std())
    if sa < 4.0 or sb < 4.0:
        return 0.0  # flat/featureless panel — NCC is meaningless
    za, zb = (ga - ga.mean()) / sa, (gb - gb.mean()) / sb
    return float((za * zb).mean())  # NCC in [-1, 1]; near 1.0 == same drawing


def drop_near_identical_cuts(
    cuts: Sequence[Dict[str, Any]],
    images_by_file: Dict[str, np.ndarray],
    *,
    thresh: float = 0.96,
    min_area_ratio: float = 0.7,
    protect: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Drop the LATER of any pair of SIMILAR-SIZED, near-identical cuts.

    Catches the case the containment filter (drop_visual_duplicate_cuts) cannot:
    two SEPARATE panels of roughly the same size with the same framing and only
    tiny differences (the Ch20 g0003 'reaction face with ?' pair p000013 /
    p000016 — area_ratio ~1.0, so neither is "the small one contained in the
    big one"). We resize both full images to 64x64 grayscale and compare with
    normalized cross-correlation; a pair is a near-dup only when similarity
    >= *thresh* AND their areas are close (area ratio >= *min_area_ratio*), so a
    seam fragment (small-in-big, low area ratio) is left for the containment
    filter. The EARLIER cut is kept, the later dropped, freed time redistributed.
    Conservative by design: 0.96 NCC means the same drawing barely changed —
    two distinct panels (different characters/scenes) score far lower and survive.
    *protect* files (system cards) are never dropped.
    """
    prot = protect or set()
    dropped: List[str] = []
    n = len(cuts)
    for i in range(n):
        fi = str(cuts[i]["file"])
        if fi in dropped:
            continue
        for j in range(i + 1, n):
            fj = str(cuts[j]["file"])
            if fj in dropped or fi == fj or fj in prot:
                continue
            a, b = images_by_file.get(fi), images_by_file.get(fj)
            if a is None or b is None:
                continue
            area_a = a.shape[0] * a.shape[1]
            area_b = b.shape[0] * b.shape[1]
            ratio = min(area_a, area_b) / max(1, max(area_a, area_b))
            if ratio < min_area_ratio:
                continue  # different-sized seam pair — not our case
            if _near_identical_similarity(a, b) >= thresh:
                dropped.append(fj)  # keep the earlier cut, drop the later
    return _redistribute(cuts, dropped), dropped


# ---------------------------------------------------------------------------
# 3. uniform light border trim (pure)
# ---------------------------------------------------------------------------

def content_bbox(
    img: np.ndarray,
    *,
    light_thresh: int = 215,
    uniform_frac: float = 0.97,
    max_trim_frac: float = 0.18,
) -> Tuple[int, int, int, int]:
    """(x1, y1, x2, y2) of the artwork after trimming uniform LIGHT margins.

    Only near-white/page-grey borders are trimmed (the baked page margin);
    dark art and the panel's own outline are content. Trim per side is capped
    at max_trim_frac so a mostly-white panel can never be eaten.
    """
    gray = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float64)
    H, W = gray.shape[:2]
    light = gray >= light_thresh

    def run(mean_fn, limit: int) -> int:
        n = 0
        while n < limit and mean_fn(n) >= uniform_frac:
            n += 1
        return n

    cap_y, cap_x = int(H * max_trim_frac), int(W * max_trim_frac)
    top = run(lambda r: light[r, :].mean(), cap_y)
    bot = run(lambda r: light[H - 1 - r, :].mean(), cap_y)
    left = run(lambda c: light[:, c].mean(), cap_x)
    right = run(lambda c: light[:, W - 1 - c].mean(), cap_x)
    return (left, top, W - right, H - bot)


# ---------------------------------------------------------------------------
# 2. oval-aware bubble mask + inpaint (pure given an image)
# ---------------------------------------------------------------------------

def _bubble_text(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    *,
    pad: int = 4,
) -> Tuple[np.ndarray, Optional[int], Optional[np.ndarray]]:
    """(text_mask, fill_value, interior_mask) for one bubble box.

    User direction: the bubble (shape + outline) STAYS; only its text is
    blanked with the bubble's own flat color — no inpainting, so no smears.
    The interior is the near-white (or near-black, shout bubbles) connected
    component around the box centre; text = contrasting pixels safely inside
    that component's filled contour (eroded clear of the outline ring).
    """
    H, W = img.shape[:2]
    x1 = max(0, int(box[0]) - pad)
    y1 = max(0, int(box[1]) - pad)
    x2 = min(W, int(box[2]) + pad)
    y2 = min(H, int(box[3]) + pad)
    mask = np.zeros((H, W), np.uint8)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return mask, None, None

    gray = img[y1:y2, x1:x2].mean(axis=2) if img.ndim == 3 else img[y1:y2, x1:x2]
    gray = gray.astype(np.uint8)

    def centre_component(binary: np.ndarray) -> Optional[np.ndarray]:
        n, labels = cv2.connectedComponents(binary.astype(np.uint8))
        h, w = binary.shape
        cy, cx = h // 2, w // 2
        win = labels[max(0, cy - h // 6): cy + h // 6 + 1,
                     max(0, cx - w // 6): cx + w // 6 + 1]
        vals, counts = np.unique(win[win > 0], return_counts=True)
        if len(vals) == 0:
            return None
        return (labels == vals[np.argmax(counts)]).astype(np.uint8)

    white = centre_component(gray >= 225)
    black = centre_component(gray <= 35)
    if white is not None and (black is None or white.sum() >= black.sum()):
        comp, is_white = white, True
    elif black is not None:
        comp, is_white = black, False
    else:
        return mask, None, None

    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None, None
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    inside = cv2.erode(
        filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    if is_white:
        text = ((gray <= 170) & (inside > 0)).astype(np.uint8)
        fill = int(np.median(gray[comp > 0])) if comp.any() else 250
    else:
        text = ((gray >= 90) & (inside > 0)).astype(np.uint8)
        fill = int(np.median(gray[comp > 0])) if comp.any() else 10

    text = cv2.dilate(
        text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    text &= inside  # the dilation must never reach the outline ring
    mask[y1:y2, x1:x2] = text * 255

    inside_full = np.zeros((H, W), np.uint8)
    inside_full[y1:y2, x1:x2] = inside
    return mask, fill, inside_full


def bubble_text_mask(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """uint8 mask (255 = blank) of the TEXT inside a bubble box."""
    return _bubble_text(img, box)[0]


def _merge_word_clusters(
    rects: Sequence[Tuple[int, int, int, int]],
    gap: int = 14,
) -> List[Tuple[int, int, int, int]]:
    """Union word rects that sit within *gap* px of each other — one cluster
    per text block, so the surround ring samples the bubble, not neighbors."""
    work = [list(r) for r in rects]
    merged = True
    while merged:
        merged = False
        out: List[List[int]] = []
        for r in work:
            for o in out:
                if (min(r[2], o[2]) - max(r[0], o[0]) > -gap
                        and min(r[3], o[3]) - max(r[1], o[1]) > -gap):
                    o[0] = min(o[0], r[0]); o[1] = min(o[1], r[1])
                    o[2] = max(o[2], r[2]); o[3] = max(o[3], r[3])
                    merged = True
                    break
            else:
                out.append(r)
        work = out
    return [tuple(r) for r in work]


def _flat_surround_fill(
    img: np.ndarray,
    rect: Tuple[int, int, int, int],
    pad: int = 10,
) -> Optional[int]:
    """Fill value when *rect* sits on a uniform near-white/near-black surround
    (an undetected bubble interior); None when the surround is artwork."""
    h, w = img.shape[:2]
    gray = img.mean(axis=2) if img.ndim == 3 else img
    x1, y1, x2, y2 = [int(v) for v in rect]
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    ring = np.ones((ry2 - ry1, rx2 - rx1), bool)
    ring[(y1 - ry1):(y2 - ry1), (x1 - rx1):(x2 - rx1)] = False
    vals = gray[ry1:ry2, rx1:rx2][ring]
    if vals.size < 30:
        return None
    med = float(np.median(vals))
    if med >= 232 and float((vals >= 215).mean()) >= 0.85:
        return int(med)
    if med <= 30 and float((vals <= 50).mean()) >= 0.85:
        return int(med)
    return None


def _flatten_blank_bubble_residue(
    out: np.ndarray,
    box: Tuple[int, int, int, int],
    fill: Optional[int],
) -> None:
    """Final cleanup for bubbles that are already blank.

    The oval mask intentionally avoids outlines, but clipped/spiky bubbles can
    leave faint gray anti-aliased text just outside that mask while still inside
    the viewer-visible blank bubble. Flatten only a safe inset rectangle, and
    only when that interior is already white/black and low-ink.
    """
    if fill is None:
        return
    gray = out.mean(axis=2) if out.ndim == 3 else out.astype(float)
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    dx = max(4, int(0.12 * max(1, x2 - x1)))
    dy = max(4, int(0.12 * max(1, y2 - y1)))
    rx1, ry1 = max(0, x1 + dx), max(0, y1 + dy)
    rx2, ry2 = min(w, x2 - dx), min(h, y2 - dy)
    if rx2 <= rx1 or ry2 <= ry1:
        return
    roi = gray[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return
    if int(fill) >= 128:
        white_frac = float((roi >= 235).mean())
        ink_frac = float((roi <= 120).mean())
        if white_frac >= 0.70 and ink_frac < 0.03:
            mask = (roi >= 140) & (roi < 235)
            if mask.any():
                out[ry1:ry2, rx1:rx2][mask] = fill
    else:
        black_frac = float((roi <= 25).mean())
        ink_frac = float((roi >= 180).mean())
        if black_frac >= 0.70 and ink_frac < 0.03:
            mask = (roi > 25) & (roi <= 120)
            if mask.any():
                out[ry1:ry2, rx1:rx2][mask] = fill


def clean_scene_image(
    img: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    """Remove the text inside each bubble; the bubble itself stays.

    Primary method (the user's original approach): inpaint the exact OCR word
    rects that fall inside the bubble interior — regions that small heal
    invisibly, it reads as "the text was simply removed". Fallback when OCR
    missed a bubble entirely: blank contrasting pixels with the bubble's own
    flat color. A residue sweep then flattens anything still deviating from
    the fill inside the interior (missed glyphs, anti-aliased ghosts).

    Word boxes OUTSIDE every detected bubble are blanked only when their
    surround is a uniform near-white/black void — a bubble the detector
    missed (spiky scream balloons). Text embedded in artwork keeps its
    textured surround and survives.
    """
    words = [tuple(int(v) for v in t) for t in (text_boxes or [])]
    if not boxes and not words:
        return img
    out = img.copy()
    for b in boxes:
        tmask, fill, inside = _bubble_text(out, b)
        if inside is None:
            continue
        wmask = np.zeros(out.shape[:2], np.uint8)
        for (wx1, wy1, wx2, wy2) in words:
            pad = 5  # cover anti-aliased stroke edges beyond the tight OCR box
            wmask[max(0, wy1 - pad):wy2 + pad, max(0, wx1 - pad):wx2 + pad] = 255
        gate = inside > 0
        # A bubble clipped by the panel edge (its body cut off by the panel
        # boundary) has its text flush against that edge, where the eroded
        # interior can't reach — the inside-gate alone leaves the glyphs (IE
        # ch1 p000111 "JOINING OUR CLASS."). The detector vouched for this box
        # and OCR pinned the words, so also admit word pixels inside the
        # bubble's inner region: inset to spare the outline ring, but flush on
        # the clipped side(s).
        H_, W_ = out.shape[:2]
        bx1, by1, bx2, by2 = (int(v) for v in b)
        edge_tol = 2
        if bx1 <= edge_tol or by1 <= edge_tol or bx2 >= W_ - edge_tol or by2 >= H_ - edge_tol:
            ox = max(4, int(0.06 * (bx2 - bx1)))
            oy = max(4, int(0.06 * (by2 - by1)))
            ix1 = bx1 if bx1 <= edge_tol else bx1 + ox
            iy1 = by1 if by1 <= edge_tol else by1 + oy
            ix2 = bx2 if bx2 >= W_ - edge_tol else bx2 - ox
            iy2 = by2 if by2 >= H_ - edge_tol else by2 - oy
            if ix2 > ix1 and iy2 > iy1:
                inner = np.zeros((H_, W_), bool)
                inner[max(0, iy1):iy2, max(0, ix1):ix2] = True
                gate = gate | inner
        wmask = cv2.bitwise_and(wmask, gate.astype(np.uint8) * 255)
        if wmask.any() and fill is not None:
            # flat fill with the bubble's own interior color: on a flat
            # interior this is exact removal — nothing to ghost or smear
            out[wmask > 0] = fill
        elif fill is not None and tmask.any():
            out[tmask > 0] = fill
        if fill is not None:
            # residue sweep — but only on genuinely flat interiors, so a
            # false-positive detector box on artwork is never flattened
            g = out.mean(axis=2) if out.ndim == 3 else out
            flat = (np.abs(g.astype(int) - int(fill)) <= 15) & (inside > 0)
            n_inside = int((inside > 0).sum())
            if n_inside and flat.sum() / n_inside >= 0.80:
                residue = (inside > 0) & ~flat
                if residue.any():
                    out[residue] = fill
            _flatten_blank_bubble_residue(out, b, fill)

    if words:
        grown = [(int(b[0]) - 6, int(b[1]) - 6, int(b[2]) + 6, int(b[3]) + 6)
                 for b in boxes]

        def covered(wr: Tuple[int, int, int, int]) -> bool:
            wx1, wy1, wx2, wy2 = wr
            wa = max(1, (wx2 - wx1) * (wy2 - wy1))
            for (bx1, by1, bx2, by2) in grown:
                ix = max(0, min(wx2, bx2) - max(wx1, bx1))
                iy = max(0, min(wy2, by2) - max(wy1, by1))
                if ix * iy >= 0.5 * wa:
                    return True
            return False

        orphans = [w for w in words if not covered(w)]
        for cl in _merge_word_clusters(orphans):
            fill = _flat_surround_fill(out, cl)
            if fill is not None:
                h, w = out.shape[:2]
                x1, y1 = max(0, cl[0] - 4), max(0, cl[1] - 4)
                x2, y2 = min(w, cl[2] + 4), min(h, cl[3] + 4)
                out[y1:y2, x1:x2] = fill
    return out


def clean_panel_image(
    img: np.ndarray,
    panel_kind: str,
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    """Bubble-text removal that honors panel_kind (the D3 husk fix).

    Blanking a speech bubble on STORY artwork leaves an empty white husk that
    reads as broken on screen (IE p000007 "...SHIT!! / I CAN'T MOVE": real art,
    blanked=True). The dialogue IS voiced, but the user's direction (option b,
    zero smear) is to keep the drawn text in the art rather than gut it — so a
    `story` panel is returned byte-identical (a fresh copy, never blanked).
    Every other kind (caption / document / system / bubble-dominated) is blanked
    via clean_scene_image exactly as before.
    """
    if str(panel_kind or "").strip().lower() == "story":
        return img.copy()
    blist = list(boxes or [])
    if not blist and not (text_boxes or []):
        return img.copy()
    return clean_scene_image(img, blist, text_boxes=text_boxes)


def bubble_coverage(
    shape: Tuple[int, ...],
    boxes: Sequence[Tuple[int, int, int, int]],
) -> float:
    """Fraction of the panel covered by bubble boxes (union, downscaled grid)."""
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0 or not boxes:
        return 0.0
    s = 4
    grid = np.zeros((max(1, h // s), max(1, w // s)), np.uint8)
    for (x1, y1, x2, y2) in boxes:
        grid[max(0, int(y1) // s): max(0, int(y2) // s),
             max(0, int(x1) // s): max(0, int(x2) // s)] = 1
    return float(grid.mean())


def art_content_score(
    img: np.ndarray,
    bubble_boxes: Sequence[Tuple[int, int, int, int]],
) -> float:
    """Fraction of edge pixels OUTSIDE the bubble regions — how much actual
    artwork detail a (cleaned) panel offers. Empty-bubble husks over gradients
    score near zero; real art scores an order of magnitude higher. This is the
    gate that catches panels which only become worthless AFTER text cleaning."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    keep = np.ones(gray.shape, bool)
    for (x1, y1, x2, y2) in bubble_boxes:
        keep[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = False
    n = int(keep.sum())
    if n == 0:
        return 0.0
    return float((edges > 0)[keep].sum()) / n


def drop_bubble_dominated_cuts(
    cuts: Sequence[Dict[str, Any]],
    coverage_by_file: Dict[str, float],
    *,
    max_coverage: float = 0.45,
    exempt: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Drop cuts that are mostly bubble/text (a cleaned bubble-only panel is a
    near-blank blob on screen). *exempt* files (system-message/status panels —
    story beats) are never dropped. Never
    empties a shot — the least bubbly cut survives."""
    ex = exempt or set()
    over = [c for c in cuts
            if str(c["file"]) not in ex
            and coverage_by_file.get(str(c["file"]), 0.0) >= max_coverage]
    if not over:
        return list(cuts), []
    dropped = [str(c["file"]) for c in over]
    if len(dropped) == len(cuts):
        keeper = min(cuts, key=lambda c: coverage_by_file.get(str(c["file"]), 0.0))
        dropped = [f for f in dropped if f != str(keeper["file"])]
        if not dropped:
            return list(cuts), []
    return _redistribute(cuts, dropped), dropped


def filter_protected_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
    protected: Sequence[Tuple[int, int, int, int]],
    *,
    max_overlap: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    """Remove bubble boxes that mostly overlap a protected (system_box) region —
    system-window text is read aloud by the script and must stay visible."""
    out: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        bx1, by1, bx2, by2 = b
        barea = max(1, (bx2 - bx1) * (by2 - by1))
        hit = False
        for (px1, py1, px2, py2) in protected:
            ix = max(0, min(bx2, px2) - max(bx1, px1))
            iy = max(0, min(by2, py2) - max(by1, py1))
            if (ix * iy) / barea >= max_overlap:
                hit = True
                break
        if not hit:
            out.append(b)
    return out


def split_on_white_bands(
    img: np.ndarray,
    *,
    min_band_h: int = 40,
    white_thresh: int = 225,
    white_frac: float = 0.93,
    min_part_h: int = 16,
    pad: int = 12,
) -> List[Tuple[int, int]]:
    """(y1, y2) content spans of an over-merged crop, split at wide internal
    white bands (the dead page-void between stacked panels). One span = no
    split. Spans are padded and clipped."""
    gray = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float64)
    H = gray.shape[0]
    white_rows = (gray >= white_thresh).mean(axis=1) >= white_frac

    spans: List[Tuple[int, int]] = []
    y = 0
    while y < H:
        if not white_rows[y]:
            start = y
            while y < H and not white_rows[y]:
                y += 1
            spans.append((start, y))
        else:
            y += 1

    # merge spans separated by thin white gaps (< min_band_h = not a real band)
    merged: List[Tuple[int, int]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < min_band_h:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)

    merged = [(a, b) for a, b in merged if b - a >= min_part_h]
    if len(merged) <= 1:
        return [(0, H)]
    return [(max(0, a - pad), min(H, b + pad)) for a, b in merged]


def filter_content_parts(
    img: np.ndarray,
    parts: Sequence[Tuple[int, int]],
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    min_h: int = 120,
    max_bubble_cov: float = 0.5,
    min_midtone_frac: float = 0.15,
    min_art_score: float = 0.012,
) -> List[Tuple[int, int]]:
    """Keep only the REAL-art parts of a split scene.

    Discards parts that are (a) too short, (b) mostly covered by detected
    bubbles, (c) near-binary black+white — spiky scream/SFX bubbles evade the
    bubble detector but have almost no midtones — or (d) edge-dead gradient
    husks (midtone-rich backgrounds with no actual line art)."""
    gray_full = img.mean(axis=2) if img.ndim == 3 else img
    out: List[Tuple[int, int]] = []
    for (a, b) in parts:
        if (b - a) < min_h:
            continue
        part_boxes = [(x1, y1 - a, x2, y2 - a)
                      for (x1, y1, x2, y2) in boxes
                      if min(y2, b) - max(y1, a) > 0]
        if bubble_coverage((b - a, img.shape[1]), part_boxes) >= max_bubble_cov:
            continue
        g = gray_full[a:b]
        midtone = float(((g > 60) & (g < 200)).mean())
        if midtone < min_midtone_frac:
            continue
        if art_content_score(img[a:b], part_boxes) < min_art_score:
            continue
        out.append((a, b))
    return out


def dead_box_recrop(
    img: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    max_blank_frac: float = 0.35,
    min_h: int = 120,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop away large now-blank caption/bubble boxes that dominate a panel.

    After text blanking, big rectangular caption boxes become empty white
    voids (ghost remnants included) that can fill most of the frame while a
    thin strip of real art survives (user report #22: feet strip + two huge
    blanked boxes). When boxes cover more than *max_blank_frac* of the panel,
    crop to the largest band of rows whose art lives OUTSIDE the boxes.
    NOT yet wired into main — see handover."""
    h, w = img.shape[:2]
    info: Dict[str, Any] = {"blank_box_frac": 0.0, "recropped": False}
    if h == 0 or w == 0 or not boxes:
        return img, info

    info["blank_box_frac"] = bubble_coverage((h, w), boxes)
    if info["blank_box_frac"] < max_blank_frac:
        return img, info

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 50, 150) > 0
    outside = np.ones((h, w), bool)
    for (x1, y1, x2, y2) in boxes:
        # pad past the box border strokes so they don't count as "art"
        outside[max(0, int(y1) - 6):min(h, int(y2) + 6),
                max(0, int(x1) - 6):min(w, int(x2) + 6)] = False

    row_art = (edges & outside).sum(axis=1) / np.maximum(1, outside.sum(axis=1))
    content = row_art > 0.01

    best: Tuple[int, int] = (0, 0)
    y = 0
    while y < h:
        if content[y]:
            start = y
            while y < h and (content[y] or (y - start < 20)):
                y += 1
            if (y - start) > (best[1] - best[0]):
                best = (start, y)
        else:
            y += 1

    if best[1] - best[0] >= min_h:
        a = max(0, best[0] - 10)
        b = min(h, best[1] + 10)
        # an edge-rich band can still be a binary scream bubble (radiating
        # black/white spikes, the Nano p000020 case) — real art has midtones
        # AND color; anti-aliased spikes fake midtones but stay chroma-zero
        band = gray[a:b]
        midtone = float(((band > 60) & (band < 200)).mean())
        chroma_ok = True
        if img.ndim == 3:
            sub = img[a:b].astype(int)
            chroma = float(np.maximum(
                np.maximum(np.abs(sub[..., 0] - sub[..., 1]),
                           np.abs(sub[..., 1] - sub[..., 2])),
                np.abs(sub[..., 0] - sub[..., 2])).mean())
            chroma_ok = chroma >= 5.0
        if midtone >= 0.15 and chroma_ok:
            info["recropped"] = True
            info["band"] = (a, b)
            return img[a:b], info
    return img, info


def select_panel_crops(
    img: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    text_rich: bool,
    no_split: bool = False,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """The writer's crop decision for one CLEANED panel: dead-box recrop →
    white-band split → content filter. Returns one part (possibly recropped)
    or two parts (split2). Document panels pass through whole."""
    info: Dict[str, Any] = {"recropped": False, "blank_box_frac": 0.0}
    if not text_rich:
        img2, dead = dead_box_recrop(img, boxes)
        info.update(dead)
        if dead.get("recropped"):
            a, b = dead["band"]
            boxes = [(x1, max(0, y1 - a), x2, min(b - a, y2 - a))
                     for (x1, y1, x2, y2) in boxes
                     if min(y2, b) - max(y1, a) > 0]
            img = img2

    spans = ([(0, int(img.shape[0]))] if no_split
             else split_spans_for_panel(img, text_rich=text_rich))
    if len(spans) > 1:
        content = filter_content_parts(img, spans, boxes)
        if len(content) == 2:
            return [img[a:b] for (a, b) in content], info
        if len(content) == 1:
            a, b = content[0]
            return [img[a:b]], info
    return [img], info


def speech_shaped_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
    panel_w: int,
    *,
    max_aspect: float = 3.5,
    max_w_frac: float = 0.85,
) -> List[Tuple[int, int, int, int]]:
    """Only boxes shaped like speech bubbles. The bubble detector also boxes
    full-width UI rows (the ORV app list) and caption strips — wide flat
    rectangles are not speech, and must not make a document look dialogue."""
    out: List[Tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2) in boxes:
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if w >= max_w_frac * max(1, panel_w):
            continue
        if w >= max_aspect * h:
            continue
        out.append((x1, y1, x2, y2))
    return out


def doc_like(
    text_coverage: float,
    n_words: int,
    word_boxes: Sequence[Tuple[int, int, int, int]],
    bubble_boxes: Sequence[Tuple[int, int, int, int]],
    *,
    min_coverage: float = 0.22,
    min_words: int = 15,
    max_in_bubble_frac: float = 0.5,
    min_outside_words: int = 8,
) -> bool:
    """Is this a DOCUMENT panel (app screen / stats page) or just wordy?

    Word count alone misclassifies dialogue-heavy panels as documents (15+
    words is two speech bubbles), which keeps their dialogue ON SCREEN while
    the narration speaks the same lines. A document's words live OUTSIDE
    speech bubbles; dialogue's words live inside them. Mixed panels (speech
    bubble over an app screen, ORV p000025) stay documents when the
    outside-bubble text is substantial on its own."""
    if not (float(text_coverage) >= min_coverage or int(n_words) >= min_words):
        return False
    if not word_boxes or not bubble_boxes:
        return True
    grown = [(x1 - 6, y1 - 6, x2 + 6, y2 + 6)
             for (x1, y1, x2, y2) in bubble_boxes]
    inside = 0
    for (wx1, wy1, wx2, wy2) in word_boxes:
        wa = max(1, (wx2 - wx1) * (wy2 - wy1))
        for (bx1, by1, bx2, by2) in grown:
            ix = max(0, min(wx2, bx2) - max(wx1, bx1))
            iy = max(0, min(wy2, by2) - max(wy1, by1))
            if ix * iy >= 0.5 * wa:
                inside += 1
                break
    outside = len(word_boxes) - inside
    return (inside / len(word_boxes) < max_in_bubble_frac
            or outside >= min_outside_words)


_TEXT_CONTEXT_SUBJECT_TERMS = (
    "speech bubble",
    "bubble",
    "thought bubble",
    "text bubble",
    "caption",
    "text",
    "sfx",
    "sound effect",
    "onomatopoeia",
)

_MINOR_FRAGMENT_SUBJECT_TERMS = (
    "hair",
    "character's hair",
    "character hair",
)

_STORY_VISUAL_SUBJECT_TERMS = (
    "character",
    "man",
    "woman",
    "person",
    "boy",
    "girl",
    "doctor",
    "prince",
    "face",
    "figure",
    "body",
    "head",
    "eyes",
    "hand",
    "hands",
    "foot",
    "feet",
)


def _looks_like_title_text(ocr: str, text_coverage: float) -> bool:
    ocr = str(ocr or "").strip()
    if not ocr or "..." in ocr or any(c in ocr for c in "~!?"):
        return False
    words = [w for w in re.split(r"[^A-Za-z0-9']+", ocr)
             if any(c.isalpha() for c in w)]
    letters = [c for c in ocr if c.isalpha()]
    if not (2 <= len(words) <= 8) or not letters:
        return False
    if sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return False
    return float(text_coverage or 0.0) < 0.20


def text_context_only_panel(vitem: Dict[str, Any]) -> bool:
    """True when the panel's only usable signal is text/bubble content.

    The OCR still belongs in narration context, but after dialogue blanking the
    image is not a story visual. This closes the gap where panel_understand can
    stamp a pure thought bubble as panel_kind=story.
    """
    kind = str(vitem.get("panel_kind") or "").strip().lower()
    ocr = str(vitem.get("ocr_clean") or "").strip()
    text_cov = float(vitem.get("text_coverage") or 0.0)
    subjects = [str(s or "").strip().lower()
                for s in (vitem.get("subjects") or []) if str(s or "").strip()]
    if kind in {"caption", "empty"}:
        return True
    if not subjects:
        return False

    def is_text_subject(subj: str) -> bool:
        return any(term in subj for term in _TEXT_CONTEXT_SUBJECT_TERMS)

    def is_minor_fragment_subject(subj: str) -> bool:
        s = subj.strip().lower()
        return (s in _MINOR_FRAGMENT_SUBJECT_TERMS
                or s.endswith("'s hair")
                or s.endswith(" hair"))

    has_text_subject = any(is_text_subject(s) for s in subjects)
    has_real_subject = any(
        not is_text_subject(s) and not is_minor_fragment_subject(s)
        for s in subjects)
    has_text_signal = bool(ocr) or text_cov >= 0.02 or bool(vitem.get("text_only"))
    if has_text_subject and not has_real_subject and has_text_signal:
        has_bubble_subject = any("bubble" in s for s in subjects)
        return bool(has_bubble_subject or not _looks_like_title_text(ocr, text_cov))
    if _looks_like_title_text(ocr, text_cov):
        return False
    return False


def story_visual_panel(vitem: Dict[str, Any]) -> bool:
    """A story panel with a real visual subject, even when text/bubbles also
    occupy much of the frame. This protects chibi/info and reaction panels from
    being mistaken for blank text husks after dialogue removal, while pure
    bubble-only/context panels still drop through empty_bubble_panel()."""
    if str(vitem.get("panel_kind") or "").strip().lower() != "story":
        return False
    if text_context_only_panel(vitem):
        return False
    subjects = [str(s or "").strip().lower()
                for s in (vitem.get("subjects") or []) if str(s or "").strip()]

    def is_text_subject(subj: str) -> bool:
        return any(term in subj for term in _TEXT_CONTEXT_SUBJECT_TERMS)

    def is_minor_fragment_subject(subj: str) -> bool:
        s = subj.strip().lower()
        return (s in _MINOR_FRAGMENT_SUBJECT_TERMS
                or s.endswith("'s hair")
                or s.endswith(" hair"))

    for subj in subjects:
        if is_text_subject(subj) or is_minor_fragment_subject(subj):
            continue
        if any(term in subj for term in _STORY_VISUAL_SUBJECT_TERMS):
            return True
    return False


def empty_bubble_panel(
    vitem: Dict[str, Any],
    *,
    max_text_coverage: float = 0.10,
    max_words: int = 10,
) -> bool:
    """Deterministic junk signal from panel understanding.

    `panel_kind=empty` means the understanding found no story-bearing art. A
    pure bubble/text panel can also be mislabeled as story; in both cases the
    cleaned cut becomes a blank bubble blob on screen and must be covered by a
    neighboring story panel instead of rendered directly.
    """
    if text_context_only_panel(vitem):
        return True
    if str(vitem.get("panel_kind") or "").strip().lower() != "empty":
        return False
    subjects = [str(s).lower() for s in (vitem.get("subjects") or [])]
    has_bubble_subject = any("bubble" in s for s in subjects)
    ocr = str(vitem.get("ocr_clean") or "")
    words = [w for w in re.split(r"[^A-Za-z0-9']+", ocr)
             if any(c.isalpha() for c in w)]
    low_text = (float(vitem.get("text_coverage") or 0.0) <= max_text_coverage
                and len(words) <= max_words)
    return has_bubble_subject or low_text


def split_spans_for_panel(img: np.ndarray, *, text_rich: bool) -> List[Tuple[int, int]]:
    """Spans for the splitter. Document-like panels (the ORV in-story app
    list — many text rows) are NEVER split: white gaps between rows would
    shred them into sub-min_h fragments and discard story content."""
    if text_rich:
        return [(0, int(img.shape[0]))]
    return split_on_white_bands(img)


def panel_recoverable(
    img: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    min_art_score: float = 0.012,
    text_rich: bool = False,
) -> bool:
    """The drop-vs-recrop decision for a CLEANED panel: dropped ONLY when no
    region holds real content. Text-rich (document) panels are judged WHOLE
    by edge detail — text glyphs ARE their content; everything else is judged
    by its best split part, which the writer then recrops to."""
    if text_rich:
        # document panels: their text/UI IS the content — never exclude the
        # detector's (often false-positive) boxes from the score, else a
        # boxed-over stats page reads as blank (the ORV p000003 case)
        return art_content_score(img, []) >= min_art_score
    spans = split_spans_for_panel(img, text_rich=False)
    parts = filter_content_parts(img, spans, boxes, min_art_score=min_art_score)
    if parts:
        return True
    # every part can fail individually (bubble-dominated span, bright glow
    # span) while the WHOLE panel is real art — the writer keeps the whole
    # image when no part qualifies, so judge that same image (IE p000039).
    # Guards (measured on the real misses):
    #  - midtone >= 0.08, the established binary-card line;
    #  - chroma evidence: monochrome panels at this point are spike bursts /
    #    blanked-bubble blobs, never color-webtoon art (Nano p000020 has
    #    chroma 0.0 yet midtone 0.13 from anti-aliasing);
    #  - boxes PADDED before edge exclusion: empty-bubble outline rims sit
    #    just outside the detector boxes and fake an art score on otherwise
    #    edge-dead gradients (IE p000008 curtain).
    gray = img.mean(axis=2) if img.ndim == 3 else img
    midtone = float(((gray > 60) & (gray < 200)).mean())
    if img.ndim == 3:
        b = img[..., 0].astype(int)
        g2 = img[..., 1].astype(int)
        r = img[..., 2].astype(int)
        chroma = float(np.maximum(np.maximum(np.abs(b - g2), np.abs(g2 - r)),
                                  np.abs(b - r)).mean())
    else:
        chroma = 0.0
    padded = [(x1 - 8, y1 - 8, x2 + 8, y2 + 8) for (x1, y1, x2, y2) in boxes]
    if (midtone >= 0.08 and chroma >= 5.0
            and art_content_score(img, padded) >= min_art_score):
        return True
    # blank caption boxes can dominate coverage while a thin band of real art
    # survives outside them (#22) — recoverable iff dead_box_recrop rescues it
    cropped, dead = dead_box_recrop(img, boxes)
    return bool(dead.get("recropped")) and art_content_score(cropped, []) >= min_art_score


def exempt_from_drop(
    *,
    recoverable: bool,
    sys_box: bool,
    title_card: bool,
    rich: bool,
    visual_story: bool,
    panel_kind: Optional[str],
    has_ocr: bool,
) -> bool:
    """Whether a cut is protected from the bubble/husk drop gate.

    Document panels (their text IS the content) and real story visuals are always
    exempt. A SYSTEM / title card is ALSO unconditionally exempt: its text sits on
    a flat card (NOT inside an inpainted bubble), so the text IS the on-screen
    story beat and the card must always be shown — even when blanking would leave
    it "empty-looking" (a notification on a plain white background: Nano ch1
    p000114 "7TH GENERATION NANO MACHINE", non-recoverable after its text is
    blanked). This does NOT shield an empty DIALOGUE bubble: that panel is marked
    panel_kind=caption by the understanding and excluded UPSTREAM (panel_understand
    + story_group fold its text into the adjacent art's narration), so it never
    reaches this gate as a "system" husk (Nano ch1 p000020). The broad
    'story/caption carries OCR' exemption stays gated on `recoverable` — a
    contentless caption husk (no recoverable art after cleaning) is NOT shielded
    and still drops."""
    if rich:
        return True
    if visual_story:
        return True
    if sys_box or title_card or panel_kind == "system":
        return True
    if recoverable and panel_kind in ("story", "caption") and has_ocr:
        return True
    return False


_SCENE_NUM_RE = re.compile(r"(\d+)")


def _scene_num(fname: str) -> int:
    m = _SCENE_NUM_RE.search(os.path.basename(str(fname)))
    return int(m.group(1)) if m else -1


def substitute_garbage_sole_cuts(
    cuts_by_segment: Dict[str, List[Dict[str, Any]]],
    coverage_by_file: Dict[str, float],
    *,
    durations: Dict[str, float],
    exempt: Optional[set] = None,
    min_cov: float = 0.99,
    order: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str, str]]]:
    """A segment whose ONLY cut is hard garbage (chrome cover, husk, cross-seg
    duplicate — score >= *min_cov*, not *exempt*) must never ship that garbage.

    Rather than swapping in the numerically-nearest KEPT panel — which is
    STORY-BLIND and put the WRONG art under the narration (IE Bai Xue: the
    transfer-student line ran over an unrelated sports panel) — HOLD the nearest
    GOOD panel, preferring the one just BEFORE it (story-adjacent), falling back
    to the next good panel at the chapter head. A held image with the narration
    running over it reads as deliberate coverage; QA's montage + semantic judge
    already exempt holds. A garbage segment with no good panel anywhere keeps
    its least-bad cut so the shot is never empty."""
    ex = exempt or set()
    seq = list(order) if order else list(cuts_by_segment.keys())
    out = {k: list(v) for k, v in cuts_by_segment.items()}
    subs: List[Tuple[str, str, str]] = []

    def _is_garbage(seg: str) -> bool:
        cuts = cuts_by_segment.get(seg) or []
        return (len(cuts) == 1
                and str(cuts[0].get("file")) not in ex
                and coverage_by_file.get(str(cuts[0].get("file")), 0.0) >= min_cov)

    # nearest GOOD (non-garbage) shown panel in each direction, one scan each
    prev_good: Dict[str, Optional[str]] = {}
    g: Optional[str] = None
    for seg in seq:
        prev_good[seg] = g
        if not _is_garbage(seg) and (cuts_by_segment.get(seg)):
            g = str(cuts_by_segment[seg][-1].get("file"))
    next_good: Dict[str, Optional[str]] = {}
    g = None
    for seg in reversed(seq):
        next_good[seg] = g
        if not _is_garbage(seg) and (cuts_by_segment.get(seg)):
            g = str(cuts_by_segment[seg][-1].get("file"))

    # A stretch of narration-only caption boxes must not freeze on one panel
    # (IE ch1: p93 held 4x/33s). Cover each caption by HOLDING a story-adjacent
    # real scene, cycling so no on-screen image (held or real) repeats more than
    # twice in a row. The candidate pool is the upcoming scene (forward bridge)
    # plus the recent scenes (newest first) — so a mid-chapter run alternates
    # before/after while an END-of-chapter cliffhanger run (no scene after)
    # replays recent scenes. Agnostic: keys on coverage geometry, not pixels.
    prev_shown: Optional[str] = None       # last file actually put on screen
    run_len = 0                            # consecutive count of prev_shown
    recent: List[str] = []                 # recent distinct real panels, oldest→newest
    for seg in seq:
        cuts = cuts_by_segment.get(seg) or []
        if not _is_garbage(seg):
            if cuts:
                f = str(cuts[-1].get("file"))
                run_len = run_len + 1 if f == prev_shown else 1
                prev_shown = f
                if f in recent:
                    recent.remove(f)
                recent.append(f)
                del recent[:-3]
            continue
        # preference: the scene being narrated (prev good), then the upcoming
        # scene, then recent scenes newest-first — all story-adjacent.
        prefs: List[str] = []
        for p in (prev_good.get(seg), next_good.get(seg), *reversed(recent)):
            if p and p not in prefs:
                prefs.append(p)
        if not prefs:
            continue                       # no good panel anywhere — keep cut
        top = prefs[0]
        if top != prev_shown or run_len < 2:
            hold = top                     # coherent: stay on the narrated scene
        else:                              # would freeze (>2 in a row) — cycle
            hold = next((p for p in prefs if p != prev_shown), top)
        run_len = run_len + 1 if hold == prev_shown else 1
        prev_shown = hold
        old = str(cuts[0].get("file"))
        dur = round(float(durations.get(seg)
                          or cuts[0].get("dur") or 0.0), 4)
        out[seg] = [{"file": hold, "start": 0.0, "dur": dur, "held": True}]
        subs.append((seg, old, hold))
    return out, subs


def cap_repeats_with_holds(
    cuts_by_segment: Dict[str, List[Dict[str, Any]]],
    *,
    durations: Dict[str, float],
    order: Sequence[str],
    exempt: Optional[set] = None,
    cap: int = 2,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str]]]:
    """A panel may carry at most *cap* segments. Walking the timeline, cuts
    whose file already showed *cap* times are dropped; a segment left with
    nothing HOLDS the previous segment's last panel (held=True) — the
    narrator keeps talking over a held image, the way a human editor covers
    a starved tail, instead of looping panels (IE ch1 alternation). Holds
    are intentional: QA exempts them. sys/doc files (*exempt*) never count."""
    ex = exempt or set()
    out: Dict[str, List[Dict[str, Any]]] = {}
    holds: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {}
    last_idx: Dict[str, int] = {}
    group_shown: Dict[str, set] = {}
    prev_file: Optional[str] = None
    for i, seg in enumerate(order):
        # group key: g####_p## -> g#### (segments outside that scheme are their
        # own group, so the per-group rule below is a no-op for them).
        grp = re.sub(r"_p\d+$", "", str(seg))
        seen = group_shown.setdefault(grp, set())
        cuts = list(cuts_by_segment.get(seg) or [])
        kept: List[Dict[str, Any]] = []
        for c in cuts:
            if c.get("held"):
                kept.append(c)     # already a substitute-hold — pass through
                continue
            f = str(c.get("file"))
            # radius 3 matches QA's 4-segment degenerate window. The single
            # allocation invariant: NO panel — not even an exempt sys/doc card
            # — is re-emitted as a fresh cut inside the window; it HOLDS the
            # previous panel instead (kills the IE ABA-dups, which were all
            # sys/doc panels reappearing 2 segments apart). Exemption relaxes
            # only the GLOBAL cap, so a true system card may still recur far
            # apart (outside the window).
            near = f in last_idx and (i - last_idx[f]) <= 3
            # GROUP-global cap for non-exempt panels: once a panel has been
            # shown in this group it is NOT re-emitted later in the same group,
            # even non-adjacently (gap > radius) — it would otherwise replay
            # with the same animation (IE p000091 idx89&93, p000109 idx106&110).
            # The previous distinct panel HOLDS that slot instead. Exempt
            # system/title cards are unaffected (they may legitimately recur).
            reused = f not in ex and f in seen
            if not near and not reused and (f in ex or counts.get(f, 0) < cap):
                kept.append(c)
                counts[f] = counts.get(f, 0) + 1
                last_idx[f] = i
                if f not in ex:
                    seen.add(f)
        if not kept and cuts:
            if prev_file is None:
                kept = [cuts[0]]            # nothing to hold yet
                f0 = str(cuts[0].get("file"))
                counts[f0] = counts.get(f0, 0) + 1
            else:
                dur = round(float(durations.get(seg) or sum(
                    float(c.get("dur") or 0.0) for c in cuts)), 4)
                kept = [{"file": prev_file, "start": 0.0, "dur": dur,
                         "held": True,
                         # held frame: ONE static shot (no Ken Burns) so a panel
                         # repeated over consecutive segments doesn't restart a
                         # fresh pan each time (the eye-panel-3x bug).
                         "motion": {"mode": "static",
                                    "zoom": {"start": 1.0, "end": 1.0},
                                    "strength": 0.0}}]
                holds.append((seg, prev_file))
        elif kept and len(kept) < len(cuts):
            # SOME-but-not-all cuts dropped: reflow the survivors across the
            # FULL segment window. Without this the survivors keep their original
            # start/dur and the dropped cut's span becomes a NO-CUT time hole —
            # which renders as the #000 background, a black screen (g0003_p06
            # front-gap, g0018_p37 / g0022_p16 tail-gaps). Survivors tile the
            # whole window contiguously: no gap, no overlap. (Same math as
            # _redistribute, applied in-place by identity so a repeated filename
            # inside one segment can't drop the wrong instance.)
            start0 = float(cuts[0].get("start") or 0.0)
            total = sum(float(c.get("dur") or 0.0) for c in cuts)
            surv_total = sum(float(c.get("dur") or 0.0) for c in kept)
            scale = (total / surv_total) if surv_total > 0 else 1.0
            t = start0
            reflowed: List[Dict[str, Any]] = []
            for c in kept:
                d = round(float(c.get("dur") or 0.0) * scale, 4)
                reflowed.append({**c, "start": round(t, 4), "dur": d})
                t += d
            kept = reflowed
        out[seg] = kept
        if kept and not kept[-1].get("held"):
            prev_file = str(kept[-1].get("file"))
    for seg, cuts in cuts_by_segment.items():
        out.setdefault(seg, list(cuts))
    return out, holds


_JUNK_PROMPT = """You are a video editor's eye for a manhwa recap. This image
is ONE cut that would appear on screen for several seconds.

IMPORTANT: every word in text boxes/bubbles is ALREADY READ ALOUD by the
narrator — text alone never justifies screen time. Judge the ARTWORK.

Is the artwork a MEANINGFUL story visual (characters, faces, action,
setting, a styled system-message card) — or JUNK that would look broken on
screen (empty/blanked speech bubbles dominating the frame, a flat gradient/
curtain/glow with no drawn subject even if a small text box sits on it, a
sliver fragment, leftover panel scraps)?
Reply ONLY JSON: {"keep": true/false, "reason": "<short>"}"""


def judge_cut_visuals(files: Sequence[str], clean_dir: str, *,
                      exempt: Optional[set] = None,
                      model: str = "gemma4:26b",
                      cache_path: Optional[str] = None,
                      reuse: bool = False) -> Dict[str, str]:
    """Per-cut VISUAL quality judge — the question no geometric rule fully
    answers ('is this panel worth screen time?'). Returns {file: reason}
    for junk cuts. Fail-soft: no ollama -> judges nothing. sys/doc exempt.

    The verdict is per-PANEL (the artwork), so it is STABLE across heal cycles
    (re-narration changes words, not panels). `cache_path` persists the
    verdicts; `reuse=True` (heal cycles) returns them WITHOUT any model call,
    so a heal cycle no longer re-pays ~one Gemma vision call per shown cut (the
    bulk of render_prep's per-cycle cost). The initial pass (reuse=False) always
    judges fresh and (re)writes the cache, so it never goes stale across runs."""
    ex = exempt or set()
    junk: Dict[str, str] = {}
    cache: Dict[str, Any] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}
    if reuse and cache:
        for f in files:
            if f in ex:
                continue
            v = cache.get(f)
            if isinstance(v, dict) and v.get("keep") is False:
                junk[f] = str(v.get("reason") or "")[:120]
        return junk
    try:
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from ollama_compat import chat as _chat
    except Exception:
        return junk
    new_cache: Dict[str, Any] = {}
    for f in files:
        if f in ex:
            continue
        path = os.path.join(clean_dir, f)
        if not os.path.exists(path):
            continue
        try:
            resp = _chat(model=model, think=False,
                         messages=[{"role": "user", "content": _JUNK_PROMPT,
                                    "images": [path]}],
                         options={"temperature": 0, "num_predict": 150})
            raw = str(resp["message"]["content"] or "")
            m = re.search(r"\{.*\}", raw, re.S)
            v = json.loads(m.group(0)) if m else {}
            keep = v.get("keep")
            new_cache[f] = {"keep": keep,
                            "reason": str(v.get("reason") or "")[:120]}
            if keep is False:
                junk[f] = new_cache[f]["reason"]
        except Exception:
            continue
    if cache_path:
        try:
            with open(cache_path, "w") as _cf:
                json.dump(new_cache, _cf)
        except Exception:
            pass
    return junk


# ---------------------------------------------------------------------------
# plan rewrite (pure)
# ---------------------------------------------------------------------------

_STATIC_MOTION = {"mode": "static", "zoom": {"start": 1.0, "end": 1.0},
                  "strength": 0.0}


# One slow Ken Burns spanning a merged same-image run. The on-screen zoom/pan RATE
# = (delta) / (run duration), so spreading a fixed delta over a longer run makes a
# longer merge move SLOWER. Kept gentle (small zoom + pan) so a held panel drifts.
_MERGE_ZOOM_START = 1.0
_MERGE_ZOOM_END = 1.1
_MERGE_BIAS_START = {"x": 0.3, "y": 0.15}
_MERGE_BIAS_END = {"x": -0.3, "y": -0.15}
_MERGE_STRENGTH = 0.6


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _kenburns_slice(f0: float, f1: float) -> Dict[str, Any]:
    """Motion for the slice [f0, f1] (fractions of a same-image run's total
    duration) of ONE continuous slow Ken Burns. Slicing keeps zoom + pan
    CONTINUOUS across the run's cuts (each slice starts where the previous ended),
    so the repeated image reads as a single slow move, never a restart."""
    return {
        "mode": "kenburns",
        "strength": _MERGE_STRENGTH,
        "ease": "ease_in_out",
        "start_bias": {"x": round(_lerp(_MERGE_BIAS_START["x"], _MERGE_BIAS_END["x"], f0), 4),
                       "y": round(_lerp(_MERGE_BIAS_START["y"], _MERGE_BIAS_END["y"], f0), 4)},
        "end_bias": {"x": round(_lerp(_MERGE_BIAS_START["x"], _MERGE_BIAS_END["x"], f1), 4),
                     "y": round(_lerp(_MERGE_BIAS_START["y"], _MERGE_BIAS_END["y"], f1), 4)},
        "zoom": {"start": round(_lerp(_MERGE_ZOOM_START, _MERGE_ZOOM_END, f0), 4),
                 "end": round(_lerp(_MERGE_ZOOM_START, _MERGE_ZOOM_END, f1), 4)},
    }


def _item_sole_image(item: Dict[str, Any]) -> Optional[str]:
    """The single source image an item shows end-to-end, or None when the item is
    branding, has no cuts, shows a split (file2/layout), or shows more than one
    image — none of those can join a cross-item same-image run."""
    if item.get("branding"):
        return None
    cuts = item.get("cuts") or []
    if not cuts or any(c.get("file2") or c.get("layout") for c in cuts):
        return None
    files = {str(c.get("file") or "") for c in cuts}
    files.discard("")
    return next(iter(files)) if len(files) == 1 else None


def _collapse_same_image_cuts_within_item(cuts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Within ONE item, collapse a maximal run of consecutive same-image cuts into
    a single cut whose dur is the sum, carrying one slow Ken Burns over the full
    span. (A cross-item run can't collapse — each item is its own renderer Sequence
    with its own audio — so that case is handled by continuous slices instead.)"""
    out: List[Dict[str, Any]] = []
    i, n = 0, len(cuts)
    while i < n:
        c = cuts[i]
        f = str(c.get("file") or "")
        if not f or c.get("file2") or c.get("layout"):
            out.append(c)
            i += 1
            continue
        j = i + 1
        while (j < n and str(cuts[j].get("file") or "") == f
               and not cuts[j].get("file2") and not cuts[j].get("layout")):
            j += 1
        run = cuts[i:j]
        if len(run) >= 2:
            total = round(sum(float(x.get("dur") or 0.0) for x in run), 4)
            out.append({**run[0], "file": f, "dur": total, "held": True,
                        "motion": _kenburns_slice(0.0, 1.0)})
        else:
            out.append(c)
        i = j
    return out


def merge_consecutive_same_image_cuts(plan: Dict[str, Any]) -> Dict[str, Any]:
    """AGNOSTIC: when the SAME source image is shown across consecutive cuts, show
    it ONCE with ONE slow Ken Burns spanning the full merged duration — NOT static,
    NOT a re-animated loop, NOT N frozen holds.

    Replaces the earlier static-on-repeat behavior. Two cases:
      - within ONE item: consecutive same-image cuts collapse to a single cut whose
        dur is the sum, with one slow Ken Burns over that duration.
      - across CONSECUTIVE items (the production case: a panel held over several
        per-panel narration segments by cap_repeats_with_holds): each item keeps
        its own audio + duration (timing UNTOUCHED), but the run shares ONE
        continuous slow Ken Burns sliced by cumulative time — so the still image
        pans/zooms slowly and continuously across the whole run instead of
        freezing or restarting. A cut can't span items (each item is its own
        renderer Sequence with its own audio), so the continuous slice is how a
        single slow move is expressed across the merged segments.

    Composes with cap_repeats_with_holds / merge_consecutive_duplicate_narration:
    they supply the held same-image cuts; this then animates the whole run as one
    slow move (it overrides their interim static motion — no double-handling)."""
    tl = (plan or {}).get("timeline") or []
    # 1) within-item collapse (one item with repeated cuts -> one merged cut)
    for item in tl:
        if item.get("branding"):
            continue
        cuts = item.get("cuts") or []
        if len(cuts) >= 2:
            item["cuts"] = _collapse_same_image_cuts_within_item(cuts)
    # 2) cross-item continuous Ken Burns over a run sharing one image
    n = len(tl)
    i = 0
    while i < n:
        img = _item_sole_image(tl[i])
        if img is None:
            i += 1
            continue
        j = i + 1
        while j < n and _item_sole_image(tl[j]) == img:
            j += 1
        if j - i >= 2:
            run = tl[i:j]
            durs = [max(0.0, float((it.get("cuts") or [{}])[0].get("dur")
                                   or it.get("duration_sec") or 0.0)) for it in run]
            total = sum(durs) or float(len(run))
            acc = 0.0
            for it, d in zip(run, durs):
                f0 = acc / total
                acc += d
                cut = (it.get("cuts") or [None])[0]
                if cut is not None:
                    cut["motion"] = _kenburns_slice(f0, acc / total)
        i = j
    return plan


def _norm_tts_text(text: Any) -> str:
    """Normalize a segment's narration for duplicate comparison: drop a leading
    [mood] tag, lowercase, collapse to alphanumeric tokens."""
    s = re.sub(r"^\s*\[[^\]]+\]\s*", "", str(text or "")).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def merge_consecutive_duplicate_narration(plan: Dict[str, Any]) -> Dict[str, Any]:
    """AGNOSTIC: two consecutive timeline segments carrying the SAME narration are
    one spoken line voiced over two panels (the p95/p96 'Ancestor...?' bug).
    Collapse each later duplicate to ONE static held cut of the FIRST segment's
    image, so the repeated line reads as one continuous held shot — never a second
    animated panel, never a re-played pan. (The narration-level dedup upstream
    removes the duplicate at the source; this is the render-side safety net.)
    Branding items reset the run. Empty/whitespace narration never counts as a
    duplicate."""
    prev_text: Optional[str] = None
    prev_img: Optional[str] = None
    for it in (plan or {}).get("timeline") or []:
        if it.get("branding"):
            prev_text, prev_img = None, None
            continue
        text = _norm_tts_text(it.get("tts_text"))
        cuts = it.get("cuts") or []
        cur_img = str((cuts[-1].get("file") if cuts else
                       it.get("primary_scene_file")) or "")
        if text and text == prev_text and prev_img:
            dur = round(float(it.get("duration_sec") or 0.0), 4)
            it["cuts"] = [{"file": prev_img, "start": 0.0, "dur": dur,
                           "held": True, "motion": dict(_STATIC_MOTION)}]
            # prev_text / prev_img unchanged so a 3rd identical line also holds
        else:
            if text:
                prev_text = text
            if cur_img:
                prev_img = cur_img
    return plan


def rewrite_plan(
    plan: Dict[str, Any],
    *,
    scenes_subdir: str,
    scene_dims: Dict[str, Dict[str, int]],
    cuts_by_segment: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    out = json.loads(json.dumps(plan))
    out["scenes_subdir"] = scenes_subdir
    out["scene_dims"] = scene_dims
    for item in out.get("timeline") or []:
        seg = item.get("segment_id")
        if seg in cuts_by_segment:
            item["cuts"] = cuts_by_segment[seg]
    return out


def insert_branding_items(
    plan: Dict[str, Any],
    *,
    intro_dur: float,
    outro_dur: float,
    intro_pad: float = 1.0,
    outro_pad: float = 3.0,
    which: str = "both",
) -> Dict[str, Any]:
    """Insert the channel intro (after the first story beat — the hook plays
    first, then the brand moment over the panel the story paused on) and the
    end-card outro. All later timings shift by the intro length; the renderer
    matches items on ``branding`` and supplies the bundled audio/visuals.
    Zero durations = no-op (assets missing).

    *which*: "both" (single-chapter video, default) | "intro" | "outro" |
    "none" — bundle segments use intro for the FIRST chapter, outro for the
    LAST, none for middles, so a concatenated season carries exactly one
    intro and one outro."""
    out = json.loads(json.dumps(plan))
    tl = out.get("timeline") or []
    if not tl:
        return out

    if which not in ("both", "intro", "outro", "none"):
        raise ValueError(f"branding which={which!r}")
    # channel decision (2026-06-15): NO intro on any video — videos open on the
    # story, outro only. The intro arg + the "intro"/"both" modes are kept for
    # compat but never insert an intro; "intro" (a bundle's first segment) now
    # carries no branding at all.
    intro_dur = 0.0
    if which in ("none", "intro"):
        outro_dur = 0.0

    new_tl: List[Dict[str, Any]] = list(tl)
    if intro_dur > 0:
        first = tl[0]
        d = round(intro_dur + intro_pad, 4)
        cuts = first.get("cuts") or []
        hold_file = str(cuts[-1].get("file")) if cuts else str(first.get("primary_scene_file") or "")
        intro_item = {
            "segment_id": "branding_intro",
            "branding": "intro",
            "start_sec": first["end_sec"],
            "duration_sec": d,
            "end_sec": round(float(first["end_sec"]) + d, 4),
            "cuts": [{"file": hold_file, "start": 0.0, "dur": d}] if hold_file else [],
        }
        new_tl = [first, intro_item]
        for item in tl[1:]:
            it = dict(item)
            it["start_sec"] = round(float(item["start_sec"]) + d, 4)
            it["end_sec"] = round(float(item["end_sec"]) + d, 4)
            new_tl.append(it)

    if outro_dur > 0:
        last_end = float(new_tl[-1]["end_sec"])
        d = round(outro_dur + outro_pad, 4)
        new_tl.append({
            "segment_id": "branding_outro",
            "branding": "outro",
            "start_sec": round(last_end, 4),
            "duration_sec": d,
            "end_sec": round(last_end + d, 4),
            "cuts": [],
        })

    out["timeline"] = new_tl
    out["total_duration_sec"] = float(new_tl[-1]["end_sec"])
    return out


SPEECH_MODES = {"spoken", "shout", "inner_thought"}


def speech_mode_files(beats_obj: Dict[str, Any]) -> set:
    """Scene files Gemini classified as SPEECH panels (bubble_mode spoken/
    shout/inner_thought). On these, a system_box detection is presumed a false
    positive and must not shield the speech bubbles from text cleaning; real
    system windows live on panels Gemini saw as none/narration."""
    out: set = set()
    for b in beats_obj.get("beats") or []:
        for e in b.get("scene_selection") or []:
            if str(e.get("bubble_mode") or "").strip().lower() in SPEECH_MODES:
                out.add(str(e.get("scene_file") or ""))
    out.discard("")
    return out


def _wav_duration_sec(path: str) -> float:
    import wave
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_bubble_detector(device: str):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "manhwa-cropper"))
    from manhwa_cropper.detectors.bubbles import BubbleDetector
    return BubbleDetector(device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--scenes-manifest", required=True)
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--vision-manifest", default="",
                    help="manifest.vision.json — its text_coverage/text_only feed "
                         "the bubble-dominance gate (default: <episode>/manifest.vision.json)")
    ap.add_argument("--out-plan", default="", help="default: <plan>.clean.json next to --plan")
    # 0.20: edge-clipped/small bubbles score low, and false positives are
    # harmless by construction (no white/black interior -> untouched).
    ap.add_argument("--bubble-conf", type=float, default=0.20)
    ap.add_argument("--no-bubbles", action="store_true", help="skip bubble inpainting")
    ap.add_argument("--reuse-clean", action="store_true",
                    help="heal-cycle fast path: reuse the cached per-cut visual "
                         "judge verdicts (panels are unchanged between heal "
                         "cycles) instead of re-paying the Gemma vision pass")
    ap.add_argument("--no-trim", action="store_true", help="skip border trimming")
    ap.add_argument("--no-branding", action="store_true",
                    help="skip channel intro/outro insertion (alias for "
                         "--branding none)")
    ap.add_argument("--branding", choices=["both", "intro", "outro", "none"],
                    default="both",
                    help="bundle segments: first chapter=intro, last=outro, "
                         "middles=none; default both (single-chapter video)")
    ap.add_argument("--no-split", action="store_true",
                    help="skip splitting over-merged crops on white bands")
    ap.add_argument("--series-title", default="",
                    help="series title for cover/title-page chrome detection")
    ap.add_argument("--min-art-score", type=float, default=0.012,
                    help="cuts whose CLEANED panel has less edge detail than "
                         "this are dropped (empty-bubble husks)")
    ap.add_argument("--panel-weights",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "assets", "models",
                        "webtoon_panels.pt"),
                    help="trained webtoon YOLO — its system_box class protects "
                         "system-message panels from the bubble gate/blanking")
    ap.add_argument("--branding-dir",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "assets", "branding", "origin-power"),
                    help="dir holding intro.wav / outro.wav (channel constants)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)
    with open(args.scenes_manifest, "r", encoding="utf-8") as f:
        scenes_m = json.load(f)

    geom: Dict[str, Dict[str, float]] = {}
    for s in scenes_m.get("scenes") or []:
        box = s.get("box_px_xyxy") or [0, 0, 0, 0]
        gy0 = float(s.get("chunk_global_y0") or 0.0)
        geom[str(s.get("out_file"))] = {
            "x1": float(box[0]), "y1": gy0 + float(box[1]),
            "x2": float(box[2]), "y2": gy0 + float(box[3]),
        }

    # vision text metrics — the EXISTING "text domain" measurement: a panel
    # that is text_only or mostly OCR text is as bad on screen as a bubble blob.
    vision_path = args.vision_manifest or os.path.join(args.episode_dir, "manifest.vision.json")
    text_score: Dict[str, float] = {}
    vision_item: Dict[str, Dict[str, Any]] = {}
    word_boxes_by_file: Dict[str, List[Tuple[int, int, int, int]]] = {}
    if os.path.exists(vision_path):
        with open(vision_path, "r", encoding="utf-8") as f:
            for it in json.load(f).get("items") or []:
                sf = str(it.get("scene_file") or "")
                tc = float(it.get("text_coverage") or 0.0)
                text_score[sf] = 1.0 if it.get("text_only") else tc
                vision_item[sf] = {"ocr_clean": it.get("ocr_clean"),
                                   "text_only": it.get("text_only"),
                                   "text_coverage": tc,
                                   # carry the understanding's verdict so the
                                   # is_chrome_scene chokepoint defers to it
                                   "panel_kind": it.get("panel_kind"),
                                   # + the subjects, so an in-world screen the
                                   # understanding rescued chrome->story keeps
                                   # its on-screen text (see _is_inworld_screen)
                                   "subjects": it.get("subjects") or []}
                w = float(it.get("width") or 0)
                h = float(it.get("height") or 0)
                if w > 0 and h > 0:
                    word_boxes_by_file[sf] = [
                        (int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h))
                        for wd in ((it.get("vision") or {}).get("ocr_words") or [])
                        for b in [wd.get("bbox") or []]
                        if len(b) == 4
                    ]

    beats_path = os.path.join(args.episode_dir, "manifest.beats.json")
    speech_files: set = set()
    if os.path.exists(beats_path):
        with open(beats_path, "r", encoding="utf-8") as f:
            speech_files = speech_mode_files(json.load(f))

    # Panels the understanding labeled a SYSTEM card (an in-world notification /
    # status window — Nano ch1 p000114 "7TH GENERATION NANO MACHINE"). Their TEXT
    # is the on-screen story beat, so they are kept + shown UNCONDITIONALLY: never
    # husk-dropped (exempt_from_drop), never seam/visual deduped away (protect),
    # never bubble-blanked (_cleaned), never sent to the visual judge (sysf). The
    # empty DIALOGUE bubble is a different panel_kind (caption), excluded upstream.
    system_files = {f for f, v in vision_item.items()
                    if str(v.get("panel_kind") or "").strip().lower() == "system"}

    scenes_dir = os.path.join(args.episode_dir, "scenes")
    img_cache: Dict[str, Optional[np.ndarray]] = {}

    def _img(fname: str) -> Optional[np.ndarray]:
        if fname not in img_cache:
            img_cache[fname] = cv2.imread(os.path.join(scenes_dir, fname))
        return img_cache[fname]

    detector = None
    if not args.no_bubbles:
        detector = _load_bubble_detector(args.device)
    boxes_cache: Dict[str, List[Tuple[int, int, int, int]]] = {}

    def _boxes(fname: str) -> List[Tuple[int, int, int, int]]:
        if detector is None:
            return []
        if fname not in boxes_cache:
            img = _img(fname)
            boxes_cache[fname] = [] if img is None else [
                (int(x1), int(y1), int(x2), int(y2))
                for (x1, y1, x2, y2, _s) in detector.detect(
                    img, imgsz=1024, conf=args.bubble_conf)
            ]
        return boxes_cache[fname]

    # system_box detections from OUR trained model (works on crops, mAP .843):
    # they veto both the dominance gate and text blanking. Fail-soft when the
    # weights are missing — protection off, loudly.
    panel_model = None
    if os.path.exists(args.panel_weights):
        from ultralytics import YOLO
        panel_model = YOLO(args.panel_weights)
    else:
        print(f"[warn] panel weights missing ({args.panel_weights}) — "
              "system-message protection DISABLED")
    sys_cache: Dict[str, List[Tuple[int, int, int, int]]] = {}

    def _sys_boxes(fname: str) -> List[Tuple[int, int, int, int]]:
        if panel_model is None:
            return []
        if fname not in sys_cache:
            img = _img(fname)
            out: List[Tuple[int, int, int, int]] = []
            if img is not None:
                r = panel_model.predict(img, conf=0.30, device=args.device, verbose=False)[0]
                if r.boxes is not None:
                    for (x1, y1, x2, y2), c in zip(
                            r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                        if int(c) == 1:  # system_box
                            out.append((int(x1), int(y1), int(x2), int(y2)))
            sys_cache[fname] = out
        return sys_cache[fname]

    from scene_chrome import is_chrome_scene, needs_image_stats  # sibling tool

    # cleaned-image cache: cleaning result is needed BOTH by the blankness
    # gate (what does the viewer see after text removal?) and the writer.
    cleaned_cache: Dict[str, Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int]]]] = {}

    def _text_rich(fname: str) -> bool:
        words = word_boxes_by_file.get(fname, [])
        img = _img(fname)
        panel_w = img.shape[1] if img is not None else 0
        return doc_like(text_score.get(fname, 0.0), len(words), words,
                        speech_shaped_boxes(_boxes(fname), panel_w))

    def _panel_kind(fname: str) -> str:
        return str(vision_item.get(fname, {}).get("panel_kind") or "").strip().lower()

    def _is_inworld_screen(fname: str) -> bool:
        """An in-world device/app screen the understanding rescued chrome->story
        (panel_understand stamps subjects=['an in-world screen']): its on-screen
        text IS the story content — an episode list, a feed (ORV ep1 p000003,
        the "no one reads it" webnovel list). Treat it like a document: keep
        that text, blank only the speech bubble(s) over it. doc_like can't see
        it because the detector mis-boxes the UI rows as bubbles, so the screen
        looks dialogue-dominated; the understanding's marker is the reliable
        signal."""
        subj = vision_item.get(fname, {}).get("subjects") or []
        return any("in-world screen" in str(s).lower() for s in subj)

    def _is_title_card(fname: str) -> bool:
        """Styled title/system card (SYSTEM ACTIVATION., AGE: 3 YEARS) — short
        mostly-caps phrase on a flat (white/black) frame. These
        are story beats: the timeline protects them from the LLM's 'redundant'
        verdict, and render_prep must NOT then drop them as low-art text.
        Same signal as prep_qa/timeline_planner."""
        vit = vision_item.get(fname, {})
        ocr = str(vit.get("ocr_clean") or "").strip()
        if is_chrome_scene(vit, series_title=args.series_title or None):
            return False
        if empty_bubble_panel(vit):
            return False
        if not ocr or "..." in ocr or any(c in ocr for c in "~!?"):
            return False
        words = [w for w in re.split(r"[^A-Za-z0-9']+", ocr)
                 if any(c.isalpha() for c in w)]
        letters = [c for c in ocr if c.isalpha()]
        if not (2 <= len(words) <= 8) or not letters:
            return False
        if sum(c.isupper() for c in letters) / len(letters) < 0.8:
            return False
        if float(vit.get("text_coverage") or 0.0) >= 0.20:
            return False
        img = _img(fname)
        if img is None:
            return False
        g = img.mean(axis=2)
        return float(((g > 235) | (g < 25)).mean()) >= 0.6

    def _cleaned(fname: str) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        if fname not in cleaned_cache:
            img = _img(fname)
            if img is None:
                cleaned_cache[fname] = (None, [])
            elif ((_is_title_card(fname) or _panel_kind(fname) == "system")
                  and fname not in speech_files):
                # title/system card: the styled text IS the content (SKY
                # CORPORATION, age cards, the "7TH GENERATION NANO MACHINE"
                # notification) — never blank it, or the card ships empty
                cleaned_cache[fname] = (img.copy(), [])
            elif _is_inworld_screen(fname) or (_text_rich(fname) and fname not in speech_files):
                # DOCUMENT panel (word-rich, no speech per Gemini) OR a rescued
                # in-world screen (episode list / feed): its on-screen text IS
                # the content and must survive — but a speech bubble floating
                # OVER it (ORV p000025 stats page, p000003 reader comment) is
                # dialogue. An in-world screen takes this path even in speech
                # mode: its comment IS voiced (so the bubble blanks below) but
                # the SCREEN behind it is the story and must be kept — the
                # speech_files guard only applies to plain document panels.
                # like any other: blank ONLY words inside speech-SHAPED
                # boxes; UI rows (wide flat detector boxes) and all
                # outside-bubble text stay untouched. No orphan pass here.
                sboxes = speech_shaped_boxes(
                    _boxes(fname), img.shape[1])
                words = word_boxes_by_file.get(fname) or []
                grown = [(x1 - 6, y1 - 6, x2 + 6, y2 + 6)
                         for (x1, y1, x2, y2) in sboxes]

                def _in_speech(wr):
                    wx1, wy1, wx2, wy2 = wr
                    wa = max(1, (wx2 - wx1) * (wy2 - wy1))
                    for (bx1, by1, bx2, by2) in grown:
                        ix = max(0, min(wx2, bx2) - max(wx1, bx1))
                        iy = max(0, min(wy2, by2) - max(wy1, by1))
                        if ix * iy >= 0.5 * wa:
                            return True
                    return False

                inwords = [w for w in words if _in_speech(w)]
                out = (clean_scene_image(img.copy(), sboxes, text_boxes=inwords)
                       if (sboxes and inwords) else img.copy())
                cleaned_cache[fname] = (out, [])
            else:
                protected = [] if fname in speech_files else _sys_boxes(fname)
                boxes = filter_protected_boxes(_boxes(fname), protected)
                words = list(word_boxes_by_file.get(fname) or [])
                if protected and words:
                    # words inside protected system boxes are KEPT text — the
                    # orphan-word path must never see (and blank) them
                    def _in_protected(wr):
                        wx1, wy1, wx2, wy2 = wr
                        wa = max(1, (wx2 - wx1) * (wy2 - wy1))
                        for (bx1, by1, bx2, by2) in protected:
                            ix = max(0, min(wx2, bx2) - max(wx1, bx1))
                            iy = max(0, min(wy2, by2) - max(wy1, by1))
                            if ix * iy >= 0.5 * wa:
                                return True
                        return False
                    words = [w for w in words if not _in_protected(w)]
                # orphan-word blanking needs the cleaner even with zero
                # detected bubbles (spiky balloons evade the detector) — BUT a
                # STORY-art panel keeps its drawn text (clean_panel_image, the D3
                # husk fix): blanking a bubble on real artwork leaves an empty
                # white husk. caption/system/doc panels are still blanked.
                story = _panel_kind(fname) == "story" and fname not in system_files
                out = clean_panel_image(img, _panel_kind(fname), boxes,
                                        text_boxes=words)
                cleaned_cache[fname] = (out, [] if story else boxes)
        return cleaned_cache[fname]

    # Chrome is decided at the single chokepoint (scene_chrome.is_chrome_scene),
    # which now defers to the understanding's panel_kind (carried on vision_item).
    # So a 'story' panel is never scored as chrome here — no per-module exempt set
    # is needed, and genuine husks/blanks are still dropped on their own merits.

    # 1. drop bad cuts per shot — seam duplicates (geometric, then VISUAL
    # containment), then bubble/text-dominated panels, then CHROME
    # (publisher/cover/counter pages) and post-clean HUSKS (panels with no
    # art detail left once their bubbles are emptied).
    cuts_by_segment: Dict[str, List[Dict[str, Any]]] = {}
    all_dropped: List[str] = []
    cov_all: Dict[str, float] = {}
    exempt_all: set = set()
    for item in plan.get("timeline") or []:
        cuts = item.get("cuts") or []
        new_cuts, dropped = drop_contained_duplicate_cuts(
            cuts, geom, protect=system_files)
        if len(new_cuts) > 1:
            imgs = {str(c["file"]): _img(str(c["file"])) for c in new_cuts}
            imgs = {k: v for k, v in imgs.items() if v is not None}
            new_cuts, vdropped = drop_visual_duplicate_cuts(
                new_cuts, imgs, protect=system_files)
            dropped = list(dropped) + vdropped
            # near-identical SAME-SIZE pair (the 'reaction face with ?' repeat):
            # the containment filter above only catches small-in-big seam dups.
            if len(new_cuts) > 1:
                imgs = {k: v for k, v in imgs.items()
                        if k in {str(c["file"]) for c in new_cuts}}
                new_cuts, ndropped = drop_near_identical_cuts(
                    new_cuts, imgs, protect=system_files)
                dropped = list(dropped) + ndropped
                if ndropped:
                    print(f"[ok] {item.get('segment_id')}: "
                          f"near_identical_dropped={ndropped}")
        if new_cuts:
            cov: Dict[str, float] = {}
            exempt: set = set()
            for c in new_cuts:
                f = str(c["file"])
                img = _img(f)
                bub = bubble_coverage(img.shape, _boxes(f)) if img is not None else 0.0
                score = max(bub, text_score.get(f, 0.0))
                vit = vision_item.get(f, {})
                mid = None
                if img is not None and needs_image_stats(
                        str(vit.get("ocr_clean") or "")):
                    g = img.mean(axis=2)
                    mid = float(((g > 60) & (g < 200)).mean())
                if empty_bubble_panel(vit):
                    score = 1.0  # understanding says no story art; cover it
                elif is_chrome_scene(vit, series_title=args.series_title or None,
                                   midtone_frac=mid):
                    score = 1.0  # chrome (per the understanding-aware chokepoint)
                else:
                    visual_story = story_visual_panel(vit)
                    cimg, cboxes = _cleaned(f)
                    rich = _text_rich(f)
                    recoverable = (cimg is None) or panel_recoverable(
                        cimg, cboxes, min_art_score=args.min_art_score,
                        text_rich=rich)
                    # Deterministic empty-bubble husk: blanked bubbles DOMINATE the
                    # frame and the panel carried NO text (a curtain/gradient with
                    # empty outlines, no drawn subject — IE p000010, which a faint
                    # gradient lets sneak past the art-score). Can't over-drop: a
                    # real atmospheric shot has no bubbles (coverage ~0); a real
                    # dialogue panel carried text (text_coverage > 0).
                    if (cimg is not None and cboxes
                            and bubble_coverage(cimg.shape, cboxes) >= 0.20
                            and float(vit.get("text_coverage") or 0.0) <= 0.02):
                        recoverable = False
                    if visual_story:
                        recoverable = True
                    if not recoverable:
                        score = 1.0  # no recoverable region after cleaning
                    # System / title / document cards are story beats whose TEXT is
                    # the content — it sits on a flat card, NOT in an inpainted
                    # bubble, so it survives cleaning. But the system-card / title-
                    # card protection must NOT shield a CONTENTLESS HUSK (an empty
                    # bubble blanked to a plain background): those protections apply
                    # only when the panel is still recoverable, so a sys-tagged husk
                    # drops and a real neighbour holds its place (exempt_from_drop).
                    sys_box = bool(img is not None
                                   and bubble_coverage(img.shape, _sys_boxes(f)) >= 0.02)
                    if exempt_from_drop(
                            recoverable=recoverable, sys_box=sys_box,
                            title_card=_is_title_card(f), rich=rich,
                            visual_story=visual_story,
                            panel_kind=vit.get("panel_kind"),
                            has_ocr=bool(str(vit.get("ocr_clean") or "").strip())):
                        exempt.add(f)
                cov[f] = score
            new_cuts, bdropped = drop_bubble_dominated_cuts(new_cuts, cov, exempt=exempt)
            dropped = list(dropped) + bdropped
            cov_all.update(cov)
            exempt_all |= exempt
        cuts_by_segment[item["segment_id"]] = new_cuts
        all_dropped.extend(dropped)

    # consecutive shown cuts must differ — the artist's blow-up/repeat panels
    # land in NEIGHBORING segments and the per-segment dedup never sees them.
    # Substitution can CREATE new adjacencies, so iterate to a fixpoint.
    order = [str(it.get("segment_id")) for it in plan.get("timeline") or []]
    durations = {str(it.get("segment_id")): float(it.get("duration_sec") or 0.0)
                 for it in plan.get("timeline") or []}

    # compare what the WRITER will emit: margins dilute template matching,
    # so trim first (the keyboard pair only matches post-trim)
    trimmed_cache: Dict[str, Optional[np.ndarray]] = {}

    def _trimmed_clean(f: str) -> Optional[np.ndarray]:
        if f not in trimmed_cache:
            img = _cleaned(f)[0]
            if img is not None and not args.no_trim:
                tx1, ty1, tx2, ty2 = content_bbox(img)
                img = img[ty1:ty2, tx1:tx2]
            trimmed_cache[f] = img
        return trimmed_cache[f]

    for _round in range(3):
        cuts_by_segment, xdropped = drop_cross_segment_duplicate_cuts(
            cuts_by_segment, order, _trimmed_clean, thresh=0.84,
            coverage_by_file=cov_all, exempt=exempt_all, protect=system_files)
        for seg, f in xdropped:
            sole = (len(cuts_by_segment[seg]) == 1
                    and str(cuts_by_segment[seg][0]["file"]) == f)
            print(f"[ok] {seg}: cross-segment duplicate {f}"
                  + (" -> forcing substitution" if sole else " dropped"))
            if sole:
                cov_all[f] = 1.0      # sole survivor is a dup
                exempt_all.discard(f)
            else:
                all_dropped.append(f)

        # sole-cut segments whose survivor is hard garbage (chrome cover,
        # husk, cross-segment duplicate) show the nearest kept story panel
        cuts_by_segment, subs = substitute_garbage_sole_cuts(
            cuts_by_segment, cov_all, durations=durations, exempt=exempt_all,
            order=order)
        for seg, old, new in subs:
            all_dropped.append(old)
            print(f"[ok] {seg}: garbage sole cut {old} -> SUBSTITUTED {new}")
        if not xdropped and not subs:
            break

    shown = sorted({c["file"] for cs in cuts_by_segment.values() for c in cs})

    # 2+3. clean + trim shown scenes into scenes_clean/
    clean_dir = os.path.join(args.episode_dir, "scenes_clean")
    os.makedirs(clean_dir, exist_ok=True)

    scene_dims: Dict[str, Dict[str, int]] = {}
    split_map: Dict[str, Tuple[str, str]] = {}
    bubbles_cleaned = 0

    def _write_part(name: str, part: np.ndarray, doc: bool = False,
                    sys_panel: bool = False, blanked: bool = False) -> None:
        if not args.no_trim:
            tx1, ty1, tx2, ty2 = content_bbox(part)
            part = part[ty1:ty2, tx1:tx2]
        cv2.imwrite(os.path.join(clean_dir, name), part,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        ph, pw = part.shape[:2]
        # doc: document/UI panels — the renderer must never cover-crop their
        # text (full-bleed) and never scroll them; contain-fit only.
        # sys/blanked: QA metadata — system-message panels keep their text by
        # design; blanked panels had bubble text removed (narration replaces it)
        scene_dims[name] = {"w": int(pw), "h": int(ph), "doc": bool(doc),
                            "sys": bool(sys_panel), "blanked": bool(blanked)}

    for fname in shown:
        img, boxes = _cleaned(fname)
        if img is None:
            print(f"[warn] unreadable scene, kept original reference: {fname}")
            continue
        img = img.copy()
        bubbles_cleaned += len(boxes)

        # over-merged crops: dead-box recrop first (blank caption voids, #22),
        # then split at wide white voids; parts that are just floating
        # (now-empty) bubbles are discarded, two real parts render side by
        # side, a single real part crops the void away entirely.
        # Document-like panels (text-rich) are never recropped or split.
        rich = _text_rich(fname)
        orig = _img(fname)
        sysf = bool((orig is not None
                     and bubble_coverage(orig.shape, _sys_boxes(fname)) >= 0.02)
                    or _is_title_card(fname)
                    or _panel_kind(fname) == "system")   # sys cards are protected
        blanked = bool(boxes) or (not rich and not sysf
                                  and bool(word_boxes_by_file.get(fname)))
        parts, pinfo = select_panel_crops(img, boxes, text_rich=rich,
                                          no_split=args.no_split)
        if pinfo.get("recropped"):
            print(f"[ok] {fname}: DEAD-BOX recrop "
                  f"blank_frac={pinfo['blank_box_frac']:.2f}")
        if len(parts) == 2:
            stem, ext = os.path.splitext(fname)
            names = (f"{stem}_a{ext}", f"{stem}_b{ext}")
            for nm, part in zip(names, parts):
                _write_part(nm, part, doc=rich, sys_panel=sysf, blanked=blanked)
            split_map[fname] = names
            print(f"[ok] {fname}: SPLIT -> {names[0]} + {names[1]} (split2)")
            continue

        _write_part(fname, parts[0], doc=rich, sys_panel=sysf, blanked=blanked)
        print(f"[ok] {fname}: bubbles={len(boxes)} -> "
              f"{scene_dims[fname]['w']}x{scene_dims[fname]['h']}")

    # AI visual judge on the CLEANED cuts (voids only exist post-blanking):
    # junk (empty-bubble husks, flat glows, slivers) is DROPPED; the repeat
    # cap then refills/holds. The judge that asks what no geometry can:
    # "is this panel worth screen time?"
    # For SPLIT panels, judge each written HALF (_a/_b): the original filename
    # is never written to scenes_clean/, so judging by it skipped split panels
    # entirely and let a junk gradient/husk half survive (g0026 p044_b).
    judged: List[str] = []
    for f in shown:
        judged.extend(split_map.get(f, (f,)))
    junk = judge_cut_visuals(
        [f for f in judged
         if not (scene_dims.get(f) or {}).get("sys")
         and not (scene_dims.get(f) or {}).get("doc")],
        clean_dir, exempt=exempt_all,
        cache_path=os.path.join(clean_dir, ".cut_judge_cache.json"),
        reuse=args.reuse_clean)
    # operator drops: one click on the dashboard bans a panel for good
    mdp = os.path.join(args.episode_dir, "manual_drops.json")
    if os.path.exists(mdp):
        try:
            with open(mdp, "r", encoding="utf-8") as fh:
                for f in json.load(fh) or []:
                    junk[str(f)] = "operator drop (dashboard)"
        except Exception:
            pass
    def _cut_is_junk(f: str) -> bool:
        # drop a cut when its file is junk (single panel or operator-dropped
        # original), or when BOTH split halves are junk; a single junk half
        # collapses the split to the survivor (handled in the split pass below)
        if f in junk:
            return True
        if f in split_map:
            a, b = split_map[f]
            return a in junk and b in junk
        return False

    if junk:
        for f, why in sorted(junk.items()):
            print(f"[ok] visual judge: DROPPING {f} — {why}")

        def _drop_junk_cuts(cs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            # Redistribute the freed time so a judge-dropped cut never leaves a
            # BLACK GAP (the survivors re-spread to fill the voiceover-locked
            # group window) — same contract as the seam/husk drop passes. Before
            # this, the judge drop alone was the one path that removed a cut
            # WITHOUT reflowing, so a mid-group drop left a hole (Nano g0001
            # p000003 -> 3.6s black at 7.3-10.9s).
            junk_files = [str(c.get("file")) for c in cs
                          if _cut_is_junk(str(c.get("file")))]
            if not junk_files:
                return cs
            survivors = [c for c in cs
                         if not _cut_is_junk(str(c.get("file")))]
            if not survivors:
                return cs   # whole segment is junk — holds/substitution cover it
            return _redistribute(cs, junk_files)

        cuts_by_segment = {seg: _drop_junk_cuts(cs)
                           for seg, cs in cuts_by_segment.items()}

    # repeat cap + holds (also covers segments emptied by the judge — their
    # neighbor's panel holds while the narration continues)
    cuts_by_segment, holds = cap_repeats_with_holds(
        cuts_by_segment, durations=durations, order=order,
        exempt=exempt_all, cap=2)
    for seg, f in holds:
        print(f"[ok] {seg}: repeat cap -> HOLDING previous panel {f}")

    # split scenes render side-by-side — but if the judge killed ONE half,
    # collapse to the surviving half (drops the junk gradient/husk half, e.g.
    # g0026 p044_b) instead of rendering a broken split
    for cs in cuts_by_segment.values():
        for c in cs:
            f = str(c.get("file"))
            if f not in split_map:
                continue
            a, b = split_map[f]
            a_junk, b_junk = a in junk, b in junk
            if a_junk and not b_junk:
                c["file"] = b
                c.pop("file2", None); c.pop("layout", None)
            elif b_junk and not a_junk:
                c["file"] = a
                c.pop("file2", None); c.pop("layout", None)
            else:
                c["file"], c["file2"] = a, b
                c["layout"] = "split2"

    out_plan = rewrite_plan(plan, scenes_subdir="scenes_clean",
                            scene_dims=scene_dims,
                            cuts_by_segment=cuts_by_segment)
    # consecutive segments with the SAME narration -> hold the first image (the
    # p95/p96 dup); then collapse ANY consecutive same-image run (held or planned)
    # into ONE slow Ken Burns spanning the merged duration (audio/timing intact).
    out_plan = merge_consecutive_duplicate_narration(out_plan)
    out_plan = merge_consecutive_same_image_cuts(out_plan)

    outro_dur = 0.0
    which = "none" if args.no_branding else args.branding
    if which != "none":
        # NO intro on any video (channel decision) — only the outro is read/added.
        outro_dur = _wav_duration_sec(os.path.join(args.branding_dir, "outro.wav"))
        out_plan = insert_branding_items(out_plan, intro_dur=0.0,
                                         outro_dur=outro_dur, which=which)

    out_path = args.out_plan or (os.path.splitext(args.plan)[0] + ".clean.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_plan, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote={out_path} shown={len(shown)} "
          f"seam_dups_dropped={sorted(set(all_dropped))} bubbles_inpainted={bubbles_cleaned} "
          f"branding=outro:{outro_dur:.1f}s (no intro) "
          f"total={out_plan.get('total_duration_sec', 0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
