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

# `from studio...` must work even when spawned as a bare script without
# PYTHONPATH (the worker does; pipeline._run_tool sets it) — same bootstrap
# prep_qa.py uses.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from manifest_io import write_manifest  # noqa: E402


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


def _near_identical_similarity(a: np.ndarray, b: np.ndarray, *, size: int = 64,
                               boxes_a: Sequence[Tuple[int, int, int, int]] = (),
                               boxes_b: Sequence[Tuple[int, int, int, int]] = ()) -> float:
    """Full-image similarity in [0,1] for two SIMILAR-SIZED panels.

    Both images are downscaled to a fixed *size*x*size* grayscale grid (so a
    few px of size mismatch don't matter) and compared with normalized
    cross-correlation. NCC keys on STRUCTURE, not absolute brightness, so a
    global tone shift between two genuinely-different panels never reads as a
    match; only the same drawing, barely changed, scores near 1.0. Returns 0.0
    when either image is featureless (flat) — a zero-variance NCC is undefined
    and would spuriously match every other flat panel.

    *boxes_a*/*boxes_b* (optional): detected bubble boxes to neutralize
    (`_mask_bubbles_for_hash`) before the NCC. Cleaning removes only the text
    inside a bubble — the OUTLINE stays — so identical art under different
    dialogue scores below the gate; masking the bubbles pushes it back to ~1.0.
    """
    if len(boxes_a):
        a = _mask_bubbles_for_hash(a, boxes_a)
    if len(boxes_b):
        b = _mask_bubbles_for_hash(b, boxes_b)

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
    boxes_by_file: Optional[Dict[str, Sequence[Tuple[int, int, int, int]]]] = None,
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
    *protect* files (system cards) are never dropped. *boxes_by_file* (optional):
    per-file bubble boxes, masked before the NCC so identical art under different
    dialogue bubbles (the assassin p054/p055 pair) still scores as a near-dup.
    """
    prot = protect or set()
    bbf = boxes_by_file or {}
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
            if _near_identical_similarity(
                    a, b, boxes_a=bbf.get(fi, ()), boxes_b=bbf.get(fj, ())) >= thresh:
                dropped.append(fj)  # keep the earlier cut, drop the later
    return _redistribute(cuts, dropped), dropped


def _dhash8_bgr(img: np.ndarray,
                boxes: Sequence[Tuple[int, int, int, int]] = ()) -> int:
    """8x8 difference hash of a BGR (cv2) image — the perceptual hash for the
    cross-segment near-identical drop below. Shift-tolerant where NCC is not: the
    source-repeated eye p090/p095 (a re-drawn crop, not a pixel copy) measures
    hamming 3 here but only 0.88 NCC (below the 0.96 near-identical gate), while
    the pipeline's dhash64 gives 24 (it is tuned for exact chunk-overlap dups).
    Distinct panels score 38-40 and the flash-bisection halves 23 — both far above
    the drop threshold, so the pass never touches them.

    *boxes* (optional): detected bubble boxes to neutralize
    (`_mask_bubbles_for_hash`) before hashing, so two identical drawings that
    differ ONLY in their dialogue bubbles (whose outlines survive text cleaning)
    hash the same instead of splitting apart."""
    if len(boxes):
        img = _mask_bubbles_for_hash(img, boxes)
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    diff = g[:, 1:] > g[:, :-1]
    out = 0
    for bit in diff.flatten():
        out = (out << 1) | int(bit)
    return out


def _mask_bubbles_for_hash(
    img: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Neutralize each detected speech/caption bubble so its OUTLINE and SHAPE
    stop perturbing a perceptual hash / NCC. Cleaning removes only the text
    INSIDE a bubble — the outline stays, by design (`clean_scene_image`) — so two
    panels of identical art with different dialogue bubbles hash DIFFERENTLY and
    slip the dedup ladder (the assassin p054/p055, crying p104/p105 pairs). For
    each box we fill it: the ring-median flat-surround fill when the box sits on a
    uniform void (`_flat_surround_fill`), else a Telea inpaint over the box so the
    region blends into its artwork neighbourhood.

    HASH-ONLY — the result is NEVER written to disk, so the pipeline rule "clean
    text only, never inpaint the shown image" does not apply here. Empty *boxes*
    is a no-op (returns the input unchanged)."""
    if img is None or not len(boxes):
        return img
    out = img.copy()
    h, w = out.shape[:2]
    inpaint_mask: Optional[np.ndarray] = None
    for rect in boxes:
        x1, y1, x2, y2 = (int(v) for v in rect)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        fill = _flat_surround_fill(out, (x1, y1, x2, y2))
        if fill is not None:
            out[y1:y2, x1:x2] = fill          # flat void: match the surround tone
        else:
            if inpaint_mask is None:
                inpaint_mask = np.zeros((h, w), np.uint8)
            inpaint_mask[y1:y2, x1:x2] = 255  # art surround: inpaint below
    if inpaint_mask is not None:
        src = out if out.dtype == np.uint8 else np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.inpaint(src, inpaint_mask, 3, cv2.INPAINT_TELEA)
    return out


def normalized_ocr_text(s: Any) -> str:
    """Lowercase, collapse everything non-alphanumeric — the normalization the
    OCR-containment twin test compares under (OCR of the same dialogue differs
    in punctuation/casing between panels)."""
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def ocr_dialogue_contained(a: Any, b: Any) -> bool:
    """One panel's OCR dialogue, normalized, is a substring of the other's —
    the artist "echo" signature (the p000054/p000055 pair repeats the tail of
    the previous panel's line in a re-drawn close-up). Both sides must carry
    text and the contained side must be substantial (>= 8 chars normalized), so
    an empty/`...`/interjection panel never trivially "contains" into
    everything."""
    na, nb = normalized_ocr_text(a), normalized_ocr_text(b)
    if not na or not nb:
        return False
    small, big = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(small) >= 8 and small in big


def twin_verdict(ham: int, ocr_a: Any = "", ocr_b: Any = "", *,
                 ham_max: int = 8, ham_max_contained: int = 14) -> bool:
    """THE shared shown-twin test (render_prep invariant pass + prep_qa
    `dup_shown` tripwire import this so enforcement and QA can never drift).

    *ham* is the bubble-masked 8x8 dhash distance of the two RAW panels
    (`_dhash8_bgr` over the full `scenes/` image with `_mask_bubbles_for_hash`
    boxes — never the shown crops, so crop geometry can't manufacture or hide a
    twin). Twins: ham <= *ham_max* (the ladder's existing 8), OR ham <=
    *ham_max_contained* AND the OCR dialogue of one panel contains the other's
    (echo pairs redraw the art slightly — masked ham 12-14 measured on the
    p000054/p000055 pair — but repeat the dialogue verbatim)."""
    if ham <= ham_max:
        return True
    return ham <= ham_max_contained and ocr_dialogue_contained(ocr_a, ocr_b)


def drop_cross_segment_near_identical_cuts(
    cuts_by_segment: Dict[str, List[Dict[str, Any]]],
    order: Sequence[str],
    get_img,
    *,
    ham_max: int = 8,
    min_area_ratio: float = 0.7,
    exempt: Optional[set] = None,
    get_boxes=None,
    get_raw_img=None,
    get_raw_boxes=None,
    on_recrop=None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str]],
           List[Tuple[str, str, str]]]:
    """Drop a SOURCE-REPEATED panel shown again across a segment boundary — the
    near-identical eye close-up the comic draws twice a few beats apart (p090 at
    g0017's tail, p095 at g0018's head). The containment pass can't see it (same
    size, neither contains the other), the per-segment near-identical pass never
    compares across segments, and NCC underscores the shifted crop (0.88) — so an
    8x8 perceptual dhash decides (hamming <= *ham_max*; measured 3 for that pair vs
    23 for the flash bisection and 38+ for distinct art).

    Takes NO narrated `protect` set on purpose: a narrated panel IS droppable here,
    because the later twin is dropped ONLY when its segment keeps >= 1 other cut.
    The narration then plays over the panels it still shows, so the held-image
    regression (a narrated segment emptied -> a neighbour holds 12-16s) cannot
    happen — that guard is what makes overriding the narrated protection safe.
    *exempt* files (system cards, blank/chrome) and split cuts are never a duplicate
    and never a comparison reference. Only CONSECUTIVE shown cuts are compared, so a
    deliberate flashback callback to a much earlier panel is never `prev` in view
    and always survives. The EARLIER twin is kept; freed time is redistributed
    within the affected segment (audio/timing intact).

    *get_boxes*(f) (optional): per-file bubble boxes, masked before the dhash
    (`_mask_bubbles_for_hash`) so identical art under different dialogue bubbles is
    still recognized. When the later twin is the SOLE cut of its segment (dropping
    it would empty the segment and hold a neighbour), it is instead CANONICALIZED
    to the earlier twin — c["file"] is rewritten to the earlier file, its own
    audio/duration untouched — so the now-consecutive same-image sole cuts fold
    into ONE continuous Ken-Burns pan downstream (merge_consecutive_same_image_cuts)
    with no held image and no re-cut. Returns (out, dropped, canonicalized) where
    canonicalized is a list of (segment, from_file, to_file) for logging.

    MANUFACTURED-TWIN GUARD (*get_raw_img*/*get_raw_boxes*/*on_recrop*): the
    pass compares the SHOWN images, and the cleaner can manufacture a twin out
    of two DISTINCT panels by cropping one to the region they share (the
    p000090/p000095 eye: crop ham 3, raw masked ham ~22-26 — canonicalizing
    swapped p095's art away and ONE image held ~24s). So before the sole-cut
    canonicalize, the RAW panels (*get_raw_img*, bubble-masked via
    *get_raw_boxes*) must ALSO be twins (ham <= *ham_max*); when the crops
    match but the raws don't, the cut is NOT canonicalized — *on_recrop*(seg,
    file) is called instead so the caller re-writes that cut's clean image as
    the FULL panel (the raw art is distinct; showing it whole is strictly
    better than a twin crop or a 24s hold). Raw images unavailable -> the
    guard stays out of the way (legacy canonicalize behavior)."""
    ex = exempt or set()
    gb = get_boxes or (lambda f: ())
    out = {k: list(v) for k, v in cuts_by_segment.items()}
    dropped: List[Tuple[str, str]] = []
    canonicalized: List[Tuple[str, str, str]] = []
    prev_file: Optional[str] = None
    prev_hash: Optional[int] = None
    for seg in order:
        cuts = out.get(seg) or []
        kept: List[Dict[str, Any]] = []
        n = len(cuts)
        for ci, c in enumerate(cuts):
            f = str(c.get("file"))
            if f in ex or c.get("file2") or c.get("layout"):
                kept.append(c)          # blank/system/split: not a dup or a ref
                continue
            img = get_img(f)
            h = _dhash8_bgr(img, gb(f)) if img is not None else None
            near = False
            if (prev_file and prev_file != f and prev_hash is not None
                    and h is not None):
                pa = get_img(prev_file)
                if pa is not None:
                    aa = pa.shape[0] * pa.shape[1]
                    ab = img.shape[0] * img.shape[1]
                    ratio = min(aa, ab) / max(1, max(aa, ab))
                    if (ratio >= min_area_ratio
                            and (prev_hash ^ h).bit_count() <= ham_max):
                        near = True
            survivors_if_drop = len(kept) + (n - ci - 1)
            if near and survivors_if_drop >= 1:
                dropped.append((seg, f))
                continue                # keep the earlier twin as the reference
            if near and prev_file and prev_file not in ex:
                # SOLE cut of its segment (survivors_if_drop < 1): dropping it
                # would empty the segment and hold a neighbour 12-16s. Instead
                # canonicalize to the earlier twin — same image, its OWN audio/
                # duration untouched — so the now-consecutive same-image sole cuts
                # fold into ONE continuous Ken-Burns pan downstream
                # (merge_consecutive_same_image_cuts). prev_file/prev_hash stay so
                # a run of near-dup sole cuts all canonicalize to the first twin.
                #
                # ... but ONLY when the RAW panels are twins too. A crop-twin
                # whose raws are distinct is the cleaner's manufacture (the
                # p090/p095 eye band): canonicalizing it hides real art behind
                # a ~24s single-image hold. Re-crop to the full panel instead.
                if get_raw_img is not None:
                    ra, rb = get_raw_img(prev_file), get_raw_img(f)
                    if ra is not None and rb is not None:
                        grb = get_raw_boxes or (lambda _f: ())
                        raw_ham = (_dhash8_bgr(ra, grb(prev_file))
                                   ^ _dhash8_bgr(rb, grb(f))).bit_count()
                        if raw_ham > ham_max:
                            print(f"[dedup-guard] {seg}: crop-twin but "
                                  f"raw-distinct — re-cropped to full panel "
                                  f"instead of canonicalize")
                            if on_recrop is not None:
                                on_recrop(seg, f)
                            kept.append(c)
                            prev_file, prev_hash = f, h
                            continue
                    else:
                        print(f"[dedup-guard] {seg}: raw image(s) unavailable "
                              f"for {prev_file!r}/{f!r} — guard skipped, "
                              f"canonicalizing on crop-twin alone (legacy "
                              f"behavior)")
                canonicalized.append((seg, f, prev_file))
                c["file"] = prev_file
                kept.append(c)
                continue
            kept.append(c)
            if h is not None:
                prev_file, prev_hash = f, h
        if len(kept) != len(cuts) and kept:
            removed = [str(c.get("file")) for c in cuts if c not in kept]
            out[seg] = _redistribute(cuts, removed)
    return out, dropped, canonicalized


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


# Round-2 E2 (bubble-clean residue): stylized lettering Apple-OCR cannot read
# stays visible in "cleaned" bubbles (Nano ch1 p000023 "JANG?", p000026
# "END.", p000076, p000099 full dialogue) — the same OCR blindness already
# proven on painted SFX. A speech bubble interior with DENSE STROKES but
# empty OCR is invisible text by construction (a genuinely empty bubble is a
# flat void). Canny edge ratio over the bubble's interior component — the
# impact_lettering gate style. Measured on the real fixtures: p000099 raw
# interiors 0.060/0.079; genuinely blank (cleaned) interiors 0.000 → 0.030
# is a 2x margin under the weakest positive.
BUBBLE_STROKE_DENSITY_MIN = 0.030


def bubble_stroke_density(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
) -> float:
    """Fraction of Canny edge pixels inside *box*'s bubble interior
    component. 0.0 when the box has no white/black interior (fail-soft —
    a detector false-positive on artwork never scores). SINGLE authority
    for the invisible-text check: clean_scene_image's residue net and
    prep_qa's bubble_text_residue WARN both call this."""
    _tmask, _fill, inside = _bubble_text(img, box)
    if inside is None:
        return 0.0
    n = int((inside > 0).sum())
    if not n:
        return 0.0
    gray = img.mean(axis=2).astype(np.uint8) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    return float(((edges > 0) & (inside > 0)).sum()) / float(n)


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
    *,
    residue_net: bool = False,
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
        # Round-2 E2 residue net: OCR found NOTHING in this SPEECH-SHAPED
        # bubble, yet its interior shows dense strokes — stylized lettering
        # OCR cannot read (p000099: the contrast fill catches glyph cores but
        # leaves readable anti-aliased ghosts at 240-254, inside BOTH sweeps'
        # tolerance bands). Measured BEFORE any fill; the flatten runs after
        # the normal passes.
        dense_invisible = (
            residue_net and fill is not None and not wmask.any()
            and speech_shaped_boxes([tuple(int(v) for v in b)], W_)
            and bubble_stroke_density(out, b) >= BUBBLE_STROKE_DENSITY_MIN)
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
            if dense_invisible:
                # invisible-text bubble: flatten the WHOLE interior with the
                # bubble's own flat color (the existing removal mechanism,
                # without the tolerance bands the 240-254 ghost halos slip
                # through). Interior-only — outline/shape stay; a speech
                # bubble's interior is flat paper by design, so this is
                # exactly what a properly cleaned bubble looks like.
                out[inside > 0] = fill

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


def edge_recrop_window(
    img: np.ndarray,
    bubbles: Sequence[Tuple[int, int, int, int]],
    protected: Sequence[Tuple[int, int, int, int]] = (),
    *,
    edge_touch_frac: float = 0.10,
    max_cut_frac: float = 0.50,
    min_keep_px: int = 320,
    dominance: float = 0.55,
    flat_std: float = 18.0,
) -> Tuple[int, int]:
    """(y0, y1) of the SHOWN window after trimming bubble-dominated edge bands.

    The tag-driven re-crop (owner decision 2026-07-16): a balloon stack parked
    against the top/bottom edge of a panel crop is dialogue chrome, not art —
    the shown frame tightens to the art region and the balloons never appear
    (their words already ride the narration). A band is only cut when
    (a) its bubbles touch the edge, (b) nothing protected (system windows)
    intersects it, (c) it is bubble-DOMINATED: union coverage >= *dominance*
    OR the non-bubble remainder is flat background (std < *flat_std*), and
    (d) at least *min_keep_px* and half the panel survive. Bubbles overlapping
    real art (mid-panel) never trigger a cut — they stay, drawn as-is.
    Vertical only.  # ponytail: add left/right bands if a real case shows up.
    """
    h, w = img.shape[0], img.shape[1]
    y0, y1 = 0, h

    gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.ndim == 3 else img)

    def _band_ok(a: int, b: int) -> bool:
        if b <= a:
            return False
        for (px1, py1, px2, py2) in protected:
            if py2 > a and py1 < b:
                return False
        band = np.zeros((b - a, w), bool)
        for (bx1, by1, bx2, by2) in bubbles:
            iy0, iy1 = max(a, int(by1)) - a, min(b, int(by2)) - a
            if iy1 > iy0:
                band[iy0:iy1, max(0, int(bx1)):max(0, int(bx2))] = True
        cover = float(band.mean())
        if cover >= dominance:
            return True
        rest = gray[a:b][~band]
        if rest.size == 0:
            return True
        if float(rest.std()) < flat_std:
            return True
        # backdrop gradients (sky, blur) beat the flat test but carry no ART:
        # bubble-masked edge density of the band, on the min-art-score scale
        # (balloon outlines are masked out; a face/action in the band fires
        # edges and vetoes the cut). Measured on nano p000023: balloon band
        # 0.0115 vs the face region 0.0419 — 3.6x separation at 0.015.
        edges = cv2.Canny(gray[a:b], 50, 150)
        edges[band] = 0
        return float(edges.mean()) / 255.0 < 0.015

    # Chained cuts: a balloon STACK reaches the edge through its first member
    # (p000023: balloon 2 starts mid-balloon-1, not at the edge) — grow the
    # cut while another bubble starts above/below the current line.
    top_edge = int(h * edge_touch_frac)
    top_cut = 0
    changed = True
    while changed:
        changed = False
        for (_bx1, by1, _bx2, by2) in bubbles:
            reach = top_edge if top_cut == 0 else top_cut
            end = min(int(by2), int(h * max_cut_frac))
            if int(by1) <= reach and end > top_cut:
                top_cut = end
                changed = True
    if top_cut > 0 and _band_ok(0, top_cut):
        y0 = top_cut

    bot_edge = int(h * (1.0 - edge_touch_frac))
    bot_cut = h
    changed = True
    while changed:
        changed = False
        for (_bx1, by1, by2_start_guard, by2) in [
                (b[0], b[1], max(int(b[1]), int(h * (1.0 - max_cut_frac))), b[3])
                for b in bubbles]:
            reach = bot_edge if bot_cut == h else bot_cut
            if int(by2) >= reach and by2_start_guard < bot_cut:
                bot_cut = by2_start_guard
                changed = True
    if bot_cut < h and _band_ok(bot_cut, h):
        y1 = bot_cut

    if y1 - y0 < max(min_keep_px, h // 2):
        return 0, h
    return int(y0), int(y1)


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
    midtone_min: float = 0.15,
    chroma_min: float = 5.0,
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
        if img.ndim == 3 and chroma_min > 0.0:
            sub = img[a:b].astype(int)
            chroma = float(np.maximum(
                np.maximum(np.abs(sub[..., 0] - sub[..., 1]),
                           np.abs(sub[..., 1] - sub[..., 2])),
                np.abs(sub[..., 0] - sub[..., 2])).mean())
            chroma_ok = chroma >= chroma_min
        if midtone >= midtone_min and chroma_ok:
            info["recropped"] = True
            info["band"] = (a, b)
            return img[a:b], info
    return img, info


def husk_recrop_decision(
    img: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    display_sec: float,
    max_hold_sec: float,
    neighbor_crops: Sequence[Tuple[str, Optional[np.ndarray]]] = (),
    blank_frac_min: float = 0.40,
    ham_max: int = 8,
    min_h: int = 120,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """V3 husk-crop policy: the cleaner blanked bubbles covering more than
    *blank_frac_min* of a CHOSEN crop AND that crop will hold the screen past
    max_hold_sec/2 (the 22.8s eye whose lower half was one huge cleaned-empty
    bubble) -> re-crop to the art band, excluding the dead bubble region
    (dead_box_recrop's band logic with its color guards relaxed — BW/line-art
    holds fail the chroma gate by construction, midtone floor kept low) —
    UNLESS the re-crop would become a shown-crop dhash twin (ham <= *ham_max*)
    of a neighbor's crop within the 3-cut window (*neighbor_crops* — that is
    exactly how the p090/p095 manufactured-twin was born), in which case the
    full crop is kept and the V1/V2 ken variety carries the watchability.

    Returns (crop, info): info.husk_recropped True with info.band=(y0,y1) in
    input coords on success; info.refused_twin=(neighbor, ham) when the twin
    guard vetoed; blank_frac always measured. Never touches doc/system panels
    upstream (their cleaned path emits no boxes, so this no-ops)."""
    info: Dict[str, Any] = {"husk_recropped": False, "blank_frac": 0.0,
                            "refused_twin": None, "band": None}
    if img is None or img.size == 0 or not boxes:
        return img, info
    info["blank_frac"] = bubble_coverage(img.shape, boxes)
    if info["blank_frac"] <= blank_frac_min:
        return img, info
    if display_sec <= max_hold_sec / 2.0:
        return img, info
    img2, dead = dead_box_recrop(img, boxes, max_blank_frac=blank_frac_min,
                                 min_h=min_h, midtone_min=0.05,
                                 chroma_min=0.0)
    if not dead.get("recropped"):
        return img, info
    h2 = _dhash8_bgr(img2)
    for nf, nimg in neighbor_crops:
        if nimg is None or nimg.size == 0:
            continue
        ham = (h2 ^ _dhash8_bgr(nimg)).bit_count()
        if ham <= ham_max:
            info["refused_twin"] = (str(nf), int(ham))
            return img, info
    info["husk_recropped"] = True
    info["band"] = dead.get("band")
    return img2, info


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
    seen: set = set()  # every non-exempt file ever emitted (GLOBAL, not per-group)
    prev_file: Optional[str] = None
    for i, seg in enumerate(order):
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
            # GLOBAL cap for non-exempt panels: once a panel has been shown
            # ANYWHERE it is NOT re-emitted again — not later in the same group,
            # AND not in a later group, at any distance (gap > radius). The
            # cross-group case otherwise replays the same image with the same
            # animation (group N shows it, group N+1 shows it again — the
            # within-group p000091 idx89&93 / p000109 idx106&110 dups were just
            # the same bug inside one group). The previous distinct panel HOLDS
            # that slot instead. Consecutive same-image runs are caught by
            # `near` (held → merge_consecutive re-animates them as ONE continuous
            # Ken Burns), so this never breaks the held-run case. Exempt
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


def _kenburns_slice(f0: float, f1: float,
                    ease: str = "ease_in_out") -> Dict[str, Any]:
    """Motion for the slice [f0, f1] (fractions of a same-image run's total
    duration) of ONE continuous slow Ken Burns. Slicing keeps zoom + pan
    CONTINUOUS across the run's cuts (each slice starts where the previous ended),
    so the repeated image reads as a single slow move, never a restart.

    `ease` must vary by position in the run: the renderer applies the easing
    over EACH cut's own duration, so a mid-run "ease_in_out" slice decelerates
    to a stop and re-accelerates at every segment boundary — a visible
    stop-start pulse on the held image. Head slice eases in, interior slices
    run linear, tail slice eases out; only a sole slice keeps ease_in_out."""
    return {
        "mode": "kenburns",
        "strength": _MERGE_STRENGTH,
        "ease": ease,
        "start_bias": {"x": round(_lerp(_MERGE_BIAS_START["x"], _MERGE_BIAS_END["x"], f0), 4),
                       "y": round(_lerp(_MERGE_BIAS_START["y"], _MERGE_BIAS_END["y"], f0), 4)},
        "end_bias": {"x": round(_lerp(_MERGE_BIAS_START["x"], _MERGE_BIAS_END["x"], f1), 4),
                     "y": round(_lerp(_MERGE_BIAS_START["y"], _MERGE_BIAS_END["y"], f1), 4)},
        "zoom": {"start": round(_lerp(_MERGE_ZOOM_START, _MERGE_ZOOM_END, f0), 4),
                 "end": round(_lerp(_MERGE_ZOOM_START, _MERGE_ZOOM_END, f1), 4)},
    }


# ---------------------------------------------------------------------------
# V1/V2 ken variety: long static holds + perceptual echoes (2026-07 review)
# ---------------------------------------------------------------------------

# Renderer branch thresholds — parity with remotion/src/plan.ts. Wide/tall
# panels take Cut.tsx branches with a BUILT-IN per-cut drift that ignores the
# motion dict, so ken variety can't reach them (and they are never "static").
_WIDE_COVER_MIN_ASPECT = 1.3
_TALL_SCROLL_MIN_ASPECT = 2.0

_KV_STRENGTH = 0.8          # pan strength for variety sub-cuts
_KV_SPLIT3_FACTOR = 1.5     # display > 1.5x cap -> 3 sub-cuts, else 2
_KV_WEIGHTS = {2: (0.45, 0.55), 3: (0.35, 0.35, 0.30)}
_KV_MIN_SUBCUT_SEC = 2.0    # never manufacture a flash_cut


def _motion_honored_dims(dims_entry: Optional[Dict[str, Any]]) -> bool:
    """True when Cut.tsx's DEFAULT contain branch renders this file — the only
    branch that honors zoom/bias, i.e. where ken variety is expressible. doc
    panels render contain but must stay still (text); wide/tall run their own
    built-in drift. Missing dims -> assume default branch (fail toward the fix;
    prep_qa missing_dims blocks separately)."""
    d = dims_entry or {}
    if d.get("doc"):
        return False
    w, h = float(d.get("w") or 0.0), float(d.get("h") or 0.0)
    if h > 0 and w / h >= _WIDE_COVER_MIN_ASPECT:
        return False
    if w > 0 and h / w >= _TALL_SCROLL_MIN_ASPECT:
        return False
    return True


def focal_point_for_crop(
    img: Optional[np.ndarray],
    dead_boxes: Sequence[Tuple[int, int, int, int]] = (),
    face_center: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, str]:
    """Deterministic focal point (fx, fy in 0..1, + source tag) for a SHOWN
    crop. Priority: a known FACE center (vision targets, largest face) -> the
    densest ART cell of a 4x4 edge-energy grid with dead regions suppressed:
    the blanked bubble/word boxes the cleaner computed (*dead_boxes*, crop
    coords) are zeroed, and near-flat white/black pixels (a blanked bubble the
    boxes missed) carry no energy by construction. Ties keep the first cell in
    top-left scan order; unreadable/blank crops fall back to upper-middle
    (0.5, 0.4) — the same "manhwa subjects sit high" default the planner
    uses."""
    if face_center is not None:
        fx = min(max(float(face_center[0]), 0.0), 1.0)
        fy = min(max(float(face_center[1]), 0.0), 1.0)
        return fx, fy, "face"
    if img is None or img.size == 0 or min(img.shape[:2]) < 8:
        return 0.5, 0.4, "default"
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    energy = np.abs(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F, ksize=3))
    energy[(g > 235) | (g < 20)] = 0.0
    for (x1, y1, x2, y2) in dead_boxes:
        x1, y1 = max(0, int(x1) - 4), max(0, int(y1) - 4)
        x2, y2 = min(w, int(x2) + 4), min(h, int(y2) + 4)
        if x2 > x1 and y2 > y1:
            energy[y1:y2, x1:x2] = 0.0
    n = 4
    best, bfx, bfy = 0.0, 0.5, 0.4
    for gy in range(n):
        for gx in range(n):
            y0, y1 = (h * gy) // n, (h * (gy + 1)) // n
            x0, x1 = (w * gx) // n, (w * (gx + 1)) // n
            if y1 <= y0 or x1 <= x0:
                continue
            s = float(energy[y0:y1, x0:x1].mean())
            if s > best + 1e-9:
                best = s
                bfx = (x0 + x1) / 2.0 / w
                bfy = (y0 + y1) / 2.0 / h
    if best <= 0.0:
        return 0.5, 0.4, "default"
    return float(bfx), float(bfy), "art"


def _focal_pan_bias(fx: float, fy: float) -> Dict[str, float]:
    """Pan bias landing the focal point centered — the same translate
    convention as timeline_planner.face_end_bias (verified against Cut.tsx
    biasOffset): a focal RIGHT of center needs a NEGATIVE x bias; BELOW center
    a POSITIVE y bias (Cut.tsx negates y)."""
    bx = max(-1.0, min(1.0, -(fx - 0.5) / 0.5))
    by = max(-1.0, min(1.0, (fy - 0.5) / 0.5))
    return {"x": round(bx, 3), "y": round(by, 3)}


def _scaled_bias(b: Dict[str, float], k: float) -> Dict[str, float]:
    return {"x": round(b["x"] * k, 3), "y": round(b["y"] * k, 3)}


def ken_variety_motions(n: int, fx: float, fy: float) -> List[Dict[str, Any]]:
    """2-3 DISTINCT ken/motion regions over ONE panel, in the exact schema
    Cut.tsx consumes (mode/strength/ease/start_bias/end_bias/zoom, focus_y for
    the tall branch): WIDE establish (whole frame, gentle drift toward the
    focal) -> TIGHT push onto the focal band (zoom-in) -> (n==3) PULL back out
    to the counter band (zoom-out). Zooms stay inside Cut.tsx's clamp
    [1.0, MAX_ZOOM_CAP=1.35]; the sub-cuts differ by zoom LEVEL and DIRECTION
    even when the focal point is dead-center (bias degenerates to 0)."""
    fb = _focal_pan_bias(fx, fy)
    fy01 = min(max(float(fy), 0.0), 1.0)
    wide = {"mode": "kenburns", "strength": _KV_STRENGTH, "ease": "ease_in",
            "start_bias": {"x": 0.0, "y": 0.0},
            "end_bias": _scaled_bias(fb, 0.35),
            "zoom": {"start": 1.02, "end": 1.09},
            "focus_y": 0.5, "ken_region": "wide"}
    tight = {"mode": "kenburns", "strength": _KV_STRENGTH,
             "ease": "ease_out" if n <= 2 else "linear",
             "start_bias": _scaled_bias(fb, 0.7), "end_bias": fb,
             "zoom": {"start": 1.18, "end": 1.32},
             "focus_y": round(fy01, 3), "ken_region": "tight"}
    if n <= 2:
        return [wide, tight]
    pull = {"mode": "kenburns", "strength": _KV_STRENGTH, "ease": "ease_out",
            "start_bias": _scaled_bias(fb, 0.8),
            "end_bias": _scaled_bias(fb, -0.45),
            "zoom": {"start": 1.24, "end": 1.05},
            "focus_y": round(1.0 - fy01, 3), "ken_region": "pull"}
    return [wide, tight, pull]


def _rewrite_item_camera_for_ken(item: Dict[str, Any],
                                 motions: List[Dict[str, Any]],
                                 dims: Dict[str, Any]) -> None:
    """The renderer clamps every cut's zoom to the ITEM-level camera:
    Shot.tsx passes item.camera into Cut.tsx's zoomCap = min(MAX_ZOOM_CAP,
    camera.max_zoom), further min'd to TEXT_ZOOM_CAP=1.06 when
    camera.avoid_text_zoom (remotion/src/plan.ts). The planner mirrors
    max_zoom off the item's ORIGINAL motion end (<=~1.16; 1.0 static) and
    defaults avoid_text_zoom True — so without this rewrite the ken-variety
    tight push (1.18->1.32) renders FLAT at 1.06 and the long-hold defect
    ships invisibly. Raise max_zoom to cover the new sub-cut zooms; clear
    avoid_text_zoom ONLY when every panel the item shows has nothing
    readable left to protect: bubble text already blanked off (scene_dims
    'blanked') and not a doc text panel. Stamped panel_kind=='system' cards
    never reach the ken passes (skip_files), and the pixel-level 'sys' flag
    is deliberately NOT consulted (the system-box YOLO overfires on
    SFX/bubble text). Items with no camera are never clamped below
    MAX_ZOOM_CAP — left untouched."""
    cam = item.get("camera")
    if not isinstance(cam, dict) or not cam:
        return
    cam = dict(cam)
    need = max((max(float((m.get("zoom") or {}).get("start") or 1.0),
                    float((m.get("zoom") or {}).get("end") or 1.0))
                for m in motions), default=1.0)
    if float(cam.get("max_zoom") or 0.0) < need:
        cam["max_zoom"] = round(float(need), 3)
    if cam.get("avoid_text_zoom"):
        files = [str(x) for c in (item.get("cuts") or [])
                 for x in (c.get("file"), c.get("file2")) if x]

        def _text_gone(f: str) -> bool:
            d = dims.get(f) or {}
            return bool(d.get("blanked")) and not d.get("doc")

        if files and all(_text_gone(f) for f in files):
            cam["avoid_text_zoom"] = False
    item["camera"] = cam


def split_long_hold_cuts(
    plan: Dict[str, Any],
    *,
    max_hold_sec: float,
    focal_for_file,
    skip_files: Optional[set] = None,
) -> Tuple[Dict[str, Any], List[Tuple[str, str, float, int, str]]]:
    """V1 fix: ONE cut displaying one file statically past
    [render].max_same_image_hold_sec is unwatchable even when the panel
    legitimately OWNS its narration (the 22.8s eye + cleaned-empty bubble;
    16.5s two-blank-bubble panel). Split that display into 2-3 sub-cuts of the
    SAME file with DIFFERENT ken/motion regions (deterministic focal via
    *focal_for_file*(f) -> (fx, fy, source)); durations split proportionally
    and sum EXACTLY to the original — audio/timing/ownership untouched,
    nothing dropped or merged.

    MUST run AFTER merge_consecutive_same_image_cuts / the twin-invariant
    re-merge: those passes collapse same-image runs and would fold the
    sub-cuts straight back into one cut (they now also guard on ken_variety,
    defensively). Skips: branding, split2 (file2/layout), stamped system cards
    (*skip_files* — their on-screen text needs stillness) and files whose
    renderer branch ignores the motion dict (_motion_honored_dims: doc, wide
    cover-drift, tall scroll). Sub-cuts carry ken_variety=True — prep_qa
    exempts them from repeat_cut/held_repeat, and the long_hold static
    ceiling keys on their absence."""
    skip = skip_files or set()
    dims = (plan or {}).get("scene_dims") or {}
    logs: List[Tuple[str, str, float, int, str]] = []
    for item in (plan or {}).get("timeline") or []:
        if item.get("branding"):
            continue
        cuts = item.get("cuts") or []
        if not cuts:
            continue
        out_cuts: List[Dict[str, Any]] = []
        split_motions: List[Dict[str, Any]] = []
        changed = False
        for c in cuts:
            f = str(c.get("file") or "")
            dur = float(c.get("dur") or 0.0)
            if (not f or c.get("file2") or c.get("layout")
                    or c.get("ken_variety") or dur <= max_hold_sec
                    or f in skip or not _motion_honored_dims(dims.get(f))):
                out_cuts.append(c)
                continue
            n = 3 if dur > _KV_SPLIT3_FACTOR * max_hold_sec else 2
            while n >= 2 and dur * min(_KV_WEIGHTS[n]) < _KV_MIN_SUBCUT_SEC:
                n -= 1
            if n < 2:
                out_cuts.append(c)      # cap too small to split safely
                continue
            fx, fy, src = focal_for_file(f)
            motions = ken_variety_motions(n, fx, fy)
            start = float(c.get("start") or 0.0)
            acc = 0.0
            for k, (wgt, m) in enumerate(zip(_KV_WEIGHTS[n], motions)):
                d_k = (round(dur - acc, 4) if k == n - 1
                       else round(dur * wgt, 4))
                out_cuts.append({**c, "start": round(start + acc, 4),
                                 "dur": d_k, "motion": m,
                                 "ken_variety": True})
                acc = round(acc + d_k, 4)
            # rounding the earlier weights can shave the remainder to
            # 1.9999s at a pathological cap — a manufactured flash_cut.
            # Shift the epsilon off the previous sub-cut (the while-guard
            # above leaves it >=0.33s of slack at the feasibility boundary)
            # so the floor holds and the total stays EXACT.
            last, prev = out_cuts[-1], out_cuts[-2]
            eps = round(_KV_MIN_SUBCUT_SEC - float(last["dur"]), 4)
            if eps > 0:
                prev["dur"] = round(float(prev["dur"]) - eps, 4)
                last["dur"] = round(float(last["dur"]) + eps, 4)
                last["start"] = round(float(last["start"]) - eps, 4)
            split_motions.extend(motions)
            changed = True
            logs.append((str(item.get("segment_id") or ""), f, dur, n, src))
        if changed:
            item["cuts"] = out_cuts
            # the item-level camera cap must not flatten the sub-cut zooms
            _rewrite_item_camera_for_ken(item, split_motions, dims)
    return plan, logs


def ken_differentiate_echo_pairs(
    plan: Dict[str, Any],
    get_img,
    get_boxes,
    get_raw_img,
    get_raw_boxes,
    *,
    focal_for_file,
    skip_files: Optional[set] = None,
    window: int = 3,
    ham_max: int = 8,
) -> Tuple[Dict[str, Any], List[Tuple[str, str, str, str, int, int]]]:
    """V2 fix: two nearby shown cuts whose SHOWN CROPS are bubble-masked
    dhash twins (ham <= *ham_max*) while their RAW panels are NOT (raw masked
    ham > *ham_max*) — the artist zoom-echo (p000044 re-crops p000043's lower
    half) and the husk re-crop class. The masked-RAW invariant correctly keeps
    them as distinct panels; the viewer reads a stutter. Give the pair
    DIFFERENT ken regions via the V1 focal machinery (earlier cut WIDE, later
    cut TIGHT on its own focal) so the repeat reads as intentional emphasis.
    NEVER drops, NEVER merges narration — motion only; ownership absolute.

    A member is only re-aimed when its motion is actually expressible: sole
    display of its file (not part of a same-image run whose continuous slices
    would stutter), default renderer branch (_motion_honored_dims), not
    already ken_variety/differentiated. System cards (*skip_files*) exempt as
    usual. Pairs compared within a sliding *window* of shown cuts; split
    halves (no raw image) are skipped — raw-distinctness can't be proven."""
    skip = skip_files or set()
    dims = (plan or {}).get("scene_dims") or {}
    # (segment_id, cut ref, item ref). NOTE: deliberate divergence from
    # prep_qa.iter_shown_cuts (its echo window includes split2 file2 halves):
    # enforcement only walks cuts whose motion it could rewrite; QA measures
    # the shown stream.
    ent: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for item in (plan or {}).get("timeline") or []:
        if item.get("branding"):
            continue
        for c in item.get("cuts") or []:
            f = str(c.get("file") or "")
            if f and not c.get("file2") and not c.get("layout"):
                ent.append((str(item.get("segment_id") or ""), c, item))

    counts: Dict[str, int] = {}
    for _seg, c, _it in ent:
        counts[str(c["file"])] = counts.get(str(c["file"]), 0) + 1

    ch: Dict[str, Optional[int]] = {}
    rh: Dict[str, Optional[int]] = {}

    def _ch(f: str) -> Optional[int]:
        if f not in ch:
            img = get_img(f)
            ch[f] = None if img is None else _dhash8_bgr(img, get_boxes(f))
        return ch[f]

    def _rh(f: str) -> Optional[int]:
        if f not in rh:
            img = get_raw_img(f)
            rh[f] = None if img is None else _dhash8_bgr(img, get_raw_boxes(f))
        return rh[f]

    def _modifiable(f: str, c: Dict[str, Any]) -> bool:
        return (counts.get(f, 0) == 1 and not c.get("ken_variety")
                and not c.get("echo_differentiated")
                and _motion_honored_dims(dims.get(f)))

    logs: List[Tuple[str, str, str, str, int, int]] = []
    seen_pairs: set = set()
    for j in range(len(ent)):
        seg_j, cj, it_j = ent[j]
        fj = str(cj["file"])
        if fj in skip:
            continue
        for i in range(max(0, j - (window - 1)), j):
            seg_i, ci, it_i = ent[i]
            fi = str(ci["file"])
            if not fi or fi == fj or fi in skip:
                continue
            key = tuple(sorted((fi, fj)))
            if key in seen_pairs:
                continue
            if _ch(fi) is None or _ch(fj) is None:
                continue
            sham = (_ch(fi) ^ _ch(fj)).bit_count()   # type: ignore[operator]
            if sham > ham_max:
                continue
            if _rh(fi) is None or _rh(fj) is None:
                continue                             # can't prove raw-distinct
            rham = (_rh(fi) ^ _rh(fj)).bit_count()   # type: ignore[operator]
            if rham <= ham_max:
                continue                             # raw twins: invariant's job
            seen_pairs.add(key)
            if _modifiable(fi, ci):
                fx, fy, _s = focal_for_file(fi)
                ci["motion"] = {**ken_variety_motions(2, fx, fy)[0],
                                "echo_pair": fj}
                ci["echo_differentiated"] = True
                _rewrite_item_camera_for_ken(it_i, [ci["motion"]], dims)
            if _modifiable(fj, cj):
                fx, fy, _s = focal_for_file(fj)
                cj["motion"] = {**ken_variety_motions(2, fx, fy)[1],
                                "echo_pair": fi}
                cj["echo_differentiated"] = True
                _rewrite_item_camera_for_ken(it_j, [cj["motion"]], dims)
            logs.append((seg_i, fi, seg_j, fj, sham, rham))
    return plan, logs


def _item_sole_image(item: Dict[str, Any]) -> Optional[str]:
    """The single source image an item shows end-to-end, or None when the item is
    branding, has no cuts, shows a split (file2/layout), carries a V1
    ken-variety split (deliberate distinct regions — folding them into one
    continuous slice would undo the fix), or shows more than one image — none
    of those can join a cross-item same-image run."""
    if item.get("branding"):
        return None
    cuts = item.get("cuts") or []
    if not cuts or any(c.get("file2") or c.get("layout")
                       or c.get("ken_variety") for c in cuts):
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
        if not f or c.get("file2") or c.get("layout") or c.get("ken_variety"):
            # ken_variety sub-cuts are DELIBERATE distinct regions over one
            # image (V1) — collapsing them back would undo the fix
            out.append(c)
            i += 1
            continue
        j = i + 1
        while (j < n and str(cuts[j].get("file") or "") == f
               and not cuts[j].get("file2") and not cuts[j].get("layout")
               and not cuts[j].get("ken_variety")):
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


def protect_narrated_from_junk(junk: Dict[str, str],
                               narrated_files: set,
                               *, also_protect: Optional[set] = None) -> Dict[str, str]:
    """Drop narrated panels from the visual judge's *junk* set (mutates + returns
    it). A panel that owns its own spoken line is a story beat the writer chose to
    describe; after the caption-fold fix narrated panels are all real `story` art,
    so the judge calling one 'flat glow / abstract' (an action-impact or energy/
    flash climax like the golden transformation burst) is a FALSE POSITIVE —
    dropping it makes a neighbour HOLD for 12-16s while the narrator describes a
    shot never shown. Operator manual_drops are applied AFTER this and still win.

    *also_protect* extends the spare set with panels that must survive the judge
    regardless of narration — stamped panel_kind=='system' cards, whose on-screen
    text is the story beat: dropping one trades a cosmetic flag for a blocking
    system_card_unshown (mirrors the substitute-garbage path which already spares
    system_files via its exempt set)."""
    protect = set(narrated_files or set()) | set(also_protect or set())
    for f in [f for f in junk if f in protect]:
        junk.pop(f, None)
    return junk


def narrated_files_from_plan(plan: Dict[str, Any]) -> set:
    """Every panel a narrated segment SHOWS — must NEVER be dropped as a
    'duplicate' by the seam / visual / near-identical / cross-segment dedup (the
    panel-collapse regression dropped distinct narrated panels). This is EVERY
    file in a narrated item's span (`scene_files`), not just the primary: a
    multi-panel flow span (e.g. [smirk, transformation-flash]) shows both, so a
    later span panel must be protected too — else the seam dedup drops it and the
    span renders only its head (the flash p074 dropped, smirk p073 held). A TRUE
    exact same-file consecutive run still folds via merge_consecutive_same_image_cuts;
    caption / held / fallback panels carry no narration of their own and stay
    droppable. Basenames, branding skipped."""
    out: set = set()
    for it in (plan or {}).get("timeline") or []:
        if not isinstance(it, dict) or it.get("branding"):
            continue
        if not str(it.get("tts_text") or "").strip():
            continue
        for f in (it.get("scene_files") or []):
            b = os.path.basename(str(f or ""))
            if b:
                out.add(b)
        pf = os.path.basename(str(it.get("primary_scene_file") or ""))
        if pf:                          # legacy items without scene_files
            out.add(pf)
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
            for k, (it, d) in enumerate(zip(run, durs)):
                f0 = acc / total
                acc += d
                cut = (it.get("cuts") or [None])[0]
                if cut is not None:
                    ease = ("ease_in" if k == 0 else
                            "ease_out" if k == len(run) - 1 else "linear")
                    cut["motion"] = _kenburns_slice(f0, acc / total, ease=ease)
        i = j
    return plan


def enforce_shown_twin_invariant(
    plan: Dict[str, Any],
    get_raw_img,
    *,
    get_raw_boxes=None,
    get_ocr=None,
    skip_files: Optional[set] = None,
    window: int = 8,
    ham_max: int = 8,
    ham_max_contained: int = 14,
) -> Tuple[Dict[str, Any], List[Tuple[str, str, str, int, bool]]]:
    """FINAL invariant pass — "no two shown panels may be twins" — run after
    every other dedup/merge pass, immediately before the clean plan is written.

    The dedup ladder is a stack of per-pass gates (protect_files narrated
    protection, min_area_ratio, consecutive-only comparisons) and each gate is
    a bypass route: the p000054/p000055 echo pair shipped BOTH panels because
    narrated protection shielded every per-segment pass and the cross-segment
    pass bailed at its area gate (0.56 < 0.7) before the bubble-masked hashing
    ever ran. This pass closes the class: it compares the RAW PANELS
    (`scenes/` file, full image, bubbles masked) — never the shown crops, so
    crop geometry can neither manufacture (D4) nor hide (D3) a twin — for
    every pair of shown cuts within a sliding *window* of shown cuts. Twin =
    `twin_verdict` (masked ham <= *ham_max*, or <= *ham_max_contained* with
    OCR dialogue containment — the echo-pair signature). Same-file pairs are
    DELIBERATELY skipped (`fi == fj` never reaches the twin test) — no fold is
    needed since a same-file recurrence is not two DIFFERENT panels shipping
    together, it is one panel shown twice, which is already someone else's
    job at every distance: consecutive runs are
    merge_consecutive_same_image_cuts', far-apart repeats are capped by
    cap_repeats_with_holds/held_repeat_flags, and an excessive same-file span
    standing in for art it doesn't own is long_hold_flags'. This is an
    intentional deviation from an earlier brief's "(plus any same-file pairs
    anywhere)" ask, not an oversight.

    Resolution is MERGE, never drop-narration: the twins collapse to ONE file
    — the richer panel (more bubbles, else larger raw area, else the earlier)
    — by rewriting every cut of the loser file to the survivor. Every
    narration line still lands on a shown image (cuts/audio/durations/
    scene_files untouched); the caller re-runs
    merge_consecutive_same_image_cuts so a now-consecutive run animates as one
    continuous slow pan. Never touched: system cards (*skip_files*, their
    on-screen text IS the beat — two notifications share a UI frame), doc
    panels (plan scene_dims doc flag, kept text differs under an identical
    frame), split cuts (file2/layout — a deliberate side-by-side layout), and
    files with no raw scene image (split halves). Mutates *plan* in place;
    returns (plan, folds) with folds = [(segment_of_later_cut, loser_file,
    survivor_file, masked_ham, containment)]."""
    gb = get_raw_boxes or (lambda _f: ())
    go = get_ocr or (lambda _f: "")
    skip = set(skip_files or ())
    dims = (plan or {}).get("scene_dims") or {}

    entries: List[Tuple[str, Dict[str, Any]]] = []   # (segment_id, cut) shown order
    for item in (plan or {}).get("timeline") or []:
        if item.get("branding"):
            continue
        for c in item.get("cuts") or []:
            entries.append((str(item.get("segment_id") or ""), c))

    hashes: Dict[str, Optional[int]] = {}

    def _h(f: str) -> Optional[int]:
        if f not in hashes:
            img = get_raw_img(f)
            hashes[f] = None if img is None else _dhash8_bgr(img, gb(f))
        return hashes[f]

    def _eligible(f: str) -> bool:
        if not f or f in skip or (dims.get(f) or {}).get("doc"):
            return False
        return _h(f) is not None

    def _richness(f: str) -> Tuple[int, int]:
        img = get_raw_img(f)
        area = int(img.shape[0] * img.shape[1]) if img is not None else 0
        return (len(gb(f) or ()), area)

    alias: Dict[str, str] = {}                       # loser -> survivor

    def _resolve(f: str) -> str:
        while f in alias:
            f = alias[f]
        return f

    folds: List[Tuple[str, str, str, int, bool]] = []
    n = len(entries)
    for j in range(n):
        seg_j, cj = entries[j]
        if cj.get("file2") or cj.get("layout"):
            continue                                 # split layout: never folded
        fj = _resolve(str(cj.get("file") or ""))
        if not _eligible(fj):
            continue
        for i in range(max(0, j - window), j):
            ci = entries[i][1]
            if ci.get("file2") or ci.get("layout"):
                continue
            fi = _resolve(str(ci.get("file") or ""))
            if fi == fj or not _eligible(fi):
                continue
            ham = (_h(fi) ^ _h(fj)).bit_count()      # type: ignore[operator]
            contained = ocr_dialogue_contained(go(fi), go(fj))
            if not twin_verdict(ham, go(fi), go(fj), ham_max=ham_max,
                                ham_max_contained=ham_max_contained):
                continue
            survivor, loser = ((fi, fj) if _richness(fi) >= _richness(fj)
                               else (fj, fi))
            alias[loser] = survivor
            folds.append((seg_j, loser, survivor, ham, contained))
            fj = survivor
            break                                    # cut folded; next entry

    if alias:
        for _seg, c in entries:
            f = str(c.get("file") or "")
            if f in alias and not c.get("file2") and not c.get("layout"):
                c["file"] = _resolve(f)
    return plan, folds


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
    # channel decision (2026-06-15): NO intro on any video. (2026-06-29): NO
    # outro either — every video now ends on the last STORY panel. The only
    # branding left in-frame is the corner watermark overlay, drawn by the
    # renderer (remotion/src/Branding.tsx), NOT a timeline item. The intro/outro
    # args + the which modes are kept for caller compat but never insert anything.
    intro_dur = 0.0
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
    ap.add_argument("--bubble-shown-mode", choices=["keep", "husk"],
                    default="keep",
                    help="story panels on screen: 'keep' (default; owner "
                         "2026-07-16) trims bubble-dominated EDGE bands off "
                         "the shown frame (tag-driven re-crop via the trained "
                         "detector) and leaves remaining bubbles AS DRAWN — "
                         "no text erasure, no white husks; 'husk' is the "
                         "legacy erase-text-keep-bubble behavior")
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
    ap.add_argument("--max-hold-sec", type=float, default=10.0,
                    help="[render].max_same_image_hold_sec — one file shown "
                         "statically past this is split into ken-varied "
                         "sub-cuts (V1); half of it gates the husk re-crop "
                         "(V3)")
    ap.add_argument("--panel-weights",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "assets", "models",
                        "webtoon_panels_v3.pt"),
                    help="trained webtoon YOLO — its system class (by name: "
                         "system_box/system_ui) protects system-message panels "
                         "from the bubble gate/blanking; v3 doesn't fire on "
                         "plain text cards like the legacy model did")
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
                    # largest FACE target (px, original scene coords) — the V1
                    # ken-variety focal prefers it (same pick as the planner's
                    # pick_face_target: largest, tie toward frame center)
                    best = None
                    for t in (it.get("targets") or []):
                        if not isinstance(t, dict) or t.get("type") != "face":
                            continue
                        bb = t.get("bbox") or []
                        if len(bb) != 4:
                            continue
                        area = max(0.0, float(bb[2]) - float(bb[0])) * \
                            max(0.0, float(bb[3]) - float(bb[1]))
                        cx = (float(bb[0]) + float(bb[2])) / 2.0
                        cy = (float(bb[1]) + float(bb[3])) / 2.0
                        cen = (cx - 0.5) ** 2 + (cy - 0.5) ** 2
                        if best is None or (-area, cen) < (best[0], best[1]):
                            best = (-area, cen, (cx * w, cy * h))
                    if best is not None:
                        vision_item[sf]["face_px"] = best[2]

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

    # Every panel that OWNS a distinct narration segment must survive the dedup —
    # a panel carrying its own spoken line is never a "duplicate" (only a TRUE
    # same-file consecutive run folds, via merge_consecutive_same_image_cuts).
    # This is the no-drop-distinct guarantee that keeps all 113 narrated panels.
    narrated_files = narrated_files_from_plan(plan)
    protect_files = system_files | narrated_files
    if narrated_files:
        print(f"[ok] protecting {len(narrated_files)} narration-bearing panel(s) "
              f"from the duplicate-drop")

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
    sys_ids: set = set()
    bubble_ids: set = set()
    if os.path.exists(args.panel_weights):
        from ultralytics import YOLO
        from studio.detect.yolo_panels import system_class_ids
        panel_model = YOLO(args.panel_weights)
        names = getattr(panel_model, "names", None) or {}
        sys_ids = system_class_ids(names)
        # voice containers by NAME — v3: speech_bubble/radio/speech_background;
        # legacy: speech_bubble. Feed the shown-frame edge re-crop.
        bubble_ids = {i for i, n in dict(names).items()
                      if n in ("speech_bubble", "radio", "speech_background")}
    else:
        print(f"[warn] panel weights missing ({args.panel_weights}) — "
              "system-message protection DISABLED")
    el_cache: Dict[str, Dict[str, List[Tuple[int, int, int, int]]]] = {}

    def _element_boxes(fname: str) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """One trained-model pass per shown file: system + bubble boxes."""
        if fname not in el_cache:
            out: Dict[str, List[Tuple[int, int, int, int]]] = {
                "system": [], "bubbles": []}
            img = _img(fname)
            if panel_model is not None and img is not None:
                r = panel_model.predict(img, conf=0.30, device=args.device, verbose=False)[0]
                if r.boxes is not None:
                    for (x1, y1, x2, y2), c in zip(
                            r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                        box = (int(x1), int(y1), int(x2), int(y2))
                        if int(c) in sys_ids:
                            out["system"].append(box)
                        elif int(c) in bubble_ids:
                            out["bubbles"].append(box)
            el_cache[fname] = out
        return el_cache[fname]

    def _sys_boxes(fname: str) -> List[Tuple[int, int, int, int]]:
        return _element_boxes(fname)["system"]

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
                # keep-mode: the screen ships AS DRAWN (bubble text included).
                if args.bubble_shown_mode == "keep":
                    cleaned_cache[fname] = (img.copy(), [])
                    return cleaned_cache[fname]
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
            elif args.bubble_shown_mode == "keep":
                # TAG-DRIVEN SHOWN FRAME (owner 2026-07-16): trim bubble-
                # dominated edge bands via the trained detector's bubble boxes
                # (the balloon stack over p000023's face vanishes; the face
                # fills the frame); every surviving bubble ships AS DRAWN —
                # its dialogue already rides the narration. No text erasure,
                # no white husks, no residue nets on the story path.
                els = _element_boxes(fname)
                ry0, ry1 = edge_recrop_window(img, els["bubbles"],
                                              protected=els["system"])
                out = img[ry0:ry1].copy() if (ry0, ry1) != (0, img.shape[0]) \
                    else img.copy()
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
                # orphan-word blanking needs the cleaner even with zero detected
                # bubbles (spiky balloons evade the detector). Clean the TEXT
                # ONLY — keep the bubble shape, never inpaint/blur. System panels
                # are kept whole above (their styled text IS the content);
                # bubble-only panels are folded to narration upstream, so neither
                # reaches this default path. residue_net: an OCR-empty bubble
                # whose interior shows dense strokes carries stylized text OCR
                # can't read (p000099) — flatten it anyway. Story panels only;
                # the doc/in-world path above never enables the net.
                out = clean_scene_image(img, boxes, text_boxes=words,
                                        residue_net=True)
                cleaned_cache[fname] = (out, boxes)
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
            cuts, geom, protect=protect_files)
        if len(new_cuts) > 1:
            imgs = {str(c["file"]): _img(str(c["file"])) for c in new_cuts}
            imgs = {k: v for k, v in imgs.items() if v is not None}
            new_cuts, vdropped = drop_visual_duplicate_cuts(
                new_cuts, imgs, protect=protect_files)
            dropped = list(dropped) + vdropped
            # near-identical SAME-SIZE pair (the 'reaction face with ?' repeat):
            # the containment filter above only catches small-in-big seam dups.
            if len(new_cuts) > 1:
                imgs = {k: v for k, v in imgs.items()
                        if k in {str(c["file"]) for c in new_cuts}}
                # this pass runs on the ORIGINAL _img scenes, so _boxes (original
                # coords) align — mask the bubbles so identical art under
                # different dialogue (the assassin pair) still reads as a near-dup
                new_cuts, ndropped = drop_near_identical_cuts(
                    new_cuts, imgs, protect=protect_files,
                    boxes_by_file={k: _boxes(k) for k in imgs})
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

    # exempt_from_drop only runs in the MULTI-cut branch above, so a SOLE-cut
    # system card never enters exempt_all — the first substitute_garbage_sole_cuts
    # (a flat status card reads as high-coverage 'garbage') then swaps it out for a
    # neighbour (ORV p000004: planner cut it, render_prep substituted p000003).
    # Seed exempt_all with system_files so EVERY exempt_all-based drop below (both
    # substitutes, cross-segment dedup, repeat-cap) spares stamped system cards —
    # the render half of the "in-world system cards are always shown" guarantee.
    exempt_all |= system_files

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
            coverage_by_file=cov_all, exempt=exempt_all, protect=protect_files)
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

    # cross-segment NEAR-IDENTICAL: a same-size source-repeat the containment loop
    # can't catch (the p090/p095 eye, drawn again a few beats later). Drops the
    # later twin ONLY when its segment keeps another cut, so a narrated panel goes
    # without ever emptying a segment (no held-image). Exempt is SYSTEM cards only,
    # NOT exempt_all: that set holds rich/visual_story STORY panels (the eye is one)
    # -- exactly the near-dup candidates we must collapse. Sparing system cards
    # avoids collapsing two DISTINCT notifications that share a UI frame (identical
    # low-freq dhash, different text); the retain-a-cut guard is the held-image
    # safety.
    # hash the image the renderer SHOWS. On --reuse-clean the cached scenes_clean/
    # crop is what ships (and what the viewer saw as the dup); the fresh re-clean
    # can differ wildly (ch1 eye p095: cached crop 395px vs re-clean 832px -> dhash
    # 3 vs 28), so hashing the re-clean missed it. Fall back to the fresh clean
    # when no cache exists (first full run writes it from the same source).
    _ni_clean_dir = os.path.join(args.episode_dir, "scenes_clean")

    def _shown_img(f):
        if args.reuse_clean:
            p = os.path.join(_ni_clean_dir, os.path.basename(str(f)))
            if os.path.exists(p):
                im = cv2.imread(p)
                if im is not None:
                    return im
        return _trimmed_clean(f)

    def _shown_boxes(f):
        # bubble boxes aligned to the SHOWN (trimmed-clean) crop _shown_img hashes:
        # the cleaner already returns them in original-scene coords (_cleaned[1]),
        # so offset by the content_bbox trim _trimmed_clean applies. Empty for
        # system/title/doc panels (the cleaner returns no boxes there). On the
        # --reuse-clean path the cached crop may differ in scale, so the offset is
        # approximate there; the strict hamming gate keeps that from mis-dropping.
        cl, boxes = _cleaned(f)
        if cl is None or not boxes or args.no_trim:
            return boxes
        tx1, ty1, _tx2, _ty2 = content_bbox(cl)
        return [(x1 - tx1, y1 - ty1, x2 - tx1, y2 - ty1)
                for (x1, y1, x2, y2) in boxes]
    # force_full_panel: cuts the manufactured-twin guard bounced off the
    # canonicalize path (crop-twin, raw-distinct) — their clean image is
    # written as the FULL cleaned panel below (no dead-box recrop, no split),
    # so the distinct art ships instead of a twin crop / 24s hold.
    force_full_panel: set = set()
    cuts_by_segment, nidrop, nicanon = drop_cross_segment_near_identical_cuts(
        cuts_by_segment, order, _shown_img,
        exempt=system_files, get_boxes=_shown_boxes,
        get_raw_img=_img, get_raw_boxes=_boxes,
        on_recrop=lambda seg, f: force_full_panel.add(f))
    for seg, f in nidrop:
        all_dropped.append(f)
        print(f"[ok] {seg}: cross-segment near-identical {f} dropped")
    for seg, f, canon in nicanon:
        print(f"[ok] {seg}: cross-segment near-identical sole cut {f} -> "
              f"canonicalized to {canon} (folds into one continuous pan)")

    shown = sorted({c["file"] for cs in cuts_by_segment.values() for c in cs})

    # V1/V3 display accounting: max CONSECUTIVE on-screen seconds per file at
    # this point in the ladder (holds are added later, so this is a floor of
    # the final hold length) + each file's neighbours within the 3-cut window
    # (the V3 husk re-crop twin guard).
    _seq_files: List[Tuple[str, float]] = []
    for _seg in order:
        for c in cuts_by_segment.get(_seg) or []:
            _seq_files.append((str(c.get("file") or ""),
                               float(c.get("dur") or 0.0)))
    disp_sec: Dict[str, float] = {}
    _si = 0
    while _si < len(_seq_files):
        _sj, _tot = _si, _seq_files[_si][1]
        while (_sj + 1 < len(_seq_files)
               and _seq_files[_sj + 1][0] == _seq_files[_si][0]):
            _sj += 1
            _tot += _seq_files[_sj][1]
        _f0 = _seq_files[_si][0]
        disp_sec[_f0] = max(disp_sec.get(_f0, 0.0), _tot)
        _si = _sj + 1
    neigh: Dict[str, set] = {}
    _files_only = [f for f, _d in _seq_files]
    for _idx, _f0 in enumerate(_files_only):
        for _k in range(max(0, _idx - 2), min(len(_files_only), _idx + 3)):
            _g = _files_only[_k]
            if _g and _f0 and _g != _f0:
                neigh.setdefault(_f0, set()).add(_g)

    pre_crop_cache: Dict[str, Optional[np.ndarray]] = {}

    def _pre_crop(g: str) -> Optional[np.ndarray]:
        """A neighbour's chosen crop BEFORE the V3 husk policy — what the twin
        guard compares against. Deterministic + order-independent (comparing
        against post-V3 crops would make outcomes depend on the processing
        order of `shown`)."""
        if g not in pre_crop_cache:
            im, bx = _cleaned(g)
            if im is None:
                pre_crop_cache[g] = None
            elif g in force_full_panel:
                pre_crop_cache[g] = im
            else:
                try:
                    prt, _pi = select_panel_crops(
                        im.copy(), bx, text_rich=_text_rich(g),
                        no_split=args.no_split)
                    pre_crop_cache[g] = prt[0]
                except Exception:
                    pre_crop_cache[g] = im
        return pre_crop_cache[g]

    def _husk_neighbors(fname: str) -> List[Tuple[str, Optional[np.ndarray]]]:
        return [(g, _pre_crop(g)) for g in sorted(neigh.get(fname, ()))]

    # 2+3. clean + trim shown scenes into scenes_clean/
    clean_dir = os.path.join(args.episode_dir, "scenes_clean")
    os.makedirs(clean_dir, exist_ok=True)

    scene_dims: Dict[str, Dict[str, int]] = {}
    split_map: Dict[str, Tuple[str, str]] = {}
    bubbles_cleaned = 0
    # shown-space geometry for the V1 focal machinery + V2 echo hashing, all
    # in the WRITTEN crop's coordinates: bubble boxes (hash masking), bubble+
    # word dead regions and face centers (focal selection). Only paths whose
    # geometry is known pass them; others rely on the focal's flat-region
    # suppression fallback.
    shown_bubble_boxes: Dict[str, List[Tuple[int, int, int, int]]] = {}
    shown_dead_boxes: Dict[str, List[Tuple[int, int, int, int]]] = {}
    shown_face: Dict[str, Tuple[float, float]] = {}

    def _write_part(name: str, part: np.ndarray, doc: bool = False,
                    sys_panel: bool = False, blanked: bool = False,
                    part_bubbles: Sequence[Tuple[int, int, int, int]] = (),
                    part_words: Sequence[Tuple[int, int, int, int]] = (),
                    part_face: Optional[Tuple[float, float]] = None) -> None:
        tx1 = ty1 = 0
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

        def _remap(bxs):
            out = []
            for (bx1, by1, bx2, by2) in bxs:
                x1c, y1c = max(0, int(bx1) - tx1), max(0, int(by1) - ty1)
                x2c, y2c = min(pw, int(bx2) - tx1), min(ph, int(by2) - ty1)
                if x2c > x1c and y2c > y1c:
                    out.append((x1c, y1c, x2c, y2c))
            return out

        shown_bubble_boxes[name] = _remap(part_bubbles)
        shown_dead_boxes[name] = _remap(part_bubbles) + _remap(part_words)
        if part_face is not None and pw > 0 and ph > 0:
            fxp, fyp = float(part_face[0]) - tx1, float(part_face[1]) - ty1
            if 0.0 <= fxp < pw and 0.0 <= fyp < ph:
                shown_face[name] = (fxp / pw, fyp / ph)

    def _band_remap(bxs, a2, b2):
        return [(bx1, max(0, by1 - a2), bx2, min(b2 - a2, by2 - a2))
                for (bx1, by1, bx2, by2) in bxs
                if min(by2, b2) - max(by1, a2) > 0]

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
        face_px = (vision_item.get(fname) or {}).get("face_px")
        words = word_boxes_by_file.get(fname) or []

        def _husk_pass(chosen):
            """V3: re-crop a blank-bubble-dominated crop that will hold the
            screen long, unless it would twin a 3-cut-window neighbour.
            Returns (crop, bubbles, words, face) in the crop's coords."""
            part2, hinfo = husk_recrop_decision(
                chosen, boxes, display_sec=disp_sec.get(fname, 0.0),
                max_hold_sec=args.max_hold_sec,
                neighbor_crops=_husk_neighbors(fname))
            if hinfo.get("husk_recropped"):
                a2, b2 = hinfo["band"]
                print(f"[ok] {fname}: HUSK re-crop "
                      f"blank_frac={hinfo['blank_frac']:.2f} "
                      f"display={disp_sec.get(fname, 0.0):.1f}s "
                      f"band=({a2},{b2})")
                f2 = None
                if face_px is not None and a2 <= face_px[1] < b2:
                    f2 = (face_px[0], face_px[1] - a2)
                return (part2, _band_remap(boxes, a2, b2),
                        _band_remap(words, a2, b2), f2)
            if hinfo.get("refused_twin"):
                nf, hv = hinfo["refused_twin"]
                print(f"[ok] {fname}: husk re-crop REFUSED — would twin "
                      f"neighbour {nf} (ham={hv}); keeping full crop "
                      "(ken variety covers the hold)")
            return chosen, boxes, words, face_px

        if fname in force_full_panel:
            # manufactured-twin guard: the aggressive recrop turned this panel
            # into a hash-twin of a DISTINCT neighbour — write it whole (border
            # trim only, in _write_part) so its real art is what ships. V3 may
            # still re-crop the husk when the band clears the twin guard.
            part, pbub, pwords, pface = _husk_pass(img)
            _write_part(fname, part, doc=rich, sys_panel=sysf, blanked=blanked,
                        part_bubbles=pbub, part_words=pwords, part_face=pface)
            print(f"[ok] {fname}: FULL-PANEL rewrite (dedup-guard) -> "
                  f"{scene_dims[fname]['w']}x{scene_dims[fname]['h']}")
            continue
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

        if (not pinfo.get("recropped")
                and parts[0].shape[:2] == img.shape[:2]):
            # geometry unchanged — boxes/words/face are valid in crop coords;
            # the V3 husk policy applies to exactly this surviving-husk case
            part, pbub, pwords, pface = _husk_pass(parts[0])
            _write_part(fname, part, doc=rich, sys_panel=sysf, blanked=blanked,
                        part_bubbles=pbub, part_words=pwords, part_face=pface)
        else:
            # select_panel_crops already re-cropped/split geometry — the dead
            # region is gone; shown-space geometry unknown (focal falls back
            # to flat-region suppression)
            _write_part(fname, parts[0], doc=rich, sys_panel=sysf,
                        blanked=blanked)
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
    # A panel that OWNS its own narration line is a story beat the writer chose
    # to describe — show it (see protect_narrated_from_junk). Stamped system
    # cards (panel_kind=='system') are spared too — their on-screen text IS the
    # beat, and dropping one here would surface a blocking system_card_unshown
    # (same intent as the substitute path's system_files exempt below). Operator
    # manual_drops below still WIN (the human overrides the writer).
    protect_narrated_from_junk(junk, narrated_files, also_protect=system_files)
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

        # a sole-cut junk segment has no survivor to redistribute to — and under
        # per-panel 1:1 EVERY segment is sole-cut, so without this an operator/
        # heal drop (manual_drops.json) never actually left the screen: the
        # coverage-based substitute pass upstream knows nothing about junk.
        # Give junk files coverage 1.0 and reuse the SAME story-adjacent hold
        # logic; the later same-image merge collapses "good panel + held copy"
        # into one slow pan spanning both narration lines.
        # Junk OUTRANKS the keep-worthy exemptions the bubble pass granted rich
        # art panels (that is the point of an explicit drop — same semantics as
        # _drop_junk_cuts' multi-cut path and the cross-seg-dup discard above),
        # EXCEPT for genuine system cards: holding one away would only trade a
        # cosmetic flag for a CRITICAL system_card_unshown downstream. Keyed on
        # system_files (the stamped panel_kind system_card_unshown itself gates
        # on) — NOT scene_dims' pixel-level "sys" flag, which the system-box
        # YOLO trips on mere SFX/bubble text (Nano ch1: all three cross_dup
        # STORY panels carried sys:True and stayed exempt).
        cuts_by_segment, junk_subs = substitute_garbage_sole_cuts(
            cuts_by_segment, {f: 1.0 for f in junk},
            durations=durations,
            exempt=exempt_all - (set(junk) - system_files), order=order)
        for seg, old, new in junk_subs:
            all_dropped.append(old)
            print(f"[ok] {seg}: junk sole cut {old} -> HOLDING {new}")

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

    # FINAL shown-twin INVARIANT — after every pass above, before the plan is
    # written: no two shown panels may be masked-raw twins. Catches whatever
    # slipped the ladder's gates (narrated protection + the 0.7 area gate let
    # the p054/p055 echo pair ship BOTH panels). Folds rewrite the later twin's
    # cuts to the richer panel — narration lines/audio untouched — then the
    # same-image merge re-runs so the folded run pans as ONE continuous move.
    out_plan, twin_folds = enforce_shown_twin_invariant(
        out_plan, _img, get_raw_boxes=_boxes,
        get_ocr=lambda f: str(vision_item.get(f, {}).get("ocr_clean") or ""),
        skip_files=system_files)
    for seg, loser, survivor, ham_v, contained in twin_folds:
        print(f"[dedup-invariant] {seg}: {loser} folded into {survivor} "
              f"(masked ham={ham_v}, containment={contained})")
    if twin_folds:
        out_plan = merge_consecutive_same_image_cuts(out_plan)

    # V1: a single static display longer than the cap is unwatchable even on
    # a panel that OWNS its narration (the 22.8s eye + cleaned-empty bubble)
    # — split it into 2-3 ken-varied sub-cuts. MUST run after the same-image
    # merges above (they would collapse the sub-cuts back into one).
    _shown_cache: Dict[str, Optional[np.ndarray]] = {}

    def _shown_clean_file(f: str) -> Optional[np.ndarray]:
        if f not in _shown_cache:
            _shown_cache[f] = cv2.imread(os.path.join(clean_dir, f))
        return _shown_cache[f]

    def _focal(f: str) -> Tuple[float, float, str]:
        return focal_point_for_crop(
            _shown_clean_file(f),
            dead_boxes=shown_dead_boxes.get(f, ()),
            face_center=shown_face.get(f))

    out_plan, kv_logs = split_long_hold_cuts(
        out_plan, max_hold_sec=args.max_hold_sec, focal_for_file=_focal,
        skip_files=system_files)
    for seg, f, dur, n_sub, src in kv_logs:
        print(f"[ok] {seg}: ken-variety split {f} ({dur:.1f}s static > "
              f"{args.max_hold_sec:.1f}s cap) -> {n_sub} sub-cuts "
              f"(focal={src})")

    # V2: shown-crop echo pairs (crop twins whose RAW panels are distinct —
    # the artist zoom-echo / husk re-crop class) get DIFFERENT ken regions so
    # the repeat reads as intentional emphasis, never a stutter. Motion only;
    # nothing dropped, no narration merged.
    out_plan, echo_logs = ken_differentiate_echo_pairs(
        out_plan, _shown_clean_file,
        lambda f: shown_bubble_boxes.get(f, ()),
        _img, _boxes, focal_for_file=_focal, skip_files=system_files)
    for seg_i, fi, seg_j, fj, sham, rham in echo_logs:
        print(f"[ok] {seg_j}: perceptual echo {fj} ~ {fi} ({seg_i}) "
              f"crop_ham={sham} raw_ham={rham} -> ken differentiation")

    which = "none" if args.no_branding else args.branding
    if which != "none":
        # No intro AND no outro timeline items anymore (channel decision) — the
        # call is kept so the --branding contract is unchanged for bundle
        # callers, but insert_branding_items now inserts nothing; the video ends
        # on its last story panel (only the corner watermark overlay remains).
        out_plan = insert_branding_items(out_plan, intro_dur=0.0,
                                         outro_dur=0.0, which=which)

    # prep_qa gates its bubble-interior checks (visible_text & co.) on this:
    # in keep mode surviving bubbles ship AS DRAWN, so readable bubble text on
    # a shown frame is design, not a blanking miss.
    out_plan["bubble_shown_mode"] = args.bubble_shown_mode

    out_path = args.out_plan or (os.path.splitext(args.plan)[0] + ".clean.json")
    script_path = os.path.join(args.episode_dir, "manifest.script.json")
    beats_path = os.path.join(args.episode_dir, "manifest.beats.json")
    tts_index_path = os.path.join(args.episode_dir, "tts", "tts_index.json")
    write_manifest(out_path, out_plan,
                   inputs=(script_path, beats_path, tts_index_path),
                   tool="render_prep")

    print(f"[ok] wrote={out_path} shown={len(shown)} "
          f"seam_dups_dropped={sorted(set(all_dropped))} bubbles_inpainted={bubbles_cleaned} "
          f"branding=none (ends on last story panel) "
          f"total={out_plan.get('total_duration_sec', 0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
