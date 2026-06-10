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
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Drop cuts whose GLOBAL box is >= contain_frac inside another cut's box.

    geom_by_file: {file: {x1,y1,x2,y2}} in global page pixels. The smaller box
    is the fragment; the complete panel survives. Freed time is redistributed
    proportionally so the shot window stays fully covered.
    """
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
                if small_file not in dropped:
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
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Drop the smaller cut of any pair whose pixels match inside the larger."""
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
            if ratio <= area_ratio_max and visually_contained(small, big, thresh=thresh):
                dropped.append(small_f)
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

def bubble_mask_in_box(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    *,
    pad: int = 4,
) -> np.ndarray:
    """uint8 mask (255 = erase) of the bubble inside a detector box.

    The bubble interior is the near-white (or near-black, for shout bubbles)
    connected component around the box centre — NOT the whole box, so art
    around the oval survives. Morphological close swallows the text strokes;
    dilation swallows the outline ring. Clipped to the padded box.
    """
    H, W = img.shape[:2]
    x1 = max(0, int(box[0]) - pad)
    y1 = max(0, int(box[1]) - pad)
    x2 = min(W, int(box[2]) + pad)
    y2 = min(H, int(box[3]) + pad)
    mask = np.zeros((H, W), np.uint8)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return mask

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
    comp = None
    if white is not None and (black is None or white.sum() >= black.sum()):
        comp = white
    elif black is not None:
        comp = black
    if comp is None:
        return mask

    comp = cv2.morphologyEx(
        comp, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    comp = cv2.dilate(
        comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask[y1:y2, x1:x2] = comp * 255
    return mask


def clean_scene_image(
    img: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Inpaint every bubble box's oval mask; untouched pixels stay identical."""
    if not boxes:
        return img
    total = np.zeros(img.shape[:2], np.uint8)
    for b in boxes:
        total = cv2.bitwise_or(total, bubble_mask_in_box(img, b))
    if not total.any():
        return img
    return cv2.inpaint(img, total, 5, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# plan rewrite (pure)
# ---------------------------------------------------------------------------

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
    ap.add_argument("--out-plan", default="", help="default: <plan>.clean.json next to --plan")
    ap.add_argument("--bubble-conf", type=float, default=0.30)
    ap.add_argument("--no-bubbles", action="store_true", help="skip bubble inpainting")
    ap.add_argument("--no-trim", action="store_true", help="skip border trimming")
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

    scenes_dir = os.path.join(args.episode_dir, "scenes")
    img_cache: Dict[str, Optional[np.ndarray]] = {}

    def _img(fname: str) -> Optional[np.ndarray]:
        if fname not in img_cache:
            img_cache[fname] = cv2.imread(os.path.join(scenes_dir, fname))
        return img_cache[fname]

    # 1. drop seam duplicates per shot — geometric first, then VISUAL
    # containment (global coords miss stitch-overlap duplicates entirely).
    cuts_by_segment: Dict[str, List[Dict[str, Any]]] = {}
    all_dropped: List[str] = []
    for item in plan.get("timeline") or []:
        cuts = item.get("cuts") or []
        new_cuts, dropped = drop_contained_duplicate_cuts(cuts, geom)
        if len(new_cuts) > 1:
            imgs = {str(c["file"]): _img(str(c["file"])) for c in new_cuts}
            imgs = {k: v for k, v in imgs.items() if v is not None}
            new_cuts, vdropped = drop_visual_duplicate_cuts(new_cuts, imgs)
            dropped = list(dropped) + vdropped
        cuts_by_segment[item["segment_id"]] = new_cuts
        all_dropped.extend(dropped)

    shown = sorted({c["file"] for cs in cuts_by_segment.values() for c in cs})

    # 2+3. clean + trim shown scenes into scenes_clean/
    clean_dir = os.path.join(args.episode_dir, "scenes_clean")
    os.makedirs(clean_dir, exist_ok=True)

    detector = None
    if not args.no_bubbles:
        detector = _load_bubble_detector(args.device)

    scene_dims: Dict[str, Dict[str, int]] = {}
    bubbles_cleaned = 0
    for fname in shown:
        src = os.path.join(scenes_dir, fname)
        img = cv2.imread(src)
        if img is None:
            print(f"[warn] unreadable scene, kept original reference: {fname}")
            continue

        boxes: List[Tuple[int, int, int, int]] = []
        if detector is not None:
            for (bx1, by1, bx2, by2, _score) in detector.detect(
                    img, imgsz=1024, conf=args.bubble_conf):
                boxes.append((int(bx1), int(by1), int(bx2), int(by2)))
        if boxes:
            img = clean_scene_image(img, boxes)
            bubbles_cleaned += len(boxes)

        if not args.no_trim:
            x1, y1, x2, y2 = content_bbox(img)
            img = img[y1:y2, x1:x2]

        cv2.imwrite(os.path.join(clean_dir, fname), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        h, w = img.shape[:2]
        scene_dims[fname] = {"w": int(w), "h": int(h)}
        print(f"[ok] {fname}: bubbles={len(boxes)} -> {w}x{h}")

    out_plan = rewrite_plan(plan, scenes_subdir="scenes_clean",
                            scene_dims=scene_dims,
                            cuts_by_segment=cuts_by_segment)
    out_path = args.out_plan or (os.path.splitext(args.plan)[0] + ".clean.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_plan, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote={out_path} shown={len(shown)} "
          f"seam_dups_dropped={sorted(set(all_dropped))} bubbles_inpainted={bubbles_cleaned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
