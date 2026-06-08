#!/usr/bin/env python3
"""
panels_to_scenes.py

Input:
  --stitch-manifest: manifest.stitch.json (chunk images + paths)
  --panels-manifest: manifest.panels.expanded.json (or manifest.panels.json)

Output:
  --out-dir: directory of cropped scene JPGs
  --out-manifest: manifest.scenes.json

Key features:
  - crops panels from each chunk
  - optional trim that is protected-span aware
  - IMPORTANT FIX: splits "merged" crops on internal gutters BEFORE trimming,
    using gutter-likeness + protected spans so giant blank gaps disappear
  - optional dedupe, blank skipping, narration keeping
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from studio.paths import resolve_rel

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
from collections import deque


# -----------------------------
# IO helpers
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Box helpers
# -----------------------------
def norm_to_px_xyxy(b: List[float], w: int, h: int) -> List[int]:
    y0, x0, y1, x1 = [float(v) for v in b]
    y0 = clamp(y0, 0.0, 1.0)
    x0 = clamp(x0, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x1 = clamp(x1, 0.0, 1.0)
    px0 = int(round(x0 * w))
    py0 = int(round(y0 * h))
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    px0 = int(clamp(px0, 0, w - 2))
    py0 = int(clamp(py0, 0, h - 2))
    px1 = int(clamp(px1, px0 + 2, w))
    py1 = int(clamp(py1, py0 + 2, h))
    return [px0, py0, px1, py1]

def px_xyxy_to_norm(box: List[int], w: int, h: int) -> List[float]:
    x0, y0, x1, y1 = box
    return [y0 / h, x0 / w, y1 / h, x1 / w]

def pad_box_xyxy(box: List[int], w: int, h: int, pad: int) -> List[int]:
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    if x1 <= x0 + 1:
        x1 = min(w, x0 + 2)
    if y1 <= y0 + 1:
        y1 = min(h, y0 + 2)
    return [x0, y0, x1, y1]

def area_px(box: List[int]) -> int:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


# -----------------------------
# DHash for dedupe
# -----------------------------
def dhash64(im: Image.Image) -> int:
    # 9x8 grayscale
    g = im.convert("L").resize((9, 8), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    out = 0
    for i, b in enumerate(bits):
        if b:
            out |= (1 << i)
    return int(out)

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# -----------------------------
# Blank / edge metrics
# -----------------------------
def blank_score_and_edge_density(im: Image.Image, sample_w: int = 420) -> Tuple[float, float]:
    # Downsample for speed; compute "blankness" and "edge"
    rgb = im.convert("RGB")
    w, h = rgb.size
    if w > sample_w:
        nh = max(1, int(round(h * (sample_w / w))))
        rgb = rgb.resize((sample_w, nh), Image.BILINEAR)
    arr = np.asarray(rgb).astype(np.float32)  # (h,w,3)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    white = (lum >= 245).mean()
    black = (lum <= 12).mean()
    blank = float(max(white, black))
    dx = np.abs(lum[:, 1:] - lum[:, :-1]).mean() / 255.0
    dy = np.abs(lum[1:, :] - lum[:-1, :]).mean() / 255.0
    edge = float(0.5 * (dx + dy))
    return blank, edge


# -----------------------------
# Protected spans
# -----------------------------
def load_protected_spans_for_chunk(protected_dir: str, chunk_file: str, suffix: str) -> List[List[int]]:
    """
    protected_dir contains files like: chunk_0001.protected_spans.json
    Returns list of [y0,y1] in chunk pixel coords.
    """
    if not protected_dir:
        return []
    base = os.path.splitext(os.path.basename(chunk_file))[0]  # chunk_0001
    p = os.path.join(protected_dir, f"{base}{suffix}")
    if not os.path.exists(p):
        return []
    try:
        d = load_json(p)
        spans = d.get("protected_spans") or []
        out: List[List[int]] = []
        for s in spans:
            if isinstance(s, list) and len(s) == 2:
                y0, y1 = int(s[0]), int(s[1])
                if y1 > y0:
                    out.append([y0, y1])
        out.sort(key=lambda t: (t[0], t[1]))
        return out
    except Exception:
        return []

def chunk_spans_to_crop_spans(chunk_spans: List[List[int]], crop_y0: int, crop_y1: int) -> List[List[int]]:
    out: List[List[int]] = []
    for sy0, sy1 in chunk_spans:
        iy0 = max(crop_y0, sy0)
        iy1 = min(crop_y1, sy1)
        if iy1 > iy0:
            out.append([iy0 - crop_y0, iy1 - crop_y0])
    out.sort(key=lambda s: (s[0], s[1]))
    return out

def spans_intersect_y(spans: List[List[int]], y0: int, y1: int) -> bool:
    for a, b in spans:
        if b > y0 and a < y1:
            return True
    return False


# -----------------------------
# Content-aware trim (protected-span aware)
# -----------------------------
def _rgb_dist(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def _downsample_for_mask(im: Image.Image, max_w: int) -> Tuple[Image.Image, float]:
    w, h = im.size
    if w <= max_w:
        return im, 1.0
    scale = max_w / float(w)
    nh = max(1, int(round(h * scale)))
    return im.resize((max_w, nh), Image.BILINEAR), scale

def _sample_border_colors(px: List[Tuple[int, int, int]], w: int, h: int, step: int = 3) -> List[Tuple[int, int, int]]:
    reps: List[Tuple[int, int, int]] = []
    # sample top/bottom rows + left/right cols
    for x in range(0, w, step):
        reps.append(px[x])
        reps.append(px[(h-1)*w + x])
    for y in range(0, h, step):
        reps.append(px[y*w])
        reps.append(px[y*w + (w-1)])
    # de-dup coarse
    out: List[Tuple[int, int, int]] = []
    for c in reps:
        ok = True
        for k in out:
            if _rgb_dist(c, k) <= 20:
                ok = False
                break
        if ok:
            out.append(c)
    return out[:18]

def protected_limits(spans_local: List[List[int]], h: int) -> Tuple[int, int]:
    if not spans_local:
        return (h, 0)  # no restriction
    top = max(0, min(s[0] for s in spans_local))
    bot = min(h, max(s[1] for s in spans_local))
    return (top, bot)

def content_aware_trim(
    crop: Image.Image,
    *,
    protected_spans_local: Optional[List[List[int]]] = None,
    max_trim_px: int = 1600,
    max_w: int = 520,
    bg_sim_tol: int = 70,
    near_white_v: int = 245,
    near_black_v: int = 12,
    pad_px: int = 4,
    min_keep_h: int = 140,
    min_keep_w: int = 120,
    min_content_frac: float = 0.28,
) -> Tuple[Image.Image, Dict[str, Any]]:
    protected_spans_local = protected_spans_local or []
    w0, h0 = crop.size
    if w0 <= 0 or h0 <= 0:
        return crop, {"trimmed": False, "reason": "empty"}
    if h0 <= min_keep_h + 2 or w0 <= min_keep_w + 2:
        return crop, {"trimmed": False, "reason": "too_small_for_trim"}

    top_limit, bot_limit = protected_limits(protected_spans_local, h0)

    im_small, scale = _downsample_for_mask(crop.convert("RGB"), max_w=max_w)
    w, h = im_small.size
    px = list(im_small.getdata())

    reps = _sample_border_colors(px, w, h, step=3)
    if not reps:
        return crop, {"trimmed": False, "reason": "no_border_samples"}

    bg_candidate = [False] * (w * h)
    for i, (r, g, b) in enumerate(px):
        v = (r + g + b) // 3
        extreme = (v >= near_white_v) or (v <= near_black_v)
        sim = False
        if not extreme:
            for rep in reps:
                if _rgb_dist((r, g, b), rep) <= bg_sim_tol:
                    sim = True
                    break
        if extreme or sim:
            bg_candidate[i] = True

    q = deque()
    edge_bg = [False] * (w * h)

    def push_if(x: int, y: int):
        idx = y * w + x
        if bg_candidate[idx] and not edge_bg[idx]:
            edge_bg[idx] = True
            q.append((x, y))

    for x in range(w):
        push_if(x, 0)
        push_if(x, h - 1)
    for y in range(h):
        push_if(0, y)
        push_if(w - 1, y)

    while q:
        x, y = q.popleft()
        if x > 0:
            push_if(x - 1, y)
        if x + 1 < w:
            push_if(x + 1, y)
        if y > 0:
            push_if(x, y - 1)
        if y + 1 < h:
            push_if(x, y + 1)

    minx, miny = w, h
    maxx, maxy = -1, -1
    fg_count = 0
    for yy in range(h):
        row_off = yy * w
        for xx in range(w):
            idx = row_off + xx
            if not edge_bg[idx]:
                fg_count += 1
                if xx < minx: minx = xx
                if yy < miny: miny = yy
                if xx > maxx: maxx = xx
                if yy > maxy: maxy = yy

    if fg_count == 0 or maxx < minx or maxy < miny:
        return crop, {"trimmed": False, "reason": "no_foreground"}

    fg_frac = fg_count / float(w * h)
    if fg_frac < min_content_frac:
        return crop, {"trimmed": False, "reason": "fg_frac_too_low", "fg_frac": round(fg_frac, 5)}

    pad_s = max(1, int(round(pad_px * scale))) if scale != 0 else 1
    minx = max(0, minx - pad_s)
    miny = max(0, miny - pad_s)
    maxx = min(w - 1, maxx + pad_s)
    maxy = min(h - 1, maxy + pad_s)

    x0 = int(round(minx / scale))
    y0 = int(round(miny / scale))
    x1 = int(round((maxx + 1) / scale))
    y1 = int(round((maxy + 1) / scale))

    x0 = max(0, min(w0 - 2, x0))
    y0 = max(0, min(h0 - 2, y0))
    x1 = max(x0 + 2, min(w0, x1))
    y1 = max(y0 + 2, min(h0, y1))

    max_trim = int(max_trim_px)
    x0 = min(x0, max_trim)
    y0 = min(y0, max_trim)
    x1 = max(x1, w0 - max_trim)
    y1 = max(y1, h0 - max_trim)

    if protected_spans_local:
        # don't trim below top protected text, and don't trim above bottom protected text
        y0 = min(y0, top_limit)
        y1 = max(y1, bot_limit)

    new_w = x1 - x0
    new_h = y1 - y0
    if new_h < min_keep_h or new_w < min_keep_w:
        return crop, {
            "trimmed": False,
            "reason": "min_keep_guard",
            "new_w": new_w,
            "new_h": new_h,
            "protected_used": bool(protected_spans_local),
        }

    if x0 == 0 and y0 == 0 and x1 == w0 and y1 == h0:
        return crop, {"trimmed": False, "reason": "no_change"}

    trimmed = crop.crop((x0, y0, x1, y1))
    return trimmed, {
        "trimmed": True,
        "mode": "content",
        "left_px": x0,
        "top_px": y0,
        "right_px": w0 - x1,
        "bottom_px": h0 - y1,
        "old_w": w0,
        "old_h": h0,
        "new_w": trimmed.width,
        "new_h": trimmed.height,
        "fg_frac_small": round(fg_frac, 5),
        "bg_sim_tol": int(bg_sim_tol),
        "protected_used": bool(protected_spans_local),
        "protected_count": len(protected_spans_local),
        "protected_top_limit": int(top_limit) if protected_spans_local else None,
        "protected_bot_limit": int(bot_limit) if protected_spans_local else None,
    }


# -----------------------------
# Internal gutter split (THE FIX)
# -----------------------------
@dataclass
class GutterParams:
    blank_thr: float = 0.985
    edge_max: float = 0.020
    min_run_px: int = 80
    margin_px: int = 30
    max_splits: int = 4

def _row_blank_edge(im: Image.Image, max_w: int = 420) -> Tuple[np.ndarray, np.ndarray, float]:
    rgb = im.convert("RGB")
    w0, h0 = rgb.size
    scale = 1.0
    if w0 > max_w:
        scale = max_w / float(w0)
        nh = max(1, int(round(h0 * scale)))
        rgb = rgb.resize((max_w, nh), Image.BILINEAR)
    arr = np.asarray(rgb).astype(np.float32)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    blank = np.maximum((lum >= 245).mean(axis=1), (lum <= 12).mean(axis=1))
    dx = np.abs(lum[:, 1:] - lum[:, :-1]).mean(axis=1) / 255.0
    return blank, dx, scale

def _find_best_internal_gutter_run(
    crop: Image.Image,
    spans_local: List[List[int]],
    gp: GutterParams,
) -> Optional[Tuple[int, int]]:
    """
    Returns (y0, y1) in crop pixel coords for the best internal gutter run to split on.
    """
    blank_s, edge_s, scale = _row_blank_edge(crop, max_w=420)
    h_s = blank_s.shape[0]
    # map thresholds to sampled coords
    min_run_s = max(3, int(round(gp.min_run_px * scale)))
    margin_s = max(2, int(round(gp.margin_px * scale)))

    good = (blank_s >= gp.blank_thr) & (edge_s <= gp.edge_max)
    # find runs
    runs: List[Tuple[int, int, float]] = []
    i = 0
    while i < h_s:
        if not good[i]:
            i += 1
            continue
        j = i
        while j < h_s and good[j]:
            j += 1
        run_len = j - i
        if run_len >= min_run_s and i >= margin_s and j <= (h_s - margin_s):
            # avoid splitting through protected spans (convert span to sampled coords)
            if spans_local:
                # if any span intersects the run in full-res, reject
                y0_full = int(round(i / scale))
                y1_full = int(round(j / scale))
                if spans_intersect_y(spans_local, y0_full, y1_full):
                    i = j
                    continue
            # score: longer + blanker
            score = float(run_len) + float(blank_s[i:j].mean()) * 10.0
            runs.append((i, j, score))
        i = j

    if not runs:
        return None

    runs.sort(key=lambda t: t[2], reverse=True)
    i0, i1, _ = runs[0]
    y0 = int(round(i0 / scale))
    y1 = int(round(i1 / scale))
    y0 = int(clamp(y0, 0, crop.height))
    y1 = int(clamp(y1, 0, crop.height))
    if y1 <= y0 + 2:
        return None
    return (y0, y1)

def split_crop_on_gutters(
    crop: Image.Image,
    crop_box_in_chunk: List[int],
    spans_local: List[List[int]],
    gp: GutterParams,
    min_h_px: int,
) -> List[Tuple[Image.Image, List[int], List[List[int]]]]:
    """
    Iteratively split a crop into multiple sub-crops on internal gutter runs.
    Returns list of (sub_image, sub_box_in_chunk, sub_spans_local).
    """
    out: List[Tuple[Image.Image, List[int], List[List[int]]]] = [(crop, crop_box_in_chunk, spans_local)]
    for _ in range(gp.max_splits):
        changed = False
        new_out: List[Tuple[Image.Image, List[int], List[List[int]]]] = []
        for im, box, spans in out:
            found = _find_best_internal_gutter_run(im, spans, gp)
            if not found:
                new_out.append((im, box, spans))
                continue
            gy0, gy1 = found
            # split at center of gutter run
            split_y = (gy0 + gy1) // 2
            if split_y < min_h_px or (im.height - split_y) < min_h_px:
                new_out.append((im, box, spans))
                continue

            # compute sub-images + sub-boxes (chunk coords)
            x0, y0, x1, y1 = box
            top_im = im.crop((0, 0, im.width, split_y))
            bot_im = im.crop((0, split_y, im.width, im.height))

            top_box = [x0, y0, x1, y0 + split_y]
            bot_box = [x0, y0 + split_y, x1, y1]

            # split spans
            top_spans: List[List[int]] = []
            bot_spans: List[List[int]] = []
            for sy0, sy1 in spans:
                if sy1 <= split_y:
                    top_spans.append([sy0, sy1])
                elif sy0 >= split_y:
                    bot_spans.append([sy0 - split_y, sy1 - split_y])
                else:
                    # span crosses split -> do not split this crop
                    top_im = None  # mark invalid split
                    break
            if top_im is None:
                new_out.append((im, box, spans))
                continue

            new_out.append((top_im, top_box, top_spans))
            new_out.append((bot_im, bot_box, bot_spans))
            changed = True

        out = new_out
        if not changed:
            break
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-manifest", required=True)
    ap.add_argument("--panels-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)

    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--pad-px", type=int, default=4)

    ap.add_argument("--min-area-frac", type=float, default=0.002)
    ap.add_argument("--min-h-px", type=int, default=180)
    ap.add_argument("--min-w-px", type=int, default=240)

    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--dedupe-threshold", type=int, default=8)  # hamming
    ap.add_argument("--dedupe-ratio-tol", type=float, default=0.08)
    ap.add_argument("--dedupe-area-tol", type=float, default=0.10)
    ap.add_argument("--dedupe-lookback", type=int, default=80)

    ap.add_argument("--blank-threshold", type=float, default=0.985)
    ap.add_argument("--blank-max-edge", type=float, default=0.018)
    ap.add_argument("--skip-blank", action="store_true")

    ap.add_argument("--save-narration", action="store_true")  # keep blank-ish if text panels exist

    ap.add_argument("--trim-margins", action="store_true")
    ap.add_argument("--trim-max-px", type=int, default=1600)
    ap.add_argument("--trim-mask-max-w", type=int, default=520)
    ap.add_argument("--trim-bg-sim-tol", type=int, default=70)
    ap.add_argument("--trim-near-white", type=int, default=245)
    ap.add_argument("--trim-near-black", type=int, default=12)
    ap.add_argument("--trim-pad-px", type=int, default=4)
    ap.add_argument("--trim-min-keep-h", type=int, default=140)
    ap.add_argument("--trim-min-keep-w", type=int, default=120)
    ap.add_argument("--trim-min-content-frac", type=float, default=0.28)

    ap.add_argument("--protected-spans-dir", type=str, default="")
    ap.add_argument("--protected-spans-suffix", type=str, default=".protected_spans.json")

    ap.add_argument("--panel-id-mode", choices=["sequential", "by_chunk"], default="sequential")
    ap.add_argument("--progress-every", type=int, default=20)
    args = ap.parse_args()

    if not os.path.exists(args.stitch_manifest):
        raise SystemExit(f"missing stitch manifest: {args.stitch_manifest}")
    if not os.path.exists(args.panels_manifest):
        raise SystemExit(f"missing panels manifest: {args.panels_manifest}")

    stitch = load_json(args.stitch_manifest)
    panels = load_json(args.panels_manifest)

    stitch_chunks = stitch.get("chunks") or []
    if not stitch_chunks:
        raise SystemExit("stitch manifest has no chunks")

    panels_chunks = panels.get("chunks") or []
    if not panels_chunks:
        raise SystemExit("panels manifest has no chunks")

    ensure_dir(args.out_dir)

    stitch_by_file = {c.get("chunk_file"): c for c in stitch_chunks if c.get("chunk_file")}
    protected_cache: Dict[str, List[List[int]]] = {}

    # dedupe memory
    recent: List[Dict[str, Any]] = []

    scenes: List[Dict[str, Any]] = []
    global_y = 0
    chunk_global_y0: Dict[str, int] = {}

    # compute global y offsets by stitch order (fallback: manifest order)
    for ch in stitch_chunks:
        cf = ch.get("chunk_file")
        cp_stored = ch.get("chunk_path")
        cp = str(resolve_rel(args.stitch_manifest, cp_stored)) if cp_stored else ""
        if not cf or not cp or not os.path.exists(cp):
            continue
        with Image.open(cp) as im:
            h = im.height
        chunk_global_y0[cf] = global_y
        global_y += h

    # splitting parameters
    gp = GutterParams(
        blank_thr=float(args.blank_threshold),
        edge_max=float(args.blank_max_edge),
        min_run_px=90,
        margin_px=24,
        max_splits=4,
    )

    total_panels_in = 0
    written = 0
    skipped_blank = 0
    skipped_small = 0
    skipped_dedupe = 0

    seq_id = 0

    for cidx, ch in enumerate(panels_chunks):
        cf = ch.get("chunk_file")
        if not cf:
            continue
        st = stitch_by_file.get(cf) or {}
        chunk_path_stored = st.get("chunk_path") or ch.get("chunk_path")
        chunk_path = str(resolve_rel(args.stitch_manifest, chunk_path_stored)) if chunk_path_stored else ""
        if not chunk_path or not os.path.exists(chunk_path):
            continue

        boxes_norm = ch.get("panels_norm") or []
        if not isinstance(boxes_norm, list):
            continue

        with Image.open(chunk_path) as chunk_im:
            chunk_im = chunk_im.convert("RGB")
            cw, chh = chunk_im.size

            # load protected spans once per chunk
            chunk_spans: List[List[int]] = []
            if args.protected_spans_dir:
                if cf not in protected_cache:
                    protected_cache[cf] = load_protected_spans_for_chunk(
                        args.protected_spans_dir, cf, args.protected_spans_suffix
                    )
                chunk_spans = protected_cache.get(cf, [])

            for pidx, b in enumerate(boxes_norm):
                if not (isinstance(b, list) and len(b) == 4):
                    continue
                total_panels_in += 1

                box_xyxy = norm_to_px_xyxy(b, cw, chh)
                box_xyxy = pad_box_xyxy(box_xyxy, cw, chh, int(args.pad_px))

                # size guards
                if (box_xyxy[3] - box_xyxy[1]) < int(args.min_h_px) or (box_xyxy[2] - box_xyxy[0]) < int(args.min_w_px):
                    skipped_small += 1
                    continue

                # area guard
                if area_px(box_xyxy) < int(args.min_area_frac * (cw * chh)):
                    skipped_small += 1
                    continue

                crop = chunk_im.crop(tuple(box_xyxy))

                # protected spans in crop coords
                crop_local_spans = chunk_spans_to_crop_spans(chunk_spans, box_xyxy[1], box_xyxy[3])

                # THE FIX: split first (pre-trim) if a merged crop contains big internal gutters
                split_parts = split_crop_on_gutters(
                    crop=crop,
                    crop_box_in_chunk=box_xyxy,
                    spans_local=crop_local_spans,
                    gp=gp,
                    min_h_px=int(args.min_h_px),
                )

                for part_idx, (part_im, part_box, part_spans) in enumerate(split_parts):
                    # optional trim AFTER splitting
                    trim_info: Dict[str, Any] = {"trimmed": False, "reason": "disabled"}
                    if args.trim_margins:
                        part_im, trim_info = content_aware_trim(
                            part_im,
                            protected_spans_local=part_spans,
                            max_trim_px=int(args.trim_max_px),
                            max_w=int(args.trim_mask_max_w),
                            bg_sim_tol=int(args.trim_bg_sim_tol),
                            near_white_v=int(args.trim_near_white),
                            near_black_v=int(args.trim_near_black),
                            pad_px=int(args.trim_pad_px),
                            min_keep_h=int(args.trim_min_keep_h),
                            min_keep_w=int(args.trim_min_keep_w),
                            min_content_frac=float(args.trim_min_content_frac),
                        )

                    blank, edge = blank_score_and_edge_density(part_im)
                    is_blankish = (blank >= float(args.blank_threshold) and edge <= float(args.blank_max_edge))

                    # if blankish and not saving narration, skip
                    if args.skip_blank and is_blankish and not args.save_narration:
                        skipped_blank += 1
                        continue

                    # dedupe
                    dh = dhash64(part_im)
                    if args.dedupe and recent:
                        r_w, r_h = part_im.size
                        r_ratio = r_w / max(1, r_h)
                        r_area = r_w * r_h
                        dup = False
                        for prev in recent[-int(args.dedupe_lookback):]:
                            ham = hamming64(dh, prev["dhash64"])
                            if ham > int(args.dedupe_threshold):
                                continue
                            pw, ph = prev["w"], prev["h"]
                            pr = pw / max(1, ph)
                            pa = pw * ph
                            if abs(r_ratio - pr) > float(args.dedupe_ratio_tol):
                                continue
                            if abs(r_area - pa) / max(1, pa) > float(args.dedupe_area_tol):
                                continue
                            dup = True
                            break
                        if dup:
                            skipped_dedupe += 1
                            continue

                    if args.panel_id_mode == "sequential":
                        panel_id = f"p{seq_id:06d}"
                        seq_id += 1
                    else:
                        panel_id = f"c{cidx:04d}_p{pidx:04d}_{part_idx:02d}"

                    out_name = f"{panel_id}.jpg"
                    out_path = os.path.join(args.out_dir, out_name)
                    part_im.save(out_path, "JPEG", quality=int(args.jpeg_quality), optimize=True)

                    # record
                    gx0, gy0, gx1, gy1 = part_box
                    scenes.append(
                        {
                            "panel_id": panel_id,
                            "chunk_file": cf,
                            "chunk_path": os.path.abspath(chunk_path),
                            "chunk_w": cw,
                            "chunk_h": chh,
                            "chunk_global_y0": int(chunk_global_y0.get(cf, 0)),
                            "panel_index_in_chunk": int(pidx),
                            "part_index": int(part_idx),
                            "box_px_xyxy": [int(gx0), int(gy0), int(gx1), int(gy1)],
                            "box_norm": px_xyxy_to_norm([gx0, gy0, gx1, gy1], cw, chh),
                            "out_file": out_name,
                            "out_path": os.path.abspath(out_path),
                            "w": part_im.width,
                            "h": part_im.height,
                            "blank_score": float(round(blank, 6)),
                            "edge_density": float(round(edge, 6)),
                            "trim": trim_info,
                            "protected_spans_local": part_spans,
                            "dhash64": int(dh),
                            "split": {
                                "enabled": True,
                                "max_splits": int(gp.max_splits),
                                "blank_thr": float(gp.blank_thr),
                                "edge_max": float(gp.edge_max),
                            },
                        }
                    )
                    recent.append({"dhash64": dh, "w": part_im.width, "h": part_im.height})
                    written += 1

                    if args.progress_every > 0 and (written % int(args.progress_every) == 0):
                        print(f"[prog] written={written} scenes total_panels_in={total_panels_in} (blank={skipped_blank} small={skipped_small} dedupe={skipped_dedupe})")

    out_obj = {
        "source_stitch_manifest": os.path.abspath(args.stitch_manifest),
        "source_panels_manifest": os.path.abspath(args.panels_manifest),
        "out_dir": os.path.abspath(args.out_dir),
        "count_scenes": len(scenes),
        "stats": {
            "written": int(written),
            "panels_in": int(total_panels_in),
            "skipped_blank": int(skipped_blank),
            "skipped_small": int(skipped_small),
            "skipped_dedupe": int(skipped_dedupe),
        },
        "params": {
            "jpeg_quality": int(args.jpeg_quality),
            "pad_px": int(args.pad_px),
            "min_area_frac": float(args.min_area_frac),
            "min_h_px": int(args.min_h_px),
            "min_w_px": int(args.min_w_px),
            "dedupe": bool(args.dedupe),
            "dedupe_threshold": int(args.dedupe_threshold),
            "dedupe_ratio_tol": float(args.dedupe_ratio_tol),
            "dedupe_area_tol": float(args.dedupe_area_tol),
            "dedupe_lookback": int(args.dedupe_lookback),
            "blank_threshold": float(args.blank_threshold),
            "blank_max_edge": float(args.blank_max_edge),
            "skip_blank": bool(args.skip_blank),
            "save_narration": bool(args.save_narration),
            "trim_margins": bool(args.trim_margins),
            "trim_max_px": int(args.trim_max_px),
            "trim_mask_max_w": int(args.trim_mask_max_w),
            "trim_bg_sim_tol": int(args.trim_bg_sim_tol),
            "trim_near_white": int(args.trim_near_white),
            "trim_near_black": int(args.trim_near_black),
            "trim_pad_px": int(args.trim_pad_px),
            "trim_min_keep_h": int(args.trim_min_keep_h),
            "trim_min_keep_w": int(args.trim_min_keep_w),
            "trim_min_content_frac": float(args.trim_min_content_frac),
            "protected_spans_dir": args.protected_spans_dir,
            "protected_spans_suffix": args.protected_spans_suffix,
            "panel_id_mode": args.panel_id_mode,
        },
        "scenes": scenes,
    }

    dump_json(args.out_manifest, out_obj)
    print(f"[ok] wrote scenes: {os.path.abspath(args.out_manifest)}  scenes={len(scenes)}  out_dir={os.path.abspath(args.out_dir)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
