#!/usr/bin/env python3
"""
chunk_stitch_adaptive.py (robust gutters + run-based cuts + overflow)

Goal:
- Create tall stitched "chunks" from many webtoon images.
- Split chunks ONLY at safe "gutter bands" (runs) so panels/text are not cut in half.
- Gutters can be white, black, or flat/low-detail fades.
- If no safe cut is found near the target height, we delay cutting (overflow) until we find one.

Output:
  <episode>/stitch_chunks/chunk_0001.jpg ...
  <episode>/manifest.stitch.json

Manifest stays compatible:
  chunks[].sources[] contains y0/y1 offsets in chunk space.
"""

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import numpy as np
except Exception:
    np = None


def natural_key(path: str):
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    return (int(nums[0]) if nums else 0, base)


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resize_to_width(im: Image.Image, target_w: int) -> Image.Image:
    if im.width == target_w:
        return im
    scale = target_w / float(im.width)
    nh = int(round(im.height * scale))
    return im.resize((target_w, nh), Image.Resampling.LANCZOS)


# -----------------------------
# Per-row metrics (robust)
# -----------------------------
def _row_metrics_rgb(canvas: Image.Image) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Returns per row:
      white_frac[y] : fraction of near-white pixels on row y
      black_frac[y] : fraction of near-black pixels on row y
      edge_score[y] : avg abs horizontal gradient magnitude on row y (luma)
      var_luma[y]   : variance of luma along the row (flatness proxy)
    """
    w, h = canvas.size

    # thresholds (intentionally simple; we auto-calibrate with percentiles later)
    WHITE_T = 245
    BLACK_T = 12

    if np is None:
        px = canvas.load()
        white_frac = [0.0] * h
        black_frac = [0.0] * h
        edge_score = [0.0] * h
        var_luma = [0.0] * h

        step = max(1, w // 700)  # sample ~700 columns
        for y in range(h):
            whites = 0
            blacks = 0
            lumas = []
            last = None
            grad_sum = 0.0
            cnt = 0
            for x in range(0, w, step):
                r, g, b = px[x, y]
                luma = (0.299 * r + 0.587 * g + 0.114 * b)
                lumas.append(luma)
                if r >= WHITE_T and g >= WHITE_T and b >= WHITE_T:
                    whites += 1
                if r <= BLACK_T and g <= BLACK_T and b <= BLACK_T:
                    blacks += 1
                if last is not None:
                    grad_sum += abs(luma - last)
                last = luma
                cnt += 1
            white_frac[y] = whites / float(max(1, cnt))
            black_frac[y] = blacks / float(max(1, cnt))
            edge_score[y] = grad_sum / float(max(1, cnt))
            if lumas:
                m = sum(lumas) / float(len(lumas))
                var_luma[y] = sum((v - m) * (v - m) for v in lumas) / float(max(1, len(lumas)))
        return white_frac, black_frac, edge_score, var_luma

    # Numpy path (fast)
    arr = np.asarray(canvas).astype(np.float32)  # (h,w,3)

    white = (arr[:, :, 0] >= WHITE_T) & (arr[:, :, 1] >= WHITE_T) & (arr[:, :, 2] >= WHITE_T)
    black = (arr[:, :, 0] <= BLACK_T) & (arr[:, :, 1] <= BLACK_T) & (arr[:, :, 2] <= BLACK_T)

    white_frac = white.mean(axis=1)
    black_frac = black.mean(axis=1)

    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    grad = np.abs(luma[:, 1:] - luma[:, :-1])
    edge_score = grad.mean(axis=1)

    mean_l = luma.mean(axis=1)
    var_l = ((luma - mean_l[:, None]) ** 2).mean(axis=1)

    return white_frac.tolist(), black_frac.tolist(), edge_score.tolist(), var_l.tolist()


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    if np is not None:
        return float(np.percentile(np.asarray(xs, dtype=np.float32), p))
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    k = max(0, min(len(s) - 1, k))
    return float(s[k])


def _find_runs(mask: List[bool], y0: int, y1: int) -> List[Tuple[int, int]]:
    """
    Return list of runs [a,b] inclusive where mask is True within [y0,y1].
    """
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for y in range(y0, y1 + 1):
        v = mask[y]
        if v and not in_run:
            in_run = True
            start = y
        elif not v and in_run:
            runs.append((start, y - 1))
            in_run = False
    if in_run:
        runs.append((start, y1))
    return runs


def _pick_cut_y_run_based(
    canvas: Image.Image,
    target_y: int,
    base_window: int,
    min_run_px: int,
    max_window: int,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Pick a cut using gutter RUNS. Gutters can be white, black, or flat/low-detail.
    Returns: (cut_y or None, debug dict)

    Strategy:
      - Compute row metrics.
      - Auto-calibrate thresholds using percentiles.
      - Build gutter-candidate mask.
      - Search around target_y for runs >= min_run_px.
      - Choose best run near target; cut at run midpoint.
      - If none, expand search window up to max_window; if still none -> None.
    """
    w, h = canvas.size
    if h <= 2:
        return None, {"reason": "too_small"}

    target_y = max(0, min(h - 1, int(target_y)))

    white_frac, black_frac, edge_score, var_luma = _row_metrics_rgb(canvas)

    # Auto thresholds from this canvas
    white_thr = max(0.60, _percentile(white_frac, 90))  # allow content with not-perfect white
    black_thr = max(0.60, _percentile(black_frac, 90))
    edge_thr = _percentile(edge_score, 20)              # low edges
    var_thr = _percentile(var_luma, 20)                 # flat rows

    # Candidate gutter row if:
    #  - very white OR very black OR (flat + low edges)
    gutter = [False] * h
    for y in range(h):
        if white_frac[y] >= white_thr:
            gutter[y] = True
        elif black_frac[y] >= black_thr:
            gutter[y] = True
        elif (edge_score[y] <= edge_thr and var_luma[y] <= var_thr):
            gutter[y] = True

    # Expand search progressively
    win = int(base_window)
    win = max(50, win)
    max_window = max(win, int(max_window))

    best_run = None  # (score, a, b)
    best_meta = {}

    while win <= max_window:
        y0 = max(0, target_y - win)
        y1 = min(h - 1, target_y + win)
        runs = _find_runs(gutter, y0, y1)

        # filter runs by length
        good = []
        for a, b in runs:
            if (b - a + 1) >= int(min_run_px):
                good.append((a, b))

        if good:
            # score runs: prefer close to target + strong gutter signature + low edges/var
            for a, b in good:
                mid = (a + b) // 2
                dist = abs(mid - target_y)

                wf = float(sum(white_frac[a:b+1]) / max(1, (b - a + 1)))
                bf = float(sum(black_frac[a:b+1]) / max(1, (b - a + 1)))
                es = float(sum(edge_score[a:b+1]) / max(1, (b - a + 1)))
                vv = float(sum(var_luma[a:b+1]) / max(1, (b - a + 1)))

                # higher is better:
                # - closer to target (penalty)
                # - stronger white/black (bonus)
                # - lower edges/var (bonus)
                score = (wf * 1.3) + (bf * 1.3) - (es * 0.015) - (vv * 0.0008) - (dist / float(win + 1)) * 0.8

                if best_run is None or score > best_run[0]:
                    best_run = (score, a, b)
                    best_meta = {
                        "window_used": win,
                        "target_y": target_y,
                        "run_a": a,
                        "run_b": b,
                        "run_len": (b - a + 1),
                        "run_mid": mid,
                        "avg_white": wf,
                        "avg_black": bf,
                        "avg_edge": es,
                        "avg_var": vv,
                        "thr_white": white_thr,
                        "thr_black": black_thr,
                        "thr_edge": edge_thr,
                        "thr_var": var_thr,
                    }
            break

        win = int(win * 1.6)

    if best_run is None:
        return None, {
            "reason": "no_gutter_run_found",
            "target_y": target_y,
            "base_window": base_window,
            "max_window": max_window,
            "min_run_px": min_run_px,
            "thr_white": white_thr,
            "thr_black": black_thr,
            "thr_edge": edge_thr,
            "thr_var": var_thr,
        }

    _, a, b = best_run
    cut_y = int((a + b) // 2)
    cut_y = max(1, min(h - 1, cut_y))

    best_meta["reason"] = "gutter_run"
    best_meta["cut_y"] = cut_y
    return cut_y, best_meta


def _force_cut_least_damage(canvas: Image.Image, target_y: int, window: int) -> Tuple[int, Dict[str, Any]]:
    """
    Last resort: pick a row near target with the lowest edge_score and low var.
    This is still better than random/white-only cuts.
    """
    w, h = canvas.size
    target_y = max(0, min(h - 1, int(target_y)))

    white_frac, black_frac, edge_score, var_luma = _row_metrics_rgb(canvas)

    y0 = max(0, target_y - int(window))
    y1 = min(h - 1, target_y + int(window))

    best_y = target_y
    best_val = 1e18
    for y in range(y0, y1 + 1):
        # smaller is better
        val = (edge_score[y] * 1.0) + (var_luma[y] * 0.002) + (abs(y - target_y) * 0.05)
        if val < best_val:
            best_val = val
            best_y = y

    best_y = max(1, min(h - 1, int(best_y)))
    return best_y, {
        "reason": "forced_least_damage",
        "target_y": target_y,
        "window": int(window),
        "cut_y": int(best_y),
    }


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--out-dir", default="stitch_chunks")
    ap.add_argument("--max-chunk-height", type=int, default=16000)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--target-width", type=int, default=0, help="0=auto (use max width across inputs)")

    # Adaptive behavior (no per-episode tuning required)
    ap.add_argument("--search-window", type=int, default=700, help="Base search +/- px for gutter run near target")
    ap.add_argument("--max-search-window", type=int, default=3500, help="Max expanded window for finding a gutter run")
    ap.add_argument("--min-gutter-run-px", type=int, default=60, help="Minimum contiguous gutter band height")
    ap.add_argument("--overlap-px", type=int, default=700, help="Overlap between consecutive chunks")

    # Overflow behavior: if no safe cut found, allow chunk to exceed max-chunk-height up to this many px
    ap.add_argument("--max-overflow-px", type=int, default=6000, help="Allow overflow before forcing a cut")

    # Debug
    ap.add_argument("--debug-cuts", action="store_true", help="Print cut decisions")
    args = ap.parse_args()

    episode_dir = os.path.abspath(args.episode_dir)
    paths = sorted(glob.glob(os.path.join(episode_dir, args.glob)), key=natural_key)
    if not paths:
        raise SystemExit(f"No images found: {episode_dir} glob={args.glob}")

    # Determine target width
    widths = []
    sizes_by_path: Dict[str, Tuple[int, int]] = {}
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
            sizes_by_path[p] = (w, h)
            widths.append(w)
    target_w = int(args.target_width) if args.target_width and args.target_width > 0 else max(widths)

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(episode_dir, out_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load + resize pages
    pages: List[Tuple[str, Image.Image]] = []
    for p in paths:
        im = load_rgb(p)
        im = resize_to_width(im, target_w)
        pages.append((p, im))

    chunks: List[Dict[str, Any]] = []
    chunk_index = 1

    cur_sources: List[Dict[str, Any]] = []
    cur_images: List[Image.Image] = []
    cur_h = 0

    def render_canvas(images: List[Image.Image]) -> Image.Image:
        hh = sum(im.height for im in images)
        canvas = Image.new("RGB", (target_w, hh), (255, 255, 255))
        y = 0
        for im in images:
            canvas.paste(im, (0, y))
            y += im.height
        return canvas

    def flush_by_cut(cut_y: int, keep_overlap: int, cut_debug: Optional[Dict[str, Any]] = None):
        nonlocal chunk_index, cur_sources, cur_images, cur_h, chunks

        if not cur_images:
            return

        canvas = render_canvas(cur_images)
        w, h = canvas.size

        cut_y = max(1, min(h - 1, int(cut_y)))
        head = canvas.crop((0, 0, w, cut_y))

        chunk_name = f"chunk_{chunk_index:04d}.jpg"
        chunk_path = os.path.join(out_dir, chunk_name)
        head.save(chunk_path, "JPEG", quality=int(args.jpeg_quality))

        # Recompute sources y0/y1 for head
        y = 0
        head_sources: List[Dict[str, Any]] = []
        for src, im in zip(cur_sources, cur_images):
            y0 = y
            y1 = y + im.height
            if y0 >= cut_y:
                break
            s2 = dict(src)
            s2["y0"] = y0
            s2["y1"] = min(y1, cut_y)
            head_sources.append(s2)
            y = y1

        chunks.append(
            {
                "chunk_index": chunk_index,
                "chunk_file": chunk_name,
                "chunk_path": os.path.relpath(chunk_path, episode_dir),
                "chunk_w": w,
                "chunk_h": cut_y,
                "sources": head_sources,
                "cut_debug": cut_debug or {},
            }
        )
        chunk_index += 1

        # Tail for overlap
        tail_start = max(0, cut_y - int(keep_overlap))
        tail = canvas.crop((0, tail_start, w, h))

        new_images: List[Image.Image] = []
        new_sources: List[Dict[str, Any]] = []

        y = 0
        for src, im in zip(cur_sources, cur_images):
            y0 = y
            y1 = y + im.height
            y = y1

            inter0 = max(y0, tail_start)
            inter1 = min(y1, h)
            if inter1 <= inter0:
                continue

            crop0 = inter0 - y0
            crop1 = inter1 - y0
            im_part = im.crop((0, crop0, im.width, crop1))

            s2 = dict(src)
            s2["y0"] = None
            s2["y1"] = None
            new_sources.append(s2)
            new_images.append(im_part)

        cur_sources = new_sources
        cur_images = new_images
        cur_h = sum(im.height for im in cur_images)

    max_h = int(args.max_chunk_height)
    hard_cap = max_h + int(args.max_overflow_px)

    for p, im in pages:
        src = {
            "file": os.path.basename(p),
            "path": os.path.abspath(p),
            "orig_w": sizes_by_path[p][0],
            "orig_h": sizes_by_path[p][1],
            "resized_w": im.width,
            "resized_h": im.height,
            "y0": None,
            "y1": None,
        }

        # If we'd exceed max_h, attempt a safe cut using current canvas (BEFORE adding the new page)
        if cur_images and (cur_h + im.height > max_h):
            canvas = render_canvas(cur_images)
            target_y = min(max_h, canvas.size[1] - 1)

            cut_y, dbg = _pick_cut_y_run_based(
                canvas=canvas,
                target_y=target_y,
                base_window=int(args.search_window),
                min_run_px=int(args.min_gutter_run_px),
                max_window=int(args.max_search_window),
            )

            if cut_y is not None:
                if args.debug_cuts:
                    print(f"[cut] chunk={chunk_index:04d} y={cut_y} reason={dbg.get('reason')} win={dbg.get('window_used')}")
                flush_by_cut(cut_y=cut_y, keep_overlap=int(args.overlap_px), cut_debug=dbg)
            else:
                # No safe cut found near target -> allow overflow (do NOT cut yet)
                if cur_h >= hard_cap:
                    # hard cap reached -> forced cut least-damage
                    force_y, fdbg = _az = _force_cut_least_damage(canvas=canvas, target_y=target_y, window=int(args.max_search_window))
                    if args.debug_cuts:
                        print(f"[force] chunk={chunk_index:04d} y={force_y} reason={fdbg.get('reason')}")
                    flush_by_cut(cut_y=force_y, keep_overlap=int(args.overlap_px), cut_debug=fdbg)
                else:
                    if args.debug_cuts:
                        print(f"[defer] no safe cut (cur_h={cur_h}, hard_cap={hard_cap})")

        cur_sources.append(src)
        cur_images.append(im)
        cur_h += im.height

    # Flush remainder as last chunk
    if cur_images:
        canvas = render_canvas(cur_images)
        w, h = canvas.size

        chunk_name = f"chunk_{chunk_index:04d}.jpg"
        chunk_path = os.path.join(out_dir, chunk_name)
        canvas.save(chunk_path, "JPEG", quality=int(args.jpeg_quality))

        y = 0
        for i, im in enumerate(cur_images):
            cur_sources[i]["y0"] = y
            cur_sources[i]["y1"] = y + im.height
            y += im.height

        chunks.append(
            {
                "chunk_index": chunk_index,
                "chunk_file": chunk_name,
                "chunk_path": os.path.relpath(chunk_path, episode_dir),
                "chunk_w": w,
                "chunk_h": h,
                "sources": cur_sources,
                "cut_debug": {"reason": "final_flush"},
            }
        )

    out_manifest = {
        "episode_dir": ".",
        "out_dir": out_dir,
        "glob": args.glob,
        "target_width": target_w,
        "max_chunk_height": max_h,
        "adaptive": {
            "search_window": int(args.search_window),
            "max_search_window": int(args.max_search_window),
            "min_gutter_run_px": int(args.min_gutter_run_px),
            "overlap_px": int(args.overlap_px),
            "max_overflow_px": int(args.max_overflow_px),
        },
        "count_chunks": len(chunks),
        "chunks": chunks,
    }

    manifest_path = os.path.join(episode_dir, "manifest.stitch.json")
    dump_json(manifest_path, out_manifest)
    print(f"[ok] wrote={manifest_path} chunks={len(chunks)} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
