#!/usr/bin/env python3
"""
chunk_stitch_adaptive.py

Like chunk_stitch.py, but:
- chooses cut points near gutters (low-ink / low-edge / high-white rows)
- adds overlap so panels aren't cut across chunks

Output:
  <episode>/stitch_chunks/chunk_0001.jpg ...
  <episode>/manifest.stitch.json

Manifest format is compatible with your current flow:
  chunks[].sources[] contains y0/y1 offsets in chunk space.
"""

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple

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


def _row_scores_rgb(canvas: Image.Image) -> Tuple[List[float], List[float]]:
    """
    Returns:
      white_frac[y] in [0..1] : fraction of near-white pixels on row y
      edge_score[y]          : average abs gradient magnitude on row y
    """
    w, h = canvas.size

    if np is None:
        # Fallback: cheap sampling (slower, but works without numpy)
        px = canvas.load()
        white_frac = [0.0] * h
        edge_score = [0.0] * h

        step = max(1, w // 600)  # sample ~600 columns
        for y in range(h):
            white = 0
            last_luma = None
            grad_sum = 0.0
            cnt = 0
            for x in range(0, w, step):
                r, g, b = px[x, y]
                luma = (0.299 * r + 0.587 * g + 0.114 * b)
                if r >= 245 and g >= 245 and b >= 245:
                    white += 1
                if last_luma is not None:
                    grad_sum += abs(luma - last_luma)
                last_luma = luma
                cnt += 1
            white_frac[y] = white / float(max(1, cnt))
            edge_score[y] = grad_sum / float(max(1, cnt))
        return white_frac, edge_score

    # Numpy path (fast)
    arr = np.asarray(canvas).astype(np.float32)  # (h,w,3)
    # near-white mask
    white = (arr[:, :, 0] >= 245) & (arr[:, :, 1] >= 245) & (arr[:, :, 2] >= 245)
    white_frac = white.mean(axis=1)  # (h,)

    # luma for edges
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    # horizontal gradient magnitude (cheap edge proxy)
    grad = np.abs(luma[:, 1:] - luma[:, :-1])
    edge_score = grad.mean(axis=1)  # (h,)

    return white_frac.tolist(), edge_score.tolist()


def _pick_cut_y(
    canvas: Image.Image,
    target_y: int,
    search_window: int,
    white_min: float,
    edge_max: float,
) -> int:
    """
    Pick a cut line near target_y.
    Prefer rows that are:
      - high white_frac
      - low edge_score
    """
    w, h = canvas.size
    target_y = max(0, min(h - 1, target_y))

    white_frac, edge_score = _row_scores_rgb(canvas)

    y0 = max(0, target_y - search_window)
    y1 = min(h - 1, target_y + search_window)

    best_y = target_y
    best_val = -1e18

    # Normalize edge to roughly comparable scale
    # Lower is better.
    for y in range(y0, y1 + 1):
        wf = white_frac[y]
        es = edge_score[y]

        # Hard filters help avoid cutting through art
        if wf < white_min:
            continue
        if es > edge_max:
            continue

        # Score: prefer very white + very low edges + close to target
        dist = abs(y - target_y)
        val = (wf * 3.0) - (es * 0.02) - (dist / float(search_window + 1)) * 0.5
        if val > best_val:
            best_val = val
            best_y = y

    # If nothing passed filters, fall back to best "most white, least edge" without filters
    if best_val < -1e17:
        for y in range(y0, y1 + 1):
            wf = white_frac[y]
            es = edge_score[y]
            dist = abs(y - target_y)
            val = (wf * 2.0) - (es * 0.01) - (dist / float(search_window + 1)) * 0.3
            if val > best_val:
                best_val = val
                best_y = y

    return int(best_y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--out-dir", default="stitch_chunks")
    ap.add_argument("--max-chunk-height", type=int, default=16000)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--target-width", type=int, default=0, help="0=auto (use max width across inputs)")

    # Adaptive params
    ap.add_argument("--search-window", type=int, default=500, help="Search +/- this many px for a good gutter cut")
    ap.add_argument("--overlap-px", type=int, default=600, help="Overlap between consecutive chunks")
    ap.add_argument("--white-min", type=float, default=0.80, help="Min near-white fraction for a row to be considered gutter")
    ap.add_argument("--edge-max", type=float, default=22.0, help="Max edge score for a row to be considered gutter")

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

    # Build a rolling canvas from pages (append until we need to cut)
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

    def flush_by_cut(cut_y: int, keep_overlap: int):
        """
        Flush chunk [0:cut_y], keep tail [cut_y-keep_overlap : end] as next chunk start.
        """
        nonlocal chunk_index, cur_sources, cur_images, cur_h, chunks

        if not cur_images:
            return

        canvas = render_canvas(cur_images)
        w, h = canvas.size

        cut_y = max(1, min(h - 1, cut_y))
        head = canvas.crop((0, 0, w, cut_y))

        # Save head as a chunk
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
                "chunk_path": chunk_path,
                "chunk_w": w,
                "chunk_h": cut_y,
                "sources": head_sources,
            }
        )
        chunk_index += 1

        # Tail for overlap
        tail_start = max(0, cut_y - int(keep_overlap))
        tail = canvas.crop((0, tail_start, w, h))

        # Rebuild cur_images/cur_sources from tail by cropping the last pages regionally
        # We do this by walking original stacked images and taking the intersecting slice.
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
            # These y0/y1 will be re-assigned when next chunk is saved
            s2["y0"] = None
            s2["y1"] = None
            new_sources.append(s2)
            new_images.append(im_part)

        cur_sources = new_sources
        cur_images = new_images
        cur_h = sum(im.height for im in cur_images)

    # Assemble + cut
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

        # If adding this page exceeds max height, cut adaptively
        if cur_h + im.height > int(args.max_chunk_height) and cur_images:
            # Render current canvas to decide cut location
            canvas = render_canvas(cur_images)
            target_y = int(args.max_chunk_height)

            # target_y can be > current height; clamp
            target_y = min(target_y, canvas.size[1] - 1)

            cut_y = _pick_cut_y(
                canvas=canvas,
                target_y=target_y,
                search_window=int(args.search_window),
                white_min=float(args.white_min),
                edge_max=float(args.edge_max),
            )
            flush_by_cut(cut_y=cut_y, keep_overlap=int(args.overlap_px))

        cur_sources.append(src)
        cur_images.append(im)
        cur_h += im.height

    # Flush remainder as last chunk
    if cur_images:
        canvas = render_canvas(cur_images)
        w, h = canvas.size

        # Save
        chunk_name = f"chunk_{chunk_index:04d}.jpg"
        chunk_path = os.path.join(out_dir, chunk_name)
        canvas.save(chunk_path, "JPEG", quality=int(args.jpeg_quality))

        # Assign y0/y1
        y = 0
        for i, im in enumerate(cur_images):
            cur_sources[i]["y0"] = y
            cur_sources[i]["y1"] = y + im.height
            y += im.height

        chunks.append(
            {
                "chunk_index": chunk_index,
                "chunk_file": chunk_name,
                "chunk_path": chunk_path,
                "chunk_w": w,
                "chunk_h": h,
                "sources": cur_sources,
            }
        )

    out_manifest = {
        "episode_dir": episode_dir,
        "out_dir": out_dir,
        "glob": args.glob,
        "target_width": target_w,
        "max_chunk_height": int(args.max_chunk_height),
        "adaptive": {
            "search_window": int(args.search_window),
            "overlap_px": int(args.overlap_px),
            "white_min": float(args.white_min),
            "edge_max": float(args.edge_max),
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
