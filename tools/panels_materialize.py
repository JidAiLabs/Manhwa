#!/usr/bin/env python3
"""
panels_materialize.py

Materialize Gemini panel boxes (manifest.panels.json) into actual cropped images,
using stitch mapping (manifest.stitch.json) for provenance.

Inputs:
  - manifest.stitch.json (from chunk_stitch[_adaptive].py)
  - manifest.panels.json (from gemini_panel_boxes.py)

Outputs:
  - <episode>/panels/ panel_000001.jpg ...
  - <episode>/manifest.panels_flat.json

Key behavior:
  - Each chunk has panels_norm = [[x0,y0,x1,y1], ...] in normalized coords.
  - Crops are taken from the chunk image.
  - If a crop is too tall, the script can split it using original page boundaries
    recorded in stitch manifest sources (recommended default).
  - If still too tall, it window-splits by max height.
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def rect_norm_to_px(b: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    x0 = clamp01(x0); y0 = clamp01(y0); x1 = clamp01(x1); y1 = clamp01(y1)
    # ensure ordering
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    px0 = int(round(x0 * w))
    py0 = int(round(y0 * h))
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    # clamp to image
    px0 = clamp(px0, 0, w)
    px1 = clamp(px1, 0, w)
    py0 = clamp(py0, 0, h)
    py1 = clamp(py1, 0, h)
    return (px0, py0, px1, py1)


def area_px(r: Tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = r
    return max(0, x1 - x0) * max(0, y1 - y0)


def intersect_y(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def crop_and_save(im: Image.Image, r: Tuple[int, int, int, int], out_path: str, quality: int) -> None:
    x0, y0, x1, y1 = r
    if x1 <= x0 or y1 <= y0:
        return
    c = im.crop((x0, y0, x1, y1))
    c.save(out_path, "JPEG", quality=int(quality))


def build_chunk_map(stitch: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for ch in (stitch.get("chunks") or []):
        cf = ch.get("chunk_file")
        if cf:
            m[cf] = ch
    return m


def get_sources_for_crop(chunk_sources: List[Dict[str, Any]], crop_y0: int, crop_y1: int) -> List[Dict[str, Any]]:
    """
    Return list of source page segments that overlap this crop y-range, with crop-local y mapping.
    """
    out: List[Dict[str, Any]] = []
    for s in (chunk_sources or []):
        sy0 = int(s.get("y0") or 0)
        sy1 = int(s.get("y1") or 0)
        if sy1 <= sy0:
            continue
        ov = intersect_y(crop_y0, crop_y1, sy0, sy1)
        if ov <= 0:
            continue
        # crop-local mapping
        local_y0 = max(0, sy0 - crop_y0)
        local_y1 = min(crop_y1 - crop_y0, sy1 - crop_y0)
        out.append(
            {
                "file": s.get("file"),
                "path": s.get("path"),
                "chunk_y0": sy0,
                "chunk_y1": sy1,
                "crop_local_y0": local_y0,
                "crop_local_y1": local_y1,
                "resized_w": s.get("resized_w"),
                "resized_h": s.get("resized_h"),
                "orig_w": s.get("orig_w"),
                "orig_h": s.get("orig_h"),
            }
        )
    return out


def split_by_sources(crop_r: Tuple[int, int, int, int], sources: List[Dict[str, Any]], min_piece_h: int) -> List[Tuple[int, int, int, int]]:
    """
    Split a crop rectangle along source boundaries (y0/y1 of each source).
    Returns list of rects with same x-range but y-range clipped to those boundaries.
    """
    x0, y0, x1, y1 = crop_r
    if not sources or y1 - y0 <= 0:
        return [crop_r]

    # collect boundary y positions within crop
    bounds = {y0, y1}
    for s in sources:
        sy0 = int(s.get("y0") or 0)
        sy1 = int(s.get("y1") or 0)
        # clip to crop
        by0 = clamp(sy0, y0, y1)
        by1 = clamp(sy1, y0, y1)
        bounds.add(by0)
        bounds.add(by1)

    ys = sorted(bounds)
    rects: List[Tuple[int, int, int, int]] = []
    for a, b in zip(ys, ys[1:]):
        if b - a < int(min_piece_h):
            continue
        rects.append((x0, a, x1, b))

    # fallback if boundaries produced nothing
    return rects if rects else [crop_r]


def split_by_windows(crop_r: Tuple[int, int, int, int], max_h: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = crop_r
    H = y1 - y0
    if max_h <= 0 or H <= max_h:
        return [crop_r]
    rects: List[Tuple[int, int, int, int]] = []
    step = max(1, max_h - max(0, overlap))
    cur = y0
    while cur < y1:
        ny1 = min(y1, cur + max_h)
        rects.append((x0, cur, x1, ny1))
        if ny1 >= y1:
            break
        cur = cur + step
    return rects


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-manifest", required=True)
    ap.add_argument("--panels-manifest", required=True)
    ap.add_argument("--out-dir", default="", help="Default: <episode>/panels")
    ap.add_argument("--out-manifest", default="", help="Default: <episode>/manifest.panels_flat.json")

    ap.add_argument("--jpeg-quality", type=int, default=92)

    # geometry tweaks
    ap.add_argument("--pad-x", type=int, default=4, help="Pad crop left/right in pixels (chunk space).")
    ap.add_argument("--pad-y", type=int, default=4, help="Pad crop top/bottom in pixels (chunk space).")

    ap.add_argument("--min-w", type=int, default=80, help="Skip crops thinner than this.")
    ap.add_argument("--min-h", type=int, default=120, help="Skip crops shorter than this.")

    # splitting very tall crops
    ap.add_argument("--split-by-pages", action="store_true", help="Split using stitch source boundaries (recommended).")
    ap.add_argument("--max-out-height", type=int, default=1800, help="If crop piece still taller than this, window-split (0 disables).")
    ap.add_argument("--window-overlap", type=int, default=120, help="Overlap when window-splitting.")
    ap.add_argument("--min-piece-h", type=int, default=220, help="Minimum height of page-split pieces.")
    args = ap.parse_args()

    stitch = load_json(args.stitch_manifest)
    panels = load_json(args.panels_manifest)

    episode_dir = os.path.abspath(stitch.get("episode_dir") or os.path.dirname(os.path.abspath(args.stitch_manifest)))
    chunk_map = build_chunk_map(stitch)

    out_dir = args.out_dir.strip() or os.path.join(episode_dir, "panels")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_manifest = args.out_manifest.strip() or os.path.join(episode_dir, "manifest.panels_flat.json")
    out_manifest = os.path.abspath(out_manifest)

    items: List[Dict[str, Any]] = []
    panel_id = 1

    chunks_in = panels.get("chunks") or []
    for ch in chunks_in:
        chunk_file = ch.get("chunk_file")
        if not chunk_file:
            continue
        chunk_info = chunk_map.get(chunk_file)
        if not chunk_info:
            # still allow if chunk_path exists in panels manifest
            chunk_info = {"chunk_file": chunk_file, "chunk_path": ch.get("chunk_path"), "sources": []}

        chunk_path = ch.get("chunk_path") or chunk_info.get("chunk_path")
        if not chunk_path or not os.path.exists(chunk_path):
            continue

        with Image.open(chunk_path) as im:
            im = im.convert("RGB")
            W, H = im.size

            # prefer panels chunk_w/h if present
            # but trust actual image size for cropping
            sources = chunk_info.get("sources") or []

            panels_norm = ch.get("panels_norm") or []
            for idx, b in enumerate(panels_norm, 1):
                if not (isinstance(b, list) and len(b) == 4):
                    continue

                r = rect_norm_to_px(b, W, H)
                x0, y0, x1, y1 = r

                # padding
                x0 = clamp(x0 - int(args.pad_x), 0, W)
                x1 = clamp(x1 + int(args.pad_x), 0, W)
                y0 = clamp(y0 - int(args.pad_y), 0, H)
                y1 = clamp(y1 + int(args.pad_y), 0, H)
                r = (x0, y0, x1, y1)

                if (x1 - x0) < int(args.min_w) or (y1 - y0) < int(args.min_h):
                    continue

                # split strategy
                rects: List[Tuple[int, int, int, int]] = [r]

                if args.split_by_pages and sources and (y1 - y0) > int(args.max_out_height):
                    rects = split_by_sources(r, sources, min_piece_h=int(args.min_piece_h))

                # window split if still too tall
                final_rects: List[Tuple[int, int, int, int]] = []
                for rr in rects:
                    if int(args.max_out_height) > 0 and (rr[3] - rr[1]) > int(args.max_out_height):
                        final_rects.extend(split_by_windows(rr, int(args.max_out_height), int(args.window_overlap)))
                    else:
                        final_rects.append(rr)

                # write each piece
                piece_idx = 1
                for rr in final_rects:
                    if (rr[2] - rr[0]) < int(args.min_w) or (rr[3] - rr[1]) < int(args.min_h):
                        continue

                    out_name = f"panel_{panel_id:06d}.jpg"
                    out_path = os.path.join(out_dir, out_name)
                    crop_and_save(im, rr, out_path, quality=int(args.jpeg_quality))

                    ov_sources = get_sources_for_crop(sources, rr[1], rr[3])

                    items.append(
                        {
                            "panel_id": panel_id,
                            "panel_file": out_name,
                            "panel_path": out_path,
                            "chunk_file": chunk_file,
                            "chunk_path": chunk_path,
                            "chunk_w": W,
                            "chunk_h": H,
                            "panel_index_in_chunk": idx,
                            "panel_piece_index": piece_idx,
                            "bbox_norm": [round(float(x), 6) for x in b],
                            "bbox_px": [int(rr[0]), int(rr[1]), int(rr[2]), int(rr[3])],
                            "width": int(rr[2] - rr[0]),
                            "height": int(rr[3] - rr[1]),
                            "sources_overlap": ov_sources,
                        }
                    )

                    panel_id += 1
                    piece_idx += 1

    out_obj = {
        "episode_dir": episode_dir,
        "source_stitch_manifest": os.path.abspath(args.stitch_manifest),
        "source_panels_manifest": os.path.abspath(args.panels_manifest),
        "out_dir": out_dir,
        "count_panels": len(items),
        "items": items,
        "params": {
            "pad_x": int(args.pad_x),
            "pad_y": int(args.pad_y),
            "min_w": int(args.min_w),
            "min_h": int(args.min_h),
            "split_by_pages": bool(args.split_by_pages),
            "max_out_height": int(args.max_out_height),
            "window_overlap": int(args.window_overlap),
            "min_piece_h": int(args.min_piece_h),
        },
    }

    dump_json(out_manifest, out_obj)
    print(f"[ok] wrote_panels={out_dir} count={len(items)}")
    print(f"[ok] wrote_manifest={out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
