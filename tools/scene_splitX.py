#!/usr/bin/env python3
"""
Scene Splitter with Deduplication - Fixed Version
Removes duplicate panels while preserving original detection logic
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import hashlib

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class SplitConfig:
    white_mean: int = 245
    black_mean: int = 10
    std_max: int = 8
    split_run_px: int = 80
    trim_run_px: int = 200
    collapse_run_px: int = 350
    keep_gap_px: int = 40
    min_scene_h: int = 420
    max_scene_h: int = 2400
    sample_cols: int = 96


class DeduplicationTracker:
    def __init__(self, similarity_threshold: float = 0.95):
        self.seen_hashes: Set[str] = set()
        self.similarity_threshold = similarity_threshold
        
    def compute_hash(self, image: Image.Image) -> str:
        small = image.resize((32, 32), Image.Resampling.LANCZOS)
        gray = small.convert('L')
        img_bytes = np.array(gray).tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    
    def is_duplicate(self, image: Image.Image) -> bool:
        img_hash = self.compute_hash(image)
        if img_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(img_hash)
        return False


def natural_key(path: str):
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    return (int(nums[0]) if nums else 0, base)


def load_rgb(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im


def to_gray_np(im: Image.Image, sample_cols: int) -> np.ndarray:
    arr = np.asarray(im, dtype=np.uint8)
    gray = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    h, w = gray.shape
    if sample_cols >= w:
        return gray
    idx = np.linspace(0, w - 1, sample_cols).astype(np.int32)
    return gray[:, idx]


def find_blank_runs(gray_sample: np.ndarray, cfg: SplitConfig) -> List[Tuple[int, int, str]]:
    row_mean = gray_sample.mean(axis=1)
    row_std = gray_sample.std(axis=1)

    is_white = (row_mean >= cfg.white_mean) & (row_std <= cfg.std_max)
    is_black = (row_mean <= cfg.black_mean) & (row_std <= cfg.std_max)
    is_blank = is_white | is_black

    runs = []
    h = gray_sample.shape[0]
    y = 0
    while y < h:
        if not is_blank[y]:
            y += 1
            continue
        y0 = y
        kind = "white" if is_white[y] else "black"
        while y < h and is_blank[y]:
            y += 1
        y1 = y
        wcount = int(is_white[y0:y1].sum())
        bcount = int(is_black[y0:y1].sum())
        kind = "white" if wcount >= bcount else "black"
        runs.append((y0, y1, kind))
    return runs


def crop_y(im: Image.Image, y0: int, y1: int) -> Image.Image:
    y0 = max(0, y0)
    y1 = min(im.height, y1)
    if y1 <= y0:
        return im.crop((0, 0, im.width, 1))
    return im.crop((0, y0, im.width, y1))


def trim_top_bottom_blanks(im: Image.Image, cfg: SplitConfig) -> Image.Image:
    gray = to_gray_np(im, cfg.sample_cols)
    runs = find_blank_runs(gray, cfg)
    top_cut = 0
    bot_cut = im.height

    for (a, b, _) in runs:
        if a == 0 and (b - a) >= cfg.trim_run_px:
            top_cut = b
        break

    for (a, b, _) in reversed(runs):
        if b == im.height and (b - a) >= cfg.trim_run_px:
            bot_cut = a
        break

    if top_cut == 0 and bot_cut == im.height:
        return im
    return crop_y(im, top_cut, bot_cut)


def split_by_big_blank_runs(im: Image.Image, cfg: SplitConfig) -> List[Image.Image]:
    gray = to_gray_np(im, cfg.sample_cols)
    runs = find_blank_runs(gray, cfg)

    cut_positions = []
    for (a, b, _) in runs:
        run_len = b - a
        if run_len >= cfg.split_run_px:
            cut_positions.append((a, b, run_len))

    if not cut_positions:
        return [im]

    chunks = []
    cur_y = 0
    for (a, b, run_len) in cut_positions:
        if a > cur_y:
            chunks.append(crop_y(im, cur_y, a))
        if run_len >= cfg.collapse_run_px:
            gap_h = cfg.keep_gap_px
            gap = Image.new("RGB", (im.width, gap_h), (0, 0, 0))
            chunks.append(gap)
        cur_y = b
    
    if cur_y < im.height:
        chunks.append(crop_y(im, cur_y, im.height))

    out = []
    for c in chunks:
        if c.height <= 2:
            continue
        out.append(trim_top_bottom_blanks(c, cfg))
    out = [c for c in out if c.height > 10]
    return out


def stack_vertical(images: List[Image.Image]) -> Image.Image:
    w = max(im.width for im in images)
    norm = []
    for im in images:
        if im.width != w:
            nh = int(im.height * (w / im.width))
            norm.append(im.resize((w, nh), Image.Resampling.LANCZOS))
        else:
            norm.append(im)
    total_h = sum(im.height for im in norm)
    canvas = Image.new("RGB", (w, total_h), (255, 255, 255))
    y = 0
    for im in norm:
        canvas.paste(im, (0, y))
        y += im.height
    return canvas


def split_tall_scene(scene: Image.Image, cfg: SplitConfig) -> List[Image.Image]:
    if cfg.max_scene_h <= 0 or scene.height <= cfg.max_scene_h:
        return [scene]
    parts = []
    y = 0
    while y < scene.height:
        y2 = min(scene.height, y + cfg.max_scene_h)
        parts.append(crop_y(scene, y, y2))
        y = y2
    return parts


def build_scenes_from_slices(paths: List[str], cfg: SplitConfig, 
                            dedup_tracker: DeduplicationTracker) -> List[Image.Image]:
    scenes: List[Image.Image] = []
    current_parts: List[Image.Image] = []

    def flush_current():
        nonlocal current_parts, scenes
        if not current_parts:
            return
        merged = stack_vertical(current_parts)
        merged = trim_top_bottom_blanks(merged, cfg)
        
        if not dedup_tracker.is_duplicate(merged):
            for part in split_tall_scene(merged, cfg):
                if not dedup_tracker.is_duplicate(part):
                    scenes.append(part)
        
        current_parts = []

    for p in paths:
        im = load_rgb(p)
        chunks = split_by_big_blank_runs(im, cfg)

        for ch in chunks:
            is_gap = (ch.height <= cfg.keep_gap_px + 2) and (np.asarray(ch).mean() < 5)
            if is_gap:
                if current_parts:
                    flush_current()
                continue

            ch = trim_top_bottom_blanks(ch, cfg)
            if ch.height <= 10:
                continue

            current_parts.append(ch)

    flush_current()

    merged_scenes: List[Image.Image] = []
    buf: List[Image.Image] = []
    buf_h = 0

    def flush_buf():
        nonlocal buf, buf_h, merged_scenes
        if not buf:
            return
        merged = stack_vertical(buf)
        merged = trim_top_bottom_blanks(merged, cfg)
        
        if not dedup_tracker.is_duplicate(merged):
            merged_scenes.append(merged)
        
        buf = []
        buf_h = 0

    for sc in scenes:
        if not buf:
            buf = [sc]
            buf_h = sc.height
            continue
        if buf_h < cfg.min_scene_h or sc.height < int(cfg.min_scene_h * 0.6):
            buf.append(sc)
            buf_h += sc.height
        else:
            flush_buf()
            buf = [sc]
            buf_h = sc.height
    flush_buf()

    return merged_scenes


def main():
    ap = argparse.ArgumentParser(
        description="Scene Splitter with Deduplication"
    )
    ap.add_argument("input_dir", help="Input directory")
    ap.add_argument("output_dir", help="Output directory")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--split-run", type=int, default=80)
    ap.add_argument("--trim-run", type=int, default=200)
    ap.add_argument("--collapse-run", type=int, default=350)
    ap.add_argument("--keep-gap", type=int, default=40)
    ap.add_argument("--min-scene-h", type=int, default=420)
    ap.add_argument("--max-scene-h", type=int, default=2400)

    args = ap.parse_args()

    cfg = SplitConfig(
        split_run_px=args.split_run,
        trim_run_px=args.trim_run,
        collapse_run_px=args.collapse_run,
        keep_gap_px=args.keep_gap,
        min_scene_h=args.min_scene_h,
        max_scene_h=args.max_scene_h,
    )

    dedup_tracker = DeduplicationTracker()

    paths = sorted(
        glob.glob(os.path.join(args.input_dir, args.glob)), 
        key=natural_key
    )
    
    if not paths:
        raise SystemExit(f"No input images found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nProcessing {len(paths)} chunks with deduplication...\n")
    
    scenes = build_scenes_from_slices(paths, cfg, dedup_tracker)
    
    print(f"Results:")
    print(f"  Scenes created: {len(scenes)}")
    print(f"  Duplicates removed: {len(dedup_tracker.seen_hashes) - len(scenes)}")
    print(f"\nSaving scenes...\n")

    for i, sc in enumerate(scenes, 1):
        out = os.path.join(args.output_dir, f"scene_{i:04d}.jpg")
        sc.save(out, "JPEG", quality=args.jpeg_quality)
        print(f"  scene_{i:04d}.jpg ({sc.width}x{sc.height}px)")
    
    print(f"\nComplete! Wrote {len(scenes)} unique scenes to: {args.output_dir}/")


if __name__ == "__main__":
    main()

