#!/usr/bin/env python3
import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # helps with some web downloads


@dataclass
class SplitConfig:
    # "blank row" detection
    white_mean: int = 245
    black_mean: int = 10
    std_max: int = 8

    # run thresholds (pixel heights)
    split_run_px: int = 80       # run length that can separate scenes
    trim_run_px: int = 200       # run length trimmed at top/bottom of each chunk
    collapse_run_px: int = 350   # run length inside content gets collapsed
    keep_gap_px: int = 40        # if collapsing, keep this much gap

    # post-merge constraints
    min_scene_h: int = 420       # merge fragments until scene >= this height
    max_scene_h: int = 2400      # optional: split very tall scenes further (0 disables)

    # speed / robustness
    sample_cols: int = 96        # columns sampled to compute row stats


def natural_key(path: str):
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    return (int(nums[0]) if nums else 0, base)


def load_rgb(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im


def to_gray_np(im: Image.Image, sample_cols: int) -> np.ndarray:
    """Return grayscale as float32 [H,Wsample] by sampling columns to speed up."""
    arr = np.asarray(im, dtype=np.uint8)
    # luma
    gray = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    h, w = gray.shape
    if sample_cols >= w:
        return gray
    idx = np.linspace(0, w - 1, sample_cols).astype(np.int32)
    return gray[:, idx]


def find_blank_runs(gray_sample: np.ndarray, cfg: SplitConfig) -> List[Tuple[int, int, str]]:
    """
    Return list of (start_y, end_y_exclusive, kind) for consecutive blank-ish row runs.
    kind in {"white","black"} based on mean.
    """
    row_mean = gray_sample.mean(axis=1)
    # use std for "uniform"
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
        # classify run by majority in the run
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

    # top
    for (a, b, _) in runs:
        if a == 0 and (b - a) >= cfg.trim_run_px:
            top_cut = b
        break

    # bottom
    for (a, b, _) in reversed(runs):
        if b == im.height and (b - a) >= cfg.trim_run_px:
            bot_cut = a
        break

    if top_cut == 0 and bot_cut == im.height:
        return im
    return crop_y(im, top_cut, bot_cut)


def split_by_big_blank_runs(im: Image.Image, cfg: SplitConfig) -> List[Image.Image]:
    """
    Split an image into chunks using blank runs >= split_run_px.
    Also collapses internal huge blank runs >= collapse_run_px by removing most of them.
    """
    gray = to_gray_np(im, cfg.sample_cols)
    runs = find_blank_runs(gray, cfg)

    # Determine cut positions (use middle of big run)
    cut_positions = []
    for (a, b, _) in runs:
        run_len = b - a
        if run_len >= cfg.split_run_px:
            cut_positions.append((a, b, run_len))

    if not cut_positions:
        return [im]

    # Build chunks between cuts
    chunks = []
    cur_y = 0
    for (a, b, run_len) in cut_positions:
        # chunk before run
        if a > cur_y:
            chunks.append(crop_y(im, cur_y, a))
        # for the run itself: if it's a huge void, keep only a small gap
        if run_len >= cfg.collapse_run_px:
            # represent the gap as a small blank image
            gap_h = cfg.keep_gap_px
            gap = Image.new("RGB", (im.width, gap_h), (0, 0, 0))  # black separator
            chunks.append(gap)
        # move cursor after run
        cur_y = b
    # tail
    if cur_y < im.height:
        chunks.append(crop_y(im, cur_y, im.height))

    # Trim large blanks from each real chunk
    out = []
    for c in chunks:
        if c.height <= 2:
            continue
        out.append(trim_top_bottom_blanks(c, cfg))
    # Drop tiny empty
    out = [c for c in out if c.height > 10]
    return out


def stack_vertical(images: List[Image.Image]) -> Image.Image:
    w = max(im.width for im in images)
    # resize small mismatches to w
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
    # split by height windows (simple)
    parts = []
    y = 0
    while y < scene.height:
        y2 = min(scene.height, y + cfg.max_scene_h)
        parts.append(crop_y(scene, y, y2))
        y = y2
    return parts


def build_scenes_from_slices(paths: List[str], cfg: SplitConfig) -> List[Image.Image]:
    scenes: List[Image.Image] = []
    current_parts: List[Image.Image] = []

    def flush_current():
        nonlocal current_parts, scenes
        if not current_parts:
            return
        merged = stack_vertical(current_parts)
        merged = trim_top_bottom_blanks(merged, cfg)
        for part in split_tall_scene(merged, cfg):
            scenes.append(part)
        current_parts = []

    # We treat “big blank run” as separators BUT we also merge small fragments later.
    for p in paths:
        im = load_rgb(p)

        # Split within the slice by big blank runs
        chunks = split_by_big_blank_runs(im, cfg)

        for ch in chunks:
            # If the chunk is basically a separator gap, keep it *inside* a scene only if we already started
            # (we represent separators as black gaps in split_by_big_blank_runs)
            is_gap = (ch.height <= cfg.keep_gap_px + 2) and (np.asarray(ch).mean() < 5)
            if is_gap:
                # end scene if we have decent content already
                if current_parts:
                    flush_current()
                continue

            ch = trim_top_bottom_blanks(ch, cfg)
            if ch.height <= 10:
                continue

            # append
            current_parts.append(ch)

            # If we've reached a "reasonable scene size", you can flush later,
            # but keep it simple: we will merge fragments then flush on separators / end.
            # (This avoids splitting text/title into 3 pieces.)
            # no-op

    flush_current()

    # Merge rule: merge consecutive scenes that are too small
    merged_scenes: List[Image.Image] = []
    buf: List[Image.Image] = []
    buf_h = 0

    def flush_buf():
        nonlocal buf, buf_h, merged_scenes
        if not buf:
            return
        merged = stack_vertical(buf)
        merged = trim_top_bottom_blanks(merged, cfg)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--out-dir", default="scenes")
    ap.add_argument("--jpeg-quality", type=int, default=95)

    # Tunables (defaults chosen for your screenshots: big white/black padding + small title fragments)
    ap.add_argument("--split-run", type=int, default=80)
    ap.add_argument("--trim-run", type=int, default=200)
    ap.add_argument("--collapse-run", type=int, default=350)
    ap.add_argument("--keep-gap", type=int, default=40)
    ap.add_argument("--min-scene-h", type=int, default=420)
    ap.add_argument("--max-scene-h", type=int, default=0)

    args = ap.parse_args()

    cfg = SplitConfig(
        split_run_px=args.split_run,
        trim_run_px=args.trim_run,
        collapse_run_px=args.collapse_run,
        keep_gap_px=args.keep_gap,
        min_scene_h=args.min_scene_h,
        max_scene_h=args.max_scene_h,
    )

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.glob)), key=natural_key)
    if not paths:
        raise SystemExit("No input images found")

    os.makedirs(args.out_dir, exist_ok=True)

    scenes = build_scenes_from_slices(paths, cfg)

    for i, sc in enumerate(scenes, 1):
        out = os.path.join(args.out_dir, f"scene_{i:04d}.jpg")
        sc.save(out, "JPEG", quality=args.jpeg_quality)  # no optimize/progressive (avoids some Pillow issues)
    print(f"Wrote {len(scenes)} scenes to: {args.out_dir}/")


if __name__ == "__main__":
    main()
