#!/usr/bin/env python3
import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class PostCfg:
    # When to consider a scene "too long" and eligible for extra splitting
    long_scene_h: int = 2600
    long_scene_ratio: float = 4.0  # h/w

    # Separator (gutter / caption bar / fade) detection
    sample_cols: int = 128
    sep_std_max: float = 12.0
    sep_black_mean: float = 25.0
    sep_white_mean: float = 230.0
    sep_gray_std_max: float = 6.0          # for gray fades
    sep_run_px: int = 40                   # minimum consecutive rows to treat as separator band
    sep_pad_px: int = 6                    # keep a little padding around cut (trim later anyway)

    # Drop tiny / blank-ish outputs
    min_part_h: int = 280
    min_scene_hard: int = 120              # anything below this after trim is dropped
    flat_std_drop: float = 2.0             # extremely flat overall -> likely blank

    # Simple dedupe of adjacent parts
    dedupe: bool = True
    dhash_size: int = 8
    dhash_hamming_max: int = 6


def natural_key(path: str):
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    return (int(nums[0]) if nums else 0, base)


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_y(im: Image.Image, y0: int, y1: int) -> Image.Image:
    y0 = max(0, y0)
    y1 = min(im.height, y1)
    if y1 <= y0:
        return im.crop((0, 0, im.width, 1))
    return im.crop((0, y0, im.width, y1))


def to_gray_sample(im: Image.Image, sample_cols: int) -> np.ndarray:
    arr = np.asarray(im, dtype=np.uint8)
    gray = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    h, w = gray.shape
    if sample_cols >= w:
        return gray
    idx = np.linspace(0, w - 1, sample_cols).astype(np.int32)
    return gray[:, idx]


def trim_top_bottom_by_uniform_rows(im: Image.Image, cfg: PostCfg) -> Image.Image:
    """
    Trims top/bottom where rows are very uniform (std small) AND very dark/bright/flat gray.
    This is a safer trim for “fade/gradient gutters” than pure white/black only.
    """
    g = to_gray_sample(im, cfg.sample_cols)
    row_mean = g.mean(axis=1)
    row_std = g.std(axis=1)

    # uniform-ish rows
    uniform = row_std <= cfg.sep_std_max
    dark = row_mean <= cfg.sep_black_mean
    bright = row_mean >= cfg.sep_white_mean
    grayfade = (row_std <= cfg.sep_gray_std_max)  # catches smooth gradients too

    blankish = uniform & (dark | bright | grayfade)

    top = 0
    while top < im.height and blankish[top]:
        top += 1

    bot = im.height - 1
    while bot >= 0 and blankish[bot]:
        bot -= 1
    bot += 1  # exclusive

    # If we trimmed almost everything, return original (avoid nuking real content)
    if bot - top < cfg.min_scene_hard:
        return im
    if top == 0 and bot == im.height:
        return im
    return crop_y(im, top, bot)


def find_separator_runs(im: Image.Image, cfg: PostCfg) -> List[Tuple[int, int]]:
    """
    Find y-runs that look like separators:
    - low row_std AND (very dark OR very bright OR very low-std gray fade)
    """
    g = to_gray_sample(im, cfg.sample_cols)
    row_mean = g.mean(axis=1)
    row_std = g.std(axis=1)

    uniform = row_std <= cfg.sep_std_max
    dark = row_mean <= cfg.sep_black_mean
    bright = row_mean >= cfg.sep_white_mean
    grayfade = row_std <= cfg.sep_gray_std_max

    sep = uniform & (dark | bright | grayfade)

    runs: List[Tuple[int, int]] = []
    y = 0
    h = im.height
    while y < h:
        if not sep[y]:
            y += 1
            continue
        y0 = y
        while y < h and sep[y]:
            y += 1
        y1 = y
        if (y1 - y0) >= cfg.sep_run_px:
            runs.append((y0, y1))
    return runs


def split_on_separators(im: Image.Image, cfg: PostCfg) -> List[Image.Image]:
    runs = find_separator_runs(im, cfg)
    if not runs:
        return [im]

    cuts = []
    for (a, b) in runs:
        mid = (a + b) // 2
        cuts.append(mid)

    parts: List[Image.Image] = []
    cur = 0
    for c in cuts:
        y1 = max(cur, c - cfg.sep_pad_px)
        if y1 - cur >= cfg.min_part_h:
            parts.append(crop_y(im, cur, y1))
        cur = min(im.height, c + cfg.sep_pad_px)

    if im.height - cur >= cfg.min_part_h:
        parts.append(crop_y(im, cur, im.height))

    # Final trim each part
    cleaned = []
    for p in parts:
        p2 = trim_top_bottom_by_uniform_rows(p, cfg)
        if p2.height >= cfg.min_scene_hard:
            cleaned.append(p2)
    return cleaned if cleaned else [im]


def overall_flat_std(im: Image.Image) -> float:
    arr = np.asarray(im, dtype=np.uint8)
    gray = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    return float(gray.std())


def dhash(im: Image.Image, size: int = 8) -> int:
    # difference hash: resize to (size+1, size), compare adjacent pixels
    g = im.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.uint8)
    diff = a[:, 1:] > a[:, :-1]
    # pack bits into int
    bits = diff.flatten().astype(np.uint8)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedupe_adjacent(images: List[Image.Image], cfg: PostCfg) -> List[Image.Image]:
    if not cfg.dedupe or len(images) <= 1:
        return images
    out = [images[0]]
    prev_h = dhash(images[0], cfg.dhash_size)
    for im in images[1:]:
        h2 = dhash(im, cfg.dhash_size)
        if hamming(prev_h, h2) <= cfg.dhash_hamming_max:
            # drop near-duplicate
            continue
        out.append(im)
        prev_h = h2
    return out


def should_split_long(im: Image.Image, cfg: PostCfg) -> bool:
    if im.height >= cfg.long_scene_h:
        return True
    if im.width > 0 and (im.height / im.width) >= cfg.long_scene_ratio:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--glob", default="scene_*.jpg")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jpeg-quality", type=int, default=95)

    ap.add_argument("--long-scene-h", type=int, default=2600)
    ap.add_argument("--long-scene-ratio", type=float, default=4.0)

    ap.add_argument("--sep-run", type=int, default=40)
    ap.add_argument("--sep-std-max", type=float, default=12.0)
    ap.add_argument("--sep-black-mean", type=float, default=25.0)
    ap.add_argument("--sep-white-mean", type=float, default=230.0)
    ap.add_argument("--sep-gray-std-max", type=float, default=6.0)
    ap.add_argument("--min-part-h", type=int, default=280)

    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--flat-std-drop", type=float, default=2.0)

    args = ap.parse_args()

    cfg = PostCfg(
        long_scene_h=args.long_scene_h,
        long_scene_ratio=args.long_scene_ratio,
        sep_run_px=args.sep_run,
        sep_std_max=args.sep_std_max,
        sep_black_mean=args.sep_black_mean,
        sep_white_mean=args.sep_white_mean,
        sep_gray_std_max=args.sep_gray_std_max,
        min_part_h=args.min_part_h,
        dedupe=not args.no_dedupe,
        flat_std_drop=args.flat_std_drop,
    )

    paths = sorted(glob.glob(os.path.join(args.in_dir, args.glob)), key=natural_key)
    if not paths:
        raise SystemExit("No input scenes found")

    os.makedirs(args.out_dir, exist_ok=True)

    out_idx = 1
    for p in paths:
        im = load_rgb(p)

        # 1) Trim top/bottom uniform/fade margins
        im = trim_top_bottom_by_uniform_rows(im, cfg)

        # 2) Drop scenes that are basically flat/blank (your gray gradient problem)
        if im.height < cfg.min_scene_hard:
            continue
        if overall_flat_std(im) <= cfg.flat_std_drop:
            continue

        # 3) Split long scenes on separator bands
        parts = [im]
        if should_split_long(im, cfg):
            parts = split_on_separators(im, cfg)

        # 4) Dedupe adjacent parts
        parts = dedupe_adjacent(parts, cfg)

        # 5) Write
        for part in parts:
            if part.height < cfg.min_scene_hard:
                continue
            out_path = os.path.join(args.out_dir, f"scene_{out_idx:04d}.jpg")
            part.save(out_path, "JPEG", quality=args.jpeg_quality)
            out_idx += 1

    print(f"Wrote {out_idx-1} scenes to: {args.out_dir}/")


if __name__ == "__main__":
    main()
