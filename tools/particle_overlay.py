#!/usr/bin/env python3
"""
particle_overlay.py — render the channel's ambient particle overlay.

A SEAMLESSLY LOOPING snow-dust/bokeh clip on black, composited by the
renderer with screen blending (the standard technique behind "cinematic"
particle overlays — pre-rendered particles, not DOM elements):

  - gaussian bokeh sprites, depth-scaled (near = big, soft, bright, fast)
  - sub-frame motion blur (3 temporal samples per frame)
  - turbulent sway from integer-cycle sines  -> exactly periodic
  - fall speeds quantized to whole wrap-cycles -> the loop never pops
  - per-particle twinkle, slight warm tint

Run (one-time channel asset):
  .eval_venv/bin/python tools/particle_overlay.py \
      --out remotion/assets/particles.mp4 --seconds 16 --fps 30
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List

import cv2
import numpy as np


def quantized_fall_speed(depth: float, *, loop_sec: float, wrap_h: float) -> float:
    """Fall speed (px/s) whose total travel over the loop is a WHOLE number of
    wrap heights — the seamless-loop requirement. Depth 0 (far) = 1 cycle,
    depth 1 (near) = up to 3 cycles per loop."""
    cycles = 1 + int(round(depth * 2))
    return cycles * wrap_h / loop_sec


def gaussian_sprite(sigma: float) -> np.ndarray:
    """Soft radial bokeh sprite, peak 1.0, radius ~3 sigma."""
    r = max(2, int(math.ceil(sigma * 3)))
    y, x = np.mgrid[-r: r + 1, -r: r + 1].astype(np.float64)
    s = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return s


def _particles(count: int, seed: int, width: int, height: int, loop_sec: float) -> List[Dict]:
    rng = np.random.RandomState(seed)
    wrap_h = float(height + 60)
    wrap_w = float(width + 120)
    out: List[Dict] = []
    for i in range(count):
        depth = float(rng.uniform(0, 1)) ** 1.4
        out.append({
            "x0": float(rng.uniform(0, wrap_w)),
            "y0": float(rng.uniform(0, wrap_h)),
            "vy": quantized_fall_speed(depth, loop_sec=loop_sec, wrap_h=wrap_h),
            # integer cycles over the loop => sway is exactly periodic
            "k1": int(rng.randint(1, 4)),
            "k2": int(rng.randint(2, 6)),
            "kt": int(rng.randint(1, 4)),
            "p1": float(rng.uniform(0, 2 * math.pi)),
            "p2": float(rng.uniform(0, 2 * math.pi)),
            "pt": float(rng.uniform(0, 2 * math.pi)),
            "a1": float(rng.uniform(18, 46)) * (0.5 + depth),
            "a2": float(rng.uniform(6, 16)),
            "sigma": 1.2 + depth * 7.5 * float(rng.uniform(0.8, 1.2)),
            "bright": (0.30 + 0.70 * depth) * float(rng.uniform(0.75, 1.0)),
            "depth": depth,
            "wrap_h": wrap_h,
            "wrap_w": wrap_w,
        })
    # a few hero bokeh orbs — very large, very soft, dim
    for j in range(max(2, count // 18)):
        p = dict(out[j])
        p.update({"sigma": float(rng.uniform(14, 22)), "bright": 0.22,
                  "vy": quantized_fall_speed(1.0, loop_sec=loop_sec, wrap_h=wrap_h)})
        out.append(p)
    return out


def _stamp(canvas: np.ndarray, sprite: np.ndarray, cx: float, cy: float, gain: float) -> None:
    h, w = canvas.shape[:2]
    sh, sw = sprite.shape
    x1 = int(round(cx)) - sw // 2
    y1 = int(round(cy)) - sh // 2
    x2, y2 = x1 + sw, y1 + sh
    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2, sy2 = sw - max(0, x2 - w), sh - max(0, y2 - h)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    region = canvas[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    region += sprite[sy1:sy2, sx1:sx2] * gain


def render_frame(t: float, *, width: int, height: int, loop_sec: float,
                 count: int, seed: int = 11) -> np.ndarray:
    """One overlay frame (uint8 BGR, black background) at time *t*.
    render_frame(0) == render_frame(loop_sec) by construction."""
    parts = _particles(count, seed, width, height, loop_sec)
    acc = np.zeros((height, width), np.float64)
    w = 2 * math.pi / loop_sec

    # sub-frame motion blur: average three temporal samples
    for dt in (-0.011, 0.0, 0.011):
        tt = t + dt
        for p in parts:
            y = (p["y0"] + p["vy"] * tt) % p["wrap_h"] - 30
            sway = (p["a1"] * math.sin(w * p["k1"] * tt + p["p1"])
                    + p["a2"] * math.sin(w * p["k2"] * tt + p["p2"]))
            x = (p["x0"] + sway) % p["wrap_w"] - 60
            tw = 0.78 + 0.22 * math.sin(w * p["kt"] * tt + p["pt"])
            _stamp(acc, gaussian_sprite(p["sigma"]), x, y, p["bright"] * tw / 3.0)

    acc = np.clip(acc, 0.0, 1.0)
    # slight warm tint (BGR)
    frame = np.dstack([acc * 235, acc * 248, acc * 255]).astype(np.uint8)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    n = int(round(args.seconds * args.fps))
    writer = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*fourcc),
                                 args.fps, (args.width, args.height))
        if writer.isOpened():
            print(f"[ok] encoder: {fourcc}")
            break
    if writer is None or not writer.isOpened():
        raise SystemExit("no usable mp4 encoder in OpenCV")

    for i in range(n):
        frame = render_frame(i / args.fps, width=args.width, height=args.height,
                             loop_sec=args.seconds, count=args.count, seed=args.seed)
        writer.write(frame)
        if i % args.fps == 0:
            print(f"[{i}/{n}]")
    writer.release()
    print(f"[ok] wrote {args.out} ({n} frames, seamless {args.seconds:.0f}s loop)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
