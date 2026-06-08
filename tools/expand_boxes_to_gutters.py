#!/usr/bin/env python3
"""
expand_boxes_to_gutters.py

Take panel boxes (normalized) and expand them to nearby gutter boundaries.

Input:
  - manifest.stitch.json
  - manifest.panels.json (from gemini_panel_boxes.py)

Output:
  - manifest.panels.expanded.json (same schema, but panels_norm expanded)
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(p: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def norm_to_px(b: List[float], w: int, h: int) -> List[int]:
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

def px_to_norm_xyxy(box: List[int], w: int, h: int) -> List[float]:
    x0, y0, x1, y1 = box
    return [y0 / h, x0 / w, y1 / h, x1 / w]

def _rowcol_scores(im: Image.Image) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    w, h = im.size
    arr = np.asarray(im).astype(np.float32)  # (h,w,3)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    row_white = ((r >= 245) & (g >= 245) & (b >= 245)).mean(axis=1)
    row_black = ((r <= 10) & (g <= 10) & (b <= 10)).mean(axis=1)
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    dx = np.abs(lum[:, 1:] - lum[:, :-1]).mean(axis=1) / 255.0
    row_edge = dx

    col_white = ((r >= 245) & (g >= 245) & (b >= 245)).mean(axis=0)
    col_black = ((r <= 10) & (g <= 10) & (b <= 10)).mean(axis=0)

    dy = np.abs(lum[1:, :] - lum[:-1, :]).mean(axis=0) / 255.0
    col_edge = dy

    return row_white.tolist(), row_black.tolist(), row_edge.tolist(), col_white.tolist(), col_black.tolist(), col_edge.tolist()

def is_gutter_run_row(row_white: float, row_black: float, row_edge: float, white_min: float, black_min: float, edge_max: float) -> bool:
    return ((row_white >= white_min) or (row_black >= black_min)) and (row_edge <= edge_max)

def is_gutter_run_col(col_white: float, col_black: float, col_edge: float, white_min: float, black_min: float, edge_max: float) -> bool:
    return ((col_white >= white_min) or (col_black >= black_min)) and (col_edge <= edge_max)

def merge_boxes_px(boxes: List[List[int]], w: int, h: int, gap_px: int, x_overlap_min: float) -> List[List[int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # y0, x0
    merged = [boxes[0]]

    for b in boxes[1:]:
        x0, y0, x1, y1 = b
        mx0, my0, mx1, my1 = merged[-1]
        inter_x = max(0, min(x1, mx1) - max(x0, mx0))
        w_small = max(1, min(x1 - x0, mx1 - mx0))
        overlap = inter_x / w_small
        v_gap = y0 - my1
        if overlap >= x_overlap_min and v_gap >= 0 and v_gap <= gap_px:
            merged[-1] = [min(mx0, x0), min(my0, y0), max(mx1, x1), max(my1, y1)]
        else:
            merged.append(b)

    out = []
    for x0, y0, x1, y1 in merged:
        out.append([int(clamp(x0, 0, w)), int(clamp(y0, 0, h)), int(clamp(x1, 2, w)), int(clamp(y1, 2, h))])
    return out

def expand_box_to_gutters(
    box: List[int],
    row_white: List[float],
    row_black: List[float],
    row_edge: List[float],
    col_white: List[float],
    col_black: List[float],
    col_edge: List[float],
    w: int,
    h: int,
    max_expand: int,
    run_len: int,
    white_min: float,
    black_min: float,
    edge_max: float,
) -> List[int]:
    x0, y0, x1, y1 = box

    def find_up(y: int) -> int:
        limit = max(0, y - max_expand)
        yy = y
        while yy > limit:
            ok = True
            for k in range(run_len):
                yk = yy - k
                if yk < 0 or not is_gutter_run_row(row_white[yk], row_black[yk], row_edge[yk], white_min, black_min, edge_max):
                    ok = False
                    break
            if ok:
                return max(0, yy - run_len + 1)
            yy -= 1
        return y

    def find_down(y: int) -> int:
        limit = min(h - 1, y + max_expand)
        yy = y
        while yy < limit:
            ok = True
            for k in range(run_len):
                yk = yy + k
                if yk >= h or not is_gutter_run_row(row_white[yk], row_black[yk], row_edge[yk], white_min, black_min, edge_max):
                    ok = False
                    break
            if ok:
                return min(h - 1, yy + run_len - 1)
            yy += 1
        return y

    def find_left(x: int) -> int:
        limit = max(0, x - max_expand)
        xx = x
        while xx > limit:
            ok = True
            for k in range(run_len):
                xk = xx - k
                if xk < 0 or not is_gutter_run_col(col_white[xk], col_black[xk], col_edge[xk], white_min, black_min, edge_max):
                    ok = False
                    break
            if ok:
                return max(0, xx - run_len + 1)
            xx -= 1
        return x

    def find_right(x: int) -> int:
        limit = min(w - 1, x + max_expand)
        xx = x
        while xx < limit:
            ok = True
            for k in range(run_len):
                xk = xx + k
                if xk >= w or not is_gutter_run_col(col_white[xk], col_black[xk], col_edge[xk], white_min, black_min, edge_max):
                    ok = False
                    break
            if ok:
                return min(w - 1, xx + run_len - 1)
            xx += 1
        return x

    new_y0 = find_up(y0)
    new_y1 = find_down(y1 - 1) + 1
    new_x0 = find_left(x0)
    new_x1 = find_right(x1 - 1) + 1

    new_x0 = int(clamp(new_x0, 0, w - 2))
    new_y0 = int(clamp(new_y0, 0, h - 2))
    new_x1 = int(clamp(new_x1, new_x0 + 2, w))
    new_y1 = int(clamp(new_y1, new_y0 + 2, h))
    return [new_x0, new_y0, new_x1, new_y1]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-manifest", required=True)
    ap.add_argument("--panels-manifest", required=True)
    ap.add_argument("--out-panels-manifest", required=True)

    ap.add_argument("--max-expand", type=int, default=900)
    ap.add_argument("--run-len", type=int, default=18)

    ap.add_argument("--white-min", type=float, default=0.92)
    ap.add_argument("--black-min", type=float, default=0.92)
    ap.add_argument("--edge-max", type=float, default=0.020)

    ap.add_argument("--merge-gap-px", type=int, default=120)
    ap.add_argument("--merge-x-overlap", type=float, default=0.70)

    args = ap.parse_args()

    stitch = load_json(args.stitch_manifest)
    panels = load_json(args.panels_manifest)

    stitch_by_file = {c.get("chunk_file"): c for c in (stitch.get("chunks") or []) if c.get("chunk_file")}

    out = json.loads(json.dumps(panels))
    out_chunks = []

    for ch in (panels.get("chunks") or []):
        cf = ch.get("chunk_file")
        if not cf:
            out_chunks.append(ch)
            continue

        st = stitch_by_file.get(cf)
        chunk_path = (st or {}).get("chunk_path") or ch.get("chunk_path")
        if not chunk_path or not os.path.exists(chunk_path):
            out_chunks.append(ch)
            continue

        with Image.open(chunk_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            row_white, row_black, row_edge, col_white, col_black, col_edge = _rowcol_scores(im)

            boxes = ch.get("panels_norm") or []
            boxes_px = []
            for b in boxes:
                if isinstance(b, list) and len(b) == 4:
                    boxes_px.append(norm_to_px(b, w, h))

            boxes_px = merge_boxes_px(boxes_px, w, h, gap_px=int(args.merge_gap_px), x_overlap_min=float(args.merge_x_overlap))

            expanded_norm = []
            for bpx in boxes_px:
                eb = expand_box_to_gutters(
                    bpx,
                    row_white, row_black, row_edge,
                    col_white, col_black, col_edge,
                    w, h,
                    max_expand=int(args.max_expand),
                    run_len=int(args.run_len),
                    white_min=float(args.white_min),
                    black_min=float(args.black_min),
                    edge_max=float(args.edge_max),
                )
                expanded_norm.append(px_to_norm_xyxy(eb, w, h))

            new_ch = dict(ch)
            new_ch["panels_norm"] = expanded_norm
            new_ch["expanded"] = {
                "max_expand": int(args.max_expand),
                "run_len": int(args.run_len),
                "white_min": float(args.white_min),
                "black_min": float(args.black_min),
                "edge_max": float(args.edge_max),
                "merge_gap_px": int(args.merge_gap_px),
                "merge_x_overlap": float(args.merge_x_overlap),
            }
            out_chunks.append(new_ch)

    out["chunks"] = out_chunks
    dump_json(args.out_panels_manifest, out)
    print(f"[ok] wrote expanded panels: {os.path.abspath(args.out_panels_manifest)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
