#!/usr/bin/env python3
import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Data helpers
# ----------------------------

@dataclass
class Box:
    # pixel coords (x1,y1,x2,y2) inclusive-exclusive
    x1: int
    y1: int
    x2: int
    y2: int
    src: str  # "ocr_word" / "text_block" / etc.

    def clamp(self, w: int, h: int) -> "Box":
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(1, min(self.x2, w))
        y2 = max(1, min(self.y2, h))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return Box(x1, y1, x2, y2, self.src)

    def pad(self, p: int) -> "Box":
        return Box(self.x1 - p, self.y1 - p, self.x2 + p, self.y2 + p, self.src)

    def dilate(self, d: int) -> "Box":
        return self.pad(d)

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def chunk_box_to_scene_local(
    chunk_box_xyxy: Tuple[float, float, float, float],
    scene_box_xyxy: Tuple[float, float, float, float],
    min_px: int = 3,
) -> Optional[Tuple[int, int, int, int]]:
    """Map a chunk-space element box (e.g. a YOLO speech_bubble) into a scene
    crop's local pixel coordinates.

    Returns the intersection shifted to the crop origin, or None when the
    overlap is thinner than *min_px*. Slivers of bubbles poking in from
    OUTSIDE the crop are returned on purpose — those remnant arcs are exactly
    what the inpaint mask must cover.
    """
    bx1, by1, bx2, by2 = chunk_box_xyxy
    sx1, sy1, sx2, sy2 = scene_box_xyxy
    ix1, iy1 = max(bx1, sx1), max(by1, sy1)
    ix2, iy2 = min(bx2, sx2), min(by2, sy2)
    if (ix2 - ix1) < min_px or (iy2 - iy1) < min_px:
        return None
    return (int(ix1 - sx1), int(iy1 - sy1), int(ix2 - sx1), int(iy2 - sy1))


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_manifest(m: Any) -> List[Dict[str, Any]]:
    """
    Accept:
      - list[scene]
      - dict with key holding list (scenes/items/data/entries)
      - dict of dict entries keyed by filename/id
    """
    if isinstance(m, list):
        return m
    if isinstance(m, dict):
        for k in ("scenes", "items", "data", "entries"):
            if k in m and isinstance(m[k], list):
                return m[k]
        if all(isinstance(v, dict) for v in m.values()):
            return list(m.values())
    raise SystemExit("manifest format not recognized (expected list or dict containing a list)")


def find_scene_entry(manifest: List[Dict[str, Any]], filename: str) -> Optional[Dict[str, Any]]:
    for e in manifest:
        sf = e.get("scene_file")
        sp = e.get("scene_path")
        if sf and sf == filename:
            return e
        if sp and os.path.basename(sp) == filename:
            return e
    return None


def norm_bbox_to_px(b: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(round(b[0] * w))
    y1 = int(round(b[1] * h))
    x2 = int(round(b[2] * w))
    y2 = int(round(b[3] * h))
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    if x2 == x1:
        x2 += 1
    if y2 == y1:
        y2 += 1
    return x1, y1, x2, y2


# ----------------------------
# Box building
# ----------------------------

def boxes_from_text_blocks(scene: Dict[str, Any], w: int, h: int) -> List[Box]:
    out: List[Box] = []
    vision = scene.get("vision") or {}

    for b in (vision.get("text_blocks") or []):
        if isinstance(b, list) and len(b) == 4:
            x1, y1, x2, y2 = norm_bbox_to_px(b, w, h)
            out.append(Box(x1, y1, x2, y2, "text_block"))

    for t in (scene.get("targets") or []):
        if (t.get("type") == "text_block") and isinstance(t.get("bbox"), list) and len(t["bbox"]) == 4:
            x1, y1, x2, y2 = norm_bbox_to_px(t["bbox"], w, h)
            out.append(Box(x1, y1, x2, y2, "text_block_target"))

    return out


def boxes_from_ocr_words(scene: Dict[str, Any], w: int, h: int) -> List[Box]:
    out: List[Box] = []
    vision = scene.get("vision") or {}
    for ow in (vision.get("ocr_words") or []):
        b = ow.get("bbox")
        if isinstance(b, list) and len(b) == 4:
            x1, y1, x2, y2 = norm_bbox_to_px(b, w, h)
            out.append(Box(x1, y1, x2, y2, "ocr_word"))
    return out


def merge_boxes_linewise(boxes: List[Box], y_tol: int = 12, x_gap: int = 14) -> List[Box]:
    """
    Merge words into line boxes by y-closeness + x proximity.
    Prevents punctuation boxes (!) from becoming separate masks.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b.y1, b.x1))

    lines: List[List[Box]] = []
    for b in boxes:
        placed = False
        for line in lines:
            lb = line[-1]
            if abs(b.y1 - lb.y1) <= y_tol or abs(b.y2 - lb.y2) <= y_tol:
                if b.x1 <= lb.x2 + x_gap:
                    line.append(b)
                    placed = True
                    break
        if not placed:
            lines.append([b])

    merged: List[Box] = []
    for line in lines:
        x1 = min(b.x1 for b in line)
        y1 = min(b.y1 for b in line)
        x2 = max(b.x2 for b in line)
        y2 = max(b.y2 for b in line)
        merged.append(Box(x1, y1, x2, y2, "ocr_line"))

    return merged


# ----------------------------
# Region stats (bubble vs SFX)
# ----------------------------

def _roi(img_bgr: np.ndarray, box: Box) -> np.ndarray:
    return img_bgr[box.y1:box.y2, box.x1:box.x2]


def compute_region_stats(
    img_bgr: np.ndarray,
    box: Box,
    white_thr: int,
    bright_thr: int,
    sat_low_thr: int,
) -> Dict[str, float]:
    """
    Computes robust stats for deciding bubble-like background.
    Uses "bright pixels only" for std, to avoid ink/outline skewing std.
    """
    roi = _roi(img_bgr, box)
    if roi.size == 0:
        return {
            "white_frac": 0.0,
            "median_gray": 0.0,
            "std_gray": 999.0,
            "std_gray_bright": 999.0,
            "bright_frac": 0.0,
            "low_sat_frac": 0.0,
            "median_sat": 255.0,
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    white_frac = float(np.mean(gray >= white_thr))
    median_gray = float(np.median(gray))
    std_gray = float(np.std(gray))

    bright_mask = gray >= bright_thr
    bright_frac = float(np.mean(bright_mask))
    if np.any(bright_mask):
        std_gray_bright = float(np.std(gray[bright_mask]))
    else:
        std_gray_bright = 999.0

    low_sat_frac = float(np.mean(sat <= sat_low_thr))
    median_sat = float(np.median(sat))

    return {
        "white_frac": white_frac,
        "median_gray": median_gray,
        "std_gray": std_gray,
        "std_gray_bright": std_gray_bright,
        "bright_frac": bright_frac,
        "low_sat_frac": low_sat_frac,
        "median_sat": median_sat,
    }


def compute_ring_stats(
    img_bgr: np.ndarray,
    box: Box,
    ring: int,
    white_thr: int,
    bright_thr: int,
    sat_low_thr: int,
) -> Dict[str, float]:
    """
    Measures what surrounds the text:
    - For bubbles: ring is usually bright, fairly uniform (on bright pixels), low-saturation.
    - For SFX on textured/colored bg: ring is darker or higher texture or higher saturation.
    """
    h, w = img_bgr.shape[:2]
    bx = box.clamp(w, h)

    outer = bx.pad(ring).clamp(w, h)
    inner = bx.pad(1).clamp(w, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (outer.x1, outer.y1), (outer.x2 - 1, outer.y2 - 1), 255, thickness=-1)
    cv2.rectangle(mask, (inner.x1, inner.y1), (inner.x2 - 1, inner.y2 - 1), 0, thickness=-1)

    ys, xs = np.where(mask == 255)
    if len(xs) < 80:
        return {
            "ring_white_frac": 0.0,
            "ring_median_gray": 0.0,
            "ring_std_gray": 999.0,
            "ring_std_gray_bright": 999.0,
            "ring_bright_frac": 0.0,
            "ring_low_sat_frac": 0.0,
            "ring_median_sat": 255.0,
        }

    samples = img_bgr[ys, xs]
    gray = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
    hsv = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[:, 1]

    ring_white_frac = float(np.mean(gray >= white_thr))
    ring_median_gray = float(np.median(gray))
    ring_std_gray = float(np.std(gray))

    bright_mask = gray >= bright_thr
    ring_bright_frac = float(np.mean(bright_mask))
    if np.any(bright_mask):
        ring_std_gray_bright = float(np.std(gray[bright_mask]))
    else:
        ring_std_gray_bright = 999.0

    ring_low_sat_frac = float(np.mean(sat <= sat_low_thr))
    ring_median_sat = float(np.median(sat))

    return {
        "ring_white_frac": ring_white_frac,
        "ring_median_gray": ring_median_gray,
        "ring_std_gray": ring_std_gray,
        "ring_std_gray_bright": ring_std_gray_bright,
        "ring_bright_frac": ring_bright_frac,
        "ring_low_sat_frac": ring_low_sat_frac,
        "ring_median_sat": ring_median_sat,
    }


# ----------------------------
# Fill / inpaint
# ----------------------------

def pick_bubble_fill_color(img_bgr: np.ndarray, sample_box: Box, bright_thr: int) -> Tuple[int, int, int]:
    """
    Sample color from bright pixels in the sample_box (bubble interior),
    avoids picking ink/outline.
    """
    h, w = img_bgr.shape[:2]
    sb = sample_box.clamp(w, h)
    roi = _roi(img_bgr, sb)
    if roi.size == 0:
        return (255, 255, 255)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = gray >= bright_thr
    if np.any(bright):
        px = roi[bright]
        med = np.median(px, axis=0)
        return (int(med[0]), int(med[1]), int(med[2]))

    med = np.median(roi.reshape(-1, 3), axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def apply_solid_fill(img_bgr: np.ndarray, mask_box: Box, sample_box: Box, bright_thr: int) -> None:
    mb = mask_box
    col = pick_bubble_fill_color(img_bgr, sample_box, bright_thr=bright_thr)
    cv2.rectangle(img_bgr, (mb.x1, mb.y1), (mb.x2 - 1, mb.y2 - 1), col, thickness=-1)


def inpaint_regions(img_bgr: np.ndarray, mask: np.ndarray, radius: int = 7) -> np.ndarray:
    return cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_TELEA)


# ----------------------------
# Debug artifacts
# ----------------------------

def write_debug(debug_dir: str, basename: str, overlay: np.ndarray, mask: np.ndarray, info: Dict[str, Any]) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"{basename}.overlay.jpg"), overlay)
    cv2.imwrite(os.path.join(debug_dir, f"{basename}.mask.png"), mask)
    with open(os.path.join(debug_dir, f"{basename}.debug.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def draw_overlay(img_bgr: np.ndarray, kept_mask_boxes: List[Box], skipped_raw_boxes: List[Box]) -> np.ndarray:
    ov = img_bgr.copy()
    for b in skipped_raw_boxes:
        cv2.rectangle(ov, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), (0, 0, 255), 2)  # red
    for b in kept_mask_boxes:
        cv2.rectangle(ov, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), (0, 255, 0), 2)  # green
    return ov


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Remove speech-bubble text from manhwa panels using manifest OCR boxes.")
    ap.add_argument("--panels-dir", required=True, help="Directory with panel images (scenes_raw).")
    ap.add_argument("--vision-manifest", required=True, help="manifest.vision.json path.")
    ap.add_argument("--out-dir", required=True, help="Output directory for cleaned images.")
    ap.add_argument("--out-manifest", required=True, help="Output manifest path.")
    ap.add_argument("--glob", default="*.jpg", help="Glob pattern inside panels-dir.")
    ap.add_argument("--jpeg-quality", type=int, default=92)

    ap.add_argument("--use-ocr-words", action="store_true", help="Use per-word OCR boxes (then merge into line boxes).")
    ap.add_argument("--fill", choices=["solid", "inpaint"], default="solid")
    ap.add_argument("--dilate", type=int, default=12, help="Dilate boxes for removal (pixels).")

    # behavior flags
    ap.add_argument("--force", action="store_true",
                    help="More permissive for bubbles, BUT still skips likely SFX unless --disable-sfx-filter.")
    ap.add_argument("--disable-sfx-filter", action="store_true",
                    help="Remove all detected boxes (dangerous; will remove SFX too).")

    ap.add_argument("--debug-dir", default=None, help="If set, write overlay/mask/debug json per processed panel.")
    ap.add_argument("--safe-pad", type=int, default=6, help="Pad ROI for safe check (pixels).")
    ap.add_argument("--ring-size", type=int, default=8, help="Ring thickness for surrounding-background sampling.")

    # thresholds (tune here, not in code)
    ap.add_argument("--white-thr", type=int, default=225, help="Gray >= thr counts as white.")
    ap.add_argument("--bright-thr", type=int, default=170, help="Gray >= thr counts as bright (bubble interior/ring).")
    ap.add_argument("--sat-low-thr", type=int, default=45, help="HSV saturation <= thr considered low-saturation (bubble-like).")

    # bubble tests
    ap.add_argument("--roi-min-bright-frac", type=float, default=0.22, help="ROI bright fraction to consider bubble-like.")
    ap.add_argument("--roi-max-std-bright", type=float, default=28.0, help="Max std on bright pixels inside ROI.")
    ap.add_argument("--roi-min-low-sat-frac", type=float, default=0.65, help="Min low-sat fraction inside ROI.")

    ap.add_argument("--ring-min-white-frac", type=float, default=0.20, help="Ring white fraction to consider bubble-like.")
    ap.add_argument("--ring-min-median-gray", type=float, default=165.0, help="Ring median gray to consider bubble-like.")
    ap.add_argument("--ring-max-std-bright", type=float, default=26.0, help="Max std on bright pixels in ring.")
    ap.add_argument("--ring-min-low-sat-frac", type=float, default=0.70, help="Min low-sat fraction in ring.")

    # filter manifest support (skip inpainting for text-only or excluded scenes)
    ap.add_argument(
        "--filter-manifest",
        default=None,
        help="Optional manifest (e.g., manifest.filtered.json) containing items with use_for_video=false to skip inpainting.",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    manifest_raw = load_json(args.vision_manifest)
    manifest = normalize_manifest(manifest_raw)

    # Build skip set from filter manifest (by filename)
    skip_files: set[str] = set()
    if args.filter_manifest:
        fm_raw = load_json(args.filter_manifest)
        fm = normalize_manifest(fm_raw)
        for e in fm:
            if e.get("use_for_video") is False:
                sf = e.get("scene_file")
                sp = e.get("scene_path")
                if sf:
                    skip_files.add(sf)
                elif sp:
                    skip_files.add(os.path.basename(sp))

    files = sorted(glob.glob(os.path.join(args.panels_dir, args.glob)))
    if not files:
        print("[warn] no files matched")
        save_json(args.out_manifest, [])
        return 0

    out_manifest: List[Dict[str, Any]] = []
    stats = {
        "cleaned": 0,
        "cleaned_forced": 0,
        "skipped_by_filter": 0,
        "copied_no_vision": 0,
        "copied_no_text": 0,
        "copied_rejected": 0,
        "errors": 0,
        "fill": args.fill,
        "inpaint": (args.fill == "inpaint"),
    }

    for fp in files:
        base = os.path.basename(fp)

        # If filtered out, do not remove bubble text; just copy through.
        if base in skip_files:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is None:
                stats["errors"] += 1
                out_manifest.append({"scene_file": base, "out_file": None, "status": "error", "error": "cv2.imread failed"})
                continue
            out_path = os.path.join(args.out_dir, base)
            cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            out_manifest.append({"scene_file": base, "out_file": base, "status": "skipped_by_filter"})
            stats["skipped_by_filter"] += 1
            continue

        try:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("cv2.imread failed")

            h, w = img.shape[:2]
            scene = find_scene_entry(manifest, base)
            if scene is None:
                out_path = os.path.join(args.out_dir, base)
                cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                out_manifest.append({"scene_file": base, "out_file": base, "status": "copied_no_vision"})
                stats["copied_no_vision"] += 1
                continue

            # RAW candidate boxes (NO dilation yet)
            if args.use_ocr_words:
                word_boxes = [b.clamp(w, h) for b in boxes_from_ocr_words(scene, w, h)]
                cand_raw = merge_boxes_linewise(word_boxes, y_tol=12, x_gap=14)
            else:
                cand_raw = boxes_from_text_blocks(scene, w, h)

            cand_raw = [b.clamp(w, h) for b in cand_raw if b.area() > 0]

            if not cand_raw:
                out_path = os.path.join(args.out_dir, base)
                cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                out_manifest.append({"scene_file": base, "out_file": base, "status": "copied_no_text"})
                stats["copied_no_text"] += 1
                continue

            kept_raw: List[Box] = []
            skipped_raw: List[Box] = []

            debug_info: Dict[str, Any] = {
                "file": base,
                "image_wh": [w, h],
                "force": bool(args.force),
                "disable_sfx_filter": bool(args.disable_sfx_filter),
                "params": {
                    "safe_pad": args.safe_pad,
                    "ring_size": args.ring_size,
                    "white_thr": args.white_thr,
                    "bright_thr": args.bright_thr,
                    "sat_low_thr": args.sat_low_thr,
                    "roi_min_bright_frac": args.roi_min_bright_frac,
                    "roi_max_std_bright": args.roi_max_std_bright,
                    "roi_min_low_sat_frac": args.roi_min_low_sat_frac,
                    "ring_min_white_frac": args.ring_min_white_frac,
                    "ring_min_median_gray": args.ring_min_median_gray,
                    "ring_max_std_bright": args.ring_max_std_bright,
                    "ring_min_low_sat_frac": args.ring_min_low_sat_frac,
                },
                "boxes": [],
            }

            for b in cand_raw:
                safe_roi = b.pad(args.safe_pad).clamp(w, h)

                roi_stats = compute_region_stats(
                    img, safe_roi,
                    white_thr=args.white_thr,
                    bright_thr=args.bright_thr,
                    sat_low_thr=args.sat_low_thr,
                )

                ring_stats = compute_ring_stats(
                    img, safe_roi,
                    ring=args.ring_size,
                    white_thr=args.white_thr,
                    bright_thr=args.bright_thr,
                    sat_low_thr=args.sat_low_thr,
                )

                # Bubble-like if ROI background looks like a bubble interior...
                bubble_by_roi = (
                    (roi_stats["bright_frac"] >= args.roi_min_bright_frac) and
                    (roi_stats["std_gray_bright"] <= args.roi_max_std_bright) and
                    (roi_stats["low_sat_frac"] >= args.roi_min_low_sat_frac) and
                    (roi_stats["median_gray"] >= 140.0)
                )

                # ...or surrounding ring looks like bubble background (bright, uniform on bright pixels, low saturation)
                bubble_by_ring = (
                    (ring_stats["ring_white_frac"] >= args.ring_min_white_frac) and
                    (ring_stats["ring_median_gray"] >= args.ring_min_median_gray) and
                    (ring_stats["ring_std_gray_bright"] <= args.ring_max_std_bright) and
                    (ring_stats["ring_low_sat_frac"] >= args.ring_min_low_sat_frac)
                )

                is_bubble_like = bool(bubble_by_roi or bubble_by_ring)

                # Explicit SFX-likely rejection signals:
                sfx_like = (
                    (ring_stats["ring_median_gray"] < 120.0 and ring_stats["ring_white_frac"] < 0.10) or
                    (ring_stats["ring_low_sat_frac"] < 0.45 and ring_stats["ring_median_sat"] > 70.0)
                )

                decision: str
                if args.disable_sfx_filter:
                    kept_raw.append(b)
                    decision = "kept_disable_sfx_filter"
                elif args.force:
                    # force = permissive for bubbles, but do not delete obvious SFX
                    if is_bubble_like and not sfx_like:
                        kept_raw.append(b)
                        decision = "kept_force_bubble"
                    else:
                        skipped_raw.append(b)
                        decision = "skipped_force_sfx"
                else:
                    # normal mode
                    if is_bubble_like and not sfx_like:
                        kept_raw.append(b)
                        decision = "kept_safe"
                    else:
                        skipped_raw.append(b)
                        decision = "skipped_unsafe"

                debug_info["boxes"].append({
                    "raw_bbox": b.as_tuple(),
                    "safe_roi": safe_roi.as_tuple(),
                    "roi": roi_stats,
                    "ring": ring_stats,
                    "bubble_by_roi": bool(bubble_by_roi),
                    "bubble_by_ring": bool(bubble_by_ring),
                    "is_bubble_like": bool(is_bubble_like),
                    "sfx_like": bool(sfx_like),
                    "decision": decision,
                    "src": b.src,
                })

            if not kept_raw:
                out_path = os.path.join(args.out_dir, base)
                cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                out_manifest.append({"scene_file": base, "out_file": base, "status": "copied_rejected"})
                stats["copied_rejected"] += 1

                if args.debug_dir:
                    overlay = draw_overlay(img, kept_mask_boxes=[], skipped_raw_boxes=skipped_raw)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    write_debug(args.debug_dir, base, overlay, mask, debug_info)
                continue

            # DILATED MASK boxes only for kept
            kept_mask = [b.dilate(args.dilate).clamp(w, h) for b in kept_raw]

            mask = np.zeros((h, w), dtype=np.uint8)
            if args.fill == "inpaint":
                for b in kept_mask:
                    cv2.rectangle(mask, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), 255, thickness=-1)
                cleaned = inpaint_regions(img, mask, radius=7)
            else:
                cleaned = img.copy()
                for raw_b, mask_b in zip(kept_raw, kept_mask):
                    cv2.rectangle(mask, (mask_b.x1, mask_b.y1), (mask_b.x2 - 1, mask_b.y2 - 1), 255, thickness=-1)
                    # sample from safe ROI, paint on dilated mask
                    sample_box = raw_b.pad(args.safe_pad).clamp(w, h)
                    apply_solid_fill(cleaned, mask_box=mask_b, sample_box=sample_box, bright_thr=args.bright_thr)

            out_path = os.path.join(args.out_dir, base)
            cv2.imwrite(out_path, cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

            out_manifest.append({
                "scene_file": base,
                "out_file": base,
                "status": "cleaned_force" if args.force else "cleaned",
                "removed_boxes": [b.as_tuple() for b in kept_mask],
                "skipped_boxes": [b.as_tuple() for b in skipped_raw],
            })

            if args.force:
                stats["cleaned_forced"] += 1
            else:
                stats["cleaned"] += 1

            if args.debug_dir:
                overlay = draw_overlay(img, kept_mask_boxes=kept_mask, skipped_raw_boxes=skipped_raw)
                write_debug(args.debug_dir, base, overlay, mask, debug_info)

        except Exception as e:
            stats["errors"] += 1
            out_manifest.append({
                "scene_file": base,
                "out_file": None,
                "status": "error",
                "error": str(e),
            })

    save_json(args.out_manifest, out_manifest)
    print(
        f"[ok] wrote={args.out_manifest} "
        f"cleaned={stats['cleaned']} cleaned_forced={stats['cleaned_forced']} "
        f"skipped_by_filter={stats['skipped_by_filter']} "
        f"copied_no_vision={stats['copied_no_vision']} copied_no_text={stats['copied_no_text']} "
        f"copied_rejected={stats['copied_rejected']} errors={stats['errors']} "
        f"fill={stats['fill']} inpaint={stats['inpaint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
