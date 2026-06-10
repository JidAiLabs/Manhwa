#!/usr/bin/env python3
"""
panels_to_scenes.py

Crop "panel scenes" out of stitched chunk images using Gemini panel boxes.

Inputs:
  - manifest.stitch.json  (from chunk_stitch*.py)
  - manifest.panels.json  (from gemini_panel_boxes.py)

Outputs:
  - scenes_raw/ panel_000001.jpg ...
  - manifest.scenes.json  (panel -> scene mapping + geometry)

Key behaviors:
  - Reading order = chunk_index asc, then bbox y0 asc, then x0 asc
  - Optional padding around boxes
  - Optional dedupe (useful for overlapped chunks) via perceptual dHash
  - Optional "blank" filtering (skip near-white/near-black crops)

Notes:
  - This script does NOT remove text. It produces RAW panel crops.
  - You can later run vision_extract.py on scenes_raw/ to get OCR + objects/faces/labels.

Usage example:
  python3 tools/panels_to_scenes.py \
    --stitch-manifest "/.../Episode/manifest.stitch.json" \
    --panels-manifest "/.../Episode/manifest.panels.json" \
    --out-dir "/.../Episode/scenes_raw" \
    --out-manifest "/.../Episode/manifest.scenes.json" \
    --jpeg-quality 92 \
    --pad-px 6 \
    --dedupe \
    --dedupe-threshold 9 \
    --skip-blank \
    --blank-threshold 0.985
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# JSON helpers
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Geometry helpers
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def norm_to_px(b: List[float], w: int, h: int) -> List[int]:
    """
    b = [y0,x0,y1,x1] in 0..1
    returns [x0,y0,x1,y1] px ints (PIL crop format)
    """
    y0, x0, y1, x1 = [float(v) for v in b]
    y0 = clamp(y0, 0.0, 1.0)
    x0 = clamp(x0, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x1 = clamp(x1, 0.0, 1.0)

    # Convert to pixels
    px0 = int(round(x0 * w))
    py0 = int(round(y0 * h))
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))

    # Ensure valid
    px0 = clamp(px0, 0, w - 1)
    py0 = clamp(py0, 0, h - 1)
    px1 = clamp(px1, px0 + 1, w)
    py1 = clamp(py1, py0 + 1, h)

    return [int(px0), int(py0), int(px1), int(py1)]


def pad_box_xyxy(box: List[int], w: int, h: int, pad_px: int) -> List[int]:
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad_px)
    y0 = max(0, y0 - pad_px)
    x1 = min(w, x1 + pad_px)
    y1 = min(h, y1 + pad_px)
    if x1 <= x0 + 1:
        x1 = min(w, x0 + 2)
    if y1 <= y0 + 1:
        y1 = min(h, y0 + 2)
    return [x0, y0, x1, y1]


def area_norm(b: List[float]) -> float:
    y0, x0, y1, x1 = b
    return max(0.0, y1 - y0) * max(0.0, x1 - x0)


# -----------------------------
# Perceptual hash dedupe
# -----------------------------
def dhash64(im: Image.Image) -> int:
    """
    64-bit dHash:
      - resize to (9, 8)
      - compare adjacent pixels horizontally
    """
    g = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = list(g.getdata())  # row-major length 72
    bits = 0
    bitpos = 0
    for y in range(8):
        row = px[y * 9 : (y + 1) * 9]
        for x in range(8):
            bits <<= 1
            bits |= 1 if row[x] > row[x + 1] else 0
            bitpos += 1
    return bits


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# -----------------------------
# Blank detection
# -----------------------------
def blank_score(im: Image.Image) -> float:
    """
    Returns fraction of pixels that are very close to white or very close to black.
    Higher means more "blank-ish".
    """
    g = im.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
    px = list(g.getdata())
    if not px:
        return 1.0
    hi = sum(1 for v in px if v >= 250)
    lo = sum(1 for v in px if v <= 5)
    return (hi + lo) / float(len(px))


@dataclass
class PanelItem:
    chunk_index: int
    chunk_file: str
    chunk_path: str
    chunk_w: int
    chunk_h: int
    panel_idx_in_chunk: int
    bbox_norm: List[float]


def build_chunk_index(stitch: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns chunk_file -> chunk dict (includes chunk_index, chunk_path, etc.)
    """
    out: Dict[str, Dict[str, Any]] = {}
    for ch in (stitch.get("chunks") or []):
        cf = ch.get("chunk_file")
        if cf:
            out[cf] = ch
    return out


def collect_panels(stitch: Dict[str, Any], panels: Dict[str, Any]) -> List[PanelItem]:
    stitch_by_file = build_chunk_index(stitch)

    items: List[PanelItem] = []
    for ch in (panels.get("chunks") or []):
        cf = ch.get("chunk_file")
        if not cf:
            continue
        st = stitch_by_file.get(cf)
        if not st:
            # manifest mismatch (different out_dir etc.)
            continue

        chunk_index = int(st.get("chunk_index") or 0)
        chunk_path = st.get("chunk_path") or ch.get("chunk_path")
        cw = int(ch.get("chunk_w") or st.get("chunk_w") or 0)
        chh = int(ch.get("chunk_h") or st.get("chunk_h") or 0)

        boxes = ch.get("panels_norm") or []
        for i, b in enumerate(boxes):
            if not isinstance(b, list) or len(b) != 4:
                continue
            items.append(
                PanelItem(
                    chunk_index=chunk_index,
                    chunk_file=cf,
                    chunk_path=str(chunk_path),
                    chunk_w=cw,
                    chunk_h=chh,
                    panel_idx_in_chunk=i,
                    bbox_norm=[float(x) for x in b],
                )
            )

    # reading order: chunk index asc, y0 asc, x0 asc
    items.sort(key=lambda it: (it.chunk_index, it.bbox_norm[0], it.bbox_norm[1], -area_norm(it.bbox_norm)))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-manifest", required=True)
    ap.add_argument("--panels-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)

    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--pad-px", type=int, default=0)

    ap.add_argument("--min-area-frac", type=float, default=0.0, help="Extra guard: drop bboxes smaller than this norm area")

    # overlap dedupe
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--dedupe-threshold", type=int, default=9, help="dHash Hamming distance <= this => duplicate")

    # blank skip
    ap.add_argument("--skip-blank", action="store_true")
    ap.add_argument("--blank-threshold", type=float, default=0.985, help="blank_score >= this => skip scene")

    args = ap.parse_args()

    stitch = load_json(args.stitch_manifest)
    panels = load_json(args.panels_manifest)

    items = collect_panels(stitch, panels)
    if not items:
        raise SystemExit("No panel boxes found to crop. Check manifest.panels.json contains panels_norm.")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seen_hashes: List[int] = []
    written = 0
    skipped_dupe = 0
    skipped_blank = 0
    skipped_small = 0
    errors = 0

    scenes: List[Dict[str, Any]] = []

    for idx, it in enumerate(items, start=1):
        panel_id = f"panel_{idx:06d}"
        scene_file = f"{panel_id}.jpg"
        scene_path = os.path.join(out_dir, scene_file)

        try:
            with Image.open(it.chunk_path) as im:
                cw, chh = im.size

                # convert box
                if args.min_area_frac and area_norm(it.bbox_norm) < float(args.min_area_frac):
                    skipped_small += 1
                    scenes.append(
                        {
                            "panel_id": panel_id,
                            "dropped": True,
                            "drop_reason": "too_small_area",
                            "chunk_file": it.chunk_file,
                            "chunk_index": it.chunk_index,
                            "chunk_path": it.chunk_path,
                            "bbox_norm": it.bbox_norm,
                        }
                    )
                    continue

                box_xyxy = norm_to_px(it.bbox_norm, cw, chh)
                if args.pad_px and int(args.pad_px) > 0:
                    box_xyxy = pad_box_xyxy(box_xyxy, cw, chh, int(args.pad_px))

                crop = im.crop(box_xyxy).convert("RGB")

                # optional blank skip
                bs = blank_score(crop)
                if args.skip_blank and bs >= float(args.blank_threshold):
                    skipped_blank += 1
                    scenes.append(
                        {
                            "panel_id": panel_id,
                            "dropped": True,
                            "drop_reason": "blank_after_crop",
                            "blank_score": round(bs, 6),
                            "chunk_file": it.chunk_file,
                            "chunk_index": it.chunk_index,
                            "chunk_path": it.chunk_path,
                            "bbox_norm": it.bbox_norm,
                            "bbox_px_xyxy": box_xyxy,
                            "scene_file": scene_file,
                            "scene_path": scene_path,
                            "width": crop.width,
                            "height": crop.height,
                        }
                    )
                    continue

                # optional dedupe (good for overlapped chunks)
                ph = dhash64(crop)
                is_dupe = False
                if args.dedupe and seen_hashes:
                    thr = int(args.dedupe_threshold)
                    for prev in seen_hashes[-250:]:  # keep compare bounded
                        if hamming64(ph, prev) <= thr:
                            is_dupe = True
                            break

                if is_dupe:
                    skipped_dupe += 1
                    scenes.append(
                        {
                            "panel_id": panel_id,
                            "dropped": True,
                            "drop_reason": "dedupe",
                            "dhash64": int(ph),
                            "chunk_file": it.chunk_file,
                            "chunk_index": it.chunk_index,
                            "chunk_path": it.chunk_path,
                            "bbox_norm": it.bbox_norm,
                            "bbox_px_xyxy": box_xyxy,
                            "scene_file": scene_file,
                            "scene_path": scene_path,
                            "width": crop.width,
                            "height": crop.height,
                        }
                    )
                    continue

                crop.save(scene_path, "JPEG", quality=int(args.jpeg_quality))
                written += 1
                if args.dedupe:
                    seen_hashes.append(ph)

                scenes.append(
                    {
                        "panel_id": panel_id,
                        "dropped": False,
                        "chunk_file": it.chunk_file,
                        "chunk_index": it.chunk_index,
                        "chunk_path": it.chunk_path,
                        "panel_idx_in_chunk": it.panel_idx_in_chunk,
                        "bbox_norm": it.bbox_norm,            # [y0,x0,y1,x1]
                        "bbox_px_xyxy": box_xyxy,             # [x0,y0,x1,y1]
                        "scene_file": scene_file,
                        "scene_path": scene_path,
                        "width": crop.width,
                        "height": crop.height,
                        "blank_score": round(bs, 6),
                        "dhash64": int(ph),
                    }
                )

        except Exception as e:
            errors += 1
            scenes.append(
                {
                    "panel_id": panel_id,
                    "dropped": True,
                    "drop_reason": "exception",
                    "error": repr(e),
                    "chunk_file": it.chunk_file,
                    "chunk_index": it.chunk_index,
                    "chunk_path": it.chunk_path,
                    "bbox_norm": it.bbox_norm,
                }
            )

    out = {
        "source_stitch_manifest": os.path.abspath(args.stitch_manifest),
        "source_panels_manifest": os.path.abspath(args.panels_manifest),
        "out_dir": out_dir,
        "params": {
            "jpeg_quality": int(args.jpeg_quality),
            "pad_px": int(args.pad_px),
            "min_area_frac": float(args.min_area_frac),
            "dedupe": bool(args.dedupe),
            "dedupe_threshold": int(args.dedupe_threshold),
            "skip_blank": bool(args.skip_blank),
            "blank_threshold": float(args.blank_threshold),
        },
        "stats": {
            "total_panels_seen": len(items),
            "written": written,
            "skipped_dupe": skipped_dupe,
            "skipped_blank": skipped_blank,
            "skipped_small": skipped_small,
            "errors": errors,
        },
        "scenes": scenes,
    }

    dump_json(args.out_manifest, out)
    print(
        f"[ok] wrote scenes={written} "
        f"(dupe={skipped_dupe}, blank={skipped_blank}, small={skipped_small}, err={errors}) "
        f"manifest={os.path.abspath(args.out_manifest)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
