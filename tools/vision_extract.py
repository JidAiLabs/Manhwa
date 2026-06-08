#!/usr/bin/env python3
"""
vision_extract.py

Extracts Google Cloud Vision signals per scene image and writes manifest.vision.json.

Adds:
- ocr_clean (UI stripped)
- text_coverage (0..1, approx)
- text_only (heuristic; tries to avoid phone-in-hand false positives)
- keywords (simple)
- targets: a compact set of camera targets for later timeline planning
  - wide
  - text_blocks (merged line-like boxes)
  - objects (with boxes)
  - faces (optional; enabled below)
- NEW: vision.ocr_words: list of {"t": "<word>", "bbox":[x0,y0,x1,y1]} normalized
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

from google.cloud import vision  # pip install google-cloud-vision
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Helpers
# -----------------------------
def natural_key(path: str):
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    return (int(nums[0]) if nums else 0, base)


def parse_scene_id(filename: str) -> int:
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else 0


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def rect_from_vertices_px(vertices) -> Tuple[int, int, int, int]:
    vs = list(vertices or [])
    if not vs:
        return (0, 0, 0, 0)
    xs = [int(getattr(v, "x", 0) or 0) for v in vs]
    ys = [int(getattr(v, "y", 0) or 0) for v in vs]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (x0, y0, x1, y1)


def rect_from_norm_vertices(nv) -> Tuple[float, float, float, float]:
    vs = list(nv or [])
    if not vs:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [float(getattr(v, "x", 0.0) or 0.0) for v in vs]
    ys = [float(getattr(v, "y", 0.0) or 0.0) for v in vs]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (clamp01(x0), clamp01(y0), clamp01(x1), clamp01(y1))


def rect_px_to_norm(r: Tuple[int, int, int, int], w: int, h: int) -> Tuple[float, float, float, float]:
    if w <= 0 or h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    x0, y0, x1, y1 = r
    return (clamp01(x0 / w), clamp01(y0 / h), clamp01(x1 / w), clamp01(y1 / h))


def rect_area_norm(r: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = r
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def load_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# -----------------------------
# Config
# -----------------------------
@dataclass
class VisionConfig:
    max_text_chars: int = 1200
    max_labels: int = 15
    max_objects: int = 10
    max_faces: int = 6

    # OCR cleaning + heuristics
    max_keyword_count: int = 8
    text_only_min_coverage: float = 0.10  # safer default for “text card / bubble dominant”
    text_block_merge_y_px: int = 18       # how aggressively we merge OCR words into “line blocks”
    max_text_blocks: int = 12             # keep manifest compact

    # NEW: ocr_words cap
    max_ocr_words: int = 500

    # UI stripping
    ui_noise_patterns: Tuple[str, ...] = (
        r"\bLTE\b", r"\b5G\b", r"\bWi-?Fi\b", r"\bPM\b", r"\bAM\b",
        r"\bNEXT\b", r"\bPREVIOUS\b", r"\bMENU\b",
        r"\b\d{1,3}%\b",              # battery percent
        r"\b\d{1,2}:\d{2}\b",          # time like 6:50
        r"\b\d+/\d+\b",                # page indicator like 27/27
    )


# -----------------------------
# OCR cleaning / keywords
# -----------------------------
def clean_ocr_text(raw: str, cfg: VisionConfig) -> str:
    if not raw:
        return ""
    lines: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue

        noisy = False
        for pat in cfg.ui_noise_patterns:
            if re.search(pat, s, flags=re.IGNORECASE):
                noisy = True
                break
        if noisy:
            continue

        lines.append(s)

    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keywords_from_text(text: str, cfg: VisionConfig) -> List[str]:
    if not text:
        return []
    stop = {
        "the","and","to","of","is","are","this","that","i","you","we","a","an","in","on",
        "it","as","for","with","be","was","were","at","by","from","or","but","not",
        "my","your","our","their","his","her","they","them","me","he","she","him"
    }
    toks = re.findall(r"[A-Za-z']{2,}", text)
    toks = [t.lower() for t in toks if t.lower() not in stop]
    seen = set()
    out: List[str] = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= cfg.max_keyword_count:
            break
    return out


# -----------------------------
# Text coverage + blocks
# -----------------------------
def bbox_area_px(bounding_poly) -> int:
    r = rect_from_vertices_px(getattr(bounding_poly, "vertices", None))
    x0, y0, x1, y1 = r
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def compute_text_coverage_from_text_annotations(resp, img_w: int, img_h: int) -> float:
    """
    Approx: sum bbox areas of word-level annotations / image area.
    text_annotations[0] is full text; [1:] are smaller (usually words).
    Overcounts overlaps; good enough for heuristics.
    """
    if img_w <= 0 or img_h <= 0:
        return 0.0
    anns = list(getattr(resp, "text_annotations", []) or [])
    if len(anns) <= 1:
        return 0.0

    total_area = 0
    for ann in anns[1:]:
        bp = getattr(ann, "bounding_poly", None)
        if bp:
            total_area += bbox_area_px(bp)

    cov = total_area / float(img_w * img_h)
    return float(clamp01(cov))


def extract_word_boxes(resp, w: int, h: int) -> List[Tuple[float, float, float, float]]:
    anns = list(getattr(resp, "text_annotations", []) or [])
    if len(anns) <= 1:
        return []
    out: List[Tuple[float, float, float, float]] = []
    for ann in anns[1:]:
        bp = getattr(ann, "bounding_poly", None)
        if not bp:
            continue
        r_px = rect_from_vertices_px(getattr(bp, "vertices", None))
        r_n = rect_px_to_norm(r_px, w, h)
        if rect_area_norm(r_n) <= 0.00001:
            continue
        out.append(r_n)
    return out


def extract_ocr_words(resp, w: int, h: int, max_words: int = 500) -> List[Dict[str, Any]]:
    """
    Word-level OCR entries: [{"t": "YOU", "bbox":[x0,y0,x1,y1]}, ...] in normalized coords.
    """
    anns = list(getattr(resp, "text_annotations", []) or [])
    if len(anns) <= 1:
        return []
    out: List[Dict[str, Any]] = []
    for ann in anns[1:]:
        txt = (getattr(ann, "description", "") or "").strip()
        if not txt:
            continue
        bp = getattr(ann, "bounding_poly", None)
        if not bp:
            continue
        r_px = rect_from_vertices_px(getattr(bp, "vertices", None))
        r_n = rect_px_to_norm(r_px, w, h)
        if rect_area_norm(r_n) <= 0.00001:
            continue
        out.append({"t": txt, "bbox": [round(float(x), 4) for x in r_n]})
        if len(out) >= max_words:
            break
    return out


def merge_words_into_text_blocks(
    word_boxes: List[Tuple[float, float, float, float]],
    w: int,
    h: int,
    merge_y_px: int,
    max_blocks: int,
) -> List[Tuple[float, float, float, float]]:
    """
    Merge OCR word rectangles into line-like blocks by grouping words with similar y-center.
    Keeps blocks compact for later “zoom text block” targets.
    """
    if not word_boxes:
        return []

    words_px: List[Tuple[int, int, int, int]] = []
    for (x0, y0, x1, y1) in word_boxes:
        words_px.append((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))

    words_px.sort(key=lambda r: ((r[1] + r[3]) // 2, r[0]))

    lines: List[List[Tuple[int, int, int, int]]] = []
    for r in words_px:
        yc = (r[1] + r[3]) // 2
        placed = False
        for line in lines:
            r0 = line[0]
            yc0 = (r0[1] + r0[3]) // 2
            if abs(yc - yc0) <= merge_y_px:
                line.append(r)
                placed = True
                break
        if not placed:
            lines.append([r])

    blocks_px: List[Tuple[int, int, int, int]] = []
    for line in lines:
        x0 = min(r[0] for r in line)
        y0 = min(r[1] for r in line)
        x1 = max(r[2] for r in line)
        y1 = max(r[3] for r in line)
        if x1 > x0 and y1 > y0:
            blocks_px.append((x0, y0, x1, y1))

    blocks_px.sort(key=lambda r: r[1])
    merged: List[Tuple[int, int, int, int]] = []
    for r in blocks_px:
        if not merged:
            merged.append(r)
            continue
        p = merged[-1]
        gap = r[1] - p[3]
        if gap <= merge_y_px:
            overlap = max(0, min(p[2], r[2]) - max(p[0], r[0]))
            width_union = max(p[2], r[2]) - min(p[0], r[0])
            ov_ratio = (overlap / width_union) if width_union > 0 else 0.0
            if ov_ratio >= 0.35:
                merged[-1] = (min(p[0], r[0]), min(p[1], r[1]), max(p[2], r[2]), max(p[3], r[3]))
                continue
        merged.append(r)

    merged_norm = [rect_px_to_norm(r, w, h) for r in merged]
    merged_norm.sort(key=rect_area_norm, reverse=True)
    return merged_norm[:max_blocks]


# -----------------------------
# Targets (camera anchors)
# -----------------------------
def make_targets(
    text_blocks: List[Tuple[float, float, float, float]],
    objects: List[Dict[str, Any]],
    faces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    targets.append({"id": "wide", "type": "frame", "bbox": [0.0, 0.0, 1.0, 1.0]})

    for i, b in enumerate(text_blocks, 1):
        targets.append({"id": f"text_{i}", "type": "text_block", "bbox": [round(x, 4) for x in b]})

    for i, o in enumerate(objects, 1):
        bb = o.get("bbox")
        if bb and len(bb) == 4 and rect_area_norm(tuple(bb)) > 0.00001:
            targets.append({"id": f"obj_{i}", "type": "object", "name": o.get("name", ""), "bbox": [round(float(x), 4) for x in bb]})

    for i, f in enumerate(faces, 1):
        bb = f.get("bbox")
        if bb and len(bb) == 4 and rect_area_norm(tuple(bb)) > 0.00001:
            targets.append({"id": f"face_{i}", "type": "face", "bbox": [round(float(x), 4) for x in bb]})

    return targets


# -----------------------------
# Text-only heuristic (safer)
# -----------------------------
_STRONG_VISUAL_TERMS = {
    "person", "people", "man", "woman", "face", "head", "hand", "finger",
    "phone", "mobile phone", "cell phone", "smartphone", "screen",
    "monitor", "display", "computer", "laptop",
}

def has_strong_visual(labels: List[str], objects: List[str]) -> bool:
    s = " ".join((labels or []) + (objects or [])).lower()
    return any(term in s for term in _STRONG_VISUAL_TERMS)

def classify_text_only(ocr_clean: str, text_cov: float, labels: List[str], objects: List[str], cfg: VisionConfig) -> bool:
    if not ocr_clean or len(ocr_clean) < 10:
        return False
    if has_strong_visual(labels, objects):
        return False
    if text_cov >= max(0.20, cfg.text_only_min_coverage):
        return True
    l0 = " ".join(labels[:5]).lower()
    if ("text" in l0 or "font" in l0) and text_cov >= cfg.text_only_min_coverage:
        return True
    return False


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-dir", required=True, help="Directory containing scene_*.jpg")
    ap.add_argument("--glob", default="scene_*.jpg")
    ap.add_argument("--out", default="manifest.vision.json")

    ap.add_argument("--max-text-chars", type=int, default=1200)
    ap.add_argument("--max-labels", type=int, default=15)
    ap.add_argument("--max-objects", type=int, default=10)
    ap.add_argument("--max-faces", type=int, default=6)

    ap.add_argument("--text-only-min-coverage", type=float, default=0.10)
    ap.add_argument("--max-keywords", type=int, default=8)
    ap.add_argument("--max-text-blocks", type=int, default=12)
    ap.add_argument("--merge-y-px", type=int, default=18)
    ap.add_argument("--max-ocr-words", type=int, default=500)

    args = ap.parse_args()

    cfg = VisionConfig(
        max_text_chars=args.max_text_chars,
        max_labels=args.max_labels,
        max_objects=args.max_objects,
        max_faces=args.max_faces,
        text_only_min_coverage=args.text_only_min_coverage,
        max_keyword_count=args.max_keywords,
        max_text_blocks=args.max_text_blocks,
        text_block_merge_y_px=args.merge_y_px,
        max_ocr_words=args.max_ocr_words,
    )

    paths = sorted(glob.glob(os.path.join(args.scenes_dir, args.glob)), key=natural_key)
    if not paths:
        raise SystemExit(f"No images found in {args.scenes_dir} with glob={args.glob}")

    client = vision.ImageAnnotatorClient()

    items: List[Dict[str, Any]] = []
    errors = 0

    for p in paths:
        scene_file = os.path.basename(p)
        scene_id = parse_scene_id(scene_file)

        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im2:
                w, h = im2.size

            img_bytes = load_bytes(p)
            image = vision.Image(content=img_bytes)

            resp = client.annotate_image(
                {
                    "image": image,
                    "features": [
                        {"type_": vision.Feature.Type.TEXT_DETECTION},
                        {"type_": vision.Feature.Type.LABEL_DETECTION, "max_results": cfg.max_labels},
                        {"type_": vision.Feature.Type.OBJECT_LOCALIZATION, "max_results": cfg.max_objects},
                        {"type_": vision.Feature.Type.FACE_DETECTION, "max_results": cfg.max_faces},
                    ],
                }
            )

            if resp.error and resp.error.message:
                errors += 1
                items.append(
                    {
                        "scene_id": scene_id,
                        "scene_file": scene_file,
                        "scene_path": os.path.abspath(p),
                        "width": w,
                        "height": h,
                        "ocr_clean": "",
                        "text_coverage": 0.0,
                        "text_only": False,
                        "keywords": [],
                        "targets": [{"id": "wide", "type": "frame", "bbox": [0.0, 0.0, 1.0, 1.0]}],
                        "vision": {"error": resp.error.message},
                    }
                )
                continue

            full_text = ""
            if resp.text_annotations:
                full_text = resp.text_annotations[0].description or ""
            full_text = full_text.strip()
            if len(full_text) > cfg.max_text_chars:
                full_text = full_text[: cfg.max_text_chars] + "…"

            labels = [{"desc": l.description, "score": float(l.score)} for l in (resp.label_annotations or [])]

            objects_raw = list(resp.localized_object_annotations or [])
            objects: List[Dict[str, Any]] = []
            for o in objects_raw:
                bb = rect_from_norm_vertices(
                    getattr(o, "bounding_poly", None).normalized_vertices
                    if getattr(o, "bounding_poly", None) else None
                )
                objects.append(
                    {
                        "name": o.name,
                        "score": float(o.score),
                        "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                    }
                )

            faces: List[Dict[str, Any]] = []
            for fa in list(resp.face_annotations or [])[: cfg.max_faces]:
                bp = getattr(fa, "bounding_poly", None)
                r_px = rect_from_vertices_px(getattr(bp, "vertices", None) if bp else None)
                r_n = rect_px_to_norm(r_px, w, h)
                faces.append(
                    {
                        "bbox": [float(r_n[0]), float(r_n[1]), float(r_n[2]), float(r_n[3])],
                        "confidence": float(getattr(fa, "detection_confidence", 0.0) or 0.0),
                    }
                )

            ocr_clean = clean_ocr_text(full_text, cfg)
            text_cov = compute_text_coverage_from_text_annotations(resp, w, h)
            keywords = keywords_from_text(ocr_clean, cfg)

            label_names = [x["desc"] for x in labels]
            object_names = [x["name"] for x in objects]

            text_only = classify_text_only(ocr_clean, float(text_cov), label_names, object_names, cfg)

            word_boxes = extract_word_boxes(resp, w, h)
            text_blocks = merge_words_into_text_blocks(
                word_boxes=word_boxes,
                w=w,
                h=h,
                merge_y_px=cfg.text_block_merge_y_px,
                max_blocks=cfg.max_text_blocks,
            )

            targets = make_targets(text_blocks=text_blocks, objects=objects, faces=faces)

            # NEW: ocr_words
            ocr_words = extract_ocr_words(resp, w, h, max_words=cfg.max_ocr_words)

            items.append(
                {
                    "scene_id": scene_id,
                    "scene_file": scene_file,
                    "scene_path": os.path.abspath(p),
                    "width": w,
                    "height": h,
                    "ocr_clean": ocr_clean,
                    "text_coverage": round(float(text_cov), 4),
                    "text_only": bool(text_only),
                    "keywords": keywords,
                    "targets": targets,
                    "vision": {
                        "text": full_text,
                        "labels": labels,
                        "objects": objects,
                        "faces": faces,
                        "text_blocks": [[round(x, 4) for x in b] for b in text_blocks],
                        "ocr_words": ocr_words,
                    },
                }
            )

        except Exception as e:
            errors += 1
            items.append(
                {
                    "scene_id": scene_id,
                    "scene_file": scene_file,
                    "scene_path": os.path.abspath(p),
                    "ocr_clean": "",
                    "text_coverage": 0.0,
                    "text_only": False,
                    "keywords": [],
                    "targets": [{"id": "wide", "type": "frame", "bbox": [0.0, 0.0, 1.0, 1.0]}],
                    "vision": {"error": repr(e)},
                }
            )

    out_obj = {
        "scenes_dir": os.path.abspath(args.scenes_dir),
        "config": asdict(cfg),
        "count": len(items),
        "errors": errors,
        "items": items,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.scenes_dir)), args.out) \
        if not os.path.isabs(args.out) else args.out

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote={out_path} scenes={len(items)} errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
