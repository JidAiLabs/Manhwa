#!/usr/bin/env python3
# vision_anchors.py
#
# Extract semantic anchors (text / faces / objects) from a chunk image using
# Google Cloud Vision API. Outputs JSON with pixel bounding boxes.

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image


def _bbox_from_vertices_px(vertices: List[Dict[str, int]], w: int, h: int) -> Optional[List[int]]:
    """
    vertices: list of {x,y} in px (Vision returns ints, sometimes missing)
    returns [x0,y0,x1,y1] inclusive/exclusive-ish (we’ll treat as pixel bounds)
    """
    xs = []
    ys = []
    for v in vertices:
        x = int(v.get("x", 0) or 0)
        y = int(v.get("y", 0) or 0)
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(w, max(xs))
    y1 = min(h, max(ys))
    # enforce sane min size
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return [x0, y0, x1, y1]


def _bbox_from_normalized_vertices(nv_list: List[Any], w: int, h: int) -> Optional[List[int]]:
    """
    For object localization: normalized vertices in [0..1]
    """
    xs = []
    ys = []
    for nv in nv_list:
        x = float(getattr(nv, "x", 0.0) or 0.0)
        y = float(getattr(nv, "y", 0.0) or 0.0)
        xs.append(int(round(x * w)))
        ys.append(int(round(y * h)))
    if not xs or not ys:
        return None
    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(w, max(xs))
    y1 = min(h, max(ys))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return [x0, y0, x1, y1]


def _area(b: List[int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input chunk image path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output anchors.json path")

    ap.add_argument("--include-faces", action="store_true", default=True)
    ap.add_argument("--include-objects", action="store_true", default=False)

    # Text filtering
    ap.add_argument("--min-text-chars", type=int, default=2)
    ap.add_argument("--min-text-area", type=int, default=250)  # px^2

    # Face filtering
    ap.add_argument("--min-face-area", type=int, default=900)  # px^2

    # Object filtering
    ap.add_argument("--min-object-score", type=float, default=0.45)
    ap.add_argument("--min-object-area", type=int, default=1200)  # px^2
    ap.add_argument("--object-allowlist", type=str, default="person,human,face,head,man,woman,boy,girl",
                    help="comma-separated lowercase names; only these objects kept (if objects enabled)")

    args = ap.parse_args()

    in_path = os.path.abspath(args.in_path)
    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # get dimensions
    with Image.open(in_path) as im:
        w, h = im.size

    # Vision client
    from google.cloud import vision  # local import to avoid dependency if not used

    client = vision.ImageAnnotatorClient()
    content = _load_image_bytes(in_path)
    image = vision.Image(content=content)

    anchors: Dict[str, Any] = {
        "source_image": in_path,
        "image_w": w,
        "image_h": h,
        "text_boxes": [],
        "face_boxes": [],
        "object_boxes": [],
        "meta": {
            "min_text_chars": int(args.min_text_chars),
            "min_text_area": int(args.min_text_area),
            "min_face_area": int(args.min_face_area),
            "min_object_score": float(args.min_object_score),
            "min_object_area": int(args.min_object_area),
            "include_objects": bool(args.include_objects),
        },
    }

    # --- TEXT DETECTION ---
    # Use document_text_detection (better for dense comic text)
    text_resp = client.document_text_detection(image=image)
    if text_resp.error.message:
        raise RuntimeError(text_resp.error.message)

    # Vision returns many levels (pages/blocks/paragraphs/words/symbols).
    # We’ll anchor at the BLOCK level to cover speech bubbles / narration boxes robustly.
    dt = text_resp.full_text_annotation
    if dt and dt.pages:
        for page in dt.pages:
            for block in page.blocks:
                # aggregate block bbox
                bb = block.bounding_box
                v = [{"x": p.x, "y": p.y} for p in bb.vertices]
                box = _bbox_from_vertices_px(v, w, h)
                if not box:
                    continue
                if _area(box) < int(args.min_text_area):
                    continue

                # estimate block text length (sum symbols)
                char_count = 0
                for para in block.paragraphs:
                    for word in para.words:
                        char_count += len(word.symbols)

                if char_count < int(args.min_text_chars):
                    continue

                anchors["text_boxes"].append({
                    "bbox_px": box,
                    "char_count": int(char_count),
                })

    # --- FACE DETECTION ---
    if args.include_faces:
        face_resp = client.face_detection(image=image)
        if face_resp.error.message:
            raise RuntimeError(face_resp.error.message)
        for face in face_resp.face_annotations:
            v = [{"x": p.x, "y": p.y} for p in face.bounding_poly.vertices]
            box = _bbox_from_vertices_px(v, w, h)
            if not box:
                continue
            if _area(box) < int(args.min_face_area):
                continue
            anchors["face_boxes"].append({
                "bbox_px": box,
            })

    # --- OBJECT LOCALIZATION (optional) ---
    if args.include_objects:
        allow = set([s.strip().lower() for s in str(args.object_allowlist).split(",") if s.strip()])
        obj_resp = client.object_localization(image=image)
        if obj_resp.error.message:
            raise RuntimeError(obj_resp.error.message)

        for obj in obj_resp.localized_object_annotations:
            name = (obj.name or "").strip().lower()
            score = float(obj.score or 0.0)
            if score < float(args.min_object_score):
                continue
            if allow and name not in allow:
                continue

            box = _bbox_from_normalized_vertices(obj.bounding_poly.normalized_vertices, w, h)
            if not box:
                continue
            if _area(box) < int(args.min_object_area):
                continue

            anchors["object_boxes"].append({
                "name": obj.name,
                "score": round(score, 4),
                "bbox_px": box,
            })

    # sort for stability
    anchors["text_boxes"].sort(key=lambda d: (d["bbox_px"][1], d["bbox_px"][0], -(d["bbox_px"][3]-d["bbox_px"][1])))
    anchors["face_boxes"].sort(key=lambda d: (d["bbox_px"][1], d["bbox_px"][0]))
    anchors["object_boxes"].sort(key=lambda d: (d["bbox_px"][1], d["bbox_px"][0]))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)

    print(f"[ok] anchors written: {out_path} | text={len(anchors['text_boxes'])} faces={len(anchors['face_boxes'])} objs={len(anchors['object_boxes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
