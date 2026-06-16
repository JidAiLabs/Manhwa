from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json

import cv2
import numpy as np

from manhwa_cropper.detectors.bubbles import BubbleDetector
from manhwa_cropper.detectors.split_gutters import propose_scene_boxes_from_gutters
from manhwa_cropper.postprocess.smart_trim import smart_trim_with_edge_guard
from manhwa_cropper.export.writer import write_scenes

Box = Tuple[int, int, int, int]                 # x1,y1,x2,y2
DetBox = Tuple[float, float, float, float, float]  # x1,y1,x2,y2,score


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _clip_box_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Box:
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _edge_density(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(gray, 40, 120)
    return float((e > 0).mean())


def _std_gray(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def _detect_caption_text_boxes(img_bgr: np.ndarray) -> List[Box]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bright = (gray > 210).astype(np.uint8) * 255
    dark = (gray < 45).astype(np.uint8) * 255

    k = max(3, int(min(w, h) * 0.015))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, max(3, k // 2)))
    bright2 = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)
    dark2 = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.bitwise_or(bright2, dark2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Box] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < (w * h) * 0.001:
            continue
        if hh < 18 or ww < 40:
            continue
        ar = ww / max(1.0, hh)
        if ar < 0.4:
            continue
        boxes.append(_clip_box_xyxy(x, y, x + ww, y + hh, w, h))

    if not boxes:
        return []

    tmp = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        tmp[y1:y2, x1:x2] = 255

    merge_k = max(5, int(min(w, h) * 0.02))
    tmp = cv2.dilate(tmp, cv2.getStructuringElement(cv2.MORPH_RECT, (merge_k, merge_k // 2)), iterations=1)
    contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    merged: List[Box] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < (w * h) * 0.002:
            continue
        merged.append(_clip_box_xyxy(x, y, x + ww, y + hh, w, h))

    return merged


def _detect_visual_anchor_boxes(img_bgr: np.ndarray) -> List[Box]:
    """
    Key fix for "missing face": detect non-text, non-bubble "visual content" areas
    even on smooth glow backgrounds. These become anchors so trimming won't delete them.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    g = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g, 30, 110)

    # connect edges into blobs
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Box] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < (w * h) * 0.01:   # ignore tiny bits
            continue
        if hh < 80 and ww < 80:
            continue
        boxes.append(_clip_box_xyxy(x, y, x + ww, y + hh, w, h))

    # keep top 2 largest
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[:2]


def _union_boxes(boxes: List[Box], w: int, h: int, pad: int) -> Box:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return _clip_box_xyxy(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)


def _has_content_outside_union(img_bgr: np.ndarray, union: Box) -> bool:
    """
    Prevent the bad case: anchor-trim around text only while a smooth character exists elsewhere.
    We check edge density outside the union; if it's non-trivial, DO NOT anchor-trim.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = union

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(gray, 30, 110)

    outside = np.ones((h, w), dtype=np.uint8) * 255
    outside[y1:y2, x1:x2] = 0

    out_edges = e[outside > 0]
    if out_edges.size == 0:
        return False

    frac = float((out_edges > 0).mean())
    return frac >= 0.0045


def _anchor_trim_if_blank(img_bgr: np.ndarray, anchors: List[Box], pad: int = 80) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Only do anchor-trim if:
      - looks blank-ish AND
      - there is NOT meaningful content outside anchor union
    This avoids deleting faces/characters.
    """
    h, w = img_bgr.shape[:2]
    ed = _edge_density(img_bgr)

    if anchors and ed < 0.012:
        u = _union_boxes(anchors, w, h, pad=pad)
        if _has_content_outside_union(img_bgr, u):
            return img_bgr, {"anchor_trimmed": False, "edge_density": ed, "anchor_trim_blocked": True}

        ax1, ay1, ax2, ay2 = u
        cropped = img_bgr[ay1:ay2, ax1:ax2].copy()
        return cropped, {
            "anchor_trimmed": True,
            "anchor_trim_blocked": False,
            "anchor_box_xyxy": [ax1, ay1, ax2, ay2],
            "edge_density": ed,
        }

    return img_bgr, {"anchor_trimmed": False, "edge_density": ed}


def _drop_meaningless(img_bgr: np.ndarray, anchors: List[Box]) -> bool:
    if anchors:
        return False
    ed = _edge_density(img_bgr)
    sd = _std_gray(img_bgr)
    return (ed < 0.0035 and sd < 10.0)


def _merge_tiny_tails(
    crops: List[np.ndarray],
    meta: List[Dict[str, Any]],
    min_tail_h: int,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    if len(crops) <= 1:
        return crops, meta

    out_crops: List[np.ndarray] = []
    out_meta: List[Dict[str, Any]] = []

    i = 0
    while i < len(crops):
        img = crops[i]
        h = img.shape[0]

        if h < min_tail_h:
            if i + 1 < len(crops):
                nxt = crops[i + 1]
                merged = np.vstack([img, nxt])

                m = dict(meta[i + 1])
                m["merged_from"] = [meta[i].get("scene_index"), meta[i + 1].get("scene_index")]

                anchors_a = meta[i].get("anchors_xyxy", [])
                anchors_b = meta[i + 1].get("anchors_xyxy", [])
                shifted_b = [[x1, y1 + h, x2, y2 + h] for (x1, y1, x2, y2) in anchors_b]
                m["anchors_xyxy"] = [*anchors_a, *shifted_b]

                out_crops.append(merged)
                out_meta.append(m)
                i += 2
                continue

            if out_crops:
                prev_h = out_crops[-1].shape[0]
                out_crops[-1] = np.vstack([out_crops[-1], img])

                anchors = out_meta[-1].get("anchors_xyxy", [])
                anchors_this = meta[i].get("anchors_xyxy", [])
                shifted = [[x1, y1 + prev_h, x2, y2 + prev_h] for (x1, y1, x2, y2) in anchors_this]
                out_meta[-1]["anchors_xyxy"] = [*anchors, *shifted]
                out_meta[-1]["merged_from"] = (out_meta[-1].get("merged_from", []) or []) + [meta[i].get("scene_index")]
                i += 1
                continue

        out_crops.append(img)
        out_meta.append(meta[i])
        i += 1

    return out_crops, out_meta


def _split_scene_by_textbands(
    scene_xyxy: Box,
    text_boxes_global: List[Box],
    *,
    min_part_h: int,
    pad: int = 24,
) -> List[Box]:
    """
    If there's a wide-ish text band near the bottom/top of a scene, split it out as its own shot.
    This targets "HOWEVER..." and narration cards.
    """
    x1, y1, x2, y2 = scene_xyxy
    sh = y2 - y1
    if sh < (min_part_h * 2):
        return [scene_xyxy]

    # collect text boxes inside this scene
    inside: List[Box] = []
    for tb in text_boxes_global:
        tx1, ty1, tx2, ty2 = tb
        if tx2 <= x1 or tx1 >= x2 or ty2 <= y1 or ty1 >= y2:
            continue
        # clamp to scene
        cx1 = max(x1, tx1)
        cy1 = max(y1, ty1)
        cx2 = min(x2, tx2)
        cy2 = min(y2, ty2)
        inside.append((cx1, cy1, cx2, cy2))

    if not inside:
        return [scene_xyxy]

    # choose candidate "band" boxes: wide and not too tall
    candidates: List[Box] = []
    for (tx1, ty1, tx2, ty2) in inside:
        tw = tx2 - tx1
        th = ty2 - ty1
        if tw >= 0.55 * (x2 - x1) and th <= 0.30 * sh:
            candidates.append((tx1, ty1, tx2, ty2))

    if not candidates:
        return [scene_xyxy]

    # pick the lowest band (most common for "HOWEVER..." at bottom)
    candidates.sort(key=lambda b: b[1])
    band = candidates[-1]
    bx1, by1, bx2, by2 = band

    # if band is near bottom portion, split: upper scene + text scene
    if by1 >= y1 + int(0.55 * sh):
        cut = max(y1 + min_part_h, by1 - pad)
        upper = (x1, y1, x2, cut)
        lower = (x1, cut, x2, y2)
        if (upper[3] - upper[1]) >= min_part_h and (lower[3] - lower[1]) >= min_part_h:
            return [upper, lower]

    # if band near top, split: text scene + rest
    if by2 <= y1 + int(0.45 * sh):
        cut = min(y2 - min_part_h, by2 + pad)
        top = (x1, y1, x2, cut)
        bottom = (x1, cut, x2, y2)
        if (top[3] - top[1]) >= min_part_h and (bottom[3] - bottom[1]) >= min_part_h:
            return [top, bottom]

    return [scene_xyxy]


def _load_vision_text_boxes(vision_manifest: Path, scene_file: str, w: int, h: int) -> List[Box]:
    """
    manifest.vision.json items contain normalized boxes. We convert to absolute xyxy.
    Expected structure based on your file:
      data["items"][i]["scene_file"] == "chunk_0004.jpg"
      data["items"][i]["vision"]["text_blocks"] = [{"bbox":[x1,y1,x2,y2], ...}, ...]  (normalized)
    """
    try:
        data = json.loads(vision_manifest.read_text(encoding="utf-8"))
    except Exception:
        return []

    item = None
    for it in data.get("items", []):
        if it.get("scene_file") == scene_file:
            item = it
            break
    if not item:
        return []

    out: List[Box] = []
    for tb in item.get("vision", {}).get("text_blocks", []) or []:
        bb = tb.get("bbox")
        if not bb or len(bb) != 4:
            continue
        x1n, yn1, x2n, y2n = bb
        x1 = int(round(x1n * w))
        y1 = int(round(yn1 * h))
        x2 = int(round(x2n * w))
        y2 = int(round(y2n * h))
        out.append(_clip_box_xyxy(x1, y1, x2, y2, w, h))
    return out


def crop_page_to_scenes(
    image_path: Path,
    out_dir: Path,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.5,
    device: str = "cpu",
    min_scene_h: int = 220,
    min_gutter_h: int = 18,
    max_scenes: int = 120,
    enable_trim: bool = True,
    write_json: bool = False,
    vision_manifest: Optional[Path] = None,   # <-- optional
):
    img = _read_image(image_path)
    H, W = img.shape[:2]

    bubble_det = BubbleDetector(device=device)
    bubbles = bubble_det.detect(img, imgsz=imgsz, conf=conf, iou=iou)
    page_captions = _detect_caption_text_boxes(img)
    caption_as_det = [(float(x1), float(y1), float(x2), float(y2), 1.0) for (x1, y1, x2, y2) in page_captions]
    bubbles_plus_text = [*bubbles, *caption_as_det]

    # optional: vision API text blocks (global coordinates)
    vision_text_boxes: List[Box] = []
    if vision_manifest is not None and vision_manifest.exists():
        # your manifest uses original chunk file names like chunk_0004.jpg
        vision_text_boxes = _load_vision_text_boxes(vision_manifest, image_path.name, W, H)

    scene_boxes0 = propose_scene_boxes_from_gutters(
        img=img,
        bubble_boxes=bubbles_plus_text,
        min_scene_h=min_scene_h,
        min_gutter_h=min_gutter_h,
        max_scenes=max_scenes,
    )

    # Split scenes by text bands (“HOWEVER…”) using Vision text boxes when present,
    # else fallback heuristic detection later at crop-level.
    scene_boxes: List[Box] = []
    for (x1, y1, x2, y2) in scene_boxes0:
        b = _clip_box_xyxy(int(x1), int(y1), int(x2), int(y2), W, H)
        parts = _split_scene_by_textbands(b, vision_text_boxes, min_part_h=max(160, min_scene_h // 2))
        scene_boxes.extend(parts)

    # Crop scenes
    crops: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    for idx, (x1, y1, x2, y2) in enumerate(scene_boxes):
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2].copy()

        local_bubbles: List[DetBox] = []
        bubble_anchors: List[Box] = []
        for (bx1, by1, bx2, by2, s) in bubbles:
            if bx2 <= x1 or bx1 >= x2 or by2 <= y1 or by1 >= y2:
                continue
            lx1 = int(bx1 - x1)
            ly1 = int(by1 - y1)
            lx2 = int(bx2 - x1)
            ly2 = int(by2 - y1)
            local_bubbles.append((float(lx1), float(ly1), float(lx2), float(ly2), float(s)))
            bubble_anchors.append((lx1, ly1, lx2, ly2))

        # caption anchors: use Vision subset if present, else heuristic detection on crop
        caption_anchors: List[Box] = []
        if vision_text_boxes:
            for (tx1, ty1, tx2, ty2) in vision_text_boxes:
                if tx2 <= x1 or tx1 >= x2 or ty2 <= y1 or ty1 >= y2:
                    continue
                caption_anchors.append((int(tx1 - x1), int(ty1 - y1), int(tx2 - x1), int(ty2 - y1)))
        else:
            caption_anchors = _detect_caption_text_boxes(crop)

        # visual anchors: crucial to keep faces/characters
        visual_anchors = _detect_visual_anchor_boxes(crop)

        anchors: List[Box] = [*bubble_anchors, *caption_anchors, *visual_anchors]

        crops.append(crop)
        meta.append({
            "scene_index": idx,
            "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "bubbles": [list(map(float, b[:4])) + [float(b[4])] for b in local_bubbles],
            "anchors_xyxy": [list(map(int, a)) for a in anchors],
            "caption_anchors_xyxy": [list(map(int, a)) for a in caption_anchors],
            "visual_anchors_xyxy": [list(map(int, a)) for a in visual_anchors],
        })

    # Merge small tails (helps prevent hair-only slices)
    crops, meta = _merge_tiny_tails(crops, meta, min_tail_h=max(180, min_scene_h))

    final_crops: List[np.ndarray] = []
    final_meta: List[Dict[str, Any]] = []

    for crop, m in zip(crops, meta):
        anchors = [tuple(a) for a in m.get("anchors_xyxy", [])]

        crop2, anchor_info = _anchor_trim_if_blank(crop, anchors, pad=80)

        if anchor_info.get("anchor_trimmed"):
            ax1, ay1, ax2, ay2 = anchor_info["anchor_box_xyxy"]
            shifted = []
            for (x1, y1, x2, y2) in anchors:
                if x2 <= ax1 or x1 >= ax2 or y2 <= ay1 or y1 >= ay2:
                    continue
                shifted.append([x1 - ax1, y1 - ay1, x2 - ax1, y2 - ay1])
            anchors = [tuple(a) for a in shifted]
            m["anchors_xyxy"] = shifted

        if enable_trim:
            crop3, trim_info = smart_trim_with_edge_guard(
                crop2,
                anchors,
                pad_text=90,
                pad_general=22,
                max_blank_ratio=0.80,
                post_pad=32,
                edge_guard_band=30,
                edge_guard_min_edge_frac=0.010,
            )
        else:
            crop3 = crop2
            trim_info = {"trimmed": False}

        anchors_after = [tuple(a) for a in m.get("anchors_xyxy", [])]
        if _drop_meaningless(crop3, anchors_after):
            continue

        m.update(anchor_info)
        m.update(trim_info)

        final_crops.append(crop3)
        final_meta.append(m)

    write_scenes(
        out_dir=out_dir,
        stem=image_path.stem,
        crops=final_crops,
        meta=final_meta if write_json else None,
    )
