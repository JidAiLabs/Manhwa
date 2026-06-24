"""
studio/detect/yolo_panels.py

YOLO-based panel detector — drop-in replacement for tools/gemini_panel_boxes.py.

Output schema is schema-compatible with gemini_panel_boxes.py:
  {"chunks": [{"chunk_file": "<basename>", "panels_norm": [[ymin,xmin,ymax,xmax], ...]}, ...]}

Boxes are normalized 0..1, sorted top-to-bottom by ymin.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from studio.paths import resolve_rel


# ---------------------------------------------------------------------------
# Pure conversion helper
# ---------------------------------------------------------------------------

def boxes_to_panels_norm(
    px_boxes: Sequence[Tuple[float, float, float, float]],
    *,
    w: float,
    h: float,
) -> List[List[float]]:
    """Convert pixel xyxy boxes to normalised [ymin, xmin, ymax, xmax] lists.

    Args:
        px_boxes: Sequence of (x1, y1, x2, y2) pixel coordinates.
        w: Image width in pixels.
        h: Image height in pixels.

    Returns:
        List of [ymin, xmin, ymax, xmax] float lists sorted by ymin ascending.
        Values are rounded to 6 decimal places.
    """
    result: List[List[float]] = []
    for x1, y1, x2, y2 in px_boxes:
        ymin = round(float(y1) / h, 6)
        xmin = round(float(x1) / w, 6)
        ymax = round(float(y2) / h, 6)
        xmax = round(float(x2) / w, 6)
        result.append([ymin, xmin, ymax, xmax])
    result.sort(key=lambda b: b[0])
    return result


def snap_panels_to_elements(
    panels_norm: Sequence[Sequence[float]],
    element_boxes_norm: Sequence[Sequence[float]],
    *,
    min_inside_frac: float = 0.55,
) -> List[List[float]]:
    """Grow panel boxes to swallow speech bubbles/system boxes they slice.

    A panel edge cutting through a bubble leaves a bubble remnant in the crop
    (and the rest in a neighbour). Each element box is assigned to the ONE
    panel containing the largest share of its area; when that share is at
    least *min_inside_frac* but not total, that panel grows to the union.
    Elements fully inside (nothing to fix) or mostly outside every panel
    (floating in the gutter) are left alone. Output stays sorted by ymin.
    Boxes are normalized [ymin, xmin, ymax, xmax].
    """
    panels = [[float(v) for v in p] for p in panels_norm]
    for b in element_boxes_norm:
        by0, bx0, by1, bx1 = (float(v) for v in b)
        barea = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
        if barea <= 0.0:
            continue
        best_i, best_frac = -1, 0.0
        for i, (py0, px0, py1, px1) in enumerate(panels):
            iy = max(0.0, min(py1, by1) - max(py0, by0))
            ix = max(0.0, min(px1, bx1) - max(px0, bx0))
            frac = (iy * ix) / barea
            if frac > best_frac:
                best_frac, best_i = frac, i
        if best_i >= 0 and min_inside_frac <= best_frac < 1.0 - 1e-9:
            p = panels[best_i]
            panels[best_i] = [min(p[0], by0), min(p[1], bx0),
                              max(p[2], by1), max(p[3], bx1)]
    panels.sort(key=lambda p: p[0])
    return [[round(v, 6) for v in p] for p in panels]


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

_PANEL_CLASS_ID = 0  # class 0 = panel in the trained webtoon model

# Non-panel classes the webtoon model was trained on (data.yaml order:
# panel, system_box, speech_bubble, text, sfx, character). These were
# previously discarded; now emitted as elements_norm for bubble snapping
# and pixel-accurate inpaint masks.
_ELEMENT_CLASS_IDS = {1: "system_box", 2: "speech_bubble", 4: "sfx"}
# Classes whose sliced boxes should pull the panel boundary outward.
_SNAP_CLASSES = ("speech_bubble", "system_box")


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def _dedup_iou(boxes, thr: float = 0.5):
    """Drop near-duplicate boxes (the same panel seen in two overlapping
    windows), keeping the first in reading order."""
    kept: List[Tuple[float, float, float, float]] = []
    for b in sorted(boxes, key=lambda z: (z[1], z[0])):
        if not any(_iou(b, k) > thr for k in kept):
            kept.append(b)
    return kept


def _under_segmented(px_boxes, img_h: int, *, min_h: int = 8000) -> bool:
    """True when a TALL chunk was under-detected: no panels at all, ONE box
    spanning most of it (a chunk-as-panel), or too few panels for its height —
    the signature of YOLO downscaling a giant chunk until panels vanish."""
    if img_h <= min_h:
        return False
    if not px_boxes:
        return True
    if any((y2 - y1) > 0.7 * img_h for (_x1, y1, _x2, y2) in px_boxes):
        return True
    return len(px_boxes) < img_h / 4000.0


def _retile_panels(model, img_path, img_w, img_h, conf, device,
                   *, win: int = 6000, overlap: int = 600):
    """Re-detect panels in an under-segmented chunk by slicing it into vertical
    windows YOLO resolves at proper scale, offsetting boxes back to chunk coords,
    and de-duplicating the window overlaps. Returns panel boxes (x1,y1,x2,y2)."""
    import numpy as _np
    from PIL import Image as _Image
    _Image.MAX_IMAGE_PIXELS = None
    im = _Image.open(img_path).convert("RGB")
    found: List[Tuple[float, float, float, float]] = []
    y = 0
    while True:
        y1 = min(img_h, y + win)
        arr = _np.asarray(im.crop((0, y, img_w, y1)))
        res = model.predict(source=arr, conf=conf, device=device, verbose=False)[0]
        b = res.boxes
        if b is not None and len(b) > 0:
            for (x1, ty1, x2, ty2), c in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy()):
                if int(c) == _PANEL_CLASS_ID:
                    found.append((float(x1), float(ty1) + y, float(x2), float(ty2) + y))
        if y1 >= img_h:
            break
        y += win - overlap
    return _dedup_iou(found)


def detect_panels(
    stitch_manifest_path: str,
    out_path: str,
    weights: str,
    conf: float = 0.25,
    device: Optional[str] = None,
    snap: bool = True,
) -> Dict[str, Any]:
    """Run YOLO panel detection over all chunks listed in a stitch manifest.

    Args:
        stitch_manifest_path: Path to manifest.stitch.json.
        out_path: Where to write manifest.panels.json.
        weights: Path to YOLO .pt weights file.
        conf: Confidence threshold (default 0.25).
        device: Inference device ("mps", "cpu", "cuda", …).
                Defaults to "mps" if available, else "cpu".

    Returns:
        The output dict that was written to out_path.
    """
    # Lazy import so the module can be imported without ultralytics installed
    # (pure tests must not require it).
    import torch
    from ultralytics import YOLO

    # Resolve device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load manifest
    manifest_path = Path(stitch_manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        stitch = json.load(f)

    chunks = stitch.get("chunks") or []

    # Load model once
    model = YOLO(weights)

    out_chunks: List[Dict[str, Any]] = []

    for ch in chunks:
        chunk_file: str = ch.get("chunk_file") or ""
        chunk_path_stored: str = ch.get("chunk_path") or chunk_file

        # Resolve image path via resolve_rel (absolute paths pass through unchanged)
        img_path = str(resolve_rel(manifest_path, chunk_path_stored))

        basename = os.path.basename(img_path) if not chunk_file else chunk_file

        results = model.predict(
            source=img_path,
            conf=conf,
            device=device,
            verbose=False,
        )

        result = results[0]
        img_h, img_w = result.orig_shape  # (H, W)

        boxes = result.boxes
        px_boxes: List[Tuple[float, float, float, float]] = []
        el_px: Dict[str, List[Tuple[float, float, float, float]]] = {
            name: [] for name in _ELEMENT_CLASS_IDS.values()
        }

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()   # shape (N, 4)
            cls = boxes.cls.cpu().numpy()      # shape (N,)
            for (x1, y1, x2, y2), c in zip(xyxy, cls):
                ci = int(c)
                box = (float(x1), float(y1), float(x2), float(y2))
                if ci == _PANEL_CLASS_ID:
                    px_boxes.append(box)
                elif ci in _ELEMENT_CLASS_IDS:
                    el_px[_ELEMENT_CLASS_IDS[ci]].append(box)

        # RE-TILE GUARD: a tall chunk the full-chunk pass under-segmented (one box
        # spanning most of it, or too sparse) means YOLO's downscale ate the
        # panels. Re-run on vertical sub-tiles so each panel is seen at full scale.
        if _under_segmented(px_boxes, img_h):
            retiled = _retile_panels(model, img_path, img_w, img_h, conf, device)
            if len(retiled) > len(px_boxes):
                px_boxes = retiled

        panels_norm = boxes_to_panels_norm(px_boxes, w=img_w, h=img_h)
        elements_norm = {
            name: boxes_to_panels_norm(bx, w=img_w, h=img_h)
            for name, bx in el_px.items()
            if bx
        }
        if snap:
            snap_boxes = [b for name in _SNAP_CLASSES for b in elements_norm.get(name, [])]
            if snap_boxes:
                panels_norm = snap_panels_to_elements(panels_norm, snap_boxes)

        out_chunks.append(
            {
                "chunk_file": basename,
                "panels_norm": panels_norm,
                # additive: chunk-space boxes of the model's non-panel classes
                "elements_norm": elements_norm,
            }
        )

    out_obj: Dict[str, Any] = {
        "chunks": out_chunks,
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    return out_obj
