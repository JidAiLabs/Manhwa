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


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

_PANEL_CLASS_ID = 0  # class 0 = panel in the trained webtoon model


def detect_panels(
    stitch_manifest_path: str,
    out_path: str,
    weights: str,
    conf: float = 0.25,
    device: Optional[str] = None,
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

    manifest_dir = manifest_path.parent
    chunks = stitch.get("chunks") or []

    # Load model once
    model = YOLO(weights)

    out_chunks: List[Dict[str, Any]] = []

    for ch in chunks:
        chunk_file: str = ch.get("chunk_file") or ""
        chunk_path_stored: str = ch.get("chunk_path") or chunk_file

        # Resolve image path: use stored path if absolute, else relative to manifest dir
        if os.path.isabs(chunk_path_stored):
            img_path = chunk_path_stored
        else:
            img_path = str(manifest_dir / chunk_path_stored)

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

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()   # shape (N, 4)
            cls = boxes.cls.cpu().numpy()      # shape (N,)
            for (x1, y1, x2, y2), c in zip(xyxy, cls):
                if int(c) == _PANEL_CLASS_ID:
                    px_boxes.append((float(x1), float(y1), float(x2), float(y2)))

        panels_norm = boxes_to_panels_norm(px_boxes, w=img_w, h=img_h)

        out_chunks.append(
            {
                "chunk_file": basename,
                "panels_norm": panels_norm,
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
