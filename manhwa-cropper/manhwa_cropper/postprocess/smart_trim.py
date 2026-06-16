# manhwa_cropper/postprocess/smart_trim.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any
import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def _clip(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Box:
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _edge_fraction_in_band(gray: np.ndarray, band: int, side: str) -> float:
    h, w = gray.shape[:2]
    band = int(max(1, min(band, h // 3)))
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g, 40, 120)

    if side == "top":
        region = edges[0:band, :]
    elif side == "bottom":
        region = edges[h - band : h, :]
    elif side == "left":
        bandw = int(max(1, min(band, w // 3)))
        region = edges[:, 0:bandw]
    elif side == "right":
        bandw = int(max(1, min(band, w // 3)))
        region = edges[:, w - bandw : w]
    else:
        raise ValueError("side must be top/bottom/left/right")

    return float(np.count_nonzero(region)) / float(region.size)


def trim_with_text_anchor(
    img_bgr: np.ndarray,
    anchors_xyxy: List[Box],
    *,
    pad_text: int = 80,
    pad_general: int = 20,
    max_blank_ratio: float = 0.80,
) -> np.ndarray:
    """
    Content-aware trim that:
      - trims large blank margins (top/bottom/left/right) using an "ink" mask
      - guarantees anchor regions are kept (bubbles/captions)
    This avoids the classic failure mode: cropping tightly to bubbles and cutting faces below.
    """
    h, w = img_bgr.shape[:2]
    if h < 4 or w < 4:
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Edges are good for line-art; texture helps on smooth gradients.
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g, 40, 120)

    # Local contrast/texture mask (helps keep faces in gradients)
    lap = cv2.Laplacian(g, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)

    ink = ((edges > 0) | (lap > 18)).astype(np.uint8) * 255

    # Stamp anchors as "ink" so trimming cannot cut them away
    for (x1, y1, x2, y2) in anchors_xyxy:
        x1, y1, x2, y2 = _clip(x1, y1, x2, y2, w, h)
        ink[y1:y2, x1:x2] = 255

    # Light morphological connect
    k = max(3, int(min(w, h) * 0.01))
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)

    # If there's basically no ink, do nothing
    if float(np.count_nonzero(ink)) / float(ink.size) < 0.0005:
        return img_bgr

    # Helper: decide if a margin strip is "blank enough" to trim
    def blank_ratio_in_strip(side: str, t: int) -> float:
        if t <= 0:
            return 0.0
        if side == "top":
            region = ink[0:t, :]
        elif side == "bottom":
            region = ink[h - t : h, :]
        elif side == "left":
            region = ink[:, 0:t]
        elif side == "right":
            region = ink[:, w - t : w]
        else:
            raise ValueError(side)
        # blank ratio = fraction of pixels with NO ink
        return 1.0 - (float(np.count_nonzero(region)) / float(region.size))

    # Iteratively trim while margins are blank enough
    top = 0
    bottom = h
    left = 0
    right = w

    step = max(6, int(min(w, h) * 0.01))
    max_steps = 400

    def current_h() -> int:
        return bottom - top

    def current_w() -> int:
        return right - left

    # Precompute anchor union to ensure we keep some pad around text
    if anchors_xyxy:
        ax1 = min(a[0] for a in anchors_xyxy)
        ay1 = min(a[1] for a in anchors_xyxy)
        ax2 = max(a[2] for a in anchors_xyxy)
        ay2 = max(a[3] for a in anchors_xyxy)
        ax1, ay1, ax2, ay2 = _clip(ax1, ay1, ax2, ay2, w, h)
        # text pad is larger than general pad
        must_keep = _clip(ax1 - pad_text, ay1 - pad_text, ax2 + pad_text, ay2 + pad_text, w, h)
    else:
        must_keep = (0, 0, w, h)

    for _ in range(max_steps):
        changed = False

        # Stop if we'd cut into must-keep region
        mkx1, mky1, mkx2, mky2 = must_keep

        # TOP
        if top + step < bottom and top + step <= mky1:
            br = blank_ratio_in_strip("top", top + step)
            if br >= max_blank_ratio:
                top += step
                changed = True

        # BOTTOM
        if bottom - step > top and bottom - step >= mky2:
            # compute blank ratio in bottom strip relative to full image;
            # we just use the strip thickness = (h - (bottom-step))
            t = h - (bottom - step)
            br = blank_ratio_in_strip("bottom", t)
            if br >= max_blank_ratio:
                bottom -= step
                changed = True

        # LEFT
        if left + step < right and left + step <= mkx1:
            br = blank_ratio_in_strip("left", left + step)
            if br >= max_blank_ratio:
                left += step
                changed = True

        # RIGHT
        if right - step > left and right - step >= mkx2:
            t = w - (right - step)
            br = blank_ratio_in_strip("right", t)
            if br >= max_blank_ratio:
                right -= step
                changed = True

        # Prevent over-trimming
        if current_h() < 80 or current_w() < 80:
            break

        if not changed:
            break

    # Final pad: keep a bit of context (helps faces not get clipped tight)
    left2 = max(0, left - pad_general)
    right2 = min(w, right + pad_general)
    top2 = max(0, top - pad_general)
    bottom2 = min(h, bottom + pad_general)

    left2, top2, right2, bottom2 = _clip(left2, top2, right2, bottom2, w, h)
    return img_bgr[top2:bottom2, left2:right2].copy()


def smart_trim_with_edge_guard(
    img_bgr: np.ndarray,
    anchors_xyxy: List[Box],
    *,
    pad_text: int = 80,
    pad_general: int = 20,
    max_blank_ratio: float = 0.80,
    post_pad: int = 28,
    edge_guard_band: int = 28,
    edge_guard_min_edge_frac: float = 0.012,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calls trim_with_text_anchor(), then applies an edge-guard padding
    so hair/faces aren't cut tight in glow/gradients.
    Returns (image, info).
    """
    h0, w0 = img_bgr.shape[:2]
    gray0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    pre_edge = {
        "top": _edge_fraction_in_band(gray0, edge_guard_band, "top"),
        "bottom": _edge_fraction_in_band(gray0, edge_guard_band, "bottom"),
        "left": _edge_fraction_in_band(gray0, edge_guard_band, "left"),
        "right": _edge_fraction_in_band(gray0, edge_guard_band, "right"),
    }

    trimmed = trim_with_text_anchor(
        img_bgr,
        anchors_xyxy,
        pad_text=pad_text,
        pad_general=pad_general,
        max_blank_ratio=max_blank_ratio,
    )

    crop1 = trimmed
    h1, w1 = crop1.shape[:2]
    if h1 < 2 or w1 < 2:
        return img_bgr, {"trimmed": False, "reason": "degenerate_after_trim"}

    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    post_edge = {
        "top": _edge_fraction_in_band(gray1, edge_guard_band, "top"),
        "bottom": _edge_fraction_in_band(gray1, edge_guard_band, "bottom"),
        "left": _edge_fraction_in_band(gray1, edge_guard_band, "left"),
        "right": _edge_fraction_in_band(gray1, edge_guard_band, "right"),
    }

    add_top = post_pad if max(pre_edge["top"], post_edge["top"]) >= edge_guard_min_edge_frac else 0
    add_bottom = post_pad if max(pre_edge["bottom"], post_edge["bottom"]) >= edge_guard_min_edge_frac else 0
    add_left = post_pad if max(pre_edge["left"], post_edge["left"]) >= edge_guard_min_edge_frac else 0
    add_right = post_pad if max(pre_edge["right"], post_edge["right"]) >= edge_guard_min_edge_frac else 0

    if any([add_top, add_bottom, add_left, add_right]):
        crop2 = cv2.copyMakeBorder(
            crop1, add_top, add_bottom, add_left, add_right, borderType=cv2.BORDER_REPLICATE
        )
    else:
        crop2 = crop1

    info = {
        "trimmed": True,
        "pre_edge": pre_edge,
        "post_edge": post_edge,
        "post_pad": {"top": add_top, "bottom": add_bottom, "left": add_left, "right": add_right},
        "base_shape": [int(h0), int(w0)],
        "trimmed_shape": [int(h1), int(w1)],
    }
    return crop2, info
