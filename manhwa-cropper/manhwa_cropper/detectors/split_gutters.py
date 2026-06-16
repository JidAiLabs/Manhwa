import cv2
import numpy as np

def _row_scores(img_bgr: np.ndarray):
    """
    Compute per-row content score using:
    - edge density
    - gradient magnitude
    - darkness density (helps when art is dark on white)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 40, 120)
    edges = (edges > 0).astype(np.uint8)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    dark = (gray < 210).astype(np.uint8)  # mild threshold

    # normalize scores per row
    edge_score = edges.mean(axis=1)
    grad_score = (grad.mean(axis=1) / (grad.max() + 1e-6))
    dark_score = dark.mean(axis=1)

    # weighted sum
    score = (0.55 * edge_score) + (0.30 * grad_score) + (0.15 * dark_score)
    return score

def _bubble_cut_mask(H: int, bubble_boxes, pad: int = 10):
    """
    Returns a boolean array per row indicating rows that should NOT be used as cut lines
    because they intersect a (padded) bubble.
    """
    mask = np.zeros((H,), dtype=bool)
    for (x1, y1, x2, y2, s) in bubble_boxes:
        y1p = max(0, int(y1) - pad)
        y2p = min(H - 1, int(y2) + pad)
        mask[y1p:y2p+1] = True
    return mask

def _find_gutter_bands(score, forbid_rows, min_gutter_h: int, quantile: float = 0.18):
    """
    Gutter bands are contiguous low-score rows (below threshold) that are not forbidden by bubbles.
    """
    thr = float(np.quantile(score, quantile))
    low = (score <= thr) & (~forbid_rows)

    bands = []
    H = len(score)
    i = 0
    while i < H:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < H and low[j]:
            j += 1
        if (j - i) >= min_gutter_h:
            bands.append((i, j))  # [i, j)
        i = j
    return bands

def _choose_cut_lines(bands):
    """
    Use band midpoints as cut lines.
    """
    cuts = []
    for (a, b) in bands:
        cuts.append((a + b) // 2)
    return cuts

def propose_scene_boxes_from_gutters(
    img: np.ndarray,
    bubble_boxes,
    min_scene_h: int = 220,
    min_gutter_h: int = 18,
    max_scenes: int = 120,
):
    H, W = img.shape[:2]
    score = _row_scores(img)
    forbid = _bubble_cut_mask(H, bubble_boxes, pad=12)

    bands = _find_gutter_bands(score, forbid, min_gutter_h=min_gutter_h, quantile=0.18)
    cuts = _choose_cut_lines(bands)

    # Always include boundaries
    cuts = [0] + [c for c in cuts if 0 < c < H] + [H]

    # Build scenes between cuts, merging tiny fragments
    scenes = []
    start = cuts[0]
    for c in cuts[1:]:
        if (c - start) < min_scene_h:
            # too small: postpone cut (merge)
            continue
        scenes.append((0, start, W, c))
        start = c

        if len(scenes) >= max_scenes:
            break

    # Tail
    if start < H and (H - start) >= min_scene_h and len(scenes) < max_scenes:
        scenes.append((0, start, W, H))

    # If we found nothing (edge case), fallback to whole image
    if not scenes:
        scenes = [(0, 0, W, H)]

    # Cleanup: remove duplicates/zero-height
    cleaned = []
    last_y2 = -1
    for (x1, y1, x2, y2) in scenes:
        if y2 <= y1:
            continue
        if y2 == last_y2:
            continue
        cleaned.append((x1, y1, x2, y2))
        last_y2 = y2
    return cleaned
