#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from PIL import Image, ImageDraw

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# -----------------------------
# Helpers / data types
# -----------------------------
@dataclass
class Band:
    y0: int
    y1: int
    kind: str  # "text" / "sfx" / etc.


@dataclass
class Segment:
    y0: int
    y1: int

    @property
    def h(self) -> int:
        return max(0, self.y1 - self.y0)


# Span position labels relative to narration bands
LABEL_TOP_OUTSIDE = "TOP_OUTSIDE"
LABEL_BOT_OUTSIDE = "BOT_OUTSIDE"
LABEL_BETWEEN = "BETWEEN"


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def norm_bbox_to_px(b: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    # b = [x0,y0,x1,y1] normalized
    x0 = int(round(b[0] * w))
    y0 = int(round(b[1] * h))
    x1 = int(round(b[2] * w))
    y1 = int(round(b[3] * h))
    x0 = clamp(x0, 0, w)
    x1 = clamp(x1, 0, w)
    y0 = clamp(y0, 0, h)
    y1 = clamp(y1, 0, h)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def merge_ranges(ranges: List[Tuple[int, int]], merge_gap: int) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda t: (t[0], t[1]))
    out = [list(ranges[0])]
    for a, b in ranges[1:]:
        if a <= out[-1][1] + merge_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(int(x[0]), int(x[1])) for x in out]


def segment_by_bands(H: int, bands: List[Tuple[int, int]]) -> List[Segment]:
    # segments are the NON-band vertical spans
    if not bands:
        return [Segment(0, H)]
    segs: List[Segment] = []
    cur = 0
    for y0, y1 in bands:
        if y0 > cur:
            segs.append(Segment(cur, y0))
        cur = max(cur, y1)
    if cur < H:
        segs.append(Segment(cur, H))
    # remove zero/negative
    segs = [s for s in segs if s.h > 0]
    return segs


def edge_density_score(pil_img: Image.Image) -> float:
    """
    Rough "content" score: fraction of edge pixels (Canny).
    Falls back to luminance stddev if cv2 is unavailable.
    """
    if cv2 is None:
        g = pil_img.convert("L")
        hist = g.histogram()
        n = sum(hist)
        if n == 0:
            return 0.0
        mean = sum(i * c for i, c in enumerate(hist)) / n
        var = sum(((i - mean) ** 2) * c for i, c in enumerate(hist)) / n
        return float(math.sqrt(var)) / 255.0

    import numpy as np  # type: ignore
    g = pil_img.convert("L")
    arr = np.array(g)
    max_w = 600
    if arr.shape[1] > max_w:
        scale = max_w / arr.shape[1]
        arr = cv2.resize(
            arr,
            (int(arr.shape[1] * scale), int(arr.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    edges = cv2.Canny(arr, 50, 150)
    return float((edges > 0).mean())


def is_speedlines_only(crop: Image.Image) -> bool:
    """
    Heuristic to detect "speedlines / motion background" spans.
    These have HIGH edge density but very DIRECTIONAL, uniform texture
    with few closed contours — i.e. they look like radial/parallel lines on
    a flat(ish) background rather than real scene content.

    With cv2: checks gradient-angle concentration (low circular std → directional).
    Without cv2: high edge but low luminance std → plain lines on flat bg.
    """
    score = edge_density_score(crop)
    if score < 0.04:
        return False  # not even high-edge; definitely not speedlines

    if cv2 is not None:
        import numpy as np
        g = crop.convert("L")
        arr = np.array(g)
        max_w = 400
        if arr.shape[1] > max_w:
            scale = max_w / arr.shape[1]
            arr = cv2.resize(
                arr,
                (int(arr.shape[1] * scale), int(arr.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        gx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        threshold = mag.max() * 0.2 if mag.max() > 0 else 1.0
        mask = mag > threshold
        if mask.sum() < 100:
            return False
        angles = np.arctan2(np.abs(gy[mask]), np.abs(gx[mask])) * 180.0 / np.pi
        angle_std = float(np.std(angles))
        # Very uniform gradient angles = directional lines (speedlines / hatching)
        if angle_std < 28 and score > 0.06:
            return True
        return False
    else:
        # Fallback without cv2:
        # High edge + low overall luminance std → featureless line texture
        g = crop.convert("L")
        hist = g.histogram()
        n = sum(hist)
        if n == 0:
            return False
        mean = sum(i * c for i, c in enumerate(hist)) / n
        var = sum(((i - mean) ** 2) * c for i, c in enumerate(hist)) / n
        lum_std = math.sqrt(var)
        # Score high but luminance barely varies → lines on flat bg
        if score > 0.07 and lum_std < 45:
            return True
        return False


def is_background_texture(crop: Image.Image) -> bool:
    """
    Detect uniform noise/grain/halftone backgrounds that fool edge_density_score
    into thinking the span is content-rich.  The key insight:

      Real content edges (character outlines, panel borders, art lines)
      persist when the image is aggressively downsampled.

      High-frequency noise/grain/halftone edges DISAPPEAR under heavy
      downsampling because they are sub-pixel-scale patterns.

    If edge density drops by >65% from full-res to a thumbnail, the span
    is classified as background texture.
    """
    score_full = edge_density_score(crop)
    if score_full < 0.05:
        return False  # not high-edge, not a texture issue

    # Aggressively downsample to ~80px wide
    w, h = crop.size
    target_w = 80
    if w > target_w:
        scale = target_w / w
        thumb = crop.resize(
            (target_w, max(20, int(h * scale))), Image.LANCZOS
        )
    else:
        thumb = crop

    score_small = edge_density_score(thumb)

    # Noise/texture: edge density collapses under downsampling
    # Real content: edge density stays relatively stable
    if score_full > 0 and score_small < score_full * 0.35:
        return True
    return False


# -----------------------------
# SFX detection (heuristics)
# -----------------------------
_SFX_WORDLIST = {
    "grr", "grrr", "growl", "skree", "skreee", "crash", "bam", "pow", "dart", "thud",
    "slash", "whoosh", "wham", "bang", "kick", "rip", "grip"
}


def looks_like_sfx_text(text: str) -> bool:
    """
    Decide if OCR text is SFX (onomatopoeia) vs narration/dialogue.
    Conservative: we only label as SFX when pretty confident.
    """
    if not text:
        return False
    t = text.strip()
    t_clean = re.sub(r"[^A-Za-z]", "", t).lower()

    if t_clean in _SFX_WORDLIST:
        return True

    letters = re.sub(r"[^A-Za-z]", "", t)
    if len(letters) >= 2 and len(letters) <= 8:
        has_space = " " in t
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(1, len(letters))
        if (not has_space) and upper_ratio > 0.8:
            if t_clean not in {"im", "iam", "ill", "youre", "dont", "cant"}:
                return True

    if re.search(r"(.)\1\1+", t_clean) and len(t_clean) <= 10:
        return True

    return False


def classify_text_block_as_sfx(
    block_px: Tuple[int, int, int, int],
    scene_w: int,
    scene_h: int,
    ocr_words: List[Dict[str, Any]],
) -> bool:
    """
    Uses geometry + words within the block to determine SFX.
    SFX tends to be smaller and not full-width, placed mid-frame.
    Narration boxes tend to be wide, near top/bottom, multi-word.
    """
    x0, y0, x1, y1 = block_px
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    area = bw * bh
    width_ratio = bw / scene_w
    height_ratio = bh / scene_h
    y_center = (y0 + y1) / 2 / scene_h

    words_in = []
    for w in ocr_words or []:
        bb = w.get("bbox")
        if not bb or len(bb) != 4:
            continue
        wx0, wy0, wx1, wy1 = norm_bbox_to_px(bb, scene_w, scene_h)
        cx = (wx0 + wx1) / 2
        cy = (wy0 + wy1) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            words_in.append(str(w.get("t", "")).strip())

    joined = " ".join([w for w in words_in if w]).strip()

    if looks_like_sfx_text(joined):
        return True

    if width_ratio < 0.35 and height_ratio < 0.08 and area < (scene_w * scene_h) * 0.02:
        if 0.12 < y_center < 0.88:
            if (
                len(words_in) <= 3
                and len(joined.replace(" ", "")) <= 10
                and joined.upper() == joined
            ):
                return True

    return False


# -----------------------------
# Main cropping logic
# -----------------------------
def get_text_blocks(item: Dict[str, Any]) -> List[List[float]]:
    v = item.get("vision") or {}
    if isinstance(v.get("text_blocks"), list) and v["text_blocks"]:
        return v["text_blocks"]
    blocks = []
    for t in item.get("targets") or []:
        if t.get("type") == "text_block" and isinstance(t.get("bbox"), list):
            blocks.append(t["bbox"])
    return blocks


def get_ocr_words(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = item.get("vision") or {}
    ow = v.get("ocr_words") or []
    return ow if isinstance(ow, list) else []


def build_cut_bands(
    item: Dict[str, Any],
    scene_w: int,
    scene_h: int,
    text_pad_px: int,
    band_width_ratio: float,
) -> Tuple[List[Tuple[int, int]], List[Band]]:
    """
    Build y-bands from NON-SFX text blocks only.
    Returns (cut_ranges, all_bands) where cut_ranges are the merged padded
    regions of narration/dialog text blocks.
    """
    text_blocks = get_text_blocks(item)
    ocr_words = get_ocr_words(item)

    all_bands: List[Band] = []
    for b in text_blocks:
        x0, y0, x1, y1 = norm_bbox_to_px(b, scene_w, scene_h)
        bw = max(1, x1 - x0)

        if (bw / scene_w) < band_width_ratio * 0.20:
            continue

        is_sfx = classify_text_block_as_sfx(
            (x0, y0, x1, y1), scene_w, scene_h, ocr_words
        )
        kind = "sfx" if is_sfx else "text"

        if kind == "text":
            yy0 = clamp(y0 - text_pad_px, 0, scene_h)
            yy1 = clamp(y1 + text_pad_px, 0, scene_h)
            if yy1 > yy0:
                all_bands.append(Band(yy0, yy1, kind))
        else:
            all_bands.append(Band(y0, y1, kind))

    cut_ranges = [(b.y0, b.y1) for b in all_bands if b.kind == "text"]
    cut_ranges = merge_ranges(cut_ranges, merge_gap=max(6, int(scene_h * 0.006)))
    return cut_ranges, all_bands


# ---------------------------------------------------------------
# Step 1 – Label each segment relative to narration cut bands
# ---------------------------------------------------------------
def label_segments(
    segs: List[Segment],
    cut_bands: List[Tuple[int, int]],
) -> List[Tuple[Segment, str]]:
    """
    Assign each segment one of:
      TOP_OUTSIDE  – above the first cut band
      BOT_OUTSIDE  – below the last cut band
      BETWEEN      – sandwiched between two cut bands

    If there are no cut bands every segment gets TOP_OUTSIDE
    (no narration context exists to define BETWEEN).
    """
    if not cut_bands:
        return [(s, LABEL_TOP_OUTSIDE) for s in segs]

    first_band_y0 = cut_bands[0][0]
    last_band_y1 = cut_bands[-1][1]

    result: List[Tuple[Segment, str]] = []
    for s in segs:
        cy = (s.y0 + s.y1) / 2
        if cy < first_band_y0:
            label = LABEL_TOP_OUTSIDE
        elif cy > last_band_y1:
            label = LABEL_BOT_OUTSIDE
        else:
            label = LABEL_BETWEEN
        result.append((s, label))
    return result


# ---------------------------------------------------------------
# Step 5 helper – find lowest-edge row for splitting a segment
# ---------------------------------------------------------------
def find_best_split_row(
    img: Image.Image,
    seg: Segment,
    window: int = 20,
) -> int:
    """
    Scan the middle third of *seg* for the horizontal strip with the
    lowest edge density — a natural "gutter" to split on.
    """
    W, _ = img.size
    seg_h = seg.y1 - seg.y0
    search_start = seg.y0 + seg_h // 3
    search_end = seg.y1 - seg_h // 3

    if search_start >= search_end:
        return (seg.y0 + seg.y1) // 2

    best_row = (seg.y0 + seg.y1) // 2
    best_score = float("inf")

    step = max(1, window // 4)
    for row in range(search_start, search_end, step):
        row_y0 = max(seg.y0, row - window // 2)
        row_y1 = min(seg.y1, row + window // 2)
        if row_y1 <= row_y0:
            continue
        strip = img.crop((0, row_y0, W, row_y1))
        sc = edge_density_score(strip)
        if sc < best_score:
            best_score = sc
            best_row = row

    return best_row


# ---------------------------------------------------------------
# Core rule-based segment selector  (replaces choose_segments_by_score)
# ---------------------------------------------------------------
def rule_based_select_segments(
    img: Image.Image,
    segs: List[Segment],
    cut_bands: List[Tuple[int, int]],
    max_shots: int,
    min_h_px: int = 90,
    max_h_ratio: float = 0.85,
    blank_thr: float = 0.010,
    outside_h_ratio_max: float = 0.25,
    outside_edge_low: float = 0.030,
) -> Tuple[List[Segment], List[str]]:
    """
    Rule-based segment selection following a deterministic decision tree:

    Step 1  Label spans as TOP_OUTSIDE / BETWEEN / BOT_OUTSIDE.
    Step 2  Always keep BETWEEN spans (drop only if truly blank AND tiny).
    Step 3  Keep/trim OUTSIDE spans based on size, edge, speedlines heuristic.
    Step 4  Enforce max_shots by dropping OUTSIDE first, then merging BETWEEN.
    Step 5  Split over-tall segments at natural seam rows.
    Step 6  Final blank guard – replace only OUTSIDE-only sets, never BETWEEN.

    Returns (segments, labels) both sorted by y0.
    """
    W, H = img.size

    if not segs:
        return [], []

    def crop_score(s: Segment) -> float:
        if s.h < min_h_px:
            return 0.0
        return edge_density_score(img.crop((0, s.y0, W, s.y1)))

    # ── Step 1: label ─────────────────────────────────────────────
    labeled = label_segments(segs, cut_bands)

    # Pre-compute the best edge score among all BETWEEN spans.
    # Used for relative suppression of small OUTSIDE spans (Step 3).
    between_scores_all = [
        crop_score(s) for s, l in labeled if l == LABEL_BETWEEN
    ]
    best_between_score = max(between_scores_all) if between_scores_all else 0.0

    # ── Step 2 & 3: structural keep/drop rules ────────────────────
    kept: List[Tuple[Segment, str]] = []

    num_cut_bands = len(cut_bands)

    for s, label in labeled:
        h_ratio = s.h / max(1, H)

        if label == LABEL_BETWEEN:
            # Rule 2.1 – drop if truly blank AND very tiny (< 5% of frame)
            sc = crop_score(s)
            if sc < blank_thr and h_ratio < 0.05:
                continue  # drop: blank micro-spacer

            # Rule 2.2 – drop if background noise/texture with no real content.
            # BETWEEN spans with high edge density from grain/halftone are caught
            # here; real panel content survives aggressive downsampling.
            if h_ratio >= 0.05:  # only test spans large enough to be meaningful
                if is_background_texture(img.crop((0, s.y0, W, s.y1))):
                    continue  # drop: background texture, not real content

            kept.append((s, label))

        else:  # TOP_OUTSIDE or BOT_OUTSIDE
            # Rule 3.1 / 3.2
            sc = crop_score(s)

            # Always keep large outside spans
            if h_ratio >= outside_h_ratio_max:
                kept.append((s, label))
                continue

            # If no BETWEEN spans will survive, keep to avoid empty output
            between_survivors = sum(
                1 for ss, ll in labeled
                if ll == LABEL_BETWEEN and (
                    crop_score(ss) >= blank_thr or ss.h / max(1, H) >= 0.10
                )
            )
            if between_survivors == 0:
                kept.append((s, label))
                continue

            # Drop blank-ish outside spans
            if sc < outside_edge_low:
                continue  # drop

            # Drop speedlines-only background spans (high edge but no content)
            if is_speedlines_only(img.crop((0, s.y0, W, s.y1))):
                continue  # drop

            # Relative suppression: if strong BETWEEN content exists, a small
            # OUTSIDE span must be substantially content-rich to justify keeping.
            # This eliminates redundant slivers (scene 8 top claws, scene 24
            # "NOW" overflow) that have real edges but are dwarfed by BETWEEN.
            OUTSIDE_RELATIVE_THR = 0.78
            if best_between_score > 0 and sc < best_between_score * OUTSIDE_RELATIVE_THR:
                continue  # drop: outshone by dominant BETWEEN content

            kept.append((s, label))

    # Fallback: if nothing kept, fall back to the highest-scoring segment
    if not kept:
        best = max(segs, key=crop_score)
        return [best], [LABEL_TOP_OUTSIDE]

    # ── Step 4: enforce max_shots without destroying BETWEEN ──────
    if len(kept) > max_shots:
        between_spans = [(s, l) for s, l in kept if l == LABEL_BETWEEN]
        outside_spans = [(s, l) for s, l in kept if l != LABEL_BETWEEN]

        # Drop outside spans weakest-first
        outside_spans.sort(key=lambda x: crop_score(x[0]))
        while len(between_spans) + len(outside_spans) > max_shots and outside_spans:
            outside_spans.pop(0)

        kept = between_spans + outside_spans

        # If still over budget, merge adjacent BETWEEN spans (never drop them)
        if len(kept) > max_shots:
            kept_sorted = sorted(kept, key=lambda x: x[0].y0)
            while len(kept_sorted) > max_shots:
                merged = False
                for i in range(len(kept_sorted) - 1):
                    if (
                        kept_sorted[i][1] == LABEL_BETWEEN
                        and kept_sorted[i + 1][1] == LABEL_BETWEEN
                    ):
                        s1, s2 = kept_sorted[i][0], kept_sorted[i + 1][0]
                        merged_seg = Segment(s1.y0, s2.y1)
                        kept_sorted[i] = (merged_seg, LABEL_BETWEEN)
                        kept_sorted.pop(i + 1)
                        merged = True
                        break
                if not merged:
                    # Last resort: drop weakest OUTSIDE, or weakest overall
                    out_idx = next(
                        (i for i, (_, l) in enumerate(kept_sorted) if l != LABEL_BETWEEN),
                        None,
                    )
                    if out_idx is not None:
                        kept_sorted.pop(out_idx)
                    else:
                        weakest = min(range(len(kept_sorted)), key=lambda i: crop_score(kept_sorted[i][0]))
                        kept_sorted.pop(weakest)
            kept = kept_sorted

    # ── Step 5: split over-tall BETWEEN segments at natural seam ────
    # OUTSIDE spans are NEVER split here — they are either kept or dropped
    # as atomic units.  Splitting an OUTSIDE span produces sibling OUTSIDE
    # children which then confuse Step 6's blank-guard, causing the
    # "missing middle" bug (scene 5) and the "one scene → two shots" bug
    # (scenes 1, 6, 16).
    result: List[Tuple[Segment, str]] = []
    for s, label in kept:
        h_ratio = s.h / max(1, H)
        # Only split BETWEEN spans, and only when genuinely over-tall
        if (
            label == LABEL_BETWEEN
            and h_ratio > max_h_ratio
            and s.h >= min_h_px * 2
            and len(kept) < max_shots  # don't split if already at budget
        ):
            split_row = find_best_split_row(img, s)
            top_child = Segment(s.y0, split_row)
            bot_child = Segment(split_row, s.y1)
            # Keep both children (splitting a BETWEEN is structurally motivated)
            for child in (top_child, bot_child):
                if child.h >= min_h_px:
                    result.append((child, label))
        else:
            result.append((s, label))

    # ── Step 6: blank guard – OUTSIDE-only result sets only ───────
    # Never replace BETWEEN spans; only apply when ALL kept are OUTSIDE.
    all_outside = all(l != LABEL_BETWEEN for _, l in result)
    if all_outside and result:
        seg_scores = [(crop_score(s), s) for s in segs]
        best_all_sc, best_all_seg = max(seg_scores, key=lambda x: x[0])
        kept_max_sc = max(crop_score(s) for s, _ in result) if result else 0.0
        if best_all_sc > 0 and kept_max_sc < best_all_sc * 0.60 and best_all_sc >= blank_thr:
            result = [(best_all_seg, LABEL_TOP_OUTSIDE)]

    # Sort by position and unzip
    result.sort(key=lambda x: x[0].y0)
    final_segs = [s for s, _ in result]
    final_labels = [l for _, l in result]
    return final_segs, final_labels


def make_debug_image(
    scene_img: Image.Image,
    cut_bands: List[Tuple[int, int]],
    kept: List[Segment],
    all_bands: List[Band],
    kept_labels: Optional[List[str]] = None,
) -> Image.Image:
    img = scene_img.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    W, H = img.size

    # Label colours: BETWEEN=green, TOP_OUTSIDE=lime, BOT_OUTSIDE=yellow
    label_colour = {
        LABEL_BETWEEN: (0, 220, 0),
        LABEL_TOP_OUTSIDE: (180, 255, 0),
        LABEL_BOT_OUTSIDE: (255, 220, 0),
    }

    for i, s in enumerate(kept):
        lbl = (kept_labels[i] if kept_labels and i < len(kept_labels) else LABEL_BETWEEN)
        colour = label_colour.get(lbl, (0, 255, 0))
        d.rectangle([2, s.y0, W - 3, s.y1], outline=colour, width=3)

    # Draw cut bands in red
    for y0, y1 in cut_bands:
        d.rectangle([2, y0, W - 3, y1], outline=(255, 0, 0), width=3)

    # Draw SFX blocks (non-cut) in blue
    for b in all_bands:
        if b.kind == "sfx":
            d.rectangle([2, b.y0, W - 3, b.y1], outline=(0, 120, 255), width=2)

    return img


def write_shots(
    scene_path: str,
    scene_id: int,
    out_dir: str,
    kept: List[Segment],
    kept_labels: List[str],
    manifest_out: Dict[str, Any],
    debug_img: Optional[Image.Image],
    debug: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(scene_path).convert("RGB")
    W, H = img.size

    scene_entry: Dict[str, Any] = {
        "scene_id": scene_id,
        "scene_file": os.path.basename(scene_path),
        "scene_path": scene_path,
        "width": W,
        "height": H,
        "shots": [],
    }

    for i, s in enumerate(sorted(kept, key=lambda z: z.y0), start=1):
        crop = img.crop((0, s.y0, W, s.y1))
        shot_name = f"scene_{scene_id:04d}_p01_s{i:02d}.jpg"
        shot_path = os.path.join(out_dir, shot_name)
        crop.save(shot_path, quality=95)

        lbl = kept_labels[i - 1] if i - 1 < len(kept_labels) else ""
        scene_entry["shots"].append(
            {
                "shot_file": shot_name,
                "shot_path": shot_path,
                "span_label": lbl,
                "bbox_px": [0, s.y0, W, s.y1],
                "bbox_norm": [0.0, round(s.y0 / H, 6), 1.0, round(s.y1 / H, 6)],
            }
        )

    if debug and debug_img is not None:
        dbg_name = f"scene_{scene_id:04d}_debug.jpg"
        dbg_path = os.path.join(out_dir, dbg_name)
        debug_img.save(dbg_path, quality=95)
        scene_entry["debug_file"] = dbg_name
        scene_entry["debug_path"] = dbg_path

    manifest_out["items"].append(scene_entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-manifest", required=True, help="manifest.vision.json")
    ap.add_argument("--out-dir", required=True, help="output shots directory")
    ap.add_argument("--max-shots", type=int, default=3)
    ap.add_argument("--text-pad", type=int, default=14)
    ap.add_argument("--bubble-pad", type=int, default=18, help="reserved (not used)")
    ap.add_argument(
        "--band-width-ratio",
        type=float,
        default=0.55,
        help="min width ratio to treat as meaningful text block",
    )
    ap.add_argument(
        "--max-h-ratio",
        type=float,
        default=0.85,
        help="h_ratio above which a kept BETWEEN segment is split at its natural seam (OUTSIDE spans are never split)",
    )
    ap.add_argument(
        "--outside-h-max",
        type=float,
        default=0.25,
        help="h_ratio below which an OUTSIDE span is subject to trimming rules",
    )
    ap.add_argument(
        "--outside-edge-low",
        type=float,
        default=0.030,
        help="edge score below which an OUTSIDE span is considered blank-ish and dropped",
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    with open(args.vision_manifest, "r", encoding="utf-8") as f:
        mani = json.load(f)

    items = mani.get("items") or []
    out_manifest: Dict[str, Any] = {
        "source_manifest": args.vision_manifest,
        "out_dir": args.out_dir,
        "shots_strategy": {
            "engine": "rule_based_decision_tree",
            "cut_on_text": "non_sfx_only",
            "between_rule": "always_keep_unless_blank_and_tiny",
            "outside_rule": "trim_by_size_edge_speedlines",
            "max_shots_enforcement": "drop_outside_first_then_merge_between",
            "split_on_h_ratio": args.max_h_ratio,
            "split_applies_to": "BETWEEN_only_never_OUTSIDE",
            "max_shots": args.max_shots,
            "text_pad_px": args.text_pad,
            "band_width_ratio": args.band_width_ratio,
            "outside_h_max": args.outside_h_max,
            "outside_edge_low": args.outside_edge_low,
        },
        "items": [],
    }

    total_shots = 0

    for it in items:
        scene_id = int(it.get("scene_id"))
        scene_path = it.get("scene_path") or it.get("scene_file")
        if not scene_path or not os.path.exists(scene_path):
            print(f"[scene {scene_id}] missing file: {scene_path}")
            continue

        img = Image.open(scene_path)
        W, H = img.size

        cut_bands, all_bands = build_cut_bands(
            it, W, H,
            text_pad_px=args.text_pad,
            band_width_ratio=args.band_width_ratio,
        )
        segs = segment_by_bands(H, cut_bands)

        kept, kept_labels = rule_based_select_segments(
            img,
            segs,
            cut_bands,
            max_shots=args.max_shots,
            min_h_px=90,
            max_h_ratio=args.max_h_ratio,
            blank_thr=0.010,
            outside_h_ratio_max=args.outside_h_max,
            outside_edge_low=args.outside_edge_low,
        )

        dbg = (
            make_debug_image(img, cut_bands, kept, all_bands, kept_labels)
            if args.debug
            else None
        )

        kept_scores = []
        for s in kept:
            if s.h >= 90:
                kept_scores.append(
                    round(edge_density_score(img.crop((0, s.y0, W, s.y1))), 4)
                )
            else:
                kept_scores.append(0.0)

        print(
            f"[scene {scene_id}] cut_bands={cut_bands} segs={len(segs)} "
            f"kept={len(kept)} labels={kept_labels} scores={kept_scores}"
        )

        write_shots(
            scene_path, scene_id, args.out_dir,
            kept, kept_labels, out_manifest, dbg, args.debug,
        )
        total_shots += len(kept)

    out_path = os.path.join(args.out_dir, "manifest.smartcrop.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote shots={total_shots}  manifest={out_path}")


if __name__ == "__main__":
    main()
