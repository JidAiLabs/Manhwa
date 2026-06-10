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
    Used ONLY to avoid selecting huge blank gradients.
    """
    if cv2 is None:
        # fallback: luminance stddev as a weak proxy
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
    # downscale for speed
    max_w = 600
    if arr.shape[1] > max_w:
        scale = max_w / arr.shape[1]
        arr = cv2.resize(arr, (int(arr.shape[1] * scale), int(arr.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(arr, 50, 150)
    return float((edges > 0).mean())


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

    # single short "word" onomatopoeia
    if t_clean in _SFX_WORDLIST:
        return True

    # mostly uppercase, very short, few letters, no spaces -> typical SFX
    letters = re.sub(r"[^A-Za-z]", "", t)
    if len(letters) >= 2 and len(letters) <= 8:
        has_space = (" " in t)
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(1, len(letters))
        if (not has_space) and upper_ratio > 0.8:
            # avoid classifying "I'M" etc as SFX
            if t_clean not in {"im", "iam", "ill", "youre", "dont", "cant"}:
                return True

    # repeated letters (e.g., "GRRRR", "NOOOO") often SFX
    if re.search(r"(.)\1\1+", t_clean) and len(t_clean) <= 10:
        return True

    return False


def classify_text_block_as_sfx(block_px: Tuple[int, int, int, int], scene_w: int, scene_h: int, ocr_words: List[Dict[str, Any]]) -> bool:
    """
    Uses geometry + words within the block to determine SFX.
    - SFX tends to be smaller and not full-width, placed mid-frame.
    - Narration boxes tend to be wide, near top/bottom, multi-word.
    """
    x0, y0, x1, y1 = block_px
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    area = bw * bh
    width_ratio = bw / scene_w
    height_ratio = bh / scene_h
    y_center = (y0 + y1) / 2 / scene_h

    # collect words that fall inside this block (approx overlap by bbox center)
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

    joined = " ".join([w for w in words_in if w])
    joined = joined.strip()

    # Strong SFX if the text content looks like SFX
    if looks_like_sfx_text(joined):
        return True

    # Geometry hints (we only mark SFX if it also "looks" like SFX)
    # Small-ish, not wide, often central
    if width_ratio < 0.35 and height_ratio < 0.08 and area < (scene_w * scene_h) * 0.02:
        if 0.12 < y_center < 0.88:
            # and the words are short / few
            if len(words_in) <= 3 and len(joined.replace(" ", "")) <= 10 and joined.upper() == joined:
                return True

    return False


# -----------------------------
# Main cropping logic
# -----------------------------
def get_text_blocks(item: Dict[str, Any]) -> List[List[float]]:
    # prefer vision.text_blocks if present, else from targets
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
    We also filter out ultra-narrow blocks (likely noise) by width ratio.
    """
    text_blocks = get_text_blocks(item)
    ocr_words = get_ocr_words(item)

    all_bands: List[Band] = []
    for b in text_blocks:
        x0, y0, x1, y1 = norm_bbox_to_px(b, scene_w, scene_h)
        bw = max(1, x1 - x0)

        # ignore very narrow blocks (UI debris / tiny marks)
        if (bw / scene_w) < band_width_ratio * 0.20:
            continue

        is_sfx = classify_text_block_as_sfx((x0, y0, x1, y1), scene_w, scene_h, ocr_words)
        kind = "sfx" if is_sfx else "text"

        # Only non-SFX blocks become cut bands
        if kind == "text":
            yy0 = clamp(y0 - text_pad_px, 0, scene_h)
            yy1 = clamp(y1 + text_pad_px, 0, scene_h)
            if yy1 > yy0:
                all_bands.append(Band(yy0, yy1, kind))
        else:
            all_bands.append(Band(y0, y1, kind))

    # merge only the cut bands (kind="text")
    cut_ranges = [(b.y0, b.y1) for b in all_bands if b.kind == "text"]
    cut_ranges = merge_ranges(cut_ranges, merge_gap=max(6, int(scene_h * 0.006)))
    return cut_ranges, all_bands

def choose_segments_by_score(
    img: Image.Image,
    segs: List[Segment],
    max_shots: int,
    abs_thr: float = 0.010,
    rel_thr: float = 0.35,
    min_h_px: int = 90,
) -> Tuple[List[Segment], List[float]]:
    """
    Pick segments by content score (edge density), NOT by height.

    Rules:
      - Compute score for each segment.
      - Candidate segments are those with:
          score >= abs_thr AND score >= max_score * rel_thr
        (drops big flat gradients)
      - If no candidates:
          keep the single best-score segment.
      - Rank by score (with mild height tie-break).
      - Smart dominance suppression:
          If the best segment is large and the 2nd best is both small and weak,
          export only the best (avoids redundant tiny/no-meaning shots).
      - Otherwise, keep additional segments only if they are meaningful vs best,
        until max_shots.
    """
    if not segs:
        return [], []

    W, H = img.size

    # 1) score all segments
    scores: List[float] = []
    for s in segs:
        if s.h < min_h_px:
            scores.append(0.0)
            continue
        crop = img.crop((0, s.y0, W, s.y1))
        scores.append(edge_density_score(crop))

    max_score = max(scores) if scores else 0.0

    # 2) gate candidates (non-blank-ish)
    cand_idx = [
        i for i, (s, sc) in enumerate(zip(segs, scores))
        if s.h >= min_h_px
        and sc >= abs_thr
        and sc >= (max_score * rel_thr if max_score > 0 else abs_thr)
    ]

    # 3) if everything looks blank, keep best-score segment
    if not cand_idx:
        best_i = int(max(range(len(segs)), key=lambda i: scores[i]))
        return [segs[best_i]], [scores[best_i]]

    # 4) rank by score with mild height tie-break
    def rank(i: int) -> float:
        h_norm = segs[i].h / max(1, H)
        return scores[i] * (0.85 + 0.15 * h_norm)

    cand_idx.sort(key=rank, reverse=True)

    best_i = cand_idx[0]
    best_sc = scores[best_i]
    best_h_ratio = segs[best_i].h / max(1, H)

    # --- Smart dominance suppression (safe for 2-part and 3+ part) ---
    # Collapse to 1 shot ONLY if:
    #   - best is quite large, AND
    #   - second is both small AND weak vs best
    DOM_BEST_H = 0.58          # best is "large"
    DOM_SECOND_H_MAX = 0.28    # second is "small"
    DOM_SECOND_SC_RATIO = 0.70 # second is "weak" vs best

    if len(cand_idx) == 1:
        return [segs[best_i]], [scores[best_i]]

    second_i = cand_idx[1]
    second_sc = scores[second_i]
    second_h_ratio = segs[second_i].h / max(1, H)

    if (best_h_ratio >= DOM_BEST_H) and (
        (second_h_ratio <= DOM_SECOND_H_MAX)
        and (best_sc > 0 and second_sc < best_sc * DOM_SECOND_SC_RATIO)
    ):
        return [segs[best_i]], [scores[best_i]]

    # 5) keep best, then add only meaningful extras up to max_shots
    keep_idx = [best_i]

    SCORE_KEEP_RATIO = 0.72   # keep if score >= 72% of best
    MIN_H_KEEP_RATIO = 0.32   # or keep if segment height >= 32% of full height

    for i in cand_idx[1:]:
        if len(keep_idx) >= max(1, max_shots):
            break

        h_ratio = segs[i].h / max(1, H)
        sc = scores[i]

        if (best_sc > 0 and sc >= best_sc * SCORE_KEEP_RATIO) or (h_ratio >= MIN_H_KEEP_RATIO):
            keep_idx.append(i)

    kept = sorted([segs[i] for i in keep_idx], key=lambda s: s.y0)
    kept_scores = [scores[i] for i in keep_idx]
    return kept, kept_scores

def make_debug_image(
    scene_img: Image.Image,
    cut_bands: List[Tuple[int, int]],
    kept: List[Segment],
    all_bands: List[Band],
) -> Image.Image:
    img = scene_img.copy().convert("RGB")
    d = ImageDraw.Draw(img)

    W, H = img.size

    # draw kept segments in green
    for s in kept:
        d.rectangle([2, s.y0, W - 3, s.y1], outline=(0, 255, 0), width=3)

    # draw cut bands in red
    for y0, y1 in cut_bands:
        d.rectangle([2, y0, W - 3, y1], outline=(255, 0, 0), width=3)

    # draw SFX blocks (non-cut) in blue (approx as band)
    for b in all_bands:
        if b.kind == "sfx":
            d.rectangle([2, b.y0, W - 3, b.y1], outline=(0, 120, 255), width=2)

    return img


def write_shots(
    scene_path: str,
    scene_id: int,
    out_dir: str,
    kept: List[Segment],
    manifest_out: Dict[str, Any],
    debug_img: Optional[Image.Image],
    debug: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(scene_path).convert("RGB")
    W, H = img.size

    scene_entry = {
        "scene_id": scene_id,
        "scene_file": os.path.basename(scene_path),
        "scene_path": scene_path,
        "width": W,
        "height": H,
        "shots": []
    }

    for i, s in enumerate(sorted(kept, key=lambda z: z.y0), start=1):
        crop = img.crop((0, s.y0, W, s.y1))
        shot_name = f"scene_{scene_id:04d}_p01_s{i:02d}.jpg"
        shot_path = os.path.join(out_dir, shot_name)
        crop.save(shot_path, quality=95)

        scene_entry["shots"].append({
            "shot_file": shot_name,
            "shot_path": shot_path,
            "bbox_px": [0, s.y0, W, s.y1],
            "bbox_norm": [0.0, s.y0 / H, 1.0, s.y1 / H]
        })

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
    ap.add_argument("--bubble-pad", type=int, default=18, help="reserved (not used in this version)")
    ap.add_argument("--band-width-ratio", type=float, default=0.55, help="min width ratio to treat as meaningful text block")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    with open(args.vision_manifest, "r", encoding="utf-8") as f:
        mani = json.load(f)

    items = mani.get("items") or []
    out_manifest = {
        "source_manifest": args.vision_manifest,
        "out_dir": args.out_dir,
        "shots_strategy": {
            "cut_on_text": "non_sfx_only",
            "rule": "if 1 band -> 2 seg keep largest; if 2 bands -> 3 seg drop smallest; else drop smallest until max_shots",
            "max_shots": args.max_shots,
            "text_pad_px": args.text_pad,
            "band_width_ratio": args.band_width_ratio,
            "blank_guard": True
        },
        "items": []
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
            band_width_ratio=args.band_width_ratio
        )
        segs = segment_by_bands(H, cut_bands)

        kept, kept_scores = choose_segments_by_score(
            img,
            segs,
            max_shots=args.max_shots,
            abs_thr=0.010,   # tune if needed
            rel_thr=0.35,    # tune if needed
            min_h_px=90
        )

        # ---- Blankness guard (prevents “big empty gradient” outputs) ----
        # If the kept segments are mostly blank AND we had at least one non-SFX text band,
        # fallback to full frame (keeps the narration rather than outputting empty).
        if cut_bands and kept:
            W, H = img.size
            all_scores = []
            for s in segs:
                crop = img.crop((0, s.y0, W, s.y1))
                all_scores.append(edge_density_score(crop))

            best_all = max(all_scores) if all_scores else 0.0
            best_kept = max(
                edge_density_score(img.crop((0, s.y0, W, s.y1))) for s in kept
            ) if kept else 0.0

            # If what we kept is significantly worse than what's available, switch to best segment
            if best_all > 0.0 and best_kept < (best_all * 0.60) and best_all >= 0.010:
                bi = int(max(range(len(segs)), key=lambda i: all_scores[i]))
                kept = [segs[bi]]

        dbg = make_debug_image(img, cut_bands, kept, all_bands) if args.debug else None

        print(
            f"[scene {scene_id}] part=1 cut_bands={cut_bands} segs={len(segs)} "
            f"kept={len(kept)} kept_scores={[round(x, 4) for x in kept_scores]}"
        )
        #print(f"[scene {scene_id}] part=1 cut_bands={cut_bands} segs={len(segs)} kept={len(kept)}")
        write_shots(scene_path, scene_id, args.out_dir, kept, out_manifest, dbg, args.debug)

        total_shots += len(kept)

    out_path = os.path.join(args.out_dir, "manifest.smartcrop.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote shots={total_shots} manifest={out_path}")


if __name__ == "__main__":
    main()
