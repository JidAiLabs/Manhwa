#!/usr/bin/env python3
# anchors_to_protected_spans.py
#
# Convert Vision anchors (text/face/object boxes) into merged protected vertical spans.
# Output is a list of [y0,y1] spans in pixels (0 <= y < image_h).

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(p: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def merge_spans(spans: List[Tuple[int, int]], gap_px: int = 0) -> List[List[int]]:
    """
    Merge spans that overlap or are within gap_px.
    spans are (y0,y1) with y0<y1
    """
    if not spans:
        return []
    spans = sorted(spans, key=lambda t: (t[0], t[1]))
    out: List[List[int]] = []
    cur0, cur1 = spans[0]
    for y0, y1 in spans[1:]:
        if y0 <= cur1 + gap_px:
            cur1 = max(cur1, y1)
        else:
            out.append([cur0, cur1])
            cur0, cur1 = y0, y1
    out.append([cur0, cur1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True, help="anchors.json from vision_anchors.py")
    ap.add_argument("--out", required=True, help="protected_spans.json output")

    # padding per anchor type (px)
    ap.add_argument("--pad-text", type=int, default=30)
    ap.add_argument("--pad-face", type=int, default=40)
    ap.add_argument("--pad-object", type=int, default=30)

    # merging tuning
    ap.add_argument("--merge-gap", type=int, default=12, help="merge spans closer than this px gap")
    ap.add_argument("--min-span-h", type=int, default=18, help="drop spans smaller than this height after padding")

    args = ap.parse_args()

    anchors = load_json(args.anchors)
    H = int(anchors.get("image_h") or 0)
    if H <= 0:
        raise SystemExit("anchors.json missing image_h")

    spans: List[Tuple[int, int]] = []

    def add_box_span(box: List[int], pad: int):
        x0, y0, x1, y1 = [int(v) for v in box]
        y0p = clamp(y0 - pad, 0, H)
        y1p = clamp(y1 + pad, 0, H)
        if y1p - y0p >= int(args.min_span_h):
            spans.append((y0p, y1p))

    for t in anchors.get("text_boxes") or []:
        add_box_span(t["bbox_px"], int(args.pad_text))

    for f in anchors.get("face_boxes") or []:
        add_box_span(f["bbox_px"], int(args.pad_face))

    for o in anchors.get("object_boxes") or []:
        add_box_span(o["bbox_px"], int(args.pad_object))

    merged = merge_spans(spans, gap_px=int(args.merge_gap))

    out = {
        "source_anchors": os.path.abspath(args.anchors),
        "image_h": H,
        "params": {
            "pad_text": int(args.pad_text),
            "pad_face": int(args.pad_face),
            "pad_object": int(args.pad_object),
            "merge_gap": int(args.merge_gap),
            "min_span_h": int(args.min_span_h),
        },
        "protected_spans": merged,
        "stats": {
            "raw_spans": len(spans),
            "merged_spans": len(merged),
        }
    }

    dump_json(args.out, out)
    print(f"[ok] protected spans written: {os.path.abspath(args.out)} | raw={len(spans)} merged={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
