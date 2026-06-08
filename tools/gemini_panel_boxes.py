#!/usr/bin/env python3
"""
gemini_panel_boxes.py (robust / schema-first)

Input: manifest.stitch.json
Output: manifest.panels.json

For each chunk, asks Gemini for panel boxes in normalized coords [ymin,xmin,ymax,xmax].

Default safety:
  - drop_fullwidth_bands defaults to True (removes common "slanted border / thin strip" junk)
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np

from google import genai
from google.genai import types


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    cand = text[s : e + 1]
    try:
        return json.loads(cand)
    except Exception:
        return None

def _part_text(s: str) -> types.Part:
    try:
        return types.Part.from_text(text=s)
    except TypeError:
        return types.Part.from_text(s)

def _part_image_jpeg(b: bytes) -> types.Part:
    try:
        return types.Part.from_bytes(bytes=b, mime_type="image/jpeg")
    except TypeError:
        return types.Part.from_bytes(data=b, mime_type="image/jpeg")

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def area(b: List[float]) -> float:
    y0, x0, y1, x1 = b
    return max(0.0, y1 - y0) * max(0.0, x1 - x0)

def iou(a: List[float], b: List[float]) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    inter = max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)
    if inter <= 0:
        return 0.0
    ua = area(a) + area(b) - inter
    return inter / ua if ua > 0 else 0.0

def nms(boxes: List[List[float]], iou_thr: float) -> List[List[float]]:
    if not boxes:
        return []
    boxes = [list(map(float, b)) for b in boxes]
    boxes.sort(key=lambda b: area(b), reverse=True)
    kept: List[List[float]] = []
    for b in boxes:
        ok = True
        for k in kept:
            if iou(b, k) >= iou_thr:
                ok = False
                break
        if ok:
            kept.append(b)
    return kept

def sanitize_boxes(boxes: List[Any]) -> List[List[float]]:
    out: List[List[float]] = []
    for b in boxes or []:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        y0, x0, y1, x1 = [float(v) for v in b]
        y0, x0, y1, x1 = clamp01(y0), clamp01(x0), clamp01(y1), clamp01(x1)
        if y1 <= y0 or x1 <= x0:
            continue
        out.append([y0, x0, y1, x1])
    return out

def call_panels(
    client: genai.Client,
    model: str,
    image_path: str,
    max_panels: int,
    max_output_tokens: int,
    temperature: float,
    schema: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    system = (
        "You detect webtoon/manhwa PANEL rectangles in a vertical scroll image.\n"
        "A PANEL is a distinct comic frame separated by gutters/whitespace.\n"
        "Return ONLY valid JSON matching the schema. No extra text.\n"
        "IMPORTANT:\n"
        "- Do NOT return thin horizontal bands, text lines, or speech bubbles.\n"
        "- Do NOT return gutters/margins.\n"
        "- Each bbox must tightly cover the FULL panel artwork.\n"
        "- Return in reading order.\n"
        "- Use [ymin,xmin,ymax,xmax] normalized to 0..1.\n"
    )

    user = {
        "task": "Return bounding boxes for all PANELS (comic frames) in this image.",
        "bbox_format": "[ymin,xmin,ymax,xmax] normalized 0..1",
        "rules": [
            "Panels must be full comic frames separated by gutters/whitespace.",
            "Exclude speech bubbles unless inside a panel (still bbox the panel).",
            "Avoid thin strips: every bbox should cover a meaningful frame, not a band.",
            "Do not output duplicates.",
            f"Return at most {max_panels} panels.",
        ],
    }

    parts = [
        _part_text("INPUT_JSON:" + json.dumps(user, ensure_ascii=False, separators=(",", ":"))),
        _part_image_jpeg(img_bytes),
    ]

    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=max_output_tokens,
        ),
    )

    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, dict):
        return parsed, (resp.text or "")

    raw = resp.text or ""
    try:
        return json.loads(raw), raw
    except Exception:
        obj = _extract_json_object(raw)
        return obj, raw

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-manifest", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--project", required=True)
    ap.add_argument("--location", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")

    ap.add_argument("--min-area-frac", type=float, default=0.010)
    ap.add_argument("--min-h-frac", type=float, default=0.035)
    ap.add_argument("--min-w-frac", type=float, default=0.40)

    # default True (so thin full-width junk is removed without passing any flag)
    ap.add_argument("--drop-fullwidth-bands", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--max-panels-per-chunk", type=int, default=120)
    ap.add_argument("--nms-iou", type=float, default=0.55)

    ap.add_argument("--max-output-tokens", type=int, default=2400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--debug-filter", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.stitch_manifest):
        raise SystemExit(f"stitch manifest not found: {args.stitch_manifest}")

    stitch = load_json(args.stitch_manifest)
    chunks = stitch.get("chunks") or []
    if not chunks:
        raise SystemExit("No chunks found in stitch manifest")

    schema = {
        "type": "OBJECT",
        "properties": {
            "panels": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "bbox": {
                            "type": "ARRAY",
                            "items": {"type": "NUMBER"},
                            "minItems": 4,
                            "maxItems": 4,
                        }
                    },
                    "required": ["bbox"],
                },
            }
        },
        "required": ["panels"],
    }

    existing_by_chunk: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.out):
        try:
            existing = load_json(args.out)
            for it in (existing.get("chunks") or []):
                cf = it.get("chunk_file")
                if cf:
                    existing_by_chunk[cf] = it
        except Exception:
            existing_by_chunk = {}

    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    out_chunks: List[Dict[str, Any]] = []
    ok_cnt = 0
    fail_cnt = 0

    for ch in chunks:
        chunk_file = ch.get("chunk_file")
        chunk_path = ch.get("chunk_path")
        if not chunk_file or not chunk_path:
            continue

        if chunk_file in existing_by_chunk and not existing_by_chunk[chunk_file].get("error"):
            out_chunks.append(existing_by_chunk[chunk_file])
            ok_cnt += 1
            continue

        with Image.open(chunk_path) as im:
            cw, chh = im.size

        result_obj: Optional[Dict[str, Any]] = None
        raw_text = ""

        for _attempt in range(args.retries + 1):
            obj, raw = call_panels(
                client=client,
                model=args.model,
                image_path=chunk_path,
                max_panels=int(args.max_panels_per_chunk),
                max_output_tokens=int(args.max_output_tokens),
                temperature=float(args.temperature),
                schema=schema,
            )
            raw_text = raw

            if isinstance(obj, dict) and isinstance(obj.get("panels"), list):
                result_obj = obj
                break

            repair_prompt = {"instruction": "Re-output ONLY valid JSON matching the schema.", "last_output": (raw_text or "")[:5000]}
            resp = client.models.generate_content(
                model=args.model,
                contents=[types.Content(role="user", parts=[_part_text(json.dumps(repair_prompt, ensure_ascii=False))])],
                config=types.GenerateContentConfig(
                    system_instruction="You are a strict JSON formatter. Output valid JSON only.",
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=int(args.max_output_tokens),
                ),
            )
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, dict) and isinstance(parsed.get("panels"), list):
                result_obj = parsed
                raw_text = resp.text or raw_text
                break

        if result_obj is None:
            fail_cnt += 1
            out_chunks.append(
                {
                    "chunk_file": chunk_file,
                    "chunk_w": cw,
                    "chunk_h": chh,
                    "chunk_path": chunk_path,
                    "error": "parse_failed",
                    "raw": (raw_text or "")[:3000],
                    "panels_norm": [],
                }
            )
            print(f"[warn] {chunk_file} parse_failed")
            continue

        panels_list = []
        for p in (result_obj.get("panels") or []):
            if isinstance(p, dict) and "bbox" in p:
                panels_list.append(p["bbox"])
        boxes = sanitize_boxes(panels_list)

        raw_cnt = len(boxes)
        boxes0 = boxes[:]

        min_h = float(args.min_h_frac)
        min_w = float(args.min_w_frac)

        filtered = []
        for y0, x0, y1, x1 in boxes:
            h = (y1 - y0)
            w = (x1 - x0)
            if h < min_h:
                continue
            if w < min_w:
                continue
            if args.drop_fullwidth_bands and x0 <= 0.02 and x1 >= 0.98 and h < 0.10:
                continue
            filtered.append([y0, x0, y1, x1])

        boxes = filtered
        after_hw = len(boxes)

        if not boxes and boxes0:
            boxes = boxes0

        min_area = float(args.min_area_frac)
        boxes = [b for b in boxes if area(b) >= min_area]
        after_area = len(boxes)

        boxes = nms(boxes, float(args.nms_iou))
        after_nms = len(boxes)

        if args.debug_filter:
            print(f"[dbg] {chunk_file} raw={raw_cnt} after_hw={after_hw} after_area={after_area} after_nms={after_nms}")

        out_chunks.append(
            {
                "chunk_file": chunk_file,
                "chunk_w": cw,
                "chunk_h": chh,
                "chunk_path": chunk_path,
                "panels_norm": boxes,
            }
        )
        ok_cnt += 1
        print(f"[ok] {chunk_file} panels={len(boxes)}")

    out_obj = {
        "source_stitch_manifest": os.path.abspath(args.stitch_manifest),
        "project": args.project,
        "location": args.location,
        "model": args.model,
        "params": {
            "min_area_frac": float(args.min_area_frac),
            "min_h_frac": float(args.min_h_frac),
            "min_w_frac": float(args.min_w_frac),
            "drop_fullwidth_bands": bool(args.drop_fullwidth_bands),
            "max_panels_per_chunk": int(args.max_panels_per_chunk),
            "nms_iou": float(args.nms_iou),
            "max_output_tokens": int(args.max_output_tokens),
            "temperature": float(args.temperature),
            "retries": int(args.retries),
            "resume": bool(args.resume),
        },
        "count_chunks": len(out_chunks),
        "stats": {"ok": ok_cnt, "failed": fail_cnt},
        "chunks": out_chunks,
    }

    dump_json(args.out, out_obj)
    print(f"[ok] wrote={args.out} chunks={len(out_chunks)} ok={ok_cnt} failed={fail_cnt}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
