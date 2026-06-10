#!/usr/bin/env python3
"""
scene_group_builder.py (corrected)

Deterministic grouping of scene_*.jpg into "shots" (groups) while preserving order.

Key fixes:
- Adds HARD STOP merge gates:
  - never merge across heavy-text boundary (text_coverage >= heavy_text_cov)
  - never merge across long OCR boundary (ocr_len >= long_ocr_len)
  This prevents runaway "merge almost everything" behavior.
- Keeps merges conservative and capped with max_group_len.
- Still merges ONLY consecutive scenes. No LLM calls.

Inputs:
- manifest.vision.json from vision_extract.py

Outputs:
- manifest.groups.json with:
  - shots: list of grouped scene_files (order preserved)
  - summary: counts + group size stats
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# IO
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Helpers
# -----------------------------
def scene_num(scene_file: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(scene_file or ""))
    return int(m.group(1)) if m else -1


def get_labels_objects(item: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    v = item.get("vision") or {}
    labels = []
    for x in (v.get("labels") or []):
        d = x.get("desc")
        if d:
            labels.append(str(d))
    objects = []
    for x in (v.get("objects") or []):
        n = x.get("name")
        if n:
            objects.append(str(n))
    return labels, objects


def normalize_items(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = manifest.get("items") or []
    out: List[Dict[str, Any]] = []

    for it in items:
        sf = it.get("scene_file")
        if not sf:
            continue

        labels, objects = get_labels_objects(it)

        ocr = (it.get("ocr_clean") or "").strip()
        kw = it.get("keywords") if isinstance(it.get("keywords"), list) else []
        try:
            text_cov = float(it.get("text_coverage") or 0.0)
        except Exception:
            text_cov = 0.0

        out.append(
            {
                "scene_id": int(it.get("scene_id") or scene_num(sf) or 0),
                "scene_file": sf,
                "scene_path": it.get("scene_path"),
                "ocr_clean": ocr,
                "ocr_len": len(ocr),
                "text_only": bool(it.get("text_only")),
                "text_coverage": text_cov,
                "keywords": [str(x) for x in kw if x],
                "labels": labels[:15],
                "objects": objects[:15],
                "error": (it.get("vision") or {}).get("error"),
            }
        )

    out.sort(key=lambda x: (x["scene_id"], x["scene_file"]))
    return out


def _kw_overlap(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    sa = {x.lower() for x in a if isinstance(x, str) and x}
    sb = {x.lower() for x in b if isinstance(x, str) and x}
    return len(sa.intersection(sb))


def _is_low_text_scene(s: Dict[str, Any], low_text_cov: float, short_ocr_len: int) -> bool:
    return (float(s.get("text_coverage") or 0.0) <= low_text_cov) or (int(s.get("ocr_len") or 0) <= short_ocr_len)


def _is_heavy_text_scene(s: Dict[str, Any], heavy_text_cov: float, long_ocr_len: int) -> bool:
    return (float(s.get("text_coverage") or 0.0) >= heavy_text_cov) or (int(s.get("ocr_len") or 0) >= long_ocr_len)


def _hard_stop_merge(prev: Dict[str, Any], cur: Dict[str, Any], heavy_text_cov: float, long_ocr_len: int) -> bool:
    """
    Prevents runaway merging:
    - do not merge across heavy-text panels
    - do not merge across long OCR panels
    """
    if float(prev.get("text_coverage") or 0.0) >= heavy_text_cov:
        return True
    if float(cur.get("text_coverage") or 0.0) >= heavy_text_cov:
        return True
    if int(prev.get("ocr_len") or 0) >= long_ocr_len:
        return True
    if int(cur.get("ocr_len") or 0) >= long_ocr_len:
        return True
    return False


# -----------------------------
# Grouping logic
# -----------------------------
def group_scenes(
    scenes: List[Dict[str, Any]],
    *,
    max_group_len: int = 3,
    max_merge_text_run: int = 2,
    merge_single_text_only: bool = True,
    low_text_cov: float = 0.18,
    heavy_text_cov: float = 0.42,
    short_ocr_len: int = 25,
    long_ocr_len: int = 140,
    min_keyword_overlap: int = 2,
) -> List[Dict[str, Any]]:
    shots: List[Dict[str, Any]] = []
    i = 0
    shot_id = 1

    def mk_shot(scene_files: List[str], reason: Optional[str], merged_from: List[int]) -> Dict[str, Any]:
        return {"shot_id": shot_id, "scene_files": scene_files, "why_merge": reason, "scene_ids": merged_from}

    def can_attach(prev: Dict[str, Any], cur: Dict[str, Any]) -> Tuple[bool, str]:
        if prev.get("error") or cur.get("error"):
            return False, "vision_error_boundary"

        # HARD STOP gates (the important fix)
        if _hard_stop_merge(prev, cur, heavy_text_cov=heavy_text_cov, long_ocr_len=long_ocr_len):
            return False, "hard_stop_heavy_or_long"

        # Merge consecutive text_only (short run only, enforced later too)
        if prev.get("text_only") and cur.get("text_only"):
            return True, "consecutive_text_only"

        # Attach lone text_only to neighbor only if neighbor is not heavy/long (already gated) and has context
        if merge_single_text_only and (prev.get("text_only") or cur.get("text_only")):
            # allow if the non-text_only side is low-text visual or has some OCR
            if _is_low_text_scene(prev, low_text_cov, short_ocr_len) or _is_low_text_scene(cur, low_text_cov, short_ocr_len):
                return True, "single_text_only_attached"
            if int(prev.get("ocr_len") or 0) >= 15 or int(cur.get("ocr_len") or 0) >= 15:
                return True, "single_text_only_attached"
            return False, "single_text_only_not_attached"

        # Low-text visual flow: merge small sequences of action/reaction
        if _is_low_text_scene(prev, low_text_cov, short_ocr_len) and _is_low_text_scene(cur, low_text_cov, short_ocr_len):
            return True, "low_text_visual_flow"

        # Keyword continuity (but not enough alone to chain forever, and hard stop already applied)
        ov = _kw_overlap(prev.get("keywords") or [], cur.get("keywords") or [])
        if ov >= min_keyword_overlap:
            return True, "keyword_continuity"

        return False, "no_merge_signal"

    while i < len(scenes):
        s = scenes[i]

        if s.get("error"):
            shots.append(
                {"shot_id": shot_id, "scene_files": [s["scene_file"]], "why_merge": None, "scene_ids": [s["scene_id"]], "note": f"vision_error={s['error']}"}
            )
            shot_id += 1
            i += 1
            continue

        group = [s]
        files = [s["scene_file"]]
        ids = [s["scene_id"]]
        reason_used: Optional[str] = None

        j = i + 1
        while j < len(scenes) and len(group) < max_group_len:
            prev = group[-1]
            cur = scenes[j]

            ok, why = can_attach(prev, cur)
            if not ok:
                break

            # enforce max consecutive text_only run INSIDE group
            if cur.get("text_only"):
                run = 0
                k = len(group) - 1
                while k >= 0 and group[k].get("text_only"):
                    run += 1
                    k -= 1
                if run >= max_merge_text_run:
                    break

            group.append(cur)
            files.append(cur["scene_file"])
            ids.append(cur["scene_id"])
            if reason_used is None:
                reason_used = why

            j += 1

        shots.append(mk_shot(files, None if len(files) == 1 else (reason_used or "merged"), ids))
        shot_id += 1
        i = j

    for idx, sh in enumerate(shots, 1):
        sh["shot_id"] = idx

    return shots


def build_summary(scenes: List[Dict[str, Any]], shots: List[Dict[str, Any]]) -> Dict[str, Any]:
    sizes = [len(s.get("scene_files") or []) for s in shots]
    return {
        "num_scenes": len(scenes),
        "num_shots": len(shots),
        "num_merged_shots": sum(1 for s in shots if len(s.get("scene_files") or []) > 1),
        "avg_group_size": round(sum(sizes) / max(1, len(sizes)), 3) if sizes else 1.0,
        "max_group_size": max(sizes) if sizes else 1,
        "avg_text_coverage": round(sum(float(s.get("text_coverage") or 0.0) for s in scenes) / max(1, len(scenes)), 4),
    }


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--out", default="manifest.groups.json")

    ap.add_argument("--max-group-len", type=int, default=3)
    ap.add_argument("--max-merge-text-run", type=int, default=2)
    ap.add_argument("--merge-single-text-only", action="store_true")

    ap.add_argument("--low-text-cov", type=float, default=0.18)
    ap.add_argument("--heavy-text-cov", type=float, default=0.42)
    ap.add_argument("--short-ocr-len", type=int, default=25)
    ap.add_argument("--long-ocr-len", type=int, default=140)
    ap.add_argument("--min-keyword-overlap", type=int, default=2)

    args = ap.parse_args()

    man = load_json(args.vision_manifest)
    scenes = normalize_items(man)

    shots = group_scenes(
        scenes,
        max_group_len=max(2, int(args.max_group_len)),
        max_merge_text_run=max(1, int(args.max_merge_text_run)),
        merge_single_text_only=bool(args.merge_single_text_only),
        low_text_cov=float(args.low_text_cov),
        heavy_text_cov=float(args.heavy_text_cov),
        short_ocr_len=max(0, int(args.short_ocr_len)),
        long_ocr_len=max(10, int(args.long_ocr_len)),
        min_keyword_overlap=max(0, int(args.min_keyword_overlap)),
    )

    out_obj = {
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "grouping": {
            "method": "deterministic_consecutive_merge_only_corrected",
            "max_group_len": int(args.max_group_len),
            "max_merge_text_run": int(args.max_merge_text_run),
            "merge_single_text_only": bool(args.merge_single_text_only),
            "low_text_cov": float(args.low_text_cov),
            "heavy_text_cov": float(args.heavy_text_cov),
            "short_ocr_len": int(args.short_ocr_len),
            "long_ocr_len": int(args.long_ocr_len),
            "min_keyword_overlap": int(args.min_keyword_overlap),
        },
        "summary": build_summary(scenes, shots),
        "shots": shots,
    }

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.vision_manifest)), out_path)

    dump_json(out_path, out_obj)
    print(f"[ok] wrote={out_path} shots={len(shots)} merged_shots={out_obj['summary']['num_merged_shots']} avg_group={out_obj['summary']['avg_group_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
