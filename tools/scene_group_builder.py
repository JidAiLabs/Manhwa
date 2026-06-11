#!/usr/bin/env python3
"""
scene_group_builder.py (v3.1)

Deterministic grouping of panel_*.jpg into "shots" (groups) while preserving order.
Conservative: merges ONLY consecutive panels; avoids giant groups.

Inputs:
- manifest.vision.json produced by vision_extract.py (items[*])

Outputs:
- manifest.groups.json with:
  - shots: list of groups (scene_files)
  - summary: counts, avg_group_size, max_group_size

No LLM calls.
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
    labels: List[str] = []
    for x in (v.get("labels") or []):
        d = x.get("desc")
        if d:
            labels.append(str(d))
    objects: List[str] = []
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

        out.append(
            {
                "scene_id": int(it.get("scene_id") or scene_num(sf) or 0),
                "scene_file": sf,
                "scene_path": it.get("scene_path"),
                "ocr_clean": ocr,
                "ocr_len": len(ocr),
                "text_only": bool(it.get("text_only")),
                "text_coverage": float(it.get("text_coverage") or 0.0),
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
    sa = {x.lower() for x in a if isinstance(x, str)}
    sb = {x.lower() for x in b if isinstance(x, str)}
    return len(sa.intersection(sb))


def _is_low_text_scene(s: Dict[str, Any], low_text_cov: float, short_ocr_len: int) -> bool:
    return (float(s.get("text_coverage") or 0.0) <= low_text_cov) or (int(s.get("ocr_len") or 0) <= short_ocr_len)


def _is_heavy_text_scene(s: Dict[str, Any], heavy_text_cov: float, long_ocr_len: int) -> bool:
    return (float(s.get("text_coverage") or 0.0) >= heavy_text_cov) or (int(s.get("ocr_len") or 0) >= long_ocr_len)


# -----------------------------
# Grouping logic
# -----------------------------
def group_scenes(
    scenes: List[Dict[str, Any]],
    *,
    max_group_len: int = 4,
    merge_single_text_only: bool = True,
    max_merge_text_run: int = 2,  # <-- restored for CLI compat
    low_text_cov: float = 0.18,
    heavy_text_cov: float = 0.42,
    short_ocr_len: int = 25,
    long_ocr_len: int = 140,
    min_keyword_overlap: int = 2,
) -> List[Dict[str, Any]]:
    """
    Returns shots: list of groups of consecutive scene_files.

    Philosophy:
    - Hard cap group size (Gemini cost control)
    - Never cross vision_error boundaries
    - Heavy-text panels are boundaries (unless strong keyword continuity)
    - Low-text sequences can merge (action/reaction flow)
    - Short dialog adjacent to visual can merge (reaction line + action)
    - Keyword overlap can merge when neither side is heavy-text
    - max_merge_text_run caps consecutive text_only within a group
    """
    shots: List[Dict[str, Any]] = []
    i = 0
    shot_id = 1

    def mk_shot(files: List[str], ids: List[int], why: Optional[str]) -> Dict[str, Any]:
        return {"shot_id": shot_id, "scene_files": files, "scene_ids": ids, "why_merge": why}

    def can_attach(prev: Dict[str, Any], cur: Dict[str, Any]) -> Tuple[bool, str]:
        if prev.get("error") or cur.get("error"):
            return False, "vision_error_boundary"

        prev_heavy = _is_heavy_text_scene(prev, heavy_text_cov, long_ocr_len)
        cur_heavy = _is_heavy_text_scene(cur, heavy_text_cov, long_ocr_len)

        # Heavy text boundary unless strong continuity
        if prev_heavy or cur_heavy:
            ov = _kw_overlap(prev.get("keywords") or [], cur.get("keywords") or [])
            if ov >= max(3, min_keyword_overlap + 1):
                return True, "heavy_text_keyword_continuity"
            return False, "heavy_text_boundary"

        # text_only handling (conservative)
        if merge_single_text_only and (prev.get("text_only") or cur.get("text_only")):
            # attach if other side is low-text (reaction/action) or has meaningful OCR
            if _is_low_text_scene(prev, low_text_cov, short_ocr_len) or _is_low_text_scene(cur, low_text_cov, short_ocr_len):
                return True, "text_only_attached_to_visual"
            if (prev.get("ocr_len", 0) >= 15) and (cur.get("ocr_len", 0) >= 15):
                return True, "text_only_attached_to_dialog"
            return False, "text_only_boundary"

        # Low-text visual flow (action/reaction streak)
        if _is_low_text_scene(prev, low_text_cov, short_ocr_len) and _is_low_text_scene(cur, low_text_cov, short_ocr_len):
            return True, "low_text_visual_flow"

        # Visual + short dialog adjacency
        if _is_low_text_scene(prev, low_text_cov, short_ocr_len) and (cur.get("ocr_len", 0) <= max(45, short_ocr_len * 2)):
            return True, "visual_then_short_dialog"
        if _is_low_text_scene(cur, low_text_cov, short_ocr_len) and (prev.get("ocr_len", 0) <= max(45, short_ocr_len * 2)):
            return True, "short_dialog_then_visual"

        # Keyword continuity (only when both are not heavy-text, already ensured)
        ov = _kw_overlap(prev.get("keywords") or [], cur.get("keywords") or [])
        if ov >= min_keyword_overlap:
            return True, "keyword_continuity"

        return False, "no_merge_signal"

    while i < len(scenes):
        s = scenes[i]

        if s.get("error"):
            shots.append(
                {
                    "shot_id": shot_id,
                    "scene_files": [s["scene_file"]],
                    "scene_ids": [s["scene_id"]],
                    "why_merge": None,
                    "note": f"vision_error={s['error']}",
                }
            )
            shot_id += 1
            i += 1
            continue

        files = [s["scene_file"]]
        ids = [s["scene_id"]]
        why_used: Optional[str] = None

        # track consecutive text_only inside current group
        text_only_streak = 1 if s.get("text_only") else 0

        j = i + 1
        prev_scene = s
        while j < len(scenes) and len(files) < max_group_len:
            cur = scenes[j]

            ok, why = can_attach(prev_scene, cur)
            if not ok:
                break

            # cap consecutive text_only within group
            if max_merge_text_run > 0:
                if cur.get("text_only"):
                    if text_only_streak >= max_merge_text_run:
                        break
                # else allowed (reset below)

            files.append(cur["scene_file"])
            ids.append(cur["scene_id"])

            # update streak
            if cur.get("text_only"):
                text_only_streak += 1
            else:
                text_only_streak = 0

            if why_used is None:
                why_used = why

            prev_scene = cur
            j += 1

        if len(files) == 1:
            shots.append(mk_shot(files, ids, None))
        else:
            shots.append(mk_shot(files, ids, why_used or "merged"))

        shot_id += 1
        i = j

    for idx, sh in enumerate(shots, 1):
        sh["shot_id"] = idx

    return shots


def build_summary(scenes: List[Dict[str, Any]], shots: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_scenes = len(scenes)
    sizes = [len(s.get("scene_files") or []) for s in shots]
    merged_shots = [s for s in shots if len(s.get("scene_files", [])) > 1]

    return {
        "num_scenes": num_scenes,
        "num_shots": len(shots),
        "num_merged_shots": len(merged_shots),
        "num_merged_scenes": (sum(len(s["scene_files"]) for s in merged_shots) - len(merged_shots)),
        "avg_text_coverage": round(sum(float(s.get("text_coverage") or 0.0) for s in scenes) / max(1, num_scenes), 4),
        "avg_group_size": round(sum(sizes) / max(1, len(sizes)), 3) if sizes else 1.0,
        "max_group_size": max(sizes) if sizes else 1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-manifest", required=True, help="Path to manifest.vision.json")
    ap.add_argument("--out", default="manifest.groups.json", help="Output JSON path")
    ap.add_argument("--max-group-len", type=int, default=4)

    ap.add_argument("--merge-single-text-only", action="store_true")
    ap.add_argument("--max-merge-text-run", type=int, default=2, help="Max consecutive text_only allowed inside a group")

    ap.add_argument("--keep-chrome", action="store_true",
                    help="keep publication chrome scenes (publisher logos, cover/"
                         "title pages, chapter-number cards, view counters). By "
                         "default chrome is EXCLUDED before grouping so it is "
                         "never narrated or shown.")
    ap.add_argument("--series-title", default="",
                    help="series title for cover/title-page chrome detection")
    ap.add_argument("--low-text-cov", type=float, default=0.18)
    ap.add_argument("--heavy-text-cov", type=float, default=0.42)
    ap.add_argument("--short-ocr-len", type=int, default=25)
    ap.add_argument("--long-ocr-len", type=int, default=140)
    ap.add_argument("--min-keyword-overlap", type=int, default=2)

    args = ap.parse_args()

    man = load_json(args.vision_manifest)
    scenes = normalize_items(man)

    chrome_dropped: List[str] = []
    if not args.keep_chrome:
        from scene_chrome import is_chrome_scene, needs_image_stats

        def _midtone(s: Dict[str, Any]) -> Optional[float]:
            """Midtone fraction for OCR-blind chrome (stylized number cards)
            and watermark-vs-cover disambiguation (single site hit). Computed
            only for those signatures — a few image reads per chapter."""
            if (not needs_image_stats(str(s.get("ocr_clean") or ""))
                    or not s.get("scene_path")):
                return None
            try:
                from PIL import Image
                import numpy as np
                im = np.asarray(Image.open(s["scene_path"]).convert("L"))
                return float(((im > 60) & (im < 200)).mean())
            except Exception:
                return None

        keep: List[Dict[str, Any]] = []
        for s in scenes:
            if is_chrome_scene(s, series_title=args.series_title or None,
                               midtone_frac=_midtone(s)):
                chrome_dropped.append(str(s["scene_file"]))
            else:
                keep.append(s)
        scenes = keep
        if chrome_dropped:
            print(f"[chrome] excluded {len(chrome_dropped)}: {chrome_dropped}")

    shots = group_scenes(
        scenes,
        max_group_len=max(2, int(args.max_group_len)),
        merge_single_text_only=bool(args.merge_single_text_only),
        max_merge_text_run=max(0, int(args.max_merge_text_run)),
        low_text_cov=float(args.low_text_cov),
        heavy_text_cov=float(args.heavy_text_cov),
        short_ocr_len=max(0, int(args.short_ocr_len)),
        long_ocr_len=max(10, int(args.long_ocr_len)),
        min_keyword_overlap=max(0, int(args.min_keyword_overlap)),
    )

    out_obj = {
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "chrome_excluded": chrome_dropped,
        "grouping": {
            "method": "deterministic_consecutive_merge_only_v3_1",
            "max_group_len": max(2, int(args.max_group_len)),
            "merge_single_text_only": bool(args.merge_single_text_only),
            "max_merge_text_run": max(0, int(args.max_merge_text_run)),
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
    if not os.path.isabs(out_path) and not os.path.dirname(out_path):
        # bare filename → place it next to the vision manifest. Paths WITH a
        # directory part are respected as given: the old unconditional join
        # doubled relative paths (<ep>/<ep>/manifest.groups.json) and silently
        # left the real manifest stale.
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.vision_manifest)), out_path)

    dump_json(out_path, out_obj)
    s = out_obj["summary"]
    print(f"[ok] wrote={out_path} shots={s['num_shots']} merged_shots={s['num_merged_shots']} avg_group={s['avg_group_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
