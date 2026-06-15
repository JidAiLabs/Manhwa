#!/usr/bin/env python3
"""story_group.py — Pass 2 of the understanding-first pipeline.

Group panels by UNDERSTANDING, not by gutters. Reads the per-panel descriptions
(Pass 1, panel_understand) in reading order and segments them into STORY BEATS:
a beat = a run of CONSECUTIVE panels that form one moment/shot. New beat at a
scene/location change, a time jump, a flashback start/end, or a topic shift;
near-identical consecutive panels merge into one montage beat. Each beat is
tagged segment (present|flashback|dream) + arc_label — so flashbacks are native.

This REPLACES the position/threshold grouping in scene_group_builder.py. Output
is a byte-compatible manifest.groups.json: top-level `shots` with per-shot
`shot_id` (contiguous int) + `scene_files`, plus extra `segment`/`arc_label`
tags that downstream ignores safely (timeline can carry the flashback tag).

Coverage is an invariant: every non-chrome panel lands in exactly one shot
(repair_to_shots reconstructs a consecutive partition from the model's intent).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from gemini_narrative_pass import (                                   # noqa: E402
    load_json, dump_json, _call_model_with_backoff)

_SEGMENTS = ("present", "flashback", "dream")

GROUP_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "beats": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "scene_files": {"type": "ARRAY", "items": {"type": "STRING"}},
                "segment": {"type": "STRING", "enum": list(_SEGMENTS)},
                "arc_label": {"type": "STRING"},
                "why": {"type": "STRING"},
            },
            "required": ["scene_files"],
        }},
    },
    "required": ["beats"],
}

SYSTEM = (
    "You are a manhwa recap editor. You get a numbered sequence of panel "
    "descriptions from ONE chapter, in reading order. Segment them into STORY "
    "BEATS for the recap.\n"
    "A beat = a run of CONSECUTIVE panels that form one moment/shot. Start a NEW "
    "beat at: a scene or location change, a time jump, a FLASHBACK start OR end, "
    "or a clear topic shift. Group near-identical consecutive panels (e.g. a "
    "multi-panel action or a slow reveal) into ONE beat.\n"
    "For each beat return: scene_files (its consecutive panels, in order), "
    "segment (present | flashback | dream — MARK flashbacks and dreams), "
    "arc_label (a 2-4 word label for the scene). Cover EVERY panel exactly once, "
    "in order. Prefer MORE, tighter beats over a few large ones — each beat "
    "becomes one narration line, so it should be one clear moment."
)


def _norm_segment(s: Any) -> str:
    s = str(s or "").strip().lower()
    return s if s in _SEGMENTS else "present"


def build_grouping_payload(panels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure: the numbered, ordered description sequence the grouper reasons over."""
    return {"panels": [{
        "n": i,
        "scene_file": p.get("scene_file"),
        "description": (p.get("description") or "")[:300],
        "action": (p.get("action") or "")[:160],
        "setting": (p.get("setting") or "")[:80],
        "dialogue": (p.get("dialogue") or "")[:160],
        "intensity": p.get("intensity") or "",
    } for i, p in enumerate(panels)]}


def repair_to_shots(scene_order: List[str], model_beats: List[Dict[str, Any]],
                    *, max_beat_len: int = 4) -> List[Dict[str, Any]]:
    """Pure + robust: reconstruct a CONSECUTIVE partition of scene_order from the
    model's grouping intent — guarantees coverage (every panel in exactly one
    shot, in order) no matter how the model mis-orders/omits. A new shot starts
    when the model's beat changes OR the run hits max_beat_len. Unassigned panels
    continue the current beat (never dropped)."""
    assign: Dict[str, tuple] = {}
    for bi, b in enumerate(model_beats or []):
        seg, arc = _norm_segment(b.get("segment")), str(b.get("arc_label") or "").strip()
        for sf in (b.get("scene_files") or []):
            assign.setdefault(str(sf), (bi, seg, arc))

    shots: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for sf in scene_order:
        info = assign.get(sf)
        if info is not None:
            bi, seg, arc = info
        elif cur is not None:                       # unassigned → continue beat
            bi, seg, arc = cur["_bi"], cur["segment"], cur["arc_label"]
        else:
            bi, seg, arc = -1, "present", ""
        if (cur is None or bi != cur["_bi"]
                or len(cur["scene_files"]) >= max_beat_len):
            cur = {"_bi": bi, "scene_files": [], "segment": seg, "arc_label": arc}
            shots.append(cur)
        cur["scene_files"].append(sf)

    return [{"shot_id": i, "scene_files": s["scene_files"],
             "segment": s["segment"], "arc_label": s["arc_label"]}
            for i, s in enumerate(shots, 1)]


def group_panels(panels: List[Dict[str, Any]], call_fn: Callable[..., Any],
                 *, max_beat_len: int = 4) -> List[Dict[str, Any]]:
    """Group the (non-chrome, ordered) panels into story shots. `call_fn(payload)
    -> parsed dict|None` is injected (real model, or stub in tests)."""
    if not panels:
        return []
    parsed = call_fn(build_grouping_payload(panels))
    beats = (parsed or {}).get("beats") if isinstance(parsed, dict) else None
    return repair_to_shots([p.get("scene_file") for p in panels],
                           beats or [], max_beat_len=max_beat_len)


def _midtone(item: Dict[str, Any]) -> Optional[float]:
    from scene_chrome import needs_image_stats
    if (not needs_image_stats(str(item.get("ocr_clean") or ""))
            or not item.get("scene_path")):
        return None
    try:
        from PIL import Image
        import numpy as np
        im = np.asarray(Image.open(item["scene_path"]).convert("L"))
        return float(((im > 60) & (im < 200)).mean())
    except Exception:
        return None


def chrome_files(vision_items: List[Dict[str, Any]], series_title: str) -> set:
    from scene_chrome import is_chrome_scene
    out = set()
    for it in vision_items:
        sf = it.get("scene_file")
        if sf and is_chrome_scene(it, series_title=series_title or None,
                                  midtone_frac=_midtone(it)):
            out.add(sf)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--understood", required=True,
                    help="manifest.panels.understood.json (Pass 1 output)")
    ap.add_argument("--vision-manifest", required=True,
                    help="for chrome exclusion + scene paths")
    ap.add_argument("--out", required=True, help="manifest.groups.json")
    ap.add_argument("--series-title", default="", help="chrome BAN (cover/title)")
    ap.add_argument("--backend", choices=["vertex", "ollama"], default="ollama")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="")
    ap.add_argument("--max-beat-len", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--keep-chrome", action="store_true")
    args = ap.parse_args()

    understood = load_json(args.understood)
    vision = load_json(args.vision_manifest)
    panels = [p for p in (understood.get("panels") or []) if p.get("scene_file")]

    vmap = {it.get("scene_file"): it for it in (vision.get("items") or [])}
    chrome = set() if args.keep_chrome else chrome_files(
        list(vmap.values()), args.series_title)
    if chrome:
        print(f"[chrome] excluded {len(chrome)}: {sorted(chrome)}")
    story = [p for p in panels if p.get("scene_file") not in chrome]

    client = None
    model = args.ollama_model
    if args.backend == "vertex":
        from google import genai
        if not args.project or not args.location:
            raise SystemExit("--project/--location required for --backend vertex")
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)
        model = args.model

    def call_fn(payload: Dict[str, Any]):
        parsed, _raw, _usage = _call_model_with_backoff(
            client=client, model=model, system_instruction=SYSTEM,
            user_payload=payload, image_paths=[], response_schema=GROUP_SCHEMA,
            max_output_tokens=2400, temperature=args.temperature,
            backoff_max=60.0, backend=args.backend)
        return parsed

    shots = group_panels(story, call_fn, max_beat_len=args.max_beat_len)
    out = {
        "source_understood": os.path.abspath(args.understood),
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "chrome_excluded": sorted(chrome),
        "grouping": {"method": "understanding_first_v1",
                     "max_beat_len": args.max_beat_len},
        "summary": {"num_scenes": len(story), "num_shots": len(shots),
                    "flashback_shots": sum(1 for s in shots
                                           if s["segment"] != "present")},
        "shots": shots,
    }
    dump_json(args.out, out)
    print(f"[ok] wrote={args.out} scenes={len(story)} shots={len(shots)} "
          f"(was position-grouped; now story-grouped) chrome={len(chrome)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
