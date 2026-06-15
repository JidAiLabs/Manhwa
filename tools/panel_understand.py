#!/usr/bin/env python3
"""panel_understand.py — Pass 1 of the understanding-first pipeline.

Describe EVERY panel (multimodal): what is literally happening, who is in it,
the dialogue, the setting, the intensity. One record per panel = **full
coverage by construction** — nothing can be merged or dropped before it has been
understood. This output feeds the story-grouper (Pass 2, which segments the
sequence into story-sized beats + flashback boundaries) and the per-beat
narrator (Pass 3).

It reuses the battle-tested multimodal call from gemini_narrative_pass
(`_call_model_with_backoff`: ollama/Gemma or Vertex, schema-constrained, 429-safe).

Out: manifest.panels.understood.json = {panels:[{scene_file, description,
subjects[], action, dialogue, setting, intensity}]}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from gemini_narrative_pass import (                                   # noqa: E402
    load_json, dump_json, _call_model_with_backoff)

# Gemini-style schema (UPPERCASE enums) — _call_model converts it for Ollama.
PANEL_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "description": {"type": "STRING"},
        "subjects": {"type": "ARRAY", "items": {"type": "STRING"}},
        "action": {"type": "STRING"},
        "dialogue": {"type": "STRING"},
        "setting": {"type": "STRING"},
        "intensity": {"type": "STRING",
                      "enum": ["calm", "tense", "intense", "explosive"]},
    },
    "required": ["description", "action", "intensity"],
}

SYSTEM = (
    "You are a manhwa recap analyst. You see ONE webtoon panel image plus its "
    "OCR text. Describe what is LITERALLY happening in this panel — specific and "
    "vivid, but strictly faithful to what is shown (never invent characters or "
    "events). Return JSON:\n"
    "  description: 1-2 concrete sentences of the action/scene in this panel.\n"
    "  subjects: the characters / creatures / key objects visible.\n"
    "  action: the single key event or beat of this panel.\n"
    "  dialogue: any spoken line or caption, copied VERBATIM from the OCR; '' if "
    "none. Do not paraphrase dialogue.\n"
    "  setting: where/what the scene is (a train, a city street, a flashback "
    "screen, etc.).\n"
    "  intensity: calm | tense | intense | explosive.\n"
    "The 'previous_panels' field is context for continuity only — describe THIS "
    "panel, not the previous ones."
)


def build_payload(panel: Dict[str, Any], prev_descs: List[str]) -> Dict[str, Any]:
    """Pure: the per-panel model input (OCR + cheap vision signals + rolling
    context for continuity). Image is attached separately by the caller."""
    v = panel.get("vision") or {}
    labels = [x.get("desc") for x in (v.get("labels") or []) if x.get("desc")]
    objects = [x.get("name") for x in (v.get("objects") or []) if x.get("name")]
    return {
        "scene_file": panel.get("scene_file"),
        "ocr": (panel.get("ocr_clean") or "")[:900],
        "labels": labels[:12],
        "objects": objects[:12],
        "previous_panels": [d for d in prev_descs[-2:] if d],
    }


def assemble_record(scene_file: str, parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure: normalize one model result into a panel record. A parse failure is
    recorded (never silently dropped) so resume can re-run just that panel."""
    if not isinstance(parsed, dict):
        return {"scene_file": scene_file, "description": "", "subjects": [],
                "action": "", "dialogue": "", "setting": "",
                "intensity": "unknown", "error": "parse_failed"}
    inten = str(parsed.get("intensity") or "").lower()
    return {
        "scene_file": scene_file,
        "description": str(parsed.get("description") or "").strip(),
        "subjects": [str(s) for s in (parsed.get("subjects") or []) if s],
        "action": str(parsed.get("action") or "").strip(),
        "dialogue": str(parsed.get("dialogue") or "").strip(),
        "setting": str(parsed.get("setting") or "").strip(),
        "intensity": inten if inten in
        ("calm", "tense", "intense", "explosive") else "unknown",
    }


def understand_panels(items: List[Dict[str, Any]], call_fn: Callable[..., Any],
                      *, log: Callable[[str], None] = lambda _m: None,
                      prior: Optional[Dict[str, Dict[str, Any]]] = None
                      ) -> List[Dict[str, Any]]:
    """Describe each panel in order, threading rolling context. `call_fn(payload,
    image_path) -> parsed dict|None` is injected (the real model, or a stub in
    tests). `prior` (scene_file -> good record) lets --resume skip done panels."""
    prior = prior or {}
    out: List[Dict[str, Any]] = []
    prev_descs: List[str] = []
    for it in items:
        sf = it.get("scene_file")
        if not sf:
            continue
        done = prior.get(sf)
        if done and done.get("description") and not done.get("error"):
            out.append(done)
            prev_descs.append(done.get("description", ""))
            continue
        payload = build_payload(it, prev_descs)
        parsed = call_fn(payload, it.get("scene_path"))
        rec = assemble_record(sf, parsed)
        if rec.get("error"):
            log(f"[panel] {sf}: parse failed")
        out.append(rec)
        prev_descs.append(rec.get("description", ""))
    return out


def _scene_items_in_order(vision: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = [it for it in (vision.get("items") or []) if it.get("scene_file")]
    items.sort(key=lambda it: (int(it.get("scene_id") or 0),
                               str(it.get("scene_file"))))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vertex", "ollama"], default="ollama")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-output-tokens", type=int, default=400)
    ap.add_argument("--resume", action="store_true",
                    help="keep good panel records in --out, redo only failures")
    args = ap.parse_args()

    vision = load_json(args.vision_manifest)
    items = _scene_items_in_order(vision)
    if not items:
        raise SystemExit("no vision items (expected key: items)")

    client = None
    model = args.ollama_model
    if args.backend == "vertex":
        from google import genai
        if not args.project or not args.location:
            raise SystemExit("--project/--location required for --backend vertex")
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)
        model = args.model

    prior: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.out):
        try:
            prior = {p.get("scene_file"): p for p in
                     (load_json(args.out).get("panels") or [])
                     if p.get("scene_file")}
        except Exception:
            prior = {}

    def call_fn(payload: Dict[str, Any], scene_path: Optional[str]):
        parsed, _raw, _usage = _call_model_with_backoff(
            client=client, model=model, system_instruction=SYSTEM,
            user_payload=payload, image_paths=[scene_path] if scene_path else [],
            response_schema=PANEL_SCHEMA, max_output_tokens=args.max_output_tokens,
            temperature=args.temperature, backoff_max=60.0, backend=args.backend)
        return parsed

    panels = understand_panels(items, call_fn,
                               log=lambda m: print(m, flush=True), prior=prior)
    dump_json(args.out, {
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "model": model, "count": len(panels), "panels": panels})
    ok = sum(1 for p in panels if p.get("description") and not p.get("error"))
    print(f"[ok] wrote={args.out} panels={len(panels)} understood={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
