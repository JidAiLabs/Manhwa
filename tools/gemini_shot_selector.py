#!/usr/bin/env python3
"""
gemini_shot_selector.py (Vertex AI / Gemini, 429-safe)

Purpose
- Takes smart_cropper output (manifest.smartcrop.json) + vision manifest (manifest.vision.json)
- For each scene, sends candidate shot images + vision signals to Gemini
- Gemini returns:
    - which shots to KEEP (vs redundant/empty)
    - per-shot short narrative (for later video)
    - optional reason + score

Design goals
- Reuse the same robust structure as gemini_narrative_pass.py:
  - Part.from_text / Part.from_bytes compatibility
  - resp.parsed when available; fallback JSON extraction
  - Repair pass on parse failure
  - 429 RESOURCE_EXHAUSTED exponential backoff + jitter
  - Resume mode (keep good results; regen missing/errored)
  - Incremental checkpoint writes

Requirements
  pip install -U google-genai pillow
Auth
  gcloud auth application-default login

Example
  python3 gemini_shot_selector.py \
    --shots-manifest "/path/to/shots/manifest.smartcrop.json" \
    --vision-manifest "/path/to/manifest.vision.json" \
    --out "/path/to/shots/manifest.smartcrop.selected.json" \
    --project "<GCP_PROJECT>" \
    --location "us-central1" \
    --model "gemini-2.5-flash" \
    --min-sleep 1.2 \
    --max-images-per-scene 3 \
    --resume \
    --checkpoint-every 10
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import ClientError


# ----------------------------
# IO helpers
# ----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ----------------------------
# Robust JSON extraction fallback
# ----------------------------
def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    candidate = text[s : e + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


# ----------------------------
# SDK-safe Part builders
# ----------------------------
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


# ----------------------------
# Model calls (+ 429 backoff)
# ----------------------------
def _call_model(
    *,
    client: genai.Client,
    model: str,
    system_instruction: str,
    user_payload: Dict[str, Any],
    image_paths: List[str],
    response_schema: Dict[str, Any],
    max_output_tokens: int,
    temperature: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    parts: List[types.Part] = []
    parts.append(_part_text("INPUT_JSON:\n" + json.dumps(user_payload, ensure_ascii=False)))

    for p in image_paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            parts.append(_part_image_jpeg(f.read()))

    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
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
        return _extract_json_object(raw), raw


def _call_model_with_backoff(
    *,
    client: genai.Client,
    model: str,
    system_instruction: str,
    user_payload: Dict[str, Any],
    image_paths: List[str],
    response_schema: Dict[str, Any],
    max_output_tokens: int,
    temperature: float,
    backoff_max: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    attempt = 0
    while True:
        try:
            return _call_model(
                client=client,
                model=model,
                system_instruction=system_instruction,
                user_payload=user_payload,
                image_paths=image_paths,
                response_schema=response_schema,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except ClientError as e:
            msg = str(e)
            if ("429" not in msg) and ("RESOURCE_EXHAUSTED" not in msg):
                raise
            sleep_s = min(backoff_max, (2 ** min(attempt, 6)) + random.random() * 0.8)
            print(f"[warn] 429 RESOURCE_EXHAUSTED. sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)
            attempt += 1


# ----------------------------
# Manifest utils
# ----------------------------
def _build_vision_map(vision_manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Keyed by scene_file.
    vision_manifest format: {"items": [{"scene_file": "...", "scene_path": "...", "vision": {...}, ...}, ...]}
    """
    items = vision_manifest.get("items") or []
    return {it.get("scene_file"): it for it in items if it.get("scene_file")}


def _read_smartcrop_items(shots_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    smart_cropper output: {"items": [{"scene_id":..,"scene_file":..,"shots":[...]} ...]}
    """
    items = shots_manifest.get("items") or []
    return items if isinstance(items, list) else []


def _scene_signals_from_vision(vision_item: Dict[str, Any]) -> Dict[str, Any]:
    v = vision_item.get("vision") or {}
    labels = [x.get("desc") for x in (v.get("labels") or []) if isinstance(x, dict) and x.get("desc")]
    objects = [x.get("name") for x in (v.get("objects") or []) if isinstance(x, dict) and x.get("name")]

    return {
        "ocr_clean": (vision_item.get("ocr_clean") or "")[:1200],
        "text_only": bool(vision_item.get("text_only")),
        "text_coverage": vision_item.get("text_coverage"),
        "keywords": vision_item.get("keywords") if isinstance(vision_item.get("keywords"), list) else [],
        "labels": labels[:20],
        "objects": objects[:20],
        # Optional: pass text blocks / ocr words if present (can help the model)
        "text_blocks": (v.get("text_blocks") or [])[:50],
        "ocr_words": (v.get("ocr_words") or [])[:250],
    }


def _shot_payload_from_scene(scene_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for idx, sh in enumerate(scene_entry.get("shots") or []):
        out.append(
            {
                "shot_index": idx,
                "shot_file": sh.get("shot_file"),
                "shot_path": sh.get("shot_path"),
                "bbox_px": sh.get("bbox_px"),
                "bbox_norm": sh.get("bbox_norm"),
            }
        )
    return out


def _select_shot_images(scene_entry: Dict[str, Any], max_images: int) -> List[str]:
    """
    Attach shot images. Usually shots <= 3. If more, take first N by y order.
    (We keep this deterministic; Gemini decides keep/drop.)
    """
    shots = scene_entry.get("shots") or []
    if not shots or max_images <= 0:
        return []
    # already in y-order in smartcrop output; but ensure stable
    paths = [s.get("shot_path") for s in shots if s.get("shot_path")]
    return paths[:max_images]


# ----------------------------
# Merge model output back into manifest
# ----------------------------
def _apply_decision_to_scene(scene_entry: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds per-shot:
      - use_for_video (bool)
      - narrative (str)
      - score (float)
      - reason (str)
    Keeps original shot fields intact.
    """
    shots = scene_entry.get("shots") or []
    dec_shots = decision.get("shots") or []

    # index by shot_index
    by_idx: Dict[int, Dict[str, Any]] = {}
    for ds in dec_shots:
        try:
            i = int(ds.get("shot_index"))
        except Exception:
            continue
        by_idx[i] = ds

    # must keep at least one shot (hard guard)
    any_keep = False
    for i in range(len(shots)):
        ds = by_idx.get(i) or {}
        keep = bool(ds.get("keep")) if "keep" in ds else False
        if keep:
            any_keep = True
            break

    if not any_keep and shots:
        # if model dropped everything, force-keep the first shot
        first = by_idx.get(0) or {}
        first["keep"] = True
        first.setdefault("reason", "forced_keep_at_least_one")
        first.setdefault("score", 0.50)
        first.setdefault("narrative", "")
        by_idx[0] = first

    # apply
    for i, sh in enumerate(shots):
        ds = by_idx.get(i) or {}
        sh["use_for_video"] = bool(ds.get("keep")) if "keep" in ds else False
        sh["score"] = float(ds.get("score")) if ds.get("score") is not None else None
        sh["reason"] = ds.get("reason") if isinstance(ds.get("reason"), str) else ""
        sh["narrative"] = ds.get("narrative") if isinstance(ds.get("narrative"), str) else ""

    scene_entry["shots"] = shots
    return scene_entry


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--shots-manifest", required=True, help="manifest.smartcrop.json from smart_cropper.py")
    ap.add_argument("--vision-manifest", required=True, help="manifest.vision.json")
    ap.add_argument("--out", required=True, help="manifest.smartcrop.selected.json output path")

    ap.add_argument("--project", required=True)
    ap.add_argument("--location", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")

    ap.add_argument("--min-sleep", type=float, default=1.2, help="Sleep between scenes to avoid 429 bursts")
    ap.add_argument("--max-images-per-scene", type=int, default=3, help="Cap shot images attached per scene (0=none)")
    ap.add_argument("--backoff-max", type=float, default=60.0, help="Max seconds for 429 backoff sleep")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="Write output every N scenes")

    ap.add_argument("--max-scenes", type=int, default=0, help="0 = all")
    ap.add_argument("--resume", action="store_true", help="If out exists, keep good decisions and only regen missing/errored")
    ap.add_argument("--retries", type=int, default=2, help="Retries per scene on parse/validation failure")
    ap.add_argument("--max-output-tokens", type=int, default=1400)
    args = ap.parse_args()

    shots_m = load_json(args.shots_manifest)
    vision_m = load_json(args.vision_manifest)

    scenes = _read_smartcrop_items(shots_m)
    if not scenes:
        raise SystemExit("No scenes found in shots manifest (expected key: items)")

    vision_by_file = _build_vision_map(vision_m)

    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    # System prompt tuned for your use-case: decide keep/drop + short narrative per kept shot.
    system = (
        "You are a manhwa recap editor AND shot selector.\n"
        "Input: a single scene split into candidate vertical shots, plus OCR/vision signals.\n"
        "Task:\n"
        "1) Decide which shots are WORTH KEEPING for a recap video.\n"
        "   - Drop redundant shots that are mostly empty background, transition gradients, or tiny fragments.\n"
        "   - Keep multiple shots only if each adds distinct visual story content.\n"
        "   - Must keep at least ONE shot.\n"
        "2) For each kept shot, write a short narrative line (1–2 sentences) describing what is visible.\n"
        "   - Be faithful to the image.\n"
        "   - Do NOT transcribe dialogue bubbles; summarize instead.\n"
        "3) Output ONLY valid JSON matching the schema.\n"
    )

    # Response schema: decision per shot
    decision_schema = {
        "type": "OBJECT",
        "properties": {
            "scene_id": {"type": "INTEGER"},
            "scene_file": {"type": "STRING"},
            "shots": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "shot_index": {"type": "INTEGER"},
                        "keep": {"type": "BOOLEAN"},
                        "score": {"type": "NUMBER"},  # 0..1
                        "reason": {"type": "STRING"},
                        "narrative": {"type": "STRING"},
                    },
                    "required": ["shot_index", "keep", "score", "reason", "narrative"],
                },
            },
        },
        "required": ["scene_id", "scene_file", "shots"],
    }

    # Resume support
    existing_by_scene_id: Dict[int, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.out):
        try:
            existing = load_json(args.out)
            for it in (existing.get("items") or []):
                sid = int(it.get("scene_id") or 0)
                # treat as good if no error and at least one shot use_for_video is True
                good = False
                for sh in (it.get("shots") or []):
                    if sh.get("use_for_video") is True:
                        good = True
                        break
                if sid and good and not it.get("error"):
                    existing_by_scene_id[sid] = it
        except Exception:
            existing_by_scene_id = {}

    max_scenes = args.max_scenes if args.max_scenes > 0 else len(scenes)

    out_items: List[Dict[str, Any]] = []
    parse_errors = 0
    regenerated = 0

    def write_checkpoint() -> None:
        tmp_obj = {
            "source_shots_manifest": os.path.abspath(args.shots_manifest),
            "source_vision_manifest": os.path.abspath(args.vision_manifest),
            "model": args.model,
            "count_scenes": len(out_items),
            "stats": {"parse_errors": parse_errors, "regenerated": regenerated},
            "items": sorted(out_items, key=lambda x: int(x.get("scene_id") or 0)),
        }
        dump_json(args.out, tmp_obj)

    for scene_entry in scenes[:max_scenes]:
        scene_id = int(scene_entry.get("scene_id") or 0)
        scene_file = scene_entry.get("scene_file") or ""
        if not scene_id or not scene_file:
            continue

        if scene_id in existing_by_scene_id:
            out_items.append(existing_by_scene_id[scene_id])
            continue

        vision_item = vision_by_file.get(scene_file) or {}
        signals = _scene_signals_from_vision(vision_item)
        shots_payload = _shot_payload_from_scene(scene_entry)
        img_paths = _select_shot_images(scene_entry, args.max_images_per_scene)

        user_payload = {
            "scene_id": scene_id,
            "scene_file": scene_file,
            "scene_path": scene_entry.get("scene_path"),
            "scene_size": {"width": scene_entry.get("width"), "height": scene_entry.get("height")},
            "vision_signals": signals,
            "candidate_shots": shots_payload,
            "rules": {
                "must_keep_at_least_one": True,
                "drop_redundant_empty_background": True,
                "avoid_dialogue_transcription": True,
                "narrative_length": "1-2 sentences per kept shot",
            },
        }

        decision: Optional[Dict[str, Any]] = None
        raw_text = ""

        for attempt in range(args.retries + 1):
            obj, raw = _call_model_with_backoff(
                client=client,
                model=args.model,
                system_instruction=system,
                user_payload=user_payload,
                image_paths=img_paths,
                response_schema=decision_schema,
                max_output_tokens=args.max_output_tokens,
                temperature=0.2,
                backoff_max=args.backoff_max,
            )
            raw_text = raw

            if isinstance(obj, dict) and int(obj.get("scene_id") or 0) == scene_id:
                # basic validation: has shots
                if isinstance(obj.get("shots"), list) and obj["shots"]:
                    decision = obj
                    break

            # Repair pass: strict json formatter
            repair_payload = {
                "scene_id": scene_id,
                "scene_file": scene_file,
                "last_output": (raw_text or "")[:4000],
                "instruction": "Re-output the decision as VALID JSON matching the schema exactly. No extra text.",
            }
            obj2, raw2 = _call_model_with_backoff(
                client=client,
                model=args.model,
                system_instruction="You are a strict JSON formatter. Output valid JSON only.",
                user_payload=repair_payload,
                image_paths=[],
                response_schema=decision_schema,
                max_output_tokens=args.max_output_tokens,
                temperature=0.0,
                backoff_max=args.backoff_max,
            )
            raw_text = raw2
            if isinstance(obj2, dict) and int(obj2.get("scene_id") or 0) == scene_id:
                if isinstance(obj2.get("shots"), list) and obj2["shots"]:
                    decision = obj2
                    regenerated += 1
                    break

        if decision is None:
            parse_errors += 1
            # fallback: keep first shot, no narrative
            fallback = dict(scene_entry)
            for i, sh in enumerate(fallback.get("shots") or []):
                sh["use_for_video"] = (i == 0)
                sh["score"] = 0.50 if i == 0 else 0.0
                sh["reason"] = "fallback_parse_failed"
                sh["narrative"] = ""
            fallback["error"] = "parse_failed_after_retries"
            out_items.append(fallback)
        else:
            enriched = dict(scene_entry)
            enriched = _apply_decision_to_scene(enriched, decision)
            out_items.append(enriched)

        # Throttle between scenes (burst prevention)
        if args.min_sleep > 0:
            time.sleep(args.min_sleep + random.random() * 0.25)

        # Checkpoint frequently
        if args.checkpoint_every > 0 and (len(out_items) % args.checkpoint_every == 0):
            write_checkpoint()

        kept_n = 0
        for sh in (out_items[-1].get("shots") or []):
            if sh.get("use_for_video") is True:
                kept_n += 1
        print(f"[scene {scene_id}] shots={len(scene_entry.get('shots') or [])} kept={kept_n}")

    out_items.sort(key=lambda x: int(x.get("scene_id") or 0))
    out_obj = {
        "source_shots_manifest": os.path.abspath(args.shots_manifest),
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "model": args.model,
        "count_scenes": len(out_items),
        "stats": {"parse_errors": parse_errors, "regenerated": regenerated},
        "items": out_items,
    }
    dump_json(args.out, out_obj)
    print(f"[ok] wrote={args.out} scenes={len(out_items)} parse_errors={parse_errors} regenerated={regenerated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
