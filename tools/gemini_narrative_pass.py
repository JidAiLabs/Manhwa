#!/usr/bin/env python3
"""
gemini_narrative_pass.py (429-safe)

Fixes:
- SDK-compatible Part.from_text / Part.from_bytes calls
- Uses resp.parsed when available, else robust JSON extraction
- Repair pass on parse failure
- Resume mode supported (keeps good beats, regenerates missing/errored)
- 429 RESOURCE_EXHAUSTED backoff with jitter
- Throttle between groups (min-sleep + jitter)
- Cap images per group (select lowest text_coverage panels first)
- Incremental checkpoint writes (checkpoint-every)

Requires:
  pip install -U google-genai
Auth:
  gcloud auth application-default login
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import ClientError

# Shared keep/redundant + bubble/intensity normalization (sibling tool module).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_selection import normalize_scene_selection  # noqa: E402
from usage_cost import UsageAccumulator  # noqa: E402


def _usage_from_resp(resp: Any) -> Dict[str, int]:
    """Extract exact (input, output, cached) token counts from a Gemini response."""
    um = getattr(resp, "usage_metadata", None)
    return {
        "input": int(getattr(um, "prompt_token_count", 0) or 0),
        "output": int(getattr(um, "candidates_token_count", 0) or 0),
        "cached": int(getattr(um, "cached_content_token_count", 0) or 0),
    }


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_groups(groups_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(groups_manifest.get("shots"), list):
        return groups_manifest["shots"]
    if isinstance(groups_manifest.get("groups"), list):
        return groups_manifest["groups"]
    return []


def _build_vision_map(vision_manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    items = vision_manifest.get("items") or []
    return {it.get("scene_file"): it for it in items if it.get("scene_file")}


def _pack_group_payload(group: Dict[str, Any], vision_items_by_file: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    scene_files = group.get("scene_files") or []
    scenes: List[Dict[str, Any]] = []

    for sf in scene_files:
        it = vision_items_by_file.get(sf) or {}
        v = it.get("vision") or {}
        labels = [x.get("desc") for x in (v.get("labels") or []) if x.get("desc")]
        objects = [x.get("name") for x in (v.get("objects") or []) if x.get("name")]

        scenes.append(
            {
                "scene_file": sf,
                "ocr_clean": (it.get("ocr_clean") or "")[:900],
                "text_only": bool(it.get("text_only")),
                "text_coverage": it.get("text_coverage"),
                "keywords": it.get("keywords") if isinstance(it.get("keywords"), list) else [],
                "labels": labels[:15],
                "objects": objects[:15],
            }
        )

    return {
        "group_id": int(group.get("shot_id") or group.get("group_id") or 0),
        "scene_files": scene_files,
        "scenes_signals": scenes,
        "why_merge": group.get("why_merge"),
    }


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
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, int]]:
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

    usage = _usage_from_resp(resp)
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, dict):
        return parsed, (resp.text or ""), usage

    raw = resp.text or ""
    try:
        return json.loads(raw), raw, usage
    except Exception:
        return _extract_json_object(raw), raw, usage


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
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, int]]:
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


def _select_images_for_group(
    payload: Dict[str, Any],
    vision_by_file: Dict[str, Dict[str, Any]],
    max_images: int,
) -> List[str]:
    if max_images <= 0:
        return []

    candidates: List[Tuple[float, str]] = []
    for sf in payload.get("scene_files") or []:
        it = vision_by_file.get(sf) or {}

        # NEW: skip images for scenes excluded from production
        if it.get("use_for_video") is False:
            continue

        sp = it.get("scene_path")
        if not sp:
            continue

        tc = it.get("text_coverage")
        try:
            score = float(tc) if tc is not None else 0.30
        except Exception:
            score = 0.30

        # Lower text coverage first (more visually informative)
        candidates.append((score, sp))

    candidates.sort(key=lambda x: x[0])
    img_paths = [p for _, p in candidates]
    return img_paths[:max_images]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-manifest", required=True)
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--project", required=True)
    ap.add_argument("--location", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")

    ap.add_argument("--min-sleep", type=float, default=1.2, help="Sleep between groups to avoid 429 bursts")
    ap.add_argument("--max-images-per-group", type=int, default=3, help="Cap images attached per group (0=none)")
    ap.add_argument("--backoff-max", type=float, default=60.0, help="Max seconds for 429 backoff sleep")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="Write output every N groups")

    ap.add_argument("--max-groups", type=int, default=0, help="0 = all")
    ap.add_argument("--resume", action="store_true", help="If out exists, keep good beats and only regen errors/missing")
    ap.add_argument("--retries", type=int, default=2, help="Retries per group on parse/validation failure")
    ap.add_argument("--max-output-tokens", type=int, default=2400)
    args = ap.parse_args()

    groups_m = load_json(args.groups_manifest)
    vision_m = load_json(args.vision_manifest)

    groups = _read_groups(groups_m)
    if not groups:
        raise SystemExit("No groups/shots found (expected key: shots or groups)")

    vision_by_file = _build_vision_map(vision_m)

    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    system = (
        "You are a YouTube manhwa recap story editor.\n"
        "Given consecutive scene images + OCR, produce ONE structured beat for that group.\n"
        "Be faithful to visible content.\n"
        "Avoid excessive poetic language.\n"
        "End with a strong hook line.\n"
        "Rendering hints: avoid zooming into text bubbles; focus faces/hands/key objects/wide.\n"
        "\n"
        "ALSO judge each panel for the recap video (scene_selection, one entry per scene_file):\n"
        "  role: DEFAULT to 'keep'. Only mark a panel 'redundant' when it is genuinely\n"
        "    expendable — i.e. ONE of these clearly holds:\n"
        "      (a) DUPLICATE: it shows essentially the SAME moment as another panel here (a\n"
        "          near-identical repeat, or a barely-different frame of one continuous motion); OR\n"
        "      (b) CROPPED FRAGMENT: it is a partial/cut-off version of another panel — a face or\n"
        "          body sliced at a panel edge, a thin sliver, a stitch-seam fragment.\n"
        "    For a duplicate pair, KEEP the one with the most COMPLETE framing and mark the other\n"
        "    'redundant'. Do NOT drop a panel merely for being a minor reaction, a transition, or\n"
        "    'for brevity' — distinct panels (even small ones) stay 'keep'. Most panels are 'keep';\n"
        "    only the true duplicates and cropped fragments are 'redundant'.\n"
        "  bubble_mode: the dominant speech-bubble style — 'spoken' (smooth oval, said aloud),\n"
        "    'inner_thought' (jagged/cloud, thinking), 'narration' (rectangular caption box),\n"
        "    'shout' (spiky), or 'none' if no bubble.\n"
        "  intensity: the emotional energy — 'calm', 'tense', 'intense', or 'explosive'.\n"
        "Return ONLY valid JSON matching the provided schema. No extra text.\n"
    )

    beat_schema = {
        "type": "OBJECT",
        "properties": {
            "group_id": {"type": "INTEGER"},
            "scene_files": {"type": "ARRAY", "items": {"type": "STRING"}},
            "beat_title": {"type": "STRING"},
            "what_happens": {"type": "STRING"},
            "emotional_turn": {"type": "STRING"},
            "conflict_or_stakes": {"type": "STRING"},
            "reveals_or_info": {"type": "STRING"},
            "hook": {"type": "STRING"},
            "mood_words": {"type": "ARRAY", "items": {"type": "STRING"}},
            "rendering_hints": {
                "type": "OBJECT",
                "properties": {
                    "avoid_text_zoom": {"type": "BOOLEAN"},
                    "preferred_focus": {"type": "STRING"},
                    "camera_motion": {"type": "STRING"},
                },
                "required": ["avoid_text_zoom", "preferred_focus", "camera_motion"],
            },
            "scene_selection": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "scene_file": {"type": "STRING"},
                        "role": {"type": "STRING"},          # keep | redundant
                        "bubble_mode": {"type": "STRING"},   # spoken|inner_thought|narration|shout|none
                        "intensity": {"type": "STRING"},     # calm|tense|intense|explosive
                        "reason": {"type": "STRING"},
                    },
                    "required": ["scene_file", "role", "bubble_mode", "intensity"],
                },
            },
        },
        "required": [
            "group_id",
            "scene_files",
            "beat_title",
            "what_happens",
            "emotional_turn",
            "conflict_or_stakes",
            "reveals_or_info",
            "hook",
            "mood_words",
            "rendering_hints",
            "scene_selection",
        ],
    }

    existing_by_id: Dict[int, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.out):
        try:
            existing = load_json(args.out)
            for b in (existing.get("beats") or []):
                gid = int(b.get("group_id") or 0)
                if gid and not b.get("error"):
                    existing_by_id[gid] = b
        except Exception:
            existing_by_id = {}

    max_groups = args.max_groups if args.max_groups > 0 else len(groups)

    beats_out: List[Dict[str, Any]] = []
    parse_errors = 0
    regenerated = 0
    usage = UsageAccumulator(args.model)

    def write_checkpoint() -> None:
        tmp_obj = {
            "source_groups_manifest": os.path.abspath(args.groups_manifest),
            "source_vision_manifest": os.path.abspath(args.vision_manifest),
            "model": args.model,
            "count_beats": len(beats_out),
            "stats": {"parse_errors": parse_errors, "regenerated": regenerated},
            "beats": sorted(beats_out, key=lambda x: int(x.get("group_id") or 0)),
        }
        dump_json(args.out, tmp_obj)

    for g in groups[:max_groups]:
        gid = int(g.get("shot_id") or g.get("group_id") or 0)
        if not gid:
            continue

        if gid in existing_by_id:
            beats_out.append(existing_by_id[gid])
            continue

        payload = _pack_group_payload(g, vision_by_file)
        img_paths = _select_images_for_group(payload, vision_by_file, args.max_images_per_group)

        beat: Optional[Dict[str, Any]] = None
        raw_text = ""

        for _ in range(args.retries + 1):
            obj, raw, u = _call_model_with_backoff(
                client=client,
                model=args.model,
                system_instruction=system,
                user_payload=payload,
                image_paths=img_paths,
                response_schema=beat_schema,
                max_output_tokens=args.max_output_tokens,
                temperature=0.2,
                backoff_max=args.backoff_max,
            )
            usage.add(input_tokens=u["input"], output_tokens=u["output"], cached_tokens=u.get("cached", 0))
            raw_text = raw

            # Accept any content-bearing dict; we KNOW the group_id (loop var) and
            # scene_files (payload), so stamp them ourselves rather than forcing the
            # model to echo group_id correctly — that mismatch was driving needless
            # repair retries (~70% extra calls) with no quality benefit.
            if isinstance(obj, dict) and (obj.get("what_happens") or obj.get("beat_title")):
                obj["group_id"] = gid
                obj["scene_files"] = payload["scene_files"]
                beat = obj
                break

            repair_payload = {
                "group_id": gid,
                "scene_files": payload["scene_files"],
                "last_output": (raw_text or "")[:4000],
                "instruction": "Re-output the beat as VALID JSON matching the schema exactly. No extra text.",
            }
            obj2, raw2, u2 = _call_model_with_backoff(
                client=client,
                model=args.model,
                system_instruction="You are a strict JSON formatter. Output valid JSON only.",
                user_payload=repair_payload,
                image_paths=[],
                response_schema=beat_schema,
                max_output_tokens=args.max_output_tokens,
                temperature=0.0,
                backoff_max=args.backoff_max,
            )
            usage.add(input_tokens=u2["input"], output_tokens=u2["output"], cached_tokens=u2.get("cached", 0))
            raw_text = raw2
            if isinstance(obj2, dict) and (obj2.get("what_happens") or obj2.get("beat_title")):
                obj2["group_id"] = gid
                obj2["scene_files"] = payload["scene_files"]
                beat = obj2
                break

        if beat is None:
            parse_errors += 1
            beat = {
                "group_id": gid,
                "scene_files": payload["scene_files"],
                "beat_title": "Beat",
                "what_happens": "Unable to parse model output.",
                "emotional_turn": "unknown",
                "conflict_or_stakes": "unknown",
                "reveals_or_info": "unknown",
                "hook": "Something shifts…",
                "mood_words": ["uncertain"],
                "rendering_hints": {
                    "avoid_text_zoom": True,
                    "preferred_focus": "wide",
                    "camera_motion": "slow_pan",
                },
                "scene_selection": [],
                "error": "parse_failed_after_retries",
            }

        # Guarantee exactly one sanitized selection entry per scene (defaults to
        # 'keep' so a parse gap never silently drops a panel).
        beat["scene_selection"] = normalize_scene_selection(
            beat.get("scene_selection"), payload["scene_files"]
        )
        beats_out.append(beat)

        # Throttle between groups (burst prevention)
        if args.min_sleep > 0:
            time.sleep(args.min_sleep + random.random() * 0.25)

        # Checkpoint frequently
        if args.checkpoint_every > 0 and (len(beats_out) % args.checkpoint_every == 0):
            write_checkpoint()

    beats_out.sort(key=lambda x: int(x.get("group_id") or 0))
    out_obj = {
        "source_groups_manifest": os.path.abspath(args.groups_manifest),
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "model": args.model,
        "count_beats": len(beats_out),
        "stats": {
            "parse_errors": parse_errors,
            "regenerated": regenerated,
            "usage": {
                "calls": usage.calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "est_cost_usd": round(usage.cost(), 4),
            },
        },
        "beats": beats_out,
    }
    dump_json(args.out, out_obj)
    print(f"[ok] wrote={args.out} beats={len(beats_out)} parse_errors={parse_errors} regenerated={regenerated}")
    print(usage.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
