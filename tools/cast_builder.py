"""
tools/cast_builder.py — build a chapter CAST registry so recap narration names the
same character consistently instead of re-introducing "a figure" every scene.

One Gemini multimodal pass over all OCR + a sampling of panels -> manifest.cast.json:
  {cast: [{id, canonical_name, aliases[], role, visual_description, is_protagonist}]}

Names come from the dialogue/OCR (proper names or address terms like "Ancestor-nim");
unnamed-but-recurring figures get a stable descriptive handle. gemini_narrative_pass
then threads this in (--cast) so every group's narration matches faces to the cast.

Auth mirrors gemini_narrative_pass (Vertex AI via the gcp-vision SA key).

  V=.eval_venv/bin/python
  $V tools/cast_builder.py \
      --groups-manifest  ongoing/nano-machine/Chapter_1/manifest.groups.json \
      --vision-manifest  ongoing/nano-machine/Chapter_1/manifest.vision.json \
      --out              ongoing/nano-machine/Chapter_1/manifest.cast.json \
      --project <proj> --location us-central1
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _vision_items(vision: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = vision.get("items") or vision.get("scenes") or []
    if isinstance(items, dict):
        items = list(items.values())
    return items


def _sample_images(items: List[Dict[str, Any]], cap: int) -> List[str]:
    """Pick visually-informative panels spread across the chapter (low text coverage
    first within an even stride), returning on-disk scene_paths."""
    withpath = [it for it in items if it.get("scene_path")]
    if not withpath:
        return []
    # even stride across the chapter so we see characters from start to end
    stride = max(1, len(withpath) // max(1, cap))
    picked = withpath[::stride][:cap]

    def tc(it: Dict[str, Any]) -> float:
        try:
            return float(it.get("text_coverage")) if it.get("text_coverage") is not None else 0.3
        except Exception:
            return 0.3
    picked.sort(key=tc)  # most visual first
    return [str(it["scene_path"]) for it in picked]


def _all_ocr(items: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for it in items:
        o = (it.get("ocr_clean") or "").strip()
        if o:
            out.append(f"{it.get('scene_file')}: {o[:200]}")
    return out


CAST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "cast": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "canonical_name": {"type": "STRING"},
                    "aliases": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "role": {"type": "STRING"},
                    "visual_description": {"type": "STRING"},
                    "is_protagonist": {"type": "BOOLEAN"},
                },
                "required": ["id", "canonical_name", "role", "visual_description", "is_protagonist"],
            },
        }
    },
    "required": ["cast"],
}

SYSTEM = (
    "You are building a CAST LIST for ONE webtoon/manhwa chapter so a recap video can name "
    "the same character consistently instead of calling them 'a figure' every scene.\n"
    "You are given a sampling of panels (across the whole chapter) plus all OCR/dialogue text.\n"
    "Identify the recurring or story-important characters. For EACH:\n"
    "  id: short snake_case handle (e.g. 'protagonist', 'dying_master', 'hooded_leader').\n"
    "  canonical_name: a REAL name if one appears in the dialogue/OCR (a proper name, or an\n"
    "    address term actually used such as 'Ancestor-nim'); otherwise a short descriptive\n"
    "    handle ('the young descendant', 'the dying old master'). This is what narration will call them.\n"
    "  aliases: other ways they're named/addressed in the text.\n"
    "  role: one of protagonist | antagonist | ally | mentor | minor | group (for a faceless band).\n"
    "  visual_description: appearance cues a downstream model can MATCH in a panel — age, hair,\n"
    "    clothing, weapon, distinctive features. Be concrete.\n"
    "  is_protagonist: true for the single main character the recap follows.\n"
    "Ignore anonymous crowds. Prefer 6-12 entries. Return ONLY JSON matching the schema."
)


def _img_part(path: str) -> Optional[types.Part]:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return None
    try:
        return types.Part.from_bytes(data=data, mime_type="image/jpeg")
    except TypeError:
        return types.Part.from_bytes(bytes=data, mime_type="image/jpeg")


def _text_part(s: str) -> types.Part:
    try:
        return types.Part.from_text(text=s)
    except TypeError:
        return types.Part.from_text(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-manifest", required=True)
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--max-images", type=int, default=24)
    args = ap.parse_args()

    project = args.project
    if not project:
        keys = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if keys and os.path.exists(keys):
            project = json.loads(open(keys).read()).get("project_id", "")
    if not project:
        raise SystemExit("No --project and could not derive project_id from GOOGLE_APPLICATION_CREDENTIALS")

    vision = _load(args.vision_manifest)
    items = _vision_items(vision)
    images = _sample_images(items, args.max_images)
    ocr = _all_ocr(items)

    parts: List[types.Part] = [_text_part(SYSTEM)]
    parts.append(_text_part("ALL DIALOGUE / OCR IN THE CHAPTER (scene_file: text):\n" + "\n".join(ocr)))
    parts.append(_text_part(f"\n{len(images)} sample panels follow (spread across the chapter):"))
    for p in images:
        ip = _img_part(p)
        if ip is not None:
            parts.append(ip)

    client = genai.Client(vertexai=True, project=project, location=args.location)
    resp = client.models.generate_content(
        model=args.model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=CAST_SCHEMA,
        ),
    )
    cast = json.loads(resp.text)
    cast["_meta"] = {"model": args.model, "images_used": len(images), "ocr_lines": len(ocr)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cast, f, ensure_ascii=False, indent=2)

    members = cast.get("cast") or []
    print(f"[ok] {args.out} — {len(members)} cast members")
    for c in members:
        star = " *PROTAGONIST*" if c.get("is_protagonist") else ""
        print(f"  - {c.get('canonical_name')} ({c.get('role')}){star}: {c.get('visual_description','')[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
