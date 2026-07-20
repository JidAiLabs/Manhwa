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
import sys
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from manifest_io import write_manifest                                # noqa: E402


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


def recurring_figures(understood: Any, min_panels: int = 3) -> List[str]:
    """Recurring-figure candidate lines from the per-panel understanding —
    evidence-driven cast coverage (2026-07-20 story-state wave: the blue-sash
    helper who performs the ch1 kill recurred across panels yet was absent
    from the sampled-images cast, so the identity gate had nothing to attach
    the strike to).

    Pure code, no model call: subjects are clustered by their cast_identity
    appearance tokens (cluster core = INTERSECTION of member token sets, so a
    cluster is defined by what its subjects have in COMMON and can never
    drift into swallowing everyone); a cluster seen in >= min_panels distinct
    panels yields one line '<most common subject wording> — seen in N panels
    (files…)'. The prompt then requires each listed figure to appear in the
    cast (entry or alias) — the model decides which."""
    from collections import Counter

    from cast_identity import _looks_person, _subject_tokens
    clusters: List[Dict[str, Any]] = []
    for p in ((understood or {}).get("panels") or []):
        if not isinstance(p, dict):
            continue
        fn = str(p.get("scene_file") or "")
        for s in (p.get("subjects") or []):
            s = str(s).strip()
            if not _looks_person(s):
                continue                 # props/scenery are not cast material
            toks = _subject_tokens(s)
            if len(toks) < 2:
                continue                 # generic-only: no identity evidence
            best, best_core = None, ()
            for c in clusters:
                core = c["tokens"] & toks
                if len(core) >= 2 and len(core) > len(best_core):
                    best, best_core = c, core
            if best is None:
                clusters.append({"tokens": set(toks), "files": {fn},
                                 "reps": Counter([s])})
            else:
                best["tokens"] = set(best_core)
                best["files"].add(fn)
                best["reps"][s] += 1
    out: List[str] = []
    for c in sorted(clusters, key=lambda c: -len(c["files"])):
        if len(c["files"]) < min_panels:
            continue
        rep = c["reps"].most_common(1)[0][0]
        files = ", ".join(sorted(c["files"])[:3])
        out.append(f"{rep} — seen in {len(c['files'])} panels ({files}…)")
        if len(out) >= 8:
            break
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
    "    handle ('the dying old master'). This is what narration will call them.\n"
    "    EXCEPTION — the PROTAGONIST: if the main character has no real name in the text,\n"
    "    canonical_name MUST be exactly 'our protagonist' (recap-native). NEVER a descriptive\n"
    "    phrase like 'the web novel reader' or 'the young man'.\n"
    "  aliases: other ways they're named/addressed in the text.\n"
    "  role: one of protagonist | antagonist | ally | mentor | minor | group (for a faceless band).\n"
    "  visual_description: appearance cues a downstream model can MATCH in a panel — age, hair,\n"
    "    clothing, weapon, distinctive features. Be concrete.\n"
    "  is_protagonist: true for the single main character the recap follows.\n"
    "Ignore anonymous crowds. Prefer 6-14 entries. Return ONLY JSON matching the schema."
)

RECURRING_RULE = (
    "RECURRING FIGURES OBSERVED (from per-panel analysis of the whole chapter):\n"
    "{FIGURES}\n"
    "Every listed recurring figure MUST appear in the cast — either as its own "
    "entry or as an alias/visual_description of an existing member. A recurring "
    "figure who performs or receives violence is story-important by definition; "
    "never omit them."
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
    ap.add_argument("--backend", choices=["vertex", "ollama"], default="vertex")
    ap.add_argument("--max-images", type=int, default=24)
    ap.add_argument("--understood", default="",
                    help="manifest.panels.understood.json — recurring-figure "
                         "coverage (every recurring figure must land in the cast)")
    args = ap.parse_args()

    project = args.project
    if args.backend == "vertex" and not project:
        keys = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if keys and os.path.exists(keys):
            project = json.loads(open(keys).read()).get("project_id", "")
    if args.backend == "vertex" and not project:
        raise SystemExit("No --project and could not derive project_id from GOOGLE_APPLICATION_CREDENTIALS")

    vision = _load(args.vision_manifest)
    items = _vision_items(vision)
    images = _sample_images(items, args.max_images)
    ocr = _all_ocr(items)

    figures_block = ""
    if args.understood and os.path.exists(args.understood):
        try:
            figs = recurring_figures(_load(args.understood))
        except Exception:
            figs = []
        if figs:
            figures_block = RECURRING_RULE.replace(
                "{FIGURES}", "\n".join(f"  - {f}" for f in figs))

    parts: List[types.Part] = [_text_part(SYSTEM)]
    if figures_block:
        parts.append(_text_part(figures_block))
    parts.append(_text_part("ALL DIALOGUE / OCR IN THE CHAPTER (scene_file: text):\n" + "\n".join(ocr)))
    parts.append(_text_part(f"\n{len(images)} sample panels follow (spread across the chapter):"))
    for p in images:
        ip = _img_part(p)
        if ip is not None:
            parts.append(ip)

    if args.backend == "ollama":
        import ollama
        text_blobs = [SYSTEM]
        if figures_block:
            text_blobs.append(figures_block)
        text_blobs += ["ALL DIALOGUE / OCR IN THE CHAPTER (scene_file: text):\n"
                       + "\n".join(ocr),
                       f"\n{len(images)} sample panels follow:"]
        import re as _re

        def _lower_types(o):
            if isinstance(o, dict):
                return {k: (v.lower() if k == "type" and isinstance(v, str)
                            else _lower_types(v))
                        for k, v in o.items() if k != "propertyOrdering"}
            if isinstance(o, list):
                return [_lower_types(x) for x in o]
            return o

        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from ollama_compat import chat as _ollama_chat
        resp = _ollama_chat(
            model=args.model,
            messages=[{"role": "user", "content": "\n".join(text_blobs),
                       "images": [str(pth) for pth in images]}],
            format=_lower_types(CAST_SCHEMA), think=False,
            options={"temperature": 0.2, "num_ctx": 16384,
                     "num_predict": 2400})
        cast = json.loads(resp["message"]["content"])
    else:
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
    inputs = [args.groups_manifest, args.vision_manifest]
    if args.understood and os.path.exists(args.understood):
        inputs.append(args.understood)
    write_manifest(args.out, cast, inputs=tuple(inputs),
                   tool="cast_builder",
                   extra_meta={"model": args.model, "images_used": len(images),
                               "ocr_lines": len(ocr)})

    members = cast.get("cast") or []
    print(f"[ok] {args.out} — {len(members)} cast members")
    for c in members:
        star = " *PROTAGONIST*" if c.get("is_protagonist") else ""
        print(f"  - {c.get('canonical_name')} ({c.get('role')}){star}: {c.get('visual_description','')[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
