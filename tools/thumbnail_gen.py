#!/usr/bin/env python3
"""
thumbnail_gen.py — YouTube thumbnail via Nano Banana Pro (Gemini 3 Pro Image).

Generates ONE upload thumbnail per chapter in the proven recap-channel style:
dramatic weak-vs-powerful split composition, the REAL characters lifted from
reference panels of the chapter, BEFORE/AFTER labels (or a custom hook via
--hook-text). NO series titles or chapter numbers — recap channels never
render the licensed name; titles/hooks come from tools/youtube_meta.py.

Reference panels are auto-picked from manifest.beats.json scene_selection
(earliest kept calm/tense panel = "weak", last kept intense panel = climax),
overridable with --refs. Uses the ORIGINAL scenes/ art (the model is told to
ignore speech bubbles and on-page text).

Auth: Vertex AI with the repo service account (same as the beats stage);
falls back to a GEMINI_API_KEY client if Vertex does not serve the model.

Cost: ~$0.13-0.24 per generated image (1K/2K vs 4K). Prints what it does.

Usage:
  python tools/thumbnail_gen.py --episode-dir ongoing/nano-machine/Chapter_1 \
      --series-title "Nano Machine" [--chapter-label "Chapter 1"] \
      [--hook-text "..."] [--refs p000004.jpg,p000113.jpg] [--size 2K]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

_INTENSITY_RANK = {"calm": 0, "tense": 1, "intense": 2}


def pick_reference_scenes(beats_obj: Dict[str, Any], *,
                          max_refs: int = 3) -> List[str]:
    """Weak-early + climax-late kept panels from the beats scene_selection."""
    sel: List[Tuple[str, str]] = []  # (scene_file, intensity) kept, in order
    for beat in beats_obj.get("beats") or []:
        for s in beat.get("scene_selection") or []:
            if str(s.get("role") or "keep") != "keep":
                continue
            sel.append((str(s.get("scene_file") or ""),
                        str(s.get("intensity") or "calm")))
    if not sel:
        return []
    weak = next((f for f, i in sel if i in ("calm", "tense")), sel[0][0])
    strong = next((f for f, i in reversed(sel) if i == "intense"),
                  sel[-1][0])
    refs = [weak, strong]
    # a mid-chapter intense panel adds identity/action context
    mids = [f for f, i in sel[len(sel) // 3: 2 * len(sel) // 3]
            if i == "intense" and f not in refs]
    if mids:
        refs.append(mids[len(mids) // 2])
    out: List[str] = []
    for f in refs:
        if f and f not in out:
            out.append(f)
    return out[:max_refs]


def build_prompt(hook_text: str = "") -> str:
    """Thumbnail prompt. NO series titles, NO chapter numbers anywhere —
    recap channels never show the licensed name (the user's directive).
    Default text is the BEFORE/AFTER pair; *hook_text* swaps in one short
    impact phrase with a pointer arrow instead."""
    if hook_text:
        text_block = (f'- ONE short impact phrase in huge yellow capital '
                      f'letters with a thick black outline, placed in the '
                      f'upper area: "{hook_text.upper()}", with a small '
                      f'hand-drawn yellow arrow pointing from it toward the '
                      f'powered-up character.')
    else:
        text_block = ('- "BEFORE" in huge yellow capital letters with a '
                      'thick black outline at the bottom of the LEFT half.\n'
                      '- "AFTER" in the same style at the bottom of the '
                      'RIGHT half.')
    return f"""Create a YouTube thumbnail (16:9) for a webtoon recap video.

COMPOSITION — dramatic transformation split:
- LEFT HALF: the protagonist at his weakest — beaten, desperate, close-up on the face, cold blue/teal grading, dim moody lighting.
- RIGHT HALF: the SAME protagonist transformed and powerful — confident or furious, glowing eyes, energy aura, fiery orange/red rim lighting, dynamic angle.
- A sharp diagonal energy crack/lightning seam separates the halves.

CHARACTER FIDELITY:
- Use the EXACT character designs from the reference images: same face, hairstyle, hair color, clothing. Repaint them in clean, high-detail webtoon/anime style — do NOT copy panel crops literally.
- IGNORE and DO NOT reproduce any speech bubbles, Korean/Chinese/English on-page text, watermarks, or UI from the references.

TEXT (render EXACTLY this and nothing else — no series name, no chapter or episode numbers, no logos):
{text_block}

STYLE: maximum-contrast YouTube thumbnail look — saturated colors, crisp edges, cinematic glow, particles/sparks, vignette. No watermark, no borders."""


def _load_ref_images(episode_dir: str, refs: List[str], *,
                     max_side: int = 1024):
    from PIL import Image
    out = []
    for fn in refs:
        path = os.path.join(episode_dir, "scenes", fn)
        if not os.path.exists(path):
            print(f"[warn] ref missing, skipped: {path}")
            continue
        im = Image.open(path).convert("RGB")
        s = max(im.size)
        if s > max_side:
            im = im.resize((max(1, im.width * max_side // s),
                            max(1, im.height * max_side // s)))
        out.append(im)
    return out


def _make_client(location: str):
    """Vertex with the repo SA first; GEMINI_API_KEY client as fallback."""
    from google import genai
    attempts = []
    repo_sa = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "keys", "gcp-vision.json")
    sa = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or repo_sa
    if not os.path.exists(sa) and os.path.exists(repo_sa):
        print(f"[warn] GOOGLE_APPLICATION_CREDENTIALS stale ({sa}) — "
              f"using repo key {repo_sa}")
        sa = repo_sa
    if os.path.exists(sa):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa
        try:
            project = json.load(open(sa))["project_id"]
            attempts.append(("vertex", genai.Client(
                vertexai=True, project=project, location=location)))
        except Exception as e:  # pragma: no cover - env specific
            print(f"[warn] vertex client failed: {e}")
    if os.environ.get("GEMINI_API_KEY"):
        attempts.append(("api-key", genai.Client(
            api_key=os.environ["GEMINI_API_KEY"])))
    if not attempts:
        print("[err] no auth: neither a service-account key on disk nor "
              "GEMINI_API_KEY in the environment")
    return attempts


def generate(episode_dir: str, *, hook_text: str, refs: List[str],
             models: List[str], location: str, aspect: str, size: str,
             out_path: str) -> Optional[str]:
    from google.genai import types

    prompt = build_prompt(hook_text)
    images = _load_ref_images(episode_dir, refs)
    if not images:
        print("[err] no usable reference images")
        return None
    cfg = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect, image_size=size),
    )

    last_err: Optional[Exception] = None
    for kind, client in _make_client(location):
        for model in models:
            try:
                print(f"[..] {kind} client, model={model}, "
                      f"refs={refs} size={size}")
                resp = client.models.generate_content(
                    model=model, contents=[prompt, *images], config=cfg)
                parts = (resp.candidates[0].content.parts
                         if resp.candidates else [])
                for part in parts:
                    data = getattr(getattr(part, "inline_data", None),
                                   "data", None)
                    if data:
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path, "wb") as f:
                            f.write(data)
                        return model
                texts = [p.text for p in parts if getattr(p, "text", None)]
                last_err = RuntimeError(
                    f"no image part returned ({model}): {texts[:1]}")
                print(f"[warn] {last_err}")
            except Exception as e:
                last_err = e
                print(f"[warn] {kind}/{model}: {e}")
    print(f"[err] all attempts failed: {last_err}")
    return None


def write_youtube_jpg(master_png: str, yt_path: str,
                      *, max_bytes: int = 2_000_000) -> None:
    """YouTube wants <=2MB; 1280x720 JPEG from the 2K master."""
    from PIL import Image
    im = Image.open(master_png).convert("RGB").resize((1280, 720))
    for q in (92, 88, 84, 78, 70):
        im.save(yt_path, "JPEG", quality=q, optimize=True)
        if os.path.getsize(yt_path) <= max_bytes:
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--hook-text", default="",
                    help="short impact phrase overlay (default: BEFORE/AFTER "
                         "labels); series titles are NEVER rendered")
    ap.add_argument("--refs", default="",
                    help="comma-separated scene files (default: auto from "
                         "beats scene_selection)")
    ap.add_argument("--models", default="gemini-3-pro-image,"
                                        "gemini-3-pro-image-preview")
    ap.add_argument("--location", default="global")
    ap.add_argument("--aspect", default="16:9")
    ap.add_argument("--size", default="2K", choices=["1K", "2K", "4K"])
    ap.add_argument("--out", default="",
                    help="default <episode>/render/thumbnail.png")
    args = ap.parse_args()

    ep = args.episode_dir.rstrip("/")

    refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    if not refs:
        bp = os.path.join(ep, "manifest.beats.json")
        if os.path.exists(bp):
            with open(bp, "r", encoding="utf-8") as f:
                refs = pick_reference_scenes(json.load(f))
    if not refs:
        print("[err] no refs (no beats manifest?) — pass --refs")
        return 1

    out_png = args.out or os.path.join(ep, "render", "thumbnail.png")
    model = generate(ep, hook_text=args.hook_text, refs=refs,
                     models=[m.strip() for m in args.models.split(",")],
                     location=args.location, aspect=args.aspect,
                     size=args.size, out_path=out_png)
    if not model:
        return 1
    yt = os.path.splitext(out_png)[0] + "_yt.jpg"
    write_youtube_jpg(out_png, yt)
    print(f"[ok] model={model} master={out_png} "
          f"youtube={yt} ({os.path.getsize(yt) // 1024}KB) "
          f"[cost] ~$0.13-0.24/image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
