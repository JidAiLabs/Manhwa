#!/usr/bin/env python3
"""
thumbnail_build.py — orchestrate one thumbnail from the publish concept:
  concept (style + hook + refs) -> Nano Banana ART (text-free) -> deterministic
  text overlay -> 1280x720 jpg.

The ART carries NO text (the model never renders the licensed name); all words
are the branded overlay. Reuses thumbnail_gen (Nano Banana + ref picking),
thumbnail_styles (the chosen module's composition), thumbnail_overlay (text).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from thumbnail_styles import style_for                                  # noqa: E402
from thumbnail_overlay import render_overlay                            # noqa: E402
import thumbnail_gen as tg                                              # noqa: E402


def render_thumbnail(concept: Dict[str, Any], *, ref_episode_dir: str,
                     out_dir: str, models: List[str], location: str = "global",
                     size: str = "2K", refs: List[str] = None) -> Dict[str, Any]:
    """Core: an explicit concept (style + hook + refs) + the chapter dir that
    HOLDS the reference scene files -> text-free Nano-Banana art + deterministic
    overlay -> <out_dir>/thumbnail_yt.jpg. ref_episode_dir only resolves the
    scene crops; out_dir is independent (a chapter's render/ OR a series dir)."""
    style = concept.get("style", "power_reveal")
    art_prompt = tg.build_art_prompt(style_for(style)["art_prompt"])

    refs = refs or concept.get("refs") or []
    if not refs:
        beats = json.load(open(os.path.join(ref_episode_dir,
                                            "manifest.beats.json")))
        refs = tg.pick_reference_scenes(beats)

    os.makedirs(out_dir, exist_ok=True)
    art_path = os.path.join(out_dir, "thumbnail_art.png")
    used = tg.generate(ref_episode_dir, hook_text="", refs=refs, models=models,
                       location=location, aspect="16:9", size=size,
                       out_path=art_path, prompt_override=art_prompt)
    if not used:
        raise RuntimeError("Nano Banana returned no image")

    out = os.path.join(out_dir, "thumbnail_yt.jpg")
    render_overlay(art_path, out, hook=concept.get("hook", ""),
                   style_overlay=concept.get("style_overlay")
                   or style_for(style)["overlay"],
                   speech=concept.get("speech") or [])
    return {"style": style, "hook": concept.get("hook"), "model": used,
            "art": art_path, "thumbnail": out, "refs": refs}


def build_thumbnail(episode_dir: str, *, models: List[str],
                    location: str = "global", size: str = "2K",
                    refs: List[str] = None) -> Dict[str, Any]:
    """Chapter mode: the concept (render/publish_meta.json) and the ref scene
    files both live in one chapter's dir. (The channel ships ONE thumbnail per
    SERIES — see render_thumbnail / the series_thumbnail worker job — but this
    per-chapter path stays for single-chapter previews and tests.)"""
    render = os.path.join(episode_dir, "render")
    concept = json.load(open(os.path.join(render, "publish_meta.json")))
    return render_thumbnail(concept, ref_episode_dir=episode_dir,
                            out_dir=render, models=models, location=location,
                            size=size, refs=refs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", default="", help="chapter mode: reads "
                    "<dir>/render/publish_meta.json + that chapter's scenes")
    ap.add_argument("--concept", default="", help="series mode: explicit "
                    "concept json (from publish_concept --episode-dirs)")
    ap.add_argument("--ref-episode-dir", default="", help="series mode: the "
                    "CLIMAX chapter dir that holds the ref scene crops")
    ap.add_argument("--out-dir", default="", help="series mode: where to write "
                    "thumbnail_yt.jpg (e.g. dist/series_<id>)")
    ap.add_argument("--models", default="gemini-3-pro-image,gemini-3-pro-image-preview")
    ap.add_argument("--location", default="global")
    ap.add_argument("--size", default="2K", choices=["1K", "2K", "4K"])
    ap.add_argument("--refs", default="", help="comma-separated scene files")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    refs = [r for r in args.refs.split(",") if r] or None
    if args.concept:
        concept = json.load(open(args.concept))
        ref_ep = args.ref_episode_dir or os.path.dirname(
            os.path.dirname(os.path.abspath(args.concept)))
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.concept))
        rep = render_thumbnail(concept, ref_episode_dir=ref_ep, out_dir=out_dir,
                               models=models, location=args.location,
                               size=args.size, refs=refs)
    elif args.episode_dir:
        rep = build_thumbnail(args.episode_dir, models=models,
                              location=args.location, size=args.size, refs=refs)
    else:
        ap.error("need --episode-dir (chapter) or --concept (series)")
    print(f"[ok] thumbnail={rep['thumbnail']} style={rep['style']} "
          f"hook={rep['hook']!r} model={rep['model']} refs={rep['refs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
