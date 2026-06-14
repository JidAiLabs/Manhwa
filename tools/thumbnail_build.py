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


def build_thumbnail(episode_dir: str, *, models: List[str],
                    location: str = "global", size: str = "2K",
                    refs: List[str] = None) -> Dict[str, Any]:
    """Generate the styled, overlaid thumbnail for a chapter that already has a
    publish_meta.json (run publish_concept first). Returns a small report."""
    render = os.path.join(episode_dir, "render")
    concept = json.load(open(os.path.join(render, "publish_meta.json")))
    style = concept.get("style", "power_reveal")
    art_prompt = tg.build_art_prompt(style_for(style)["art_prompt"])

    if not refs:
        beats = json.load(open(os.path.join(episode_dir, "manifest.beats.json")))
        refs = tg.pick_reference_scenes(beats)

    art_path = os.path.join(render, "thumbnail_art.png")
    used = tg.generate(episode_dir, hook_text="", refs=refs, models=models,
                       location=location, aspect="16:9", size=size,
                       out_path=art_path, prompt_override=art_prompt)
    if not used:
        raise RuntimeError("Nano Banana returned no image")

    out = os.path.join(render, "thumbnail_yt.jpg")
    render_overlay(art_path, out, hook=concept.get("hook", ""),
                   style_overlay=concept.get("style_overlay")
                   or style_for(style)["overlay"],
                   speech=concept.get("speech") or [])
    return {"style": style, "hook": concept.get("hook"), "model": used,
            "art": art_path, "thumbnail": out, "refs": refs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--models", default="gemini-3-pro-image,gemini-3-pro-image-preview")
    ap.add_argument("--location", default="global")
    ap.add_argument("--size", default="2K", choices=["1K", "2K", "4K"])
    ap.add_argument("--refs", default="", help="comma-separated scene files")
    args = ap.parse_args()
    rep = build_thumbnail(
        args.episode_dir, models=[m for m in args.models.split(",") if m],
        location=args.location, size=args.size,
        refs=[r for r in args.refs.split(",") if r] or None)
    print(f"[ok] thumbnail={rep['thumbnail']} style={rep['style']} "
          f"hook={rep['hook']!r} model={rep['model']} refs={rep['refs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
