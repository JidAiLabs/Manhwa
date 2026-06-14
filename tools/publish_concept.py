#!/usr/bin/env python3
"""
publish_concept.py — ONE coherent publish package per unit (single chapter now;
bundle range later): title + thumbnail hook + thumbnail style + synopsis +
hashtags + description + pinned comment.

Coherence by construction: title and the thumbnail label both read from the same
concept, so they can't drift. Copyright-safe: the licensed series name never
appears in title / description / thumbnail — only in the PINNED COMMENT (user
decision). $0 — local Gemma for the copy, deterministic for style/templates.

Output: <episode>/render/publish_meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from thumbnail_styles import select_style, style_for          # noqa: E402
from youtube_meta import chapter_digest, extract_json          # noqa: E402

# Channel-static boilerplate (edit Patreon / email once). Real series name is NOT
# here — it goes only in the pinned comment.
CHANNEL = {
    "name": "OriginPower Manhwa Recap",
    "patreon": "https://www.patreon.com/originpowermanhwa",
    "email": "originpowermanhwa@gmail.com",
}
_DISCLAIMER = (
    "I don't own the manhwa/artwork. All rights to their respective owners. "
    "For any concern or removal, contact {email} before a copyright claim.")
_BASE_TAGS = ("manhwa recap, manhwa, webtoon, manhwa recaps, manga recap, "
              "manhua recap, anime recap, recap, manhwa summary, webtoon recap")


def build_concept_prompt(digest: str, banned: str, style: str) -> str:
    return (
        "You write copyright-safe metadata for a manhwa RECAP video. NEVER use "
        f"this licensed title (or any part of it): {banned or '(none)'}.\n"
        f"Chosen thumbnail style: {style} (the hook should suit it).\n\n"
        "From the STORY DIGEST, return ONLY JSON:\n"
        "{\n"
        '  "title": "clickbait recap title, 60-95 chars, trope-based, CAPS for '
        'emphasis, NO real names",\n'
        '  "hooks": ["3 thumbnail labels, 1-4 words each, punchy"],\n'
        '  "synopsis": "2-4 sentence teaser with emojis, trope framing, NO real '
        'names",\n'
        '  "hashtags": ["6-10 hashtags incl #manhwa #manga + genre/theme"]\n'
        "}\n\nSTORY DIGEST:\n" + digest)


_DIGIT = re.compile(r"\d|\bS+\b|\brank\b|\blevel\b|\blvl\b|\bSSS?\b", re.I)


def pick_hook(hooks: List[str], style: str) -> str:
    """Choose the thumbnail label that best fits the style (deterministic)."""
    hooks = [str(h).strip() for h in (hooks or []) if str(h).strip()]
    if not hooks:
        return ""
    if style == "stat_callout":
        for h in hooks:
            if _DIGIT.search(h):
                return h
    if style == "before_after":
        for h in hooks:
            if "|" in h:
                return h
    return hooks[0]


def build_description(synopsis: str, hashtags: List[str]) -> str:
    tags = " ".join(t if t.startswith("#") else "#" + t.lstrip("#")
                    for t in (hashtags or []) if str(t).strip())
    return "\n\n".join(filter(None, [
        synopsis.strip(),
        tags,
        f"▶ Patreon: {CHANNEL['patreon']}",
        f"📩 Business: {CHANNEL['email']}",
        _DISCLAIMER.format(email=CHANNEL["email"]),
        "Tags: " + _BASE_TAGS,
    ]))


def pinned_comment(real_title: str, official_link: str = "") -> str:
    t = (real_title or "").strip() or "(see description)"
    tail = f" — read the official release: {official_link}" if official_link else \
           " — please support the official release."
    return f"Manhwa: {t}{tail}"


_INTENSITY = {"calm": 0, "unknown": 0, "tense": 1, "intense": 2, "explosive": 3}


def bundle_digest(beats_objs: List[Dict[str, Any]], *,
                  per_chapter_chars: int = 700) -> str:
    """Aggregate MANY chapters into one arc digest that fits the LLM context:
    a compact per-chapter summary (hooks + the punchiest beats), so the title
    can span the whole arc (setup -> payoff), which a single chapter can't."""
    parts: List[str] = []
    for i, b in enumerate(beats_objs, 1):
        lines: List[str] = []
        for bt in b.get("beats") or []:
            t = (str(bt.get("hook") or "").strip()
                 or str(bt.get("what_happens") or "").strip())
            if t:
                lines.append(t)
        blob = " ".join(lines)[:per_chapter_chars]
        if blob:
            parts.append(f"[Chapter {i}] {blob}")
    return "\n".join(parts)


def select_bundle_climax(beats_objs: List[Dict[str, Any]]):
    """Pick the single most thumbnail-worthy moment across the bundle: the
    highest-intensity kept beat. Returns (chapter_index, scene_files) so the
    thumbnail art comes from the ARC's climax — usually NOT chapter 1."""
    best = (-1, 0, [])  # (intensity, chapter_index, scene_files)
    for ci, b in enumerate(beats_objs):
        for bt in b.get("beats") or []:
            scenes = [s for s in (bt.get("scene_selection") or [])
                      if isinstance(s, dict)]
            inten = max((_INTENSITY.get(str(s.get("intensity") or "").lower(), 0)
                         for s in scenes), default=0)
            if inten > best[0]:
                files = [str(s.get("scene_file")) for s in scenes
                         if s.get("role", "keep") != "redundant" and s.get("scene_file")]
                best = (inten, ci, files[:3])
    return best[1], best[2]


def assemble_concept(beats_obj: Dict[str, Any], llm: Dict[str, Any], *,
                     series_title: str, genre: str = "",
                     official_link: str = "") -> Dict[str, Any]:
    """Build the concept from beats (style) + the LLM copy. Pure/testable."""
    style = select_style(beats_obj, genre=genre)
    hooks = llm.get("hooks") or []
    hook = pick_hook(hooks, style)
    synopsis = str(llm.get("synopsis") or "").strip()
    hashtags = llm.get("hashtags") or ["#manhwa", "#manga", "#manhwarecap"]
    return {
        "title": str(llm.get("title") or "").strip(),
        "style": style,
        "style_overlay": style_for(style)["overlay"],
        "hook": hook,
        "hooks": hooks,
        "synopsis": synopsis,
        "hashtags": hashtags,
        "description": build_description(synopsis, hashtags),
        "pinned_comment": pinned_comment(series_title, official_link),
    }


def _fmt_ts(sec: float) -> str:
    s = int(sec)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parts_timestamps(durations: List[float],
                     labels: Optional[List[str]] = None) -> List[str]:
    """YouTube-chapter 'Parts' list: cumulative offsets, first MUST be 0:00."""
    out: List[str] = []
    t = 0.0
    for i, d in enumerate(durations):
        out.append(f"{_fmt_ts(t)} {labels[i] if labels else f'Part {i + 1}'}")
        t += float(d or 0.0)
    return out


def build_bundle_concept(beats_list: List[Dict[str, Any]], llm: Dict[str, Any],
                         *, durations: List[float], series_title: str,
                         genre: str = "", official_link: str = "",
                         labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Concept for a VIDEO (bundle of N chapters): arc title/synopsis from the
    aggregated chapters, style+refs from the bundle's CLIMAX chapter, and the
    Parts (YouTube-chapter) timestamps appended to the description."""
    climax_ci, refs = select_bundle_climax(beats_list)
    style_beats = beats_list[climax_ci] if beats_list else {}
    c = assemble_concept(style_beats, llm, series_title=series_title,
                         genre=genre, official_link=official_link)
    c["parts"] = parts_timestamps(durations, labels)
    c["climax_chapter_index"] = climax_ci
    c["refs"] = refs                      # thumbnail art comes from the climax
    c["description"] = c["description"] + "\n\n" + "\n".join(c["parts"])
    return c


def _gemma(prompt: str, model: str) -> Dict[str, Any]:
    from ollama_compat import chat as _chat
    resp = _chat(model=model, think=False,
                 messages=[{"role": "user", "content": prompt}],
                 options={"temperature": 0.8, "num_predict": 800})
    return extract_json((resp.get("message") or {}).get("content") or "") or {}


def _plan_duration(ep: str) -> float:
    for fn in ("render.plan.clean.json", "render.plan.json"):
        try:
            return float(json.load(open(os.path.join(ep, fn))).get("total_duration_sec") or 0.0)
        except Exception:
            continue
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", default="", help="single-chapter mode")
    ap.add_argument("--episode-dirs", default="", help="comma-separated chapter "
                    "dirs = a BUNDLE/video (arc title + Parts). Use for videos.")
    ap.add_argument("--series-title", default="", help="licensed title — BAN list "
                    "(never in title/desc/thumb) + pinned-comment credit")
    ap.add_argument("--genre", default="")
    ap.add_argument("--official-link", default="")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.episode_dirs:
        eps = [e for e in args.episode_dirs.split(",") if e]
        beats_list = [json.load(open(os.path.join(e, "manifest.beats.json"))) for e in eps]
        durations = [_plan_duration(e) for e in eps]
        style = select_style(beats_list[select_bundle_climax(beats_list)[0]] if beats_list else {},
                             genre=args.genre)
        llm = _gemma(build_concept_prompt(bundle_digest(beats_list), args.series_title, style),
                     args.ollama_model)
        concept = build_bundle_concept(beats_list, llm, durations=durations,
                                       series_title=args.series_title, genre=args.genre,
                                       official_link=args.official_link)
        out = args.out or os.path.join(eps[0], "render", "bundle_publish_meta.json")
    else:
        if not args.episode_dir:
            ap.error("need --episode-dir (single) or --episode-dirs (bundle)")
        beats_obj = json.load(open(os.path.join(args.episode_dir, "manifest.beats.json")))
        style = select_style(beats_obj, genre=args.genre)
        llm = _gemma(build_concept_prompt(chapter_digest(beats_obj), args.series_title, style),
                     args.ollama_model)
        concept = assemble_concept(beats_obj, llm, series_title=args.series_title,
                                   genre=args.genre, official_link=args.official_link)
        out = args.out or os.path.join(args.episode_dir, "render", "publish_meta.json")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(concept, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote={out} style={concept['style']} "
          f"hook={concept['hook']!r} title={concept['title']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
