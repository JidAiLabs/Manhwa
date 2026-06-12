#!/usr/bin/env python3
"""
youtube_meta.py — copyright-safe YouTube title + thumbnail hooks per chapter.

Recap channels never publish the licensed series name: titles are trope
hooks ("He Was Mocked as the Weakest, but His Sync Rate Broke All Records",
"BETRAYED BY HIS FAMILY, He Receives A SECRET AI SYSTEM") and thumbnails
carry one short impact phrase. This tool reads the chapter's beats
(narration + per-beat hooks) and asks Gemini Flash for:

  title        — clickbait YouTube title, NO series/character real names,
                 emphasis words in CAPS, ~60-95 chars
  hooks        — 3 short thumbnail hook options (1-4 words, punchy)
  description  — 2-3 sentence upload description + hashtags (no series name)

Output: <episode>/render/youtube_meta.json (+ console). Cost: ~$0.001.

Usage:
  python tools/youtube_meta.py --episode-dir ongoing/<series>/<chapter>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)


def chapter_digest(beats_obj: Dict[str, Any], *, max_chars: int = 6000) -> str:
    """Compact story digest for the copywriter prompt."""
    lines: List[str] = []
    for b in beats_obj.get("beats") or []:
        for key in ("hook", "what_happens", "narration"):
            v = str(b.get(key) or "").strip()
            if v:
                lines.append(v)
                break
    return "\n".join(lines)[:max_chars]


def build_meta_prompt(digest: str, banned: str = "") -> str:
    ban = (f"\nBANNED WORDS (the licensed series name — never use them in "
           f"title, hooks, or description, not even partially): {banned}\n"
           if banned else "")
    return f"""You write YouTube packaging for a webtoon/manhwa RECAP channel.

STORY DIGEST of the chapter (first chapter of the series):
---
{digest}
---
{ban}

Write JSON with EXACTLY these keys:
{{
  "title": "...",        // clickbait recap title, 60-95 chars. NEVER use the
                         // series name or any character's real name — describe
                         // the trope/premise instead ("He...", "When a ...").
                         // Put 2-4 emphasis words in ALL CAPS. End styles like
                         // " - Manhwa Recap" are allowed.
  "hooks": ["...", "...", "..."],  // 3 thumbnail text options, 1-4 words,
                         // punchy ALL-CAPS-friendly ("SECRET AI SYSTEM",
                         // "OP DEMON BABY!", "GENIUS", "RANK 1")
  "description": "..."   // 2-3 sentences + 4-6 hashtags (#manhwa #manhwarecap
                         // etc). No series or character real names.
}}

Return ONLY the JSON object."""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """First {...} block in a model reply, tolerant of code fences."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--series-title", default="",
                    help="INTERNAL ban-list only — the licensed name must "
                         "never appear in the output")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--backend", choices=["vertex", "ollama"],
                    default="ollama")
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--out", default="",
                    help="default <episode>/render/youtube_meta.json")
    args = ap.parse_args()

    ep = args.episode_dir.rstrip("/")
    bp = os.path.join(ep, "manifest.beats.json")
    if not os.path.exists(bp):
        print(f"[err] no beats manifest at {bp}")
        return 1
    with open(bp, "r", encoding="utf-8") as f:
        digest = chapter_digest(json.load(f))

    meta: Optional[Dict[str, Any]] = None
    if args.backend == "ollama":
        try:
            import ollama
            resp = ollama.chat(model=args.model, think=False,
                               messages=[{"role": "user", "content":
                                          build_meta_prompt(digest,
                                                            args.series_title)}],
                               options={"temperature": 0.8,
                                        "num_predict": 1200})
            meta = extract_json(resp["message"]["content"] or "")
        except Exception as e:
            print(f"[warn] ollama/{args.model}: {e}")
    if not meta:
        from thumbnail_gen import _make_client
        for kind, client in _make_client(args.location):
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[build_meta_prompt(digest, args.series_title)])
                meta = extract_json(resp.text or "")
                if meta and meta.get("title"):
                    break
            except Exception as e:
                print(f"[warn] {kind}: {e}")
    if not meta:
        print("[err] no usable metadata returned")
        return 1

    out = args.out or os.path.join(ep, "render", "youtube_meta.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[ok] {out}")
    print("  title:", meta.get("title"))
    for h in meta.get("hooks") or []:
        print("  hook :", h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
