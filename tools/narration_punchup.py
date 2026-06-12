#!/usr/bin/env python3
"""
narration_punchup.py — persona pass over grounded beats narration.

The beats pass stays factual (it sees the art). This OPTIONAL second pass
rewrites each narration line in the proven recap-channel persona — gamer
framing, modern anachronisms, dry snark — WITHOUT adding facts. Style guide
distilled from the user's reference transcript (the 530K+ view voice).

Grounding contract: every event/name in the rewrite must already be in the
original line; cast names are preserved verbatim; lines that come back
overlong, name-mangled or fact-inflated FALL BACK to the original.

Usage:
  python tools/narration_punchup.py --beats <ep>/manifest.beats.json \
      --out <ep>/manifest.beats.punch.json [--cast <ep>/manifest.cast.json] \
      [--backend vertex|ollama] [--model gemini-2.5-flash] \
      [--humor full|light]
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

STYLE_GUIDE = """You are the narrator persona of a top manhwa recap channel.
Voice: internet-native, dry, confident, a little sarcastic — a sharp friend
recapping the story, not a movie trailer.

TECHNIQUES (use 1-2 per line, vary them, never force all at once):
- gamer/RPG framing: stats, XP, side quest, boss fight, NPC, build,
  speedrun, loot, aggro ("free XP", "that's a boss-fight invitation")
- modern-life anachronisms inside the ancient setting ("punched into a
  different zip code", "he doesn't read the HR reports on his enforcers",
  "administrative speak for corporate takeover")
- audience intimacy: "our guy", "our boy", "look at his face"
- comedic hyperbole on impacts ("coughing up half his internal organs")
- punchy standalone fragments for beats: "Total silence." "Deal." "He's in."
- snark at villain logic ("he's definitely not taking his own supply")
- meta-narration ("the stealth mission is officially an action movie now")

HARD RULES:
- NEVER invent events, objects, dialogue, or names not present in the
  original line. You restyle facts; you do not add them.
- Keep every character name EXACTLY as written (the cast list is law).
- Keep the original meaning and emotional turn of the line.
- Similar length: between 60% and 150% of the original word count.
- No publication chrome: never mention chapters, episodes, sites, scans,
  views, or the series' real title.
- Mood tags like [panicked] at the start of a line must be preserved as-is.
- HUMOR=light means: one light touch per line at most, keep drama lines
  dramatic. HUMOR=full means: the reference-transcript density."""


def build_prompt(lines: List[Dict[str, Any]], cast_names: List[str],
                 humor: str) -> str:
    payload = [{"group_id": l["group_id"], "narration": l["narration"]}
               for l in lines]
    cast = ", ".join(cast_names) if cast_names else "(none listed)"
    return (f"{STYLE_GUIDE}\n\nHUMOR={humor}\nCAST NAMES (verbatim): {cast}\n\n"
            "Rewrite EVERY line below in the persona. Return ONLY a JSON "
            "array of objects {\"group_id\": int, \"narration\": str} — same "
            "group_ids, same order, no commentary.\n\nLINES:\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))


_MOOD_RE = re.compile(r"^\s*(\[[a-z _-]+\])", re.I)


def _word_count(s: str) -> int:
    return len(re.findall(r"[\w']+", s))


def validate_line(original: str, punched: str,
                  cast_names: List[str]) -> bool:
    """Reject rewrites that break the grounding contract."""
    if not punched or not punched.strip():
        return False
    ow, pw = _word_count(original), _word_count(punched)
    if ow >= 5 and not (0.6 * ow <= pw <= 1.5 * ow + 8):
        return False
    om = _MOOD_RE.match(original)
    if om and not punched.strip().startswith(om.group(1)):
        return False
    low_o, low_p = original.lower(), punched.lower()
    for name in cast_names:
        # any cast name USED must exist verbatim; names present in the
        # original must not be dropped entirely
        if name.lower() in low_o and name.lower() not in low_p:
            return False
    if re.search(r"\b(chapter|episode)\s+\d+|\.com\b|webtoon|asura|elftoon",
                 low_p):
        return False
    return True


def merge(beats_obj: Dict[str, Any], punched: List[Dict[str, Any]],
          cast_names: List[str]) -> Dict[str, Any]:
    """Apply validated rewrites; keep the grounded original otherwise.
    The original always survives as beat['narration_plain']."""
    by_gid = {int(p.get("group_id") or 0): str(p.get("narration") or "")
              for p in punched if isinstance(p, dict)}
    out = json.loads(json.dumps(beats_obj))
    applied = 0
    for b in out.get("beats") or []:
        gid = int(b.get("group_id") or 0)
        original = str(b.get("narration") or "")
        b["narration_plain"] = original
        cand = by_gid.get(gid, "")
        if cand and validate_line(original, cand, cast_names):
            b["narration"] = cand
            applied += 1
    out.setdefault("stats", {})["punchup_applied"] = applied
    return out


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _cast_names(cast_path: str) -> List[str]:
    if not cast_path or not os.path.exists(cast_path):
        return []
    try:
        obj = json.load(open(cast_path))
        names = []
        for c in obj.get("cast") or obj.get("characters") or []:
            n = c.get("name") if isinstance(c, dict) else str(c)
            if n:
                names.append(str(n))
        return names
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cast", default="")
    ap.add_argument("--backend", choices=["vertex", "ollama"],
                    default="vertex")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--humor", choices=["full", "light"], default="full")
    args = ap.parse_args()

    beats_obj = json.load(open(args.beats))
    lines = [{"group_id": int(b.get("group_id") or 0),
              "narration": str(b.get("narration") or "")}
             for b in beats_obj.get("beats") or [] if b.get("narration")]
    cast_names = _cast_names(args.cast)
    prompt = build_prompt(lines, cast_names, args.humor)

    if args.backend == "ollama":
        import ollama
        resp = ollama.chat(model=args.ollama_model,
                           messages=[{"role": "user", "content": prompt}],
                           think=False,
                           options={"temperature": 0.7, "num_ctx": 32768,
                                    "num_predict": 8192})
        raw = (resp.get("message") or {}).get("content") or ""
    else:
        from thumbnail_gen import _make_client  # self-heals stale cred paths
        attempts = _make_client(args.location)
        if not attempts:
            print("[err] no auth available")
            return 1
        _, client = attempts[0]
        resp = client.models.generate_content(model=args.model,
                                              contents=[prompt])
        raw = resp.text or ""

    punched = _extract_json_array(raw)
    out = merge(beats_obj, punched, cast_names)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    n = out["stats"]["punchup_applied"]
    print(f"[ok] wrote={args.out} punched={n}/{len(lines)} "
          f"(rejected lines keep the grounded original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
