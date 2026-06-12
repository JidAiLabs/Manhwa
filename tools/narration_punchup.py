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

BASE_PERSONA = """You are the narrator persona of a top manhwa recap channel.
Voice: internet-native, dry, confident, a little sarcastic — a sharp friend
recapping the story, not a movie trailer.

GENRE-NEUTRAL TECHNIQUES (use 1-2 per line, vary them, never force all):
- gamer/RPG framing: stats, XP, side quest, boss fight, NPC, build,
  speedrun, loot, aggro ("free XP", "that's a boss-fight invitation")
- audience intimacy: "our guy", "our boy", "look at his face"
- comedic hyperbole on impacts ("coughing up half his internal organs")
- punchy standalone fragments for beats: "Total silence." "Deal." "He's in."
- snark at villain logic ("he's definitely not taking his own supply")
- meta-narration ("the stealth mission is officially an action movie now")
- vary line openings; filler openers like "Okay, so" at most ONCE per
  chapter, never on consecutive lines

HARD RULES:
- NEVER invent events, objects, dialogue, or names not present in the
  original line. You restyle facts; you do not add them.
- Keep every character name EXACTLY as written (the cast list is law).
- Keep the original meaning and emotional turn of the line — an injured
  character stays injured, a defeat stays a defeat.
- Similar length: between 60% and 150% of the original word count.
- No publication chrome: never mention chapters, episodes, sites, scans,
  views, or the series' real title.
- Mood tags like [panicked] at the start of a line must be preserved as-is.
- HUMOR=light means: one light touch per line at most, keep drama lines
  dramatic. HUMOR=full means: the reference-transcript density."""

# The comedy AXIS is genre-specific: a murim joke misfires in modern Seoul.
GENRE_ADDONS = {
    "murim": """GENRE: murim/wuxia (ancient martial world).
Comedy axis = the gap between the ancient setting and modern concepts:
modern-life anachronisms land hardest here ("punched into a different zip code", "he doesn't read the HR reports on his enforcers", "sect politics =
corporate org-chart drama"). Cultivation/qi/sect jargon is fair game for
snark ("30 years of qi per pill — supplements have gotten serious").""",
    "modern": """GENRE: modern-world (apocalypse/hunter/regression in a
contemporary setting). The world ALREADY has phones and subways — ancient-
setting anachronism jokes do NOT apply. Comedy axis = mundane daily life vs
supernatural stakes ("the apocalypse started before his commute ended",
"monster attacks and his first thought is the deposit on his flat"). If the
protagonist knows the story/future, lean on reader/meta jokes ("he has the
walkthrough; everyone else is playing blind").""",
    "system": """GENRE: system/reincarnation/regression with game windows.
Comedy axis = treating life as a game UI played absurdly well: tutorial and
newbie framing ("skipping the tutorial", "day-one patch notes"), absurd
contrast between the protagonist's situation and power ("a literal infant
grinding stat points"), deadpan quest-log narration of dramatic moments.""",
}


def genre_key(genre_text: str) -> str:
    g = (genre_text or "").lower()
    if any(k in g for k in ("murim", "wuxia", "martial", "cultivat")):
        return "murim"
    # the SETTING governs the anachronism axis: a modern-world regression
    # story jokes about commutes, not ancient sects
    if any(k in g for k in ("modern", "apocalypse", "hunter", "urban")):
        return "modern"
    if any(k in g for k in ("system", "reincarnat", "regress", "rebirth")):
        return "system"
    return "generic"


def build_prompt(lines: List[Dict[str, Any]], cast_names: List[str],
                 humor: str, genre: str = "") -> str:
    payload = [{"group_id": l["group_id"], "narration": l["narration"]}
               for l in lines]
    cast = ", ".join(cast_names) if cast_names else "(none listed)"
    addon = GENRE_ADDONS.get(genre_key(genre), "")
    guide = BASE_PERSONA + ("\n\n" + addon if addon else "")
    return (f"{guide}\n\nHUMOR={humor}\nCAST NAMES (verbatim): {cast}\n\n"
            "Rewrite EVERY line below in the persona. Return ONLY a JSON "
            "array of objects {\"group_id\": int, \"narration\": str} — same "
            "group_ids, same order, no commentary.\n\nLINES:\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))


_MOOD_RE = re.compile(r"^\s*(\[[a-z _-]+\])", re.I)


def _word_count(s: str) -> int:
    return len(re.findall(r"[\w']+", s))


_UI_TOKENS = {"read", "ep", "episode", "episodes", "comments", "comment",
              "views", "view", "likes", "like", "subscribe", "next", "prev",
              "previous", "tap", "menu", "notice", "unread"}


def _caption_words_by_group(ep_dir: str,
                            beats_obj: Dict[str, Any]) -> Dict[int, set]:
    """Per-group caption word sets (text_only/recovered panels, UI tokens
    stripped) — the punch pass must never paraphrase the monologue away."""
    try:
        v = json.load(open(os.path.join(ep_dir, "manifest.vision.json")))
        items = {str(i.get("scene_file")): i for i in v.get("items") or []}
    except Exception:
        return {}
    rec: set = set()
    try:
        sc = json.load(open(os.path.join(ep_dir, "manifest.scenes.json")))
        rec = {str(s.get("out_file")) for s in sc.get("scenes") or []
               if s.get("recovered")}
    except Exception:
        pass
    out: Dict[int, List[set]] = {}
    for b in beats_obj.get("beats") or []:
        sets: List[set] = []
        for sf in b.get("scene_files") or []:
            it = items.get(str(sf)) or {}
            if not (it.get("text_only") or str(sf) in rec):
                continue
            words = {w for w in re.sub(
                r"[^a-z0-9]+", " ",
                str(it.get("ocr_clean") or "").lower()).split()
                if not w.isdigit() and w not in _UI_TOKENS}
            # PER SCENE, matching prep_qa's caption_unvoiced — a group with
            # two captions must keep BOTH, not 50% of their union
            if len(words) >= 4:
                sets.append(words)
        if sets:
            out[int(b.get("group_id") or 0)] = sets
    return out


def validate_line(original: str, punched: str,
                  cast_names: List[str], *,
                  required: Any = None) -> bool:
    """Reject rewrites that break the grounding contract."""
    if not punched or not punched.strip():
        return False
    if required:
        req_sets = ([required] if isinstance(required, (set, frozenset))
                    else list(required))
        pwords = set(re.sub(r"[^a-z0-9]+", " ", punched.lower()).split())
        for rs in req_sets:
            if rs and len(set(rs) & pwords) / max(1, len(set(rs))) < 0.5:
                return False    # a caption paraphrased away
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
          cast_names: List[str],
          caption_words: Any = None) -> Dict[str, Any]:
    """Apply validated rewrites; keep the grounded original otherwise.
    The original always survives as beat['narration_plain']; groups whose
    panels carry captions reject any rewrite that drops the caption words."""
    by_gid = {int(p.get("group_id") or 0): str(p.get("narration") or "")
              for p in punched if isinstance(p, dict)}
    caption_words = caption_words or {}
    out = json.loads(json.dumps(beats_obj))
    applied = 0
    for b in out.get("beats") or []:
        gid = int(b.get("group_id") or 0)
        original = str(b.get("narration_plain") or b.get("narration") or "")
        b["narration_plain"] = original
        cand = by_gid.get(gid, "").replace("*", "")  # md emphasis -> TTS-safe
        if cand and validate_line(original, cand, cast_names,
                                  required=caption_words.get(gid)):
            b["narration"] = cand
            applied += 1
        else:
            # rejection RESTORES the grounded line — on an already-punched
            # file the old punch must not survive a failed re-validation
            b["narration"] = original
    out.setdefault("stats", {})["punchup_applied"] = applied
    return out


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Tolerant of code fences, leading prose, trailing junk, and a
    truncated tail (salvages every complete object). A strict regex here
    silently discarded 11 good punched lines once — never again."""
    t = re.sub(r"```(?:json)?", " ", text or "")
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        except Exception:
            pass
    out: List[Dict[str, Any]] = []
    for om in re.finditer(r"\{[^{}]*\}", t):
        try:
            d = json.loads(om.group(0))
        except Exception:
            continue
        if isinstance(d, dict) and "group_id" in d:
            out.append(d)
    return out


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
    ap.add_argument("--episode-dir", default="",
                    help="enables caption protection (vision+scenes manifests)")
    ap.add_argument("--cast", default="")
    ap.add_argument("--backend", choices=["vertex", "ollama"],
                    default="ollama")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--humor", choices=["full", "light"], default="full")
    ap.add_argument("--genre", default="",
                    help="series genre text (murim/modern/system axes); "
                         "auto-read from --script section_genre_mode if given")
    ap.add_argument("--script", default="",
                    help="manifest.script.json for genre auto-detection")
    args = ap.parse_args()
    if not args.genre and args.script and os.path.exists(args.script):
        try:
            sc = json.load(open(args.script))
            modes = [str(x.get("section_genre_mode") or "")
                     for x in sc.get("sections") or []]
            modes = [m for m in modes if m and m != "unknown"]
            if modes:
                args.genre = max(set(modes), key=modes.count)
        except Exception:
            pass

    beats_obj = json.load(open(args.beats))
    # idempotent: always punch from the GROUNDED line — re-running on an
    # already-punched file must not punch the punch (closed-loop drift)
    lines = [{"group_id": int(b.get("group_id") or 0),
              "narration": str(b.get("narration_plain")
                               or b.get("narration") or "")}
             for b in beats_obj.get("beats") or []
             if (b.get("narration_plain") or b.get("narration"))]
    cast_names = _cast_names(args.cast)
    prompt = build_prompt(lines, cast_names, args.humor, genre=args.genre)

    if args.backend == "ollama":
        import ollama  # noqa: F401 — availability probe
        from ollama_compat import chat as _ollama_chat
        resp = _ollama_chat(model=args.ollama_model,
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
    cap_words = (_caption_words_by_group(args.episode_dir, beats_obj)
                 if args.episode_dir else {})
    out = merge(beats_obj, punched, cast_names, caption_words=cap_words)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    n = out["stats"]["punchup_applied"]
    print(f"[ok] wrote={args.out} punched={n}/{len(lines)} "
          f"(rejected lines keep the grounded original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
