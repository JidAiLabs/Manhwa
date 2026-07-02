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
import copy
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from beats_segments import (  # noqa: E402
    beat_segments,
    has_native_segments,
    segment_entries,
    write_segment_lines,
)
from narration_consistency import strip_chrome_opener  # noqa: E402
from niche_modules import register_block  # noqa: E402
from recap_style import (  # noqa: E402
    RECAP_STYLE_RULES,
    dedupe_consecutive_panel_lines,
    is_spoken_fragment,
    neutralize_identity_reveal_leaks,
    repair_spoken_fragments,
)

BASE_PERSONA = """You are the narrator persona of a top manhwa recap channel.
Voice: internet-native, dry, confident, a little sarcastic — a sharp friend
recapping the story, not a movie trailer.

GENRE-NEUTRAL TECHNIQUES (choose at most one when it helps; never force one):
- audience intimacy: "our guy", "our boy", "look at his face"
- comedic hyperbole on impacts ("coughing up half his internal organs")
- punchy standalone fragments for beats: "Total silence." "Deal." "He's in."
- snark at villain logic ("he's definitely not taking his own supply")
- meta-narration ("the stealth mission is officially an action movie now")
- vary line openings; filler openers like "Okay, so" at most ONCE per
  chapter, never on consecutive lines
The comedy/framing AXIS is set by the GENRE block below — it is NOT neutral.
Use only framing that fits THIS manhwa's world.

HARD RULES:
- NEVER invent events, objects, dialogue, or names not present in the
  original line. You RESTYLE the facts; you do not add them. Keep the SAME
  subjects and the SAME counts the line gives you — don't rename what's drawn
  or turn a few into a crowd.
- STAY IN THIS MANHWA'S WORLD: never bolt on a mechanic it doesn't have. The
  GENRE block decides what framing fits; outside a literal game/system world
  there is no "server", "respawn", or "XP" to reach for.
- Keep every character name EXACTLY as written (the cast list is law).
- Paraphrase dialogue in clean narrator language. Never preserve or create a
  quoted run of ALL-CAPS bubble OCR, a truncated fragment, or onomatopoeia.
- Keep the original meaning and emotional turn of the line — an injured
  character stays injured, a defeat stays a defeat.
- Compress visible-only drag aggressively. A rewrite may be 35% of the original
  when it preserves the panel's action/stakes; never pad to match source length.
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
snark ("30 years of qi per pill — supplements have gotten serious").
This world has NO game system — game/RPG framing (XP, respawn, aggro, boss
fights, loot) is off-genre here; never reach for it.""",
    "modern": """GENRE: modern-world (apocalypse/hunter/regression in a
contemporary setting). The world ALREADY has phones and subways — ancient-
setting anachronism jokes do NOT apply. Comedy axis = mundane daily life vs
supernatural stakes ("the apocalypse started before his commute ended",
"monster attacks and his first thought is the deposit on his flat"). If the
protagonist knows the story/future, lean on reader/meta jokes ("he has the
walkthrough; everyone else is playing blind").
This is a REAL world, not a game: never frame its monsters or stakes as game
mechanics — no "aggro", "respawn", "boss fight", "server", "XP". The monsters
are real; say what they are.""",
    "system": """GENRE: system/reincarnation/regression with game windows.
Comedy axis = treating life as a game UI played absurdly well: tutorial and
newbie framing ("skipping the tutorial", "day-one patch notes"), absurd
contrast between the protagonist's situation and power ("a literal infant
grinding stat points"), deadpan quest-log narration of dramatic moments.
Game/RPG framing (XP, boss fights, aggro, loot, respawn, quest log) IS the
native voice here — it's literally how this world works; use it freely.""",
}


# intensity ranking from beats' scene_selection — the deterministic signal that
# sets the persona TEMPERATURE: the channel voice is always on for every beat; the
# tag only tunes it — DRAMATIC drops the jokes (never the voice), CONNECTIVE/COMIC
# run warmer.
_INTENSITY_RANK = {"": 0, "unknown": 0, "calm": 0, "tense": 1,
                   "intense": 2, "explosive": 3}

_COMIC_CUE_RE = re.compile(
    r"\b("
    r"mock|mocking|taunt|taunting|tease|ridicule|laugh|laughter|howl|"
    r"smirk|cocky|smug|manic|humiliat|embarrass|bald|hair disappear|"
    r"where did all your hair|face[- ]?slap|fooling everyone"
    r")\b",
    re.I,
)
_QUOTED_SPEECH_RE = re.compile(
    r"(?:[\"“][^\"”]{2,}[\"”]|(?<!\w)'[^']{2,}'(?!\w))")


def _has_quoted_speech(text: str) -> bool:
    return bool(_QUOTED_SPEECH_RE.search(str(text or "")))


def _comic_cue_score(beat: Dict[str, Any]) -> int:
    """Cheap visual-gag detector for the cinematic/persona blend.

    High panel intensity alone should not suppress humor when the beat itself is
    explicitly a joke, taunt, humiliation, or face-slap. This keeps fight/death
    beats serious while letting recap-channel persona fire on drawn comic relief.
    """
    if not isinstance(beat, dict):
        return 0
    fields: List[str] = []
    for key in ("narration_plain", "narration", "what_happens", "beat_title",
                "hook", "stakes"):
        v = beat.get(key)
        if isinstance(v, str):
            fields.append(v)
    fields.extend(str(x) for x in (beat.get("mood_words") or []) if x)
    for e in beat.get("scene_selection") or []:
        if isinstance(e, dict):
            fields.extend(str(e.get(k) or "") for k in (
                "visual_summary", "ocr_clean", "dialogue", "bubble_mode"))
    blob = " ".join(fields)
    return len(_COMIC_CUE_RE.findall(blob))


def classify_beats(beats_obj: Dict[str, Any]) -> Dict[int, str]:
    """Per-group DRAMATIC/CONNECTIVE label from the strongest scene intensity in
    the beat. The channel voice is always on for ALL beats; the tag only sets the
    TEMPERATURE of that voice — DRAMATIC (intense/explosive) drops the jokes but
    keeps the voice, CONNECTIVE runs warm and witty, and COMIC beats require a short
    grounded punch because the art/text is already playing the moment for mockery or
    humiliation. Deterministic — no LLM."""
    out: Dict[int, str] = {}
    for b in (beats_obj or {}).get("beats") or []:
        try:
            gid = int(b.get("group_id") or 0)
        except (TypeError, ValueError):
            continue
        ranks = [_INTENSITY_RANK.get(str(s.get("intensity") or "").lower(), 0)
                 for s in (b.get("scene_selection") or []) if isinstance(s, dict)]
        if _comic_cue_score(b) > 0:
            out[gid] = "COMIC"
        else:
            out[gid] = "DRAMATIC" if (max(ranks) if ranks else 0) >= 2 else "CONNECTIVE"
    return out


def classify_panel_lines(beats_obj: Dict[str, Any]) -> Dict[tuple, str]:
    """Per-segment style guard so one explosive frame does not mute a whole
    group. A segment is one panel on legacy manifests, a 1-4 panel flow span
    on adaptive beats — a span takes the MAX intensity across its panels
    (peaks preserved)."""
    out: Dict[tuple, str] = {}
    for beat in (beats_obj or {}).get("beats") or []:
        try:
            gid = int(beat.get("group_id") or 0)
        except (TypeError, ValueError):
            continue
        selection = {
            str(item.get("scene_file") or ""): item
            for item in beat.get("scene_selection") or []
            if isinstance(item, dict)
        }
        for i, (seg, entry) in enumerate(
                zip(beat_segments(beat), segment_entries(beat))):
            line = str(entry.get("line_plain") or seg["line"])
            if _COMIC_CUE_RE.search(line):
                out[(gid, i)] = "COMIC"
                continue
            rank = max(
                (_INTENSITY_RANK.get(
                    str((selection.get(f) or {}).get("intensity") or "").lower(), 0)
                 for f in seg["span"]),
                default=0)
            out[(gid, i)] = "DRAMATIC" if rank >= 2 else "CONNECTIVE"
    return out


CINEMATIC_RULES = """THE CHANNEL VOICE IS THE BASELINE — write EVERY line in the
persona: internet-native, dry, confident, a little arrogant — a sharp friend
recapping the story, not a movie trailer narrator. This voice is ALWAYS ON, even on
grave beats; it never switches off. Use the DRAMATIC/CONNECTIVE/COMIC tag ONLY to set
the TEMPERATURE of that voice, never to remove it:
- DRAMATIC (intense/explosive, somber, tragic, danger): keep the voice and its
  confidence, but DROP THE JOKES — no winks or deflating asides; let the menace,
  stakes, and consequence land in the same dry, characterful voice.
- CONNECTIVE / mundane-aside: the voice runs warm and witty — this is where asides,
  light hyperbole, and intimate stand-ins ("our guy"/"our boy") land most.
- COMIC (mockery, humiliation, a visual gag, a face-slap): the beat is already a joke
  — add ONE sharp recap-channel punch so it lands. The punch must be clearly
  figurative/framing, never a new story event.
Cinematic phrasing (strong verbs, rhythm, stakes) is the floor for EVERY line; it does
NOT mean adding weather, lighting, hair, mist, or trailer-grade atmosphere the viewer
can already see. The NICHE TEMPERATURE block (when present) further tunes how
hot/cold/funny this voice runs.
STORY CAPTIONS / narration-box text: WEAVE them into the line in the story's own
first-person voice — you MAY rephrase for flow, but keep their MEANING and any key
phrase, and never read a caption robotically as a bare standalone fragment.
Keep every grounding rule: no invented facts, cast names verbatim, caption meaning
preserved, mood tags preserved, no chrome."""


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
                 humor: str, genre: str = "",
                 classes: Optional[Dict[Any, str]] = None,
                 story_context: str = "", niche: str = "",
                 niche_secondary: str = "") -> str:
    """Build the LLM prompt for either per-beat or per-panel lines.

    When *lines* contain ``panel_index``, the contract is per-panel:
    the LLM must return the SAME array with ``{group_id, panel_index,
    narration}`` objects in the same order. Without ``panel_index`` the
    legacy per-beat contract (``{group_id, narration}``) applies.
    """
    cast = ", ".join(cast_names) if cast_names else "(none listed)"
    addon = GENRE_ADDONS.get(genre_key(genre), "")
    guide = BASE_PERSONA + ("\n\n" + addon if addon else "")
    nblock = register_block(niche, niche_secondary)
    if nblock:
        guide += "\n\n" + nblock
    guide += "\n\n" + RECAP_STYLE_RULES
    if story_context:
        guide += ("\n\nWHOLE-CHAPTER STORY SPINE (context only; invent nothing):\n"
                  + story_context)
    per_panel = lines and "panel_index" in lines[0]
    if humor == "cinematic":
        guide += "\n\n" + CINEMATIC_RULES
        cls = classes or {}
        if per_panel:
            payload = [{"group_id": l["group_id"],
                        "panel_index": l["panel_index"],
                        "style": cls.get(
                            (int(l["group_id"]), int(l["panel_index"])),
                            cls.get(int(l["group_id"]), "CONNECTIVE")),
                        "must_paraphrase_dialogue":
                            _has_quoted_speech(l["narration"]),
                        "narration": l["narration"]} for l in lines]
        else:
            payload = [{"group_id": l["group_id"],
                        "style": cls.get(int(l["group_id"]), "CONNECTIVE"),
                        "narration": l["narration"]} for l in lines]
    else:
        if per_panel:
            payload = [{"group_id": l["group_id"],
                        "panel_index": l["panel_index"],
                        "must_paraphrase_dialogue":
                            _has_quoted_speech(l["narration"]),
                        "narration": l["narration"]} for l in lines]
        else:
            payload = [{"group_id": l["group_id"], "narration": l["narration"]}
                       for l in lines]
    if per_panel:
        return_schema = (
            "{\"group_id\": int, \"panel_index\": int, \"narration\": str} — "
            "SAME length, SAME group_id+panel_index pairs, same order, "
            "rewrite each narration line in the persona, NEVER merge or drop "
            "lines"
        )
    else:
        return_schema = (
            "{\"group_id\": int, \"narration\": str} — same "
            "group_ids, same order, no commentary"
        )
    pace_rule = (
        "\nPACING CONTRACT: choose length from the panel's narrative job, not "
        "from a chapter average. A sword clash, blink reaction, or clean impact "
        "can be one sharp beat. A reveal, inner decision, rule explanation, or "
        "main-story turn can breathe longer. Do not pad ordinary panels, and do "
        "not compress important thought just to be short.\n"
        if per_panel else "")
    return (f"{guide}\n\nHUMOR={humor}\nCAST NAMES (verbatim): {cast}\n"
            f"{pace_rule}\n"
            "Rewrite EVERY line below in the persona. When "
            "must_paraphrase_dialogue=true, convey the speech or thought "
            "INDIRECTLY in clean narrator language and use NO quotation marks; "
            "never read bubble OCR aloud. Do not invent quoted dialogue or "
            "quoted thoughts. Ensure every output is natural, "
            "grammatical spoken English AND an independently speakable complete "
            "clause: never begin with ellipsis/lowercase continuation and never "
            "end with a comma, colon, or semicolon. Return ONLY a JSON "
            f"array of objects {return_schema}.\n\nLINES:\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))


_MOOD_RE = re.compile(r"^\s*(\[[a-z _-]+\])", re.I)

# Span word budget (spec 2026-07-02 §3.1: "the budget survives punchup").
# Tiny arithmetic duplicated from gemini_narrative_pass.validate_segments —
# importing that module would pull google-genai into this ollama-first tool.
# A span of N panels must carry enough words that its ONE clip holds every
# panel >= 2.0s at 135 wpm; the ceiling only guards absurd bloat (see the
# writer validator's rationale — 6.0 hard-failed gemma's natural rhythm).
_SPAN_WPM = 135.0                 # == gemini_narrative_pass.WPM / script default
_SPAN_MIN_SEC_PER_PANEL = 2.0     # planner's per-panel on-screen floor
_SPAN_MAX_SEC_PER_PANEL = 10.0    # parity-pinned to the writer validator


def span_budget_ok(n_panels: int, text: str) -> bool:
    """True when `text` fits its span's duration-aware word budget
    (N*2.0s <= words/(wpm/60) <= N*6.0s). Leading mood tags don't count."""
    n = max(1, int(n_panels or 0))
    body = _MOOD_RE.sub("", str(text or "")).strip()
    sec = len(body.split()) / (_SPAN_WPM / 60.0)
    return (n * _SPAN_MIN_SEC_PER_PANEL) <= sec <= (n * _SPAN_MAX_SEC_PER_PANEL)


def _word_count(s: str) -> int:
    return len(re.findall(r"[\w']+", s))


def _has_repeated_sentence_loop(text: str) -> bool:
    parts = [p.strip().lower() for p in re.split(r"[.!?]+", text or "")
             if p.strip()]
    if len(parts) < 4:
        return False
    seen: Dict[str, int] = {}
    for p in parts:
        words = re.findall(r"[\w']+", p)
        if len(words) < 2:
            continue
        key = " ".join(words[:12])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 4:
            return True
    return False


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
                  required: Any = None, max_ratio: float = 1.5,
                  forbid_quotes: bool = False,
                  forbid_fragments: bool = False) -> bool:
    """Reject rewrites that break the grounding contract.

    ``max_ratio`` is accepted for old callers but no longer controls narrative
    length. Pace belongs to the panel: a fast action may be tiny, while a
    thought/reveal panel may need room. This gate only catches broken output,
    chrome, name/caption loss, quote leaks, and obvious model loops.
    """
    if not punched or not punched.strip():
        return False
    if _has_repeated_sentence_loop(punched):
        return False
    if forbid_quotes and _has_quoted_speech(punched):
        return False
    if forbid_fragments and is_spoken_fragment(punched):
        return False
    if required:
        req_sets = ([required] if isinstance(required, (set, frozenset))
                    else list(required))
        pwords = set(re.sub(r"[^a-z0-9]+", " ", punched.lower()).split())
        for rs in req_sets:
            if rs and len(set(rs) & pwords) / max(1, len(set(rs))) < 0.5:
                return False    # a caption paraphrased away
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
          caption_words: Any = None,
          classes: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """Apply validated rewrites; keep the grounded original otherwise.
    The original always survives as beat['narration_plain']; groups whose
    panels carry captions reject any rewrite that drops the caption words.
    Panel length is not gated here; ``classes`` is accepted for compatibility
    with older callers."""
    by_gid = {int(p.get("group_id") or 0): str(p.get("narration") or "")
              for p in punched if isinstance(p, dict)}
    caption_words = caption_words or {}
    classes = classes or {}
    out = json.loads(json.dumps(beats_obj))
    applied = 0
    for b in out.get("beats") or []:
        gid = int(b.get("group_id") or 0)
        original = str(b.get("narration_plain") or b.get("narration") or "")
        b["narration_plain"] = original
        cand = by_gid.get(gid, "").replace("*", "")  # md emphasis -> TTS-safe
        if cand and validate_line(original, cand, cast_names,
                                  required=caption_words.get(gid),
                                  forbid_quotes=True):
            b["narration"] = cand
            applied += 1
        else:
            # rejection RESTORES the grounded line — on an already-punched
            # file the old punch must not survive a failed re-validation
            b["narration"] = original
        # scrub series-intro/title-card chrome at the SOURCE so the script, plan
        # and audio all inherit the same clean narration (no cross-stage desync
        # that would trip narration_stale). Title-agnostic; spares story nouns.
        b["narration"] = strip_chrome_opener(b["narration"])
        b["narration_plain"] = strip_chrome_opener(b["narration_plain"])
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


def _batch_lines(lines: List[Dict[str, Any]], batch_size: int,
                 *, max_payload_chars: int = 42000) -> List[List[Dict[str, Any]]]:
    """Split only for model transport/context safety.

    A positive ``batch_size`` is an explicit operator override. The default
    adaptive path has no opinion about story pacing or panel importance; it just
    keeps the JSON payload comfortably inside the model context.
    """
    if not lines:
        return []
    if batch_size and batch_size > 0:
        size = max(1, int(batch_size))
        return [lines[i:i + size] for i in range(0, len(lines), size)]
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0
    limit = max(8000, int(max_payload_chars or 42000))
    for line in lines:
        line_chars = len(json.dumps(line, ensure_ascii=False)) + 2
        if current and current_chars + line_chars > limit:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(line)
        current_chars += line_chars
    if current:
        batches.append(current)
    return batches


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


def _story_context(story_path: str) -> str:
    if not story_path or not os.path.exists(story_path):
        return ""
    try:
        obj = json.load(open(story_path))
    except Exception:
        return ""
    parts = [str(obj.get("logline") or "").strip(),
             str(obj.get("premise") or "").strip()]
    return "\n".join(p for p in parts if p)


def _load_niche(episode_dir, explicit_primary="", explicit_secondary=""):
    """Explicit args win; else read <episode_dir>/manifest.series.json; else ('','')."""
    if explicit_primary:
        return (explicit_primary, explicit_secondary)
    try:
        with open(os.path.join(episode_dir, "manifest.series.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        return (str(d.get("niche_primary") or ""),
                str(d.get("niche_secondary") or ""))
    except Exception:
        return ("", "")


def infer_genre_from_content(beats_obj: Dict[str, Any], ep_dir: str = "") -> str:
    """Read the manhwa TYPE off the chapter's own content so the persona adapts
    without any per-series config: a game/system world (status windows, skills,
    quests) keeps its game voice; a murim world (sect/qi/cultivation) its wuxia
    snark; everything else is treated as a real, modern world. This is the
    'understanding of the manhwa type' driving the persona — it classifies the
    WORLD from what's on the page, it does not blacklist narration words. Reads the
    grounded narration plus, when available, the raw OCR (a cleaner signal)."""
    blob = " ".join(str(b.get("narration_plain") or b.get("narration") or "")
                    for b in (beats_obj or {}).get("beats") or []).lower()
    try:
        if ep_dir:
            v = json.load(open(os.path.join(ep_dir, "manifest.vision.json")))
            blob += " " + " ".join(str(i.get("ocr_clean") or "")
                                   for i in (v.get("items") or [])).lower()
    except Exception:
        pass
    system = ("status window", "status screen", "notification window", "level up",
              "leveled up", " skill ", "skill tree", " quest", "system message",
              "stat point", "stat window", "dungeon", "awaken", " mana ", "[skill]",
              "[level", "ding!", "you have")
    murim = ("sect", " qi ", "martial art", "cultivat", "murim", "meridian",
             "dao ", "pavilion", "inner energy", "clan ", "ancestor")
    ss = sum(1 for w in system if w in blob)
    ms = sum(1 for w in murim if w in blob)
    if ss >= 2 and ss >= ms:
        return "system"
    if ms >= 2 and ms > ss:
        return "murim"
    return "modern"


def build_panel_payload(beats_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten every beat's narration segments into a per-segment list for the
    LLM (a segment = one panel on legacy manifests, a 1-4 panel flow span on
    adaptive beats).

    Each entry is ``{group_id, panel_index, narration}`` where ``narration``
    is the grounded original (``line_plain`` when a previous punch stamped
    one) — idempotent, always punches from the grounded original even on a
    re-run. ``panel_index`` is the SEGMENT index, matching the enumeration
    classify_panel_lines and apply_panel_punchup use.
    """
    out: List[Dict[str, Any]] = []
    for b in beats_obj.get("beats") or []:
        gid = int(b.get("group_id") or 0)
        for i, (seg, entry) in enumerate(
                zip(beat_segments(b), segment_entries(b))):
            narration = str(entry.get("line_plain") or seg["line"])
            out.append({"group_id": gid, "panel_index": i, "narration": narration})
    return out


def apply_panel_punchup(
    beat: Dict[str, Any],
    rewrites: Dict[tuple, str],
    cast_names: Optional[List[str]] = None,
    caption_words: Optional[Dict[int, Any]] = None,
    classes: Optional[Dict[Any, str]] = None,
) -> int:
    """Apply validated per-segment rewrites in-place; set line_plain; rejoin
    narration. A segment is one panel on legacy manifests, a 1-4 panel flow
    span on adaptive beats.

    *rewrites* maps ``(group_id, segment_index) -> candidate string`` (the
    index enumerates beat_segments(beat), same as build_panel_payload).
    Per-line grounding gate (validate_line) mirrors the per-beat merge()
    logic. On native-segments beats a rewrite must ALSO keep the span word
    budget (spec §3.1 — the duration arithmetic survives punchup) or the
    grounded original is kept, exactly like the caption-preservation
    fallback. Legacy manifests keep today's no-budget behavior.
    The joined beat["narration"] and beat["narration_plain"] are always
    updated. Returns the count of segments whose rewrite was accepted.
    """
    gid = int(beat.get("group_id") or 0)
    cast_names = cast_names or []
    caption_words = caption_words or {}
    classes = classes or {}

    required = caption_words.get(gid)

    segs = beat_segments(beat)
    entries = segment_entries(beat)   # the actual mutable entry dicts
    if not entries:
        return 0
    budget_gated = has_native_segments(beat)
    accepted = 0
    lines: List[str] = []
    plains: List[str] = []
    for i, (seg, entry) in enumerate(zip(segs, entries)):
        original = str(entry.get("line_plain") or seg["line"])
        cand = str(rewrites.get((gid, i), "") or "").replace("*", "")
        if cand and validate_line(original, cand, cast_names,
                                  required=required,
                                  forbid_quotes=True,
                                  forbid_fragments=True) \
                and (not budget_gated
                     or span_budget_ok(len(seg["span"]), cand)):
            line = cand
            accepted += 1
        else:
            line = original
        # strip_chrome_opener can empty a chrome-only line; keep the original
        # then (an empty line would delete the segment and orphan its span —
        # write_segment_lines refuses it).
        lines.append(strip_chrome_opener(line) or line)
        plains.append(strip_chrome_opener(original) or original)

    write_segment_lines(beat, lines)   # entry lines + the narration join
    for entry, plain in zip(entries, plains):
        entry["line_plain"] = plain
    beat["narration_plain"] = " ".join(plains)
    return accepted


def _vision_by_file(ep_dir: str) -> Dict[str, Dict[str, Any]]:
    """{scene_file: vision item} from manifest.vision.json (for the identity
    backstop's concealment/OCR signals). Empty on any failure."""
    if not ep_dir:
        return {}
    try:
        v = json.load(open(os.path.join(ep_dir, "manifest.vision.json")))
        return {str(i.get("scene_file")): i for i in v.get("items") or []
                if i.get("scene_file")}
    except Exception:
        return {}


def _understood_by_file(ep_dir: str) -> Dict[str, Dict[str, Any]]:
    """{scene_file: understood panel} from manifest.panels.understood.json (the
    per-panel subjects/description the identity backstop matches against the
    protagonist fingerprint). Empty on any failure."""
    if not ep_dir:
        return {}
    try:
        u = json.load(open(os.path.join(ep_dir, "manifest.panels.understood.json")))
        return {str(p.get("scene_file")): p for p in u.get("panels") or []
                if p.get("scene_file")}
    except Exception:
        return {}


def _cast_obj(cast_path: str) -> Dict[str, Any]:
    """Full manifest.cast.json object (needed for protagonist names + visual
    fingerprint); _cast_names() only returns the bare names."""
    if not cast_path or not os.path.exists(cast_path):
        return {"cast": []}
    try:
        obj = json.load(open(cast_path))
        return obj if isinstance(obj, dict) else {"cast": []}
    except Exception:
        return {"cast": []}


def apply_post_punchup_backstop(
    out: Dict[str, Any],
    cast_obj: Dict[str, Any],
    vision_by_file: Dict[str, Any],
    understood_by_file: Dict[str, Any],
) -> Dict[str, int]:
    """Re-assert the identity/dedup backstop AFTER the persona pass, in place.

    The beats pass already neutralizes a concealed/unresolved figure's identity
    BEFORE this punchup runs — but the persona rewrite can re-attach a
    protagonist handle ("our guy") to that same still-concealed figure, so the
    mis-ID would resurface in the FINAL beats that ship. Re-running
    neutralize_identity_reveal_leaks (+ the consecutive-dup merge), exactly as
    gemini_narrative_pass does at ITS tail, makes the persona pass unable to leave
    a concealed figure tagged as the protagonist or ship a duplicated consecutive
    line. Agnostic: cues/handles are generic; identity comes only from the cast +
    the per-panel understanding."""
    n_ident = neutralize_identity_reveal_leaks(
        out, cast_obj or {"cast": []}, vision_by_file or {},
        understood_by_file or {})
    n_dups = dedupe_consecutive_panel_lines(out)
    stats = out.setdefault("stats", {})
    stats["identity_reveals_neutralized"] = n_ident
    stats["consecutive_dups_merged"] = n_dups
    return {"identity_reveals_neutralized": n_ident,
            "consecutive_dups_merged": n_dups}


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
    ap.add_argument("--humor", choices=["full", "light", "cinematic"],
                    default="full")
    ap.add_argument("--genre", default="",
                    help="series genre text (murim/modern/system axes); "
                         "auto-read from --script section_genre_mode if given")
    ap.add_argument("--script", default="",
                    help="manifest.script.json for genre auto-detection")
    ap.add_argument("--niche", default="",
                    help="manhwa niche A/B/C/D; default reads --episode-dir/manifest.series.json")
    ap.add_argument("--niche-secondary", default="")
    ap.add_argument("--story", default="",
                    help="manifest.story.json for whole-chapter spine context")
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("STUDIO_PUNCHUP_BATCH_SIZE", "0")),
                    help="per-panel lines per rewrite call; 0=auto by context")
    ap.add_argument("--batch-workers", type=int,
                    default=int(os.environ.get("STUDIO_PUNCHUP_WORKERS", "2")),
                    help="parallel Ollama rewrite calls (Vertex stays serial)")
    ap.add_argument("--num-ctx", type=int,
                    default=int(os.environ.get("STUDIO_PUNCHUP_NUM_CTX", "16384")),
                    help="Ollama context window for rewrite calls")
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
    # the persona follows the manhwa TYPE: when no genre is given (the beated stage
    # runs before the script exists), read it from the chapter's own content.
    if not args.genre:
        args.genre = infer_genre_from_content(beats_obj, args.episode_dir)

    cast_names = _cast_names(args.cast)
    cap_words = (_caption_words_by_group(args.episode_dir, beats_obj)
                 if args.episode_dir else {})

    # Per-segment path: any beat with narration segments (native adaptive
    # `segments` OR legacy per-panel lines) activates this mode. Fall back to
    # the legacy per-beat path for old manifests without either shape.
    use_per_panel = any(beat_segments(b)
                        for b in beats_obj.get("beats") or [])
    if args.humor == "cinematic":
        classes = (classify_panel_lines(beats_obj)
                   if use_per_panel else classify_beats(beats_obj))
    else:
        classes = {}

    if use_per_panel:
        lines = build_panel_payload(beats_obj)
    else:
        # idempotent: always punch from the GROUNDED line — re-running on an
        # already-punched file must not punch the punch (closed-loop drift)
        lines = [{"group_id": int(b.get("group_id") or 0),
                  "narration": str(b.get("narration_plain")
                                   or b.get("narration") or "")}
                 for b in beats_obj.get("beats") or []
                 if (b.get("narration_plain") or b.get("narration"))]

    payload_chars = max(8000, int(args.num_ctx * 2.6))
    batches = (_batch_lines(lines, args.batch_size,
                            max_payload_chars=payload_chars)
               if use_per_panel else [lines])
    story_context = _story_context(args.story)
    niche_p, niche_s = _load_niche(args.episode_dir, args.niche,
                                   args.niche_secondary)

    def _prompt(batch: List[Dict[str, Any]], index: int) -> str:
        return build_prompt(
            batch, cast_names, args.humor, genre=args.genre,
            classes=classes,
            story_context=story_context,
            niche=niche_p, niche_secondary=niche_s)

    if args.backend == "ollama":
        import ollama  # noqa: F401 — availability probe
        from ollama_compat import chat as _ollama_chat

        def _run_ollama(item: tuple[int, List[Dict[str, Any]]]):
            index, batch = item
            resp = _ollama_chat(
                model=args.ollama_model,
                messages=[{"role": "user", "content": _prompt(batch, index)}],
                think=False,
                options={
                    "temperature": 0.7,
                    "num_ctx": args.num_ctx,
                    # Transport guard only: enough room for flexible pacing
                    # without turning max tokens into a style rule.
                    "num_predict": max(900, len(batch) * 90),
                })
            raw = (resp.get("message") or {}).get("content") or ""
            return index, _extract_json_array(raw)

        work = list(enumerate(batches))
        workers = max(1, min(int(args.batch_workers or 1), len(work)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_ollama, work))
        punched = [row for _i, rows in sorted(results) for row in rows]
    else:
        from thumbnail_gen import _make_client  # self-heals stale cred paths
        attempts = _make_client(args.location)
        if not attempts:
            print("[err] no auth available")
            return 1
        _, client = attempts[0]
        punched = []
        for index, batch in enumerate(batches):
            resp = client.models.generate_content(
                model=args.model, contents=[_prompt(batch, index)])
            punched.extend(_extract_json_array(resp.text or ""))

    if use_per_panel:
        # Build (group_id, panel_index) -> narration lookup from the LLM response
        rewrites = {(int(p.get("group_id") or 0), int(p.get("panel_index") or 0)):
                    str(p.get("narration") or "")
                    for p in punched if isinstance(p, dict)
                    and "panel_index" in p}
        out = copy.deepcopy(beats_obj)
        applied = 0
        for b in out.get("beats") or []:
            applied += apply_panel_punchup(b, rewrites, cast_names=cast_names,
                                           caption_words=cap_words, classes=classes)
        out.setdefault("stats", {})["punchup_applied"] = applied
    else:
        out = merge(beats_obj, punched, cast_names, caption_words=cap_words,
                    classes=classes)
        applied = out["stats"]["punchup_applied"]

    out.setdefault("stats", {})["spoken_fragments_repaired"] = (
        repair_spoken_fragments(out))

    # POST-PUNCHUP IDENTITY BACKSTOP: the persona pass runs AFTER the beats pass
    # already neutralized concealed-figure identities, and it can re-introduce a
    # protagonist handle ("our guy") on a still-unresolved figure. Re-apply the
    # identity + consecutive-dup backstop here so a mis-ID can never survive into
    # the final beats that ship.
    apply_post_punchup_backstop(
        out, _cast_obj(args.cast),
        _vision_by_file(args.episode_dir),
        _understood_by_file(args.episode_dir))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote={args.out} punched={applied}/{len(lines)} "
          f"(rejected lines keep the grounded original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
