#!/usr/bin/env python3
"""
script_expander.py (CONSOLIDATED)
- Visual-anchored manhwa recap writing (no camera language)
- Genre inference + trope pack injection
- ElevenLabs v3 leading mood tags + repair
- Semantic SFX cues extraction (labels, not file paths)
- Deterministic segment_id = g####_p## for downstream TTS/timeline alignment

Requires:
  pip install -U openai

Run:
  python3 script_expander.py --beats manifest.beats.json --vision manifest.vision.json --out manifest.script.json --model gpt-4.1-mini --resume
"""

import argparse
import inspect
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# Shared exact-token + estimated-cost accounting (sibling tool module).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usage_cost import UsageAccumulator  # noqa: E402


def _openai_usage(resp: Any) -> Dict[str, int]:
    """Exact (input, output) token counts from an OpenAI response.

    Handles both the Responses API (input_tokens/output_tokens) and Chat
    Completions (prompt_tokens/completion_tokens).
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input": 0, "output": 0, "cached": 0}
    inp = getattr(u, "input_tokens", None)
    if inp is None:
        inp = getattr(u, "prompt_tokens", 0)
    out = getattr(u, "output_tokens", None)
    if out is None:
        out = getattr(u, "completion_tokens", 0)
    # cached input tokens: Responses API -> input_tokens_details.cached_tokens;
    # Chat Completions -> prompt_tokens_details.cached_tokens.
    cached = 0
    for attr in ("input_tokens_details", "prompt_tokens_details"):
        d = getattr(u, attr, None)
        if d is not None:
            cached = int(getattr(d, "cached_tokens", 0) or 0)
            break
    return {"input": int(inp or 0), "output": int(out or 0), "cached": cached}

# =============================================================================
# ElevenLabs v3 tags (leading tag must be one of these)
# =============================================================================
V3_VALID_TAGS = {
    "calm", "tense", "urgent", "excited", "awe", "sad",
    "whisper", "angry", "nervous", "panicked", "serious"
}
_LEADING_TAG_RE = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*")

def _split_leading_bracket_tag(text: str) -> Tuple[Optional[str], str]:
    """
    If text starts with [TAG], return (tag, rest). Else (None, text).
    Keeps the original casing in rest; tag is returned as raw inside brackets.
    """
    if not isinstance(text, str):
        return None, ""
    m = _LEADING_TAG_RE.match(text)
    if not m:
        return None, text.strip()
    tag = (m.group(1) or "").strip()
    rest = text[m.end():].lstrip()
    return tag, rest

def _remove_leading_bracket_tag_only(text: str) -> str:
    tag, rest = _split_leading_bracket_tag(text)
    return rest if tag is not None else (text or "").strip()

# =============================================================================
# Semantic SFX labels (NOT file paths)
# =============================================================================
SFX_SEMANTIC_LABELS = {
    "MONSTER_GROWL": "Generic monster/creature growl",
    "DRAGON_VOICE": "Dragon roar/screech (any dragon type)",
    "BEAST_ROAR": "Large beast roar",
    "CREATURE_SCREECH": "High-pitched creature sound",
    "SWORD_CLASH": "Metal weapon clash",
    "IMPACT_HEAVY": "Heavy impact/hit",
    "IMPACT_BODY": "Body impact/fall",
    "WEAPON_WHOOSH": "Weapon swing through air",
    "FOOTSTEP_HEAVY": "Heavy footstep or stomp",
    "FOOTSTEPS_RUN": "Running footsteps",
    "MOVEMENT_DASH": "Quick dash/sprint sound",
    "EXPLOSION": "Explosion sound",
    "BUILDING_COLLAPSE": "Structure collapsing",
    "GROUND_CRACK": "Ground/ice cracking",
    "RUMBLE": "Ground rumbling/trembling",
    "WIND_GUST": "Wind gust",
    "WATER_SPLASH": "Water splash",
    "FIRE_CRACKLE": "Fire burning",
    "BREATHING_HEAVY": "Heavy breathing/gasping",
    "GASP": "Sharp gasp",
    "AMB_BATTLE": "Battle ambience (distant)",
    "AMB_CAVE": "Cave ambience",
    "AMB_WIND": "Wind ambience",
    "AMB_RAIN": "Rain ambience",
    "AMB_CROWD": "Crowd ambience",
}

ONOMATOPOEIA_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bGRRR+\b", re.I), "MONSTER_GROWL"),
    (re.compile(r"\bSKREEE+\b", re.I), "DRAGON_VOICE"),
    (re.compile(r"\bGRA+H+\b|\bGRAAA+\b", re.I), "BEAST_ROAR"),
    (re.compile(r"\bSCREECH\b|\bSCREEE+\b", re.I), "CREATURE_SCREECH"),
    (re.compile(r"\bCLANG\b|\bCLINK\b|\bCLASH\b", re.I), "SWORD_CLASH"),
    (re.compile(r"\bWHAM\b|\bBAM\b|\bCRASH\b|\bSMASH\b|\bTHUD\b", re.I), "IMPACT_HEAVY"),
    (re.compile(r"\bKICK\b|\bSLAM\b|\bBODY\b\s+\bHIT\b", re.I), "IMPACT_BODY"),
    (re.compile(r"\bWHOOSH\b|\bSWISH\b", re.I), "WEAPON_WHOOSH"),
    (re.compile(r"\bSTOMP\b|\bSTEP\b", re.I), "FOOTSTEP_HEAVY"),
    (re.compile(r"\bRUN\b|\bFOOTSTEPS\b", re.I), "FOOTSTEPS_RUN"),
    (re.compile(r"\bDASH\b|\bBLURT\b|\bBLUR\b", re.I), "MOVEMENT_DASH"),
    (re.compile(r"\bBOOM\b|\bEXPLODE\b|\bEXPLOSION\b", re.I), "EXPLOSION"),
    (re.compile(r"\bCOLLAPSE\b|\bCRUMBLE\b", re.I), "BUILDING_COLLAPSE"),
    (re.compile(r"\bCRACK\b|\bSHATTER\b", re.I), "GROUND_CRACK"),
    (re.compile(r"\bRUMBLE\b|\bTREMBLE\b", re.I), "RUMBLE"),
    (re.compile(r"\bGUST\b|\bWIND\b", re.I), "WIND_GUST"),
    (re.compile(r"\bSPLASH\b", re.I), "WATER_SPLASH"),
    (re.compile(r"\bCRACKLE\b|\bFIRE\b", re.I), "FIRE_CRACKLE"),
    (re.compile(r"\bHUFF\b|\bPANT\b|\bGASP\b", re.I), "BREATHING_HEAVY"),
]

# =============================================================================
# Anti-camera-language
# =============================================================================
# NOTE: "our protagonist"/"our hero"/"the character(s)" used to live here, but
# R4 reconciliation ALLOWS persona terms like "protagonist", "MC", "the MC",
# "antagonist". We still ban generic camera/we-our-us speak. The only
# "our"/"we" forms banned are the camera-voice phrases ("we see", etc.).
BANNED_PHRASES = [
    "we see", "we witness", "we watch", "let's", "let us",
    "the camera", "wide shot", "close-up", "camera pulls", "camera zooms",
    "shot reveals", "shot shows", "the scene shows", "the scene reveals",
    "establishing shot", "medium shot", "pov shot", "quick cuts",
]
CAMERA_WORDS = [
    "camera", "shot", "wide", "close-up", "pan", "zoom", "frame",
    "angle", "pulls back", "cuts to", "reveals", "focuses on",
    "establishing", "medium shot", "pov", "quick cuts"
]

# =============================================================================
# Manhwa-recap vocabulary (R4) — curated lexicon injected into the prompt so the
# narration uses genre-native terms naturally (only where the scene supports it).
# =============================================================================
MANHWA_JARGON: List[str] = [
    "protagonist",
    "antagonist",
    "MC (main character)",
    "the MC",
    "regressor / regression / return",
    "hunter",
    "awakening / awakener",
    "S-rank / F-rank (power ranking)",
    "gate / dungeon",
    "murim",
    "cultivation / cultivator",
    "qi / mana",
    "dantian",
    "breakthrough",
    "aura farming (presence overwhelming before any motion)",
    "dumbfounded",
    "face-slap / face-slapping (an arrogant elite humbled)",
    "OP (overpowered)",
    "power scaling",
    "sect",
]

def _manhwa_jargon_block() -> str:
    """Render the jargon lexicon as a single bullet list for prompt injection."""
    return "\n".join(f"- {term}" for term in MANHWA_JARGON)

# =============================================================================
# Voice settings (Eleven v3 constraint: stability must be 0.0, 0.5, or 1.0)
# =============================================================================
def _quantize_v3_stability(x: float) -> float:
    x = float(x)
    if x < 0.25:
        return 0.0
    if x < 0.75:
        return 0.5
    return 1.0

DEFAULT_V3_VOICE = {
    "stability": 0.5,
    "style": 0.35,
    "speed": 1.08,
    "similarity_boost": 0.78,
    "use_speaker_boost": False,
}

V3_PRESET_BY_TAG: Dict[str, Dict[str, Any]] = {
    "calm": {"stability": 0.5, "style": 0.25, "speed": 1.02},
    "serious": {"stability": 0.5, "style": 0.25, "speed": 1.04},
    "tense": {"stability": 0.5, "style": 0.45, "speed": 1.06},
    "urgent": {"stability": 0.0, "style": 0.60, "speed": 1.12},
    "excited": {"stability": 0.0, "style": 0.65, "speed": 1.12},
    "awe": {"stability": 0.5, "style": 0.45, "speed": 1.03},
    "sad": {"stability": 1.0, "style": 0.30, "speed": 0.96},
    "whisper": {"stability": 1.0, "style": 0.25, "speed": 0.98},
    "angry": {"stability": 0.0, "style": 0.70, "speed": 1.06},
    "nervous": {"stability": 0.5, "style": 0.45, "speed": 1.08},
    "panicked": {"stability": 0.0, "style": 0.75, "speed": 1.16},
}

def _build_voice_settings_for_tag(tag: str) -> Dict[str, Any]:
    t = (tag or "serious").strip().lower()
    base = dict(DEFAULT_V3_VOICE)
    preset = V3_PRESET_BY_TAG.get(t, {})
    for k, v in preset.items():
        base[k] = v
    base["stability"] = _quantize_v3_stability(float(base.get("stability", 0.5)))
    return base

# =============================================================================
# Small utils
# =============================================================================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", str(s or "")))

def _count_words(paras: List[str]) -> int:
    return sum(_words(str(p)) for p in (paras or []) if isinstance(p, str))

def _within_tolerance(actual: int, target: int, tol: float) -> bool:
    if target <= 0:
        return True
    lo = int(target * (1.0 - tol))
    hi = int(target * (1.0 + tol))
    return lo <= actual <= hi

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def _safe_join_lines(lines: List[str], max_items: int = 6) -> str:
    out: List[str] = []
    for s in (lines or [])[:max_items]:
        t = re.sub(r"\s+", " ", str(s or "")).strip()
        if t:
            out.append(t)
    return " | ".join(out)

def build_story_so_far(prior_sections: List[Dict[str, Any]], max_chars: int = 600) -> str:
    """
    R3 — cross-section continuity. Build a compact running synopsis of what has
    happened in the recap so far, from already-generated sections.

    Pure helper (no LLM): concatenates each prior section's key narration
    (section_summary, falling back to its script paragraphs) plus its
    cliffhanger, into a single compact string truncated to `max_chars`.

    Empty input -> "" so the caller can skip injecting a STORY SO FAR block.
    """
    if not prior_sections:
        return ""

    fragments: List[str] = []
    for sec in prior_sections:
        if not isinstance(sec, dict):
            continue

        # Prefer the model's own section_summary; fall back to joined paragraphs.
        summary = re.sub(r"\s+", " ", str(sec.get("section_summary") or "")).strip()
        if not summary:
            paras = sec.get("script_paragraphs") or []
            joined = " ".join(
                re.sub(r"\s+", " ", str(p)).strip()
                for p in paras
                if isinstance(p, str) and p.strip()
            )
            summary = joined.strip()

        if summary:
            fragments.append(summary)

        cliff = re.sub(r"\s+", " ", str(sec.get("cliffhanger_line") or "")).strip()
        if cliff:
            fragments.append(cliff)

    synopsis = " ".join(f for f in fragments if f).strip()
    synopsis = re.sub(r"\s+", " ", synopsis).strip()
    if not synopsis:
        return ""

    if max_chars and len(synopsis) > max_chars:
        synopsis = synopsis[: max_chars - 1].rstrip() + "…"
    return synopsis

# =============================================================================
# Genre inference + trope pack (YOUR functions, kept)
# =============================================================================
def _infer_genre_mode(beats_chunk: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for b in (beats_chunk or []):
        if not isinstance(b, dict):
            continue
        parts.append(
            (
                str(b.get("beat_title") or "")
                + " "
                + str(b.get("what_happens") or "")
                + " "
                + str(b.get("conflict_or_stakes") or "")
                + " "
                + str(b.get("reveals_or_info") or "")
                + " "
                + " ".join([str(x) for x in (b.get("mood_words") or []) if x])
            ).strip()
        )

    blob = " ".join([p for p in parts if p]).lower()

    hunter_hits = [
        "hunter", "dungeon", "gate", "raid", "monster", "awakener", "awakening",
        "system", "quest", "level", "rank", "skill", "mana", "artifact",
        "constellation", "star stream"
    ]
    cook_hits = ["cook", "cooking", "recipe", "kitchen", "chef", "taste", "flavor", "restaurant", "ingredients"]
    city_hits = ["city", "build", "building", "construction", "mayor", "village", "kingdom", "territory", "economy", "tax"]
    slice_hits = ["school", "class", "home", "date", "friends", "daily", "everyday", "workplace", "office", "neighbor"]

    def score(words: List[str]) -> int:
        return sum(1 for w in words if w in blob)

    sh = score(hunter_hits)
    sc = score(cook_hits)
    sb = score(city_hits)
    ss = score(slice_hits)

    best = max(sh, sc, sb, ss)
    if best <= 1:
        return "unknown"
    if best == sh:
        return "hunter"
    if best == sc:
        return "cooking"
    if best == sb:
        return "city_building"
    return "slice_of_life"

def _trope_lines_for_genre(genre: str) -> List[str]:
    g = (genre or "").strip().lower()
    if g != "hunter":
        return [
            f"Genre: {g}. Write like a top-tier manhwa recap writer in this genre.",
            "Rule of Cool: punchy action balanced with brief internal weight (fear, grit, resolve).",
        ]

    return [
        "Genre: hunter/system fantasy. Write like a top-tier manhwa recap writer in this genre.",
        "Aura Farming (ONLY when visually implied): presence hits BEFORE movement; describe pressure (air thickening, cracks, shadows).",
        "Energy Mechanics (Qi/Mana) (ONLY when implied): cultivation, dantian/meridians/breakthroughs; earned through pain; heart demons as obstacles.",
        "Rule of Cool action balanced with internal monologue that humanizes struggle against fate.",
        "Face-Slapping trope (ONLY if shown): arrogant elites get humiliated by hidden strength.",
        "System windows: when a UI/stat window appears in OCR, READ key lines aloud (selectively) so viewers understand power/scaling.",
    ]

# =============================================================================
# Vision OCR integration
# =============================================================================
def _build_vision_map(vision_manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    items = vision_manifest.get("items") or []
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            sf = it.get("scene_file")
            if sf:
                out[str(sf)] = it
    return out

def _ocr_to_lines(ocr_clean: str, max_lines: int = 10, max_chars_each: int = 90) -> List[str]:
    if not isinstance(ocr_clean, str) or not ocr_clean.strip():
        return []
    raw = re.split(r"[\n\r]+", ocr_clean.strip())
    lines: List[str] = []
    for r in raw:
        s = re.sub(r"\s+", " ", r).strip()
        if not s:
            continue
        if len(s) <= 1:
            continue
        if len(s) > max_chars_each:
            s = s[: max_chars_each - 1].rstrip() + "…"
        lines.append(s)
        if len(lines) >= max_lines:
            break
    return lines

def _scene_visual_weak(vision_item: Dict[str, Any]) -> bool:
    if not isinstance(vision_item, dict):
        return True
    if bool(vision_item.get("text_only")):
        return True
    tc = vision_item.get("text_coverage")
    try:
        if tc is not None and float(tc) >= 0.62:
            return True
    except Exception:
        pass
    return False

# =============================================================================
# Narrative cleaning (no camera language)
# =============================================================================
def strip_camera_language(text: str) -> str:
    if not text:
        return ""
    text = re.sub(
        r"^(a |the )?(wide shot|close-up|shot|angle|frame|pan|zoom)\s+(of\s+|reveals?\s+|shows?\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*the camera (pulls back|zooms|pans|focuses on|reveals|shows)[^.!?]*[,.]?\s*",
        ". ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"quick cuts (reveal|show)[^.!?]*[,.]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,.!?])\s*", r"\1 ", text)
    text = re.sub(r"\.+\s*\.+", ".", text)
    return text.strip()

def fix_ellipses(text: str) -> str:
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"\s+\.\.\.", "...", text)
    text = re.sub(r"\.\.\.\s+\.\.\.", "...", text)
    sentences = re.split(r"([.!?])", text)
    result: List[str] = []
    for sent in sentences:
        if sent.count("...") > 2:
            parts = sent.split("...")
            if len(parts) > 3:
                sent = parts[0] + "... " + "".join(parts[1:-1]) + "... " + parts[-1]
        result.append(sent)
    return "".join(result)

def clean_narration_post_llm(text: str) -> str:
    text = strip_camera_language(text)
    text = fix_ellipses(text)
    text = re.sub(r"\bwe (see|watch|witness)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def validate_paragraph_quality(paragraphs: List[str]) -> List[str]:
    issues: List[str] = []
    for i, para in enumerate(paragraphs or []):
        pl = para.lower()
        for cam_word in CAMERA_WORDS:
            if cam_word in pl:
                issues.append(f"Para {i+1}: camera word '{cam_word}'")
        for banned in BANNED_PHRASES:
            if banned in pl:
                issues.append(f"Para {i+1}: banned phrase '{banned}'")
        wc = _words(para)
        if wc < 10:
            issues.append(f"Para {i+1}: too short ({wc} words)")
        elif wc > 150:
            issues.append(f"Para {i+1}: too long ({wc} words)")
        if para.count("...") > 2:
            issues.append(f"Para {i+1}: too many ellipses")
    return issues

# =============================================================================
# TTS tag validation/repair
# =============================================================================
def _has_valid_leading_tts_tag(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    m = _LEADING_TAG_RE.match(s)
    if not m:
        return False
    tag = (m.group(1) or "").strip().lower()
    return tag in V3_VALID_TAGS

def _all_tts_have_valid_tags(paras: List[str]) -> bool:
    if not isinstance(paras, list) or not paras:
        return False
    return all(_has_valid_leading_tts_tag(str(p)) for p in paras)

def _sanitize_single_leading_tts_tag(para: str) -> str:
    """
    Enforce exactly ONE *leading* v3 mood tag from V3_VALID_TAGS.
    Preserve any other bracket tags in the body (e.g. [SHOUTING], [SIGH], [PAUSE]).
    If the leading tag is not a valid mood tag, treat it as a body tag and prepend a default mood.
    """
    if not isinstance(para, str):
        return ""

    raw = para.strip()
    if not raw:
        return "[serious]"

    leading_tag, rest = _split_leading_bracket_tag(raw)

    # If the paragraph starts with a non-mood bracket tag (e.g. [SHOUTING]),
    # keep it as part of the content and add a mood tag in front.
    if leading_tag is not None:
        mood_candidate = leading_tag.strip().lower()
        if mood_candidate in V3_VALID_TAGS:
            # Valid leading mood: keep it, but ensure there isn't another mood tag immediately duplicated
            # (We do NOT delete other tags elsewhere)
            return f"[{mood_candidate}] {rest}".strip()
        else:
            # Not a mood tag; keep it in content
            rest2 = f"[{leading_tag}] {rest}".strip() if rest else f"[{leading_tag}]"
            # Choose a default mood based on the content (including that tag)
            inferred = _sanitize_single_leading_tts_tag(f"[serious] {rest2}")
            # inferred will already return with a single leading mood
            return inferred

    # No leading tag at all: infer mood from content and prepend it
    clean_lower = raw.lower()

    if any(w in clean_lower for w in ["panic", "panicked", "run", "now", "seconds"]):
        tag = "panicked"
    elif any(w in clean_lower for w in ["urgent", "hurry", "immediately"]):
        tag = "urgent"
    elif any(w in clean_lower for w in ["whisper", "hush", "quiet"]):
        tag = "whisper"
    elif any(w in clean_lower for w in ["sad", "tear", "cry", "regret", "loss"]):
        tag = "sad"
    elif any(w in clean_lower for w in ["rage", "furious", "anger"]):
        tag = "angry"
    elif any(w in clean_lower for w in ["fight", "attack", "strike", "clash", "explode"]):
        tag = "excited"
    elif any(w in clean_lower for w in ["awe", "miracle", "reveal", "divine"]):
        tag = "awe"
    elif any(w in clean_lower for w in ["tense", "dread", "threat", "monster"]):
        tag = "tense"
    elif any(w in clean_lower for w in ["nervous", "shaking", "tremble"]):
        tag = "nervous"
    else:
        tag = "serious"

    return f"[{tag}] {raw}".strip()

def _ensure_tts_tags_from_beats(beats_chunk: List[Dict[str, Any]], tts_paragraphs: List[str]) -> List[str]:
    """
    Ensure each paragraph starts with a valid V3 mood tag.
    Preserve any non-mood bracket tags in the body.
    """
    out: List[str] = []
    for i, para in enumerate(tts_paragraphs or []):
        beat = beats_chunk[i] if i < len(beats_chunk or []) else {}
        mood_words = beat.get("mood_words") or []
        mood_blob = " ".join([str(x) for x in mood_words if x]).lower()
        emo = str(beat.get("emotional_turn") or "").lower()
        stake = str(beat.get("conflict_or_stakes") or "").lower()
        blob = f"{mood_blob} {emo} {stake}"

        if "whisper" in blob or "hush" in blob or "secret" in blob:
            tag = "whisper"
        elif any(k in blob for k in ["panic", "terror", "danger"]):
            tag = "panicked"
        elif any(k in blob for k in ["sad", "loss", "regret", "mourning"]):
            tag = "sad"
        elif any(k in blob for k in ["angry", "rage", "furious"]):
            tag = "angry"
        elif any(k in blob for k in ["awe", "reveal", "miracle", "divine"]):
            tag = "awe"
        elif any(k in blob for k in ["fight", "battle", "clash", "attack", "explosion", "blood"]):
            tag = "excited"
        elif any(k in blob for k in ["tense", "dread", "ominous", "threat"]):
            tag = "tense"
        elif any(k in blob for k in ["nervous", "uneasy", "shaking"]):
            tag = "nervous"
        else:
            tag = "serious"

        p = str(para or "").strip()
        if not p:
            out.append(f"[{tag}]")
            continue

        leading, rest = _split_leading_bracket_tag(p)
        if leading is not None and leading.strip().lower() in V3_VALID_TAGS:
            # Already has valid leading mood tag; keep it
            out.append(f"[{leading.strip().lower()}] {rest}".strip())
        else:
            # No valid mood leading tag: prepend mood tag, preserve original text (including leading non-mood tag)
            out.append(f"[{tag}] {p}".strip())

    return out

# =============================================================================
# SSML helper (optional)
# =============================================================================
_TAG_RE = re.compile(r"\[[^\[\]]+\]")

def _strip_v3_tags(text: str) -> str:
    return re.sub(_TAG_RE, "", text or "").strip()

def _insert_breaks_ssml(paragraph: str, break_s: float = 0.6, max_breaks: int = 4) -> str:
    txt = _strip_v3_tags(paragraph)
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return ""
    parts = re.split(r"([.!?])", txt)
    out: List[str] = []
    breaks = 0
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if chunk:
            out.append(chunk + punct)
            if punct and breaks < max_breaks:
                out.append(f' <break time="{break_s:.1f}s" /> ')
                breaks += 1
    ssml = "".join(out).strip()
    ssml = re.sub(r"\s+", " ", ssml).strip()
    return ssml

# =============================================================================
# OpenAI response helpers (structured outputs + fallback)
# =============================================================================
def _resp_to_text(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    choices = getattr(resp, "choices", None)
    if isinstance(choices, list) and choices:
        msg = getattr(resp.choices[0], "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""

def _resp_to_json_or_text(resp: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Returns (parsed_json, raw_text).
    Handles:
      - Responses API with structured outputs (resp.output[*].content[*].parsed)
      - Responses API with output_text
      - Chat Completions content
    """
    # 1) Responses API: parsed
    out = getattr(resp, "output", None)
    if isinstance(out, list) and out:
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list) and content:
                for c in content:
                    parsed = getattr(c, "parsed", None)
                    if isinstance(parsed, dict):
                        txt = getattr(c, "text", "") or ""
                        return parsed, str(txt)
                    # sometimes "parsed" may live on the item itself
                    parsed2 = getattr(item, "parsed", None)
                    if isinstance(parsed2, dict):
                        txt = getattr(c, "text", "") or ""
                        return parsed2, str(txt)

    # 2) Responses API: output_text
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return None, t.strip()

    # 3) Chat Completions
    choices = getattr(resp, "choices", None)
    if isinstance(choices, list) and choices:
        msg = getattr(resp.choices[0], "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        if isinstance(content, str) and content.strip():
            return None, content.strip()

    return None, ""

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    candidate = text[s : e + 1].strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None

def _responses_supports_response_format(client: OpenAI) -> bool:
    try:
        fn = client.responses.create
    except Exception:
        return False
    try:
        sig = inspect.signature(fn)
        return "response_format" in sig.parameters
    except Exception:
        return False

def _chat_supports_response_format(client: OpenAI) -> bool:
    try:
        fn = client.chat.completions.create
        sig = inspect.signature(fn)
        return "response_format" in sig.parameters
    except Exception:
        return False

def _call_chat_json(
    client: OpenAI,
    model: str,
    system: str,
    user_payload: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    temperature: float,
    max_output_tokens: int,
    usage_acc: Optional[UsageAccumulator] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    schema_hint = json.dumps(schema, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Return ONLY valid JSON. No markdown, no extra text.\n"
                "It MUST match this JSON Schema:\n"
                f"{schema_hint}\n\n"
                "INPUT_JSON:\n" + json.dumps(user_payload, ensure_ascii=False)
            ),
        },
    ]
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    if _chat_supports_response_format(client):
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    if usage_acc is not None:
        u = _openai_usage(resp)
        usage_acc.add(input_tokens=u["input"], output_tokens=u["output"], cached_tokens=u.get("cached", 0))
    raw = _resp_to_text(resp).strip()
    try:
        return (json.loads(raw) if raw else None), raw
    except Exception:
        return _extract_json_object(raw), raw

def _call_openai_json(
    client: OpenAI,
    model: str,
    system: str,
    user_payload: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    temperature: float,
    max_output_tokens: int,
    usage_acc: Optional[UsageAccumulator] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:

    if _responses_supports_response_format(client):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": "INPUT_JSON:\n" + json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "script_section", "schema": schema, "strict": True},
            },
        )

        if usage_acc is not None:
            u = _openai_usage(resp)
            usage_acc.add(input_tokens=u["input"], output_tokens=u["output"], cached_tokens=u.get("cached", 0))

        parsed, raw = _resp_to_json_or_text(resp)
        if isinstance(parsed, dict):
            return parsed, raw

        raw = (raw or "").strip()
        if not raw:
            return None, ""

        try:
            return json.loads(raw), raw
        except Exception:
            return _extract_json_object(raw), raw

    # fallback to chat
    return _call_chat_json(
        client=client,
        model=model,
        system=system,
        user_payload=user_payload,
        schema=schema,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        usage_acc=usage_acc,
    )

# =============================================================================
# Shots constraints helpers
# =============================================================================
def _allowed_files_by_beat_id(payload: Dict[str, Any]) -> Dict[int, List[str]]:
    m: Dict[int, List[str]] = {}
    for b in payload.get("beats") or []:
        bid = int(b.get("beat_id") or 0)
        allowed = b.get("allowed_scene_files") or b.get("scene_files") or []
        if isinstance(allowed, list):
            m[bid] = [str(x) for x in allowed if x]
        else:
            m[bid] = []
    return m

def _shots_scene_files_valid(obj: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[bool, str]:
    allowed_map = _allowed_files_by_beat_id(payload)
    shots = obj.get("shots") or []
    if not isinstance(shots, list):
        return False, "shots_not_list"

    for s in shots:
        if not isinstance(s, dict):
            continue
        bid = int(s.get("beat_id") or 0)
        allowed = allowed_map.get(bid, [])

        scene_files = s.get("scene_files") or []
        if not isinstance(scene_files, list):
            return False, f"shot_beat_id={bid}_scene_files_not_list"
        for x in [str(v) for v in scene_files if v]:
            if x not in allowed:
                return False, f"shot_beat_id={bid}_illegal_scene_file={x}"

        fb = s.get("fallback_scene_files") or []
        if not isinstance(fb, list):
            return False, f"shot_beat_id={bid}_fallback_scene_files_not_list"
        for x in [str(v) for v in fb if v]:
            if x not in allowed:
                return False, f"shot_beat_id={bid}_illegal_fallback_scene_file={x}"

    return True, "ok"

def _shots_count_matches_paras(obj: Dict[str, Any]) -> bool:
    paras = obj.get("script_paragraphs") or []
    shots = obj.get("shots") or []
    return isinstance(paras, list) and isinstance(shots, list) and len(paras) == len(shots)

def _ensure_paragraph_coverage(
    beats: List[Dict[str, Any]], paragraphs: List[str]
) -> List[str]:
    """Guarantee one narration paragraph per beat (no silently-dropped beats).

    The model occasionally returns fewer paragraphs than the section has beats;
    ``_build_default_shots_from_payload`` then uses ``n = min(beats, paragraphs)``
    and drops the trailing beats, leaving those groups SILENT in the video. Pad
    the paragraph list to one-per-beat using a deterministic fallback from each
    uncovered beat's ``what_happens`` (never truncates extra paragraphs).
    """
    paras = [str(p) for p in (paragraphs or [])]
    bl = beats or []
    for i in range(len(paras), len(bl)):
        b = bl[i] if isinstance(bl[i], dict) else {}
        wh = str(b.get("what_happens") or "").strip()
        paras.append(wh or "The scene continues.")
    return paras


def _build_default_shots_from_payload(payload: Dict[str, Any], script_paragraphs: List[str], *, wpm: int) -> List[Dict[str, Any]]:
    beats = payload.get("beats") or []
    shots: List[Dict[str, Any]] = []
    n = min(len(beats), len(script_paragraphs))

    for i in range(n):
        b = beats[i]
        if not isinstance(b, dict):
            continue

        gid = int(b.get("group_id") or 0)
        bid = int(b.get("beat_id") or (i + 1))

        allowed = b.get("allowed_scene_files") or b.get("scene_files") or []
        if not isinstance(allowed, list):
            allowed = []
        allowed = [str(x) for x in allowed if x]

        scene_files = allowed[:3]
        fallback = allowed[1:3] if len(allowed) >= 2 else allowed[:1]

        rh = b.get("rendering_hints") or {}
        if not isinstance(rh, dict):
            rh = {}
        avoid_text_zoom = bool(rh.get("avoid_text_zoom", True))
        preferred_focus = str(rh.get("preferred_focus") or "wide")
        camera_motion = str(rh.get("camera_motion") or "slow_pan")

        wc = _words(script_paragraphs[i])
        est_sec = (wc / max(80, int(wpm))) * 60.0
        est_sec = _clamp(est_sec, 3.0, 18.0)

        min_hold = _clamp(est_sec * 0.80, 2.5, 16.0)
        max_hold = _clamp(est_sec * 1.25, 3.5, 22.0)

        ocr_map = b.get("ocr_snippets_by_scene_file") or {}
        diag: List[str] = []
        if isinstance(ocr_map, dict):
            for sf in scene_files:
                lines = ocr_map.get(sf) or []
                if isinstance(lines, list):
                    for ln in lines[:2]:
                        s = str(ln).strip()
                        if s:
                            diag.append(s)
                if len(diag) >= 3:
                    break

        weak = b.get("weak_scene_files") or []
        weak_set = set([str(x) for x in weak if x])
        is_optional = any(sf in weak_set for sf in scene_files)

        shots.append(
            {
                "beat_id": bid,
                "group_id": gid,
                "scene_files": scene_files,
                "fallback_scene_files": fallback,
                "duration_s": float(est_sec),
                "min_hold_s": float(min_hold),
                "max_hold_s": float(max_hold),
                "camera": camera_motion,
                "focus": preferred_focus,
                "avoid_text_zoom": avoid_text_zoom,
                "use_dialogue": bool(diag),
                "dialogue_snippets": diag[:3],
                "is_optional": bool(is_optional),
                "notes": "auto_shots_from_beats",
            }
        )
    return shots

def _normalize_shots(shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in shots or []:
        if not isinstance(s, dict):
            continue
        diag = s.get("dialogue_snippets") or []
        if not isinstance(diag, list):
            diag = []
        out.append(
            {
                "beat_id": int(s.get("beat_id") or 0),
                "group_id": int(s.get("group_id") or 0),
                "scene_files": [str(x) for x in (s.get("scene_files") or []) if x],
                "fallback_scene_files": [str(x) for x in (s.get("fallback_scene_files") or []) if x],
                "duration_s": float(s.get("duration_s") or 0.0),
                "min_hold_s": float(s.get("min_hold_s") or 0.0),
                "max_hold_s": float(s.get("max_hold_s") or 0.0),
                "camera": str(s.get("camera") or ""),
                "focus": str(s.get("focus") or ""),
                "avoid_text_zoom": bool(s.get("avoid_text_zoom", True)),
                "use_dialogue": bool(s.get("use_dialogue", False)),
                "dialogue_snippets": [str(x) for x in diag if x],
                "is_optional": bool(s.get("is_optional", False)),
                "notes": str(s.get("notes") or ""),
            }
        )
    return out

# =============================================================================
# Section chunking / targets
# =============================================================================
def _chunk_beats(beats: List[Dict[str, Any]], beats_per_section: int) -> List[List[Dict[str, Any]]]:
    beats_per_section = max(1, int(beats_per_section or 6))
    return [beats[i : i + beats_per_section] for i in range(0, len(beats), beats_per_section)]

def _estimate_words(min_minutes: int, max_minutes: int, wpm: int) -> int:
    min_words = max(350, int(min_minutes * wpm))
    max_words = max(min_words + 150, int(max_minutes * wpm))
    return random.randint(min_words, max_words)

# =============================================================================
# Semantic SFX extraction for a section (from shots dialogue/ocr snippets)
# =============================================================================
def _build_sfx_cues_for_section(section_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    shots = section_obj.get("shots") or []
    if not isinstance(shots, list):
        return cues

    for shot in shots:
        if not isinstance(shot, dict):
            continue
        bid = int(shot.get("beat_id") or 0)
        gid = int(shot.get("group_id") or 0)

        snippets = shot.get("dialogue_snippets") or []
        if not isinstance(snippets, list):
            snippets = []

        joined = " ".join(str(x) for x in snippets if x).strip()
        if not joined:
            continue

        for pattern, sfx_label in ONOMATOPOEIA_MAP:
            matches = pattern.findall(joined)
            if not matches:
                continue

            for match in matches:
                token = str(match).strip()
                if not token:
                    continue

                high_intensity = {"DRAGON_VOICE", "BEAST_ROAR", "EXPLOSION", "IMPACT_HEAVY"}
                intensity = 0.8 if sfx_label in high_intensity else 0.6
                if len(token) >= 6 and sfx_label in {"MONSTER_GROWL", "DRAGON_VOICE", "BEAST_ROAR"}:
                    intensity = 0.9

                cues.append(
                    {
                        "beat_id": bid,
                        "group_id": gid,
                        "sfx_label": sfx_label,
                        "token": token,
                        "placement": "sync",
                        "intensity": float(_clamp(intensity, 0.0, 1.0)),
                    }
                )

    # dedupe
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in cues:
        key = (c["beat_id"], c["group_id"], c["sfx_label"], str(c["token"]).upper())
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

# =============================================================================
# Validation schema (section)
# =============================================================================
def _validate_section_json(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    # sfx_cues is optional because you build it post-hoc
    required = [
        "section_index",
        "word_target",
        "section_genre_mode",
        "section_summary",
        "script_paragraphs",
        "tts_paragraphs_v3",
        "pronunciation_lexemes",
        "shots",
        "cliffhanger_line",
    ]
    for k in required:
        if k not in obj:
            return False

    if not isinstance(obj["script_paragraphs"], list):
        return False
    if not isinstance(obj["tts_paragraphs_v3"], list):
        return False
    if not isinstance(obj["shots"], list):
        return False
    if not isinstance(obj["pronunciation_lexemes"], list):
        return False

    # Do NOT require tags here; you repair tags later
    # Do NOT require sfx_cues here; you fill it later

    # Basic length alignment: if mismatch, we'll fix shots later; allow it through
    return True

# =============================================================================
# SYSTEM PROMPT TEMPLATE (your enhanced “no camera language” version)
# =============================================================================
ENHANCED_SYSTEM_TEMPLATE = """You are an elite Manhwa recap scriptwriter for YouTube, specializing in fast-paced, visually-driven storytelling.

=== CORE MISSION ===
Turn beats + OCR + scene data into a recap that feels like a movie trailer, not a book report.

=== NARRATIVE VOICE (CRITICAL) ===
- Pure story narration - NO camera language
- NEVER use: "the camera", "wide shot", "close-up", "shot reveals", "we see", "we watch"
- NEVER use generic camera-speak "we/our/us" (e.g. "we see", "our view"). You MAY
  use persona terms like "the protagonist", "the MC", "the antagonist".
- Write like: "At 55 years old, the warrior gasps for breath..." (direct story)
- NOT like: "A close-up reveals the warrior's exhausted face..." (cinematography)

=== NO OCR-ECHO / NO REPETITION (HARD RULE) ===
- NEVER quote UI/interface text. NEVER narrate view counts, comment counts, likes,
  episode/chapter numbers, site names, or watermarks — these are app chrome, not story.
- PARAPHRASE dialogue into narration in your own words; do not transcribe it.
- DIALOGUE & INTERNAL MONOLOGUE: rephrase into narration in your own words —
  do NOT transcribe. Do not reuse any run of 5+ consecutive words from the
  panel's OCR/dialogue. (e.g. OCR "IT IS NOT OVER YET, I WILL DESTROY YOU" ->
  "he refuses to fall, swearing to tear his enemy apart", NOT a copy of the line.)
- KEEP AS-IS (do NOT paraphrase): short, punchy DIRECT lines (e.g. "Come here.",
  "You're dead.", "It's him!") and proper nouns/titles (character names,
  "Sky Corporation", "7th Generation Nano Machine"). Name or quote these
  directly — only longer dialogue/monologue must be reworded.
- NEVER repeat the same phrase, sentence, or fact across paragraphs — if a beat
  echoes an earlier one, advance it or rephrase with new emphasis.
- Treat OCR text as a NOISY hint, not ground truth — OCR misreads are common
  (e.g. 'I' for '1'); never build narration around garbled or counter-like fragments.

=== TRUTHFULNESS (NON-NEGOTIABLE) ===
- ONLY use info from provided beats and OCR snippets
- If unclear, narrate ambiguity briefly instead of inventing
- Never invent names/relationships/plot points not present

=== VISUAL ANCHORING (MANDATORY EVERY PARAGRAPH) ===
Each paragraph MUST contain at least ONE anchor:
1) A concrete visual detail (blood, torn sleeve, glowing eyes), OR
2) A concrete action/event description grounded in the beat.

Formula: [Anchor] -> [cause/reveal] -> [emotional weight/consequence]

=== CONTINUITY ===
A STORY SO FAR summary may be provided. Maintain consistency with it; do NOT
re-introduce already-established characters/facts as if new; carry emotional
threads forward.

=== FLASHBACK / TIME-SHIFT ===
If a beat or its OCR indicates a flashback, memory, dream, or time-shift (cues
like 'YEARS AGO', 'THE PAST', 'FLASHBACK', 'EARLIER', a visibly younger
character, or a scene-break into backstory), narrate it as past framing
('Years earlier…', 'In a memory…') and signal the return to the present.
Do not narrate a flashback in the same flat present-tense as the main timeline.

=== MANHWA-RECAP VOCABULARY ===
Use this manhwa-recap vocabulary NATURALLY where the visuals/beats support it
(e.g. call the lead 'the MC' or 'our protagonist's rival the antagonist'; when
presence overwhelms before motion, 'aura farming'; when an arrogant elite is
humbled, a 'face-slap'). Do NOT force terms that don't fit the scene.
{JARGON_LEXICON}

=== PACING ===
- Each paragraph: 2–5 sentences depending on beat intensity
- Action beats: short punchy sentences
- Emotion/reveal: slightly longer, let it breathe
- Keep concrete, avoid abstract phrasing

=== SYSTEM/STAT WINDOWS ===
- If OCR suggests UI/stat window, read key lines aloud (selectively) so viewers understand power/abilities

=== TTS FORMATTING (ElevenLabs v3) ===
- Each tts_paragraphs_v3 item MUST start with exactly ONE mood tag:
  [calm] [tense] [urgent] [excited] [awe] [sad] [whisper] [angry] [nervous] [panicked] [serious]
- Write for the ear: strong punctuation, clean clauses
- Natural dialogue integration: use context-aware verbs (mutters, shouts, asks)

=== SHOTS (STRICT) ===
- shots[*].scene_files and fallback_scene_files MUST be chosen ONLY from each beat's allowed list
- Never invent filenames

=== LENGTH CONTROL ===
- Target about {WORD_TARGET} words total for script_paragraphs (±{TOL_PCT}%)

{GENRE_FLAVOR}

IMPORTANT:
- Output JSON must include 'sfx_cues' as an array (can be empty)
- NO camera language anywhere in narration
Return ONLY valid JSON matching the schema. No extra text.
"""

# =============================================================================
# Main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", required=True, help="Path to manifest.beats.json")
    ap.add_argument("--vision", default="", help="Optional: manifest.vision.json")
    ap.add_argument("--out", required=True, help="Output manifest.script.json")
    ap.add_argument("--model", default="gpt-4.1-mini")

    ap.add_argument("--min-minutes", type=int, default=9)
    ap.add_argument("--max-minutes", type=int, default=11)
    ap.add_argument("--wpm", type=int, default=135)
    ap.add_argument("--beats-per-section", type=int, default=6)

    ap.add_argument("--duration-mode", choices=["soft", "none"], default="soft")
    ap.add_argument("--words-per-beat", type=int, default=110)

    ap.add_argument("--force-genre", default="", help="Force section_genre_mode (e.g. hunter)")
    ap.add_argument("--max-output-tokens", type=int, default=2600)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--word-tolerance", type=float, default=0.10)

    ap.add_argument("--emit-ssml", action="store_true", help="If set, emit tts_paragraphs_ssml")

    ap.add_argument("--debug-dir", default="", help="If set, dump raw LLM outputs per section here")

    args = ap.parse_args()

    beats_m = load_json(args.beats)
    beats = beats_m.get("beats") or []
    if not isinstance(beats, list) or not beats:
        raise SystemExit("No beats found in beats manifest")

    # sort beats by group_id
    beats.sort(key=lambda x: int(x.get("group_id") or 0))

    # attach stable beat_id if missing
    for idx, b in enumerate(beats):
        if isinstance(b, dict) and not b.get("beat_id"):
            b["beat_id"] = idx + 1

    vision_by_file: Dict[str, Dict[str, Any]] = {}
    if args.vision and os.path.exists(args.vision):
        try:
            vision_by_file = _build_vision_map(load_json(args.vision))
        except Exception:
            vision_by_file = {}

    sections = _chunk_beats(beats, args.beats_per_section)

    # resume
    existing_sections: Dict[int, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.out):
        try:
            existing = load_json(args.out)
            for sec in (existing.get("sections") or []):
                idx = int(sec.get("section_index") or -1)
                if idx >= 0:
                    existing_sections[idx] = sec
        except Exception:
            existing_sections = {}

    # total word target
    if args.duration_mode == "none":
        total_word_target = max(350, int(len(beats) * int(args.words_per_beat)))
    else:
        total_word_target = _estimate_words(args.min_minutes, args.max_minutes, args.wpm)

    # per section allocation proportional to beats count
    total_beats = len(beats)
    per_section_targets: List[int] = []
    acc = 0
    for chunk in sections:
        frac = (len(chunk) / total_beats) if total_beats else 0.0
        tgt = max(220, int(round(total_word_target * frac)))
        per_section_targets.append(tgt)
        acc += tgt
    if per_section_targets:
        per_section_targets[-1] = max(220, per_section_targets[-1] + (total_word_target - acc))

    # schema (section output)
    section_schema = {
        "type": "object",
        "properties": {
            "section_index": {"type": "integer"},
            "word_target": {"type": "integer"},
            "section_genre_mode": {"type": "string"},
            "section_summary": {"type": "string"},
            "script_paragraphs": {"type": "array", "items": {"type": "string"}},
            "tts_paragraphs_v3": {"type": "array", "items": {"type": "string"}},
            "pronunciation_lexemes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"grapheme": {"type": "string"}, "alias": {"type": "string"}},
                    "required": ["grapheme", "alias"],
                    "additionalProperties": False,
                },
            },
            "shots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "beat_id": {"type": "integer"},
                        "group_id": {"type": "integer"},
                        "scene_files": {"type": "array", "items": {"type": "string"}},
                        "fallback_scene_files": {"type": "array", "items": {"type": "string"}},
                        "duration_s": {"type": "number"},
                        "min_hold_s": {"type": "number"},
                        "max_hold_s": {"type": "number"},
                        "camera": {"type": "string"},
                        "focus": {"type": "string"},
                        "avoid_text_zoom": {"type": "boolean"},
                        "use_dialogue": {"type": "boolean"},
                        "dialogue_snippets": {"type": "array", "items": {"type": "string"}},
                        "is_optional": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "beat_id",
                        "group_id",
                        "scene_files",
                        "fallback_scene_files",
                        "duration_s",
                        "min_hold_s",
                        "max_hold_s",
                        "camera",
                        "focus",
                        "avoid_text_zoom",
                        "use_dialogue",
                        "dialogue_snippets",
                        "is_optional",
                        "notes",
                    ],
                    "additionalProperties": False,
                },
            },
            "cliffhanger_line": {"type": "string"},
            "sfx_cues": {"type": "array", "items": {"type": "object"}},
        },
        "required": [
            "section_index",
            "word_target",
            "section_genre_mode",
            "section_summary",
            "script_paragraphs",
            "tts_paragraphs_v3",
            "pronunciation_lexemes",
            "shots",
            "cliffhanger_line",
            "sfx_cues",
        ],
        "additionalProperties": False,
    }

    client = OpenAI()
    usage = UsageAccumulator(args.model)
    out_sections: List[Dict[str, Any]] = []
    regenerated = 0
    parse_errors = 0

    for section_index, chunk in enumerate(sections):
        if section_index in existing_sections and not existing_sections[section_index].get("error"):
            out_sections.append(existing_sections[section_index])
            continue

        if section_index in existing_sections and existing_sections[section_index].get("error"):
            regenerated += 1

        word_target = per_section_targets[section_index] if section_index < len(per_section_targets) else 900
        genre_mode = (args.force_genre or "").strip() or _infer_genre_mode(chunk)
        trope_lines = _trope_lines_for_genre(genre_mode)
        genre_flavor = "\n".join([f"- {t}" for t in trope_lines])

        system = (
            ENHANCED_SYSTEM_TEMPLATE
            .replace("{JARGON_LEXICON}", _manhwa_jargon_block())
            .replace("{WORD_TARGET}", str(word_target))
            .replace("{TOL_PCT}", str(int(args.word_tolerance * 100)))
            .replace("{GENRE_FLAVOR}", "=== GENRE FLAVOR (ONLY WHEN SUPPORTED BY VISUALS/OCR) ===\n" + genre_flavor)
        )

        # R3 — cross-section continuity: build a STORY SO FAR synopsis from the
        # sections already generated in THIS run (resumed sections included via
        # out_sections), and thread it into the user payload for this section.
        story_so_far = build_story_so_far(out_sections, max_chars=600)

        # Build payload beats with OCR previews
        payload_beats: List[Dict[str, Any]] = []
        for b in chunk:
            gid = int(b.get("group_id") or 0)
            beat_id = int(b.get("beat_id") or 0)

            scene_files = b.get("scene_files") or []
            if not isinstance(scene_files, list):
                scene_files = []

            ocr_by_scene: Dict[str, List[str]] = {}
            weak_scenes: List[str] = []
            for sf in scene_files:
                it = vision_by_file.get(str(sf)) or {}
                lines = _ocr_to_lines(str(it.get("ocr_clean") or ""), max_lines=10, max_chars_each=90)
                if lines:
                    ocr_by_scene[str(sf)] = lines
                if it and _scene_visual_weak(it):
                    weak_scenes.append(str(sf))

            ocr_preview: List[str] = []
            for sf in scene_files[:3]:
                if sf in ocr_by_scene:
                    ocr_preview.append(f"{sf}: {_safe_join_lines(ocr_by_scene[sf], max_items=5)}")

            payload_beats.append(
                {
                    "beat_id": beat_id,
                    "group_id": gid,
                    "scene_files": scene_files,
                    "allowed_scene_files": scene_files,
                    "beat_title": b.get("beat_title") or "",
                    "what_happens": b.get("what_happens") or "",
                    "emotional_turn": b.get("emotional_turn") or "",
                    "conflict_or_stakes": b.get("conflict_or_stakes") or "",
                    "reveals_or_info": b.get("reveals_or_info") or "",
                    "hook": b.get("hook") or "",
                    "mood_words": b.get("mood_words") or [],
                    "rendering_hints": b.get("rendering_hints")
                    or {"avoid_text_zoom": True, "preferred_focus": "wide", "camera_motion": "slow_pan"},
                    "ocr_snippets_by_scene_file": ocr_by_scene,
                    "ocr_preview": ocr_preview,
                    "weak_scene_files": weak_scenes,
                }
            )

        payload = {
            "section_index": section_index,
            "word_target": word_target,
            "section_genre_mode_hint": genre_mode,
            "beats": payload_beats,
        }
        # R3 — only include the continuity block when there's prior context, so
        # the very first section's payload is unchanged.
        if story_so_far:
            payload["STORY SO FAR"] = story_so_far

        obj: Optional[Dict[str, Any]] = None
        raw: str = ""

        for _attempt in range(args.retries + 1):
            o1, r1 = _call_openai_json(
                client=client,
                model=args.model,
                system=system,
                user_payload=payload,
                schema=section_schema,
                temperature=0.5,
                max_output_tokens=args.max_output_tokens,
                usage_acc=usage,
            )
            raw = r1

            if args.debug_dir:
                os.makedirs(args.debug_dir, exist_ok=True)
                dump_json(
                    os.path.join(args.debug_dir, f"section_{section_index}_attempt_{_attempt}.json"),
                    {"raw": raw, "parsed": o1 if isinstance(o1, dict) else None},
                )

            if isinstance(o1, dict):
                o1.setdefault("sfx_cues", [])
                o1.setdefault("pronunciation_lexemes", [])
                o1.setdefault("shots", [])
                o1.setdefault("script_paragraphs", [])
                o1.setdefault("tts_paragraphs_v3", [])
                o1.setdefault("cliffhanger_line", "Something shifts…")
                o1.setdefault("section_summary", "")
                o1.setdefault("section_genre_mode", genre_mode)

            if _validate_section_json(o1):
                # post-clean narration + enforce tag rules again
                script_paras = [clean_narration_post_llm(str(p)) for p in (o1.get("script_paragraphs") or [])]
                # COVERAGE: never let the model silently drop beats — one
                # paragraph per beat. The shots-repair below (paras != shots count)
                # then rebuilds shots covering every beat/group.
                script_paras = _ensure_paragraph_coverage(chunk, script_paras)
                o1["script_paragraphs"] = script_paras

                tts_v3 = list(o1.get("tts_paragraphs_v3") or [])
                # clean content part but preserve leading tag
                cleaned_tts: List[str] = []
                for p in tts_v3:
                    p = str(p or "").strip()
                    if not p:
                        cleaned_tts.append("[serious] ")
                        continue
                    leading, rest = _split_leading_bracket_tag(p)
                    if leading is not None:
                        # Preserve rest (including any inline tags); only clean narration text
                        rest = clean_narration_post_llm(rest)
                        cleaned_tts.append(f"[{leading}] {rest}".strip())
                    else:
                        cleaned_tts.append(clean_narration_post_llm(p))

                # COVERAGE: keep one TTS paragraph per narration paragraph so the
                # backfilled beats also get voiced (and don't desync the clips).
                for j in range(len(cleaned_tts), len(script_paras)):
                    cleaned_tts.append("[serious] " + script_paras[j])
                o1["tts_paragraphs_v3"] = cleaned_tts

                # tag repair
                if not _all_tts_have_valid_tags(o1.get("tts_paragraphs_v3") or []):
                    o1["tts_paragraphs_v3"] = _ensure_tts_tags_from_beats(payload_beats, o1.get("tts_paragraphs_v3") or [])
                o1["tts_paragraphs_v3"] = [_sanitize_single_leading_tts_tag(p) for p in (o1.get("tts_paragraphs_v3") or [])]

                # shots repair if needed
                ok_files, _ = _shots_scene_files_valid(o1, payload)
                ok_count = _shots_count_matches_paras(o1)
                if (not ok_files) or (not ok_count):
                    o1["shots"] = _build_default_shots_from_payload(payload, o1.get("script_paragraphs") or [], wpm=int(args.wpm))
                    o1["shots"] = _normalize_shots(o1["shots"])

                # word target check (soft) + quality issues (soft)
                wc = _count_words(o1.get("script_paragraphs") or [])
                ok_words = _within_tolerance(wc, word_target, args.word_tolerance)
                quality_issues = validate_paragraph_quality(o1.get("script_paragraphs") or [])

                if ok_words and not quality_issues and _validate_section_json(o1):
                    obj = o1
                    break

                # If quality issues exist, still accept but continue attempts if available
                if _attempt == args.retries:
                    obj = o1
                    break

        if obj is None:
            parse_errors += 1
            obj = {
                "section_index": section_index,
                "word_target": word_target,
                "section_genre_mode": genre_mode,
                "section_summary": "Unable to generate section due to JSON parse/validation failures.",
                "script_paragraphs": [],
                "tts_paragraphs_v3": [],
                "pronunciation_lexemes": [],
                "shots": [],
                "cliffhanger_line": "Something shifts…",
                "sfx_cues": [],
                "error": "parse_failed_after_retries",
                "raw_excerpt": (raw or "")[:1200],
            }

        obj["section_genre_mode"] = genre_mode
        obj["shots"] = _normalize_shots(obj.get("shots") or [])

        # build semantic SFX cues from dialogue snippets
        obj["sfx_cues"] = _build_sfx_cues_for_section(obj)

        # Add SSML helper (optional)
        if args.emit_ssml:
            obj["tts_paragraphs_ssml"] = [
                _insert_breaks_ssml(str(p), break_s=0.6, max_breaks=4)
                for p in (obj.get("tts_paragraphs_v3") or [])
        ]

        # Add deterministic segment_id + tts_meta + audio_events aligned to paragraphs
        # segment_id is based on group_id and paragraph index within this section
        shots = obj.get("shots") or []
        paras = obj.get("script_paragraphs") or []
        tts_v3 = obj.get("tts_paragraphs_v3") or []

        tts_meta: List[Dict[str, Any]] = []
        audio_events: List[List[Dict[str, Any]]] = []

        n = min(len(shots), len(paras), len(tts_v3))
        for i in range(n):
            gid = int((shots[i] or {}).get("group_id") or 0)
            segment_id = f"g{gid:04d}_p{i:02d}"
            shots[i]["segment_id"] = segment_id

            # leading tag => voice settings
            m = _LEADING_TAG_RE.match(str(tts_v3[i] or ""))
            tag = (m.group(1).lower() if m else "serious")
            if tag not in V3_VALID_TAGS:
                tag = "serious"

            tts_meta.append(
                {"segment_id": segment_id, "voice_settings": _build_voice_settings_for_tag(tag)}
            )

            # per-paragraph semantic SFX events: filter cues matching this beat/group
            bid = int((shots[i] or {}).get("beat_id") or 0)
            gid2 = int((shots[i] or {}).get("group_id") or 0)
            evs: List[Dict[str, Any]] = []
            for cue in obj.get("sfx_cues") or []:
                if not isinstance(cue, dict):
                    continue
                if int(cue.get("beat_id") or 0) == bid and int(cue.get("group_id") or 0) == gid2:
                    evs.append(
                        {
                            "type": "sfx",
                            "sfx_label": str(cue.get("sfx_label") or ""),
                            "token": str(cue.get("token") or ""),
                            "placement": str(cue.get("placement") or "sync"),
                            "intensity": float(cue.get("intensity") or 0.7),
                        }
                    )
            audio_events.append(evs)

        obj["shots"] = shots
        obj["tts_meta"] = tts_meta
        obj["audio_events"] = audio_events

        out_sections.append(obj)

    out_obj = {
        "schema_version": "script_manifest_v3",
        "source_beats_manifest": os.path.abspath(args.beats),
        "source_vision_manifest": os.path.abspath(args.vision) if args.vision else "",
        "model": args.model,
        "minutes_range": {"min": args.min_minutes, "max": args.max_minutes},
        "duration_mode": args.duration_mode,
        "words_per_beat": int(args.words_per_beat),
        "wpm": args.wpm,
        "beats_per_section": args.beats_per_section,
        "force_genre": (args.force_genre or "").strip(),
        "word_target_total": total_word_target,
        "section_word_targets": per_section_targets,
        "stats": {
            "parse_errors": parse_errors,
            "regenerated": regenerated,
            "usage": {
                "calls": usage.calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "est_cost_usd": round(usage.cost(), 4),
            },
        },
        "sections": out_sections,
    }

    dump_json(args.out, out_obj)
    print(f"[ok] wrote={args.out} sections={len(out_sections)} parse_errors={parse_errors} regenerated={regenerated}")
    print(usage.summary())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
