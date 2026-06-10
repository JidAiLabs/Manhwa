#!/usr/bin/env python3
"""
script_expander.py (enhanced: VISUAL-ANCHORED + PACED + TROPE PACK + TTS V3)

Key changes vs your current version:
- Replaces system prompt with a visually-anchored, trailer-like recap style.
- Explicitly allows 2–5 sentences per paragraph (beat-dependent), not rigid 2–4.
- Adds guidance to READ system/stat windows aloud (selectively) when OCR indicates it.
- Re-integrates your earlier trope guidance ("Aura Farming", "Qi/Mana mechanics", etc.)
  under strict truthfulness: only when visually implied by panels/OCR.
- Keeps: strict filename legality, 1 paragraph per beat, emotion-tag requirement for v3.

Install:
  pip install -U openai

Run:
  python3 tools/script_expander.py --beats ... --vision ... --out ... --model gpt-4.1-mini --resume
"""

import argparse
import inspect
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# -----------------------------
# Small utils
# -----------------------------
def _words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", str(s or "")))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _count_words(paras: List[str]) -> int:
    txt = " ".join([p for p in (paras or []) if isinstance(p, str)])
    return len(re.findall(r"\b\w+\b", txt))


def _within_tolerance(actual: int, target: int, tol: float) -> bool:
    if target <= 0:
        return True
    return abs(actual - target) <= int(target * tol)


def _safe_join_lines(lines: List[str], max_items: int = 8) -> str:
    out: List[str] = []
    for s in (lines or [])[:max_items]:
        t = re.sub(r"\s+", " ", str(s or "")).strip()
        if t:
            out.append(t)
    return " | ".join(out)


# -----------------------------
# TTS drama tags
# -----------------------------
_ALLOWED_TTS_TAGS = ["calm", "tense", "urgent", "excited", "awe", "sad", "whisper", "angry"]
_LEADING_TTS_TAG_RE = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*")


def _has_valid_leading_tts_tag(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    m = _LEADING_TTS_TAG_RE.match(s)
    if not m:
        return False
    tag = (m.group(1) or "").strip().lower()
    return tag in _ALLOWED_TTS_TAGS


def _all_tts_have_valid_tags(paras: List[str]) -> bool:
    if not isinstance(paras, list) or not paras:
        return False
    return all(_has_valid_leading_tts_tag(str(p)) for p in paras)


def _sanitize_single_leading_tts_tag(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return ""
    tags = re.findall(r"^\s*(\[[^\[\]]+\]\s*)+", s)
    if tags:
        s2 = re.sub(r"^\s*(\[[^\[\]]+\]\s*)+", "", s).strip()
        lead = tags[0].lower()
        chosen = None
        for t in _ALLOWED_TTS_TAGS:
            if f"[{t}]" in lead:
                chosen = t
                break
        if not chosen:
            chosen = "calm"
        return f"[{chosen}] {s2}".strip()
    return f"[calm] {s}".strip()


def _ensure_tts_tags_from_beats(
    beats_chunk: List[Dict[str, Any]],
    tts_paragraphs: List[str],
) -> List[str]:
    out: List[str] = []
    n = min(len(beats_chunk or []), len(tts_paragraphs or []))
    for i in range(n):
        t = str(tts_paragraphs[i] or "").strip()
        if _has_valid_leading_tts_tag(t):
            out.append(t)
            continue

        b = beats_chunk[i] if i < len(beats_chunk) else {}
        mood = " ".join([str(x) for x in (b.get("mood_words") or []) if x]).lower()
        emo = str(b.get("emotional_turn") or "").lower()
        stake = str(b.get("conflict_or_stakes") or "").lower()
        blob = f"{mood} {emo} {stake}".strip()

        if any(k in blob for k in ["whisper", "quiet", "hush", "secret"]):
            tag = "whisper"
        elif any(k in blob for k in ["grief", "loss", "sad", "mourning", "tear", "despair"]):
            tag = "sad"
        elif any(k in blob for k in ["rage", "furious", "anger", "wrath"]):
            tag = "angry"
        elif any(k in blob for k in ["awe", "wonder", "reveal", "revelation", "miracle", "divine"]):
            tag = "awe"
        elif any(k in blob for k in ["panic", "run", "urgent", "immediate", "now", "seconds"]):
            tag = "urgent"
        elif any(k in blob for k in ["fight", "battle", "clash", "attack", "charge", "explosion", "blood"]):
            tag = "excited"
        elif any(k in blob for k in ["tense", "dread", "ominous", "stalk", "creep", "threat"]):
            tag = "tense"
        else:
            tag = "calm"

        out.append(f"[{tag}] {t}".strip())

    for j in range(n, len(tts_paragraphs or [])):
        t = str(tts_paragraphs[j] or "").strip()
        if _has_valid_leading_tts_tag(t):
            out.append(t)
        else:
            out.append(f"[calm] {t}".strip())
    return out


# -----------------------------
# IO helpers
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Text + JSON helpers
# -----------------------------
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


def _resp_to_text(resp: Any) -> str:
    if resp is None:
        return ""

    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    out = getattr(resp, "output", None)
    if isinstance(out, list) and out:
        chunks: List[str] = []
        try:
            for item in out:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for c in content:
                    c_type = getattr(c, "type", None)
                    c_text = getattr(c, "text", None)
                    if c_type == "output_text" and isinstance(c_text, str):
                        chunks.append(c_text)
            joined = "\n".join([x for x in chunks if x and x.strip()]).strip()
            if joined:
                return joined
        except Exception:
            pass

    choices = getattr(resp, "choices", None)
    if isinstance(choices, list) and choices:
        msg = getattr(resp, "choices")[0].message
        content = getattr(msg, "content", None) if msg is not None else None
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def _contains_banned_phrases(paras: List[str]) -> List[str]:
    text = "\n".join([p for p in (paras or []) if isinstance(p, str)]).lower()
    banned = [
        r"\bwe\b",
        r"\bour\b",
        r"\bus\b",
        r"\bcharacters\b",
        r"\bcharacter\b",
        r"\bwe see\b",
        r"\bnext we see\b",
        r"\bthe scene opens\b",
    ]
    hits: List[str] = []
    for pat in banned:
        if re.search(pat, text):
            hits.append(pat)
    return hits


# -----------------------------
# OpenAI call helpers (compat)
# -----------------------------
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
) -> Tuple[Optional[Dict[str, Any]], str]:
    schema_hint = json.dumps(schema, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Return ONLY valid JSON. No markdown, no extra text.\n"
                "It MUST match this JSON Schema (structure, not literal):\n"
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
                "json_schema": {
                    "name": "script_section",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        raw = _resp_to_text(resp).strip()
        try:
            return (json.loads(raw) if raw else None), raw
        except Exception:
            return _extract_json_object(raw), raw

    return _call_chat_json(
        client=client,
        model=model,
        system=system,
        user_payload=user_payload,
        schema=schema,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


# -----------------------------
# Vision OCR integration
# -----------------------------
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


# -----------------------------
# Beats + shots constraints (anti-hallucination)
# -----------------------------
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


# -----------------------------
# Deterministic fallback shots builder
# -----------------------------
def _build_default_shots_from_payload(
    payload: Dict[str, Any],
    script_paragraphs: List[str],
    *,
    wpm: int,
) -> List[Dict[str, Any]]:
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
    for s in shots:
        if not isinstance(s, dict):
            continue
        beat_id = int(s.get("beat_id") or 0)
        group_id = int(s.get("group_id") or 0)
        scene_files = s.get("scene_files") or []
        if not isinstance(scene_files, list):
            scene_files = []
        fallback = s.get("fallback_scene_files") or []
        if not isinstance(fallback, list):
            fallback = []
        diag = s.get("dialogue_snippets") or []
        if not isinstance(diag, list):
            diag = []

        out.append(
            {
                "beat_id": beat_id,
                "group_id": group_id,
                "scene_files": [str(x) for x in scene_files if x],
                "fallback_scene_files": [str(x) for x in fallback if x],
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


# -----------------------------
# Script planning helpers
# -----------------------------
def _chunk_beats(beats: List[Dict[str, Any]], beats_per_section: int) -> List[List[Dict[str, Any]]]:
    if beats_per_section <= 0:
        beats_per_section = 6
    chunks: List[List[Dict[str, Any]]] = []
    for i in range(0, len(beats), beats_per_section):
        chunks.append(beats[i : i + beats_per_section])
    return chunks


def _estimate_words(min_minutes: int, max_minutes: int, wpm: int) -> int:
    min_words = max(350, int(min_minutes * wpm))
    max_words = max(min_words + 150, int(max_minutes * wpm))
    return random.randint(min_words, max_words)


# -----------------------------
# Genre detection (light heuristic)
# -----------------------------
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


# -----------------------------
# TTS helpers (SSML)
# -----------------------------
_TAG_RE = re.compile(r"\[[^\[\]]+\]")


def _strip_v3_tags(text: str) -> str:
    return re.sub(_TAG_RE, "", text or "").strip()


def _insert_breaks_ssml(paragraph: str, break_s: float = 0.6, max_breaks: int = 4) -> str:
    txt = _strip_v3_tags(paragraph)
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return ""

    parts = re.split(r"([.!?])", txt)
    out = []
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


def _validate_section_json(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    required = [
        "section_index",
        "word_target",
        "section_summary",
        "script_paragraphs",
        "tts_paragraphs_v3",
        "shots",
        "cliffhanger_line",
        "section_genre_mode",
        "pronunciation_lexemes",
    ]
    for k in required:
        if k not in obj:
            return False
    if not isinstance(obj["script_paragraphs"], list):
        return False
    if not isinstance(obj["tts_paragraphs_v3"], list):
        return False
    if len(obj["tts_paragraphs_v3"]) != len(obj["script_paragraphs"]):
        return False
    if not isinstance(obj["shots"], list):
        return False
    if not isinstance(obj["pronunciation_lexemes"], list):
        return False
    if not _all_tts_have_valid_tags(obj["tts_paragraphs_v3"]):
        return False
    return True


# -----------------------------
# Main
# -----------------------------
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
    args = ap.parse_args()

    if args.duration_mode == "soft":
        if args.min_minutes <= 0 or args.max_minutes <= 0 or args.max_minutes < args.min_minutes:
            raise SystemExit("Invalid minutes range. Expect: 0 < min-minutes <= max-minutes")
    else:
        if args.words_per_beat <= 0:
            raise SystemExit("Invalid --words-per-beat. Expect > 0")

    beats_m = load_json(args.beats)
    beats = beats_m.get("beats") or []
    if not isinstance(beats, list) or not beats:
        raise SystemExit("No beats found in beats manifest")

    vision_by_file: Dict[str, Dict[str, Any]] = {}
    if args.vision and os.path.exists(args.vision):
        try:
            vision_m = load_json(args.vision)
            vision_by_file = _build_vision_map(vision_m)
        except Exception:
            vision_by_file = {}

    beats.sort(key=lambda x: int(x.get("group_id") or 0))

    for idx, b in enumerate(beats):
        if isinstance(b, dict):
            b["_beat_id"] = idx + 1

    sections = _chunk_beats(beats, args.beats_per_section)

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

    if args.duration_mode == "none":
        total_word_target = max(350, int(len(beats) * int(args.words_per_beat)))
    else:
        total_word_target = _estimate_words(args.min_minutes, args.max_minutes, args.wpm)

    total_beats = len(beats)
    per_section_targets: List[int] = []
    acc = 0
    for chunk in sections:
        frac = (len(chunk) / total_beats) if total_beats else 0
        tgt = max(220, int(round(total_word_target * frac)))
        per_section_targets.append(tgt)
        acc += tgt
    if per_section_targets:
        delta = total_word_target - acc
        per_section_targets[-1] = max(220, per_section_targets[-1] + delta)

    client = OpenAI()

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
        ],
        "additionalProperties": False,
    }

    system_template = (
        "You are an elite Manhwa recap scriptwriter for YouTube, specializing in fast-paced, visually-driven storytelling.\n\n"
        "=== CORE MISSION ===\n"
        "Turn beats + OCR + scene data into a recap that feels like a movie trailer, not a book report.\n\n"
        "=== TRUTHFULNESS (NON-NEGOTIABLE) ===\n"
        "- ONLY use info from the provided beats and OCR snippets.\n"
        "- If unclear, narrate ambiguity briefly instead of inventing.\n"
        "- Never invent names/relationships/plot points not present.\n"
        "- If OCR shows dialogue/UI text, you MAY quote short fragments (<= 6 words).\n\n"
        "=== NARRATIVE VOICE ===\n"
        "- Third-person cinematic narrator.\n"
        "- Never use: we/our/us.\n"
        "- Never say: character(s).\n"
        "- Prefer roles: the warrior, the boy, the injured girl, the hunter, the survivor.\n\n"
        "=== VISUAL ANCHORING (MANDATORY EVERY PARAGRAPH) ===\n"
        "Each paragraph MUST contain at least ONE anchor derived from the visuals/OCR:\n"
        "1) An OCR quote (1–6 words) in quotes, OR\n"
        "2) A concrete visual detail (blood, torn sleeve, cave mouth, sign, glowing eyes), OR\n"
        "3) A camera/composition cue (close-up, wide shot, quick cuts).\n"
        "Formula: [Visual anchor] -> [cause/reveal] -> [emotional weight/consequence].\n\n"
        "=== PACING ===\n"
        "- Each paragraph may be 2–5 sentences depending on the beat.\n"
        "- Action: short punchy sentences.\n"
        "- Emotion/reveal: slightly longer, let it breathe.\n"
        "- Avoid abstract phrasing; keep it concrete.\n\n"
        "=== SYSTEM/STAT WINDOWS ===\n"
        "- If OCR suggests a UI/stat window, read key lines aloud (selectively) so viewers understand power/abilities.\n\n"
        "=== TTS FORMATTING (Eleven v3) ===\n"
        "- Each tts_paragraphs_v3 item MUST start with exactly ONE mood tag:\n"
        "  [calm] [tense] [urgent] [excited] [awe] [sad] [whisper] [angry]\n"
        "- You may use a few inline tags sparingly (e.g. [short pause], [whispers]) but do NOT overuse.\n"
        "- Write for the ear: strong punctuation, clean clauses.\n\n"
        "=== SHOTS (STRICT) ===\n"
        "- shots[*].scene_files and fallback_scene_files MUST be chosen ONLY from each beat's allowed list.\n"
        "- Never invent filenames.\n\n"
        "=== LENGTH CONTROL ===\n"
        "- Target about {WORD_TARGET} words total for script_paragraphs (±{TOL_PCT}%).\n\n"
        "Return ONLY valid JSON matching the schema. No extra text.\n"
    )

    out_sections: List[Dict[str, Any]] = []
    regenerated = 0
    parse_errors = 0

    for section_index, chunk in enumerate(sections):
        if section_index in existing_sections and not existing_sections[section_index].get("error"):
            sec = existing_sections[section_index]
            if isinstance(sec.get("shots"), list) and len(sec.get("shots") or []) > 0:
                out_sections.append(sec)
                continue

        if section_index in existing_sections and existing_sections[section_index].get("error"):
            regenerated += 1

        word_target = per_section_targets[section_index] if section_index < len(per_section_targets) else 900
        genre_mode = (args.force_genre or "").strip() or _infer_genre_mode(chunk)

        trope_lines = _trope_lines_for_genre(genre_mode)
        system = (
            system_template.replace("{WORD_TARGET}", str(word_target))
            .replace("{TOL_PCT}", str(int(args.word_tolerance * 100)))
            + "\n=== GENRE FLAVOR (ONLY WHEN SUPPORTED BY VISUALS/OCR) ===\n"
            + "\n".join([f"- {t}" for t in trope_lines])
            + "\n"
        )

        payload_beats: List[Dict[str, Any]] = []
        for b in chunk:
            gid = int(b.get("group_id") or 0)
            beat_id = int(b.get("_beat_id") or 0)

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

            # compact preview for the model to encourage quoting small fragments
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
            )
            raw = r1

            if _validate_section_json(o1):
                paras = o1.get("script_paragraphs") or []
                wc = _count_words(paras)
                banned_hits = _contains_banned_phrases(paras)

                ok_words = _within_tolerance(wc, word_target, args.word_tolerance)
                ok_files, _ = _shots_scene_files_valid(o1, payload)
                ok_count = _shots_count_matches_paras(o1)
                ok_tts_tags = _all_tts_have_valid_tags(o1.get("tts_paragraphs_v3") or [])

                if ok_words and not banned_hits and ok_files and ok_count and ok_tts_tags:
                    obj = o1
                    break

                if not ok_tts_tags:
                    o1["tts_paragraphs_v3"] = _ensure_tts_tags_from_beats(
                        beats_chunk=payload_beats,
                        tts_paragraphs=list(o1.get("tts_paragraphs_v3") or []),
                    )
                    o1["tts_paragraphs_v3"] = [
                        _sanitize_single_leading_tts_tag(p) for p in (o1.get("tts_paragraphs_v3") or [])
                    ]
                    ok_tts_tags2 = _all_tts_have_valid_tags(o1.get("tts_paragraphs_v3") or [])
                    if ok_tts_tags2 and ok_words and not banned_hits and ok_files and ok_count:
                        obj = o1
                        break

                if (not ok_files) or (not ok_count):
                    o1["shots"] = _build_default_shots_from_payload(
                        payload=payload,
                        script_paragraphs=list(o1.get("script_paragraphs") or []),
                        wpm=int(args.wpm),
                    )
                    o1["shots"] = _normalize_shots(o1["shots"])
                    ok_files2, _ = _shots_scene_files_valid(o1, payload)
                    ok_count2 = _shots_count_matches_paras(o1)
                    ok_tts_tags3 = _all_tts_have_valid_tags(o1.get("tts_paragraphs_v3") or [])
                    if ok_files2 and ok_count2 and ok_words and not banned_hits and ok_tts_tags3:
                        obj = o1
                        break

            repair_payload2 = {
                "section_index": section_index,
                "word_target": word_target,
                "last_output": (raw or "")[:6000],
                "instruction": (
                    "Re-output ONLY valid JSON matching the schema. No extra text.\n"
                    "Requirements:\n"
                    "- Third-person narrator; no we/our/us\n"
                    "- Never say character(s)\n"
                    "- Every paragraph must include at least one visual anchor (OCR quote or concrete visible detail or camera cue)\n"
                    "- tts_paragraphs_v3: exactly ONE leading mood tag per paragraph\n"
                ),
            }
            o2, r2 = _call_openai_json(
                client=client,
                model=args.model,
                system="You are a strict JSON formatter and editor. Output valid JSON only.",
                user_payload=repair_payload2,
                schema=section_schema,
                temperature=0.0,
                max_output_tokens=args.max_output_tokens,
            )
            raw = r2
            if isinstance(o2, dict):
                if not _all_tts_have_valid_tags(o2.get("tts_paragraphs_v3") or []):
                    o2["tts_paragraphs_v3"] = _ensure_tts_tags_from_beats(
                        beats_chunk=payload_beats,
                        tts_paragraphs=list(o2.get("tts_paragraphs_v3") or []),
                    )
                ok_files4, _ = _shots_scene_files_valid(o2, payload)
                ok_count4 = _shots_count_matches_paras(o2)
                if (not ok_files4) or (not ok_count4):
                    o2["shots"] = _build_default_shots_from_payload(
                        payload=payload,
                        script_paragraphs=list(o2.get("script_paragraphs") or []),
                        wpm=int(args.wpm),
                    )
                    o2["shots"] = _normalize_shots(o2["shots"])
                if _validate_section_json(o2):
                    obj = o2
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
                "error": "parse_failed_after_retries",
                "raw_excerpt": (raw or "")[:1200],
            }

        if isinstance(obj.get("shots"), list):
            obj["shots"] = _normalize_shots(obj["shots"])

        obj["section_genre_mode"] = genre_mode

        ok_files_final, _ = _shots_scene_files_valid(obj, payload)
        ok_count_final = _shots_count_matches_paras(obj)
        if (not ok_files_final) or (not ok_count_final):
            obj["shots"] = _build_default_shots_from_payload(
                payload=payload,
                script_paragraphs=list(obj.get("script_paragraphs") or []),
                wpm=int(args.wpm),
            )
            obj["shots"] = _normalize_shots(obj["shots"])

        if not _all_tts_have_valid_tags(obj.get("tts_paragraphs_v3") or []):
            obj["tts_paragraphs_v3"] = _ensure_tts_tags_from_beats(
                beats_chunk=payload_beats,
                tts_paragraphs=list(obj.get("tts_paragraphs_v3") or []),
            )
        obj["tts_paragraphs_v3"] = [
            _sanitize_single_leading_tts_tag(p) for p in (obj.get("tts_paragraphs_v3") or [])
        ]

        tts_v3 = obj.get("tts_paragraphs_v3") or []
        if not isinstance(tts_v3, list):
            tts_v3 = []
        obj["tts_paragraphs_ssml"] = [
            _insert_breaks_ssml(str(p), break_s=0.6, max_breaks=4) for p in tts_v3
        ]

        out_sections.append(obj)

    out_obj = {
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
        "stats": {"parse_errors": parse_errors, "regenerated": regenerated},
        "sections": out_sections,
    }

    dump_json(args.out, out_obj)
    print(
        f"[ok] wrote={args.out} sections={len(out_sections)} "
        f"parse_errors={parse_errors} regenerated={regenerated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
