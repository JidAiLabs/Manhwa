#!/usr/bin/env python3
"""
gemini_narrative_pass.py (429-safe)

Fixes:
- SDK-compatible Part.from_text / Part.from_bytes calls
- Uses resp.parsed when available, else robust JSON extraction
- Repair pass on parse failure
- Resume mode supported (keeps good beats, regenerates missing/errored)
- 429 RESOURCE_EXHAUSTED backoff with jitter
- Throttle between groups (min-sleep + jitter)
- Cap images per group (select lowest text_coverage panels first)
- Incremental checkpoint writes (checkpoint-every)

Requires:
  pip install -U google-genai
Auth:
  gcloud auth application-default login
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import ClientError

# Shared keep/redundant + bubble/intensity normalization (sibling tool module).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_selection import normalize_scene_selection  # noqa: E402
from usage_cost import UsageAccumulator  # noqa: E402
from narration_safe_rules import SAFE_NARRATION_RULES  # noqa: E402
from recap_style import (  # noqa: E402
    RECAP_STYLE_RULES,
    dedupe_consecutive_panel_lines,
    is_shot_description,
    neutralize_identity_reveal_leaks,
    repair_spoken_fragments,
)

# --- meta-garbage narration guard --------------------------------------------
# Ch20 g0014: a panel's OCR was a long run of underscores (a garbage SFX scan).
# The narration model, fed that corruption, returned VALID JSON whose narration
# was META-COMMENTARY about parsing/JSON — and it got voiced. The beat's `error`
# was None (the JSON parsed), so nothing caught it. This detector flags a
# "narration" that is clearly the model talking about its own input/JSON rather
# than telling the story.
_META_STRONG_SIGNALS = (
    r"malformed\s+json",
    r"json\s+fragment",
    r"scene_files",
    r"object\s+schema",
    r"valid\s+json",
    r"underscore\s+characters?",
    r"\bjson\b",
    r"\bschema\b",
    r"\bunderscores?\b",
)
_META_WEAK_SIGNALS = (
    r"data\s+structure",
    r"reconstruct\s+the",
    r"the\s+input\s+was",
    r"parsing\s+the",
    r"the\s+task\s+is\s+to",
    r"integrity\s+of\s+the",
)
_META_STRONG_RE = re.compile("|".join(_META_STRONG_SIGNALS), re.IGNORECASE)
_META_WEAK_RE = re.compile("|".join(_META_WEAK_SIGNALS), re.IGNORECASE)

# --- repeated-phrase detector ------------------------------------------------
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "as", "by", "from", "is", "was", "are", "were", "be",
    "been", "being", "it", "its", "he", "she", "his", "her", "they",
    "their", "this", "that", "his", "her", "its", "our", "your", "my",
    "into", "through", "across", "over", "under", "up", "down", "out",
    "not", "no", "nor", "so", "yet", "both", "each", "than", "too",
    "very", "just", "even", "still", "also", "then", "there", "here",
})


def repeated_phrases(
    lines: List[str],
    n: int = 3,
    min_count: int = 2,
) -> List[Tuple[str, int]]:
    """Return (phrase, count) for size-n n-grams of non-stopwords occurring
    >= min_count times across all narration lines, sorted by count desc.

    Useful for QA flagging of heavy atmospheric repetition in a chapter's
    narration. Does NOT gate the pipeline — call site decides what to do.
    """
    from collections import Counter
    import re as _re

    counts: Counter = Counter()
    for line in lines:
        words = [w for w in _re.findall(r"[a-z]+", line.lower())
                 if w not in _STOPWORDS]
        for i in range(len(words) - n + 1):
            counts[" ".join(words[i:i + n])] += 1

    return sorted(
        [(phrase, count) for phrase, count in counts.items()
         if count >= min_count],
        key=lambda x: x[1],
        reverse=True,
    )


def _is_meta_garbage(text: str) -> bool:
    """True when the 'narration' is clearly the model talking about JSON/parsing/
    its own input rather than the story. Requires at least one STRONG signal
    (json / schema / scene_files / underscore) to avoid false positives on real
    narration that merely mentions a 'structure' or 'task'."""
    if not text:
        return False
    return bool(_META_STRONG_RE.search(text))


def _clean_fallback_narration(beat_title: str, what_happens: str) -> str:
    """Last-resort narration when the model keeps returning meta-garbage: use
    what_happens if it is NOT itself meta-garbage, else the beat_title if clean,
    else a neutral one-line bridge. NEVER returns meta-garbage."""
    for cand in (what_happens, beat_title):
        c = (cand or "").strip()
        if c and not _is_meta_garbage(c):
            return c
    return "The scene shifts."


# Convey dialogue in the NARRATOR'S clean words. The on-screen bubble text is raw
# OCR — ALL-CAPS, frequently mis-read, truncated mid-word, or a pure sound effect —
# so copying it verbatim reads as garbled shouting ("KILL HIM!", "SERVES YOU RIGHT!
# Mon", "...SINCE OUR COMRA"). Paraphrase what is said into the recap voice instead.
_DIALOGUE_RULE = (
    "DIALOGUE: PARAPHRASE the bulk of what a character SAYS or THINKS into the "
    "NARRATOR'S OWN clean words — but DO quote occasionally for impact. A few "
    "SHORT (<=6 words), COMPLETE, punchy real lines per chapter — a threat, a "
    "taunt, a key line, a name — land harder than any paraphrase (e.g. he mutters "
    "'I can't move.', she spits 'Damn you.', he sneers that it 'serves them "
    "right'). Quote where a real line hits hard; paraphrase everything else. Write "
    "EVERY quote in clean sentence case attributed to who says it. NEVER copy raw "
    "on-screen / OCR text verbatim (it is ALL-CAPS, mis-read, or truncated mid-"
    "word, so it reads as garbled shouting); NEVER quote a sound effect or "
    "onomatopoeia (huh, ugh, keuk, ack, grr, a raw scream); NEVER quote an "
    "incomplete, trailing-off fragment such as 'Ancestor...?' — finish the thought "
    "in your own words instead. NEVER voice publication chrome — ads, credits, "
    "'subscribe/follow/join our Discord', watermarks, scanlator or site names."
)


def _usage_from_resp(resp: Any) -> Dict[str, int]:
    """Extract exact (input, output, cached) token counts from a Gemini response."""
    um = getattr(resp, "usage_metadata", None)
    return {
        "input": int(getattr(um, "prompt_token_count", 0) or 0),
        "output": int(getattr(um, "candidates_token_count", 0) or 0),
        "cached": int(getattr(um, "cached_content_token_count", 0) or 0),
    }


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_groups(groups_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(groups_manifest.get("shots"), list):
        return groups_manifest["shots"]
    if isinstance(groups_manifest.get("groups"), list):
        return groups_manifest["groups"]
    return []


def _build_vision_map(vision_manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    items = vision_manifest.get("items") or []
    return {it.get("scene_file"): it for it in items if it.get("scene_file")}


def _load_cast_list(cast_path: str) -> List[Dict[str, Any]]:
    """Load manifest.cast.json -> its `cast` array (list of members). Empty list
    on a missing/unreadable/malformed file (never raises). Reused by the cast
    block AND the per-beat token resolver so the cast is read once."""
    if not cast_path or not os.path.exists(cast_path):
        return []
    try:
        with open(cast_path, "r", encoding="utf-8") as f:
            cast = json.load(f)
    except Exception:
        return []
    members = cast.get("cast") if isinstance(cast, dict) else None
    return members if isinstance(members, list) else []


def _build_cast_block(cast_path: str) -> str:
    """Render manifest.cast.json into a prompt block the narration uses to name
    characters consistently. Empty string when no cast file is given.

    The role is rendered as `(role)` NOT `[role]`: a bracketed `[protagonist]`
    reads like a canonical reference token and the model copies it verbatim into
    the narration, where the TTS then voices the literal '[protagonist]'."""
    cast = _load_cast_list(cast_path)
    if not cast:
        return ""
    lines = [
        "CHAPTER CAST — name these consistently; match each figure by appearance. "
        "Refer to each character by their NAME or a natural pronoun inline — NEVER "
        "output a bracketed token like [protagonist] or [antagonist]; never invent "
        "a generic descriptor (e.g. 'an injured man') for a character who is in "
        "this cast:"
    ]
    for c in cast:
        name = c.get("canonical_name") or c.get("id") or "?"
        role = c.get("role") or ""
        desc = (c.get("visual_description") or "").strip()
        aliases = ", ".join(c.get("aliases") or [])
        tag = f" (aka {aliases})" if aliases else ""
        lines.append(f"  - {name} ({role}){tag}: {desc}")
    lines.append("")  # trailing blank so it reads cleanly before the next section
    return "\n".join(lines) + "\n"


# Words that mark an alias as a generic descriptor / epithet rather than a usable
# proper name (we never substitute a bracketed token with "this bastard").
_NON_NAME_WORDS = frozenset({
    "this", "that", "the", "a", "an", "bastard", "guy", "man", "woman", "boy",
    "girl", "kid", "old", "young", "person", "figure", "one", "thing", "stranger",
    "people", "lady", "gentleman", "mister", "sir", "fellow", "dude",
})


def _proper_name_alias(aliases: List[str]) -> Optional[str]:
    """Pick the first alias that looks like a usable PROPER NAME: capitalized,
    1-4 tokens, and free of generic/role words ('bastard', 'man', 'this', ...).
    Returns None if none qualifies (caller falls back to canonical_name)."""
    for a in aliases or []:
        a = str(a or "").strip()
        if not a or not a[0].isupper():
            continue
        toks = a.split()
        if not (1 <= len(toks) <= 4):
            continue
        if any(t.strip(".,'").lower() in _NON_NAME_WORDS for t in toks):
            continue
        return a
    return None


def _cast_member_reference(member: Dict[str, Any]) -> str:
    """The text a bracketed token for this cast member should become: a proper-
    name alias when one exists, else the canonical_name (recap-native, e.g.
    'the antagonist')."""
    return _proper_name_alias(member.get("aliases") or []) or \
        str(member.get("canonical_name") or member.get("id") or "").strip()


def _resolve_cast_tokens(text: str, cast: List[Dict[str, Any]]) -> str:
    """Safety net: rewrite any bracketed `[token]` the model copied into the
    narration into readable prose, so the TTS never voices a literal token.

    (a) A token matching a cast member's role / id / canonical_name (case-
        insensitive, '_' and ' ' interchangeable) becomes that member's
        reference (proper-name alias, else canonical_name).
    (b) Any REMAINING bracket token is stripped to its inner words (e.g.
        '[someone] runs' -> 'someone runs'). We NEVER blank a line: an unknown
        token degrades to readable inner text, not emptiness.
    The possessive form `[protagonist]'s` is preserved (only the bracket part is
    rewritten, the trailing 's stays)."""
    if not text or "[" not in text:
        return text

    def _norm(s: str) -> str:
        return re.sub(r"[\s_]+", " ", str(s or "").strip().lower())

    lookup: Dict[str, str] = {}
    for m in cast or []:
        ref = _cast_member_reference(m)
        if not ref:
            continue
        for key in (m.get("role"), m.get("id"), m.get("canonical_name")):
            k = _norm(key)
            if k:
                lookup.setdefault(k, ref)

    def _sub(match: "re.Match[str]") -> str:
        inner = match.group(1).strip()
        hit = lookup.get(_norm(inner))
        if hit is not None:
            return hit
        # unknown token: keep the inner words (readable), drop the brackets.
        return inner

    return re.sub(r"\[([^\[\]]*)\]", _sub, text)


def _build_story_block(story_path: str) -> str:
    """Render manifest.story.json (the chapter spine: logline + premise + ordered
    arc) into a prompt block, so every beat is written as part of the WHOLE story
    instead of an isolated panel caption. Empty string when no spine is given."""
    if not story_path or not os.path.exists(story_path):
        return ""
    try:
        with open(story_path, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        return ""
    logline = str(s.get("logline") or "").strip()
    premise = str(s.get("premise") or "").strip()
    arc = s.get("arc") if isinstance(s.get("arc"), list) else []
    if not (logline or premise or arc):
        return ""
    lines = ["CHAPTER STORY SPINE — the whole arc this recap tells. Write EVERY "
             "beat as part of THIS story (place it in the arc, pay off setups, "
             "call back to earlier beats) so the recap reads as ONE connected "
             "story, not isolated panel descriptions. Use the spine for "
             "through-line + context ONLY — never state anything not visible in "
             "the current beat's panels:"]
    if logline:
        lines.append(f"  LOGLINE: {logline}")
    if premise:
        lines.append(f"  PREMISE: {premise}")
    if arc:
        lines.append("  ARC (beats in order):")
        for a in arc:
            gid = a.get("group_id")
            lab = str(a.get("arc_label") or "").strip()
            seg = str(a.get("segment") or "present")
            tag = "" if seg == "present" else f" [{seg}]"
            lines.append(f"    beat {gid}: {lab}{tag}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _pack_group_payload(
    group: Dict[str, Any],
    vision_items_by_file: Dict[str, Dict[str, Any]],
    understand_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    scene_files = group.get("scene_files") or []
    scenes: List[Dict[str, Any]] = []
    understand_by_file = understand_by_file or {}

    for sf in scene_files:
        it = vision_items_by_file.get(sf) or {}
        understood = understand_by_file.get(sf) or {}
        v = it.get("vision") or {}
        labels = [x.get("desc") for x in (v.get("labels") or []) if x.get("desc")]
        objects = [x.get("name") for x in (v.get("objects") or []) if x.get("name")]

        scenes.append(
            {
                "scene_file": sf,
                "ocr_clean": (it.get("ocr_clean") or "")[:900],
                "text_only": bool(it.get("text_only")),
                "text_coverage": it.get("text_coverage"),
                "keywords": it.get("keywords") if isinstance(it.get("keywords"), list) else [],
                "labels": labels[:15],
                "objects": objects[:15],
                # Full paid understanding, including panels omitted from the
                # image attachment cap. This is the narration's factual source;
                # vision OCR/labels are supporting signals, not a substitute.
                "description": str(understood.get("description") or "")[:500],
                "action": str(understood.get("action") or "")[:240],
                "setting": str(understood.get("setting") or "")[:160],
                "dialogue": str(understood.get("dialogue") or "")[:320],
                "panel_kind": str(understood.get("panel_kind")
                                  or it.get("panel_kind") or ""),
                "intensity": str(understood.get("intensity") or ""),
                "subjects": (
                    understood.get("subjects")
                    if isinstance(understood.get("subjects"), list)
                    else (it.get("subjects")
                          if isinstance(it.get("subjects"), list) else [])),
            }
        )

    return {
        "group_id": int(group.get("shot_id") or group.get("group_id") or 0),
        "scene_files": scene_files,
        "scenes_signals": scenes,
        # this beat's place in the arc + its PACE (intensity drives line length:
        # punchy for intense/explosive, fuller for calm/tense). story_group emits
        # arc_label/segment/intensity; the old code looked for a non-existent
        # 'why_merge' and dropped the lot.
        "arc_label": group.get("arc_label"),
        "segment": group.get("segment") or "present",
        "intensity": group.get("intensity") or "tense",
    }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    candidate = text[s : e + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _part_text(s: str) -> types.Part:
    try:
        return types.Part.from_text(text=s)
    except TypeError:
        return types.Part.from_text(s)


def _part_image_jpeg(b: bytes) -> types.Part:
    try:
        return types.Part.from_bytes(bytes=b, mime_type="image/jpeg")
    except TypeError:
        return types.Part.from_bytes(data=b, mime_type="image/jpeg")


def _schema_to_json_schema(s: Any) -> Any:
    """Gemini response_schema (UPPERCASE type enums) -> standard JSON Schema
    for Ollama's structured-output `format` parameter."""
    if isinstance(s, dict):
        out = {}
        for k, v in s.items():
            if k == "propertyOrdering":
                continue
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            else:
                out[k] = _schema_to_json_schema(v)
        return out
    if isinstance(s, list):
        return [_schema_to_json_schema(x) for x in s]
    return s


def _call_model(
    *,
    client: Optional[genai.Client],
    model: str,
    system_instruction: str,
    user_payload: Dict[str, Any],
    image_paths: List[str],
    response_schema: Dict[str, Any],
    max_output_tokens: int,
    temperature: float,
    backend: str = "vertex",
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, int]]:
    if backend == "ollama":
        # local open model (Gemma 4 et al.) via the Ollama server — same
        # contract: system + INPUT_JSON + panel images -> schema'd JSON
        import ollama
        msg: Dict[str, Any] = {
            "role": "user",
            "content": "INPUT_JSON:\n" + json.dumps(user_payload, ensure_ascii=False),
        }
        images = [p for p in image_paths if p and os.path.exists(p)]
        if images:
            msg["images"] = images
        from ollama_compat import chat as _ollama_chat
        resp = _ollama_chat(
            model=model,
            messages=[{"role": "system", "content": system_instruction}, msg],
            format=_schema_to_json_schema(response_schema),
            think=False,  # Gemma 4 thinks by default and burns the budget
            options={"temperature": temperature,
                     "num_predict": max_output_tokens,
                     # 16k thrashed gemma's SWA cache (full prompt re-processing
                     # every call -> ~32min wedge). Beats prompts measure ~1-7.5k
                     # tokens, so 8k fits with headroom and matches the understand
                     # stage's working cap. Env-tunable.
                     "num_ctx": int(os.environ.get("STUDIO_BEATS_NUM_CTX", "8192"))},
        )
        raw = (resp.get("message") or {}).get("content") or ""
        usage = {"input": int(resp.get("prompt_eval_count") or 0),
                 "output": int(resp.get("eval_count") or 0), "cached": 0}
        try:
            return json.loads(raw), raw, usage
        except Exception:
            return _extract_json_object(raw), raw, usage

    parts: List[types.Part] = []
    parts.append(_part_text("INPUT_JSON:\n" + json.dumps(user_payload, ensure_ascii=False)))

    for p in image_paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            parts.append(_part_image_jpeg(f.read()))

    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
        ),
    )

    usage = _usage_from_resp(resp)
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, dict):
        return parsed, (resp.text or ""), usage

    raw = resp.text or ""
    try:
        return json.loads(raw), raw, usage
    except Exception:
        return _extract_json_object(raw), raw, usage


# Wall-clock bound on the 429 retry loop (only the vertex/gemini backend can 429;
# ollama — the production default — never hits this). Generous enough for a
# transient quota dip, bounded so it can't stall a lane forever.
_MODEL_429_DEADLINE_SEC = int(os.environ.get("STUDIO_MODEL_429_DEADLINE_SEC", "900") or "900")


# Transient local-LLM (ollama) disconnects: an ollama restart/crash/overload drops
# the connection mid-request (httpx.RemoteProtocolError / ConnectError). These are
# recoverable — retry with backoff so a blip (or a reboot's ollama reload) doesn't
# fail the whole chapter. The hard-watchdog TimeoutError is deliberately NOT here:
# a genuine stall should fail-soft and move the lane on, not retry-loop.
_TRANSIENT_LLM_EXC: tuple = (ConnectionError,)
try:
    import httpx as _httpx
    _TRANSIENT_LLM_EXC = _TRANSIENT_LLM_EXC + (_httpx.TransportError,)
except Exception:
    pass


def _call_model_with_backoff(
    *,
    client: Optional[genai.Client],
    model: str,
    system_instruction: str,
    user_payload: Dict[str, Any],
    image_paths: List[str],
    response_schema: Dict[str, Any],
    max_output_tokens: int,
    temperature: float,
    backoff_max: float,
    backend: str = "vertex",
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, int]]:
    attempt = 0
    # BOUND the 429 retry: a quota cliff during a 300-chapter run must NOT loop
    # forever — after the deadline, raise so the stage fails and the lane moves on.
    deadline = time.time() + _MODEL_429_DEADLINE_SEC
    while True:
        try:
            return _call_model(
                client=client,
                model=model,
                system_instruction=system_instruction,
                user_payload=user_payload,
                image_paths=image_paths,
                response_schema=response_schema,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                backend=backend,
            )
        except ClientError as e:
            msg = str(e)
            if ("429" not in msg) and ("RESOURCE_EXHAUSTED" not in msg):
                raise
            if time.time() >= deadline:
                print(f"[error] 429 RESOURCE_EXHAUSTED persisted > "
                      f"{_MODEL_429_DEADLINE_SEC}s — giving up (stage fails).")
                raise
            sleep_s = min(backoff_max, (2 ** min(attempt, 6)) + random.random() * 0.8)
            print(f"[warn] 429 RESOURCE_EXHAUSTED. sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)
            attempt += 1
        except _TRANSIENT_LLM_EXC as e:
            # ollama dropped the connection mid-request (restart/crash/overload) —
            # transient. Retry with backoff, bounded by the same deadline so a
            # persistently-down server eventually fails the stage and the lane moves on.
            if time.time() >= deadline:
                print(f"[error] local-LLM transient error persisted > "
                      f"{_MODEL_429_DEADLINE_SEC}s — giving up (stage fails): {type(e).__name__}")
                raise
            sleep_s = min(backoff_max, (2 ** min(attempt, 6)) + random.random() * 0.8)
            print(f"[warn] local-LLM disconnect ({type(e).__name__}: {str(e)[:80]}). "
                  f"sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)
            attempt += 1


def _generate_beat_for_group(
    *,
    client: Any,
    model: str,
    system_instruction: str,
    payload: Dict[str, Any],
    image_paths: List[str],
    beat_schema: Any,
    gid: Any,
    retries: int,
    max_output_tokens: int,
    backoff_max: float,
    backend: str = "vertex",
    usage: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Run the model accept loop for one group. Returns a content-bearing beat
    dict (group_id + scene_files stamped) or None if every attempt failed to
    parse. Guards against two silent corruptions:
      - EMPTY narration: retry, last-attempt fall back to what_happens.
      - META-GARBAGE narration (the Ch20 g0014 bug — the model narrates about
        JSON/parsing/underscores instead of the story): retry the FULL
        generation; on the last attempt fall back to a CLEAN line
        (what_happens if not itself garbage, else a neutral bridge). The
        meta-garbage line is NEVER kept as the narration."""

    def _acc(u: Dict[str, int]) -> None:
        if usage is not None:
            usage.add(input_tokens=u["input"], output_tokens=u["output"],
                      cached_tokens=u.get("cached", 0))

    scene_files = payload.get("scene_files", [])
    raw_text = ""

    for _attempt in range(retries + 1):
        obj, raw, u = _call_model_with_backoff(
            client=client,
            model=model,
            system_instruction=system_instruction,
            user_payload=payload,
            image_paths=image_paths,
            response_schema=beat_schema,
            max_output_tokens=max_output_tokens,
            temperature=0.2,
            backoff_max=backoff_max,
            backend=backend,
        )
        _acc(u)
        raw_text = raw

        # Accept any content-bearing dict; we KNOW the group_id (loop var) and
        # scene_files (payload), so stamp them ourselves rather than forcing the
        # model to echo group_id correctly — that mismatch was driving needless
        # repair retries (~70% extra calls) with no quality benefit.
        if isinstance(obj, dict) and (obj.get("what_happens") or obj.get("beat_title")):
            narr = (obj.get("narration") or "").strip()
            # Guard: an EMPTY narration (seen on action beats) OR a META-GARBAGE
            # narration (the model talking about JSON/parsing its own corrupted
            # input) must not be silently accepted — retry the full generation
            # for a real line, and only on the last attempt fall back to a clean
            # line so it's never blank and never voiced as garbage.
            if not narr or _is_meta_garbage(narr):
                if _attempt < retries:
                    continue
                obj["narration"] = _clean_fallback_narration(
                    obj.get("beat_title") or "", obj.get("what_happens") or "")
            obj["group_id"] = gid
            obj["scene_files"] = scene_files
            return obj

        repair_payload = {
            "group_id": gid,
            "scene_files": scene_files,
            "last_output": (raw_text or "")[:4000],
            "instruction": "Re-output the beat as VALID JSON matching the schema exactly. No extra text.",
        }
        obj2, raw2, u2 = _call_model_with_backoff(
            client=client,
            model=model,
            system_instruction="You are a strict JSON formatter. Output valid JSON only.",
            user_payload=repair_payload,
            image_paths=[],
            response_schema=beat_schema,
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            backoff_max=backoff_max,
            backend=backend,
        )
        _acc(u2)
        raw_text = raw2
        if isinstance(obj2, dict) and (obj2.get("what_happens") or obj2.get("beat_title")):
            # A repaired beat can still carry meta-garbage narration — scrub it.
            if _is_meta_garbage((obj2.get("narration") or "").strip()):
                obj2["narration"] = _clean_fallback_narration(
                    obj2.get("beat_title") or "", obj2.get("what_happens") or "")
            obj2["group_id"] = gid
            obj2["scene_files"] = scene_files
            return obj2

    return None


def _select_images_for_group(
    payload: Dict[str, Any],
    vision_by_file: Dict[str, Dict[str, Any]],
    max_images: int,
) -> List[str]:
    if max_images <= 0:
        return []

    candidates: List[Tuple[float, str]] = []
    for sf in payload.get("scene_files") or []:
        it = vision_by_file.get(sf) or {}

        # NEW: skip images for scenes excluded from production
        if it.get("use_for_video") is False:
            continue

        sp = it.get("scene_path")
        if not sp:
            continue

        tc = it.get("text_coverage")
        try:
            score = float(tc) if tc is not None else 0.30
        except Exception:
            score = 0.30

        # Lower text coverage first (more visually informative)
        candidates.append((score, sp))

    candidates.sort(key=lambda x: x[0])
    img_paths = [p for _, p in candidates]
    return img_paths[:max_images]


def build_beat_schema() -> dict:
    """Return the Gemini response schema for a narrative beat."""
    return {
        "type": "OBJECT",
        "properties": {
            "group_id": {"type": "INTEGER"},
            "scene_files": {"type": "ARRAY", "items": {"type": "STRING"}},
            "beat_title": {"type": "STRING"},
            "what_happens": {"type": "STRING"},
            "narration": {"type": "STRING"},
            "panel_narration": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "scene_file": {"type": "STRING"},
                        "line": {"type": "STRING"},
                    },
                    "required": ["scene_file", "line"],
                },
            },
            "emotional_turn": {"type": "STRING"},
            "conflict_or_stakes": {"type": "STRING"},
            "reveals_or_info": {"type": "STRING"},
            "hook": {"type": "STRING"},
            "mood_words": {"type": "ARRAY", "items": {"type": "STRING"}},
            "rendering_hints": {
                "type": "OBJECT",
                "properties": {
                    "avoid_text_zoom": {"type": "BOOLEAN"},
                    "preferred_focus": {"type": "STRING"},
                    "camera_motion": {"type": "STRING"},
                },
                "required": ["avoid_text_zoom", "preferred_focus", "camera_motion"],
            },
            "scene_selection": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "scene_file": {"type": "STRING"},
                        "role": {"type": "STRING"},          # keep | redundant
                        "bubble_mode": {"type": "STRING"},   # spoken|inner_thought|narration|shout|none
                        "intensity": {"type": "STRING"},     # calm|tense|intense|explosive
                        "reason": {"type": "STRING"},
                    },
                    "required": ["scene_file", "role", "bubble_mode", "intensity"],
                },
            },
        },
        "required": [
            "group_id",
            "scene_files",
            "beat_title",
            "what_happens",
            "narration",
            "panel_narration",
            "emotional_turn",
            "conflict_or_stakes",
            "reveals_or_info",
            "hook",
            "mood_words",
            "rendering_hints",
            "scene_selection",
        ],
    }


def align_panel_narration(scene_files, model_panels, understand_by_file=None):
    """Return exactly one {scene_file, line} per surviving scene_file, in order.

    Match the model's returned lines to panels by scene_file; fall back to
    positional fill for any panel the model didn't key; pad any still-missing
    panel with a grounded line from the understanding (description/action/
    subjects); fold overflow lines into the LAST panel so nothing is lost. Never
    invents a panel absent from scene_files. Guarantees len(out)==len(scene_files).
    """
    understand_by_file = understand_by_file or {}
    files = [f for f in (scene_files or []) if f]
    file_set = set(files)
    keyed: Dict[str, str] = {}
    leftover: List[str] = []
    for item in (model_panels or []):
        if not isinstance(item, dict):
            continue
        line = str(item.get("line") or item.get("narration") or "").strip()
        if not line:
            continue
        sf = str(item.get("scene_file") or "").strip()
        if sf in file_set and sf not in keyed:
            keyed[sf] = line
        else:
            leftover.append(line)
    for f in files:                       # positional fill for unkeyed panels
        if f not in keyed and leftover:
            keyed[f] = leftover.pop(0)
    for f in files:                       # grounded pad — never empty, never camera prose
        if f not in keyed:
            u = understand_by_file.get(f) or {}
            action = str(u.get("action") or "").strip()
            desc = str(u.get("description") or "").strip()
            subj = ", ".join(str(s) for s in (u.get("subjects") or []) if s).strip()
            # D4: the understanding `description` is often camera/shot framing
            # ("A close-up shot shows..."). NEVER copy that verbatim. Prefer the
            # concrete action, then a NON-camera description, then the named
            # subjects; if everything usable is camera prose or empty, leave a
            # short heal-flaggable bridge instead of reading the picture.
            keyed[f] = next(
                (c for c in (action, desc, subj)
                 if c and not is_shot_description(c)),
                "The moment holds.")
    out = [{"scene_file": f, "line": keyed[f]} for f in files]
    if leftover and out:                  # fold any remaining overflow into the last panel
        out[-1]["line"] = (out[-1]["line"] + " " + " ".join(leftover)).strip()
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the ArgumentParser for gemini_narrative_pass."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-manifest", required=True)
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--project", default="",
                    help="GCP project (required for --backend vertex)")
    ap.add_argument("--location", default="",
                    help="Vertex location (required for --backend vertex)")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--backend", choices=["vertex", "ollama"], default="vertex",
                    help="ollama = local open model (Gemma 4) via the Ollama "
                         "server; no GCP creds, $0")
    ap.add_argument("--ollama-model", default="gemma4:26b")

    ap.add_argument("--min-sleep", type=float, default=1.2, help="Sleep between groups to avoid 429 bursts")
    ap.add_argument("--max-images-per-group", type=int, default=3, help="Cap images attached per group (0=none)")
    ap.add_argument("--backoff-max", type=float, default=60.0, help="Max seconds for 429 backoff sleep")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="Write output every N groups")

    ap.add_argument("--max-groups", type=int, default=0, help="0 = all")
    ap.add_argument("--resume", action="store_true", help="If out exists, keep good beats and only regen errors/missing")
    ap.add_argument("--retries", type=int, default=2, help="Retries per group on parse/validation failure")
    ap.add_argument("--max-output-tokens", type=int, default=2400)
    ap.add_argument("--cast", default="", help="Optional manifest.cast.json for consistent character naming + dialogue attribution")
    ap.add_argument("--story", default="", help="Optional manifest.story.json (chapter spine: logline + ordered arc) so each beat advances ONE connected story")
    ap.add_argument("--corrections", default="", help="Optional JSON {group_id: note}; force-regen those groups with the note appended (closed-loop grounding gate)")
    ap.add_argument("--understood", default="",
                    help="manifest.panels.understood.json for per-panel pad grounding")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    groups_m = load_json(args.groups_manifest)
    vision_m = load_json(args.vision_manifest)
    understood_m = load_json(args.understood) if args.understood and os.path.exists(args.understood) else {}
    u_by_file = {p.get("scene_file"): p for p in (understood_m.get("panels") or []) if p.get("scene_file")}

    groups = _read_groups(groups_m)
    if not groups:
        raise SystemExit("No groups/shots found (expected key: shots or groups)")

    vision_by_file = _build_vision_map(vision_m)

    if args.backend == "ollama":
        client = None
        args.model = args.ollama_model
    else:
        if not args.project or not args.location:
            raise SystemExit("--project/--location are required for --backend vertex")
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)

    system = (
        "You are a YouTube manhwa recap story editor.\n"
        "Given consecutive scene images + OCR, produce ONE structured beat for that group.\n"
        "Be faithful to visible content.\n"
        "Avoid excessive poetic language.\n"
        "End with a strong hook line.\n"
        "Rendering hints: avoid zooming into text bubbles; focus faces/hands/key objects/wide.\n"
        "\n"
        "For EACH file in scene_files, in order, WRITE ONE narration line in "
        "'panel_narration' as {scene_file, line}. Give EVERY panel its own line — "
        "a quick action panel gets a punchy phrase, a pivotal/quiet panel gets a "
        "fuller cinematic sentence; match length to what the panel shows. The lines "
        "must FLOW as one continuous story (continue from previous_narration), not "
        "isolated captions. Then set 'narration' to all the lines joined with a space.\n"
        "    - PACE = INPUT_JSON.intensity (the beat's energy) AND how many panels this beat\n"
        "      spans. A MULTI-PANEL action or shock beat (a fight, a reveal, a power awakening\n"
        "      shown across SEVERAL panels) is a CINEMATIC SET-PIECE — give it the FULLEST\n"
        "      treatment: build the moment across the panels with vivid, sensory drama — the\n"
        "      impact, the reaction, the dread, the stakes — so the montage has room to LAND.\n"
        "      Do NOT compress a multi-panel action climax into one efficient line; that beat\n"
        "      is the moment the audience came for, so make the words MATCH the screen time\n"
        "      those panels take. Keep lines SHORT and punchy ONLY for a SINGLE dramatic panel\n"
        "      (one hit, one cut). A 'calm' or 'tense' beat earns reflective, scene-setting\n"
        "      narration — the stakes, what the character feels. NEVER let a big multi-panel\n"
        "      moment feel thin; NEVER pad a genuinely quiet single panel. Match the scene's\n"
        "      SCALE and energy.\n"
        "    - GROUND it strictly in THESE panels — describe only what is actually drawn here.\n"
        "      Invent NOTHING: no event/motion/outcome not shown, and NO setting that isn't\n"
        "      visible (never 'chandeliers', 'a grand hall', 'marble', 'parchment' unless on the page).\n"
        "      USE THE UNDERSTANDING: each panel's INPUT_JSON.scenes_signals carries its\n"
        "      description, action, setting, dialogue, subjects, panel_kind, and intensity.\n"
        "      These fields cover even a panel omitted by the image cap. Treat them as the\n"
        "      factual source: name the listed subjects in those words. Do not rename\n"
        "      them (if it says 'beast' it is a beast, not a 'hound'), do not change their number\n"
        "      (two stay two, never 'a pack/swarm'), and do not add a creature/person not listed.\n"
        "      Do NOT invent a SYSTEM the world lacks (no 'server'/'game'/'respawn' on a real scene).\n"
        "    - IDENTITY + NAMES: NAME established CHAPTER CAST members so the audience can\n"
        "      follow who is who — recognition is the priority. NAME the protagonist (or a\n"
        "      relaxed stand-in like 'our guy') normally on HIS OWN panels, even when a\n"
        "      separate mysterious figure is on screen nearby. Reserve a grounded NEUTRAL\n"
        "      handle ('the stranger', 'the intruder') ONLY for a figure THIS panel itself\n"
        "      presents as genuinely concealed — transformed, masked, hooded, glowing,\n"
        "      silhouetted, disguised, or newly-arrived (e.g. 'gear unlike anything') — and not\n"
        "      yet matched to a known character. Do NOT neutralize an ESTABLISHED character\n"
        "      just because a concealed figure appears, and do NOT keep calling a clearly-shown,\n"
        "      already-known character 'the stranger'. A power/transformation reveal of an\n"
        "      UNKNOWN figure is a mystery to preserve — but once the story's own text or the\n"
        "      character's established look identifies someone, use their name. Once introduced,\n"
        "      ration the protagonist's real name and usually use pronouns or a relaxed stand-in.\n"
        "    - DIALOGUE — quote selectively, recap-style: PARAPHRASE the bulk into narration but\n"
        "      DO quote occasionally for impact. QUOTE a SHORT (<=6 words), COMPLETE, punchy real\n"
        "      line (a threat, a name, a key line) in clean sentence case, attributed — e.g. he\n"
        "      mutters 'I can't move.', she spits 'Damn you.'. A few such quotes per chapter land\n"
        "      hard; paraphrase everything else. Do NOT quote a whole long bubble; NEVER stack two\n"
        "      long quotes in a row. Good: the Assassins sneer that his 'peasant blood' changes\n"
        "      nothing -> a painless death. inner_thought -> render as the character's thought (at\n"
        "      most one short quote). NEVER quote UI text/watermarks/counters/sound-effects, raw\n"
        "      ALL-CAPS/garbled OCR, or a trailing-off stub ('Ancestor...?') — only real,\n"
        "      complete, sentence-case character speech.\n"
        "    - ACTION beats (a fight, a knife drawn, a strike — few words, lots of motion) are the\n"
        "      CLIMAX: describe the PHYSICAL action vividly and grounded — who draws/strikes/dodges\n"
        "      what, and the stakes (e.g. 'Prince Cheon finally rips his hidden knife free to defend\n"
        "      himself'). Do NOT skip them or retreat into vague atmosphere.\n"
        "    - Present tense, active voice; cinematic but accurate. NEVER name the\n"
        "      shot/camera/panel/image/frame; NEVER begin 'A close-up shot shows...'\n"
        "      or 'The panel shows...'. Narrate the STORY, not the picture.\n"
        "    - PUBLICATION CHROME: if a panel is a series cover, title/chapter-number card,\n"
        "      publisher or studio logo, app UI screen, or credits page — do NOT describe it.\n"
        "      Never narrate 'the chapter opens with...', view counts, or studio names.\n"
        "      Write the narration from the STORY panels only; if a group contains only\n"
        "      chrome, write a one-line bridge into the story instead.\n"
        "    - NARRATIVE CAPTIONS ARE NOT CHROME — a text-only panel or box with the\n"
        "      author's monologue / scene-setting / transition text (e.g. 'BACK THEN,\n"
        "      I HAD NO IDEA.', 'ON THE DAY I FINISHED THE WEB NOVEL...') is the\n"
        "      STORY'S VOICE — WEAVE it into your narration in the character's first\n"
        "      person. You MAY rephrase for flow and fold it together with what's\n"
        "      drawn, but KEEP its meaning and any key line; NEVER drop a caption and\n"
        "      NEVER read one robotically as a bare, thin fragment. A beat that is\n"
        "      ONLY a caption plus an effect/transition panel STILL earns a full,\n"
        "      vivid, grounded line — carry the caption's thought INTO the moment on\n"
        "      screen (the crash, the screech) instead of stopping at the caption.\n"
        "      FRAGMENTS: a caption ending in '...' (e.g. 'AND I...') is HALF A\n"
        "      SENTENCE that continues on the next panel/group. NEVER quote the stub\n"
        "      as a standalone thought — write narration that flows INTO the\n"
        "      continuation (end your line mid-momentum so the next beat completes it).\n"
        "      Even so, your line MUST end on a COMPLETE clause — NEVER let the whole\n"
        "      narration trail off on a dangling quoted stub or bare '...' (do NOT end\n"
        "      with e.g. 'Wait a sec...' or 'What the—'); finish the thought in your\n"
        "      own words.\n"
        "    - CONTINUITY: INPUT_JSON.previous_narration holds the line(s) the narrator\n"
        "      JUST SPOKE. Continue that flow: never re-introduce characters or\n"
        "      re-describe the setting already established, never start with the same\n"
        "      opening words as the previous line, and if the previous line ended\n"
        "      mid-thought, your first words must complete it.\n"
        "    - TONAL CONTINUITY: it is ONE narrator telling ONE continuous story, not\n"
        "      separate clips. Do NOT hard-jump the energy between beats — when this\n"
        "      beat's intensity is far from the line just spoken (a calm aside right\n"
        "      after an explosive fight, or the reverse), EASE in with a short bridge\n"
        "      ('and then, just like that, the chaos stilled...' / 'but the quiet\n"
        "      didn't last—') so the pace flows. Match the energy, but TRANSITION into\n"
        "      it; never start cold in a wildly different tone from the previous line.\n"
        "    - VOCABULARY FRESHNESS: do NOT reuse the same atmospheric or descriptive\n"
        "      words you already used in previous_narration. If you wrote 'moon',\n"
        "      'shadow', 'pale', or 'mist' earlier in the chapter, find fresh phrasing\n"
        "      now — describe what is concretely drawn (a scar, a fist, a doorway)\n"
        "      rather than reaching for generic atmosphere. Avoid stock clichés such as\n"
        "      'under the pale moonlight', 'shadows dance', 'mist rolls in'. Vary the\n"
        "      vocabulary: one strong specific image beats three recycled mood words.\n"
        "    - STORY SPINE: a CHAPTER STORY SPINE (logline + the ordered arc) is given\n"
        "      below, and INPUT_JSON.arc_label is THIS beat's place in it. Write the\n"
        "      line to ADVANCE that story — connect it to what came before, set up what\n"
        "      comes next, and carry the chapter's through-line so the recap is ONE\n"
        "      story (e.g. tie 'I know how this goes' back to the years he spent reading\n"
        "      it alone). The spine is CONTEXT only — assert nothing not visible in THESE\n"
        "      panels, and keep captions verbatim.\n"
        "\n"
        "{CAST_BLOCK}"
        "{STORY_SPINE}"
        "ALSO judge each panel for the recap video (scene_selection, one entry per scene_file):\n"
        "  role: DEFAULT to 'keep'. Only mark a panel 'redundant' when it is genuinely\n"
        "    expendable — i.e. ONE of these clearly holds:\n"
        "      (a) DUPLICATE: it shows essentially the SAME moment as another panel here (a\n"
        "          near-identical repeat, or a barely-different frame of one continuous motion); OR\n"
        "      (b) CROPPED FRAGMENT: it is a partial/cut-off version of another panel — a face or\n"
        "          body sliced at a panel edge, a thin sliver, a stitch-seam fragment; OR\n"
        "      (c) TEXT/BUBBLE PANEL: it is dominated by a speech bubble or SFX text with little\n"
        "          distinct artwork — once bubbles are cleaned it is near-blank, so it adds nothing\n"
        "          visually (its words still get woven into the narration). Mark it 'redundant'.\n"
        "    For a duplicate pair, KEEP the one with the most COMPLETE framing and mark the other\n"
        "    'redundant'. Do NOT drop a panel merely for being a minor reaction, a transition, or\n"
        "    'for brevity' — distinct panels (even small ones) stay 'keep'. Most panels are 'keep';\n"
        "    only the true duplicates and cropped fragments are 'redundant'.\n"
        "  bubble_mode: the dominant speech-bubble style — 'spoken' (smooth oval, said aloud),\n"
        "    'inner_thought' (jagged/cloud, thinking), 'narration' (rectangular caption box),\n"
        "    'shout' (spiky), or 'none' if no bubble.\n"
        "  intensity: the emotional energy — 'calm', 'tense', 'intense', or 'explosive'.\n"
        "Return ONLY valid JSON matching the provided schema. No extra text.\n"
    )
    cast_block = _build_cast_block(args.cast)
    # Same cast list (loaded once) feeds the per-beat token resolver, which scrubs
    # any bracketed cast token the model copied into the final narration.
    cast_list = _load_cast_list(args.cast)
    story_block = _build_story_block(args.story)
    system = system.replace("{CAST_BLOCK}", cast_block)
    system = system.replace("{STORY_SPINE}", story_block)
    # Generator-side advertiser-safety rules ride the narration prompt so the
    # narration is brand-safe at the source; the sanitize-pass NET still runs
    # downstream regardless.
    system = (system + "\n\n" + SAFE_NARRATION_RULES + "\n\n"
              + _DIALOGUE_RULE + "\n\n" + RECAP_STYLE_RULES)
    corrections: Dict[int, str] = {}
    if args.corrections and os.path.exists(args.corrections):
        try:
            corrections = {int(k): str(v) for k, v in json.load(open(args.corrections)).items()}
        except Exception:
            corrections = {}

    beat_schema = build_beat_schema()

    existing_by_id: Dict[int, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.out):
        try:
            existing = load_json(args.out)
            for b in (existing.get("beats") or []):
                gid = int(b.get("group_id") or 0)
                if gid and not b.get("error"):
                    existing_by_id[gid] = b
        except Exception:
            existing_by_id = {}

    max_groups = args.max_groups if args.max_groups > 0 else len(groups)

    beats_out: List[Dict[str, Any]] = []
    parse_errors = 0
    regenerated = 0
    usage = UsageAccumulator(args.model)

    def write_checkpoint() -> None:
        tmp_obj = {
            "source_groups_manifest": os.path.abspath(args.groups_manifest),
            "source_vision_manifest": os.path.abspath(args.vision_manifest),
            "model": args.model,
            "count_beats": len(beats_out),
            "stats": {"parse_errors": parse_errors, "regenerated": regenerated},
            "beats": sorted(beats_out, key=lambda x: int(x.get("group_id") or 0)),
        }
        dump_json(args.out, tmp_obj)

    for g in groups[:max_groups]:
        gid = int(g.get("shot_id") or g.get("group_id") or 0)
        if not gid:
            continue

        # Resume keeps good beats — UNLESS this group has a correction queued
        # (closed-loop grounding gate), in which case we force a regen.
        if gid in existing_by_id and gid not in corrections:
            beats_out.append(existing_by_id[gid])
            continue

        sys_g = system
        if gid in corrections:
            sys_g = sys_g + (
                "\n\nCORRECTION FOR THIS GROUP — the previous narration had this problem:\n  "
                + corrections[gid] + "\n"
                "Rewrite the 'narration' to FIX it: stay strictly to what is visible here plus the "
                "panel's actual dialogue, COVER every on-panel caption in full, keep the cast names, "
                "assert nothing not shown, and never leave the narration empty.\n"
            )
            regenerated += 1

        payload = _pack_group_payload(g, vision_by_file, u_by_file)
        # rolling context: the last spoken lines ride along so each beat
        # CONTINUES the story instead of re-opening it (and completes any
        # fragment the previous caption left hanging)
        prev = [str(b.get("narration") or "")
                for b in beats_out[-2:] if b.get("narration")]
        if prev:
            payload["previous_narration"] = prev
        img_paths = _select_images_for_group(payload, vision_by_file, args.max_images_per_group)

        beat = _generate_beat_for_group(
            client=client,
            model=args.model,
            system_instruction=sys_g,
            payload=payload,
            image_paths=img_paths,
            beat_schema=beat_schema,
            gid=gid,
            retries=args.retries,
            max_output_tokens=args.max_output_tokens,
            backoff_max=args.backoff_max,
            backend=args.backend,
            usage=usage,
        )

        if beat is None:
            parse_errors += 1
            beat = {
                "group_id": gid,
                "scene_files": payload["scene_files"],
                "beat_title": "Beat",
                "what_happens": "Unable to parse model output.",
                "emotional_turn": "unknown",
                "conflict_or_stakes": "unknown",
                "reveals_or_info": "unknown",
                "hook": "Something shifts…",
                "mood_words": ["uncertain"],
                "rendering_hints": {
                    "avoid_text_zoom": True,
                    "preferred_focus": "wide",
                    "camera_motion": "slow_pan",
                },
                "scene_selection": [],
                "error": "parse_failed_after_retries",
            }

        # Strip any bracketed cast token the model copied into the narration so
        # the TTS never voices a literal '[protagonist]'. Conservative — never
        # blanks a line; an unknown token degrades to its readable inner words.
        if beat.get("narration"):
            beat["narration"] = _resolve_cast_tokens(beat["narration"], cast_list)

        # Normalize panel_narration: exactly one line per surviving scene_file.
        # Runs on BOTH normal and fallback beats (the fallback has no panel_narration
        # so align_panel_narration will pad every panel from u_by_file / defaults).
        # We derive narration from the panel lines here, overwriting what the model
        # joined so the joined string stays in sync with the per-panel lines.
        # narration_plain (owned by the punchup stage) is NOT set.
        surviving = [f for f in (beat.get("scene_files") or payload["scene_files"]) if f]
        beat["panel_narration"] = align_panel_narration(
            surviving, beat.get("panel_narration"), u_by_file)
        assert len(beat["panel_narration"]) == len(surviving), (
            f"panel_narration/scene_files mismatch in group {gid}")
        beat["narration"] = " ".join(p["line"] for p in beat["panel_narration"]).strip() or beat.get("narration", "")

        # Guarantee exactly one sanitized selection entry per scene (defaults to
        # 'keep' so a parse gap never silently drops a panel).
        beat["scene_selection"] = normalize_scene_selection(
            beat.get("scene_selection"), payload["scene_files"]
        )
        beats_out.append(beat)

        # Throttle between groups (burst prevention)
        if args.min_sleep > 0:
            time.sleep(args.min_sleep + random.random() * 0.25)

        # Checkpoint frequently
        if args.checkpoint_every > 0 and (len(beats_out) % args.checkpoint_every == 0):
            write_checkpoint()

    beats_out.sort(key=lambda x: int(x.get("group_id") or 0))
    identity_reveals_neutralized = neutralize_identity_reveal_leaks(
        {"beats": beats_out}, {"cast": cast_list}, vision_by_file, u_by_file)
    spoken_fragments_repaired = repair_spoken_fragments({"beats": beats_out})
    # an exact-duplicate consecutive panel line (p95/p96 'Ancestor...?') must not
    # ship twice — merge the duplicate panel out so the line is voiced once.
    consecutive_dups_merged = dedupe_consecutive_panel_lines({"beats": beats_out})
    out_obj = {
        "source_groups_manifest": os.path.abspath(args.groups_manifest),
        "source_vision_manifest": os.path.abspath(args.vision_manifest),
        "model": args.model,
        "count_beats": len(beats_out),
        "stats": {
            "parse_errors": parse_errors,
            "regenerated": regenerated,
            "identity_reveals_neutralized": identity_reveals_neutralized,
            "spoken_fragments_repaired": spoken_fragments_repaired,
            "consecutive_dups_merged": consecutive_dups_merged,
            "usage": {
                "calls": usage.calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "est_cost_usd": round(usage.cost(), 4),
            },
        },
        "beats": beats_out,
    }
    dump_json(args.out, out_obj)
    print(f"[ok] wrote={args.out} beats={len(beats_out)} parse_errors={parse_errors} regenerated={regenerated}")
    print(usage.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
