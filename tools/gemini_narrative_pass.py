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
from manifest_io import write_manifest  # noqa: E402
from narration_safe_rules import SAFE_NARRATION_RULES  # noqa: E402
from niche_modules import register_block  # noqa: E402
from recap_style import (  # noqa: E402
    RECAP_STYLE_RULES,
    dedupe_consecutive_panel_lines,
    ends_terminal,
    is_shot_description,
    mentions_figures_leak,
    mentions_image_file,
    mentions_impact_marker,
    mentions_mood_tag_leak,
    neutralize_identity_reveal_leaks,
    repair_spoken_fragments,
)
from beats_segments import (  # noqa: E402
    beat_segments,
    has_native_segments,
    write_segment_lines,
)
from span_align import span_align_pass  # noqa: E402

# --- adaptive flow segments (spec 2026-07-02) ---------------------------------
# The writer emits beats[].segments[] = [{"span": [scene_files...], "line": ...}]
# — a flow passage spans 2-4 consecutive panels voiced as ONE clip; solo lines
# stay for money shots / system cards. Deterministic guardrails live OUTSIDE the
# LLM in validate_segments(); constants are code, not config.
SPAN_CAP = 4        # max panels one segment may span (4 x 6.0s = 24s max clip)
WPM = 135           # word-budget arithmetic; matches script_expander's default
# HARD gates only — the budget's job is to reject the UNSHIPPABLE, not to
# teach taste (that lives in the prompt's word guidance). History: 6.0s max
# hard-failed gemma's money-shot holds (18/21 ch1 beats fell back); 10.0 still
# rejected 8/21 marginal cases the pipeline handles fine — a thin clip is
# floor-EXTENDED by the planner (each panel still gets >=2.0s on screen; short
# hold tail), and a 12s/panel hold was routine in the per-panel era (ran to
# 18s). Egregious bounds only: sub-1s/panel is unspeakable coverage; >15s is
# a parked monologue.
_SEG_MIN_SEC_PER_PANEL = 1.0
_SEG_MAX_SEC_PER_PANEL = 15.0
_MOOD_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]")

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
    # prose-first hands the model scene_file names as sentence tags — a file
    # name leaking into the narration ("…to conclude at p000032.jpg") is the
    # model narrating its bookkeeping, the same meta family
    r"\S+\.(?:jpe?g|png|webp)\b",
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


def demote_backfilled_error(beat: Dict[str, Any]) -> Dict[str, Any]:
    """A GROUP-level JSON parse failure sets `error`, but the narration is still
    backfilled (one valid line per surviving scene_file — `panel_narration` in
    per_panel mode, singleton-span `segments` in adaptive mode). Once those
    lines exist the beat carries REAL narration — rename the parse-failure flag to
    `group_parse_error` so no downstream stage (script_expander, prep QA, resume)
    silences a beat that has valid lines, while keeping the telemetry. No-op on a
    healthy beat, and on an error beat with no usable lines the flag stays so it
    is still regenerated/handled as a failure."""
    if beat.get("error") and (beat.get("panel_narration") or beat.get("segments")):
        beat["group_parse_error"] = beat.pop("error")
    return beat


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
    "incomplete fragment that trails off on an ellipsis — finish the thought "
    "in your own words instead. NEVER voice publication chrome — ads, credits, "
    "'subscribe/follow/join our Discord', watermarks, scanlator or site names. "
    "When a real line is short and iconic — a threat, a taunt, a name — prefer "
    "QUOTING it (clean sentence case, attributed) over paraphrasing."
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
        "this cast. Where an entry shows 'SAY: ...', speak THOSE words — the "
        "name before it is an internal label for telling look-alikes apart and "
        "must never be read aloud (a viewer should hear 'one of the assassins', "
        "never 'the Assassin Member'):"
    ]
    for c in cast:
        name = c.get("canonical_name") or c.get("id") or "?"
        # SAY the spoken form, not the identity key. canonical_name has to
        # tell look-alikes apart ("Assassin Member" vs "Assassin Leader"),
        # which makes it a catalogue label; read aloud it produced "The
        # Assassin Member watches him with a sharp gaze". Older casts carry
        # no spoken_name and fall back to the key, as before.
        spoken = (c.get("spoken_name") or "").strip()
        role = c.get("role") or ""
        desc = (c.get("visual_description") or "").strip()
        aliases = ", ".join(c.get("aliases") or [])
        tag = f" (aka {aliases})" if aliases else ""
        say = (f" — SAY: {spoken}"
               if spoken and spoken.lower() != name.lower() else "")
        lines.append(f"  - {name} ({role}){tag}{say}: {desc}")
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


def _non_camera_description(understood: Optional[Dict[str, Any]]) -> str:
    """The panel's cleanest NON-camera factual signal — the ONE ladder shared by
    the narration writer's input (`_pack_group_payload`) and the grounded-pad
    stand-in (`_grounded_pad_line`), kept in a single place (DRY).

    D4: an understanding's `description` is often camera/shot framing ("A
    close-up shot shows...") or rendering-effect prose ("...creating motion
    blur"); handed to the writer VERBATIM it gets echoed back as a
    `shot_description` narration line (and the skipped-panel pad reuses it). So
    prefer the concrete `action`, then a non-camera `description`, then the named
    `subjects`, then a neutral `subjects`+`setting` summary — NEVER a
    camera/effect phrase. Returns "" when nothing usable remains (callers supply
    their own ultimate fallback). The returned string is complete — never
    truncated (callers apply their own length caps)."""
    u = understood or {}
    action = str(u.get("action") or "").strip()
    desc = str(u.get("description") or "").strip()
    subjects = [str(s).strip() for s in (u.get("subjects") or []) if str(s).strip()]
    setting = str(u.get("setting") or "").strip()
    # 1) the concrete action, then 2) a non-camera description.
    for c in (action, desc):
        if c and not is_shot_description(c):
            return c
    # 3) the named subjects (dropping any that themselves read as a shot phrase),
    # 4) enriched with a non-camera setting into a neutral summary.
    clean_subjects = [s for s in subjects if not is_shot_description(s)]
    if clean_subjects:
        summary = ", ".join(clean_subjects)
        if setting and not is_shot_description(setting):
            summary = f"{summary} in {setting}"
        if not is_shot_description(summary):
            return summary
    # 4b) no usable subjects — a bare non-camera setting still beats camera prose.
    if setting and not is_shot_description(setting):
        return setting
    return ""


def _pack_group_payload(
    group: Dict[str, Any],
    vision_items_by_file: Dict[str, Dict[str, Any]],
    understand_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    figures_by_file: Optional[Dict[str, List[Dict[str, str]]]] = None,
    echo_of: Optional[Dict[str, str]] = None,
    ledger_facts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scene_files = group.get("scene_files") or []
    scenes: List[Dict[str, Any]] = []
    understand_by_file = understand_by_file or {}
    figures_by_file = figures_by_file or {}
    echo_of = echo_of or {}

    for sf in scene_files:
        it = vision_items_by_file.get(sf) or {}
        understood = understand_by_file.get(sf) or {}
        v = it.get("vision") or {}
        labels = [x.get("desc") for x in (v.get("labels") or []) if x.get("desc")]
        objects = [x.get("name") for x in (v.get("objects") or []) if x.get("name")]
        _desc = str(understood.get("description") or "").strip()
        _action = str(understood.get("action") or "").strip()

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
                # D4 INPUT sanitization: never hand the writer a camera/shot
                # description or a rendering-effect action — it echoes them back
                # as `shot_description` narration. Keep a clean description;
                # otherwise fall through the shared non-camera ladder. A
                # camera-phrased action is dropped (its content, if any, resurfaces
                # via the description ladder).
                "description": (_desc if _desc and not is_shot_description(_desc)
                                else _non_camera_description(understood))[:500],
                "action": (_action if _action and not is_shot_description(_action)
                           else "")[:240],
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
        # Eyes wave: the DETECTOR-stamped impact verdict (panel_understand)
        # must reach the writer as an unmissable per-panel marker — OCR sees
        # none of the painted SFX, so without this the writer under-reads a
        # stab panel as calm. Keys exist ONLY when the signal does, keeping
        # the payload byte-compatible for every unflagged panel.
        if (understood.get("impact_sfx") or {}).get("present"):
            scenes[-1]["impact_sfx"] = "[IMPACT SFX on panel]"
        _sow = str(understood.get("strikes_or_weapons") or "").strip().lower()
        if _sow and _sow != "none":
            scenes[-1]["strikes_or_weapons"] = _sow
        # Round-2 identity fix: the panel's cast-resolved FIGURES (deterministic
        # keyword evidence, tools/cast_identity.py) ride the payload so the
        # writer names actors from GROUND truth, not vibes ("the assassin draws
        # his steel" over Cheon's counter-draw). Key exists ONLY when a cast
        # manifest resolved something — byte-compatible otherwise.
        figs = figures_by_file.get(sf) or []
        if figs:
            scenes[-1]["figures"] = [
                (f["name"] if f.get("name") and f["name"] != "unknown"
                 else f"unknown ({str(f.get('evidence') or '')[:40]})")
                for f in figs[:4]]
        # pu_v4: the analyst itself hedged on this panel — the writer must not
        # upgrade that uncertainty into a concrete event (key exists only when
        # flagged, byte-compatible otherwise).
        if understood.get("uncertain"):
            scenes[-1]["uncertain"] = True
        # FIRST-PERSON dialogue is spoken BY a character about themselves, and
        # a reaction shot often draws the LISTENER instead. Binding the words
        # to whoever is in frame produced "the mysterious figure's vision
        # begins to fail" over the dying protagonist's own line.
        if _FIRST_PERSON_RE.search(str(understood.get("dialogue") or "")):
            scenes[-1]["dialogue_voice"] = (
                "first-person — spoken BY a character ABOUT THEMSELVES; the "
                "speaker is often NOT the figure drawn here")
        # zoom/echo: the later panel of a story_group echo pair repeats the
        # earlier one's art re-framed — one moment, narrated once.
        if sf in echo_of:
            scenes[-1]["echo_of"] = echo_of[sf]

    payload = {
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
    # Story-state ledger (2026-07-20): this beat's slice of the chapter fact
    # record — dialogue-arbitrated action directions, who is dead by now,
    # banned handles, answered questions. Key exists ONLY when a ledger does
    # — byte-compatible otherwise.
    if ledger_facts:
        payload["facts"] = ledger_facts
    return payload


# First-person speech markers (generic English, OCR is upper-case).
_FIRST_PERSON_RE = re.compile(
    r"\b(I|I'm|I've|I'll|my|me|mine|myself)\b", re.IGNORECASE)

_MD_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S)


def _extract_json_value(text: str) -> Any:
    """Fence/prose/think-tolerant JSON extraction — object OR array.

    2026-07-16: the old object-only find('{')..rfind('}') fallback was dead
    code under ollama's format-constrained decoding, and WRONG the moment a
    backend free-writes: a grouping response is a JSON ARRAY, so the slice
    glued the beat objects together without their brackets and never parsed.
    Backend-agnostic by design (ollama / MLX shim / vertex raw fallback)."""
    if not text:
        return None
    t = _THINK_BLOCK_RE.sub("", str(text))
    m = _MD_FENCE_RE.search(t)
    if m:
        t = m.group(1)
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for o, c in (("[", "]"), ("{", "}")):
        s, e = t.find(o), t.rfind(c)
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except Exception:
                continue
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    v = _extract_json_value(text)
    return v if isinstance(v, dict) else None


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


def _bumped_num_ctx(err_str: str, cur_ctx: int, num_predict: int,
                    ctx_max: int = 16384) -> Optional[int]:
    """If *err_str* is an ollama context-exceed error, return a num_ctx that fits
    the reported prompt + a generation/headroom margin (rounded up to 1k, capped
    at *ctx_max*); else None. Lets a rare oversized beats group retry at a
    fit-to-prompt context instead of hard-failing the whole chapter, while normal
    groups stay at the small default (no gemma SWA-cache thrash)."""
    if "context" not in err_str.lower():
        return None
    m = (re.search(r"\((\d+)\s*tokens\)", err_str)
         or re.search(r"n_prompt_tokens[\"\s:]+(\d+)", err_str))
    if not m:
        return None
    need = int(m.group(1)) + max(0, int(num_predict)) + 1024
    fit = min(int(ctx_max), ((need + 1023) // 1024) * 1024)
    return fit if fit > int(cur_ctx) else None


# A single over-tall panel (ORV Ep1 ~4623x800) OOMs gemma's Metal VISION encoder
# at full res — hit in BOTH the understanding pass AND the beats pass, since both
# attach panel images here. Downscale to _MODEL_MAX_IMG_H before the local model
# sees them (understanding-only vs beats makes no difference — the encoder is the
# same). Records/scene_sha keep the original scene; this touches only the bytes
# sent to gemma. nano's tallest (~2400px) works, so cap at 2560. Env-tunable.
# (STUDIO_UNDERSTAND_MAX_H kept as the name — it's the same knob, now shared.)
_MODEL_MAX_IMG_H = int(os.environ.get("STUDIO_UNDERSTAND_MAX_H", "2560"))
# Several tall images in ONE call STACK in GPU memory — the beats pass sends up
# to 3 panels/group (--max-images-per-group), and 3x2560px still OOM'd ORV. So
# bound the TOTAL height across a call: 1 image keeps the full cap, N images
# split this budget (floor 1024px each). ORV group of 3 tall panels → ~1365px.
_MODEL_TOTAL_IMG_H = int(os.environ.get("STUDIO_MODEL_TOTAL_IMG_H", "4096"))


def _images_height_cap(n_images: int) -> int:
    """Per-image height cap for a call sending *n_images*: full cap for one, a
    split of the total budget (floor 1024) for several. Bounds stacked vision
    memory so a group of tall panels can't OOM Metal even after per-image scale."""
    if n_images <= 1:
        return _MODEL_MAX_IMG_H
    return min(_MODEL_MAX_IMG_H, max(1024, _MODEL_TOTAL_IMG_H // n_images))


def _model_safe_image(image_path: Optional[str], max_h: Optional[int] = None
                      ) -> Tuple[Optional[str], Optional[str]]:
    """(path_to_send, temp_to_cleanup). Downscale an over-tall image to *max_h*
    (default _MODEL_MAX_IMG_H) so the local vision encoder can't OOM; return the
    original (temp=None) when it already fits or on any error (fail-soft)."""
    cap = int(max_h or _MODEL_MAX_IMG_H)
    if not image_path or not os.path.exists(image_path):
        return image_path, None
    try:
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(image_path) as im:
            w, h = im.size
            if h <= cap:
                return image_path, None
            nw = max(1, round(w * cap / h))
            small = im.convert("RGB").resize((nw, cap), Image.LANCZOS)
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".jpg", prefix="pu_safe_")
        os.close(fd)
        small.save(tmp, "JPEG", quality=90)
        return tmp, tmp
    except Exception:
        return image_path, None


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
        _img_tmps: List[str] = []
        if images:
            _cap = _images_height_cap(len(images))   # split the budget across a stacked group
            _safe: List[str] = []
            for _p in images:
                _sp, _tmp = _model_safe_image(_p, _cap)
                if _sp:
                    _safe.append(_sp)
                if _tmp:
                    _img_tmps.append(_tmp)
            msg["images"] = _safe
        from ollama_compat import chat as _ollama_chat
        # 16k thrashed gemma's SWA cache (full prompt re-processing every call ->
        # ~32min wedge), so beats run at a small default (8k) that fits the typical
        # ~1-7.5k prompt. An oversized group (many panels) can overflow it -> we
        # catch the ollama context-exceed error and retry THAT call at a
        # fit-to-prompt num_ctx (capped), so small groups stay fast and a big group
        # never hard-fails the whole chapter. Both env-tunable.
        ctx0 = int(os.environ.get("STUDIO_BEATS_NUM_CTX", "8192"))
        ctx_max = int(os.environ.get("STUDIO_BEATS_NUM_CTX_MAX", "16384"))
        _kw = dict(
            model=model,
            messages=[{"role": "system", "content": system_instruction}, msg],
            format=_schema_to_json_schema(response_schema),
            think=False,  # Gemma 4 thinks by default and burns the budget
            options={"temperature": temperature,
                     "num_predict": max_output_tokens,
                     "num_ctx": ctx0},
        )
        try:
            try:
                resp = _ollama_chat(**_kw)
            except Exception as e:
                nb = _bumped_num_ctx(str(e), ctx0, max_output_tokens, ctx_max)
                if nb is None:
                    raise
                print(f"[beats] prompt exceeds num_ctx {ctx0}; retry at num_ctx={nb}",
                      file=sys.stderr)
                _kw["options"]["num_ctx"] = nb
                resp = _ollama_chat(**_kw)
            raw = (resp.get("message") or {}).get("content") or ""
            usage = {"input": int(resp.get("prompt_eval_count") or 0),
                     "output": int(resp.get("eval_count") or 0), "cached": 0}
            try:
                return json.loads(raw), raw, usage
            except Exception:
                return _extract_json_value(raw), raw, usage
        finally:
            for _t in _img_tmps:
                try:
                    os.remove(_t)
                except OSError:
                    pass

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
        return _extract_json_value(raw), raw, usage


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


def build_beat_schema(segmentation: str = "adaptive") -> dict:
    """Return the Gemini response schema for a narrative beat.

    adaptive: narration comes back as `segments` — ordered {span, line}
    passages whose spans partition the group's scene_files (the shape a
    span-PINNED correction regen must keep speaking).
    prose (the free-generation shape when the tool runs adaptive):
    `narration` is ONE connected passage and `sentences` [{text, panels}]
    is that same passage split and panel-tagged — the spans are derived in
    code by segments_from_sentences(), never by the model.
    per_panel: the legacy 1-line-per-panel `panel_narration` schema,
    byte-identical to the pre-segments tool."""
    schema = {
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
    if segmentation == "prose":
        # prose-first free generation: the passage rides the existing
        # `narration` field (generated EARLY — before the split, so the
        # model authors flowing prose with zero bookkeeping interleaved);
        # `sentences` lands LAST in property order, a mechanical re-split of
        # the passage it already wrote.
        props = schema["properties"]
        del props["panel_narration"]
        props["sentences"] = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "panels": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["text", "panels"],
            },
        }
        schema["required"] = ["sentences" if k == "panel_narration" else k
                              for k in schema["required"]]
    elif segmentation != "per_panel":
        # segments REPLACES panel_narration as the one narration shape the
        # model returns (per_panel keeps the legacy schema byte-identical).
        props = schema["properties"]
        del props["panel_narration"]
        props["segments"] = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "span": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "line": {"type": "STRING"},
                },
                "required": ["span", "line"],
            },
        }
        schema["required"] = ["segments" if k == "panel_narration" else k
                              for k in schema["required"]]
    return schema


def _grounded_pad_line(f, understand_by_file):
    """A grounded stand-in line for a panel the model left uncovered.
    D4: the understanding `description` is often camera/shot framing
    ("A close-up shot shows..."). NEVER copy that verbatim. Delegates to the
    shared `_non_camera_description` ladder (action → non-camera description →
    subjects → subjects+setting summary); if everything usable is camera prose
    or empty, leave a short heal-flaggable bridge instead of reading the
    picture."""
    return (_non_camera_description((understand_by_file or {}).get(f) or {})
            or "The moment holds.")


# A system card's line must SAY the card, never describe it (the prompt bans
# it; gemma still parrots the understanding's "A plain white panel featuring the
# blue text 'SKY CORPORATION.'" into the solo span). Deterministic guard.
_CARD_DESC_RE = re.compile(
    r"\b(?:panel|card|screen|box|frame|window|display)\b[^.]{0,80}\b(?:text|reads?|"
    r"featur\w*|displays?|contains?|shows?|announc\w*|appears?|centered)\b"
    r"|\b(?:text|words?|letters?)\s+(?:appears?|is displayed|reads?|flash\w*)\b"
    r"|\bcentered in the frame\b|\bplain (?:white |black )?(?:panel|card|screen)\b",
    re.IGNORECASE)


def _speak_card(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return t
    t = t[:1].upper() + t[1:].lower()
    return t if t.endswith((".", "!", "?")) else t + "."


def _card_words(text: str):
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 2}


def system_card_line(f, understand_by_file, line):
    """The line to voice on solo system panel *f*: the model's line when it
    voices the card (shares content, doesn't describe it), else the card's own
    text (understanding dialogue / OCR), sentence-cased. No card text -> the
    line as given."""
    u = (understand_by_file or {}).get(f) or {}
    card = str(u.get("dialogue") or u.get("ocr_clean") or "").strip()
    if not card:
        return line
    ln = str(line or "").strip()
    if ln and not _CARD_DESC_RE.search(ln) and (_card_words(ln) & _card_words(card)):
        return ln
    return _speak_card(card)


def auto_repair_segments(segs, surviving, kinds, understand_by_file=None):
    """Deterministic STRUCTURAL repair before validation — the model's prose
    is never rewritten, only spans are adjusted (real ch1: wholesale singleton
    fallback threw away whole beats of good narration over one bad span):
      - a system panel inside a multi-panel span is EXTRACTED as its own solo
        (the first story run keeps the line; extracted/split parts get a
        grounded pad),
      - panels the model SKIPPED are inserted as grounded-pad singletons at
        their reading-order position (unambiguous positions only — a skip
        INSIDE a span's range still fails validation and goes to the model
        repair re-ask).
    Anything else (unknown/duplicate/out-of-order/cap/budget) is left for
    validate_segments. Returns a new list; the input is not mutated."""
    surviving = [f for f in (surviving or []) if f]
    order = {f: i for i, f in enumerate(surviving)}
    out = []
    for s in segs or []:
        span = list(s.get("span") or [])
        line = s.get("line")
        is_sys = [str(kinds.get(f) or "").lower() == "system" for f in span]
        if len(span) > 1 and any(is_sys):
            runs, cur = [], []
            for f, sysf in zip(span, is_sys):
                if sysf:
                    if cur:
                        runs.append(cur)
                        cur = []
                    runs.append([f])
                else:
                    cur.append(f)
            if cur:
                runs.append(cur)
            line_used = False
            for run in runs:
                keep = (not line_used
                        and str(kinds.get(run[0]) or "").lower() != "system")
                out.append({"span": run,
                            "line": line if keep else
                            _grounded_pad_line(run[0], understand_by_file)})
                line_used = line_used or keep
        else:
            out.append({"span": span, "line": line})
    covered = {f for s in out for f in (s.get("span") or [])}
    for f in [f for f in surviving if f not in covered]:
        pos = order[f]
        idx = len(out)
        for i, s in enumerate(out):
            sp = s.get("span") or []
            if sp and order.get(sp[0], 10 ** 9) > pos:
                idx = i
                break
        out.insert(idx, {"span": [f],
                         "line": _grounded_pad_line(f, understand_by_file)})
    # system cards speak their text (a describing or padded line is replaced)
    for s in out:
        span = list(s.get("span") or [])
        if len(span) == 1 and str(kinds.get(span[0]) or "").lower() == "system":
            s["line"] = system_card_line(span[0], understand_by_file, s.get("line"))
    return out


def segments_from_sentences(sentences, surviving, kinds, u_by_file=None):
    """Prose-first (2026-07-03): derive adaptive segments from a beat written
    as ONE connected passage plus panel-tagged sentences. The model authors
    prose — the grouped-era deliverable — and ALL span structure is computed
    here, deterministically:
      - the earliest sentence to tag a story panel owns it; a tag that would
        move ownership backwards is absorbed into the running span (owners
        never regress, so spans stay in reading order);
      - untagged story panels ride the previous sentence's span (leading
        ones ride the first) — the voice keeps speaking while the art
        advances, exactly the grouped-era pacing;
      - an untagged sentence rides the previous tagged sentence's segment
        (its text joins that line; leading untagged text joins the first);
      - system panels are always solo: a sentence tagging EXACTLY that card
        (and nothing else) supplies its line, otherwise a grounded pad;
      - a run over SPAN_CAP keeps its line on the first SPAN_CAP panels and
        the overflow rides the NEXT sentence's span (pads only at the tail).
    Returns None when nothing usable came back — no story-panel tags, or one
    lone sentence claiming more story panels than a span may hold — so the
    caller re-asks, then falls back. Pure; never mutates its inputs."""
    files = [f for f in (surviving or []) if f]
    if not files:
        return None
    kinds = kinds or {}
    idx = {f: i for i, f in enumerate(files)}
    is_sys = [str(kinds.get(f) or "").lower() == "system" for f in files]
    n_story = sum(1 for s in is_sys if not s)

    sents: List[Dict[str, Any]] = []
    for s in (sentences or []):
        if not isinstance(s, dict):
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        story_tags, sys_tags = set(), set()
        for p in (s.get("panels") or []):
            j = idx.get(os.path.basename(str(p or "").strip()))
            if j is None:
                continue
            (sys_tags if is_sys[j] else story_tags).add(j)
        sents.append({"text": text, "tags": sorted(story_tags),
                      "sys": sorted(sys_tags)})
    # sentence-integrity rejoin (2026-07-06, class C): a "sentence" the model
    # split mid-thought (no terminal punctuation) is HALF a sentence — fold
    # the next one back into it (text joined, tags unioned) so no derived
    # line can dangle mid-clause. A dangling FINAL sentence has no neighbor
    # to rejoin; the fragment net (repair_spoken_fragments) amputates its
    # stub and the truncated_line QA flag heals what survives.
    rejoined: List[Dict[str, Any]] = []
    for s in sents:
        pure_sys = not s["tags"] and len(s["sys"]) == 1
        prev_pure_sys = bool(rejoined) and not rejoined[-1]["tags"] and (
            len(rejoined[-1]["sys"]) == 1)
        # never fold a system card's OWN line into a dangling predecessor (the
        # card would lose its dedicated line to a pad) — and never fold the
        # NEXT sentence into a dangling system-card line either (the card
        # would swallow a story sentence, leaving an unpunctuated
        # concatenation on the card and no line at all for the story panel).
        # System solos are a wall in both directions; a dangling pure-sys
        # sentence is left for the fragment net to period-close on its own.
        if (rejoined and not ends_terminal(rejoined[-1]["text"])
                and not pure_sys and not prev_pure_sys):
            prev = rejoined[-1]
            prev["text"] = (prev["text"] + " " + s["text"]).strip()
            prev["tags"] = sorted(set(prev["tags"]) | set(s["tags"]))
            prev["sys"] = sorted(set(prev["sys"]) | set(s["sys"]))
        else:
            rejoined.append(s)
    sents = rejoined
    if not sents:
        return None

    # a sentence tagging EXACTLY one system card (and no story panel) IS that
    # card's line; it takes no further part in the prose folding
    sys_line: Dict[int, str] = {}
    body: List[Dict[str, Any]] = []
    for s in sents:
        if not s["tags"] and len(s["sys"]) == 1 and s["sys"][0] not in sys_line:
            sys_line[s["sys"][0]] = s["text"]
        else:
            body.append(s)

    tagged = [s for s in body if s["tags"]]
    if n_story and not tagged:
        return None
    if not n_story and not sys_line:
        return None
    if len(tagged) == 1 and n_story > SPAN_CAP:
        # one lone sentence would own EVERY story panel — beyond what a span
        # may hold, and pads would eat the beat. Re-ask for a real split.
        return None

    # ownership: earliest tagger wins; then owners never regress
    owner: List[Optional[int]] = [None] * len(files)
    for si, s in enumerate(body):
        for j in s["tags"]:
            if owner[j] is None:
                owner[j] = si
    run = -1
    for j in range(len(files)):
        if is_sys[j] or owner[j] is None:
            continue
        owner[j] = max(owner[j], run)
        run = owner[j]
    prev: Optional[int] = None                    # untagged panels ride left…
    for j in range(len(files)):
        if is_sys[j]:
            continue
        if owner[j] is None:
            owner[j] = prev
        else:
            prev = owner[j]
    nxt: Optional[int] = None                     # …leading ones ride right
    for j in range(len(files) - 1, -1, -1):
        if is_sys[j]:
            continue
        if owner[j] is None:
            owner[j] = nxt
        else:
            nxt = owner[j]

    # each owning sentence's segment text: its own sentence plus every
    # following non-owning sentence up to the next owner, in passage order
    owners_used = {o for o in owner if o is not None}
    line_parts: Dict[int, List[str]] = {o: [] for o in owners_used}
    pending: List[str] = []
    cur: Optional[int] = None
    for si, s in enumerate(body):
        if si in line_parts:
            cur = si
            if pending:
                line_parts[cur].extend(pending)
                pending = []
            line_parts[cur].append(s["text"])
        elif cur is not None:
            line_parts[cur].append(s["text"])
        else:
            pending.append(s["text"])

    def _pad(j: int) -> str:
        return _grounded_pad_line(files[j], u_by_file)

    segs: List[Dict[str, Any]] = []
    used: set = set()                             # a line is voiced ONCE
    carry: List[int] = []                         # cap overflow -> next span
    j = 0
    while j < len(files):
        if is_sys[j]:
            for c in carry:                       # overflow never crosses a card
                segs.append({"span": [files[c]], "line": _pad(c)})
            carry = []
            segs.append({"span": [files[j]],
                         "line": sys_line.get(j) or _pad(j)})
            j += 1
            continue
        o = owner[j]
        k = j
        while k < len(files) and not is_sys[k] and owner[k] == o:
            k += 1
        run_idx = carry + list(range(j, k))
        carry = []
        text = ("" if o is None or o in used
                else " ".join(line_parts.get(o) or []).strip())
        if text:
            used.add(o)
        if len(run_idx) > SPAN_CAP:
            head, rest = run_idx[:SPAN_CAP], run_idx[SPAN_CAP:]
            segs.append({"span": [files[x] for x in head],
                         "line": text or _pad(head[0])})
            if k < len(files) and not is_sys[k]:
                carry = rest                      # ride the next sentence's span
            else:
                for x in rest:
                    segs.append({"span": [files[x]], "line": _pad(x)})
        elif run_idx:
            segs.append({"span": [files[x] for x in run_idx],
                         "line": text or _pad(run_idx[0])})
        j = k
    return segs


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
            keyed[f] = _grounded_pad_line(f, understand_by_file)
    out = [{"scene_file": f, "line": keyed[f]} for f in files]
    if leftover and out:                  # fold any remaining overflow into the last panel
        out[-1]["line"] = (out[-1]["line"] + " " + " ".join(leftover)).strip()
    return out


def glue_echo_spans(segs, echo_of, surviving):
    """An echo panel (a zoom re-frame of its original — story_group
    echo_pairs) always rides its ORIGINAL's span so the pair is voiced as ONE
    moment, never two events. Deterministic: only the safe adjacent case is
    glued (the echo directly follows its original in reading order, so the
    consecutive-partition invariant is preserved by construction); a
    non-adjacent echo is left alone (render_prep's ken echo restyle still
    covers presentation). If the echo owned a whole segment, that segment's
    line JOINS the original's line — words are never dropped. Pure."""
    if not echo_of or not segs:
        return segs
    pos = {f: i for i, f in enumerate([f for f in (surviving or []) if f])}
    out = [dict(s, span=[str(f) for f in (s.get("span") or []) if f])
           for s in segs if isinstance(s, dict)]
    for later, earlier in echo_of.items():
        if pos.get(later) is None or pos.get(earlier) is None:
            continue
        if pos[later] != pos[earlier] + 1:
            continue                    # non-adjacent: not safely glueable
        ie = il = None
        for i, s in enumerate(out):
            if earlier in s["span"]:
                ie = i
            if later in s["span"]:
                il = i
        if ie is None or il is None or ie == il:
            continue
        out[il]["span"] = [f for f in out[il]["span"] if f != later]
        out[ie]["span"].append(later)
        if not out[il]["span"]:
            extra = str(out[il].get("line") or "").strip()
            if extra:
                out[ie]["line"] = (str(out[ie].get("line") or "").strip()
                                   + " " + extra).strip()
            del out[il]
    return out


# Deterministic identity gate — extracted to identity_gate.py (2026-07-20
# story-state wave) so narration_punchup's post-punchup backstop can re-run
# the same gate after the persona rewrite. Re-exported here for callers/tests.
from identity_gate import (  # noqa: E402
    _PROT_HANDLE_RE,
    _figure_handle,
    _neutral_from_evidence,
    enforce_actor_handles,
)


# a voiced line must END: terminal punctuation (or an ellipsis / dash
# cliffhanger), optionally followed by closing quotes/brackets
_LINE_END_RE = re.compile(r"""(?:[.!?…]|\.\.\.|[—–-])[\s"'”’)\]]*$""")


def validate_segments(segments, scene_files, kinds, wpm: float = WPM,
                      echo_of=None) -> List[str]:
    """Deterministic guardrails for adaptive flow segments — pure, no LLM.

    Returns human-readable errors ([] = valid) so a failing beat can be
    re-asked with the exact problems appended to the prompt:
      1. spans partition scene_files EXACTLY, in reading order (no skip,
         overlap, or unknown file — the panel-collapse regression stays
         impossible);
      2. len(span) <= SPAN_CAP;
      3. a panel_kind == "system" file is always a SOLO span (cards keep their
         own clip; `kinds` maps scene_file -> panel_kind);
      4. duration-aware word budget: N*2.0s <= words/(wpm/60) <= N*6.0s per
         segment (N = span size) — reject too-thin AND too-fat lines;
      5. every line non-empty, no bracket-mood prefix (the packer adds moods)
         and no BARE mood-word prefix either (e.g. "Dramatic: He's…" — the
         round-3 leak; the packer adds moods, the writer never should).
    """
    errors: List[str] = []
    segs = segments if isinstance(segments, list) else []
    files = [f for f in (scene_files or []) if f]
    kinds = kinds or {}
    echo_of = echo_of or {}
    words_per_sec = float(wpm) / 60.0

    # belt-check: glue_echo_spans runs before validation, so a split ADJACENT
    # echo pair here is a logic bug (or a span-pinned regen trying to
    # re-split) — never a model style choice. Non-adjacent pairs mirror the
    # glue's skip (they can't be moved without breaking the consecutive
    # partition; render_prep's ken restyle covers their presentation).
    span_of = {f: i for i, seg in enumerate(segs) if isinstance(seg, dict)
               for f in (seg.get("span") or [])}
    pos = {f: i for i, f in enumerate(files)}
    for _later, _earlier in echo_of.items():
        if (pos.get(_later) is not None and pos.get(_earlier) is not None
                and pos[_later] == pos[_earlier] + 1
                and _later in span_of and _earlier in span_of
                and span_of[_later] != span_of[_earlier]):
            errors.append(f"echo pair split across segments: {_later} is a "
                          f"zoom re-frame of {_earlier} and must ride its "
                          "span (one voiced moment)")

    covered: List[str] = []
    for i, seg in enumerate(segs):
        span = ([str(f) for f in (seg.get("span") or []) if f]
                if isinstance(seg, dict) else [])
        line = (str(seg.get("line") or "").strip()
                if isinstance(seg, dict) else "")
        if not span:
            errors.append(f"segment {i}: empty span")
            continue
        covered.extend(span)
        n = len(span)
        # echo riders add no independent watch time — they don't count
        # against the cap (only relevant when the glue overflowed it)
        n_cap = sum(1 for f in span if f not in echo_of)
        if n_cap > SPAN_CAP:
            errors.append(f"segment {i}: span of {n_cap} panels exceeds the "
                          f"cap of {SPAN_CAP}")
        for f in span:
            if str(kinds.get(f) or "") == "system" and n > 1:
                errors.append(f"segment {i}: system panel {f} must be a "
                              "solo span")
        if not line:
            errors.append(f"segment {i}: empty line")
            continue
        if _MOOD_PREFIX_RE.match(line):
            errors.append(f"segment {i}: line must not start with a bracket "
                          "mood tag")
        if mentions_mood_tag_leak(line):
            leak_word = line.split()[0] if line.split() else ""
            errors.append(f"segment {i}: line opens with a bare mood/tone "
                          f"word ({leak_word!r}) followed by a fresh "
                          "sentence — that is a leaked label, never story; "
                          "drop it and start the line with the real sentence")
        if mentions_image_file(line):
            errors.append(f"segment {i}: line names an image file — file "
                          "names are tags, never narration; narrate what "
                          "HAPPENS across these panels instead")
        solo_card = (n == 1 and str(kinds.get(span[0]) or "") == "system")
        if not solo_card and not _LINE_END_RE.search(line):
            # (a system card's own text may lack a period — the fragment net
            # period-closes it later; story lines must END)
            errors.append(f"segment {i}: line ends mid-sentence "
                          f"({line[-40:]!r}) — the thought was cut off; finish "
                          "it on a complete clause")
        if mentions_impact_marker(line):
            errors.append(f"segment {i}: line echoes the impact-SFX bracket "
                          "marker verbatim — describe the strike/stab/blow "
                          "itself, never the bracket tag")
        if mentions_figures_leak(line):
            errors.append(f"segment {i}: line echoes the unresolved-figure "
                          "'unknown (...)' payload wrapper verbatim — use "
                          "neutral phrasing (the masked figure, the man in "
                          "the hood) instead, never the raw evidence text")
        if is_shot_description(line):
            # same detector prep_qa ERRORs on — enforcing it here turns a
            # multi-cycle heal burn into one cheap repair re-ask at source
            errors.append(f"segment {i}: line describes the artwork/camera/"
                          "a visual effect instead of the story — narrate "
                          "what happens and its consequence, never how the "
                          "panel is drawn")
        n_words = len(line.split())
        sec = n_words / words_per_sec
        if sec < n * _SEG_MIN_SEC_PER_PANEL:
            errors.append(
                f"segment {i}: too thin — {n_words} words (~{sec:.1f}s) cannot "
                f"hold {n} panel(s) on screen (needs >= "
                f"{n * _SEG_MIN_SEC_PER_PANEL:.0f}s of voice; add words or "
                "shrink the span)")
        elif sec > n * _SEG_MAX_SEC_PER_PANEL:
            # state the cap in WORDS — models follow an explicit word count
            # far more reliably than a seconds figure (2026-07-16: the
            # seconds-only phrasing re-asked into another fat line, and the
            # fallback then shipped it -> a 21s single-panel hold)
            max_words = int(n * _SEG_MAX_SEC_PER_PANEL * words_per_sec)
            errors.append(
                f"segment {i}: too fat — {n_words} words (~{sec:.1f}s) over "
                f"{n} panel(s); rewrite this line in AT MOST {max_words} "
                "words (or widen the span)")

    if covered != files:
        cov_set, file_set = set(covered), set(files)
        missing = [f for f in files if f not in cov_set]
        unknown = [f for f in covered if f not in file_set]
        dups = sorted({f for f in covered if covered.count(f) > 1})
        if missing:
            errors.append("spans skip panel(s): " + ", ".join(missing))
        if unknown:
            errors.append("spans name unknown panel(s): " + ", ".join(unknown))
        if dups:
            errors.append("spans repeat panel(s): " + ", ".join(dups))
        if not (missing or unknown or dups):
            errors.append("spans are out of reading order: "
                          + " -> ".join(covered) + " != " + " -> ".join(files))
    return errors


def _segment_repair_block(errors: List[str]) -> str:
    """The ONE repair re-ask: the exact validator errors appended to the prompt."""
    return (
        "\n\nSEGMENT REPAIR — your previous answer's segments were INVALID:\n  - "
        + "\n  - ".join(errors)
        + "\nRe-write the beat fixing EXACTLY these problems. The spans must "
          "cover every scene_file exactly once, in reading order, each span at "
          f"most {SPAN_CAP} panels, system cards solo, and each line sized to "
          "its span's word budget.\n")


def _prose_repair_block(errors: List[str]) -> str:
    """Repair re-ask for the prose-first shape: the derived segments (or the
    tagging itself) failed — ask for a fresh passage + tagged sentences."""
    return (
        "\n\nNARRATION REPAIR — your previous answer could not be used:\n  - "
        + "\n  - ".join(errors)
        + "\nRe-write the beat fixing EXACTLY these problems: one connected "
          "'narration' passage, then the SAME passage split into 'sentences' "
          "in order, each tagged with the 1-4 CONSECUTIVE scene_file(s) it "
          "speaks over. Tag every scene_file under some sentence; give a "
          "system/notification card its own short sentence; size each "
          "sentence to the screen time its panels deserve.\n")


def _pinned_span_block(segments: List[Dict[str, Any]]) -> str:
    """Correction-regen prompt block when the existing beat carries native
    segments (spec 3.5): the spans are FIXED — a re-split would renumber the
    sibling segment_ids (g####_p##) and churn the per-clip TTS cache
    (audio_stale). The writer rewrites LINES within the locked spans; only a
    full beats re-run (no --resume) may change spans."""
    rows = [f"  segment {i + 1} covers: {', '.join(s['span'])}"
            for i, s in enumerate(segments)]
    return (
        "\n\nFIXED SEGMENTATION FOR THIS REWRITE — this beat's segments are "
        "LOCKED (each segment is already voiced as its own cached clip):\n"
        + "\n".join(rows) + "\n"
        f"Return EXACTLY {len(segments)} segments with EXACTLY these spans, "
        "in this order — rewrite only each segment's line. Never merge, "
        "split, reorder, or drop a span.\n")


def enforce_pinned_spans(beat, prev_beat, gid):
    """Span-pin gate for a corrections regen (spec 3.5): when the previous
    beat carries native segments, the regenerated beat must carry EXACTLY the
    same spans (same partition, same order). A compliant rewrite (same spans,
    new lines) is adopted — returns the fresh beat. ANY re-split — including
    the singleton fallback — is rejected: returns the PREVIOUS beat unchanged
    (logged), so the whole manifest stays self-consistent and no segment_id is
    ever renumbered by a heal. Lines may change; spans may not."""
    pinned = (beat_segments(prev_beat)
              if has_native_segments(prev_beat) else [])
    if not pinned:
        return beat
    new_spans = [s["span"] for s in beat_segments(beat)]
    if new_spans == [s["span"] for s in pinned]:
        return beat
    print(f"[segments] span-pin g{gid:04d}: regen re-split "
          f"({len(new_spans)} vs {len(pinned)} segments) — kept previous "
          "lines (spans are locked under --corrections)")
    return prev_beat


def finalize_adaptive_beat(beat, surviving, kinds, u_by_file, gid,
                           reask_fn=None, allow_flow_nudge=True,
                           derive_fn=None, allow_span_align=True,
                           echo_of=None):
    """Adaptive mode: normalize + validate the model's segments; on failure do
    ONE repair re-ask (reask_fn(errors) -> repaired beat or None); still failing
    -> fall back to align_panel_narration singleton spans (never block the
    chapter; log `[segments] fallback beat gNNNN`).

    derive_fn(beat) -> raw segments list, or None/[] when the answer carries
    nothing usable. Default reads the beat's native segments; prose-first
    passes a segments_from_sentences closure. An unusable answer goes to the
    repair re-ask and then the FALLBACK path — never to a silent all-pad
    auto-repair (an all-pad "valid" beat could slip past a span pin whose
    spans are all-singleton, the exact d953fe4 poisoning).

    A VALID all-singleton answer on a >=4-panel beat gets ONE flow-nudge
    re-ask (the observed gemma failure: caption slideshow, zero spans); the
    nudged answer is adopted only when it validates AND actually flows.
    Disabled on span-pinned regens (allow_flow_nudge=False) — a nudge re-split
    would only be rejected by the pin — and in prose mode (the passage is
    flow-authored; span sizes are derived bookkeeping there).

    Writes beat['segments'] (spans partition `surviving` — the adaptive-mode
    cover assert), drops panel_narration + prose scaffolding (`sentences`),
    and rebuilds beat['narration'] as the ordered join of segment lines —
    LOAD-BEARING: caption_unvoiced / narration_stale / alignment QA and
    punchup key on it.
    """
    _derive = derive_fn or beat_segments
    echo_of = echo_of or {}

    def _norm(raw_segs):
        repaired = auto_repair_segments(raw_segs, surviving, kinds, u_by_file)
        return glue_echo_spans(repaired, echo_of, surviving)

    def _check(s):
        return validate_segments(s, surviving, kinds, echo_of=echo_of)

    raw = _derive(beat)
    if raw:
        segs = _norm(raw)
        errors = _check(segs)
    else:
        segs = []
        errors = ["the answer carried no usable narration shape (no valid "
                  "segments / panel-tagged sentences)"]
    if errors and reask_fn is not None:
        repaired = reask_fn(errors)
        if isinstance(repaired, dict):
            raw2 = _derive(repaired)
            if raw2:
                segs2 = _norm(raw2)
                if not _check(segs2):
                    segs, errors = segs2, []
    elif (not errors and allow_flow_nudge and reask_fn is not None
          and len(surviving) >= _FLOW_NUDGE_MIN_PANELS
          and all(len(s["span"]) == 1 for s in segs)):
        nudged = reask_fn([_flow_nudge_note(len(surviving))])
        if isinstance(nudged, dict):
            raw3 = _derive(nudged)
            segs3 = _norm(raw3) if raw3 else []
            if (segs3 and not _check(segs3)
                    and any(len(s["span"]) > 1 for s in segs3)):
                print(f"[segments] flow-nudge beat g{gid:04d} adopted "
                      f"({len(segs3)} segments)")
                segs = segs3
    if errors:
        print(f"[segments] fallback beat g{gid:04d} -> singleton spans "
              f"({errors[0]})")
        # Reuse whatever lines the model DID give as positional material;
        # align_panel_narration keys/fills/pads to exactly one line per panel.
        model_panels = [{"scene_file": (s.get("span") or [""])[0],
                         "line": s.get("line")} for s in segs]
        if not model_panels:
            # prose answers keep their sentence texts as positional material
            model_panels = [
                {"scene_file": "", "line": str((x or {}).get("text") or "")}
                for x in (beat.get("sentences") or []) if isinstance(x, dict)]
        aligned = align_panel_narration(surviving, model_panels, u_by_file)
        segs = glue_echo_spans(
            [{"span": [p["scene_file"]], "line": p["line"]} for p in aligned],
            echo_of, surviving)
        # Marker for the corrections caller: a pad-heavy fallback must NEVER
        # replace pinned prose — when the pin was itself all-singleton the
        # span comparison alone can't tell fallback pads from a real rewrite
        # (this poisoned 6 healed ch1 beats with "The moment holds.").
        beat["_segments_fallback"] = True
    if not errors and allow_span_align:
        # ONE-PANEL OFFSET post-pass (2026-07-06 review, dominant class): fix
        # a line leading/lagging its art by one panel by shifting span
        # boundaries — conservative margin, invariants revalidated by the
        # SAME validator, and OFF on span-pinned regens (allow_span_align is
        # False there: a shifted span would only be rejected by the pin).
        aligned, shifts = span_align_pass(
            segs, surviving, kinds, u_by_file, span_cap=SPAN_CAP,
            validate=_check)
        if shifts:
            for msg in shifts:
                print(f"[span_align] g{gid:04d}: {msg}")
            segs = aligned
    beat.pop("panel_narration", None)
    beat.pop("sentences", None)   # authoring scaffolding — segments are the contract
    beat["segments"] = segs
    covered = [f for s in segs for f in s["span"]]
    if covered != list(surviving):     # postcondition — must survive -O
        raise RuntimeError(
            f"segments/scene_files cover mismatch in group {gid}")
    beat["narration"] = (" ".join(s["line"] for s in segs).strip()
                         or beat.get("narration", ""))
    return beat


def _append_niche(system, niche="", niche_secondary=""):
    """Append the per-series niche TEMPERATURE block; no-op when no niche is set."""
    blk = register_block(niche, niche_secondary)
    return system + ("\n\n" + blk if blk else "")


# The narration-shape instruction is the ONLY part of the system prompt that
# differs between segmentation modes; every persona/grounding/caption rule
# below it is shared. _PER_PANEL_NARRATION_INSTRUCTION is byte-identical to the
# pre-segments prompt so per_panel mode stays a true escape hatch.
_PER_PANEL_NARRATION_INSTRUCTION = (
    "For EACH file in scene_files, in order, WRITE ONE narration line in "
    "'panel_narration' as {scene_file, line}. Give EVERY panel its own line — "
    "a quick action panel gets a punchy phrase, a pivotal/quiet panel gets a "
    "fuller cinematic sentence; match length to what the panel shows. The lines "
    "must FLOW as one continuous story (continue from previous_narration), not "
    "isolated captions. Then set 'narration' to all the lines joined with a space.\n"
)

def _max_words(n: int) -> int:
    return int(n * _SEG_MAX_SEC_PER_PANEL * (WPM / 60.0))


# the validator's exact arithmetic, spelled out for the model's FIRST draft
_WORD_CAP_RULE = (
    "HARD LENGTH CAP (screen time — a sentence over its cap FAILS validation "
    "and costs a rewrite): a sentence may carry AT MOST "
    f"{_max_words(1)} words PER TAGGED PANEL — one panel <={_max_words(1)}, "
    f"two <={_max_words(2)}, three <={_max_words(3)}, four <={_max_words(4)} "
    "words. Count your words before answering. The taste target sits far "
    "below the cap: a solo moment ~5-13 words, a run ~10-15 words per panel.\n"
)

_ADAPTIVE_NARRATION_INSTRUCTION = (
    "You are the NARRATOR telling this story aloud — never a caption writer. "
    "The per-panel descriptions are RAW MATERIAL, not lines: echoing or "
    "rephrasing one is FORBIDDEN, and openers like 'The character…' / 'The "
    "scene shows…' are BANNED — name people (cast or persona handles) and "
    "narrate stakes, momentum, consequence.\n"
    "Write 'segments': an ORDERED list of {span, line}. A span = 1-4 "
    "CONSECUTIVE scene_files; its line is voiced as ONE clip over those "
    "panels. Every scene_file appears in EXACTLY ONE span, in order — never "
    "skip, repeat, or reorder.\n"
    "DEFAULT TO FLOW: group panels carrying one action, traversal, "
    "progression, or caption-run into a 2-4 panel span with ONE flowing "
    "passage (clauses lean across panels; end mid-momentum, not mid-word). A "
    "typical 5-8 panel beat = 2-4 segments of mixed sizes.\n"
    "SOLO is the earned exception — a close-up, reveal, punchline, "
    "dialogue-heavy panel, or system card. A beat of nothing but solos is a "
    "slideshow, not narration.\n"
    "Never enumerate panels — 'in the next panel' and the like are BANNED.\n"
    "WORD BUDGET (the voice carries the span's screen time): solo ≈5-13 "
    "words; 2-panel ≈10-26; 3-panel ≈25-40; 4-panel ≈30-50. Never a thin "
    "line stretched over panels, never a bloated line parked on one. "
    + _WORD_CAP_RULE
    + "Lines FLOW as one continuous story (continue from previous_narration). "
    "Set 'narration' to the segment lines joined with a space.\n"
)

# Prose-first free generation (2026-07-03) — restores the grouped-era voice
# the user approved on 2026-06-16: the writer authors ONE connected passage
# per beat (narration flows, panels pace underneath) and tags each sentence
# with the panel(s) it rides over; segments_from_sentences() then derives the
# span partition DETERMINISTICALLY. Asking gemma to invent {span, line}
# partitions while writing prose produced caption slideshows (ch1 2026-07-03:
# 67% singletons, "He's definitely not happy about that.") across three
# prompt-level fixes — segmentation is bookkeeping, so it lives in code now.
# _ADAPTIVE_NARRATION_INSTRUCTION stays for span-PINNED correction regens,
# where the model must think in segments (locked spans, lines rewritten).
_PROSE_NARRATION_INSTRUCTION = (
    "You are the NARRATOR telling this story aloud — never a caption writer. "
    "The per-panel descriptions are RAW MATERIAL, not lines: echoing or "
    "rephrasing one is FORBIDDEN, and openers like 'The character…' / 'The "
    "scene shows…' are BANNED — name people (cast or persona handles) and "
    "narrate stakes, momentum, consequence.\n"
    "Write 'narration' FIRST: ONE connected passage — the exact sentences a "
    "narrator SPEAKS over these panels, walking the beat's moments IN ORDER "
    "as one continuous story. Let each moment's CONTENT set its share: when "
    "several CONSECUTIVE panels show one continuous action, traversal, or "
    "progression (a fall, a chase, a charge, a caption run), write ONE "
    "fuller sentence that carries the WHOLE run — never slice it into one "
    "thin sentence per panel; a moment that lands alone (a close-up, a "
    "reveal, a punchline) earns its own tight sentence — never a list of "
    "captions, never 'in the next panel'. Cover every panel's moment inside "
    "the passage, weaving captions and dialogue in as you go.\n"
    "THEN split that same passage into 'sentences': one entry per sentence, "
    "in order, text EXACTLY as written in the passage. Tag each entry with "
    "'panels': ALL the scene_file(s) that sentence speaks over, in reading "
    "order — a run-carrying sentence tags EACH of its 2-4 files; a "
    "solo-moment sentence tags one. Tag every scene_file under some "
    "sentence; give a system/notification card its own short sentence. File "
    "names belong ONLY in 'panels' — NEVER in the narration or a sentence's "
    "text.\n"
    "A panel that is JUST a speech bubble or caption on a plain background "
    "(panel_kind 'caption') is TEXT, not a picture: WEAVE its words into the "
    "sentence of the nearest drawn panel — never write a standalone sentence "
    "about a bubble alone.\n"
    # HARD numeric caps in the PRIMARY prompt (2026-07-17): the caps only
    # lived in the rejection note, so 15/25 groups paid a full second model
    # call to learn them (~9 min/chapter). Derived from the same constants
    # validate_segments enforces — they cannot drift.
    + _WORD_CAP_RULE
)

# A structurally-VALID all-singleton answer on a big beat is the observed
# gemma failure mode (it mirrors the per-panel input listing and parrots the
# descriptions — "valid garbage", zero flow). One targeted nudge re-ask; the
# model may insist (some beats ARE legitimately all-solo), then we accept.
_FLOW_NUDGE_MIN_PANELS = 4


def _flow_nudge_note(n_panels: int) -> str:
    return (f"All {n_panels} panels came back as isolated single-panel captions "
            "— that reads as a slideshow, not narration. Combine the panels "
            "that carry one continuous action/progression into 2-4 panel FLOW "
            "spans with ONE flowing passage each (per the FLOW criteria); keep "
            "solo only for moments that land harder alone.")


def _default_segmentation() -> str:
    """Tool default for --segmentation: env STUDIO_NARR_SEGMENTATION wins when
    valid, else 'adaptive'. argparse validates `choices` only for CLI-provided
    values, so a garbage env var must be normalized here."""
    v = (os.environ.get("STUDIO_NARR_SEGMENTATION") or "").strip().lower()
    return v if v in ("adaptive", "per_panel") else "adaptive"


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
    ap.add_argument("--ledger", default="",
                    help="Optional manifest.ledger.json (chapter story-state: "
                         "arbitrated action directions, deaths, banned handles, "
                         "answered questions) — per-beat FACTS block + gate")
    ap.add_argument("--niche", default="")
    ap.add_argument("--niche-secondary", default="")
    ap.add_argument("--segmentation", choices=["adaptive", "per_panel"],
                    default=_default_segmentation(),
                    help="adaptive = flow segments spanning 1-4 panels voiced "
                         "as one clip each (spec 2026-07-02); per_panel = the "
                         "legacy 1-line-per-panel path, byte-compatible")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    groups_m = load_json(args.groups_manifest)
    vision_m = load_json(args.vision_manifest)
    understood_m = load_json(args.understood) if args.understood and os.path.exists(args.understood) else {}
    u_by_file = {p.get("scene_file"): p for p in (understood_m.get("panels") or []) if p.get("scene_file")}
    # Story-state ledger — fail-soft: {} keeps every downstream byte-compatible
    ledger_m = (load_json(args.ledger)
                if args.ledger and os.path.exists(args.ledger) else {})

    groups = _read_groups(groups_m)
    if not groups:
        raise SystemExit("No groups/shots found (expected key: shots or groups)")
    # zoom/echo pairs from story_group: {later: earlier} — the later panel is
    # the SAME artwork re-framed; it rides the original's span, one voiced line
    echo_of = {str(b): str(a)
               for a, b in (groups_m.get("echo_pairs") or []) if a and b}

    vision_by_file = _build_vision_map(vision_m)

    if args.backend == "ollama":
        client = None
        args.model = args.ollama_model
    else:
        if not args.project or not args.location:
            raise SystemExit("--project/--location are required for --backend vertex")
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)

    system_body = (
        "You are a YouTube manhwa recap story editor.\n"
        "Given consecutive scene images + OCR, produce ONE structured beat for that group.\n"
        "Be faithful to visible content.\n"
        "Avoid excessive poetic language.\n"
        "End with a strong hook line.\n"
        "Rendering hints: avoid zooming into text bubbles; focus faces/hands/key objects/wide.\n"
        "\n"
        "{NARR_INSTRUCTION}"
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
        "      NEVER upgrade specificity beyond the understanding: 'stained' stays stained (not\n"
        "      blood), 'a dark shape' stays a shape. A panel marked 'uncertain' is one the analyst\n"
        "      could not identify — narrate it just as vaguely or fold it into the surrounding\n"
        "      motion; NEVER give an uncertain subject an action, attacker, weapon, or identity.\n"
        "      ECHO PANELS: a panel marked 'echo_of' repeats that earlier panel's art re-framed\n"
        "      (an artist zoom for emphasis). It is the SAME single moment — write ONE flowing\n"
        "      passage covering both; never introduce the echo as a new event or give it its own\n"
        "      sentence.\n"
        "    - IDENTITY + NAMES: NAME established CHAPTER CAST members so the audience can\n"
        "      follow who is who — recognition is the priority. NAME the protagonist (or a\n"
        "      relaxed stand-in like 'our guy') normally on HIS OWN panels, even when a\n"
        "      separate mysterious figure is on screen nearby. 'Our guy' / 'our boy' /\n"
        "      'our man' refers to EXACTLY ONE person — the protagonist. NEVER use it for\n"
        "      any other character: a helper, ally, stranger, or rescuer gets their cast\n"
        "      name or a neutral handle, no matter how sympathetic they are. Reserve a grounded NEUTRAL\n"
        "      handle ('the stranger', 'the intruder') ONLY for a figure THIS panel itself\n"
        "      presents as genuinely concealed — transformed, masked, hooded, glowing,\n"
        "      silhouetted, disguised, or newly-arrived (e.g. 'gear unlike anything') — and not\n"
        "      yet matched to a known character. Do NOT neutralize an ESTABLISHED character\n"
        "      just because a concealed figure appears, and do NOT keep calling a clearly-shown,\n"
        "      already-known character 'the stranger'. A power/transformation reveal of an\n"
        "      UNKNOWN figure is a mystery to preserve — but once the story's own text or the\n"
        "      character's established look identifies someone, use their name. Once introduced,\n"
        "      ration the protagonist's real name and usually use pronouns or a relaxed stand-in.\n"
        "    - FIGURES ARE GROUND TRUTH: when a panel's scenes_signals entry carries a\n"
        "      'figures' list, those are the characters ACTUALLY IN that panel, resolved from\n"
        "      the chapter cast's appearance. Name actors ONLY from that list: NEVER attribute\n"
        "      an action, weapon, wound, or thought to a cast member the panel's figures do not\n"
        "      include (if figures says the prince, it is the prince drawing the blade — not an\n"
        "      assassin). When a figure is 'unknown (…)', use neutral phrasing (the masked\n"
        "      figure, the man in the hood) — never guess a name for an unknown.\n"
        "    - FACTS ARE THE CHAPTER RECORD: when INPUT_JSON.facts exists it is the chapter's\n"
        "      established story record, reconciled against the DIALOGUE (the highest-trust\n"
        "      evidence). When facts and a panel's visual description disagree about WHO does\n"
        "      WHAT to WHOM, the FACTS WIN — narrate the [dialogue-arbitrated] direction from\n"
        "      facts.actions, never the panel's guess. NEVER narrate an action by anyone in\n"
        "      facts.dead_by_now (they are already dead at this beat); NEVER use a handle in\n"
        "      facts.banned_handles (e.g. 'the leader' after the leader died — a surviving\n"
        "      underling does NOT inherit the title); NEVER re-ask a question facts.answered\n"
        "      already resolves — build on the answer instead.\n"
        "    - WHO IS SPEAKING: a panel marked 'dialogue_voice' carries FIRST-PERSON\n"
        "      speech — the words are ABOUT THE SPEAKER, and manhwa often draws the\n"
        "      LISTENER reacting instead. Attribute such a line using facts/figures and\n"
        "      the surrounding story, NEVER automatically to the figure drawn in that\n"
        "      panel. A dying character's 'my sight is fading' belongs to the dying\n"
        "      character even when someone else is on screen.\n"
        "    - DIALOGUE — quote selectively, recap-style: PARAPHRASE the bulk into narration but\n"
        "      DO quote occasionally for impact. QUOTE a SHORT (<=6 words), COMPLETE, punchy real\n"
        "      line (a threat, a name, a key line) in clean sentence case, attributed — e.g. he\n"
        "      mutters 'I can't move.', she spits 'Damn you.'. A few such quotes per chapter land\n"
        "      hard; paraphrase everything else. Do NOT quote a whole long bubble; NEVER stack two\n"
        "      long quotes in a row. Good: the attackers sneer that his lineage changes\n"
        "      nothing -> a painless death. inner_thought -> render as the character's thought (at\n"
        "      most one short quote). NEVER quote UI text/watermarks/counters/sound-effects, raw\n"
        "      ALL-CAPS/garbled OCR, or a stub that trails off mid-word on an\n"
        "      ellipsis — only real,\n"
        "      complete, sentence-case character speech.\n"
        "    - ACTION beats (a fight, a knife drawn, a strike — few words, lots of motion) are the\n"
        "      CLIMAX: describe the PHYSICAL action vividly and grounded — who draws/strikes/dodges\n"
        "      what, and the stakes (e.g. 'he finally rips his hidden blade free to defend\n"
        "      himself'). Do NOT skip them or retreat into vague atmosphere.\n"
        "    - Present tense, active voice; cinematic but accurate. NEVER name the\n"
        "      shot/camera/panel/image/frame; NEVER begin 'A close-up shot shows...'\n"
        "      or 'The panel shows...'. Narrate the STORY, not the picture.\n"
        "    - RENDERING IS NOT STORY: narrate the ACTION and its impact/stakes,\n"
        "      never HOW the panel is DRAWN. NEVER describe visual effects or\n"
        "      rendering — no 'motion blur', 'speed lines', 'blurry streaks',\n"
        "      'creating ... effects', 'is depicted', 'the panel/image shows'. For\n"
        "      an action/motion panel (a strike, a dash, an impact), say WHAT\n"
        "      happens and the consequence (who strikes whom, the force, the\n"
        "      result) — e.g. 'He whips his blade around in a vicious arc' — not\n"
        "      'a sword is being swung with motion blur'.\n"
        "    - PUBLICATION CHROME: if a panel is a series cover, title/chapter-number card,\n"
        "      publisher or studio logo, app UI screen, or credits page — do NOT describe it.\n"
        "      Never narrate 'the chapter opens with...', view counts, or studio names.\n"
        "      Write the narration from the STORY panels only; if a group contains only\n"
        "      chrome, write a one-line bridge into the story instead.\n"
        "    - NARRATIVE CAPTIONS ARE NOT CHROME — a text-only panel or box with the\n"
        "      author's monologue / scene-setting / transition text — a retrospective\n"
        "      or scene-setting line in the narrator's own voice — is the\n"
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
        "    - SYSTEM CARDS SPEAK THEIR TEXT: when a panel is an in-world system/\n"
        "      notification card (panel_kind 'system'), voice the card's ACTUAL words —\n"
        "      verbatim or tightly paraphrased (read out what the card announces).\n"
        "      NEVER describe the card as an object or interface:\n"
        "      'a white panel appears with the text…', 'a panel displays…', 'text appears\n"
        "      on screen' are BANNED — say what the card SAYS, never how it is drawn.\n"
        "    - CONTINUITY: INPUT_JSON.previous_narration holds the line(s) the narrator\n"
        "      JUST SPOKE. Continue that flow: never re-introduce characters or\n"
        "      re-describe the setting already established, never start with the same\n"
        "      opening words as the previous line, and if the previous line ended\n"
        "      mid-thought, your first words must complete it.\n"
        "      BRIDGE REQUIREMENT: when previous_narration exists, your FIRST sentence\n"
        "      must CONNECT to it — open with a consequence, reaction, or contrast\n"
        "      ('But…', 'Before he can react…', 'That focus shatters when…', a pronoun\n"
        "      continuation) — and NEVER open cold with a scene reset ('The scene\n"
        "      shows…', 'In a dark ravine, a figure…', 'We see…'). The bridge is still a\n"
        "      complete, independently speakable sentence.\n"
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
    # Round-2 identity fix: deterministic panel→cast FIGURE resolution at the
    # writer seam (cast exists only from the beated stage — AFTER understanding
    # — so resolution happens at read time, tools/cast_identity.py; prep_qa's
    # actor_mismatch gate shares the same authority). {} without cast/understood.
    figures_by_file: Dict[str, List[Dict[str, str]]] = {}
    actor_nouns: Dict[str, Any] = {}
    protagonist_names: set = set()
    if cast_list and u_by_file:
        from cast_identity import actor_noun_map, resolve_figures_by_file
        from identity_gate import protagonist_names as _prot_names
        from identity_gate import spoken_names as _spoken_names
        # Ledger dead-sets: a killed entity must stop resolving on later
        # panels (the oracle fix — a dead leader kept claiming look-alike
        # assassin panels via the faction tie).
        excluded_by_file: Dict[str, set] = {}
        if ledger_m:
            from story_ledger import dead_sets_by_file
            excluded_by_file = dead_sets_by_file(
                ledger_m, [p.get("scene_file")
                           for p in (understood_m.get("panels") or [])
                           if p.get("scene_file")])
        figures_by_file = resolve_figures_by_file(
            understood_m, cast_list, excluded_by_file=excluded_by_file)
        actor_nouns = actor_noun_map(cast_list)
        protagonist_names = _prot_names(cast_list)
        spoken_map = _spoken_names(cast_list)
    story_block = _build_story_block(args.story)
    system_body = system_body.replace("{CAST_BLOCK}", cast_block)
    system_body = system_body.replace("{STORY_SPINE}", story_block)
    # Generator-side advertiser-safety rules ride the narration prompt so the
    # narration is brand-safe at the source; the sanitize-pass NET still runs
    # downstream regardless.
    system_body = (system_body + "\n\n" + SAFE_NARRATION_RULES + "\n\n"
                   + _DIALOGUE_RULE + "\n\n" + RECAP_STYLE_RULES)
    # resolve niche: explicit CLI args win; else read the episode manifest next to --out
    niche_p, niche_s = args.niche, args.niche_secondary
    if not niche_p:
        try:
            with open(os.path.join(os.path.dirname(args.out), "manifest.series.json"),
                      encoding="utf-8") as _f:
                _d = json.load(_f)
            niche_p = str(_d.get("niche_primary") or "")
            niche_s = str(_d.get("niche_secondary") or "")
        except Exception:
            niche_p, niche_s = "", ""
    system_body = _append_niche(system_body, niche_p, niche_s)

    # Free generation writes PROSE-FIRST under adaptive (one connected passage
    # + panel-tagged sentences; spans derived in code) or the legacy per-panel
    # shape; a span-PINNED correction regen keeps the direct-segments schema +
    # instruction (locked spans, lines rewritten) — the verified heal path.
    if args.segmentation == "per_panel":
        system_free = system_body.replace(
            "{NARR_INSTRUCTION}", _PER_PANEL_NARRATION_INSTRUCTION)
        schema_free = build_beat_schema("per_panel")
    else:
        system_free = system_body.replace(
            "{NARR_INSTRUCTION}", _PROSE_NARRATION_INSTRUCTION)
        schema_free = build_beat_schema("prose")
    system_pinned = system_body.replace(
        "{NARR_INSTRUCTION}", _ADAPTIVE_NARRATION_INSTRUCTION)
    schema_pinned = build_beat_schema("adaptive")

    corrections: Dict[int, str] = {}
    if args.corrections and os.path.exists(args.corrections):
        try:
            corrections = {int(k): str(v) for k, v in json.load(open(args.corrections)).items()}
        except Exception:
            corrections = {}

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

    # The span pin exists SOLELY to protect the per-clip TTS cache from
    # segment_id renumbering (spec 3.5). Before anything is voiced there is
    # nothing to protect — and holding a rewrite to exact span reproduction
    # made the heal loop 0-for-9 on real ch1 (every good rewrite discarded,
    # the flagged line kept). Corrections therefore pin only when the episode
    # carries a TTS index; unvoiced chapters heal through the free prose path.
    pin_spans = os.path.exists(os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "tts", "tts_index.json"))

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

        pin_prev: Optional[Dict[str, Any]] = None
        if gid in corrections and pin_spans:
            # Span-pinned heal (spec 3.5): when the beat being corrected
            # already carries native segments AND the episode is voiced, its
            # spans are FIXED — the rewrite may only change LINES (a re-split
            # would renumber the sibling segment_ids -> per-clip TTS cache
            # churn + audio_stale). Pinning derives from the EXISTING beat's
            # shape, never from --segmentation; without --resume there is no
            # existing beat, so a full beats re-run keeps freedom to re-split.
            prev = existing_by_id.get(gid)
            if (prev is not None and has_native_segments(prev)
                    and beat_segments(prev)):
                pin_prev = prev
        sys_g = system_pinned if pin_prev is not None else system_free
        schema_g = schema_pinned if pin_prev is not None else schema_free
        if gid in corrections:
            # The rewrite instruction must speak the ACTIVE schema: a pinned
            # regen returns segments[].line; free adaptive returns the prose
            # passage + tagged sentences — telling it to fix a field it does
            # not emit yields malformed answers → validation fallback → pads
            # (poisoned 6 healed ch1 beats).
            if pin_prev is not None:
                _fix_target = "every 'segments' line"
            elif args.segmentation == "per_panel":
                _fix_target = "the 'narration'"
            else:
                _fix_target = "the 'narration' passage and its tagged 'sentences'"
            sys_g = sys_g + (
                "\n\nCORRECTION FOR THIS GROUP — the previous narration had this problem:\n  "
                + corrections[gid] + "\n"
                f"Rewrite {_fix_target} to FIX it: stay strictly to what is visible here plus the "
                "panel's actual dialogue, COVER every on-panel caption in full, keep the cast names, "
                "assert nothing not shown, and never leave a line empty.\n"
            )
            if pin_prev is not None:
                sys_g += _pinned_span_block(beat_segments(pin_prev))
            regenerated += 1

        payload = _pack_group_payload(
            g, vision_by_file, u_by_file,
            figures_by_file=figures_by_file,
            echo_of=echo_of,
            ledger_facts=(ledger_m.get("beat_facts") or {}).get(
                f"g{gid:04d}") if ledger_m else None)
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
            beat_schema=schema_g,
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

        all_files = [f for f in (beat.get("scene_files")
                                 or payload["scene_files"]) if f]
        if args.segmentation == "per_panel":
            surviving = all_files          # legacy escape hatch: byte-compatible
            # Normalize panel_narration: exactly one line per surviving scene_file.
            # Runs on BOTH normal and fallback beats (the fallback has no panel_narration
            # so align_panel_narration will pad every panel from u_by_file / defaults).
            # We derive narration from the panel lines here, overwriting what the model
            # joined so the joined string stays in sync with the per-panel lines.
            # narration_plain (owned by the punchup stage) is NOT set.
            beat.pop("segments", None)
            beat["panel_narration"] = align_panel_narration(
                surviving, beat.get("panel_narration"), u_by_file)
            assert len(beat["panel_narration"]) == len(surviving), (
                f"panel_narration/scene_files mismatch in group {gid}")
            beat["narration"] = " ".join(p["line"] for p in beat["panel_narration"]).strip() or beat.get("narration", "")
        else:
            # Caption-only panels (pure speech bubbles: panel_kind == 'caption')
            # are TEXT, not visuals. The writer SEES them in the payload and
            # weaves their words in, but they must NEVER own a shown slot/clip:
            # after bubble-cleaning they are blank, so at render they junk-drop
            # and HOLD a neighbour (the 13.8s held-eye g0017) and each inflates
            # the solo count. Exclude them from the shown-panel partition; a
            # sentence that tagged only captions has no surviving tag, so the
            # splitter's untagged-fold routes its line into the adjacent visual
            # segment (the caption's words survive; its blank frame does not).
            # Guard: never empty the beat — a rare all-caption group keeps its
            # panels so something shows.
            _vis = [f for f in all_files
                    if str((u_by_file.get(f) or {}).get("panel_kind")
                            or "").lower() != "caption"]
            surviving = _vis if _vis else all_files
            beat["scene_files"] = surviving
            # Adaptive flow segments: validate the model's spans; ONE repair
            # re-ask with the exact errors; still failing -> singleton fallback
            # (mirrors the per_panel backfill — the chapter never blocks). A
            # parse-failed beat skips the re-ask: the model already exhausted
            # its retries, so go straight to the grounded singleton fallback.
            kinds = {f: str(((u_by_file.get(f) or {}).get("panel_kind")) or "")
                     for f in surviving}

            def _reask(errors: List[str]) -> Optional[Dict[str, Any]]:
                block = (_segment_repair_block(errors) if pin_prev is not None
                         else _prose_repair_block(errors))
                return _generate_beat_for_group(
                    client=client, model=args.model,
                    system_instruction=sys_g + block,
                    payload=payload, image_paths=img_paths,
                    beat_schema=schema_g, gid=gid, retries=0,
                    max_output_tokens=args.max_output_tokens,
                    backoff_max=args.backoff_max, backend=args.backend,
                    usage=usage)

            def _derive(b: Dict[str, Any]):
                # prose tags win when present; a pinned regen's answer (or a
                # legacy manifest) still counts via its native segments
                segs = segments_from_sentences(
                    b.get("sentences"), surviving, kinds, u_by_file)
                return segs if segs is not None else beat_segments(b)

            finalize_adaptive_beat(
                beat, surviving, kinds, u_by_file, gid,
                reask_fn=None if beat.get("error") else _reask,
                # the span-size nudge is direct-segments machinery: prose
                # authors flow in the passage itself, and a pinned nudge
                # re-split would only be rejected by enforce_pinned_spans
                allow_flow_nudge=False,
                derive_fn=_derive,
                # a span-pinned heal may change LINES only — an offset shift
                # would re-split and be rejected wholesale by the pin
                allow_span_align=pin_prev is None,
                echo_of=echo_of)

        # Corrections regen of a native-segments beat: adopt the rewrite ONLY
        # if it kept the pinned spans AND is a real rewrite; a validation
        # fallback (pad-heavy singletons) can accidentally MATCH an
        # all-singleton pin, so it must explicitly keep the previous beat.
        if pin_prev is not None:
            if beat.pop("_segments_fallback", False):
                print(f"[segments] span-pin g{gid:04d}: regen fell back to "
                      "pads — kept previous lines")
                beat = pin_prev
            else:
                beat = enforce_pinned_spans(beat, pin_prev, gid)
        else:
            fell_back = beat.pop("_segments_fallback", False)
            prev0 = existing_by_id.get(gid) if gid in corrections else None
            if fell_back and prev0 is not None:
                # UNPINNED corrections (unvoiced episode): a re-split rewrite
                # is welcome, but pads must never replace real lines — the
                # same poisoning family the pin guards against.
                print(f"[segments] corrections g{gid:04d}: regen fell back "
                      "to pads — kept previous lines")
                beat = prev0

        # The per-panel backfill above gives even a parse-failed beat valid lines;
        # demote the silencing `error` flag so those lines actually reach render.
        demote_backfilled_error(beat)

        # Resolve bracketed cast tokens PER LINE — the narration-string resolve
        # above is rebuilt away when the finalizers re-join the lines, so a
        # '[protagonist]' inside a segment line would otherwise reach the TTS.
        if cast_list:
            seg_lines = [s["line"] for s in beat_segments(beat)]
            fixed = [_resolve_cast_tokens(x, cast_list) for x in seg_lines]
            if fixed != seg_lines and all(x.strip() for x in fixed):
                write_segment_lines(beat, fixed)

        # Deterministic identity gate (2026-07-16): protagonist handles and
        # subject-position actor-nouns must match the span's resolved figures;
        # unambiguous mismatches are rewritten to what the panel shows.
        # Covers primary writes AND heal re-rolls (same code path); stamped
        # for the validation rewind's precision measurement.
        if figures_by_file and actor_nouns:
            rw = enforce_actor_handles(beat, figures_by_file, actor_nouns,
                                       protagonist_names,
                                       ledger=ledger_m or None,
                                       spoken=spoken_map)
            if rw:
                beat["actor_rewrites"] = rw
                for msg in rw:
                    print(f"[identity] g{gid:04d}: {msg}")

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
    write_manifest(args.out, out_obj, inputs=(args.groups_manifest, args.cast),
                   tool="gemini_narrative_pass")
    print(f"[ok] wrote={args.out} beats={len(beats_out)} parse_errors={parse_errors} regenerated={regenerated}")
    print(usage.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
