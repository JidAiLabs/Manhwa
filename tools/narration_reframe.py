#!/usr/bin/env python3
"""narration_reframe — the LLM REFRAME step of the advertiser-safety pass.

The regex sanitizer (narration_sanitize.py) splits risk into three actions:
  - replace : deterministic safe swaps (already applied in report.text)
  - flag    : context-sensitive — a dumb swap would break grammar/meaning, so
              the line needs a small LLM rewrite that softens per the note
  - block   : must not publish as-is — needs an LLM rewrite down to implication
              level, or the chapter HALTS

This module owns ONLY the flag/block rewrite. It builds a prompt from the
shared writing constraints (narration_safe_rules.SAFE_NARRATION_RULES) plus the
specific notes of the hits that fired on THIS line, and asks an injected model
to rewrite the single narration line keeping the SAME story meaning but clean.

The model call is INJECTED as ``call_fn`` so this is unit-testable with a stub
(no live model / network). The pipeline passes a ``call_fn`` that wraps
``gemini_narrative_pass._call_model_with_backoff`` with the resolved
ollama/Gemma (or Vertex) backend — see studio/pipeline.py _stage_scripted.

    call_fn(system: str, user_payload: dict, schema: dict, max_tokens: int) -> dict|None

It must return a parsed JSON object (or None on failure) matching ``schema``
(``{"narration": "<rewritten line>"}``). On any miss the ORIGINAL line is
returned unchanged so the re-sanitize step still gates it (a block that can't be
softened stays a block → recorded UNRESOLVED, never silently published).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from narration_safe_rules import SAFE_NARRATION_RULES  # noqa: E402


# call_fn(system, user_payload, schema, max_tokens) -> parsed dict | None
ReframeCallFn = Callable[[str, Dict[str, Any], Dict[str, Any], int], Optional[Dict[str, Any]]]

# Schema the reframe model must satisfy (kept tiny + permissive: a single
# rewritten line). Shaped like gemini_narrative_pass schemas (OBJECT/STRING) so
# the same Vertex/ollama call path validates it without translation.
REFRAME_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {"narration": {"type": "STRING"}},
    "required": ["narration"],
}

_REFRAME_SYSTEM_HEADER = (
    "You rewrite ONE manhwa-recap narration line so it is advertiser-safe for "
    "YouTube monetization, while keeping the SAME story meaning, the same "
    "characters, and the same beat. This is a softening rewrite, not a "
    "summary: do not add new events, do not drop the story point, do not change "
    "who does what. Return ONLY the rewritten line."
)

# Per-category guidance mirroring SAFE_NARRATION_RULES, used to AMPLIFY the
# hit-specific notes (so even a terse denylist note carries the full intent).
_CATEGORY_GUIDANCE: Dict[str, str] = {
    "sexual": (
        "Keep any sexual content at IMPLICATION level only — never explicit, no "
        "anatomy. Imply ('they crossed a line', 'spent the night', 'were "
        "intimate') or skip the beat."
    ),
    "self_harm_suicide": (
        "Frame self-harm/suicide carefully and NON-graphically; never name a "
        "method ('ended her own life', 'couldn't go on', 'chose to leave this "
        "world')."
    ),
    "violence": (
        "Soften violence: keep the consequence, drop the graphic method/gore "
        "('didn't survive', 'was taken out', 'the clan was wiped out')."
    ),
    "profanity": (
        "Paraphrase insults/profanity rather than voicing them; no slurs."
    ),
    "substance": (
        "Refer to 'substances'/'narcotics' generically; imply rather than detail."
    ),
    "slurs": (
        "Remove the slur entirely; paraphrase any insult."
    ),
}


def build_reframe_prompt(line: str, hits: List[Any]) -> Dict[str, Any]:
    """Build (system, user_payload) for one line's reframe.

    Pure + deterministic so the prompt can be unit-asserted. ``hits`` are
    Sanitizer ``Hit`` objects (or anything exposing ``.category``/``.note``);
    their notes (deduped, in first-seen order) plus the matching category
    guidance become the concrete softening instructions for THIS line.
    """
    notes: List[str] = []
    seen_notes = set()
    cats: List[str] = []
    seen_cats = set()
    for h in hits or []:
        note = (getattr(h, "note", None) or "").strip()
        if note and note not in seen_notes:
            seen_notes.add(note)
            notes.append(note)
        cat = (getattr(h, "category", None) or "").strip()
        if cat and cat not in seen_cats:
            seen_cats.add(cat)
            cats.append(cat)

    guidance = [_CATEGORY_GUIDANCE[c] for c in cats if c in _CATEGORY_GUIDANCE]

    parts: List[str] = [_REFRAME_SYSTEM_HEADER, "", SAFE_NARRATION_RULES]
    if guidance:
        parts += ["", "APPLY THESE (the risks this line triggered):"]
        parts += [f"- {g}" for g in guidance]
    if notes:
        parts += ["", "SPECIFIC NOTES FOR THIS LINE:"]
        parts += [f"- {n}" for n in notes]
    system = "\n".join(parts)

    user_payload = {
        "line": line,
        "instruction": (
            "Rewrite the line above to satisfy every rule, keeping the same "
            "story meaning. Output JSON: {\"narration\": \"<rewritten line>\"}."
        ),
    }
    return {"system": system, "user_payload": user_payload}


def reframe_line(
    line: str,
    hits: List[Any],
    call_fn: ReframeCallFn,
    *,
    max_tokens: int = 220,
) -> str:
    """Rewrite ONE narration line to soften the flagged/blocked risks.

    Returns the rewritten line, or the ORIGINAL ``line`` unchanged when there is
    nothing to do (no hits) or the model call fails / returns an empty line —
    so the caller's re-sanitize step always has a concrete line to re-judge and
    an un-softenable block is never silently dropped.
    """
    if not line or not hits:
        return line
    prompt = build_reframe_prompt(line, hits)
    try:
        obj = call_fn(prompt["system"], prompt["user_payload"], REFRAME_SCHEMA, max_tokens)
    except Exception:
        return line
    if not isinstance(obj, dict):
        return line
    new_line = obj.get("narration")
    if not isinstance(new_line, str) or not new_line.strip():
        return line
    return new_line.strip()
