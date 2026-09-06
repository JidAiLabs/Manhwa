#!/usr/bin/env python3
"""
tools/story_pass.py — read the WHOLE chapter once, from its dialogue.

Why this exists (2026-07-20, measured):

The pipeline used to reconstruct the plot from 114 independent per-panel image
calls, each seeing ONE picture with no story context ("describe THIS panel,
not the previous ones"), and then spend 25 serial writer calls reassembling a
narrative from those fragments plus heal cycles repairing the incoherence.
That cost ~79 minutes on Nano Machine ch1 and still got the chapter's central
event backwards: it narrated the assassin leader killing the protagonist when
the protagonist had killed an assassin.

But the plot is already written down. The chapter's whole OCR is ~6.7k
characters (~1.7k tokens) — manhwa is dialogue-driven, and characters SAY what
happened. One call over the ordered transcript produced, in 187 seconds, the
correct synopsis, the cast with per-character FATES ("Assassin Member: killed
by Prince Cheon", "Assassin Leader: alive; retreats"), and the key events with
actor -> target attribution and the verbatim line proving each.

So: dialogue is the highest-trust channel for PLOT; images stay authoritative
for what a panel LOOKS like. This stage owns the former.

Output: manifest.chapter_story.json
  {synopsis, cast: [{name, role, fate}], events: [{panels, actor, does,
   target, evidence}]}

story_ledger.py consumes it to derive per-beat facts (deaths propagate, dead
role-holders' titles get banned), so every existing consumer — the writer's
FACTS block, the identity gate, prep_qa's dead_actor/role_stale — keeps its
contract while the weak per-window arbitration goes away.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from manifest_io import write_manifest  # noqa: E402

# Bump when the prompt/schema changes materially (stamped into the manifest).
PROMPT_VERSION = "sp_v1"

STORY_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "synopsis": {"type": "STRING"},
        "cast": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "role": {"type": "STRING"},
                "fate": {"type": "STRING"},
            },
            "required": ["name", "role", "fate"]}},
        "events": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "panels": {"type": "STRING"},
                "actor": {"type": "STRING"},
                "does": {"type": "STRING"},
                "target": {"type": "STRING"},
                "evidence": {"type": "STRING"},
            },
            "required": ["panels", "actor", "does", "target", "evidence"]}},
    },
    "required": ["synopsis", "cast", "events"],
}

SYSTEM = (
    "You are reading ONE chapter of a manhwa. Below is every panel in reading "
    "order with the dialogue printed on it (verbatim OCR). Panels marked "
    "'(no text)' are wordless art.\n\n"
    "From the DIALOGUE and the panel order, reconstruct what actually "
    "happens. Dialogue is the highest-trust evidence: what characters SAY "
    "about events tells you who did what, including for the wordless panels "
    "between them. A later line that describes an earlier moment (a taunt, a "
    "shocked question, an accusation about who killed whom) SETTLES that "
    "moment — trust it over any guess.\n\n"
    "Return:\n"
    "  synopsis: 5-8 sentences — the chapter's real plot, in order.\n"
    "  cast: every recurring character. name = what to call them (a real "
    "name from the dialogue if there is one, else a short descriptive "
    "handle); role = their part in the story; fate = what has happened to "
    "them BY THE END of this chapter (alive, killed, wounded, fled, "
    "unknown). Be exact about who dies — a character who dies must say so, "
    "and a character who survives must not.\n"
    "  events: the chapter's KEY events in order. For each: panels (the "
    "panel range, e.g. 'p000036-p000037'), the ACTOR, what they DO, the "
    "TARGET, and the verbatim dialogue line that proves it. Be especially "
    "precise about violence: who strikes whom, and who dies. If the evidence "
    "does not settle an event, leave it out rather than guessing.\n"
    "Report ONLY what this chapter establishes. Return ONLY JSON matching "
    "the schema. No extra text."
)


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_transcript(vision: Any, understood: Any = None,
                     max_chars: int = 220) -> str:
    """The chapter as an ordered panel transcript. Publication chrome and
    blank panels are skipped when an understanding is available (they carry
    watermarks/credits, never plot); everything else rides, text or not, so
    wordless beats keep their place in the reading order."""
    kinds = {p.get("scene_file"): str(p.get("panel_kind") or "")
             for p in ((understood or {}).get("panels") or [])
             if isinstance(p, dict)}
    lines: List[str] = []
    for it in (vision or {}).get("items") or []:
        sf = str(it.get("scene_file") or "")
        if not sf:
            continue
        kind = kinds.get(sf, "")
        if kind in ("chrome", "empty"):
            continue
        text = (it.get("ocr_clean") or "").strip().replace("\n", " ")
        lines.append(f"{sf} [{kind or 'story'}]"
                     + (f' "{text[:max_chars]}"' if text else " (no text)"))
    return "\n".join(lines)


def build_story(transcript: str, call_fn) -> Dict[str, Any]:
    """Pure-ish: one call, normalized output. Raises on an unusable answer —
    the caller decides whether that is fatal (the pipeline treats a missing
    chapter story as non-fatal and falls back to the old path)."""
    raw = call_fn(SYSTEM + "\n\nCHAPTER TRANSCRIPT:\n" + transcript) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"story pass returned {type(raw).__name__}, not an object")
    synopsis = str(raw.get("synopsis") or "").strip()
    cast = [{"name": str(c.get("name") or "").strip(),
             "role": str(c.get("role") or "").strip(),
             "fate": str(c.get("fate") or "").strip()}
            for c in (raw.get("cast") or []) if isinstance(c, dict)
            and str(c.get("name") or "").strip()]
    events = [{"panels": str(e.get("panels") or "").strip(),
               "actor": str(e.get("actor") or "").strip(),
               "does": str(e.get("does") or "").strip(),
               "target": str(e.get("target") or "").strip(),
               "evidence": str(e.get("evidence") or "").strip()[:220]}
              for e in (raw.get("events") or []) if isinstance(e, dict)
              and str(e.get("does") or "").strip()]
    if not synopsis and not events:
        raise ValueError("story pass produced neither synopsis nor events")
    return {"synopsis": synopsis, "cast": cast, "events": events,
            "prompt_version": PROMPT_VERSION}


def _ollama_call(model: str, num_ctx: int):
    from ollama_compat import chat as _chat

    def _lower(o):
        if isinstance(o, dict):
            return {k: (v.lower() if k == "type" and isinstance(v, str)
                        else _lower(v))
                    for k, v in o.items() if k != "propertyOrdering"}
        if isinstance(o, list):
            return [_lower(x) for x in o]
        return o

    schema = _lower(STORY_SCHEMA)

    def _call(prompt: str) -> Dict[str, Any]:
        r = _chat(model=model, think=False,
                  messages=[{"role": "user", "content": prompt}],
                  format=schema,
                  options={"temperature": 0.2, "num_ctx": num_ctx,
                           "num_predict": 2200})
        return json.loads(r["message"]["content"])
    return _call


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--understood", default="",
                    help="manifest.panels.understood.json — skips chrome/empty "
                         "panels; optional (transcript still works without it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["ollama"], default="ollama",
                    help="deprecated no-op: local ollama is the only backend")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--num-ctx", type=int,
                    default=int(os.environ.get("STUDIO_STORY_NUM_CTX", "16384")))
    ap.add_argument("--retries", type=int, default=3,
                    help="re-ask on an unusable answer (the call is cheap and "
                         "non-deterministic; a single bad roll must not cost "
                         "the chapter its story)")
    args = ap.parse_args()

    vision = _load(args.vision_manifest)
    understood = (_load(args.understood)
                  if args.understood and os.path.exists(args.understood) else None)
    transcript = build_transcript(vision, understood)
    n_panels = transcript.count("\n") + 1 if transcript else 0
    print(f"[story] transcript: {n_panels} panels, {len(transcript)} chars")
    if not transcript.strip():
        raise SystemExit("[story] empty transcript — no panels to read")

    call = _ollama_call(args.model, args.num_ctx)
    # RETRY: one unparseable roll must not drop the whole chapter back to the
    # ledger's per-window fallback (nano ch1 job 68 did exactly that — the
    # stage exited 1 and the chapter silently lost its story). The call is
    # cheap and non-deterministic, so a re-ask is the right response.
    story = None
    for attempt in range(1, args.retries + 1):
        try:
            story = build_story(transcript, call)
            break
        except Exception as e:
            print(f"[story] attempt {attempt}/{args.retries} unusable: {e}")
            if attempt == args.retries:
                raise
    assert story is not None

    inputs = [args.vision_manifest]
    if args.understood and os.path.exists(args.understood):
        inputs.append(args.understood)
    write_manifest(args.out, story, inputs=tuple(inputs), tool="story_pass",
                   extra_meta={"model": args.model,
                               "prompt_version": PROMPT_VERSION,
                               "transcript_chars": len(transcript)})
    print(f"[ok] wrote={args.out} cast={len(story['cast'])} "
          f"events={len(story['events'])}")
    for c in story["cast"]:
        print(f"  - {c['name']} [{c['role']}] fate={c['fate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
