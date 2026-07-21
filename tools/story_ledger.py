#!/usr/bin/env python3
"""
tools/story_ledger.py — chapter STORY-STATE LEDGER (2026-07-20 root-cause wave).

THE structural gap behind the nano ch1 narration-logic errors (inverted kill,
dead leader re-labeled "the leader", two "our guys", re-asked questions): no
component carried per-character narrative state across panels or groups — the
writer had 2 beats of prose memory, and a beat split could strand the panels
that DISPROVE a visual misread (p000033 "the masked figure strikes the young
man") away from the dialogue that carries the truth (p000037 "HOW DID A KID …
KILL ONE OF OUR MEMBERS?").

This tool builds manifest.ledger.json ONCE per chapter at the beated stage
(after cast_builder, before the writer):

  code   — entity table (cast + derived recurring figures), deterministic
           resolution of every pu_v5 structured action's actor/target to an
           entity (cast_identity is the identity oracle), the chapter
           dialogue digest;
  model  — ONE text-only schema-constrained call: events[] (deaths/defeats,
           reveals, role transitions, answered questions — each with a
           verbatim evidence_quote) + overrides[] where the DIALOGUE
           contradicts a panel's visual attribution. Dialogue is the highest-
           trust channel; the model arbitrates, it never writes prose;
  code   — apply overrides, derive beat_facts per group (deaths propagate to
           all LATER groups; a dead unique-role holder bans their role handle
           downstream), write the manifest atomically with input shas.

Consumers: gemini_narrative_pass (FACTS block per beat + ledger-aware
identity gate), narration_punchup's backstop, prep_qa (dead_actor/role_stale),
narration_heal (fact-carrying correction notes).

Fail-soft: an unparseable arbitration yields a purely-visual ledger
(events=[], overrides=[]) with a loud log — never a crashed beated stage.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from cast_identity import (  # noqa: E402
    cast_profiles,
    resolve_figures,
    resolve_name,
)
from identity_gate import _neutral_from_evidence  # noqa: E402
from manifest_io import write_manifest  # noqa: E402

EVENT_TYPES = ["death", "defeat", "reveal", "role_transition",
               "question_answered"]

LEDGER_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "events": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "type": {"type": "STRING", "enum": EVENT_TYPES},
                "scene_file": {"type": "STRING"},
                "subject": {"type": "STRING"},
                "detail": {"type": "STRING"},
                "evidence_quote": {"type": "STRING"},
            },
            "required": ["type", "scene_file", "subject", "detail",
                         "evidence_quote"]}},
        "overrides": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "scene_file": {"type": "STRING"},
                "actor": {"type": "STRING"},
                "target": {"type": "STRING"},
                "reason": {"type": "STRING"},
            },
            "required": ["scene_file", "actor", "target", "reason"]}},
    },
    "required": ["events", "overrides"],
}

SYSTEM = (
    "You are the story-fact ARBITER for one webtoon/manhwa chapter. You are "
    "given the chapter's ENTITIES, each panel's analyzed actions (who does "
    "what to whom — a VISUAL guess that can be wrong), and every line of "
    "DIALOGUE in reading order. DIALOGUE IS THE HIGHEST-TRUST EVIDENCE: what "
    "characters literally say about what happened outranks a visual guess "
    "(e.g. assassins asking 'how did the kid kill one of our members?' PROVES "
    "the kid killed an assassin, whatever a fight panel looked like).\n"
    "Return JSON with:\n"
    "  events: the chapter's established FACTS, each anchored to the panel "
    "where it becomes true, with a VERBATIM evidence_quote from the dialogue "
    "(or a short panel-evidence phrase when no dialogue covers it):\n"
    "    - death / defeat: a character is killed / decisively taken down. "
    "subject = who died/lost (MUST copy an entity name exactly).\n"
    "    - reveal: an identity or hidden fact is established (e.g. a figure "
    "states who they are). subject = the entity revealed.\n"
    "    - role_transition: a unique role changes hands or ends (a leader "
    "dies, a master is replaced).\n"
    "    - question_answered: a story question a character asked that got a "
    "definitive answer (detail = 'Q -> A' in one line).\n"
    "  overrides: panels whose analyzed actor/target CONTRADICTS the dialogue "
    "evidence — give the corrected actor and target (entity names, or "
    "'unclear'), and the reason quoting the dialogue that proves it. Only "
    "override when the dialogue genuinely proves the direction; when evidence "
    "is thin, leave the panel alone.\n"
    "Report ONLY facts the chapter itself establishes. Return ONLY JSON "
    "matching the schema. No extra text."
)


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_groups(groups_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(groups_manifest.get("shots"), list):
        return groups_manifest["shots"]
    if isinstance(groups_manifest.get("groups"), list):
        return groups_manifest["groups"]
    return []


def _panels(understood: Any) -> List[Dict[str, Any]]:
    return [p for p in ((understood or {}).get("panels") or [])
            if isinstance(p, dict) and p.get("scene_file")]


# --- entity table -------------------------------------------------------------

def build_entities(understood: Any, cast: Any) -> List[Dict[str, Any]]:
    """Cast members + DERIVED entities for recurring figures the cast still
    missed (belt-and-suspenders under cast_builder's recurring-figure rule).
    A derived entity gets a neutral speakable handle from its most common
    subject wording; resolution profiles for both kinds come from
    cast_identity so every downstream match uses the same oracle."""
    from cast_builder import recurring_figures
    members = cast.get("cast") if isinstance(cast, dict) else cast
    entities: List[Dict[str, Any]] = []
    for m in (members or []):
        if not isinstance(m, dict):
            continue
        name = str(m.get("canonical_name") or "").strip()
        if not name:
            continue
        entities.append({
            "id": str(m.get("id") or name),
            "canonical_name": name,
            "source": "cast",
            "role": str(m.get("role") or "").strip().lower(),
            "is_protagonist": bool(m.get("is_protagonist")),
            "aliases": [str(a) for a in (m.get("aliases") or [])],
            "visual_description": str(m.get("visual_description") or ""),
        })
    profiles = cast_profiles({"cast": members or []})
    seen_handles = {e["canonical_name"].lower() for e in entities}
    for i, line in enumerate(recurring_figures(understood)):
        rep = line.split(" — seen in ", 1)[0].strip()
        figs = resolve_figures({"subjects": [rep]}, profiles)
        if any(f.get("name") != "unknown" for f in figs):
            continue                     # a cast member already covers it
        handle = _neutral_from_evidence([{"evidence": rep}])
        if handle.lower() in seen_handles or handle == "the figure":
            continue
        seen_handles.add(handle.lower())
        entities.append({
            "id": f"derived_{i}",
            "canonical_name": handle,
            "source": "derived",
            "role": "derived",
            "is_protagonist": False,
            "aliases": [],
            "visual_description": rep,
        })
    return entities


def entity_profiles(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """cast_identity profiles over the WHOLE entity table (cast + derived) —
    derived entities resolve by their observed appearance wording."""
    return cast_profiles({"cast": [
        {"canonical_name": e["canonical_name"], "aliases": e.get("aliases"),
         "role": e.get("role"),
         "visual_description": e.get("visual_description")}
        for e in entities]})


def _resolve_name(text: str, profiles: List[Dict[str, Any]]) -> str:
    """One actor/target string -> entity canonical name, 'unclear', or ''.

    Delegates to cast_identity.resolve_name — THE shared oracle. Its
    faction-tie rule is what stops two look-alike assassins from erasing each
    other into 'unclear' (which is how the writer once received the fact
    'unclear lunges at our protagonist' and narrated the inversion)."""
    text = str(text or "").strip()
    if not text:
        return ""
    if text.lower() == "unclear":
        return "unclear"
    name, _ev = resolve_name(text, profiles)
    return "unclear" if name == "unknown" else name


def build_panel_actions(understood: Any, profiles: List[Dict[str, Any]]
                        ) -> List[Dict[str, Any]]:
    """Deterministic: every pu_v5 structured action with actor/target resolved
    to entity names. Panels without `actions` (pre-pu_v5) contribute nothing —
    the ledger is still useful for deaths/roles from dialogue alone."""
    out: List[Dict[str, Any]] = []
    for p in _panels(understood):
        for a in (p.get("actions") or []):
            if not isinstance(a, dict):
                continue
            verb = str(a.get("verb") or "").strip()
            if not verb:
                continue
            out.append({
                "scene_file": str(p["scene_file"]),
                "actor": _resolve_name(a.get("actor"), profiles),
                "verb": verb,
                "target": _resolve_name(a.get("target"), profiles),
                "evidence": "visual",
                "raw": {"actor": str(a.get("actor") or ""),
                        "target": str(a.get("target") or "")},
            })
    return out


# --- the ONE arbitration call -------------------------------------------------

def build_digest(entities: List[Dict[str, Any]],
                 panel_actions: List[Dict[str, Any]],
                 understood: Any,
                 panels: Optional[List[Dict[str, Any]]] = None) -> str:
    """Compact digest the arbiter reads: entities (always the full cast, so
    names anchor), plus the actions and dialogue of *panels* (default: the
    whole chapter). Windowing the panels is what makes anchoring reliable —
    one whole-chapter call had to cross-reference ~100 actions against every
    line of dialogue at once and returned three trivial events, missing an
    explicit kill stated in the dialogue."""
    panels = _panels(understood) if panels is None else panels
    window = {str(p["scene_file"]) for p in panels}
    lines: List[str] = ["ENTITIES:"]
    for e in entities:
        tag = " (PROTAGONIST)" if e.get("is_protagonist") else ""
        lines.append(f"  - {e['canonical_name']}{tag} [{e.get('role')}] "
                     f"look: {e.get('visual_description', '')[:120]}")
    lines.append("ANALYZED PANEL ACTIONS (visual guesses, may be wrong):")
    by_file: Dict[str, List[str]] = {}
    for a in panel_actions:
        if a["scene_file"] in window:
            by_file.setdefault(a["scene_file"], []).append(
                f"{a['actor'] or '?'} {a['verb']} {a['target'] or ''}".strip())
    for p in panels:
        sf = str(p["scene_file"])
        acts = "; ".join(by_file.get(sf, []))
        act = str(p.get("action") or "")[:100]
        lines.append(f"  {sf} [{p.get('panel_kind')}] {acts or act}")
    lines.append("DIALOGUE (verbatim OCR, reading order — the highest-trust "
                 "evidence):")
    for p in panels:
        d = str(p.get("dialogue") or "").strip()
        if d:
            lines.append(f"  {p['scene_file']}: {d[:200]}")
    return "\n".join(lines)


def _windows(panels: List[Dict[str, Any]], size: int = 14, overlap: int = 4
             ) -> List[List[Dict[str, Any]]]:
    """Overlapping reading-order windows. The overlap matters: a kill and the
    dialogue that proves it can straddle a boundary (nano ch1's strike is at
    p000033, the 'how did a kid kill one of our members' line at p000037)."""
    if not panels:
        return []
    step = max(1, size - overlap)
    out = [panels[i:i + size] for i in range(0, len(panels), step)
           if panels[i:i + size]]
    return out or [panels]


def arbitrate(digest: str, *, backend: str, model: str, project: str = "",
              location: str = "us-central1", num_ctx: int = 16384
              ) -> Dict[str, Any]:
    """The single model call. Raises on transport/parse errors — the caller
    fail-softs to a visual-only ledger."""
    if backend == "ollama":
        from ollama_compat import chat as _ollama_chat

        def _lower_types(o):
            if isinstance(o, dict):
                return {k: (v.lower() if k == "type" and isinstance(v, str)
                            else _lower_types(v))
                        for k, v in o.items() if k != "propertyOrdering"}
            if isinstance(o, list):
                return [_lower_types(x) for x in o]
            return o

        resp = _ollama_chat(
            model=model,
            messages=[{"role": "user", "content": SYSTEM + "\n\n" + digest}],
            format=_lower_types(LEDGER_SCHEMA), think=False,
            options={"temperature": 0.2, "num_ctx": num_ctx,
                     "num_predict": 2000})
        return json.loads(resp["message"]["content"])
    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True, project=project, location=location)
    resp = client.models.generate_content(
        model=model,
        contents=[SYSTEM + "\n\n" + digest],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=LEDGER_SCHEMA,
        ),
    )
    return json.loads(resp.text)


# --- deterministic derivation ---------------------------------------------------

def normalize_events(raw: Any, entities: List[Dict[str, Any]],
                     understood: Any, profiles: Optional[List[Dict[str, Any]]]
                     = None, log=print) -> List[Dict[str, Any]]:
    """Anchor each event to a real entity + a real panel, and DEDUPE.

    The subject is matched exactly first, then resolved through the identity
    oracle ('one of our members' / 'an assassin member' must land on the
    faction entity). Every rejection is LOGGED — the first version dropped
    non-matching subjects silently, so a missing death looked identical to a
    model that never claimed one."""
    names = {e["canonical_name"].lower(): e["canonical_name"]
             for e in entities}
    files = {str(p["scene_file"]) for p in _panels(understood)}
    profiles = profiles if profiles is not None else entity_profiles(entities)
    out: List[Dict[str, Any]] = []
    seen: Set[tuple] = set()
    for ev in (raw or []):
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type") or "").strip()
        raw_subj = str(ev.get("subject") or "").strip()
        sf = str(ev.get("scene_file") or "").strip()
        subj = names.get(raw_subj.lower())
        if not subj and raw_subj:
            got, _ev = resolve_name(raw_subj, profiles)
            subj = None if got == "unknown" else got
        if etype not in EVENT_TYPES:
            log(f"[ledger] dropped event: unknown type {etype!r}")
            continue
        if not subj:
            log(f"[ledger] dropped {etype}: subject {raw_subj!r} matches no "
                "entity (cast coverage gap?)")
            continue
        if sf not in files:
            log(f"[ledger] dropped {etype} for {subj!r}: scene_file {sf!r} "
                "is not a panel of this chapter")
            continue
        key = (etype, sf, subj)
        if key in seen:                  # overlapping windows see it twice
            continue
        seen.add(key)
        out.append({"type": etype, "scene_file": sf, "subject": subj,
                    "detail": str(ev.get("detail") or "").strip()[:300],
                    "evidence_quote":
                        str(ev.get("evidence_quote") or "").strip()[:200]})
    return out


def apply_overrides(panel_actions: List[Dict[str, Any]], raw: Any,
                    entities: List[Dict[str, Any]],
                    profiles: Optional[List[Dict[str, Any]]] = None) -> int:
    """Flip actor/target on dialogue-arbitrated panels, in place. Entity names
    resolve through the oracle (exact first, then appearance/faction), so a
    natural phrasing like 'the assassin member' still anchors; an override for
    a panel with no recorded action ADDS one (the misread panel may have
    shipped only free text)."""
    names = {e["canonical_name"].lower(): e["canonical_name"]
             for e in entities}
    profiles = profiles if profiles is not None else entity_profiles(entities)

    def _n(v: str) -> str:
        v = str(v or "").strip()
        if not v:
            return ""
        hit = names.get(v.lower())
        if hit:
            return hit
        got, _ev = resolve_name(v, profiles)
        return "unclear" if got == "unknown" else got

    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for a in panel_actions:
        by_file.setdefault(a["scene_file"], []).append(a)
    applied = 0
    for ov in (raw or []):
        if not isinstance(ov, dict):
            continue
        sf = str(ov.get("scene_file") or "").strip()
        actor, target = _n(ov.get("actor")), _n(ov.get("target"))
        if not sf or not (actor or target):
            continue
        verb = str(ov.get("verb") or "").strip()
        acts = by_file.get(sf)
        if acts:
            for a in acts:
                a["actor"], a["target"] = actor, target
                if verb:                       # the OUTCOME, from the story
                    a["verb"] = verb
                a["evidence"] = "dialogue_arbitrated"
                a["reason"] = str(ov.get("reason") or "").strip()[:200]
        else:
            entry = {"scene_file": sf, "actor": actor,
                     "verb": verb or "acts on",
                     "target": target, "evidence": "dialogue_arbitrated",
                     "reason": str(ov.get("reason") or "").strip()[:200],
                     "raw": {}}
            panel_actions.append(entry)
            by_file.setdefault(sf, []).append(entry)
        applied += 1
    return applied


def _unique_role_handles(dead: str, entities: List[Dict[str, Any]],
                         alive: Set[str]) -> List[str]:
    """Handles that stop being sayable once *dead* is gone: 'the <token>' for
    each of the dead entity's name tokens no LIVING entity shares ('the
    leader' after the leader dies — a surviving assassin must not inherit
    it)."""
    from cast_identity import _name_tokens
    by_name = {e["canonical_name"]: e for e in entities}
    m = by_name.get(dead)
    if not m:
        return []
    mine = _name_tokens({"canonical_name": m["canonical_name"],
                         "aliases": m.get("aliases")})
    for other in entities:
        if other["canonical_name"] == dead:
            continue
        if other["canonical_name"] not in alive:
            continue
        mine -= _name_tokens({"canonical_name": other["canonical_name"],
                              "aliases": other.get("aliases")})
    return sorted(f"the {t}" for t in mine)


def build_beat_facts(groups: List[Dict[str, Any]],
                     events: List[Dict[str, Any]],
                     panel_actions: List[Dict[str, Any]],
                     entities: List[Dict[str, Any]],
                     understood: Any) -> Dict[str, Dict[str, Any]]:
    """{group_id: facts} — pure derivation. Deaths propagate to every LATER
    group (the group containing the death narrates it; afterwards the dead
    cannot act and their unique role handle is banned). Answered questions
    propagate the same way."""
    order = {str(p["scene_file"]): i for i, p in enumerate(_panels(understood))}
    profiles = entity_profiles(entities)
    acts_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for a in panel_actions:
        acts_by_file.setdefault(a["scene_file"], []).append(a)
    u_by_file = {str(p["scene_file"]): p for p in _panels(understood)}
    all_names = {e["canonical_name"] for e in entities}

    def _ev_index(ev: Dict[str, Any]) -> int:
        return order.get(ev["scene_file"], -1)

    facts: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        gid = int(g.get("shot_id") or 0)
        files = [str(f) for f in (g.get("scene_files") or [])]
        idxs = [order[f] for f in files if f in order]
        start = min(idxs) if idxs else -1
        present: Set[str] = set()
        for f in files:
            for fig in resolve_figures(u_by_file.get(f), profiles):
                if fig.get("name") and fig["name"] != "unknown":
                    present.add(fig["name"])
        actions = []
        for f in files:
            for a in acts_by_file.get(f, []):
                # A story verb is a whole phrase that usually names its own
                # object ("kills an assassin using a deceptive strike"), so
                # re-appending the target would read as word salad. Only add
                # it when the verb does not already refer to it.
                verb, tgt = a["verb"], a["target"]
                if tgt and _mentions(verb, tgt):
                    tgt = ""
                s = " ".join(x for x in (a["actor"] or "someone", verb, tgt)
                             if x).strip()
                mark = (" [dialogue-arbitrated]"
                        if a.get("evidence") == "dialogue_arbitrated" else "")
                actions.append(f"{f}: {s}{mark}")
        dead_before = sorted({ev["subject"] for ev in events
                              if ev["type"] == "death"
                              and 0 <= _ev_index(ev) < start})
        alive = all_names - set(dead_before)
        banned: List[str] = []
        for d in dead_before:
            banned += _unique_role_handles(d, entities, alive)
        answered = [ev["detail"] for ev in events
                    if ev["type"] == "question_answered"
                    and 0 <= _ev_index(ev) < start]
        facts[f"g{gid:04d}"] = {
            "present": sorted(present),
            "actions": actions,
            "dead_by_now": dead_before,
            "banned_handles": sorted(set(banned)),
            "answered": answered,
        }
    return facts


# Generic English death vocabulary — NOT series content.
#
# HISTORY (2026-07-20, a bug I shipped): this used to be matched against the
# free text of any event, and a death was recorded whenever the word "kill"
# appeared. The story pass reported "Assassin Leader explains that they must
# kill Cheon" — a THREAT — and the ledger recorded the PROTAGONIST as dead at
# p000017. His handles were then banned and the identity gate rewrote "our
# guy" into "the assassin leader", inverting the narration.
#
# So a kill word is now necessary but NOT sufficient. Deaths come from the
# story pass's structured per-character `fate` field (its explicit answer to
# "what happened to this character"), events only supply the anchor panel, and
# anything phrased as intent//attempt/order is never a death.
_DEATH_RE = re.compile(
    r"\b(kill(?:s|ed)|slain|slew|murder(?:s|ed)|execut(?:es|ed)|"
    r"died|dies|dead|deceased|killed\s+off|struck\s+down|cut\s+down|corpse)\b",
    re.IGNORECASE)

# Intent, attempt, order, or plan — a stated wish to kill is not a killing.
_INTENT_RE = re.compile(
    r"\b(must|will|would|shall|should|could|can|may|might|"
    r"wants?\s+to|plans?\s+to|intends?\s+to|tries?\s+to|attempts?\s+to|"
    r"seeks?\s+to|hopes?\s+to|decides?\s+to|about\s+to|going\s+to|"
    r"orders?|commands?|threatens?|vows?|swears?|promises?|"
    r"to\s+kill|for\s+killing|so\s+they\s+can)\b", re.IGNORECASE)


# Explicit survival. A fate sentence often mentions SOMEONE ELSE's death:
# "Alive; retreats after the Prince kills one of his men" is the LEADER
# surviving, and reading its "kills" as his own death is the same mistake in
# mirror image. Survival stated about this character always wins.
_SURVIVES_RE = re.compile(
    r"\b(alive|survives?|survived|lives|living|escapes?|escaped|flees|fled|"
    r"retreats?|retreated|withdraws?|unharmed|unhurt|recovers?|recovered|"
    r"spared|rescued|saved|wounded\s+but|injured\s+but|near\s+death\s+but|"
    r"unknown)\b", re.IGNORECASE)


def is_completed_death(text: str) -> bool:
    """Pure: does *text* say THIS character died, and that it already happened?

    Three gates, all necessary:
      - a death word is present ("killed", "died", "slain");
      - it is not intent/attempt/order ("must kill", "vows to kill");
      - the text does not state survival ("alive", "retreats", "survives").

    "kills an assassin through a deceptive strike" -> True (as an EVENT verb;
    the caller pairs it with the victim's own fate). "Killed by Prince Cheon"
    -> True. "they must kill Cheon" -> False. "Alive; retreats after the
    Prince kills one of his men" -> False."""
    t = str(text or "")
    if not _DEATH_RE.search(t):
        return False
    return not _INTENT_RE.search(t) and not _SURVIVES_RE.search(t)


def _panel_range(text: str, ordered_files: List[str]) -> List[str]:
    """'p000036-p000037' (or 'p000036.jpg') -> the scene_files it spans.

    The story pass echoes panel ids from the transcript, which may or may not
    carry the extension, so matching is by stem against the real reading
    order. An unparseable or unknown range yields [] — the event is then
    dropped and LOGGED rather than anchored to a guess."""
    stems = {os.path.splitext(f)[0]: f for f in ordered_files}
    order = {f: i for i, f in enumerate(ordered_files)}
    found = [stems[m] for m in re.findall(r"p\d+", str(text or ""))
             if m in stems]
    if not found:
        return []
    if len(found) == 1:
        return found
    lo, hi = min(order[f] for f in found), max(order[f] for f in found)
    return ordered_files[lo:hi + 1]


def _mentions(verb: str, target: str) -> bool:
    """Does *verb* already refer to *target*? Compares identity tokens, so
    'kills an assassin using a deceptive strike' covers 'the masked
    assassin' (shared 'assassin') and the target is not appended twice."""
    from cast_identity import _name_tokens
    vt = {t for t in re.findall(r"[a-z]+", str(verb or "").lower())}
    tt = _name_tokens({"canonical_name": str(target or ""), "aliases": []})
    return bool(tt) and bool(tt & vt)


def _is_silent(text: Any) -> bool:
    """A panel carries no usable dialogue: empty, or punctuation/SFX-only
    ('?!', '...'). Fewer than 3 letters or digits means nothing was said."""
    return len(re.findall(r"[A-Za-z0-9]", str(text or ""))) < 3


def extend_over_silent(span: List[str], ordered: List[str],
                       dialogue: Dict[str, Any],
                       anchored: Set[str], max_back: int = 4) -> List[str]:
    """Widen an event's span BACKWARD across adjacent wordless panels.

    THE fix for the nano ch1 inversion surviving the story pass. The story
    pass reads text, so it anchors an event where the DIALOGUE is — the kill
    landed on p000036-37 ("serves you right", "how did a kid kill one of our
    members"). But the strike is DRAWN at p000033, which is wordless, so it
    received no attribution at all and the writer fell back to the per-panel
    understanding, which had the direction backwards.

    A silent action panel immediately before a dialogue panel is the moment
    that dialogue is reacting to, so it belongs to that event. Extension stops
    at the first panel that speaks (its own event owns it), at another event's
    anchor, or after *max_back* panels — a line reacts to the action just
    before it, not to something fifteen panels earlier, and without the cap a
    long wordless stretch would be swallowed whole by the next event."""
    if not span:
        return span
    pos = {f: i for i, f in enumerate(ordered)}
    start = pos.get(span[0])
    if start is None:
        return span
    i = start - 1
    while i >= 0 and (start - i) <= max_back:
        fn = ordered[i]
        if fn in anchored or not _is_silent(dialogue.get(fn)):
            break
        span = [fn] + span
        i -= 1
    return span


def facts_from_chapter_story(story: Any, entities: List[Dict[str, Any]],
                             understood: Any,
                             profiles: List[Dict[str, Any]],
                             log=print) -> tuple:
    """(raw_events, raw_overrides) derived DETERMINISTICALLY from the
    whole-chapter story pass — no model call here.

    This replaces the per-window arbitration, which had to cross-reference
    ~100 visual guesses against every line of dialogue in 12 separate calls
    and still missed a kill the dialogue stated outright. The story pass sees
    the entire chapter at once, so its events already carry actor -> target
    and the proving line; all that is left is anchoring them to panels."""
    ordered = [str(p["scene_file"]) for p in _panels(understood)]
    events: List[Dict[str, Any]] = []
    overrides: List[Dict[str, Any]] = []

    # STEP 1 — who actually died? The story pass answers this per character in
    # `fate`. That is the ONLY death authority: a character dies when their own
    # fate says so, never because some event sentence contains the word "kill"
    # (a villain's threat is not a death).
    dead: Dict[str, str] = {}
    for c in ((story or {}).get("cast") or []):
        if not isinstance(c, dict):
            continue
        fate = str(c.get("fate") or "")
        if not is_completed_death(fate):
            continue
        who, _e = resolve_name(str(c.get("name") or ""), profiles)
        if who == "unknown":
            log(f"[ledger] {c.get('name')!r} is {fate[:50]!r} but matches no "
                "entity — that death cannot propagate")
            continue
        dead[who] = fate

    # STEP 2 — walk the events for panel attribution, and anchor each death to
    # the panel where the story says it happens.
    dialogue = {str(p["scene_file"]): p.get("dialogue")
                for p in _panels(understood)}
    raw_spans = [_panel_range(ev.get("panels"), ordered)
                 for ev in ((story or {}).get("events") or [])
                 if isinstance(ev, dict)]
    anchored = {f for s in raw_spans for f in s}
    for ev in ((story or {}).get("events") or []):
        if not isinstance(ev, dict):
            continue
        span = _panel_range(ev.get("panels"), ordered)
        does = str(ev.get("does") or "")
        if not span:
            log(f"[ledger] story event not anchorable to panels "
                f"({ev.get('panels')!r}): {does[:60]!r}")
            continue
        # the wordless action panels this dialogue is reacting to
        anchor = span[0]
        span = extend_over_silent(span, ordered, dialogue, anchored)
        if span[0] != anchor:
            log(f"[ledger] event at {anchor} extended back to {span[0]} "
                f"over silent panel(s): {does[:50]!r}")
        actor, _a = resolve_name(str(ev.get("actor") or ""), profiles)
        target, _t = resolve_name(str(ev.get("target") or ""), profiles)
        # direction: every panel in the span gets the story's attribution
        if actor != "unknown" or target != "unknown":
            for fn in span:
                overrides.append({
                    "scene_file": fn,
                    "actor": "" if actor == "unknown" else actor,
                    "target": "" if target == "unknown" else target,
                    # WHAT happened, not just who. Without this the override
                    # replaced actor/target but left the per-panel VISUAL verb
                    # in place, so the writer was handed "our protagonist
                    # lunges at the masked assassin" when the story said he
                    # KILLS him — corrected the actor and lost the outcome,
                    # and the chapter's turning point never reached the page.
                    "verb": does[:120],
                    "reason": f"chapter story: {does[:80]} | "
                              f"{str(ev.get('evidence') or '')[:80]}"})
        # A death is recorded ONLY when the fate sheet already says this
        # character died AND this event describes the killing actually
        # happening. Both gates must pass; either alone was the old bug.
        if (target in dead and is_completed_death(does)
                and not any(e["subject"] == target for e in events)):
            events.append({"type": "death", "scene_file": span[-1],
                           "subject": target,
                           "detail": f"{does[:150]} (fate: {dead[target][:60]})",
                           "evidence_quote": str(ev.get("evidence") or "")[:200]})

    for who, fate in dead.items():
        if not any(e["subject"] == who for e in events):
            log(f"[ledger] {who!r} is {fate[:50]!r} per the cast sheet but no "
                "event anchors that death to a panel — it will NOT propagate")
    return events, overrides


def dead_sets_by_file(ledger: Any, ordered_files: List[str]
                      ) -> Dict[str, Set[str]]:
    """{scene_file: entities dead STRICTLY BEFORE this panel} — the excluded
    sets cast_identity.resolve_figures_by_file takes so a killed look-alike
    (the leader) stops claiming later panels. {} keys are omitted."""
    order = {str(f): i for i, f in enumerate(ordered_files)}
    deaths = sorted(
        (order[str(ev.get("scene_file"))], str(ev.get("subject")))
        for ev in ((ledger or {}).get("events") or [])
        if ev.get("type") == "death" and str(ev.get("scene_file")) in order)
    out: Dict[str, Set[str]] = {}
    for f, i in order.items():
        dead = {s for di, s in deaths if di < i}
        if dead:
            out[f] = dead
    return out


def build_ledger(understood: Any, groups_m: Any, cast: Any,
                 arbitrate_fn=None, log=print,
                 chapter_story: Any = None) -> Dict[str, Any]:
    """The whole pipeline minus I/O — injectable arbitrate_fn for tests.

    When *chapter_story* is present (tools/story_pass.py), its events are the
    fact source and NO model call is made here: one whole-chapter read beats
    12 windowed arbitrations that each saw a slice. arbitrate_fn remains the
    fallback for chapters with no story pass."""
    entities = build_entities(understood, cast)
    profiles = entity_profiles(entities)
    panel_actions = build_panel_actions(understood, profiles)
    groups = _read_groups(groups_m or {})
    events: List[Dict[str, Any]] = []
    overrides_applied = 0
    if chapter_story:
        raw_events, raw_overrides = facts_from_chapter_story(
            chapter_story, entities, understood, profiles, log=log)
        events = normalize_events(raw_events, entities, understood,
                                  profiles=profiles, log=log)
        overrides_applied = apply_overrides(panel_actions, raw_overrides,
                                            entities, profiles=profiles)
        log(f"[ledger] from chapter story: {len(events)} death event(s), "
            f"{overrides_applied} panel attribution(s) — 0 model calls")
    elif arbitrate_fn is not None:
        raw_events: List[Any] = []
        raw_overrides: List[Any] = []
        windows = _windows(_panels(understood))
        failed = 0
        for i, win in enumerate(windows):
            try:
                digest = build_digest(entities, panel_actions, understood,
                                      panels=win)
                raw = arbitrate_fn(digest) or {}
                raw_events += list(raw.get("events") or [])
                raw_overrides += list(raw.get("overrides") or [])
            except Exception as e:      # one bad window must not lose the rest
                failed += 1
                log(f"[ledger] window {i + 1}/{len(windows)} "
                    f"({win[0]['scene_file']}..{win[-1]['scene_file']}) "
                    f"arbitration FAILED ({e!r}) — its facts are missing")
        if failed == len(windows):
            log("[ledger] ARBITRATION FAILED on EVERY window — writing a "
                "visual-only ledger (events=[]); direction overrides and "
                "death propagation are OFF for this chapter")
        events = normalize_events(raw_events, entities, understood,
                                  profiles=profiles, log=log)
        overrides_applied = apply_overrides(panel_actions, raw_overrides,
                                            entities, profiles=profiles)
        log(f"[ledger] {len(windows)} window(s), {failed} failed -> "
            f"{len(events)} event(s), {overrides_applied} override(s)")
    return {
        "entities": entities,
        "panel_actions": panel_actions,
        "events": events,
        "beat_facts": build_beat_facts(groups, events, panel_actions,
                                       entities, understood),
        "stats": {"overrides_applied": overrides_applied,
                  "events": len(events),
                  "actions": len(panel_actions)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--understood", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--cast", required=True)
    ap.add_argument("--chapter-story", default="",
                    help="manifest.chapter_story.json (story_pass). When "
                         "present its events ARE the facts and no model call "
                         "is made here.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vertex", "ollama"],
                    default="ollama")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--num-ctx", type=int,
                    default=int(os.environ.get("STUDIO_LEDGER_NUM_CTX",
                                               "16384")))
    args = ap.parse_args()

    understood = _load(args.understood)
    groups_m = _load(args.groups)
    cast = _load(args.cast)

    def _arb(digest: str) -> Dict[str, Any]:
        return arbitrate(digest, backend=args.backend, model=args.model,
                         project=args.project, location=args.location,
                         num_ctx=args.num_ctx)

    story = (_load(args.chapter_story)
             if args.chapter_story and os.path.exists(args.chapter_story)
             else None)
    ledger = build_ledger(understood, groups_m, cast, arbitrate_fn=_arb,
                          chapter_story=story)
    inputs = [args.understood, args.groups, args.cast]
    if story is not None:
        inputs.append(args.chapter_story)
    write_manifest(args.out, ledger, inputs=tuple(inputs),
                   tool="story_ledger",
                   extra_meta={"model": args.model, "backend": args.backend})
    s = ledger["stats"]
    print(f"[ok] wrote={args.out} entities={len(ledger['entities'])} "
          f"actions={s['actions']} events={s['events']} "
          f"overrides={s['overrides_applied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
