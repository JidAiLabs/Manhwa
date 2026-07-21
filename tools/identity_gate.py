#!/usr/bin/env python3
"""identity_gate.py — deterministic actor-handle gate over narration lines.

Extracted from gemini_narrative_pass (2026-07-20 story-state wave) so BOTH the
writer AND narration_punchup's post-punchup backstop can run the same gate:
punchup's persona rewrite (temp 0.7) runs AFTER the writer's gate and used to
re-attach wrong-character handles ("our guy" on the helper) with no re-check —
the self-documented ordering hole this extraction closes.

Identity evidence comes ONLY from tools/cast_identity.py resolutions (the
single deterministic identity oracle); this module never guesses.
"""
from __future__ import annotations

import re
import sys
import os
from typing import Any, Dict, List, Optional, Set

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from beats_segments import beat_segments, write_segment_lines  # noqa: E402
from cast_identity import subject_actor_nouns_ex  # noqa: E402

_PROT_HANDLE_RE = re.compile(r"\bour (?:guy|boy|man|protagonist)\b",
                             re.IGNORECASE)
_HANDLE_STOP = frozenset({"a", "an", "the", "in", "with", "and", "of", "on"})


def protagonist_names(cast: Any) -> Set[str]:
    """Canonical names of the protagonist member(s) of a cast manifest (full
    dict or bare members list) — the single referent 'our guy' may claim."""
    if isinstance(cast, dict):
        cast = cast.get("cast")
    names = {
        str(m.get("canonical_name") or "").strip()
        for m in (cast or []) if isinstance(m, dict)
        and (m.get("is_protagonist")
             or str(m.get("role") or "").lower() == "protagonist"
             or str(m.get("canonical_name") or "") == "our protagonist")}
    names.discard("")
    return names


def _neutral_from_evidence(figs) -> str:
    """A grounded neutral handle from an unknown figure's evidence subject —
    'a masked figure in a dark hooded cloak' -> 'the masked figure'.
    Gerunds/verbs are skipped ('a person wearing…' must yield 'the person',
    never 'the person wearing' — job-48 g0018)."""
    person_nouns = ("figure", "person", "man", "woman", "warrior",
                    "stranger", "fighter")
    for f in figs or []:
        ev = str((f or {}).get("evidence") or "").strip().lower()
        toks = [t for t in re.findall(r"[a-z']+", ev)
                if t not in _HANDLE_STOP and not t.endswith("ing")]
        # anchor on the person-noun: 'a masked figure in a dark hooded
        # cloak' -> 'the masked figure'; 'a person wearing a light blue
        # hooded jacket' -> 'the person'
        for i, t in enumerate(toks):
            if t in person_nouns:
                return "the " + ((toks[i - 1] + " ") if i else "") + t
        if len(toks) >= 2:
            return "the " + " ".join(toks[:2])
        if toks:
            return "the " + toks[0]
    return "the figure"


def spoken_names(cast: Any) -> Dict[str, str]:
    """{canonical_name: spoken_name} for cast entries that carry one. The
    gate REWRITES lines, so it must speak the same prose the writer does —
    otherwise it reintroduces the catalogue label ('the Assassin Member')
    it was meant to keep out."""
    if isinstance(cast, dict):
        cast = cast.get("cast")
    out: Dict[str, str] = {}
    for m in (cast or []):
        if not isinstance(m, dict):
            continue
        key = str(m.get("canonical_name") or "").strip()
        say = str(m.get("spoken_name") or "").strip()
        if key and say:
            out[key] = say
    return out


def _figure_handle(name: str, spoken: Optional[Dict[str, str]] = None) -> str:
    """Speakable handle for a resolved cast name: the cast's own spoken form
    when it has one, else 'unnamed assassin' -> 'the assassin'; the
    cast_builder protagonist convention keeps its own handle; real names are
    used as-is."""
    if spoken and name in spoken:
        return spoken[name]
    if name.lower().startswith("unnamed "):
        return "the " + name.split(" ", 1)[1]
    return name


def enforce_actor_handles(beat, figures_by_file, noun_map, protagonist_names,
                          ledger=None, spoken=None):
    """Deterministic identity gate (2026-07-16 wave): a line may claim the
    protagonist ('our guy'/'our protagonist'/a protagonist name-noun) ONLY
    when the span's cast_identity-resolved figures include the protagonist;
    any subject-position actor-noun disjoint from the span's figures is
    rewritten noun-for-noun to what the panel actually shows — but only in
    the UNAMBIGUOUS case (exactly one resolved figure, singular noun; the
    all-unknown span gets a neutral evidence-derived handle). Everything
    else is left for the actor_mismatch heal net. Shares its pattern
    authority with prep_qa (cast_identity.subject_actor_nouns/actor_noun_map)
    so guard and QA can never disagree. Returns 'old -> new' descriptions;
    lines are edited in place via write_segment_lines (spans untouched).

    2026-07-20 story-state wave — an optional *ledger* (manifest.ledger.json
    object) upgrades three formerly-hands-off cases:
      - multi-figure span: when the ledger's panel_actions attribute the
        span's action to exactly ONE living entity, that entity's handle
        breaks the tie (the old gate bailed on every fight panel);
      - a noun whose cast members are ALL in this beat's dead_by_now is
        rewritten even if a stale figure resolution still lists them;
      - zero-figure span: a protagonist handle is re-pointed when the beat's
        facts place the protagonist absent AND the span has one clear actor.
    Without a ledger, behavior is byte-identical to the 2026-07-16 gate."""
    segs = beat_segments(beat)
    if not segs:
        return []
    facts: dict = {}
    actions_by_file: Dict[str, List[dict]] = {}
    if ledger:
        gid = int(beat.get("group_id") or 0)
        facts = (ledger.get("beat_facts") or {}).get(f"g{gid:04d}") or {}
        for a in (ledger.get("panel_actions") or []):
            fn = str(a.get("scene_file") or "")
            if fn:
                actions_by_file.setdefault(fn, []).append(a)
    dead_now = set(facts.get("dead_by_now") or [])
    present = set(facts.get("present") or [])
    rewrites: List[str] = []
    lines = [s["line"] or "" for s in segs]
    new_lines = list(lines)
    for i, s in enumerate(segs):
        line = new_lines[i]
        if not line:
            continue
        span = s["span"] or []
        span_figs = [f for fn in span
                     for f in (figures_by_file.get(fn) or [])]
        # living entities the ledger says ACT in this span (tie-breaker)
        span_actors = {str(a.get("actor"))
                       for fn in span for a in actions_by_file.get(fn, [])
                       if a.get("actor")
                       and a.get("actor") != "unclear"} - dead_now
        if not span_figs:
            # no figure ground truth -> hands off, UNLESS the ledger places
            # the protagonist absent from this beat and names one clear actor
            if (facts and present and protagonist_names
                    and not (present & protagonist_names)
                    and len(span_actors) == 1
                    and _PROT_HANDLE_RE.search(line)):
                repl = _figure_handle(next(iter(span_actors)), spoken)
                new = _PROT_HANDLE_RE.sub(repl, line, count=1)
                if new != line:
                    rewrites.append(
                        f"protagonist handle -> {repl!r} (ledger)")
                    new_lines[i] = new
            continue
        named = {f["name"] for f in span_figs
                 if f.get("name") and f["name"] != "unknown"}

        def _repl_for() -> str:
            if len(named) == 1:
                return _figure_handle(next(iter(named)), spoken)
            if not named:
                return _neutral_from_evidence(span_figs)
            if len(span_actors) == 1:         # ledger breaks the tie
                return _figure_handle(next(iter(span_actors)), spoken)
            return ""                         # multi-figure: ambiguous

        # 1. protagonist handle over a span that doesn't resolve them
        if (protagonist_names and not (named & protagonist_names)
                and _PROT_HANDLE_RE.search(line)):
            repl = _repl_for()
            if repl:
                new = _PROT_HANDLE_RE.sub(repl, line, count=1)
                if new != line:
                    rewrites.append(f"protagonist handle -> {repl!r}")
                    line = new
        # 2. subject-position actor-noun disjoint from the span's figures
        #    (singular only; plurals are the actor_count heal net's job).
        #    A noun whose members are ALL dead by this beat is never a valid
        #    subject even when a stale resolution still lists them.
        for noun, members, plural in subject_actor_nouns_ex(line, noun_map):
            all_dead = bool(members) and members <= dead_now
            if plural or ((members & named) and not all_dead):
                continue
            if named & protagonist_names and members & named:
                continue
            repl = _repl_for()
            if not repl or repl.lower().find(noun) >= 0:
                continue                      # ambiguous / no-op rewrite
            pat = re.compile(
                r"\b(?:(?:our|the|a|an)\s+)?" + re.escape(noun)
                + r"(?P<poss>'s)?\b", re.IGNORECASE)
            new, n = pat.subn(
                lambda m: repl + (m.group("poss") or ""), line, count=1)
            if n and new != line:
                rewrites.append(f"'{noun}' -> {repl!r}"
                                + (" (dead actor)" if all_dead else ""))
                line = new
        new_lines[i] = line
    if new_lines != lines and all(x.strip() for x in new_lines):
        write_segment_lines(beat, new_lines)
    return rewrites
