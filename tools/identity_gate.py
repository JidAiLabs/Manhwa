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
from cast_identity import (_HANDLE_LEAD, subject_actor_nouns_ex,  # noqa: E402
                           subject_person_count)

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


def name_forms(cast: Any) -> Dict[str, List[str]]:
    """{canonical_name: [canonical, aliases…, spoken_name]} — every written form
    of a member, so a multi-word name is recognised whichever form the line
    uses (the cap_protagonist_name lesson: a rewriter built from one name needs
    every alias)."""
    if isinstance(cast, dict):
        cast = cast.get("cast")
    out: Dict[str, List[str]] = {}
    for m in (cast or []):
        if not isinstance(m, dict):
            continue
        key = str(m.get("canonical_name") or "").strip()
        if not key:
            continue
        forms = [key] + [str(a).strip() for a in (m.get("aliases") or [])
                         if str(a).strip()]
        say = str(m.get("spoken_name") or "").strip()
        if say:
            forms.append(say)
        out[key] = list(dict.fromkeys(forms))
    return out


def _full_name_at(line: str, start: int, end: int, members: Set[str],
                  names: Dict[str, List[str]]):
    """(match, proper) for the longest multi-word name of *members* written in
    *line* that covers line[start:end], else (None, False). *proper* = the
    form carries a capitalised word ("the Book of Command") rather than being
    a descriptive handle ("the hooded leader")."""
    forms = []
    for mbr in members:
        for f in (names or {}).get(mbr) or [mbr]:
            words = re.findall(r"[\w'’-]+", f)
            while words and words[0].lower() in _HANDLE_LEAD:
                words = words[1:]
            if len(words) >= 2:
                forms.append((words, any(w[:1].isupper() for w in words)))
    lead = "|".join(sorted(_HANDLE_LEAD))
    for words, proper in sorted(forms, key=lambda x: -len(x[0])):
        pat = re.compile(r"\b(?:(?:" + lead + r")\s+)?"
                         + r"\s+".join(map(re.escape, words))
                         + r"(?P<poss>'s)?\b", re.IGNORECASE)
        for m in pat.finditer(line):
            if m.start() <= start and end <= m.end():
                return m, proper
    return None, False


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


def _base(f: Any) -> str:
    return os.path.basename(str(f or ""))


def solo_mc_resolver(identity: Any, understood_by_file: Dict[str, Any],
                     text_by_file: Optional[Dict[str, str]] = None,
                     fold_reach: int = 3):
    """span -> the protagonist's name when the PICTURE shows him alone on every
    panel the line voices, else None. None (no resolver) without an identity.

    A panel is solo when the image pass confirmed him on its one head, the
    understanding lists one person, and nothing is written on it (speech,
    caption, OCR: an off-panel speaker may be who the line is about). A panel
    with no person and no text does not veto. The window is the span's range
    in chapter order widened over adjacent non-story panels, as far as
    prep_qa._covered_panels reaches: a caption folds its words into the
    neighbouring line, so the line may be about whoever the caption names."""
    if not identity:
        return None
    recs = identity.get("panels", identity) if isinstance(identity, dict) else {}
    texts = {_base(f): str(t or "") for f, t in (text_by_file or {}).items()}
    order: List[str] = []
    kind: Dict[str, str] = {}
    solo: Dict[str, str] = {}
    for f, u in (understood_by_file or {}).items():
        b = _base(f)
        u = u if isinstance(u, dict) else {}
        order.append(b)
        kind[b] = str(u.get("panel_kind") or "").lower()
        written = (str(u.get("dialogue") or "") + texts.get(b, "")).strip()
        persons = sum(subject_person_count(str(x)) for x in (u.get("subjects") or []))
        rec = recs.get(b) if isinstance(recs, dict) else None
        if written:
            continue
        if persons == 0 and not (rec or {}).get("heads"):
            solo[b] = ""
        elif (rec and rec.get("mc") and rec.get("heads") == 1 and persons == 1
              and not rec.get("others") and rec.get("names")
              and kind[b] not in ("system", "caption")):
            solo[b] = str(rec["names"][0])
    at = {b: i for i, b in enumerate(order)}

    def resolve(span) -> Optional[str]:
        idx = [at.get(_base(f)) for f in span or []]
        if not idx or None in idx:
            return None
        lo, hi = min(idx), max(idx)
        for step in (-1, 1):
            k, n = (lo if step < 0 else hi) + step, 0
            while 0 <= k < len(order) and n < fold_reach and kind[order[k]] != "story":
                lo, hi, k, n = min(lo, k), max(hi, k), k + step, n + 1
        got = {solo.get(b) for b in order[lo:hi + 1]}
        if None in got:
            return None
        got.discard("")
        return got.pop() if len(got) == 1 else None
    return resolve


def enforce_actor_handles(beat, figures_by_file, noun_map, protagonist_names,
                          ledger=None, spoken=None, kinds=None, names=None,
                          solo_mc=None):
    """Deterministic identity gate — positive evidence only (2026-10-10).

    Two rules over each segment's subject-position, singular actor-nouns
    (pattern authority shared with prep_qa: cast_identity.subject_actor_nouns):
      * DEAD ACTOR (story ledger, 2026-07-20): a noun whose cast members are
        all dead by this beat is rewritten to who the span shows — the single
        resolved figure, the ledger's single actor, or a neutral handle;
      * SOLO PROTAGONIST: when solo_mc(span) (solo_mc_resolver) says the
        picture shows only the protagonist on every panel the line voices, a
        DESCRIPTIVE noun for someone else ("the guard") becomes his handle.
    A proper name the writer wrote ("Namwoon", "the Book of Command") is
    never rewritten: 88% from the image is not enough to overrule a name the
    writer read on the page; prep_qa's actor_mismatch reports it instead.

    Removed (measured right ~1 time in 4 over 54 graded rewrites, 2026-10-09):
    re-pointing protagonist handles, rewriting toward keyword-resolved
    figures, neutral handles on all-unknown spans, the ledger tie-break
    outside dead actors, and the old/replace/shadow name modes. Without an
    identity manifest the gate rewrites nothing but dead actors.
    Returns 'old -> new' descriptions; lines are edited in place via
    write_segment_lines (spans untouched)."""
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
    rewrites: List[str] = []
    lines = [s["line"] or "" for s in segs]
    new_lines = list(lines)
    for i, s in enumerate(segs):
        line = new_lines[i]
        if not line:
            continue
        span = s["span"] or []
        # A SYSTEM CARD is text on a screen, not a claim about who is drawn
        # (ORV Ep210 "NAME: DOKJA KIM" became "Name: the blue digital").
        if span and kinds and all(
                str((kinds or {}).get(fn) or "") == "system" for fn in span):
            continue
        span_figs = [f for fn in span
                     for f in (figures_by_file.get(fn) or [])]
        span_actors = {str(a.get("actor"))
                       for fn in span for a in actions_by_file.get(fn, [])
                       if a.get("actor")
                       and a.get("actor") != "unclear"} - dead_now
        named = {f["name"] for f in span_figs
                 if f.get("name") and f["name"] != "unknown"}
        solo = solo_mc(span) if solo_mc else None

        def _dead_repl() -> str:
            if solo:
                return _figure_handle(solo, spoken)
            if len(named) == 1:
                return _figure_handle(next(iter(named)), spoken)
            if not span_figs:
                return ""
            if not named:
                return _neutral_from_evidence(span_figs)
            if len(span_actors) == 1:         # ledger breaks the tie
                return _figure_handle(next(iter(span_actors)), spoken)
            return ""                         # multi-figure: ambiguous

        for noun, members, plural in subject_actor_nouns_ex(line, noun_map):
            if plural:
                continue                      # the actor_count heal net's job
            if bool(members) and members <= dead_now:
                if named & protagonist_names and members & named:
                    continue
                repl, why = _dead_repl(), " (dead actor)"
            elif solo and solo not in members:
                repl, why = _figure_handle(solo, spoken), ""
            else:
                continue
            if not repl or noun in repl.lower():
                continue                      # ambiguous / no-op rewrite
            pat = re.compile(
                r"\b(?:(?P<art>our|the|a|an)\s+)?(?P<noun>" + re.escape(noun)
                + r")(?P<poss>'s)?\b", re.IGNORECASE)
            mo = pat.search(line)
            if not mo:
                continue
            # one word of a multi-word name ("The Book of Command"): swapping it
            # alone left "Choohino of Command" (TT ch102) — the name is the unit
            full, proper = _full_name_at(
                line, mo.start("noun"), mo.end("noun"), members, names)
            if proper or (not why and not mo.group("art")
                          and line[mo.start("noun")].isupper()):
                continue                      # a proper name: never rewritten
            if full is not None:
                mo = full
            new = (line[:mo.start()] + repl + (mo.group("poss") or "")
                   + line[mo.end():])
            if new != line:
                rewrites.append(f"'{noun}' -> {repl!r}{why}")
                line = new
        new_lines[i] = line
    if new_lines != lines and all(x.strip() for x in new_lines):
        write_segment_lines(beat, new_lines)
    return rewrites
