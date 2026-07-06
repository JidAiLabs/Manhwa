#!/usr/bin/env python3
"""span_align.py — deterministic narration<->panel span alignment (one authority).

The 73-segment human vision review of Nano ch1 (2026-07-06) found the DOMINANT
narration defect class was a ONE-PANEL OFFSET in action runs: a line leads or
lags its span by one panel (an impact line voiced over the pre-impact panel;
an "eyes widen" line voiced one panel after the eyes). Per the prose-first
doctrine, spans are derived IN CODE by the splitter — so the fix is also code:
an affinity score between a segment's line and candidate windows {assigned,
shifted -1, shifted +1} over the panels' UNDERSTANDING records, and

  * ``span_align_pass`` — the splitter post-pass: shift a boundary only when
    the shifted placement STRICTLY dominates by a margin AND every span
    invariant holds (contiguous partition, non-empty, cap, system solos
    untouched, no neighbor line's affinity damaged);
  * ``window_affinities`` — the SAME scoring, exposed for prep_qa's
    ``narration_offset`` check, so the splitter and QA can never disagree.

Also home of the impact lexicon (moved from prep_qa so both this module and
prep_qa share it without a circular import; prep_qa re-exports it).

Pure functions, deterministic, no I/O, no model calls.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# impact lexicon (single authority — prep_qa imports from here)
# ---------------------------------------------------------------------------
# Word FORMS a narration line uses when it actually narrates a strike/stab/
# blow/wound. Full-form list on purpose (no stemming surprises). Generic
# violent-contact senses only; ambiguous motion senses ("ran through the
# market") are NOT included. Over-matching can only SUPPRESS the
# impact_mismatch flag, never create one — the safe direction for a gate.
_IMPACT_LEXEMES = frozenset({
    "strike", "strikes", "striking", "struck",
    "stab", "stabs", "stabbed", "stabbing",
    "pierce", "pierces", "pierced", "piercing",
    "slash", "slashes", "slashed", "slashing",
    "blow", "blows",
    "hit", "hits", "hitting",
    "smash", "smashes", "smashed", "smashing",
    "crash", "crashes", "crashed", "crashing",
    "impact", "impacts", "impacted", "impacting",
    "plunge", "plunges", "plunged", "plunging",
    "thrust", "thrusts", "thrusting",
    "impale", "impales", "impaled", "impaling",
    "skewer", "skewers", "skewered", "skewering",
    "slam", "slams", "slammed", "slamming",
    "punch", "punches", "punched", "punching",
    "kick", "kicks", "kicked", "kicking",
    "clash", "clashes", "clashed", "clashing",
    "shatter", "shatters", "shattered", "shattering",
    "burst", "bursts", "bursting",
    "explode", "explodes", "exploded", "exploding", "explosion", "explosive",
    "gash", "gashes", "gashed", "gashing",
    "wound", "wounds", "wounded", "wounding",
    "blood", "bloody", "bloodied",
    "bleed", "bleeds", "bleeding", "bled",
    "cut", "cuts", "cutting",
    "sever", "severs", "severed", "severing",
    "cleave", "cleaves", "cleaved", "cleaving", "cleft",
    "bash", "bashes", "bashed", "bashing",
    "pummel", "pummels", "pummeled", "pummelling", "pummeling",
    "drive the blade", "drives the blade", "drove the blade",
})
_IMPACT_LEXEME_RE = re.compile(
    r"\b(?:" + "|".join(sorted(re.escape(w) for w in _IMPACT_LEXEMES)) + r")\b",
    re.IGNORECASE)


def has_impact_lexeme(line: str) -> bool:
    """True when the narration line carries ANY impact-class word form."""
    return bool(_IMPACT_LEXEME_RE.search(str(line or "")))


# ---------------------------------------------------------------------------
# affinity: line <-> panel window, over understanding records
# ---------------------------------------------------------------------------
# Shift a boundary only when the shifted window BEATS the assigned one by at
# least this margin (affinity is 0..~1.5: token-overlap fraction + impact
# term). Conservative on purpose — a missed fix costs one mediocre pacing
# moment; a wrong shift actively misaligns a good line.
SPAN_ALIGN_MARGIN = 0.35
# Both-sides impact agreement (line narrates a strike AND the window carries a
# detector/understanding impact signal) is the strongest placement evidence.
_IMPACT_BONUS = 0.5
# One-sided impact (line narrates a strike over a window with no impact
# signal, or vice versa) argues the line belongs elsewhere.
_IMPACT_PENALTY = 0.25

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_STOPWORDS = frozenset({
    "the", "and", "but", "for", "nor", "not", "with", "from", "into", "onto",
    "over", "under", "out", "off", "down", "then", "than", "while", "when",
    "where", "who", "whom", "what", "why", "how", "his", "her", "their",
    "its", "him", "them", "she", "they", "this", "that", "these", "those",
    "are", "was", "were", "been", "being", "has", "have", "had", "does",
    "did", "will", "would", "could", "should", "can", "may", "might", "must",
    "just", "even", "only", "very", "too", "also", "still", "yet", "now",
    "here", "there", "all", "some", "any", "each", "one", "two", "you",
    "your", "our", "guy", "boy", "man", "woman", "character", "figure",
    "panel", "scene", "moment", "suddenly", "before", "after", "about",
    "like", "gets", "goes", "comes", "begins", "starts", "himself",
    "herself", "itself", "toward", "towards", "against", "around",
    "through", "along",
})


def _stem(word: str) -> str:
    """Light deterministic stemmer — enough to match 'dragging'~'drags',
    'strikes'~'striking', 'eyes'~'eye'. Not linguistic; impact wording is
    matched by the full-form lexicon above, not by stems."""
    w = word.lower().rstrip("'")
    if w.endswith("'s"):
        w = w[:-2]
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if len(w) >= 4 and w[-1] == w[-2]:      # dragg -> drag, stabb -> stab
        w = w[:-1]
    if len(w) >= 4 and w.endswith("e"):     # strike -> strik (= striking's stem)
        w = w[:-1]
    return w


def _tokens(text: Any) -> set:
    """Significant word stems of a text blob."""
    out = set()
    for w in _WORD_RE.findall(str(text or "")):
        if len(w) < 3 or w.lower() in _STOPWORDS:
            continue
        out.add(_stem(w))
    return out


def _panel_blob(rec: Dict[str, Any]) -> str:
    """The understanding text a line can legitimately overlap with: subjects +
    action + description + setting + SFX lettering. Dialogue included — a line
    often paraphrases the span's speech."""
    rec = rec or {}
    parts = [str(rec.get(k) or "") for k in
             ("action", "description", "setting", "dialogue", "sfx_text")]
    subs = rec.get("subjects")
    if isinstance(subs, (list, tuple)):
        parts.extend(str(s) for s in subs)
    return " ".join(p for p in parts if p)


def panel_has_impact(rec: Dict[str, Any]) -> bool:
    """Impact signal on ONE panel: the detector stamp (impact_sfx.present) or
    impact wording in the understanding's action/SFX fields."""
    rec = rec or {}
    if (rec.get("impact_sfx") or {}).get("present"):
        return True
    blob = " ".join(str(rec.get(k) or "")
                    for k in ("action", "sfx_text", "strikes_or_weapons"))
    return has_impact_lexeme(blob)


def affinity(line: str, window: Sequence[str],
             u_by_file: Dict[str, Dict[str, Any]]) -> float:
    """How well `line` fits the panels in `window` (understanding records via
    `u_by_file`). overlap in [0,1] (fraction of the line's significant stems
    found in the window's understanding) + a signed impact-correspondence
    term. Deterministic; empty inputs score 0."""
    line_toks = _tokens(line)
    if not line_toks or not window:
        return 0.0
    recs = [(u_by_file or {}).get(f) or {} for f in window]
    win_toks: set = set()
    for rec in recs:
        win_toks |= _tokens(_panel_blob(rec))
    score = len(line_toks & win_toks) / len(line_toks)
    line_imp = has_impact_lexeme(line)
    win_imp = any(panel_has_impact(rec) for rec in recs)
    if line_imp and win_imp:
        score += _IMPACT_BONUS
    elif line_imp != win_imp:
        score -= _IMPACT_PENALTY
    return score


def _window(files: List[str], start: int, n: int, sys_set: set
            ) -> Optional[List[str]]:
    """files[start:start+n] as a candidate placement, or None when it runs off
    the beat or would cross a system panel (cards are walls, never candidate
    art for a story line)."""
    if start < 0 or start + n > len(files):
        return None
    win = files[start:start + n]
    if any(f in sys_set for f in win):
        return None
    return win


def _system_files(files: Sequence[str], kinds: Dict[str, Any]) -> set:
    """Files whose understanding panel_kind is 'system' — the SINGLE
    authority for span_align's system-wall checks. window_affinities' ±1
    window gate and _package's cascade wall (system spans are never crossed,
    however far the deficit rotates past that immediate window) both call
    this so they can never disagree about which files are cards."""
    return {f for f in files
            if str((kinds or {}).get(f) or "").lower() == "system"}


def window_affinities(segments: Sequence[Dict[str, Any]], i: int,
                      files: List[str], kinds: Dict[str, Any],
                      u_by_file: Dict[str, Dict[str, Any]]
                      ) -> Tuple[float, Optional[float], Optional[float]]:
    """(own, shifted-1, shifted+1) affinities for segment ``i`` — THE shared
    scoring both the splitter post-pass and prep_qa's narration_offset check
    use, so they can never disagree. ``files`` is the beat's ordered panel
    list (the concatenation of its spans). A system-solo segment scores
    (own, None, None) — cards never move."""
    sys_set = _system_files(files, kinds)
    span = [str(f) for f in (segments[i].get("span") or [])]
    line = str(segments[i].get("line") or "")
    own = affinity(line, span, u_by_file)
    if (not span or any(f in sys_set for f in span)
            or span[0] not in files):
        return own, None, None
    start = files.index(span[0])
    n = len(span)
    minus = _window(files, start - 1, n, sys_set)
    plus = _window(files, start + 1, n, sys_set)
    return (own,
            affinity(line, minus, u_by_file) if minus is not None else None,
            affinity(line, plus, u_by_file) if plus is not None else None)


# ---------------------------------------------------------------------------
# the splitter post-pass
# ---------------------------------------------------------------------------

def _invariants_ok(spans: List[List[str]], files: List[str],
                   kinds: Dict[str, Any], span_cap: int) -> bool:
    """The span invariants the splitter's validator enforces (contiguous
    partition, non-empty, cap, system solo) — checked structurally here so a
    package can never be applied broken even when no validator is passed."""
    if any(not s for s in spans):
        return False
    if [f for s in spans for f in s] != list(files):
        return False
    for s in spans:
        if len(s) > span_cap:
            return False
        for f in s:
            if (str((kinds or {}).get(f) or "").lower() == "system"
                    and len(s) > 1):
                return False
    return True


def _package(spans: List[List[str]], i: int, direction: int, span_cap: int,
             kinds: Dict[str, Any]) -> Optional[List[List[str]]]:
    """Candidate spans after a one-panel boundary shift at segment ``i``.

    direction +1: seg i absorbs the NEXT segment's head panel (the voice keeps
    speaking while the art advances one panel sooner); the deficit cascades
    right through singleton neighbors until a multi-panel span absorbs it.
    direction -1: mirror image toward the run start. None when the cascade
    runs off the end, the grown span would exceed the cap, OR the cascade
    would touch a system-card span — system solos are WALLS in the cascade,
    never crossed however far the deficit rotates past the immediate ±1
    window that window_affinities checks."""
    out = [list(s) for s in spans]
    if direction not in (-1, 1) or not (0 <= i < len(out)):
        return None
    if len(out[i]) + 1 > span_cap:
        return None
    sys_set = _system_files([f for s in spans for f in s], kinds)
    if any(f in sys_set for f in out[i]):
        return None
    if direction == 1:
        j = i + 1
        if j >= len(out) or any(f in sys_set for f in out[j]):
            return None
        out[i].append(out[j][0])
        while j < len(out):
            if len(out[j]) >= 2:
                out[j] = out[j][1:]
                return out
            if j + 1 >= len(out):
                return None                      # singleton at the run end
            if any(f in sys_set for f in out[j + 1]):
                return None                       # system wall — never crossed
            out[j] = [out[j + 1][0]]
            j += 1
        return None
    j = i - 1
    if j < 0 or any(f in sys_set for f in out[j]):
        return None
    out[i].insert(0, out[j][-1])
    while j >= 0:
        if len(out[j]) >= 2:
            out[j] = out[j][:-1]
            return out
        if j - 1 < 0:
            return None
        if any(f in sys_set for f in out[j - 1]):
            return None                           # system wall — never crossed
        out[j] = [out[j - 1][-1]]
        j -= 1
    return None


def span_align_pass(segments: List[Dict[str, Any]], files: Sequence[str],
                    kinds: Dict[str, Any],
                    u_by_file: Dict[str, Dict[str, Any]], *,
                    margin: float = SPAN_ALIGN_MARGIN, span_cap: int = 4,
                    validate=None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Deterministic post-pass over freshly derived segments: fix one-panel
    line/art offsets by shifting span boundaries. Returns (segments, logs) —
    the input list is never mutated; logs carry one line per applied shift.

    A shift at segment i is applied ONLY when ALL hold:
      * the shifted same-size window beats the assigned one by >= ``margin``
        (and beats the other direction);
      * after the boundary shift (seg i grows by the absorbed panel; the
        deficit cascades through singleton neighbors; the first multi-panel
        span absorbs it) segment i's affinity still improves by >= margin and
        NO other affected segment's affinity gets worse;
      * every span invariant holds (partition/cap/solo — plus the caller's
        full validator when ``validate`` is passed: validate(segs) -> [] ok).
    One left-to-right pass; each segment moves at most once (no ping-pong).
    System solos never move and are never crossed."""
    segs = [{**s, "span": [str(f) for f in (s.get("span") or [])]}
            for s in (segments or [])]
    files = [str(f) for f in (files or [])]
    if not segs or not files:
        return segs, []
    logs: List[str] = []
    locked: set = set()
    i = 0
    while i < len(segs):
        if i in locked:
            i += 1
            continue
        own, minus, plus = window_affinities(segs, i, files, kinds, u_by_file)
        best_dir = 0
        if plus is not None and plus >= own + margin and plus > (
                minus if minus is not None else float("-inf")):
            best_dir = 1
        elif minus is not None and minus >= own + margin and minus > (
                plus if plus is not None else float("-inf")):
            best_dir = -1
        if not best_dir:
            i += 1
            continue
        old_spans = [s["span"] for s in segs]
        new_spans = _package(old_spans, i, best_dir, span_cap, kinds)
        if new_spans is None or not _invariants_ok(new_spans, files, kinds,
                                                   span_cap):
            i += 1
            continue
        affected = [k for k in range(len(segs))
                    if new_spans[k] != old_spans[k]]
        if locked & set(affected):
            i += 1
            continue
        ok = True
        for k in affected:
            line_k = str(segs[k].get("line") or "")
            before = affinity(line_k, old_spans[k], u_by_file)
            after = affinity(line_k, new_spans[k], u_by_file)
            if k == i:
                ok = ok and after >= before + margin
            else:
                ok = ok and after >= before
        if ok and validate is not None:
            candidate = [{**s, "span": list(new_spans[k])}
                         for k, s in enumerate(segs)]
            ok = not validate(candidate)
        if not ok:
            i += 1
            continue
        for k in affected:
            segs[k]["span"] = list(new_spans[k])
            locked.add(k)
        logs.append(
            f"segment {i} shifted {'+1' if best_dir == 1 else '-1'} "
            f"(affinity {own:.2f} -> "
            f"{(plus if best_dir == 1 else minus):.2f}); "
            f"boundaries moved on segments {affected}")
        i = max(affected) + 1
    return segs, logs
