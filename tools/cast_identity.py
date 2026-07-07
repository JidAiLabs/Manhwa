#!/usr/bin/env python3
"""cast_identity.py — deterministic cast-grounded FIGURE resolution.

SINGLE AUTHORITY for "who is actually in this panel": the round-2 vision
review's dominant residual (~6 findings) was identity misattribution — "the
assassin draws his steel" over Prince Cheon's counter-draw, a dying prince's
eye narrated as "an assassin's eye", a departed assassin given the
descendant's inner thoughts. The writer named actors from vibes because its
payload carried only generic subjects ("a person in light robes").

manifest.cast.json (cast_builder, beated stage) carries each character's
appearance (`visual_description`) + names/aliases. Understanding records
(panel_understand, grouped stage — runs BEFORE cast exists) carry per-panel
subjects/description text. This module joins the two AT READ TIME with
deterministic keyword evidence — no model call (the failure mode being killed
IS model misattribution), no artifact mutation (stamping understood.json at
beated would invert the groups←understood freshness edge in studio/deps.py).

Consumers:
  - gemini_narrative_pass._pack_group_payload → per-panel `figures` list, so
    the narrator names actors from ground truth;
  - prep_qa.actor_mismatch_flags → a line whose actor-noun contradicts its
    span's resolved figures (ERROR, heal-target — measured before blocking).

Calibrated on the real Nano Machine ch1 cast: protagonist = light/white/grey
clothing + purple hair (aliases "Prince Cheon", "descendant"); assassins =
dark brown HOODED cloaks + face masks + swords; stranger = blue/white HOODIE
(hoodie is stranger-exclusive and deliberately NOT collapsed into hood/hooded).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")

# tokens that carry no identity evidence (function words + hedges cast_builder
# tends to emit: "possibly white or grey clothing", "often seen wielding")
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "with", "in", "on", "at", "to",
    "from", "that", "this", "his", "her", "their", "its", "who", "whose",
    "is", "are", "was", "were", "be", "been", "being", "as", "by", "for",
    "often", "seen", "possibly", "looking", "wearing", "wears", "worn",
    "appears", "appearing", "mentioned", "suddenly", "only", "reveals",
    "revealing", "matching", "carrying", "wielding", "becomes", "colored",
    "coloured", "very", "some", "any", "no", "not", "but", "into", "over",
})

# generic person-words: appearance evidence they are NOT, and as narration
# nouns they are sanctioned neutral handles ("our guy", "the man") — never
# noun-map keys (mapping "guy" → stranger would flag the protagonist's
# sanctioned stand-in as a mismatch).
_GENERIC_PERSON = frozenset({
    "guy", "guys", "man", "men", "woman", "women", "person", "people",
    "figure", "figures", "character", "characters", "one", "individual",
    "boy", "girl", "male", "female",
})

# generic descriptors excluded from the NOUN map (adjectives / hedge words
# that ride cast names: "unnamed assassin", "mysterious stranger", "the
# strange guy", "dying ancestor") — generic English, not series content.
_GENERIC_DESCRIPTOR = frozenset({
    "unnamed", "mysterious", "strange", "young", "old", "elderly", "dying",
    "our", "unknown", "little", "big", "tall", "short",
})

# spelling/variant normalization (deterministic, tiny)
_VARIANTS = {"grey": "gray", "reddish": "red", "blackish": "black",
             "whitish": "white", "greyish": "gray", "grayish": "gray"}

# garment-class nouns → every member also emits the class marker "garment",
# so "light robes" (understanding) meets "light-colored clothing" (cast).
# "hoodie" is deliberately ALSO kept as its own raw token (stranger-exclusive
# vs the assassins' "hooded" cloaks).
_GARMENT = frozenset({
    "robe", "clothing", "clothes", "cloak", "tunic", "garment", "garments",
    "attire", "outfit", "uniform", "gown", "coat", "jacket", "hoodie",
    "armor", "armour", "dress", "shirt",
})

# color adjectives eligible for the color→garment pairing bonus
_COLORS = frozenset({
    "white", "black", "gray", "red", "blue", "green", "purple", "brown",
    "dark", "light", "pale", "crimson", "golden", "yellow", "silver",
})


def _singular(tok: str) -> str:
    """Cheap, safe singularization: assassins→assassin, robes→robe. The
    guard list keeps non-plural s-enders whole (mysterious, glass, focus,
    basis) — 'mysteriou' once leaked into the noun map as a matchable key."""
    if (len(tok) > 3 and tok.endswith("s")
            and not tok.endswith(("ss", "us", "is", "ous"))):
        return tok[:-1]
    return tok


def _norm(tok: str) -> str:
    tok = tok.lower().rstrip("'")
    if tok.endswith("'s"):
        tok = tok[:-2]
    tok = _singular(tok)
    return _VARIANTS.get(tok, tok)


def _tokens(text: str) -> List[str]:
    return [_norm(t) for t in _WORD_RE.findall(str(text or ""))]


def _informative(toks: Sequence[str]) -> List[str]:
    return [t for t in toks if t and t not in _STOPWORDS]


def _color_garment_pairs(toks: Sequence[str], window: int = 6
                         ) -> Set[Tuple[str, str]]:
    """(color, 'garment') associations: a color token within *window*
    informative tokens BEFORE a garment token ("dark brown hooded cloaks" →
    dark→garment, brown→garment)."""
    pairs: Set[Tuple[str, str]] = set()
    for i, t in enumerate(toks):
        if t in _GARMENT:
            for c in toks[max(0, i - window):i]:
                if c in _COLORS:
                    pairs.add((c, "garment"))
    return pairs


def _members(cast: Any) -> List[Dict[str, Any]]:
    """Accept the full manifest dict OR the bare members list; [] fail-soft."""
    if isinstance(cast, dict):
        cast = cast.get("cast")
    return [m for m in (cast or []) if isinstance(m, dict)]


def _name_tokens(member: Dict[str, Any]) -> Set[str]:
    """Identity NOUNS for one member: canonical_name + aliases + id words,
    minus stopwords / generic person-words / generic descriptors."""
    raw: List[str] = []
    raw += _tokens(member.get("canonical_name") or "")
    for a in member.get("aliases") or []:
        raw += _tokens(a)
    raw += _tokens(str(member.get("id") or "").replace("_", " "))
    return {t for t in raw
            if t not in _STOPWORDS and t not in _GENERIC_PERSON
            and t not in _GENERIC_DESCRIPTOR and len(t) > 1}


def cast_profiles(cast: Any) -> List[Dict[str, Any]]:
    """[{name, name_tokens, appearance, pairs}] per cast member.

    appearance = informative tokens of visual_description (garment-class
    members add the 'garment' marker); pairs = color→garment associations."""
    profiles: List[Dict[str, Any]] = []
    for m in _members(cast):
        name = str(m.get("canonical_name") or m.get("id") or "").strip()
        if not name:
            continue
        toks = _informative(_tokens(m.get("visual_description") or ""))
        appearance = set(toks)
        appearance.update("garment" for t in toks if t in _GARMENT)
        profiles.append({
            "name": name,
            "name_tokens": _name_tokens(m),
            "appearance": appearance,
            "pairs": _color_garment_pairs(toks),
        })
    return profiles


def _subject_tokens(text: str) -> Set[str]:
    toks = _informative(_tokens(text))
    out = set(toks)
    out.update("garment" for t in toks if t in _GARMENT)
    return out


def _score(profile: Dict[str, Any], text: str) -> Tuple[float, List[str]]:
    """(score, evidence tokens) of *text* against one profile. Name-token
    hits dominate (10 each); a color→garment pair match adds 2 on top; each
    shared appearance token = 1 (counted once, name hits excluded)."""
    toks = _subject_tokens(text)
    stream = _informative(_tokens(text))
    ev: List[str] = []
    score = 0.0
    name_hits = sorted(profile["name_tokens"] & toks)
    for nt in name_hits:
        score += 10.0
        ev.append(nt)
    pair_hits = profile["pairs"] & _color_garment_pairs(stream)
    for c, _g in sorted(pair_hits):
        score += 2.0
        ev.append(f"{c}+garment")
    shared = sorted((profile["appearance"] & toks) - set(name_hits))
    score += float(len(shared))
    ev += shared
    return score, ev


_PERSONISH = _GENERIC_PERSON | {"stranger", "warrior", "assassin", "prince",
                                "king", "lord", "master", "servant"}


def _looks_person(text: str) -> bool:
    toks = set(_tokens(text))
    return bool(toks & _PERSONISH) or bool(toks & _GARMENT)


def resolve_figures(understanding: Optional[Dict[str, Any]],
                    profiles: Sequence[Dict[str, Any]]
                    ) -> List[Dict[str, str]]:
    """[{cast_name|'unknown', evidence}] for ONE understood panel record.

    Each `subjects[]` entry resolves independently (a subject string describes
    ONE figure); the panel's description/action are scanned for NAME tokens
    only (they mix several figures' features — appearance-matching the blob
    would cross-attribute). Resolution requires score >= 2 AND a strict
    margin of 1 over the runner-up; ties resolve to 'unknown', never a guess
    (the failure mode being killed is misattribution)."""
    u = understanding or {}
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def _add(name: str, evidence: str) -> None:
        key = name if name != "unknown" else f"unknown:{evidence}"
        if key not in seen:
            seen.add(key)
            out.append({"name": name, "evidence": evidence})

    subjects = [str(s).strip() for s in (u.get("subjects") or [])
                if str(s).strip()]
    by_name = {p["name"]: p for p in profiles}
    for subj in subjects:
        scored = sorted(((_score(p, subj), p["name"]) for p in profiles),
                        key=lambda t: (-t[0][0], t[1]))
        if scored:
            (best, ev), name = scored[0]
            runner = scored[1][0][0] if len(scored) > 1 else 0.0
            if best >= 2.0 and best - runner >= 1.0:
                _add(name, f"{subj[:60]} ~ {'+'.join(ev[:4])}")
                continue
            # SAME-FACTION tie (the assassin leader vs the assassin group —
            # near-identical appearance BY DESIGN): when every tied top
            # scorer shares a name-token, they are one narrative identity;
            # resolve to the first tied member (cast order) instead of
            # throwing the identity away as 'unknown'.
            if best >= 2.0:
                tied = [n for (s, _e), n in scored if s == best]
                if len(tied) > 1:
                    common = set.intersection(
                        *(by_name[n]["name_tokens"] for n in tied))
                    if common:
                        _add(name, f"{subj[:60]} ~ faction:"
                                   f"{'+'.join(sorted(common)[:2])}")
                        continue
        if _looks_person(subj):
            _add("unknown", subj[:60])

    blob = " ".join(str(u.get(k) or "") for k in ("description", "action",
                                                  "dialogue"))
    blob_toks = set(_tokens(blob))
    for p in profiles:
        hits = sorted(p["name_tokens"] & blob_toks)
        if hits:
            _add(p["name"], f"named: {'+'.join(hits[:3])}")
    return out


def resolve_figures_by_file(understood_obj: Any, cast: Any
                            ) -> Dict[str, List[Dict[str, str]]]:
    """{scene_file: figures} over a whole manifest.panels.understood.json.
    {} when either side is missing/empty — consumers stay silent."""
    profiles = cast_profiles(cast)
    if not profiles:
        return {}
    out: Dict[str, List[Dict[str, str]]] = {}
    for p in ((understood_obj or {}).get("panels") or []):
        if isinstance(p, dict) and p.get("scene_file"):
            out[str(p["scene_file"])] = resolve_figures(p, profiles)
    return out


def actor_noun_map(cast: Any) -> Dict[str, Set[str]]:
    """{actor-noun: set(canonical_names)} derived from the cast manifest —
    NO hardcoded per-series word list. From the real Nano ch1 cast this
    yields assassin→{unnamed assassin, the assassins}, prince/cheon/
    descendant→{our protagonist}, stranger→{unnamed stranger}, …"""
    nouns: Dict[str, Set[str]] = {}
    for m in _members(cast):
        name = str(m.get("canonical_name") or m.get("id") or "").strip()
        if not name:
            continue
        for t in _name_tokens(m):
            nouns.setdefault(t, set()).add(name)
    return nouns


_SENT_SPLIT_RE = re.compile(r"[.!?…]+")


def subject_actor_nouns(line: str, noun_map: Dict[str, Set[str]]
                        ) -> List[Tuple[str, Set[str]]]:
    """Actor-nouns of *line* used in SUBJECT position: within the first 5
    word-tokens of a sentence, or possessive-marked within the first 7
    ("an assassin's eye…"). Late mentions are usually objects/off-panel
    references ("their blades meant for the ancestor") — deliberately not
    flagged; this is the precision lever that keeps actor_mismatch a
    measured heal-target, not an FP fountain."""
    hits: List[Tuple[str, Set[str]]] = []
    seen: Set[str] = set()
    for sent in _SENT_SPLIT_RE.split(str(line or "")):
        raw = _WORD_RE.findall(sent)
        for i, w in enumerate(raw[:7]):
            possessive = w.lower().endswith("'s") or w.lower().endswith("s'")
            if i >= 5 and not possessive:
                continue
            t = _norm(w)
            if t in noun_map and t not in seen:
                seen.add(t)
                hits.append((t, set(noun_map[t])))
    return hits
