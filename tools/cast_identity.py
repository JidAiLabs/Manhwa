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

# worn-item words (and their -ed adjectives) that describe APPEARANCE, not
# identity, when they appear inside a cast NAME ("the hooded leader", "the
# masked assassin" identify by leader/assassin — job-48 g0014/15: the
# light-blue-hooded arrival name-hit the dark-cloaked leader on 'hooded'
# (+10) and the identity gate then rewrote the protagonist's transformation
# reveal to the villain).
_WORN = frozenset({"hood", "hooded", "mask", "masked", "veil", "veiled",
                   "cloaked", "robed", "armored", "armoured", "helmeted"})


def _is_appearance_word(t: str) -> bool:
    if t in _COLORS or t in _GARMENT or t in _WORN:
        return True
    return t.endswith("ed") and (t[:-2] in _GARMENT or t[:-2] in _WORN)


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


# color CLASSES for evidence matching (2026-08-18, nano ch1 job 149): the cast
# builder wrote "light-colored tunic" for the prince while every panel subject
# says "white robe" — light/white/pale are one class, dark/black another, so a
# vocabulary coin-flip can no longer hand the prince to the blue-haired master.
_COLOR_CLASS = {"white": "light", "pale": "light", "light": "light",
                "black": "dark", "dark": "dark"}


def _color_class(c: str) -> str:
    return _COLOR_CLASS.get(c, c)


_SHADE = frozenset({"light", "dark", "pale"})


def _drop_shade_modifiers(toks: Sequence[str]) -> List[str]:
    """'light blue jacket' / 'dark red cloak': light/dark/pale directly
    before another color are SHADE modifiers, not the garment's color — drop
    them so 'light blue' cannot pair with a 'white' (light-class) tunic."""
    out: List[str] = []
    for i, t in enumerate(toks):
        if t in _SHADE and i + 1 < len(toks) and toks[i + 1] in _COLORS and toks[i + 1] not in _SHADE:
            continue
        out.append(t)
    return out


_HAIR = frozenset({"hair", "haired"})


def _color_garment_pairs(toks: Sequence[str], window: int = 6
                         ) -> Set[Tuple[str, str]]:
    """(color-class, 'garment') associations: a color token within *window*
    informative tokens BEFORE a garment token ("dark brown hooded cloaks" →
    dark→garment, brown→garment)."""
    pairs: Set[Tuple[str, str]] = set()
    toks = _drop_shade_modifiers(list(toks))
    for i, t in enumerate(toks):
        if t in _GARMENT:
            for c in toks[max(0, i - window):i]:
                if c in _COLORS:
                    pairs.add((_color_class(c), "garment"))
    return pairs


def _hair_colors(toks: Sequence[str], window: int = 4) -> Set[str]:
    """Color classes said of HAIR ("messy, long dark purple hair" → {dark,
    purple}; "blue-haired" → {blue}). Hair color is stable identity evidence
    (clothes change, hair mostly doesn't) — matched at +3, clashed at -3."""
    out: Set[str] = set()
    # hair keeps shade words: subjects say "dark hair" for "dark purple hair"
    for i, t in enumerate(toks):
        if t in _HAIR:
            for c in toks[max(0, i - window):i]:
                if c in _COLORS:
                    out.add(_color_class(c))
    return out


def _members(cast: Any) -> List[Dict[str, Any]]:
    """Accept the full manifest dict OR the bare members list; [] fail-soft."""
    if isinstance(cast, dict):
        cast = cast.get("cast")
    return [m for m in (cast or []) if isinstance(m, dict)]


def _name_tokens(member: Dict[str, Any]) -> Set[str]:
    """Identity NOUNS for one member: canonical_name + aliases, minus
    stopwords / generic person-words / generic descriptors.

    `id` words are deliberately EXCLUDED (round-2 review, class C): ids are
    pipeline slugs (assassin_group, assassin_leader) whose structural parts
    ("group", "leader") are role/count descriptors, not identity evidence —
    "a group of villagers" must never hard-claim the assassins, and a line
    naming an unrelated "leader" must never stamp the assassin leader in.
    canonical_name/aliases already carry the real nouns (assassin, member,
    prince, cheon, …); cast_builder's schema requires canonical_name
    non-empty, so no evidence is lost by dropping the id."""
    raw: List[str] = []
    raw += _tokens(member.get("canonical_name") or "")
    for a in member.get("aliases") or []:
        raw += _tokens(a)
    return {t for t in raw
            if t not in _STOPWORDS and t not in _GENERIC_PERSON
            and t not in _GENERIC_DESCRIPTOR and not _is_appearance_word(t)
            and len(t) > 1}


def _specific(toks: Sequence[str]) -> List[str]:
    """*toks* minus generic person/descriptor words — appearance evidence
    they are NOT (module comment above): 'young' and 'man' must never by
    themselves clear the resolution score bar."""
    return [t for t in toks
            if t not in _GENERIC_PERSON and t not in _GENERIC_DESCRIPTOR]


def cast_profiles(cast: Any) -> List[Dict[str, Any]]:
    """[{name, name_tokens, appearance, pairs}] per cast member.

    appearance = informative, NON-GENERIC tokens of visual_description
    (garment-class members add the 'garment' marker); pairs = color→garment
    associations. Every point of `_score` traces back to a name hit, a
    pair, or one of these appearance tokens, so filtering generic words out
    here is what makes "score >= 2.0" mean ">= 1 piece of SPECIFIC evidence"
    — 'a young man' (both generic) must score 0, never hard-claim a cast
    member on vibes alone."""
    profiles: List[Dict[str, Any]] = []
    for m in _members(cast):
        name = str(m.get("canonical_name") or m.get("id") or "").strip()
        if not name:
            continue
        toks = _informative(_tokens(m.get("visual_description") or ""))
        specific = _specific(_drop_shade_modifiers(toks))
        appearance = {_color_class(t) for t in specific}
        appearance.update("garment" for t in specific if t in _GARMENT)
        # owner-declared traits this member NEVER shows (series registry
        # `not`, e.g. Dokja: ["glasses"]) — normalised exactly like
        # appearance so the two sides meet on the same token
        ftoks = _informative(_tokens(" ".join(
            str(x) for x in (m.get("not") or []))))
        forbid = {_color_class(t) for t in _specific(_drop_shade_modifiers(ftoks))}
        role = str(m.get("role") or "").strip().lower()
        raw_name = [t.lower() for t in _WORD_RE.findall(name)]
        profiles.append({
            "name": name,
            "name_tokens": _name_tokens(m),
            "appearance": appearance,
            "pairs": _color_garment_pairs(toks),
            "hair": _hair_colors(toks),
            "forbid": forbid,
            "role": role,
            # the FACTION member of a same-faction tie: role says so, or the
            # canonical name itself is plural ('the assassins') — cast_builder
            # often labels a group 'antagonist' instead of 'group'.
            "is_group": (role == "group" or any(
                len(t) > 3 and t.endswith("s") and _singular(t) != t
                for t in raw_name)),
        })
    return profiles


def _subject_tokens(text: str) -> Set[str]:
    """Non-generic informative tokens of *text* (+ the 'garment' class
    marker) — the SAME generic exclusion as cast_profiles' appearance set,
    so a subject built entirely of generic words ('a young man') can never
    share evidence with a profile."""
    specific = _specific(_drop_shade_modifiers(_informative(_tokens(text))))
    out = {_color_class(t) for t in specific}
    out.update("garment" for t in specific if t in _GARMENT)
    return out


# COLORED LIGHTING: a flashback / glow / silhouette / monochrome panel tints
# EVERYTHING one color ("a young person with glowing light blue hair and skin"
# was the purple-haired prince under a blue tint — nano ch1 p000019 — and it
# resolved to the blue-haired master, whose lines then swallowed the assassin
# leader's). Skin is never blue: when a subject says a color of SKIN, or
# glow/tint/silhouette/monochrome, its colors describe the light, not the
# person — color evidence (hair + garment pairs) is withheld; names still hit.
_TINT_RE = re.compile(
    r"\b(?:glow|glowing|glows|tint|tinted|silhouette|silhouetted|monochrome"
    r"|monochromatic|backlit|shadowed|greyscale|grayscale|sepia)\b"
    r"|\b(?:blue|red|green|purple|yellow|golden|orange|pink|light|dark|pale"
    r"|white|gray|grey|black|crimson|silver)(?:-|\s+)(?:\w+\s+)?skin\b",
    re.IGNORECASE)


_LIGHTING_TOKENS = frozenset({"glow", "glowing", "glows", "skin", "tint", "tinted",
                              "silhouette", "silhouetted", "monochrome", "backlit",
                              "shadowed", "surrounded", "energy", "aura", "light"})


def _color_evidence_valid(text: str) -> bool:
    return not _TINT_RE.search(str(text or ""))


def _score(profile: Dict[str, Any], text: str) -> Tuple[float, List[str]]:
    """(score, evidence tokens) of *text* against one profile. Name-token
    hits dominate (10 each); a color→garment pair match adds 2 on top; each
    shared appearance token = 1 (counted once, name hits excluded). Under
    colored lighting (see _TINT_RE) color evidence is withheld."""
    toks = _subject_tokens(text)
    stream = _informative(_tokens(text))
    ev: List[str] = []
    score = 0.0
    name_hits = sorted(profile["name_tokens"] & toks)
    for nt in name_hits:
        score += 10.0
        ev.append(nt)
    forbidden = sorted((profile.get("forbid") or set()) & toks)
    if forbidden:
        # the owner says this member never shows these: appearance alone can
        # not resolve through them (-5 = one pair + hair), a name still can
        score -= 5.0
        ev.append("not:" + "+".join(forbidden))
    if not _color_evidence_valid(text):
        # tinted panel: only names + non-color, non-lighting appearance
        # tokens count (glow/skin/the bare garment marker describe the light
        # or nothing identity-bearing)
        shared = sorted((profile["appearance"] & toks) - set(name_hits)
                        - _COLORS - {"light", "dark", "garment"} - _LIGHTING_TOKENS)
        score += float(len(shared))
        ev += shared + ["tinted"]
        return score, ev
    subj_pairs = _color_garment_pairs(stream)
    pair_hits = profile["pairs"] & subj_pairs
    for c, _g in sorted(pair_hits):
        score += 2.0
        ev.append(f"{c}+garment")
    # COLOR CLASH: both sides dress their garments in colors and share NONE —
    # strong mismatch evidence ('light blue hooded jacket' must not resolve
    # to the 'dark hooded cloak' leader). A penalty, not a veto: multi-outfit
    # characters stay resolvable via name/appearance dominance.
    if profile["pairs"] and subj_pairs and not pair_hits:
        score -= 3.0
        ev.append("color-clash")
    # HAIR color: stable identity evidence — one shared hair color +3, hair
    # colors on both sides sharing none -3 (a purple-haired subject is not
    # the light-blue-haired master, whatever robe he wears)
    subj_hair = _hair_colors(stream)
    prof_hair = profile.get("hair") or set()
    if prof_hair and subj_hair:
        if prof_hair & subj_hair:
            score += 3.0
            ev.append("hair:" + "+".join(sorted(prof_hair & subj_hair)))
        else:
            score -= 3.0
            ev.append("hair-clash")
    shared = sorted((profile["appearance"] & toks) - set(name_hits))
    score += float(len(shared))
    ev += shared
    return score, ev


# Person-denoting role/occupation nouns, used only to decide whether a subject
# string describes a PERSON (so an unresolved one is recorded as an unknown
# figure rather than ignored as a prop). Deliberately broad across genres —
# the earlier list was murim/fantasy-only, which read a doctor or a detective
# as scenery. Generic English, no series content.
_PERSONISH = _GENERIC_PERSON | {
    # family / relations
    "mother", "father", "son", "daughter", "brother", "sister", "child",
    "wife", "husband", "parent", "elder", "grandfather", "grandmother",
    # martial / fantasy
    "stranger", "warrior", "assassin", "prince", "princess", "king", "queen",
    "lord", "lady", "master", "servant", "knight", "swordsman", "fighter",
    "monk", "priest", "priestess", "mage", "wizard", "witch", "healer",
    "hunter", "ranger", "thief", "bandit", "mercenary", "guard", "soldier",
    "general", "commander", "captain", "chief", "noble", "commoner", "slave",
    "demon", "god", "goddess",
    # modern / everyday
    "doctor", "nurse", "teacher", "student", "officer", "detective", "agent",
    "spy", "driver", "worker", "clerk", "manager", "boss", "employee",
    "scientist", "engineer", "reporter", "lawyer", "judge", "merchant",
    "shopkeeper", "chef", "athlete", "idol", "singer", "actor", "artist",
    "pilot", "sailor", "neighbour", "neighbor",
    # story roles
    "hero", "villain", "rival", "friend", "enemy", "ally", "companion",
    "partner", "leader", "member", "survivor", "victim", "witness",
}


def _looks_person(text: str) -> bool:
    toks = set(_tokens(text))
    return bool(toks & _PERSONISH) or bool(toks & _GARMENT)


def _overlap(a: Set[str], b: Set[str]) -> float:
    """Overlap coefficient |A∩B| / min(|A|,|B|). Jaccard under-reads the
    real look-alike pair (8/11 = 0.73 after token normalisation); overlap
    reads it as 8/9 = 0.89."""
    if not a or not b:
        return 0.0
    return len(a & b) / float(min(len(a), len(b)))


def resolve_name(text: str, profiles: Sequence[Dict[str, Any]]
                 ) -> Tuple[str, str]:
    """(canonical_name|'unknown', evidence) for ONE figure-describing string.

    THE single resolution rule, shared by resolve_figures (panel subjects) and
    story_ledger (structured action actors/targets) — two copies drifted once
    and the ledger's copy erased every look-alike assassin to 'unclear',
    handing the writer facts like "unclear lunges at our protagonist".

    Requires score >= 2 AND a margin of 1 over the runner-up. A SAME-FACTION
    tie (every tied member shares a name token — near-identical appearance BY
    DESIGN) is one narrative identity, resolved to the LEAST SPECIFIC tied
    member: the faction/group entry when one exists, else the member with the
    fewest identity tokens. Evidence that fits 'the assassin' and 'the
    assassin leader' equally cannot claim the leader — that asymmetry is what
    kept re-crowning a dead leader from a generic hooded-cloak subject."""
    text = str(text or "").strip()
    if not text or not profiles:
        return "unknown", text[:60]
    by_name = {p["name"]: p for p in profiles}
    scored = sorted(((_score(p, text), p["name"]) for p in profiles),
                    key=lambda t: (-t[0][0], t[1]))
    (best, ev), name = scored[0]
    runner = scored[1][0][0] if len(scored) > 1 else 0.0
    if len(scored) > 1 and best >= 2.0 and runner >= 2.0 and best - runner < 3.0:
        # LOOK-ALIKE guard (ORV Ep128, 2026-09-06): cast_builder gave Dokja
        # and Michio token-identical descriptions ("dark hair, glasses, black
        # shirt over a white t-shirt, dark jacket") and the oracle then split
        # them on ONE incidental word — "messy" → Dokja, "button" → Michio —
        # naming the wrong man for a whole chapter. Two profiles whose
        # appearance sets nearly coincide, with no name evidence on either
        # side and less than one real piece of evidence (a pair or hair)
        # between them, are one look — unknown, never a coin flip. Faction
        # ties (shared name tokens) keep their own rule below.
        pa, pb = by_name[name], by_name[scored[1][1]]
        toks = _subject_tokens(text)
        if (not (pa["name_tokens"] & toks) and not (pb["name_tokens"] & toks)
                and not (pa["name_tokens"] & pb["name_tokens"])
                and _overlap(pa["appearance"], pb["appearance"]) >= 0.8):
            return "unknown", f"{text[:60]} ~ lookalike:{name}|{scored[1][1]}"
    if best >= 2.0 and best - runner >= 1.0:
        return name, f"{text[:60]} ~ {'+'.join(ev[:4])}"
    if best >= 2.0:
        tied = [n for (s, _e), n in scored if s == best]
        if len(tied) > 1:
            common = set.intersection(*(by_name[n]["name_tokens"]
                                        for n in tied))
            if common:
                pick = sorted(tied, key=lambda n: (
                    not by_name[n].get("is_group"),
                    len(by_name[n]["name_tokens"]), n))[0]
                return pick, (f"{text[:60]} ~ faction:"
                              f"{'+'.join(sorted(common)[:2])}")
    return "unknown", text[:60]


def resolve_figures(understanding: Optional[Dict[str, Any]],
                    profiles: Sequence[Dict[str, Any]],
                    excluded: Optional[Set[str]] = None
                    ) -> List[Dict[str, str]]:
    """[{cast_name|'unknown', evidence}] for ONE understood panel record.

    Each `subjects[]` entry resolves independently (a subject string describes
    ONE figure); the panel's description/action are scanned for NAME tokens
    only (they mix several figures' features — appearance-matching the blob
    would cross-attribute). Resolution requires score >= 2 AND a strict
    margin of 1 over the runner-up; ties resolve to 'unknown', never a guess
    (the failure mode being killed is misattribution).

    *excluded* names never resolve (story-ledger dead set: a killed leader
    must stop claiming later look-alike panels — 2026-07-20 wave)."""
    u = understanding or {}
    if excluded:
        profiles = [p for p in profiles if p["name"] not in excluded]
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def _add(name: str, evidence: str) -> None:
        key = name if name != "unknown" else f"unknown:{evidence}"
        if key not in seen:
            seen.add(key)
            out.append({"name": name, "evidence": evidence})

    subjects = [str(s).strip() for s in (u.get("subjects") or [])
                if str(s).strip()]
    for subj in subjects:
        name, ev = resolve_name(subj, profiles)
        if name != "unknown":
            _add(name, ev)
        elif _looks_person(subj):
            _add("unknown", subj[:60])

    blob = " ".join(str(u.get(k) or "") for k in ("description", "action",
                                                  "dialogue"))
    blob_toks = set(_tokens(blob))
    for p in profiles:
        hits = sorted(p["name_tokens"] & blob_toks)
        if hits:
            _add(p["name"], f"named: {'+'.join(hits[:3])}")
    return out


def resolve_figures_by_file(understood_obj: Any, cast: Any,
                            excluded_by_file: Optional[
                                Dict[str, Set[str]]] = None
                            ) -> Dict[str, List[Dict[str, str]]]:
    """{scene_file: figures} over a whole manifest.panels.understood.json.
    {} when either side is missing/empty — consumers stay silent.
    *excluded_by_file* maps scene_file -> cast names that must not resolve
    there (the story ledger's per-panel dead set)."""
    profiles = cast_profiles(cast)
    if not profiles:
        return {}
    out: Dict[str, List[Dict[str, str]]] = {}
    for p in ((understood_obj or {}).get("panels") or []):
        if isinstance(p, dict) and p.get("scene_file"):
            fn = str(p["scene_file"])
            out[fn] = resolve_figures(
                p, profiles, excluded=(excluded_by_file or {}).get(fn))
    return out


def shares_faction(names_a: Any, names_b: Any, cast: Any) -> bool:
    """True when any member of *names_a* shares an identity token with any of
    *names_b* — they are the SAME narrative faction and appearance evidence
    cannot separate them ('the assassin leader' vs 'the masked assassin' both
    carry 'assassin').

    resolve_name deliberately answers such a tie with the LEAST specific
    member, so a generic hooded panel resolves to the plain member. Any gate
    comparing a line's actor-noun against that resolution must therefore treat
    a same-faction pair as agreement, not a mismatch — otherwise every line
    that says 'leader' over an ambiguous panel is flagged wrong. (Measured:
    that produced 21 false actor_mismatch errors on nano ch1, which dragged 18
    of 25 groups into the heal loop.)"""
    by_name = {m_name: toks for m_name, toks in (
        (str(m.get("canonical_name") or m.get("id") or "").strip(),
         _name_tokens(m)) for m in _members(cast)) if m_name}
    a = set().union(*(by_name.get(n, set()) for n in (names_a or []))) \
        if names_a else set()
    b = set().union(*(by_name.get(n, set()) for n in (names_b or []))) \
        if names_b else set()
    return bool(a & b)


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

# Quoted dialogue is stripped before sentence-splitting: a name inside what a
# character SAYS is who they talk ABOUT, not the narrator's claim about this
# line's actor ("the stranger sneers 'the prince dies tonight'" claims only
# the stranger — "prince" is the stranger's own words, not the narrator
# naming an actor). The apostrophe doubles as a contraction/possessive mark
# ("can't", "assassin's") with no surrounding space, so only a quote-shaped
# apostrophe/quote-mark — whitespace-or-start before the opener, whitespace/
# punctuation/end after the closer — is treated as a delimiter; an in-word
# apostrophe never matches and is left alone.
_QUOTED_SPAN_RE = re.compile(
    r"(?:(?<=\s)|^)['\"‘“]"
    r"[^'\"’”]*"
    r"['\"’”](?=[\s.,!?;:]|$)"
)



# --- grammatical role of an actor-noun (2026-08-18) --------------------------
# The gates used to decide "who this line says is acting" from raw token
# POSITION (first five words), so a cast noun inside a partitive of-PP ("One of
# the assassins"), a copular predicate ("They were the bastards"), a direct
# object ("The news hits the students") or a fronted PP was asserted to be the
# panel's actor. All four were audited false positives on real chapters.
# These sets are closed-class ENGLISH function words — no story vocabulary.
_NP_START = frozenset("""a an the this that these those my your his her its our
    their some any several many few both each every another no all most either
    neither such one two three four five six seven eight nine ten he she it
    they we i you him them us me there""".split())
_COORD = frozenset("and or nor but plus".split())
_PREP = frozenset("""of in on at from with without about among amongst between
    against by into onto over under toward towards through across near behind
    beside beyond like than for to upon within during around off out up down
    along past despite amid amidst beneath above below inside outside""".split())
_SUBORD = frozenset("""as when while if because since although though unless
    until whenever whereas after before once wherever whether""".split())
_RELAT = frozenset("that which who whom whose".split())
_PRE_SUBJ_ADV = frozenset("""even only just still already then now soon again
    always never often once yet perhaps maybe meanwhile instead however indeed
    also too almost nearly barely hardly quite rather very so thus therefore
    here there back suddenly""".split())
# collective / plural-person markers: one subject STRING can denote many people
_COLLECTIVE = frozenset("""group crowd pair trio duo band throng mob gang team
    squad row line cluster bunch handful couple dozen dozens hundreds audience
    class army horde swarm party assembly gathering ranks masses lineup
    formation""".split())
_IRREG_PLURAL_PERSON = frozenset("people men women children folk folks youths".split())
_QUANT_MANY = frozenset("""several many multiple numerous various few both all
    some others two three four five six seven eight nine ten""".split())


def _is_pre_subject_adverb(tok: str) -> bool:
    return tok in _PRE_SUBJ_ADV or (len(tok) > 3 and tok.endswith("ly"))


def _subject_zone_flags(sent: str):
    """[(match, in_subject_zone)] for each word of *sent*. The zone is the
    clause's subject region: it opens at the clause start, closes at the first
    preposition / relativizer / second NP, and reopens after a clause break
    (comma, semicolon, dash) or a subordinator."""
    out = []
    open_, seen_np, prev_tok, prev_end = True, False, "", 0
    for m in _WORD_RE.finditer(sent):
        gap = sent[prev_end:m.start()]
        tok = m.group(0).lower()
        if any(ch in gap for ch in ",;:—–") or (prev_end and "--" in gap):
            open_, seen_np = True, False
        if tok in _SUBORD:
            open_, seen_np, prev_tok, prev_end = True, False, tok, m.end()
            continue
        if tok in _RELAT or tok in _PREP:
            open_ = False
            prev_tok, prev_end = tok, m.end()
            continue
        if tok in _NP_START:
            if seen_np and prev_tok not in _COORD and "," not in gap:
                open_ = False          # a SECOND noun phrase: object/predicate
            seen_np = True
        out.append((m, open_))
        if (open_ and tok not in _NP_START and tok not in _COORD
                and not _is_pre_subject_adverb(tok)):
            seen_np = True             # a bare subject head closes the slot
        prev_tok, prev_end = tok, m.end()
    return out


def group_member_names(cast: Any) -> Set[str]:
    """Canonical names of cast members that ARE a group ('the students', role
    'group'). Their handle is plural BY IDENTITY, so a line naming them can
    never be a count violation."""
    return {p["name"] for p in cast_profiles(cast) if p.get("is_group")}


def subject_person_count(text: str) -> int:
    """0 = not a person, 1 = one person, 2 = MORE THAN ONE (a sentinel, not a
    headcount). A panel analyst writes a drawn crowd as ONE subject string ("a
    group of people with dark hair"), which the old capacity count read as one
    figure and then reported as "every panel shows ONE figure"."""
    if not _looks_person(text):
        return 0
    toks = [t.lower() for t in _WORD_RE.findall(str(text or ""))]
    for t in toks:
        if t in _COLLECTIVE or t in _IRREG_PLURAL_PERSON or t in _QUANT_MANY:
            return 2
        if t not in _PERSONISH and _singular(t) != t and _singular(t) in _PERSONISH:
            return 2                   # a regular plural person noun
    return 1


def subject_actor_nouns_ex(line: str, noun_map: Dict[str, Set[str]]
                           ) -> List[Tuple[str, Set[str], bool]]:
    """Like subject_actor_nouns but each hit carries a PLURAL bit — whether
    the raw token was a plural form before _norm singularized it ("his
    assassins" → ('assassin', …, True)). The actor_count guard reads it; a
    possessive ("assassin's") is never plural."""
    hits: List[Tuple[str, Set[str], bool]] = []
    seen: Set[str] = set()
    clean_line = _QUOTED_SPAN_RE.sub(" ", str(line or ""))
    for sent in _SENT_SPLIT_RE.split(clean_line):
        for i, (m, in_zone) in enumerate(_subject_zone_flags(sent)[:7]):
            w = m.group(0)
            lw = w.lower()
            possessive = lw.endswith("'s") or lw.endswith("s'")
            if i >= 5 and not possessive:
                continue
            if not in_zone:
                continue               # object / predicate / PP-embedded
            t = _norm(w)
            if t in noun_map and t not in seen:
                base = lw.rstrip("'")
                if base.endswith("'s"):
                    base = base[:-2]
                plural = (not possessive and base != t
                          and _singular(base) == t)
                seen.add(t)
                hits.append((t, set(noun_map[t]), plural))
    return hits


def subject_actor_nouns(line: str, noun_map: Dict[str, Set[str]]
                        ) -> List[Tuple[str, Set[str]]]:
    """Actor-nouns of *line* used in SUBJECT position: within the first 5
    word-tokens of a sentence, or possessive-marked within the first 7
    ("an assassin's eye…"). Late mentions are usually objects/off-panel
    references ("their blades meant for the ancestor") — deliberately not
    flagged; this is the precision lever that keeps actor_mismatch a
    measured heal-target, not an FP fountain.

    Collapsed by RESOLVED MEMBER-SET: a multi-token name ("Cheon Yoo Jong"
    → cheon/yoo/jong, all ONE cast member) is a single mention, not three —
    otherwise the identity gates (actor_mismatch, dead_actor) fire once per
    name word for the same person (nano ch6: 1 issue reported as 3). The
    plural-aware _ex form stays per-token on purpose — actor_count reads the
    plural bit of each individual word."""
    out: List[Tuple[str, Set[str]]] = []
    seen_members: Set[frozenset] = set()
    for t, members, _plural in subject_actor_nouns_ex(line, noun_map):
        key = frozenset(members)
        if key in seen_members:
            continue
        seen_members.add(key)
        out.append((t, members))
    return out
