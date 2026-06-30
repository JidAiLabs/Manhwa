# tools/niche_modules.py
"""Manhwa niche (a/b/c/d) classification + per-niche persona temperature blocks.

The niche is the user's "Manhwa Fresh" Niche Module:
  A = Isekai / Power-Fantasy   B = Romance / Drama
  C = Dark-Action / Revenge    D = Comedy / Slice-of-Life

classify_niche() is deterministic (a weighted keyword map over the source's genre
tags, synopsis as a weak tiebreak) — NO LLM call, so it never reintroduces the
grading non-determinism we are trying to remove. pick_primary_secondary() applies
the 0.5x margin rule in ONE place (used by the add-series caller and the tests).
"""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

# Per-niche keyword weights. Tags are matched case-insensitively as substrings
# against each genre tag (and, at lower weight, the synopsis). Weights:
# 3 = a defining signal, 2 = strong, 1 = supporting. A tag may contribute to more
# than one niche (e.g. "martial arts" -> A power-fantasy).
_NICHE_KEYWORDS: Mapping[str, Mapping[str, int]] = {
    "A": {  # Isekai / Power-Fantasy
        "isekai": 3, "reincarnat": 3, "regression": 3, "regressor": 3,
        "returner": 3, "system": 3, "leveling": 3, "level up": 3, "level-up": 3,
        "dungeon": 2, "tower": 2, "hunter": 2, "awakening": 2,
        "cultivation": 3, "murim": 2, "martial art": 2, "sect": 2,
        "status window": 3, "overpowered": 3, "rpg": 2,
        "constellation": 2, "summoner": 1, "necromancer": 1,
    },
    "B": {  # Romance / Drama
        "romance": 3, "romantic": 3, "love": 2, "drama": 2, "josei": 3,
        "shoujo": 3, "shojo": 3, "otome": 3, "melodrama": 2,
        "marriage": 2, "family drama": 2,
    },
    "C": {  # Dark-Action / Revenge
        "revenge": 3, "vengeance": 3, "action": 3, "thriller": 2, "horror": 2,
        "tragedy": 2, "mature": 2, "gore": 2, "psychological": 2, "war": 1,
        "crime": 2, "mafia": 2, "murder": 2, "villain": 2, "assassin": 2,
        "betrayal": 2, "bloody": 2, "dark": 2, "seinen": 1, "military": 1,
    },
    "D": {  # Comedy / Slice-of-Life
        "comedy": 3, "gag": 3, "parody": 3, "slice of life": 3, "slice-of-life": 3,
        "humor": 2, "humour": 2, "wholesome": 2, "healing": 2, "cooking": 1,
        "everyday": 1,
    },
}

# Tags carrying no genre signal — ignored so they don't dilute scoring.
_CHROME_TAGS = {"webtoon", "manhwa", "manhua", "manga", "full color", "colored",
                "long strip", "adaptation", "web comic", "webcomic"}

_SYNOPSIS_WEIGHT = 0.34  # a synopsis hit is worth ~1/3 of a genre-tag hit (tiebreak)


def _norm(s: str) -> str:
    return (s or "").lower().strip()


def _score_text(text: str, weight_scale: float, acc: dict) -> None:
    low = _norm(text)
    if not low or low in _CHROME_TAGS:
        return
    for niche, kw in _NICHE_KEYWORDS.items():
        for term, w in kw.items():
            if term in low:
                acc[niche] = acc.get(niche, 0.0) + w * weight_scale


def classify_niche(genres: Sequence[str],
                   synopsis: str = "") -> List[Tuple[str, float]]:
    """Return [(niche, score), ...] ranked by score descending; [] if nothing maps."""
    acc: dict = {}
    for tag in (genres or []):
        _score_text(str(tag), 1.0, acc)
    if synopsis:
        _score_text(str(synopsis), _SYNOPSIS_WEIGHT, acc)
    return sorted(
        ((n, s) for n, s in acc.items() if s > 0.0),
        key=lambda kv: (-kv[1], kv[0]),  # score desc, niche letter for stable ties
    )


def pick_primary_secondary(ranked: List[Tuple[str, float]]):
    """Apply the 0.5x margin rule -> (primary|None, secondary|None). ONE home for
    the rule so the add-series caller and the tests agree."""
    if not ranked:
        return (None, None)
    primary, p_score = ranked[0]
    secondary = None
    if len(ranked) > 1 and ranked[1][1] >= 0.5 * p_score:
        secondary = ranked[1][0]
    return (primary, secondary)


# --- per-niche persona TEMPERATURE blocks (injected by narration_punchup, Chunk 2)
# These modulate the ALWAYS-ON base voice; they never replace it.
NICHE_REGISTERS: Mapping[str, str] = {
    "A": (
        "NICHE TEMPERATURE — Isekai/Power-Fantasy (HYPE): lean into the power gap "
        "and the climb; the gap is the point. Cocky understatement about how "
        "outmatched everyone else is; confidence high, jokes welcome. Use "
        "game/system framing ONLY if the world is literally a system; for a "
        "murim/cultivation power-fantasy use realm/cultivation framing instead."
    ),
    "B": (
        "NICHE TEMPERATURE — Romance/Drama (WARM): intimacy over spectacle. Slow on "
        "glances, confessions, betrayals; texture is wry warmth, never a gamer joke "
        "over an emotional beat. Stakes are relational, not combat; keep the wit gentle."
    ),
    "C": (
        "NICHE TEMPERATURE — Dark-Action/Revenge (COLD): controlled menace, short "
        "hard clauses; let cruelty and consequence land. The wit stays but turns "
        "grim/ironic, not light — arrogance reads as intimidation, not comedy. Jokes "
        "thin out; stay restrained on grief and death."
    ),
    "D": (
        "NICHE TEMPERATURE — Comedy/Slice-of-Life (FUNNY): persona-forward, the "
        "funniest register. Frequent asides, absurd contrast, deadpan, audience "
        "winks — humor is the DEFAULT here, not seasoning."
    ),
}


def register_block(primary: str, secondary: str = "") -> str:
    """Compose the register prompt: primary governs; secondary flavors matching beats."""
    primary = (primary or "").upper().strip()
    secondary = (secondary or "").upper().strip()
    if primary not in NICHE_REGISTERS:
        return ""  # no niche -> base voice only
    out = "PRIMARY " + NICHE_REGISTERS[primary]
    if secondary in NICHE_REGISTERS and secondary != primary:
        out += ("\nSECONDARY (flavor only the beats that match it) "
                + NICHE_REGISTERS[secondary])
    return out


def demo() -> None:
    """Runnable self-check: assert representative tag-sets rank correctly."""
    assert classify_niche(["Isekai", "Fantasy"])[0][0] == "A"
    assert classify_niche(["Romance", "Drama"])[0][0] == "B"
    r = classify_niche(["Action", "Martial Arts", "Murim", "Mature"])
    assert pick_primary_secondary(r) == ("C", "A"), r
    assert classify_niche(["Comedy", "Slice of Life"])[0][0] == "D"
    assert classify_niche([], "") == []
    assert pick_primary_secondary([("C", 5.0), ("A", 2.0)]) == ("C", None)
    assert register_block("C", "A").startswith("PRIMARY") and "SECONDARY" in register_block("C", "A")
    print("niche_modules demo OK")


if __name__ == "__main__":
    demo()
