# tests/test_niche_modules.py
from tools import niche_modules as nm


def _primary(genres, synopsis=""):
    ranked = nm.classify_niche(genres, synopsis)
    return ranked[0][0] if ranked else None


def _with_secondary(genres, synopsis=""):
    """(primary, secondary|None) via the module's shared margin rule."""
    return nm.pick_primary_secondary(nm.classify_niche(genres, synopsis))


def test_pure_genres_map_to_single_niche():
    assert _primary(["Isekai", "Fantasy"]) == "A"
    assert _primary(["System", "Leveling", "Dungeon"]) == "A"
    assert _primary(["Romance", "Drama"]) == "B"
    assert _primary(["Revenge", "Tragedy", "Mature"]) == "C"
    assert _primary(["Comedy", "Slice of Life"]) == "D"


def test_nano_style_murim_action_is_C_primary_A_secondary():
    # A murim/action/revenge manhwa (Nano Machine class): action+mature -> C,
    # martial-arts/murim -> A, with A >= 0.5*C so it surfaces as secondary.
    primary, secondary = _with_secondary(
        ["Action", "Martial Arts", "Murim", "Mature", "Fantasy"])
    assert primary == "C"
    assert secondary == "A"


def test_single_dominant_genre_has_no_secondary():
    primary, secondary = _with_secondary(["Comedy", "Slice of Life", "Comedy"])
    assert primary == "D"
    assert secondary is None


def test_unmappable_or_empty_returns_empty():
    assert nm.classify_niche([], "") == []
    assert nm.classify_niche(["Webtoon", "Full Color"], "") == []  # chrome tags only


def test_synopsis_is_a_weak_tiebreak_only():
    ranked = nm.classify_niche([], "A betrayed swordsman returns for bloody revenge.")
    assert ranked and ranked[0][0] == "C"
    tag_ranked = nm.classify_niche(["Romance"], "revenge revenge revenge")
    assert tag_ranked[0][0] == "B"  # one genre tag outweighs three synopsis hits


def test_registers_exist_and_are_nonempty_for_all_four():
    for key in ("A", "B", "C", "D"):
        assert key in nm.NICHE_REGISTERS
        assert isinstance(nm.NICHE_REGISTERS[key], str)
        assert nm.NICHE_REGISTERS[key].strip()


def test_ranked_scores_are_descending():
    ranked = nm.classify_niche(["Action", "Martial Arts", "Comedy"])
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_pick_primary_secondary_applies_half_margin():
    assert nm.pick_primary_secondary([("C", 5.0), ("A", 4.0)]) == ("C", "A")
    assert nm.pick_primary_secondary([("C", 5.0), ("A", 2.0)]) == ("C", None)  # 2 < 2.5
    assert nm.pick_primary_secondary([("D", 6.0)]) == ("D", None)
    assert nm.pick_primary_secondary([]) == (None, None)


def test_register_block_primary_and_secondary():
    blk = nm.register_block("C", "A")
    assert blk.startswith("PRIMARY") and "SECONDARY" in blk
    assert nm.register_block("", "") == ""        # no niche -> base voice only
    assert "SECONDARY" not in nm.register_block("D", "")  # primary only
