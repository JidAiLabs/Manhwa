"""Deterministic thumbnail-style selection from beats signals."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "thumbnail_styles",
    Path(__file__).resolve().parent.parent / "tools" / "thumbnail_styles.py")
ts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ts)  # type: ignore[union-attr]


def _beats(*texts, intensity="calm", bubble=None):
    return {"beats": [{"group_id": i + 1, "what_happens": t,
                       "scene_selection": [{"intensity": intensity,
                                            "bubble_mode": bubble}]}
                      for i, t in enumerate(texts)]}


def test_system_genre_or_ui_picks_stat_callout():
    assert ts.select_style(_beats("he checks his status window and skill list"),
                           genre="system regression") == "stat_callout"
    assert ts.select_style(_beats("his level hits 9999 and rank S")) == "stat_callout"


def test_monster_picks_vs_monster():
    assert ts.select_style(_beats("he faces a giant dragon at the tower top")) == "vs_monster"


def test_feat_object_picks_feat_object():
    assert ts.select_style(_beats("nobody can lift the 80kg weight he trains with")) == "feat_object"


def test_transformation_picks_before_after():
    assert ts.select_style(_beats("the weakest boy trained 100x and grew stronger")) == "before_after"


def test_humiliation_picks_humiliation():
    assert ts.select_style(_beats("he humiliates the professors who mocked him")) == "humiliation"


def test_default_is_power_reveal():
    assert ts.select_style(_beats("a quiet afternoon, nothing special happens")) == "power_reveal"


def test_every_style_has_art_prompt_and_overlay():
    for name, mod in ts.STYLE_MODULES.items():
        assert mod.get("art_prompt") and isinstance(mod.get("overlay"), dict), name
    assert ts.style_for("nonexistent") == ts.STYLE_MODULES[ts.DEFAULT_STYLE]


# --- hook designs: the label layer, counted from the owner's examples -------

_BUILT_DESIGNS = ("nametag", "nametag_headline", "system_window")


def test_hook_designs_declare_grammar_overlay_and_art_clause():
    for name in _BUILT_DESIGNS:
        d = ts.HOOK_DESIGNS[name]
        assert isinstance(d["label_grammar"], str) and d["label_grammar"].strip()
        assert isinstance(d["overlay"], dict) and d["overlay"]
        assert isinstance(d["art_clause"], str) and d["art_clause"].strip()


def test_hook_design_grammar_carries_no_worked_example():
    """An example of the exact thing being asked for is an answer, not a format
    hint: a quoted sample label shipped verbatim on a live thumbnail."""
    import re
    for name, d in ts.HOOK_DESIGNS.items():
        assert not re.search(r'"[A-Z0-9][A-Z0-9 #!>\-]{2,}"', d["label_grammar"]), name


# --- a design may not put its label where it puts the face ------------------
# Owner, 2026-09-22, first system_window card: "the label cover the face on
# right img?". The design said label_pos=upper_right AND told the art to put the
# face at 68% across, 42% down: the corner label's column (40% of the width,
# two stacked lines down to ~45% of the height) sat on it BY CONSTRUCTION.

def _label_box(pos):
    """The area a corner label can fill (thumbnail_overlay: a 40% column from
    its edge, two stacked lines from y=0.08)."""
    return {"upper_left": (0.00, 0.00, 0.45, 0.48),
            "upper_right": (0.55, 0.00, 1.00, 0.48)}[pos]


def test_no_design_puts_its_label_on_the_face():
    for name, d in ts.HOOK_DESIGNS.items():
        o = d["overlay"]
        x0, y0, x1, y1 = _label_box(o["label_pos"])
        fx, fy = o["arrow_to"]
        assert not (x0 <= fx <= x1 and y0 <= fy <= y1), \
            "%s: label %s covers the face at %s" % (name, o["label_pos"], o["arrow_to"])


def test_a_card_sits_clear_of_the_label_and_of_the_face():
    for name, d in ts.HOOK_DESIGNS.items():
        card = d["overlay"].get("card")
        if not card:
            continue
        (cx, cy), (cw, ch) = card["pos"], card["size"]
        lx0, ly0, lx1, ly1 = _label_box(d["overlay"]["label_pos"])
        overlaps_label = cx < lx1 and cx + cw > lx0 and cy < ly1 and cy + ch > ly0
        assert not overlaps_label, name
        fx, fy = d["overlay"]["arrow_to"]
        assert not (cx <= fx <= cx + cw and cy <= fy <= cy + ch), name


def test_the_headline_design_asks_for_a_painted_bottom_not_a_blank_bar():
    clause = ts.HOOK_DESIGNS["nametag_headline"]["art_clause"].lower()
    assert "bottom fifth" not in clause           # that wording painted an empty strip
    assert "bottom edge" in clause
