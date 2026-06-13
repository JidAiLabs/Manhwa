"""
tests/test_scene_selection.py

TDD for tools/scene_selection.py — the shared, pure logic for the Gemini
scene-understanding pass (SP2 #2 real fix + semantic dedup).

Two functions, used by two tools so the contract stays in one place:
  - normalize_scene_selection(raw, scene_files): gemini_narrative_pass sanitizes
    the model's per-scene judgments into exactly one entry per scene, defaulting
    SAFELY to "keep" (never drop a panel unless the model explicitly says so).
  - choose_kept_scenes(scene_files, selection, max_keep): timeline_planner picks
    which panels to actually show, dropping "redundant" ones FIRST (instead of the
    old arbitrary cut), preserving original order.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "scene_selection",
    Path(__file__).resolve().parent.parent / "tools" / "scene_selection.py",
)
ss = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ss)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# normalize_scene_selection
# ---------------------------------------------------------------------------

def test_normalize_one_entry_per_scene_in_order():
    raw = [
        {"scene_file": "b.jpg", "role": "redundant"},
        {"scene_file": "a.jpg", "role": "keep"},
    ]
    out = ss.normalize_scene_selection(raw, ["a.jpg", "b.jpg", "c.jpg"])
    assert [e["scene_file"] for e in out] == ["a.jpg", "b.jpg", "c.jpg"]
    assert out[0]["role"] == "keep"
    assert out[1]["role"] == "redundant"
    # c.jpg had no model entry → defaults to keep (safe)
    assert out[2]["role"] == "keep"


def test_normalize_defaults_are_safe():
    out = ss.normalize_scene_selection([], ["x.jpg"])
    e = out[0]
    assert e["role"] == "keep"                 # never drop by default
    assert e["bubble_mode"] == "unknown"
    assert e["intensity"] == "unknown"


def test_normalize_sanitizes_bad_values():
    raw = [{"scene_file": "x.jpg", "role": "garbage",
            "bubble_mode": "nonsense", "intensity": "??"}]
    e = ss.normalize_scene_selection(raw, ["x.jpg"])[0]
    assert e["role"] == "keep"                 # invalid role → keep
    assert e["bubble_mode"] == "unknown"
    assert e["intensity"] == "unknown"


def test_normalize_accepts_valid_enums():
    raw = [{"scene_file": "x.jpg", "role": "redundant",
            "bubble_mode": "inner_thought", "intensity": "explosive"}]
    e = ss.normalize_scene_selection(raw, ["x.jpg"])[0]
    assert e["role"] == "redundant"
    assert e["bubble_mode"] == "inner_thought"
    assert e["intensity"] == "explosive"


def test_normalize_ignores_unknown_scene_files():
    raw = [{"scene_file": "ghost.jpg", "role": "redundant"}]
    out = ss.normalize_scene_selection(raw, ["a.jpg"])
    assert len(out) == 1 and out[0]["scene_file"] == "a.jpg"
    assert out[0]["role"] == "keep"


# ---------------------------------------------------------------------------
# choose_kept_scenes
# ---------------------------------------------------------------------------

def _sel(roles):
    # roles: dict scene_file -> role
    return [{"scene_file": k, "role": v, "bubble_mode": "unknown", "intensity": "unknown"}
            for k, v in roles.items()]


def test_choose_drops_redundant_first():
    files = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    sel = _sel({"a.jpg": "keep", "b.jpg": "redundant", "c.jpg": "keep", "d.jpg": "redundant"})
    # only room for 2 → keep the two 'keep' scenes, drop both redundant
    out = ss.choose_kept_scenes(files, sel, max_keep=2)
    assert out == ["a.jpg", "c.jpg"]


def test_choose_keepers_in_original_order():
    files = ["a.jpg", "b.jpg", "c.jpg"]
    sel = _sel({"a.jpg": "keep", "b.jpg": "keep", "c.jpg": "redundant"})
    out = ss.choose_kept_scenes(files, sel, max_keep=3)
    # redundant c is DROPPED even though there's room — keepers hold longer
    assert out == ["a.jpg", "b.jpg"]


def test_choose_does_not_pad_with_redundant():
    files = ["a.jpg", "b.jpg", "c.jpg"]
    sel = _sel({"a.jpg": "keep", "b.jpg": "redundant", "c.jpg": "redundant"})
    # only 1 keeper; redundant panels are NOT padded back in → just the keeper,
    # which then gets the whole shot's time (longer hold, no on-screen dup)
    out = ss.choose_kept_scenes(files, sel, max_keep=2)
    assert out == ["a.jpg"]


def test_choose_protects_title_card_marked_redundant():
    files = ["a.jpg", "card.jpg", "b.jpg"]
    sel = _sel({"a.jpg": "keep", "card.jpg": "redundant", "b.jpg": "keep"})
    # SKY CORPORATION-class card: protected → kept despite 'redundant', in order
    out = ss.choose_kept_scenes(files, sel, max_keep=3, protected={"card.jpg"})
    assert out == ["a.jpg", "card.jpg", "b.jpg"]


def test_choose_protected_card_survives_full_budget():
    files = ["a.jpg", "b.jpg", "card.jpg"]
    sel = _sel({"a.jpg": "keep", "b.jpg": "keep", "card.jpg": "redundant"})
    out = ss.choose_kept_scenes(files, sel, max_keep=2, protected={"card.jpg"})
    assert "card.jpg" in out                 # mandatory card never dropped


def test_choose_falls_back_to_files_when_no_keepers():
    files = ["a.jpg", "b.jpg", "c.jpg"]
    sel = _sel({"a.jpg": "redundant", "b.jpg": "redundant", "c.jpg": "redundant"})
    # all redundant → show the first max_keep so the shot isn't empty
    out = ss.choose_kept_scenes(files, sel, max_keep=2)
    assert out == ["a.jpg", "b.jpg"]


def test_choose_always_returns_at_least_one():
    files = ["a.jpg", "b.jpg"]
    sel = _sel({"a.jpg": "redundant", "b.jpg": "redundant"})
    out = ss.choose_kept_scenes(files, sel, max_keep=0)
    assert out == ["a.jpg"]                      # never show an empty shot


def test_choose_empty_files_returns_empty():
    assert ss.choose_kept_scenes([], [], max_keep=3) == []
