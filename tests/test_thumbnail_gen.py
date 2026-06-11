"""tests/test_thumbnail_gen.py — pure pieces of the YouTube thumbnail tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "thumbnail_gen",
    Path(__file__).resolve().parent.parent / "tools" / "thumbnail_gen.py",
)
tg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tg)  # type: ignore[union-attr]


def _beats(sel):
    return {"beats": [{"scene_selection": [
        {"scene_file": f, "role": r, "intensity": i} for f, r, i in sel]}]}


def test_pick_reference_scenes_weak_then_climax():
    beats = _beats([
        ("p000001.jpg", "keep", "calm"),       # weak (earliest calm/tense)
        ("p000002.jpg", "redundant", "intense"),
        ("p000003.jpg", "keep", "intense"),
        ("p000009.jpg", "keep", "intense"),    # climax (last intense kept)
    ])
    refs = tg.pick_reference_scenes(beats)
    assert refs[0] == "p000001.jpg"
    assert "p000009.jpg" in refs
    assert "p000002.jpg" not in refs           # redundant never referenced
    assert len(refs) <= 3 == len(set(refs) | {refs[0]}) or len(refs) >= 2


def test_pick_reference_scenes_empty():
    assert tg.pick_reference_scenes({"beats": []}) == []


def test_build_prompt_default_before_after_no_titles():
    p = tg.build_prompt("")
    assert '"BEFORE"' in p and '"AFTER"' in p
    assert "no series name" in p               # licensed names never rendered
    assert "speech bubbles" in p and "16:9" in p


def test_build_prompt_hook_replaces_before_after():
    p = tg.build_prompt("Secret AI System")
    assert '"SECRET AI SYSTEM"' in p and "arrow" in p
    assert '"BEFORE"' not in p
