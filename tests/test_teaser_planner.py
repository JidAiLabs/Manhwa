"""
tests/test_teaser_planner.py

Unit tests for tools/teaser_planner.py — the bundle-level arc-teaser planner.

Chunk 2 scope: the deterministic, $0, pure Stage-1 scoring layer:
  - eligible_panels  (panel eligibility / flattening)
  - score_window     (signal scoring of one contiguous window)
  - score_windows    (spoiler guard + window enumeration + non-overlapping shortlist)

tools/ is not a package, so the module is loaded by path (the repo's standard
tool-test idiom).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "teaser_planner",
    Path(__file__).resolve().parent.parent / "tools" / "teaser_planner.py",
)
tp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tp)  # type: ignore[union-attr]


# ---------------------------------------------------------------- Task 3
def test_eligible_panels_skips_chrome_empty_error():
    panels = [
        {"scene_file": "a", "panel_kind": "story", "intensity": "calm"},
        {"scene_file": "b", "panel_kind": "chrome", "intensity": "calm"},
        {"scene_file": "c", "panel_kind": "empty", "intensity": "calm"},
        {"scene_file": "d", "panel_kind": "system", "intensity": "tense"},
        {"scene_file": "e", "panel_kind": "story", "error": "parse_failed"},
    ]
    out = tp.eligible_panels(panels)
    assert [p["scene_file"] for p in out] == ["a", "d"]


# ---------------------------------------------------------------- Task 4
def test_score_window_high_stakes_beats_calm():
    hot = [{"scene_file": f"h{i}", "panel_kind": "story", "intensity": "explosive",
            "description": "the entrance exam begins", "action": "a clan heir humiliates him",
            "dialogue": "you have no badge", "subjects": ["heir", "prince"]} for i in range(5)]
    calm = [{"scene_file": f"c{i}", "panel_kind": "story", "intensity": "calm",
             "description": "they eat lunch quietly", "action": "", "dialogue": "",
             "subjects": ["prince"]} for i in range(5)]
    assert tp.score_window(hot)["score"] > tp.score_window(calm)["score"]
