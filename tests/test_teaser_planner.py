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


# ---------------------------------------------------------------- Task 5
def test_payoff_tail_excluded():
    seq = [{"scene_file": f"p{i}", "panel_kind": "story", "intensity": "calm",
            "description": "x", "action": "", "dialogue": "", "subjects": []} for i in range(10)]
    wins = tp.score_windows(seq, min_panels=2, max_panels=3, payoff_tail_frac=0.2, shortlist_n=5)
    # last 20% (p8,p9) must not appear in any returned window
    assert all(p["scene_file"] not in ("p8", "p9") for w in wins for p in w["panels"])


def test_windows_respect_max_panels_and_nonoverlap():
    seq = [{"scene_file": f"p{i}", "panel_kind": "story", "intensity": "tense",
            "description": "exam", "action": "fight", "dialogue": "", "subjects": []} for i in range(12)]
    wins = tp.score_windows(seq, min_panels=4, max_panels=10, payoff_tail_frac=0.2, shortlist_n=3)
    assert wins and all(4 <= len(w["panels"]) <= 10 for w in wins)
    # non-overlapping shortlist
    spans = sorted((w["start"], w["end"]) for w in wins)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_no_windows_when_too_few_panels():
    seq = [{"scene_file": "p0", "panel_kind": "story", "intensity": "calm",
            "description": "x", "action": "", "dialogue": "", "subjects": []}]
    assert tp.score_windows(seq, min_panels=4, max_panels=10, payoff_tail_frac=0.2, shortlist_n=3) == []
