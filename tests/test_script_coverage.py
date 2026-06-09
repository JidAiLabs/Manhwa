"""
tests/test_script_coverage.py

TDD for the missing-narration fix in tools/script_expander.py.

Root cause: the model sometimes returns fewer narration paragraphs than the
section has beats; _build_default_shots_from_payload then uses
n = min(len(beats), len(paragraphs)) and SILENTLY drops the trailing beats
(observed: section 0 had 6 beats but 3 paragraphs -> groups 4,5,6 got no
narration). _ensure_paragraph_coverage pads paragraphs to one-per-beat using a
deterministic fallback from each uncovered beat's what_happens.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "script_expander",
    Path(__file__).resolve().parent.parent / "tools" / "script_expander.py",
)
se = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(se)  # type: ignore[union-attr]


def test_coverage_pads_missing_paragraphs_from_what_happens():
    beats = [
        {"group_id": 1, "what_happens": "A duel begins."},
        {"group_id": 2, "what_happens": "The hero is wounded."},
        {"group_id": 3, "what_happens": "A masked figure appears."},
    ]
    paras = ["A duel begins under the moon."]   # model only covered beat 1
    out = se._ensure_paragraph_coverage(beats, paras)
    assert len(out) == 3                          # one per beat now
    assert out[0] == "A duel begins under the moon."   # original kept
    assert "wounded" in out[1]                    # backfilled from beat 2
    assert "masked figure" in out[2]              # backfilled from beat 3


def test_coverage_unchanged_when_already_complete():
    beats = [{"group_id": 1, "what_happens": "x"}, {"group_id": 2, "what_happens": "y"}]
    paras = ["one", "two"]
    assert se._ensure_paragraph_coverage(beats, paras) == ["one", "two"]


def test_coverage_does_not_truncate_extra_paragraphs():
    beats = [{"group_id": 1, "what_happens": "x"}]
    paras = ["one", "two", "three"]
    assert se._ensure_paragraph_coverage(beats, paras) == ["one", "two", "three"]


def test_coverage_fallback_when_beat_has_no_what_happens():
    beats = [{"group_id": 1, "what_happens": ""}, {"group_id": 2}]
    out = se._ensure_paragraph_coverage(beats, [])
    assert len(out) == 2
    assert all(p.strip() for p in out)            # never an empty paragraph
