"""beats_segments: the ONE shared reader/writer for beats[].segments.

Adaptive flow narration (spec 2026-07-02): the beats writer emits
`beats[].segments[] = [{"span": [scene_files...], "line": "..."}]`. Every
consumer reads through `beat_segments()` (native segments, or legacy
`panel_narration` adapted to singleton spans) and every mutator writes lines
back through `write_segment_lines()` (shape-aware, so the teaser's legacy
`{"beats":[{"panel_narration": ...}]}` round-trip keeps working).
"""
from __future__ import annotations

import pytest

from tools.beats_segments import (
    beat_segments,
    has_native_segments,
    write_segment_lines,
)


# ---------------------------------------------------------------------------
# beat_segments — reader
# ---------------------------------------------------------------------------

def test_native_segments_returned_with_basename_spans():
    beat = {"segments": [
        {"span": ["/abs/scenes/p000012.jpg"], "line": "Solo line."},
        {"span": ["p000013.jpg", "scenes/p000014.jpg"], "line": "Flow across two."},
    ]}
    assert beat_segments(beat) == [
        {"span": ["p000012.jpg"], "line": "Solo line."},
        {"span": ["p000013.jpg", "p000014.jpg"], "line": "Flow across two."},
    ]


def test_legacy_panel_narration_adapts_to_singleton_spans_in_order():
    beat = {"panel_narration": [
        {"scene_file": "p1.jpg", "line": "First."},
        {"scene_file": "p2.jpg", "line": "Second."},
    ]}
    assert beat_segments(beat) == [
        {"span": ["p1.jpg"], "line": "First."},
        {"span": ["p2.jpg"], "line": "Second."},
    ]


def test_neither_shape_returns_empty():
    assert beat_segments({}) == []
    assert beat_segments({"narration": "joined only"}) == []
    assert beat_segments(None) == []          # defensive: not a dict


def test_malformed_native_entries_skipped():
    beat = {"segments": [
        {"span": ["p1.jpg"], "line": "Good."},
        {"span": [], "line": "No span."},          # empty span
        {"span": ["p2.jpg"], "line": ""},          # empty line
        {"line": "Missing span."},                 # no span key
        "not-a-dict",
        {"span": ["p3.jpg"], "line": "Also good."},
    ]}
    assert beat_segments(beat) == [
        {"span": ["p1.jpg"], "line": "Good."},
        {"span": ["p3.jpg"], "line": "Also good."},
    ]


def test_malformed_legacy_entries_skipped():
    beat = {"panel_narration": [
        {"scene_file": "p1.jpg", "line": "Good."},
        {"scene_file": "", "line": "No file."},
        {"scene_file": "p2.jpg"},                  # no line
    ]}
    assert beat_segments(beat) == [{"span": ["p1.jpg"], "line": "Good."}]


def test_native_segments_win_over_legacy_panel_narration():
    beat = {"segments": [{"span": ["a.jpg", "b.jpg"], "line": "Flow."}],
            "panel_narration": [{"scene_file": "a.jpg", "line": "Old."}]}
    assert beat_segments(beat) == [{"span": ["a.jpg", "b.jpg"], "line": "Flow."}]


def test_reader_returns_copies_not_references():
    beat = {"segments": [{"span": ["a.jpg"], "line": "Line."}]}
    segs = beat_segments(beat)
    segs[0]["line"] = "MUTATED"
    segs[0]["span"].append("x.jpg")
    assert beat["segments"][0] == {"span": ["a.jpg"], "line": "Line."}


# ---------------------------------------------------------------------------
# write_segment_lines — shape-aware writer
# ---------------------------------------------------------------------------

def test_write_native_updates_lines_in_order_and_rebuilds_join():
    beat = {"segments": [{"span": ["a.jpg"], "line": "Old solo."},
                         {"span": ["b.jpg", "c.jpg"], "line": "Old flow."}],
            "narration": "Old solo. Old flow."}
    write_segment_lines(beat, ["New solo.", "New flow."])
    assert [s["line"] for s in beat["segments"]] == ["New solo.", "New flow."]
    assert beat["narration"] == "New solo. New flow."
    assert beat["segments"][1]["span"] == ["b.jpg", "c.jpg"]   # spans untouched


def test_write_legacy_updates_panel_narration_in_place():
    beat = {"panel_narration": [{"scene_file": "a.jpg", "line": "Old 1."},
                                {"scene_file": "b.jpg", "line": "Old 2."}],
            "narration": "Old 1. Old 2."}
    write_segment_lines(beat, ["New 1.", "New 2."])
    assert [p["line"] for p in beat["panel_narration"]] == ["New 1.", "New 2."]
    assert beat["narration"] == "New 1. New 2."
    assert "segments" not in beat   # shape preserved (teaser round-trip)


def test_write_length_mismatch_raises():
    beat = {"segments": [{"span": ["a.jpg"], "line": "One."}]}
    with pytest.raises(ValueError):
        write_segment_lines(beat, ["One.", "Two."])     # a mutator never re-splits


def test_write_targets_exactly_what_the_reader_returned():
    # round-trip invariant: lines map onto the entries beat_segments yields,
    # malformed entries stay untouched.
    beat = {"panel_narration": [
        {"scene_file": "a.jpg", "line": "Good."},
        {"scene_file": "", "line": "Malformed."},
        {"scene_file": "b.jpg", "line": "Also good."},
    ]}
    write_segment_lines(beat, ["New A.", "New B."])
    assert beat["panel_narration"][0]["line"] == "New A."
    assert beat["panel_narration"][1]["line"] == "Malformed."
    assert beat["panel_narration"][2]["line"] == "New B."
    assert beat["narration"] == "New A. New B."


def test_write_empty_line_rejected():
    # an empty line would silently delete the segment on the next read (= a
    # re-split); the writer refuses it.
    beat = {"segments": [{"span": ["a.jpg"], "line": "One."}]}
    with pytest.raises(ValueError):
        write_segment_lines(beat, [""])


def test_write_returns_the_beat():
    beat = {"segments": [{"span": ["a.jpg"], "line": "One."}]}
    assert write_segment_lines(beat, ["Two words here."]) is beat


# ---------------------------------------------------------------------------
# has_native_segments — shape detection (retires per-panel-era post-processing)
# ---------------------------------------------------------------------------

def test_has_native_segments_true_only_for_segments_shape():
    assert has_native_segments(
        {"segments": [{"span": ["a.jpg"], "line": "One."}]}) is True
    assert has_native_segments(
        {"panel_narration": [{"scene_file": "a.jpg", "line": "One."}]}) is False
    assert has_native_segments({"segments": []}) is False
    assert has_native_segments({}) is False
    assert has_native_segments(None) is False
