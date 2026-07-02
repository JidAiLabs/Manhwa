#!/usr/bin/env python3
"""beats_segments.py — the ONE shared reader/writer for a beat's narration segments.

Adaptive flow narration (spec 2026-07-02) replaces the strict per-panel
`panel_narration` list with `segments`: an ordered list of
{"span": [scene_files...], "line": "..."} where a span covers 1-4 consecutive
panels voiced as ONE clip.

Every consumer reads a beat's narration through `beat_segments()` — native
`segments` win; a legacy beat carrying only `panel_narration` (old manifests,
the teaser's synthetic beat) is adapted to singleton spans. Every mutator
(punchup, recap_style repairs, heal) writes repaired lines back through
`write_segment_lines()`, which round-trips WHICHEVER shape the beat carries —
a mutator may edit lines, never re-split spans — and rebuilds
`beat["narration"]` as the ordered join of segment lines (the load-bearing
derived view caption/staleness QA and punchup key on).

Pure functions, no I/O.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List


def _basename(path: Any) -> str:
    return os.path.basename(str(path or "").strip())


def _native_ok(seg: Any) -> bool:
    """A well-formed native segment: dict with a non-empty line + span."""
    if not isinstance(seg, dict):
        return False
    span = seg.get("span")
    has_span = isinstance(span, list) and any(_basename(f) for f in span)
    return has_span and bool(str(seg.get("line") or "").strip())


def _legacy_ok(item: Any) -> bool:
    """A well-formed legacy panel_narration entry: scene_file + line."""
    return (isinstance(item, dict)
            and bool(_basename(item.get("scene_file")))
            and bool(str(item.get("line") or "").strip()))


def has_native_segments(beat: Any) -> bool:
    """True when the beat carries native adaptive `segments` (the new shape).

    Consumers use this to retire per-panel-era post-processing that flow spans
    supersede (e.g. the script packer's short-line merger, consecutive-dup
    panel removal) WITHOUT changing behavior for legacy `panel_narration`
    manifests — shape detection, not a flag."""
    if not isinstance(beat, dict):
        return False
    native = beat.get("segments")
    return isinstance(native, list) and bool(native)


def beat_segments(beat: Any) -> List[Dict[str, Any]]:
    """Return the beat's narration segments: [{"span": [basenames...], "line": str}].

    Native `beat["segments"]` wins; a legacy beat with only `panel_narration`
    yields singleton spans in order; neither -> []. Malformed entries (missing/
    empty line or span) are skipped. Spans are normalized to basenames. The
    returned list is a normalized COPY — write changes back via
    write_segment_lines(), never by mutating it.
    """
    if not isinstance(beat, dict):
        return []
    native = beat.get("segments")
    if isinstance(native, list) and native:
        return [{"span": [_basename(f) for f in seg["span"] if _basename(f)],
                 "line": str(seg["line"]).strip()}
                for seg in native if _native_ok(seg)]
    return [{"span": [_basename(item["scene_file"])],
             "line": str(item["line"]).strip()}
            for item in (beat.get("panel_narration") or [])
            if _legacy_ok(item)]


def write_segment_lines(beat: Dict[str, Any], lines: List[str]) -> Dict[str, Any]:
    """Write repaired narration lines back into whichever shape the beat carries.

    `lines` maps 1:1 onto the entries `beat_segments(beat)` returns (malformed
    entries are skipped on read, so they stay untouched on write). A length
    mismatch or an empty line raises ValueError — a mutator may edit lines,
    never re-split spans or silently delete a segment. Rebuilds
    `beat["narration"]` as the ordered join. Returns the beat.
    """
    lines = [str(x or "").strip() for x in (lines or [])]
    if any(not x for x in lines):
        raise ValueError("write_segment_lines: empty line (would delete a segment)")

    native = beat.get("segments")
    if isinstance(native, list) and native:
        targets: List[Dict[str, Any]] = [s for s in native if _native_ok(s)]
    else:
        targets = [p for p in (beat.get("panel_narration") or []) if _legacy_ok(p)]
    if len(targets) != len(lines):
        raise ValueError(
            f"write_segment_lines: {len(lines)} line(s) for {len(targets)} "
            "segment(s) — a mutator may edit lines, never re-split")

    for entry, line in zip(targets, lines):
        entry["line"] = line
    beat["narration"] = " ".join(lines).strip() or beat.get("narration", "")
    return beat
