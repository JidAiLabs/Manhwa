"""
tests/test_script_expander_segments.py

Chunk 2 (adaptive flow narration): the verbatim packer emits ONE paragraph +
ONE shot per narration SEGMENT — a flow span of 1-4 consecutive panels voiced
as a single clip. `shots[].scene_files` carries the span (order kept), the
mood tag escalates from the MAX panel intensity across the span, and the
short-line merger (`tts_merge_short`) is retired for segments-shaped beats
(flow spans supersede it — running both would double-merge).

Legacy `panel_narration` beats keep today's behavior byte-for-byte (singleton
spans via beat_segments; merger still available).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent / "tools"

_SPEC = importlib.util.spec_from_file_location(
    "script_expander", _TOOLS / "script_expander.py")
se = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(se)  # type: ignore[union-attr]

_RP_SPEC = importlib.util.spec_from_file_location(
    "render_prep", _TOOLS / "render_prep.py")
rp = importlib.util.module_from_spec(_RP_SPEC)
_RP_SPEC.loader.exec_module(rp)  # type: ignore[union-attr]


FLOW_LINE = ("He's plummeting down the ravine, every impact stacking, "
             "until the bottom finally catches him and the pain catches up.")
SOLO_LINE = "The quest window flares bright inside his vision."


def _segments_beat(gid=7, error=None):
    beat = {
        "group_id": gid,
        "beat_id": 1,
        "beat_title": "Fall",
        "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"],
        "narration": f"{FLOW_LINE} {SOLO_LINE}",
        "what_happens": "He falls; the system pings.",
        "hook": "The count begins.",
        "mood_words": [],
        "segments": [
            {"span": ["p1.jpg", "p2.jpg", "p3.jpg"], "line": FLOW_LINE},
            {"span": ["p4.jpg"], "line": SOLO_LINE},
        ],
    }
    if error:
        beat["error"] = error
    return beat


def _payload(gid=7, files=("p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg")):
    return {"beats": [{"group_id": gid, "beat_id": 1,
                       "allowed_scene_files": list(files),
                       "scene_files": list(files),
                       "ocr_snippets_by_scene_file": {}}]}


# ---- one paragraph + one shot per segment -----------------------------------

def test_segments_beat_two_spans_two_paragraphs_two_shots():
    sec = se._build_verbatim_section(
        section_index=0, chunk=[_segments_beat()], payload=_payload(),
        word_target=120, genre_mode="action")
    assert len(sec["script_paragraphs"]) == 2
    assert len(sec["tts_paragraphs_v3"]) == 2
    shots = sec["shots"]
    assert len(shots) == 2
    # span order kept, basenames
    assert shots[0]["scene_files"] == ["p1.jpg", "p2.jpg", "p3.jpg"]
    assert shots[1]["scene_files"] == ["p4.jpg"]
    assert "plummeting" in sec["script_paragraphs"][0].lower()
    assert "quest window" in sec["script_paragraphs"][1].lower()


def test_segments_span_head_is_first_scene_file_and_protected_in_plan():
    """The span head leads the shot's scene_files, so the planner's
    primary_scene_file (= scene_files[0]) is the span head — and
    narrated_files_from_plan protects it against the dedup passes."""
    sec = se._build_verbatim_section(
        section_index=0, chunk=[_segments_beat()], payload=_payload(),
        word_target=120, genre_mode="action")
    head = sec["shots"][0]["scene_files"][0]
    assert head == "p1.jpg"
    # the planner emits primary_scene_file = segment_scene_files[0]; the
    # protection reader keys on exactly that field.
    plan = {"timeline": [{
        "segment_id": "g0007_p00",
        "tts_text": sec["script_paragraphs"][0],
        "primary_scene_file": head,
        "scene_files": sec["shots"][0]["scene_files"],
    }]}
    assert head in rp.narrated_files_from_plan(plan)


def test_segments_mood_tag_uses_max_intensity_across_span():
    """Span mood = MAX panel intensity across the span (peaks preserved): a
    calm-calm-EXPLOSIVE span escalates, the calm solo does not."""
    beat = _segments_beat()
    beat["scene_selection"] = [
        {"scene_file": "p1.jpg", "role": "keep", "intensity": "calm"},
        {"scene_file": "p2.jpg", "role": "keep", "intensity": "calm"},
        {"scene_file": "p3.jpg", "role": "keep", "intensity": "explosive"},
        {"scene_file": "p4.jpg", "role": "keep", "intensity": "calm"},
    ]
    sec = se._build_verbatim_section(
        section_index=0, chunk=[beat], payload=_payload(),
        word_target=120, genre_mode="action")
    tts = sec["tts_paragraphs_v3"]
    assert len(tts) == 2
    span_tag = se._split_leading_bracket_tag(tts[0])[0]
    solo_tag = se._split_leading_bracket_tag(tts[1])[0]
    assert span_tag == "excited"     # explosive peak inside the span escalates
    assert solo_tag != "excited"     # the calm solo keeps its own mood


def test_segments_caps_normalization_applied_per_line():
    beat = _segments_beat()
    beat["segments"][1]["line"] = 'One assassin sneers, "KILL HIM!"'
    sec = se._build_verbatim_section(
        section_index=0, chunk=[beat], payload=_payload(),
        word_target=120, genre_mode="action")
    assert sec["script_paragraphs"][1] == 'One assassin sneers, "Kill him!"'
    tag, rest = se._split_leading_bracket_tag(sec["tts_paragraphs_v3"][1])
    assert tag in se.V3_VALID_TAGS
    assert rest == sec["script_paragraphs"][1]


# ---- tts_merge_short retired for segments-shaped beats ----------------------

def test_merge_short_is_noop_for_segments_beats():
    """Flow spans supersede the short-line merger: a segments-shaped beat keeps
    one paragraph/shot per segment even with tts_merge_short=True."""
    beat = {
        "group_id": 3,
        "scene_files": ["p1.jpg", "p2.jpg"],
        "narration": "Run. Hide.",
        "segments": [
            {"span": ["p1.jpg"], "line": "Run."},
            {"span": ["p2.jpg"], "line": "Hide."},
        ],
    }
    sec = se._build_verbatim_section(
        section_index=0, chunk=[beat],
        payload=_payload(gid=3, files=("p1.jpg", "p2.jpg")),
        word_target=120, genre_mode="action", tts_merge_short=True)
    assert len(sec["shots"]) == 2, "merger must not double-merge flow segments"
    assert [s["scene_files"] for s in sec["shots"]] == [["p1.jpg"], ["p2.jpg"]]


def test_merge_short_still_merges_legacy_panel_narration():
    """Legacy manifests keep the old behavior: short adjacent panel lines merge
    when tts_merge_short=True."""
    beat = {
        "group_id": 3,
        "scene_files": ["p1.jpg", "p2.jpg"],
        "narration": "Run. Hide.",
        "panel_narration": [
            {"scene_file": "p1.jpg", "line": "Run."},
            {"scene_file": "p2.jpg", "line": "Hide."},
        ],
    }
    sec = se._build_verbatim_section(
        section_index=0, chunk=[beat],
        payload=_payload(gid=3, files=("p1.jpg", "p2.jpg")),
        word_target=120, genre_mode="action", tts_merge_short=True)
    assert len(sec["shots"]) == 1, "legacy merge path must keep working"
    assert sec["shots"][0]["scene_files"] == ["p1.jpg", "p2.jpg"]


# ---- error beats: segments backfill honored (panel-collapse guard) ----------

def test_error_beat_with_valid_segments_keeps_every_span():
    beat = _segments_beat(error="parse_failed_after_retries")
    sec = se._build_verbatim_section(
        section_index=0, chunk=[beat], payload=_payload(),
        word_target=120, genre_mode="action")
    assert len(sec["shots"]) == 2
    union = {f for s in sec["shots"] for f in (s.get("scene_files") or [])}
    assert union == {"p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"}
    joined = " ".join(sec["script_paragraphs"]).lower()
    assert "the scene continues" not in joined


# ---- CLI: segment_ids enumerate paragraphs (fewer segments renumber) --------

def test_cli_segment_ids_enumerate_segments(tmp_path):
    beats = {"beats": [_segments_beat()]}
    beats_p = tmp_path / "manifest.beats.json"
    out_p = tmp_path / "manifest.script.json"
    beats_p.write_text(json.dumps(beats))

    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    r = subprocess.run(
        [sys.executable, str(_TOOLS / "script_expander.py"),
         "--beats", str(beats_p), "--out", str(out_p),
         "--narration-source", "gemini_verbatim"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"expander failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"

    obj = json.loads(out_p.read_text())
    sec = obj["sections"][0]
    assert [s["segment_id"] for s in sec["shots"]] == ["g0007_p00", "g0007_p01"]
    assert sec["shots"][0]["scene_files"] == ["p1.jpg", "p2.jpg", "p3.jpg"]
    assert len(sec["script_paragraphs"]) == 2
    assert len(sec["tts_meta"]) == 2
