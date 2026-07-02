"""Adaptive flow segments (spec 2026-07-02): the beats writer emits
beats[].segments[] = [{"span": [scene_files...], "line": "..."}] — flow
passages spanning 2-4 consecutive panels voiced as ONE clip, solo lines where
a moment lands. A deterministic validator enforces exact cover, span cap,
system-solo, and the duration-aware word budget; on failure ONE repair re-ask,
then the align_panel_narration singleton fallback (never block the chapter).

per_panel mode short-circuits to the legacy path byte-compatibly (covered by
the existing narrative-pass tests + the e2e here).
"""
from __future__ import annotations

import json
import sys

import pytest

import tools.gemini_narrative_pass as gnp


def _words(n: int) -> str:
    """A line of exactly n words."""
    return " ".join(["word"] * n)


# ---------------------------------------------------------------------------
# validate_segments — deterministic guardrails (pure)
# ---------------------------------------------------------------------------

FILES = ["p1.jpg", "p2.jpg", "p3.jpg"]
KINDS = {f: "story" for f in FILES}


def test_valid_partition_passes():
    segs = [{"span": ["p1.jpg", "p2.jpg"], "line": _words(18)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    assert gnp.validate_segments(segs, FILES, KINDS) == []


def test_skipped_panel_flagged():
    segs = [{"span": ["p1.jpg", "p2.jpg"], "line": _words(18)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("skip" in e and "p3.jpg" in e for e in errs)


def test_repeated_panel_flagged():
    segs = [{"span": ["p1.jpg", "p2.jpg"], "line": _words(18)},
            {"span": ["p2.jpg"], "line": _words(8)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("repeat" in e and "p2.jpg" in e for e in errs)


def test_unknown_panel_flagged():
    segs = [{"span": ["p1.jpg", "zzz.jpg"], "line": _words(18)},
            {"span": ["p2.jpg"], "line": _words(8)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("unknown" in e and "zzz.jpg" in e for e in errs)


def test_out_of_order_flagged():
    segs = [{"span": ["p2.jpg"], "line": _words(8)},
            {"span": ["p1.jpg"], "line": _words(8)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("order" in e for e in errs)


def test_span_cap_enforced():
    files = [f"p{i}.jpg" for i in range(1, 6)]           # 5 panels
    segs = [{"span": files, "line": _words(30)}]
    errs = gnp.validate_segments(segs, files, {f: "story" for f in files})
    assert gnp.SPAN_CAP == 4
    assert any("cap" in e for e in errs)


def test_system_panel_must_be_solo():
    kinds = dict(KINDS, **{"p2.jpg": "system"})
    flow = [{"span": ["p1.jpg", "p2.jpg"], "line": _words(18)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(flow, FILES, kinds)
    assert any("system" in e and "p2.jpg" in e for e in errs)
    # solo system card is fine
    solo = [{"span": ["p1.jpg"], "line": _words(8)},
            {"span": ["p2.jpg"], "line": _words(8)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    assert gnp.validate_segments(solo, FILES, kinds) == []


def test_word_budget_rejects_thin_and_fat():
    # WPM=135 -> 2.25 words/s; budget = N*2.0s .. N*6.0s per segment
    assert gnp.WPM == 135
    thin = [{"span": ["p1.jpg"], "line": _words(3)},          # 1.33s < 2.0s
            {"span": ["p2.jpg"], "line": _words(8)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(thin, FILES, KINDS)
    assert any("thin" in e for e in errs)

    fat = [{"span": ["p1.jpg"], "line": _words(20)},          # 8.9s > 6.0s
           {"span": ["p2.jpg"], "line": _words(8)},
           {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(fat, FILES, KINDS)
    assert any("fat" in e for e in errs)

    # a flow span too thin for its panel count (10 words over 3 panels = 4.4s < 6s)
    thin_flow = [{"span": FILES, "line": _words(10)}]
    errs = gnp.validate_segments(thin_flow, FILES, KINDS)
    assert any("thin" in e for e in errs)


def test_word_budget_boundaries_inclusive():
    # 9 words / 2.25 wps = 4.0s == 2 panels * 2.0s -> allowed
    segs = [{"span": ["p1.jpg", "p2.jpg"], "line": _words(9)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    assert gnp.validate_segments(segs, FILES, KINDS) == []


def test_empty_line_and_mood_prefix_flagged():
    segs = [{"span": ["p1.jpg"], "line": ""},
            {"span": ["p2.jpg"], "line": "[tense] " + _words(8)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("empty" in e for e in errs)
    assert any("mood" in e or "bracket" in e for e in errs)


def test_validator_reports_multiple_errors():
    segs = [{"span": ["p1.jpg"], "line": _words(3)},
            {"span": ["p3.jpg"], "line": _words(8)}]          # thin + skips p2
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert len(errs) >= 2


# ---------------------------------------------------------------------------
# finalize_adaptive_beat — validate, ONE repair re-ask, singleton fallback
# ---------------------------------------------------------------------------

U_BY_FILE = {f: {"action": f"He crosses toward {f}."} for f in FILES}

GOOD_SEGMENTS = [{"span": ["p1.jpg", "p2.jpg"], "line": _words(18)},
                 {"span": ["p3.jpg"], "line": _words(8)}]
BAD_SEGMENTS = [{"span": ["p1.jpg", "p2.jpg"], "line": _words(18)}]   # skips p3


def test_valid_segments_kept_without_reask():
    calls = []

    def reask(errors):
        calls.append(errors)
        return None

    beat = {"group_id": 7, "scene_files": FILES,
            "segments": [dict(s) for s in GOOD_SEGMENTS],
            "panel_narration": [{"scene_file": "p1.jpg", "line": "stale"}],
            "narration": "model join"}
    gnp.finalize_adaptive_beat(beat, FILES, KINDS, U_BY_FILE, 7, reask_fn=reask)
    assert calls == []                                   # no re-ask needed
    assert beat["segments"] == GOOD_SEGMENTS            # normalized copy kept
    assert "panel_narration" not in beat                 # segments replaces it
    assert beat["narration"] == " ".join(s["line"] for s in GOOD_SEGMENTS)


def test_bad_then_good_repair_reask_adopts_fixed_segments():
    calls = []

    def reask(errors):
        calls.append(list(errors))
        return {"group_id": 7, "scene_files": FILES,
                "segments": [dict(s) for s in GOOD_SEGMENTS]}

    beat = {"group_id": 7, "scene_files": FILES,
            "segments": [dict(s) for s in BAD_SEGMENTS]}
    gnp.finalize_adaptive_beat(beat, FILES, KINDS, U_BY_FILE, 7, reask_fn=reask)
    assert len(calls) == 1                               # exactly ONE re-ask
    assert any("p3.jpg" in e for e in calls[0])          # errors passed through
    assert beat["segments"] == GOOD_SEGMENTS


def test_bad_bad_falls_back_to_singleton_spans(capsys):
    def reask(errors):
        return {"group_id": 7, "scene_files": FILES,
                "segments": [dict(s) for s in BAD_SEGMENTS]}   # still bad

    beat = {"group_id": 7, "scene_files": FILES,
            "segments": [dict(s) for s in BAD_SEGMENTS]}
    gnp.finalize_adaptive_beat(beat, FILES, KINDS, U_BY_FILE, 7, reask_fn=reask)
    spans = [s["span"] for s in beat["segments"]]
    assert spans == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]      # exact singleton cover
    assert all(s["line"] for s in beat["segments"])           # padded, never empty
    assert beat["narration"] == " ".join(s["line"] for s in beat["segments"])
    assert "fallback beat g0007" in capsys.readouterr().out   # logged


def test_parse_error_beat_skips_reask_and_falls_back():
    beat = {"group_id": 7, "scene_files": FILES,
            "error": "parse_failed_after_retries"}            # no segments at all
    gnp.finalize_adaptive_beat(beat, FILES, KINDS, U_BY_FILE, 7, reask_fn=None)
    assert [s["span"] for s in beat["segments"]] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert all(s["line"] for s in beat["segments"])


def test_demote_backfilled_error_honors_segments_shape():
    beat = {"group_id": 7, "error": "parse_failed_after_retries",
            "segments": [{"span": ["p1.jpg"], "line": "He falls hard tonight."}]}
    out = gnp.demote_backfilled_error(beat)
    assert "error" not in out
    assert out["group_parse_error"] == "parse_failed_after_retries"


# ---------------------------------------------------------------------------
# schema + CLI + prompt
# ---------------------------------------------------------------------------

def test_beat_schema_adaptive_has_segments_not_panel_narration():
    schema = gnp.build_beat_schema()                     # tool default = adaptive
    props = schema["properties"]
    assert "segments" in props and "panel_narration" not in props
    item = props["segments"]["items"]["properties"]
    assert set(item) >= {"span", "line"}
    assert "segments" in schema["required"]
    assert "panel_narration" not in schema["required"]
    assert "narration" in props                          # the join stays


def test_beat_schema_per_panel_is_legacy():
    schema = gnp.build_beat_schema("per_panel")
    assert "panel_narration" in schema["properties"]
    assert "segments" not in schema["properties"]
    assert "panel_narration" in schema["required"]


def test_cli_default_adaptive_env_overrides_flag_wins(monkeypatch):
    base = ["--groups-manifest", "g.json", "--vision-manifest", "v.json",
            "--out", "o.json"]
    monkeypatch.delenv("STUDIO_NARR_SEGMENTATION", raising=False)
    assert gnp.build_arg_parser().parse_args(base).segmentation == "adaptive"

    monkeypatch.setenv("STUDIO_NARR_SEGMENTATION", "per_panel")
    assert gnp.build_arg_parser().parse_args(base).segmentation == "per_panel"

    # explicit flag wins over env
    args = gnp.build_arg_parser().parse_args(base + ["--segmentation", "adaptive"])
    assert args.segmentation == "adaptive"

    # garbage env normalizes to adaptive (argparse skips choices on defaults)
    monkeypatch.setenv("STUDIO_NARR_SEGMENTATION", "bogus")
    assert gnp.build_arg_parser().parse_args(base).segmentation == "adaptive"


def test_adaptive_prompt_criteria_and_bans():
    text = gnp._ADAPTIVE_NARRATION_INSTRUCTION
    assert "segments" in text and "span" in text
    assert "FLOW" in text and "SOLO" in text
    assert "in the next panel" in text                   # named as BANNED
    assert "BANNED" in text or "banned" in text
    assert "WORD BUDGET" in text
    # the legacy instruction stays available for per_panel byte-compat
    assert "EVERY panel its own line" in gnp._PER_PANEL_NARRATION_INSTRUCTION


# ---------------------------------------------------------------------------
# main() e2e with a stubbed model — writer output shape per mode
# ---------------------------------------------------------------------------

def _write_manifests(tmp_path, files=tuple(FILES), system_files=()):
    groups = {"shots": [{"shot_id": 7, "scene_files": list(files),
                         "arc_label": "opening", "intensity": "tense"}]}
    vision = {"items": [{"scene_file": f, "ocr_clean": "", "vision": {}}
                        for f in files]}
    understood = {"panels": [
        {"scene_file": f,
         "description": f"A figure moves near {f}.",
         "action": f"He crosses toward {f}.",
         "panel_kind": "system" if f in system_files else "story",
         "intensity": "tense", "subjects": ["the prince"]} for f in files]}
    g = tmp_path / "groups.json"
    v = tmp_path / "vision.json"
    u = tmp_path / "understood.json"
    g.write_text(json.dumps(groups))
    v.write_text(json.dumps(vision))
    u.write_text(json.dumps(understood))
    return g, v, u


def _run_main(tmp_path, monkeypatch, responses, extra_argv=()):
    """Drive gnp.main() with a stubbed model that returns `responses` in order
    (the last response repeats if the tool asks again)."""
    g, v, u = _write_manifests(tmp_path)
    out = tmp_path / "beats.json"
    calls = []

    def stub(**kw):
        calls.append(kw)
        obj = responses[min(len(calls) - 1, len(responses) - 1)]
        return dict(obj), "raw", {"input": 1, "output": 1, "cached": 0}

    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    monkeypatch.delenv("STUDIO_NARR_SEGMENTATION", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "gemini_narrative_pass.py", "--groups-manifest", str(g),
        "--vision-manifest", str(v), "--out", str(out),
        "--understood", str(u), "--backend", "ollama",
        "--min-sleep", "0", *extra_argv])
    assert gnp.main() == 0
    return json.loads(out.read_text()), calls


_GOOD_MODEL_BEAT = {
    "beat_title": "Opening", "what_happens": "He crosses the hall.",
    "narration": "model join placeholder for the accept loop only.",
    "segments": [
        {"span": ["p1.jpg", "p2.jpg"],
         "line": "He plummets down the ravine, every impact stacking, until "
                 "the bottom finally catches him and the pain arrives."},
        {"span": ["p3.jpg"],
         "line": "The stranger's eyes snap open in the dark."},
    ],
    "scene_selection": [],
}


def test_main_adaptive_emits_segments_and_join(tmp_path, monkeypatch):
    out, calls = _run_main(tmp_path, monkeypatch, [_GOOD_MODEL_BEAT])
    assert len(calls) == 1                               # no re-ask
    beat = out["beats"][0]
    assert "panel_narration" not in beat                 # segments replaces it
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]
    assert beat["narration"] == " ".join(
        s["line"] for s in beat["segments"])             # load-bearing join


def test_main_adaptive_repair_reask_then_adopts(tmp_path, monkeypatch):
    bad = dict(_GOOD_MODEL_BEAT,
               segments=[_GOOD_MODEL_BEAT["segments"][0]])    # skips p3.jpg
    out, calls = _run_main(tmp_path, monkeypatch, [bad, _GOOD_MODEL_BEAT])
    assert len(calls) == 2                               # ONE repair re-ask
    assert "SEGMENT REPAIR" in calls[1]["system_instruction"]
    assert "p3.jpg" in calls[1]["system_instruction"]    # exact errors appended
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]


def test_main_adaptive_bad_bad_singleton_fallback(tmp_path, monkeypatch):
    bad = dict(_GOOD_MODEL_BEAT,
               segments=[_GOOD_MODEL_BEAT["segments"][0]])    # skips p3.jpg
    out, calls = _run_main(tmp_path, monkeypatch, [bad, bad])
    assert len(calls) == 2                               # asked once, re-asked once
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]              # never blocks the chapter
    assert all(s["line"] for s in beat["segments"])


def test_main_per_panel_stays_legacy_shape(tmp_path, monkeypatch):
    legacy = {
        "beat_title": "Opening", "what_happens": "He crosses the hall.",
        "narration": "join placeholder.",
        "panel_narration": [
            {"scene_file": "p1.jpg", "line": "He steps into the hall."},
            {"scene_file": "p2.jpg", "line": "The doors slam shut behind."},
            {"scene_file": "p3.jpg", "line": "A blade glints in the dark."},
        ],
        "scene_selection": [],
    }
    out, calls = _run_main(tmp_path, monkeypatch, [legacy],
                           extra_argv=("--segmentation", "per_panel"))
    assert len(calls) == 1
    beat = out["beats"][0]
    assert "segments" not in beat                        # byte-compatible legacy
    assert [p["scene_file"] for p in beat["panel_narration"]] == FILES
    assert beat["narration"] == " ".join(
        p["line"] for p in beat["panel_narration"])
