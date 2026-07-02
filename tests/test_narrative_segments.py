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
    # WPM=135 -> 2.25 words/s; budget = N*1.0s .. N*15.0s per segment (the
    # ceiling is a lenient bloat guard — 6.0s hard-failed gemma's natural
    # money-shot rhythm on real ch1 and 18/21 beats fell back)
    assert gnp.WPM == 135
    thin = [{"span": ["p1.jpg"], "line": _words(2)},          # 0.89s < 1.0s
            {"span": ["p2.jpg"], "line": _words(8)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(thin, FILES, KINDS)
    assert any("thin" in e for e in errs)

    ok_hold = [{"span": ["p1.jpg"], "line": _words(20)},      # 8.9s: a money-
               {"span": ["p2.jpg"], "line": _words(8)},       # shot hold, VALID
               {"span": ["p3.jpg"], "line": _words(8)}]
    assert gnp.validate_segments(ok_hold, FILES, KINDS) == []

    fat = [{"span": ["p1.jpg"], "line": _words(35)},          # 15.6s > 15.0s
           {"span": ["p2.jpg"], "line": _words(8)},
           {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(fat, FILES, KINDS)
    assert any("fat" in e for e in errs)

    # a flow span too thin for its panel count (6 words over 3 panels = 2.7s < 3s)
    thin_flow = [{"span": FILES, "line": _words(6)}]
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
    segs = [{"span": ["p1.jpg"], "line": _words(2)},
            {"span": ["p3.jpg"], "line": _words(8)}]          # thin + skips p2
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert len(errs) >= 2


# ---------------------------------------------------------------------------
# finalize_adaptive_beat — validate, ONE repair re-ask, singleton fallback
# ---------------------------------------------------------------------------

U_BY_FILE = {f: {"action": f"He crosses toward {f}."} for f in FILES}

GOOD_SEGMENTS = [{"span": ["p1.jpg", "p2.jpg"], "line": _words(18)},
                 {"span": ["p3.jpg"], "line": _words(8)}]
# out-of-order span — auto_repair_segments never reorders, so this still
# reaches the model repair re-ask / singleton fallback paths
BAD_SEGMENTS = [{"span": ["p2.jpg", "p1.jpg"], "line": _words(18)},
                {"span": ["p3.jpg"], "line": _words(8)}]


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
    assert any("order" in e for e in calls[0])           # errors passed through
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


_BAD_ORDER_SEGMENTS = [
    {"span": ["p2.jpg", "p1.jpg"], "line": _GOOD_MODEL_BEAT["segments"][0]["line"]},
    {"span": ["p3.jpg"], "line": "The stranger watches from the ridge."},
]


def test_main_adaptive_repair_reask_then_adopts(tmp_path, monkeypatch):
    bad = dict(_GOOD_MODEL_BEAT, segments=list(_BAD_ORDER_SEGMENTS))
    out, calls = _run_main(tmp_path, monkeypatch, [bad, _GOOD_MODEL_BEAT])
    assert len(calls) == 2                               # ONE repair re-ask
    assert "SEGMENT REPAIR" in calls[1]["system_instruction"]
    assert "order" in calls[1]["system_instruction"]     # exact errors appended
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]


def test_main_adaptive_bad_bad_singleton_fallback(tmp_path, monkeypatch):
    bad = dict(_GOOD_MODEL_BEAT, segments=list(_BAD_ORDER_SEGMENTS))
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


# ---------------------------------------------------------------------------
# span-pinned heal regen (Chunk 3, spec 3.5): a corrected group whose EXISTING
# beat carries native segments must keep its spans — the writer rewrites LINES
# only. A re-split would renumber sibling segment_ids -> per-clip TTS cache
# churn + audio_stale. Violations fall back to the previous lines (logged);
# only a full beats re-run (no --resume) may change spans.
# ---------------------------------------------------------------------------

PREV_FLOW = ("He drops through the canopy, bounces off two branches, and "
             "lands where nobody thought to watch.")
PREV_SOLO = "The stranger's eyes snap open in the dark."


def _prev_segments_beat():
    return {
        "group_id": 7, "scene_files": list(FILES),
        "beat_title": "Opening", "what_happens": "He crosses the hall.",
        "segments": [
            {"span": ["p1.jpg", "p2.jpg"], "line": PREV_FLOW},
            {"span": ["p3.jpg"], "line": PREV_SOLO},
        ],
        "narration": PREV_FLOW + " " + PREV_SOLO,
        "scene_selection": [],
    }


def _run_corrections(tmp_path, monkeypatch, responses, prev_beat,
                     extra_argv=()):
    """Drive main() over an EXISTING beats.json with a correction queued for
    group 7 (--resume --corrections) and a stubbed model."""
    g, v, u = _write_manifests(tmp_path)
    out = tmp_path / "beats.json"
    out.write_text(json.dumps({"count_beats": 1, "beats": [prev_beat]}))
    corr = tmp_path / "corr.json"
    corr.write_text(json.dumps({"7": "Weave the caption into the narration."}))
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
        "--min-sleep", "0", "--resume", "--corrections", str(corr),
        *extra_argv])
    assert gnp.main() == 0
    return json.loads(out.read_text()), calls


def test_corrections_prompt_pins_the_existing_spans(tmp_path, monkeypatch):
    rewrite = dict(_GOOD_MODEL_BEAT)          # same spans, new lines
    _, calls = _run_corrections(tmp_path, monkeypatch, [rewrite],
                                _prev_segments_beat())
    sysi = calls[0]["system_instruction"]
    assert "CORRECTION FOR THIS GROUP" in sysi
    assert "FIXED SEGMENTATION" in sysi                    # spans are locked
    assert "p1.jpg, p2.jpg" in sysi and "p3.jpg" in sysi   # exact spans listed


def test_corrections_compliant_rewrite_same_spans_adopted(tmp_path,
                                                          monkeypatch):
    rewrite = dict(_GOOD_MODEL_BEAT)          # spans [p1,p2],[p3]; fresh lines
    out, calls = _run_corrections(tmp_path, monkeypatch, [rewrite],
                                  _prev_segments_beat())
    assert len(calls) == 1
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]                  # spans preserved
    lines = [s["line"] for s in beat["segments"]]
    assert lines == [s["line"] for s in _GOOD_MODEL_BEAT["segments"]]
    assert lines[0] != PREV_FLOW                           # rewrite adopted
    assert beat["narration"] == " ".join(lines)            # join rebuilt


def test_corrections_resplit_keeps_previous_lines(tmp_path, monkeypatch,
                                                  capsys):
    # a VALID partition that differs from the pinned spans — must NOT be
    # adopted (it would renumber sibling segment_ids -> clip-cache churn)
    resplit = dict(_GOOD_MODEL_BEAT, segments=[
        {"span": ["p1.jpg"], "line": _words(8)},
        {"span": ["p2.jpg", "p3.jpg"], "line": _words(18)},
    ])
    out, calls = _run_corrections(tmp_path, monkeypatch, [resplit],
                                  _prev_segments_beat())
    assert len(calls) == 1                    # validator-valid: no re-ask
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]                  # spans identical
    assert [s["line"] for s in beat["segments"]] == [PREV_FLOW, PREV_SOLO]
    assert beat["narration"] == PREV_FLOW + " " + PREV_SOLO
    assert "span-pin" in capsys.readouterr().out           # fallback logged


def test_corrections_singleton_fallback_cannot_resplit_pinned_beat(
        tmp_path, monkeypatch):
    # model answers stay INVALID (out-of-order span; auto-repair never
    # reorders) -> repair re-ask, then the singleton fallback — which is
    # itself a re-split of the pinned flow span, so the previous lines win.
    bad = dict(_GOOD_MODEL_BEAT, segments=list(_BAD_ORDER_SEGMENTS))
    out, calls = _run_corrections(tmp_path, monkeypatch, [bad, bad],
                                  _prev_segments_beat())
    assert len(calls) == 2                    # asked once, re-asked once
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]
    assert [s["line"] for s in beat["segments"]] == [PREV_FLOW, PREV_SOLO]


def test_corrections_pin_holds_even_under_per_panel_flag(tmp_path,
                                                         monkeypatch):
    # pinning derives from the EXISTING beat's shape, not --segmentation:
    # a per_panel-mode regen of a native-segments beat may not singletonize it
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
    out, _ = _run_corrections(tmp_path, monkeypatch, [legacy],
                              _prev_segments_beat(),
                              extra_argv=("--segmentation", "per_panel"))
    beat = out["beats"][0]
    assert [s["span"] for s in beat.get("segments") or []] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]                  # spans preserved
    assert [s["line"] for s in beat["segments"]] == [PREV_FLOW, PREV_SOLO]


def test_corrections_legacy_prev_beat_keeps_todays_behavior(tmp_path,
                                                            monkeypatch):
    # the pin only exists for native-segments beats: a legacy per-panel beat
    # under corrections regenerates exactly as today (adaptive re-write,
    # fresh spans allowed — the pre-flow manifests never had spans to keep)
    prev = {
        "group_id": 7, "scene_files": list(FILES),
        "beat_title": "Opening", "what_happens": "He crosses the hall.",
        "panel_narration": [
            {"scene_file": "p1.jpg", "line": "Old line one for panel one."},
            {"scene_file": "p2.jpg", "line": "Old line two for panel two."},
            {"scene_file": "p3.jpg", "line": "Old line three for panel three."},
        ],
        "narration": "Old line one. Old line two. Old line three.",
        "scene_selection": [],
    }
    out, calls = _run_corrections(tmp_path, monkeypatch, [_GOOD_MODEL_BEAT],
                                  prev)
    assert len(calls) == 1
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]                  # fresh spans OK
    assert "FIXED SEGMENTATION" not in calls[0]["system_instruction"]


# ---- flow-nudge: a VALID all-singleton answer on a big beat gets ONE nudge ---

FILES5 = ["q1.jpg", "q2.jpg", "q3.jpg", "q4.jpg", "q5.jpg"]
KINDS5 = {f: "story" for f in FILES5}
U5 = {f: {"description": f"what panel {f} shows"} for f in FILES5}


def _five_solos():
    return [{"span": [f], "line": f"Panel {i} keeps the story moving forward."}
            for i, f in enumerate(FILES5)]


_MIXED5 = [
    {"span": ["q1.jpg", "q2.jpg", "q3.jpg"],
     "line": ("He tears through the underbrush, blade out, and the whole "
              "hunt turns on him in one breath of moving steel.")},
    {"span": ["q4.jpg"], "line": "The masked hunter finally shows himself."},
    {"span": ["q5.jpg"], "line": "And our guy realizes nobody is leaving."},
]


def test_all_singleton_big_beat_gets_flow_nudge_adopted(capsys):
    calls = []

    def reask(errors):
        calls.append(list(errors))
        return {"group_id": 9, "scene_files": FILES5,
                "segments": [dict(s) for s in _MIXED5]}

    beat = {"group_id": 9, "scene_files": FILES5, "segments": _five_solos()}
    gnp.finalize_adaptive_beat(beat, FILES5, KINDS5, U5, 9, reask_fn=reask)
    assert len(calls) == 1
    assert any("single-panel captions" in e for e in calls[0])
    assert [len(s["span"]) for s in beat["segments"]] == [3, 1, 1]
    assert "flow-nudge beat g0009 adopted" in capsys.readouterr().out


def test_flow_nudge_model_insists_all_solo_keeps_original():
    calls = []

    def reask(errors):
        calls.append(1)
        return {"group_id": 9, "scene_files": FILES5,
                "segments": _five_solos()}     # valid but still all-solo

    original = _five_solos()
    beat = {"group_id": 9, "scene_files": FILES5,
            "segments": [dict(s) for s in original]}
    gnp.finalize_adaptive_beat(beat, FILES5, KINDS5, U5, 9, reask_fn=reask)
    assert calls == [1]                        # nudged once, not looped
    assert beat["segments"] == original        # the model insisted; accepted


def test_flow_nudge_invalid_answer_keeps_original():
    def reask(errors):
        return {"group_id": 9, "scene_files": FILES5,
                "segments": [{"span": FILES5, "line": "too big a span"}]}

    original = _five_solos()
    beat = {"group_id": 9, "scene_files": FILES5,
            "segments": [dict(s) for s in original]}
    gnp.finalize_adaptive_beat(beat, FILES5, KINDS5, U5, 9, reask_fn=reask)
    assert beat["segments"] == original


def test_no_flow_nudge_below_min_panels():
    calls = []

    def reask(errors):
        calls.append(1)
        return None

    beat = {"group_id": 7, "scene_files": FILES,
            "segments": [dict(s) for s in GOOD_SEGMENTS]}
    gnp.finalize_adaptive_beat(beat, FILES, KINDS, U_BY_FILE, 7, reask_fn=reask)
    assert calls == []                         # 3-panel beat: no nudge


def test_no_flow_nudge_on_pinned_regen():
    calls = []

    def reask(errors):
        calls.append(1)
        return None

    beat = {"group_id": 9, "scene_files": FILES5, "segments": _five_solos()}
    gnp.finalize_adaptive_beat(beat, FILES5, KINDS5, U5, 9, reask_fn=reask,
                               allow_flow_nudge=False)
    assert calls == []


def test_adaptive_prompt_pins_anti_parrot_and_flow_default():
    text = gnp._ADAPTIVE_NARRATION_INSTRUCTION
    assert "RAW MATERIAL" in text
    assert "DEFAULT TO FLOW" in text
    assert "The character" in text             # named as a banned opener
    assert "FLOW" in text and "SOLO" in text   # original pins still hold
    assert "in the next panel" in text


# ---- auto_repair_segments: structural repair keeps the model's prose --------

def test_auto_repair_inserts_skipped_panel_as_grounded_pad():
    segs = [{"span": ["p1.jpg", "p2.jpg"], "line": _words(18)}]   # skips p3
    out = gnp.auto_repair_segments(segs, FILES, KINDS, U_BY_FILE)
    assert [s["span"] for s in out] == [["p1.jpg", "p2.jpg"], ["p3.jpg"]]
    assert out[0]["line"] == _words(18)                  # prose untouched
    assert out[1]["line"] == "He crosses toward p3.jpg."  # grounded pad
    assert gnp.validate_segments(out, FILES, KINDS) == []


def test_auto_repair_extracts_system_panel_from_span():
    kinds = dict(KINDS, **{"p2.jpg": "system"})
    segs = [{"span": ["p1.jpg", "p2.jpg", "p3.jpg"], "line": _words(18)}]
    out = gnp.auto_repair_segments(segs, FILES, kinds, U_BY_FILE)
    assert [s["span"] for s in out] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert out[0]["line"] == _words(18)                  # line stays on story head
    assert out[1]["line"] == "He crosses toward p2.jpg."  # card gets the pad
    assert gnp.validate_segments(out, FILES, kinds) == []


def test_auto_repair_never_reorders_a_bad_span():
    out = gnp.auto_repair_segments(list(_BAD_ORDER_SEGMENTS), FILES, KINDS,
                                   U_BY_FILE)
    errs = gnp.validate_segments(out, FILES, KINDS)
    assert any("order" in e for e in errs)               # still model-repair turf


def test_main_adaptive_skip_is_auto_repaired_without_reask(tmp_path,
                                                           monkeypatch):
    # the OLD wholesale-fallback case: a skipped panel now costs one padded
    # singleton, not the whole beat's prose — and no model re-ask at all
    bad = dict(_GOOD_MODEL_BEAT,
               segments=[_GOOD_MODEL_BEAT["segments"][0]])    # skips p3.jpg
    out, calls = _run_main(tmp_path, monkeypatch, [bad])
    assert len(calls) == 1                                # no re-ask needed
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]
    assert beat["segments"][0]["line"] == \
        _GOOD_MODEL_BEAT["segments"][0]["line"]           # prose preserved
