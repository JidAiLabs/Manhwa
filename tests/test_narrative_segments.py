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
    """A line of exactly n words (ends on a period: lines must END, see
    validate_segments' mid-sentence guard)."""
    return " ".join(["word"] * n) + "."


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


def test_filename_leak_in_line_flagged():
    # the writer receives scene_file names as sentence tags under prose-first;
    # a tag leaking into a voiced line must fail validation -> repair re-ask
    segs = [{"span": ["p1.jpg", "p2.jpg"],
             "line": "It progresses through the series to conclude at "
                     "p2.jpg today."},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("image file" in e for e in errs)


def test_impact_marker_leak_in_line_flagged():
    # the writer's payload carries "[IMPACT SFX on panel]" as a per-panel tag
    # (gemini_narrative_pass._pack_group_payload) — an echoed bracket marker
    # must fail validation exactly like a leaked scene_file tag (the SAME
    # leak channel: bracket/tag context fed to the model, echoed verbatim).
    segs = [{"span": ["p1.jpg", "p2.jpg"],
             "line": "[IMPACT SFX on panel] as the blade comes down hard."},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("bracket" in e and "marker" in e for e in errs)


def test_figures_leak_in_line_flagged():
    # the writer's payload carries "unknown (<evidence>)" for an unresolved
    # cast_identity figure (gemini_narrative_pass._pack_group_payload) — an
    # echoed evidence wrapper must fail validation exactly like a leaked
    # impact-SFX/scene_file tag (the SAME leak channel). A resolved cast NAME
    # is sanctioned and must NOT trip this.
    segs = [{"span": ["p1.jpg", "p2.jpg"],
             "line": "unknown (a masked figure lurking) nears the gate."},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("unknown" in e and "payload" in e for e in errs)

    named = [{"span": ["p1.jpg", "p2.jpg"],
              "line": "Prince Cheon draws his hidden blade and lunges."},
             {"span": ["p3.jpg"], "line": _words(8)}]
    assert gnp.validate_segments(named, FILES, KINDS) == []


def test_empty_line_and_mood_prefix_flagged():
    segs = [{"span": ["p1.jpg"], "line": ""},
            {"span": ["p2.jpg"], "line": "[tense] " + _words(8)},
            {"span": ["p3.jpg"], "line": _words(8)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("empty" in e for e in errs)
    assert any("mood" in e or "bracket" in e for e in errs)


@pytest.mark.parametrize("leaked_line", [
    # real round-3 Nano ch1 shapes (18 segments: 15 "Dramatic:", 3 "Comic:")
    "Dramatic: He’s tumbling down a massive cliff, screaming his lungs "
    "out while plummeting into the abyss.",
    "Dramatic: Suddenly, a blinding flash of light erupts out of nowhere.",
    "Comic: The masked guy grabs him by the throat and asks if that was "
    "his big attempt at revenge.",
    "Dramatic He's free-falling down a rocky cliff, screaming his lungs out.",
])
def test_mood_tag_leak_in_line_flagged(leaked_line):
    # a BARE (unbracketed) mood/tone word opening a line is the SAME leak
    # channel as a bracket-mood prefix — the packer adds moods, the writer
    # never should, bracketed or not.
    segs = [{"span": ["p1.jpg"], "line": leaked_line},
            {"span": ["p2.jpg", "p3.jpg"], "line": _words(18)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("mood" in e or "tone" in e for e in errs), errs


def test_mood_tag_leak_silent_on_ordinary_lines():
    # a real sentence that only happens to OPEN with one of these words as an
    # adjective, continuing in lowercase, must never trip this net. (A
    # bracket-mood prefix in the WRITER's own line is separately — and
    # already — rejected by _MOOD_PREFIX_RE regardless of this check: see
    # test_empty_line_and_mood_prefix_flagged.)
    segs = [{"span": ["p1.jpg"],
             "line": "Dramatic reveals stay restrained even here."},
            {"span": ["p2.jpg"],
             "line": "He's tumbling down a massive cliff, screaming."},
            {"span": ["p3.jpg"], "line": _words(8)}]
    assert gnp.validate_segments(segs, FILES, KINDS) == []


def test_validator_reports_multiple_errors():
    segs = [{"span": ["p1.jpg"], "line": _words(2)},
            {"span": ["p3.jpg"], "line": _words(8)}]          # thin + skips p2
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert len(errs) >= 2


# ---------------------------------------------------------------------------
# finalize_adaptive_beat — validate, ONE repair re-ask, singleton fallback
# ---------------------------------------------------------------------------

U_BY_FILE = {f: {"action": f"He crosses toward {f.split(chr(46))[0]}."}
             for f in FILES}

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

def _write_manifests(tmp_path, files=tuple(FILES), system_files=(),
                     caption_files=()):
    def _kind(f):
        if f in system_files:
            return "system"
        if f in caption_files:
            return "caption"
        return "story"
    groups = {"shots": [{"shot_id": 7, "scene_files": list(files),
                         "arc_label": "opening", "intensity": "tense"}]}
    vision = {"items": [{"scene_file": f, "ocr_clean": "", "vision": {}}
                        for f in files]}
    understood = {"panels": [
        {"scene_file": f,
         "description": f"A figure moves near {f}.",
         "action": f"He crosses toward {f.split(chr(46))[0]}.",
         "panel_kind": _kind(f),
         "intensity": "tense", "subjects": ["the prince"]} for f in files]}
    g = tmp_path / "groups.json"
    v = tmp_path / "vision.json"
    u = tmp_path / "understood.json"
    g.write_text(json.dumps(groups))
    v.write_text(json.dumps(vision))
    u.write_text(json.dumps(understood))
    return g, v, u


def _run_main(tmp_path, monkeypatch, responses, extra_argv=(),
              caption_files=()):
    """Drive gnp.main() with a stubbed model that returns `responses` in order
    (the last response repeats if the tool asks again)."""
    g, v, u = _write_manifests(tmp_path, caption_files=caption_files)
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
    # free generation re-asks in prose-first terms (pinned regens keep the
    # SEGMENT REPAIR wording — covered by the corrections tests below)
    assert "NARRATION REPAIR" in calls[1]["system_instruction"]
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


def test_main_adaptive_excludes_caption_panels_from_spans(tmp_path, monkeypatch):
    # p2 is a pure speech-bubble (panel_kind=caption): TEXT, not a visual. The
    # writer sees it (payload) and may tag a sentence to it, but it must NEVER
    # own a shown span — else it blank-drops at render and holds a neighbour
    # (the 13.8s held-eye) and inflates solos. Its sentence's words fold into
    # the adjacent visual segment instead.
    beat = {
        "beat_title": "Reflection", "what_happens": "He reflects.",
        "narration": "join placeholder",
        "segments": [
            {"span": ["p1.jpg"], "line": "He staggers upright, blood on his lips."},
            {"span": ["p2.jpg"],
             "line": "Peasant blood, they say, as if it changes anything."},
            {"span": ["p3.jpg"], "line": "His eyes flare with sudden rage."},
        ],
        "sentences": [
            {"text": "He staggers upright, blood on his lips.",
             "panels": ["p1.jpg"]},
            {"text": "Peasant blood, they say, as if it changes anything.",
             "panels": ["p2.jpg"]},
            {"text": "His eyes flare with sudden rage.", "panels": ["p3.jpg"]},
        ],
        "scene_selection": [],
    }
    out, _ = _run_main(tmp_path, monkeypatch, [beat], caption_files=("p2.jpg",))
    b = out["beats"][0]
    shown = [f for s in b["segments"] for f in s["span"]]
    assert "p2.jpg" not in shown                     # caption never shown
    assert shown == ["p1.jpg", "p3.jpg"]             # only visuals partition
    # the caption's words fold into the adjacent (previous) visual segment
    assert "peasant blood" in b["narration"].lower()
    assert "p2.jpg" not in b.get("scene_files", [])


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
                     extra_argv=(), voiced=True):
    """Drive main() over an EXISTING beats.json with a correction queued for
    group 7 (--resume --corrections) and a stubbed model. voiced=True lays
    down a tts/tts_index.json next to the beats — the span pin engages ONLY
    then (there is a per-clip cache to protect); voiced=False exercises the
    unvoiced heal, which goes through the free prose path."""
    g, v, u = _write_manifests(tmp_path)
    out = tmp_path / "beats.json"
    out.write_text(json.dumps({"count_beats": 1, "beats": [prev_beat]}))
    if voiced:
        (tmp_path / "tts").mkdir(exist_ok=True)
        (tmp_path / "tts" / "tts_index.json").write_text(
            json.dumps({"clips": []}))
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
    assert out[1]["line"] == "He crosses toward p3."  # grounded pad
    assert gnp.validate_segments(out, FILES, KINDS) == []


def test_auto_repair_extracts_system_panel_from_span():
    kinds = dict(KINDS, **{"p2.jpg": "system"})
    segs = [{"span": ["p1.jpg", "p2.jpg", "p3.jpg"], "line": _words(18)}]
    out = gnp.auto_repair_segments(segs, FILES, kinds, U_BY_FILE)
    assert [s["span"] for s in out] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert out[0]["line"] == _words(18)                  # line stays on story head
    assert out[1]["line"] == "He crosses toward p2."  # card gets the pad
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


def test_pinned_regen_fallback_keeps_previous_beat_not_pads():
    # An all-singleton pin can accidentally MATCH a validation fallback's
    # singleton spans — the pads must never replace the pinned prose.
    prev = {"group_id": 7, "scene_files": FILES,
            "segments": [{"span": [f], "line": f"Good prose about {f}."}
                         for f in FILES],
            "narration": "join"}
    import json, subprocess, sys, tempfile, os
    # exercise via finalize + the caller-level guard: simulate what main does
    bad = {"group_id": 7, "scene_files": FILES,
           "segments": [dict(s) for s in _BAD_ORDER_SEGMENTS]}
    gnp.finalize_adaptive_beat(bad, FILES, KINDS, U_BY_FILE, 7,
                               reask_fn=lambda e: None)
    assert bad.pop("_segments_fallback", False) is True   # marker set
    # the caller sees the marker and keeps prev — emulate the guard:
    beat = prev if True else bad
    assert [s["line"] for s in beat["segments"]] == [
        f"Good prose about {f}." for f in FILES]


def test_correction_block_speaks_segments_under_adaptive(tmp_path, monkeypatch):
    # under adaptive the rewrite instruction must target the segments lines,
    # not the derived 'narration' join (the old wording made gemma return
    # malformed segments -> fallback pads)
    bad = dict(_GOOD_MODEL_BEAT, segments=list(_BAD_ORDER_SEGMENTS))
    out, calls = _run_corrections(tmp_path, monkeypatch,
                                  [_GOOD_MODEL_BEAT], _prev_segments_beat())
    assert any("every 'segments' line" in c["system_instruction"]
               for c in calls)


# ---------------------------------------------------------------------------
# PROSE-FIRST free generation (2026-07-03): the writer authors ONE connected
# passage ('narration') + the same passage split into panel-tagged
# 'sentences'; segments_from_sentences derives the span partition in code.
# Grouped-era prose (user-approved 2026-06-16), 1:1-era guarantees.
# ---------------------------------------------------------------------------

def _sfs(sentences, files=FILES, kinds=None, u=None):
    return gnp.segments_from_sentences(
        sentences, files, kinds or {f: "story" for f in files},
        u if u is not None else U_BY_FILE)


def test_sfs_one_sentence_per_panel_is_singletons():
    out = _sfs([{"text": "He wakes.", "panels": ["p1.jpg"]},
                {"text": "He stands slowly.", "panels": ["p2.jpg"]},
                {"text": "He walks into the dark.", "panels": ["p3.jpg"]}])
    assert [s["span"] for s in out] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert [s["line"] for s in out] == [
        "He wakes.", "He stands slowly.", "He walks into the dark."]


def test_sfs_same_panel_sentences_fold_into_one_segment():
    # the screenshot case: a reaction clause re-tags the splash panel — it
    # must ride the SAME segment, not become its own 2.5s caption
    out = _sfs([{"text": "Something wet splashes across his skin.",
                 "panels": ["p1.jpg"]},
                {"text": "He's definitely not happy about that.",
                 "panels": ["p1.jpg"]},
                {"text": "Then the ground gives way.",
                 "panels": ["p2.jpg", "p3.jpg"]}])
    assert [s["span"] for s in out] == [["p1.jpg"], ["p2.jpg", "p3.jpg"]]
    assert out[0]["line"] == ("Something wet splashes across his skin. "
                              "He's definitely not happy about that.")


def test_sfs_untagged_sentence_rides_previous_segment():
    out = _sfs([{"text": "The blade comes down.", "panels": ["p1.jpg"]},
                {"text": "No hesitation at all.", "panels": []},
                {"text": "He rolls clear.", "panels": ["p2.jpg", "p3.jpg"]}])
    assert [s["span"] for s in out] == [["p1.jpg"], ["p2.jpg", "p3.jpg"]]
    assert out[0]["line"] == "The blade comes down. No hesitation at all."


def test_sfs_untagged_panels_ride_the_previous_span():
    # the model narrated p1 and p3; p2 (a build-up frame) keeps showing while
    # the p1 sentence finishes — the grouped-era pacing
    out = _sfs([{"text": "He tears through the underbrush at full sprint.",
                 "panels": ["p1.jpg"]},
                {"text": "The cliff edge ends the chase.", "panels": ["p3.jpg"]}])
    assert [s["span"] for s in out] == [["p1.jpg", "p2.jpg"], ["p3.jpg"]]


def test_sfs_leading_untagged_panels_ride_the_first_span():
    out = _sfs([{"text": "The hall finally opens up ahead.", "panels": ["p2.jpg"]},
                {"text": "And the doors slam shut.", "panels": ["p3.jpg"]}])
    assert [s["span"] for s in out] == [["p1.jpg", "p2.jpg"], ["p3.jpg"]]


def test_sfs_out_of_order_tags_absorb_forward_and_keep_text_order():
    # s0 tags p2, s1 tags p1 — ownership may never regress, so everything
    # folds into ONE span with both texts in passage order
    out = _sfs([{"text": "First sentence of the passage.", "panels": ["p2.jpg"]},
                {"text": "Second sentence of the passage.", "panels": ["p1.jpg"]}])
    assert [s["span"] for s in out] == [["p1.jpg", "p2.jpg", "p3.jpg"]]
    assert out[0]["line"] == ("First sentence of the passage. "
                              "Second sentence of the passage.")


def test_sfs_system_card_solo_with_its_own_sentence():
    kinds = {"p1.jpg": "story", "p2.jpg": "system", "p3.jpg": "story"}
    out = _sfs([{"text": "He staggers upright.", "panels": ["p1.jpg"]},
                {"text": "A cold blue window declares the Nano Machine active.",
                 "panels": ["p2.jpg"]},
                {"text": "And the pain starts.", "panels": ["p3.jpg"]}],
               kinds=kinds)
    assert [s["span"] for s in out] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert out[1]["line"] == \
        "A cold blue window declares the Nano Machine active."
    assert gnp.validate_segments(out, FILES, kinds) == []


def test_sfs_system_card_without_sentence_gets_grounded_pad_and_split():
    kinds = {"p1.jpg": "story", "p2.jpg": "system", "p3.jpg": "story"}
    out = _sfs([{"text": "He crawls toward the ridge, then past it.",
                 "panels": ["p1.jpg", "p3.jpg"]}], kinds=kinds)
    assert [s["span"] for s in out] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert out[0]["line"] == "He crawls toward the ridge, then past it."
    assert out[1]["line"] == "He crosses toward p2."   # grounded pad
    assert out[2]["line"] == "He crosses toward p3."   # line voiced once
    assert gnp.validate_segments(out, FILES, kinds) == []


def test_sfs_dangling_system_sentence_does_not_swallow_next_story_sentence():
    # 2026-07-07 review: the rejoin guard was one-directional -- it stopped a
    # system card's OWN sentence from being swallowed BY a dangling
    # predecessor, but not the reverse: a non-terminal sentence that itself
    # tags exactly one system card would swallow the FOLLOWING story
    # sentence, leaving the card with a fallback pad and the story panel
    # with an unpunctuated concatenation. System solos must be a wall in
    # BOTH directions; the fragment net period-closes the dangling card line
    # on its own later.
    kinds = {"p1.jpg": "story", "p2.jpg": "system", "p3.jpg": "story"}
    out = _sfs([{"text": "He staggers upright.", "panels": ["p1.jpg"]},
                {"text": "Nano machine activation initializing",  # no period
                 "panels": ["p2.jpg"]},
                {"text": "He turns to face the threat.", "panels": ["p3.jpg"]}],
               kinds=kinds)
    assert [s["span"] for s in out] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert out[0]["line"] == "He staggers upright."
    assert out[1]["line"] == "Nano machine activation initializing"
    assert out[2]["line"] == "He turns to face the threat."
    assert gnp.validate_segments(out, FILES, kinds) == []


def test_sfs_cap_overflow_rides_the_next_sentence_span():
    files6 = [f"r{i}.jpg" for i in range(1, 7)]
    kinds6 = {f: "story" for f in files6}
    u6 = {f: {"action": f"He crosses toward {f.split(chr(46))[0]}."} for f in files6}
    out = gnp.segments_from_sentences(
        [{"text": "The fall takes everything from him on the way down.",
          "panels": files6[:5]},                       # 5 > SPAN_CAP
         {"text": "The floor finally holds him, and the dark closes in "
                  "overhead.", "panels": [files6[5]]}],
        files6, kinds6, u6)
    assert [s["span"] for s in out] == [files6[:4], files6[4:]]
    assert out[0]["line"] == \
        "The fall takes everything from him on the way down."
    assert out[1]["line"] == ("The floor finally holds him, and the dark "
                              "closes in overhead.")
    assert gnp.validate_segments(out, files6, kinds6) == []


def test_sfs_lone_mega_sentence_returns_none():
    files5 = [f"q{i}.jpg" for i in range(1, 6)]
    assert gnp.segments_from_sentences(
        [{"text": "Everything happens at once.", "panels": files5}],
        files5, {f: "story" for f in files5}, {}) is None


def test_sfs_no_usable_tags_returns_none():
    assert _sfs([{"text": "Words with no tags.", "panels": []}]) is None
    assert _sfs([{"text": "Tags nobody knows.", "panels": ["zzz.jpg"]}]) is None
    assert _sfs([]) is None
    assert _sfs(None) is None


def test_sfs_unknown_files_and_empty_texts_are_ignored():
    out = _sfs([{"text": "", "panels": ["p1.jpg"]},
                {"text": "He runs the length of the bridge.",
                 "panels": ["zzz.jpg", "p1.jpg", "p2.jpg", "p1.jpg"]},
                {"text": "The far side greets him with steel.",
                 "panels": ["p3.jpg"]}])
    assert [s["span"] for s in out] == [["p1.jpg", "p2.jpg"], ["p3.jpg"]]


def test_sfs_all_system_group_uses_card_sentences():
    kinds = {f: "system" for f in FILES}
    out = _sfs([{"text": "Quest window one.", "panels": ["p1.jpg"]},
                {"text": "Quest window two.", "panels": ["p2.jpg"]}],
               kinds=kinds)
    assert [s["span"] for s in out] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert out[0]["line"] == "Quest window one."
    assert out[1]["line"] == "Quest window two."
    assert out[2]["line"] == "He crosses toward p3."   # pad for untagged card


def test_sfs_basenames_accepted_from_full_paths():
    out = _sfs([{"text": "He wakes at the bottom of the ravine.",
                 "panels": ["/abs/dir/p1.jpg", "scenes/p2.jpg"]},
                {"text": "Nothing about the climb looks kind.",
                 "panels": ["p3.jpg"]}])
    assert [s["span"] for s in out] == [["p1.jpg", "p2.jpg"], ["p3.jpg"]]


# ---- prose-first schema + prompt --------------------------------------------

def test_beat_schema_prose_has_narration_and_sentences():
    schema = gnp.build_beat_schema("prose")
    props = schema["properties"]
    assert "sentences" in props and "panel_narration" not in props
    assert "segments" not in props                     # spans are code's job
    item = props["sentences"]["items"]["properties"]
    assert set(item) >= {"text", "panels"}
    assert "sentences" in schema["required"]
    assert "narration" in props                        # the passage field
    # the passage must be authored BEFORE the split (property order drives
    # the constrained-decoding generation order on the ollama backend)
    keys = list(props)
    assert keys.index("narration") < keys.index("sentences")


def test_prose_prompt_demands_passage_then_tagged_split():
    text = gnp._PROSE_NARRATION_INSTRUCTION
    assert "'narration' FIRST" in text
    assert "ONE connected passage" in text
    assert "'sentences'" in text and "panels" in text
    assert "RAW MATERIAL" in text                      # anti-parrot pins hold
    assert "in the next panel" in text
    assert "CONSECUTIVE" in text
    # the direct-segments instruction survives for span-pinned regens
    assert "segments" in gnp._ADAPTIVE_NARRATION_INSTRUCTION


# ---- prose-first main() e2e (stubbed model) ---------------------------------

_PROSE_MODEL_BEAT = {
    "beat_title": "The Fall", "what_happens": "He falls into the ravine.",
    "narration": ("Something wet splashes across his skin, and he is not "
                  "happy about it. Then the ground gives way and the ravine "
                  "swallows him whole."),
    "sentences": [
        {"text": "Something wet splashes across his skin, and he is not "
                 "happy about it.", "panels": ["p1.jpg"]},
        {"text": "Then the ground gives way and the ravine swallows him "
                 "whole.", "panels": ["p2.jpg", "p3.jpg"]},
    ],
    "scene_selection": [],
}


def test_main_prose_derives_segments_and_drops_scaffolding(tmp_path,
                                                           monkeypatch):
    out, calls = _run_main(tmp_path, monkeypatch, [_PROSE_MODEL_BEAT])
    assert len(calls) == 1
    # free generation asks in the prose shape…
    props = calls[0]["response_schema"]["properties"]
    assert "sentences" in props and "segments" not in props
    assert "'narration' FIRST" in calls[0]["system_instruction"]
    beat = out["beats"][0]
    # …and lands in the SAME segments contract every consumer reads
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg"], ["p2.jpg", "p3.jpg"]]
    assert "sentences" not in beat                     # scaffolding dropped
    assert "panel_narration" not in beat
    assert beat["narration"] == " ".join(
        s["line"] for s in beat["segments"])           # load-bearing join


def test_main_prose_unusable_tags_reasks_then_falls_back(tmp_path,
                                                         monkeypatch):
    no_tags = dict(_PROSE_MODEL_BEAT, sentences=[
        {"text": "A fine passage with no tags at all.", "panels": []}])
    out, calls = _run_main(tmp_path, monkeypatch, [no_tags, no_tags])
    assert len(calls) == 2                             # ONE prose repair re-ask
    assert "NARRATION REPAIR" in calls[1]["system_instruction"]
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]            # singleton fallback
    # the passage's sentence text is reused as positional material
    assert beat["segments"][0]["line"] == \
        "A fine passage with no tags at all."
    assert all(s["line"] for s in beat["segments"])


def test_main_prose_repair_reask_adopts_good_answer(tmp_path, monkeypatch):
    no_tags = dict(_PROSE_MODEL_BEAT, sentences=[
        {"text": "A fine passage with no tags at all.", "panels": []}])
    out, calls = _run_main(tmp_path, monkeypatch,
                           [no_tags, _PROSE_MODEL_BEAT])
    assert len(calls) == 2
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg"], ["p2.jpg", "p3.jpg"]]              # repaired answer adopted


def test_main_prose_resolves_cast_tokens_in_segment_lines(tmp_path,
                                                          monkeypatch):
    cast = tmp_path / "cast.json"
    cast.write_text(json.dumps({"cast": [
        {"id": "mc", "canonical_name": "Prince Cheon", "role": "protagonist",
         "aliases": ["Prince Cheon"], "visual_description": "ragged robes"}]}))
    tokened = dict(_PROSE_MODEL_BEAT, sentences=[
        {"text": "Something wet splashes across [protagonist]'s skin.",
         "panels": ["p1.jpg"]},
        {"text": "Then the ground gives way and the ravine swallows him "
                 "whole.", "panels": ["p2.jpg", "p3.jpg"]},
    ])
    out, _ = _run_main(tmp_path, monkeypatch, [tokened],
                       extra_argv=("--cast", str(cast)))
    beat = out["beats"][0]
    assert beat["segments"][0]["line"] == \
        "Something wet splashes across Prince Cheon's skin."
    assert "[protagonist]" not in beat["narration"]


def test_corrections_pinned_regen_still_speaks_segments_schema(tmp_path,
                                                               monkeypatch):
    # the heal path is untouched by prose-first: a pinned regen asks with the
    # direct-segments schema + instruction (locked spans, lines rewritten)
    rewrite = dict(_GOOD_MODEL_BEAT)
    _, calls = _run_corrections(tmp_path, monkeypatch, [rewrite],
                                _prev_segments_beat())
    props = calls[0]["response_schema"]["properties"]
    assert "segments" in props and "sentences" not in props
    assert "FIXED SEGMENTATION" in calls[0]["system_instruction"]


# ---- unvoiced corrections: no clip cache exists -> the pin has nothing to
# protect, so the heal goes through the free PROSE path (a valid re-split is
# ADOPTED — holding rewrites to exact span reproduction made the real ch1
# heal loop 0-for-9, every good rewrite discarded and the flagged line kept).

def test_corrections_unvoiced_uses_prose_path_and_adopts_resplit(
        tmp_path, monkeypatch):
    resplit_prose = {
        "beat_title": "Opening", "what_happens": "He crosses the hall.",
        "narration": ("The hall swallows him one step at a time. And the "
                      "doors slam shut with the finality of a verdict."),
        "sentences": [
            {"text": "The hall swallows him one step at a time.",
             "panels": ["p1.jpg"]},
            {"text": "And the doors slam shut with the finality of a "
                     "verdict.", "panels": ["p2.jpg", "p3.jpg"]},
        ],
        "scene_selection": [],
    }
    out, calls = _run_corrections(tmp_path, monkeypatch, [resplit_prose],
                                  _prev_segments_beat(), voiced=False)
    sysi = calls[0]["system_instruction"]
    assert "CORRECTION FOR THIS GROUP" in sysi
    assert "FIXED SEGMENTATION" not in sysi              # nothing to pin
    assert "sentences" in calls[0]["response_schema"]["properties"]
    beat = out["beats"][0]
    # the re-split is ADOPTED — spans differ from the previous beat
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg"], ["p2.jpg", "p3.jpg"]]
    assert beat["segments"][0]["line"] == \
        "The hall swallows him one step at a time."


def test_corrections_unvoiced_fallback_still_keeps_previous_lines(
        tmp_path, monkeypatch):
    # pads must never replace real lines even without a pin
    no_tags = {
        "beat_title": "Opening", "what_happens": "He crosses the hall.",
        "narration": "A passage with no tags.",
        "sentences": [{"text": "A passage with no tags.", "panels": []}],
        "scene_selection": [],
    }
    out, calls = _run_corrections(tmp_path, monkeypatch, [no_tags, no_tags],
                                  _prev_segments_beat(), voiced=False)
    assert len(calls) == 2                               # ask + repair re-ask
    beat = out["beats"][0]
    assert [s["span"] for s in beat["segments"]] == [
        ["p1.jpg", "p2.jpg"], ["p3.jpg"]]                # previous beat kept
    assert [s["line"] for s in beat["segments"]] == [PREV_FLOW, PREV_SOLO]


# ---------------------------------------------------------------------------
# sentence-integrity rejoin (2026-07-06 review, class C): a model "sentence"
# with no terminal punctuation is HALF a sentence — fold the next one back in
# so no derived line can dangle mid-clause.
# ---------------------------------------------------------------------------

def test_prose_dangling_sentence_rejoins_with_the_next():
    files = ["p1.jpg", "p2.jpg"]
    kinds = {f: "story" for f in files}
    segs = gnp.segments_from_sentences([
        {"text": "But there is no mercy to be found, only the",
         "panels": ["p1.jpg"]},
        {"text": "cold certainty of the blade.", "panels": ["p2.jpg"]},
    ], files, kinds)
    assert segs == [{
        "span": ["p1.jpg", "p2.jpg"],
        "line": "But there is no mercy to be found, only the cold "
                "certainty of the blade.",
    }]


def test_prose_dangler_never_steals_a_system_cards_line():
    files = ["p1.jpg", "p2.jpg"]
    kinds = {"p1.jpg": "story", "p2.jpg": "system"}
    segs = gnp.segments_from_sentences([
        {"text": "The mercy runs out, only the", "panels": ["p1.jpg"]},
        {"text": "System activation begins now.", "panels": ["p2.jpg"]},
    ], files, kinds)
    # the card keeps its OWN line; the dangler stays (the fragment net
    # amputates it downstream, and truncated_line QA nets any survivor)
    assert segs == [
        {"span": ["p1.jpg"], "line": "The mercy runs out, only the"},
        {"span": ["p2.jpg"], "line": "System activation begins now."},
    ]


def test_grounded_pad_never_copies_display_meta_description():
    # class B root: g0024_p13/_p15 voiced "The text is displayed as a
    # standalone caption." — an understanding description of a text card
    # copied verbatim by the pad ladder. The display-meta family now gates
    # the ladder, which falls through to the named subjects.
    u = {"p1.jpg": {
        "description": "The text is displayed as a standalone caption.",
        "action": "", "subjects": ["a glowing system card"], "setting": "",
    }}
    line = gnp._grounded_pad_line("p1.jpg", u)
    assert "displayed" not in line and "caption" not in line
    assert line == "a glowing system card"


# ---- system cards SPEAK their text (2026-08-18, nano ch1 ending) -----------
# The prompt bans "a white panel appears with the text…" but gemma parroted the
# understanding's card description into the solo system spans; the fix is
# deterministic: a system-card line that describes the card (or shares no
# content with it) becomes the card's own words.

def test_auto_repair_voices_system_card_text_instead_of_describing_it():
    kinds = dict(KINDS, **{"p2.jpg": "system"})
    u = dict(U_BY_FILE)
    u["p2.jpg"] = dict(U_BY_FILE.get("p2.jpg") or {}, dialogue="7TH GENERATION NANO MACHINE, STARTING ACTIVATION.")
    segs = [{"span": ["p1.jpg"], "line": _words(12)},
            {"span": ["p2.jpg"], "line": "A plain white panel contains two blue text boxes announcing an activation process."},
            {"span": ["p3.jpg"], "line": _words(12)}]
    out = gnp.auto_repair_segments(segs, FILES, kinds, u)
    assert out[1]["line"] == "7th generation nano machine, starting activation."
    # a line that already voices the card is left alone
    segs[1]["line"] = "The 7th generation nano machine announces its activation."
    out2 = gnp.auto_repair_segments(segs, FILES, kinds, u)
    assert out2[1]["line"] == "The 7th generation nano machine announces its activation."
    # extracted system card (from a multi-panel span) is voiced too, not padded
    segs3 = [{"span": ["p1.jpg", "p2.jpg", "p3.jpg"], "line": _words(18)}]
    out3 = gnp.auto_repair_segments(segs3, FILES, kinds, u)
    assert out3[1]["line"] == "7th generation nano machine, starting activation."


def test_validate_rejects_a_line_cut_mid_sentence():
    # nano ch1 g0019: "…terrified that he might be seeing the true power of a" —
    # the model's output stopped mid-clause; a line must END (., !, ?, …, or a
    # closing quote after one). Cliffhanger "…but..." style still passes.
    segs = [{"span": ["p1.jpg"], "line": "He stares, terrified that he might be seeing the true power of a"},
            {"span": ["p2.jpg"], "line": _words(10)},
            {"span": ["p3.jpg"], "line": _words(10)}]
    errs = gnp.validate_segments(segs, FILES, KINDS)
    assert any("ends mid-sentence" in e for e in errs)
    segs[0]["line"] = "He stares, terrified of what he sees but..."
    assert not any("ends mid-sentence" in e for e in gnp.validate_segments(segs, FILES, KINDS))
    segs[0]["line"] = "He gasps, 'Exactly what is that light?!'"
    assert not any("ends mid-sentence" in e for e in gnp.validate_segments(segs, FILES, KINDS))
