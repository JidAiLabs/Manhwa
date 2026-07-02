"""
tests/test_prep_qa_spans.py

Chunk 3 (adaptive flow narration): prep_qa becomes span-aware.

The strict 1:1 panel↔line era is over — a beat's narration is `segments[]`
where one line may span 1-4 consecutive panels. The count-shaped invariants
become COVER checks: every shown story panel belongs to exactly ONE segment
span (`panel_uncovered` / `panel_double_covered`, both ERROR). Grounding,
caption and alignment checks are verified span-aware (they key on the plan
montage / the narration join — supersets of any single panel), and
shot_description_flags iterates `beat_segments` so flow-span lines are
still screened for camera prose.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "prep_qa",
    Path(__file__).resolve().parent.parent / "tools" / "prep_qa.py",
)
pq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pq)  # type: ignore[union-attr]


# ---- helpers ----------------------------------------------------------------

def _plan(items, dims=None):
    return {"timeline": items, "scenes_subdir": "scenes_clean",
            "total_duration_sec": sum(i.get("duration_sec", 0) for i in items),
            "scene_dims": dims or {}}


def _item(seg, files, dur=8.0, **kw):
    cuts = [{"file": f, "start": i * dur / max(1, len(files)),
             "dur": dur / max(1, len(files))} for i, f in enumerate(files)]
    d = {"segment_id": seg, "cuts": cuts, "duration_sec": dur,
         "tts_text": kw.pop("tts_text", "A quiet morning passes."),
         "tts_audio": kw.pop("tts_audio", f"/tts/{seg}.wav")}
    d.update(kw)
    return d


def _seg_beat(gid, segments, scene_files=None):
    """Native-segments beat: segments = [(span_list, line), ...]."""
    segs = [{"span": list(span), "line": line} for span, line in segments]
    return {
        "group_id": gid,
        "scene_files": scene_files or [f for span, _ in segments for f in span],
        "segments": segs,
        "narration": " ".join(line for _, line in segments),
    }


FLOW_LINE = ("He plummets down the ravine, every impact stacking, until the "
             "bottom finally catches him and the pain arrives.")
SOLO_LINE = "The stranger's eyes snap open in the dark."


# ---- span cover checks: panel_uncovered / panel_double_covered ---------------

def test_cover_clean_partition_passes():
    beats = {"beats": [_seg_beat(1, [(["p1.jpg", "p2.jpg"], FLOW_LINE),
                                     (["p3.jpg"], SOLO_LINE)])]}
    plan = _plan([_item("g0001_p00", ["p1.jpg", "p2.jpg"]),
                  _item("g0001_p01", ["p3.jpg"])])
    assert pq.span_cover_flags(plan, beats) == []


def test_cover_uncovered_shown_panel_errors():
    # extra.jpg is on screen but belongs to NO segment span — narration
    # doesn't carry it; the old 1:1 count assert caught this class.
    beats = {"beats": [_seg_beat(1, [(["p1.jpg", "p2.jpg"], FLOW_LINE)])]}
    plan = _plan([_item("g0001_p00", ["p1.jpg", "p2.jpg", "extra.jpg"])])
    fl = pq.span_cover_flags(plan, beats)
    assert [f["code"] for f in fl] == ["panel_uncovered"]
    assert fl[0]["severity"] == pq.ERROR
    assert fl[0]["scene"] == "extra.jpg"
    assert fl[0]["segment_id"] == "g0001_p00"      # where it is shown


def test_cover_double_covered_panel_errors():
    # two beats both claim p2.jpg in their spans — the panel would be paced
    # under two different clips; the beats manifest is inconsistent.
    beats = {"beats": [
        _seg_beat(1, [(["p1.jpg", "p2.jpg"], FLOW_LINE)]),
        _seg_beat(2, [(["p2.jpg", "p3.jpg"], FLOW_LINE)]),
    ]}
    plan = _plan([_item("g0001_p00", ["p1.jpg", "p2.jpg"]),
                  _item("g0002_p01", ["p2.jpg", "p3.jpg"])])
    fl = pq.span_cover_flags(plan, beats)
    assert [f["code"] for f in fl] == ["panel_double_covered"]
    assert fl[0]["severity"] == pq.ERROR
    assert fl[0]["scene"] == "p2.jpg"
    assert "g0001" in fl[0]["detail"] and "g0002" in fl[0]["detail"]


def test_cover_exempts_protected_injected_card():
    # inject_missing_protected shows a narration-less system/doc card by
    # design (it belongs to NO span) — never a panel_uncovered.
    beats = {"beats": [_seg_beat(1, [(["p1.jpg"], SOLO_LINE)])]}
    plan = _plan([_item("g0001_p00", ["p1.jpg", "sys.jpg"])],
                 dims={"sys.jpg": {"w": 100, "h": 100, "sys": True}})
    assert pq.span_cover_flags(plan, beats) == []


def test_cover_exempts_vision_stamped_system_card():
    # same exemption when only the understanding stamped panel_kind="system"
    # (raw planner plans carry no scene_dims sys marker)
    beats = {"beats": [_seg_beat(1, [(["p1.jpg"], SOLO_LINE)])]}
    plan = _plan([_item("g0001_p00", ["p1.jpg", "sys.jpg"])])
    vitems = {"sys.jpg": {"panel_kind": "system"}}
    assert pq.span_cover_flags(plan, beats, vitems) == []


def test_cover_exempts_held_and_branding_cuts():
    beats = {"beats": [_seg_beat(1, [(["p1.jpg"], SOLO_LINE)])]}
    held_item = _item("g0001_p00", ["p1.jpg"])
    held_item["cuts"].append({"file": "prev.jpg", "dur": 3.0, "held": True})
    branding = {"segment_id": "branding_intro", "branding": "intro",
                "duration_sec": 7.0,
                "cuts": [{"file": "logo.jpg", "start": 0.0, "dur": 7.0}]}
    plan = _plan([branding, held_item])
    assert pq.span_cover_flags(plan, beats) == []


def test_cover_split_halves_map_to_parent_panel():
    # render_prep split2 shows p000031_a/_b; the span covers p000031.jpg
    beats = {"beats": [_seg_beat(1, [(["p000031.jpg"], SOLO_LINE)])]}
    item = _item("g0001_p00", ["p000031_a.jpg"])
    item["cuts"][0]["file2"] = "p000031_b.jpg"
    plan = _plan([item])
    assert pq.span_cover_flags(plan, beats) == []


def test_cover_skips_manifests_without_any_segments():
    # a pre-per-panel manifest (beats carry only `narration`) has no span
    # info to assert cover against — the check must stay silent, not flood
    beats = {"beats": [{"group_id": 1, "narration": "Old-style beat.",
                        "scene_files": ["p1.jpg", "p2.jpg"]}]}
    plan = _plan([_item("g0001_p00", ["p1.jpg", "p2.jpg"])])
    assert pq.span_cover_flags(plan, beats) == []


def test_cover_legacy_panel_narration_adapts_to_singletons():
    beats = {"beats": [{"group_id": 1, "scene_files": ["p1.jpg", "p2.jpg"],
                        "panel_narration": [
                            {"scene_file": "p1.jpg", "line": SOLO_LINE},
                            {"scene_file": "p2.jpg", "line": SOLO_LINE}],
                        "narration": SOLO_LINE + " " + SOLO_LINE}]}
    good = _plan([_item("g0001_p00", ["p1.jpg"]),
                  _item("g0001_p01", ["p2.jpg"])])
    assert pq.span_cover_flags(good, beats) == []
    bad = _plan([_item("g0001_p00", ["p1.jpg", "stray.jpg"])])
    assert [f["code"] for f in pq.span_cover_flags(bad, beats)] == [
        "panel_uncovered"]


def test_cover_dropped_span_panel_is_not_flagged():
    # spec 3.5: a visual drop shrinks the span's CUT list, narration untouched
    # — the dropped panel is simply not shown; cover only judges SHOWN panels.
    beats = {"beats": [_seg_beat(
        1, [(["p1.jpg", "p2.jpg", "p3.jpg"], FLOW_LINE)])]}
    plan = _plan([_item("g0001_p00", ["p1.jpg", "p3.jpg"])])   # p2 dropped
    assert pq.span_cover_flags(plan, beats) == []


# ---- alignment_flags: the narration join keeps the stale gate working --------

def test_alignment_clean_on_segments_beat_with_span_items():
    # per-segment plan items re-join per group before comparing to the beat
    # narration (the ordered join of segment lines) — no false narration_stale
    beats = {"beats": [_seg_beat(1, [(["p1.jpg", "p2.jpg"], FLOW_LINE),
                                     (["p3.jpg"], SOLO_LINE)])]}
    groups = {"shots": [{"group_id": 1}]}
    script = {"narration_source": "gemini_verbatim"}
    plan = _plan([_item("g0001_p00", ["p1.jpg", "p2.jpg"], tts_text=FLOW_LINE),
                  _item("g0001_p01", ["p3.jpg"], tts_text=SOLO_LINE)])
    assert pq.alignment_flags(plan, beats, groups, script) == []


def test_alignment_still_catches_stale_segments_beat():
    beats = {"beats": [_seg_beat(1, [(["p1.jpg", "p2.jpg"], FLOW_LINE)])]}
    groups = {"shots": [{"group_id": 1}]}
    script = {"narration_source": "gemini_verbatim"}
    plan = _plan([_item("g0001_p00", ["p1.jpg", "p2.jpg"],
                        tts_text="A completely different chapter about "
                                 "dragons at sea and their ancient war.")])
    assert "narration_stale" in [
        f["code"] for f in pq.alignment_flags(plan, beats, groups, script)]


# ---- grounding: a span segment is judged against ALL its span panels --------

def test_grounding_judges_span_item_against_all_cuts(monkeypatch, tmp_path):
    import sys
    import types
    import json as _json
    calls = []

    def chat(**kw):
        calls.append([Path(p).name for p in kw["messages"][0]["images"]])
        return {"message": {"content": _json.dumps({"ok": True, "issue": ""})}}

    fake = types.ModuleType("ollama")
    fake.chat = chat
    monkeypatch.setitem(sys.modules, "ollama", fake)
    monkeypatch.setenv("STUDIO_QA_CONC", "1")
    for f in ("p1.jpg", "p2.jpg"):
        (tmp_path / f).write_bytes(b"jpg")
    plan = {"timeline": [_item("g0001_p00", ["p1.jpg", "p2.jpg"],
                               tts_text=FLOW_LINE)]}
    assert pq.grounding_flags(plan, str(tmp_path)) == []
    assert calls == [["p1.jpg", "p2.jpg"]]     # ONE judge call, BOTH panels


def test_grounding_cache_key_includes_the_cut_set(monkeypatch, tmp_path):
    # same narration over a DIFFERENT cut set must re-judge (cache key =
    # model + text + shown files), so span growth/shrink is never stale-hit
    import sys
    import types
    import json as _json
    calls = {"n": 0}

    def chat(**kw):
        calls["n"] += 1
        return {"message": {"content": _json.dumps({"ok": True, "issue": ""})}}

    fake = types.ModuleType("ollama")
    fake.chat = chat
    monkeypatch.setitem(sys.modules, "ollama", fake)
    monkeypatch.setenv("STUDIO_QA_CONC", "1")
    for f in ("p1.jpg", "p2.jpg"):
        (tmp_path / f).write_bytes(b"jpg")
    cache = str(tmp_path / ".gcache.json")

    span_plan = {"timeline": [_item("g0001_p00", ["p1.jpg", "p2.jpg"],
                                    tts_text=FLOW_LINE)]}
    pq.grounding_flags(span_plan, str(tmp_path), cache_path=cache)
    assert calls["n"] == 1
    pq.grounding_flags(span_plan, str(tmp_path), cache_path=cache)
    assert calls["n"] == 1                     # unchanged span: cache hit

    solo_plan = {"timeline": [_item("g0001_p00", ["p1.jpg"],
                                    tts_text=FLOW_LINE)]}
    pq.grounding_flags(solo_plan, str(tmp_path), cache_path=cache)
    assert calls["n"] == 2                     # cut set changed: re-judged


# ---- caption_unvoiced: the join covers captions on ANY span panel -----------

def _caption_vitems(caption_file, text):
    return {caption_file: {"text_only": True, "ocr_clean": text}}


def test_caption_on_second_span_panel_voiced_in_flow_line_passes():
    cap = "BACK THEN I HAD NO IDEA WHAT WAS COMING"
    line = ("Back then he had no idea what was coming, and the ravine kept "
            "swallowing him whole.")
    beats = {"beats": [_seg_beat(1, [(["p1.jpg", "cap.jpg"], line)])]}
    assert pq.caption_unvoiced_flags(beats, _caption_vitems("cap.jpg", cap)) == []


def test_caption_on_second_span_panel_unvoiced_flags():
    cap = "BACK THEN I HAD NO IDEA WHAT WAS COMING"
    beats = {"beats": [_seg_beat(1, [(["p1.jpg", "cap.jpg"], FLOW_LINE)])]}
    fl = pq.caption_unvoiced_flags(beats, _caption_vitems("cap.jpg", cap))
    assert [f["code"] for f in fl] == ["caption_unvoiced"]
    assert fl[0]["severity"] == pq.ERROR
    assert fl[0]["scene"] == "cap.jpg"


# ---- shot_description_flags: iterates beat_segments (flow spans included) ----

def test_shot_description_flags_fire_on_flow_span_line():
    beats = {"beats": [_seg_beat(5, [
        (["p1.jpg", "p2.jpg"],
         "The panel focuses on the ruined courtyard as he falls."),
        (["p3.jpg"], "He clenches his fists."),
    ])]}
    fl = pq.shot_description_flags(beats)
    assert [f["code"] for f in fl] == ["shot_description"]
    assert fl[0]["severity"] == pq.ERROR
    assert fl[0]["segment_id"] == "g0005"          # healable per group
    assert fl[0]["scene"] == "p1.jpg"              # span head carries the thumb


def test_shot_description_flags_clean_on_segments_beat():
    beats = {"beats": [_seg_beat(1, [(["p1.jpg", "p2.jpg"], FLOW_LINE),
                                     (["p3.jpg"], SOLO_LINE)])]}
    assert pq.shot_description_flags(beats) == []
