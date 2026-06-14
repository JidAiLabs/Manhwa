"""
tests/test_narration_consistency.py — deterministic audio↔narration drift gate.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "narration_consistency",
    Path(__file__).resolve().parent.parent / "tools" / "narration_consistency.py",
)
nc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nc)  # type: ignore[union-attr]


# ---- normalization + fingerprint ----------------------------------------

def test_normalize_strips_leading_tags_ws_and_case():
    assert nc.normalize_narration("[excited]  He   RUNS.") == "he runs."
    assert nc.normalize_narration("[mad] [fast] Go!") == "go!"
    assert nc.normalize_narration("He runs.") == "he runs."


def test_sha_stable_under_tag_ws_case_but_differs_on_edit():
    a = nc.narration_sha("[excited] He runs for his life.")
    b = nc.narration_sha("He runs   for his life.")      # tag + spacing only
    c = nc.narration_sha("HE RUNS FOR HIS LIFE.")        # case only
    d = nc.narration_sha("He sprints for his life.")     # real word change
    assert a == b == c          # spoken content identical
    assert a != d               # genuine edit -> different fingerprint


# ---- consistency over a plan + index ------------------------------------

def _plan(*segs):
    return {"timeline": [{"segment_id": s, "tts_text": t} for s, t in segs]}


def _index(*clips):
    return {"clips": [dict(segment_id=s, **kw) for s, kw in clips]}


def test_fresh_when_sha_matches():
    plan = _plan(("g0001_p00", "[tense] He runs."))
    idx = _index(("g0001_p00", {"text_sha": nc.narration_sha("He runs.")}))
    r = nc.audio_consistency(plan, idx)
    assert r == {"fresh": ["g0001_p00"], "stale": [], "missing": []}
    assert nc.is_voiced_current(plan, idx) is True


def test_stale_when_text_changed():
    plan = _plan(("g0001_p00", "He sprints away."))
    idx = _index(("g0001_p00", {"text_sha": nc.narration_sha("He runs.")}))
    r = nc.audio_consistency(plan, idx)
    assert r["stale"] == ["g0001_p00"] and not r["fresh"]
    assert nc.is_voiced_current(plan, idx) is False


def test_missing_when_no_clip():
    plan = _plan(("g0009_p02", "New beat with no audio."))
    r = nc.audio_consistency(plan, _index())
    assert r["missing"] == ["g0009_p02"]


def test_clip_without_text_sha_is_stale_even_with_stored_text():
    # only text_sha proves freshness; a stored source/sent_text is NOT trusted
    # (a rewrite-without-resynth producer would otherwise look fresh). Forces a
    # one-time re-voice that backfills text_sha — self-healing migration.
    plan = _plan(("g0001_p00", "He runs."))
    assert nc.audio_consistency(
        plan, _index(("g0001_p00", {"sent_text": "He runs."})))["stale"] == ["g0001_p00"]
    assert nc.audio_consistency(
        plan, _index(("g0001_p00", {"duration_sec": 3.0})))["stale"] == ["g0001_p00"]


def test_inline_tags_are_ignored_in_fingerprint():
    # a mid-sentence delivery tag must not look like a content change
    assert nc.narration_sha("He fights [beat] harder.") == nc.narration_sha("He fights harder.")


def test_branding_and_silent_segments_ignored():
    plan = {"timeline": [
        {"branding": True, "segment_id": "intro"},
        {"segment_id": "g0001_p00", "tts_text": ""},          # held/silent
        {"segment_id": "g0002_p01", "tts_text": "Real line."},
    ]}
    idx = _index(("g0002_p01", {"text_sha": nc.narration_sha("Real line.")}))
    r = nc.audio_consistency(plan, idx)
    assert r["fresh"] == ["g0002_p01"] and not r["stale"] and not r["missing"]
