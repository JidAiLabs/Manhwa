"""tests/test_death_audit.py — the measurement that made the case for sp_v2.

Before: 36 killed fates, 7 anchored. The audit is how we know the sweep
worked, so it has to count the same way the ledger does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import death_audit as da  # noqa: E402

_ENTS = [{"id": "beast_lord", "canonical_name": "Beast Lord",
          "role": "antagonist", "aliases": ["the beast"],
          "visual_description": "a woman with long dark hair"},
         {"id": "captain", "canonical_name": "the Captain", "role": "ally",
          "aliases": [], "visual_description": "a man in a heavy coat"}]


def _chapter(tmp_path: Path, name: str, cast, events, version="sp_v2") -> Path:
    ep = tmp_path / "ongoing" / "orv" / name
    ep.mkdir(parents=True)
    (ep / "manifest.chapter_story.json").write_text(json.dumps(
        {"cast": cast, "events": [], "_meta": {"prompt_version": version}}))
    (ep / "manifest.ledger.json").write_text(json.dumps(
        {"entities": _ENTS, "events": events}))
    return ep


def test_a_dies_at_anchor_counts_as_anchored_and_names_its_source(tmp_path):
    ep = _chapter(
        tmp_path, "Episode_107",
        [{"name": "Beast Lord", "role": "antagonist", "fate": "eradicated",
          "dies_at": "p000024.jpg", "after_death": "absent"}],
        [{"type": "death", "subject": "Beast Lord",
          "scene_file": "p000024.jpg", "anchor_source": "dies_at"}])
    rec = da.audit_chapter(str(ep))
    assert rec["number"] == 107.0 and rec["prompt_version"] == "sp_v2"
    assert len(rec["killed"]) == 1                  # counted despite the verb
    assert rec["killed"][0]["anchor_source"] == "dies_at"
    assert rec["killed"][0]["anchored_at"] == "p000024.jpg"


def test_an_unanchored_death_is_the_thing_being_measured(tmp_path):
    ep = _chapter(
        tmp_path, "Episode_49",
        [{"name": "Beast Lord", "role": "antagonist",
          "fate": "killed by the Captain", "dies_at": "", "after_death": ""}],
        [], version="sp_v1")
    rec = da.audit_chapter(str(ep))
    assert rec["killed"][0]["anchored_at"] == ""
    assert "NOT anchored" in da._fmt(rec)


def test_a_name_the_ledger_cannot_resolve_is_reported_separately(tmp_path):
    ep = _chapter(
        tmp_path, "Episode_50",
        [{"name": "Maruyama and Amano", "role": "x", "fate": "killed",
          "dies_at": "", "after_death": ""}], [])
    rec = da.audit_chapter(str(ep))
    assert rec["killed"][0]["resolved"] == "unknown"
    assert "matches no entity" in da._fmt(rec)


def test_a_chapter_without_a_story_or_ledger_is_skipped(tmp_path):
    ep = tmp_path / "ongoing" / "orv" / "Episode_1"
    ep.mkdir(parents=True)
    assert da.audit_chapter(str(ep)) == {}


def test_contradictions_read_the_shipped_narration_with_no_model_call(tmp_path):
    ep = _chapter(
        tmp_path, "Episode_108",
        [{"name": "Beast Lord", "role": "antagonist", "fate": "killed",
          "dies_at": "p000024.jpg", "after_death": "absent"}],
        [{"type": "death", "subject": "Beast Lord",
          "scene_file": "p000024.jpg", "anchor_source": "dies_at"}])
    led = json.loads((ep / "manifest.ledger.json").read_text())
    led["beat_facts"] = {"g0013": {"dead_by_now": ["Beast Lord"]}}
    (ep / "manifest.ledger.json").write_text(json.dumps(led))
    (ep / "manifest.cast.json").write_text(json.dumps({"cast": _ENTS}))
    (ep / "manifest.beats.json").write_text(json.dumps({"beats": [
        {"group_id": 13, "narration": "The beast lunges at the Captain.",
         "segments": [{"span": ["p000043.jpg"],
                       "line": "The beast lunges at the Captain."}]}]}))
    rec = da.audit_chapter(str(ep), contradictions=True)
    assert rec["contradictions"] == ["dead_actor"]
