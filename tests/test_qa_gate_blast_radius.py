"""tools/qa_gate_blast_radius.py — the corpus measurement a gate must pass
before it may block, park or heal.

Pins the two lessons the rewrite encodes: detectors are bound by PARAMETER
NAME (a multi-arg gate like actor_mismatch is measured, a detector whose
inputs are not on disk is LISTED, never skipped silently), and counts are
broken out per title so a single-title corpus cannot pass for a fleet.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tests.test_cast_identity import CAST, _UNDERSTOOD, _beats

_SPEC = importlib.util.spec_from_file_location(
    "qa_gate_blast_radius",
    Path(__file__).resolve().parents[1] / "tools" / "qa_gate_blast_radius.py")
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)

_VISION = {"items": [
    {"scene_file": "p000010.jpg", "ocr_clean": "", "subjects": [], "vision": {}},
    {"scene_file": "p000011.jpg", "ocr_clean": "", "subjects": [], "vision": {}},
]}

ASSASSIN_OVER_PRINCE = _beats(("The assassin draws his steel.", ["p000010.jpg"]))
CORRECT = _beats(("The prince rips his hidden knife free.", ["p000010.jpg"]))


def _chapter(root: Path, title: str, ch: str, *, beats, report=None) -> Path:
    d = root / "ongoing" / title / ch
    d.mkdir(parents=True)
    (d / "manifest.beats.json").write_text(json.dumps(beats))
    (d / "manifest.panels.understood.json").write_text(json.dumps(_UNDERSTOOD))
    (d / "manifest.cast.json").write_text(json.dumps(CAST))
    (d / "manifest.vision.json").write_text(json.dumps(_VISION))
    if report is not None:
        (d / "prep_qa.json").write_text(json.dumps(report))
    return d


def _run(capsys, *argv) -> str:
    rc = tool.main(list(argv))
    assert rc == 0
    return capsys.readouterr().out


def test_multi_arg_detector_is_measured_with_a_titles_column(tmp_path, capsys):
    # actor_mismatch takes (beats, understood, cast[, vitems]) — the old tool
    # called det(beats) and swallowed the TypeError, so this row never existed.
    _chapter(tmp_path, "nano-machine", "Chapter_1", beats=ASSASSIN_OVER_PRINCE)
    _chapter(tmp_path, "omniscient-reader", "Episode_1", beats=CORRECT)
    out = _run(capsys, "--root", str(tmp_path), "--code", "actor_mismatch")
    row = next(l for l in out.splitlines() if l.startswith("actor_mismatch"))
    assert "WARN" in row
    assert "1/2" in row                      # one of the two titles carries it
    assert "· nano-machine: 1 chapter(s)" in out
    assert "2 chapters across 2 title(s)" in out


def test_detector_needing_missing_input_is_listed_not_skipped_silently(
        tmp_path, capsys):
    _chapter(tmp_path, "nano-machine", "Chapter_1", beats=CORRECT)
    out = _run(capsys, "--root", str(tmp_path))
    assert "SKIPPED" in out
    assert "audio_flags: plan, tts_index" in out          # no plan, no tts on disk
    assert "image_flags:" in out                          # per-image, never a chapter gate


def test_from_reports_tallies_what_is_on_disk_at_any_severity(tmp_path, capsys):
    _chapter(tmp_path, "nano-machine", "Chapter_1", beats=CORRECT, report={
        "flags": [
            {"code": "visible_text", "severity": "WARN", "scene": "p000010.jpg",
             "segment_id": "g0001", "detail": "ink"},
            {"code": "visible_text", "severity": "WARN", "scene": "p000011.jpg",
             "segment_id": "g0001", "detail": "ink"},
            {"code": "cut_gap", "severity": "ERROR", "scene": "",
             "segment_id": "g0002", "detail": "gap"}]})
    _chapter(tmp_path, "omniscient-reader", "Episode_1", beats=CORRECT,
             report={"flags": []})
    out = _run(capsys, "--root", str(tmp_path), "--from-reports")
    rows = {l.split()[0]: l for l in out.splitlines()
            if l.startswith(("visible_text", "cut_gap"))}
    assert "WARN" in rows["visible_text"] and "     2      1" in rows["visible_text"]
    assert "ERROR" in rows["cut_gap"] and "YES" in rows["cut_gap"]   # blocking column
    assert "SKIPPED" not in out                            # nothing ran, nothing skipped


def test_sample_dump_gives_the_grader_the_line_and_the_panels(tmp_path, capsys):
    _chapter(tmp_path, "nano-machine", "Chapter_1", beats=ASSASSIN_OVER_PRINCE)
    out_path = tmp_path / "sample.txt"
    _run(capsys, "--root", str(tmp_path), "--code", "actor_mismatch",
         "--sample", "1", "--sample-out", str(out_path))
    text = out_path.read_text()
    assert "The assassin draws his steel." in text          # the FULL line
    assert "bleeding from the mouth" in text                # the panel's subject
    assert "resolved=[('our protagonist'" in text           # what the oracle saw
    assert "cast 'assassin':" in text                       # the entry behind the noun
    assert "verdict: " in text


def test_footer_states_the_bar_and_warns_on_a_single_title(tmp_path, capsys):
    _chapter(tmp_path, "nano-machine", "Chapter_1", beats=CORRECT,
             report={"flags": []})
    out = _run(capsys, "--root", str(tmp_path), "--from-reports")
    assert "precision >= 0.80 over >= 20 graded flags" in out
    assert "single-title corpus" in out
