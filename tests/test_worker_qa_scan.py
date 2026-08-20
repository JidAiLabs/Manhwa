"""tests/test_worker_qa_scan.py

The dashboard's standalone QA re-scan must run the SAME gate set as the
prepare it is re-checking. Dropping --semantic did not merely hide grounding
WARNs: without the judge, caption_unvoiced loses its arbiter, so a caption the
writer carries by PARAPHRASE (a WARN on the prepare) came back a hard ERROR on
the re-scan (ORV Ep1 g0009, 2026-08-20). A cheap scan that disagrees with the
authoritative one is worse than no scan.
"""
from __future__ import annotations

import io
import sqlite3
import types

import studio.worker as w


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE stage_run (id INTEGER PRIMARY KEY, chapter_id "
                "INTEGER, stage TEXT, duration_sec REAL, ok INTEGER, "
                "meta_json TEXT)")
    return con


def _scan_args(monkeypatch, *, semantic_heal: bool) -> list:
    seen: list = []
    monkeypatch.setattr(w, "_chapter", lambda con, cid: {
        "id": cid, "series_id": 1, "ep_dir": "/nope"})
    monkeypatch.setattr(w, "_series_title", lambda con, sid: "Series")
    monkeypatch.setattr(w, "_stream", lambda args, log, **kw: seen.extend(args) or 0)
    monkeypatch.setattr(w, "_qa_verdict", lambda ep, started_at=0.0: w.QAVerdict(
        ok=True, blocking=set(), codes=set(), report={"flags": []}, reason=""))
    monkeypatch.setattr(w, "_stamp_plan_sha", lambda ep, verdict: None)
    monkeypatch.setattr("studio.config.load", lambda: types.SimpleNamespace(
        max_same_image_hold_sec=10.0, semantic_heal=semantic_heal))
    w._h_qa_scan(_con(), {"chapter_id": 7}, io.StringIO())
    return seen


def test_qa_scan_runs_the_semantic_gate(monkeypatch):
    args = _scan_args(monkeypatch, semantic_heal=False)
    assert "--semantic" in args          # the caption arbiter lives behind this
    assert "--semantic-heal" not in args


def test_qa_scan_mirrors_semantic_heal_config(monkeypatch):
    assert "--semantic-heal" in _scan_args(monkeypatch, semantic_heal=True)
