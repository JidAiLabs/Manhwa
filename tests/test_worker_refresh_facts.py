"""tests/test_worker_refresh_facts.py

The refresh_facts job is the sp_v2 rollout path: one gemma re-read of a
chapter + a ledger rebuild, with the narration left alone. It runs on the GPU
lane (it IS a gemma call) and the chapter lease keeps it off that chapter's
own prepare.
"""
from __future__ import annotations

import io
import sqlite3

import studio.worker as w
from studio.dashboard import jobs


def _run(monkeypatch, payload):
    seen: list = []
    enqueued: list = []
    monkeypatch.setattr(w, "_chapter", lambda con, cid: {
        "id": cid, "series_id": 3, "number": 108.0, "ep_dir": "/nope"})
    monkeypatch.setattr(w, "_series_env", lambda con, sid: None)
    monkeypatch.setattr(w, "_stream",
                        lambda args, log, **kw: seen.extend(args) or 0)
    monkeypatch.setattr(jobs, "enqueue",
                        lambda con, t, **kw: enqueued.append((t, kw)) or 99)
    w._h_refresh_facts(sqlite3.connect(":memory:"),
                       {"chapter_id": 7, "payload": payload}, io.StringIO())
    return seen, enqueued


def test_the_job_refreshes_that_one_chapter_by_number(monkeypatch):
    seen, enqueued = _run(monkeypatch, {})
    assert seen[1:] == ["-m", "studio", "refresh-facts", "3",
                        "--chapters", "108.0"]
    assert "--force" not in seen
    assert enqueued == []               # the sweep does not re-narrate


def test_force_and_then_prepare_ride_in_the_payload(monkeypatch):
    seen, enqueued = _run(monkeypatch, {"force": True, "then_prepare": True})
    assert "--force" in seen
    assert enqueued and enqueued[0][0] == "prepare"
    assert enqueued[0][1]["chapter_id"] == 7
    assert enqueued[0][1]["priority"] == 2      # behind an operator's own work


def test_a_failed_refresh_fails_the_job(monkeypatch):
    monkeypatch.setattr(w, "_chapter", lambda con, cid: {
        "id": cid, "series_id": 3, "number": 108.0, "ep_dir": "/nope"})
    monkeypatch.setattr(w, "_series_env", lambda con, sid: None)
    monkeypatch.setattr(w, "_stream", lambda args, log, **kw: 1)
    try:
        w._h_refresh_facts(sqlite3.connect(":memory:"),
                           {"chapter_id": 7, "payload": {}}, io.StringIO())
        assert False, "should have raised"
    except RuntimeError as e:
        assert "exited 1" in str(e)


def test_it_shares_the_gemma_lane_and_has_a_handler():
    assert jobs.LANES["refresh_facts"] == "gpu"
    assert "refresh_facts" in w.HANDLERS
