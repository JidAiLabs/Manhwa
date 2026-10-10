"""A real pause (2026-10-10): freezing the worker with SIGSTOP tripped its
wall-clock watchdogs and failed a prepare. The file <repo>/.worker_pause stops
lanes from CLAIMING new jobs; running jobs finish normally. Its content lists
the lanes to pause ("gpu,tts"); empty = every lane."""
import sqlite3

import studio.worker as w


def _claims(monkeypatch, tmp_path, flag, lane):
    monkeypatch.setattr(w, "REPO", tmp_path)
    if flag is not None:
        (tmp_path / ".worker_pause").write_text(flag)
    seen = []
    monkeypatch.setattr(w.jobs, "claim_next", lambda con, lane=None: seen.append(lane) or None)
    w.run_once(sqlite3.connect(":memory:"), lane=lane)
    return seen


def test_no_flag_claims_as_usual(monkeypatch, tmp_path):
    assert _claims(monkeypatch, tmp_path, None, "gpu") == ["gpu"]


def test_an_empty_flag_pauses_every_lane(monkeypatch, tmp_path):
    for lane in ("gpu", "tts", "cpu", "api"):
        assert _claims(monkeypatch, tmp_path, "", lane) == []


def test_a_lane_list_pauses_only_those_lanes(monkeypatch, tmp_path):
    assert _claims(monkeypatch, tmp_path, "gpu, tts\n", "gpu") == []
    assert _claims(monkeypatch, tmp_path, "gpu, tts\n", "tts") == []
    assert _claims(monkeypatch, tmp_path, "gpu, tts\n", "cpu") == ["cpu"]
