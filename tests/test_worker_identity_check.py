"""identity_check: the per-series validation of the picture + gemma identity
runs as a worker job on the GPU lane (it IS a stream of gemma calls), never
beside the worker — 2026-10-10 an out-of-band gemma loop coincided with a
Metal GPU hang that burned a prepare's retries."""
from __future__ import annotations

import io
import sqlite3

import pytest

import studio.worker as w
from studio.dashboard import jobs


def test_identity_check_is_a_gpu_lane_job_with_a_handler():
    assert jobs.LANES["identity_check"] == "gpu"
    assert w.HANDLERS["identity_check"] is w._h_identity_check


def test_the_job_runs_the_check_for_its_series(monkeypatch):
    seen = []
    monkeypatch.setattr(w, "_stream", lambda args, log, **kw: seen.extend(args) or 0)
    w._h_identity_check(sqlite3.connect(":memory:"),
                        {"payload": {"series_slug": "dragon-devouring-mage", "n": 30}},
                        io.StringIO())
    i = seen.index("--check-series")
    assert seen[i + 1].endswith("ongoing/dragon-devouring-mage")
    assert seen[seen.index("--verify-ccip") + 1].endswith("cast/dragon-devouring-mage.exemplars.json")
    assert seen[seen.index("--n") + 1] == "30"
    assert seen[1].endswith("tools/panel_identity.py")


def test_a_failed_check_fails_the_job(monkeypatch):
    monkeypatch.setattr(w, "_stream", lambda args, log, **kw: 2)
    with pytest.raises(RuntimeError):
        w._h_identity_check(sqlite3.connect(":memory:"),
                            {"payload": {"series_slug": "x"}}, io.StringIO())


def test_the_check_uses_exemplars_confirmed_on_the_series_page(monkeypatch, tmp_path):
    ex = tmp_path / "ongoing" / "clan" / ".identity" / "exemplars.json"
    ex.parent.mkdir(parents=True)
    ex.write_text("{}")
    monkeypatch.setattr(w, "REPO", tmp_path)
    seen = []
    monkeypatch.setattr(w, "_stream", lambda args, log, **kw: seen.extend(args) or 0)
    w._h_identity_check(sqlite3.connect(":memory:"), {"payload": {"series_slug": "clan"}},
                        io.StringIO())
    assert seen[seen.index("--verify-ccip") + 1] == str(ex)


def test_propose_writes_the_proposal_then_checks_gemmas_choice(monkeypatch, tmp_path):
    assert jobs.LANES["identity_propose"] == "gpu"
    assert w.HANDLERS["identity_propose"] is w._h_identity_propose
    monkeypatch.setattr(w, "REPO", tmp_path)
    calls = []

    def stream(args, log, **kw):
        calls.append(args)
        if "--auto" in args:
            import json
            from pathlib import Path
            p = Path(args[args.index("--auto") + 1])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(
                {"protagonist": ["a", "b"], "decoy": ["c", "d"], "candidates": []}))
        return 0
    monkeypatch.setattr(w, "_stream", stream)
    w._h_identity_propose(sqlite3.connect(":memory:"), {"payload": {"series_slug": "clan"}},
                          io.StringIO())
    prop = tmp_path / "ongoing" / "clan" / ".identity" / "exemplars.proposed.json"
    assert calls[0][calls[0].index("--auto") + 1] == str(prop)
    chk = calls[1]
    assert chk[chk.index("--verify-ccip") + 1] == str(prop)
    assert chk[chk.index("--out-dir") + 1] == str(tmp_path / "dist" / "identity_check" / "proposed")
    assert not (tmp_path / "cast").exists()                # nothing switched on


def test_propose_without_a_choice_runs_no_check(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "REPO", tmp_path)
    calls = []

    def stream(args, log, **kw):
        calls.append(args)
        if "--auto" in args:
            from pathlib import Path
            p = Path(args[args.index("--auto") + 1])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{"candidates": [{}, {}]}')
        return 0
    monkeypatch.setattr(w, "_stream", stream)
    w._h_identity_propose(sqlite3.connect(":memory:"), {"payload": {"series_slug": "clan"}},
                          io.StringIO())
    assert len(calls) == 1
