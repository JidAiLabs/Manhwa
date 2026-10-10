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
