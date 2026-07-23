"""tests/test_gpu_doctor.py — the classify verdict that prevents the misdiagnosis
(a live launchd server read as an orphan to kill)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gpu_doctor", Path(__file__).resolve().parent.parent / "tools" / "gpu_doctor.py")
gd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gd)  # type: ignore[union-attr]


def test_classify_verdicts():
    launchd = {100, 101}      # originpower.* launchd-owned pids
    live = {500}              # pgids of running jobs

    # a launchd job itself, and its child worker -> MANAGED (the shim case)
    assert gd.classify({"pid": 100, "ppid": 1, "pgid": 100}, launchd, live) == "MANAGED"
    assert gd.classify({"pid": 102, "ppid": 100, "pgid": 100}, launchd, live) == "MANAGED"
    # a subprocess of a RUNNING job (pgid matches) -> ACTIVE
    assert gd.classify({"pid": 501, "ppid": 1, "pgid": 500}, launchd, live) == "ACTIVE"
    # a repo model process with no launchd owner and no live job -> ORPHAN
    assert gd.classify({"pid": 9, "ppid": 1, "pgid": 9}, launchd, live) == "ORPHAN"
    # PPID 1 alone is NOT orphan evidence — launchd IS pid 1 (the exact trap)
    assert gd.classify({"pid": 100, "ppid": 1, "pgid": 100}, launchd, live) != "ORPHAN"


def test_launchd_pids_includes_homebrew_ollama():
    """The parse MUST pick up ANY launchd label — esp. homebrew's ollama — or a
    live model server gets mislabeled ORPHAN (the bug this whole tool prevents).
    Non-running rows ('-') are skipped."""
    sample = "\n".join([
        "PID\tStatus\tLabel",
        "2711\t0\thomebrew.mxcl.ollama",
        "26666\t0\tapplication.com.originpower.worker",
        "-\t0\tcom.apple.some.idle.service",
    ])
    pids = gd._parse_launchd_pids(sample)
    assert 2711 in pids and 26666 in pids       # ollama + our worker = MANAGED
    assert all(isinstance(p, int) for p in pids)
    # the "-" (not running) row contributes nothing
    assert len(pids) == 2
