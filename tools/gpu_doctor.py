#!/usr/bin/env python3
"""tools/gpu_doctor.py — one glance at every model/GPU process this project owns
and its verdict, so nobody has to guess (or misdiagnose) again:

  MANAGED  a launchd-owned service (or its child) — leave it, it's supervised.
  ACTIVE   a subprocess of a RUNNING job (pgid matches the jobs table) — leave it.
  ORPHAN   a repo model process with no supervisor and no live job — safe to reap.

This exists because a LIVE launchd shim (2x16GB, actively serving) was once read
as "orphaned, kill it" off a bare `PPID 1` — which on macOS just means launchd.
Run on the host holding the processes:

    python tools/gpu_doctor.py            # human table + reap suggestions
    python tools/gpu_doctor.py --json     # machine-readable
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# This project's model/GPU engines + GPU-bound pipeline stages.
_SIG = re.compile(
    r"ollama|mlx_ollama_shim|mlx_vlm|mlx_audio|panel_understand|"
    r"gemini_narrative_pass|story_group|local_tts|qwen|prep_qa|blender", re.I)


def _run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=25).stdout
    except Exception:
        return ""


def _parse_launchd_pids(text: str) -> set:
    """Every PID launchd currently supervises, from `launchctl list` output —
    ANY label, not just originpower.*. That MUST include homebrew's ollama
    (`homebrew.mxcl.ollama`): the whole point is "is launchd supervising it",
    and a bare PPID 1 does NOT prove otherwise (launchd IS pid 1). Missing this
    is exactly how a live server gets mislabeled ORPHAN. Non-running rows show
    "-" for the PID and are skipped."""
    pids = set()
    for line in text.splitlines():
        f = line.split("\t")
        if f and f[0].lstrip("-").isdigit():
            pids.add(int(f[0]))
    return pids


def _launchd_pids() -> set:
    return _parse_launchd_pids(_run(["launchctl", "list"]))


def _live_pgids(db: str) -> set:
    try:
        import sqlite3
        con = sqlite3.connect(db)
        return {r[0] for r in con.execute(
            "SELECT pgid FROM job WHERE state IN ('running','cancelling') "
            "AND pgid IS NOT NULL")}
    except Exception:
        return set()


def _ports() -> dict:
    out = _run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"])
    m: dict = {}
    for line in out.splitlines()[1:]:
        f = line.split()
        if len(f) >= 9 and f[1].isdigit():
            m.setdefault(int(f[1]), []).append(f[8].rsplit(":", 1)[-1])
    return m


def _footprint_mb(pid: int):
    m = re.search(r"phys_footprint:\s*([\d.]+)\s*([KMG]B)",
                  _run(["footprint", str(pid)]))
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    return v * {"KB": 1 / 1024, "MB": 1, "GB": 1024}.get(u, 1)


def _procs() -> list:
    out = _run(["ps", "-axo", "pid=,ppid=,pgid=,etime=,command="])
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, pgid, etime, cmd = parts
        if REPO not in cmd and "ollama" not in cmd.lower():
            continue
        if not _SIG.search(cmd):
            continue
        try:
            procs.append({"pid": int(pid), "ppid": int(ppid),
                          "pgid": int(pgid), "etime": etime, "cmd": cmd})
        except ValueError:
            continue
    return procs


def classify(proc: dict, launchd_pids: set, live_pgids: set) -> str:
    """Verdict for one process — pure, the unit under test. A launchd job OR its
    direct child is MANAGED; a member of a running job's process group is ACTIVE;
    anything else that matched our signatures is an ORPHAN safe to reap."""
    if proc["pid"] in launchd_pids or proc["ppid"] in launchd_pids:
        return "MANAGED"
    if proc.get("pgid") in live_pgids:
        return "ACTIVE"
    return "ORPHAN"


def report(db: "str | None" = None) -> list:
    db = db or os.path.join(REPO, "studio.db")
    launchd, live, ports = _launchd_pids(), _live_pgids(db), _ports()
    rows = []
    for p in _procs():
        rows.append({**p, "verdict": classify(p, launchd, live),
                     "ports": ports.get(p["pid"], []),
                     "mem_mb": _footprint_mb(p["pid"])})
    return rows


def _short(cmd: str) -> str:
    m = re.search(r"tools/(\w+)\.py", cmd)
    if m:
        return m.group(1)
    return "ollama" if "ollama" in cmd.lower() else cmd[:38]


def main(argv) -> int:
    rows = report()
    if "--json" in argv:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no model/GPU processes found for this project.")
        return 0
    print(f"{'PID':>7} {'VERDICT':<8} {'MEM':>8} {'PORTS':<10} {'UP':<12} WHAT")
    for r in sorted(rows, key=lambda x: (x["verdict"] != "ORPHAN", x["pid"])):
        mem = ("?" if not r["mem_mb"] else
               f"{r['mem_mb'] / 1024:.1f}GB" if r["mem_mb"] >= 1024
               else f"{r['mem_mb']:.0f}MB")
        print(f"{r['pid']:>7} {r['verdict']:<8} {mem:>8} "
              f"{','.join(r['ports']) or '-':<10} {r['etime']:<12} {_short(r['cmd'])}")
    orphans = [r for r in rows if r["verdict"] == "ORPHAN"]
    if orphans:
        print(f"\n⚠ {len(orphans)} ORPHAN(s) — repo model process, no launchd "
              "owner, no live job. Safe to reap:")
        print("   kill " + " ".join(str(r["pid"]) for r in orphans))
        print("MANAGED / ACTIVE are NOT orphans — leave them.")
    else:
        print("\n✓ no orphans — every model process is launchd-managed or tied "
              "to a live job.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
