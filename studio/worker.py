"""studio worker — the queue executor (run in its own terminal/launchd).

Claims ONE job at a time from studio.db (serial GPU policy), executes it,
streams output to logs/jobs/<id>.log, records per-stage durations into
stage_run, and enforces the gates: render needs a passing QA scan + your
approval; concat needs bundle approval. The dashboard never executes
anything — it only inserts job/approval rows that this process consumes.

Run:  .eval_venv/bin/python -m studio worker
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO

from studio.dashboard import bundles, gates, jobs

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".eval_venv" / "bin" / "python")


@contextlib.contextmanager
def record_stage(con: sqlite3.Connection, *, chapter_id: Optional[int],
                 stage: str, series_id: Optional[int] = None):
    """Wraps any stage execution: stage_run row with duration + ok flag."""
    t0 = time.time()
    ok = 1
    try:
        yield
    except BaseException:
        ok = 0
        raise
    finally:
        con.execute(
            "INSERT INTO stage_run (chapter_id, stage, duration_sec, ok, "
            "meta_json) VALUES (?,?,?,?, json_object('series_id', ?))",
            (chapter_id, stage, round(time.time() - t0, 2), ok, series_id))
        con.commit()


def _chapter(con: sqlite3.Connection, chapter_id: int) -> Dict[str, Any]:
    r = con.execute("SELECT id, series_id, number, label, ep_dir, status "
                    "FROM chapter WHERE id=?", (chapter_id,)).fetchone()
    if not r:
        raise RuntimeError(f"chapter {chapter_id} not in catalog")
    return dict(zip(("id", "series_id", "number", "label", "ep_dir",
                     "status"), r))


def _series_title(con: sqlite3.Connection, series_id: int) -> str:
    r = con.execute("SELECT title FROM series WHERE id=?",
                    (series_id,)).fetchone()
    return r[0] if r else ""


def _stream(cmd, log: TextIO, cwd: str = str(REPO)) -> int:
    log.write("$ " + " ".join(str(c) for c in cmd) + "\n")
    log.flush()
    p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                         text=True)
    return p.wait()


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

def _h_chain(con: sqlite3.Connection, job: Dict[str, Any], log: TextIO) -> None:
    """Run pipeline stages for one chapter up to payload['target'] via the
    studio CLI (it owns config, creds, resumability)."""
    ch = _chapter(con, job["chapter_id"])
    target = job["payload"].get("target", "planned")
    with record_stage(con, chapter_id=ch["id"], stage=f"chain:{target}",
                      series_id=ch["series_id"]):
        rc = _stream([PY, "-m", "studio", "run", str(ch["series_id"]),
                      "--chapters", str(int(ch["number"])),
                      "--until", target], log)
        if rc != 0:
            raise RuntimeError(f"studio run exited {rc}")


def _h_qa_scan(con: sqlite3.Connection, job: Dict[str, Any], log: TextIO) -> None:
    ch = _chapter(con, job["chapter_id"])
    title = _series_title(con, ch["series_id"])
    t0 = time.time()
    rc = _stream([PY, str(REPO / "tools" / "prep_qa.py"),
                  "--episode-dir", ch["ep_dir"] or "",
                  "--series-title", title], log)
    con.execute(
        "INSERT INTO stage_run (chapter_id, stage, duration_sec, ok, "
        "meta_json) VALUES (?,?,?,?, json_object('series_id', ?))",
        (ch["id"], "qa_scan", round(time.time() - t0, 2),
         1 if rc == 0 else 0, ch["series_id"]))
    con.commit()
    if rc != 0:
        raise RuntimeError("prep-QA found ERROR-severity flags "
                           f"(exit {rc}) — see report in {ch['ep_dir']}")


def _h_render_segment(con: sqlite3.Connection, job: Dict[str, Any],
                      log: TextIO) -> None:
    allowed, why = gates.render_allowed(con, job["chapter_id"])
    if not allowed:
        raise RuntimeError(f"render blocked: {why}")
    ch = _chapter(con, job["chapter_id"])
    branding = job["payload"].get("branding", "both")
    ep = Path(ch["ep_dir"] or "")
    with record_stage(con, chapter_id=ch["id"], stage="render_segment",
                      series_id=ch["series_id"]):
        rc = _stream([PY, str(REPO / "tools" / "render_prep.py"),
                      "--plan", str(ep / "render.plan.json"),
                      "--scenes-manifest", str(ep / "manifest.scenes.json"),
                      "--episode-dir", str(ep),
                      "--series-title", _series_title(con, ch["series_id"]),
                      "--branding", branding], log)
        if rc != 0:
            raise RuntimeError(f"render_prep exited {rc}")
        out = ep / "render" / f"segment_{branding}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        rc = _stream(["npx", "remotion", "render", "src/index.ts",
                      "RecapVideo", str(out),
                      f"--props={ep / 'render.plan.clean.json'}",
                      f"--public-dir={ep}", "--concurrency=8", "--crf=22"],
                     log, cwd=str(REPO / "remotion"))
        if rc != 0:
            raise RuntimeError(f"remotion exited {rc}")
        con.execute("UPDATE chapter SET status='rendered' WHERE id=?",
                    (ch["id"],))
        con.commit()


def _h_concat(con: sqlite3.Connection, job: Dict[str, Any], log: TextIO) -> None:
    allowed, why = gates.concat_allowed(con, job["bundle_id"])
    if not allowed:
        raise RuntimeError(f"concat blocked: {why}")
    bid = job["bundle_id"]
    segs = []
    for cid in bundles.bundle_chapters(con, bid):
        ch = _chapter(con, cid)
        rdir = Path(ch["ep_dir"] or "") / "render"
        found = sorted(rdir.glob("segment_*.mp4")) or sorted(rdir.glob("*.mp4"))
        if not found:
            raise RuntimeError(f"chapter {cid} has no rendered segment")
        segs.append(str(found[0]))
    out_dir = REPO / "dist" / f"bundle_{bid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bundle.mp4"
    argv, listfile = bundles.concat_cmd(segs, str(out))
    lf = out_dir / "concat.txt"
    lf.write_text(listfile)
    argv[argv.index("LISTFILE")] = str(lf)
    with record_stage(con, chapter_id=None, stage="concat"):
        rc = _stream(argv, log)
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")
    con.execute("UPDATE bundle SET state='concatenated', output_path=? "
                "WHERE id=?", (str(out), bid))
    con.commit()


def _h_refresh(con: sqlite3.Connection, job: Dict[str, Any], log: TextIO) -> None:
    rc = _stream([PY, "-m", "studio", "refresh"]
                 + (["--series", str(job["series_id"])] if job["series_id"]
                    else []), log)
    if rc != 0:
        raise RuntimeError(f"refresh exited {rc}")


HANDLERS: Dict[str, Callable[[sqlite3.Connection, Dict[str, Any], TextIO], None]] = {
    "chain": _h_chain,
    "qa_scan": _h_qa_scan,
    "render_segment": _h_render_segment,
    "concat": _h_concat,
    "refresh": _h_refresh,
}


def run_once(con: sqlite3.Connection, *, handlers=None,
             log_dir: str = "logs/jobs") -> bool:
    handlers = HANDLERS if handlers is None else handlers
    job = jobs.claim_next(con)
    if not job:
        return False
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{job['id']}-{job['type']}.log")
    jobs.set_log(con, job["id"], log_path)
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            handler = handlers.get(job["type"])
            if handler is None:
                raise RuntimeError(f"no handler for job type {job['type']!r}")
            handler(con, job, log)
        jobs.finish(con, job["id"], ok=True)
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("\n" + traceback.format_exc())
        jobs.finish(con, job["id"], ok=False, error=str(e))
    return True


def _heartbeat(con: sqlite3.Connection) -> None:
    con.execute("UPDATE job SET started_at=datetime('now') "
                "WHERE type='heartbeat'")
    if con.total_changes == 0 or con.execute(
            "SELECT COUNT(*) FROM job WHERE type='heartbeat'").fetchone()[0] == 0:
        con.execute("INSERT INTO job (type, state, started_at) "
                    "VALUES ('heartbeat','running',datetime('now'))")
    con.commit()


def main(db_path: str = "studio.db") -> int:
    from studio.catalog.db import connect
    con = connect(db_path)
    print(f"[worker] serial queue on {db_path} — ctrl-c to stop")
    try:
        while True:
            _heartbeat(con)
            if not run_once(con):
                time.sleep(2)
    except KeyboardInterrupt:
        print("\n[worker] stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "studio.db"))
