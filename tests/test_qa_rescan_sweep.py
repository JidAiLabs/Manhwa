"""tools/qa_rescan_sweep.py — the one-shot event that un-parks chapters left
behind by the pre-2026-09-07 autopilot policy. Classifies the way the worker
will read the report, enqueues qa_scan only for the ones that will advance,
and is idempotent (jobs.enqueue dedupes)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from studio.catalog.db import connect

_SPEC = importlib.util.spec_from_file_location(
    "qa_rescan_sweep",
    Path(__file__).resolve().parents[1] / "tools" / "qa_rescan_sweep.py")
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _seed(tmp_path):
    db = tmp_path / "s.db"
    con = connect(db)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at, autopilot) VALUES (1,'asura','https://x','s','S','t',1)")
    for cid, codes in ((5, ["visible_text"]), (6, ["cut_gap"]),
                       (7, ["visible_text"])):
        ep = tmp_path / f"ep{cid}"
        ep.mkdir()
        (ep / "prep_qa.json").write_text(json.dumps({"flags": [
            {"code": c, "severity": "ERROR"} for c in codes]}))
        con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                    "status, ep_dir, updated_at) VALUES (?,1,?,?,'https://x',"
                    "'scripted',?,'t')", (cid, cid, f"Ch {cid}", str(ep)))
        con.execute("INSERT INTO stage_run (chapter_id, stage, duration_sec, "
                    "ok, meta_json) VALUES (?,'qa_scan',1.0,1,'{}')", (cid,))
    # chapter 7 already has a live job -> not a candidate
    con.execute("INSERT INTO job (type, chapter_id, state, priority) "
                "VALUES ('prepare', 7, 'running', 1)")
    con.commit()
    return db, con


def _run(capsys, db, *argv) -> str:
    assert tool.main(["--db", str(db), *argv]) == 0
    return capsys.readouterr().out


def test_dry_run_classifies_like_the_worker(tmp_path, capsys):
    db, con = _seed(tmp_path)
    out = _run(capsys, db)
    rows = {l.split()[0]: l for l in out.splitlines() if l.strip()[:1].isdigit()}
    assert "WILL ADVANCE" in rows["5"] and "visible_text" in rows["5"]
    assert "WILL BLOCK" in rows["6"] and "cut_gap" in rows["6"]
    assert "7" not in rows                          # live job -> skipped
    assert "1 will advance, 1 will block" in out
    assert "dry-run" in out
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='qa_scan'"
                       ).fetchone()[0] == 0


def test_apply_enqueues_only_the_advancing_and_is_idempotent(tmp_path, capsys):
    db, con = _seed(tmp_path)
    _run(capsys, db, "--apply")
    jobs = con.execute("SELECT chapter_id, priority FROM job WHERE "
                       "type='qa_scan' ORDER BY chapter_id").fetchall()
    assert jobs == [(5, 1)]
    _run(capsys, db, "--apply")                     # dedupe: nothing new
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='qa_scan'"
                       ).fetchone()[0] == 1
    _run(capsys, db, "--apply", "--include-blocked")
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='qa_scan'"
                       ).fetchone()[0] == 2


def test_limit_smokes_one(tmp_path, capsys):
    db, con = _seed(tmp_path)
    con.execute("UPDATE chapter SET status='planned' WHERE id=6")
    con.execute("UPDATE chapter SET ep_dir=? WHERE id=6",
                (str(tmp_path / "ep5"),))              # make 6 advance too
    con.commit()
    out = _run(capsys, db, "--apply", "--limit", "1")
    assert "1 qa_scan job(s) enqueued" in out
