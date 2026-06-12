"""Worker: claims serially, logs, enforces gates, records stage timings."""
import time

from studio.catalog.db import connect
from studio.dashboard import gates, jobs
from studio import worker


def _con(tmp_path):
    return connect(tmp_path / "s.db")


def test_run_once_executes_and_logs(tmp_path):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "stub", chapter_id=3)
    seen = {}

    def stub(c, job, log):
        log.write("hello from stub\n")
        seen["job"] = job["id"]

    assert worker.run_once(con, handlers={"stub": stub},
                           log_dir=str(tmp_path / "logs")) is True
    row = [r for r in jobs.queue_view(con) if r["id"] == jid]
    assert not row or row[0]["state"] != "running"
    done = con.execute("SELECT state, log_path FROM job WHERE id=?",
                       (jid,)).fetchone()
    assert done[0] == "done" and seen["job"] == jid
    assert "hello from stub" in open(done[1]).read()


def test_run_once_idle_returns_false(tmp_path):
    con = _con(tmp_path)
    assert worker.run_once(con, handlers={}, log_dir=str(tmp_path)) is False


def test_handler_exception_fails_job(tmp_path):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "boom", chapter_id=1)

    def boom(c, job, log):
        raise RuntimeError("kaput")

    worker.run_once(con, handlers={"boom": boom}, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "kaput" in err


def test_render_segment_gate_refusal(tmp_path):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "render_segment", chapter_id=9)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "QA" in err


def test_recording_wrapper_writes_stage_run(tmp_path):
    con = _con(tmp_path)
    with worker.record_stage(con, chapter_id=4, stage="stitched",
                             series_id=2):
        time.sleep(0.01)
    row = con.execute("SELECT stage, ok, duration_sec FROM stage_run "
                      "WHERE chapter_id=4").fetchone()
    assert row[0] == "stitched" and row[1] == 1 and row[2] > 0


def test_recording_wrapper_records_failure(tmp_path):
    con = _con(tmp_path)
    try:
        with worker.record_stage(con, chapter_id=5, stage="beated"):
            raise ValueError("x")
    except ValueError:
        pass
    ok = con.execute("SELECT ok FROM stage_run WHERE chapter_id=5").fetchone()[0]
    assert ok == 0


def test_chain_past_scripted_requires_voice_approval(tmp_path):
    """run->planned crosses the voiceover line: blocked until the user
    approves the narration (gate='voice'); run->scripted is never gated."""
    con = _con(tmp_path)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','u','s','S','t')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES "
                "(1,1,1,'Ch 1','u','scripted','/tmp/x','t')")
    con.commit()
    jid = jobs.enqueue(con, "chain", chapter_id=1,
                       payload={"target": "planned"})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "narration" in err


# ---- stale-narration self-heal (mechanical heal only; prose never judged) --

def _seed_chapter(con, tmp_path, status="voiced_failed"):
    ep = tmp_path / "ep"
    ep.mkdir(exist_ok=True)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','https://x','s','S','t')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES (5,1,1,'Ch 1',"
                "'https://x/1',?,?,'t')", (status, str(ep)))
    con.commit()
    return ep


def test_qa_error_codes_reads_report(tmp_path):
    import json
    ep = tmp_path
    (ep / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "narration_stale", "severity": "ERROR"},
        {"code": "flash_cut", "severity": "WARN"}]}))
    assert worker._qa_error_codes(ep) == {"narration_stale"}
    assert worker._qa_error_codes(tmp_path / "nope") == set()


def test_run_prep_and_qa_heal_aware_returns_instead_of_raising(
        tmp_path, monkeypatch):
    import json
    import pytest
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    (ep / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "narration_stale", "severity": "ERROR"}]}))
    monkeypatch.setattr(worker, "_stream", lambda cmd, log:
                        1 if any("prep_qa.py" in str(c) for c in cmd) else 0)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep)}
    log = open(tmp_path / "log.txt", "w")
    codes = worker._run_prep_and_qa(con, ch, log, heal_aware=True)
    assert codes == {"narration_stale"}
    with pytest.raises(RuntimeError):
        worker._run_prep_and_qa(con, ch, log, heal_aware=False)


def test_prepare_self_heals_stale_narration(tmp_path, monkeypatch):
    con = _con(tmp_path)
    _seed_chapter(con, tmp_path)
    calls = []
    monkeypatch.setattr(worker, "_stream", lambda cmd, log:
                        (calls.append(" ".join(map(str, cmd))), 0)[1])
    qa_results = [{"narration_stale"}, set()]
    qa_calls = []

    def fake_qa(c, ch, log, **kw):
        qa_calls.append(1)
        return qa_results.pop(0)
    monkeypatch.setattr(worker, "_run_prep_and_qa", fake_qa)
    jid = jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "done", err
    assert len(qa_calls) == 2
    assert con.execute("SELECT status FROM chapter WHERE id=5"
                       ).fetchone()[0] == "beated"
    assert sum("--until scripted" in c for c in calls) >= 2
    assert sum("timeline_planner.py" in c for c in calls) >= 2


def test_prepare_heal_beats_incomplete_demotes_to_grouped(
        tmp_path, monkeypatch):
    con = _con(tmp_path)
    _seed_chapter(con, tmp_path)
    monkeypatch.setattr(worker, "_stream", lambda cmd, log: 0)
    qa_results = [{"beats_incomplete", "narration_stale"}, set()]
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: qa_results.pop(0))
    jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    assert con.execute("SELECT status FROM chapter WHERE id=5"
                       ).fetchone()[0] == "grouped"


def test_prepare_heal_gives_up_after_one_cycle(tmp_path, monkeypatch):
    con = _con(tmp_path)
    _seed_chapter(con, tmp_path)
    monkeypatch.setattr(worker, "_stream", lambda cmd, log: 0)
    qa_results = [{"narration_stale"}, {"narration_stale"}]
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: qa_results.pop(0))
    jid = jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "stale" in err
