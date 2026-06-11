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
