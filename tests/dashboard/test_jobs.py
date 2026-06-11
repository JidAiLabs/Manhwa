"""Serial job queue over studio.db — the worker claims ONE job at a time."""
from studio.catalog.db import connect
from studio.dashboard import jobs


def _con(tmp_path):
    return connect(tmp_path / "s.db")


def test_enqueue_and_serial_claim(tmp_path):
    con = _con(tmp_path)
    a = jobs.enqueue(con, "chain", chapter_id=1, payload={"target": "voiced"})
    b = jobs.enqueue(con, "qa_scan", chapter_id=1)
    j = jobs.claim_next(con)
    assert j["id"] == a and j["state"] == "running" and j["started_at"]
    # SERIAL: nothing else claimable while one runs
    assert jobs.claim_next(con) is None
    jobs.finish(con, a, ok=True)
    j2 = jobs.claim_next(con)
    assert j2["id"] == b


def test_priority_orders_before_id(tmp_path):
    con = _con(tmp_path)
    a = jobs.enqueue(con, "chain", chapter_id=1)
    b = jobs.enqueue(con, "chain", chapter_id=2)
    jobs.bump(con, b)                       # priority 99 < 100
    assert jobs.claim_next(con)["id"] == b


def test_cancel_only_queued(tmp_path):
    con = _con(tmp_path)
    a = jobs.enqueue(con, "chain", chapter_id=1)
    assert jobs.cancel(con, a) is True
    b = jobs.enqueue(con, "chain", chapter_id=2)
    jobs.claim_next(con)
    assert jobs.cancel(con, b) is False     # running -> not cancellable
    assert jobs.queue_view(con)[0]["state"] == "running"


def test_finish_failure_records_error(tmp_path):
    con = _con(tmp_path)
    a = jobs.enqueue(con, "render_segment", chapter_id=1)
    jobs.claim_next(con)
    jobs.finish(con, a, ok=False, error="needs render approval")
    row = [r for r in jobs.queue_view(con) if r["id"] == a][0]
    assert row["state"] == "failed" and "approval" in row["error"]
