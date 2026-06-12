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


def test_queue_view_keeps_recent_finished_jobs(tmp_path):
    """Done jobs must stay visible (with logs) — they should not vanish the
    moment they finish (user lost their first QA scan this way)."""
    con = _con(tmp_path)
    a = jobs.enqueue(con, "qa_scan", chapter_id=1)
    jobs.claim_next(con)
    jobs.finish(con, a, ok=True)
    view = jobs.queue_view(con)
    assert any(r["id"] == a and r["state"] == "done" for r in view)


def test_lane_claims_run_in_parallel_but_serial_within_lane(tmp_path):
    """Assembly line: gpu (prepare/voice), cpu (render/concat), api lanes
    each run ONE job — three jobs total can be active simultaneously."""
    con = _con(tmp_path)
    g = jobs.enqueue(con, "voiceover", chapter_id=1)
    c = jobs.enqueue(con, "render_segment", chapter_id=2)
    a = jobs.enqueue(con, "refresh", series_id=1)
    g2 = jobs.enqueue(con, "prepare", chapter_id=3)
    assert jobs.claim_next(con, lane="gpu")["id"] == g
    assert jobs.claim_next(con, lane="gpu") is None        # gpu busy
    assert jobs.claim_next(con, lane="cpu")["id"] == c     # cpu free
    assert jobs.claim_next(con, lane="api")["id"] == a
    jobs.finish(con, g, ok=True)
    assert jobs.claim_next(con, lane="gpu")["id"] == g2


def test_legacy_claim_without_lane_is_fully_serial(tmp_path):
    con = _con(tmp_path)
    jobs.enqueue(con, "voiceover", chapter_id=1)
    jobs.enqueue(con, "render_segment", chapter_id=2)
    assert jobs.claim_next(con) is not None
    assert jobs.claim_next(con) is None
