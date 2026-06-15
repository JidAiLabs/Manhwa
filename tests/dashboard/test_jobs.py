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


def test_lane_claims_respect_per_lane_width(tmp_path):
    """Assembly line with WIDTH: gpu (gemma) runs 2 prepares at once, while a
    voiceover runs IN PARALLEL in its own tts (qwen) lane — not blocked behind
    the prepares; cpu stays exclusive."""
    con = _con(tmp_path)
    v = jobs.enqueue(con, "voiceover", chapter_id=1)        # tts lane
    c = jobs.enqueue(con, "render_segment", chapter_id=2)   # cpu lane
    a = jobs.enqueue(con, "refresh", series_id=1)           # api lane
    g1 = jobs.enqueue(con, "prepare", chapter_id=3)         # gpu lane
    g2 = jobs.enqueue(con, "prepare", chapter_id=4)         # gpu lane
    g3 = jobs.enqueue(con, "prepare", chapter_id=5)         # gpu lane (3rd, waits)
    assert jobs.claim_next(con, lane="gpu")["id"] == g1
    assert jobs.claim_next(con, lane="gpu")["id"] == g2     # gpu width 2
    assert jobs.claim_next(con, lane="gpu") is None         # gpu full -> g3 waits
    assert jobs.claim_next(con, lane="tts")["id"] == v      # voiceover runs in PARALLEL
    assert jobs.claim_next(con, lane="tts") is None         # tts width 1
    assert jobs.claim_next(con, lane="cpu")["id"] == c      # cpu free
    assert jobs.claim_next(con, lane="api")["id"] == a
    jobs.finish(con, g1, ok=True)
    assert jobs.claim_next(con, lane="gpu")["id"] == g3     # freed slot -> 3rd prepare


def test_claim_race_lost_returns_none(tmp_path):
    con = _con(tmp_path)
    j = jobs.enqueue(con, "prepare", chapter_id=1)
    con.execute("UPDATE job SET state='running' WHERE id=?", (j,))
    con.commit()                       # sibling thread won the claim
    assert jobs.claim_next(con, lane="gpu") is None


def test_orphan_requeue_at_boot(tmp_path):
    from studio import worker
    con = _con(tmp_path)
    j = jobs.enqueue(con, "prepare", chapter_id=1)
    con.execute("UPDATE job SET state='running' WHERE id=?", (j,))
    con.commit()                       # worker died mid-job
    n = worker.requeue_orphans(con)
    assert n == 1
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (j,)).fetchone()[0] == "queued"


def test_legacy_claim_without_lane_is_fully_serial(tmp_path):
    con = _con(tmp_path)
    jobs.enqueue(con, "voiceover", chapter_id=1)
    jobs.enqueue(con, "render_segment", chapter_id=2)
    assert jobs.claim_next(con) is not None
    assert jobs.claim_next(con) is None


def test_every_worker_handler_has_a_lane():
    """The worker only runs per-lane loops (the serial lane=None path is unused),
    so a job type in HANDLERS but missing from LANES can NEVER be claimed — it
    queues forever. This regressed once for publish_meta; guard it for good."""
    from studio import worker
    missing = [t for t in worker.HANDLERS if t not in jobs.LANES]
    assert not missing, f"handlers with no lane (would queue forever): {missing}"
    bad = [(t, l) for t, l in jobs.LANES.items() if l not in jobs.LANE_WIDTH]
    assert not bad, f"lanes pointing at an unknown width bucket: {bad}"


def test_series_thumbnail_is_claimable_on_its_lane(tmp_path):
    con = _con(tmp_path)
    jobs.enqueue(con, "series_thumbnail", series_id=1)
    j = jobs.claim_next(con, lane=jobs.LANES["series_thumbnail"])
    assert j and j["type"] == "series_thumbnail" and j["series_id"] == 1
