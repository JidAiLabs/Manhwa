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


def test_cancel_queued_outright_running_requests_cancelling(tmp_path):
    con = _con(tmp_path)
    a = jobs.enqueue(con, "chain", chapter_id=1)
    assert jobs.cancel(con, a) is True          # queued -> cancelled outright
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (a,)).fetchone()[0] == "cancelled"
    b = jobs.enqueue(con, "chain", chapter_id=2)
    jobs.claim_next(con)                         # b -> running
    assert jobs.cancel(con, b) is True          # running -> cancelling (worker kills it)
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (b,)).fetchone()[0] == "cancelling"


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


# ---- chapter lease: mutual exclusion across lanes for the SAME chapter -----

def test_claim_skips_chapter_with_running_job(tmp_path):
    """Chapter lease: a chapter with an already-RUNNING job is skipped for ANY
    other lane's queued job on that chapter — but the skip does not BLOCK the
    lane; the next eligible queued job (a different chapter) is still claimed."""
    con = _con(tmp_path)
    a_prep = jobs.enqueue(con, "prepare", chapter_id=1)
    assert jobs.claim_next(con, lane="gpu")["id"] == a_prep   # A's prepare running
    a_voice = jobs.enqueue(con, "voiceover", chapter_id=1)    # same chapter, diff lane
    b_voice = jobs.enqueue(con, "voiceover", chapter_id=2)    # different chapter
    claimed = jobs.claim_next(con, lane="tts")
    assert claimed is not None and claimed["id"] == b_voice   # A skipped, B claimed
    # A's voiceover is still sitting queued, not lost or silently claimed
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (a_voice,)).fetchone()[0] == "queued"


def test_claim_lease_released_on_finish(tmp_path):
    con = _con(tmp_path)
    a_prep = jobs.enqueue(con, "prepare", chapter_id=1)
    jobs.claim_next(con, lane="gpu")
    a_voice = jobs.enqueue(con, "voiceover", chapter_id=1)
    assert jobs.claim_next(con, lane="tts") is None      # leased while prepare runs
    jobs.finish(con, a_prep, ok=True)
    claimed = jobs.claim_next(con, lane="tts")
    assert claimed is not None and claimed["id"] == a_voice


def test_claim_update_guard_blocks_race(tmp_path):
    """A true SELECT-then-UPDATE race window is hard to fabricate in a
    single-threaded synchronous test — but the same lease guard lives on the
    UPDATE, so this asserts its observable, end-to-end behavior: a chapter
    with a running job blocks ALL its other queued jobs (even same-lane), and
    claim_next moves on to the next eligible candidate instead of returning
    None just because the head of the queue is leased out."""
    con = _con(tmp_path)
    jobs.enqueue(con, "prepare", chapter_id=1)
    jobs.claim_next(con, lane="gpu")                      # A's prepare -> running
    a_qa = jobs.enqueue(con, "qa_scan", chapter_id=1)     # same lane, same chapter
    b_prep = jobs.enqueue(con, "prepare", chapter_id=2)   # different chapter
    claimed = jobs.claim_next(con, lane="gpu")
    assert claimed is not None and claimed["id"] == b_prep
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (a_qa,)).fetchone()[0] == "queued"


# ---- enqueue dedupe: double-clicks / retries must not pile up duplicates --

def test_enqueue_dedupes_identical_queued(tmp_path):
    con = _con(tmp_path)
    a = jobs.enqueue(con, "prepare", chapter_id=1)
    b = jobs.enqueue(con, "prepare", chapter_id=1)
    assert a == b
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='prepare' AND "
                       "chapter_id=1").fetchone()[0] == 1


def test_enqueue_dedupe_off_inserts(tmp_path):
    con = _con(tmp_path)
    a = jobs.enqueue(con, "prepare", chapter_id=1)
    b = jobs.enqueue(con, "prepare", chapter_id=1, dedupe=False)
    assert a != b
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='prepare' AND "
                       "chapter_id=1").fetchone()[0] == 2


def test_enqueue_dedupes_against_a_RUNNING_job(tmp_path):
    """The dedupe key covers queued AND live rows. Matching only 'queued' meant
    the guard stopped working the instant the worker claimed the job, so a
    second browser tab or an impatient double-click enqueued a duplicate of a
    ~40 GPU-minute prepare. (The worker's own auto-retry does not rely on
    slipping past this — it passes dedupe=False explicitly; see below.)"""
    con = _con(tmp_path)
    a = jobs.enqueue(con, "prepare", chapter_id=1)
    jobs.claim_next(con, lane="gpu")                     # a -> running
    assert jobs.enqueue(con, "prepare", chapter_id=1) == a
    jobs.cancel(con, a)                                  # a -> cancelling
    assert jobs.enqueue(con, "prepare", chapter_id=1) == a, \
        "a cancelling job's work is still on the books"
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='prepare'"
                       ).fetchone()[0] == 1


def test_retry_style_reenqueue_forces_a_fresh_row(tmp_path):
    """The worker's auto-retry MUST get its own row: dedupe ignores the
    payload, so folding onto an existing row would drop the incremented
    _attempt counter and loop a failing chapter forever."""
    con = _con(tmp_path)
    a = jobs.enqueue(con, "prepare", chapter_id=1)
    jobs.claim_next(con, lane="gpu")
    b = jobs.enqueue(con, "prepare", chapter_id=1,
                     payload={"_attempt": 1}, dedupe=False)
    assert b != a
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (b,)).fetchone()[0] == "queued"


def test_enqueue_series_scope_key_includes_series(tmp_path):
    con = _con(tmp_path)
    a = jobs.enqueue(con, "refresh", series_id=1)
    b = jobs.enqueue(con, "refresh", series_id=2)
    assert a != b
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='refresh'"
                       ).fetchone()[0] == 2
    # same series_id DOES dedupe (series-scoped key)
    c = jobs.enqueue(con, "refresh", series_id=1)
    assert c == a


def test_enqueue_dedupe_adopts_lower_priority(tmp_path):
    """A dedupe hit adopts the more urgent (lower) priority so expedite call
    sites keep working. Raising priority is never done (a later low-urgency
    duplicate must not demote an expedited job)."""
    con = _con(tmp_path)
    # enqueue with default priority 100
    a = jobs.enqueue(con, "prepare", chapter_id=1)
    row = con.execute("SELECT priority FROM job WHERE id=?", (a,)).fetchone()
    assert row[0] == 100
    # enqueue same key with priority 1 (more urgent) → returns a, priority becomes 1
    b = jobs.enqueue(con, "prepare", chapter_id=1, priority=1)
    assert b == a
    row = con.execute("SELECT priority FROM job WHERE id=?", (a,)).fetchone()
    assert row[0] == 1
    # enqueue same key with priority 50 (less urgent) → returns a, priority STAYS 1
    c = jobs.enqueue(con, "prepare", chapter_id=1, priority=50)
    assert c == a
    row = con.execute("SELECT priority FROM job WHERE id=?", (a,)).fetchone()
    assert row[0] == 1


def test_all_gemma_jobs_share_the_gpu_lane(tmp_path):
    """Every job that makes a local gemma call must be on the gpu lane, or two
    gemma contexts collide on one GPU and Metal OOMs (the width-2 prepare
    failure). Teaser + thumbnail + metadata all call gemma, so they belong
    here — NOT on a separate lane that runs them concurrently with a prepare."""
    for t in ("prepare", "qa_scan", "chain",
              "plan_teaser", "series_thumbnail", "publish_meta"):
        assert jobs.LANES[t] == "gpu", f"{t} must be on the gpu lane"
    # qwen TTS stays on its OWN lane so a voiceover still overlaps a prepare
    assert jobs.LANES["voiceover"] == "tts"
    # ffmpeg/remotion + network jobs use no local model
    assert jobs.LANES["render_segment"] == "cpu"
    assert jobs.LANES["refresh"] == "api"


def test_teaser_and_prepare_do_not_run_concurrently(tmp_path, monkeypatch):
    """With the gpu lane at width 1 (the Mini), a prepare and a teaser can
    never both hold a gemma slot at once — they queue, they don't collide."""
    monkeypatch.setitem(jobs.LANE_WIDTH, "gpu", 1)   # the Mini's setting
    con = _con(tmp_path)
    p = jobs.enqueue(con, "prepare", chapter_id=1)
    jobs.enqueue(con, "plan_teaser", bundle_id=1)
    assert jobs.claim_next(con, lane="gpu")["id"] == p
    assert jobs.claim_next(con, lane="gpu") is None, \
        "teaser must wait for the prepare's gemma slot, not run beside it"
