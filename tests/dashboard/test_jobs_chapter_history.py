"""chapter_history: ONE row per chapter for the dashboard, per-stage time =
the LATEST attempt per stage (last-run cost, NOT the lifetime sum across every
re-run) (replaces per-JOB spam)."""
from studio.catalog.db import connect
from studio.dashboard import jobs


def _seed(con):
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','u','nano','Nano Machine','t')")
    for cid, num in [(8, 8), (9, 9)]:
        con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                    "status, updated_at, season) VALUES "
                    "(?,1,?,?,?,'planned','t',1)", (cid, num, f"Chapter {num}",
                                                    f"u{num}"))


def _stage(con, cid, stage, dur, ok=1):
    con.execute("INSERT INTO stage_run (chapter_id, stage, duration_sec, ok) "
                "VALUES (?,?,?,?)", (cid, stage, dur, ok))


def test_chapter_history_latest_per_stage_and_orders_recent_first(tmp_path):
    con = connect(tmp_path / "s.db")
    _seed(con)
    # Ch8: a prepare pass + a voiceover pass -> qa_scan and prepped run TWICE.
    # The row must show the LATEST attempt per stage (last-run cost), NOT the
    # lifetime sum across both passes.
    _stage(con, 8, "chain:scripted", 600)     # 10 min prep (once)
    _stage(con, 8, "qa_scan", 240)            # prepare QA  (earlier attempt)
    _stage(con, 8, "prepped", 90)             #             (earlier attempt)
    _stage(con, 8, "voiced", 480)             # 8 min voice
    _stage(con, 8, "prepped", 480)            # voiceover render-prep (LATEST)
    _stage(con, 8, "qa_scan", 840)            # voiceover QA          (LATEST)
    _stage(con, 8, "render_segment", 570)
    _stage(con, 9, "chain:scripted", 300)     # Ch9 only prepped (most recent)
    con.commit()

    h = jobs.chapter_history(con)
    assert [r["chapter_id"] for r in h] == [9, 8]     # most-recently-active first
    ch8 = next(r for r in h if r["chapter_id"] == 8)
    bd = {s["label"]: s["sec"] for s in ch8["breakdown"]}
    assert bd["QA"] == 840                             # qa_scan: LATEST attempt only
    assert bd["render-prep"] == 480                   # prepped: LATEST attempt only
    assert bd["prep"] == 600 and bd["voice"] == 480 and bd["render"] == 570
    assert ch8["total_sec"] == 600 + 480 + 480 + 840 + 570   # latest-per-stage sum
    assert [s["label"] for s in ch8["breakdown"]] == [
        "prep", "voice", "render-prep", "QA", "render"]   # canonical order
    assert ch8["scope_name"] == "Nano Machine · Chapter 8"


def test_chapter_history_total_is_last_run_not_lifetime(tmp_path):
    """A chapter re-run many times during debugging must show the LAST run's
    cost, not the ever-growing lifetime sum. Same stage, 3 attempts → the row
    reports the latest (0.5s), never the 630.5s lifetime total."""
    con = connect(tmp_path / "s.db")
    _seed(con)
    _stage(con, 8, "chain:scripted", 600)     # first (slow) debug run
    _stage(con, 8, "chain:scripted", 30)      # second run
    _stage(con, 8, "chain:scripted", 0.5)     # latest run (the one to show)
    con.commit()

    ch8 = next(r for r in jobs.chapter_history(con) if r["chapter_id"] == 8)
    prep = next(s for s in ch8["breakdown"] if s["label"] == "prep")
    assert prep["sec"] == 0.5                  # latest attempt, NOT 600+30+0.5
    assert ch8["total_sec"] == 0.5             # total = latest-per-stage, not 630.5


def test_chapter_history_flags_failed_stage(tmp_path):
    con = connect(tmp_path / "s.db")
    _seed(con)
    _stage(con, 8, "qa_scan", 100, ok=0)              # a failed QA stage
    con.commit()
    qa = next(s for s in jobs.chapter_history(con)[0]["breakdown"]
              if s["label"] == "QA")
    assert qa["ok"] == 0


def test_chapter_history_recovered_stage_not_flagged(tmp_path):
    """A stage that FAILED then succeeded on a later attempt shows ok (no '!') —
    the row reflects the LATEST outcome, not the worst-ever. (The fetch-bug
    chapters that recovered must stop flashing '!' once rendered.)"""
    con = connect(tmp_path / "s.db")
    _seed(con)
    _stage(con, 8, "chain:scripted", 100, ok=0)   # original failed attempt
    _stage(con, 8, "chain:scripted", 600, ok=1)   # recovery succeeded
    con.commit()
    prep = next(s for s in jobs.chapter_history(con)[0]["breakdown"]
                if s["label"] == "prep")
    assert prep["ok"] == 1                          # latest attempt won → no "!"
    assert prep["sec"] == 600                        # latest attempt's time, not sum


def test_failed_chapters_only_lists_dead_lettered(tmp_path):
    """A chapter is listed ONLY when its MOST RECENT job failed — a chapter that
    recovered (later success) or has a pending retry queued is excluded."""
    con = connect(tmp_path / "s.db")
    _seed(con)  # series 1 + chapters 8, 9 (status 'planned')
    con.execute("INSERT INTO chapter (id,series_id,number,label,url,status,"
                "updated_at,season) VALUES (10,1,10,'Chapter 10','u10',"
                "'visioned','t',1)")

    def J(cid, state, err=None):
        con.execute("INSERT INTO job (type,series_id,chapter_id,payload_json,"
                    "priority,state,error) VALUES ('prepare',1,?,'{}',100,?,?)",
                    (cid, state, err))

    J(8, "failed", "boom"); J(8, "queued")          # Ch8: pending retry
    J(9, "failed", "exited 1")                        # Ch9: DEAD (latest=failed)
    J(10, "failed", "x"); J(10, "done")              # Ch10: recovered
    con.commit()

    dead = jobs.failed_chapters(con, 1)
    assert [d["chapter_id"] for d in dead] == [9]    # only the dead-lettered one
    assert dead[0]["error"] == "exited 1"
    assert jobs.failed_chapters(con, 999) == []      # other series: none


def test_chapter_history_excludes_running_chapter(tmp_path):
    """A chapter with a RUNNING job is shown live in 'running', NOT in recent
    chapters (its stage total would be partial → confusing). It reappears once
    the job is no longer running."""
    con = connect(tmp_path / "s.db")
    _seed(con)
    _stage(con, 8, "chain:scripted", 600)
    _stage(con, 9, "chain:scripted", 500)
    con.commit()
    assert {r["chapter_id"] for r in jobs.chapter_history(con)} == {8, 9}
    jobs.enqueue(con, "prepare", chapter_id=8, series_id=1)
    jobs.claim_next(con)                              # Ch8 -> running
    assert [r["chapter_id"] for r in jobs.chapter_history(con)] == [9]
