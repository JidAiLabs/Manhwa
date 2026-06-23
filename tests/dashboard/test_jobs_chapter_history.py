"""chapter_history: ONE row per chapter for the dashboard, per-stage time
SUMMED across the prepare/voiceover/heal re-runs (replaces per-JOB spam)."""
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


def test_chapter_history_sums_stages_and_orders_recent_first(tmp_path):
    con = connect(tmp_path / "s.db")
    _seed(con)
    # Ch8: a prepare pass + a voiceover pass -> qa_scan and prepped run TWICE
    _stage(con, 8, "chain:scripted", 600)     # 10 min prep (once)
    _stage(con, 8, "qa_scan", 240)            # prepare QA
    _stage(con, 8, "prepped", 90)
    _stage(con, 8, "voiced", 480)             # 8 min voice
    _stage(con, 8, "prepped", 480)            # voiceover render-prep
    _stage(con, 8, "qa_scan", 840)            # voiceover QA
    _stage(con, 8, "render_segment", 570)
    _stage(con, 9, "chain:scripted", 300)     # Ch9 only prepped (most recent)
    con.commit()

    h = jobs.chapter_history(con)
    assert [r["chapter_id"] for r in h] == [9, 8]     # most-recently-active first
    ch8 = next(r for r in h if r["chapter_id"] == 8)
    bd = {s["label"]: s["sec"] for s in ch8["breakdown"]}
    assert bd["QA"] == 240 + 840                       # qa_scan SUMMED across passes
    assert bd["render-prep"] == 90 + 480              # prepped SUMMED
    assert bd["prep"] == 600 and bd["voice"] == 480 and bd["render"] == 570
    assert ch8["total_sec"] == 600 + 240 + 90 + 480 + 480 + 840 + 570
    assert [s["label"] for s in ch8["breakdown"]] == [
        "prep", "voice", "render-prep", "QA", "render"]   # canonical order
    assert ch8["scope_name"] == "Nano Machine · Chapter 8"


def test_chapter_history_flags_failed_stage(tmp_path):
    con = connect(tmp_path / "s.db")
    _seed(con)
    _stage(con, 8, "qa_scan", 100, ok=0)              # a failed QA stage
    con.commit()
    qa = next(s for s in jobs.chapter_history(con)[0]["breakdown"]
              if s["label"] == "QA")
    assert qa["ok"] == 0
