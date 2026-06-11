"""ETA model: series medians -> global medians -> measured seeds."""
from studio.catalog.db import connect
from studio.dashboard import eta


def _con(tmp_path):
    return connect(tmp_path / "s.db")


def _run(con, chapter_id, stage, dur, series_id=None):
    con.execute(
        "INSERT INTO stage_run (chapter_id, stage, duration_sec, ok, meta_json)"
        " VALUES (?,?,?,1, json_object('series_id', ?))",
        (chapter_id, stage, dur, series_id))
    con.commit()


def test_seed_fallback_when_no_data(tmp_path):
    con = _con(tmp_path)
    assert eta.stage_eta(con, "voiced") == eta.SEED_SEC["voiced"]


def test_global_median_overrides_seed(tmp_path):
    con = _con(tmp_path)
    for d in (100, 200, 900):
        _run(con, 1, "voiced", d)
    assert eta.stage_eta(con, "voiced") == 200


def test_series_median_overrides_global(tmp_path):
    con = _con(tmp_path)
    _run(con, 1, "voiced", 999, series_id=7)
    _run(con, 2, "voiced", 111, series_id=8)
    assert eta.stage_eta(con, "voiced", series_id=8) == 111


def test_chapter_eta_sums_remaining(tmp_path):
    con = _con(tmp_path)
    total = eta.chapter_eta(con, 1, ["planned", "prepped", "qa_scan"])
    assert total == (eta.SEED_SEC["planned"] + eta.SEED_SEC["prepped"]
                     + eta.SEED_SEC["qa_scan"])


def test_fmt():
    assert eta.fmt_eta(440) == "7:20"
    assert eta.fmt_eta(5760) == "1.6 h"
    assert eta.fmt_eta(60 * 60 * 24 * 26) == "26 days"
