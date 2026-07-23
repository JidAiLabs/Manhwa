"""The live Series-table overlay: the LATEST job per chapter drives
running/queued/failed visibility on top of the persisted stage — so a voicing
chapter no longer reads as a bare 'scripted', and a dead one no longer hides its
failure (the two staleness bugs the owner reported)."""
import sqlite3

from studio.dashboard import jobs


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(
        "CREATE TABLE chapter(id INTEGER PRIMARY KEY, series_id INT,"
        " number REAL, status TEXT);"
        "CREATE TABLE job(id INTEGER PRIMARY KEY, chapter_id INT,"
        " type TEXT, state TEXT);")
    return con


def test_chapter_activity_latest_job_wins():
    con = _db()
    con.executescript(
        "INSERT INTO chapter VALUES (1,9,2,'scripted'),(2,9,6,'scripted'),"
        " (3,9,1,'rendered'),(4,9,3,'downloaded');"
        "INSERT INTO job VALUES"
        " (10,1,'voiceover','running'),"
        " (11,2,'prepare','failed'),"
        " (20,3,'render_segment','done'),"
        # ch4: an older failure then a NEWER queued retry -> latest (queued) wins
        " (30,4,'prepare','failed'),(31,4,'prepare','queued'),"
        " (99,NULL,'heartbeat','running');")
    act = jobs.chapter_activity(con, 9)
    assert act[1] == {"type": "voiceover", "state": "running", "label": "voicing"}
    assert act[2]["state"] == "failed" and act[2]["label"] == "preparing"
    assert act[3]["state"] == "done"
    assert act[4]["state"] == "queued"          # newer retry supersedes failure
    # a heartbeat row (no chapter) never leaks in
    assert all(v["type"] != "heartbeat" for v in act.values())
    # a chapter with NO jobs simply isn't in the map (row shows its plain stage)
    assert set(act) == {1, 2, 3, 4}


def test_job_label_maps_known_and_falls_through():
    assert jobs.job_label("voiceover") == "voicing"
    assert jobs.job_label("render_segment") == "rendering"
    assert jobs.job_label("weird_type") == "weird_type"
    assert jobs.job_label("") == "?"
