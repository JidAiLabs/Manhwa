"""Dashboard catalog migrations: new tables + chapter.season (additive)."""
import sqlite3

from studio.catalog.db import connect


def test_new_tables_and_season_column(tmp_path):
    con = connect(tmp_path / "s.db")
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"job", "stage_run", "approval", "bundle", "bundle_chapter",
            "discovery_title"} <= names
    cols = {r[1] for r in con.execute("PRAGMA table_info(chapter)")}
    assert "season" in cols


def test_existing_db_upgraded(tmp_path):
    p = tmp_path / "old.db"
    raw = sqlite3.connect(p)
    raw.execute("CREATE TABLE chapter (id INTEGER PRIMARY KEY, number REAL)")
    raw.commit()
    raw.close()
    con = connect(p)
    cols = {r[1] for r in con.execute("PRAGMA table_info(chapter)")}
    assert "season" in cols


def test_job_pgid_column_added(tmp_path):
    """pgid (additive, nullable) persists the live child's process-group id for
    a running job so a restarted worker can reap survivors before requeuing."""
    p = tmp_path / "s.db"
    con = connect(p)
    cols = {r[1] for r in con.execute("PRAGMA table_info(job)")}
    assert "pgid" in cols
    # re-connecting (a worker restart against an already-migrated db) must not
    # error — the ALTER TABLE is guarded by the same PRAGMA check used for
    # every other additive column in this file.
    con2 = connect(p)
    cols2 = {r[1] for r in con2.execute("PRAGMA table_info(job)")}
    assert "pgid" in cols2


def test_approval_content_sha_column_added(tmp_path):
    """content_sha (additive, nullable) binds an approval to the content it
    approved — see studio/dashboard/gates.py:gate_sha/_approval_valid."""
    p = tmp_path / "s.db"
    con = connect(p)
    cols = {r[1] for r in con.execute("PRAGMA table_info(approval)")}
    assert "content_sha" in cols
    # re-connecting must not error (same additive-migration guard as pgid)
    con2 = connect(p)
    cols2 = {r[1] for r in con2.execute("PRAGMA table_info(approval)")}
    assert "content_sha" in cols2


def test_series_teaser_state_is_added_and_carries_bundle_state(tmp_path):
    """The teaser is one per MANHWA. It was a bundle column, so deleting a
    video destroyed its teaser and there was no way to keep one independently.
    Moving it to `series` must not silently drop an already-approved teaser."""
    import sqlite3
    from studio.catalog.db import connect

    db = tmp_path / "s.db"
    con = connect(db)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','u','s','S','t')")
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (2,'asura','u2','s2','S2','t')")
    con.execute("INSERT INTO bundle (id, series_id, title, kind, teaser_state) "
                "VALUES (1,1,'v','manual','approved')")
    con.execute("INSERT INTO bundle (id, series_id, title, kind, teaser_state) "
                "VALUES (2,2,'v','manual','none')")
    con.commit()
    # Simulate the PRE-migration shape: rows already present, column absent.
    # connect() migrates on first open, so without this the carry-forward runs
    # against an empty bundle table and proves nothing.
    con.execute("ALTER TABLE series DROP COLUMN teaser_state")
    con.commit()
    con.close()

    # re-open: connect() runs the migrations against the existing file
    con = connect(db)
    got = dict(con.execute("SELECT id, teaser_state FROM series").fetchall())
    assert got[1] == "approved", "an approved teaser was lost in the move"
    assert got[2] == "none"
