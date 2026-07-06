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
