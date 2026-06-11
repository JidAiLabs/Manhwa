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
