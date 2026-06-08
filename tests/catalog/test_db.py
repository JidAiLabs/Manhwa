from studio.catalog.db import connect

def test_schema_created(tmp_path):
    con = connect(tmp_path/"t.db")
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"series","chapter"} <= tables

def test_series_unique(tmp_path):
    import sqlite3, pytest
    con = connect(tmp_path/"t.db")
    con.execute("INSERT INTO series(source,series_url,slug,title,added_at) VALUES('a','u','s','t','now')")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO series(source,series_url,slug,title,added_at) VALUES('a','u','s2','t2','now')")
