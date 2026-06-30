# tests/catalog/test_niche.py
import sqlite3
from studio.catalog import db as catalog_db   # connect(path) -> new, migrated connection
from studio.catalog import repo


def _fresh_con(tmp_path):
    return catalog_db.connect(tmp_path / "studio.db")


def test_series_table_has_niche_columns(tmp_path):
    con = _fresh_con(tmp_path)
    cols = {row[1] for row in con.execute("PRAGMA table_info(series)")}
    assert {"niche_primary", "niche_secondary", "genres", "synopsis"} <= cols


def test_migration_is_idempotent_and_backcompat(tmp_path):
    # legacy DB WITHOUT the new columns, then let connect() migrate it.
    path = tmp_path / "old.db"
    raw = sqlite3.connect(str(path))
    raw.execute("CREATE TABLE series (id INTEGER PRIMARY KEY, source TEXT, "
                "series_url TEXT, slug TEXT, title TEXT, added_at TEXT, "
                "last_checked TEXT, poll_priority INTEGER DEFAULT 100, "
                "UNIQUE(source, series_url))")
    raw.execute("INSERT INTO series(source, series_url, slug, title, added_at) "
                "VALUES ('asura','u','s','t','now')")
    raw.commit(); raw.close()
    catalog_db.connect(path)          # 1st: ALTER-ADD the columns, must not crash
    con = catalog_db.connect(path)    # 2nd: idempotent, must not crash
    cols = {row[1] for row in con.execute("PRAGMA table_info(series)")}
    assert {"niche_primary", "niche_secondary", "genres", "synopsis"} <= cols
    assert con.execute("SELECT title FROM series WHERE id=1").fetchone()[0] == "t"


def test_upsert_and_get_roundtrip_niche(tmp_path):
    con = _fresh_con(tmp_path)
    sid = repo.upsert_series(con, source="asura", series_url="u", slug="s",
                             title="t", added_at="now",
                             niche_primary="C", niche_secondary="A",
                             genres="Action, Martial Arts", synopsis="syn")
    s = repo.get_series(con, sid)
    assert s.niche_primary == "C"
    assert s.niche_secondary == "A"
    assert s.genres == "Action, Martial Arts"
    assert s.synopsis == "syn"


def test_upsert_does_not_blank_existing_niche_on_metaless_redip(tmp_path):
    con = _fresh_con(tmp_path)
    repo.upsert_series(con, source="asura", series_url="u", slug="s", title="t",
                       added_at="now", niche_primary="C", genres="Action")
    # re-discovery with no metadata must NOT wipe the stored niche (COALESCE)
    sid = repo.upsert_series(con, source="asura", series_url="u", slug="s",
                             title="t2", added_at="now")
    s = repo.get_series(con, sid)
    assert s.niche_primary == "C"
