from studio.catalog.db import connect


def test_bundle_has_teaser_state_default_none(tmp_path):
    con = connect(tmp_path / "s.db")
    # real series columns: (source, series_url, slug, title, added_at NOT NULL, ...)
    con.execute("INSERT INTO series (source, series_url, slug, title, added_at) "
                "VALUES ('x','u','s','T', datetime('now'))")
    sid = con.execute("SELECT id FROM series").fetchone()[0]
    con.execute("INSERT INTO bundle (series_id, kind) VALUES (?, 'manual')", (sid,))
    con.commit()
    row = con.execute("SELECT teaser_state FROM bundle").fetchone()
    assert row[0] == "none"
