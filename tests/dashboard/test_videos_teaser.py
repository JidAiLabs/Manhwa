"""Teaser dashboard wiring: the Plan-teaser button enqueues a plan_teaser job,
and approve/decline set series.teaser_state — the teaser is ONE PER MANHWA,
so it survives deleting a video (the concat gate reads it off the series)."""
import pytest
from fastapi.testclient import TestClient

from studio.catalog.db import connect
from studio.dashboard import gates
from studio.dashboard.app import create_app


@pytest.fixture()
def client(tmp_path):
    db = tmp_path / "s.db"
    con = connect(db)
    # real series columns: (source, series_url, slug, title, added_at NOT NULL)
    con.execute("INSERT INTO series (source, series_url, slug, title, added_at) "
                "VALUES ('asura','https://asura.example/nano','nano',"
                "'Nano Machine', datetime('now'))")
    sid = con.execute("SELECT id FROM series").fetchone()[0]
    con.execute("INSERT INTO bundle (series_id, kind, title) "
                "VALUES (?, 'full', 'Nano — Full')", (sid,))
    con.commit()
    bid = con.execute("SELECT id FROM bundle").fetchone()[0]
    return TestClient(create_app(db_path=str(db))), con, sid, bid


def test_plan_teaser_enqueues_job(client):
    c, con, sid, bid = client
    r = c.post(f"/series/{sid}/teaser/plan", follow_redirects=False)
    assert r.status_code == 303
    assert con.execute(
        "SELECT COUNT(*) FROM job WHERE type='plan_teaser' AND series_id=?",
        (sid,)).fetchone()[0] == 1


def test_decline_sets_state(client):
    c, con, sid, bid = client
    r = c.post(f"/series/{sid}/teaser/decline", follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT teaser_state FROM series WHERE id=?",
                       (sid,)).fetchone()[0] == "declined"


def test_approve_sets_state_and_records_gate(client):
    c, con, sid, bid = client
    r = c.post(f"/series/{sid}/teaser/approve", follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT teaser_state FROM series WHERE id=?",
                       (sid,)).fetchone()[0] == "approved"
    # an explicit teaser approval is recorded (concat_allowed reads it via
    # bundle.teaser_state, not a dedicated teaser_allowed gate — that dead
    # API was deleted; approval EXISTENCE is what matters here)
    assert gates._has_approval(con, "teaser", series_id=sid) is True


def test_series_page_renders_create_teaser_button(client):
    """The teaser is made from the SERIES page now, independently of any video
    — one per manhwa, so it cannot be created or destroyed by video actions."""
    c, _con, sid, _bid = client
    r = c.get(f"/series/{sid}")
    assert r.status_code == 200
    assert "create teaser" in r.text.lower()
    assert f"/series/{sid}/teaser/plan" in r.text


def test_planned_teaser_shows_review_card(client, tmp_path, monkeypatch):
    """When a teaser is PLANNED and its manifest exists, the SERIES page shows
    the review card (hook narration + reason) with approve/decline forms."""
    import json
    c, con, sid, bid = client
    from studio.dashboard import app as _app
    monkeypatch.setattr(_app, "REPO", tmp_path)
    tdir = tmp_path / "dist" / f"series_{sid}" / "teaser"
    (tdir / "scenes").mkdir(parents=True)
    (tdir / "manifest.teaser.json").write_text(json.dumps({
        "source_chapters": [5],
        "scene_files": ["scene_0007.jpg"],
        "panel_narration": [{"scene_file": "scene_0007.jpg",
                             "line": "The exam begins."}],
        "reason": "public test + humiliation",
        "rewind_line": "But to see how he got here, we go back.",
        "spoiler_boundary": "no identity reveal"}))
    con.execute("UPDATE series SET teaser_state='planned' WHERE id=?", (sid,))
    con.commit()
    html = c.get(f"/series/{sid}").text
    assert "The exam begins." in html
    assert f"/series/{sid}/teaser/approve" in html
    assert f"/series/{sid}/teaser/decline" in html


def test_videos_page_flags_a_teaser_newer_than_the_video(tmp_path, monkeypatch):
    """"open" plays the CONCATENATED file, which carries the teaser as it was
    when the concat ran. Re-voicing the teaser therefore changes nothing the
    Videos page shows — it looked like the repair had not happened. The row now
    says so. The teaser itself is played from the SERIES page."""
    import json
    from fastapi.testclient import TestClient
    from studio.catalog.db import connect
    from studio.dashboard import app as appmod

    db = tmp_path / "s.db"
    con = connect(db)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','u','s','S','t')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at, season) VALUES "
                "(1,1,1,'Ch 1','u','rendered','',' t',1)")
    dist = tmp_path / "dist" / "bundle_1"
    dist.mkdir(parents=True)
    out = dist / "bundle.mp4"
    out.write_text("video")
    sdist = tmp_path / "dist" / "series_1"       # the teaser is per MANHWA
    sdist.mkdir(parents=True)
    teaser = sdist / "teaser.mp4"
    teaser.write_text("teaser")
    import os, time
    os.utime(out, (time.time() - 600, time.time() - 600))   # concat is OLDER
    con.execute("INSERT INTO bundle (id, series_id, title, kind, state, "
                "output_path) VALUES "
                "(1,1,'V','manual','concatenated',?)", (str(out),))
    con.execute("UPDATE series SET teaser_state='approved' WHERE id=1")
    con.commit()

    monkeypatch.setattr(appmod, "REPO", tmp_path)
    c = TestClient(appmod.create_app(db_path=str(db)))
    assert "newer than video" in c.get("/videos").text, "stale concat not flagged"
    # and the teaser itself is playable from the series page
    assert "/dist/series_1/teaser.mp4" in c.get("/series/1").text
