"""Dashboard routes: every page renders; actions only insert rows."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio.catalog.db import connect
from studio.dashboard.app import create_app


@pytest.fixture()
def client(tmp_path):
    db = tmp_path / "s.db"
    con = connect(db)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','u','nano','Nano Machine','t')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, updated_at, season) VALUES "
                "(1,1,1,'Chapter 1','u1','planned','t',1)")
    con.commit()
    return TestClient(create_app(db_path=str(db))), con


def test_all_pages_render(client):
    c, _ = client
    for path in ("/", "/series", "/series/1", "/chapter/1", "/videos",
                 "/discovery", "/health"):
        r = c.get(path)
        assert r.status_code == 200, path
    assert "Nano Machine" in c.get("/series").text


def test_post_job_inserts_queued_row(client):
    c, con = client
    r = c.post("/jobs", data={"type": "chain", "chapter_id": 1,
                              "target": "planned"}, follow_redirects=False)
    assert r.status_code == 303
    row = con.execute("SELECT type, state, payload_json FROM job").fetchone()
    assert row[0] == "chain" and row[1] == "queued" and "planned" in row[2]


def test_approve_and_chapter_lock_state(client):
    c, con = client
    assert "approval" in c.get("/chapter/1").text or "QA" in c.get("/chapter/1").text
    c.post("/approve", data={"gate": "render", "chapter_id": 1},
           follow_redirects=False)
    assert con.execute("SELECT COUNT(*) FROM approval WHERE gate='render'"
                       ).fetchone()[0] == 1


def test_bundle_create_and_videos_page(client):
    c, con = client
    r = c.post("/bundles", data={"series_id": 1, "kind": "full",
                                 "title": "Nano — Full"},
               follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT COUNT(*) FROM bundle").fetchone()[0] == 1
    assert "Nano — Full" in c.get("/videos").text


def test_cancel_route(client):
    c, con = client
    c.post("/jobs", data={"type": "qa_scan", "chapter_id": 1},
           follow_redirects=False)
    jid = con.execute("SELECT id FROM job").fetchone()[0]
    c.post(f"/jobs/{jid}/cancel", follow_redirects=False)
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (jid,)).fetchone()[0] == "cancelled"


def test_log_partial_tails_file(client, tmp_path):
    c, con = client
    log = tmp_path / "x.log"
    log.write_text("alpha\nbeta\n")
    con.execute("INSERT INTO job (type, state, log_path) VALUES "
                "('chain','running',?)", (str(log),))
    con.commit()
    jid = con.execute("SELECT id FROM job WHERE log_path IS NOT NULL"
                      ).fetchone()[0]
    assert "beta" in c.get(f"/partials/log/{jid}").text
