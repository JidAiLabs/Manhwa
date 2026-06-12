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
                "added_at) VALUES (1,'asura','https://asura.example/nano',"
                "'nano','Nano Machine','t')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, updated_at, season) VALUES (1,1,1,'Chapter 1',"
                "'https://asura.example/nano/ch1','planned','t',1)")
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


def test_approvals_auto_advance_the_pipeline(client):
    """The user's flow: approving IS the trigger. Story approval enqueues
    voiceover; voiceover approval enqueues the render."""
    c, con = client
    c.post("/approve", data={"gate": "voice", "chapter_id": 1},
           follow_redirects=False)
    c.post("/approve", data={"gate": "render", "chapter_id": 1},
           follow_redirects=False)
    types = [r[0] for r in con.execute("SELECT type FROM job ORDER BY id")]
    assert types == ["voiceover", "render_segment"]


def test_prepare_series_expands_to_per_chapter_jobs(client):
    c, con = client
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, updated_at) VALUES (2,1,2,'Ch 2','u2','discovered','t')")
    # chapter 1 already has a green QA scan -> only chapter 2 needs prep
    con.execute("INSERT INTO stage_run (chapter_id, stage, ok, duration_sec)"
                " VALUES (1,'qa_scan',1,10)")
    con.commit()
    c.post("/jobs", data={"type": "prepare_series", "series_id": 1},
           follow_redirects=False)
    rows = con.execute("SELECT type, chapter_id FROM job").fetchall()
    assert rows == [("prepare", 2)]


def test_discovery_add_creates_job_and_marks(client):
    c, con = client
    con.execute("INSERT INTO discovery_title (anilist_id, title) "
                "VALUES (42,'Some Manhwa')")
    con.commit()
    c.post("/discovery/42/add", data={"source": "asura", "url": "https://x"},
           follow_redirects=False)
    assert con.execute("SELECT status FROM discovery_title WHERE anilist_id=42"
                       ).fetchone()[0] == "in_production"
    t, payload = con.execute("SELECT type, payload_json FROM job").fetchone()
    assert t == "add_series" and "asura" in payload


def test_token_auth_when_env_set(client, monkeypatch, tmp_path):
    from studio.catalog.db import connect as _c
    from studio.dashboard.app import create_app
    from fastapi.testclient import TestClient
    monkeypatch.setenv("STUDIO_DASH_TOKEN", "sekret")
    db = tmp_path / "t.db"
    _c(db)
    c = TestClient(create_app(db_path=str(db)))
    assert c.get("/").status_code == 401              # locked
    assert "form" in c.get("/login").text             # GET shows the form
    c.post("/login", data={"token": "wrong"}, follow_redirects=False)
    assert c.get("/").status_code == 401
    c.post("/login", data={"token": "sekret"}, follow_redirects=False)
    assert c.get("/").status_code == 200              # cookie set (POST only)


def test_no_token_env_means_open(client):
    c, _ = client
    assert c.get("/").status_code == 200


def test_real_manhwa_links_on_pages(client):
    """Series board, series detail, chapter header, and discovery rows all
    link out to the real reader pages (series_url / chapter.url / AniList)."""
    c, con = client
    assert 'href="https://asura.example/nano"' in c.get("/series").text
    assert 'href="https://asura.example/nano"' in c.get("/series/1").text
    assert ('href="https://asura.example/nano/ch1"'
            in c.get("/chapter/1").text)
    con.execute("INSERT INTO discovery_title (anilist_id, title, trend_score,"
                " chapters, status, meta_json) VALUES "
                "(77,'Solo Farming',90,120,'candidate','{}')")
    con.commit()
    assert "anilist.co/manga/77" in c.get("/discovery").text


def test_unsafe_url_schemes_never_rendered(client):
    """Scraped/stored URLs are untrusted: javascript:/data: schemes must
    never reach an href, and manual discovery-add must reject them."""
    c, con = client
    con.execute("UPDATE series SET series_url='javascript:alert(1)'")
    con.execute("UPDATE chapter SET url='javascript:alert(2)'")
    con.execute("INSERT INTO discovery_title (anilist_id, title, trend_score,"
                " chapters, status, meta_json) VALUES (88,'Evil',95,10,"
                "'candidate','{\"links\":{\"asura\":{\"url\":"
                "\"javascript:alert(3)\",\"title\":\"x\",\"score\":0.9}}}')")
    con.commit()
    for path in ("/series", "/series/1", "/chapter/1", "/discovery"):
        assert "javascript:" not in c.get(path).text, path
    r = c.post("/discovery/88/add", data={"source": "asura",
               "url": "javascript:alert(4)"}, follow_redirects=False)
    assert r.status_code == 400
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='add_series'"
                       ).fetchone()[0] == 0


def test_autopilot_toggle_and_badge(client):
    c, con = client
    r = c.post("/series/1/autopilot", follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT autopilot FROM series WHERE id=1"
                       ).fetchone()[0] == 1
    assert "autopilot" in c.get("/series/1").text.lower()
    c.post("/series/1/autopilot", follow_redirects=False)   # toggles back
    assert con.execute("SELECT autopilot FROM series WHERE id=1"
                       ).fetchone()[0] == 0


def test_rebuild_route_resets_and_enqueues(client):
    """Shipped stage-code fixes only apply when the stage re-runs — the
    rebuild button demotes to 'detected' and queues a fresh prepare."""
    c, con = client
    r = c.post("/chapter/1/rebuild", follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT status FROM chapter WHERE id=1"
                       ).fetchone()[0] == "detected"
    assert con.execute("SELECT type, chapter_id FROM job").fetchone() == \
        ("prepare", 1)
