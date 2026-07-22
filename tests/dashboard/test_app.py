"""Dashboard routes: every page renders; actions only insert rows."""
import json
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
                 "/discovery", "/health", "/runs"):
        r = c.get(path)
        assert r.status_code == 200, path
    assert "Nano Machine" in c.get("/series").text


def _pass_qa(con, cid=1):
    """A chapter that has actually passed QA. /approve now ENFORCES the render
    gate rather than only hiding the button, so a render-approval test must
    set up a chapter that is genuinely approvable."""
    con.execute("INSERT INTO stage_run (chapter_id, stage, ok) "
                "VALUES (?, 'qa_scan', 1)", (cid,))
    con.commit()


def _insert_finished_job(con, jid=77, state="done"):
    con.execute(
        "INSERT INTO job (id, type, series_id, chapter_id, state, priority, "
        "created_at, started_at, finished_at, log_path) VALUES "
        "(?, 'prepare', 1, 1, ?, 1, '2026-07-16 01:00:00', "
        "'2026-07-16 01:00:00', '2026-07-16 01:12:30', 'logs/jobs/77.log')",
        (jid, state))
    con.commit()


def test_runs_page_lists_finished_job_with_duration(client):
    c, con = client
    _insert_finished_job(con)
    text = c.get("/runs").text
    assert "77" in text and "DONE" in text
    assert "Nano Machine" in text          # scope name resolved
    assert "/partials/log/77" in text      # log button survives completion
    assert "12:2" in text or "12:30" in text  # ~12m30s duration rendered (fmt_eta mm:ss)


def test_queue_partial_keeps_finished_jobs_visible(client):
    c, con = client
    _insert_finished_job(con, jid=78, state="failed")
    text = c.get("/partials/queue").text
    assert "finished" in text and "FAILED" in text
    assert "/partials/log/78" in text


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
    _pass_qa(con)
    c.post("/approve", data={"gate": "render", "chapter_id": 1},
           follow_redirects=False)
    assert con.execute("SELECT COUNT(*) FROM approval WHERE gate='render'"
                       ).fetchone()[0] == 1


def test_approve_endpoint_stores_content_sha(client, tmp_path):
    """The /approve endpoint binds the row to the CONTENT it approved, not
    just a checkbox: it must stamp content_sha from the files on disk."""
    c, con = client
    from studio.dashboard import gates
    ep = tmp_path / "ep"
    (ep / "tts").mkdir(parents=True)
    (ep / "render.plan.clean.json").write_text('{"cuts": []}')
    (ep / "tts" / "tts_index.json").write_text('{"clips": []}')
    con.execute("UPDATE chapter SET ep_dir=? WHERE id=1", (str(ep),))
    con.commit()
    _pass_qa(con)
    c.post("/approve", data={"gate": "render", "chapter_id": 1},
           follow_redirects=False)
    stored = con.execute("SELECT content_sha FROM approval WHERE gate='render' "
                         "AND chapter_id=1").fetchone()[0]
    assert stored is not None
    assert stored == gates.gate_sha("render", ep)


def test_post_revoice_restamps_stale_voice_approval(client, tmp_path):
    """post_revoice must treat an existing-but-STALE voice approval (the
    narration was healed/rewritten after it was approved) as absent: it has
    to re-stamp the gate with the CURRENT sha before enqueueing, or the
    voiceover job it just queued blocks inside the worker's sha-validating
    gate check — a manufactured dead-letter."""
    c, con = client
    from studio.dashboard import gates
    ep = tmp_path / "ep"
    ep.mkdir()
    script = ep / "manifest.script.json"
    script.write_text('{"paragraphs": ["original"]}')
    con.execute("UPDATE chapter SET ep_dir=?, status='voiced' WHERE id=1",
               (str(ep),))
    gates.approve(con, "voice", chapter_id=1,
                 content_sha=gates.gate_sha("voice", ep))
    script.write_text('{"paragraphs": ["healed"]}')      # re-narrated
    con.commit()
    c.post("/chapter/1/revoice", follow_redirects=False)
    current = gates.gate_sha("voice", ep)
    assert gates._approval_valid(con, "voice", chapter_id=1,
                                 current_sha=current)
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='voiceover' "
                       "AND chapter_id=1").fetchone()[0] == 1


def test_post_revoice_does_not_fold_onto_plain_queued_voiceover(client, tmp_path):
    """post_revoice enqueues with payload {"auto_to": "video"} -- operator
    intent to drive all the way to render. jobs.enqueue's default dedupe
    folds same (type, chapter) rows together regardless of payload, which
    would silently drop that intent onto a plain queued voiceover (e.g. from
    an earlier auto-chain step). The endpoint must pass dedupe=False so its
    own row always lands."""
    c, con = client
    from studio.dashboard import gates, jobs
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "manifest.script.json").write_text('{"paragraphs": ["original"]}')
    con.execute("UPDATE chapter SET ep_dir=?, status='voiced' WHERE id=1",
               (str(ep),))
    gates.approve(con, "voice", chapter_id=1,
                 content_sha=gates.gate_sha("voice", ep))
    plain_id = jobs.enqueue(con, "voiceover", chapter_id=1)   # pre-existing plain row
    con.commit()
    c.post("/chapter/1/revoice", follow_redirects=False)
    rows = con.execute("SELECT id, payload_json FROM job WHERE "
                       "type='voiceover' AND chapter_id=1").fetchall()
    assert len(rows) == 2
    ids = {r[0] for r in rows}
    assert plain_id in ids
    payloads = {r[1] for r in rows}
    assert '{}' in payloads
    assert any("auto_to" in p for p in payloads)


def test_post_reload_does_not_fold_onto_plain_queued_prepare(client):
    """post_reload enqueues with payload {"auto_to": "video"} -- same operator
    -intent concern as revoice above, for the reload/dead-letter path."""
    c, con = client
    from studio.dashboard import jobs
    plain_id = jobs.enqueue(con, "prepare", chapter_id=1)   # pre-existing plain row
    con.commit()
    c.post("/chapter/1/reload", follow_redirects=False)
    rows = con.execute("SELECT id, payload_json FROM job WHERE "
                       "type='prepare' AND chapter_id=1").fetchall()
    assert len(rows) == 2
    ids = {r[0] for r in rows}
    assert plain_id in ids
    payloads = {r[1] for r in rows}
    assert '{}' in payloads
    assert any("auto_to" in p for p in payloads)


def test_series_thumbnail_job_and_approval_flow(client, tmp_path, monkeypatch):
    """The series page is where the one-per-manhwa thumbnail is generated and
    approved (answers 'where will I see it to approve it'). Generate enqueues a
    series_thumbnail job; approving records a series-scoped 'thumbnail' gate."""
    c, con = client
    from studio.dashboard import app as _app, gates
    monkeypatch.setattr(_app, "REPO", tmp_path)   # hermetic: no real dist/ writes
    # generate button -> a queued series_thumbnail job carrying the series id
    r = c.post("/jobs", data={"type": "series_thumbnail", "series_id": 1},
               follow_redirects=False)
    assert r.status_code == 303
    row = con.execute("SELECT type, series_id, state FROM job WHERE "
                      "type='series_thumbnail'").fetchone()
    assert row == ("series_thumbnail", 1, "queued")
    # no image on disk yet -> the file route 404s, the page shows no <img>
    assert c.get("/thumb/series/1").status_code == 404
    assert 'src="/thumb/series/1' not in c.get("/series/1").text
    # approve -> series-scoped gate row, redirect back to the series page
    r = c.post("/approve", data={"gate": "thumbnail", "series_id": 1},
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/series/1"
    assert gates.thumbnail_approved(con, 1) is True


def test_series_thumbnail_served_and_shown_when_present(client, tmp_path, monkeypatch):
    """Once dist/series_<id>/thumbnail_yt.jpg exists, the file route serves it
    and the series page embeds it with a cache-busting version."""
    c, _ = client
    from studio.dashboard import app as _app
    monkeypatch.setattr(_app, "REPO", tmp_path)
    tdir = tmp_path / "dist" / "series_1"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "thumbnail_yt.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # tiny jpeg
    rr = c.get("/thumb/series/1")
    assert rr.status_code == 200 and "image/jpeg" in rr.headers["content-type"]
    assert 'src="/thumb/series/1?v=' in c.get("/series/1").text


def test_bundle_create_from_a_range(client):
    """The new-video form is range-based (from/to episode), not kind+title.
    Creating a video bundles the unbundled chapters in the range, auto-enqueues
    the title/description job, and — as the FIRST video for the series — the
    intro teaser."""
    c, con = client
    con.execute("UPDATE chapter SET ep_dir='/tmp/ep1', status='rendered' "
                "WHERE id=1")  # a rendered candidate
    con.commit()
    r = c.post("/bundles", data={"series_id": 1, "num_from": 1, "num_to": 1},
               follow_redirects=False)
    assert r.status_code == 303
    from studio.dashboard import bundles
    bid = con.execute("SELECT id FROM bundle").fetchone()[0]
    assert bundles.bundle_chapters(con, bid) == [1]
    types = {row[0] for row in con.execute("SELECT type FROM job")}
    assert "publish_meta" in types            # auto title/description
    assert "plan_teaser" in types             # debut video -> intro teaser
    # the provisional "Episodes …" title shows until publish_meta runs
    assert "Episodes" in c.get("/videos").text


def test_bundle_create_excludes_already_bundled_chapters(client):
    """A continuation batch must not re-offer produced chapters."""
    c, con = client
    con.execute("UPDATE chapter SET ep_dir='/tmp/ep1', status='rendered' "
                "WHERE id=1")
    con.commit()
    c.post("/bundles", data={"series_id": 1, "num_from": 1, "num_to": 1},
           follow_redirects=False)
    from studio.dashboard import bundles
    assert bundles.unbundled_chapters(con, 1) == []      # ch1 now produced
    # a second create over the same range finds nothing and makes no bundle
    r = c.post("/bundles", data={"series_id": 1, "num_from": 1, "num_to": 1},
               follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT COUNT(*) FROM bundle").fetchone()[0] == 1


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
    _pass_qa(con)
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
    # a locked GET now REDIRECTS to the login form rather than dead-ending on
    # a bare 401 — but it must still never serve the page itself
    r = c.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    assert "dashboard token" in c.get("/").text       # lands on the form
    assert "form" in c.get("/login").text
    # a POST with no cookie is still a hard 401 (no redirect dance)
    assert c.post("/jobs/1/cancel", follow_redirects=False).status_code == 401
    c.post("/login", data={"token": "wrong"}, follow_redirects=False)
    assert c.get("/", follow_redirects=False).status_code == 303
    c.post("/login", data={"token": "sekret"}, follow_redirects=False)
    assert c.get("/").status_code == 200              # cookie set (POST only)


def test_locked_htmx_request_gets_hx_redirect_not_a_silent_freeze(
        client, monkeypatch, tmp_path):
    """THE frozen-dashboard bug: htmx does not swap a non-2xx response, so a
    bare 401 made the 2s queue poller fail silently and the page kept showing
    state that could be hours old while looking perfectly alive."""
    from studio.catalog.db import connect as _c
    from studio.dashboard.app import create_app
    from fastapi.testclient import TestClient
    monkeypatch.setenv("STUDIO_DASH_TOKEN", "sekret")
    db = tmp_path / "t.db"
    _c(db)
    c = TestClient(create_app(db_path=str(db)))
    r = c.get("/partials/queue", headers={"HX-Request": "true"},
              follow_redirects=False)
    assert r.headers.get("HX-Redirect") == "/login"


def test_login_next_cannot_be_an_open_redirect(client, monkeypatch, tmp_path):
    from studio.catalog.db import connect as _c
    from studio.dashboard.app import create_app
    from fastapi.testclient import TestClient
    monkeypatch.setenv("STUDIO_DASH_TOKEN", "sekret")
    db = tmp_path / "t.db"
    _c(db)
    c = TestClient(create_app(db_path=str(db)))
    for hostile in ("//evil.example", "https://evil.example", "http://x"):
        r = c.post("/login", data={"token": "sekret", "next": hostile},
                   follow_redirects=False)
        assert r.headers["location"] == "/", hostile


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


def test_refresh_runs_sync_and_clears_new_pending(client, monkeypatch):
    """The manual refresh button must re-discover synchronously and clear the
    NEW alert (the old async /jobs refresh updated nothing in the view)."""
    import subprocess
    c, con = client
    con.execute("UPDATE series SET new_pending=3 WHERE id=1")
    con.commit()
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = c.post("/series/1/refresh", follow_redirects=False)
    assert r.status_code == 303
    assert "refresh" in calls["cmd"] and "--series" in calls["cmd"]   # ran sync
    assert con.execute("SELECT new_pending FROM series WHERE id=1"
                       ).fetchone()[0] == 0                            # alert cleared


def test_rebuild_route_resets_enqueues_and_clears_stale_audio(client, tmp_path):
    """Rebuild RE-NARRATES, so gen-1 audio + approvals are stale. It must demote to
    'detected', queue a fresh prepare, AND clear voice/render approvals + cached
    clips/index/preview — else render ships gen-1 audio under gen-2 captions."""
    c, con = client
    ep = tmp_path / "ongoing" / "nano" / "ch1"
    (ep / "tts" / "clips").mkdir(parents=True)
    (ep / "render").mkdir(parents=True)
    (ep / "tts" / "clips" / "g0001_p00.wav").write_bytes(b"old")
    (ep / "tts" / "tts_index.json").write_text("{}")
    (ep / "render" / "voice_preview.mp3").write_bytes(b"gen1")
    con.execute("UPDATE chapter SET ep_dir=? WHERE id=1", (str(ep),))
    con.execute("INSERT INTO approval (gate, chapter_id, note) VALUES ('voice',1,'x')")
    con.execute("INSERT INTO approval (gate, chapter_id, note) VALUES ('render',1,'x')")
    con.commit()
    r = c.post("/chapter/1/rebuild", follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT status FROM chapter WHERE id=1").fetchone()[0] == "detected"
    assert con.execute("SELECT type FROM job").fetchone()[0] == "prepare"
    assert con.execute("SELECT COUNT(*) FROM approval WHERE chapter_id=1").fetchone()[0] == 0
    assert not (ep / "render" / "voice_preview.mp3").exists()
    assert not (ep / "tts" / "tts_index.json").exists()
    assert not list((ep / "tts" / "clips").glob("*.wav"))


def test_qa_report_link_is_cache_busted(client, tmp_path):
    """prep_qa.html is a static file — without a version param browsers
    show stale reports after rebuilds (user hit this on job 25)."""
    c, con = client
    ep = tmp_path / "ongoing" / "nano" / "ch1"
    ep.mkdir(parents=True)
    (ep / "prep_qa.html").write_text("<html>report</html>")
    con.execute("UPDATE chapter SET ep_dir=? WHERE id=1", (str(ep),))
    con.commit()
    import studio.dashboard.app as app_mod
    old = app_mod.REPO
    app_mod.REPO = tmp_path
    try:
        html = c.get("/chapter/1").text
    finally:
        app_mod.REPO = old
    assert "prep_qa.html?v=" in html


def test_narration_style_selector(client):
    c, con = client
    r = c.post("/series/1/style", data={"style": "light"},
               follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT narration_style FROM series WHERE id=1"
                       ).fetchone()[0] == "light"
    assert 'value="light" selected' in c.get("/series/1").text
    r = c.post("/series/1/style", data={"style": "evil"},
               follow_redirects=False)
    assert r.status_code == 400          # only off|light|full|default


def test_drop_panel_button_bans_file_and_requeues(client, tmp_path):
    """Operator contract: see a bad visual -> click X -> panel banned in
    manual_drops.json -> prepare auto-queued."""
    import json as _json
    c, con = client
    ep = tmp_path / "ep"
    ep.mkdir()
    con.execute("UPDATE chapter SET ep_dir=? WHERE id=1", (str(ep),))
    con.commit()
    r = c.post("/chapter/1/drop", data={"file": "p000031.jpg"},
               follow_redirects=False)
    assert r.status_code == 303
    assert _json.loads((ep / "manual_drops.json").read_text()) == \
        ["p000031.jpg"]
    c.post("/chapter/1/drop", data={"file": "p000031.jpg"},
           follow_redirects=False)        # idempotent
    assert _json.loads((ep / "manual_drops.json").read_text()) == \
        ["p000031.jpg"]
    # enqueue dedupe: the second click folds onto the still-queued prepare
    # from the first click instead of piling up a duplicate job.
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='prepare'"
                       ).fetchone()[0] == 1


def test_drop_then_render_blocked_until_reprepared(client, tmp_path):
    """Operator drops a bad panel on an already-rendered+approved chapter: the
    drop writes manual_drops.json + enqueues a re-prepare (existing behavior,
    see test_drop_panel_button_bans_file_and_requeues above). Once the
    re-prepare rewrites the clean plan (simulated here the SAME way
    test_post_revoice_restamps_stale_voice_approval fakes a pipeline rewrite —
    no real subprocess chain runs), the render gate's content_sha binding
    must catch the drift: a direct render attempt — the exact gate check
    worker._h_render_segment makes before touching any tool — is blocked
    until the chapter is re-approved against the NEW plan."""
    import io
    import json as _json
    from studio.dashboard import gates
    from studio import worker

    c, con = client
    ep = tmp_path / "ep"
    (ep / "tts").mkdir(parents=True)
    (ep / "render.plan.clean.json").write_text('{"cuts": ["a"]}')
    (ep / "tts" / "tts_index.json").write_text('{"clips": ["a"]}')
    con.execute("UPDATE chapter SET ep_dir=?, status='rendered' WHERE id=1",
               (str(ep),))
    con.execute("INSERT INTO stage_run (chapter_id, stage, ok, duration_sec) "
                "VALUES (1,'qa_scan',1,10)")
    con.commit()
    c.post("/approve", data={"gate": "render", "chapter_id": 1},
           follow_redirects=False)
    assert gates.render_allowed(con, 1, str(ep))[0] is True   # baseline: clean

    r = c.post("/chapter/1/drop", data={"file": "p000031.jpg"},
               follow_redirects=False)
    assert r.status_code == 303
    assert _json.loads((ep / "manual_drops.json").read_text()) == \
        ["p000031.jpg"]
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='prepare' AND "
                       "chapter_id=1").fetchone()[0] == 1

    # Simulate the re-prepare completing: it drops the panel and rewrites
    # render.plan.clean.json with different bytes.
    (ep / "render.plan.clean.json").write_text('{"cuts": []}')

    allowed, why = gates.render_allowed(con, 1, str(ep))
    assert allowed is False and "approval" in why

    with pytest.raises(RuntimeError, match="approval"):
        worker._h_render_segment(con, {"chapter_id": 1}, io.StringIO())


def test_chapter_page_gallery_shows_flow_span_grouped(client, tmp_path,
                                                      monkeypatch):
    """Adaptive flow narration: the chapter page's narration gallery renders
    ONE row per SEGMENT — a flow span's panels grouped under its single line,
    a solo on its own row — not one merged row per beat."""
    import json
    c, con = client
    from studio.dashboard import app as _app
    monkeypatch.setattr(_app, "REPO", tmp_path)
    ep = tmp_path / "ongoing" / "nano" / "ch1"
    ep.mkdir(parents=True)
    flow = "He drops through the canopy and lands hard."
    solo = "The system window blinks awake."
    (ep / "manifest.beats.json").write_text(json.dumps({"beats": [{
        "group_id": 4,
        "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"],
        "narration": flow + " " + solo,
        "segments": [
            {"span": ["p1.jpg", "p2.jpg", "p3.jpg"], "line": flow},
            {"span": ["p4.jpg"], "line": solo},
        ]}]}))
    con.execute("UPDATE chapter SET ep_dir=? WHERE id=1", (str(ep),))
    con.commit()

    html = c.get("/chapter/1").text
    assert flow in html and solo in html
    # NOT one merged per-beat row: the joined narration never renders as one blob
    assert (flow + " " + solo) not in html

    rows = _app._gallery(str(ep))
    assert [[f["name"] for f in r["files"]] for r in rows] == [
        ["p1.jpg", "p2.jpg", "p3.jpg"], ["p4.jpg"]]
    assert [r["narration"] for r in rows] == [flow, solo]


def test_chapter_page_gallery_legacy_manifest_one_row_per_panel(client,
                                                                tmp_path,
                                                                monkeypatch):
    """Legacy panel_narration manifests render one row per panel (singleton
    spans) — same reader, no dual path."""
    import json
    c, con = client
    from studio.dashboard import app as _app
    monkeypatch.setattr(_app, "REPO", tmp_path)
    ep = tmp_path / "ongoing" / "nano" / "ch1"
    ep.mkdir(parents=True)
    (ep / "manifest.beats.json").write_text(json.dumps({"beats": [{
        "group_id": 2,
        "scene_files": ["a.jpg", "b.jpg"],
        "narration": "First. Second.",
        "panel_narration": [
            {"scene_file": "a.jpg", "line": "First."},
            {"scene_file": "b.jpg", "line": "Second."},
        ]}]}))
    con.execute("UPDATE chapter SET ep_dir=? WHERE id=1", (str(ep),))
    con.commit()

    assert c.get("/chapter/1").status_code == 200
    rows = _app._gallery(str(ep))
    assert [[f["name"] for f in r["files"]] for r in rows] == [
        ["a.jpg"], ["b.jpg"]]
    assert [r["narration"] for r in rows] == ["First.", "Second."]


def test_gallery_collapses_ken_variety_subcuts(tmp_path):
    """3 same-file sub-cuts (ken motion passes) = ONE thumb with n=3 — never
    three identical thumbnails masquerading as the duplicate-panel bug."""
    import json
    from studio.dashboard import app as _app
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "render.plan.clean.json").write_text(json.dumps({"timeline": [{
        "segment_id": "g0011_p09", "duration_sec": 21.1, "tts_text": "line",
        "cuts": [
            {"file": "p31.jpg", "dur": 7.0, "ken_variety": True},
            {"file": "p31.jpg", "dur": 7.0, "ken_variety": True},
            {"file": "p31.jpg", "dur": 7.1, "ken_variety": True},
        ]}]}))
    rows = _app._gallery(str(ep))
    assert rows[0]["files"] == [{"name": "p31.jpg", "n": 3}]


# --- Phase B: progress display told the truth about nothing ------------------

def test_timeline_stages_are_all_reachable(client):
    """Every TIMELINE entry must be satisfiable — either a catalog status or a
    stage something writes to stage_run. 'fetched' was neither, so step 1 was
    red on every chapter forever."""
    from studio.catalog.models import STATUS_RANK
    from studio.dashboard.app import TIMELINE
    # stages written to stage_run by the worker (grep record_stage / INSERT)
    written = {"prepped", "qa_scan", "render_segment", "voiced", "planned",
               "concat", "plan_teaser", "series_thumbnail"}
    for s in TIMELINE:
        assert s in STATUS_RANK or s in written, f"{s} is never satisfied"


def test_downloaded_chapter_shows_first_step_done(client):
    c, con = client
    con.execute("UPDATE chapter SET status='downloaded' WHERE id=1")
    con.commit()
    from studio.dashboard.app import _stage_timeline
    ch = dict(zip(("id", "series_id", "status", "ep_dir"),
                  con.execute("SELECT id, series_id, status, ep_dir FROM "
                              "chapter WHERE id=1").fetchone()))
    rows = {r["stage"]: r["done"] for r in _stage_timeline(con, ch)}
    assert rows["downloaded"] is True
    assert rows["stitched"] is False


def test_rendered_chapter_shows_pipeline_steps_done(client):
    """'rendered' sits ABOVE 'planned' but is not in STATUS_ORDER — it used to
    rank as unknown (0 = 'discovered'), so a finished chapter displayed as
    barely started."""
    c, con = client
    con.execute("UPDATE chapter SET status='rendered' WHERE id=1")
    con.commit()
    from studio.dashboard.app import _stage_timeline
    ch = dict(zip(("id", "series_id", "status", "ep_dir"),
                  con.execute("SELECT id, series_id, status, ep_dir FROM "
                              "chapter WHERE id=1").fetchone()))
    rows = {r["stage"]: r["done"] for r in _stage_timeline(con, ch)}
    assert rows["planned"] is True and rows["visioned"] is True


def test_failed_status_does_not_read_as_progress(client):
    c, con = client
    con.execute("UPDATE chapter SET status='voiced_failed' WHERE id=1")
    con.commit()
    from studio.dashboard.app import _stage_timeline
    ch = dict(zip(("id", "series_id", "status", "ep_dir"),
                  con.execute("SELECT id, series_id, status, ep_dir FROM "
                              "chapter WHERE id=1").fetchone()))
    rows = {r["stage"]: r["done"] for r in _stage_timeline(con, ch)}
    assert not any(rows.values())


def test_chapter_badges_render_as_html_not_literal_text(client):
    """chapter.html used typographic quotes in class=/title= attributes, which
    silently disabled the STALE MANIFESTS + QA badges."""
    c, _ = client
    body = c.get("/chapter/1").text
    assert 'class="b warn"' in body or 'class="b ok"' in body
    assert "class=”" not in body


# --- Phase E: cross-machine ---------------------------------------------------

def test_media_mount_exists_even_without_ongoing_dir(tmp_path, monkeypatch):
    """Templates emit /media/... unconditionally, but the mount used to be
    added only if ongoing/ already existed at startup — so on a fresh host
    every scene image 404'd with no explanation."""
    from studio.dashboard import app as appmod
    from studio.catalog.db import connect as _c
    from fastapi.testclient import TestClient
    monkeypatch.setattr(appmod, "REPO", tmp_path)
    db = tmp_path / "t.db"
    _c(db)
    assert not (tmp_path / "ongoing").exists()
    c = TestClient(appmod.create_app(db_path=str(db)))
    # mounted -> a MISSING file is a 404 from StaticFiles, not a routing 404
    assert (tmp_path / "ongoing").is_dir()
    assert (tmp_path / "dist").is_dir()
    routes = {getattr(r, "path", "") for r in c.app.routes}
    assert "/media" in routes and "/dist" in routes


def test_chapter_page_survives_an_ep_dir_outside_the_tree(client):
    """ep_dir is stored absolute from whichever machine fetched it, so any DB
    copied between the Air and the Mini has out-of-tree paths. That used to
    raise from relative_to and 500 the whole page."""
    c, con = client
    con.execute("UPDATE chapter SET ep_dir='/tmp/definitely-not-here' "
                "WHERE id=1")
    con.commit()
    assert c.get("/chapter/1").status_code == 200


def test_served_rel_returns_none_outside_tree():
    from studio.dashboard.app import _served_rel, REPO
    assert _served_rel(None) is None
    assert _served_rel("/tmp/nope") is None
    assert _served_rel(REPO / "ongoing" / "S" / "Ep1") == "S/Ep1"


def test_health_probes_the_ollama_host_the_worker_uses(client, monkeypatch):
    """Hardcoded localhost:11434 while the worker runs against the MLX shim on
    :11500 — the health page reported '(not running)' for a working pipeline."""
    c, _ = client
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11500")
    body = c.get("/health").text
    assert "11500" in body


def test_htmx_is_vendored_not_cdn():
    from studio.dashboard.app import HERE
    base = (HERE / "templates" / "base.html").read_text()
    assert "/static/htmx.min.js" in base
    assert "unpkg.com" not in base
    assert (HERE / "static" / "htmx.min.js").is_file()


def test_safe_next_rejects_every_off_site_destination():
    """An unvalidated post-login `next` is a textbook open redirect: the
    operator authenticates normally and the dashboard itself bounces them
    somewhere hostile with the credibility of a trusted origin. Rejecting a
    leading '//' alone is NOT enough — browsers normalise backslashes in the
    authority position, so '/\\evil' escapes a naive prefix check."""
    from studio.dashboard.app import _safe_next
    hostile = [
        "//evil.example", "https://evil.example", "http://evil.example",
        "/\\evil.example", "/\\/evil.example", "\\\\evil.example",
        "javascript:alert(1)", "  //evil.example", "/x\r\nLocation: /y",
        "", None,
    ]
    for h in hostile:
        assert _safe_next(h) == "/", h
    for ok in ("/", "/chapter/1", "/series/2?x=1", "/partials/queue"):
        assert _safe_next(ok) == ok, ok


def test_approve_render_is_refused_when_qa_is_not_green(client):
    """The gate is ENFORCED, not merely hidden in the template. Approving with
    red QA used to insert an approval row and enqueue a job that failed a
    minute later inside the worker's own gate check — a manufactured
    dead-letter plus an approval record for something never approvable."""
    c, con = client
    con.execute("INSERT INTO stage_run (chapter_id, stage, ok) "
                "VALUES (1, 'qa_scan', 0)")          # QA FAILED
    con.commit()
    c.post("/approve", data={"gate": "render", "chapter_id": 1},
           follow_redirects=False)
    assert con.execute("SELECT COUNT(*) FROM approval WHERE gate='render'"
                       ).fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='render_segment'"
                       ).fetchone()[0] == 0


def test_range_estimate_and_enqueue_agree(client):
    """They ran two different queries — the estimate omitted the
    already-queued filter — so the estimate promised more work than the button
    performed."""
    c, con = client
    from studio.dashboard import jobs
    for n in (2, 3, 4):
        con.execute("INSERT INTO chapter (series_id, number, label, url, "
                    "status, updated_at) VALUES (1,?,?,'u','downloaded','t')",
                    (n, f"Chapter {n}"))
    con.commit()
    ids = jobs.range_chapter_ids(con, 1, target="qa")
    c.post("/jobs", data={"type": "prepare_range", "series_id": 1,
                          "target": "qa"}, follow_redirects=False)
    enqueued = [r[0] for r in con.execute(
        "SELECT chapter_id FROM job WHERE type='prepare' ORDER BY chapter_id")]
    assert sorted(ids) == sorted(enqueued)
    # and a second click enqueues NOTHING new (already queued)
    assert jobs.range_chapter_ids(con, 1, target="qa") == []


def test_range_bounds_are_optional_not_sentinels(client):
    """0.0/1e12 sentinels were indistinguishable from a real selection on a
    series that HAS a chapter 0."""
    c, con = client
    from studio.dashboard import jobs
    con.execute("INSERT INTO chapter (series_id, number, label, url, status, "
                "updated_at) VALUES (1,0,'Episode 0','u','downloaded','t')")
    con.commit()
    assert len(jobs.range_chapter_ids(con, 1)) == 2            # both
    assert len(jobs.range_chapter_ids(con, 1, num_from=1)) == 1  # excl. ch 0
    assert len(jobs.range_chapter_ids(con, 1, num_to=0)) == 1    # only ch 0
    # reversed bounds are swapped, not silently empty
    assert jobs.range_chapter_ids(con, 1, num_from=1, num_to=0) == \
        jobs.range_chapter_ids(con, 1, num_from=0, num_to=1)


def test_both_range_selects_render_identical_options(client):
    """One select appended the chapter label, the other did not, so the same
    chapter read differently depending on which dropdown you looked at."""
    c, con = client
    body = c.get("/series/1").text
    import re
    selects = re.findall(r'<select name="num_(?:from|to)".*?</select>', body,
                         re.S)
    assert len(selects) == 2
    opts = [re.findall(r'<option value="([^"]*)"[^>]*>([^<]*)</option>', s)
            for s in selects]
    assert opts[0] == opts[1]


def test_chapter_page_shouts_when_the_episode_directory_is_gone(client):
    """'never downloaded' and 'the images are GONE' rendered identically —
    both just an empty gallery. The second is an emergency."""
    c, con = client
    con.execute("UPDATE chapter SET ep_dir='/tmp/gone-forever', "
                "status='planned' WHERE id=1")
    con.commit()
    body = c.get("/chapter/1").text
    assert "GONE" in body and "/tmp/gone-forever" in body


def test_gallery_honours_the_plans_scenes_subdir(client, tmp_path):
    from studio.dashboard.app import _gallery
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "render.plan.clean.json").write_text(json.dumps({
        "scenes_subdir": "scenes",
        "timeline": [{"segment_id": "g0001_p01", "tts_text": "x",
                      "cuts": [{"file": "p1.jpg"}], "duration_sec": 3}]}))
    assert _gallery(str(ep))[0]["src_dir"] == "scenes"


# --- Phase I: a stuck job must LOOK stuck -------------------------------------

def _stuck(con, jid=90, cid=1):
    """A job whose lease is old and whose process group is confirmed dead."""
    con.execute("INSERT INTO job (id, type, chapter_id, series_id, state, "
                "pgid, started_at) VALUES (?, 'prepare', ?, 1, 'running', "
                "999999, datetime('now','-3 hours'))", (jid, cid))
    con.commit()
    return jid


def test_queue_shows_a_stuck_job_as_stuck_with_a_requeue_button(client):
    """THE highest-value UI change: a job whose child died looked identical to
    a healthy long-running one, so the only cure anyone knew was restarting
    the worker."""
    c, con = client
    jid = _stuck(con)
    body = c.get("/partials/queue").text
    assert "STUCK" in body
    assert f"/jobs/{jid}/requeue" in body


def test_cancelling_job_is_visible_in_the_queue_html(client):
    """Groups were derived as 'not running and not queued', which silently
    swallowed cancelling — pressing Cancel made the row vanish."""
    c, con = client
    from studio.dashboard import jobs
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    jobs.claim_next(con, lane="gpu")
    jobs.cancel(con, jid)
    body = c.get("/partials/queue").text
    assert "CANCELLING" in body and str(jid) in body


def test_health_strip_reports_worker_down_and_recovers(client):
    c, con = client
    assert "DOWN" in c.get("/partials/health").text
    con.execute("INSERT INTO job (type, state, started_at) VALUES "
                "('heartbeat','running',datetime('now'))")
    con.commit()
    body = c.get("/partials/health").text
    assert "alive" in body and "nothing needs you" in body


def test_health_strip_counts_stuck_jobs(client):
    c, con = client
    _stuck(con)
    assert "1 STUCK" in c.get("/partials/health").text


def test_chapter_page_and_queue_agree_that_a_job_is_stuck(client):
    """Two pages computing 'is it stuck' separately is how they end up
    disagreeing — both read the same jobs.health()."""
    c, con = client
    _stuck(con)
    assert "STUCK" in c.get("/partials/queue").text
    assert "STUCK" in c.get("/chapter/1").text


def test_log_pane_follows_the_running_job(client, tmp_path):
    """The pane said '(no job selected)' until you clicked something, so a
    busy machine showed no evidence it was doing anything."""
    c, con = client
    log = tmp_path / "j.log"
    log.write_text("understanding panel 12/40\n")
    con.execute("INSERT INTO job (id, type, chapter_id, state, started_at, "
                "log_path) VALUES (91,'prepare',1,'running',datetime('now'),?)",
                (str(log),))
    con.commit()
    body = c.get("/partials/log/active").text
    assert "understanding panel 12/40" in body and "#91" in body


def test_log_active_is_calm_when_nothing_runs(client):
    c, _ = client
    assert "nothing running" in c.get("/partials/log/active").text


def test_bundle_duration_partial_and_zero_based_range(client, tmp_path):
    """The new-video duration estimate honours chapter 0 (a 0-based series must
    be able to select episode 0) and reports the projected final runtime."""
    c, con = client
    ep0, ep1 = tmp_path / "e0", tmp_path / "e1"
    for d, dur in ((ep0, 100.0), (ep1, 200.0)):
        d.mkdir()
        (d / "render.plan.clean.json").write_text(
            '{"total_duration_sec": %f}' % dur)
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES "
                "(50,1,0,'Episode 0','u','rendered',?,'t')", (str(ep0),))
    con.execute("UPDATE chapter SET status='rendered', ep_dir=? WHERE id=1",
                (str(ep1),))
    con.commit()
    # episode 0 is a real, selectable bound
    body = c.get("/partials/bundle-form?series_id=1").text
    assert 'value="0"' in body
    # from 0 to 1 = both chapters + teaser (first video) + outro
    d = c.get("/partials/bundle-duration?series_id=1&num_from=0&num_to=1").text
    assert ">2</b> chapters" in d and "intro teaser" in d
    # from 1 to 1 = only the later chapter, no episode 0
    d1 = c.get("/partials/bundle-duration?series_id=1&num_from=1&num_to=1").text
    assert ">1</b> chapters" in d1


def test_continuation_video_has_no_intro_teaser(client, tmp_path):
    """Only the FIRST video for a series carries the intro teaser."""
    c, con = client
    for cid, num, d in ((1, 1, "a"), (60, 2, "b")):
        ep = tmp_path / d
        ep.mkdir()
        con.execute("UPDATE chapter SET ep_dir=?, status='rendered' WHERE "
                    "id=?", (str(ep), cid)) \
            if cid == 1 else con.execute(
            "INSERT INTO chapter (id, series_id, number, label, url, status, "
            "ep_dir, updated_at) VALUES (?,1,?,'C','u','rendered',?,'t')",
            (cid, num, str(ep)))
    con.commit()
    c.post("/bundles", data={"series_id": 1, "num_from": 1, "num_to": 1},
           follow_redirects=False)                      # debut
    con.execute("DELETE FROM job")                       # clear debut's jobs
    con.commit()
    c.post("/bundles", data={"series_id": 1, "num_from": 2, "num_to": 2},
           follow_redirects=False)                      # continuation
    types = {r[0] for r in con.execute("SELECT type FROM job")}
    assert "publish_meta" in types                       # still auto-titled
    assert "plan_teaser" not in types                    # but NO second intro


def test_delete_bundle_frees_its_chapters_without_touching_them(client, tmp_path):
    """A stale/mis-ranged video must be deletable — its episodes go back into
    the pool (so they can be re-bundled), the chapter data is untouched, and
    the bundle's own dist artifacts are removed. Before this the only delete
    path was whole-series deletion, so a leftover 'pack' locked its chapters
    out of the video creator forever."""
    c, con = client
    ep = tmp_path / "ch"
    ep.mkdir()
    con.execute("UPDATE chapter SET ep_dir=?, status='rendered' WHERE id=1",
                (str(ep),))
    con.commit()
    c.post("/bundles", data={"series_id": 1, "num_from": 1, "num_to": 1},
           follow_redirects=False)
    from studio.dashboard import bundles
    bid = con.execute("SELECT id FROM bundle").fetchone()[0]
    assert bundles.unbundled_chapters(con, 1) == []       # chapter is locked in
    c.post(f"/bundles/{bid}/delete", follow_redirects=False)
    assert con.execute("SELECT COUNT(*) FROM bundle").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM bundle_chapter WHERE bundle_id=?",
                       (bid,)).fetchone()[0] == 0
    # the chapter itself is untouched and available again
    assert con.execute("SELECT id FROM chapter WHERE id=1").fetchone() is not None
    assert [ch["id"] for ch in bundles.unbundled_chapters(con, 1)] == [1]


def test_video_creator_offers_only_rendered_episodes(client, tmp_path):
    """A video is a concat of RENDERED segments, so a downloaded-but-not-
    rendered chapter must NOT be offered — the dropdown showed 163 unrendered
    chapters before this fix."""
    c, con = client
    # ch1: downloaded but only 'planned' (not rendered) -> NOT a candidate
    con.execute("UPDATE chapter SET ep_dir='/tmp/ep1', status='planned' "
                "WHERE id=1")
    # ch70: actually rendered -> the only candidate
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES "
                "(70,1,2,'Chapter 2','u','rendered','/tmp/ep2','t')")
    con.commit()
    from studio.dashboard import bundles
    nums = [ch["number"] for ch in bundles.unbundled_chapters(con, 1)]
    assert nums == [2.0]                      # only the rendered one
    body = c.get("/partials/bundle-form?series_id=1").text
    import re
    opts = re.findall(r'<option value="([^"]*)"', body)
    assert opts and "1" not in opts and "2" in opts   # ch1 (planned) excluded


def test_video_creator_empty_when_nothing_rendered(client):
    """With no rendered episodes the dropdown is empty and says so — not a list
    of unrendered chapters."""
    c, con = client
    con.execute("UPDATE chapter SET ep_dir='/tmp/x', status='planned' "
                "WHERE id=1")
    con.commit()
    body = c.get("/partials/bundle-form?series_id=1").text
    assert "no rendered episodes" in body
