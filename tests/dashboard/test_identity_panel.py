"""Series page: the character-identity exemplars (cast/<slug>.exemplars.json),
the validation result (dist/identity_check/<slug>.json) and its grading sheet —
so the owner sees exactly which panels gemma compares against."""
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import studio.dashboard.app as dash
from studio.catalog.db import connect

PANELS = ["ongoing/nano/Chapter_1/scenes/p1.jpg", "ongoing/nano/Chapter_2/scenes/p2.jpg",
          "ongoing/nano/Chapter_3/scenes/p3.jpg", "ongoing/nano/Chapter_4/scenes/p4.jpg"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "REPO", tmp_path)
    db = tmp_path / "s.db"
    con = connect(db)
    con.execute("INSERT INTO series (source, series_url, slug, title, added_at) "
                "VALUES ('asura','https://x/nano','nano','Nano Machine', datetime('now'))")
    con.commit()
    sid = con.execute("SELECT id FROM series").fetchone()[0]
    return TestClient(dash.create_app(db_path=str(db))), sid, tmp_path


def _setup(root, check=True):
    for p in PANELS:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(root / p)
    (root / "cast").mkdir(exist_ok=True)
    (root / "cast" / "nano.exemplars.json").write_text(json.dumps(
        {"protagonist": PANELS[:2], "decoy": PANELS[2:]}))
    if check:
        d = root / "dist" / "identity_check"
        d.mkdir(parents=True)
        (d / "nano.json").write_text(json.dumps({"n": 40, "confirmed": 31, "candidates": 900}))
        Image.new("RGB", (8, 8)).save(d / "nano_sheet.jpg")


def test_series_page_shows_the_exemplars_and_the_check(client):
    c, sid, root = client
    _setup(root)
    html = c.get(f"/series/{sid}").text
    assert "Character identity" in html
    for i in range(4):
        assert f"/identity/series/{sid}/exemplar/{i}" in html
    assert "31 of 40" in html and f"/identity/series/{sid}/check_sheet" in html
    assert "Chapter_1" in html and "look-alike" in html
    r = c.get(f"/identity/series/{sid}/exemplar/0")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert c.get(f"/identity/series/{sid}/exemplar/4").status_code == 404
    assert c.get(f"/identity/series/{sid}/check_sheet").status_code == 200


def test_a_series_without_exemplars_says_so(client):
    c, sid, root = client
    html = c.get(f"/series/{sid}").text
    assert "Character identity" in html and "no exemplars" in html
    assert c.get(f"/identity/series/{sid}/exemplar/0").status_code == 404
    assert c.get(f"/identity/series/{sid}/check_sheet").status_code == 404


def test_an_exemplar_outside_ongoing_is_never_served(client):
    c, sid, root = client
    _setup(root, check=False)
    (root / "secret.jpg").write_bytes(b"x")
    (root / "cast" / "nano.exemplars.json").write_text(json.dumps(
        {"protagonist": ["secret.jpg", PANELS[1]], "decoy": PANELS[2:]}))
    assert c.get(f"/identity/series/{sid}/exemplar/0").status_code == 404
    assert c.get(f"/identity/series/{sid}/exemplar/1").status_code == 200
