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


MORE = [f"ongoing/nano/Chapter_{c}/scenes/q{c}.jpg" for c in range(5, 11)]


def _proposal(root, chosen=0):
    for p in PANELS + MORE:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(root / p)
    h = lambda p: {"path": p, "chapter": p.split("/")[2], "panel": p.split("/")[-1],
                   "box": [0, 0, 8, 8]}
    cands = [{"protagonist": [h(p) for p in PANELS[:2] + MORE[:2]],
              "decoy": [h(p) for p in PANELS[2:]]},
             {"protagonist": [h(p) for p in MORE[2:4]], "decoy": [h(p) for p in MORE[4:6]]}]
    obj = {"seed": "exemplars", "candidates": cands,
           "auto": {"chosen": chosen, "asked_gemma": True, "answers": ["A", "B"]}}
    if chosen is not None:
        obj["protagonist"] = PANELS[:2]
        obj["decoy"] = PANELS[2:]
    d = root / "ongoing" / "nano" / ".identity"
    d.mkdir(parents=True, exist_ok=True)
    (d / "exemplars.proposed.json").write_text(json.dumps(obj))
    return d


def test_the_proposal_is_shown_and_confirming_it_switches_the_series_on(client):
    c, sid, root = client
    d = _proposal(root)
    html = c.get(f"/series/{sid}").text
    assert "Automatic proposal" in html and "gemma's choice" in html
    assert f"/identity/series/{sid}/proposed/1/decoy/1" in html
    assert c.get(f"/identity/series/{sid}/proposed/0/protagonist/3").status_code == 200
    assert c.get(f"/identity/series/{sid}/proposed/0/protagonist/4").status_code == 404
    assert c.get(f"/identity/series/{sid}/proposed/0/x/0").status_code == 404
    # swap one protagonist panel: ticks 0 and 3
    r = c.post(f"/identity/series/{sid}/confirm",
               data={"cand": "0", "prot": ["0", "3"], "decoy": ["0", "1"]},
               follow_redirects=False)
    assert r.status_code == 303
    ex = json.loads((d / "exemplars.json").read_text())
    assert ex["protagonist"] == [PANELS[0], MORE[1]] and ex["decoy"] == PANELS[2:]
    assert ex["seed"] == "exemplars" and ex["confirmed"]["as_proposed"] is False
    assert not (d / "exemplars.proposed.json").exists()
    con = connect(root / "s.db")
    assert con.execute("SELECT type, state FROM job").fetchall() == [("identity_check", "queued")]
    html = c.get(f"/series/{sid}").text
    assert "backend: <b>ccip</b>" in html and "confirmed on this page" in html
    assert f"/identity/series/{sid}/exemplar/3" in html


def test_confirm_needs_exactly_two_and_two(client):
    c, sid, root = client
    d = _proposal(root)
    for data in ({"cand": "0", "prot": ["0"], "decoy": ["0", "1"]},
                 {"cand": "0", "prot": ["0", "9"], "decoy": ["0", "1"]},
                 {"cand": "0", "prot": ["0", "1", "2"], "decoy": ["0", "1"]}):
        assert c.post(f"/identity/series/{sid}/confirm", data=data).status_code == 400
    assert c.post(f"/identity/series/{sid}/confirm",
                  data={"cand": "5", "prot": ["0", "1"], "decoy": ["0", "1"]}).status_code == 404
    assert not (d / "exemplars.json").exists()


def test_an_undecided_proposal_asks_the_person_and_discard_removes_it(client):
    c, sid, root = client
    d = _proposal(root, chosen=None)
    html = c.get(f"/series/{sid}").text
    assert "could not decide" in html and "Use person B" in html
    c.post(f"/identity/series/{sid}/proposal/discard")
    assert not (d / "exemplars.proposed.json").exists()


def test_propose_queues_the_gpu_job_once(client):
    c, sid, root = client
    c.post(f"/identity/series/{sid}/propose")
    c.post(f"/identity/series/{sid}/propose")
    con = connect(root / "s.db")
    rows = con.execute("SELECT type, payload_json FROM job").fetchall()
    assert len(rows) == 1 and rows[0][0] == "identity_propose" and "nano" in rows[0][1]
    assert "automatic proposal queued" in c.get(f"/series/{sid}").text
