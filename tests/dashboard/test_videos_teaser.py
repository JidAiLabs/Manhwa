"""Teaser dashboard wiring: the Plan-teaser button enqueues a plan_teaser job,
and approve/decline set bundle.teaser_state (the concat gate reads it)."""
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
    return TestClient(create_app(db_path=str(db))), con, bid


def test_plan_teaser_enqueues_job(client):
    c, con, bid = client
    r = c.post(f"/bundles/{bid}/teaser/plan", follow_redirects=False)
    assert r.status_code == 303
    assert con.execute(
        "SELECT COUNT(*) FROM job WHERE type='plan_teaser' AND bundle_id=?",
        (bid,)).fetchone()[0] == 1


def test_decline_sets_state(client):
    c, con, bid = client
    r = c.post(f"/bundles/{bid}/teaser/decline", follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT teaser_state FROM bundle WHERE id=?",
                       (bid,)).fetchone()[0] == "declined"


def test_approve_sets_state_and_records_gate(client):
    c, con, bid = client
    r = c.post(f"/bundles/{bid}/teaser/approve", follow_redirects=False)
    assert r.status_code == 303
    assert con.execute("SELECT teaser_state FROM bundle WHERE id=?",
                       (bid,)).fetchone()[0] == "approved"
    # an explicit teaser approval is recorded so the gate can attest to it
    assert gates.teaser_allowed(con, bid)[0] is True
