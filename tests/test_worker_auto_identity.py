"""The automatic protagonist (owner 2026-10-11: "prep first 3 chapters, then
auto-pick the protagonist, review what has been done, and then continue").
After every prepare the worker re-picks the lead of a series that has no
exemplars until two picks in a row agree (the lock), then — once the picture
profile is active — re-narrates the chapters prepared before it."""
from __future__ import annotations

import io
import json
import types
from pathlib import Path

import pytest

from studio import worker as w
from studio.catalog.db import connect
from studio.dashboard import jobs


def _series(tmp_path, n, monkeypatch, *, auto_pick=True, backend="gemma"):
    con = connect(tmp_path / "s.db")
    sd = tmp_path / "ongoing" / "s"
    con.execute("INSERT INTO series (id, source, series_url, slug, title, added_at) "
                "VALUES (1,'asura','u','s','S','t')")
    for i in range(1, n + 1):
        ep = sd / f"Chapter_{i}"
        ep.mkdir(parents=True)
        (ep / "manifest.beats.json").write_text('{"beats": []}')
        con.execute("INSERT INTO chapter (series_id, number, label, url, status, ep_dir, "
                    "updated_at) VALUES (1,?,?,'u','planned',?,'t')", (i, f"Chapter {i}", str(ep)))
    con.commit()
    monkeypatch.setattr(w, "REPO", tmp_path)
    monkeypatch.setattr(w, "_beats_cfg", lambda: types.SimpleNamespace(
        identity_auto_pick=auto_pick, identity_backend_for=lambda slug, confirmed=False: backend))
    return con, sd


def _ch(con, n):
    return w._chapter(con, con.execute("SELECT id FROM chapter WHERE number=?", (n,)).fetchone()[0])


def _locked(sd, *, by="auto", chapters=4, active=True):
    d = sd / ".identity"
    d.mkdir(parents=True, exist_ok=True)
    (d / "exemplars.json").write_text(json.dumps(
        {"confirmed": {"by": by, "chapters": chapters, "at": "2026-10-11T00:00:00+00:00"}}))
    (d / "profile.json").write_text(json.dumps({"status": "active" if active else "provisional"}))


def _no_pick(*a, **k):
    pytest.fail("no automatic pick expected")


def test_a_series_without_exemplars_is_picked_and_locked(tmp_path, monkeypatch):
    con, sd = _series(tmp_path, 4, monkeypatch)
    calls = []

    def stream(args, log, **kw):
        calls.append(args)
        out = Path(args[args.index("--lock") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"confirmed": {"by": "auto", "chapters": 4}}))
        return 0
    monkeypatch.setattr(w, "_stream", stream)
    w._auto_identity(con, _ch(con, 4), None, io.StringIO())
    args = calls[0]
    assert args[1].endswith("tools/identity_exemplars.py")
    assert args[args.index("--series") + 1] == str(sd)
    assert args[args.index("--auto") + 1] == str(sd / ".identity" / "exemplars.proposed.json")
    assert args[args.index("--lock") + 1] == str(sd / ".identity" / "exemplars.json")
    row = con.execute("SELECT series_id, payload_json FROM job WHERE type='identity_check'").fetchone()
    assert row[0] == 1 and json.loads(row[1])["series_slug"] == "s"


def test_no_lock_queues_no_check(tmp_path, monkeypatch):
    con, sd = _series(tmp_path, 4, monkeypatch)
    monkeypatch.setattr(w, "_stream", lambda args, log, **kw: 0)
    w._auto_identity(con, _ch(con, 4), None, io.StringIO())
    assert not con.execute("SELECT 1 FROM job").fetchone()


@pytest.mark.parametrize("why", ["hand-picked", "confirmed", "auto off", "backend off"])
def test_no_automatic_pick(tmp_path, monkeypatch, why):
    con, sd = _series(tmp_path, 4, monkeypatch, auto_pick=why != "auto off",
                      backend="off" if why == "backend off" else "gemma")
    if why == "hand-picked":
        (tmp_path / "cast").mkdir()
        (tmp_path / "cast" / "s.exemplars.json").write_text("{}")
    if why == "confirmed":
        _locked(sd, by=None, active=False)
    monkeypatch.setattr(w, "_stream", _no_pick)
    w._auto_identity(con, _ch(con, 4), None, io.StringIO())


@pytest.mark.parametrize("fail", [lambda *a, **k: 2, lambda *a, **k: 1 / 0])
def test_a_failed_pick_never_fails_the_prepare(tmp_path, monkeypatch, fail):
    con, sd = _series(tmp_path, 4, monkeypatch)
    monkeypatch.setattr(w, "_stream", fail)
    log = io.StringIO()
    w._auto_identity(con, _ch(con, 4), None, log)
    assert not con.execute("SELECT 1 FROM job").fetchone()


def test_chapters_prepared_before_the_lock_are_re_narrated_once_active(tmp_path, monkeypatch):
    con, sd = _series(tmp_path, 6, monkeypatch)
    _locked(sd)
    (sd / "Chapter_5" / "manifest.identity.json").write_text("{}")   # prepared with the pictures
    (sd / "Chapter_3" / ".identity_redo").write_text("")             # re-narrated once already
    monkeypatch.setattr(w, "_stream", _no_pick)
    voice = jobs.enqueue(con, "voiceover", chapter_id=_ch(con, 2)["id"])
    w._auto_identity(con, _ch(con, 6), "video", io.StringIO())
    rows = con.execute("SELECT c.number, j.payload_json FROM job j JOIN chapter c "
                       "ON c.id=j.chapter_id WHERE j.type='prepare' AND j.state='queued'").fetchall()
    # not 3 (done once), not 5 (has the pictures), not 6 (its own prepare is still running)
    assert sorted(n for n, _ in rows) == [1, 2, 4]
    assert all(json.loads(p)["auto_to"] == "video" for _, p in rows)
    assert con.execute("SELECT state FROM job WHERE id=?", (voice,)).fetchone()[0] == "cancelled"
    assert (sd / "Chapter_1" / ".identity_redo").exists()
    assert _ch(con, 1)["status"] == "grouped"
    assert not (sd / "Chapter_1" / "manifest.beats.json").exists()


def test_a_chapter_with_a_running_job_waits_for_the_next_prepare(tmp_path, monkeypatch):
    con, sd = _series(tmp_path, 3, monkeypatch)
    _locked(sd)
    monkeypatch.setattr(w, "_stream", _no_pick)
    jid = jobs.enqueue(con, "render", chapter_id=_ch(con, 1)["id"])
    con.execute("UPDATE job SET state='running' WHERE id=?", (jid,))
    con.commit()
    w._auto_identity(con, _ch(con, 3), None, io.StringIO())
    nums = [r[0] for r in con.execute("SELECT c.number FROM job j JOIN chapter c ON "
                                      "c.id=j.chapter_id WHERE j.type='prepare'")]
    assert nums == [2] and not (sd / "Chapter_1" / ".identity_redo").exists()


@pytest.mark.parametrize("lock", [dict(by=None), dict(chapters=40), dict(active=False)])
def test_no_re_narration_for_an_owner_pick_a_late_lock_or_an_inactive_profile(
        tmp_path, monkeypatch, lock):
    con, sd = _series(tmp_path, 4, monkeypatch)
    _locked(sd, **lock)
    monkeypatch.setattr(w, "_stream", _no_pick)
    w._auto_identity(con, _ch(con, 4), None, io.StringIO())
    assert not con.execute("SELECT 1 FROM job WHERE type='prepare'").fetchone()
