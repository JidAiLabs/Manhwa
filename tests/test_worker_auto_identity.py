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


def _series(tmp_path, n, monkeypatch, *, auto_pick=True, backend="gemma", narrated=True):
    con = connect(tmp_path / "s.db")
    sd = tmp_path / "ongoing" / "s"
    con.execute("INSERT INTO series (id, source, series_url, slug, title, added_at) "
                "VALUES (1,'asura','u','s','S','t')")
    for i in range(1, n + 1):
        ep = sd / f"Chapter_{i}"
        ep.mkdir(parents=True)
        if narrated:
            (ep / "manifest.beats.json").write_text('{"beats": []}')
        con.execute("INSERT INTO chapter (series_id, number, label, url, status, ep_dir, "
                    "updated_at) VALUES (1,?,?,'u','planned',?,'t')", (i, f"Chapter {i}", str(ep)))
    con.commit()
    monkeypatch.setattr(w, "REPO", tmp_path)
    monkeypatch.setattr(w, "_beats_cfg", lambda: types.SimpleNamespace(
        identity_auto_pick=auto_pick, identity_python="/id/py",
        identity_backend_for=lambda slug, confirmed=False: backend))
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


# ---- read first, narrate after the lock (owner 2026-10-11: re-running the first
# chapters "sounds like waste of time"): a new series' chapters are only READ
# (panels + faces) until the protagonist is locked and its picture profile is
# active; then they are narrated once, with the pictures.
def _reader(sd, calls, *, lock_at=None, active=True):
    """_stream double: the ccip tool records a chapter's faces (one .npz each) and,
    given --exemplars, refreshes the profile; the lock tool locks once *lock_at*
    chapters have faces; `studio fetch/run` do nothing."""
    def stream(args, log, **kw):
        args = [str(a) for a in args]
        calls.append(args)
        tool = Path(args[1]).name
        d = sd / ".identity"
        if tool == "panel_identity_ccip.py":
            d.mkdir(parents=True, exist_ok=True)
            (d / (Path(args[args.index("--episode-dir") + 1]).name + ".npz")).write_bytes(b"")
            if "--exemplars" in args:
                (d / "profile.json").write_text(json.dumps(
                    {"status": "active" if active else "provisional"}))
        elif tool == "identity_exemplars.py" and lock_at and len(list(d.glob("*.npz"))) >= lock_at:
            Path(args[args.index("--lock") + 1]).write_text(
                json.dumps({"confirmed": {"by": "auto", "chapters": lock_at}}))
        return 0
    return stream


def _job(con, n, auto_to="video"):
    return {"id": 999, "chapter_id": _ch(con, n)["id"], "payload": {"auto_to": auto_to}}


def _queue(con, *nums):
    for n in nums:
        jobs.enqueue(con, "prepare", chapter_id=_ch(con, n)["id"], payload={"auto_to": "video"})


def _faces(sd, *nums):
    (sd / ".identity").mkdir(parents=True, exist_ok=True)
    for n in nums:
        (sd / ".identity" / f"Chapter_{n}.npz").write_bytes(b"")


def _waiting(sd, *nums):
    for n in nums:
        (sd / f"Chapter_{n}" / ".waiting_for_protagonist").write_text('{"auto_to": "video"}')


def _released(con):
    return sorted(r[0] for r in con.execute(
        "SELECT c.number FROM job j JOIN chapter c ON c.id=j.chapter_id "
        "WHERE j.type='prepare' AND j.state='queued' AND j.priority=40"))


def test_a_new_series_is_read_first_and_waits_for_the_protagonist(tmp_path, monkeypatch):
    con, sd = _series(tmp_path, 3, monkeypatch, narrated=False)
    _queue(con, 2, 3)
    calls = []
    monkeypatch.setattr(w, "_stream", _reader(sd, calls))
    assert w._read_first(con, _ch(con, 1), _job(con, 1), io.StringIO()) is True
    runs = [c for c in calls if c[1:4] == ["-m", "studio", "run"]]
    assert len(runs) == 1 and runs[0][runs[0].index("--until") + 1] == "grouped"
    ccip = [c for c in calls if c[1].endswith("tools/panel_identity_ccip.py")]
    assert ccip and "--index-only" in ccip[0] and ccip[0][0] == "/id/py"
    assert "--exemplars" not in ccip[0]
    mark = sd / "Chapter_1" / ".waiting_for_protagonist"
    assert json.loads(mark.read_text())["auto_to"] == "video"


def test_the_lock_releases_the_chapters_read_so_far(tmp_path, monkeypatch):
    con, sd = _series(tmp_path, 5, monkeypatch, narrated=False)
    _faces(sd, 1, 2, 3)
    _waiting(sd, 1, 2, 3)
    _queue(con, 5)
    calls = []
    monkeypatch.setattr(w, "_stream", _reader(sd, calls, lock_at=4))
    assert w._read_first(con, _ch(con, 4), _job(con, 4), io.StringIO()) is False   # narrate 4 now
    refresh = [c for c in calls if c[1].endswith("panel_identity_ccip.py") and "--exemplars" in c]
    assert refresh and refresh[0][refresh[0].index("--exemplars") + 1] == str(
        sd / ".identity" / "exemplars.json")
    assert _released(con) == [1, 2, 3]
    assert not list(sd.glob("*/.waiting_for_protagonist"))
    assert all(json.loads(p)["auto_to"] == "video" for (p,) in con.execute(
        "SELECT payload_json FROM job WHERE priority=40"))


def test_a_locked_series_waits_until_its_profile_is_active(tmp_path, monkeypatch):
    con, sd = _series(tmp_path, 3, monkeypatch, narrated=False)
    _locked(sd, active=False)
    _queue(con, 3)
    calls = []
    monkeypatch.setattr(w, "_stream", _reader(sd, calls, active=False))
    assert w._read_first(con, _ch(con, 2), _job(con, 2), io.StringIO()) is True
    assert any("--exemplars" in c for c in calls)


@pytest.mark.parametrize("why", ["nothing else queued", "twelve chapters read"])
def test_waiting_ends_and_the_chapters_are_narrated_the_old_way(tmp_path, monkeypatch, why):
    con, sd = _series(tmp_path, 13, monkeypatch, narrated=False)
    _waiting(sd, 1, 2)
    if why == "twelve chapters read":
        _faces(sd, *range(1, 12))
        _queue(con, 13)
    calls = []
    monkeypatch.setattr(w, "_stream", _reader(sd, calls))
    n = 3 if why == "nothing else queued" else 12
    assert w._read_first(con, _ch(con, n), _job(con, n), io.StringIO()) is False
    assert _released(con) == [1, 2]


@pytest.mark.parametrize("why", ["hand-picked", "already narrated", "auto off"])
def test_no_reading_ahead(tmp_path, monkeypatch, why):
    con, sd = _series(tmp_path, 3, monkeypatch, narrated=why == "already narrated",
                      auto_pick=why != "auto off")
    if why == "hand-picked":
        (tmp_path / "cast").mkdir()
        (tmp_path / "cast" / "s.exemplars.json").write_text("{}")
    _queue(con, 3)
    monkeypatch.setattr(w, "_stream", _no_pick)
    assert w._read_first(con, _ch(con, 1), _job(con, 1), io.StringIO()) is False


def test_a_ready_series_releases_chapters_still_waiting(tmp_path, monkeypatch):
    # e.g. the owner confirmed the protagonist on the Series page meanwhile
    con, sd = _series(tmp_path, 3, monkeypatch)
    (tmp_path / "cast").mkdir()
    (tmp_path / "cast" / "s.exemplars.json").write_text("{}")
    (sd / "Chapter_1" / "manifest.beats.json").unlink()
    _waiting(sd, 1)
    monkeypatch.setattr(w, "_stream", _no_pick)
    w._auto_identity(con, _ch(con, 3), None, io.StringIO())
    assert _released(con) == [1]


def test_prepare_stops_after_reading_a_waiting_chapter(tmp_path, monkeypatch):
    con, sd = _series(tmp_path, 2, monkeypatch, narrated=False)
    monkeypatch.setattr(w, "_read_first", lambda *a, **k: True)
    monkeypatch.setattr(w, "_stream", _no_pick)
    w._h_prepare(con, _job(con, 1), io.StringIO())


def test_a_released_chapter_is_narrated_not_held_again(tmp_path, monkeypatch):
    # released the old way (waiting ended): its own prepare must not wait again
    con, sd = _series(tmp_path, 3, monkeypatch, narrated=False)
    _queue(con, 3)
    monkeypatch.setattr(w, "_stream", _no_pick)
    job = {"id": 1, "chapter_id": _ch(con, 1)["id"], "payload": {"auto_to": "video", "narrate": True}}
    assert w._read_first(con, _ch(con, 1), job, io.StringIO()) is False
    _waiting(sd, 2)
    w._release_waiting(con, 1, io.StringIO())
    (p,) = con.execute("SELECT payload_json FROM job WHERE priority=40").fetchone()
    assert json.loads(p) == {"auto_to": "video", "narrate": True}
