"""tests/test_worker_advance.py

_advance_after_prepare decides what a GREEN prepare advances to. Both bugs it
has shipped stranded a chapter silently, so each branch is pinned here.

The one that bit ORV Ep0 (job 178): when the narration is UNCHANGED, the voice
approval stays valid, and both voice branches key on an INVALID one — while the
render branch lives in the voiceover handler, which never runs because there is
nothing to re-voice. A re-prepare that rebuilds the plan then leaves the shipped
mp4 stale with nothing able to refresh it.
"""
from __future__ import annotations

import io
import sqlite3

import pytest

import studio.worker as w
from studio.dashboard import gates


def _con(autopilot: int = 1) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE series (id INTEGER PRIMARY KEY, autopilot INTEGER);
        CREATE TABLE approval (id INTEGER PRIMARY KEY, gate TEXT, series_id INT,
            chapter_id INT, bundle_id INT, created_at TEXT, note TEXT,
            content_sha TEXT);
        CREATE TABLE job (id INTEGER PRIMARY KEY, type TEXT, series_id INT,
            chapter_id INT, bundle_id INT, payload_json TEXT, state TEXT
            DEFAULT 'queued', priority INT DEFAULT 100, created_at TEXT,
            started_at TEXT, finished_at TEXT, log_path TEXT, error TEXT,
            pgid INT);
    """)
    con.execute("INSERT INTO series (id, autopilot) VALUES (1, ?)", (autopilot,))
    con.commit()
    return con


_CH = {"id": 7, "series_id": 1, "ep_dir": "/ep"}
_GREEN = None  # built per-test (QAVerdict is frozen-ish, cheap to rebuild)


def _verdict() -> "w.QAVerdict":
    return w.QAVerdict(ok=True, blocking=set(), codes=set(),
                       report={"flags": []}, reason="")


def _shas(monkeypatch, *, voice: str, render: str) -> None:
    monkeypatch.setattr(gates, "gate_sha",
                        lambda gate, ep: {"voice": voice, "render": render}[gate])


def _queued(con) -> list:
    return [r[0] for r in con.execute("SELECT type FROM job ORDER BY id")]


def test_unapproved_narration_queues_the_voiceover(monkeypatch):
    con = _con()
    _shas(monkeypatch, voice="V1", render="R1")
    w._advance_after_prepare(con, _CH, _verdict(), None, io.StringIO())
    assert _queued(con) == ["voiceover"]
    assert gates.voice_allowed(con, 7, "/ep")[0] is True


def test_unchanged_narration_with_a_rebuilt_plan_queues_the_render(monkeypatch):
    """The ORV Ep0 fall-through: voice approval still valid, plan sha moved."""
    con = _con()
    _shas(monkeypatch, voice="V1", render="R2")
    gates.approve(con, "voice", chapter_id=7, content_sha="V1")   # still valid
    gates.approve(con, "render", chapter_id=7, content_sha="R1")  # now stale
    w._advance_after_prepare(con, _CH, _verdict(), None, io.StringIO())
    assert _queued(con) == ["render_segment"]
    # the fresh approval binds to the CURRENT plan, not the rendered one
    row = con.execute("SELECT content_sha FROM approval WHERE gate='render' "
                      "ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == "R2"


def test_nothing_queued_when_the_shipped_video_is_current(monkeypatch):
    con = _con()
    _shas(monkeypatch, voice="V1", render="R1")
    gates.approve(con, "voice", chapter_id=7, content_sha="V1")
    gates.approve(con, "render", chapter_id=7, content_sha="R1")
    w._advance_after_prepare(con, _CH, _verdict(), None, io.StringIO())
    assert _queued(con) == []


def test_never_renders_a_chapter_that_was_never_voiced(monkeypatch):
    """gate_sha('render') is None without BOTH a plan and a tts_index."""
    con = _con()
    monkeypatch.setattr(gates, "gate_sha",
                        lambda gate, ep: "V1" if gate == "voice" else None)
    gates.approve(con, "voice", chapter_id=7, content_sha="V1")
    w._advance_after_prepare(con, _CH, _verdict(), None, io.StringIO())
    assert _queued(con) == []


def test_autopilot_off_advances_nothing(monkeypatch):
    con = _con(autopilot=0)
    _shas(monkeypatch, voice="V1", render="R2")
    gates.approve(con, "voice", chapter_id=7, content_sha="V1")
    w._advance_after_prepare(con, _CH, _verdict(), None, io.StringIO())
    assert _queued(con) == []


@pytest.mark.parametrize("auto_to", ["voice", "video"])
def test_bulk_request_is_the_story_approval(monkeypatch, auto_to):
    con = _con(autopilot=0)          # bulk advances even with autopilot off
    _shas(monkeypatch, voice="V1", render="R1")
    w._advance_after_prepare(con, _CH, _verdict(), auto_to, io.StringIO())
    assert _queued(con) == ["voiceover"]
