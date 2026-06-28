"""Worker-side teaser: _h_teaser plans + renders the synthetic teaser episode
(the chapter render TOOL chain runs on it — NOT the chapter-keyed handlers),
and _h_concat prepends the approved teaser.mp4 to the bundle concat.

Everything that would shell out (the planner, render_prep, remotion via
worker._stream; script_expander/local_tts/timeline_planner via
pipeline._run_tool) is monkeypatched — no real subprocess runs. worker.REPO is
pointed at tmp_path so dist/ writes stay hermetic.
"""

from __future__ import annotations

import io
from pathlib import Path

from studio.catalog.db import connect
from studio.catalog import repo

FIXED_NOW = "2026-06-28T00:00:00+00:00"


def _bundle(con, tmp_path, n=2):
    """A series + n rendered chapters + a manual bundle linking them."""
    sid = repo.upsert_series(con, "test", "https://x.test/s", "t-series", "T",
                             added_at=FIXED_NOW)
    cids = []
    for i in range(1, n + 1):
        ep = tmp_path / f"ch{i}"
        (ep / "render").mkdir(parents=True)
        (ep / "render" / "segment_none.mp4").write_bytes(b"\x00")
        cid = repo.upsert_chapter(con, sid, float(i), f"Ch {i}",
                                  f"https://x.test/c{i}", updated_at=FIXED_NOW)
        repo.set_chapter_status(con, cid, "rendered", ep_dir=str(ep),
                                updated_at=FIXED_NOW)
        cids.append(cid)
    con.execute("INSERT INTO bundle (series_id, kind) VALUES (?, 'manual')", (sid,))
    bid = con.execute("SELECT id FROM bundle").fetchone()[0]
    for pos, cid in enumerate(cids):
        con.execute("INSERT INTO bundle_chapter (bundle_id, chapter_id, position) "
                    "VALUES (?,?,?)", (bid, cid, pos))
    con.commit()
    return sid, bid, cids


# ---------------------------------------------------------------------------
# Task 10: plan_teaser lane + _h_teaser handler
# ---------------------------------------------------------------------------

def test_plan_teaser_is_a_claimable_lane():
    """A handler that isn't in LANES queues forever — guard the regression."""
    from studio.dashboard import jobs
    assert "plan_teaser" in jobs.LANES


def test_h_teaser_plans_and_sets_state(tmp_path, monkeypatch):
    import studio.worker as w
    import studio.pipeline as pl

    con = connect(tmp_path / "s.db")
    _sid, bid, _cids = _bundle(con, tmp_path, n=2)
    monkeypatch.setattr(w, "REPO", tmp_path)        # hermetic dist/
    out_dir = tmp_path / "dist" / f"bundle_{bid}" / "teaser"

    stream_calls: list = []
    tool_calls: list = []

    def fake_stream(argv, log, **k):
        sargv = [str(a) for a in argv]
        stream_calls.append(sargv)
        # planner: write the teaser manifest the next step gates on
        if any("teaser_planner.py" in a for a in sargv):
            od = Path(sargv[sargv.index("--out-dir") + 1])
            od.mkdir(parents=True, exist_ok=True)
            (od / "manifest.teaser.json").write_text("{}")
        # remotion: write the rendered segment the copy step gates on
        if "remotion" in sargv:
            (out_dir / "render").mkdir(parents=True, exist_ok=True)
            (out_dir / "render" / "segment_none.mp4").write_bytes(b"\x00")
        return 0

    monkeypatch.setattr(w, "_stream", fake_stream)
    monkeypatch.setattr(pl, "_run_tool",
                        lambda script, args, **k: tool_calls.append(script) or None)

    w._h_teaser(con, {"bundle_id": bid, "payload": {}}, io.StringIO())

    assert con.execute("SELECT teaser_state FROM bundle WHERE id=?",
                       (bid,)).fetchone()[0] == "planned"
    assert (tmp_path / "dist" / f"bundle_{bid}" / "teaser.mp4").exists()
    # the chapter render tool chain ran on the synthetic teaser dir
    assert {"script_expander.py", "local_tts_from_manifest.py",
            "timeline_planner.py"} <= set(tool_calls)
    # the planner was invoked via the worker subprocess layer
    assert any("teaser_planner.py" in a for c in stream_calls for a in c)
    # a plan_teaser stage_run was recorded (chapter_id NULL, bundle-scoped)
    assert con.execute("SELECT COUNT(*) FROM stage_run WHERE stage='plan_teaser'"
                       ).fetchone()[0] == 1


def test_h_teaser_no_teaser_leaves_state_none(tmp_path, monkeypatch):
    """Planner selects no window (writes no manifest.teaser.json) -> the render
    chain is skipped, teaser_state stays 'none', and concat stays unblocked."""
    import studio.worker as w
    import studio.pipeline as pl

    con = connect(tmp_path / "s.db")
    _sid, bid, _cids = _bundle(con, tmp_path, n=2)
    monkeypatch.setattr(w, "REPO", tmp_path)
    # planner "succeeds" but writes nothing -> no-teaser
    monkeypatch.setattr(w, "_stream", lambda argv, log, **k: 0)
    ran: list = []
    monkeypatch.setattr(pl, "_run_tool",
                        lambda script, args, **k: ran.append(script))

    w._h_teaser(con, {"bundle_id": bid, "payload": {}}, io.StringIO())

    assert con.execute("SELECT teaser_state FROM bundle WHERE id=?",
                       (bid,)).fetchone()[0] == "none"
    assert ran == []                                  # render chain not run
    assert not (tmp_path / "dist" / f"bundle_{bid}" / "teaser.mp4").exists()


def test_plan_teaser_registered_in_handlers():
    import studio.worker as w
    assert w.HANDLERS.get("plan_teaser") is w._h_teaser
