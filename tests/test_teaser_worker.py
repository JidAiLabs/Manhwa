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
import types
from pathlib import Path

from studio.catalog.db import connect
from studio.catalog import repo

FIXED_NOW = "2026-06-28T00:00:00+00:00"


def _fake_cfg(**overrides):
    """A cfg double covering every attribute _h_teaser/_autostart_intro_if_ready
    read — Task 13 wiring widened that surface (spoiler/cost-guard params +
    narration_sanitize + teaser_enabled), so tests that pin down one field
    still need the rest present. Callers override only what they care about."""
    base = dict(
        beats_backend="vertex", beats_model="gemini-2.5-flash",
        teaser_model="gemini-2.5-flash", teaser_max_hook_scan_chapters=12,
        teaser_shortlist_n=4, teaser_min_panels=4, teaser_max_hook_panels=10,
        teaser_payoff_tail_frac=0.2, teaser_max_seconds=90,
        teaser_enabled=True, narration_sanitize=False,
        script_model="gpt-5-nano", tts_backend="chatterbox", tts_voice_ref="",
        tts_kokoro_voice="", tts_python="", tts_speed=1.0)
    base.update(overrides)
    return types.SimpleNamespace(**base)


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


# ---------------------------------------------------------------------------
# Task 11: _h_concat prepends the APPROVED teaser
# ---------------------------------------------------------------------------

def test_h_concat_prepends_teaser_when_approved(tmp_path, monkeypatch):
    import studio.worker as w
    from studio.dashboard import gates

    con = connect(tmp_path / "s.db")
    _sid, bid, _cids = _bundle(con, tmp_path, n=2)
    monkeypatch.setattr(w, "REPO", tmp_path)
    con.execute("UPDATE bundle SET teaser_state='approved' WHERE id=?", (bid,))
    con.commit()
    gates.approve(con, "concat", bundle_id=bid)     # else the gate raises first
    teaser_mp4 = tmp_path / "dist" / f"bundle_{bid}" / "teaser.mp4"
    teaser_mp4.parent.mkdir(parents=True, exist_ok=True)
    teaser_mp4.write_bytes(b"\x00")

    captured: dict = {}
    monkeypatch.setattr(w, "_stream", lambda argv, log, **k: 0)
    monkeypatch.setattr(
        w.bundles, "concat_cmd",
        lambda segs, out: (captured.update(segs=list(segs)),
                           (["ffmpeg", "LISTFILE"], ""))[1])

    w._h_concat(con, {"bundle_id": bid}, io.StringIO())

    assert captured["segs"][0].endswith("teaser.mp4")       # prepended first
    assert len(captured["segs"]) == 3                       # teaser + 2 chapters


def test_h_concat_no_teaser_when_declined(tmp_path, monkeypatch):
    """A declined teaser is NOT prepended even if teaser.mp4 is on disk."""
    import studio.worker as w
    from studio.dashboard import gates

    con = connect(tmp_path / "s.db")
    _sid, bid, _cids = _bundle(con, tmp_path, n=2)
    monkeypatch.setattr(w, "REPO", tmp_path)
    con.execute("UPDATE bundle SET teaser_state='declined' WHERE id=?", (bid,))
    con.commit()
    gates.approve(con, "concat", bundle_id=bid)
    teaser_mp4 = tmp_path / "dist" / f"bundle_{bid}" / "teaser.mp4"
    teaser_mp4.parent.mkdir(parents=True, exist_ok=True)
    teaser_mp4.write_bytes(b"\x00")

    captured: dict = {}
    monkeypatch.setattr(w, "_stream", lambda argv, log, **k: 0)
    monkeypatch.setattr(
        w.bundles, "concat_cmd",
        lambda segs, out: (captured.update(segs=list(segs)),
                           (["ffmpeg", "LISTFILE"], ""))[1])

    w._h_concat(con, {"bundle_id": bid}, io.StringIO())

    assert not captured["segs"][0].endswith("teaser.mp4")
    assert len(captured["segs"]) == 2                       # chapters only


# ---------------------------------------------------------------------------
# Task 13: config wiring + backend dispatch + sanitize gate
# ---------------------------------------------------------------------------

def test_h_teaser_passes_spoiler_and_cost_guard_params(tmp_path, monkeypatch):
    """teaser_min_panels/max_hook_panels/payoff_tail_frac/max_seconds must
    reach the planner's argv. The planner's own --payoff-tail-frac argparse
    default is 0.0 (guard OFF) — if this wiring is missing, the configured
    0.20 spoiler guard silently never engages in production."""
    import studio.worker as w

    con = connect(tmp_path / "s.db")
    _sid, bid, _cids = _bundle(con, tmp_path, n=2)
    monkeypatch.setattr(w, "REPO", tmp_path)
    monkeypatch.setattr(w, "_beats_cfg", lambda: (
        _fake_cfg(teaser_min_panels=6, teaser_max_hook_panels=9,
                  teaser_payoff_tail_frac=0.33, teaser_max_seconds=45),
        "proj", "loc"))
    stream_calls: list = []
    monkeypatch.setattr(w, "_stream", lambda argv, log, **k:
                        stream_calls.append([str(a) for a in argv]) or 0)

    w._h_teaser(con, {"bundle_id": bid, "payload": {}}, io.StringIO())

    argv = next(c for c in stream_calls if any("teaser_planner.py" in a for a in c))
    assert argv[argv.index("--min-panels") + 1] == "6"
    assert argv[argv.index("--max-hook-panels") + 1] == "9"
    assert argv[argv.index("--payoff-tail-frac") + 1] == "0.33"
    assert argv[argv.index("--max-seconds") + 1] == "45"


def _stream_with_teaser_manifest(out_dir):
    """A fake_stream that satisfies the planner-writes-a-manifest gate and the
    remotion-writes-a-segment gate, so _h_teaser runs past both early exits."""
    def fake_stream(argv, log, **k):
        sargv = [str(a) for a in argv]
        if any("teaser_planner.py" in a for a in sargv):
            od = Path(sargv[sargv.index("--out-dir") + 1])
            od.mkdir(parents=True, exist_ok=True)
            (od / "manifest.teaser.json").write_text("{}")
        if "remotion" in sargv:
            (out_dir / "render").mkdir(parents=True, exist_ok=True)
            (out_dir / "render" / "segment_none.mp4").write_bytes(b"\x00")
        return 0
    return fake_stream


def test_h_teaser_elevenlabs_backend_routes_to_elevenlabs_tool(
        tmp_path, monkeypatch):
    """[tts].backend='elevenlabs' must route to elevenlabs_tts_from_manifest.py
    (mirrors pipeline._stage_voiced's dispatch) — local_tts_from_manifest's
    argparse doesn't accept 'elevenlabs' as a --backend choice, so the old
    unconditional local-CLI call exited 2 whenever this backend was configured."""
    import studio.worker as w
    import studio.pipeline as pl

    con = connect(tmp_path / "s.db")
    _sid, bid, _cids = _bundle(con, tmp_path, n=2)
    monkeypatch.setattr(w, "REPO", tmp_path)
    monkeypatch.setattr(w, "_beats_cfg", lambda: (
        _fake_cfg(tts_backend="elevenlabs"), "proj", "loc"))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-123")
    out_dir = tmp_path / "dist" / f"bundle_{bid}" / "teaser"
    monkeypatch.setattr(w, "_stream", _stream_with_teaser_manifest(out_dir))
    tool_calls: list = []
    monkeypatch.setattr(
        pl, "_run_tool",
        lambda script, args, **k: tool_calls.append((script, list(args))))

    w._h_teaser(con, {"bundle_id": bid, "payload": {}}, io.StringIO())

    names = [s for s, _ in tool_calls]
    assert "elevenlabs_tts_from_manifest.py" in names
    assert "local_tts_from_manifest.py" not in names
    ev_args = next(a for s, a in tool_calls
                  if s == "elevenlabs_tts_from_manifest.py")
    assert ev_args[ev_args.index("--voice-id") + 1] == "voice-123"


def test_h_teaser_local_backend_includes_speed(tmp_path, monkeypatch):
    """A non-1.0 cfg.tts_speed must reach local_tts_from_manifest's argv —
    the old call omitted --speed unconditionally, so teaser narration always
    voiced at 1.0x regardless of the chapter TTS tempo."""
    import studio.worker as w
    import studio.pipeline as pl

    con = connect(tmp_path / "s.db")
    _sid, bid, _cids = _bundle(con, tmp_path, n=2)
    monkeypatch.setattr(w, "REPO", tmp_path)
    monkeypatch.setattr(w, "_beats_cfg", lambda: (
        _fake_cfg(tts_backend="chatterbox", tts_speed=1.15), "proj", "loc"))
    out_dir = tmp_path / "dist" / f"bundle_{bid}" / "teaser"
    monkeypatch.setattr(w, "_stream", _stream_with_teaser_manifest(out_dir))
    tool_calls: list = []
    monkeypatch.setattr(
        pl, "_run_tool",
        lambda script, args, **k: tool_calls.append((script, list(args))))

    w._h_teaser(con, {"bundle_id": bid, "payload": {}}, io.StringIO())

    local_args = next(a for s, a in tool_calls if s == "local_tts_from_manifest.py")
    assert local_args[local_args.index("--speed") + 1] == "1.15"


def test_h_teaser_sanitize_before_tts_unresolved_fails(tmp_path, monkeypatch):
    """narration_sanitize_pass must run on the teaser script BEFORE any TTS
    tool call, and unresolved advertiser-safety blocks FAIL the job outright
    (never voice unsanitized) — the teaser previously skipped the sanitizer
    entirely, so a hard BLOCK line could reach TTS unfiltered."""
    import json
    import pytest
    import studio.worker as w
    import studio.pipeline as pl

    con = connect(tmp_path / "s.db")
    _sid, bid, _cids = _bundle(con, tmp_path, n=2)
    monkeypatch.setattr(w, "REPO", tmp_path)
    monkeypatch.setattr(w, "_beats_cfg", lambda: (
        _fake_cfg(narration_sanitize=True), "proj", "loc"))

    def fake_stream(argv, log, **k):
        sargv = [str(a) for a in argv]
        if any("teaser_planner.py" in a for a in sargv):
            od = Path(sargv[sargv.index("--out-dir") + 1])
            od.mkdir(parents=True, exist_ok=True)
            (od / "manifest.teaser.json").write_text("{}")
        if any("narration_sanitize_pass.py" in a for a in sargv):
            marker = Path(sargv[sargv.index("--marker") + 1])
            marker.write_text(json.dumps({"unresolved_blocks": [
                {"segment_id": "g0001_p01", "matched": "slur"}]}))
            return 2                      # tool contract: 2 = unresolved recorded
        return 0

    monkeypatch.setattr(w, "_stream", fake_stream)
    tool_calls: list = []
    monkeypatch.setattr(
        pl, "_run_tool", lambda script, args, **k: tool_calls.append(script))

    with pytest.raises(RuntimeError, match="unresolved"):
        w._h_teaser(con, {"bundle_id": bid, "payload": {}}, io.StringIO())

    # script_expander ran (sanitize needs its output) but NO TTS/timeline
    # tool ever fired -> the block landed strictly before voicing.
    assert tool_calls == ["script_expander.py"]


def test_autostart_intro_skipped_when_teaser_disabled(tmp_path, monkeypatch):
    """teaser_enabled=False gates ONLY the auto-start detection path — the
    manual dashboard 'Plan teaser' button (post_teaser_plan in app.py, which
    enqueues plan_teaser directly with no cfg check at all) stays
    unconditional, per the brief's explicit requirement."""
    import studio.worker as w
    from studio.dashboard import jobs

    con = connect(tmp_path / "s.db")
    sid, bid, cids = _bundle(con, tmp_path, n=2)
    con.execute("UPDATE series SET autopilot=1 WHERE id=?", (sid,))
    con.commit()
    monkeypatch.setattr(w, "_beats_cfg", lambda: (
        _fake_cfg(teaser_enabled=False), "proj", "loc"))

    # AUTO-START: disabled -> no plan_teaser even though fully rendered +
    # autopilot is on (both other preconditions satisfied).
    w._autostart_intro_if_ready(con, cids[-1], io.StringIO())
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser'"
                       ).fetchone()[0] == 0

    # MANUAL: the button's own enqueue call is untouched by this flag.
    jobs.enqueue(con, "plan_teaser", bundle_id=bid, payload={})
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser' "
                       "AND bundle_id=?", (bid,)).fetchone()[0] == 1
