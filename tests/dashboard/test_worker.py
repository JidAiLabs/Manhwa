"""Worker: claims serially, logs, enforces gates, records stage timings."""
import time

from studio.catalog.db import connect
from studio.dashboard import gates, jobs
from studio import worker


def _con(tmp_path):
    return connect(tmp_path / "s.db")


def test_run_once_executes_and_logs(tmp_path):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "stub", chapter_id=3)
    seen = {}

    def stub(c, job, log):
        log.write("hello from stub\n")
        seen["job"] = job["id"]

    assert worker.run_once(con, handlers={"stub": stub},
                           log_dir=str(tmp_path / "logs")) is True
    row = [r for r in jobs.queue_view(con) if r["id"] == jid]
    assert not row or row[0]["state"] != "running"
    done = con.execute("SELECT state, log_path FROM job WHERE id=?",
                       (jid,)).fetchone()
    assert done[0] == "done" and seen["job"] == jid
    assert "hello from stub" in open(done[1]).read()


def test_run_once_idle_returns_false(tmp_path):
    con = _con(tmp_path)
    assert worker.run_once(con, handlers={}, log_dir=str(tmp_path)) is False


def test_handler_exception_fails_job(tmp_path):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "boom", chapter_id=1)

    def boom(c, job, log):
        raise RuntimeError("kaput")

    worker.run_once(con, handlers={"boom": boom}, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "kaput" in err


def test_run_once_operator_cancel_marks_cancelled_no_retry(tmp_path):
    """A RUNNING job marked 'cancelling' (dashboard) whose subprocess the monitor
    kills -> the handler raises -> run_once records it 'cancelled', NOT failed,
    and does NOT auto-retry (an operator cancel is intentional)."""
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "boom", chapter_id=1)

    def boom(c, job, log):
        c.execute("UPDATE job SET state='cancelling' WHERE id=?", (job["id"],))
        c.commit()
        raise RuntimeError("killed by cancel monitor")   # = killed subprocess

    worker.run_once(con, handlers={"boom": boom}, log_dir=str(tmp_path))
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (jid,)).fetchone()[0] == "cancelled"
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='boom' AND "
                       "state='queued'").fetchone()[0] == 0   # no auto-retry


def test_render_segment_gate_refusal(tmp_path):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "render_segment", chapter_id=9)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "QA" in err


def test_recording_wrapper_writes_stage_run(tmp_path):
    con = _con(tmp_path)
    with worker.record_stage(con, chapter_id=4, stage="stitched",
                             series_id=2):
        time.sleep(0.01)
    row = con.execute("SELECT stage, ok, duration_sec FROM stage_run "
                      "WHERE chapter_id=4").fetchone()
    assert row[0] == "stitched" and row[1] == 1 and row[2] > 0


def test_recording_wrapper_records_failure(tmp_path):
    con = _con(tmp_path)
    try:
        with worker.record_stage(con, chapter_id=5, stage="beated"):
            raise ValueError("x")
    except ValueError:
        pass
    ok = con.execute("SELECT ok FROM stage_run WHERE chapter_id=5").fetchone()[0]
    assert ok == 0


def test_chain_past_scripted_requires_voice_approval(tmp_path):
    """run->planned crosses the voiceover line: blocked until the user
    approves the narration (gate='voice'); run->scripted is never gated."""
    con = _con(tmp_path)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','u','s','S','t')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES "
                "(1,1,1,'Ch 1','u','scripted','/tmp/x','t')")
    con.commit()
    jid = jobs.enqueue(con, "chain", chapter_id=1,
                       payload={"target": "planned"})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "narration" in err


# ---- stale-narration self-heal (mechanical heal only; prose never judged) --

def _seed_chapter(con, tmp_path, status="voiced_failed"):
    ep = tmp_path / "ep"
    ep.mkdir(exist_ok=True)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'asura','https://x','s','S','t')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES (5,1,1,'Ch 1',"
                "'https://x/1',?,?,'t')", (status, str(ep)))
    con.commit()
    return ep


def test_qa_error_codes_reads_report(tmp_path):
    import json
    ep = tmp_path
    (ep / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "narration_stale", "severity": "ERROR"},
        {"code": "flash_cut", "severity": "WARN"}]}))
    assert worker._qa_error_codes(ep) == {"narration_stale"}
    assert worker._qa_error_codes(tmp_path / "nope") == set()


def test_run_prep_and_qa_heal_aware_returns_instead_of_raising(
        tmp_path, monkeypatch):
    import json
    import pytest
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    (ep / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "missing_audio", "severity": "ERROR"}]}))   # a BLOCKING code
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw:
                        1 if any("prep_qa.py" in str(c) for c in cmd) else 0)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep)}
    log = open(tmp_path / "log.txt", "w")
    codes = worker._run_prep_and_qa(con, ch, log, heal_aware=True)
    assert codes == {"missing_audio"}
    with pytest.raises(RuntimeError):
        worker._run_prep_and_qa(con, ch, log, heal_aware=False)


def test_prepare_auto_heals_red_qa_to_green(tmp_path, monkeypatch):
    # first QA is red -> the targeted auto-heal runs -> green -> job done
    con = _con(tmp_path)
    _seed_chapter(con, tmp_path)
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: {"caption_unvoiced"})
    healed = []
    monkeypatch.setattr(worker, "_heal_to_green",
                        lambda c, ch, ep, log: healed.append(1))
    monkeypatch.setattr(worker, "_qa_error_codes", lambda ep: set())  # green now
    jid = jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "done", err
    assert healed == [1]


def test_prepare_fails_if_heal_cannot_reach_green(tmp_path, monkeypatch):
    con = _con(tmp_path)
    _seed_chapter(con, tmp_path)
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: {"montage_degenerate"})
    monkeypatch.setattr(worker, "_heal_to_green", lambda c, ch, ep, log: None)
    monkeypatch.setattr(worker, "_qa_error_codes",
                        lambda ep: {"montage_degenerate"})   # BLOCKING, still red
    jid = jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "auto-heal" in err


def test_heal_to_green_regenerates_only_flagged_then_stops(tmp_path, monkeypatch):
    # the loop runs narration_heal -> if it wrote corrections, regen those groups
    # + re-derive + re-QA; stops the cycle corrections come back empty
    import json
    import types
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep), "number": 1}
    seq = [{"3": "cover the caption"}, {}]   # cycle1 has 1 group, cycle2 none

    def fake_stream(cmd, log, **kw):
        s = " ".join(map(str, cmd))
        if "narration_heal.py" in s:
            out = cmd[cmd.index("--out") + 1]
            json.dump(seq.pop(0) if seq else {}, open(out, "w"))
        return 0
    monkeypatch.setattr(worker, "_stream", fake_stream)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (
        types.SimpleNamespace(beats_model="m", beats_backend="ollama",
                              punchup="cinematic", script_model="s"), "p", "l"))
    regen = []
    monkeypatch.setattr(worker, "_regen_flagged",
                        lambda *a, **k: regen.append(1))
    monkeypatch.setattr(worker, "_run_prep_and_qa", lambda *a, **k: set())
    worker._heal_to_green(con, ch, ep, open(tmp_path / "log.txt", "w"))
    assert regen == [1]          # exactly one heal cycle, then corrections empty


def test_heal_to_green_fast_qa_then_final_semantic(tmp_path, monkeypatch):
    import json
    import types
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    (ep / "prep_qa.json").write_text(json.dumps({"flags": []}))
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep), "number": 1}
    seq = [{"3": "cover the caption"}, {}]
    heal_cmds = []
    qa_cmds = []

    def fake_stream(cmd, log, **kw):
        s = " ".join(map(str, cmd))
        if "narration_heal.py" in s:
            heal_cmds.append(cmd)
            out = cmd[cmd.index("--out") + 1]
            json.dump(seq.pop(0) if seq else {}, open(out, "w"))
        if "prep_qa.py" in s:
            qa_cmds.append(cmd)
        return 0

    monkeypatch.setattr(worker, "_stream", fake_stream)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (
        types.SimpleNamespace(beats_model="m", beats_backend="ollama",
                              punchup="cinematic", script_model="s"), "p", "l"))
    worker._heal_to_green(con, ch, ep, open(tmp_path / "log.txt", "w"))

    assert all("--include-grounding-warn" not in c for c in heal_cmds)
    assert len(qa_cmds) == 2
    assert "--semantic" not in qa_cmds[0]
    assert "--semantic" in qa_cmds[1]


def _heal_cfg():
    import types
    return (types.SimpleNamespace(beats_model="m", beats_backend="ollama",
                                  punchup="off", script_model="s",
                                  narration_source="gemini_verbatim",
                                  semantic_heal=False), "p", "l")


def test_heal_to_green_narration_stale_only_rescripts_and_stops(
        tmp_path, monkeypatch):
    """narration_stale is NOT re-narratable: when it's the ONLY remaining ERROR
    (narration_heal returns no corrections), the heal must re-run the SCRIPTED
    stage (script_expander) + re-plan ONCE and stop — it must not loop to the
    cap re-narrating, which can never clear a staleness flag (the 2.6h bug)."""
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    (ep / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "narration_stale", "severity": "ERROR",
         "segment_id": "g0001_p00"}]}))
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep), "number": 1}

    calls = []

    def fake_stream(cmd, log, **kw):
        s = " ".join(map(str, cmd))
        calls.append(s)
        if "narration_heal.py" in s:                 # nothing is healable now
            json.dump({}, open(cmd[cmd.index("--out") + 1], "w"))
        return 0

    monkeypatch.setattr(worker, "_stream", fake_stream)
    monkeypatch.setattr(worker, "_beats_cfg", _heal_cfg)
    monkeypatch.setattr(worker, "_series_env", lambda c, sid: None)
    regen = []
    monkeypatch.setattr(worker, "_regen_flagged", lambda *a, **k: regen.append(1))
    qa = []
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda *a, **k: qa.append(1) or set())

    worker._heal_to_green(con, ch, ep, open(tmp_path / "log.txt", "w"))

    assert any("script_expander.py" in c for c in calls)     # scripted re-run
    assert any("timeline_planner.py" in c for c in calls)    # + re-plan
    assert regen == []                                       # NO re-narration
    assert sum("narration_heal.py" in c for c in calls) == 1  # not _HEAL_MAX
    assert qa == [1]                                         # one re-QA, then stop


def test_heal_to_green_stops_early_when_error_set_repeats(tmp_path, monkeypatch):
    """If a regen cycle leaves the ERROR set unchanged, the loop stops early
    instead of burning all _HEAL_MAX cycles on an identical re-narration."""
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    # a healable ERROR the (stubbed) regen never clears -> identical every cycle
    (ep / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "caption_unvoiced", "severity": "ERROR",
         "segment_id": "g0001_p00", "detail": "missing: 'HELLO'"}]}))
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep), "number": 1}

    def fake_stream(cmd, log, **kw):
        s = " ".join(map(str, cmd))
        if "narration_heal.py" in s:
            json.dump({"1": "cover the caption"},
                      open(cmd[cmd.index("--out") + 1], "w"))
        return 0

    monkeypatch.setattr(worker, "_stream", fake_stream)
    monkeypatch.setattr(worker, "_beats_cfg", _heal_cfg)
    monkeypatch.setattr(worker, "_series_env", lambda c, sid: None)
    regen = []
    monkeypatch.setattr(worker, "_regen_flagged", lambda *a, **k: regen.append(1))
    # re-QA leaves prep_qa.json unchanged -> the ERROR set repeats every cycle
    monkeypatch.setattr(worker, "_run_prep_and_qa", lambda *a, **k: set())

    worker._heal_to_green(con, ch, ep, open(tmp_path / "log.txt", "w"))

    # cycle 1 regens; cycle 2 sees the same ERROR set and stops -> 1 regen, far
    # fewer than _HEAL_MAX
    assert regen == [1]


# ---- autopilot: spotless QA advances without human clicks -------------------

def _autopilot_series(con, tmp_path, *, autopilot=1, flags=()):
    import json
    ep = _seed_chapter(con, tmp_path)
    con.execute("UPDATE series SET autopilot=? WHERE id=1", (autopilot,))
    (ep / "prep_qa.json").write_text(json.dumps(
        {"flags": [dict(f) for f in flags]}))
    con.commit()
    return ep


def test_autopilot_clean_report_advances_to_voice(tmp_path, monkeypatch):
    con = _con(tmp_path)
    _autopilot_series(con, tmp_path, flags=[
        {"code": "flash_cut", "severity": "WARN"}])   # ordinary WARN ok
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: set())
    jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    n_appr = con.execute("SELECT COUNT(*) FROM approval WHERE gate='voice' "
                         "AND chapter_id=5 AND note='autopilot'").fetchone()[0]
    n_jobs = con.execute("SELECT COUNT(*) FROM job WHERE type='voiceover' "
                         "AND chapter_id=5").fetchone()[0]
    assert (n_appr, n_jobs) == (1, 1)


def test_autopilot_blocked_by_semantic_mismatch(tmp_path, monkeypatch):
    con = _con(tmp_path)
    _autopilot_series(con, tmp_path, flags=[
        {"code": "narration_mismatch", "severity": "WARN"}])
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: set())
    jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    assert con.execute("SELECT COUNT(*) FROM approval WHERE chapter_id=5"
                       ).fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='voiceover'"
                       ).fetchone()[0] == 0


def test_autopilot_off_changes_nothing(tmp_path, monkeypatch):
    con = _con(tmp_path)
    _autopilot_series(con, tmp_path, autopilot=0, flags=[])
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: set())
    jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    assert con.execute("SELECT COUNT(*) FROM approval").fetchone()[0] == 0


def test_autopilot_voiceover_advances_to_render(tmp_path, monkeypatch):
    con = _con(tmp_path)
    _autopilot_series(con, tmp_path, flags=[])
    from studio.dashboard import gates as g
    g.approve(con, "voice", chapter_id=5, note="autopilot")
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: set())
    jobs.enqueue(con, "voiceover", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    state, err = con.execute(
        "SELECT state, error FROM job WHERE type='voiceover'").fetchone()
    assert state == "done", err
    n_appr = con.execute("SELECT COUNT(*) FROM approval WHERE gate='render' "
                         "AND chapter_id=5 AND note='autopilot'").fetchone()[0]
    n_jobs = con.execute("SELECT COUNT(*) FROM job WHERE "
                         "type='render_segment'").fetchone()[0]
    assert (n_appr, n_jobs) == (1, 1)


# --- last-resort visual heal must BLOCK when it can't actually drop the panel ---

def test_heal_visual_drops_blocks_when_over_cap(tmp_path, monkeypatch):
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    # 4 blank_crop ERRORs, n_cuts=4 -> cap=max(3,int(0.25*4))=3 -> 4 > cap
    (ep / "prep_qa.json").write_text(json.dumps({"n_cuts": 4, "flags": [
        {"code": "blank_crop", "severity": "ERROR", "scene": f"p{i}.jpg"}
        for i in range(4)]}))
    called = []
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda *a, **k: called.append(1) or set())
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep)}
    stuck = worker._heal_visual_drops(con, ch, ep, open(tmp_path / "l.txt", "w"))
    assert stuck == {"blank_crop"}        # over cap -> panels remain -> block
    assert called == []                   # never even attempted the drop


def test_heal_visual_drops_blocks_on_noop_drop(tmp_path, monkeypatch):
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    (ep / "prep_qa.json").write_text(json.dumps({"n_cuts": 10, "flags": [
        {"code": "blank_crop", "severity": "ERROR", "scene": "a.jpg"}]}))
    (ep / "manual_drops.json").write_text(json.dumps(["a.jpg"]))  # already dropped
    monkeypatch.setattr(worker, "_run_prep_and_qa", lambda *a, **k: set())
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep)}
    stuck = worker._heal_visual_drops(con, ch, ep, open(tmp_path / "l.txt", "w"))
    assert stuck == {"blank_crop"}        # drop was a no-op (sole cut) -> block


def test_heal_visual_drops_stays_green_when_drop_succeeds(tmp_path, monkeypatch):
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    (ep / "prep_qa.json").write_text(json.dumps({"n_cuts": 10, "flags": [
        {"code": "blank_crop", "severity": "ERROR", "scene": "b.jpg"}]}))

    def fake_reprep(c, ch, log, **kw):    # the re-prep removes the dropped panel
        (ep / "prep_qa.json").write_text(json.dumps({"n_cuts": 9, "flags": []}))
        return set()
    monkeypatch.setattr(worker, "_run_prep_and_qa", fake_reprep)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep)}
    stuck = worker._heal_visual_drops(con, ch, ep, open(tmp_path / "l.txt", "w"))
    assert stuck == set()                 # drop succeeded -> nothing to block
    assert json.loads((ep / "manual_drops.json").read_text()) == ["b.jpg"]
