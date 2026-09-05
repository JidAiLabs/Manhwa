"""Worker: claims serially, logs, enforces gates, records stage timings."""
import time
import types

from studio.catalog.db import connect
from studio.dashboard import gates, jobs
from studio import worker


def _con(tmp_path):
    return connect(tmp_path / "s.db")


def _fake_cfg(**overrides):
    """Cfg double for _h_teaser's config surface (mirrors the factory of the
    same name in tests/test_teaser_worker.py). Callers override only what
    they care about."""
    base = dict(
        beats_backend="ollama", beats_model="gemma", teaser_model="gemma",
        teaser_max_hook_scan_chapters=3, teaser_shortlist_n=4,
        teaser_min_panels=4, teaser_max_hook_panels=10,
        teaser_payoff_tail_frac=0.2, teaser_max_seconds=90,
        narration_sanitize=False,
        script_model="gpt", tts_backend="kokoro", tts_voice_ref=None,
        tts_kokoro_voice=None, tts_python=None)
    base.update(overrides)
    return types.SimpleNamespace(**base)


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


def test_transient_failure_is_auto_retried(tmp_path):
    """A TRANSIENT failure (network blip / exit 124 / timeout) raises a plain
    RuntimeError -> the worker re-enqueues the job so an unattended run recovers."""
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "boom", chapter_id=1)

    def boom(c, job, log):
        raise RuntimeError("ffmpeg exited 124 (timeout)")   # transient

    worker.run_once(con, handlers={"boom": boom}, log_dir=str(tmp_path))
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (jid,)).fetchone()[0] == "failed"
    # a retry was re-enqueued (auto-retry path)
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='boom' AND "
                       "state='queued'").fetchone()[0] == 1


def test_deterministic_qa_block_is_not_retried(tmp_path):
    """A DETERMINISTIC QA-BLOCKING failure (NonRetryableError) parks the chapter:
    fail ONCE, never re-enqueue — re-running the identical pipeline only reproduces
    the block (the system_card_unshown 520->521->522 ~90min burn)."""
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "boom", chapter_id=1)

    def boom(c, job, log):
        raise worker.NonRetryableError(
            "prep-QA has BLOCKING errors after auto-heal (['system_card_unshown'])")

    worker.run_once(con, handlers={"boom": boom}, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "system_card_unshown" in err
    # NO auto-retry re-enqueued (and the error has no "auto-retry N/3" suffix)
    assert "auto-retry" not in err
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='boom' AND "
                       "state='queued'").fetchone()[0] == 0


def test_prepare_deterministic_qa_block_does_not_burn_retries(tmp_path, monkeypatch):
    """End-to-end: a prepare whose QA stays RED on a BLOCKING code after auto-heal
    fails once and is NOT re-queued — the retry-burn is gone."""
    con = _con(tmp_path)
    _seed_chapter(con, tmp_path)
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: None)
    monkeypatch.setattr(worker, "_heal_to_green", lambda c, ch, ep, log: None)
    monkeypatch.setattr(worker, "_heal_visual_drops",
                        lambda c, ch, ep, log: set())
    # the FINAL verdict (built fresh after heal) is the gate now, not a disk
    # re-read -> stub it directly instead of _qa_error_codes (heal-convergence
    # only, no longer a gate's source of truth).
    monkeypatch.setattr(worker, "_qa_verdict", lambda ep, **kw: worker.QAVerdict(
        ok=False, blocking={"system_card_unshown"},
        codes={"system_card_unshown"},
        report={"flags": [{"code": "system_card_unshown", "severity": "ERROR"}]},
        reason="blocking QA codes: ['system_card_unshown']"))   # stays red
    jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    state, err = con.execute(
        "SELECT state, error FROM job WHERE type='prepare' ORDER BY id LIMIT 1"
    ).fetchone()
    assert state == "failed" and "system_card_unshown" in err
    # the burn fix: a deterministic QA block parks the chapter, never re-queues
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='prepare' AND "
                       "state='queued'").fetchone()[0] == 0


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
        {"code": "ghost_text", "severity": "WARN"}]}))
    assert worker._qa_error_codes(ep) == {"narration_stale"}
    assert worker._qa_error_codes(tmp_path / "nope") == set()


# ---- QAVerdict: one fail-closed gate replacing two disagreeing policies ----

def test_qa_verdict_missing_report_blocks(tmp_path):
    v = worker._qa_verdict(tmp_path / "nope", started_at=time.time())
    assert v.ok is False and v.blocking == {"qa_report_invalid"}
    assert v.report is None


def test_qa_verdict_corrupt_report_blocks(tmp_path):
    (tmp_path / "prep_qa.json").write_text("{not valid json")
    v = worker._qa_verdict(tmp_path, started_at=time.time())
    assert v.ok is False and v.blocking == {"qa_report_invalid"}
    assert v.report is None


def test_qa_verdict_stale_report_blocks(tmp_path):
    import json
    import os
    report = tmp_path / "prep_qa.json"
    report.write_text(json.dumps({"flags": []}))
    started_at = time.time()
    stale = started_at - 10.0     # well past the 1.0s grace window
    os.utime(report, (stale, stale))
    v = worker._qa_verdict(tmp_path, started_at=started_at)
    assert v.ok is False and v.blocking == {"qa_report_invalid"}
    assert v.report is None


def test_qa_verdict_fresh_clean_report_passes(tmp_path):
    import json
    started_at = time.time()
    (tmp_path / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "ghost_text", "severity": "WARN"}]}))
    v = worker._qa_verdict(tmp_path, started_at=started_at)
    assert v.ok is True and v.blocking == set() and v.codes == set()
    assert v.report is not None


def test_structural_codes_block(tmp_path):
    import json
    started_at = time.time()
    (tmp_path / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "cut_gap", "severity": "ERROR"}]}))
    v = worker._qa_verdict(tmp_path, started_at=started_at)
    assert v.ok is False and v.blocking == {"cut_gap"}


def test_env_demotes_blocking_code(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("STUDIO_QA_NONBLOCKING", "cut_gap")
    started_at = time.time()
    (tmp_path / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "cut_gap", "severity": "ERROR"}]}))
    v = worker._qa_verdict(tmp_path, started_at=started_at)
    # demoted out of `blocking` but still visible in `codes` (heal/logging still
    # sees it -- the env hatch silences the GATE, not the signal).
    assert v.ok is True and v.blocking == set() and v.codes == {"cut_gap"}


def test_autopilot_uses_run_verdict_not_disk(tmp_path):
    import json
    con = _con(tmp_path)
    ep = _autopilot_series(con, tmp_path, flags=[])   # writes a clean report
    ch = {"id": 5, "series_id": 1}

    clean_verdict = worker.QAVerdict(ok=True, blocking=set(), codes=set(),
                                     report={"flags": []}, reason="")
    (ep / "prep_qa.json").write_text(json.dumps(
        {"flags": [{"code": "missing_audio", "severity": "ERROR"}]}))  # disk dirty
    assert worker._autopilot_clean(con, ch, clean_verdict) is True

    dirty_verdict = worker.QAVerdict(
        ok=False, blocking={"missing_audio"}, codes={"missing_audio"},
        report={"flags": [{"code": "missing_audio", "severity": "ERROR"}]},
        reason="x")
    (ep / "prep_qa.json").write_text(json.dumps({"flags": []}))   # disk clean
    assert worker._autopilot_clean(con, ch, dirty_verdict) is False


def test_qa_stage_meta_records_plan_sha(tmp_path, monkeypatch):
    import hashlib
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    (ep / "prep_qa.json").write_text(json.dumps({"flags": []}))
    (ep / "render.plan.clean.json").write_text('{"cuts": []}')
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep)}
    verdict = worker._run_prep_and_qa(con, ch, open(tmp_path / "l.txt", "w"))
    assert verdict.ok is True
    expected = hashlib.sha256(
        (ep / "render.plan.clean.json").read_bytes()).hexdigest()
    assert worker._last_qa_plan_sha(con, 5) == expected


def test_h_qa_scan_cosmetic_error_passes_blocking_fails(tmp_path, monkeypatch):
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path, status="voiced")
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)

    (ep / "prep_qa.json").write_text(json.dumps(
        {"flags": [{"code": "chrome_leak", "severity": "ERROR"}]}))  # cosmetic
    jid1 = jobs.enqueue(con, "qa_scan", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l1"))
    state1, err1 = con.execute("SELECT state, error FROM job WHERE id=?",
                               (jid1,)).fetchone()
    assert state1 == "done", err1

    (ep / "prep_qa.json").write_text(json.dumps(
        {"flags": [{"code": "cut_gap", "severity": "ERROR"}]}))  # structural
    jid2 = jobs.enqueue(con, "qa_scan", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l2"))
    state2, err2 = con.execute("SELECT state, error FROM job WHERE id=?",
                               (jid2,)).fetchone()
    assert state2 == "failed" and "cut_gap" in err2
    # verify blocking code is NOT logged in the cosmetic line
    log_file = con.execute("SELECT log_path FROM job WHERE id=?", (jid2,)).fetchone()[0]
    log_content = open(log_file).read()
    cosmetic_lines = [l for l in log_content.split('\n')
                      if '[qa] non-blocking QA flags' in l]
    if cosmetic_lines:
        assert "cut_gap" not in cosmetic_lines[0]


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
    verdict = worker._run_prep_and_qa(con, ch, log, heal_aware=True)
    assert verdict.blocking == {"missing_audio"} and not verdict.ok
    assert verdict.codes == {"missing_audio"}
    with pytest.raises(RuntimeError):
        worker._run_prep_and_qa(con, ch, log, heal_aware=False)


def test_prepare_auto_heals_red_qa_to_green(tmp_path, monkeypatch):
    # first QA is red -> the targeted auto-heal runs -> green -> job done
    con = _con(tmp_path)
    _seed_chapter(con, tmp_path)
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: None)
    healed = []
    monkeypatch.setattr(worker, "_heal_to_green",
                        lambda c, ch, ep, log: healed.append(1))
    # the FINAL verdict (built fresh after heal) is the gate now -> stub it
    # directly to say "green now" instead of the no-longer-authoritative
    # _qa_error_codes.
    monkeypatch.setattr(worker, "_qa_verdict", lambda ep, **kw: worker.QAVerdict(
        ok=True, blocking=set(), codes=set(), report={"flags": []}, reason=""))
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
                        lambda c, ch, log, **kw: None)
    monkeypatch.setattr(worker, "_heal_to_green", lambda c, ch, ep, log: None)
    # the FINAL verdict is the gate now -> stub it directly, still red.
    monkeypatch.setattr(worker, "_qa_verdict", lambda ep, **kw: worker.QAVerdict(
        ok=False, blocking={"montage_degenerate"}, codes={"montage_degenerate"},
        report={"flags": [{"code": "montage_degenerate", "severity": "ERROR"}]},
        reason="blocking QA codes: ['montage_degenerate']"))   # still red
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


def _run_heal_capture(tmp_path, monkeypatch):
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
    return heal_cmds, qa_cmds


def test_heal_to_green_fast_qa_then_final_semantic(tmp_path, monkeypatch):
    # semantic_heal OFF via the env hatch (repo toml default is ON since the
    # 2026-07-16 wave; STUDIO_SEMANTIC_HEAL=0 is the per-run rollback lever)
    monkeypatch.setenv("STUDIO_SEMANTIC_HEAL", "0")
    heal_cmds, qa_cmds = _run_heal_capture(tmp_path, monkeypatch)
    assert all("--include-grounding-warn" not in c for c in heal_cmds)
    assert len(qa_cmds) == 2
    assert "--semantic" not in qa_cmds[0]
    assert "--semantic" in qa_cmds[1]


def test_heal_to_green_semantic_default_passes_grounding_warn(
        tmp_path, monkeypatch):
    # production default (semantic_heal=true): grounding WARNs are healable
    # and every heal-cycle QA runs the semantic judge
    monkeypatch.delenv("STUDIO_SEMANTIC_HEAL", raising=False)
    heal_cmds, qa_cmds = _run_heal_capture(tmp_path, monkeypatch)
    assert heal_cmds and all("--include-grounding-warn" in c
                             for c in heal_cmds)
    assert qa_cmds and all("--semantic-heal" in c or "--semantic" in c
                           for c in qa_cmds)


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
        {"code": "ghost_text", "severity": "WARN"}])   # ordinary WARN ok
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


def test_bulk_auto_to_reapproves_stale_voice_and_enqueues(tmp_path, monkeypatch):
    """A prior 'bulk' voice approval whose content_sha no longer matches the
    script (a heal rewrote it after that approval) must not wedge the auto_to
    chain: the old existence-only guard skipped BOTH the re-approve and the
    enqueue when a stale row was already there, silently stalling the bulk
    run. ensure_approval treats the stale row as absent, so both fire again."""
    con = _con(tmp_path)
    ep = _autopilot_series(con, tmp_path, flags=[
        {"code": "ghost_text", "severity": "WARN"}])
    (ep / "manifest.script.json").write_text('{"paragraphs": ["healed"]}')
    gates.approve(con, "voice", chapter_id=5, note="bulk",
                 content_sha="stale-sha-from-before-the-heal")
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: set())
    jobs.enqueue(con, "prepare", chapter_id=5, payload={"auto_to": "voice"})
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='voiceover' "
                       "AND chapter_id=5").fetchone()[0] == 1   # no stall
    stored = con.execute(
        "SELECT content_sha FROM approval WHERE gate='voice' AND "
        "chapter_id=5 ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert stored == gates.gate_sha("voice", ep)                # fresh sha
    assert gates.voice_allowed(con, 5, ep) == (True, "")


def test_autopilot_voice_approval_stores_content_sha(tmp_path, monkeypatch):
    """The autopilot voice approval must bind to the script CONTENT, not just
    insert a checkbox row — content_sha has to be the real gate_sha."""
    con = _con(tmp_path)
    ep = _autopilot_series(con, tmp_path, flags=[
        {"code": "ghost_text", "severity": "WARN"}])
    (ep / "manifest.script.json").write_text('{"paragraphs": ["line"]}')
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: set())
    jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    stored = con.execute("SELECT content_sha FROM approval WHERE gate='voice' "
                         "AND chapter_id=5 AND note='autopilot'").fetchone()[0]
    assert stored is not None
    assert stored == gates.gate_sha("voice", ep)


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


def test_voiceover_rewinds_planned_chapter_so_it_actually_voices(
        tmp_path, monkeypatch):
    # `studio run` resumes by status: an explicit voiceover job on a chapter
    # already at 'planned' silently NO-OPS the voiced stage — narration edited
    # after voicing then ships with stale audio (ch1 2026-07-03, and the
    # estimate-mode QA is blind to clip drift by design). The handler must
    # rewind to 'scripted' BEFORE running so the voiced stage really runs.
    con = _con(tmp_path)
    _autopilot_series(con, tmp_path, autopilot=0, flags=[])
    con.execute("UPDATE chapter SET status='planned' WHERE id=5")
    con.commit()
    from studio.dashboard import gates as g
    g.approve(con, "voice", chapter_id=5, note="fix rotation")
    status_at_run = {}

    def stream(cmd, log, **kw):
        if "run" in cmd:
            status_at_run["v"] = con.execute(
                "SELECT status FROM chapter WHERE id=5").fetchone()[0]
        return 0

    monkeypatch.setattr(worker, "_stream", stream)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: set())
    jobs.enqueue(con, "voiceover", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    state, err = con.execute(
        "SELECT state, error FROM job WHERE type='voiceover'").fetchone()
    assert state == "done", err
    assert status_at_run["v"] == "scripted"   # rewound, not resume-skipped


def test_autopilot_voiceover_advances_to_render(tmp_path, monkeypatch):
    con = _con(tmp_path)
    _autopilot_series(con, tmp_path, flags=[])
    from studio.dashboard import gates as g
    g.approve(con, "voice", chapter_id=5, note="autopilot")
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: worker.QAVerdict(
                            ok=True, blocking=set(), codes=set(),
                            report={"flags": []}, reason=""))
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


def test_autopilot_render_approval_stores_content_sha(tmp_path, monkeypatch):
    """Same binding requirement as the voice autopilot approval, but for the
    render gate's two-file sha (plan.clean + tts_index)."""
    con = _con(tmp_path)
    ep = _autopilot_series(con, tmp_path, flags=[])
    from studio.dashboard import gates as g
    g.approve(con, "voice", chapter_id=5, note="autopilot")
    (ep / "tts").mkdir(parents=True, exist_ok=True)
    (ep / "render.plan.clean.json").write_text('{"cuts": []}')
    (ep / "tts" / "tts_index.json").write_text('{"clips": []}')
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: worker.QAVerdict(
                            ok=True, blocking=set(), codes=set(),
                            report={"flags": []}, reason=""))
    jobs.enqueue(con, "voiceover", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    stored = con.execute("SELECT content_sha FROM approval WHERE gate='render' "
                         "AND chapter_id=5 AND note='autopilot'").fetchone()[0]
    assert stored is not None
    assert stored == g.gate_sha("render", ep)


def test_voiceover_auto_to_reapproves_stale_render_and_enqueues(tmp_path, monkeypatch):
    """A prior 'bulk' render approval whose content_sha no longer matches the
    plan/tts (a plan regeneration after that approval) must not wedge the
    auto_to=video chain: the old existence-only guard skipped BOTH the
    re-approve and the enqueue when a stale row was already there, silently
    stalling the render auto-advance. ensure_approval treats the stale row as
    absent, so both fire again."""
    con = _con(tmp_path)
    ep = _autopilot_series(con, tmp_path, flags=[])
    from studio.dashboard import gates as g
    g.approve(con, "voice", chapter_id=5, note="autopilot")
    (ep / "tts").mkdir(parents=True, exist_ok=True)
    (ep / "render.plan.clean.json").write_text('{"cuts": []}')
    (ep / "tts" / "tts_index.json").write_text('{"clips": []}')
    # Insert a stale render approval from before the plan was regenerated
    g.approve(con, "render", chapter_id=5, note="bulk",
             content_sha="stale-sha-from-before-plan-regen")
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda c, ch, log, **kw: worker.QAVerdict(
                            ok=True, blocking=set(), codes=set(),
                            report={"flags": []}, reason=""))
    jobs.enqueue(con, "voiceover", chapter_id=5, payload={"auto_to": "video"})
    worker.run_once(con, handlers=worker.HANDLERS,
                    log_dir=str(tmp_path / "l"))
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='render_segment' "
                       "AND chapter_id=5").fetchone()[0] == 1   # no stall
    stored = con.execute(
        "SELECT content_sha FROM approval WHERE gate='render' AND "
        "chapter_id=5 ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert stored == g.gate_sha("render", ep)                # fresh sha


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


def test_heal_visual_drops_drops_cross_dup_later_copy(tmp_path, monkeypatch):
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    # cross_dup flags the LATER near-duplicate; the drop keeps the earlier good
    # panel, whose hold plays under the dropped panel's narration line.
    (ep / "prep_qa.json").write_text(json.dumps({"n_cuts": 10, "flags": [
        {"code": "cross_dup", "severity": "ERROR", "scene": "p000044.jpg",
         "detail": "near-duplicate of the previous cut (p000043.jpg in g0009_p16)"}]}))

    def fake_reprep(c, ch, log, **kw):    # the re-prep removes the dropped panel
        (ep / "prep_qa.json").write_text(json.dumps({"n_cuts": 9, "flags": []}))
        return set()
    monkeypatch.setattr(worker, "_run_prep_and_qa", fake_reprep)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep)}
    stuck = worker._heal_visual_drops(con, ch, ep, open(tmp_path / "l.txt", "w"))
    assert stuck == set()
    # auto-drops now land in their OWN file (not the human manual_drops.json),
    # keyed by the QA code so render_prep labels them honestly (not "operator").
    assert json.loads((ep / "auto_drops.json").read_text()) == {"p000044.jpg": "cross_dup"}
    assert not (ep / "manual_drops.json").exists()


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
    assert json.loads((ep / "auto_drops.json").read_text()) == {"b.jpg": "blank_crop"}


def test_heal_visual_drops_never_drops_a_narrated_sole_cut(tmp_path, monkeypatch):
    """The nano ch6 root cause: a droppable visual flag on a panel that OWNS a
    spoken line must NOT be auto-dropped — dropping it orphans the line into a
    long_hold. It's left in place (its non-blocking flag stays); nothing dropped."""
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    # a plan where p000017 is the SOLE shown art of a narration-bearing segment
    (ep / "render.plan.clean.json").write_text(json.dumps({"timeline": [
        {"segment_id": "g0005_p13", "scene_files": ["p000017.jpg"],
         "tts_text": "The heavy tread of boots echoes through the courtyard."}]}))
    (ep / "prep_qa.json").write_text(json.dumps({"n_cuts": 10, "flags": [
        {"code": "near_dup", "severity": "ERROR", "scene": "p000017.jpg"}]}))

    def _boom(*a, **k):                    # re-prep must NOT be reached
        raise AssertionError("should not re-prep — nothing droppable")
    monkeypatch.setattr(worker, "_run_prep_and_qa", _boom)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep)}
    stuck = worker._heal_visual_drops(con, ch, ep, open(tmp_path / "l.txt", "w"))
    assert stuck == set()                  # the narrated panel is protected, not stuck
    assert not (ep / "auto_drops.json").exists()   # nothing auto-dropped


# ---- auto-intro: a fully-rendered bundle auto-plans teaser + intro-ch1 ------

import io as _io


def _seed_bundle(con, *, autopilot=1, n=2, teaser_state="none",
                 status="rendered", ep_root=None):
    """A series with `n` chapters all at `status`, grouped into one bundle."""
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at, autopilot, teaser_state) "
                "VALUES (1,'asura','u','s','S','t',?,?)",
                (autopilot, teaser_state))
    for i in range(1, n + 1):
        ed = (str(ep_root / f"ep{i}") if ep_root is not None else f"/tmp/ep{i}")
        con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                    "status, ep_dir, updated_at) VALUES (?,1,?,?,?,?,?,'t')",
                    (i, i, f"Ch {i}", f"u{i}", status, ed))
    con.execute("INSERT INTO bundle (id, series_id, kind, title) "
                "VALUES (1,1,'manual','S — pack')")
    for pos, i in enumerate(range(1, n + 1)):
        con.execute("INSERT INTO bundle_chapter (bundle_id, chapter_id, "
                    "position) VALUES (1,?,?)", (i, pos))
    con.commit()
    return 1


def _teaser_on(monkeypatch):
    """Pin [teaser].enabled for a test instead of reading production config.

    _autostart_intro_if_ready gates on _beats_cfg().teaser_enabled, so these
    tests were asserting behaviour that depended on studio.toml — they broke the
    moment the owner turned the auto-start off. A unit test must supply its own
    config; only the gate is read, the rest of the path queries the DB.
    """
    monkeypatch.setattr(
        worker, "_beats_cfg",
        lambda: (types.SimpleNamespace(teaser_enabled=True), None, None))


def test_autostart_intro_enqueues_plan_teaser_once(tmp_path, monkeypatch):
    """All chapters rendered + autopilot on + teaser_state none -> exactly one
    plan_teaser (carrying auto_intro), and a second pass does NOT re-enqueue."""
    _teaser_on(monkeypatch)
    con = _con(tmp_path)
    bid = _seed_bundle(con, autopilot=1)
    worker._autostart_intro_if_ready(con, 2, _io.StringIO())   # last ch rendered
    n = con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser' AND "
                    "series_id=?", (1,)).fetchone()[0]
    assert n == 1
    import json as _j
    pj = con.execute("SELECT payload_json FROM job WHERE type='plan_teaser'"
                     ).fetchone()[0]
    assert _j.loads(pj).get("auto_intro_bundle") == 1
    # second pass (e.g. another render-completion check) must NOT double-enqueue
    worker._autostart_intro_if_ready(con, 2, _io.StringIO())
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser' AND "
                       "series_id=?", (1,)).fetchone()[0] == 1


def test_autostart_intro_skips_when_teaser_state_not_none(tmp_path):
    con = _con(tmp_path)
    _seed_bundle(con, autopilot=1, teaser_state="planned")
    worker._autostart_intro_if_ready(con, 2, _io.StringIO())
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser'"
                       ).fetchone()[0] == 0


def test_autostart_intro_skipped_when_autopilot_off(tmp_path):
    """Autopilot OFF -> the manual path is preserved (no auto plan_teaser)."""
    con = _con(tmp_path)
    _seed_bundle(con, autopilot=0)
    worker._autostart_intro_if_ready(con, 2, _io.StringIO())
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser'"
                       ).fetchone()[0] == 0


def test_autostart_intro_skips_when_not_all_rendered(tmp_path):
    con = _con(tmp_path)
    _seed_bundle(con, autopilot=1)
    con.execute("UPDATE chapter SET status='voiced' WHERE id=2")   # one pending
    con.commit()
    worker._autostart_intro_if_ready(con, 1, _io.StringIO())
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser'"
                       ).fetchone()[0] == 0


def test_teaser_auto_mode_approves_and_enqueues_concat(tmp_path, monkeypatch):
    """A plan_teaser carrying auto_intro: when the teaser renders, the worker
    auto-approves (teaser_state='approved') and enqueues the intro+ch1 concat
    with NO manual approval."""
    con = _con(tmp_path)
    bid = _seed_bundle(con, autopilot=1, ep_root=tmp_path)
    monkeypatch.setattr(worker, "REPO", tmp_path)
    # pre-stage the synthetic teaser dir + its render output so the handler's
    # existence checks pass with every subprocess mocked out.
    tdir = tmp_path / "dist" / "series_1" / "teaser"
    (tdir / "render").mkdir(parents=True)
    (tdir / "manifest.teaser.json").write_text("{}")
    (tdir / "render" / "segment_none.mp4").write_text("v")
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_fake_cfg(), "proj", "loc"))
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    import studio.pipeline as _pl
    monkeypatch.setattr(_pl, "_run_tool", lambda *a, **k: 0, raising=False)
    jobs.enqueue(con, "plan_teaser", series_id=1,
                 payload={"auto_intro_bundle": bid})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    assert con.execute("SELECT teaser_state FROM series WHERE id=?",
                       (bid,)).fetchone()[0] == "approved"
    cj = con.execute("SELECT payload_json FROM job WHERE type='concat' AND "
                     "bundle_id=?", (bid,)).fetchall()
    assert len(cj) == 1
    import json as _j
    assert _j.loads(cj[0][0]).get("intro_ch1") is True


def test_teaser_manual_mode_parks_for_review(tmp_path, monkeypatch):
    """No auto_intro flag (manual Plan-teaser click) -> teaser_state='planned'
    and NO concat enqueued: the manual review path is preserved."""
    con = _con(tmp_path)
    bid = _seed_bundle(con, autopilot=1, ep_root=tmp_path)
    monkeypatch.setattr(worker, "REPO", tmp_path)
    tdir = tmp_path / "dist" / "series_1" / "teaser"
    (tdir / "render").mkdir(parents=True)
    (tdir / "manifest.teaser.json").write_text("{}")
    (tdir / "render" / "segment_none.mp4").write_text("v")
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_fake_cfg(), "proj", "loc"))
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    import studio.pipeline as _pl
    monkeypatch.setattr(_pl, "_run_tool", lambda *a, **k: 0, raising=False)
    jobs.enqueue(con, "plan_teaser", series_id=1)              # no auto_intro
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    assert con.execute("SELECT teaser_state FROM series WHERE id=?",
                       (bid,)).fetchone()[0] == "planned"
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='concat'"
                       ).fetchone()[0] == 0


def test_concat_intro_ch1_builds_final_with_aac(tmp_path, monkeypatch):
    """The intro+ch1 concat = teaser + the bundle's FIRST chapter, written to
    dist/bundle_<id>/intro_ch1_FINAL.mp4 with -c:v copy -c:a aac -b:a 192k
    -movflags +faststart (the QuickTime-mute-safe flags)."""
    con = _con(tmp_path)
    bid = _seed_bundle(con, autopilot=1, teaser_state="approved",
                       ep_root=tmp_path)
    ep1 = tmp_path / "ep1" / "render"
    ep1.mkdir(parents=True)
    (ep1 / "segment_both.mp4").write_text("v")
    from studio.dashboard import gates as g
    g.approve(con, "concat", bundle_id=bid)
    monkeypatch.setattr(worker, "REPO", tmp_path)
    bdir = tmp_path / "dist" / f"bundle_{bid}"
    bdir.mkdir(parents=True)
    # the teaser is per MANHWA now — dist/series_<id>/teaser.mp4
    sdir = tmp_path / "dist" / "series_1"
    sdir.mkdir(parents=True)
    (sdir / "teaser.mp4").write_text("t")
    captured = {}

    def fake_stream(cmd, log, **kw):
        captured["cmd"] = list(cmd)
        return 0
    monkeypatch.setattr(worker, "_stream", fake_stream)
    jobs.enqueue(con, "concat", bundle_id=bid, payload={"intro_ch1": True})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    cmd = captured["cmd"]
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[cmd.index("-b:a") + 1] == "192k"
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert str(bdir / "intro_ch1_FINAL.mp4") in cmd
    lf = (bdir / "intro_ch1_concat.txt").read_text()
    assert "teaser.mp4" in lf and "segment_both.mp4" in lf
    assert lf.index("teaser.mp4") < lf.index("segment_both.mp4")   # teaser first


# ---- orphan-reap on restart: pgid persistence + identity-checked kill ------


def test_requeue_orphans_reaps_persisted_pgid(tmp_path, monkeypatch):
    """A running job with a persisted pgid (a worker died mid-job) must have
    its child reaped BEFORE the row goes back to 'queued' — otherwise the old
    child and the fresh retry write the same artifacts concurrently."""
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    con.execute("UPDATE job SET state='running', pgid=12345 WHERE id=?", (jid,))
    con.commit()
    calls = []
    monkeypatch.setattr(worker, "_reap_pgid",
                        lambda pgid: calls.append(pgid) or True)
    n = worker.requeue_orphans(con)
    assert calls == [12345]
    assert n == 1
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (jid,)).fetchone()[0] == "queued"


def test_requeue_orphans_skips_reap_when_no_pgid(tmp_path, monkeypatch):
    """A running job that never spawned a child (or already cleared its pgid)
    must NOT trigger a reap attempt — nothing to identity-check or kill."""
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    con.execute("UPDATE job SET state='running' WHERE id=?", (jid,))  # pgid NULL
    con.commit()
    calls = []
    monkeypatch.setattr(worker, "_reap_pgid",
                        lambda pgid: calls.append(pgid) or True)
    n = worker.requeue_orphans(con)
    assert calls == []
    assert n == 1


def test_stream_persists_and_clears_pgid(tmp_path):
    """_stream's persist/clear responsibility, tested directly against the
    helper (every existing worker test monkeypatches _stream itself, so there
    is no realistic way to exercise the real subprocess path here — and the
    brief is explicit that a subprocess harness isn't worth building just for
    this)."""
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    worker._record_pgid(con, jid, 99999)
    assert con.execute("SELECT pgid FROM job WHERE id=?",
                       (jid,)).fetchone()[0] == 99999
    worker._record_pgid(con, jid, None)
    assert con.execute("SELECT pgid FROM job WHERE id=?",
                       (jid,)).fetchone()[0] is None


# ---- _reap_pgid: identity-checked kill (never kill blind on pid reuse) -----

def test_reap_pgid_identity_mismatch_skips(monkeypatch):
    """A ps line that does NOT contain this repo's path means the pgid was
    recycled by an unrelated process (or the group has nothing of ours left
    in it) — _reap_pgid must skip it, never kill on a guess."""
    import subprocess
    import types
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="/usr/bin/some-other-process\n", returncode=0))
    killed = []
    monkeypatch.setattr(worker.os, "killpg",
                        lambda pgid, sig: killed.append(pgid))
    assert worker._reap_pgid(999999) is False
    assert killed == []


def test_reap_pgid_kills_matching_repo_child(monkeypatch):
    import subprocess
    import types
    cmdline = f"{worker.PY} {worker.REPO}/tools/render_prep.py --plan x\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout=cmdline, returncode=0))
    killed = []
    monkeypatch.setattr(worker.os, "killpg",
                        lambda pgid, sig: killed.append(pgid))
    assert worker._reap_pgid(4242) is True
    assert killed == [4242]


def test_reap_pgid_tolerates_already_gone_process(monkeypatch):
    """killpg on an already-reaped pgid (ProcessLookupError) or an all-zombie
    process group (PermissionError — reproduced live on macOS: killpg on a
    pgid whose only member is a defunct leader raises EPERM, not ESRCH) must
    not raise. requeue_orphans runs at worker boot; a crash there would be
    worse than leaving a stale lease for one cycle."""
    import subprocess
    import types
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout=f"{worker.REPO}/tools/x.py\n", returncode=0))

    def boom(pgid, sig):
        raise PermissionError("Operation not permitted")
    monkeypatch.setattr(worker.os, "killpg", boom)
    assert worker._reap_pgid(4242) is False


def test_render_segment_triggers_auto_intro_for_last_chapter(
        tmp_path, monkeypatch):
    """End-to-end: rendering the LAST chapter of an autopilot bundle flips it to
    'rendered' AND enqueues the auto plan_teaser (the hook is wired into the
    render-completion path)."""
    _teaser_on(monkeypatch)
    con = _con(tmp_path)
    bid = _seed_bundle(con, autopilot=1, status="rendered", ep_root=tmp_path)
    # ch2 is the one we're about to render: pending + gated green
    (tmp_path / "ep2").mkdir()
    con.execute("UPDATE chapter SET status='voiced' WHERE id=2")
    con.execute("INSERT INTO stage_run (chapter_id, stage, duration_sec, ok) "
                "VALUES (2,'qa_scan',1.0,1)")
    from studio.dashboard import gates as g
    g.approve(con, "render", chapter_id=2)
    con.commit()
    monkeypatch.setattr(worker, "REPO", tmp_path)
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    jobs.enqueue(con, "render_segment", chapter_id=2,
                 payload={"branding": "both"})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    assert con.execute("SELECT status FROM chapter WHERE id=2"
                       ).fetchone()[0] == "rendered"
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser' AND "
                       "series_id=?", (1,)).fetchone()[0] == 1


def test_render_branding_defaults_both(tmp_path, monkeypatch):
    """A render_segment job with no 'branding' in its payload must default to
    'both' — even when the series has legacy branding/intro.mp4 assets on
    disk. The old default flipped to 'none' (-> single.mp4 via intro+outro
    concat) whenever has_branding_segs was true, but that path is dead:
    render_prep has hard-forced intro/outro durations to 0 since 2026-06-29,
    so branding is ALWAYS 'both' in practice now."""
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path, status="voiced")
    con.execute("INSERT INTO stage_run (chapter_id, stage, duration_sec, ok) "
                "VALUES (5,'qa_scan',1.0,1)")
    from studio.dashboard import gates as g
    g.approve(con, "render", chapter_id=5)
    con.commit()
    monkeypatch.setattr(worker, "REPO", tmp_path)
    # legacy branding assets present -> pre-fix code picked "none" here
    bdir = tmp_path / "assets" / "branding" / "series" / "1"
    bdir.mkdir(parents=True)
    (bdir / "intro.mp4").write_text("i")
    (bdir / "outro.mp4").write_text("o")
    calls = []

    def fake_stream(cmd, log, **kw):
        calls.append([str(a) for a in cmd])
        return 0
    monkeypatch.setattr(worker, "_stream", fake_stream)
    jobs.enqueue(con, "render_segment", chapter_id=5, payload={})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    state, err = con.execute(
        "SELECT state, error FROM job WHERE type='render_segment'").fetchone()
    assert state == "done", err
    render_prep_cmd, remotion_cmd = calls[0], calls[1]
    assert render_prep_cmd[render_prep_cmd.index("--branding") + 1] == "both"
    out = str(ep / "render" / "segment_both.mp4")
    assert out in remotion_cmd
    assert not (ep / "render" / "single.mp4").exists()   # dead branch never runs


# ---- Task 10: render exactly the QA'd plan (skip render_prep on sha match) --

def test_render_skips_prep_on_matching_sha(tmp_path, monkeypatch):
    """The stamped plan_sha from the passing qa_scan matches the on-disk clean
    plan -> render_prep (which re-rolls the Gemma visual judge) must be
    skipped entirely; only remotion runs, against the QA'd bytes."""
    import hashlib
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path, status="voiced")
    (ep / "render.plan.clean.json").write_text('{"cuts": ["qad"]}')
    sha = hashlib.sha256(
        (ep / "render.plan.clean.json").read_bytes()).hexdigest()
    con.execute(
        "INSERT INTO stage_run (chapter_id, stage, duration_sec, ok, "
        "meta_json) VALUES (5,'qa_scan',1.0,1, json_object('plan_sha', ?))",
        (sha,))
    from studio.dashboard import gates as g
    g.approve(con, "render", chapter_id=5)
    con.commit()
    monkeypatch.setattr(worker, "REPO", tmp_path)
    calls = []
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw:
                        calls.append([str(a) for a in cmd]) or 0)
    jobs.enqueue(con, "render_segment", chapter_id=5, payload={})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    state, err = con.execute(
        "SELECT state, error FROM job WHERE type='render_segment'").fetchone()
    assert state == "done", err
    assert not any("render_prep.py" in " ".join(c) for c in calls)
    assert any("remotion" in c for c in calls)


def test_render_fails_on_plan_drift(tmp_path, monkeypatch):
    """A stamped plan_sha that no longer matches the on-disk clean plan (it
    changed since QA blessed it) must hard-fail — never silently render
    different bytes than what QA validated."""
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path, status="voiced")
    (ep / "render.plan.clean.json").write_text('{"cuts": ["changed"]}')
    con.execute(
        "INSERT INTO stage_run (chapter_id, stage, duration_sec, ok, "
        "meta_json) VALUES (5,'qa_scan',1.0,1, "
        "json_object('plan_sha', 'deadbeef00000000'))")
    from studio.dashboard import gates as g
    g.approve(con, "render", chapter_id=5)
    con.commit()
    monkeypatch.setattr(worker, "REPO", tmp_path)
    calls = []
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw:
                        calls.append([str(a) for a in cmd]) or 0)
    jobs.enqueue(con, "render_segment", chapter_id=5, payload={})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    state, err = con.execute(
        "SELECT state, error FROM job WHERE type='render_segment'").fetchone()
    assert state == "failed" and "sha drift" in err
    assert not any("remotion" in c for c in calls)


def test_render_missing_clean_plan_is_drift(tmp_path, monkeypatch):
    """A stamped plan_sha with NO render.plan.clean.json on disk (wiped, or
    never regenerated) must be treated as drift, not silently skipped."""
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path, status="voiced")
    con.execute(
        "INSERT INTO stage_run (chapter_id, stage, duration_sec, ok, "
        "meta_json) VALUES (5,'qa_scan',1.0,1, "
        "json_object('plan_sha', 'deadbeef00000000'))")
    from studio.dashboard import gates as g
    g.approve(con, "render", chapter_id=5)
    con.commit()
    monkeypatch.setattr(worker, "REPO", tmp_path)
    calls = []
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw:
                        calls.append([str(a) for a in cmd]) or 0)
    jobs.enqueue(con, "render_segment", chapter_id=5, payload={})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    state, err = con.execute(
        "SELECT state, error FROM job WHERE type='render_segment'").fetchone()
    assert state == "failed" and "sha drift" in err
    assert calls == []


def test_render_legacy_no_sha_uses_reuse_clean(tmp_path, monkeypatch):
    """A pre-refactor stage_run never stamped plan_sha. Render must fall back
    to running render_prep (today's behavior) but WITH --reuse-clean, so a
    legacy chapter doesn't silently re-roll the Gemma visual judge."""
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path, status="voiced")
    con.execute("INSERT INTO stage_run (chapter_id, stage, duration_sec, ok) "
                "VALUES (5,'qa_scan',1.0,1)")   # legacy row: no meta_json
    from studio.dashboard import gates as g
    g.approve(con, "render", chapter_id=5)
    con.commit()
    monkeypatch.setattr(worker, "REPO", tmp_path)
    calls = []
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw:
                        calls.append([str(a) for a in cmd]) or 0)
    jobs.enqueue(con, "render_segment", chapter_id=5, payload={})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))
    state, err = con.execute(
        "SELECT state, error FROM job WHERE type='render_segment'").fetchone()
    assert state == "done", err
    render_prep_cmd, remotion_cmd = calls[0], calls[1]
    assert "render_prep.py" in " ".join(render_prep_cmd)
    assert "--reuse-clean" in render_prep_cmd
    assert "remotion" in remotion_cmd


# ---- Task 4: heal re-sanitizes so the marker matches the rewritten script --

def test_rescript_triggers_sanitize(tmp_path, monkeypatch):
    """_rescript rewrites manifest.script.json via script_expander; the
    sanitize marker it leaves behind now describes stale text. _rescript must
    re-sanitize so a heal iteration converges on an already-fresh marker
    instead of deferring the work to the voiced-stage freshness backstop."""
    import types
    ep = tmp_path / "ep"
    ep.mkdir()

    calls = []
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw:
                        calls.append(" ".join(map(str, cmd))) or 0)
    sanitize_calls = []
    monkeypatch.setattr(worker, "_run_sanitize",
                        lambda ep_, log_: sanitize_calls.append(ep_))

    cfg = types.SimpleNamespace(script_model="s", narration_source="gemini_verbatim",
                               narration_sanitize=True)
    worker._rescript(ep, cfg, None, open(tmp_path / "log.txt", "w"))

    assert any("script_expander.py" in c for c in calls)
    assert sanitize_calls == [ep]


def test_rescript_skips_sanitize_when_config_disabled(tmp_path, monkeypatch):
    """Mirrors _stage_scripted's own gating exactly: the pipeline introduces
    no new config knob, so narration_sanitize=False must suppress the heal
    re-sanitize the same way it suppresses the scripted-stage one."""
    import types
    ep = tmp_path / "ep"
    ep.mkdir()

    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 0)
    sanitize_calls = []
    monkeypatch.setattr(worker, "_run_sanitize",
                        lambda ep_, log_: sanitize_calls.append(ep_))

    cfg = types.SimpleNamespace(script_model="s", narration_source="gemini_verbatim",
                               narration_sanitize=False)
    worker._rescript(ep, cfg, None, open(tmp_path / "log.txt", "w"))

    assert sanitize_calls == []


def test_run_sanitize_shells_narration_sanitize_pass(tmp_path, monkeypatch):
    """_run_sanitize shells narration_sanitize_pass.py with the same
    --script/--seed/--marker contract pipeline._run_sanitize_pass uses, and
    tolerates rc==2 ('unresolved blocks recorded' — the marker is written
    either way; the voiced-stage gate is what enforces it, not this helper)."""
    import types
    ep = tmp_path / "ep"
    ep.mkdir()
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (
        types.SimpleNamespace(beats_backend="ollama", beats_model="m"), "p", "l"))
    seen = {}

    def fake_stream(cmd, log, **kw):
        seen["cmd"] = cmd
        return 2   # unresolved blocks recorded -- must NOT raise

    monkeypatch.setattr(worker, "_stream", fake_stream)

    worker._run_sanitize(ep, open(tmp_path / "log.txt", "w"))

    cmd = seen["cmd"]
    assert "narration_sanitize_pass.py" in cmd[1]
    assert cmd[cmd.index("--script") + 1] == str(ep / "manifest.script.json")
    assert cmd[cmd.index("--seed") + 1] == ep.name
    assert cmd[cmd.index("--marker") + 1] == str(ep / "manifest.sanitize.json")
    assert cmd[cmd.index("--reframe-backend") + 1] == "ollama"


def test_run_sanitize_raises_on_real_failure(tmp_path, monkeypatch):
    """A crash (bad manifest, missing backend) is not the tolerated rc==2 —
    it must surface as the heal-step failure it is, not a silent pass."""
    import types
    import pytest
    ep = tmp_path / "ep"
    ep.mkdir()
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (
        types.SimpleNamespace(beats_backend="ollama", beats_model="m"), "p", "l"))
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw: 1)
    with pytest.raises(RuntimeError):
        worker._run_sanitize(ep, open(tmp_path / "log.txt", "w"))


# ---------------------------------------------------------------------------
# Two-lane concurrency: the chapter lease must hold under REAL threads
# ---------------------------------------------------------------------------

def test_two_lanes_never_claim_same_chapter_concurrently(tmp_path):
    """Two lane threads (gpu + tts), each with its own real sqlite3 connection
    to the SAME db file, hammer claim_next in lock-step via Barriers (no
    sleeps — deterministic). Each round: both threads attempt to claim at the
    same instant (barrier #1); once BOTH claims have landed (barrier #2), the
    'running' table is inspected directly for the invariant the chapter lease
    exists to guarantee — no chapter_id ever has more than one running row —
    THEN the claimed jobs are released (barrier #3) before the next round.
    Checking between claim and finish (rather than bucketing by round) is
    what actually catches an overlap; finishing immediately after claiming
    would let a broken lease hide behind "already done by the time the other
    thread got there"."""
    import threading

    db_path = tmp_path / "s.db"
    con = connect(db_path)
    for ch in (1, 2):
        jobs.enqueue(con, "prepare", chapter_id=ch)      # gpu lane
        jobs.enqueue(con, "voiceover", chapter_id=ch)    # tts lane
    con.close()

    rounds = 8             # comfortably more than the 4 jobs so misses still drain
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    claimed_ids: list = []
    violations: list = []

    def run(lane):
        c = connect(db_path)
        for _ in range(rounds):
            barrier.wait(timeout=5)                      # 1: claim at the same instant
            job = jobs.claim_next(c, lane=lane)
            barrier.wait(timeout=5)                      # 2: both claims have landed
            dupes = c.execute(
                "SELECT chapter_id FROM job WHERE state='running' "
                "GROUP BY chapter_id HAVING COUNT(*) > 1").fetchall()
            with lock:
                if dupes:
                    violations.append(dupes)
                if job:
                    claimed_ids.append(job["id"])
            if job:
                jobs.finish(c, job["id"], ok=True)        # release the lease
            barrier.wait(timeout=5)                       # 3: next round starts clean
        c.close()

    t1 = threading.Thread(target=run, args=("gpu",))
    t2 = threading.Thread(target=run, args=("tts",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert violations == [], f"chapter lease broken: {violations}"
    # sanity: all 4 jobs were claimed exactly once combined (no double-claim,
    # no lost job) -- a vacuous pass (nothing ever claimed) would not catch a
    # broken lease.
    assert sorted(claimed_ids) == sorted(set(claimed_ids))
    assert len(claimed_ids) == 4


# ---------------------------------------------------------------------------
# requeue_orphans: reap must precede the requeue UPDATE (never let a fresh
# retry and a still-alive old child write the same artifacts at once)
# ---------------------------------------------------------------------------

def test_orphan_reap_precedes_requeue_update(tmp_path, monkeypatch):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    con.execute("UPDATE job SET state='running', pgid=777 WHERE id=?", (jid,))
    con.commit()

    order = []
    monkeypatch.setattr(worker, "_reap_pgid",
                        lambda pgid: order.append(("reap", pgid)) or True)

    class _OrderSpy:
        """Duck-typed connection wrapper: records WHEN the requeue UPDATE runs
        relative to _reap_pgid, then delegates to the real connection."""
        def __init__(self, real):
            self._real = real

        def execute(self, sql, *a, **k):
            if sql.strip().startswith("UPDATE job SET state='queued'"):
                order.append(("requeue_update", None))
            return self._real.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._real, name)

    worker.requeue_orphans(_OrderSpy(con))
    assert [kind for kind, _ in order] == ["reap", "requeue_update"]


# ---------------------------------------------------------------------------
# Dispatch audit: _h_prepare must actually invoke timeline_planner.py (closes
# the "deleted _plan() would still pass the suite" gap -- every other prepare
# test stubs _stream with a blanket `lambda cmd, log, **kw: 0` and never
# inspects `cmd`, so silently dropping the _plan() call would pass them all)
# ---------------------------------------------------------------------------

def test_h_prepare_dispatches_timeline_planner(tmp_path, monkeypatch):
    con = _con(tmp_path)
    _seed_chapter(con, tmp_path)
    calls = []

    def fake_stream(cmd, log, **kw):
        calls.append(cmd)
        return 0
    monkeypatch.setattr(worker, "_stream", fake_stream)
    monkeypatch.setattr(worker, "_run_prep_and_qa", lambda c, ch, log, **kw: None)
    monkeypatch.setattr(worker, "_heal_to_green", lambda c, ch, ep, log: None)
    monkeypatch.setattr(worker, "_heal_visual_drops", lambda c, ch, ep, log: set())
    monkeypatch.setattr(worker, "_qa_verdict", lambda ep, **kw: worker.QAVerdict(
        ok=True, blocking=set(), codes=set(), report={"flags": []}, reason=""))
    jobs.enqueue(con, "prepare", chapter_id=5)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path / "l"))

    dispatched = [" ".join(map(str, cmd)) for cmd in calls]
    assert any("timeline_planner.py" in s for s in dispatched), dispatched


# ---------------------------------------------------------------------------
# Report-shape guard: _error_codes_from_report must never crash on a
# well-formed-JSON-but-wrong-shape report (a top-level array, or non-list
# flags) -- it should read as "nothing to flag", not blow up with AttributeError
# ---------------------------------------------------------------------------

def test_error_codes_from_report_rejects_non_dict_report():
    assert worker._error_codes_from_report([1, 2, 3]) == set()
    assert worker._error_codes_from_report("not even a list") == set()


def test_error_codes_from_report_rejects_non_list_flags():
    assert worker._error_codes_from_report({"flags": "oops"}) == set()


def test_qa_verdict_wrong_shape_report_blocks(tmp_path):
    """End-to-end: prep_qa.json on disk is a bare JSON array (valid JSON, wrong
    top-level shape) -- _qa_verdict must fail CLOSED (the same qa_report_invalid
    path as a corrupt/missing report), not read as a clean pass just because
    _error_codes_from_report tolerantly returns an empty set for a non-dict
    report (that tolerance is for the heal-loop caller, not this gate)."""
    import json
    (tmp_path / "prep_qa.json").write_text(json.dumps([1, 2, 3]))
    v = worker._qa_verdict(tmp_path, started_at=time.time())
    assert v.ok is False and v.blocking == {"qa_report_invalid"}
    assert v.report is None


# ---------------------------------------------------------------------------
# _qa_verdict mtime boundary: mtime == started_at - 1.0 is NOT "< started_at
# - 1.0" -- exactly at the boundary must pass; just below must block.
# ---------------------------------------------------------------------------

def test_qa_verdict_mtime_exactly_at_boundary_passes(tmp_path):
    import json
    import os
    report = tmp_path / "prep_qa.json"
    report.write_text(json.dumps({"flags": []}))
    started_at = time.time()
    boundary = started_at - 1.0
    os.utime(report, (boundary, boundary))
    v = worker._qa_verdict(tmp_path, started_at=started_at)
    assert v.ok is True and v.report is not None


def test_qa_verdict_mtime_just_below_boundary_blocks(tmp_path):
    import json
    import os
    report = tmp_path / "prep_qa.json"
    report.write_text(json.dumps({"flags": []}))
    started_at = time.time()
    just_below = started_at - 1.0 - 0.2
    os.utime(report, (just_below, just_below))
    v = worker._qa_verdict(tmp_path, started_at=started_at)
    assert v.ok is False and v.blocking == {"qa_report_invalid"}


def test_impact_mismatch_blocks_and_env_hatch_demotes(tmp_path, monkeypatch):
    """Eyes wave: impact_mismatch is in the blocking set (a narrated segment
    contradicting detector-verified impact SFX must not ship un-healed), and
    the STUDIO_QA_NONBLOCKING hatch can demote it without a deploy."""
    import json
    assert "impact_mismatch" in worker._CRITICAL_QA_CODES
    started_at = time.time()
    (tmp_path / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "impact_mismatch", "severity": "ERROR"}]}))
    v = worker._qa_verdict(tmp_path, started_at=started_at)
    assert v.ok is False and v.blocking == {"impact_mismatch"}
    monkeypatch.setenv("STUDIO_QA_NONBLOCKING", "impact_mismatch")
    v2 = worker._qa_verdict(tmp_path, started_at=started_at)
    assert v2.ok is True and v2.blocking == set()
    assert v2.codes == {"impact_mismatch"}     # hatch silences the gate, not the signal


def test_regen_flagged_passes_understanding_to_the_writer(tmp_path, monkeypatch):
    """Eyes wave: the heal regen MUST hand gemini_narrative_pass the
    understanding manifest (--understood), exactly like the primary beated
    stage does — the impact_mismatch heal-then-block design depends on the
    re-rolled group's payload carrying the [IMPACT SFX on panel] marker,
    which _pack_group_payload can only build from understood records."""
    import types
    ep = tmp_path / "ep"
    ep.mkdir()
    calls = []
    monkeypatch.setattr(worker, "_stream", lambda cmd, log, **kw:
                        calls.append(" ".join(map(str, cmd))) or 0)
    cfg = types.SimpleNamespace(beats_model="gemma", beats_backend="ollama",
                                punchup="off", script_model="s",
                                narration_source="gemini_verbatim",
                                narration_sanitize=False)
    worker._regen_flagged(ep, cfg, "", "", str(ep / "corr.json"), None,
                          open(tmp_path / "log.txt", "w"))
    gnp = [c for c in calls if "gemini_narrative_pass.py" in c]
    assert gnp, "heal regen must invoke the narration writer"
    assert "--understood" in gnp[0]
    assert str(ep / "manifest.panels.understood.json") in gnp[0]


def test_heal_treadmill_guard_stops_on_identical_corrections(
        tmp_path, monkeypatch):
    """Job-49 class: WARN-driven corrections stay byte-identical cycle after
    cycle (strictly-better keeps reverting the regen) while the ERROR set
    drifts — the old guard never fired and the loop burned all 4 cycles
    re-narrating the same groups. Guard 2 stops after ONE wasted repeat."""
    import itertools
    import json
    import types
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep), "number": 1}
    monkeypatch.setenv("STUDIO_SEMANTIC_HEAL", "1")

    same_corr = {"3": "bridge from the previous line"}
    err_cycle = itertools.cycle([["impact_mismatch"], ["panel_substituted"]])

    def fake_stream(cmd, log, **kw):
        s = " ".join(map(str, cmd))
        if "narration_heal.py" in s:
            out = cmd[cmd.index("--out") + 1]
            json.dump(same_corr, open(out, "w"))
        return 0

    def fake_qa(con_, ch_, log_, **kw):
        (ep / "prep_qa.json").write_text(json.dumps({"flags": [
            {"code": c, "severity": "ERROR", "segment_id": "g0003"}
            for c in next(err_cycle)]}))
        return set()

    regen = []
    monkeypatch.setattr(worker, "_stream", fake_stream)
    monkeypatch.setattr(worker, "_run_prep_and_qa", fake_qa)
    monkeypatch.setattr(worker, "_regen_flagged",
                        lambda *a, **k: regen.append(1))
    monkeypatch.setattr(worker, "_replan", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (
        types.SimpleNamespace(beats_model="m", beats_backend="ollama",
                              punchup="cinematic", script_model="s"), "p", "l"))
    fake_qa(con, ch, None)                       # seed the initial QA report
    logf = tmp_path / "log.txt"
    worker._heal_to_green(con, ch, ep, open(logf, "w"))
    text = logf.read_text()
    assert regen == [1]                          # exactly ONE regen, no treadmill
    assert "identical corrections" in text


def test_heal_repreps_once_for_render_fixable_remnants(tmp_path, monkeypatch):
    """Job-54 class: zero narration corrections but a long_hold ERROR remains
    — the plan is stale and only a render_prep re-run (cap-aware passes) can
    clear it. The heal loop must re-prep ONCE instead of returning."""
    import json
    import types
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep), "number": 1}
    monkeypatch.setenv("STUDIO_SEMANTIC_HEAL", "1")
    (ep / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": "long_hold", "severity": "ERROR", "segment_id": "g0019"}]}))

    def fake_stream(cmd, log, **kw):
        s = " ".join(map(str, cmd))
        if "narration_heal.py" in s:
            out = cmd[cmd.index("--out") + 1]
            json.dump({}, open(out, "w"))          # nothing narration-healable
        return 0

    preps = []
    monkeypatch.setattr(worker, "_stream", fake_stream)
    monkeypatch.setattr(worker, "_run_prep_and_qa",
                        lambda *a, **k: preps.append(k) or set())
    monkeypatch.setattr(worker, "_regen_flagged",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not re-narrate")))
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (
        types.SimpleNamespace(beats_model="m", beats_backend="ollama",
                              punchup="cinematic", script_model="s"), "p", "l"))
    logf = tmp_path / "log.txt"
    worker._heal_to_green(con, ch, ep, open(logf, "w"))
    assert len(preps) == 1 and preps[0].get("reuse_clean") is True
    assert "render-fixable" in logf.read_text()


def test_heal_repreps_render_fixable_after_stuck_stop(tmp_path, monkeypatch):
    """Job-55 class: the render-fixable re-prep must ALSO run on the stuck/
    cap loop-exit path — a treadmill stop with a long_hold left the stale
    plan blocking."""
    import itertools
    import json
    import types
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep), "number": 1}
    monkeypatch.setenv("STUDIO_SEMANTIC_HEAL", "1")
    same_corr = {"3": "bridge it"}
    err_cycle = itertools.cycle([["long_hold", "impact_mismatch"],
                                 ["long_hold", "narration_offset"]])

    def fake_stream(cmd, log, **kw):
        s = " ".join(map(str, cmd))
        if "narration_heal.py" in s:
            json.dump(same_corr, open(cmd[cmd.index("--out") + 1], "w"))
        return 0

    def fake_qa(con_, ch_, log_, **kw):
        preps.append(kw)
        (ep / "prep_qa.json").write_text(json.dumps({"flags": [
            {"code": c, "severity": "ERROR", "segment_id": "g0003"}
            for c in next(err_cycle)]}))
        return set()

    preps = []
    monkeypatch.setattr(worker, "_stream", fake_stream)
    monkeypatch.setattr(worker, "_run_prep_and_qa", fake_qa)
    monkeypatch.setattr(worker, "_regen_flagged", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_replan", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (
        types.SimpleNamespace(beats_model="m", beats_backend="ollama",
                              punchup="cinematic", script_model="s"), "p", "l"))
    fake_qa(con, ch, None)
    preps.clear()
    logf = tmp_path / "log.txt"
    worker._heal_to_green(con, ch, ep, open(logf, "w"))
    text = logf.read_text()
    assert "identical corrections" in text          # treadmill stop happened
    assert "render-fixable" in text                 # AND the re-prep followed
    assert preps and preps[-1].get("reuse_clean") is True


# --- Phase C: cancel actually stops the child ---------------------------------

def _spawn_dir(tmp_path):
    """_pgid_is_ours only recognises a child whose command line contains the
    repo path, so a realistic cancel test must spawn one that does."""
    return str(worker.REPO)


def test_cancel_kills_the_child_and_records_cancelled(tmp_path):
    """The acceptance test for the whole cancel path: a real child, a real
    cancel, and a real corpse check. Before this, cancel marked the row
    'cancelling', the row vanished from every query, and the child kept
    running to completion."""
    import os
    import subprocess
    import threading
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "sleeper", chapter_id=1)
    pgids = {}

    def sleeper(c, job, log):
        # a command line containing the repo path, so the pid-reuse guard
        # recognises it as ours
        rc = worker._stream(
            ["/bin/sh", "-c", f"cd {worker.REPO} && sleep 600"], log)
        log.write(f"rc={rc}\n")

    def capture():
        ccon = connect(tmp_path / "s.db")    # sqlite objects are thread-bound
        for _ in range(200):
            r = ccon.execute("SELECT pgid FROM job WHERE id=?",
                             (jid,)).fetchone()
            if r and r[0]:
                pgids["pgid"] = r[0]
                return
            time.sleep(0.05)

    threading.Thread(target=capture, daemon=True).start()
    threading.Thread(target=worker._cancel_monitor,
                     args=(str(tmp_path / "s.db"),), daemon=True).start()

    def cancel_soon():
        for _ in range(100):
            if pgids.get("pgid"):
                break
            time.sleep(0.05)
        ccon = connect(tmp_path / "s.db")
        jobs.cancel(ccon, jid)

    threading.Thread(target=cancel_soon, daemon=True).start()
    assert worker.run_once(con, handlers={"sleeper": sleeper},
                           log_dir=str(tmp_path / "logs")) is True

    state, pgid = con.execute("SELECT state, pgid FROM job WHERE id=?",
                              (jid,)).fetchone()
    assert state == "cancelled", f"job ended {state!r}, not cancelled"
    assert pgid is None, "pgid must be cleared so nothing reaps a stranger"
    # and the child is really gone
    spawned = pgids.get("pgid")
    assert spawned, "the test never observed a live child"
    for _ in range(40):
        out = subprocess.run(["ps", "-o", "command=", "-g", str(spawned)],
                             capture_output=True, text=True)
        if "sleep 600" not in out.stdout:
            break
        time.sleep(0.25)
    else:
        os.killpg(spawned, 9)
        raise AssertionError("the cancelled child survived")


def test_finish_cannot_clobber_a_cancel(tmp_path):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "stub", chapter_id=1)
    jobs.claim_next(con, lane=None)
    assert jobs.cancel(con, jid) is True
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (jid,)).fetchone()[0] == "cancelling"
    assert jobs.finish(con, jid, ok=True) is False
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (jid,)).fetchone()[0] == "cancelling"


def test_cancelling_still_holds_its_lane_and_chapter_lease(tmp_path):
    """A cancelling job's child is still alive, so it must keep occupying its
    lane and its chapter — otherwise a second job starts writing the same
    chapter's manifests while the first is still being killed."""
    con = _con(tmp_path)
    a = jobs.enqueue(con, "prepare", chapter_id=1)
    jobs.enqueue(con, "qa_scan", chapter_id=1)
    assert jobs.claim_next(con, lane="gpu")["id"] == a
    jobs.cancel(con, a)
    assert con.execute("SELECT state FROM job WHERE id=?",
                       (a,)).fetchone()[0] == "cancelling"
    assert jobs.claim_next(con, lane="gpu") is None, \
        "a cancelling job must still hold the chapter lease"


def test_cancelling_job_stays_visible_in_the_queue(tmp_path):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    jobs.claim_next(con, lane="gpu")
    jobs.cancel(con, jid)
    assert jid in [j["id"] for j in jobs.queue_view(con)], \
        "cancel must not make a job disappear from the UI"


def test_cancel_monitor_deadline_frees_a_wedged_lane(tmp_path, monkeypatch):
    """No live child and the handler never checkpoints -> the row must still
    go terminal, or the lane is leased forever."""
    import threading
    monkeypatch.setattr(worker, "_CANCEL_GRACE_SEC", 0)
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    jobs.claim_next(con, lane="gpu")
    jobs.cancel(con, jid)
    threading.Thread(target=worker._cancel_monitor,
                     args=(str(tmp_path / "s.db"),), daemon=True).start()
    for _ in range(60):
        if con.execute("SELECT state FROM job WHERE id=?",
                       (jid,)).fetchone()[0] == "cancelled":
            break
        time.sleep(0.25)
    else:
        raise AssertionError("cancelling job never went terminal")


def test_lane_thread_survives_a_database_error(tmp_path, monkeypatch):
    """One 'database is locked' used to kill a lane thread permanently while
    the process stayed alive, so launchd never restarted it."""
    import sqlite3 as _sq
    import threading
    con = _con(tmp_path)
    calls = {"n": 0}
    real = jobs.claim_next

    def flaky(c, lane=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sq.OperationalError("database is locked")
        return real(c, lane=lane)

    monkeypatch.setattr(jobs, "claim_next", flaky)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                if not worker.run_once(con, handlers={},
                                       log_dir=str(tmp_path / "logs")):
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.05)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    for _ in range(60):
        if calls["n"] >= 2:
            break
        time.sleep(0.05)
    stop.set()
    assert calls["n"] >= 2, "the lane never claimed again after the error"


def test_orphan_reap_covers_cancelling_rows(tmp_path, monkeypatch):
    """requeue_orphans used to flip cancelling->cancelled WITHOUT reaping,
    destroying the pgid — the only handle on that child."""
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    con.execute("UPDATE job SET state='cancelling', pgid=4242 WHERE id=?",
                (jid,))
    con.commit()
    reaped = []
    monkeypatch.setattr(worker, "_reap_pgid",
                        lambda p: reaped.append(p) or True)
    worker.requeue_orphans(con)
    assert reaped == [4242], "the cancelling job's child was never reaped"
    assert con.execute("SELECT state, pgid FROM job WHERE id=?",
                       (jid,)).fetchone() == ("cancelled", None)


def test_health_reports_a_dead_child_as_stale(tmp_path):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    # a pgid that cannot exist, started long ago
    con.execute("UPDATE job SET state='running', pgid=999999, "
                "started_at=datetime('now','-3 hours') WHERE id=?", (jid,))
    con.commit()
    h = jobs.health(con)
    assert [s["id"] for s in h["stale"]] == [jid]
    assert h["lanes"]["gpu"]["busy"] == 1


def test_health_does_not_call_the_heartbeat_row_stale(tmp_path):
    """The heartbeat row is 'running' by design and would otherwise report
    itself permanently stuck."""
    con = _con(tmp_path)
    con.execute("INSERT INTO job (type, state, started_at) VALUES "
                "('heartbeat','running',datetime('now','-3 hours'))")
    con.commit()
    h = jobs.health(con)
    assert h["stale"] == [] and h["live"] == 0
    assert h["worker_down"] is True


def test_requeue_reaps_before_requeuing(tmp_path, monkeypatch):
    con = _con(tmp_path)
    jid = jobs.enqueue(con, "prepare", chapter_id=1)
    con.execute("UPDATE job SET state='running', pgid=4242 WHERE id=?", (jid,))
    con.commit()
    order = []
    monkeypatch.setattr(worker, "_reap_pgid",
                        lambda p: order.append(("reap", p)) or True)
    assert jobs.requeue(con, jid) is True
    order.append(("requeued", None))
    assert order == [("reap", 4242), ("requeued", None)]
    assert con.execute("SELECT state, pgid, started_at FROM job WHERE id=?",
                       (jid,)).fetchone() == ("queued", None, None)


def test_retry_does_not_dedupe_onto_a_stale_attempt_counter(tmp_path):
    """dedupe ignores the payload, so a retry folding onto an existing queued
    row would reset _attempt and loop past max_attempts forever."""
    con = _con(tmp_path)
    jobs.enqueue(con, "boom", chapter_id=1)          # a plain queued row
    jid = jobs.enqueue(con, "boom", chapter_id=1, dedupe=False)
    con.execute("UPDATE job SET state='running' WHERE id=?", (jid,))
    con.commit()

    def boom(c, job, log):
        raise RuntimeError("transient")

    con.execute("UPDATE job SET payload_json='{\"_attempt\": 1}' WHERE id=?",
                (jid,))
    con.commit()
    job = dict(zip([c.strip() for c in jobs._COLS.split(",")],
                   con.execute(f"SELECT {jobs._COLS} FROM job WHERE id=?",
                               (jid,)).fetchone()))
    job["payload"] = {"_attempt": 1}
    rows_before = con.execute("SELECT COUNT(*) FROM job").fetchone()[0]
    try:
        boom(con, job, None)
    except RuntimeError:
        pass
    # the real path: run_once's retry branch must create a NEW row carrying
    # _attempt=2, not fold onto the pre-existing queued duplicate
    rid = jobs.enqueue(con, "boom", chapter_id=1,
                       payload={"_attempt": 2}, priority=1, dedupe=False)
    assert con.execute("SELECT COUNT(*) FROM job").fetchone()[0] == rows_before + 1
    import json as _json
    assert _json.loads(con.execute("SELECT payload_json FROM job WHERE id=?",
                                   (rid,)).fetchone()[0])["_attempt"] == 2


# --- auto-propose publish assets after N chapters -----------------------------

def _cfg_pub(**kw):
    base = dict(publish_auto_after_chapters=3, teaser_enabled=True)
    base.update(kw)
    return _fake_cfg(**base)


def _series_with_prepared(con, tmp_path, n, *, sid=1):
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (?,'asura','u','s','S','t')", (sid,))
    for i in range(1, n + 1):
        ep = tmp_path / f"ch{i}"
        ep.mkdir()
        (ep / "manifest.beats.json").write_text('{"beats": []}')
        con.execute("INSERT INTO chapter (series_id, number, label, url, "
                    "status, ep_dir, updated_at) VALUES (?,?,?,'u','beated',"
                    "?,'t')", (sid, i, f"Chapter {i}", str(ep)))
    con.commit()


def test_autopropose_thumbnail_once_threshold_reached(tmp_path, monkeypatch):
    con = _con(tmp_path)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_cfg_pub(), "p", "l"))
    _series_with_prepared(con, tmp_path, 3)
    import io
    worker._autopropose_publish_if_ready(con, 1, io.StringIO())
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='series_thumbnail' "
                       "AND series_id=1").fetchone()[0] == 1


def test_autopropose_does_nothing_below_threshold(tmp_path, monkeypatch):
    con = _con(tmp_path)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_cfg_pub(), "p", "l"))
    _series_with_prepared(con, tmp_path, 2)          # threshold is 3
    import io
    worker._autopropose_publish_if_ready(con, 1, io.StringIO())
    assert con.execute("SELECT COUNT(*) FROM job").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM bundle").fetchone()[0] == 0


def test_autopropose_is_idempotent(tmp_path, monkeypatch):
    """Firing on every prepare must not pile up duplicate jobs or bundles."""
    con = _con(tmp_path)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_cfg_pub(), "p", "l"))
    _series_with_prepared(con, tmp_path, 3)
    import io
    for _ in range(4):
        worker._autopropose_publish_if_ready(con, 1, io.StringIO())
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='series_thumbnail'"
                       ).fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser'"
                       ).fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM bundle").fetchone()[0] == 1


def test_autopropose_creates_a_debut_bundle_of_exactly_n_chapters(tmp_path,
                                                                  monkeypatch):
    con = _con(tmp_path)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_cfg_pub(), "p", "l"))
    _series_with_prepared(con, tmp_path, 5)          # threshold 3 -> first 3
    import io
    worker._autopropose_publish_if_ready(con, 1, io.StringIO())
    bid = con.execute("SELECT id FROM bundle WHERE series_id=1").fetchone()[0]
    from studio.dashboard import bundles
    assert len(bundles.bundle_chapters(con, bid)) == 3


def test_autopropose_teaser_is_a_proposal_not_auto_intro(tmp_path, monkeypatch):
    """The teaser must land as a REVIEW proposal (teaser_state -> planned via
    the handler), never auto-approved behind the operator's back."""
    con = _con(tmp_path)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_cfg_pub(), "p", "l"))
    _series_with_prepared(con, tmp_path, 3)
    import io, json as _json
    worker._autopropose_publish_if_ready(con, 1, io.StringIO())
    payload = con.execute("SELECT payload_json FROM job WHERE "
                          "type='plan_teaser'").fetchone()[0]
    assert "auto_intro" not in _json.loads(payload)


def test_autopropose_respects_an_existing_bundle_layout(tmp_path, monkeypatch):
    """If the operator already made bundles, don't auto-create a debut one."""
    con = _con(tmp_path)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_cfg_pub(), "p", "l"))
    _series_with_prepared(con, tmp_path, 3)
    from studio.dashboard import bundles
    bundles.create_bundle(con, 1, "full", title="mine")
    import io
    worker._autopropose_publish_if_ready(con, 1, io.StringIO())
    # thumbnail still proposed, but NO new bundle and NO teaser job
    assert con.execute("SELECT COUNT(*) FROM bundle").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser'"
                       ).fetchone()[0] == 0


def test_autopropose_skips_thumbnail_that_already_exists(tmp_path, monkeypatch):
    con = _con(tmp_path)
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_cfg_pub(), "p", "l"))
    _series_with_prepared(con, tmp_path, 3)
    thumb = worker.REPO / "dist" / "series_1" / "thumbnail_yt.jpg"
    thumb.parent.mkdir(parents=True, exist_ok=True)
    thumb.write_bytes(b"x")
    try:
        import io
        worker._autopropose_publish_if_ready(con, 1, io.StringIO())
        assert con.execute("SELECT COUNT(*) FROM job WHERE "
                           "type='series_thumbnail'").fetchone()[0] == 0
    finally:
        thumb.unlink()
        try:
            thumb.parent.rmdir()
        except OSError:
            pass


def test_autopropose_off_when_threshold_zero(tmp_path, monkeypatch):
    con = _con(tmp_path)
    monkeypatch.setattr(worker, "_beats_cfg",
                        lambda: (_cfg_pub(publish_auto_after_chapters=0), "p", "l"))
    _series_with_prepared(con, tmp_path, 5)
    import io
    worker._autopropose_publish_if_ready(con, 1, io.StringIO())
    assert con.execute("SELECT COUNT(*) FROM job").fetchone()[0] == 0


def test_debut_bundle_creation_survives_concurrent_prepares(tmp_path,
                                                            monkeypatch):
    """THE race the review caught: gpu width is 2 and the chapter lease is
    per-chapter, so two different chapters of one series prepare concurrently
    on two connections. A plain check-then-create would make TWO debut bundles
    and TWO teaser proposals. create_debut_bundle_once serializes on
    BEGIN IMMEDIATE, so exactly one bundle + one teaser survive."""
    import threading
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_cfg_pub(), "p", "l"))
    con = _con(tmp_path)
    _series_with_prepared(con, tmp_path, 3)
    db = str(tmp_path / "s.db")

    barrier = threading.Barrier(2)
    errors = []

    def fire():
        import io
        c = connect(db)                      # its OWN connection, like a lane
        try:
            barrier.wait(timeout=5)
            worker._autopropose_publish_if_ready(c, 1, io.StringIO())
        except Exception as e:               # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=fire) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=20)

    assert not errors, errors
    assert con.execute("SELECT COUNT(*) FROM bundle WHERE series_id=1"
                       ).fetchone()[0] == 1, "debut bundle double-created"
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='plan_teaser'"
                       ).fetchone()[0] == 1, "teaser double-proposed"


def test_failed_thumbnail_proposal_is_not_re_fired_every_prepare(tmp_path,
                                                                 monkeypatch):
    """A permanently-failing thumbnail (e.g. no API key) must not re-enqueue on
    every subsequent chapter prepare — propose ONCE, ever."""
    monkeypatch.setattr(worker, "_beats_cfg", lambda: (_cfg_pub(), "p", "l"))
    con = _con(tmp_path)
    _series_with_prepared(con, tmp_path, 3)
    import io
    worker._autopropose_publish_if_ready(con, 1, io.StringIO())
    # the proposed thumbnail job fails
    con.execute("UPDATE job SET state='failed' WHERE type='series_thumbnail'")
    con.commit()
    worker._autopropose_publish_if_ready(con, 1, io.StringIO())  # next prepare
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='series_thumbnail'"
                       ).fetchone()[0] == 1


def test_autostart_intro_leaves_a_failed_proposed_teaser_alone(tmp_path,
                                                               monkeypatch):
    """A proposed teaser that FAILED leaves teaser_state='none'. The autopilot
    auto-intro path must NOT then re-plan it as an auto_intro (which would
    auto-approve a teaser the operator never reviewed)."""
    monkeypatch.setattr(worker, "_beats_cfg",
                        lambda: (_cfg_pub(teaser_enabled=True), "p", "l"))
    con = _con(tmp_path)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at, autopilot) VALUES (1,'a','u','s','S','t',1)")
    ep = tmp_path / "ch1"
    ep.mkdir()
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES (1,1,1,'C1','u','rendered',"
                "?,'t')", (str(ep),))
    bid = con.execute("INSERT INTO bundle (series_id, kind, teaser_state) "
                      "VALUES (1,'manual','none')").lastrowid
    con.execute("INSERT INTO bundle_chapter (bundle_id, chapter_id, position) "
                "VALUES (?,1,0)", (bid,))
    # a prior PROPOSED teaser that failed
    con.execute("INSERT INTO job (type, bundle_id, state) VALUES "
                "('plan_teaser', ?, 'failed')", (bid,))
    con.commit()
    import io
    worker._autostart_intro_if_ready(con, 1, io.StringIO())
    autos = con.execute(
        "SELECT COUNT(*) FROM job WHERE type='plan_teaser' AND bundle_id=? "
        "AND state='queued'", (bid,)).fetchone()[0]
    assert autos == 0, "autopilot re-planned a failed proposal as auto_intro"


def test_heal_visual_drops_never_drops_a_narrated_MULTI_panel_span(tmp_path, monkeypatch):
    """The wedge (audited 2026-08-18): the worker protected only SOLE cuts,
    while render_prep.protect_narrated_from_junk protects EVERY panel of a
    narrated span. So a droppable flag on the 2nd panel of a 2-panel narrated
    span was written to auto_drops.json, render_prep refused the drop, the flag
    survived, and the next pass hit 'already dropped but STILL flagged —
    BLOCKING' — a chapter wedged forever on a drop that can never take effect."""
    import json
    con = _con(tmp_path)
    ep = _seed_chapter(con, tmp_path)
    (ep / "render.plan.clean.json").write_text(json.dumps({"timeline": [
        {"segment_id": "g0007_p02", "scene_files": ["p000030.jpg", "p000031.jpg"],
         "tts_text": "He lunges, and the blade comes down."}]}))
    (ep / "prep_qa.json").write_text(json.dumps({"n_cuts": 10, "flags": [
        {"code": "cross_dup", "severity": "ERROR", "scene": "p000031.jpg"}]}))

    def _boom(*a, **k):
        raise AssertionError("should not re-prep — the panel is narrated")
    monkeypatch.setattr(worker, "_run_prep_and_qa", _boom)
    ch = {"id": 5, "series_id": 1, "ep_dir": str(ep)}
    stuck = worker._heal_visual_drops(con, ch, ep, open(tmp_path / "l.txt", "w"))
    assert stuck == set()                          # not blocking
    assert not (ep / "auto_drops.json").exists()   # and no futile drop written


# ---- approval gates are permanent, not transient -------------------------

def _seed_scripted(con, cid=41):
    con.execute("INSERT OR IGNORE INTO series (id, source, series_url, slug, "
                "title, added_at) VALUES (1,'asura','u','s','S','t')")
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES "
                "(?,1,1,'Ch 1','u','scripted','/tmp/nope','t')", (cid,))
    con.commit()


def test_voiceover_approval_block_is_not_retried(tmp_path):
    """An approval gate cannot be cleared by re-running — only by a human.

    ORV Ep64 burned all three retries inside the 14 minutes before the owner
    approved, leaving three IDENTICAL failures in the job list that read like a
    real fault. Same rule as test_deterministic_qa_block_is_not_retried: fail
    once, park it, say why.
    """
    con = _con(tmp_path)
    _seed_scripted(con, cid=41)
    jid = jobs.enqueue(con, "voiceover", chapter_id=41)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed" and "blocked" in err
    assert "auto-retry" not in err, f"approval block was retried: {err}"
    queued = con.execute("SELECT COUNT(*) FROM job WHERE type='voiceover' AND "
                         "state='queued'").fetchone()[0]
    assert queued == 0, "a retry was re-enqueued for a permanent block"


def test_render_approval_block_is_not_retried(tmp_path):
    con = _con(tmp_path)
    _seed_scripted(con, cid=42)
    jid = jobs.enqueue(con, "render_segment", chapter_id=42)
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed"
    assert "auto-retry" not in err, f"render gate was retried: {err}"
    assert con.execute("SELECT COUNT(*) FROM job WHERE type='render_segment' "
                       "AND state='queued'").fetchone()[0] == 0


def test_chain_voice_gate_block_is_not_retried(tmp_path):
    con = _con(tmp_path)
    _seed_scripted(con, cid=43)
    jid = jobs.enqueue(con, "chain", chapter_id=43, payload={"target": "planned"})
    worker.run_once(con, handlers=worker.HANDLERS, log_dir=str(tmp_path))
    state, err = con.execute("SELECT state, error FROM job WHERE id=?",
                             (jid,)).fetchone()
    assert state == "failed"
    assert "auto-retry" not in err, f"chain voice gate was retried: {err}"
