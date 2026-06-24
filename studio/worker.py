"""studio worker — the queue executor (run in its own terminal/launchd).

Claims ONE job at a time from studio.db (serial GPU policy), executes it,
streams output to logs/jobs/<id>.log, records per-stage durations into
stage_run, and enforces the gates: render needs a passing QA scan + your
approval; concat needs bundle approval. The dashboard never executes
anything — it only inserts job/approval rows that this process consumes.

Run:  .eval_venv/bin/python -m studio worker
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

from studio.dashboard import bundles, gates, jobs

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".eval_venv" / "bin" / "python")


@contextlib.contextmanager
def record_stage(con: sqlite3.Connection, *, chapter_id: Optional[int],
                 stage: str, series_id: Optional[int] = None):
    """Wraps any stage execution: stage_run row with duration + ok flag."""
    t0 = time.time()
    ok = 1
    try:
        yield
    except BaseException:
        ok = 0
        raise
    finally:
        con.execute(
            "INSERT INTO stage_run (chapter_id, stage, duration_sec, ok, "
            "meta_json) VALUES (?,?,?,?, json_object('series_id', ?))",
            (chapter_id, stage, round(time.time() - t0, 2), ok, series_id))
        con.commit()


def _chapter(con: sqlite3.Connection, chapter_id: int) -> Dict[str, Any]:
    r = con.execute("SELECT id, series_id, number, label, ep_dir, status "
                    "FROM chapter WHERE id=?", (chapter_id,)).fetchone()
    if not r:
        raise RuntimeError(f"chapter {chapter_id} not in catalog")
    return dict(zip(("id", "series_id", "number", "label", "ep_dir",
                     "status"), r))


def _series_title(con: sqlite3.Connection, series_id: int) -> str:
    r = con.execute("SELECT title FROM series WHERE id=?",
                    (series_id,)).fetchone()
    return r[0] if r else ""


# Catastrophic-hang BACKSTOP: any single stage shell-out is bounded by a generous
# wall-clock. The longest legit single call is the fresh understand+narrate pass
# (~50 min), so the default is well above that — this only kills a TRULY wedged
# child (the 48-min TTS freeze, an infinite 429 loop, a dead CDN socket) so one
# bad chapter fails and the lane moves on instead of stalling the whole run. The
# per-clip TTS watchdog handles the fine-grained case; this is the coarse net.
_STAGE_TIMEOUT_SEC = int(os.environ.get("STUDIO_STAGE_TIMEOUT_SEC", "5400") or "5400")


import threading

# --- operator cancel of a RUNNING job -----------------------------------------
# run_once stamps the current job id on its lane thread (_CUR); _stream registers
# the live subprocess under that id in _ACTIVE. The cancel-monitor thread kills
# the process tree of any job the dashboard marked 'cancelling'; the killed
# subprocess makes the handler raise, and run_once then records the job
# 'cancelled' (NOT failed) so an operator cancel never auto-retries.
_CUR = threading.local()
_ACTIVE: "Dict[int, subprocess.Popen]" = {}
_ACTIVE_LK = threading.Lock()


def _cancel_monitor(db_path: str) -> None:
    import signal
    from studio.catalog.db import connect
    mcon = connect(db_path)
    while True:
        try:
            for (jid,) in mcon.execute(
                    "SELECT id FROM job WHERE state='cancelling'").fetchall():
                with _ACTIVE_LK:
                    p = _ACTIVE.get(jid)
                if p is not None and p.poll() is None:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
        except Exception:
            pass
        time.sleep(3)


def _stream(cmd, log: TextIO, cwd: str = str(REPO),
            env: Optional[Dict[str, str]] = None,
            timeout: Optional[int] = None) -> int:
    import signal
    log.write("$ " + " ".join(str(c) for c in cmd) + "\n")
    log.flush()
    to = _STAGE_TIMEOUT_SEC if timeout is None else timeout
    # own process group so we can kill the WHOLE tree (remotion's chrome, the TTS
    # python, ffmpeg children), not just the direct child.
    p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                         text=True, env=env, start_new_session=True)
    jid = getattr(_CUR, "job_id", None)        # operator-cancel registry
    if jid is not None:
        with _ACTIVE_LK:
            _ACTIVE[jid] = p
    try:
        return p.wait(timeout=to)
    except subprocess.TimeoutExpired:
        log.write(f"\n[worker] HANG BACKSTOP: stage exceeded {to}s wall-clock — "
                  f"killing the process tree and failing the stage.\n")
        log.flush()
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            p.kill()
        try:
            p.wait(timeout=30)
        except Exception:
            pass
        return 124  # conventional timeout exit code -> stage fails, lane continues
    finally:
        if jid is not None:
            with _ACTIVE_LK:
                _ACTIVE.pop(jid, None)


def _series_env(con: sqlite3.Connection,
                series_id: Optional[int]) -> Optional[Dict[str, str]]:
    """Per-series narration style rides into pipeline subprocesses as
    STUDIO_PUNCHUP (config env override) — thread-safe across parallel
    lanes, unlike mutating os.environ."""
    if not series_id:
        return None
    r = con.execute("SELECT narration_style FROM series WHERE id=?",
                    (series_id,)).fetchone()
    style = (r[0] or "").strip() if r else ""
    if style in ("off", "light", "full"):
        return {**os.environ, "STUDIO_PUNCHUP": style}
    return None


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

# codes prep_qa emits that the worker may self-heal by re-running pipeline
# stages (prose is never edited in place): mechanical staleness re-scripts;
# a dangling-fragment narration re-writes its beats once (the writer now
# carries rolling context + fragment rules, so a re-roll usually lands).
STALE_CODES = {"beats_incomplete", "narration_stale", "fragment_dangle",
               "caption_unvoiced"}

# QA ERRORs that mean a BROKEN video — these (and only these) block a chapter
# after auto-heal. Everything else (cross_dup, fragment_dangle, visible_text,
# caption_unvoiced, chrome_leak, …) is a cosmetic/quality nit: the heal tries to
# fix it, but if it can't, the chapter still SHIPS with a WARN for review rather
# than hard-failing a whole recap + hours of work over a repeated panel.
_CRITICAL_QA_CODES = {
    "audio_index_missing", "audio_missing", "audio_stale", "missing_audio",
    "missing_file", "missing_dims", "stale_dims",
    "empty_item", "montage_degenerate", "beats_incomplete",
    # manifest freshness — stale or missing manifests block render
    "stale_manifest", "missing_manifest",
    # a whole stitch chunk rendered as one panel (detection under-segmented) —
    # heal can't fix a crop, so block → re-stitch/re-detect rather than ship it
    "chunk_as_panel",
}


def _qa_error_codes(ep: Path) -> set:
    try:
        report = json.loads((Path(ep) / "prep_qa.json").read_text())
    except Exception:
        return set()
    return {f.get("code") for f in report.get("flags") or []
            if f.get("severity") == "ERROR"}


def _autopilot_clean(con: sqlite3.Connection, ch: Dict[str, Any]) -> bool:
    """Autopilot advances ONLY on a spotless report: the series opted in,
    zero ERRORs, and zero semantic narration_mismatch warnings. Anything
    else waits for a human — manage by exception."""
    r = con.execute("SELECT autopilot FROM series WHERE id=?",
                    (ch["series_id"],)).fetchone()
    if not (r and r[0]):
        return False
    try:
        report = json.loads(
            (Path(ch["ep_dir"] or "") / "prep_qa.json").read_text())
    except Exception:
        return False
    flags = report.get("flags") or []
    if any(f.get("severity") == "ERROR" for f in flags):
        return False
    return not any(f.get("code") == "narration_mismatch" for f in flags)


def _run_prep_and_qa(con: sqlite3.Connection, ch: Dict[str, Any],
                     log: TextIO, *, branding: str = "both",
                     heal_aware: bool = False, reuse_clean: bool = False,
                     semantic: bool = True) -> set:
    """render_prep + prep_qa for a chapter; records the qa_scan stage.
    Returns the ERROR flag codes. heal_aware=True lets the caller handle
    stale-narration codes instead of failing the job outright."""
    ep = Path(ch["ep_dir"] or "")
    title = _series_title(con, ch["series_id"])
    with record_stage(con, chapter_id=ch["id"], stage="prepped",
                      series_id=ch["series_id"]):
        prep_args = [PY, str(REPO / "tools" / "render_prep.py"),
                     "--plan", str(ep / "render.plan.json"),
                     "--scenes-manifest", str(ep / "manifest.scenes.json"),
                     "--episode-dir", str(ep), "--series-title", title,
                     "--branding", branding]
        if reuse_clean:
            # heal cycle: panels are unchanged, only narration moved — reuse the
            # cached per-cut visual-judge verdicts instead of re-paying the Gemma
            # pass (cuts a heal cycle from ~8 min to ~1).
            prep_args.append("--reuse-clean")
        rc = _stream(prep_args, log)
        if rc != 0:
            raise RuntimeError(f"render_prep exited {rc}")
    t0 = time.time()
    qa_args = [PY, str(REPO / "tools" / "prep_qa.py"),
               "--episode-dir", str(ep), "--series-title", title]
    from studio.config import load as _load_cfg
    cfg = _load_cfg()
    if semantic:
        qa_args.append("--semantic")
    if cfg.semantic_heal:
        qa_args.append("--semantic-heal")   # QA-eyes: grounding_weak -> auto-heal
    rc = _stream(qa_args, log)
    codes = _qa_error_codes(ep)
    critical = codes & _CRITICAL_QA_CODES
    # qa_scan 'ok' gates the render — it's ok when no BLOCKING error remains;
    # cosmetic ERRORs (cross_dup, fragment_dangle, visible_text, …) don't fail it.
    con.execute(
        "INSERT INTO stage_run (chapter_id, stage, duration_sec, ok, "
        "meta_json) VALUES (?,?,?,?, json_object('series_id', ?))",
        (ch["id"], "qa_scan", round(time.time() - t0, 2),
         0 if critical else 1, ch["series_id"]))
    con.commit()
    if critical and not heal_aware:
        raise RuntimeError(f"prep-QA found BLOCKING flags ({sorted(critical)}) — "
                           f"open the report in {ep}")
    return codes


def _beats_cfg():
    """(cfg, project, location) for the beats/punchup/script tools, sourced the
    same way _stage_beated does (SA key project, no gcloud)."""
    from studio.config import load as _load_cfg
    cfg = _load_cfg()
    keys = REPO / "keys" / "gcp-vision.json"
    project = (json.loads(keys.read_text()).get("project_id", "")
               if keys.exists() else os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    return cfg, project, location


def _regen_flagged(ep: Path, cfg, project: str, location: str,
                   corr_path: str, env, log: TextIO) -> None:
    """Re-narrate ONLY the corrected groups from their panels (--resume keeps
    every other line), then re-apply persona + re-derive the verbatim script."""
    beats, cast = str(ep / "manifest.beats.json"), str(ep / "manifest.cast.json")
    vision = str(ep / "manifest.vision.json")
    preheal = ep / "manifest.beats.preheal.json"
    if getattr(cfg, "semantic_heal", False):
        preheal.write_text(Path(beats).read_text())   # snapshot the pre-heal lines
    gargs = [PY, str(REPO / "tools" / "gemini_narrative_pass.py"),
             "--groups-manifest", str(ep / "manifest.groups.json"),
             "--vision-manifest", vision, "--out", beats,
             "--project", project, "--location", location,
             "--model", cfg.beats_model, "--cast", cast,
             "--story", str(ep / "manifest.story.json"),
             "--resume", "--corrections", corr_path, "--max-images-per-group", "6"]
    if cfg.beats_backend == "ollama":
        gargs += ["--backend", "ollama", "--ollama-model", cfg.beats_model]
    if _stream(gargs, log, env=env) != 0:
        raise RuntimeError("gemini_narrative_pass (heal) failed")
    if (cfg.punchup or "off") != "off":
        pargs = [PY, str(REPO / "tools" / "narration_punchup.py"),
                 "--beats", beats, "--out", beats, "--cast", cast,
                 "--episode-dir", str(ep), "--humor", cfg.punchup]
        if cfg.beats_backend == "ollama":
            pargs += ["--backend", "ollama", "--ollama-model", cfg.beats_model]
        else:
            pargs += ["--backend", "vertex", "--model", cfg.beats_model,
                      "--project", project, "--location", location]
        _stream(pargs, log, env=env)
    if getattr(cfg, "semantic_heal", False):
        # strictly-better safeguard: keep each regenerated line ONLY if a judge
        # rules it beats the pre-heal line on the panel; else revert to the
        # original. Auto-heal can then only improve or hold a beat, never make
        # it worse (closes the closed-loop-degrades-good-lines risk).
        aargs = [PY, str(REPO / "tools" / "narration_accept_better.py"),
                 "--old", str(preheal), "--new", beats,
                 "--scenes-dir", str(ep / "scenes_clean"),
                 "--vision-manifest", vision, "--out", beats]
        if cfg.beats_backend == "ollama":
            aargs += ["--backend", "ollama", "--ollama-model", cfg.beats_model]
        else:
            aargs += ["--backend", "vertex", "--model", cfg.beats_model,
                      "--project", project, "--location", location]
        _stream(aargs, log, env=env)
    sargs = [PY, str(REPO / "tools" / "script_expander.py"), "--beats", beats,
             "--vision", vision, "--out", str(ep / "manifest.script.json"),
             "--model", cfg.script_model, "--narration-source",
             "gemini_verbatim", "--cast", cast]
    if getattr(cfg, "narration_microbeats", False):
        sargs += ["--microbeats", "--microbeat-max-words",
                  str(getattr(cfg, "narration_microbeat_max_words", 28))]
    if _stream(sargs, log, env=env) != 0:
        raise RuntimeError("script_expander (heal) failed")


_HEAL_MAX = 4


def _heal_to_green(con: sqlite3.Connection, ch: Dict[str, Any], ep: Path,
                   log: TextIO) -> None:
    """Auto-heal: regenerate ONLY the QA-flagged groups from their panels and
    re-derive, up to _HEAL_MAX cycles, until no narration-healable ERROR remains.
    A failing line is re-narrated from the art — never dropped to satisfy QA."""
    corr = ep / "heal_corrections.json"
    from studio.config import load as _load_cfg
    cfg = _load_cfg()
    semantic_heal = bool(getattr(cfg, "semantic_heal", False))
    project = location = env = None
    used_fast_qa = False
    for cycle in range(1, _HEAL_MAX + 1):
        heal_args = [PY, str(REPO / "tools" / "narration_heal.py"),
                     "--qa", str(ep / "prep_qa.json"), "--out", str(corr)]
        if semantic_heal:
            heal_args.append("--include-grounding-warn")
        _stream(heal_args, log)
        try:
            ncorr = len(json.loads(corr.read_text()))
        except Exception:
            ncorr = 0
        if ncorr == 0:
            if used_fast_qa:
                log.write("[heal] final semantic QA scan after mechanical "
                          "heal cycle(s)\n")
                _run_prep_and_qa(con, ch, log, heal_aware=True,
                                 reuse_clean=True, semantic=True)
            log.write("[heal] no narration-healable flags remain\n")
            return
        if project is None:                   # load cloud/ollama routing lazily
            cfg, project, location = _beats_cfg()
            env = _series_env(con, ch["series_id"])
        log.write(f"[heal] cycle {cycle}/{_HEAL_MAX}: re-narrating {ncorr} "
                  "flagged group(s) from their panels\n")
        _regen_flagged(ep, cfg, project, location, str(corr), env, log)
        with record_stage(con, chapter_id=ch["id"], stage="planned",
                          series_id=ch["series_id"]):
            if _stream([PY, str(REPO / "tools" / "timeline_planner.py"),
                        "--groups", str(ep / "manifest.groups.json"),
                        "--beats", str(ep / "manifest.beats.json"),
                        "--script", str(ep / "manifest.script.json"),
                        "--vision", str(ep / "manifest.vision.json"),
                        "--out", str(ep / "render.plan.json"),
                        "--mode", "narrated", "--min-cut-sec", "3.5"], log) != 0:
                raise RuntimeError("timeline_planner (heal) failed")
        cycle_semantic = semantic_heal
        used_fast_qa = used_fast_qa or not cycle_semantic
        _run_prep_and_qa(con, ch, log, heal_aware=True, reuse_clean=True,
                         semantic=cycle_semantic)
    if used_fast_qa:
        log.write("[heal] final semantic QA scan after hitting heal cap\n")
        _run_prep_and_qa(con, ch, log, heal_aware=True,
                         reuse_clean=True, semantic=True)
    log.write(f"[heal] hit the {_HEAL_MAX}-cycle cap\n")


# QA ERROR codes that re-narration CAN'T fix — the panel itself is the problem
# (blank/void crop, a leaked dead caption box, bubble text the blanker missed).
# The last-resort heal DROPS those panels instead of failing the whole chapter.
_VISUAL_DROPPABLE = {"blank_crop", "dead_box_leak", "visible_text", "ghost_text"}


def _heal_visual_drops(con: sqlite3.Connection, ch: Dict[str, Any], ep: Path,
                       log: TextIO) -> None:
    """Last-resort heal for QA ERRORs re-narration can't touch: DROP the
    offending panel (manual_drops.json, the same mechanism as the dashboard drop
    button) + re-prep. Bounded to <=25% of cuts so a chapter is never gutted —
    a slightly shorter recap beats a dead chapter. Runs only after narration
    heal; spotless chapters never reach here."""
    mdp = ep / "manual_drops.json"
    for _pass in range(2):
        try:
            report = json.loads((ep / "prep_qa.json").read_text())
        except Exception:
            return
        flags = report.get("flags") or []
        drop = {Path(str(f.get("scene"))).name for f in flags
                if f.get("severity") == "ERROR"
                and f.get("code") in _VISUAL_DROPPABLE and f.get("scene")}
        if not drop:
            return                            # no drop-able visual error left
        n_cuts = int(report.get("n_cuts") or 0)
        cap = max(3, int(0.25 * n_cuts))
        if len(drop) > cap:
            log.write(f"[visual-heal] {len(drop)} drop-able ERROR panels exceed "
                      f"the {cap}/{n_cuts}-cut cap — leaving for manual review\n")
            return
        try:
            existing = set(map(str, json.loads(mdp.read_text()))) if mdp.exists() \
                else set()
        except Exception:
            existing = set()
        if drop <= existing:                  # already dropped, still flagged
            return
        mdp.write_text(json.dumps(sorted(existing | drop), indent=1))
        log.write(f"[visual-heal] auto-dropping {len(drop)} QA-flagged panel(s) "
                  f"+ re-prepping: {sorted(drop)}\n")
        _run_prep_and_qa(con, ch, log, heal_aware=True, reuse_clean=True)


def _h_prepare(con: sqlite3.Connection, job: Dict[str, Any], log: TextIO) -> None:
    """Everything up to a reviewable QA REPORT, with NO voiceover:
    chain → scripted, then an ESTIMATED-timing plan (planner runs without
    audio), prep, and the QA scan. The chapter lands as 'QA ready' for the
    story approval."""
    ch = _chapter(con, job["chapter_id"])
    with record_stage(con, chapter_id=ch["id"], stage="chain:scripted",
                      series_id=ch["series_id"]):
        rc = _stream([PY, "-m", "studio", "fetch", str(ch["series_id"]),
                      "--chapters", str(int(ch["number"]))], log)
        if rc != 0:
            raise RuntimeError(f"studio fetch exited {rc}")
        rc = _stream([PY, "-m", "studio", "run", str(ch["series_id"]),
                      "--chapters", str(int(ch["number"])),
                      "--until", "scripted"], log,
                     env=_series_env(con, ch["series_id"]))
        if rc != 0:
            raise RuntimeError(f"studio run exited {rc}")
    ch = _chapter(con, job["chapter_id"])  # ep_dir may have been set
    ep = Path(ch["ep_dir"] or "")

    def _plan() -> None:
        with record_stage(con, chapter_id=ch["id"], stage="planned",
                          series_id=ch["series_id"]):
            rc = _stream([PY, str(REPO / "tools" / "timeline_planner.py"),
                          "--groups", str(ep / "manifest.groups.json"),
                          "--beats", str(ep / "manifest.beats.json"),
                          "--script", str(ep / "manifest.script.json"),
                          "--vision", str(ep / "manifest.vision.json"),
                          "--out", str(ep / "render.plan.json"),
                          "--mode", "narrated", "--min-cut-sec", "3.5"], log)
            if rc != 0:
                raise RuntimeError(f"timeline_planner exited {rc}")

    _plan()
    _run_prep_and_qa(con, ch, log, heal_aware=True)
    # AUTO-HEAL: re-narrate ONLY the QA-flagged groups from their panels
    # (corrections + --resume keep every good line), re-derive and re-QA in a
    # loop until green. A failing line is never DROPPED to satisfy QA. Runs
    # unconditionally — it self-gates on the corrections map, so it also catches
    # chrome/meta leaks that QA only WARNs about (the channel voices no chrome).
    _heal_to_green(con, ch, ep, log)
    # then the LAST-RESORT visual heal: errors re-narration can't fix (blank
    # crops, dead-box leaks, missed bubble text) get the offending panel dropped
    # + a re-prep, bounded so a chapter is never gutted.
    _heal_visual_drops(con, ch, ep, log)
    codes = _qa_error_codes(ep)
    blocking = codes & _CRITICAL_QA_CODES
    if blocking:
        raise RuntimeError(
            f"prep-QA has BLOCKING errors after auto-heal ({sorted(blocking)}) — "
            f"open the report in {ep}")
    if codes:
        log.write("[qa] proceeding with non-blocking QA flags after heal "
                  f"(cosmetic, flagged for review): {sorted(codes)}\n")
    # bulk "run range to stage X": a prepare job may carry auto_to (voice|video).
    # The bulk request IS the story approval, so advance past the voice gate up to
    # the requested target — QA still had to be green (we raised above otherwise).
    auto_to = (job.get("payload") or {}).get("auto_to")
    if auto_to in ("voice", "video") and not gates._has_approval(
            con, "voice", chapter_id=ch["id"]):
        log.write(f"[bulk] auto_to={auto_to}: QA green -> voiceover queued\n")
        gates.approve(con, "voice", chapter_id=ch["id"], note="bulk")
        jobs.enqueue(con, "voiceover", chapter_id=ch["id"],
                     payload={"auto_to": auto_to})
    elif _autopilot_clean(con, ch) and not gates._has_approval(
            con, "voice", chapter_id=ch["id"]):
        log.write("[autopilot] QA spotless → story auto-approved, "
                  "voiceover queued\n")
        gates.approve(con, "voice", chapter_id=ch["id"], note="autopilot")
        jobs.enqueue(con, "voiceover", chapter_id=ch["id"])


def _h_voiceover(con: sqlite3.Connection, job: Dict[str, Any],
                 log: TextIO) -> None:
    """After STORY approval: voice the narration, rebuild the plan with real
    audio timing, re-prep, machine-re-scan QA (must stay green), and build a
    listenable preview. The user then approves the VOICEOVER, which triggers
    the render."""
    allowed, why = gates.voice_allowed(con, job["chapter_id"])
    if not allowed:
        raise RuntimeError(f"voiceover blocked: {why}")
    ch = _chapter(con, job["chapter_id"])
    with record_stage(con, chapter_id=ch["id"], stage="voiced",
                      series_id=ch["series_id"]):
        rc = _stream([PY, "-m", "studio", "run", str(ch["series_id"]),
                      "--chapters", str(int(ch["number"])),
                      "--until", "planned"], log,
                     env=_series_env(con, ch["series_id"]))
        if rc != 0:
            raise RuntimeError(f"studio run exited {rc}")
    ch = _chapter(con, job["chapter_id"])
    # reuse_clean=True: panels are UNCHANGED from prepare (voicing alters only
    # audio/timing, not art), so reuse the prepare-time per-panel visual-judge
    # verdicts (.cut_judge_cache.json) instead of re-paying ~1 gemma call/panel
    # — the dominant voiceover-time render-prep cost. The plan is still rebuilt
    # with REAL audio timing; junk-drops stay identical to the reviewed prepare
    # state (verdicts are per-panel + stable). Same mechanism the heal cycles use.
    _run_prep_and_qa(con, ch, log, reuse_clean=True)
    ep = Path(ch["ep_dir"] or "")
    clips = sorted((ep / "tts" / "clips").glob("*.wav"))
    if clips:
        import subprocess as _sp
        rdir = ep / "render"
        rdir.mkdir(parents=True, exist_ok=True)
        # a short pause BETWEEN groups so beats don't slam together (the preview
        # was a gapless concat). format-match the clips so the concat demuxer is
        # happy.
        sil = rdir / "_gap.wav"
        try:
            meta = _sp.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=sample_rate,channels", "-of",
                 "csv=p=0", str(clips[0])],
                capture_output=True, text=True).stdout.strip().split(",")
            sr = meta[0] if meta and meta[0] else "24000"
            cl = "stereo" if (len(meta) > 1 and meta[1] == "2") else "mono"
        except Exception:
            sr, cl = "24000", "mono"
        _stream(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                 f"anullsrc=r={sr}:cl={cl}", "-t", "0.4", str(sil)], log)
        lines, prev = [], None
        for c in clips:
            g = c.name.split("_")[0]                 # 'g0001' from g0001_p00.wav
            if prev is not None and g != prev and sil.exists():
                lines.append(f"file '{sil}'\n")
            lines.append(f"file '{c}'\n")
            prev = g
        lst = rdir / "voice_preview.txt"
        lst.write_text("".join(lines))
        _stream(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                 "-safe", "0", "-i", str(lst), "-codec:a", "libmp3lame",
                 "-b:a", "96k", str(rdir / "voice_preview.mp3")],
                log)
    auto_to = (job.get("payload") or {}).get("auto_to")
    if auto_to == "video" and not gates._has_approval(
            con, "render", chapter_id=ch["id"]):
        log.write("[bulk] auto_to=video: voiced QA green -> render queued\n")
        gates.approve(con, "render", chapter_id=ch["id"], note="bulk")
        jobs.enqueue(con, "render_segment", chapter_id=ch["id"],
                     payload={"branding": "both"})
    elif _autopilot_clean(con, ch) and not gates._has_approval(
            con, "render", chapter_id=ch["id"]):
        log.write("[autopilot] voiced QA spotless → render queued\n")
        gates.approve(con, "render", chapter_id=ch["id"], note="autopilot")
        jobs.enqueue(con, "render_segment", chapter_id=ch["id"],
                     payload={"branding": "both"})


def _series_branding_dir(series_id: int) -> Path:
    return REPO / "assets" / "branding" / "series" / str(series_id)


def _h_branding_segments(con: sqlite3.Connection, job: Dict[str, Any],
                         log: TextIO) -> None:
    """Render the per-series standalone intro.mp4/outro.mp4 ONCE (intro
    overlay held over the series thumbnail; outro end-card draws itself).
    Afterwards every video at every granularity is a pure concat."""
    import shutil as _sh

    import cv2

    from studio.dashboard import bundles as _b
    sid = job["series_id"]
    row = con.execute("SELECT ep_dir FROM chapter WHERE series_id=? AND "
                      "ep_dir IS NOT NULL LIMIT 1", (sid,)).fetchone()
    if not row:
        raise RuntimeError("series has no processed chapter to source the "
                           "thumbnail from")
    thumb = Path(row[0]) / "render" / "thumbnail.png"
    if not thumb.exists():
        raise RuntimeError(f"series thumbnail missing: {thumb} — generate "
                           "it first (tools/thumbnail_gen.py)")
    bdir = _series_branding_dir(sid)
    bdir.mkdir(parents=True, exist_ok=True)
    _sh.copy(thumb, bdir / "thumb.jpg")
    img = cv2.imread(str(bdir / "thumb.jpg"))
    h, w = img.shape[:2]
    intro_wav = REPO / "assets" / "branding" / "origin-power" / "intro.wav"
    outro_wav = REPO / "assets" / "branding" / "origin-power" / "outro.wav"
    import wave

    def _dur(p):
        with wave.open(str(p)) as f:
            return f.getnframes() / f.getframerate()

    iplan = _b.branding_intro_plan("thumb.jpg", w, h, intro_dur=_dur(intro_wav))
    oplan = _b.branding_outro_plan(outro_dur=_dur(outro_wav))
    (bdir / "intro.plan.json").write_text(json.dumps(iplan))
    (bdir / "outro.plan.json").write_text(json.dumps(oplan))
    for name, plan in (("intro", "intro.plan.json"), ("outro", "outro.plan.json")):
        rc = _stream(["npx", "remotion", "render", "src/index.ts",
                      "RecapVideo", str(bdir / f"{name}.mp4"),
                      f"--props={bdir / plan}", f"--public-dir={bdir}",
                      "--concurrency=8", "--crf=22"], log,
                     cwd=str(REPO / "remotion"))
        if rc != 0:
            raise RuntimeError(f"remotion {name} exited {rc}")


def _h_add_series(con: sqlite3.Connection, job: Dict[str, Any],
                  log: TextIO) -> None:
    src = job["payload"].get("source", "")
    url = job["payload"].get("url", "")
    if not src or not url:
        raise RuntimeError("add_series needs source + url")
    rc = _stream([PY, "-m", "studio", "add-series", src, url], log)
    if rc != 0:
        raise RuntimeError(f"add-series exited {rc}")


def _h_chain(con: sqlite3.Connection, job: Dict[str, Any], log: TextIO) -> None:
    """Run pipeline stages for one chapter up to payload['target'] via the
    studio CLI (it owns config, creds, resumability). Targets at or past
    'voiced' cross the narration-review line and need the voice gate."""
    from studio.catalog.models import STATUS_ORDER
    ch = _chapter(con, job["chapter_id"])
    target = job["payload"].get("target", "scripted")
    try:
        crosses_voice = (STATUS_ORDER.index(target)
                         >= STATUS_ORDER.index("voiced"))
    except ValueError:
        crosses_voice = True   # unknown target: fail safe, require approval
    if crosses_voice:
        allowed, why = gates.voice_allowed(con, ch["id"])
        if not allowed:
            raise RuntimeError(f"voiceover blocked: {why}")
    with record_stage(con, chapter_id=ch["id"], stage=f"chain:{target}",
                      series_id=ch["series_id"]):
        rc = _stream([PY, "-m", "studio", "run", str(ch["series_id"]),
                      "--chapters", str(int(ch["number"])),
                      "--until", target], log,
                     env=_series_env(con, ch["series_id"]))
        if rc != 0:
            raise RuntimeError(f"studio run exited {rc}")


def _h_qa_scan(con: sqlite3.Connection, job: Dict[str, Any], log: TextIO) -> None:
    ch = _chapter(con, job["chapter_id"])
    title = _series_title(con, ch["series_id"])
    t0 = time.time()
    rc = _stream([PY, str(REPO / "tools" / "prep_qa.py"),
                  "--episode-dir", ch["ep_dir"] or "",
                  "--series-title", title], log)
    con.execute(
        "INSERT INTO stage_run (chapter_id, stage, duration_sec, ok, "
        "meta_json) VALUES (?,?,?,?, json_object('series_id', ?))",
        (ch["id"], "qa_scan", round(time.time() - t0, 2),
         1 if rc == 0 else 0, ch["series_id"]))
    con.commit()
    if rc != 0:
        raise RuntimeError("prep-QA found ERROR-severity flags "
                           f"(exit {rc}) — see report in {ch['ep_dir']}")


def _h_render_segment(con: sqlite3.Connection, job: Dict[str, Any],
                      log: TextIO) -> None:
    allowed, why = gates.render_allowed(con, job["chapter_id"])
    if not allowed:
        raise RuntimeError(f"render blocked: {why}")
    ch = _chapter(con, job["chapter_id"])
    bdir = _series_branding_dir(ch["series_id"])
    has_branding_segs = (bdir / "intro.mp4").exists()
    branding = job["payload"].get("branding") or (
        "none" if has_branding_segs else "both")
    ep = Path(ch["ep_dir"] or "")
    with record_stage(con, chapter_id=ch["id"], stage="render_segment",
                      series_id=ch["series_id"]):
        rc = _stream([PY, str(REPO / "tools" / "render_prep.py"),
                      "--plan", str(ep / "render.plan.json"),
                      "--scenes-manifest", str(ep / "manifest.scenes.json"),
                      "--episode-dir", str(ep),
                      "--series-title", _series_title(con, ch["series_id"]),
                      "--branding", branding], log)
        if rc != 0:
            raise RuntimeError(f"render_prep exited {rc}")
        out = ep / "render" / f"segment_{branding}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        rc = _stream(["npx", "remotion", "render", "src/index.ts",
                      "RecapVideo", str(out),
                      f"--props={ep / 'render.plan.clean.json'}",
                      f"--public-dir={ep}", "--concurrency=8", "--crf=22"],
                     log, cwd=str(REPO / "remotion"))
        if rc != 0:
            raise RuntimeError(f"remotion exited {rc}")
        if branding == "none" and has_branding_segs:
            # the chapter's standalone SINGLE video = intro + segment + outro
            single = ep / "render" / "single.mp4"
            segs = bundles.wrap_with_branding(
                [str(out)], str(bdir / "intro.mp4"), str(bdir / "outro.mp4"))
            lst = ep / "render" / "single_concat.txt"
            lst.write_text("".join(f"file '{s_}'\n" for s_ in segs))
            rc = _stream(["ffmpeg", "-y", "-loglevel", "error", "-f",
                          "concat", "-safe", "0", "-i", str(lst), "-c",
                          "copy", str(single)], log)
            if rc != 0:
                raise RuntimeError(f"single concat exited {rc}")
        con.execute("UPDATE chapter SET status='rendered' WHERE id=?",
                    (ch["id"],))
        con.commit()


def _h_concat(con: sqlite3.Connection, job: Dict[str, Any], log: TextIO) -> None:
    allowed, why = gates.concat_allowed(con, job["bundle_id"])
    if not allowed:
        raise RuntimeError(f"concat blocked: {why}")
    bid = job["bundle_id"]
    segs = []
    for cid in bundles.bundle_chapters(con, bid):
        ch = _chapter(con, cid)
        rdir = Path(ch["ep_dir"] or "") / "render"
        found = sorted(rdir.glob("segment_*.mp4")) or sorted(rdir.glob("*.mp4"))
        if not found:
            raise RuntimeError(f"chapter {cid} has no rendered segment")
        segs.append(str(found[0]))
    srow = con.execute("SELECT series_id FROM bundle WHERE id=?",
                       (bid,)).fetchone()
    bdir = _series_branding_dir(srow[0]) if srow else None
    if bdir is not None:
        segs = bundles.wrap_with_branding(
            segs, str(bdir / "intro.mp4"), str(bdir / "outro.mp4"))
    out_dir = REPO / "dist" / f"bundle_{bid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bundle.mp4"
    argv, listfile = bundles.concat_cmd(segs, str(out))
    lf = out_dir / "concat.txt"
    lf.write_text(listfile)
    argv[argv.index("LISTFILE")] = str(lf)
    with record_stage(con, chapter_id=None, stage="concat"):
        rc = _stream(argv, log)
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")
    con.execute("UPDATE bundle SET state='concatenated', output_path=? "
                "WHERE id=?", (str(out), bid))
    con.commit()


def _h_discovery_scan(con: sqlite3.Connection, job: Dict[str, Any],
                      log: TextIO) -> None:
    """AniList trends + auto-link source URLs + YouTube coverage per title."""
    from studio.dashboard import discovery
    n = discovery.scan(con, log=lambda *a: (log.write(" ".join(map(str, a))
                                                      + "\n"), log.flush()))
    log.write(f"scanned {n} titles\n")


def _h_refresh(con: sqlite3.Connection, job: Dict[str, Any], log: TextIO) -> None:
    rc = _stream([PY, "-m", "studio", "refresh"]
                 + (["--series", str(job["series_id"])] if job["series_id"]
                    else []), log)
    if rc != 0:
        raise RuntimeError(f"refresh exited {rc}")


def _h_publish_meta(con: sqlite3.Connection, job: Dict[str, Any],
                    log: TextIO) -> None:
    """BUNDLE (video) metadata: arc title + description + Parts (YouTube-chapter
    timestamps) + pinned comment, generated from ALL the bundle's chapters — a
    single chapter can't carry the arc. Copyright-safe (real name only in the
    pinned comment). The thumbnail is series-level (1 per manhwa), generated
    separately. $0 (local Gemma)."""
    bid = job["bundle_id"]
    srow = con.execute("SELECT series_id FROM bundle WHERE id=?", (bid,)).fetchone()
    if not srow:
        raise RuntimeError(f"bundle {bid} not found")
    sid = srow[0]
    title = _series_title(con, sid)
    eps: List[str] = []
    for cid in bundles.bundle_chapters(con, bid):
        ch = _chapter(con, cid)
        if ch and ch.get("ep_dir") and (
                Path(ch["ep_dir"]) / "manifest.beats.json").exists():
            eps.append(str(ch["ep_dir"]))
    if not eps:
        raise RuntimeError(f"bundle {bid} has no chapters with beats yet")
    out_dir = REPO / "dist" / f"bundle_{bid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = _stream([PY, str(REPO / "tools" / "publish_concept.py"),
                  "--episode-dirs", ",".join(eps), "--series-title", title,
                  "--out", str(out_dir / "publish_meta.json")],
                 log, env=_series_env(con, sid))
    if rc != 0:
        raise RuntimeError(f"publish_concept exited {rc}")


def _h_series_thumbnail(con: sqlite3.Connection, job: Dict[str, Any],
                        log: TextIO) -> None:
    """ONE thumbnail per manhwa (series), reused across every video. Built from
    the arc's CLIMAX — the highest-intensity beat across the series' processed
    chapters: a $0 local-Gemma concept (style + hook + climax refs) -> a
    text-free Nano-Banana background -> branded overlay -> dist/series_<id>/
    thumbnail_yt.jpg. Copyright-safe (no licensed name in the image). A fresh
    build clears any prior approval so the APPROVED badge always refers to the
    image currently on disk."""
    sid = job["series_id"]
    if not sid:
        raise RuntimeError("series_thumbnail needs series_id")
    title = _series_title(con, sid)
    rows = con.execute(
        "SELECT ep_dir FROM chapter WHERE series_id=? AND ep_dir IS NOT NULL "
        "ORDER BY number", (sid,)).fetchall()
    eps = [r[0] for r in rows
           if r[0] and (Path(r[0]) / "manifest.beats.json").exists()]
    if not eps:
        raise RuntimeError("no processed chapters yet — prepare at least one "
                           "chapter (narration) before generating a thumbnail")
    out_dir = REPO / "dist" / f"series_{sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_path = out_dir / "concept.json"
    env = _series_env(con, sid)
    with record_stage(con, chapter_id=None, stage="series_thumbnail",
                      series_id=sid):
        # 1) coherent arc concept (style + hook + climax refs) — $0 local Gemma
        rc = _stream([PY, str(REPO / "tools" / "publish_concept.py"),
                      "--episode-dirs", ",".join(eps), "--series-title", title,
                      "--out", str(concept_path)], log, env=env)
        if rc != 0:
            raise RuntimeError(f"publish_concept exited {rc}")
        concept = json.loads(concept_path.read_text())
        ci = int(concept.get("climax_chapter_index") or 0)
        ref_ep = eps[ci] if 0 <= ci < len(eps) else eps[0]
        # 2) text-free art (Nano Banana, ~$0.13) + deterministic branded overlay
        rc = _stream([PY, str(REPO / "tools" / "thumbnail_build.py"),
                      "--concept", str(concept_path),
                      "--ref-episode-dir", ref_ep,
                      "--out-dir", str(out_dir)], log, env=env)
        if rc != 0:
            raise RuntimeError(f"thumbnail_build exited {rc}")
    # 3) a freshly built thumbnail must be re-approved before it's "the one"
    con.execute("DELETE FROM approval WHERE gate='thumbnail' AND series_id=?",
                (sid,))
    con.commit()


HANDLERS: Dict[str, Callable[[sqlite3.Connection, Dict[str, Any], TextIO], None]] = {
    "discovery_scan": _h_discovery_scan,
    "prepare": _h_prepare,
    "voiceover": _h_voiceover,
    "publish_meta": _h_publish_meta,
    "series_thumbnail": _h_series_thumbnail,
    "add_series": _h_add_series,
    "branding_segments": _h_branding_segments,
    "chain": _h_chain,
    "qa_scan": _h_qa_scan,
    "render_segment": _h_render_segment,
    "concat": _h_concat,
    "refresh": _h_refresh,
}


def run_once(con: sqlite3.Connection, *, handlers=None,
             log_dir: str = "logs/jobs",
             lane: "str | None" = None) -> bool:
    handlers = HANDLERS if handlers is None else handlers
    job = jobs.claim_next(con, lane=lane)
    if not job:
        return False
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{job['id']}-{job['type']}.log")
    jobs.set_log(con, job["id"], log_path)
    _CUR.job_id = job["id"]                     # for the operator-cancel monitor
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            handler = handlers.get(job["type"])
            if handler is None:
                raise RuntimeError(f"no handler for job type {job['type']!r}")
            handler(con, job, log)
        jobs.finish(con, job["id"], ok=True)
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("\n" + traceback.format_exc())
        # OPERATOR CANCEL: the dashboard marked this running job 'cancelling' and
        # the monitor killed its subprocess -> record it 'cancelled' (NOT failed)
        # so an operator cancel never auto-retries.
        row = con.execute("SELECT state FROM job WHERE id=?",
                          (job["id"],)).fetchone()
        if row and row[0] in ("cancelling", "cancelled"):
            con.execute("UPDATE job SET state='cancelled', "
                        "finished_at=datetime('now'), error='cancelled by "
                        "operator' WHERE id=?", (job["id"],))
            con.commit()
            return True
        # Self-healing: re-enqueue a transiently-failed job (bounded) so an
        # unattended run recovers on its own instead of stalling on a blip. The
        # retry jumps to the FRONT (STUDIO_RETRY_PRIORITY, default 1) so a failed
        # chapter is re-attempted BEFORE new queued work (user directive). After
        # max attempts it stays failed and is surfaced on the Series tab for a
        # manual reload (see jobs.failed_chapters).
        max_attempts = int(os.environ.get("STUDIO_JOB_MAX_ATTEMPTS", "3"))
        retry_priority = int(os.environ.get("STUDIO_RETRY_PRIORITY", "1"))
        attempt = int((job.get("payload") or {}).get("_attempt", 0))
        if job["type"] != "heartbeat" and attempt + 1 < max_attempts:
            payload = dict(job.get("payload") or {})
            payload["_attempt"] = attempt + 1
            rid = jobs.enqueue(con, job["type"],
                               series_id=job.get("series_id"),
                               chapter_id=job.get("chapter_id"),
                               bundle_id=job.get("bundle_id"),
                               payload=payload,
                               priority=retry_priority)
            jobs.finish(con, job["id"], ok=False,
                        error=f"{str(e)[:260]} — auto-retry {attempt + 2}/{max_attempts} (job {rid})")
        else:
            jobs.finish(con, job["id"], ok=False, error=str(e)[:300])
    finally:
        _CUR.job_id = None
    return True


def _heartbeat(con: sqlite3.Connection) -> None:
    con.execute("UPDATE job SET started_at=datetime('now') "
                "WHERE type='heartbeat'")
    if con.total_changes == 0 or con.execute(
            "SELECT COUNT(*) FROM job WHERE type='heartbeat'").fetchone()[0] == 0:
        con.execute("INSERT INTO job (type, state, started_at) "
                    "VALUES ('heartbeat','running',datetime('now'))")
    con.commit()


def requeue_orphans(con: sqlite3.Connection) -> int:
    """Jobs left 'running' by a dead worker process would block their lane
    forever — at boot (one worker per host) they all go back to queued."""
    cur = con.execute("UPDATE job SET state='queued', started_at=NULL "
                      "WHERE state='running' AND type!='heartbeat'")
    # a cancel in flight when the worker died -> just record it cancelled
    con.execute("UPDATE job SET state='cancelled', finished_at=datetime('now') "
                "WHERE state='cancelling'")
    con.commit()
    return cur.rowcount


def main(db_path: str = "studio.db") -> int:
    from studio.catalog.db import connect
    con = connect(db_path)
    import threading
    orphans = requeue_orphans(con)
    if orphans:
        print(f"[worker] requeued {orphans} orphaned running job(s)")
    widths = dict(jobs.LANE_WIDTH)
    print(f"[worker] lanes {widths} on {db_path} — ctrl-c to stop")
    threading.Thread(target=_cancel_monitor, args=(db_path,),
                     daemon=True).start()

    def lane_loop(lane: str) -> None:
        lcon = connect(db_path)
        while True:
            if not run_once(lcon, lane=lane):
                time.sleep(2)

    try:
        for lane, width in widths.items():
            extra = width - 1 if lane == "gpu" else width
            for _ in range(max(0, extra)):
                threading.Thread(target=lane_loop, args=(lane,),
                                 daemon=True).start()
        while True:                      # gpu slot #1 on the main thread
            _heartbeat(con)
            if not run_once(con, lane="gpu"):
                time.sleep(2)
    except KeyboardInterrupt:
        print("\n[worker] stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "studio.db"))
