"""
studio/pipeline.py

Per-chapter stage orchestration.

Drives a downloaded chapter through deterministic pipeline stages, advancing
catalog status after each.  Designed to be RESUMABLE (re-run after failure
restarts at the failed stage) and IDEMPOTENT (re-run on a completed chapter
does nothing).
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Callable

from studio.catalog import repo
from studio.catalog.models import STATUS_ORDER, fail_status, next_status, Chapter
from studio.config import Config


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MissingCredential(Exception):
    """Raised when a stage requires a credential that is not available."""

    def __init__(self, stage: str, what_to_set: str) -> None:
        self.stage = stage
        self.what_to_set = what_to_set
        super().__init__(
            f"Stage '{stage}' requires credential: {what_to_set}"
        )


# ---------------------------------------------------------------------------
# Tool runner (single monkeypatch point for tests)
# ---------------------------------------------------------------------------

def _run_tool(script_name: str, args_list: list[str], *, python_exe: str = "") -> None:
    """Run a tool script via a Python interpreter.

    ``script_name`` is the bare filename (e.g. ``chunk_stitch_adaptive.py``).
    The script is looked up relative to the ``tools/`` directory at repo root.
    ``python_exe`` overrides the interpreter (used for the local-TTS venv, whose
    torch pin conflicts with YOLO's); empty = the pipeline's own interpreter.
    """
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "tools" / script_name
    exe = python_exe or sys.executable
    cmd = [exe, str(script_path)] + args_list
    # The break-fixed tools `import studio.paths`, but they run as standalone
    # scripts here, so the repo root must be on PYTHONPATH for the subprocess.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(cmd, check=True, env=env)
    # check=True raises CalledProcessError on non-zero exit


# ---------------------------------------------------------------------------
# Credential checkers
# ---------------------------------------------------------------------------

import os

def _check_vertex_adc() -> None:
    """Raise MissingCredential if Vertex AI ADC is not configured."""
    # GOOGLE_APPLICATION_CREDENTIALS or gcloud default credentials file
    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if not creds_file and not adc_path.exists():
        raise MissingCredential(
            "beated",
            "GOOGLE_APPLICATION_CREDENTIALS or `gcloud auth application-default login`",
        )


def _check_openai() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise MissingCredential("scripted", "OPENAI_API_KEY")


def _check_elevenlabs() -> None:
    if not os.environ.get("ELEVENLABS_API_KEY"):
        raise MissingCredential("voiced", "ELEVENLABS_API_KEY")


# ---------------------------------------------------------------------------
# Derived-manifest staleness (studio/deps.py via manifest_freshness)
# ---------------------------------------------------------------------------

def _artifact_is_stale(ep_dir: Path, artifact: str) -> bool:
    """True iff *artifact* exists but is stale per its deps-declared inputs.

    The edge compare (sha stamps > mtime fallback) lives in ONE place —
    tools/manifest_freshness.artifact_is_stale — and is reused here, not
    duplicated. tools/ is not a package, so shim it onto sys.path the way
    studio/dashboard/app.py does (computed from __file__, deliberately NOT
    _REPO_ROOT, which tests monkeypatch to a nonexistent root)."""
    tools_dir = str(Path(__file__).resolve().parent.parent / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from manifest_freshness import artifact_is_stale
    return artifact_is_stale(str(ep_dir), artifact)


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------
# Each entry: (stage_name, runner_fn, output_marker_relative, next_status_str)
# runner_fn signature: (ep_dir: Path, cfg: Config) -> None
# Stages are keyed by the status that means "this stage has been done".


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ep_paths(ep_dir: Path) -> dict:
    """Canonical manifest/dir paths within an episode directory."""
    return {
        "stitch": ep_dir / "manifest.stitch.json",
        "chunks": ep_dir / "stitch_chunks",
        "panels": ep_dir / "manifest.panels.json",
        "panels_expanded": ep_dir / "manifest.panels.expanded.json",
        "scenes": ep_dir / "scenes",
        "scenes_manifest": ep_dir / "manifest.scenes.json",
        "vision": ep_dir / "manifest.vision.json",
        "groups": ep_dir / "manifest.groups.json",
        "understood": ep_dir / "manifest.panels.understood.json",
        "cast": ep_dir / "manifest.cast.json",
        "beats": ep_dir / "manifest.beats.json",
        "script": ep_dir / "manifest.script.json",
        "tts_dir": ep_dir / "tts",
        "tts_index": ep_dir / "tts" / "tts_index.json",
        "plan": ep_dir / "render.plan.json",
    }


def write_series_manifest(ep_dir, niche_primary, niche_secondary):
    """Persist the per-series niche next to the chapter's other manifests so the
    narration tools (narration_punchup, gemini_narrative_pass) auto-read it."""
    import json, os
    with open(os.path.join(ep_dir, "manifest.series.json"), "w", encoding="utf-8") as f:
        json.dump({"niche_primary": niche_primary or "",
                   "niche_secondary": niche_secondary or ""}, f)


def _stage_stitch(ep_dir: Path, cfg: Config) -> None:
    p = _ep_paths(ep_dir)
    _run_tool("chunk_stitch_adaptive.py",
              ["--episode-dir", str(ep_dir), "--glob", "*.jpg", "--out-dir", str(p["chunks"])])


def _stage_detect(ep_dir: Path, cfg: Config) -> None:
    p = _ep_paths(ep_dir)
    if cfg.detect_backend == "yolo":
        from studio.detect.yolo_panels import detect_panels
        detect_panels(str(p["stitch"]), str(p["panels"]), str(cfg.yolo_weights))
    else:
        raise RuntimeError(
            f"detect_backend '{cfg.detect_backend}' needs Vertex auth; SP1 supports 'yolo'")
    _run_tool("expand_boxes_to_gutters.py",
              ["--stitch-manifest", str(p["stitch"]),
               "--panels-manifest", str(p["panels"]),
               "--out-panels-manifest", str(p["panels_expanded"])])


def _stage_scened(ep_dir: Path, cfg: Config) -> None:
    p = _ep_paths(ep_dir)
    _run_tool("panels_to_scenes.py",
              ["--stitch-manifest", str(p["stitch"]),
               "--panels-manifest", str(p["panels_expanded"]),
               "--out-dir", str(p["scenes"]),
               "--out-manifest", str(p["scenes_manifest"]),
               # Quality: drop near-duplicate crops, skip blank/text-only panels,
               # and trim white OR black margins (keeps content + bubbles).
               # --dedupe-overlap additionally removes overlapping sub-region
               # crops of the same tall panel that perceptual-hash dedupe misses.
               "--dedupe", "--skip-blank", "--trim-margins", "--dedupe-overlap"])
    # SEAM RECONCILE (scene-level, upstream of vision): merge panels a chunk cut
    # bisected into two near-duplicate slices, so the same drawing is never shown
    # twice. In-place rewrite of manifest.scenes.json + scenes/. No new status.
    _run_tool("reconcile_seam_panels.py",
              ["--scenes-manifest", str(p["scenes_manifest"]),
               "--stitch-manifest", str(p["stitch"]),
               "--scenes-dir", str(p["scenes"])])


def _stage_visioned(ep_dir: Path, cfg: Config) -> None:
    # OCR runs on-device via Apple Vision (free) — no Google credential needed.
    p = _ep_paths(ep_dir)
    _run_tool("vision_extract.py",
              ["--scenes-dir", str(p["scenes"]), "--glob", "*.jpg",
               "--out", str(p["vision"]),
               "--ocr-backend", cfg.vision_backend])


def _stage_grouped(ep_dir: Path, cfg: Config) -> None:
    """Understanding-first grouping (replaces the old position/gutter merge):
      Pass 1 panel_understand — describe EVERY panel multimodally (full coverage
                                by construction).
      Pass 2 story_group     — group by that understanding into story-sized beats
                                with flashback/scene tags.
    Output marker stays manifest.groups.json (byte-compatible shots[]). Honors
    .narration_keepbase (reuse existing groups so a kept narration stays aligned)."""
    import json
    p = _ep_paths(ep_dir)
    if (ep_dir / ".narration_keepbase").exists() and p["groups"].exists():
        print(f"[grouped] keep-base present -> reuse {p['groups'].name}, "
              "skip re-understanding/re-grouping")
        return
    understood = ep_dir / "manifest.panels.understood.json"
    if cfg.beats_backend == "ollama":
        backend = ["--backend", "ollama", "--ollama-model", cfg.beats_model]
    else:
        keys = _REPO_ROOT / "keys" / "gcp-vision.json"
        if keys.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(keys)
            project = json.loads(keys.read_text()).get("project_id", "")
        else:
            _check_vertex_adc()
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        backend = ["--backend", "vertex", "--model", cfg.beats_model,
                   "--project", project, "--location", location]
    understand_args = ["--vision-manifest", str(p["vision"]),
                       "--out", str(understood), "--resume"] + backend
    if cfg.beats_backend == "ollama":
        # Parallelize the per-panel understand to THIS machine's OLLAMA_NUM_PARALLEL
        # (Mini=4, Air=2) — measured ~2x on the chapter's slowest stage. A no-op
        # (sequential) when OLLAMA_NUM_PARALLEL is unset/1, and capped so a
        # mis-set env can't oversubscribe the GPU into OOM.
        try:
            conc = max(1, min(int(os.environ.get("OLLAMA_NUM_PARALLEL", "1") or "1"), 6))
        except ValueError:
            conc = 1
        if conc > 1:
            understand_args += ["--concurrency", str(conc)]
    _run_tool("panel_understand.py", understand_args)
    _run_tool("story_group.py",
              ["--understood", str(understood),
               "--vision-manifest", str(p["vision"]),
               "--out", str(p["groups"])] + backend)


def _stage_beated(ep_dir: Path, cfg: Config) -> None:
    # Prefer the repo's gcp service-account key for Vertex Gemini auth (no gcloud
    # needed). A service account can only authenticate its OWN project, so use
    # the project_id baked into the key, not whatever GOOGLE_CLOUD_PROJECT says.
    import json
    keys = _REPO_ROOT / "keys" / "gcp-vision.json"
    if keys.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(keys)
        project = json.loads(keys.read_text()).get("project_id", "")
    else:
        _check_vertex_adc()
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    p = _ep_paths(ep_dir)
    # keep-base: reuse the EXISTING beats' exact wording as the grounded base
    # (no LLM regeneration), so a hand-picked / approved descriptive take is
    # preserved verbatim instead of being re-rolled differently on every
    # re-prepare. The persona punchup below still (re)applies the channel voice
    # + source scrub. This is how a restored or frozen narration survives the
    # pipeline. Drop the marker (or delete beats) to regenerate from scratch.
    keep_base = (ep_dir / ".narration_keepbase").exists() and p["beats"].exists()
    if keep_base:
        print(f"[beated] keep-base marker present -> reuse {p['beats'].name}, "
              "skipping cast + beats regeneration")
    else:
        # Whole-chapter STORY PASS runs FIRST: read the chapter's own dialogue
        # once, in reading order, and get the plot — synopsis, cast with fates,
        # events with actor->target and the proving line. Measured on nano ch1:
        # one call, ~3 min, correct; the per-panel path spent ~79 min and got
        # the chapter's central kill backwards.
        #
        # It precedes cast_builder because its NAMES are read from what the
        # characters actually say: cast_builder alone produced canonical_name
        # 'our protagonist' while "PRINCE CHEON" sat in the dialogue (so the
        # audience never heard his name), and it omitted the stranger entirely
        # (so nothing could resolve him and the narration called him "our
        # guy"). Fail-soft: without it the ledger falls back to per-window
        # arbitration and cast_builder to its own judgement.
        chapter_story = ep_dir / "manifest.chapter_story.json"
        story_stale = (chapter_story.exists()
                       and _artifact_is_stale(ep_dir,
                                              "manifest.chapter_story.json"))
        if not chapter_story.exists() or story_stale:
            story_args = ["--vision-manifest", str(p["vision"]),
                          "--understood", str(p["understood"]),
                          "--out", str(chapter_story),
                          "--project", project, "--location", location,
                          "--model", cfg.beats_model]
            story_args += (["--backend", "ollama"]
                           if cfg.beats_backend == "ollama"
                           else ["--backend", "vertex"])
            try:
                _run_tool("story_pass.py", story_args)
            except Exception as e:      # never block the chapter on it
                print(f"[beated] story pass FAILED ({e}) -> the ledger falls "
                      "back to per-window arbitration")
        cast_stale = (p["cast"].exists()
                      and _artifact_is_stale(ep_dir, "manifest.cast.json"))
        if cast_stale:
            print("[beated] manifest.cast.json predates its groups/vision "
                  "inputs -> rebuilding cast (stale names would leak into "
                  "the narration)")
        if not p["cast"].exists() or cast_stale:
            # One call → chapter cast registry (manifest.cast.json) so the
            # narration names the same character consistently. Skipped when the
            # file exists AND is fresh w.r.t. its deps-declared inputs, so a
            # beated retry never re-pays for it — but a re-grouped chapter
            # never reuses a cast built from the old grouping.
            cast_args = ["--groups-manifest", str(p["groups"]),
                         "--vision-manifest", str(p["vision"]),
                         "--out", str(p["cast"]),
                         # recurring-figure coverage: every figure the per-panel
                         # analysis saw repeatedly must land in the cast
                         "--understood", str(p["understood"]),
                         # names + completeness from the chapter's dialogue;
                         # this pass supplies the appearance it cannot see
                         "--chapter-story", str(chapter_story),
                         "--project", project, "--location", location,
                         "--model", cfg.beats_model]
            if cfg.beats_backend == "ollama":
                cast_args += ["--backend", "ollama"]
            _run_tool("cast_builder.py", cast_args)
        # Story-state ledger (2026-07-20): projects the chapter story onto
        # per-beat facts (deaths propagate, dead role-holders' titles get
        # banned) — the record the writer, identity gate, and QA all read.
        # Skip-iff-fresh like cast.
        ledger = ep_dir / "manifest.ledger.json"
        ledger_stale = (ledger.exists()
                        and _artifact_is_stale(ep_dir, "manifest.ledger.json"))
        if not ledger.exists() or ledger_stale:
            ledger_args = ["--understood", str(p["understood"]),
                           "--groups", str(p["groups"]),
                           "--cast", str(p["cast"]),
                           "--chapter-story", str(chapter_story),
                           "--out", str(ledger),
                           "--project", project, "--location", location,
                           "--model", cfg.beats_model]
            if cfg.beats_backend == "ollama":
                ledger_args += ["--backend", "ollama"]
            else:
                ledger_args += ["--backend", "vertex"]
            _run_tool("story_ledger.py", ledger_args)
        beats_args = ["--groups-manifest", str(p["groups"]),
                      "--vision-manifest", str(p["vision"]),
                      "--out", str(p["beats"]),
                      "--project", project, "--location", location,
                      "--model", cfg.beats_model,
                      "--cast", str(p["cast"]),
                      # chapter spine (logline + arc) from story_group -> beats
                      # connect into one story instead of isolated panel captions
                      "--story", str(ep_dir / "manifest.story.json"),
                      "--understood", str(p["understood"]),
                      # chapter fact record -> per-beat FACTS block + gate
                      "--ledger", str(ep_dir / "manifest.ledger.json"),
                      # adaptive flow segments vs legacy per_panel — passed
                      # explicitly so config (not the tool's env default) rules
                      "--segmentation", cfg.segmentation or "adaptive"]
        if cfg.beats_backend == "ollama":
            beats_args += ["--backend", "ollama",
                           "--ollama-model", cfg.beats_model]
        _run_tool("gemini_narrative_pass.py",
                  beats_args + [
                   # Panels per group the writer SEES. Not cheap: measured on
                   # the MLX shim, a writer call costs ~69s at 6 images vs
                   # ~49s at 3 (~30% of the chapter's dominant serial stage).
                   # Every panel's understanding still rides scenes_signals,
                   # so an unattached panel is described, just not re-seen;
                   # scene_selection defaults it to 'keep' and render_prep's
                   # dedup (not this judgment) is what drops true twins.
                   "--max-images-per-group", "3"])
    if (cfg.punchup or "off") != "off":
        # persona pass over the grounded beats, in place: narration gets the
        # channel voice, narration_plain keeps the grounded line, and groups
        # carrying captions reject rewrites that drop the caption words.
        punch_args = ["--beats", str(p["beats"]), "--out", str(p["beats"]),
                      "--cast", str(p["cast"]),
                      "--story", str(ep_dir / "manifest.story.json"),
                      "--episode-dir", str(ep_dir),
                      "--humor", cfg.punchup]
        if cfg.beats_backend == "ollama":
            punch_args += ["--backend", "ollama",
                           "--ollama-model", cfg.beats_model]
        else:
            punch_args += ["--backend", "vertex", "--model", cfg.beats_model,
                           "--project", project, "--location", location]
        _run_tool("narration_punchup.py", punch_args)


def _stage_scripted(ep_dir: Path, cfg: Config) -> None:
    p = _ep_paths(ep_dir)
    src = cfg.narration_source or "gemini_verbatim"
    args = ["--beats", str(p["beats"]), "--vision", str(p["vision"]), "--out", str(p["script"]),
            "--model", cfg.script_model, "--narration-source", src]
    if src == "gemini_verbatim":
        # Deterministic materialization of the image-grounded Gemini narration
        # (A/B winner) — zero LLM calls, so no OpenAI credential gate. --cast
        # keeps proper nouns cased when shout-caps OCR dialogue is normalized.
        args += ["--cast", str(p["cast"])]
    else:
        _check_openai()
    _run_tool("script_expander.py", args)

    # ADVERTISER-SAFETY: the narration in manifest.script.json is now FINAL (the
    # exact text the voiced stage reads from sections[].tts_paragraphs_v3). Run
    # the sanitize+reframe pass over it IN PLACE — deterministic safe swaps, then
    # an LLM reframe of any flagged/blocked line (softened per the denylist
    # notes) using the same Gemma/Vertex backend the beated stage resolved, then
    # re-sanitize. It writes manifest.sanitize.json; _stage_voiced refuses to
    # voice when that marker lists unresolved blocks. ON by default (safety).
    if cfg.narration_sanitize:
        _run_sanitize_pass(ep_dir, cfg, p)


def _run_sanitize_pass(ep_dir: Path, cfg: Config, p: dict) -> None:
    """Run narration_sanitize_pass over the script manifest. The reframe LLM
    backend mirrors _stage_beated (ollama Gemma, or Vertex via the repo SA key
    project). Seed = ep dir name so swap rotation is deterministic per chapter."""
    import json
    sanitize_args = ["--script", str(p["script"]),
                     "--seed", ep_dir.name,
                     "--marker", str(ep_dir / "manifest.sanitize.json")]
    if cfg.beats_backend == "ollama":
        sanitize_args += ["--reframe-backend", "ollama",
                          "--reframe-model", cfg.beats_model]
    else:
        keys = _REPO_ROOT / "keys" / "gcp-vision.json"
        if keys.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(keys)
            project = json.loads(keys.read_text()).get("project_id", "")
        else:
            _check_vertex_adc()
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        sanitize_args += ["--reframe-backend", "vertex",
                          "--reframe-model", cfg.beats_model,
                          "--project", project, "--location", location]
    # exit 2 = UNRESOLVED blocks remain. We DON'T fail the scripted stage on
    # that (the marker is written either way and the QA/voiced gate enforces it);
    # but a genuine crash (missing backend, bad manifest) must surface. The tool
    # only returns {0,2}; raise on anything else.
    try:
        _run_tool("narration_sanitize_pass.py", sanitize_args)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 2:
            print("[scripted] sanitize pass left UNRESOLVED blocks -> "
                  "manifest.sanitize.json written; voiced stage will refuse")
            return
        raise


def _read_sanitize_marker(marker_path: Path) -> "dict | None":
    """Parsed manifest.sanitize.json, or None if missing/unreadable/corrupt.
    Shared by the freshness backstop and _read_sanitize_unresolved so 'can't
    be trusted' means the same thing in both places."""
    import json
    if not marker_path.exists():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sanitize_marker_stale(marker_path: Path, script_path: Path) -> bool:
    """True when manifest.sanitize.json can no longer be trusted to reflect
    the CURRENT manifest.script.json: missing, unparseable, or older than the
    script (a heal/rewrite replaced the script without re-running the
    sanitizer). Callers must treat 'stale' as 'unknown', never as 'clean' —
    fail-closed, matching the voiced gate's own posture."""
    if _read_sanitize_marker(marker_path) is None:
        return True
    try:
        return marker_path.stat().st_mtime < script_path.stat().st_mtime
    except OSError:
        return True


def _read_sanitize_unresolved(marker_path: Path) -> list:
    """Unresolved advertiser-safety blocks from manifest.sanitize.json (written
    by narration_sanitize_pass). Missing/unreadable marker → []. Safe to call
    on its own only AFTER the freshness backstop in _stage_voiced has already
    guaranteed the marker exists and is current — in isolation this function
    cannot distinguish 'clean' from 'never ran'."""
    data = _read_sanitize_marker(marker_path)
    if data is None:
        return []
    return [b for b in (data.get("unresolved_blocks") or []) if isinstance(b, dict)]


def _tts_dispatch(cfg: Config, script_path: Path, tts_dir: Path) -> None:
    """TTS backend dispatch shared by _stage_voiced and worker._h_teaser: any
    local backend (chatterbox[-turbo]/kokoro) runs free with no credential;
    'elevenlabs' is credential-checked. Both call sites build the same argv
    shape from a Config and a script/out-dir pair — only the paths differ."""
    backend = (cfg.tts_backend or "elevenlabs").lower()
    if backend != "elevenlabs":   # any local backend (chatterbox[-turbo]/kokoro)
        # Free local TTS — no credential needed. Same tts_index.json contract.
        args = ["--script", str(script_path), "--out-dir", str(tts_dir),
                "--backend", backend]
        if cfg.tts_voice_ref:
            args += ["--voice-ref", cfg.tts_voice_ref]
        if float(getattr(cfg, "tts_speed", 1.0) or 1.0) != 1.0:
            args += ["--speed", str(cfg.tts_speed)]
        if backend == "kokoro" and cfg.tts_kokoro_voice:
            args += ["--kokoro-voice", cfg.tts_kokoro_voice]
        # Local TTS deps (torch 2.6) conflict with YOLO's torch, so run it in its
        # own venv when configured (config.tts_python); falls back to ours.
        _run_tool("local_tts_from_manifest.py", args, python_exe=cfg.tts_python)
    else:
        _check_elevenlabs()
        voice = os.environ.get("ELEVENLABS_VOICE_ID", "")
        _run_tool("elevenlabs_tts_from_manifest.py",
                  ["--script", str(script_path), "--out-dir", str(tts_dir), "--voice-id", voice])


def _stage_voiced(ep_dir: Path, cfg: Config) -> None:
    p = _ep_paths(ep_dir)
    # ADVERTISER-SAFETY GATE: refuse to spend TTS on a chapter whose narration
    # still carries a hard BLOCK the reframe couldn't soften (slurs, sexual
    # violence, explicit anatomy). The scripted stage records these in
    # manifest.sanitize.json; raising here makes run_chapter set the chapter to
    # 'voiced_failed' with this error (the existing error/status path), so the
    # worker surfaces it and never voices. A clean chapter has no unresolved
    # blocks and proceeds normally.
    if cfg.narration_sanitize:
        marker_path = ep_dir / "manifest.sanitize.json"
        # FRESHNESS BACKSTOP (fail-closed): a heal rewrite (worker._rescript)
        # can replace manifest.script.json without ever re-running the
        # sanitizer, and a voiceover-rewind resume-by-status run SKIPS
        # _stage_scripted entirely once its own marker (manifest.script.json)
        # already exists — so a stale/missing/corrupt sanitize marker must
        # never be read as "clean". Re-run the SAME pass _stage_scripted uses,
        # right here, before ever consulting it.
        if _sanitize_marker_stale(marker_path, p["script"]):
            _run_sanitize_pass(ep_dir, cfg, p)
            if _read_sanitize_marker(marker_path) is None:
                raise RuntimeError(
                    "sanitize marker missing/unreadable after re-run — "
                    "refusing to voice")
        unresolved = _read_sanitize_unresolved(marker_path)
        if unresolved:
            preview = ", ".join(
                f"{b.get('segment_id', '?')}:'{b.get('matched', '')}'"
                for b in unresolved[:5])
            raise RuntimeError(
                f"voiced blocked: narration sanitize left {len(unresolved)} "
                f"unresolved advertiser-safety BLOCK(s) [{preview}] — "
                f"see {marker_path}")
    _tts_dispatch(cfg, p["script"], p["tts_dir"])


def _stage_planned(ep_dir: Path, cfg: Config) -> None:
    # Blender render is a manual follow step; the terminal pipeline output is
    # render.plan.json (produced by timeline_planner, needs no API creds).
    p = _ep_paths(ep_dir)
    _run_tool("timeline_planner.py",
              ["--groups", str(p["groups"]), "--beats", str(p["beats"]),
               "--script", str(p["script"]), "--vision", str(p["vision"]),
               "--tts-index", str(p["tts_index"]),
               "--out", str(p["plan"]), "--mode", "narrated",
               # Each shown picture gets >= 3.5s; excess panels in a shot are dropped.
               "--min-cut-sec", "3.5"])


# Ordered list of (result_status, runner_fn, output_marker_relpath)
# "result_status" = status after this stage completes successfully
_STAGE_TABLE: list[tuple[str, Callable[[Path, Config], None], str]] = [
    ("stitched",  _stage_stitch,   "manifest.stitch.json"),
    ("detected",  _stage_detect,   "manifest.panels.expanded.json"),
    ("scened",    _stage_scened,   "manifest.scenes.json"),
    ("visioned",  _stage_visioned, "manifest.vision.json"),
    ("grouped",   _stage_grouped,  "manifest.groups.json"),
    ("beated",    _stage_beated,   "manifest.beats.json"),
    ("scripted",  _stage_scripted, "manifest.script.json"),
    ("voiced",    _stage_voiced,   "tts/tts_index.json"),
    ("planned",   _stage_planned,  "render.plan.json"),
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_chapter(
    con: sqlite3.Connection,
    chapter: Chapter,
    cfg: Config,
    *,
    now_fn: Callable[[], str],
    until: str | None = None,
) -> None:
    """Drive *chapter* through pipeline stages starting from its current status.

    - RESUMABLE: re-running after a ``*_failed`` status restarts the failed stage.
    - IDEMPOTENT: if all output markers exist and status is already past a stage, skip it.

    Args:
        con: Open catalog DB connection.
        chapter: Chapter dataclass (must have id, status, ep_dir set).
        cfg: Studio config.
        now_fn: Callable returning current ISO timestamp string (injected; never
                calls datetime directly).
    """
    if chapter.ep_dir is None:
        raise ValueError(f"Chapter {chapter.id} has no ep_dir — must be downloaded first")

    ep_dir = Path(chapter.ep_dir)

    # Resolve the current "progress" status — strip _failed suffix if present
    current_status = chapter.status
    if current_status.endswith("_failed"):
        # Resume from the failed stage: treat as if we're at the stage just before it
        failed_stage = current_status[: -len("_failed")]
        # Find the predecessor status (what we need to be at to run failed_stage)
        try:
            failed_idx = STATUS_ORDER.index(failed_stage)
        except ValueError:
            raise ValueError(f"Unknown failed stage '{failed_stage}' in status '{current_status}'")
        # We want to run starting from failed_stage, so effective current status
        # is the one before it
        effective_status = STATUS_ORDER[failed_idx - 1] if failed_idx > 0 else "discovered"
    else:
        effective_status = current_status

    # Walk the stage table and execute stages that haven't been completed yet
    for result_status, runner_fn, marker_rel in _STAGE_TABLE:
        # honor --until: stop once the next stage would pass the target
        if until is not None:
            try:
                if STATUS_ORDER.index(result_status) > STATUS_ORDER.index(until):
                    break
            except ValueError:
                raise ValueError(f"Unknown --until status '{until}'")
        marker_path = ep_dir / marker_rel

        # Skip stages already completed only when the stage's output marker is
        # actually present. A cleaned/restored workspace can leave the catalog at
        # "grouped" while manifest.groups.json is missing; trusting status alone
        # just fails later in beated with a confusing FileNotFoundError.
        try:
            result_idx = STATUS_ORDER.index(result_status)
            current_idx = STATUS_ORDER.index(effective_status)
        except ValueError:
            continue

        if result_idx <= current_idx:
            if marker_path.exists():
                continue
            print(f"[resume] catalog status is past {result_status}, but "
                  f"{marker_rel} is missing; rebuilding that stage")

        # This stage needs to run.  Check idempotency: if marker exists AND
        # the catalog status is already at or past result_status, skip.
        if marker_path.exists() and result_idx <= current_idx:
            # Redundant check (covered above) but kept for clarity
            continue

        # Run the stage
        try:
            runner_fn(ep_dir, cfg)
        except MissingCredential as exc:
            repo.set_chapter_status(
                con,
                chapter.id,
                fail_status(exc.stage),
                error=str(exc),
                updated_at=now_fn(),
            )
            return
        except Exception as exc:
            repo.set_chapter_status(
                con,
                chapter.id,
                fail_status(result_status),
                error=str(exc),
                updated_at=now_fn(),
            )
            return

        # Stage succeeded — advance catalog status
        repo.set_chapter_status(
            con,
            chapter.id,
            result_status,
            updated_at=now_fn(),
        )
        # Update effective_status so next iteration's index comparison is correct
        effective_status = result_status
