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

def _run_tool(script_name: str, args_list: list[str]) -> None:
    """Run a tool script via the current Python interpreter.

    ``script_name`` is the bare filename (e.g. ``chunk_stitch_adaptive.py``).
    The script is looked up relative to the ``tools/`` directory at repo root.
    """
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "tools" / script_name
    cmd = [sys.executable, str(script_path)] + args_list
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
        "beats": ep_dir / "manifest.beats.json",
        "script": ep_dir / "manifest.script.json",
        "tts_dir": ep_dir / "tts",
        "tts_index": ep_dir / "tts" / "tts_index.json",
        "plan": ep_dir / "render.plan.json",
    }


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
               "--out-manifest", str(p["scenes_manifest"])])


def _stage_visioned(ep_dir: Path, cfg: Config) -> None:
    # Google Vision needs a service-account key. Prefer the repo's own key when
    # present — it's canonical and overrides any stale GOOGLE_APPLICATION_CREDENTIALS
    # left in the environment (e.g. an old path from before the repo moved).
    keys = _REPO_ROOT / "keys" / "gcp-vision.json"
    if keys.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(keys)
    p = _ep_paths(ep_dir)
    _run_tool("vision_extract.py",
              ["--scenes-dir", str(p["scenes"]), "--glob", "*.jpg", "--out", str(p["vision"])])


def _stage_grouped(ep_dir: Path, cfg: Config) -> None:
    p = _ep_paths(ep_dir)
    _run_tool("scene_group_builder.py",
              ["--vision-manifest", str(p["vision"]), "--out", str(p["groups"])])


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
    _run_tool("gemini_narrative_pass.py",
              ["--groups-manifest", str(p["groups"]),
               "--vision-manifest", str(p["vision"]),
               "--out", str(p["beats"]),
               "--project", project, "--location", location])


def _stage_scripted(ep_dir: Path, cfg: Config) -> None:
    _check_openai()
    p = _ep_paths(ep_dir)
    _run_tool("script_expander.py",
              ["--beats", str(p["beats"]), "--vision", str(p["vision"]), "--out", str(p["script"])])


def _stage_voiced(ep_dir: Path, cfg: Config) -> None:
    _check_elevenlabs()
    p = _ep_paths(ep_dir)
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "")
    _run_tool("elevenlabs_tts_from_manifest.py",
              ["--script", str(p["script"]), "--out-dir", str(p["tts_dir"]), "--voice-id", voice])


def _stage_planned(ep_dir: Path, cfg: Config) -> None:
    # Blender render is a manual follow step; the terminal pipeline output is
    # render.plan.json (produced by timeline_planner, needs no API creds).
    p = _ep_paths(ep_dir)
    _run_tool("timeline_planner.py",
              ["--groups", str(p["groups"]), "--beats", str(p["beats"]),
               "--script", str(p["script"]), "--vision", str(p["vision"]),
               "--tts-index", str(p["tts_index"]),
               "--out", str(p["plan"]), "--mode", "narrated"])


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
        # Skip stages already completed (result_status <= effective_status in order)
        try:
            result_idx = STATUS_ORDER.index(result_status)
            current_idx = STATUS_ORDER.index(effective_status)
        except ValueError:
            continue

        if result_idx <= current_idx:
            # Already past this stage — verify idempotency via marker
            # (even if marker is missing we trust the catalog status)
            continue

        # This stage needs to run.  Check idempotency: if marker exists AND
        # the catalog status is already at or past result_status, skip.
        marker_path = ep_dir / marker_rel
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
