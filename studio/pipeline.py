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
    result = subprocess.run(cmd, check=True)
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


def _stage_stitch(ep_dir: Path, cfg: Config) -> None:
    _run_tool("chunk_stitch_adaptive.py", [str(ep_dir)])


def _stage_detect(ep_dir: Path, cfg: Config) -> None:
    stitch_manifest = ep_dir / "manifest.stitch.json"
    panels_manifest = ep_dir / "manifest.panels.json"
    if cfg.detect_backend == "yolo":
        from studio.detect.yolo_panels import detect_panels
        detect_panels(
            str(stitch_manifest),
            str(panels_manifest),
            str(cfg.yolo_weights),
        )
    _run_tool("expand_boxes_to_gutters.py", [str(ep_dir)])


def _stage_scened(ep_dir: Path, cfg: Config) -> None:
    _run_tool("panels_to_scenes.py", [str(ep_dir)])


def _stage_visioned(ep_dir: Path, cfg: Config) -> None:
    _run_tool("vision_extract.py", [str(ep_dir), "--glob", "*.jpg"])


def _stage_grouped(ep_dir: Path, cfg: Config) -> None:
    _run_tool("scene_group_builder.py", [str(ep_dir)])


def _stage_beated(ep_dir: Path, cfg: Config) -> None:
    _check_vertex_adc()
    _run_tool("timeline_planner.py", [str(ep_dir)])


def _stage_scripted(ep_dir: Path, cfg: Config) -> None:
    _check_openai()
    _run_tool("script_expander.py", [str(ep_dir)])


def _stage_voiced(ep_dir: Path, cfg: Config) -> None:
    _check_elevenlabs()
    _run_tool("elevenlabs_tts_from_manifest.py", [str(ep_dir)])


def _stage_planned(ep_dir: Path, cfg: Config) -> None:
    _run_tool("blender_vse_from_plan.py", [str(ep_dir)])


# Ordered list of (result_status, runner_fn, output_marker_relpath)
# "result_status" = status after this stage completes successfully
_STAGE_TABLE: list[tuple[str, Callable[[Path, Config], None], str]] = [
    ("stitched",  _stage_stitch,   "manifest.stitch.json"),
    ("detected",  _stage_detect,   "manifest.panels.json"),
    ("scened",    _stage_scened,   "manifest.scenes.json"),
    ("visioned",  _stage_visioned, "manifest.vision.json"),
    ("grouped",   _stage_grouped,  "manifest.groups.json"),
    ("beated",    _stage_beated,   "manifest.beat.json"),
    ("scripted",  _stage_scripted, "manifest.script.json"),
    ("voiced",    _stage_voiced,   "manifest.voiced.json"),
    ("planned",   _stage_planned,  "manifest.plan.json"),
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
