"""
tests/test_pipeline.py

Tests for studio.pipeline.run_chapter.

Monkeypatches _run_tool and detect_panels so no real tools are invoked.
Each stub simply touches the expected output marker file.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from studio.catalog.db import connect
from studio.catalog import repo
from studio.catalog.models import Chapter
from studio.config import Config, SiteCfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXED_NOW = "2026-06-09T00:00:00+00:00"


def _now():
    return FIXED_NOW


def _make_cfg(tmp_path: Path) -> Config:
    return Config(
        sites={},
        yolo_weights=tmp_path / "fake.pt",
        detect_backend="yolo",
        gallerydl_sleep=0.0,
        # production uses local Gemma — grouped (now LLM: understand+story-group)
        # needs no cloud cred on this backend; the cred wall stays at beated.
        beats_backend="ollama",
    )


def _make_chapter(con: sqlite3.Connection, ep_dir: Path, status: str = "downloaded") -> Chapter:
    sid = repo.upsert_series(con, "test", "https://x.test/s", "test-series", "Test Series", added_at=FIXED_NOW)
    cid = repo.upsert_chapter(con, sid, 1.0, "Ch 1", "https://x.test/c1", updated_at=FIXED_NOW)
    repo.set_chapter_status(con, cid, status, ep_dir=str(ep_dir), updated_at=FIXED_NOW)
    return repo.get_chapter(con, cid)


# ---------------------------------------------------------------------------
# Marker file names per stage (must match pipeline._STAGE_TABLE)
# ---------------------------------------------------------------------------

MARKERS = {
    "stitched":  "manifest.stitch.json",
    "detected":  "manifest.panels.expanded.json",
    "scened":    "manifest.scenes.json",
    "visioned":  "manifest.vision.json",
    "grouped":   "manifest.groups.json",
}


# ---------------------------------------------------------------------------
# Stub builders
# ---------------------------------------------------------------------------

def _tool_stub(ep_dir_ref: list[Path]):
    """Return a _run_tool stub that touches the expected marker in ep_dir_ref[0]."""
    import studio.pipeline as pipeline_mod

    # Map script name → marker filename
    SCRIPT_TO_MARKER = {
        "chunk_stitch_adaptive.py":   "manifest.stitch.json",
        "expand_boxes_to_gutters.py": "manifest.panels.expanded.json",
        "panels_to_scenes.py":        "manifest.scenes.json",
        "reconcile_seam_panels.py":   "manifest.scenes.json",
        "vision_extract.py":          "manifest.vision.json",
        "scene_group_builder.py":     "manifest.groups.json",   # legacy grouper
        "panel_understand.py":        "manifest.panels.understood.json",  # Pass 1
        "story_group.py":             "manifest.groups.json",   # Pass 2 (grouped marker)
        "cast_builder.py":            "manifest.cast.json",
        "gemini_narrative_pass.py":   "manifest.beats.json",
        "narration_punchup.py":       "manifest.beats.json",
        "narration_sanitize_pass.py": "manifest.beats.json",
        "script_expander.py":         "manifest.script.json",
        "elevenlabs_tts_from_manifest.py": "tts/tts_index.json",
        "local_tts_from_manifest.py": "tts/tts_index.json",
        "timeline_planner.py":        "render.plan.json",
        "blender_vse_from_plan.py":   "render.plan.json",
    }

    calls: list[str] = []
    call_args: list[tuple[str, list[str]]] = []

    def stub(script_name: str, args_list: list[str]) -> None:
        calls.append(script_name)
        call_args.append((script_name, list(args_list)))
        marker = SCRIPT_TO_MARKER.get(script_name)
        if marker and ep_dir_ref[0] is not None:
            mp = ep_dir_ref[0] / marker
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.touch()

    stub.calls = calls  # type: ignore[attr-defined]
    stub.call_args = call_args  # type: ignore[attr-defined]
    return stub


def _detect_stub(ep_dir_ref: list[Path]):
    """Return a detect_panels stub that touches manifest.panels.json."""
    calls: list = []

    def stub(stitch_manifest_path, out_path, weights, **kwargs):
        calls.append(out_path)
        Path(out_path).touch()

    stub.calls = calls  # type: ignore[attr-defined]
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _block_cred_gated_stages(monkeypatch, pipeline_mod) -> None:
    """Patch credential checkers so cred-gated stages (beated+) always raise.

    This keeps SP1-only tests hermetic regardless of local gcloud/env state.
    """
    from pathlib import Path as _Path

    def _raise_beated():
        raise pipeline_mod.MissingCredential("beated", "GOOGLE_APPLICATION_CREDENTIALS")

    def _raise_scripted():
        raise pipeline_mod.MissingCredential("scripted", "OPENAI_API_KEY")

    def _raise_voiced():
        raise pipeline_mod.MissingCredential("voiced", "ELEVENLABS_API_KEY")

    # Point _REPO_ROOT at a key-less path so _stage_beated takes the
    # _check_vertex_adc() branch (otherwise it would auth via the real repo's
    # keys/gcp-vision.json and the cred-gate would never fire).
    monkeypatch.setattr(pipeline_mod, "_REPO_ROOT", _Path("/nonexistent-studio-test-root"))
    monkeypatch.setattr(pipeline_mod, "_check_vertex_adc", _raise_beated)
    monkeypatch.setattr(pipeline_mod, "_check_openai", _raise_scripted)
    monkeypatch.setattr(pipeline_mod, "_check_elevenlabs", _raise_voiced)


class TestRunChapterFullProgress:
    """downloaded → stitched → detected → scened → visioned → grouped"""

    def test_advances_through_all_sp1_stages(self, tmp_path, monkeypatch):
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        ep_dir_ref = [ep_dir]

        tool_stub = _tool_stub(ep_dir_ref)
        detect_stub = _detect_stub(ep_dir_ref)

        monkeypatch.setattr(pipeline_mod, "_run_tool", tool_stub)
        monkeypatch.setattr("studio.detect.yolo_panels.detect_panels", detect_stub)
        _block_cred_gated_stages(monkeypatch, pipeline_mod)

        con = connect(tmp_path / "test.db")
        chapter = _make_chapter(con, ep_dir, status="downloaded")

        cfg = _make_cfg(tmp_path)
        pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)

        final = repo.get_chapter(con, chapter.id)
        # beated_failed because credentials are blocked — grouped is last success
        assert final.status == "beated_failed", f"Expected 'beated_failed', got '{final.status}'"

    def test_sp1_stages_complete_grouped(self, tmp_path, monkeypatch):
        """All SP1 stages complete; grouped is last successful status before cred wall."""
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        ep_dir_ref = [ep_dir]

        tool_stub = _tool_stub(ep_dir_ref)
        detect_stub = _detect_stub(ep_dir_ref)

        monkeypatch.setattr(pipeline_mod, "_run_tool", tool_stub)
        monkeypatch.setattr("studio.detect.yolo_panels.detect_panels", detect_stub)
        _block_cred_gated_stages(monkeypatch, pipeline_mod)

        con = connect(tmp_path / "test.db")
        chapter = _make_chapter(con, ep_dir, status="downloaded")
        cfg = _make_cfg(tmp_path)
        pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)

        # grouped marker must exist (tool stub touched it)
        assert (ep_dir / "manifest.groups.json").exists()
        # stitched → detected → scened → visioned → grouped all ran
        assert set(tool_stub.calls) >= {
            "chunk_stitch_adaptive.py",
            "expand_boxes_to_gutters.py",
            "panels_to_scenes.py",
            "vision_extract.py",
            "panel_understand.py",       # grouped = Pass 1 understand
            "story_group.py",            # + Pass 2 story-group (replaces scene_group_builder)
        }

    def test_catalog_statuses_in_order(self, tmp_path, monkeypatch):
        """Each intermediate status should be set in catalog order."""
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        ep_dir_ref = [ep_dir]

        statuses_set: list[str] = []
        orig_set_status = repo.set_chapter_status

        # Patch BEFORE _make_chapter so we only capture pipeline-driven calls
        monkeypatch.setattr(pipeline_mod, "_run_tool", _tool_stub(ep_dir_ref))
        monkeypatch.setattr("studio.detect.yolo_panels.detect_panels", _detect_stub(ep_dir_ref))
        _block_cred_gated_stages(monkeypatch, pipeline_mod)

        def tracking_set_status(con, cid, status, *, error=None, ep_dir=None, updated_at):
            statuses_set.append(status)
            orig_set_status(con, cid, status, error=error, ep_dir=ep_dir, updated_at=updated_at)

        con = connect(tmp_path / "test.db")
        chapter = _make_chapter(con, ep_dir, status="downloaded")

        monkeypatch.setattr(repo, "set_chapter_status", tracking_set_status)

        cfg = _make_cfg(tmp_path)
        pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)

        # SP1 stages succeed, then beated_failed stops the pipeline
        assert statuses_set == ["stitched", "detected", "scened", "visioned", "grouped", "beated_failed"]

    def test_reconcile_runs_between_scened_and_visioned(self, tmp_path, monkeypatch):
        """reconcile_seam_panels.py is invoked AFTER panels_to_scenes.py and BEFORE
        vision_extract.py (scene-level seam merge, upstream of vision)."""
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        ep_dir_ref = [ep_dir]
        tool_stub = _tool_stub(ep_dir_ref)
        detect_stub = _detect_stub(ep_dir_ref)
        monkeypatch.setattr(pipeline_mod, "_run_tool", tool_stub)
        monkeypatch.setattr("studio.detect.yolo_panels.detect_panels", detect_stub)
        _block_cred_gated_stages(monkeypatch, pipeline_mod)

        con = connect(tmp_path / "test.db")
        chapter = _make_chapter(con, ep_dir, status="downloaded")
        pipeline_mod.run_chapter(con, chapter, _make_cfg(tmp_path), now_fn=_now)

        calls = tool_stub.calls
        assert "reconcile_seam_panels.py" in calls
        assert (calls.index("panels_to_scenes.py")
                < calls.index("reconcile_seam_panels.py")
                < calls.index("vision_extract.py"))


class TestIdempotency:
    """Second run_chapter call on a completed chapter does nothing."""

    def test_no_tools_called_on_second_run(self, tmp_path, monkeypatch):
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        ep_dir_ref = [ep_dir]

        tool_stub = _tool_stub(ep_dir_ref)
        detect_stub = _detect_stub(ep_dir_ref)

        monkeypatch.setattr(pipeline_mod, "_run_tool", tool_stub)
        monkeypatch.setattr("studio.detect.yolo_panels.detect_panels", detect_stub)

        con = connect(tmp_path / "test.db")
        chapter = _make_chapter(con, ep_dir, status="downloaded")
        cfg = _make_cfg(tmp_path)

        # First run — completes pipeline
        pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)
        first_call_count = len(tool_stub.calls)

        # Second run — should be a no-op
        chapter2 = repo.get_chapter(con, chapter.id)
        pipeline_mod.run_chapter(con, chapter2, cfg, now_fn=_now)
        second_call_count = len(tool_stub.calls)

        assert second_call_count == first_call_count, (
            f"Second run invoked tools: {tool_stub.calls[first_call_count:]}"
        )

    def test_missing_marker_for_completed_status_reruns_that_stage(self, tmp_path, monkeypatch):
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        ep_dir_ref = [ep_dir]
        # Catalog says grouped, but the grouped marker is gone. Earlier markers
        # are intact, so only grouped should be rebuilt before beated is tried.
        for marker in ["manifest.stitch.json", "manifest.panels.expanded.json",
                       "manifest.scenes.json", "manifest.vision.json"]:
            (ep_dir / marker).touch()

        tool_stub = _tool_stub(ep_dir_ref)
        monkeypatch.setattr(pipeline_mod, "_run_tool", tool_stub)
        monkeypatch.setattr("studio.detect.yolo_panels.detect_panels",
                            _detect_stub(ep_dir_ref))
        _block_cred_gated_stages(monkeypatch, pipeline_mod)

        con = connect(tmp_path / "test.db")
        chapter = _make_chapter(con, ep_dir, status="grouped")
        cfg = _make_cfg(tmp_path)

        pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)

        assert "panel_understand.py" in tool_stub.calls
        assert "story_group.py" in tool_stub.calls
        assert "chunk_stitch_adaptive.py" not in tool_stub.calls
        assert (ep_dir / "manifest.groups.json").exists()
        assert repo.get_chapter(con, chapter.id).status == "beated_failed"


class TestResumability:
    """Failure at a stage → status=<stage>_failed; re-run resumes from that stage."""

    def test_failure_sets_failed_status(self, tmp_path, monkeypatch):
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        ep_dir_ref = [ep_dir]

        def failing_tool_stub(script_name: str, args_list: list[str]) -> None:
            if script_name == "panels_to_scenes.py":
                raise RuntimeError("scened tool exploded")
            # Touch markers for earlier stages
            SCRIPT_TO_MARKER = {
                "chunk_stitch_adaptive.py":   "manifest.stitch.json",
                "expand_boxes_to_gutters.py": "manifest.panels.expanded.json",
            }
            marker = SCRIPT_TO_MARKER.get(script_name)
            if marker:
                (ep_dir / marker).touch()

        monkeypatch.setattr(pipeline_mod, "_run_tool", failing_tool_stub)
        monkeypatch.setattr("studio.detect.yolo_panels.detect_panels", _detect_stub(ep_dir_ref))

        con = connect(tmp_path / "test.db")
        chapter = _make_chapter(con, ep_dir, status="downloaded")
        cfg = _make_cfg(tmp_path)

        pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)

        failed_ch = repo.get_chapter(con, chapter.id)
        assert failed_ch.status == "scened_failed"
        assert "scened tool exploded" in (failed_ch.error or "")

    def test_resume_from_failed_stage(self, tmp_path, monkeypatch):
        """After scened_failed, re-running with a good stub advances past scened."""
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        ep_dir_ref = [ep_dir]

        # Set up a chapter that is already at scened_failed
        # with stitched+detected markers already present
        (ep_dir / "manifest.stitch.json").touch()
        (ep_dir / "manifest.panels.expanded.json").touch()

        con = connect(tmp_path / "test.db")
        sid = repo.upsert_series(con, "test", "https://x.test/s2", "test-s2", "Test S2", added_at=FIXED_NOW)
        cid = repo.upsert_chapter(con, sid, 1.0, "Ch 1", "https://x.test/c1", updated_at=FIXED_NOW)
        repo.set_chapter_status(con, cid, "scened_failed", ep_dir=str(ep_dir), error="boom", updated_at=FIXED_NOW)
        chapter = repo.get_chapter(con, cid)

        # Now patch with a good stub; block cred-gated stages so grouped is terminal
        tool_stub = _tool_stub(ep_dir_ref)
        detect_stub = _detect_stub(ep_dir_ref)
        monkeypatch.setattr(pipeline_mod, "_run_tool", tool_stub)
        monkeypatch.setattr("studio.detect.yolo_panels.detect_panels", detect_stub)
        _block_cred_gated_stages(monkeypatch, pipeline_mod)

        cfg = _make_cfg(tmp_path)
        pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)

        resumed_ch = repo.get_chapter(con, chapter.id)
        # scened → visioned → grouped succeed, then beated_failed stops it
        assert resumed_ch.status == "beated_failed", f"Expected 'beated_failed', got '{resumed_ch.status}'"
        # Confirm scened and beyond were run (stitched/detected were skipped)
        assert "panels_to_scenes.py" in tool_stub.calls
        assert "chunk_stitch_adaptive.py" not in tool_stub.calls


class TestMissingCredential:
    """Cred-gated stages raise MissingCredential and set *_failed status."""

    def test_missing_openai_sets_scripted_failed_for_legacy_source(self, tmp_path, monkeypatch):
        """legacy narration_source still requires OPENAI_API_KEY (gemini_verbatim,
        the default, does not — covered by TestScriptedNarrationSource)."""
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()

        # Pre-populate markers up through grouped
        for marker in ["manifest.stitch.json", "manifest.panels.expanded.json",
                       "manifest.scenes.json", "manifest.vision.json",
                       "manifest.groups.json"]:
            (ep_dir / marker).touch()

        con = connect(tmp_path / "test.db")
        sid = repo.upsert_series(con, "test", "https://x.test/s3", "test-s3", "Test S3", added_at=FIXED_NOW)
        cid = repo.upsert_chapter(con, sid, 1.0, "Ch 1", "https://x.test/c1", updated_at=FIXED_NOW)
        repo.set_chapter_status(con, cid, "grouped", ep_dir=str(ep_dir), updated_at=FIXED_NOW)
        chapter = repo.get_chapter(con, cid)

        cfg = Config(
            sites={},
            yolo_weights=tmp_path / "fake.pt",
            detect_backend="yolo",
            gallerydl_sleep=0.0,
            narration_source="legacy",
        )

        # Ensure no env credentials present
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        # Stub _run_tool (shouldn't be called for cred-gated stages)
        tool_calls: list[str] = []

        def stub(script_name, args_list, **kwargs):
            tool_calls.append(script_name)

        monkeypatch.setattr(pipeline_mod, "_run_tool", stub)

        # Fake ADC file present so vertex passes, but OPENAI absent
        fake_adc = tmp_path / "adc.json"
        fake_adc.write_text("{}")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(fake_adc))

        pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)

        ch = repo.get_chapter(con, chapter.id)
        # beated stage runs (vertex cred ok via env), scripted fails (no OPENAI)
        assert ch.status == "scripted_failed"
        assert "OPENAI_API_KEY" in (ch.error or "")


# ---------------------------------------------------------------------------
# Cast-aware beated stage + gemini_verbatim scripted stage (narration v2)
# ---------------------------------------------------------------------------

def _capturing_stub(ep_dir: Path):
    """_run_tool stub that records (script, args) and touches each stage's
    real output marker so run_chapter keeps advancing."""
    SCRIPT_TO_MARKER = {
        "cast_builder.py":            "manifest.cast.json",
        "gemini_narrative_pass.py":   "manifest.beats.json",
        "script_expander.py":         "manifest.script.json",
        "elevenlabs_tts_from_manifest.py": "tts/tts_index.json",
        "local_tts_from_manifest.py": "tts/tts_index.json",
        "timeline_planner.py":        "render.plan.json",
    }
    calls: list[tuple[str, list[str]]] = []

    def stub(script_name, args_list, **kwargs):
        calls.append((script_name, list(args_list)))
        marker = SCRIPT_TO_MARKER.get(script_name)
        if marker:
            mp = ep_dir / marker
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.touch()

    stub.calls = calls  # type: ignore[attr-defined]
    return stub


def _fake_repo_root_with_key(tmp_path: Path) -> Path:
    """Repo root whose keys/gcp-vision.json lets _stage_beated auth hermetically."""
    root = tmp_path / "fakeroot"
    (root / "keys").mkdir(parents=True)
    (root / "keys" / "gcp-vision.json").write_text('{"project_id": "test-proj"}')
    return root


def _chapter_at(con, ep_dir: Path, status: str, markers: list[str]):
    for m in markers:
        mp = ep_dir / m
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.touch()
    sid = repo.upsert_series(con, "test", f"https://x.test/{status}", f"t-{status}", "T", added_at=FIXED_NOW)
    cid = repo.upsert_chapter(con, sid, 1.0, "Ch 1", "https://x.test/c1", updated_at=FIXED_NOW)
    repo.set_chapter_status(con, cid, status, ep_dir=str(ep_dir), updated_at=FIXED_NOW)
    return repo.get_chapter(con, cid)


_GROUPED_MARKERS = ["manifest.stitch.json", "manifest.panels.expanded.json",
                    "manifest.scenes.json", "manifest.vision.json",
                    "manifest.groups.json"]


class TestBeatedCastWiring:
    """_stage_beated builds the chapter cast (once) and threads --cast through."""

    def _run(self, tmp_path, monkeypatch, *, pre_cast: bool):
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        if pre_cast:
            (ep_dir / "manifest.cast.json").touch()

        monkeypatch.setattr(pipeline_mod, "_REPO_ROOT", _fake_repo_root_with_key(tmp_path))
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "to-be-overwritten")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        stub = _capturing_stub(ep_dir)
        monkeypatch.setattr(pipeline_mod, "_run_tool", stub)

        con = connect(tmp_path / "test.db")
        chapter = _chapter_at(con, ep_dir, "grouped", _GROUPED_MARKERS)
        pipeline_mod.run_chapter(con, chapter, _make_cfg(tmp_path), now_fn=_now)
        return stub, repo.get_chapter(con, chapter.id), ep_dir

    def test_cast_built_before_narrative_pass_and_flag_passed(self, tmp_path, monkeypatch):
        stub, ch, ep_dir = self._run(tmp_path, monkeypatch, pre_cast=False)
        names = [n for n, _ in stub.calls]
        assert "cast_builder.py" in names
        assert names.index("cast_builder.py") < names.index("gemini_narrative_pass.py")

        cast_args = dict(zip(*[iter(next(a for n, a in stub.calls if n == "cast_builder.py"))] * 2))
        assert cast_args["--out"].endswith("manifest.cast.json")

        gem_args = next(a for n, a in stub.calls if n == "gemini_narrative_pass.py")
        assert "--cast" in gem_args
        assert gem_args[gem_args.index("--cast") + 1] == str(ep_dir / "manifest.cast.json")
        # beats keep the canonical unified filename (no .narr.json fork)
        assert gem_args[gem_args.index("--out") + 1] == str(ep_dir / "manifest.beats.json")

    def test_existing_cast_skips_cast_builder(self, tmp_path, monkeypatch):
        stub, ch, ep_dir = self._run(tmp_path, monkeypatch, pre_cast=True)
        names = [n for n, _ in stub.calls]
        assert "cast_builder.py" not in names
        gem_args = next(a for n, a in stub.calls if n == "gemini_narrative_pass.py")
        assert "--cast" in gem_args


class TestScriptedNarrationSource:
    """Default narration_source=gemini_verbatim voices the Gemini line without
    OpenAI; the stage passes --narration-source and --cast to script_expander."""

    def test_verbatim_scripted_runs_without_openai_key(self, tmp_path, monkeypatch):
        import studio.pipeline as pipeline_mod

        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        stub = _capturing_stub(ep_dir)
        monkeypatch.setattr(pipeline_mod, "_run_tool", stub)

        con = connect(tmp_path / "test.db")
        chapter = _chapter_at(con, ep_dir, "beated",
                              _GROUPED_MARKERS + ["manifest.beats.json", "manifest.cast.json"])
        cfg = _make_cfg(tmp_path)
        assert cfg.narration_source == "gemini_verbatim"   # the production default
        pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)

        se_args = next(a for n, a in stub.calls if n == "script_expander.py")
        assert se_args[se_args.index("--narration-source") + 1] == "gemini_verbatim"
        assert se_args[se_args.index("--cast") + 1] == str(ep_dir / "manifest.cast.json")

        ch = repo.get_chapter(con, chapter.id)
        # scripted succeeded with NO OpenAI key; pipeline stops at the TTS cred wall
        assert ch.status == "voiced_failed"
        assert "ELEVENLABS_API_KEY" in (ch.error or "")

class TestConfigNarrationSource:
    def test_load_parses_models_narration_source(self, tmp_path):
        from studio.config import load
        toml = tmp_path / "studio.toml"
        toml.write_text('[models]\nnarration_source = "legacy"\n')
        assert load(toml).narration_source == "legacy"

    def test_default_is_gemini_verbatim(self, tmp_path):
        from studio.config import load
        toml = tmp_path / "studio.toml"
        toml.write_text("")
        assert load(toml).narration_source == "gemini_verbatim"


class TestMissingEpDir:
    """Chapter without ep_dir raises ValueError immediately."""

    def test_no_ep_dir_raises(self, tmp_path):
        import studio.pipeline as pipeline_mod

        con = connect(tmp_path / "test.db")
        sid = repo.upsert_series(con, "test", "https://x.test/s4", "test-s4", "Test S4", added_at=FIXED_NOW)
        cid = repo.upsert_chapter(con, sid, 1.0, "Ch 1", "https://x.test/c1", updated_at=FIXED_NOW)
        chapter = repo.get_chapter(con, cid)  # ep_dir is None

        cfg = _make_cfg(tmp_path)
        with pytest.raises(ValueError, match="ep_dir"):
            pipeline_mod.run_chapter(con, chapter, cfg, now_fn=_now)


# ---------------------------------------------------------------------------
# --until: stop the chain at a target status (dashboard worker contract)
# ---------------------------------------------------------------------------

def test_run_chapter_until_stops_at_target(tmp_path, monkeypatch):
    from studio import pipeline

    ep = tmp_path / "ep"
    ep.mkdir()
    con = connect(tmp_path / "s.db")
    ch = _make_chapter(con, ep, status="downloaded")

    ran = []

    def _stub(name):
        def run(ep_dir, cfg):
            ran.append(name)
            # touch the marker so the stage counts as completed
            for status, _fn, marker in pipeline._STAGE_TABLE:
                if status == name:
                    (ep_dir / marker).parent.mkdir(parents=True, exist_ok=True)
                    (ep_dir / marker).touch()
        return run

    table = [(s, _stub(s), m) for (s, _f, m) in pipeline._STAGE_TABLE]
    monkeypatch.setattr(pipeline, "_STAGE_TABLE", table)

    pipeline.run_chapter(con, ch, _make_cfg(tmp_path), now_fn=_now,
                         until="scened")
    assert ran == ["stitched", "detected", "scened"]
    assert repo.get_chapter(con, ch.id).status == "scened"


def _beated_fixture(tmp_path, monkeypatch, *, marker: bool):
    """ep dir with existing manifests (+ optional keep-base marker) + a fake gcp
    key so _stage_beated reads project without a cred check; returns the call stub."""
    import json
    import studio.pipeline as pipeline_mod
    ep = tmp_path / "ep"
    ep.mkdir()
    for m in ("manifest.beats.json", "manifest.cast.json",
              "manifest.groups.json", "manifest.vision.json"):
        (ep / m).write_text("{}")
    if marker:
        (ep / ".narration_keepbase").touch()
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "gcp-vision.json").write_text(json.dumps({"project_id": "t"}))
    monkeypatch.setattr(pipeline_mod, "_REPO_ROOT", tmp_path)
    stub = _tool_stub([ep])
    monkeypatch.setattr(pipeline_mod, "_run_tool", stub)
    cfg = Config(sites={}, yolo_weights=tmp_path / "f.pt", detect_backend="yolo",
                 gallerydl_sleep=0.0, punchup="cinematic", beats_backend="ollama")
    pipeline_mod._stage_beated(ep, cfg)
    return stub


def test_beated_keep_base_skips_regeneration_but_keeps_punchup(tmp_path, monkeypatch):
    # the marker preserves a hand-picked/approved narration verbatim (no LLM
    # re-roll) while the persona punchup still re-applies the channel voice
    stub = _beated_fixture(tmp_path, monkeypatch, marker=True)
    assert "gemini_narrative_pass.py" not in stub.calls   # no regeneration
    assert "cast_builder.py" not in stub.calls
    assert "narration_punchup.py" in stub.calls           # persona still applied


def test_beated_without_marker_regenerates(tmp_path, monkeypatch):
    stub = _beated_fixture(tmp_path, monkeypatch, marker=False)
    assert "gemini_narrative_pass.py" in stub.calls       # normal regeneration


