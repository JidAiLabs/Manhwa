"""
tests/test_manifest_freshness.py

TDD for tools/manifest_freshness.py — manifest completeness + staleness guardrail.

The headline bug: render.plan.clean.json (3 days old) sat next to fresh
manifest.beats.json; the dashboard silently rendered stale cuts.  Every test
here exercises the guardrail that catches that class of failure.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "manifest_freshness",
    Path(__file__).resolve().parent.parent / "tools" / "manifest_freshness.py",
)
mf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mf)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _touch(path: Path, mtime: float) -> None:
    """Create a file and set its mtime."""
    path.write_bytes(b"")
    os.utime(str(path), (mtime, mtime))


# ---------------------------------------------------------------------------
# HEADLINE TEST — the exact production bug that motivated this guardrail
# ---------------------------------------------------------------------------

def test_stale_plan_clean_older_than_beats_flags_stale(tmp_path):
    """render.plan.clean.json is 3 days old; manifest.beats.json is fresh.
    The guardrail must emit a stale_manifest ERROR for render.plan.clean.json.

    This is the exact production bug: stale render.plan.clean.json (mtime T0+5)
    sat next to fresh manifest.beats.json (mtime T0+3d).  Nothing caught it;
    dashboard silently rendered old cuts.

    render.plan.clean.json → manifest.beats.json is now a direct DAG edge, so
    the guardrail catches this without going through the transient render.plan.json.
    """
    T0 = 1_000_000.0   # epoch seconds (arbitrary base)
    THREE_DAYS = 3 * 86_400

    # Build a complete chain with correct mtime ordering …
    _touch(tmp_path / "manifest.vision.json",              T0)
    _touch(tmp_path / "manifest.panels.understood.json",   T0 + 1)
    _touch(tmp_path / "manifest.groups.json",              T0 + 2)
    # … except beats was regenerated 3 days after all downstream files were built
    _touch(tmp_path / "manifest.beats.json",               T0 + THREE_DAYS)
    _touch(tmp_path / "manifest.script.json",              T0 + 3)
    # render.plan.json intentionally absent (transient intermediate)
    # THE KEY STALE FILE: plan.clean predates the freshly-regenerated beats
    _touch(tmp_path / "render.plan.clean.json",            T0 + 5)

    issues = mf.verify_chapter(str(tmp_path), status="prepped")

    stale = [i for i in issues if i["code"] == "stale_manifest"]
    stale_files = {i["file"] for i in stale}

    # The headline case: plan.clean must be reported stale via the direct edge
    assert "render.plan.clean.json" in stale_files, (
        f"render.plan.clean.json not in stale flags; got: {stale}")
    # All stale issues are ERROR
    assert all(i["severity"] == "ERROR" for i in stale), stale
    # The plan.clean issue names beats as the cause
    plan_clean_issue = next(i for i in stale if i["file"] == "render.plan.clean.json")
    assert "manifest.beats.json" in plan_clean_issue["detail"]


# ---------------------------------------------------------------------------
# Fresh chain — no issues
# ---------------------------------------------------------------------------

def test_fresh_chain_produces_no_issues(tmp_path):
    """A complete chain where every output is newer than all its inputs
    must produce zero issues.  render.plan.json is intentionally absent —
    it is a transient intermediate that does not persist."""
    base = 1_000_000.0
    files = [
        "manifest.vision.json",
        "manifest.panels.understood.json",
        "manifest.groups.json",
        "manifest.beats.json",
        "manifest.script.json",
        "render.plan.clean.json",
    ]
    for i, name in enumerate(files):
        _touch(tmp_path / name, base + i)

    issues = mf.verify_chapter(str(tmp_path), status="prepped")
    assert issues == [], f"expected no issues on fresh chain, got: {issues}"


# ---------------------------------------------------------------------------
# Real-pipeline false-positive regression
# ---------------------------------------------------------------------------

def test_real_pipeline_mtimes_no_false_positive(tmp_path):
    """Encodes the EXACT real-world mtimes from a freshly-prepared Ch1 run that
    was incorrectly flagged as stale/missing before the fix.

    Real mtimes (seconds since epoch):
      vision=801.493, understood=801.464, groups=882.223, beats=4834.513,
      script=4835.313, render.plan.clean.json=4883.417, render.plan.json ABSENT.

    Key: vision is ~0.03s NEWER than understood because panel_understand.py
    stamps panel_kind back onto manifest.vision.json after writing understood.
    The old understood→vision DAG edge fired a false stale on every fresh run.
    render.plan.json is absent (transient) — the old STATUS_REQUIRED entry
    fired a false missing_manifest at status='prepped'.

    Both false positives must produce zero issues after the fix.
    """
    # Use real relative offsets so the mtime relationships match exactly
    BASE = 1_000_000.0
    _touch(tmp_path / "manifest.vision.json",             BASE + 801.493)
    _touch(tmp_path / "manifest.panels.understood.json",  BASE + 801.464)   # older than vision
    _touch(tmp_path / "manifest.groups.json",             BASE + 882.223)
    _touch(tmp_path / "manifest.beats.json",              BASE + 4834.513)
    _touch(tmp_path / "manifest.script.json",             BASE + 4835.313)
    _touch(tmp_path / "render.plan.clean.json",           BASE + 4883.417)
    # render.plan.json intentionally absent — transient intermediate

    issues = mf.verify_chapter(str(tmp_path), status="prepped")
    assert issues == [], (
        f"false positives on real Ch1 mtimes (vision newer than understood, "
        f"render.plan.json absent): {issues}")

    # Also check status='planned' — same files, same expectation
    issues_planned = mf.verify_chapter(str(tmp_path), status="planned")
    assert issues_planned == [], (
        f"false positives at status='planned': {issues_planned}")


# ---------------------------------------------------------------------------
# Missing manifest for declared status
# ---------------------------------------------------------------------------

def test_missing_script_json_for_scripted_status(tmp_path):
    """status='scripted' but manifest.script.json absent → missing_manifest ERROR."""
    base = 1_000_000.0
    _touch(tmp_path / "manifest.vision.json",            base)
    _touch(tmp_path / "manifest.panels.understood.json", base + 1)
    _touch(tmp_path / "manifest.groups.json",            base + 2)
    _touch(tmp_path / "manifest.beats.json",             base + 3)
    # manifest.script.json deliberately absent

    issues = mf.verify_chapter(str(tmp_path), status="scripted")
    missing = [i for i in issues if i["code"] == "missing_manifest"]
    assert len(missing) >= 1
    assert any(i["file"] == "manifest.script.json" for i in missing)
    assert all(i["severity"] == "ERROR" for i in missing)


# ---------------------------------------------------------------------------
# Missing ep_dir
# ---------------------------------------------------------------------------

def test_missing_ep_dir_returns_one_issue_no_exception(tmp_path):
    """A non-existent directory must return exactly one missing_manifest issue
    without raising any exception."""
    missing_dir = str(tmp_path / "does_not_exist")
    issues = mf.verify_chapter(missing_dir)
    assert len(issues) == 1
    assert issues[0]["code"] == "missing_manifest"
    assert issues[0]["severity"] == "ERROR"


# ---------------------------------------------------------------------------
# Missing input skips the edge (no false stale)
# ---------------------------------------------------------------------------

def test_absent_input_skips_stale_edge(tmp_path):
    """If an input doesn't exist, the DAG edge is skipped — no false stale_manifest
    flag should be emitted for that edge (the missing_manifest check handles it)."""
    base = 1_000_000.0
    # manifest.vision.json absent; manifest.panels.understood.json still exists
    _touch(tmp_path / "manifest.panels.understood.json", base + 100)
    # manifest.panels.understood.json is newer than its absent input — but
    # since the input doesn't exist, no stale edge should fire

    issues = mf.verify_chapter(str(tmp_path), status="grouped")

    # The missing vision file should raise a missing_manifest, NOT a stale_manifest
    stale = [i for i in issues if i["code"] == "stale_manifest"
             and i["file"] == "manifest.panels.understood.json"]
    assert stale == [], f"false stale flag emitted for edge with absent input: {stale}"


# ---------------------------------------------------------------------------
# Cast file optional: beats stale check still works for other inputs
# ---------------------------------------------------------------------------

def test_cast_file_optional_beats_stale_check_uses_groups(tmp_path):
    """manifest.cast.json is optional — if absent, the beats stale check must
    still fire when manifest.groups.json is newer than manifest.beats.json.
    The cast edge must not suppress the groups→beats stale detection."""
    base = 1_000_000.0
    _touch(tmp_path / "manifest.vision.json",            base)
    _touch(tmp_path / "manifest.panels.understood.json", base + 1)
    # beats was built BEFORE groups was last updated (simulates a re-group)
    _touch(tmp_path / "manifest.beats.json",             base + 2)
    _touch(tmp_path / "manifest.groups.json",            base + 100)  # newer than beats
    # manifest.cast.json deliberately absent (optional)

    issues = mf.verify_chapter(str(tmp_path), status="beated")
    stale = [i for i in issues if i["code"] == "stale_manifest"
             and i["file"] == "manifest.beats.json"]
    assert len(stale) == 1, (
        f"expected stale_manifest for beats (groups newer), got: {issues}")
    assert "manifest.groups.json" in stale[0]["detail"]


def test_cast_file_optional_present_and_stale_is_caught(tmp_path):
    """When manifest.cast.json IS present and newer than manifest.beats.json,
    the beats→cast staleness edge must fire normally."""
    base = 1_000_000.0
    _touch(tmp_path / "manifest.vision.json",            base)
    _touch(tmp_path / "manifest.panels.understood.json", base + 1)
    _touch(tmp_path / "manifest.groups.json",            base + 2)
    _touch(tmp_path / "manifest.beats.json",             base + 3)
    _touch(tmp_path / "manifest.cast.json",              base + 100)  # cast updated after beats

    issues = mf.verify_chapter(str(tmp_path), status="beated")
    stale = [i for i in issues if i["code"] == "stale_manifest"
             and i["file"] == "manifest.beats.json"]
    assert len(stale) == 1, (
        f"expected stale_manifest for beats (cast newer), got: {issues}")


def test_cast_file_absent_beats_fresh_otherwise_no_stale(tmp_path):
    """When cast is absent and groups is older than beats, the beats edge
    should produce no stale flag (beats is up to date)."""
    base = 1_000_000.0
    _touch(tmp_path / "manifest.vision.json",            base)
    _touch(tmp_path / "manifest.panels.understood.json", base + 1)
    _touch(tmp_path / "manifest.groups.json",            base + 2)
    _touch(tmp_path / "manifest.beats.json",             base + 3)  # newer than all inputs
    # cast absent

    issues = mf.verify_chapter(str(tmp_path), status="beated")
    stale = [i for i in issues if i["code"] == "stale_manifest"]
    assert stale == [], f"unexpected stale flags: {stale}"


# ---------------------------------------------------------------------------
# Status inference (status=None)
# ---------------------------------------------------------------------------

def test_inferred_status_from_deepest_sentinel(tmp_path):
    """With status=None, verify_chapter infers the deepest stage whose sentinel
    exists and checks that chain — a stale edge within it must still be caught.

    render.plan.json is transient; the deepest persistent sentinel is
    render.plan.clean.json (infers 'prepped').  A stale script.json (beats
    newer) must still be caught via the beats→script edge.
    """
    base = 1_000_000.0
    _touch(tmp_path / "manifest.vision.json",            base)
    _touch(tmp_path / "manifest.panels.understood.json", base + 1)
    _touch(tmp_path / "manifest.groups.json",            base + 2)
    _touch(tmp_path / "manifest.beats.json",             base + 3)
    _touch(tmp_path / "manifest.script.json",            base + 4)
    _touch(tmp_path / "render.plan.clean.json",          base + 5)
    # render.plan.json intentionally absent (transient)

    # Now make script.json stale relative to beats.json (simulate re-beat)
    os.utime(str(tmp_path / "manifest.beats.json"),
             (base + 100, base + 100))

    issues = mf.verify_chapter(str(tmp_path))   # status=None
    stale = [i for i in issues if i["code"] == "stale_manifest"]
    assert any(i["file"] == "manifest.script.json" for i in stale), (
        f"inferred status check missed stale script: {issues}")


def test_no_manifests_at_all_returns_empty(tmp_path):
    """If no sentinel exists at all, status cannot be inferred → return []."""
    issues = mf.verify_chapter(str(tmp_path))  # empty dir, status=None
    assert issues == []


# ---------------------------------------------------------------------------
# stale_video: rendered MP4 older than the clean plan it was rendered from
# ---------------------------------------------------------------------------

def _fresh_prepped_chain(root: Path, base: float) -> None:
    """Create a complete fresh prepped chain under *root* at mtimes base+0..+5."""
    files = [
        "manifest.vision.json",
        "manifest.panels.understood.json",
        "manifest.groups.json",
        "manifest.beats.json",
        "manifest.script.json",
        "render.plan.clean.json",
    ]
    for i, name in enumerate(files):
        _touch(root / name, base + i)


def test_stale_video_older_than_plan(tmp_path):
    """Video exists, plan exists, video is older → one WARN with code stale_video.
    Full chain present (status='prepped' inferred) so no missing_manifest noise."""
    T0 = 1_000_000.0
    _fresh_prepped_chain(tmp_path, T0)
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    # video predates the chain; plan.clean is the freshest file (T0+5)
    _touch(render_dir / "segment_both.mp4", T0 - 1)   # older than everything

    issues = mf.verify_chapter(str(tmp_path))
    stale = [i for i in issues if i["code"] == "stale_video"]
    assert len(stale) == 1, f"expected one stale_video, got: {issues}"
    assert stale[0]["severity"] == "WARN", stale[0]
    assert stale[0]["file"] == "render/segment_both.mp4"
    assert "render.plan.clean.json" in stale[0]["detail"]


def test_stale_video_newer_than_plan(tmp_path):
    """Video exists, plan exists, video is NEWER than the plan → no stale_video."""
    T0 = 1_000_000.0
    _fresh_prepped_chain(tmp_path, T0)
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    # video is newer than render.plan.clean.json (T0+5)
    _touch(render_dir / "segment_both.mp4", T0 + 3600)

    issues = mf.verify_chapter(str(tmp_path))
    stale = [i for i in issues if i["code"] == "stale_video"]
    assert stale == [], f"unexpected stale_video when video is current: {issues}"


def test_stale_video_absent(tmp_path):
    """Video does NOT exist, plan exists → no stale_video (not rendered ≠ stale)."""
    T0 = 1_000_000.0
    _fresh_prepped_chain(tmp_path, T0)
    # no render/segment_both.mp4

    issues = mf.verify_chapter(str(tmp_path))
    stale = [i for i in issues if i["code"] == "stale_video"]
    assert stale == [], f"absent video must not produce stale_video: {issues}"


def test_stale_video_plan_absent(tmp_path):
    """Video exists, plan does NOT exist → no stale_video (nothing to compare against)."""
    T0 = 1_000_000.0
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    _touch(render_dir / "segment_both.mp4", T0)
    # no render.plan.clean.json → status cannot be inferred, returns []

    issues = mf.verify_chapter(str(tmp_path))
    stale = [i for i in issues if i["code"] == "stale_video"]
    assert stale == [], f"absent plan must not produce stale_video: {issues}"
