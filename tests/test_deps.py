"""studio/deps.py — the ONE dependency table.

Pins: (1) internal consistency (every input is a known artifact, order is the
canonical STATUS_ORDER + prepped), (2) the derived freshness DAG is a strict
SUPERSET of the legacy hand-maintained MANIFEST_DAG (nothing that was checked
stops being checked), (3) artifacts_beyond closures per rewind target, and
(4) end-to-end equivalence: a synthetic chapter with a stale edge flags via
manifest_freshness exactly as it did pre-derivation.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from studio import deps
from studio.catalog.models import STATUS_ORDER

_SPEC = importlib.util.spec_from_file_location(
    "manifest_freshness",
    Path(__file__).resolve().parent.parent / "tools" / "manifest_freshness.py")
mf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mf)  # type: ignore[union-attr]

# manifest_freshness.MANIFEST_DAG as hand-maintained before Task 8 — the
# floor the derived DAG must never drop below.
LEGACY_GRAPH = {
    "manifest.groups.json":   {"manifest.panels.understood.json"},
    "manifest.story.json":    {"manifest.panels.understood.json"},
    "manifest.beats.json":    {"manifest.groups.json", "manifest.cast.json"},
    "manifest.script.json":   {"manifest.beats.json"},
    "render.plan.clean.json": {"manifest.script.json", "manifest.beats.json"},
}


def test_every_input_is_a_known_artifact():
    for name, a in deps.ARTIFACTS.items():
        for inp in a.inputs + a.optional:
            assert inp in deps.ARTIFACTS, f"{name} input {inp} is not a known artifact"
        assert a.stage in deps.ORDER, f"{name} stage {a.stage} not in ORDER"
        # an artifact must never be its own input
        assert name not in a.inputs + a.optional


def test_order_is_status_order_plus_prepped_not_redeclared():
    assert deps.ORDER[:len(STATUS_ORDER)] == tuple(STATUS_ORDER)
    assert deps.ORDER[len(STATUS_ORDER):] == ("prepped",)


def test_inputs_always_come_from_earlier_or_same_stage():
    for name, a in deps.ARTIFACTS.items():
        for inp in a.inputs + a.optional:
            assert (deps.ORDER.index(deps.stage_of(inp))
                    <= deps.ORDER.index(a.stage)), f"{name} <- {inp} goes backwards"


def test_dag_is_superset_of_legacy_freshness_graph():
    d = deps.dag()
    for out, legacy_inputs in LEGACY_GRAPH.items():
        assert out in d, f"legacy-checked output {out} dropped from derived DAG"
        req, opt, _sha = d[out]
        assert legacy_inputs <= set(req) | set(opt), (
            f"{out}: legacy edges {legacy_inputs} not all present in {req}+{opt}")
    # the one deliberately-sha_only edge
    req, opt, sha = d["manifest.panels.understood.json"]
    assert req == ("manifest.vision.json",) and sha is True
    # cast optionality preserved (beats edge checked only when cast exists)
    req, opt, _ = d["manifest.beats.json"]
    assert "manifest.cast.json" in opt and "manifest.groups.json" in req
    # NO tts_index <- script edge; tts_index is only an OPTIONAL input of clean
    assert "tts/tts_index.json" not in d
    req, opt, _ = d["render.plan.clean.json"]
    assert "tts/tts_index.json" in opt and "tts/tts_index.json" not in req


def test_artifacts_beyond_closures():
    assert set(deps.artifacts_beyond("scripted")) == {
        "tts/tts_index.json", "render.plan.json", "render.plan.clean.json"}
    assert set(deps.artifacts_beyond("grouped")) == {
        "manifest.cast.json", "manifest.beats.json", "manifest.script.json",
        "manifest.sanitize.json", "render.plan.json", "tts/tts_index.json",
        "render.plan.clean.json"}
    assert set(deps.artifacts_beyond("detected")) == {
        "manifest.scenes.json", "manifest.vision.json",
        "manifest.panels.understood.json", "manifest.groups.json",
        "manifest.story.json", "manifest.cast.json", "manifest.beats.json",
        "manifest.script.json", "manifest.sanitize.json", "render.plan.json",
        "tts/tts_index.json", "render.plan.clean.json"}
    # downloaded: everything in the table
    assert set(deps.artifacts_beyond("downloaded")) == set(deps.ARTIFACTS)
    # beyond the last rung: nothing
    assert deps.artifacts_beyond("prepped") == ()


def test_required_chain_matches_legacy_status_required():
    assert deps.required_chain("visioned") == ("manifest.vision.json",)
    assert deps.required_chain("grouped") == (
        "manifest.vision.json", "manifest.panels.understood.json",
        "manifest.groups.json")
    assert deps.required_chain("beated") == (
        "manifest.vision.json", "manifest.panels.understood.json",
        "manifest.groups.json", "manifest.beats.json")
    assert deps.required_chain("scripted") == (
        "manifest.vision.json", "manifest.panels.understood.json",
        "manifest.groups.json", "manifest.beats.json", "manifest.script.json")
    assert deps.required_chain("prepped") == (
        "manifest.vision.json", "manifest.panels.understood.json",
        "manifest.groups.json", "manifest.beats.json", "manifest.script.json",
        "render.plan.clean.json")


def test_freshness_statuses():
    assert deps.freshness_statuses() == (
        "visioned", "grouped", "beated", "scripted", "prepped")


def test_stage_of():
    assert deps.stage_of("manifest.vision.json") == "visioned"
    assert deps.stage_of("render.plan.clean.json") == "prepped"
    assert deps.stage_of("render.plan.json") == "planned"


# --- end-to-end equivalence: the derived DAG flags exactly as before ----------

def _touch(path: Path, mtime: float) -> None:
    path.write_bytes(b"{}")
    os.utime(str(path), (mtime, mtime))


def test_dag_equivalence_stale_edge_flags_like_legacy(tmp_path):
    """Synthetic chapter with ONE stale edge (groups newer than beats) must
    yield the same single stale_manifest flag the hand-maintained graph gave."""
    base = 1_000_000.0
    _touch(tmp_path / "manifest.vision.json",            base)
    _touch(tmp_path / "manifest.panels.understood.json", base + 1)
    _touch(tmp_path / "manifest.beats.json",             base + 2)
    _touch(tmp_path / "manifest.groups.json",            base + 100)  # newer
    issues = mf.verify_chapter(str(tmp_path), status="beated")
    stale = [i for i in issues if i["code"] == "stale_manifest"]
    assert [i["file"] for i in stale] == ["manifest.beats.json"]
    assert "manifest.groups.json" in stale[0]["detail"]
