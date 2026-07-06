"""tests/test_manifest_io.py

TDD for tools/manifest_io.py — atomic, provenance-stamped manifest IO.

Headline bugs this closes: every tools/*.py manifest write was in-place
open(path, "w") + json.dump (a SIGKILL/disk-full mid-write leaves a torn
file); no manifest recorded what inputs produced it (freshness relied on
mtimes only). This module fixes both: atomic tmp+os.replace writes, and a
_meta.inputs sha stamp future freshness/DAG tooling can build on.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "manifest_io",
    Path(__file__).resolve().parent.parent / "tools" / "manifest_io.py",
)
mio = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mio)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

def test_write_atomic_and_stamped(tmp_path):
    input_path = tmp_path / "manifest.groups.json"
    input_path.write_text('{"shots": []}', encoding="utf-8")
    out_path = tmp_path / "manifest.beats.json"

    mio.write_manifest(out_path, {"beats": []}, inputs=[input_path], tool="test_tool")

    assert out_path.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == [], f"tmp residue left behind: {leftovers}"

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["beats"] == []
    meta = data["_meta"]
    assert meta["schema"] == 1
    assert meta["tool"] == "test_tool"
    assert isinstance(meta["written_at"], str) and meta["written_at"].endswith("Z")
    assert meta["inputs"] == {"manifest.groups.json": mio.input_sha(input_path)}


def test_write_replaces_existing(tmp_path):
    out_path = tmp_path / "manifest.script.json"
    mio.write_manifest(out_path, {"sections": [1]}, tool="t")
    mio.write_manifest(out_path, {"sections": [1, 2, 3]}, tool="t")

    entries = list(tmp_path.iterdir())
    assert entries == [out_path], f"directory not clean: {entries}"
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["sections"] == [1, 2, 3]


def test_existing_meta_overwritten(tmp_path):
    out_path = tmp_path / "manifest.cast.json"
    mio.write_manifest(out_path, {"cast": [], "_meta": {"old": True, "tool": "stale"}},
                        tool="new_tool")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["_meta"]["tool"] == "new_tool"
    assert "old" not in data["_meta"]


def test_write_no_inputs_yields_empty_inputs_map(tmp_path):
    out_path = tmp_path / "manifest.story.json"
    mio.write_manifest(out_path, {"arc": []}, tool="t")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["_meta"]["inputs"] == {}


def test_write_extra_meta_merged(tmp_path):
    out_path = tmp_path / "manifest.sanitize.json"
    mio.write_manifest(out_path, {"ok": True}, tool="narration_sanitize_pass",
                        extra_meta={"schema_version": "sanitize_marker_v1"})
    meta = json.loads(out_path.read_text(encoding="utf-8"))["_meta"]
    assert meta["schema_version"] == "sanitize_marker_v1"
    assert meta["schema"] == 1  # base stamp survives the merge


def test_write_missing_input_path_skipped(tmp_path):
    """An optional input that doesn't exist on disk (e.g. --cast not passed)
    must be silently dropped, not raise."""
    out_path = tmp_path / "manifest.beats.json"
    ghost = tmp_path / "manifest.cast.json"  # never created
    mio.write_manifest(out_path, {"beats": []}, inputs=[ghost, ""])
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["_meta"]["inputs"] == {}


def test_write_default_tool_is_caller_basename(tmp_path):
    out_path = tmp_path / "manifest.x.json"
    mio.write_manifest(out_path, {"x": 1})  # no tool= given
    data = json.loads(out_path.read_text(encoding="utf-8"))
    # caller here is this test module (test_manifest_io), loaded from this file
    assert data["_meta"]["tool"] == "test_manifest_io"


# ---------------------------------------------------------------------------
# read_manifest
# ---------------------------------------------------------------------------

def test_read_missing_raises(tmp_path):
    missing = tmp_path / "manifest.beats.json"
    try:
        mio.read_manifest(missing)
        assert False, "expected ManifestError"
    except mio.ManifestError as e:
        msg = str(e)
        assert str(missing) in msg
        assert "re-run" in msg or "reset" in msg


def test_read_corrupt_raises(tmp_path):
    bad = tmp_path / "manifest.beats.json"
    bad.write_text("{not valid json", encoding="utf-8")
    try:
        mio.read_manifest(bad)
        assert False, "expected ManifestError"
    except mio.ManifestError as e:
        msg = str(e)
        assert str(bad) in msg
        assert "corrupt" in msg.lower()


def test_read_non_object_top_level_raises(tmp_path):
    bad = tmp_path / "manifest.beats.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    try:
        mio.read_manifest(bad)
        assert False, "expected ManifestError"
    except mio.ManifestError as e:
        assert str(bad) in str(e)


def test_required_keys_enforced(tmp_path):
    p = tmp_path / "manifest.beats.json"
    p.write_text(json.dumps({"count_beats": 0}), encoding="utf-8")
    try:
        mio.read_manifest(p, required_keys=("beats",))
        assert False, "expected ManifestError"
    except mio.ManifestError as e:
        msg = str(e)
        assert "beats" in msg
        assert str(p) in msg


def test_required_keys_present_passes(tmp_path):
    p = tmp_path / "manifest.beats.json"
    p.write_text(json.dumps({"beats": [{"group_id": 1}]}), encoding="utf-8")
    obj = mio.read_manifest(p, required_keys=("beats",))
    assert obj["beats"] == [{"group_id": 1}]


def test_read_never_requires_meta(tmp_path):
    """A hand-built fixture manifest with no _meta at all must still read fine
    — readers never require _meta (only producers stamp it)."""
    p = tmp_path / "manifest.timeline.json"
    p.write_text(json.dumps({"timeline": []}), encoding="utf-8")
    obj = mio.read_manifest(p, required_keys=("timeline",))
    assert obj == {"timeline": []}


# ---------------------------------------------------------------------------
# input_sha
# ---------------------------------------------------------------------------

def test_input_sha_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "f.json"
    p.write_bytes(b"hello world")
    assert mio.input_sha(p) == hashlib.sha1(b"hello world").hexdigest()


# ---------------------------------------------------------------------------
# write_manifest: tmp-file cleanup when serialization itself fails
# ---------------------------------------------------------------------------

def test_write_cleans_tmp_on_serialize_failure(tmp_path):
    """json.dump raising mid-write (a non-serializable value slipped into the
    payload) must not leave a `.tmp.<pid>` file behind — the BaseException
    handler removes it before re-raising."""
    out_path = tmp_path / "manifest.beats.json"
    try:
        mio.write_manifest(out_path, {"beats": object()}, tool="t")
        assert False, "expected a serialization TypeError"
    except TypeError:
        pass
    assert not out_path.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == [], f"tmp residue left behind: {leftovers}"
