"""tools/manifest_io.py — atomic, provenance-stamped manifest IO.

Every tools/*.py stage used to write its manifest with an in-place
open(path, "w") + json.dump: a SIGKILL or a full disk mid-write leaves a
torn file, and nothing recorded which inputs produced an output (freshness
could only compare mtimes). This module fixes both, stdlib only:

  write_manifest  — stamps a _meta block (schema/tool/written_at/input shas)
                     then writes to a temp file in the same directory and
                     os.replace()s it over the target (atomic on same fs).
  read_manifest   — raises ManifestError (not a silent empty default) on a
                     missing file, unparseable JSON, or a missing required
                     top-level key.
  input_sha       — sha1 hex of a file's bytes; the hash write_manifest
                     stamps per input path.

No schema registry, no versioning framework — schema:1 is a constant.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple, Union

PathLike = Union[str, "os.PathLike[str]"]

_REMEDY = "re-run the producing stage or studio reset"


class ManifestError(RuntimeError):
    """Raised when a manifest is missing, unparseable, or lacks required keys."""


def input_sha(path: PathLike) -> str:
    """sha1 hex digest of a file's bytes."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: PathLike, *, required_keys: Tuple[str, ...] = ()) -> Dict[str, Any]:
    spath = os.fspath(path)
    if not os.path.isfile(spath):
        raise ManifestError(f"missing manifest: {spath} ({_REMEDY})")
    try:
        with open(spath, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except (ValueError, OSError) as e:
        raise ManifestError(f"corrupt manifest: {spath}: {e} ({_REMEDY})") from e
    if not isinstance(obj, dict):
        raise ManifestError(
            f"corrupt manifest: {spath}: top-level JSON must be an object ({_REMEDY})")
    for key in required_keys:
        if key not in obj:
            raise ManifestError(f"manifest missing required key {key!r}: {spath} ({_REMEDY})")
    return obj


def write_manifest(path: PathLike, obj: Dict[str, Any], *, inputs: Iterable[PathLike] = (),
                    tool: str = "", extra_meta: Optional[Dict[str, Any]] = None) -> None:
    spath = os.fspath(path)
    if not tool:
        # best-effort: basename of whatever module called write_manifest
        caller = inspect.stack(0)[1]
        tool = os.path.splitext(os.path.basename(caller.filename))[0]

    meta: Dict[str, Any] = {
        "schema": 1,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": tool,
        "inputs": {os.path.basename(os.fspath(p)): input_sha(p)
                   for p in inputs if p and os.path.exists(p)},
    }
    if extra_meta:
        meta.update(extra_meta)
    obj["_meta"] = meta

    dirpath = os.path.dirname(spath) or "."
    os.makedirs(dirpath, exist_ok=True)
    tmp_path = f"{spath}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, spath)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
