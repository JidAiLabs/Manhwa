"""manifest_freshness.py — manifest completeness + staleness guardrail.

Detects two failure classes:
  missing_manifest (ERROR): a manifest required by the chapter's pipeline
      status is absent from disk.
  stale_manifest (ERROR): a derived manifest exists but is OLDER (mtime) than
      one of its declared upstream inputs — it was not rebuilt after its source
      changed.

The canonical bug this caught: render.plan.clean.json (3 days old) sat next to
fresh manifest.beats.json; the dashboard silently rendered the stale cuts.

Pure os.path — no imports from studio/ or tools/.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# DAG: output -> list of inputs it must be newer than.
# manifest.cast.json is OPTIONAL for beats (only checked when present).
# ---------------------------------------------------------------------------
MANIFEST_DAG: Dict[str, List[str]] = {
    "manifest.panels.understood.json": ["manifest.vision.json"],
    "manifest.groups.json":            ["manifest.panels.understood.json"],
    "manifest.story.json":             ["manifest.panels.understood.json"],
    "manifest.beats.json":             ["manifest.groups.json", "manifest.cast.json"],
    "manifest.script.json":            ["manifest.beats.json"],
    "render.plan.json":                ["manifest.script.json"],
    "render.plan.clean.json":          ["render.plan.json", "manifest.beats.json"],
}

# manifest.cast.json is optional — only staleness-checked when the file exists
_OPTIONAL_INPUTS = {"manifest.cast.json"}

# ---------------------------------------------------------------------------
# Status -> required manifest files (cumulative, deepest stage wins).
# ---------------------------------------------------------------------------
STATUS_REQUIRED: Dict[str, List[str]] = {
    "visioned": [
        "manifest.vision.json",
    ],
    "grouped": [
        "manifest.vision.json",
        "manifest.panels.understood.json",
        "manifest.groups.json",
    ],
    "beated": [
        "manifest.vision.json",
        "manifest.panels.understood.json",
        "manifest.groups.json",
        "manifest.beats.json",
    ],
    "scripted": [
        "manifest.vision.json",
        "manifest.panels.understood.json",
        "manifest.groups.json",
        "manifest.beats.json",
        "manifest.script.json",
    ],
    "planned": [
        "manifest.vision.json",
        "manifest.panels.understood.json",
        "manifest.groups.json",
        "manifest.beats.json",
        "manifest.script.json",
        "render.plan.json",
        "render.plan.clean.json",
    ],
    "prepped": [
        "manifest.vision.json",
        "manifest.panels.understood.json",
        "manifest.groups.json",
        "manifest.beats.json",
        "manifest.script.json",
        "render.plan.json",
        "render.plan.clean.json",
    ],
}

# Ordered from shallowest to deepest for inference
_STATUS_ORDER = ["visioned", "grouped", "beated", "scripted", "planned", "prepped"]

# Primary output that marks a stage as "reached" (first required manifest)
_STATUS_PRIMARY: Dict[str, str] = {
    s: files[0] for s, files in STATUS_REQUIRED.items()
}
# For inference, pick the deepest unique output per stage
_STAGE_SENTINEL: Dict[str, str] = {
    "visioned": "manifest.vision.json",
    "grouped":  "manifest.groups.json",
    "beated":   "manifest.beats.json",
    "scripted": "manifest.script.json",
    "planned":  "render.plan.json",
    "prepped":  "render.plan.clean.json",
}


def _issue(code: str, severity: str, file: str, detail: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "file": file, "detail": detail}


def verify_chapter(ep_dir: str,
                   status: Optional[str] = None) -> List[Dict[str, str]]:
    """Return a list of issue dicts: {code, severity, file, detail}.

    missing_manifest (ERROR): an expected manifest for `status` is absent.
    stale_manifest   (ERROR): a derived manifest exists but is OLDER (mtime)
        than one of its declared upstream inputs.

    `status` None → infer the deepest stage whose sentinel output exists, then
    check the full required chain up to that stage.

    Missing ep_dir → returns a single missing_manifest issue, no exception.
    """
    if not os.path.isdir(ep_dir):
        return [_issue(
            "missing_manifest", "ERROR", ep_dir,
            f"episode directory does not exist: {ep_dir}")]

    def p(name: str) -> str:
        return os.path.join(ep_dir, name)

    # ---- resolve effective status ----------------------------------------
    effective_status = status
    if effective_status is None:
        for s in reversed(_STATUS_ORDER):
            sentinel = _STAGE_SENTINEL[s]
            if os.path.exists(p(sentinel)):
                effective_status = s
                break

    if effective_status is None:
        # No manifests at all — nothing to check
        return []

    required = STATUS_REQUIRED.get(effective_status, [])

    issues: List[Dict[str, str]] = []

    # ---- completeness check -----------------------------------------------
    for name in required:
        path = p(name)
        if not os.path.exists(path):
            issues.append(_issue(
                "missing_manifest", "ERROR", name,
                f"{name} is required for status={effective_status!r} "
                f"but does not exist in {ep_dir}"))

    # ---- staleness check across the full DAG --------------------------------
    for output_name, inputs in MANIFEST_DAG.items():
        out_path = p(output_name)
        if not os.path.exists(out_path):
            continue  # nothing to check if output isn't there
        try:
            out_mtime = os.path.getmtime(out_path)
        except OSError:
            continue

        for input_name in inputs:
            if input_name in _OPTIONAL_INPUTS and not os.path.exists(p(input_name)):
                continue  # optional input absent — skip edge
            in_path = p(input_name)
            if not os.path.exists(in_path):
                continue  # input absent — skip (missing_manifest handles it)
            try:
                in_mtime = os.path.getmtime(in_path)
            except OSError:
                continue
            if out_mtime < in_mtime:
                issues.append(_issue(
                    "stale_manifest", "ERROR", output_name,
                    f"{output_name} (mtime {out_mtime:.0f}) is older than "
                    f"{input_name} (mtime {in_mtime:.0f}) — re-run the stage "
                    f"that produces {output_name}"))
                break  # one stale report per output is enough

    return issues
