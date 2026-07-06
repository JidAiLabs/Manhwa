"""manifest_freshness.py — manifest completeness + staleness guardrail.

Detects three failure classes:
  missing_manifest (ERROR): a manifest required by the chapter's pipeline
      status is absent from disk.
  corrupt_manifest (ERROR): a required manifest exists but is 0-byte or
      unparseable — presence alone never counts as complete.
  stale_manifest (ERROR): a derived manifest exists but predates one of its
      declared upstream inputs. When both sides carry manifest_io _meta
      input-sha stamps the comparison is content-precise (sha, mtime ignored);
      otherwise it falls back to mtime — except sha_only edges, which are
      skipped entirely without stamps.

The canonical bug this caught: render.plan.clean.json (3 days old) sat next to
fresh manifest.beats.json; the dashboard silently rendered the stale cuts.

The DAG, the per-status required lists and the sentinels are DERIVED from
studio/deps.py — the one dependency table — not hand-maintained here.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# tools/*.py run standalone (subprocess from studio/pipeline.py, importlib in
# tests), so put tools/ and the repo root on sys.path before studio imports —
# same shim tools/prep_qa.py uses.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
for _p in (_TOOLS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from studio import deps as _deps            # noqa: E402
from studio import paths as _studio_paths   # noqa: E402
from manifest_io import input_sha as _input_sha  # noqa: E402

# ---------------------------------------------------------------------------
# All derived from studio.deps.ARTIFACTS — the one dependency table.
# ---------------------------------------------------------------------------

# {output: (required_inputs, optional_inputs, sha_only)}. sha_only edges
# (understood ← vision: the producer re-stamps its input afterwards) are
# enforced ONLY via _meta input-sha stamps and never mtime-compared.
_DAG: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], bool]] = _deps.dag()

# Legacy-shaped view (output -> flat input list) kept for external readers.
MANIFEST_DAG: Dict[str, List[str]] = {
    out: list(req) + list(opt) for out, (req, opt, _sha) in _DAG.items()}

# Status -> required manifest files (cumulative, deepest stage wins).
# 'planned' aliases 'prepped': render.plan.json is transient (an estimate-
# phase artifact that may not persist), so the persistent sentinel/required
# chain for a planned chapter is the clean plan — exactly as before.
STATUS_REQUIRED: Dict[str, List[str]] = {
    s: list(_deps.required_chain(s)) for s in _deps.freshness_statuses()}
STATUS_REQUIRED["planned"] = list(STATUS_REQUIRED["prepped"])

# Ordered from shallowest to deepest for inference (historically:
# visioned, grouped, beated, scripted, planned, prepped).
_STATUS_ORDER = [s for s in _deps.ORDER if s in STATUS_REQUIRED]

# For inference, the deepest required output per status ('planned' without a
# clean plan is not distinguishable from 'scripted' by file presence alone,
# which is acceptable — the check chain is the same).
_STAGE_SENTINEL: Dict[str, str] = {
    s: req[-1] for s, req in STATUS_REQUIRED.items()}


def _issue(code: str, severity: str, file: str, detail: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "file": file, "detail": detail}


def _parses(path: str) -> bool:
    """True iff *path* is readable, non-empty JSON (0-byte fails json.load)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except (ValueError, OSError):
        return False


def _meta_of(path: str, cache: Dict[str, Optional[dict]]) -> Optional[dict]:
    """The manifest's _meta dict, or None (missing/corrupt/unstamped).
    Cached per verify call — each file is parsed at most once."""
    if path not in cache:
        meta = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("_meta"), dict):
                meta = obj["_meta"]
        except (ValueError, OSError):
            meta = None
        cache[path] = meta
    return cache[path]


def _edge_stale_detail(ep_dir: str, output_name: str, input_name: str,
                       sha_only: bool,
                       cache: Dict[str, Optional[dict]]) -> Optional[str]:
    """Staleness verdict for ONE existing output/input pair.

    Returns a human detail string when stale, else None. sha stamps rule when
    BOTH sides carry _meta (mtime ignored for the edge); sha_only edges are
    skipped entirely without stamps; everything else falls back to mtime.
    """
    out_path = os.path.join(ep_dir, output_name)
    in_path = os.path.join(ep_dir, input_name)

    out_meta = _meta_of(out_path, cache)
    stamped_sha = ((out_meta or {}).get("inputs") or {}).get(
        os.path.basename(input_name))
    if stamped_sha and _meta_of(in_path, cache) is not None:
        try:
            current_sha = _input_sha(in_path)
        except OSError:
            return None
        if current_sha != stamped_sha:
            return (f"{output_name} was built from a different "
                    f"{input_name} (stamped sha {stamped_sha[:12]}, now "
                    f"{current_sha[:12]}) — re-run the stage that produces "
                    f"{output_name}")
        return None          # content-precise match — mtime is irrelevant

    if sha_only:
        return None          # no stamps on a sha_only edge → never mtime-compare

    try:
        out_mtime = os.path.getmtime(out_path)
        in_mtime = os.path.getmtime(in_path)
    except OSError:
        return None
    if out_mtime < in_mtime:
        return (f"{output_name} (mtime {out_mtime:.0f}) is older than "
                f"{input_name} (mtime {in_mtime:.0f}) — re-run the stage "
                f"that produces {output_name}")
    return None


def _output_stale_detail(ep_dir: str, output_name: str,
                         cache: Dict[str, Optional[dict]]) -> Optional[str]:
    """First stale detail across all of *output_name*'s DAG edges, or None.
    Absent output or absent inputs skip their edges (missing_manifest owns
    absence); optional inputs are only checked when present."""
    entry = _DAG.get(output_name)
    if entry is None or not os.path.exists(os.path.join(ep_dir, output_name)):
        return None
    required_inputs, optional_inputs, sha_only = entry
    for input_name in required_inputs + optional_inputs:
        if not os.path.exists(os.path.join(ep_dir, input_name)):
            continue
        detail = _edge_stale_detail(ep_dir, output_name, input_name,
                                    sha_only, cache)
        if detail:
            return detail
    return None


def artifact_is_stale(ep_dir: str, artifact: str) -> bool:
    """True iff *artifact* exists and any of its DAG edges reports stale —
    the same edge compare verify_chapter uses (single implementation).
    Pipeline skip-guards (studio/pipeline.py) call this to decide rebuilds."""
    return _output_stale_detail(ep_dir, artifact, {}) is not None


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
        elif not _parses(path):
            issues.append(_issue(
                "corrupt_manifest", "ERROR", name,
                f"{name} exists but is empty or unparseable JSON — the write "
                f"was torn or truncated; re-run the stage that produces {name}"))

    # ---- staleness check across the full DAG --------------------------------
    meta_cache: Dict[str, Optional[dict]] = {}
    for output_name in _DAG:
        detail = _output_stale_detail(ep_dir, output_name, meta_cache)
        if detail:
            issues.append(_issue(
                "stale_manifest", "ERROR", output_name, detail))

    # ---- optional video freshness check ----------------------------------------
    # A rendered video older than the plan it was built from means the chapter was
    # re-prepared after the last render.  This is NOT an ERROR — a chapter that has
    # been re-prepared but not yet re-rendered is the normal state between approvals.
    # WARN so the dashboard can flag the mismatch visibly without blocking the pipeline.
    video = os.path.join(ep_dir, _studio_paths.SEGMENT_MP4)
    clean_plan = p("render.plan.clean.json")
    if os.path.exists(video) and os.path.exists(clean_plan):
        try:
            if os.path.getmtime(video) < os.path.getmtime(clean_plan):
                issues.append(_issue(
                    "stale_video", "WARN",
                    _studio_paths.SEGMENT_MP4,
                    f"{_studio_paths.SEGMENT_MP4} is older than "
                    "render.plan.clean.json"
                    " — re-voice + re-render to match the current narration"))
        except OSError:
            pass

    return issues
