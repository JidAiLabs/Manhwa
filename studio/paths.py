import os
from pathlib import Path
from typing import Optional


def resolve_rel(manifest_path, stored: str) -> Path:
    """Resolve a stored path against the manifest's directory.

    Absolute stored paths pass through unchanged (back-compat).
    """
    if os.path.isabs(stored):
        return Path(stored)
    return Path(manifest_path).parent / stored


# Canonical chapter render artifact, relative to ep_dir. Naming authority for
# consumers that need to find/serve a chapter's rendered video — keep in sync
# with studio/catalog/reconcile.py's _STATUS_MARKERS/_STAGE_ARTIFACT (which
# import this constant rather than re-hardcoding it) and, if a "rendered"
# stage is ever added there, with studio/pipeline.py's _STAGE_TABLE.
SEGMENT_MP4 = "render/segment_both.mp4"


def find_segment_mp4(ep_dir) -> Optional[Path]:
    """Locate a chapter's rendered segment under ``ep_dir/render``.

    Preference order: the canonical SEGMENT_MP4 (render/segment_both.mp4) if
    it exists; else the NEWEST-by-mtime ``render/segment_*.mp4``; else the
    newest ``render/*.mp4`` of any name; else ``None``.

    Mirrors the concat handlers' old glob fallback
    (``sorted(rdir.glob("segment_*.mp4")) or sorted(rdir.glob("*.mp4"))``)
    but picks the NEWEST match instead of the alphabetically-first one — a
    stale earlier-lettered file must never win over a fresh render.
    """
    ep_dir = Path(ep_dir)
    canonical = ep_dir / SEGMENT_MP4
    if canonical.exists():
        return canonical
    rdir = ep_dir / "render"
    for pattern in ("segment_*.mp4", "*.mp4"):
        found = sorted(rdir.glob(pattern), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        if found:
            return found[0]
    return None
