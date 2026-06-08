"""
gallery-dl backend for the manhwa pipeline.

Three public functions:

* ``gallerydl_supports(url)``  — probe whether gallery-dl can handle a URL
* ``run_download(url, tmp_dir)`` — invoke gallery-dl and download into a temp dir
* ``normalize_into(src_dir, dest_dir)`` — convert and rename raw downloads to
  a canonical ``001.jpg / 002.jpg / …`` sequence

The subprocess-level functions (gallerydl_supports / run_download) are kept
thin and well-commented so they are easy to mock in tests.  The normalizer is
pure Python and fully unit-testable without touching the network.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

from studio.sources.base import UnsupportedSource

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Invoke gallery-dl via the current interpreter so it works regardless of
# whether the venv's bin/ is on PATH (it usually isn't outside `activate`).
_GDL_CMD = [sys.executable, "-m", "gallery_dl"]

_EXTRACTOR_ERROR_PHRASES = ("no suitable extractor", "unsupported url")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def gallerydl_supports(url: str) -> bool:
    """
    Return True if gallery-dl claims it can handle *url*, False if it
    explicitly says the URL is unsupported, or raise RuntimeError for any
    other non-zero exit (e.g. network error, timeout).

    We use ``--simulate`` so no files are written during the probe.
    """
    result = subprocess.run(
        [*_GDL_CMD, "--simulate", url],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 0:
        return True

    # gallery-dl exits non-zero for many reasons; only "no extractor" means
    # "this URL is structurally unsupported" — everything else is a real error.
    stderr_lower = result.stderr.lower()
    if any(phrase in stderr_lower for phrase in _EXTRACTOR_ERROR_PHRASES):
        return False

    raise RuntimeError(
        f"gallery-dl probe failed (exit {result.returncode}): {result.stderr.strip()}"
    )


def run_download(url: str, tmp_dir: Path, sleep: float = 2.0) -> None:
    """
    Run gallery-dl to download *url* into *tmp_dir*.

    ``--write-metadata`` causes gallery-dl to emit ``.json`` sidecars
    alongside each image, which ``normalize_into`` uses for page ordering.

    Raises:
        UnsupportedSource — gallery-dl cannot find an extractor for *url*
        RuntimeError      — any other non-zero exit
    """
    result = subprocess.run(
        [
            *_GDL_CMD,
            "--dest", str(tmp_dir),
            "--sleep", str(sleep),
            "--write-metadata",
            url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        return

    stderr_lower = result.stderr.lower()
    if any(phrase in stderr_lower for phrase in _EXTRACTOR_ERROR_PHRASES):
        raise UnsupportedSource(
            f"gallery-dl has no extractor for '{url}': {result.stderr.strip()}"
        )

    raise RuntimeError(
        f"gallery-dl exited {result.returncode} for '{url}': {result.stderr.strip()}"
    )


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

def _natural_sort_key(path: Path) -> list[int | str]:
    """
    Split the filename into alternating text / integer chunks so that
    ``p2.jpg < p10.jpg`` rather than the lexicographic ``p10 < p2``.
    """
    parts: list[int | str] = []
    for chunk in re.split(r"(\d+)", path.name):
        parts.append(int(chunk) if chunk.isdigit() else chunk.lower())
    return parts


def _metadata_index(sidecar: Path) -> int | None:
    """
    Return the numeric page index stored in a gallery-dl JSON sidecar, or
    None if the sidecar does not exist / does not contain a recognisable
    numeric page field.

    gallery-dl uses the field name ``"num"`` for sequential page indices.
    We also accept ``"page"`` as a fallback.
    """
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    for field in ("num", "page"):
        val = data.get(field)
        if isinstance(val, (int, float)):
            return int(val)
    return None


def normalize_into(src_dir: Path, dest_dir: Path) -> list[Path]:
    """
    Gather all images from *src_dir* (recursively), order them, convert each
    to JPEG, and write them as ``001.jpg``, ``002.jpg``, … into *dest_dir*.

    **Ordering strategy**:

    1. *Metadata ordering* — if **every** image file has a companion ``.json``
       sidecar that carries a numeric page index, order by that index and
       validate that the sequence has no gaps.
    2. *Natural sort* — otherwise, sort by filename using natural ordering
       (so ``p2.jpg`` comes before ``p10.jpg``).

    Returns the list of written paths in page order.

    Raises:
        ValueError — metadata indices have a gap in the sequence
    """
    # Collect image files only (not sidecars or other artefacts)
    images = [
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]

    if not images:
        return []

    # --- Attempt metadata-based ordering ---
    meta_indices: list[tuple[int, Path]] = []
    for img in images:
        sidecar = img.with_suffix(".json")
        idx = _metadata_index(sidecar)
        if idx is not None:
            meta_indices.append((idx, img))

    if len(meta_indices) == len(images):
        # Every image has a sidecar with a valid index — use metadata ordering
        meta_indices.sort(key=lambda t: t[0])
        ordered = [p for _, p in meta_indices]

        # Validate contiguous sequence (1-based or 0-based, no gaps allowed)
        indices = [i for i, _ in meta_indices]
        start = indices[0]
        expected = list(range(start, start + len(indices)))
        if indices != expected:
            raise ValueError(
                f"Metadata page index sequence has a gap: {indices}"
            )
    else:
        # Fall back to natural sort
        ordered = sorted(images, key=_natural_sort_key)

    # --- Write normalised JPEGs ---
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, src_path in enumerate(ordered, start=1):
        dest_path = dest_dir / f"{i:03d}.jpg"
        with Image.open(src_path) as img:
            # Convert to RGB so we can always save as JPEG (no alpha channel)
            img.convert("RGB").save(dest_path, format="JPEG")
        written.append(dest_path)

    return written
