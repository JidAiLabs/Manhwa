"""Per-series disk use — how much of the Mini a manhwa costs.

Measured, not guessed: 2026-09-20, Omniscient Reader's 309 chapters were 48 GB
(159 MB/chapter) of which 24 GB is rendered video, and a-wimp's 46 chapters
290 MB/chapter. With ~336 GB free and ~979 chapters still to process, the
question "can we keep generating locally" has a number behind it.

Walks with scandir (no `du` subprocess, no shell) so it works the same in tests
and on the Mini. Video bytes are counted separately: they are the deliverable
and the biggest single slice, so they are the first thing to archive.
"""
from __future__ import annotations

import os
from typing import Iterable, Tuple

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm")


def measure_dir(path: str) -> Tuple[int, int]:
    """(total_bytes, video_bytes) under *path*. A missing path measures 0 —
    a deleted chapter dir is a legitimate state, not an error."""
    total = video = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            size = e.stat(follow_symlinks=False).st_size
                            total += size
                            if e.name.lower().endswith(VIDEO_EXT):
                                video += size
                    except OSError:
                        continue
        except (OSError, ValueError):
            continue
    return total, video


def measure_dirs(paths: Iterable[str]) -> Tuple[int, int]:
    """(total_bytes, video_bytes) over several dirs, each counted once."""
    total = video = 0
    for p in sorted({str(x) for x in paths if x}):
        t, v = measure_dir(p)
        total += t
        video += v
    return total, video


def fmt_bytes(n: int) -> str:
    """'48.3 GB' / '159 MB' / '0 B' — for the dashboard, not for arithmetic."""
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return ("%d %s" % (round(n), unit) if unit in ("B", "KB")
                    else "%.1f %s" % (n, unit))
        n /= 1024.0
    return "%.1f TB" % n
