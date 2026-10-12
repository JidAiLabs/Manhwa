"""Finished videos live on the archive drive (owner 2026-10-12: a Samsung T9 on
the Mini, `[render] video_archive` in studio.toml).

A video is still made on the internal disk, so a render never depends on the
drive. Once it is finished it MOVES to <archive>/<series>/<chapter>.mp4
(bundles: <archive>/_bundles/bundle_<id>.mp4), checked byte for byte, and a
link is left at the old path. The dashboard player, the bundle concat and
reconcile keep reading the old path. Drive not mounted -> nothing moves; the
video stays local until the next sweep:

    python -m studio.archive            # dry run: what would move
    python -m studio.archive --apply    # move it
"""
from __future__ import annotations

import argparse
import contextlib
import filecmp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from studio import paths

REPO = Path(__file__).resolve().parents[1]
SETTLED_SEC = 600   # the sweep skips a video written this recently: a render may be on it


def root() -> Optional[Path]:
    """The archive folder, or None when it is not configured or its drive is
    not mounted. The folder itself is never created here: a missing folder IS
    the unplugged drive (mkdir would quietly fill the internal disk instead)."""
    from studio.config import load
    r = load().video_archive
    return Path(r) if r and os.path.isdir(r) else None


def archive_video(src, dest) -> bool:
    """Move ``src`` to ``dest`` and leave a link at ``src``. False, with ``src``
    untouched, when it is already a link or anything fails."""
    src, dest = Path(src), Path(dest)
    if src.is_symlink() or not src.is_file():
        return False
    part = dest.with_name(dest.name + ".part")
    lnk = src.with_name(src.name + ".lnk")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = src.stat()
        shutil.copy2(src, part)                       # copy2 keeps the mtime
        if not filecmp.cmp(src, part, shallow=False):
            raise OSError(f"copy of {src} differs")
        after = src.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise OSError(f"{src} changed while it was copied")
        os.replace(part, dest)
        with contextlib.suppress(FileNotFoundError):
            lnk.unlink()
        os.symlink(dest, lnk)
        os.replace(lnk, src)                          # the video is never absent
        return True
    except OSError:
        for leftover in (part, lnk):
            with contextlib.suppress(OSError):
                leftover.unlink()
        return False


def archive_segment(ep_dir) -> Optional[Path]:
    """Archive a chapter's rendered video; its new path, or None."""
    r = root()
    ep = Path(ep_dir)
    dest = r / ep.parent.name / f"{ep.name}.mp4" if r else None
    return dest if dest and archive_video(ep / paths.SEGMENT_MP4, dest) else None


def archive_bundle(mp4) -> Optional[Path]:
    """Archive a bundle's concatenated video (dist/bundle_<id>/bundle.mp4)."""
    r = root()
    mp4 = Path(mp4)
    dest = r / "_bundles" / f"{mp4.parent.name}.mp4" if r else None
    return dest if dest and archive_video(mp4, dest) else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="move (default: dry run)")
    args = ap.parse_args(argv)
    r = root()
    if r is None:
        print("[archive] no archive drive mounted (studio.toml [render] video_archive)")
        return 1
    cutoff = time.time() - SETTLED_SEC
    todo = [(p, archive_segment, p.parents[1])
            for p in sorted(REPO.glob(f"ongoing/*/*/{paths.SEGMENT_MP4}"))]
    todo += [(p, archive_bundle, p) for p in sorted(REPO.glob("dist/bundle_*/bundle.mp4"))]
    todo = [t for t in todo if not t[0].is_symlink() and t[0].is_file()
            and t[0].stat().st_mtime < cutoff]
    size = sum(p.stat().st_size for p, _, _ in todo)
    print(f"[archive] {len(todo)} videos, {size / 1e9:.1f} GB -> {r}")
    if not args.apply:
        return 0
    moved = failed = 0
    for i, (p, fn, arg) in enumerate(todo, 1):
        if fn(arg):
            moved += 1
        else:
            failed += 1
            print(f"[archive] FAILED {p.relative_to(REPO)} (left in place)")
        if i % 50 == 0:
            print(f"[archive] {i}/{len(todo)}", flush=True)
    print(f"[archive] moved {moved}, failed {failed}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
