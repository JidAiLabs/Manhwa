"""studio.paths: naming authority for the chapter render artifact.

find_segment_mp4 mirrors the concat handlers' old "segment_*.mp4 else *.mp4"
glob fallback, but picks the NEWEST match by mtime instead of the
alphabetically-first one (worker._h_concat / _concat_intro_ch1 used to do
``sorted(rdir.glob(...))[0]``).
"""
import os
import time

from studio import paths


def _touch(path, mtime):
    path.write_bytes(b"\x00")
    os.utime(path, (mtime, mtime))


def test_segment_mp4_constant_is_the_canonical_relpath():
    assert paths.SEGMENT_MP4 == "render/segment_both.mp4"


def test_find_segment_mp4_prefers_canonical_segment_both(tmp_path):
    rdir = tmp_path / "render"
    rdir.mkdir()
    now = time.time()
    _touch(rdir / "segment_none.mp4", now + 100)   # newer, but not canonical
    _touch(rdir / "segment_both.mp4", now)
    assert paths.find_segment_mp4(tmp_path) == rdir / "segment_both.mp4"


def test_find_segment_mp4_falls_back_to_newest_segment_star(tmp_path):
    rdir = tmp_path / "render"
    rdir.mkdir()
    now = time.time()
    _touch(rdir / "segment_intro.mp4", now)
    _touch(rdir / "segment_outro.mp4", now + 100)   # newest wins
    assert paths.find_segment_mp4(tmp_path) == rdir / "segment_outro.mp4"


def test_find_segment_mp4_falls_back_to_newest_any_mp4(tmp_path):
    rdir = tmp_path / "render"
    rdir.mkdir()
    now = time.time()
    _touch(rdir / "preview.mp4", now)
    _touch(rdir / "single.mp4", now + 100)          # newest wins, no segment_*
    assert paths.find_segment_mp4(tmp_path) == rdir / "single.mp4"


def test_find_segment_mp4_none_when_nothing_rendered(tmp_path):
    assert paths.find_segment_mp4(tmp_path) is None         # no render/ dir
    (tmp_path / "render").mkdir()
    assert paths.find_segment_mp4(tmp_path) is None         # empty render/


def test_find_segment_mp4_skips_candidate_whose_stat_races_away(tmp_path, monkeypatch):
    """A file that vanishes in the gap between the p.exists() filter and the
    mtime stat (deleted by a concurrent rewind/cleanup) must be skipped, not
    raise OSError out of find_segment_mp4 into its caller (_h_concat). The
    first stat() (inside p.exists()) is left to succeed -- only the second
    (the explicit mtime read) is made to fail, isolating the guarded-stat
    code path from the p.exists() pre-filter."""
    from pathlib import Path

    rdir = tmp_path / "render"
    rdir.mkdir()
    now = time.time()
    gone = rdir / "segment_gone.mp4"
    _touch(gone, now + 100)          # newer mtime, but its 2nd stat() will raise
    _touch(rdir / "segment_ok.mp4", now)

    real_stat = Path.stat
    calls = {"gone": 0}

    def flaky_stat(self, *args, **kwargs):
        if self.name == "segment_gone.mp4":
            calls["gone"] += 1
            if calls["gone"] > 1:
                raise OSError("vanished mid-race")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    assert paths.find_segment_mp4(tmp_path) == rdir / "segment_ok.mp4"
