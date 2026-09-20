"""Per-series disk use: the number behind "can we keep generating locally".

Measured 2026-09-20: ORV 309 chapters = 48 GB (24 GB of it rendered video),
336 GB free on the Mini, 979 chapters still to process.
"""
from __future__ import annotations

from studio.catalog import disk


def _mk(tmp_path, name, size, sub=""):
    d = tmp_path / sub if sub else tmp_path
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"x" * size)


def test_measure_counts_video_bytes_separately(tmp_path):
    _mk(tmp_path, "001.jpg", 100)
    _mk(tmp_path, "segment_both.mp4", 500, sub="render")
    _mk(tmp_path, "clip.wav", 50, sub="tts")
    total, video = disk.measure_dir(str(tmp_path))
    assert (total, video) == (650, 500)


def test_a_missing_directory_measures_zero(tmp_path):
    assert disk.measure_dir(str(tmp_path / "gone")) == (0, 0)


def test_measure_dirs_counts_each_path_once(tmp_path):
    a = tmp_path / "a"; _mk(a, "x.mp4", 200)
    b = tmp_path / "b"; _mk(b, "y.jpg", 100)
    assert disk.measure_dirs([str(a), str(b), str(a)]) == (300, 200)


def test_fmt_bytes_reads_like_the_dashboard_shows_it():
    assert disk.fmt_bytes(0) == "0 B"
    assert disk.fmt_bytes(1536) == "2 KB"
    assert disk.fmt_bytes(159 * 1024 ** 2) == "159.0 MB"
    assert disk.fmt_bytes(48 * 1024 ** 3) == "48.0 GB"
