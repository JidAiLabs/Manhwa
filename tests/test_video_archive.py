"""Finished videos move to the archive drive (owner 2026-10-12: Samsung T9 on
the Mini) and leave a link at their old path, so every consumer still finds
them; no drive -> nothing moves."""
import os
import time

import studio.archive as arc


def _video(tmp_path, slug="s", ch="Chapter_1", data=b"mp4-bytes"):
    seg = tmp_path / "ongoing" / slug / ch / "render" / "segment_both.mp4"
    seg.parent.mkdir(parents=True)
    seg.write_bytes(data)
    old = time.time() - 3600
    os.utime(seg, (old, old))
    return seg


def test_a_video_moves_to_the_drive_and_leaves_a_link(tmp_path):
    seg = _video(tmp_path)
    mtime = seg.stat().st_mtime
    dest = tmp_path / "T9" / "s" / "Chapter_1.mp4"
    assert arc.archive_video(seg, dest)
    assert seg.is_symlink() and os.readlink(seg) == str(dest)
    assert dest.read_bytes() == b"mp4-bytes" and seg.read_bytes() == b"mp4-bytes"
    assert dest.stat().st_mtime == mtime                 # freshness checks unchanged
    assert not list(dest.parent.glob("*.part"))
    assert not arc.archive_video(seg, dest)              # already a link: no-op


def test_a_bad_copy_leaves_the_video_where_it_was(tmp_path, monkeypatch):
    seg = _video(tmp_path)
    dest = tmp_path / "T9" / "s" / "Chapter_1.mp4"
    monkeypatch.setattr(arc.filecmp, "cmp", lambda a, b, shallow=True: False)
    assert not arc.archive_video(seg, dest)
    assert not seg.is_symlink() and seg.read_bytes() == b"mp4-bytes"
    assert not dest.exists() and not list(dest.parent.glob("*"))


def test_the_chapter_video_goes_under_its_series_only_when_the_drive_is_there(
        tmp_path, monkeypatch):
    seg = _video(tmp_path)
    monkeypatch.setenv("STUDIO_VIDEO_ARCHIVE", str(tmp_path / "T9"))
    assert arc.archive_segment(seg.parents[1]) is None   # drive not mounted
    assert not seg.is_symlink()
    (tmp_path / "T9").mkdir()
    assert arc.archive_segment(seg.parents[1]) == tmp_path / "T9" / "s" / "Chapter_1.mp4"
    assert seg.is_symlink()


def test_the_sweep_moves_finished_videos_only(tmp_path, monkeypatch, capsys):
    a = _video(tmp_path, ch="Chapter_1")
    b = _video(tmp_path, ch="Chapter_2")
    os.utime(b, None)                                   # just written: a render may be on it
    bundle = tmp_path / "dist" / "bundle_7" / "bundle.mp4"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"bundle")
    os.utime(bundle, (time.time() - 3600,) * 2)
    (tmp_path / "T9").mkdir()
    monkeypatch.setenv("STUDIO_VIDEO_ARCHIVE", str(tmp_path / "T9"))
    monkeypatch.setattr(arc, "REPO", tmp_path)
    assert arc.main([]) == 0                             # dry run moves nothing
    assert not a.is_symlink() and "2 videos" in capsys.readouterr().out
    assert arc.main(["--apply"]) == 0
    assert a.is_symlink() and not b.is_symlink() and bundle.is_symlink()
    assert (tmp_path / "T9" / "_bundles" / "bundle_7.mp4").read_bytes() == b"bundle"
    assert "moved 2" in capsys.readouterr().out
