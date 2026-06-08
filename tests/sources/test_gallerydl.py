"""
Tests for the gallery-dl backend normalizer.

No network access — subprocess calls are mocked out.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from studio.sources.gallerydl import normalize_into
from studio.sources.base import UnsupportedSource


# ---------------------------------------------------------------------------
# normalize_into — natural-sort ordering
# ---------------------------------------------------------------------------

def test_normalize_orders_and_converts(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    for name in ["p10.webp", "p2.png", "p1.jpg"]:
        Image.new("RGB", (10, 10)).save(src / name)
    dest = tmp_path / "ep"
    out = normalize_into(src, dest)
    assert [p.name for p in out] == ["001.jpg", "002.jpg", "003.jpg"]
    assert all(p.suffix == ".jpg" for p in out)


def test_normalize_creates_dest_dir(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    Image.new("RGB", (5, 5)).save(src / "a.png")
    dest = tmp_path / "new" / "nested"
    normalize_into(src, dest)
    assert dest.is_dir()


def test_normalize_converts_webp_to_jpg(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    Image.new("RGB", (8, 8)).save(src / "img.webp")
    dest = tmp_path / "out"
    out = normalize_into(src, dest)
    assert out[0].suffix == ".jpg"


def test_normalize_metadata_ordering(tmp_path):
    """When .json sidecars carry a numeric 'num' field, order by that instead."""
    src = tmp_path / "raw"
    src.mkdir()
    # Write images in reverse order on disk; sidecar says true order
    for fname, page_num in [("img_b.jpg", 1), ("img_a.jpg", 2)]:
        Image.new("RGB", (6, 6)).save(src / fname)
        sidecar = src / (Path(fname).stem + ".json")
        sidecar.write_text(json.dumps({"num": page_num}))
    dest = tmp_path / "out"
    out = normalize_into(src, dest)
    # img_b (num=1) must come first → 001.jpg  ;  img_a (num=2) → 002.jpg
    assert len(out) == 2
    assert out[0].name == "001.jpg"
    assert out[1].name == "002.jpg"


def test_normalize_metadata_gap_raises(tmp_path):
    """A gap in the metadata page sequence raises ValueError."""
    src = tmp_path / "raw"
    src.mkdir()
    for fname, num in [("a.jpg", 1), ("b.jpg", 3)]:  # gap at 2
        Image.new("RGB", (4, 4)).save(src / fname)
        (src / (Path(fname).stem + ".json")).write_text(json.dumps({"num": num}))
    with pytest.raises(ValueError, match="gap"):
        normalize_into(src, tmp_path / "out")


def test_normalize_empty_src_returns_empty(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    dest = tmp_path / "out"
    out = normalize_into(src, dest)
    assert out == []


# ---------------------------------------------------------------------------
# gallerydl_supports — subprocess mocked
# ---------------------------------------------------------------------------

def test_supports_returns_true_on_zero_returncode():
    from studio.sources.gallerydl import gallerydl_supports
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result):
        assert gallerydl_supports("https://example.com/series") is True


def test_supports_returns_false_on_unsupported_url():
    from studio.sources.gallerydl import gallerydl_supports
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "error: Unsupported URL: https://example.com"
    with patch("subprocess.run", return_value=mock_result):
        assert gallerydl_supports("https://example.com/series") is False


def test_supports_returns_false_on_no_extractor():
    from studio.sources.gallerydl import gallerydl_supports
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "error: no suitable extractor for URL"
    with patch("subprocess.run", return_value=mock_result):
        assert gallerydl_supports("https://example.com/series") is False


def test_supports_raises_on_unexpected_error():
    from studio.sources.gallerydl import gallerydl_supports
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Connection refused"
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError):
            gallerydl_supports("https://example.com/series")


# ---------------------------------------------------------------------------
# run_download — subprocess mocked
# ---------------------------------------------------------------------------

def test_run_download_calls_gallery_dl(tmp_path):
    from studio.sources.gallerydl import run_download
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        run_download("https://example.com/ch1", tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "gallery-dl" in cmd
        assert str(tmp_path) in cmd


def test_run_download_raises_unsupported_source_on_extractor_error(tmp_path):
    from studio.sources.gallerydl import run_download
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "error: no suitable extractor"
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(UnsupportedSource):
            run_download("https://bad.url/ch1", tmp_path)


def test_run_download_raises_runtime_error_on_other_failure(tmp_path):
    from studio.sources.gallerydl import run_download
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "some other error"
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError):
            run_download("https://example.com/ch1", tmp_path)
