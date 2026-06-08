"""
Tests for the Webtoon source adapter.

No network access — gallery-dl subprocess is monkeypatched to return fixture data.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_result(fixture_path: Path) -> MagicMock:
    """Return a mock subprocess.CompletedProcess that writes fixture JSON to stdout."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = fixture_path.read_text()
    mock.stderr = ""
    return mock


# ---------------------------------------------------------------------------
# list_chapters
# ---------------------------------------------------------------------------

def test_list_chapters_returns_chapter_refs():
    from studio.sources.webtoon import WebtoonAdapter

    fixture = FIXTURES / "webtoon_chapters.json"
    mock_result = _make_run_result(fixture)

    with patch("subprocess.run", return_value=mock_result):
        adapter = WebtoonAdapter()
        chapters = adapter.list_chapters(
            "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
        )

    assert len(chapters) == 3  # fixture has 3 episodes
    # Should be ordered ascending by episode_no
    assert chapters[0].number == 1
    assert chapters[1].number == 2
    assert chapters[2].number == 3


def test_list_chapters_urls_are_viewer_urls():
    from studio.sources.webtoon import WebtoonAdapter

    fixture = FIXTURES / "webtoon_chapters.json"
    mock_result = _make_run_result(fixture)

    with patch("subprocess.run", return_value=mock_result):
        adapter = WebtoonAdapter()
        chapters = adapter.list_chapters(
            "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
        )

    # Each URL should be a viewer URL
    for ch in chapters:
        assert "webtoons.com" in ch.url
        assert "viewer" in ch.url


def test_list_chapters_labels_include_episode_number():
    from studio.sources.webtoon import WebtoonAdapter

    fixture = FIXTURES / "webtoon_chapters.json"
    mock_result = _make_run_result(fixture)

    with patch("subprocess.run", return_value=mock_result):
        adapter = WebtoonAdapter()
        chapters = adapter.list_chapters(
            "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
        )

    # Labels should reference the episode number
    assert "1" in chapters[0].label
    assert "2" in chapters[1].label


def test_list_chapters_calls_gallery_dl_with_minus_j():
    """Confirm we invoke gallery-dl -j (not --simulate) on the list URL."""
    from studio.sources.webtoon import WebtoonAdapter

    fixture = FIXTURES / "webtoon_chapters.json"
    mock_result = _make_run_result(fixture)

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        adapter = WebtoonAdapter()
        adapter.list_chapters(
            "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
        )
        cmd = mock_run.call_args[0][0]
        # Invoked as `python -m gallery_dl -j <url>` (module form, PATH-independent)
        assert "gallery_dl" in " ".join(cmd)
        assert "-j" in cmd


# ---------------------------------------------------------------------------
# series_meta
# ---------------------------------------------------------------------------

def test_series_meta_returns_title():
    from studio.sources.webtoon import WebtoonAdapter

    fixture = FIXTURES / "webtoon_chapters.json"
    mock_result = _make_run_result(fixture)

    with patch("subprocess.run", return_value=mock_result):
        adapter = WebtoonAdapter()
        meta = adapter.series_meta(
            "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
        )

    # Title derived from comic field 'omniscient-reader' -> 'Omniscient Reader'
    assert meta.title == "Omniscient Reader"


def test_series_meta_returns_slug():
    from studio.sources.webtoon import WebtoonAdapter

    fixture = FIXTURES / "webtoon_chapters.json"
    mock_result = _make_run_result(fixture)

    with patch("subprocess.run", return_value=mock_result):
        adapter = WebtoonAdapter()
        meta = adapter.series_meta(
            "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
        )

    assert meta.slug == "omniscient-reader"


def test_series_meta_source_id():
    from studio.sources.webtoon import WebtoonAdapter

    fixture = FIXTURES / "webtoon_chapters.json"
    mock_result = _make_run_result(fixture)

    with patch("subprocess.run", return_value=mock_result):
        adapter = WebtoonAdapter()
        meta = adapter.series_meta(
            "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
        )

    assert meta.source == "webtoon"


# ---------------------------------------------------------------------------
# download delegates to gallerydl.run_download
# ---------------------------------------------------------------------------

def test_download_calls_run_download(tmp_path):
    from studio.sources.webtoon import WebtoonAdapter
    from studio.sources.base import ChapterRef

    chapter = ChapterRef(
        number=1,
        label="Episode 1",
        url="https://www.webtoons.com/en/action/omniscient-reader/episode-0-prologue/viewer?title_no=2154&episode_no=1",
    )

    with patch("studio.sources.webtoon.run_download") as mock_dl, \
         patch("studio.sources.webtoon.normalize_into", return_value=[]) as mock_norm:
        mock_dl.return_value = None
        adapter = WebtoonAdapter()
        adapter.download(chapter, tmp_path)

    mock_dl.assert_called_once()
    call_args = mock_dl.call_args
    assert call_args[0][0] == chapter.url  # first positional arg is the URL


# ---------------------------------------------------------------------------
# Capabilities and registration
# ---------------------------------------------------------------------------

def test_capabilities():
    from studio.sources.webtoon import WebtoonAdapter
    from studio.sources.base import Capability

    adapter = WebtoonAdapter()
    assert Capability.DOWNLOAD in adapter.capabilities
    assert Capability.LIST_CHAPTERS in adapter.capabilities
    assert Capability.SERIES_META in adapter.capabilities


def test_adapter_registered():
    import studio.sources  # noqa: F401 — triggers __init__ imports
    from studio.sources.base import get

    adapter = get("webtoon")
    assert adapter is not None
    assert adapter.id == "webtoon"
