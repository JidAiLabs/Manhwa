"""
Tests for the Asura source adapter.

No network access — httpx is monkeypatched to return fixture HTML.
Fixture: tests/sources/fixtures/asura_series.html  (LIVE-captured, trimmed)
         tests/sources/fixtures/asura_chapter.html  (LIVE-captured, trimmed)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(fixture_path: Path, status_code: int = 200) -> MagicMock:
    """Return a mock httpx.Response backed by a fixture file."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = fixture_path.read_text()
    mock.raise_for_status = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# list_chapters
# ---------------------------------------------------------------------------

def test_list_chapters_returns_chapter_refs():
    from studio.sources.asura import AsuraAdapter

    fixture = FIXTURES / "asura_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = AsuraAdapter()
        chapters = adapter.list_chapters(
            "https://asurascans.com/comics/nano-machine-89829cb7"
        )

    assert len(chapters) >= 2


def test_list_chapters_ordered_ascending():
    from studio.sources.asura import AsuraAdapter

    fixture = FIXTURES / "asura_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = AsuraAdapter()
        chapters = adapter.list_chapters(
            "https://asurascans.com/comics/nano-machine-89829cb7"
        )

    numbers = [ch.number for ch in chapters]
    assert numbers == sorted(numbers), "chapters must be sorted ascending"


def test_list_chapters_first_chapter_number():
    from studio.sources.asura import AsuraAdapter

    fixture = FIXTURES / "asura_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = AsuraAdapter()
        chapters = adapter.list_chapters(
            "https://asurascans.com/comics/nano-machine-89829cb7"
        )

    assert chapters[0].number == 1.0


def test_list_chapters_last_chapter_number():
    from studio.sources.asura import AsuraAdapter

    fixture = FIXTURES / "asura_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = AsuraAdapter()
        chapters = adapter.list_chapters(
            "https://asurascans.com/comics/nano-machine-89829cb7"
        )

    assert chapters[-1].number == 315.0


def test_list_chapters_urls_are_absolute():
    from studio.sources.asura import AsuraAdapter

    fixture = FIXTURES / "asura_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = AsuraAdapter()
        chapters = adapter.list_chapters(
            "https://asurascans.com/comics/nano-machine-89829cb7"
        )

    for ch in chapters:
        assert ch.url.startswith("https://"), f"Expected absolute URL, got: {ch.url}"


def test_list_chapters_label_contains_number():
    from studio.sources.asura import AsuraAdapter

    fixture = FIXTURES / "asura_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = AsuraAdapter()
        chapters = adapter.list_chapters(
            "https://asurascans.com/comics/nano-machine-89829cb7"
        )

    assert "1" in chapters[0].label
    assert "315" in chapters[-1].label


# ---------------------------------------------------------------------------
# series_meta
# ---------------------------------------------------------------------------

def test_series_meta_returns_title():
    from studio.sources.asura import AsuraAdapter

    fixture = FIXTURES / "asura_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = AsuraAdapter()
        meta = adapter.series_meta(
            "https://asurascans.com/comics/nano-machine-89829cb7"
        )

    assert meta.title == "Nano Machine"


def test_series_meta_slug():
    from studio.sources.asura import AsuraAdapter

    fixture = FIXTURES / "asura_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = AsuraAdapter()
        meta = adapter.series_meta(
            "https://asurascans.com/comics/nano-machine-89829cb7"
        )

    assert meta.slug == "nano-machine"


def test_series_meta_source_id():
    from studio.sources.asura import AsuraAdapter

    fixture = FIXTURES / "asura_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = AsuraAdapter()
        meta = adapter.series_meta(
            "https://asurascans.com/comics/nano-machine-89829cb7"
        )

    assert meta.source == "asura"


# ---------------------------------------------------------------------------
# download (chapter page parsing + image fetching)
# ---------------------------------------------------------------------------

def test_download_extracts_image_urls(tmp_path):
    """download() fetches chapter page, extracts image URLs, streams images."""
    from studio.sources.asura import AsuraAdapter
    from studio.sources.base import ChapterRef

    chapter = ChapterRef(
        number=1,
        label="Chapter 1",
        url="https://asurascans.com/comics/nano-machine-89829cb7/chapter/1",
    )

    chapter_fixture = FIXTURES / "asura_chapter.html"
    chapter_resp = _mock_response(chapter_fixture)

    # Fake image response
    from PIL import Image
    import io
    img_buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(img_buf, format="JPEG")
    img_bytes = img_buf.getvalue()

    img_resp = MagicMock()
    img_resp.status_code = 200
    img_resp.raise_for_status = MagicMock()
    img_resp.content = img_bytes

    call_count = [0]

    def fake_get(url, **kwargs):
        call_count[0] += 1
        if "chapter" in url and call_count[0] == 1:
            return chapter_resp
        return img_resp

    with patch("httpx.get", side_effect=fake_get):
        adapter = AsuraAdapter()
        result = adapter.download(chapter, tmp_path)

    assert len(result) >= 1
    assert all(p.suffix == ".jpg" for p in result)


# ---------------------------------------------------------------------------
# Capabilities and registration
# ---------------------------------------------------------------------------

def test_capabilities():
    from studio.sources.asura import AsuraAdapter
    from studio.sources.base import Capability

    adapter = AsuraAdapter()
    assert Capability.DOWNLOAD in adapter.capabilities
    assert Capability.LIST_CHAPTERS in adapter.capabilities
    assert Capability.SERIES_META in adapter.capabilities


def test_adapter_registered():
    import studio.sources  # noqa: F401 — triggers __init__ imports
    from studio.sources.base import get

    adapter = get("asura")
    assert adapter is not None
    assert adapter.id == "asura"


# ---------------------------------------------------------------------------
# Completeness: mixed extensions + page-contiguity (the half-chapter bug)
# ---------------------------------------------------------------------------

_CDN = "https://cdn.asurascans.com/asura-images/chapters-restored/nano-machine/25"


def test_extract_keeps_mixed_extensions():
    """asura serves pages as MIXED .webp/.jpg; the old .webp-only regex dropped
    the .jpg strips → 3 of 5. Extraction must keep all five."""
    from studio.sources.asura import _extract_image_urls

    html_ = " ".join(f'"{_CDN}/{i:03d}.{ext}"' for i, ext in
                     [(1, "webp"), (2, "jpg"), (3, "webp"), (4, "webp"), (5, "jpg")])
    assert len(_extract_image_urls(html_)) == 5


def test_download_refuses_page_gap(tmp_path):
    """A gap in page numbers means extraction silently missed a strip — fail
    loud rather than ship a half-chapter QA can't see ([1,3,4] gaps at 2)."""
    from studio.sources.asura import AsuraAdapter
    from studio.sources.base import ChapterRef

    html_ = " ".join(f'"{_CDN}/{i:03d}.webp"' for i in (1, 3, 4))
    resp = MagicMock(status_code=200, text=html_)
    resp.raise_for_status = MagicMock()
    with patch("httpx.get", return_value=resp):
        with pytest.raises(RuntimeError, match="GAPS"):
            AsuraAdapter().download(
                ChapterRef(number=25, label="Chapter 25", url=_CDN + "/x"),
                tmp_path)
