"""
Tests for the Elftoon source adapter (native httpx + selectolax).

No network access — httpx.get is monkeypatched to return fixture HTML.
Fixtures: tests/sources/fixtures/elftoon_series.html  (LIVE-captured, trimmed)
          tests/sources/fixtures/elftoon_chapter.html  (LIVE-captured, trimmed)

Elftoon uses a WordPress theme with ts_reader.run() for chapter pages
(NOT standard Madara .reading-content img — images come from a JS object).
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
    from studio.sources.elftoon import ElftoonAdapter

    fixture = FIXTURES / "elftoon_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = ElftoonAdapter()
        chapters = adapter.list_chapters(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )

    assert len(chapters) >= 2


def test_list_chapters_ordered_ascending():
    from studio.sources.elftoon import ElftoonAdapter

    fixture = FIXTURES / "elftoon_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = ElftoonAdapter()
        chapters = adapter.list_chapters(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )

    numbers = [ch.number for ch in chapters]
    assert numbers == sorted(numbers), "chapters must be sorted ascending"


def test_list_chapters_first_chapter():
    from studio.sources.elftoon import ElftoonAdapter

    fixture = FIXTURES / "elftoon_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = ElftoonAdapter()
        chapters = adapter.list_chapters(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )

    assert chapters[0].number == 1.0


def test_list_chapters_last_chapter():
    from studio.sources.elftoon import ElftoonAdapter

    fixture = FIXTURES / "elftoon_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = ElftoonAdapter()
        chapters = adapter.list_chapters(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )

    # 95, NOT 98. The fixture's three highest entries (96, 97, 98) are
    # ANNOUNCED but unpublished: real titles and dates, href="#". This
    # assertion used to read 98.0, which pinned the bug — a placeholder was
    # being turned into a chapter row whose URL became "https://elftoon.com#",
    # a link that returns HTTP 200 and serves the homepage. 95 is the highest
    # chapter that actually has a page.
    assert chapters[-1].number == 95.0


def test_unpublished_chapters_never_become_rows():
    """The same three-placeholder shape appears in this fixture (96-98) and on
    the live site today (108-110), so this is the site's normal behaviour for
    upcoming chapters, not a one-off."""
    from studio.sources.base import is_fetchable_url
    from studio.sources.elftoon import ElftoonAdapter

    mock_resp = _mock_response(FIXTURES / "elftoon_series.html")
    with patch("httpx.get", return_value=mock_resp):
        chapters = ElftoonAdapter().list_chapters(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )
    assert {96.0, 97.0, 98.0}.isdisjoint({c.number for c in chapters})
    assert all(is_fetchable_url(c.url) for c in chapters)
    assert not any(c.url.rstrip("/").endswith("elftoon.com#") for c in chapters)


def test_list_chapters_urls_are_absolute():
    from studio.sources.elftoon import ElftoonAdapter

    fixture = FIXTURES / "elftoon_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = ElftoonAdapter()
        chapters = adapter.list_chapters(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )

    for ch in chapters:
        assert ch.url.startswith("https://"), f"Expected absolute URL, got: {ch.url}"


def test_list_chapters_label_contains_number():
    from studio.sources.elftoon import ElftoonAdapter

    fixture = FIXTURES / "elftoon_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = ElftoonAdapter()
        chapters = adapter.list_chapters(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )

    assert "1" in chapters[0].label
    assert "95" in chapters[-1].label      # 96-98 are unpublished placeholders


# ---------------------------------------------------------------------------
# series_meta
# ---------------------------------------------------------------------------

def test_series_meta_returns_title():
    from studio.sources.elftoon import ElftoonAdapter

    fixture = FIXTURES / "elftoon_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = ElftoonAdapter()
        meta = adapter.series_meta(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )

    assert meta.title == "Infinite Evolution From Zero"


def test_series_meta_slug():
    from studio.sources.elftoon import ElftoonAdapter

    fixture = FIXTURES / "elftoon_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = ElftoonAdapter()
        meta = adapter.series_meta(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )

    assert meta.slug == "infinite-evolution-from-zero"


def test_series_meta_source_id():
    from studio.sources.elftoon import ElftoonAdapter

    fixture = FIXTURES / "elftoon_series.html"
    mock_resp = _mock_response(fixture)

    with patch("httpx.get", return_value=mock_resp):
        adapter = ElftoonAdapter()
        meta = adapter.series_meta(
            "https://elftoon.com/manga/infinite-evolution-from-zero/"
        )

    assert meta.source == "elftoon"


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def test_download_extracts_images_from_ts_reader(tmp_path):
    """download() fetches chapter page, parses ts_reader.run() images, streams them."""
    from studio.sources.elftoon import ElftoonAdapter
    from studio.sources.base import ChapterRef
    from PIL import Image
    import io

    chapter = ChapterRef(
        number=1,
        label="Chapter 1",
        url="https://elftoon.com/infinite-evolution-from-zero-chapter-1/",
    )

    chapter_fixture = FIXTURES / "elftoon_chapter.html"
    chapter_resp = _mock_response(chapter_fixture)

    # Fake image response
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
        if call_count[0] == 1:
            return chapter_resp
        return img_resp

    with patch("httpx.get", side_effect=fake_get):
        adapter = ElftoonAdapter()
        result = adapter.download(chapter, tmp_path)

    # Fixture has 3 images
    assert len(result) == 3
    assert all(p.suffix == ".jpg" for p in result)
    assert [p.name for p in result] == ["001.jpg", "002.jpg", "003.jpg"]


def test_download_calls_normalize_into(tmp_path):
    """download() must call normalize_into to canonicalize filenames."""
    from studio.sources.elftoon import ElftoonAdapter
    from studio.sources.base import ChapterRef
    from PIL import Image
    import io

    chapter = ChapterRef(
        number=1,
        label="Chapter 1",
        url="https://elftoon.com/infinite-evolution-from-zero-chapter-1/",
    )

    chapter_fixture = FIXTURES / "elftoon_chapter.html"
    chapter_resp = _mock_response(chapter_fixture)

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
        if call_count[0] == 1:
            return chapter_resp
        return img_resp

    with patch("httpx.get", side_effect=fake_get), \
         patch("studio.sources.elftoon.normalize_into", wraps=__import__("studio.sources.gallerydl", fromlist=["normalize_into"]).normalize_into) as mock_norm:
        adapter = ElftoonAdapter()
        adapter.download(chapter, tmp_path)

    mock_norm.assert_called_once()


# ---------------------------------------------------------------------------
# Capabilities and registration
# ---------------------------------------------------------------------------

def test_capabilities():
    from studio.sources.elftoon import ElftoonAdapter
    from studio.sources.base import Capability

    adapter = ElftoonAdapter()
    assert Capability.DOWNLOAD in adapter.capabilities
    assert Capability.LIST_CHAPTERS in adapter.capabilities
    assert Capability.SERIES_META in adapter.capabilities


def test_adapter_registered():
    import studio.sources  # noqa: F401 — triggers __init__ imports
    from studio.sources.base import get

    adapter = get("elftoon")
    assert adapter is not None
    assert adapter.id == "elftoon"
