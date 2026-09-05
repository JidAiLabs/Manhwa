"""
Tests for the ArenaScan source adapter.

No network access — httpx is monkeypatched to return fixture HTML.

The fixtures encode the three traps found while probing the live site, so a
future rewrite that reintroduces any of them fails here instead of silently
shipping a half-discovered series:

  1. TWO chapter-URL schemes per series (…-chapter-167/ and …-83/). Matching
     hrefs on "-chapter-" finds only the newest ~24 of 168.
  2. An unrendered "#/chapter-{{number}}" template row in the list.
  3. TWO image-filename schemes (content hashes vs plain numbers), which need
     different orderings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_response(fixture_path: Path, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = fixture_path.read_text()
    mock.raise_for_status = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# list_chapters — the era trap
# ---------------------------------------------------------------------------

def _chapters():
    from studio.sources.arenascan import ArenascanAdapter
    resp = _mock_response(FIXTURES / "arenascan_series.html")
    with patch("studio.sources.arenascan.httpx.get", return_value=resp):
        return ArenascanAdapter().list_chapters(
            "https://arenascan.com/manga/a-test-series/")


def test_both_url_eras_are_discovered():
    """The whole point: a "-chapter-" href match would drop the back catalogue."""
    chapters = _chapters()
    urls = {c.number: c.url for c in chapters}
    assert urls[167.0].endswith("/a-test-series-chapter-167/")   # newer scheme
    assert urls[83.0].endswith("/a-test-series-83/")             # older scheme
    assert urls[1.0].endswith("/a-test-series-1/")
    assert len(chapters) == 5


def test_unrendered_template_row_is_skipped():
    for c in _chapters():
        assert "{{" not in c.url and "}}" not in c.url
        assert not c.url.rstrip("/").endswith("#")


def test_numbers_come_from_data_num_and_sort_oldest_first():
    nums = [c.number for c in _chapters()]
    assert nums == sorted(nums)
    assert nums == [0.0, 1.0, 83.0, 166.0, 167.0]   # chapter 0 is a real prologue


def test_labels_use_the_sites_own_chapternum_text():
    by_num = {c.number: c.label for c in _chapters()}
    assert by_num[167.0] == "Chapter 167"
    assert by_num[0.0] == "Chapter 0"


# ---------------------------------------------------------------------------
# series_meta
# ---------------------------------------------------------------------------

def test_series_meta_reads_title_genres_and_synopsis():
    from studio.sources.arenascan import ArenascanAdapter
    resp = _mock_response(FIXTURES / "arenascan_series.html")
    with patch("studio.sources.arenascan.httpx.get", return_value=resp):
        meta = ArenascanAdapter().series_meta(
            "https://arenascan.com/manga/a-test-series/")
    assert meta.source == "arenascan"
    assert meta.title == "A Test Series"
    assert meta.slug == "a-test-series"
    assert meta.genres == ("Drama", "Fantasy")
    assert meta.synopsis.startswith("A short synopsis")


# ---------------------------------------------------------------------------
# image extraction — the filename-scheme trap
# ---------------------------------------------------------------------------

def test_hash_filenames_keep_reader_order():
    """Content hashes carry no sequence; document order IS the reading order."""
    from studio.sources.arenascan import _extract_image_urls
    urls = _extract_image_urls((FIXTURES / "arenascan_chapter_hash.html").read_text())
    assert [u.rsplit("/", 1)[-1] for u in urls] == [
        "506f3368cc4c.webp", "248a9e1653ee.webp", "aa11bb22cc33.webp"]


def test_numeric_filenames_sort_by_value_not_lexically():
    """10 must follow 2 — a lexical sort would put it after 1."""
    from studio.sources.arenascan import _extract_image_urls
    urls = _extract_image_urls(
        (FIXTURES / "arenascan_chapter_numeric.html").read_text())
    assert [u.rsplit("/", 1)[-1] for u in urls] == [
        "0.jpg", "1.jpg", "2.jpg", "10.jpg"]


def test_ts_reader_blob_is_preferred_over_the_pattern_scan():
    from studio.sources.arenascan import _extract_image_urls
    html = (FIXTURES / "arenascan_chapter_hash.html").read_text()
    assert len(_extract_image_urls(html)) == 3      # not 6 (img tags + blob)


def test_download_raises_when_a_chapter_has_no_images(tmp_path):
    """A silent empty download would create an empty chapter dir that every
    later stage misreads as a real one."""
    from studio.sources.arenascan import ArenascanAdapter
    from studio.sources.base import ChapterRef
    resp = MagicMock(status_code=200, text="<html><body>nothing</body></html>")
    resp.raise_for_status = MagicMock()
    ref = ChapterRef(number=1.0, label="Chapter 1", url="https://arenascan.com/x-1/")
    with patch("studio.sources.arenascan.httpx.get", return_value=resp):
        with pytest.raises(RuntimeError, match="no chapter images"):
            ArenascanAdapter().download(ref, tmp_path)


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------

def test_adapter_is_registered_and_configured():
    import studio.sources  # noqa: F401  (triggers self-registration)
    from studio.config import load
    from studio.sources.base import Capability, get

    adapter = get("arenascan")
    assert adapter.id == "arenascan"
    for cap in (Capability.DOWNLOAD, Capability.LIST_CHAPTERS,
                Capability.SERIES_META):
        assert cap in adapter.capabilities
    assert load().sites["arenascan"].base_url == "https://arenascan.com"
