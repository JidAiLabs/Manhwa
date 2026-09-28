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

    # Labels carry the PUBLISHED episode number, taken from the URL slug —
    # not gallery-dl's episode_no. This assertion used to read
    # `"1" in chapters[0].label`, which pinned the off-by-one: entry 0 is the
    # prologue (slug 'episode-0-prologue', episode_no=1) and was labelled
    # "Episode 1", shifting every episode after it by one.
    assert chapters[0].label == "Episode 0 (Prologue)"
    assert chapters[1].label == "Episode 1"
    # number stays the sequence index — it is the UNIQUE key
    assert chapters[0].number == 1.0 and chapters[1].number == 2.0


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


# --- label vs number: the episode_no off-by-one -------------------------------

def test_label_comes_from_the_slug_not_the_sequence_index():
    """gallery-dl's episode_no is an internal 1-based sequence index, not the
    published episode number, so labelling from it made the PROLOGUE read as
    "Episode 1" and shifted everything after it."""
    from studio.sources.webtoon import _label_from_slug
    assert _label_from_slug("episode-0-prologue", 1) == "Episode 0 (Prologue)"
    assert _label_from_slug("episode-1", 2) == "Episode 1"
    assert _label_from_slug("ep-12-the-fall", 13) == "Episode 12 (The Fall)"


def test_multi_season_labels_carry_the_season():
    """Slug numbers COLLIDE across seasons (Tower of God: 236 collisions in
    652 episodes, since season-1-ep-0 and season-2-ep-0 both parse to 0).
    Season lives in the label, where a collision is impossible; number stays
    episode_no, which is the UNIQUE key."""
    from studio.sources.webtoon import _label_from_slug
    assert _label_from_slug("season-1-ep-0", 1) == "Season 1 Episode 0"
    assert _label_from_slug("season-2-ep-0", 80) == "Season 2 Episode 0"
    assert _label_from_slug("season-2-ep-141", 222) == "Season 2 Episode 141"


def test_unparseable_slug_falls_back_to_the_old_label():
    """Specials and titled side stories carry no number — strictly no worse
    than the previous behaviour, never a wrong guess."""
    from studio.sources.webtoon import _label_from_slug
    assert _label_from_slug("new-year-special", 55) == "Episode 55"
    assert _label_from_slug("", 7) == "Episode 7"


def test_episode_slug_extraction():
    from studio.sources.webtoon import _episode_slug
    assert _episode_slug(
        "https://www.webtoons.com/en/action/orv/episode-0-prologue/"
        "viewer?title_no=2154&episode_no=1") == "episode-0-prologue"
    assert _episode_slug("https://x.example/a/season-2-ep-5/viewer") == \
        "season-2-ep-5"


def test_number_stays_the_unique_sequence_key():
    """number must remain episode_no: it is the UNIQUE(series_id, number)
    key AND the refresh-cron upsert key. Deriving it from the slug instead
    is what would corrupt every downloaded chapter's url."""
    from studio.sources.webtoon import _parse_chapters
    entries = [
        [6, "https://w.example/c/season-1-ep-0/viewer?x=1", {"episode_no": 1}],
        [6, "https://w.example/c/season-2-ep-0/viewer?x=1", {"episode_no": 80}],
    ]
    chs = _parse_chapters(entries)
    assert [c.number for c in chs] == [1.0, 80.0]          # no collision
    assert [c.label for c in chs] == ["Season 1 Episode 0",
                                      "Season 2 Episode 0"]


def test_fixture_prologue_is_episode_zero():
    """End-to-end over the recorded fixture: entry 0 is the prologue with
    episode_no=1 and must be labelled Episode 0."""
    import json
    from studio.sources.webtoon import _parse_chapters
    entries = json.loads(
        (FIXTURES / "webtoon_chapters.json").read_text())
    chs = _parse_chapters(entries)
    assert chs[0].number == 1.0
    assert chs[0].label == "Episode 0 (Prologue)"
    assert chs[1].label == "Episode 1"



# ---------------------------------------------------------------------------
# resolve_series_url: any webtoons.com link -> the canonical list page
# (2026-09-27: the short link failed three times as "add-series exited 1")
# ---------------------------------------------------------------------------

CANON = "https://www.webtoons.com/en/action/rise-of-the-devourer/list?title_no=7818"


@pytest.mark.parametrize("link", [
    "https://www.webtoons.com/en/action/rise-of-the-devourer/list?title_no=7818&page=1",
    "https://www.webtoons.com/en/action/rise-of-the-devourer/list?title_no=7818&page=3",
    "http://webtoons.com/en/action/rise-of-the-devourer/list?title_no=7818",
    "https://m.webtoons.com/en/action/rise-of-the-devourer/list?title_no=7818",
    "https://www.webtoons.com/en/action/rise-of-the-devourer/ep-5/viewer"
    "?title_no=7818&episode_no=5",
    "  " + CANON + "  ",
])
def test_a_link_with_title_no_resolves_offline(link):
    """page= is dropped: gallery-dl starts at that page and skips the newest."""
    from studio.sources.webtoon import WebtoonAdapter
    a = WebtoonAdapter()
    with patch.object(WebtoonAdapter, "_search_raw",
                      side_effect=AssertionError("no network needed")):
        assert a.resolve_series_url(link) == CANON


def test_a_short_link_is_found_by_its_exact_path():
    from studio.sources.webtoon import WebtoonAdapter
    hits = [("Rise of the Devourer (canvas)",
             "https://www.webtoons.com/en/canvas/rise-of-the-devourer-x/list?title_no=99"),
            ("Rise of the Devourer", CANON)]
    with patch.object(WebtoonAdapter, "_search_raw", return_value=hits) as s:
        got = WebtoonAdapter().resolve_series_url(
            "https://www.webtoons.com/en/action/rise-of-the-devourer/")
    assert got == CANON
    s.assert_called_once_with("rise of the devourer")


def test_a_short_link_with_the_wrong_genre_still_finds_the_one_series():
    from studio.sources.webtoon import WebtoonAdapter
    with patch.object(WebtoonAdapter, "_search_raw",
                      return_value=[("Rise of the Devourer", CANON)]):
        assert WebtoonAdapter().resolve_series_url(
            "https://www.webtoons.com/en/fantasy/rise-of-the-devourer") == CANON


@pytest.mark.parametrize("link, why", [
    ("https://www.webtoons.com/en/action/no-such-series/", "has no series at"),
    ("https://www.webtoons.com/en/", "not a webtoons.com series link"),
    ("https://example.com/en/action/x/list?title_no=1", "not a webtoons.com"),
    ("https://www.webtoons.com/fr/action/x/", "English only"),
])
def test_a_link_that_names_no_series_is_refused_with_a_reason(link, why):
    from studio.sources.base import UnsupportedSource
    from studio.sources.webtoon import WebtoonAdapter
    with patch.object(WebtoonAdapter, "_search_raw", return_value=[
            ("Other", "https://www.webtoons.com/en/action/other/list?title_no=5")]):
        with pytest.raises(UnsupportedSource, match=why) as e:
            WebtoonAdapter().resolve_series_url(link)
    assert "title_no=" in str(e.value)          # the owner is told what to paste


def test_a_search_failure_is_transient_not_a_refusal():
    """A network blip must not become "no such series" (NonRetryable)."""
    import httpx
    from studio.sources.base import UnsupportedSource
    from studio.sources.webtoon import WebtoonAdapter
    with patch("httpx.get", side_effect=httpx.ConnectError("down")):
        with pytest.raises(httpx.ConnectError):
            WebtoonAdapter().resolve_series_url(
                "https://www.webtoons.com/en/action/rise-of-the-devourer/")
        assert WebtoonAdapter().search("rise of the devourer") == []  # discovery


def test_gallery_dl_unsupported_url_is_a_permanent_refusal():
    from studio.sources.base import UnsupportedSource
    from studio.sources.webtoon import _run_gallery_dl_j
    bad = MagicMock(returncode=64, stdout="",
                    stderr="[gallery-dl][error] Unsupported URL 'https://x'")
    with patch("subprocess.run", return_value=bad):
        with pytest.raises(UnsupportedSource):
            _run_gallery_dl_j("https://x")
    flaky = MagicMock(returncode=1, stdout="", stderr="HTTP 503")
    with patch("subprocess.run", return_value=flaky):
        with pytest.raises(RuntimeError):
            _run_gallery_dl_j("https://x")


def test_the_default_resolver_returns_the_link_as_given():
    from studio.sources.base import get
    import studio.sources  # noqa: F401
    assert get("asura").resolve_series_url(
        "  https://asurascans.com/comics/x-1  ") == "https://asurascans.com/comics/x-1"
