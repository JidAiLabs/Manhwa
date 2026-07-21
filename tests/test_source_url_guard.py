"""A placeholder link must never become a chapter row.

Elftoon lists UPCOMING chapters with href="#" (real title and date, no page
yet), plus an unrendered {{number}} template row. "#" is a non-empty string,
so a bare `if not href` let it through and the adapter stored
"https://elftoon.com#" — a URL that returns HTTP 200 and serves the HOMEPAGE,
so nothing downstream could tell it apart from a real chapter until a fetch
died inside image extraction.
"""
from selectolax.parser import HTMLParser

from studio.sources import elftoon
from studio.sources.base import is_fetchable_url


def test_rejects_placeholders_and_templates():
    for bad in ("", "   ", "#", "#/chapter-5", "https://elftoon.com#",
                "javascript:void(0)", "data:text/html,x",
                "https://elftoon.com/{{number}}", "mailto:a@b.c", None):
        assert is_fetchable_url(bad) is False, bad


def test_accepts_real_chapter_urls():
    for ok in ("https://elftoon.com/infinite-evolution-from-zero-chapter-107/",
               "/comics/nano-machine/chapter/1",
               "https://asura.example/comics/x/chapter/12",
               "https://www.webtoons.com/en/action/orv/episode-0-prologue/"
               "viewer?title_no=2154&episode_no=1"):
        assert is_fetchable_url(ok) is True, ok


# the live page shape, captured 2026-07-21: a {{number}} template row, three
# announced chapters with href="#", then real chapters
_LIVE_SHAPE = """<html><body><h1>Infinite Evolution From Zero</h1><ul>
  <li data-num="{{number}}"><span class="eph-num">
    <a href="#/chapter-{{number}}">Chapter {{number}}</a></span></li>
  <li data-num="110"><span class="eph-num"><a href="#">Chapter 110</a></span></li>
  <li data-num="109"><span class="eph-num"><a href="#">Chapter 109</a></span></li>
  <li data-num="108"><span class="eph-num"><a href="#">Chapter 108</a></span></li>
  <li data-num="107"><span class="eph-num"><a
    href="https://elftoon.com/infinite-evolution-from-zero-chapter-107/"
    >Chapter 107</a></span></li>
  <li data-num="106"><span class="eph-num"><a
    href="https://elftoon.com/infinite-evolution-from-zero-chapter-106/"
    >Chapter 106</a></span></li>
</ul></body></html>"""


def test_elftoon_skips_announced_but_unpublished_chapters():
    title, chapters = elftoon._parse_series(HTMLParser(_LIVE_SHAPE))
    assert [c.number for c in chapters] == [106.0, 107.0]
    assert all(is_fetchable_url(c.url) for c in chapters)
    assert not any(c.url.endswith("#") for c in chapters)


def test_elftoon_says_what_it_dropped(capsys):
    """Silently dropping them would read as a scraper bug: '110 chapters
    exist but only 107 are listed'."""
    elftoon._parse_series(HTMLParser(_LIVE_SHAPE))
    out = capsys.readouterr().out
    assert "108" in out and "109" in out and "110" in out
    assert "unpublished" in out
