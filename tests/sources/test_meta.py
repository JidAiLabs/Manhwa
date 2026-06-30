# tests/sources/test_meta.py
from pathlib import Path
from studio.sources.base import SeriesMeta
from studio.sources import asura, elftoon

FIXTURES = Path(__file__).parent / "fixtures"


def test_seriesmeta_defaults_are_safe():
    m = SeriesMeta(source="asura", series_url="u", title="t", slug="s")
    assert m.genres == ()       # default empty tuple, never None
    assert m.synopsis == ""


def test_elftoon_parse_genres_from_fixture():
    html = (FIXTURES / "elftoon_series.html").read_text(encoding="utf-8")
    genres = elftoon._parse_genres(html)
    assert isinstance(genres, tuple) and len(genres) >= 1
    assert any("action" in g.lower() for g in genres)  # fixture is an action title


def test_asura_parse_genres_from_fixture():
    # requires the extended asura_series.html fixture (see Files note)
    html = (FIXTURES / "asura_series.html").read_text(encoding="utf-8")
    genres = asura._parse_genres(html)
    assert isinstance(genres, tuple) and len(genres) >= 1


def test_parse_genres_failsoft_on_garbage():
    assert asura._parse_genres("<html>no genres here</html>") == ()
    assert elftoon._parse_genres("") == ()
