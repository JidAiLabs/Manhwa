"""The SOURCE comes from the pasted URL, not from a dropdown default.

Two bugs, same shape, opposite directions: the picker was a hardcoded list
that omitted arenascan, so arenascan URLs fell through to the first option
(asura) and found 0 chapters; making the list registry-driven then sorted it
alphabetically, so arenascan sat FIRST and captured every add where the
operator didn't touch the picker — a webtoons.com and an asurascans.com URL
both filed as arenascan, 0 chapters again (2026-09-06). Whichever adapter
happens to sit at the top of a list must never decide.
"""
from __future__ import annotations

import pytest

from studio.sources import base


@pytest.mark.parametrize("url,expected", [
    ("https://www.webtoons.com/en/action/a-wimps-tower-strategy-guide/"
     "list?title_no=9701&page=3", "webtoon"),
    ("https://webtoons.com/en/action/x/list?title_no=1", "webtoon"),
    ("https://asurascans.com/comics/a-wimps-strategy-guide-to-conquer-the-"
     "tower-08677664/", "asura"),
    ("https://arenascan.com/manga/i-rely-on-my-invincibility/", "arenascan"),
    ("https://elftoon.com/manga/infinite-evolution-from-zero/", "elftoon"),
])
def test_url_picks_its_own_adapter(url, expected):
    a = base.for_url(url)
    assert a is not None and a.id == expected


@pytest.mark.parametrize("url", [
    "https://example.com/manga/whatever",
    "https://notasurascans.com/comics/x",   # suffix match must not be loose
    "not-a-url",
    "",
])
def test_unknown_hosts_resolve_to_nothing_rather_than_a_default(url):
    # returning None is what lets the caller SAY it cannot handle the link,
    # instead of silently handing it to whichever adapter sorts first
    assert base.for_url(url) is None


def _shipped():
    """Adapters defined under studio/sources — REGISTRY is process-global and
    other tests register their own doubles into it."""
    import studio.sources          # noqa: F401  (adapters self-register)
    return {i: a for i, a in base.REGISTRY.items()
            if type(a).__module__.startswith("studio.sources.")}


def test_every_shipped_adapter_declares_its_domains():
    missing = [i for i, a in _shipped().items() if not a.domains]
    assert not missing, f"adapters with no domains (unroutable by URL): {missing}"
    assert set(_shipped()) >= {"webtoon", "asura", "elftoon", "arenascan"}


def test_each_declared_domain_routes_back_to_its_own_adapter():
    """No two adapters may claim the same host, and every declared domain
    must resolve to the adapter that declared it."""
    for aid, adapter in _shipped().items():
        for d in adapter.domains:
            got = base.for_url(f"https://{d}/whatever")
            assert got is not None and got.id == aid, (d, aid, got and got.id)
