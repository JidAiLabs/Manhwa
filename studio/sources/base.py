"""
Source adapter contract and registry.

Every source backend (gallery-dl, custom scrapers, …) registers an instance of
SourceAdapter here so the rest of the pipeline can interact with any source
through a uniform interface.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Flag, auto
from pathlib import Path
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------

class Capability(Flag):
    DOWNLOAD = auto()
    LIST_CHAPTERS = auto()
    SERIES_META = auto()


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChapterRef:
    """Lightweight reference to a single chapter — enough to fetch it."""
    number: float
    label: str
    url: str


def is_fetchable_url(url: str) -> bool:
    """Is this a URL we could actually download a chapter from?

    Sites list UPCOMING chapters with a placeholder link — elftoon renders
    the newest three as `href="#"` with a real title and date, plus an
    unrendered `{{number}}` template row. Naive `if not href` does not catch
    "#", because "#" is a non-empty string, so the adapter happily stored
    "https://elftoon.com" + "#".

    That is worse than a broken link: it returns HTTP 200 and serves the
    HOMEPAGE, so nothing downstream can tell it apart from a real chapter
    page until image extraction fails deep inside a fetch — after the row
    has already inflated the chapter count and joined every bulk-run range.

    An announced-but-unpublished chapter is not a chapter. Skip it; the daily
    refresh adds it for real once the site publishes a link.
    """
    u = (url or "").strip()
    if not u or u.startswith(("#", "javascript:", "data:", "mailto:")):
        return False
    if "{{" in u or "}}" in u:          # unrendered template row
        return False
    # a bare origin with nothing but a fragment ("https://site.com#")
    without_scheme = u.split("://", 1)[-1]
    path = without_scheme.split("/", 1)[1] if "/" in without_scheme else ""
    if not path.strip("#?") and "#" in u:
        return False
    return True


@dataclass(frozen=True)
class SeriesMeta:
    """Metadata for a series as returned by a source."""
    source: str
    series_url: str
    title: str
    slug: str
    genres: tuple[str, ...] = ()
    synopsis: str = ""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnsupportedSource(Exception):
    """Raised when no adapter can handle a given URL or source identifier."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SourceAdapter(ABC):
    """
    Base class for all source backends.

    Subclasses must declare class-level ``id`` and ``capabilities`` and
    implement all three abstract methods.
    """

    id: str
    capabilities: Capability
    #: Host suffixes this adapter owns ("asurascans.com"). `for_url` matches a
    #: pasted URL against these so the SOURCE is derived from the link instead
    #: of a dropdown default — see for_url() for why that matters.
    domains: tuple = ()

    @abstractmethod
    def series_meta(self, series_url: str) -> SeriesMeta:
        """Return metadata for the series at *series_url*."""

    @abstractmethod
    def list_chapters(self, series_url: str) -> list[ChapterRef]:
        """Return an ordered list of chapter references for *series_url*."""

    @abstractmethod
    def download(self, chapter: ChapterRef, dest_dir: Path) -> list[Path]:
        """
        Download *chapter* into *dest_dir*.

        Returns the list of image paths written, in page order.
        """

    def search(self, title: str) -> list[tuple[str, str]]:
        """Best-effort site search: [(series_title, series_url), ...].

        Used by discovery to auto-link AniList trends to source URLs.
        Default: no search capability. Implementations must swallow their
        own errors and return [] — site churn must never break discovery."""
        return []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, SourceAdapter] = {}


def register(cls: type[SourceAdapter]) -> type[SourceAdapter]:
    """
    Class decorator — instantiates *cls* and stores it in REGISTRY keyed by
    ``cls.id``.  Returns the class unchanged so it can still be subclassed or
    tested directly.

    Usage::

        @register
        class MyAdapter(SourceAdapter):
            id = "my-source"
            ...
    """
    REGISTRY[cls.id] = cls()
    return cls


def for_url(url: str) -> "SourceAdapter | None":
    """The adapter whose `domains` own *url*, else None.

    The source used to come from a dropdown whose first option was selected
    by default. When arenascan was added the list became alphabetical, so
    arenascan sat first and silently captured every add where the operator
    did not touch the picker: a webtoons.com and an asurascans.com URL were
    both filed as arenascan, whose parser then found 0 chapters on them
    (2026-09-06). A URL already names its site — read it from there, and let
    the picker only override.
    """
    import studio.sources          # noqa: F401  (adapters self-register)
    host = urlparse(str(url or "")).hostname or ""
    host = host.lower().removeprefix("www.")
    if not host:
        return None
    for adapter in REGISTRY.values():
        for d in adapter.domains:
            d = d.lower().removeprefix("www.")
            if host == d or host.endswith("." + d):
                return adapter
    return None


def get(adapter_id: str) -> SourceAdapter:
    """
    Return the registered adapter instance for *adapter_id*.

    Raises :class:`UnsupportedSource` if no adapter is registered.
    """
    try:
        return REGISTRY[adapter_id]
    except KeyError:
        raise UnsupportedSource(f"No adapter registered for '{adapter_id}'")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def slugify(title: str) -> str:
    """
    Convert *title* to a URL-safe slug.

    * Lowercased
    * Any run of non-alphanumeric characters replaced with a single hyphen
    * Leading/trailing hyphens stripped

    Examples::

        >>> slugify("The Beginning: After/End!")
        'the-beginning-after-end'
    """
    lowered = title.lower()
    slugged = re.sub(r"[^a-z0-9]+", "-", lowered)
    return slugged.strip("-")
