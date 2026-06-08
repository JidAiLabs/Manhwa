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


@dataclass(frozen=True)
class SeriesMeta:
    """Metadata for a series as returned by a source."""
    source: str
    series_url: str
    title: str
    slug: str


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
