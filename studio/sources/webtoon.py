"""
Webtoon source adapter.

Uses gallery-dl's built-in Webtoons extractor:
  - ``gallery-dl -j <list_url>`` yields one entry per episode (type 6)
  - Each entry: [6, viewer_url, {episode_no, comic, date, ...}]

The comic slug (e.g. 'omniscient-reader') is converted to a title by
replacing hyphens with spaces and title-casing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from studio.sources.base import (
    Capability,
    ChapterRef,
    SeriesMeta,
    SourceAdapter,
    register,
    slugify,
)
from studio.sources.gallerydl import normalize_into, run_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_gallery_dl_j(url: str) -> list:
    """Run ``gallery-dl -j <url>`` and return the parsed JSON list."""
    result = subprocess.run(
        [sys.executable, "-m", "gallery_dl", "-j", url],
        capture_output=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gallery-dl -j failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def _comic_slug_to_title(slug: str) -> str:
    """Convert 'omniscient-reader' → 'Omniscient Reader'."""
    return slug.replace("-", " ").title()


def _parse_chapters(entries: list) -> list[ChapterRef]:
    """
    Parse gallery-dl -j entries into ordered ChapterRefs.

    Each entry from the list URL is: [6, viewer_url, {episode_no, comic, ...}]
    They are ordered newest-first; we sort ascending by episode_no.
    """
    chapters: list[ChapterRef] = []
    for entry in entries:
        if not isinstance(entry, list) or len(entry) < 3:
            continue
        _type, url, meta = entry[0], entry[1], entry[2]
        if _type != 6:
            continue
        episode_no = meta.get("episode_no")
        if episode_no is None:
            continue
        chapters.append(
            ChapterRef(
                number=float(episode_no),
                label=f"Episode {episode_no}",
                url=url,
            )
        )
    # Sort ascending by episode number
    chapters.sort(key=lambda c: c.number)
    return chapters


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register
class WebtoonAdapter(SourceAdapter):
    """gallery-dl backed adapter for webtoons.com."""

    id = "webtoon"
    capabilities = Capability.DOWNLOAD | Capability.LIST_CHAPTERS | Capability.SERIES_META

    def search(self, title: str) -> list[tuple[str, str]]:
        """webtoons.com/en/search — result cards are <a class='link _card_item'>
        with the title as the first text line."""
        from urllib.parse import quote

        import httpx
        from selectolax.parser import HTMLParser
        try:
            r = httpx.get(
                "https://www.webtoons.com/en/search?keyword=" + quote(title),
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS "
                                       "X 10_15_7) AppleWebKit/537.36"},
                follow_redirects=True, timeout=15)
            out: list[tuple[str, str]] = []
            for a in HTMLParser(r.text).css("a._card_item"):
                href = a.attributes.get("href") or ""
                first_line = next((ln.strip() for ln in
                                   (a.text() or "").splitlines()
                                   if ln.strip()), "")
                if href and first_line:
                    out.append((first_line, href))
            return out[:10]
        except Exception:
            return []

    def _fetch_entries(self, series_url: str) -> list:
        return _run_gallery_dl_j(series_url)

    def list_chapters(self, series_url: str) -> list[ChapterRef]:
        entries = self._fetch_entries(series_url)
        return _parse_chapters(entries)

    def series_meta(self, series_url: str) -> SeriesMeta:
        entries = self._fetch_entries(series_url)
        chapters = _parse_chapters(entries)

        # Derive title from the 'comic' field in the first entry's metadata
        comic_slug: str = "unknown"
        for entry in entries:
            if isinstance(entry, list) and len(entry) >= 3 and entry[0] == 6:
                comic_slug = entry[2].get("comic", "unknown")
                break

        title = _comic_slug_to_title(comic_slug)
        slug = slugify(title)

        return SeriesMeta(
            source=self.id,
            series_url=series_url,
            title=title,
            slug=slug,
        )

    def download(self, chapter: ChapterRef, dest_dir: Path) -> list[Path]:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_download(chapter.url, tmp_path)
            return normalize_into(tmp_path, dest_dir)
