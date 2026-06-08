"""
Asura Scans source adapter.

Asura uses a custom Astro-based frontend (not gallery-dl compatible).
We scrape with httpx + selectolax.

Series page structure:
  - Title: <h1> text
  - Chapter list: <a href="/comics/..."> links inside a div.divide-y container
    Each link has a <span> with text "Chapter N" (the <!-- --> comment is a
    React hydration artifact stripped during HTML parse)

Chapter page structure:
  - Image URLs are embedded in a <script> block as window.__ASTRO_DATA__
    (HTML-entity-encoded JSON). The JSON has the shape:
    {"pages": [1, [[0, {"url": [0, "https://..."], ...}], ...]]}
    i.e. pages[1] is the list; each element is [0, {url: [0, url_str], ...}]
"""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser

from studio.sources.base import (
    Capability,
    ChapterRef,
    SeriesMeta,
    SourceAdapter,
    register,
    slugify,
)
from studio.sources.gallerydl import normalize_into

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_BASE_URL = "https://asurascans.com"

_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Asura serves chapter pages from a stable CDN path:
#   https://cdn.asurascans.com/asura-images/chapters[-restored]/<slug>/<ch>/NNN.webp
# (covers live under /covers/, so matching /chapters excludes them).
_CHAPTER_IMG_RE = re.compile(
    r"https://cdn\.asurascans\.com/asura-images/chapters[^\s\"'\\<>]+?/(\d+)\.webp"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_html(url: str) -> HTMLParser:
    resp = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    return HTMLParser(resp.text)


def _parse_chapter_number(text: str) -> float | None:
    """Extract the numeric part from 'Chapter 315' or 'Chapter315' → 315.0.

    The React comment (<!-- -->) between 'Chapter' and the number is stripped
    by selectolax, leaving 'Chapter315' without a space.
    """
    m = re.search(r"Chapter\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _parse_series(tree: HTMLParser, series_url: str) -> tuple[str, list[ChapterRef]]:
    """Return (title, chapters) from a parsed series page."""
    # Title from <h1>
    h1 = tree.css_first("h1")
    title = h1.text(strip=True) if h1 else "Unknown"

    # Chapters: all <a> tags that have href matching /comics/.../chapter/N
    chapters: list[ChapterRef] = []
    for a in tree.css("a[href]"):
        href = a.attributes.get("href", "")
        if "/chapter/" not in href:
            continue
        # The first span with the chapter number text.
        # We do NOT fall back to a.text() — that concatenates all child spans,
        # producing strings like "Chapter315102:..." that confuse the parser.
        span_text = ""
        for span in a.css("span"):
            t = span.text(strip=True)
            if re.search(r"Chapter\s*[\d.]+", t, re.IGNORECASE):
                span_text = t
                break
        if not span_text:
            continue
        num = _parse_chapter_number(span_text)
        if num is None:
            continue
        # Build absolute URL
        if href.startswith("http"):
            abs_url = href
        else:
            abs_url = _BASE_URL + href
        label = f"Chapter {int(num)}" if num == int(num) else f"Chapter {num}"
        chapters.append(ChapterRef(number=num, label=label, url=abs_url))

    chapters.sort(key=lambda c: c.number)
    return title, chapters


def _extract_image_urls(page_html: str) -> list[str]:
    """
    Extract ordered image URLs from the Asura chapter page.

    The page embeds window.__ASTRO_DATA__ as an HTML-entity-encoded JSON
    object inside a <script> tag.  The pages array has the shape:
      [tag, [[tag, {url: [tag, url_str], ...}], ...]]
    """
    # Decode entities + un-escape slashes (the URLs may appear inside an
    # entity-encoded JSON blob), then pull every per-page chapter image and
    # order by its numeric filename. Dedupe (same image can appear twice).
    text = html.unescape(page_html).replace("\\/", "/")
    by_num: dict[int, str] = {}
    for m in _CHAPTER_IMG_RE.finditer(text):
        n = int(m.group(1))
        by_num.setdefault(n, m.group(0))
    return [by_num[k] for k in sorted(by_num)]


def _download_images(image_urls: list[str], dest_dir: Path) -> list[Path]:
    """Stream image URLs to dest_dir as 001.jpg, 002.jpg, …"""
    from PIL import Image
    import io

    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, url in enumerate(image_urls, start=1):
        resp = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        out_path = dest_dir / f"{i:03d}.jpg"
        img.save(out_path, format="JPEG")
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register
class AsuraAdapter(SourceAdapter):
    """Native httpx+selectolax adapter for asurascans.com."""

    id = "asura"
    capabilities = Capability.DOWNLOAD | Capability.LIST_CHAPTERS | Capability.SERIES_META

    def _fetch_series_page(self, series_url: str) -> HTMLParser:
        return _get_html(series_url)

    def list_chapters(self, series_url: str) -> list[ChapterRef]:
        tree = self._fetch_series_page(series_url)
        _, chapters = _parse_series(tree, series_url)
        return chapters

    def series_meta(self, series_url: str) -> SeriesMeta:
        tree = self._fetch_series_page(series_url)
        title, _ = _parse_series(tree, series_url)
        return SeriesMeta(
            source=self.id,
            series_url=series_url,
            title=title,
            slug=slugify(title),
        )

    def download(self, chapter: ChapterRef, dest_dir: Path) -> list[Path]:
        resp = httpx.get(chapter.url, headers=_HEADERS, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        image_urls = _extract_image_urls(resp.text)
        if not image_urls:
            raise RuntimeError(
                f"No images found on Asura chapter page: {chapter.url}"
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            written = _download_images(image_urls, tmp_path)
            return normalize_into(tmp_path, dest_dir)
