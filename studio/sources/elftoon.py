"""
Elftoon source adapter (native httpx + selectolax).

Elftoon uses a WordPress-based theme. Chapter lists and chapter pages are
scraped natively — gallery-dl does not support this site.

Series page structure:
  - Title: <h1> text (or og:title meta)
  - Chapter list: ul.clstyle → li[data-num] elements (ordered newest-first)
    Each li has a .eph-num a link whose href is the absolute chapter URL.
    The chapter number is in the data-num attribute.

Chapter page structure:
  - Images are NOT in .reading-content img (the standard Madara pattern).
  - Instead they are in a ts_reader.run({...}) JavaScript call embedded in
    a <script> tag.  The object has:
      {
        "sources": [
          {"source": "Server 1", "images": ["https://...", ...]}
        ],
        ...
      }
  - We use the first source's image list.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
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

_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Matches: ts_reader.run({...});
# We capture the JSON-like object argument (everything between the first { and
# the last } before the closing paren+semicolon).
_TS_READER_RE = re.compile(r"ts_reader\.run\((\{.*?\})\);", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Elftoon can rate-limit/timeout a bulk fetch (5xx/429); THROTTLE + RETRY with
# exponential backoff so a chapter doesn't hard-fail on a transient hiccup.
_THROTTLE_SEC = float(os.environ.get("ELFTOON_THROTTLE_SEC", "0.8"))
_MAX_TRIES = int(os.environ.get("ELFTOON_MAX_TRIES", "5"))
_TRANSIENT = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
_last_req = [0.0]


def _get_retry(url: str, *, timeout: float = 30.0) -> httpx.Response:
    """GET with a global throttle + exponential-backoff retry on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_TRIES):
        wait = _THROTTLE_SEC - (time.monotonic() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.monotonic()
        try:
            resp = httpx.get(url, headers=_HEADERS,
                             follow_redirects=True, timeout=timeout)
            if resp.status_code in _TRANSIENT:
                last_exc = httpx.HTTPStatusError(
                    f"transient {resp.status_code}", request=resp.request,
                    response=resp)
            else:
                resp.raise_for_status()
                return resp
        except httpx.TransportError as e:
            last_exc = e
        if attempt < _MAX_TRIES - 1:
            time.sleep(min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1.5))
    assert last_exc is not None
    raise last_exc


def _get_html(url: str) -> HTMLParser:
    """Fetch *url* with a browser User-Agent and return a parsed HTMLParser."""
    return HTMLParser(_get_retry(url, timeout=30).text)


def _parse_series(tree: HTMLParser) -> tuple[str, list[ChapterRef]]:
    """Return (title, chapters_ascending) from a parsed Elftoon series page."""
    # --- Title ---
    h1 = tree.css_first("h1")
    if h1:
        title = h1.text(strip=True)
    else:
        og = tree.css_first('meta[property="og:title"]')
        title = og.attributes.get("content", "Unknown") if og else "Unknown"

    # --- Chapters ---
    # li[data-num] — ordered newest-first on the page. Deliberately NOT
    # anchored to a ul class: the theme dropped `clstyle` from the chapter
    # list (it now names an unrelated element), which silently yielded zero
    # chapters. data-num + a chapter link is the stable signature.
    chapters: list[ChapterRef] = []
    for li in tree.css("li[data-num]"):
        # data-num is the chapter number
        num_str = li.attributes.get("data-num", "")
        try:
            num = float(num_str)
        except ValueError:
            continue

        # Chapter URL: first <a> in .eph-num (or any <a> in the li)
        a = li.css_first(".eph-num a") or li.css_first("a[href]")
        if a is None:
            continue
        href = a.attributes.get("href", "")
        if not href:
            continue

        # URLs in elftoon_series.html are already absolute
        abs_url = href if href.startswith("http") else f"https://elftoon.com{href}"

        label = f"Chapter {int(num)}" if num == int(num) else f"Chapter {num}"
        chapters.append(ChapterRef(number=num, label=label, url=abs_url))

    # Sort ascending by chapter number
    chapters.sort(key=lambda c: c.number)
    return title, chapters


def _parse_genres(html: str) -> tuple[str, ...]:
    """Genre tags from the span.mgen anchors. Fail-soft: markup churn must NEVER
    break discovery, so any parse error yields ()."""
    try:
        tree = HTMLParser(html)
        out = [a.text(strip=True) for a in tree.css("span.mgen a")]
        return tuple(g for g in out if g)
    except Exception:
        return ()


def _parse_synopsis(html: str) -> str:
    """Synopsis from the [itemprop="description"] block. Fail-soft → ''."""
    try:
        node = HTMLParser(html).css_first('[itemprop="description"]')
        return node.text(strip=True) if node else ""
    except Exception:
        return ""


def _extract_image_urls(page_html: str) -> list[str]:
    """
    Extract ordered image URLs from an Elftoon chapter page.

    Images are embedded via ts_reader.run({sources: [{images: [...]}]}).
    We parse the JSON object from the script tag and return the first
    source's image list.
    """
    m = _TS_READER_RE.search(page_html)
    if not m:
        return []

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []

    sources = data.get("sources", [])
    if not sources:
        return []

    images = sources[0].get("images", [])
    return [img for img in images if isinstance(img, str)]


def _download_images(image_urls: list[str], dest_dir: Path) -> None:
    """Stream *image_urls* into *dest_dir* as 001.ext, 002.ext, … (before normalize)."""
    import io
    from PIL import Image

    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(image_urls, start=1):
        ext = Path(url.split("?")[0]).suffix or ".jpg"
        out_path = dest_dir / f"{i:03d}{ext}"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue                        # RESUME: already fetched, skip
        resp = _get_retry(url, timeout=60)
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        tmp = out_path.with_name(out_path.name + ".tmp")
        img.save(tmp, format="JPEG")
        os.replace(tmp, out_path)           # atomic: a SIGKILL/disk-full mid-encode
                                            # can't leave a partial file the size-only
                                            # resume above would then trust and skip


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register
class ElftoonAdapter(SourceAdapter):
    """Native httpx+selectolax adapter for elftoon.com."""

    id = "elftoon"
    capabilities = Capability.DOWNLOAD | Capability.LIST_CHAPTERS | Capability.SERIES_META

    def _fetch_series_page(self, series_url: str) -> HTMLParser:
        return _get_html(series_url)

    def search(self, title: str) -> list[tuple[str, str]]:
        """WordPress search: /?s=<q>, result anchors carry class 'series'."""
        from urllib.parse import quote
        try:
            tree = _get_html(f"https://elftoon.com/?s={quote(title)}")
            out: list[tuple[str, str]] = []
            for a in tree.css("a.series"):
                href = a.attributes.get("href") or ""
                txt = (a.text() or "").strip()
                if href and txt:
                    out.append((txt, href))
            return out[:10]
        except Exception:
            return []

    def list_chapters(self, series_url: str) -> list[ChapterRef]:
        tree = self._fetch_series_page(series_url)
        _, chapters = _parse_series(tree)
        return chapters

    def series_meta(self, series_url: str) -> SeriesMeta:
        tree = self._fetch_series_page(series_url)
        title, _ = _parse_series(tree)
        html = tree.html or ""          # reuse the fetched page — no second GET
        return SeriesMeta(
            source=self.id,
            series_url=series_url,
            title=title,
            slug=slugify(title),
            genres=_parse_genres(html),
            synopsis=_parse_synopsis(html),
        )

    def download(self, chapter: ChapterRef, dest_dir: Path) -> list[Path]:
        resp = _get_retry(chapter.url, timeout=30)
        image_urls = _extract_image_urls(resp.text)
        if not image_urls:
            raise RuntimeError(
                f"No images found on Elftoon chapter page: {chapter.url}"
            )
        # PERSISTENT staging so a failed attempt RESUMES (skip-existing) instead
        # of re-fetching every image; removed only after a full success.
        staging = dest_dir / ".staging"
        _download_images(image_urls, staging)
        out = normalize_into(staging, dest_dir)
        shutil.rmtree(staging, ignore_errors=True)
        return out
