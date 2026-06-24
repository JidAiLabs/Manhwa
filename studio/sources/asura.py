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
  - Per-page images are served from a stable CDN path embedded in the page:
    cdn.asurascans.com/asura-images/chapters[-restored]/<slug>/<ch>/NNN.webp
    We extract them by pattern and order by the numeric filename. (The page
    has no window.__ASTRO_DATA__ block despite the Astro frontend.)
"""

from __future__ import annotations

import html
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

_BASE_URL = "https://asurascans.com"

_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Asura sits behind Cloudflare; a bulk download burst trips rate-limiting
# (HTTP 522/429/5xx). THROTTLE requests and RETRY transient failures with
# exponential backoff so a chapter doesn't hard-fail on a passing hiccup.
# Tunable per host via env (lower throttle when two machines share asura).
_THROTTLE_SEC = float(os.environ.get("ASURA_THROTTLE_SEC", "0.8"))
_MAX_TRIES = int(os.environ.get("ASURA_MAX_TRIES", "5"))
_TRANSIENT = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
_last_req = [0.0]


def _get_retry(url: str, *, timeout: float = 30.0) -> httpx.Response:
    """GET with a global throttle + exponential-backoff retry on transient
    errors (Cloudflare 5xx/429, timeouts, connection resets). Raises the last
    error only after exhausting _MAX_TRIES."""
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
        except httpx.TransportError as e:        # timeouts, conn resets, DNS
            last_exc = e
        if attempt < _MAX_TRIES - 1:
            time.sleep(min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1.5))
    assert last_exc is not None
    raise last_exc

# Asura serves chapter pages from a stable CDN path:
#   https://cdn.asurascans.com/asura-images/chapters[-restored]/<slug>/<ch>/NNN.webp
# (covers live under /covers/, so matching /chapters excludes them).
_CHAPTER_IMG_RE = re.compile(
    r"https://cdn\.asurascans\.com/asura-images/chapters[^\s\"'\\<>]+/"
    r"([^/\s\"'\\<>]+?)\.(?:webp|jpg|jpeg|png|gif|avif)"  # group1 = filename stem
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_html(url: str) -> HTMLParser:
    return HTMLParser(_get_retry(url, timeout=30).text)


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
    """Ordered chapter image URLs. Matches EVERY image on the chapter's CDN path
    (any filename) so no page is silently dropped, then orders them:
      - pure-numeric "restored" names (001.webp, 002.jpg …) → by number
      - hash / NNN_pN / mixed names (later uploads, e.g. 28578a.webp, 002_p1.webp)
        → reading order = first appearance in the page (the reader lists them
        top-to-bottom; the hashes carry no sequence to sort by).
    """
    # Decode entities + un-escape slashes (URLs may sit in an entity-encoded JSON
    # blob). Dedupe by URL; a dict preserves first-appearance (reading) order.
    text = html.unescape(page_html).replace("\\/", "/")
    stems: dict[str, str] = {}          # url -> filename stem, first-seen order
    for m in _CHAPTER_IMG_RE.finditer(text):
        stems.setdefault(m.group(0), m.group(1))
    urls = list(stems)
    if urls and all(s.isdigit() for s in stems.values()):
        urls.sort(key=lambda u: int(stems[u]))
    return urls


def _download_images(image_urls: list[str], dest_dir: Path) -> list[Path]:
    """Stream image URLs to dest_dir as 001.jpg, 002.jpg, …"""
    from PIL import Image
    import io

    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, url in enumerate(image_urls, start=1):
        out_path = dest_dir / f"{i:03d}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            written.append(out_path)        # RESUME: already fetched, skip
            continue
        resp = _get_retry(url, timeout=60)
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        tmp = out_path.with_name(out_path.name + ".tmp")
        img.save(tmp, format="JPEG")
        os.replace(tmp, out_path)     # atomic: a SIGKILL/disk-full mid-encode can't
        written.append(out_path)      # leave a partial file the size-only resume keeps
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

    def search(self, title: str) -> list[tuple[str, str]]:
        """Asura SSR-embeds search results as &quot;-escaped JSON blocks
        carrying "slug"/"title" and a "public_url" (/comics/<slug>-<hash>)."""
        from urllib.parse import quote
        try:
            resp = _get_retry(f"{_BASE_URL}/series?name={quote(title)}",
                              timeout=15)
            text = resp.text.replace("&quot;", '"')
            origin = str(resp.url).split("/series")[0]
            out: list[tuple[str, str]] = []
            for m in re.finditer(
                    r'"slug":\[0,"([^"]+)"\],"title":\[0,"([^"]+)"\]', text):
                slug, t = m.group(1), m.group(2)
                pu = re.search(r'"public_url":\[0,"(/comics/'
                               + re.escape(slug) + r'[^"]*)"\]', text)
                out.append((t, origin + (pu.group(1) if pu
                                         else f"/comics/{slug}")))
            return out[:10]
        except Exception:
            return []

    def download(self, chapter: ChapterRef, dest_dir: Path) -> list[Path]:
        resp = _get_retry(chapter.url, timeout=30)
        image_urls = _extract_image_urls(resp.text)
        if not image_urls:
            raise RuntimeError(
                f"No images found on Asura chapter page: {chapter.url}"
            )
        # COMPLETENESS: the regex matches EVERY image on the chapter path, so a
        # page can't be silently dropped by extension/format. For the pure-numeric
        # "restored" scheme we additionally require contiguous 1..N — catches a
        # strip asura itself failed to serve (the .webp-only bug was [1,3,4]).
        # Hash/_pN/mixed schemes carry no sequence, so reading order from the page
        # is the only completeness signal; a gap there can't be auto-recovered.
        stems = [m.group(1) for u in image_urls
                 if (m := _CHAPTER_IMG_RE.search(u))]
        if stems and all(s.isdigit() for s in stems):
            nums = sorted(int(s) for s in stems)
            if nums != list(range(1, nums[-1] + 1)):
                missing = sorted(set(range(1, nums[-1] + 1)) - set(nums))
                raise RuntimeError(
                    f"Asura chapter pages {nums} have GAPS (missing {missing}) — "
                    f"refusing to ship a partial chapter: {chapter.url}"
                )
        # PERSISTENT staging dir so a failed attempt RESUMES (skip-existing in
        # _download_images) instead of re-fetching every image; removed only
        # after a full success normalizes into dest_dir.
        staging = dest_dir / ".staging"
        written = _download_images(image_urls, staging)
        out = normalize_into(staging, dest_dir)
        shutil.rmtree(staging, ignore_errors=True)
        return out
