"""
ArenaScan source adapter.

ArenaScan runs a WordPress "Themesia"-style reader (not gallery-dl compatible).
We scrape with httpx + selectolax, like asura and elftoon.

Series page structure (https://arenascan.com/manga/<slug>/):
  - Title: <h1> text
  - Chapter list: <div id="chapterlist"> (aka div.eplister) > li, one per
    chapter, NEWEST FIRST. Each li carries:
        data-num="167"                     -- the chapter number (authoritative)
        .chapternum  -> "Chapter 167"      -- the label
        a[href]                            -- the chapter URL, verbatim
    The whole list is server-rendered; there is no AJAX paging to follow.

  * TWO URL SCHEMES BY ERA.  The site changed its permalink format partway
    through, so a single series mixes:
        newer:  https://arenascan.com/<slug>-chapter-167/
        older:  https://arenascan.com/<slug>-83/        (no "chapter" word)
    Matching hrefs on a "-chapter-" pattern therefore finds only the newest
    couple of dozen and SILENTLY drops the rest -- a fresh series would look
    like it started at chapter 144. That is why the number comes from
    `data-num` and the href is taken verbatim, never parsed. Same failure
    family as the asura multi-filename-scheme bug.

  * The list also contains an unrendered template row (href="#/chapter-
    {{number}}"), the elftoon trap `is_fetchable_url` exists for.

Chapter page structure:
  - `ts_reader.run({...})` embeds a JSON blob whose sources[0].images is the
    ordered page list. `#readerarea img` carries the same URLs and is the
    fallback when the blob is absent or unparseable.
  - Images live on https://cdn.arenascan.com/arena-bucket/<id>/<ch>/<name>.<ext>

  * FILENAMES ALSO SPLIT BY ERA: newer chapters use unsortable content hashes
    (506f3368cc4c.webp), older ones plain numbers (0.jpg, 1.jpg). Sorting
    lexically would scramble a numeric chapter (10 before 2); sorting numerically
    is impossible for hashes. So: numeric stems sort by value, anything else
    keeps document order, which is the reader's own top-to-bottom order.
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import time
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser

from studio.sources.base import (
    Capability,
    ChapterRef,
    SeriesMeta,
    SourceAdapter,
    is_fetchable_url,
    register,
    slugify,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_BASE_URL = "https://arenascan.com"

_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ArenaScan sits behind Cloudflare (confirmed: `server: cloudflare`), so a bulk
# download burst trips rate-limiting exactly as asura's does. Same throttle +
# exponential-backoff shape, tunable per host via env.
_THROTTLE_SEC = float(os.environ.get("ARENASCAN_THROTTLE_SEC", "0.8"))
_MAX_TRIES = int(os.environ.get("ARENASCAN_MAX_TRIES", "5"))
_TRANSIENT = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
_last_req = [0.0]

# Chapter images are the only thing on the /arena-bucket/ CDN path we want;
# covers and theme assets live elsewhere.
_CHAPTER_IMG_RE = re.compile(
    r"https://cdn\.arenascan\.com/arena-bucket/[^\s\"'\\<>]+/"
    r"([^/\s\"'\\<>]+?)\.(?:webp|jpg|jpeg|png|gif|avif)"   # group1 = filename stem
)

_TS_READER_RE = re.compile(r"ts_reader\.run\((\{.*?\})\);", re.S)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_html(url: str) -> HTMLParser:
    return HTMLParser(_get_retry(url, timeout=30).text)


def _chapter_number(li) -> float | None:
    """The chapter's number, preferring the machine-readable `data-num`.

    Ladder: data-num -> .chapternum text -> None. The href is deliberately NOT
    a fallback: its format changes by era (see module docstring), so parsing it
    is what silently loses the back catalogue.
    """
    raw = (li.attributes.get("data-num") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    node = li.css_first(".chapternum")
    if node:
        m = re.search(r"([\d.]+)", node.text(strip=True))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _parse_series(tree: HTMLParser) -> tuple[str, list[ChapterRef]]:
    """Return (title, chapters ordered oldest-first) from a parsed series page."""
    h1 = tree.css_first("h1")
    title = h1.text(strip=True) if h1 else "Unknown"

    container = tree.css_first("#chapterlist") or tree.css_first("div.eplister")
    chapters: list[ChapterRef] = []
    seen: set[str] = set()
    if container is not None:
        for li in container.css("li"):
            a = li.css_first("a[href]")
            if a is None:
                continue
            href = (a.attributes.get("href") or "").strip()
            if not href.startswith("http"):
                href = _BASE_URL + "/" + href.lstrip("/")
            # the unrendered "#/chapter-{{number}}" template row, and any other
            # announced-but-unpublished placeholder
            if not is_fetchable_url(href):
                continue
            num = _chapter_number(li)
            if num is None or href in seen:
                continue
            seen.add(href)
            node = li.css_first(".chapternum")
            label = node.text(strip=True) if node else ""
            if not label:
                label = (f"Chapter {int(num)}" if num == int(num)
                         else f"Chapter {num}")
            chapters.append(ChapterRef(number=num, label=label, url=href))

    chapters.sort(key=lambda c: c.number)
    return title, chapters


def _parse_genres(page_html: str) -> tuple[str, ...]:
    """Genre tags from the /genres/<g> anchors. Fail-soft: markup churn must
    NEVER break discovery, so any parse error yields ()."""
    try:
        tree = HTMLParser(page_html)
        out = [a.text(strip=True) for a in tree.css('a[href*="genres"]')]
        return tuple(g for g in out if g)
    except Exception:
        return ()


def _parse_synopsis(page_html: str) -> str:
    """Synopsis from the og:description meta tag. Fail-soft -> ''."""
    try:
        node = HTMLParser(page_html).css_first('meta[property="og:description"]')
        return (node.attributes.get("content") or "").strip() if node else ""
    except Exception:
        return ""


def _order_urls(stems: dict[str, str]) -> list[str]:
    """Reading order for {url: filename_stem}, insertion-ordered by first
    appearance. Pure-numeric stems sort by VALUE (so 10 follows 2); hashes have
    no sequence to sort by and keep document order."""
    urls = list(stems)
    if urls and all(s.isdigit() for s in stems.values()):
        urls.sort(key=lambda u: int(stems[u]))
    return urls


def _extract_image_urls(page_html: str) -> list[str]:
    """Ordered chapter image URLs.

    Prefers the ts_reader.run() blob (the reader's own ordered page list); falls
    back to scanning the CDN pattern across the page, which also covers
    `#readerarea img` since those carry the same URLs.
    """
    text = html.unescape(page_html).replace("\\/", "/")

    m = _TS_READER_RE.search(text)
    if m:
        try:
            blob = json.loads(m.group(1))
            sources = blob.get("sources") or []
            images = (sources[0] or {}).get("images") if sources else None
            urls = [u for u in (images or []) if isinstance(u, str) and u.strip()]
            if urls:
                # dedupe, preserve the reader's order
                return list(dict.fromkeys(urls))
        except Exception:
            pass                     # fall through to the pattern scan

    stems: dict[str, str] = {}       # url -> stem, first-seen (reading) order
    for mm in _CHAPTER_IMG_RE.finditer(text):
        stems.setdefault(mm.group(0), mm.group(1))
    return _order_urls(stems)


def _download_images(image_urls: list[str], dest_dir: Path) -> list[Path]:
    """Stream image URLs to dest_dir as 001.jpg, 002.jpg, … (the site serves
    webp; the pipeline downstream expects JPEG pages)."""
    import io

    from PIL import Image

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
class ArenascanAdapter(SourceAdapter):
    """Native httpx+selectolax adapter for arenascan.com."""

    id = "arenascan"
    domains = ("arenascan.com",)
    capabilities = (Capability.DOWNLOAD | Capability.LIST_CHAPTERS
                    | Capability.SERIES_META)

    def _fetch_series_page(self, series_url: str) -> HTMLParser:
        return _get_html(series_url)

    def list_chapters(self, series_url: str) -> list[ChapterRef]:
        tree = self._fetch_series_page(series_url)
        _, chapters = _parse_series(tree)
        return chapters

    def series_meta(self, series_url: str) -> SeriesMeta:
        tree = self._fetch_series_page(series_url)
        title, _ = _parse_series(tree)
        page_html = tree.html or ""     # reuse the fetched page — no second GET
        return SeriesMeta(
            source=self.id,
            series_url=series_url,
            title=title,
            slug=slugify(title),
            genres=_parse_genres(page_html),
            synopsis=_parse_synopsis(page_html),
        )

    def download(self, chapter: ChapterRef, dest_dir: Path) -> list[Path]:
        page = _get_retry(chapter.url, timeout=30).text
        urls = _extract_image_urls(page)
        if not urls:
            raise RuntimeError(
                f"arenascan: no chapter images found at {chapter.url}")
        return _download_images(urls, Path(dest_dir))

    def search(self, title: str) -> list[tuple[str, str]]:
        """Best-effort WordPress search (?s=). Must swallow its own errors —
        site churn can never be allowed to break discovery."""
        from urllib.parse import quote
        try:
            resp = _get_retry(f"{_BASE_URL}/?s={quote(title)}", timeout=20)
            tree = HTMLParser(resp.text)
            out: list[tuple[str, str]] = []
            seen: set[str] = set()
            for a in tree.css("a[href]"):
                href = (a.attributes.get("href") or "").strip()
                if "/manga/" not in href or href in seen:
                    continue
                name = (a.attributes.get("title") or a.text(strip=True) or "").strip()
                if not name:
                    continue
                seen.add(href)
                out.append((name, href))
            return out
        except Exception:
            return []
