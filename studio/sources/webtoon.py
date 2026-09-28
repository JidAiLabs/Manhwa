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
from urllib.parse import parse_qs, urljoin, urlsplit

from studio.sources.base import (
    Capability,
    ChapterRef,
    SeriesMeta,
    SourceAdapter,
    UnsupportedSource,
    register,
    slugify,
)
from studio.sources.gallerydl import (
    _EXTRACTOR_ERROR_PHRASES,
    normalize_into,
    run_download,
)

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
        err = result.stderr.strip()
        # a link gallery-dl has no extractor for is permanent: say so, so the
        # add is not retried (run_download classifies it the same way)
        if any(p in err.lower() for p in _EXTRACTOR_ERROR_PHRASES):
            raise UnsupportedSource(f"gallery-dl cannot read '{url}': {err}")
        raise RuntimeError(f"gallery-dl -j failed (exit {result.returncode}): {err}")
    return json.loads(result.stdout)


_WT_HOSTS = {"webtoons.com", "www.webtoons.com", "m.webtoons.com"}
_PASTE_HINT = "paste the series page link (it contains title_no=)"


def _series_parts(url: str):
    """(lang, genre, slug, title_no) of a webtoons.com series, list or episode
    link; title_no is "" when the link has none. None when it is not one."""
    parts = urlsplit(str(url or "").strip())
    if (parts.hostname or "").lower() not in _WT_HOSTS:
        return None
    segs = [s for s in parts.path.split("/") if s]
    if len(segs) < 3:
        return None
    title_no = (parse_qs(parts.query).get("title_no") or [""])[0]
    return segs[0], segs[1], segs[2], title_no if title_no.isdigit() else ""


def _canonical_list_url(lang: str, genre: str, slug: str, title_no: str) -> str:
    """The one form gallery-dl reads from page 1. A pasted `page=3` made it
    start at page 3 and skip the newest episodes."""
    return (f"https://www.webtoons.com/{lang}/{genre}/{slug}/list"
            f"?title_no={title_no}")


def _comic_slug_to_title(slug: str) -> str:
    """Convert 'omniscient-reader' → 'Omniscient Reader'."""
    return slug.replace("-", " ").title()


def _genres_from_meta(meta: dict) -> tuple[str, ...]:
    """Best-effort genres from a gallery-dl episode meta dict. gallery-dl gives a
    single `genre` string (e.g. "action"); accept a `genres`/`tags` list too if a
    future extractor exposes one. Fail-soft → () so discovery never breaks."""
    try:
        raw = meta.get("genres") or meta.get("tags") or meta.get("genre")
        if not raw:
            return ()
        if isinstance(raw, str):
            raw = [raw]
        return tuple(str(g).strip() for g in raw if str(g).strip())
    except Exception:
        return ()


def _episode_slug(url: str) -> str:
    """The human-facing slug from a viewer URL.

    .../omniscient-reader/episode-0-prologue/viewer?title_no=2154
      -> 'episode-0-prologue'
    """
    parts = url.split("?")[0].rstrip("/").split("/")
    if len(parts) >= 2 and parts[-1] == "viewer":
        return parts[-2]
    return parts[-1] if parts else ""


_SEASON_SLUG_RE = re.compile(
    r"^season[-_]?(\d+)[-_]?(?:episode|ep)[-_]?(\d+)(?:[-_](.+))?$", re.I)
_EPISODE_SLUG_RE = re.compile(
    r"^(?:episode|ep|chapter|ch)[-_]?(\d+)(?:[-_](.+))?$", re.I)


def _prettify(text: str) -> str:
    """'prologue' -> 'Prologue'; 'the-beginning' -> 'The Beginning'."""
    return " ".join(w.capitalize() for w in re.split(r"[-_]+", text) if w)


def _label_from_slug(slug: str, episode_no) -> str:
    """The episode name a HUMAN would use, taken from the URL slug.

    gallery-dl's ``episode_no`` is an internal 1-based sequence index, NOT the
    published episode number, so labelling from it made every episode read one
    too high: the prologue (slug 'episode-0-prologue', episode_no=1) was
    labelled "Episode 1" and everything after it shifted by one.

    The slug carries the number the site itself shows the reader, so it is the
    label authority. It is used for the LABEL ONLY — ``number`` stays
    ``episode_no``, because that is the UNIQUE key and the refresh-cron upsert
    key, and it is the one value guaranteed unique and monotonic.

    That separation is load-bearing, not stylistic. Slug numbers COLLIDE on
    multi-season series: Tower of God yields only 338 distinct slug numbers
    across 652 episodes (236 collisions), because 'season-1-ep-0' and
    'season-2-ep-0' both parse to 0. Using them as ``number`` would violate
    UNIQUE(series_id, number) wholesale. Season is carried in the LABEL, where
    a collision is impossible.

    Falls back to the old episode_no label when a slug carries no number
    (specials, titled side stories) — strictly no worse than the old behaviour.
    """
    s = (slug or "").strip()
    m = _SEASON_SLUG_RE.match(s)
    if m:
        season, ep, suffix = m.group(1), m.group(2), m.group(3)
        label = f"Season {int(season)} Episode {int(ep)}"
        return f"{label} ({_prettify(suffix)})" if suffix else label
    m = _EPISODE_SLUG_RE.match(s)
    if m:
        ep, suffix = m.group(1), m.group(2)
        label = f"Episode {int(ep)}"
        return f"{label} ({_prettify(suffix)})" if suffix else label
    return f"Episode {episode_no}"


def _parse_chapters(entries: list) -> list[ChapterRef]:
    """
    Parse gallery-dl -j entries into ordered ChapterRefs.

    Each entry from the list URL is: [6, viewer_url, {episode_no, comic, ...}]
    They are ordered newest-first; we sort ascending by episode_no.

    ``number`` is episode_no (stable, unique, monotonic — the upsert key);
    ``label`` comes from the URL slug (what the site shows the reader). See
    _label_from_slug for why these must not be the same value.
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
                label=_label_from_slug(_episode_slug(url), episode_no),
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
    domains = ("webtoons.com",)
    capabilities = Capability.DOWNLOAD | Capability.LIST_CHAPTERS | Capability.SERIES_META

    def _search_raw(self, title: str) -> list[tuple[str, str]]:
        """webtoons.com/en/search — result cards are <a class='link _card_item'>
        with the title as the first text line. RAISES on an HTTP failure, so
        the resolver can tell a network blip from "no such series"."""
        from urllib.parse import quote

        import httpx
        from selectolax.parser import HTMLParser
        r = httpx.get(
            "https://www.webtoons.com/en/search?keyword=" + quote(title),
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS "
                                   "X 10_15_7) AppleWebKit/537.36"},
            follow_redirects=True, timeout=15)
        r.raise_for_status()
        out: list[tuple[str, str]] = []
        for a in HTMLParser(r.text).css("a._card_item"):
            href = a.attributes.get("href") or ""
            first_line = next((ln.strip() for ln in
                               (a.text() or "").splitlines()
                               if ln.strip()), "")
            if href and first_line:
                out.append((first_line, urljoin("https://www.webtoons.com/", href)))
        return out

    def search(self, title: str) -> list[tuple[str, str]]:
        """Discovery's contract: never raises, at most 10 results."""
        try:
            return self._search_raw(title)[:10]
        except Exception:
            return []

    def resolve_series_url(self, url: str) -> str:
        """Any webtoons.com link to a series -> its canonical list page.

        A link with title_no (a list page, an episode) needs no network. A
        SHORT link (`/en/action/rise-of-the-devourer/`) cannot be read: the
        site answers it, and `/list` without title_no, with HTTP 500, and
        gallery-dl calls it unsupported (jobs 2934-2936 failed three times on
        it, 2026-09-27). The site search returns the same path WITH title_no,
        so the series is found by an exact path match, never a fuzzy title."""
        got = _series_parts(url)
        if got is None:
            raise UnsupportedSource(
                f"not a webtoons.com series link: {url!r} — {_PASTE_HINT}")
        lang, genre, slug, title_no = got
        if title_no:
            return _canonical_list_url(lang, genre, slug, title_no)
        if lang != "en":
            raise UnsupportedSource(
                f"a '{lang}' link without title_no cannot be looked up (the "
                f"site search is English only) — {_PASTE_HINT}")
        exact, same_slug = [], set()
        for _title, href in self._search_raw(slug.replace("-", " ")):
            hit = _series_parts(href)
            if not hit or not hit[3] or hit[0] != lang \
                    or hit[2].lower() != slug.lower():
                continue
            if hit[1].lower() == genre.lower():
                exact.append(hit)
            same_slug.add(hit)
        if exact:
            return _canonical_list_url(*exact[0])
        if len(same_slug) == 1:            # the genre in the pasted link was off
            return _canonical_list_url(*same_slug.pop())
        raise UnsupportedSource(
            f"webtoons.com search has no series at /{lang}/{genre}/{slug}/ — "
            f"{_PASTE_HINT}")

    def _fetch_entries(self, series_url: str) -> list:
        return _run_gallery_dl_j(series_url)

    def list_chapters(self, series_url: str) -> list[ChapterRef]:
        entries = self._fetch_entries(series_url)
        return _parse_chapters(entries)

    def series_meta(self, series_url: str) -> SeriesMeta:
        entries = self._fetch_entries(series_url)
        chapters = _parse_chapters(entries)

        # Derive title (+ best-effort genres/synopsis) from the first entry's
        # metadata. gallery-dl exposes a single `genre` string per episode; no
        # description key is exposed, so synopsis falls back to '' (base voice).
        comic_slug: str = "unknown"
        meta: dict = {}
        for entry in entries:
            if isinstance(entry, list) and len(entry) >= 3 and entry[0] == 6:
                meta = entry[2] if isinstance(entry[2], dict) else {}
                comic_slug = meta.get("comic", "unknown")
                break

        title = _comic_slug_to_title(comic_slug)
        slug = slugify(title)
        genres = _genres_from_meta(meta)
        synopsis = str(meta.get("description") or meta.get("summary") or "").strip()

        return SeriesMeta(
            source=self.id,
            series_url=series_url,
            title=title,
            slug=slug,
            genres=genres,
            synopsis=synopsis,
        )

    def download(self, chapter: ChapterRef, dest_dir: Path) -> list[Path]:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_download(chapter.url, tmp_path)
            return normalize_into(tmp_path, dest_dir)
