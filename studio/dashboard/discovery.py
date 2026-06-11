"""Discovery: AniList trending + auto-linked source URLs + YouTube coverage.

The 'what to make next' instrument: per trending manhwa it knows the
summary/genres (AniList), WHERE to fetch it (source-site search, fuzzy
matched), and how saturated the recap niche already is (YouTube scan —
Data API v3 when YOUTUBE_API_KEY is set, yt-dlp search otherwise).
Everything cached in discovery_title.meta_json; every fetch is
offline-safe (failures keep the cache).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sqlite3
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

ANILIST_URL = "https://graphql.anilist.co"
TRENDING_QUERY = """
query {
  Page(perPage: 25) {
    media(type: MANGA, countryOfOrigin: "KR", sort: TRENDING_DESC,
          format_in: [MANGA, ONE_SHOT]) {
      id
      title { romaji english }
      description(asHtml: false)
      genres
      status
      averageScore
      chapters
      trending
      popularity
      countryOfOrigin
    }
  }
}
"""

_TAG_RE = re.compile(r"<[^>]+>")


def parse_trending(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    media = (((payload.get("data") or {}).get("Page") or {}).get("media")
             or [])
    for m in media:
        t = m.get("title") or {}
        desc = _TAG_RE.sub(" ", str(m.get("description") or ""))
        desc = re.sub(r"\s+", " ", desc).strip()[:600]
        out.append({
            "anilist_id": int(m.get("id")),
            "title": t.get("english") or t.get("romaji") or "?",
            "trend_score": float(m.get("trending") or 0),
            "chapters": m.get("chapters"),
            "popularity": int(m.get("popularity") or 0),
            "description": desc,
            "genres": m.get("genres") or [],
            "status": m.get("status") or "",
            "score": m.get("averageScore"),
        })
    return out


def _merge_meta(con: sqlite3.Connection, anilist_id: int,
                patch: Dict[str, Any]) -> str:
    row = con.execute("SELECT meta_json FROM discovery_title WHERE "
                      "anilist_id=?", (anilist_id,)).fetchone()
    meta = {}
    if row and row[0]:
        try:
            meta = json.loads(row[0])
        except Exception:
            meta = {}
    meta.update(patch)
    return json.dumps(meta, ensure_ascii=False)


def upsert_discovery(con: sqlite3.Connection,
                     rows: List[Dict[str, Any]]) -> int:
    for r in rows:
        meta_patch = {k: r[k] for k in ("description", "genres", "status",
                                        "score", "popularity") if k in r}
        con.execute(
            "INSERT INTO discovery_title (anilist_id, title, trend_score, "
            "chapters, fetched_at, meta_json) "
            "VALUES (?,?,?,?,datetime('now'),?) "
            "ON CONFLICT(anilist_id) DO UPDATE SET title=excluded.title, "
            "trend_score=excluded.trend_score, chapters=excluded.chapters, "
            "fetched_at=excluded.fetched_at, meta_json=excluded.meta_json",
            (r["anilist_id"], r["title"], r["trend_score"], r["chapters"],
             _merge_meta(con, r["anilist_id"], meta_patch)))
    con.commit()
    return len(rows)


def fetch_trending(con: sqlite3.Connection, client=None) -> int:
    try:
        if client is None:
            import httpx
            client = httpx.Client(timeout=8)
        resp = client.post(ANILIST_URL, json={"query": TRENDING_QUERY})
        return upsert_discovery(con, parse_trending(resp.json()))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# source linking
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def best_match(title: str,
               candidates: List[Tuple[str, str]]) -> Optional[Dict[str, Any]]:
    """Pick the candidate whose title best matches; None below 0.6."""
    nt = _norm(title)
    best: Optional[Dict[str, Any]] = None
    for cand_title, url in candidates:
        nc = _norm(cand_title)
        score = difflib.SequenceMatcher(None, nt, nc).ratio()
        if nt and (nt in nc or nc in nt):
            score = max(score, 0.92)
        if best is None or score > best["score"]:
            best = {"title": cand_title, "url": url,
                    "score": round(score, 2)}
    return best if best and best["score"] >= 0.6 else None


def link_sources(con: sqlite3.Connection, anilist_id: int, title: str,
                 searchers: Dict[str, Callable[[str],
                                               List[Tuple[str, str]]]]) -> Dict[str, Any]:
    links: Dict[str, Any] = {}
    for source, search in searchers.items():
        try:
            hit = best_match(title, search(title) or [])
        except Exception:
            hit = None
        if hit:
            links[source] = hit
    con.execute("UPDATE discovery_title SET meta_json=? WHERE anilist_id=?",
                (_merge_meta(con, anilist_id, {"links": links}), anilist_id))
    con.commit()
    return links


# ---------------------------------------------------------------------------
# YouTube coverage (competitor saturation)
# ---------------------------------------------------------------------------

def parse_ytdlp_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    entries = payload.get("entries") or []
    views = [int(e.get("view_count") or 0) for e in entries]
    channels = sorted({str(e.get("channel") or e.get("uploader") or "?")
                       for e in entries} - {"?"})
    return {"videos": len(entries), "max_views": max(views, default=0),
            "channels": channels[:6]}


def youtube_coverage(title: str, *, runner=None) -> Optional[Dict[str, Any]]:
    """Competitor recap coverage for *title*.

    YOUTUBE_API_KEY set -> Data API v3 (search.list + videos.list stats);
    otherwise yt-dlp flat search (no key). None on failure."""
    q = f"{title} manhwa recap"
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if key:
        try:
            import httpx
            s = httpx.get("https://www.googleapis.com/youtube/v3/search",
                          params={"part": "snippet", "q": q, "type": "video",
                                  "maxResults": 10, "key": key},
                          timeout=8).json()
            ids = [i["id"]["videoId"] for i in s.get("items") or []]
            if not ids:
                return {"videos": 0, "max_views": 0, "channels": []}
            v = httpx.get("https://www.googleapis.com/youtube/v3/videos",
                          params={"part": "statistics,snippet",
                                  "id": ",".join(ids), "key": key},
                          timeout=8).json()
            views = [int((i.get("statistics") or {}).get("viewCount") or 0)
                     for i in v.get("items") or []]
            channels = sorted({(i.get("snippet") or {}).get("channelTitle")
                               or "?" for i in v.get("items") or []} - {"?"})
            return {"videos": len(ids), "max_views": max(views, default=0),
                    "channels": channels[:6]}
        except Exception:
            return None
    run = runner or (lambda cmd: subprocess.run(
        cmd, capture_output=True, text=True, timeout=60))
    try:
        p = run(["yt-dlp", f"ytsearch10:{q}", "--flat-playlist", "-J",
                 "--no-warnings"])
        return parse_ytdlp_search(json.loads(p.stdout))
    except Exception:
        return None


def set_youtube(con: sqlite3.Connection, anilist_id: int,
                cov: Dict[str, Any]) -> None:
    con.execute("UPDATE discovery_title SET meta_json=? WHERE anilist_id=?",
                (_merge_meta(con, anilist_id, {"youtube": cov}), anilist_id))
    con.commit()


def mark(con: sqlite3.Connection, anilist_id: int, status: str) -> None:
    con.execute("UPDATE discovery_title SET status=? WHERE anilist_id=?",
                (status, anilist_id))
    con.commit()


def opportunity(row: Dict[str, Any]) -> bool:
    """High trend + thin competitor coverage = make this next."""
    yt = (row.get("meta") or {}).get("youtube") or {}
    if not yt:
        return False
    return (row.get("trend_score") or 0) >= 60 and (
        yt.get("videos", 0) <= 3 and yt.get("max_views", 0) < 100_000)


def listing(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = con.execute(
        "SELECT anilist_id, title, trend_score, chapters, status, meta_json "
        "FROM discovery_title ORDER BY trend_score DESC").fetchall()
    out = []
    for r in rows:
        d = dict(zip(("anilist_id", "title", "trend_score", "chapters",
                      "status", "meta_json"), r))
        try:
            d["meta"] = json.loads(d.pop("meta_json") or "{}")
        except Exception:
            d["meta"] = {}
        d["opportunity"] = opportunity(d)
        out.append(d)
    return out


def scan(con: sqlite3.Connection, *, client=None,
         searchers: Optional[Dict[str, Callable]] = None,
         yt_runner=None, log=print) -> int:
    """Full pass: trends -> per-title source links + youtube coverage."""
    n = fetch_trending(con, client=client)
    log(f"[discovery] anilist rows: {n}")
    if searchers is None:
        from studio.sources.base import get_adapter
        searchers = {}
        for sid in ("asura", "webtoon", "elftoon"):
            try:
                searchers[sid] = get_adapter(sid).search
            except Exception:
                pass
    done = 0
    for row in listing(con):
        if row["status"] == "ignored":
            continue
        links = link_sources(con, row["anilist_id"], row["title"], searchers)
        cov = youtube_coverage(row["title"], runner=yt_runner)
        if cov is not None:
            set_youtube(con, row["anilist_id"], cov)
        log(f"[discovery] {row['title']}: links={list(links)} "
            f"yt={cov and cov.get('videos')}")
        done += 1
    return done
