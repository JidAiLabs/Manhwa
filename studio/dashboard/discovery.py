"""AniList trending feed (read-only, cached) — the 'next potential manhwa'
signal. Offline-safe: failures keep the existing cache. Competitor-YouTube
coverage scan is the planned v1.1 second feed.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

ANILIST_URL = "https://graphql.anilist.co"
TRENDING_QUERY = """
query {
  Page(perPage: 25) {
    media(type: MANGA, countryOfOrigin: "KR", sort: TRENDING_DESC,
          format_in: [MANGA, ONE_SHOT]) {
      id
      title { romaji english }
      chapters
      trending
      popularity
      countryOfOrigin
    }
  }
}
"""


def parse_trending(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    media = (((payload.get("data") or {}).get("Page") or {}).get("media")
             or [])
    for m in media:
        t = m.get("title") or {}
        out.append({
            "anilist_id": int(m.get("id")),
            "title": t.get("english") or t.get("romaji") or "?",
            "trend_score": float(m.get("trending") or 0),
            "chapters": m.get("chapters"),
            "popularity": int(m.get("popularity") or 0),
        })
    return out


def upsert_discovery(con: sqlite3.Connection,
                     rows: List[Dict[str, Any]]) -> int:
    for r in rows:
        con.execute(
            "INSERT INTO discovery_title (anilist_id, title, trend_score, "
            "chapters, fetched_at) VALUES (?,?,?,?,datetime('now')) "
            "ON CONFLICT(anilist_id) DO UPDATE SET title=excluded.title, "
            "trend_score=excluded.trend_score, chapters=excluded.chapters, "
            "fetched_at=excluded.fetched_at",
            (r["anilist_id"], r["title"], r["trend_score"], r["chapters"]))
    con.commit()
    return len(rows)


def fetch_trending(con: sqlite3.Connection, client=None) -> int:
    try:
        if client is None:
            import httpx
            client = httpx.Client(timeout=6)
        resp = client.post(ANILIST_URL, json={"query": TRENDING_QUERY})
        payload = resp.json()
        return upsert_discovery(con, parse_trending(payload))
    except Exception:
        return 0  # offline / API change: keep the cache, never crash the UI


def mark(con: sqlite3.Connection, anilist_id: int, status: str) -> None:
    con.execute("UPDATE discovery_title SET status=? WHERE anilist_id=?",
                (status, anilist_id))
    con.commit()


def listing(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = con.execute(
        "SELECT anilist_id, title, trend_score, chapters, status "
        "FROM discovery_title ORDER BY trend_score DESC").fetchall()
    return [dict(zip(("anilist_id", "title", "trend_score", "chapters",
                      "status"), r)) for r in rows]
