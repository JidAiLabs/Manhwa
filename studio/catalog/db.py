import sqlite3
from pathlib import Path


def connect(path: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS series (
          id INTEGER PRIMARY KEY,
          source TEXT NOT NULL,
          series_url TEXT NOT NULL,
          slug TEXT NOT NULL,
          title TEXT NOT NULL,
          added_at TEXT NOT NULL,
          last_checked TEXT,
          poll_priority INTEGER NOT NULL DEFAULT 100,
          UNIQUE(source, series_url)
        );
        CREATE TABLE IF NOT EXISTS chapter (
          id INTEGER PRIMARY KEY,
          series_id INTEGER NOT NULL REFERENCES series(id),
          number REAL NOT NULL,
          label TEXT NOT NULL,
          url TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'discovered',
          ep_dir TEXT,
          error TEXT,
          updated_at TEXT NOT NULL,
          UNIQUE(series_id, number)
        );

        -- dashboard (2026-06-12): queue, timings, gates, bundles, discovery
        CREATE TABLE IF NOT EXISTS job (
          id INTEGER PRIMARY KEY,
          type TEXT NOT NULL,
          series_id INTEGER,
          chapter_id INTEGER,
          bundle_id INTEGER,
          payload_json TEXT DEFAULT '{}',
          state TEXT NOT NULL DEFAULT 'queued',
          priority INTEGER NOT NULL DEFAULT 100,
          created_at TEXT DEFAULT (datetime('now')),
          started_at TEXT,
          finished_at TEXT,
          log_path TEXT,
          error TEXT
        );
        CREATE TABLE IF NOT EXISTS stage_run (
          id INTEGER PRIMARY KEY,
          chapter_id INTEGER,
          stage TEXT NOT NULL,
          started_at TEXT DEFAULT (datetime('now')),
          duration_sec REAL,
          ok INTEGER,
          meta_json TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS approval (
          id INTEGER PRIMARY KEY,
          gate TEXT NOT NULL,
          series_id INTEGER,
          chapter_id INTEGER,
          bundle_id INTEGER,
          created_at TEXT DEFAULT (datetime('now')),
          note TEXT
        );
        CREATE TABLE IF NOT EXISTS bundle (
          id INTEGER PRIMARY KEY,
          series_id INTEGER NOT NULL,
          title TEXT,
          kind TEXT NOT NULL,
          season_no INTEGER,
          state TEXT NOT NULL DEFAULT 'collecting',
          output_path TEXT,
          meta_json TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS bundle_chapter (
          bundle_id INTEGER NOT NULL,
          chapter_id INTEGER NOT NULL,
          position INTEGER NOT NULL,
          PRIMARY KEY (bundle_id, chapter_id)
        );
        CREATE TABLE IF NOT EXISTS discovery_title (
          id INTEGER PRIMARY KEY,
          anilist_id INTEGER UNIQUE,
          title TEXT,
          trend_score REAL,
          chapters INTEGER,
          status TEXT NOT NULL DEFAULT 'candidate',
          fetched_at TEXT,
          meta_json TEXT DEFAULT '{}'
        );
    """)
    cols = {r[1] for r in con.execute("PRAGMA table_info(chapter)")}
    if "season" not in cols:
        con.execute("ALTER TABLE chapter ADD COLUMN season INTEGER")
    con.commit()
    return con
