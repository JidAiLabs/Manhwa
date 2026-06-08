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
    """)
    con.commit()
    return con
