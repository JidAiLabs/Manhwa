import sqlite3
from pathlib import Path


def connect(path: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys=ON")
    # WAL: the dashboard reads on EVERY request while the worker holds long
    # write transactions. Under the default rollback journal a reader blocks a
    # writer (and vice versa), and the 'database is locked' that follows used
    # to kill a worker lane thread outright. WAL lets readers run concurrently
    # with the writer; the raised busy_timeout absorbs the writer-vs-writer
    # case (worker + refresh cron + a dashboard POST).
    # NOTE: WAL means a plain `cp studio.db` backup is NOT self-consistent —
    # use `VACUUM INTO 'studio.db.bak-…'` instead.
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=15000")
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
        -- chapter-lease lookups (claim_next) filter running jobs by chapter on
        -- every claim attempt — index keeps that a cheap lookup, not a scan.
        CREATE INDEX IF NOT EXISTS job_state_chapter ON job(state, chapter_id);
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
          teaser_state TEXT NOT NULL DEFAULT 'none',
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
    scols = {r[1] for r in con.execute("PRAGMA table_info(series)")}
    if "narration_style" not in scols:
        # per-series punch-up override: off|light|full (NULL = toml default)
        con.execute("ALTER TABLE series ADD COLUMN narration_style TEXT")
    if "autopilot" not in scols:
        # manage-by-exception: spotless QA auto-advances voice/render gates
        con.execute("ALTER TABLE series ADD COLUMN autopilot INTEGER "
                    "NOT NULL DEFAULT 0")
    if "new_pending" not in scols:
        # red-alert badge: chapters a daily refresh found since you last ran the
        # series (cleared when you bulk-run it or dismiss it)
        con.execute("ALTER TABLE series ADD COLUMN new_pending INTEGER "
                    "NOT NULL DEFAULT 0")
    # niche classification (additive, nullable): primary/secondary niche codes +
    # source genres + synopsis, used to register the narration persona per series
    for _col, _typ in (("niche_primary", "TEXT"), ("niche_secondary", "TEXT"),
                       ("genres", "TEXT"), ("synopsis", "TEXT")):
        if _col not in scols:
            con.execute(f"ALTER TABLE series ADD COLUMN {_col} {_typ}")
    if "teaser_state" not in scols:
        # The arc teaser is ONE PER MANHWA, not per video. It lived only on
        # `bundle`, so deleting a video destroyed its teaser (delete_bundle
        # rmtree's dist/bundle_<id>) and there was no way to see or keep one
        # independently of a video. Carries any existing bundle-level state up
        # to its series so an already-approved teaser survives the move,
        # preferring approved > planned > declined.
        con.execute("ALTER TABLE series ADD COLUMN teaser_state TEXT "
                    "NOT NULL DEFAULT 'none'")
        con.execute(
            "UPDATE series SET teaser_state = COALESCE(("
            "  SELECT b.teaser_state FROM bundle b"
            "   WHERE b.series_id = series.id AND b.teaser_state <> 'none'"
            "   ORDER BY CASE b.teaser_state WHEN 'approved' THEN 0"
            "                                WHEN 'planned'  THEN 1"
            "                                ELSE 2 END"
            "   LIMIT 1), 'none')")
    bcols = {r[1] for r in con.execute("PRAGMA table_info(bundle)")}
    if "teaser_state" not in bcols:
        # arc-teaser sequencing: none|planned|approved|declined
        con.execute("ALTER TABLE bundle ADD COLUMN teaser_state TEXT "
                    "NOT NULL DEFAULT 'none'")
    jcols = {r[1] for r in con.execute("PRAGMA table_info(job)")}
    if "pgid" not in jcols:
        # the live child's process-group id while the job is 'running' (the
        # worker spawns with start_new_session=True, so pgid == the child's
        # pid); NULL when no child is currently active. Lets a restarted
        # worker's orphan reaper (studio.worker.requeue_orphans) identity-check
        # and kill a surviving child before its job is requeued, instead of a
        # blind UPDATE that lets the old child and the fresh retry both write
        # the same chapter's artifacts at once.
        con.execute("ALTER TABLE job ADD COLUMN pgid INTEGER")
    acols = {r[1] for r in con.execute("PRAGMA table_info(approval)")}
    if "content_sha" not in acols:
        # bind an approval to the CONTENT it approved (script bytes / plan+
        # tts_index bytes), not just a checkbox; NULL = legacy row, grandfathered
        # valid until reconciled — see studio/dashboard/gates.py:_approval_valid
        con.execute("ALTER TABLE approval ADD COLUMN content_sha TEXT")
    con.commit()
    return con
