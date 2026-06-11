# OriginPower Studio Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Local two-process control dashboard (FastAPI+Jinja+htmx UI + worker queue executor over `studio.db`) with run buttons, live logs, QA/approval gates, season/full-series bundles with intro/outro-correct concat, ETAs from measured stage timings, and AniList discovery.

**Architecture:** UI handlers only read DB / insert `job`+`approval` rows; a separate `studio worker` claims one job at a time (serial GPU), executes via existing `studio/pipeline.py` + tools, logs to `logs/jobs/<id>.log`, records `stage_run` durations. Gates enforced worker-side only.

**Tech Stack:** FastAPI, Jinja2, htmx (CDN), sqlite3 (existing catalog db.py style), httpx (AniList), ffmpeg concat. Tests: pytest + fastapi TestClient in `.eval_venv`.

**Conventions:** suite must stay green (`.eval_venv/bin/python -m pytest -q`, 382 passing now). Spec: `docs/plans/specs/2026-06-12-dashboard-design.md`. Mockup (visual contract): `docs/plans/specs/mockups/dashboard-mockup.html`.

---

### Task 1: Migrations — new tables + chapter.season

**Files:** Modify `studio/catalog/db.py` (extend `connect()`); Test `tests/dashboard/test_db_migrations.py`

- [ ] Test (fresh db has tables; legacy db without `chapter.season` gets the column):

```python
from studio.catalog.db import connect

def test_new_tables_and_season_column(tmp_path):
    con = connect(tmp_path / "s.db")
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"job", "stage_run", "approval", "bundle", "bundle_chapter",
            "discovery_title"} <= names
    cols = {r[1] for r in con.execute("PRAGMA table_info(chapter)")}
    assert "season" in cols

def test_existing_db_upgraded(tmp_path):
    import sqlite3
    p = tmp_path / "old.db"
    raw = sqlite3.connect(p)
    raw.execute("CREATE TABLE chapter (id INTEGER PRIMARY KEY, number REAL)")
    raw.commit(); raw.close()
    con = connect(p)
    cols = {r[1] for r in con.execute("PRAGMA table_info(chapter)")}
    assert "season" in cols
```

- [ ] Run: fails (tables missing). Implement in `connect()` after existing DDL:

```python
    con.executescript("""
        CREATE TABLE IF NOT EXISTS job (
            id INTEGER PRIMARY KEY, type TEXT NOT NULL,
            series_id INTEGER, chapter_id INTEGER, bundle_id INTEGER,
            payload_json TEXT DEFAULT '{}',
            state TEXT NOT NULL DEFAULT 'queued',
            priority INTEGER NOT NULL DEFAULT 100,
            created_at TEXT DEFAULT (datetime('now')),
            started_at TEXT, finished_at TEXT,
            log_path TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS stage_run (
            id INTEGER PRIMARY KEY, chapter_id INTEGER, stage TEXT NOT NULL,
            started_at TEXT DEFAULT (datetime('now')),
            duration_sec REAL, ok INTEGER, meta_json TEXT DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS approval (
            id INTEGER PRIMARY KEY, gate TEXT NOT NULL,
            series_id INTEGER, chapter_id INTEGER, bundle_id INTEGER,
            created_at TEXT DEFAULT (datetime('now')), note TEXT);
        CREATE TABLE IF NOT EXISTS bundle (
            id INTEGER PRIMARY KEY, series_id INTEGER NOT NULL,
            title TEXT, kind TEXT NOT NULL, season_no INTEGER,
            state TEXT NOT NULL DEFAULT 'collecting',
            output_path TEXT, meta_json TEXT DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS bundle_chapter (
            bundle_id INTEGER NOT NULL, chapter_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            PRIMARY KEY (bundle_id, chapter_id));
        CREATE TABLE IF NOT EXISTS discovery_title (
            id INTEGER PRIMARY KEY, anilist_id INTEGER UNIQUE,
            title TEXT, trend_score REAL, chapters INTEGER,
            status TEXT NOT NULL DEFAULT 'candidate',
            fetched_at TEXT, meta_json TEXT DEFAULT '{}');
    """)
    cols = {r[1] for r in con.execute("PRAGMA table_info(chapter)")}
    if "season" not in cols:
        con.execute("ALTER TABLE chapter ADD COLUMN season INTEGER")
    con.commit()
```

- [ ] Run tests → pass; full suite green; commit `feat(dashboard): catalog migrations`.

### Task 2: Job queue core — `studio/dashboard/jobs.py`

**Files:** Create `studio/dashboard/__init__.py`, `studio/dashboard/jobs.py`; Test `tests/dashboard/test_jobs.py`

Contract (pure sqlite3 helpers, connection passed in):

```python
enqueue(con, type, *, series_id=None, chapter_id=None, bundle_id=None,
        payload=None, priority=100) -> int
claim_next(con) -> dict | None      # None if any job state='running' (SERIAL)
                                     # else oldest queued by (priority, id);
                                     # atomically sets running+started_at
finish(con, job_id, *, ok: bool, error: str = "") -> None
cancel(con, job_id) -> bool          # only queued jobs
bump(con, job_id) -> None            # priority -= 1 (moves up)
queue_view(con) -> list[dict]        # running first, then queued by order
```

- [ ] Tests: serial claim (claim returns None while one running), FIFO by priority then id, cancel only-queued, finish stamps `finished_at`. Write → fail → implement → pass → commit `feat(dashboard): job queue core`.

### Task 3: Gates — `studio/dashboard/gates.py`

**Files:** Create `studio/dashboard/gates.py`; Test `tests/dashboard/test_gates.py`

```python
approve(con, gate, *, series_id=None, chapter_id=None, bundle_id=None, note="") -> int
latest_qa_ok(con, chapter_id) -> bool   # last stage_run stage='qa_scan' ok==1
render_allowed(con, chapter_id) -> tuple[bool, str]
    # needs latest_qa_ok AND approval(gate='render', chapter_id) -> (True,"")
    # else (False, "needs QA pass" | "needs render approval")
concat_allowed(con, bundle_id) -> tuple[bool, str]   # approval gate='concat'
```

- [ ] Tests for each refusal message + happy path. Commit `feat(dashboard): approval+QA gates`.

### Task 4: ETA model — `studio/dashboard/eta.py`

**Files:** Create `studio/dashboard/eta.py`; Test `tests/dashboard/test_eta.py`

```python
SEED_SEC = {"fetched": 30, "stitched": 40, "detected": 60, "scened": 10,
            "visioned": 120, "grouped": 5, "beated": 1980, "scripted": 5,
            "voiced": 1200, "planned": 10, "prepped": 130, "qa_scan": 120,
            "render_segment": 2400, "concat": 180}
stage_eta(con, stage, series_id=None) -> float
    # median stage_run dur (series-scoped) -> global median -> SEED_SEC
chapter_eta(con, chapter_id, remaining: list[str], series_id=None) -> float
series_eta(con, series_id, chapters_remaining: int) -> float
fmt_eta(sec) -> str                   # "7:20", "1.6 h", "26 days"
```

- [ ] Tests: fallback chain (no data → seed; series data overrides global), fmt. Commit `feat(dashboard): ETA model`.

### Task 5: `render_prep --branding` flag (bundle-correct intro/outro)

**Files:** Modify `tools/render_prep.py` (arg + `insert_branding_items(which=...)`); Test additions in `tests/test_render_prep.py`

```python
def insert_branding_items(plan, *, intro_dur, outro_dur, which="both", ...):
    # which in {"both","intro","outro","none"}; intro skipped unless
    # which in ("both","intro"); outro skipped unless ("both","outro")
ap.add_argument("--branding", choices=["both", "intro", "outro", "none"],
                default="both")   # --no-branding stays as alias for "none"
```

- [ ] Tests: `which="intro"` → intro item present, no outro; `"outro"` inverse; `"none"` untouched; default unchanged (existing tests stay green). Commit `feat(prep): --branding for bundle segments`.

### Task 6: Bundles — `studio/dashboard/bundles.py`

**Files:** Create `studio/dashboard/bundles.py`; Test `tests/dashboard/test_bundles.py`

```python
create_bundle(con, series_id, kind, *, season_no=None,
              chapter_range=None, title="") -> int
    # kind 'season': chapters WHERE season=season_no ordered by number
    # kind 'full': all chapters; 'manual': chapter_range (lo, hi)
branding_for_position(i, n) -> str       # 0->'intro', n-1->'outro', else 'none'
                                          # n==1 -> 'both'
projected_runtime_sec(con, bundle_id, plan_loader) -> float
    # sum chapter plan total_duration_sec (plan_loader(chapter) -> float|None,
    # None -> chapter ETA seed) + intro/outro once
concat_cmd(segments: list[str], out_path: str) -> list[str]
    # ffmpeg -f concat -safe 0 -i <listfile> -c copy out  (listfile written
    # by caller; function returns argv + listfile body for testability)
segments_ready(con, bundle_id, probe) -> tuple[int, int]
```

- [ ] Tests: season selection order, branding map (incl. n==1), concat argv, runtime fallback. Commit `feat(dashboard): bundles + concat planning`.

### Task 7: Worker — `studio/worker.py`

**Files:** Create `studio/worker.py`; Test `tests/dashboard/test_worker.py`

Contract:

```python
HANDLERS: dict[str, Callable[[sqlite3.Connection, dict, TextIO], None]]
# 'chain'          payload {"target": "voiced"} -> pipeline stages w/ a
#                  stage_run-recording wrapper around each stage fn
# 'qa_scan'        subprocess tools/prep_qa.py; ok = returncode==0;
#                  records stage_run('qa_scan')
# 'render_segment' GATED by gates.render_allowed; runs remotion render with
#                  prep --branding from payload; records stage_run
# 'concat'         GATED by gates.concat_allowed; ffmpeg concat
# 'refresh'        sources discovery for new chapters -> insert + badge
run_once(con, *, handlers=HANDLERS, log_dir="logs/jobs") -> bool
    # claim_next; open log; dispatch; finish(ok/error). False if idle.
main() -> loop run_once with 2s sleep; heartbeat row in job table
    # (type='heartbeat' singleton, updated each tick)
```

- [ ] Tests with INJECTED stub handlers: claims serially, writes log file, gate-refused render job → state failed + error message contains gate reason, stage_run row written by the recording wrapper, heartbeat updates. NO real pipeline calls in tests. Commit `feat(dashboard): worker executor`.

### Task 8: AniList discovery — `studio/dashboard/discovery.py`

**Files:** Create `studio/dashboard/discovery.py`; Test `tests/dashboard/test_discovery.py` (+ fixture `tests/dashboard/fixtures/anilist_trending.json`)

```python
TRENDING_QUERY = "...Page(perPage:25){media(type:MANGA,countryOfOrigin:KR|CN,
                  sort:TRENDING_DESC){id title{romaji english} chapters
                  trending popularity}}..."   # exact GraphQL in impl
parse_trending(payload: dict) -> list[dict]      # pure, fixture-tested
upsert_discovery(con, rows) -> int               # keeps existing status
fetch_trending(con, client=None) -> int          # httpx, 6s timeout,
                                                 # returns 0 + keeps cache on error
mark(con, discovery_id, status) -> None          # 'tracked'|'ignored'
```

- [ ] Tests: parse fixture, upsert preserves status='tracked' on refresh, offline → 0 without raising. Commit `feat(dashboard): AniList discovery`.

### Task 9: FastAPI app — `studio/dashboard/app.py` + templates

**Files:** Create `studio/dashboard/app.py`, `studio/dashboard/templates/{base,queue,series,chapter,videos,discovery,health}.html`, `studio/dashboard/templates/partials/{queue_table,log_tail,stage_timeline}.html`, `studio/dashboard/static/style.css` (extracted from mockup); Test `tests/dashboard/test_app.py`

Routes (UI handlers: read-only + row inserts ONLY):

```
GET  /                    queue page          GET /partials/queue (htmx 2s)
GET  /partials/log/{job_id}                   tail -c 8192 of log file
GET  /series              board               GET /series/{id} chapter list
GET  /chapter/{id}        detail: stage timeline (stage_run + ETAs + locks),
                          QA badges, gallery (groups from render.plan.clean.json:
                          tts_text + cut files -> /media thumbs), audio links
GET  /videos              bundles  POST /bundles {series_id, kind, season_no?}
GET  /discovery           table    POST /discovery/{id}/track
GET  /health              ollama tags via httpx localhost:11434 (graceful),
                          venv paths exist, disk_free, worker heartbeat age
POST /jobs                {type, chapter_id|series_id|bundle_id, payload}
POST /jobs/{id}/cancel    POST /jobs/{id}/up
POST /approve             {gate, chapter_id|bundle_id, note}
GET  /media/{path:path}   StaticFiles over ongoing/ (read-only)
app = create_app(db_path=...)   # factory for tests (tmp db)
```

- [ ] TestClient tests: every GET 200 with seeded tmp db; POST /jobs inserts queued row; POST /approve inserts approval; cancel works; /media serves a seeded file; log partial tails. Templates render with the mockup's CSS classes (visual contract). Commit `feat(dashboard): FastAPI app + pages`.

### Task 10: CLI wiring + deps

**Files:** Modify `studio/cli.py` (subcommands `dashboard` [--port 8170, --db], `worker` [--db]); Modify `.eval_venv` (pip install fastapi uvicorn jinja2 python-multipart httpx); Create `studio/dashboard/requirements.txt`; Test `tests/dashboard/test_cli_dashboard.py` (parser-level: args parse + funcs referenced)

- [ ] Commit `feat(dashboard): studio dashboard/worker commands`.

### Task 11: Integration smoke + docs

- [ ] With real `studio.db` COPY in tmp: enqueue `qa_scan` for Nano ch1, `run_once` with real handler → job done, stage_run row ok=1 (prep_qa exits 0 on Nano).
- [ ] Boot app against real db (read-only ops): `/`, `/series`, `/chapter/1`, `/videos`, `/discovery`, `/health` all 200.
- [ ] Update `.continue-here.md` + `studio/README.md` run instructions (`studio dashboard` + `studio worker`, two terminals).
- [ ] Full suite green; commit `feat(dashboard): integration smoke + docs`.

## Self-review

Spec coverage: pages 1-6 → Task 9; gates → 3+7; ETAs → 4; bundles+branding → 5+6; discovery → 8; serial queue → 2+7; new-chapter refresh → 7 ('refresh' handler) + series badges (Task 9 board reads recent chapter rows); health → 9. Type names consistent (`run_once`, `claim_next`, `render_allowed` used identically across tasks). No placeholders: route/contract tables are the implementation interface; each task carries test content.
