# Acquisition + Catalog Spine Implementation Plan

> **For agentic workers:** REQUIRED: Use subagent-driven-development (if subagents available) or executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `studio/` front-end that fetches manhwa chapters from configurable sources into the existing episode layout, tracks every chapter in a SQLite catalog, and runs a fetched chapter through the existing `tools/` pipeline to `render.plan.json` — with three pipeline break-fixes and the user's trained YOLO replacing the Gemini panel stage.

**Architecture:** A new `studio/` package with four isolated units — `catalog/` (SQLite state), `sources/` (pluggable `SourceAdapter`s + gallery-dl backend), `detect/` (YOLO panel adapter), and `pipeline.py`/`cli.py` (orchestration). The existing `tools/` scripts are reused unchanged except for three targeted break-fixes. Sources are config-driven so a dead/renamed site is a `studio.toml` edit, not code.

**Tech Stack:** Python 3.12, `pytest`, stdlib `sqlite3` + `tomllib`, `gallery-dl` (subprocess), `ultralytics`/`torch` (YOLO), `httpx` + `selectolax` (native adapter fallback), `Pillow`.

**Spec:** `docs/plans/specs/2026-06-09-acquisition-catalog-spine-design.md`

**Conventions for every task below:** TDD — write the failing test first, watch it fail, implement minimally, watch it pass, commit. Use `studio/.venv` (the eval env `.eval_venv` already has torch and can be reused/renamed). Reference @superpowers:test-driven-development and @superpowers:systematic-debugging when stuck.

---

## Chunk 1: Project Setup + Catalog

### Task 1: Repo + package scaffold

**Files:**
- Create: `.gitignore`, `studio/__init__.py`, `studio/config.py`, `studio.toml`, `pyproject.toml`, `tests/__init__.py`

- [ ] **Step 1: Init git** — Run: `git init && git branch -M main`. Expected: empty repo on `main`.
- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.eval_venv/
studio.db
runs/
out/
ongoing/*/
keys/
*.mp3
*.blend
```

- [ ] **Step 3: Write `pyproject.toml`** (minimal — package + pytest config + console script)

```toml
[project]
name = "manhwa-studio"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["gallery-dl", "ultralytics", "httpx", "selectolax", "Pillow"]

[project.scripts]
studio = "studio.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["requires_ultralytics: needs torch/ultralytics", "live: hits network (manual only)"]
```

- [ ] **Step 4: Write `studio/config.py`** — load `studio.toml` via `tomllib`, expose typed accessors.

```python
import tomllib
from pathlib import Path
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class SiteCfg:
    base_url: str

@dataclass(frozen=True)
class Config:
    sites: dict[str, SiteCfg]
    yolo_weights: Path
    detect_backend: str          # "yolo" | "gemini"
    gallerydl_sleep: float

def load(path: Path | None = None) -> Config:
    p = path or (REPO_ROOT / "studio.toml")
    data = tomllib.loads(p.read_text())
    sites = {k: SiteCfg(**v) for k, v in data.get("sources", {}).items()}
    d = data.get("detect", {})
    g = data.get("gallerydl", {})
    return Config(
        sites=sites,
        yolo_weights=Path(d.get("yolo_weights", "")).expanduser(),
        detect_backend=d.get("backend", "yolo"),
        gallerydl_sleep=float(g.get("sleep", 2.0)),
    )
```

- [ ] **Step 5: Write `studio.toml`** (the three confirmed sites + the trained model)

```toml
[sources.asura]
base_url = "https://asurascans.com"
[sources.webtoon]
base_url = "https://www.webtoons.com"
[sources.elftoon]
base_url = "https://elftoon.com"

[detect]
backend = "yolo"
yolo_weights = "/Users/anka/webtoon-ai/runs/detect/webtoon/yolo26_musgd_run/weights/best.pt"

[gallerydl]
sleep = 2.0
```

- [ ] **Step 6: Commit** — `git add -A && git commit -m "chore: scaffold studio package + config"`

### Task 2: Catalog models + status enum

**Files:**
- Create: `studio/catalog/__init__.py`, `studio/catalog/models.py`
- Test: `tests/catalog/test_models.py`

- [ ] **Step 1: Failing test** — assert the status order list is the canonical linear sequence and `next_status`/`fail_status` helpers behave.

```python
from studio.catalog.models import STATUS_ORDER, next_status, fail_status, Status

def test_status_order_is_linear():
    assert STATUS_ORDER == ["discovered","downloaded","stitched","detected","scened",
                            "visioned","grouped","beated","scripted","voiced","planned"]

def test_next_status_advances():
    assert next_status("downloaded") == "stitched"
    assert next_status("planned") is None  # terminal

def test_fail_status():
    assert fail_status("stitched") == "stitched_failed"
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/catalog/test_models.py -v` → ImportError.
- [ ] **Step 3: Implement `models.py`**

```python
from dataclasses import dataclass

STATUS_ORDER = ["discovered","downloaded","stitched","detected","scened",
                "visioned","grouped","beated","scripted","voiced","planned"]

def next_status(s: str) -> str | None:
    i = STATUS_ORDER.index(s)
    return STATUS_ORDER[i + 1] if i + 1 < len(STATUS_ORDER) else None

def fail_status(stage: str) -> str:
    return f"{stage}_failed"

@dataclass
class Series:
    id: int | None; source: str; series_url: str; slug: str; title: str
    added_at: str; last_checked: str | None = None; poll_priority: int = 100

@dataclass
class Chapter:
    id: int | None; series_id: int; number: float; label: str; url: str
    status: str = "discovered"; ep_dir: str | None = None
    error: str | None = None; updated_at: str = ""
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(catalog): status state machine + models"`

### Task 3: Catalog DB schema + migrations

**Files:**
- Create: `studio/catalog/db.py`
- Test: `tests/catalog/test_db.py`

- [ ] **Step 1: Failing test** — opening a fresh DB creates `series` + `chapter` tables with the UNIQUE constraints.

```python
from studio.catalog.db import connect

def test_schema_created(tmp_path):
    con = connect(tmp_path / "t.db")
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"series", "chapter"} <= tables

def test_series_unique(tmp_path):
    con = connect(tmp_path / "t.db")
    con.execute("INSERT INTO series(source,series_url,slug,title,added_at) VALUES('a','u','s','t','now')")
    import sqlite3, pytest
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO series(source,series_url,slug,title,added_at) VALUES('a','u','s2','t2','now')")
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `db.py`** — `connect(path)` opens sqlite with `PRAGMA foreign_keys=ON`, runs the `CREATE TABLE IF NOT EXISTS` DDL from spec §5 verbatim, returns the connection.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(catalog): sqlite schema"`

### Task 4: Catalog repo (pure CRUD + transitions)

**Files:**
- Create: `studio/catalog/repo.py`
- Test: `tests/catalog/test_repo.py`

- [ ] **Step 1: Failing tests** — cover the behaviours that matter (timestamps passed in, never generated inside):

```python
from studio.catalog.db import connect
from studio.catalog import repo

def test_upsert_series_idempotent(tmp_path):
    con = connect(tmp_path/"t.db")
    sid1 = repo.upsert_series(con, "asura", "url", "slug", "Title", added_at="2026-01-01T00:00:00Z")
    sid2 = repo.upsert_series(con, "asura", "url", "slug", "Title", added_at="2026-02-01T00:00:00Z")
    assert sid1 == sid2  # same (source,url) → no duplicate

def test_status_transition_and_resume(tmp_path):
    con = connect(tmp_path/"t.db")
    sid = repo.upsert_series(con, "a","u","s","t", added_at="t0")
    cid = repo.upsert_chapter(con, sid, 1.0, "Ch 1", "curl", updated_at="t0")
    repo.set_chapter_status(con, cid, "downloaded", updated_at="t1")
    assert repo.get_chapter(con, cid).status == "downloaded"
    repo.set_chapter_status(con, cid, "stitched_failed", error="boom", updated_at="t2")
    ch = repo.get_chapter(con, cid)
    assert ch.status == "stitched_failed" and ch.error == "boom"

def test_next_actionable_skips_planned(tmp_path):
    con = connect(tmp_path/"t.db")
    sid = repo.upsert_series(con,"a","u","s","t",added_at="t0")
    c1 = repo.upsert_chapter(con, sid, 1.0,"c1","u1",updated_at="t0")
    repo.set_chapter_status(con, c1, "planned", updated_at="t1")
    c2 = repo.upsert_chapter(con, sid, 2.0,"c2","u2",updated_at="t0")
    assert repo.next_actionable(con, sid).id == c2
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `repo.py`** — `upsert_series` (INSERT … ON CONFLICT(source,series_url) DO UPDATE returning id), `upsert_chapter` (ON CONFLICT(series_id,number)), `set_chapter_status(con, cid, status, *, error=None, ep_dir=None, updated_at)`, `get_chapter`, `get_series`, `list_series`, `list_chapters`, `next_actionable` (first chapter whose status != "planned" and not "*_failed" terminal, ordered by number). All timestamps are parameters; no `datetime` import in this module.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(catalog): repo CRUD + transitions"`

- [ ] **Step 6: CHUNK 1 REVIEW** — dispatch plan-document-reviewer over Chunk 1 + run `pytest tests/catalog -v` (all green) before proceeding.

---

## Chunk 2: Sources (contract + gallery-dl + adapters)

### Task 5: SourceAdapter contract + registry

**Files:**
- Create: `studio/sources/__init__.py`, `studio/sources/base.py`
- Test: `tests/sources/test_registry.py`

- [ ] **Step 1: Failing test** — a dummy adapter registers and resolves; capability flags work.

```python
from studio.sources import base

def test_register_and_get():
    @base.register
    class Dummy(base.SourceAdapter):
        id = "dummy"; capabilities = base.Capability.DOWNLOAD
        def series_meta(self, u): ...
        def list_chapters(self, u): return []
        def download(self, ch, d): return []
    assert isinstance(base.get("dummy"), Dummy)
    assert base.Capability.DOWNLOAD in base.get("dummy").capabilities
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `base.py`** — `Capability(Flag)`, `ChapterRef`, `SeriesMeta` dataclasses, `SourceAdapter(ABC)` (exact signatures from spec §3), `REGISTRY`, `register(cls)` (instantiates + stores by `id`), `get(id)`, `UnsupportedSource(Exception)`, and a `slugify(title)->str` helper (filesystem-safe).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(sources): adapter contract + registry"`

### Task 6: gallery-dl backend + normalizer

**Files:**
- Create: `studio/sources/gallerydl.py`
- Test: `tests/sources/test_gallerydl.py`

- [ ] **Step 1: Failing test** — the normalizer is the testable core (subprocess is mocked). Given a temp dir of out-of-order mixed-format images, it writes `001.jpg…` in page order.

```python
from studio.sources.gallerydl import normalize_into
from PIL import Image

def test_normalize_orders_and_converts(tmp_path):
    src = tmp_path/"raw"; src.mkdir()
    for name in ["p10.webp","p2.png","p1.jpg"]:
        Image.new("RGB",(10,10)).save(src/name)
    dest = tmp_path/"ep"
    out = normalize_into(src, dest)              # natural-sort fallback
    assert [p.name for p in out] == ["001.jpg","002.jpg","003.jpg"]
    assert all(p.suffix == ".jpg" for p in out)
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `gallerydl.py`:**
  - `gallerydl_supports(url) -> bool` — `subprocess.run(["gallery-dl","--simulate",url])`, False on "no suitable extractor".
  - `run_download(url, tmp_dir, sleep)` — invoke `gallery-dl --dest tmp --sleep <s> --write-metadata url`; raise `UnsupportedSource` on extractor error, generic on non-zero exit.
  - `normalize_into(src_dir, dest_dir)` — collect image files; order by gallery-dl metadata page index if sidecar present else natural-sort; convert each to RGB JPEG `NNN.jpg` (3-digit) in `dest_dir`; return paths. Raise on a detected index gap.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(sources): gallery-dl backend + normalizer"`

### Task 7: Webtoon adapter (happy path) — fixture-driven

**Files:**
- Create: `studio/sources/webtoon.py`, `tests/sources/fixtures/webtoon_chapters.json`
- Test: `tests/sources/test_webtoon.py`

> **DO NOT invent selectors.** Step 0 captures a real fixture; selectors are written against it. Webtoons.com is official + gallery-dl-supported, so this is the happy-path delegation exemplar.

- [ ] **Step 0: Capture fixture** — `gallery-dl --simulate -j "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154" > tests/sources/fixtures/webtoon_chapters.json` (gallery-dl's `-j` dumps structured metadata incl. the episode list). Inspect; this is the parse source.
- [ ] **Step 1: Failing test** — `WebtoonAdapter().list_chapters(url)` parses the fixture JSON into ordered `ChapterRef`s; `series_meta` returns title "Omniscient Reader" + a filesystem-safe slug.

```python
def test_webtoon_list_chapters(monkeypatch):
    # monkeypatch the gallery-dl -j call to read the fixture file
    url = "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
    chs = WebtoonAdapter().list_chapters(url)
    assert chs[0].number == 1 and chs[0].url.startswith("http")
    assert chs == sorted(chs, key=lambda c: c.number)
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `webtoon.py`** — `capabilities = DOWNLOAD|LIST_CHAPTERS|SERIES_META`; `list_chapters`/`series_meta` parse gallery-dl `-j` metadata (the webtoons extractor yields `episode_no`, `title`, `episode` fields); `download` delegates to `gallerydl.run_download` + `normalize_into`. Base URL from `config.sites["webtoon"]`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(sources): webtoon adapter"`

### Task 8: Asura adapter — fixture-driven

**Files:** `studio/sources/asura.py`, `tests/sources/fixtures/asura_*`, `tests/sources/test_asura.py`

- [ ] **Step 0: Capture fixture** — `gallery-dl --simulate -j https://asurascans.com/comics/nano-machine-5abb513e > tests/sources/fixtures/asura_chapters.json`. If gallery-dl errors with "no suitable extractor", instead `httpx GET` the series page → save to `tests/sources/fixtures/asura_series.html` and parse with `selectolax`. Inspect to confirm the chapter-list location.
- [ ] **Step 1: Failing test** — `AsuraAdapter().list_chapters(url)` returns ordered `ChapterRef`s from the fixture; `series_meta` returns title "Nano Machine" + a filesystem-safe slug.
- [ ] **Step 2: Run, verify fail** — `pytest tests/sources/test_asura.py -v`.
- [ ] **Step 3: Implement `asura.py`** — `capabilities = DOWNLOAD|LIST_CHAPTERS|SERIES_META`; parse the captured fixture format (gallery-dl `-j` metadata OR `selectolax` over the HTML); `download` delegates to `gallerydl.run_download`+`normalize_into` if supported, else native httpx fetch. Base URL from `config.sites["asura"]`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(sources): asura adapter"`

### Task 9: Elftoon adapter — fixture-driven, native download fallback

**Files:** `studio/sources/elftoon.py`, `tests/sources/fixtures/elftoon_*`, `tests/sources/test_elftoon.py`

This is the planned "unsupported-site" exemplar from spec §4. Elftoon runs a WordPress-**Madara** theme (`/manga/<slug>/` series pages + `<slug>-chapter-N` chapter pages), almost certainly not gallery-dl-supported, so it needs a native `download`. Reference series for fixtures: **Infinite Evolution From Zero** — `https://elftoon.com/manga/infinite-evolution-from-zero/` (the repo already has its chapter-1 tiles under `out/raw/infinite-evolution-from-zero-chapter-1_tiles/`).

- [ ] **Step 0: Probe + capture** — `gallerydl_supports("https://elftoon.com/manga/infinite-evolution-from-zero/")`. Record the boolean. Capture the series page (`httpx GET` → `tests/sources/fixtures/elftoon_series.html`) and one chapter page (`→ elftoon_chapter.html`) regardless, since native parse is the likely path. Note: Elftoon may 403 a default UA — set a browser `User-Agent`. Inspect the DOM (Madara: chapter list in `.wp-manga-chapter` anchors; page images in `.reading-content img`).
- [ ] **Step 1: Failing tests** — `ElftoonAdapter().list_chapters(url)` parses ordered `ChapterRef`s from `elftoon_series.html`; and (if unsupported) `download(chapter, dest)` with httpx mocked to serve `elftoon_chapter.html` extracts image URLs and writes `001.jpg…`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `elftoon.py`** — if `gallerydl_supports` returned **true**, mirror Task 7 delegation. If **false**, native `download`: httpx GET chapter page → `selectolax` extract `<img>` image URLs in reading order (against the captured fixture's real DOM) → stream each to a temp dir → `gallerydl.normalize_into` (reuse the normalizer for `NNN.jpg` + format conversion). `list_chapters` parses the series page. Base URL from `config.sites["elftoon"]`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(sources): elftoon adapter (native fallback)"`

- [ ] **Step: CHUNK 2 REVIEW** — plan-document-reviewer over Chunk 2; `pytest tests/sources -v` green.

---

## Chunk 3: YOLO detection adapter + pipeline break-fixes

### Task 10: YOLO panel adapter (drop-in for gemini_panel_boxes)

**Files:**
- Create: `studio/detect/__init__.py`, `studio/detect/yolo_panels.py`
- Test: `tests/detect/test_yolo_panels.py` (marked `@requires_ultralytics`), `tests/detect/test_box_convert.py` (pure, always runs)

- [ ] **Step 1: Failing pure test** — the pixel→`panels_norm` conversion matches the gemini schema order `[ymin,xmin,ymax,xmax]` normalized, sorted top-to-bottom.

```python
from studio.detect.yolo_panels import boxes_to_panels_norm

def test_convert_order_and_sort():
    # two boxes (x1,y1,x2,y2) in pixels on a 100x200 image, given out of order
    px = [(0,150,100,200),(0,0,100,50)]
    out = boxes_to_panels_norm(px, w=100, h=200)
    assert out == [[0.0,0.0,0.25,1.0],[0.75,0.0,1.0,1.0]]  # ymin,xmin,ymax,xmax; top first
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `boxes_to_panels_norm(px, w, h)` (pure) + `detect_panels(stitch_manifest, out_path, weights, conf=0.25)` that loads `ultralytics.YOLO`, runs per chunk, keeps class 0, calls the converter, and writes `manifest.panels.json` that is **schema-compatible** with `tools/gemini_panel_boxes.py` — same keys (`chunks[].chunk_file`, `chunks[].panels_norm`) and same `[ymin,xmin,ymax,xmax]` normalized box order, so `tools/expand_boxes_to_gutters.py` consumes it unchanged. (Test asserts keys + order, not byte-equality.) Paths resolved via the `resolve_rel` helper from Task 11.
- [ ] **Step 4: Run pure test → pass.** Then the `@requires_ultralytics` test runs `best.pt` on a committed sample chunk and asserts schema keys + box order.
- [ ] **Step 5: Commit** — `git commit -am "feat(detect): yolo panel adapter"`

### Task 11: B3 — relative-path resolution

**Files:**
- Create: `studio/paths.py` (the `resolve_rel` helper)
- Modify: `tools/chunk_stitch_adaptive.py` (store paths relative to manifest dir), `tools/expand_boxes_to_gutters.py`, `tools/panels_to_scenes.py`, `tools/panels_materialize.py` (resolve relative)
- Test: `tests/test_b3_relpaths.py`

- [ ] **Step 1: Failing test** — write a stitch manifest, move its directory, assert a consumer still resolves `chunk_path`.

```python
from studio.paths import resolve_rel
def test_resolve_rel_after_move(tmp_path):
    man = tmp_path/"a"/"manifest.stitch.json"; man.parent.mkdir(parents=True)
    # both the per-chunk path AND episode_dir must resolve relative to the manifest dir
    assert resolve_rel(man, "stitch_chunks/chunk_0001.jpg") == man.parent/"stitch_chunks/chunk_0001.jpg"
    assert resolve_rel(man, ".") == man.parent                     # episode_dir case
    assert resolve_rel(man, "/abs/legacy.jpg").as_posix() == "/abs/legacy.jpg"  # back-compat: absolute untouched
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `resolve_rel(manifest_path, stored)` → `Path(stored)` if absolute else `(Path(manifest_path).parent / stored)`. Then edit `chunk_stitch_adaptive.py` to store **both** `chunk_path` and `episode_dir` as `os.path.relpath(..., manifest_dir)` (episode_dir → `"."`); edit the four consumers (`expand_boxes_to_gutters`, `panels_to_scenes`, `panels_materialize`, and the panel detector) to wrap every `chunk_path`/`episode_dir` read in `resolve_rel(manifest_path, stored)`. Back-compat: absolute stored paths pass through unchanged (so existing manifests still load).
- [ ] **Step 4: Run, verify pass** + re-run any existing tool smoke if present.
- [ ] **Step 5: Commit** — `git commit -am "fix(B3): manifest paths relative to manifest dir"`

### Task 12: B1 — vision glob default

**Files:** Modify `tools/vision_extract.py:372`; Test: `tests/test_b1_glob.py`

- [ ] **Step 1: Failing test** — point `vision_extract`'s file discovery at a dir of `p000001.jpg…` and assert it finds them with the new default.
- [ ] **Step 2: Verify fail** (default `scene_*.jpg` finds zero).
- [ ] **Step 3:** Change default `--glob` to `"*.jpg"`. (Pipeline also passes explicit `--glob` — Task 14.)
- [ ] **Step 4: Verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "fix(B1): vision_extract default glob *.jpg"`

### Task 13: B2 — timeline keyed by segment_id (four touchpoints)

**Files:** Modify `tools/timeline_planner.py` (`_index_tts`, `index_script`, assembly loop, timeline-item emit); Test: `tests/test_b2_segment_id.py`

- [ ] **Step 1: Failing test** — synthetic `manifest.script.json` with group `g0001` having paragraphs `p00`,`p01` + matching `tts_index.json`; run timeline; assert **two** timeline items, each with distinct `segment_id` + its own `tts_audio`.
- [ ] **Step 2: Verify fail** (today: one item, last paragraph wins).
- [ ] **Step 3: Implement** per spec §8 B2: key `_index_tts`/`index_script` by `segment_id`; assembly iterates script shots (which carry `segment_id`) and looks up by it; emit `segment_id` into each timeline item; fall back to `group_id` when no `segment_id`.
- [ ] **Step 4: Verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "fix(B2): timeline aligns on segment_id, preserves multi-paragraph audio"`

- [ ] **Step: CHUNK 3 REVIEW** — plan-document-reviewer; `pytest tests/detect tests/test_b*.py -v` green.

---

## Chunk 4: Pipeline orchestration + CLI + smoke

### Task 14: pipeline.py — drive tools/ for one chapter, update catalog

**Files:**
- Create: `studio/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Failing test** — with every `tools/` stage monkeypatched to a stub that touches its output file, `run_chapter(con, chapter)` advances status `downloaded → … → planned` and is idempotent (second call re-runs nothing). A stub raising mid-way leaves status `<stage>_failed` and resumes on re-run.
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** `run_chapter(con, chapter, cfg)` — a stage table mapping each status→(callable, output path, next status). For each stage past the current status whose output is missing: run it (subprocess to the `tools/` script, or `detect.yolo_panels.detect_panels` when `cfg.detect_backend=="yolo"`), `set_chapter_status` on success, `fail_status` + error on exception. Pass explicit `--glob` to vision (belt-and-suspenders for B1). Cred-gated stages (`beated/scripted/voiced`) check env/ADC first and fail with the actionable message from spec §10.
- [ ] **Step 4: Verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(pipeline): per-chapter orchestration + catalog updates"`

### Task 15: cli.py — add-series / fetch / run / list / status

**Files:**
- Create: `studio/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Failing test** — `add-series` (adapter mocked) inserts series+chapters; `fetch --chapters 1` (download mocked to write `001.jpg`) sets status `downloaded` + `ep_dir`; `--force` re-invokes download; `status` prints the table.
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** argparse subcommands per spec §6 wiring `sources.get` + `catalog.repo` + `pipeline.run_chapter`. Timestamps generated here via `datetime.now(UTC).isoformat()` and passed into repo. `--chapters` parses `1-5`, `N`, `new`. `--force` propagates to `adapter.download`.
- [ ] **Step 4: Verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(cli): studio command surface"`

### Task 16: Live smoke test (manual, not CI)

**Files:** `tests/test_live_smoke.py` (marked `@live`, skipped by default)

- [ ] **Step 1:** For each of Asura→Nano Machine, Webtoon→Omniscient Reader (`title_no=2154`), Elftoon→Infinite Evolution From Zero (`/manga/infinite-evolution-from-zero/`): `studio add-series <source> <url>` → `studio fetch <id> --chapters 1` → assert `ongoing/<slug>/<label>/001.jpg` exists and status `downloaded`.
- [ ] **Step 2:** `studio run <id> --chapters 1` → assert status reaches at least `scened` (deterministic, no API creds) and scene JPGs exist.
- [ ] **Step 3:** Record per-site outcome (gallery-dl supported? native fallback used? chapter-count parsed?) in a new `docs/plans/results/2026-06-09-sp1-smoke-results.md` (keep the spec dir clean).
- [ ] **Step 4: Commit** — `git commit -am "test: live smoke for three sources"`

### Task 17: Docs cleanup

- [ ] Fix the spec's markdown-lint warnings (blank lines around headings/lists, code-fence languages). Add a short `studio/README.md` (install, the five commands, how to add a new source). Commit `docs: studio readme + spec lint`.

- [ ] **Step: CHUNK 4 REVIEW** — plan-document-reviewer; full `pytest -v` green (live tests skipped).

---

## Definition of Done

- `pytest -v` green (live/`requires_ultralytics` tests skipped in CI, runnable locally).
- `studio add-series asura https://asurascans.com/comics/nano-machine-5abb513e` records the series + chapters.
- `studio fetch <id> --chapters 1` writes `ongoing/<slug>/<label>/001.jpg…`, status `downloaded`.
- `studio run <id> --chapters 1` reaches `scened` with **no API auth** (YOLO path); reaches `planned` when Vertex/OpenAI/ElevenLabs creds are present.
- All three break-fix regression tests (B1/B2/B3) pass.
- Re-running any command is idempotent; failures resume from the failed stage.
