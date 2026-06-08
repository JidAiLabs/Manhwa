# Sub-Project 1 — Acquisition + Catalog Spine (+ Pipeline Unblock)

**Date:** 2026-06-09
**Status:** Draft for review
**Owner:** anka

---

## 1. Purpose & Scope

The existing `tools/` pipeline turns webtoon page images into a narrated video, but it has **no front-end**: nothing fetches chapters, nothing remembers what has been processed, and three wiring breaks stop a chapter from reaching `render.plan.json`. This sub-project builds the **spine** that makes a single, repeatable end-to-end run possible:

> *"Add series X from site Y → fetch chapter N into the repo's episode layout → run it through detection to scenes → (with API creds present) on to `render.plan.json` → record every step in a catalog so nothing is ever fetched or processed twice."*

### In scope

1. A **`SourceAdapter`** plugin contract + a registry.
2. A **gallery-dl download backend** that normalizes output into the existing `ongoing/<Series>/<Chapter>/NNN.jpg` episode layout.
3. **Three concrete adapters**: Asura Scans, Webtoon (official), Elftoon (config-driven base URLs). The set deliberately spans an aggregator (Asura), an official platform (Webtoon), and a niche aggregator likely needing a native extractor (Elftoon).
4. A **SQLite catalog** (`studio.db`) tracking series and chapters with a status state machine, using a *perpetually-ongoing* model (no "completed" state; decaying poll cadence field only — polling itself is a later sub-project).
5. A **manual CLI** (`studio`) with `add-series`, `list`, `fetch`, `run`, `status`.
6. A **YOLO detection adapter** that runs the user's trained `best.pt` to produce `manifest.panels.json`, a drop-in replacement for `tools/gemini_panel_boxes.py` (removes the Vertex-auth dependency from the critical path).
7. **Three pipeline break-fixes** (B1, B2, B3 below) so a fetched chapter flows to `render.plan.json`.

### Explicitly OUT of scope (future sub-projects)

- **Discovery / trending** (source-site trending, YouTube-trend reverse lookup) — Sub-Project 2.
- **Scheduler / auto-polling** of ongoing series — Sub-Project 3.
- **Narrative-quality overhaul** (R2 rephrase, R3 continuity, R4 jargon, flashbacks) — separate sub-project.
- **R1 text-suppression** using YOLO's `speech_bubble/text/sfx/system_box` classes — noted as a future extension; SP1's YOLO adapter emits panels only.
- Replacing the LLM stages (`gemini_narrative_pass`, `script_expander`, `elevenlabs`) — they remain as-is; they require their own API credentials, which are configuration, not SP1 deliverables.

---

## 2. Module Layout

A new top-level package `studio/` holds all new code; the existing `tools/` scripts are reused unchanged except for the three break-fixes.

```text
studio/
  __init__.py
  cli.py                 # `studio` entry point (argparse subcommands)
  config.py              # loads studio.toml (sites, paths, cadence defaults)
  catalog/
    __init__.py
    db.py                # SQLite connection + migrations
    models.py            # Series, Chapter dataclasses + status enum
    repo.py              # CRUD + state-transition functions
  sources/
    __init__.py
    base.py              # SourceAdapter ABC + Capability flags + registry
    gallerydl.py         # gallery-dl subprocess backend + output normalizer
    asura.py             # AsuraAdapter
    flame.py             # FlameAdapter
    bato.py              # BatoAdapter
  detect/
    __init__.py
    yolo_panels.py       # best.pt -> manifest.panels.json (drop-in for gemini_panel_boxes)
  pipeline.py            # orchestrates tools/ stages for one chapter; updates catalog
studio.toml              # user config (site base URLs, model path, api-cred presence)
tests/
  ...                    # see §9
```

The episode image layout consumed by the pipeline is unchanged: `ongoing/<Series_slug>/<Chapter_label>/001.jpg, 002.jpg, …`.

---

## 3. SourceAdapter Contract

`studio/sources/base.py`. Adapters are small and declare what they can do via capability flags; the core degrades gracefully when a capability is absent.

```python
class Capability(Flag):
    DOWNLOAD      = auto()   # can fetch a chapter's images
    LIST_CHAPTERS = auto()   # can enumerate chapters for a series
    SERIES_META   = auto()   # can return title/status/cover for a series

@dataclass(frozen=True)
class ChapterRef:
    number: float            # 1, 2, 10.5 …
    label: str               # "Ep. 30 - Silent Ground (1)" or "Chapter 315"
    url: str                 # canonical chapter URL on the source

@dataclass(frozen=True)
class SeriesMeta:
    source: str              # adapter id, e.g. "asura"
    series_url: str
    title: str
    slug: str                # filesystem-safe; derived from title

class SourceAdapter(ABC):
    id: str                       # "asura" | "flame" | "bato"
    capabilities: Capability

    @abstractmethod
    def series_meta(self, series_url: str) -> SeriesMeta: ...

    @abstractmethod
    def list_chapters(self, series_url: str) -> list[ChapterRef]: ...

    @abstractmethod
    def download(self, chapter: ChapterRef, dest_dir: Path) -> list[Path]:
        """Download chapter images into dest_dir as 001.jpg, 002.jpg …
        (zero-padded, reading order). Returns the written paths.
        MUST be idempotent: if dest_dir already holds a complete set, return it
        without re-downloading."""
```

**Registry:** `register(adapter_cls)` populates `REGISTRY: dict[str, SourceAdapter]`; `get(source_id)` resolves it. Adapters self-register on import; `sources/__init__.py` imports all three.

**Base URL is config, never hardcoded.** Each adapter reads its current base URL from `studio.toml` (e.g. `[sources.asura] base_url = "https://asurascans.com"`), so domain rotation (Asura) or a site dying (Reaper) is a config edit, not a code change.

---

## 4. gallery-dl Backend

`studio/sources/gallerydl.py`. gallery-dl is the universal download engine; adapters delegate `download()` to it where the site is supported, and only `list_chapters`/`series_meta` are written natively.

- **Invocation:** subprocess `gallery-dl --dest <tmp> --write-metadata <chapter_url>` with a per-run config (rate limit, retries, user agent) written to a temp config file. Captured exit code + stderr.
- **Support probe:** `gallerydl_supports(url) -> bool` runs `gallery-dl --simulate <url>` once; if it errors with "no suitable extractor", the adapter raises `UnsupportedSource` (caught by CLI → printed as an actionable message; that site becomes the "write a custom extractor" task).
- **Normalizer:** gallery-dl's output filenames vary per extractor. `normalize_into(tmp_dir, dest_dir)` sorts the downloaded image files by gallery-dl's page index (from the metadata sidecar, falling back to natural-sort of filenames) and copies them to `dest_dir/NNN.jpg` (3-digit zero-pad, converting webp/png → jpg via Pillow for downstream consistency, since the pipeline globs `*.jpg`).
- **Rate limiting / politeness:** a fixed `--sleep` between requests (default 2s, configurable). The backend never parallelizes requests to one host.

**Per-site reality (to verify during implementation as the first concrete task):**
- Webtoon (`webtoons.com`) — official platform, well-supported by gallery-dl (`webtoons` extractor); primary happy path. Stable HTML + a real popular/trending section (useful later for SP2).
- Asura Scans — supported historically; verify against current `asurascans.com` (domain rotates).
- Elftoon (`elftoon.com`) — niche aggregator, almost certainly **not** gallery-dl-supported → `ElftoonAdapter` implements a native `download()` (httpx + parse the chapter's image list) instead of delegating. This is the deliberate "unsupported site" exemplar; the contract keeps the native code local to one file.

> **Source churn is expected, not exceptional.** During design alone, Reaper Scans shut down (Kakao C&D), Bato.to's primary domain went offline, and Flame Comics was ruled out. This is exactly why base URLs are config and adapters are disposable — swapping a source is a `studio.toml` edit plus at most one new adapter file.

---

## 5. Catalog (SQLite)

`studio/catalog/`. Single file `studio.db` at repo root. Perpetually-ongoing model: there is **no terminal "completed" state**; a `poll_priority` integer only influences *future* scheduler cadence (SP3) and defaults to a fixed value here.

### Schema

```sql
CREATE TABLE series (
  id            INTEGER PRIMARY KEY,
  source        TEXT NOT NULL,            -- adapter id
  series_url    TEXT NOT NULL,
  slug          TEXT NOT NULL,            -- filesystem dir name
  title         TEXT NOT NULL,
  added_at      TEXT NOT NULL,            -- ISO8601 (passed in by caller; see note)
  last_checked  TEXT,                     -- ISO8601 or NULL
  poll_priority INTEGER NOT NULL DEFAULT 100,
  UNIQUE(source, series_url)
);

CREATE TABLE chapter (
  id            INTEGER PRIMARY KEY,
  series_id     INTEGER NOT NULL REFERENCES series(id),
  number        REAL NOT NULL,
  label         TEXT NOT NULL,
  url           TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'discovered',
  ep_dir        TEXT,                     -- repo-relative path once downloaded
  error         TEXT,                     -- last error message if status endswith _failed
  updated_at    TEXT NOT NULL,
  UNIQUE(series_id, number)
);
```

> **Timestamps:** the runtime forbids non-deterministic clock calls inside some contexts; all timestamps are produced by the CLI process via `datetime.now(UTC)` at the call site and passed into `repo.py` functions, never generated deep in the data layer. This keeps `repo.py` pure and testable.

### Chapter status state machine

Status is a **single linear sequence** mapped 1:1 to the critical-path pipeline stages (the orphaned `smart_cropper`/`gemini_shot_selector` branch from the audit is **not** in the critical path and has no status). Ordering is canonical so `pipeline.py`'s resume logic is unambiguous:

```text
discovered → downloaded → stitched → detected → scened → visioned
           → grouped → beated → scripted → voiced → planned
```

| status | produced by stage | output |
|--------|-------------------|--------|
| discovered | `add-series` | catalog row only |
| downloaded | `fetch` (adapter.download) | `ongoing/<slug>/<label>/NNN.jpg` |
| stitched | `chunk_stitch_adaptive.py` | `manifest.stitch.json` |
| detected | YOLO/gemini panel detect + `expand_boxes_to_gutters.py` | `manifest.panels.expanded.json` |
| scened | `panels_to_scenes.py` | scene JPGs + `manifest.scenes.json` |
| visioned | `vision_extract.py` | `manifest.vision.json` |
| grouped | `scene_group_builder.py` | `manifest.groups.json` |
| beated | `gemini_narrative_pass.py` *(Vertex creds)* | `manifest.beats.json` |
| scripted | `script_expander.py` *(OpenAI creds)* | `manifest.script.json` |
| voiced | `elevenlabs_tts_from_manifest.py` *(ElevenLabs creds)* | `tts_index.json` + clips |
| planned | `timeline_planner.py` | `render.plan.json` (terminal success) |

- Each successful stage advances `status` to the next value.
- Any stage failure sets `status = "<stage>_failed"` and records `error`; the row is **resumable** — `studio run` restarts from the failed stage, not from scratch.
- `planned` is the terminal success state. Blender render itself is a manual/optional follow step, not tracked here.
- Stages marked *(…creds)* are gated on the relevant API credential being present; absent creds fail that stage with an actionable message (§10) and leave earlier outputs intact.
- **Idempotency:** `fetch` on a chapter already `downloaded`+ is a no-op (unless `--force`, which propagates to `adapter.download` and overrides its internal idempotency to force a re-fetch); `run` skips stages whose outputs already exist on disk and whose status is already past them.

`repo.py` exposes pure functions: `upsert_series`, `upsert_chapters`, `set_chapter_status`, `next_actionable(series_id)`, `get_series`, `list_series`, `list_chapters`. No business logic in SQL beyond constraints.

---

## 6. CLI

`studio/cli.py`, entry point `studio` (run as `python -m studio` or via console script).

| Command | Behaviour |
|---------|-----------|
| `studio add-series <source> <series_url>` | Resolve `series_meta` + `list_chapters` via the adapter; upsert series + chapters as `discovered`. Prints series id + chapter count. |
| `studio list [--series ID]` | Show tracked series, or chapters of one series with their statuses. |
| `studio fetch <series_id> --chapters 1-5\|N\|new` | For each selected `discovered` chapter: create `ep_dir = ongoing/<slug>/<label>/`, call `adapter.download`, set status `downloaded`. `new` = chapters not yet `downloaded`. |
| `studio run <series_id> --chapters …` | For each chapter ≥`downloaded`: drive `pipeline.py` through the stages, advancing status; honour resume + idempotency. |
| `studio status [<series_id>]` | Summary table: per chapter, current status + any error. |

Errors surface as readable messages (`UnsupportedSource`, network failure, missing model, missing API creds) and set the appropriate `*_failed` status — never a silent pass.

---

## 7. YOLO Detection Adapter

`studio/detect/yolo_panels.py`. Replaces `tools/gemini_panel_boxes.py` in the critical path so the run needs no Vertex auth.

- **Input:** `manifest.stitch.json` (from `chunk_stitch_adaptive.py`) — list of stitched chunk images.
- **Model:** path from `studio.toml` (`[detect] yolo_weights = ".../best.pt"`); loaded once via `ultralytics.YOLO`. Device auto-selects `mps`→`cpu`.
- **Per chunk:** run inference, keep **class 0 (`panel`)** boxes at configurable `conf` (default 0.25). Convert each pixel box `(x1,y1,x2,y2)` to the schema `tools/expand_boxes_to_gutters.py` already consumes: `panels_norm = [ymin, xmin, ymax, xmax]` normalized to the chunk's W/H. Sort top-to-bottom.
- **Output:** `manifest.panels.json` — **byte-compatible** with the gemini producer's schema (`chunks[].chunk_file`, `chunks[].panels_norm`), so `expand_boxes_to_gutters.py` → `panels_to_scenes.py` run unchanged.
- The auxiliary classes (`speech_bubble/text/sfx/system_box/character`) are **not emitted in SP1** (YAGNI — no consumer yet). A one-line comment marks where a future R1/anchor sub-project would add a `manifest.detect.json`.

Selection between detectors is a config switch (`[detect] backend = "yolo" | "gemini"`); `pipeline.py` calls the chosen one. Default `yolo`.

---

## 8. Pipeline Break-Fixes

`studio/pipeline.py` drives the existing `tools/` chain for one chapter, but three breaks (found in the prior audit) must be fixed in the `tools/` scripts themselves so the chain completes:

**B1 — vision glob mismatch.** `tools/vision_extract.py` defaults `--glob "scene_*.jpg"` but `panels_to_scenes.py` writes `p000001.jpg`, so Vision finds zero files and exits.
*Fix:* change the `vision_extract.py` default to `"*.jpg"`, AND have `pipeline.py` pass an explicit `--glob` matching the scenes dir. (Two-layer: robust default + explicit call.)

**B2 — timeline TTS/script keyed by group_id, dropping multi-paragraph audio.** `tools/timeline_planner.py` `_index_tts` and `index_script` key their maps by `group_id` only, overwriting so only the last paragraph of a multi-paragraph group survives into `render.plan.json`.
*Fix — wider than the two index functions (four touchpoints):*
1. `_index_tts` and `index_script`: key by `segment_id` (`g####_p##`) instead of `group_id`.
2. The timeline **assembly loop** (currently `tts_by_gid.get(group_id)` / `script_by_gid.get(group_id)`): obtain a `segment_id` per shot and look up by it. The shot's `segment_id` comes from the shots in `manifest.script.json` (which `script_expander.py` already stamps); the groups manifest only carries `group_id`, so assembly must join script-shots (which have `segment_id`) rather than iterating group entries alone.
3. **Emit `segment_id` into each `render.plan.json` timeline item** (today they carry only `group_id`/`shot_id`) — required both for correctness and so the B2 regression test is assertable.
4. Fallback: when a shot genuinely has no `segment_id`, fall back to `group_id` keying (preserves single-paragraph-group behaviour).

**B3 — absolute, stale paths baked into manifests.** `manifest.stitch.json` stores absolute `chunk_path`/`episode_dir`; if the repo moves, downstream `open(chunk_path)` fails.
*Fix:* store paths **relative to the manifest's own directory** in `chunk_stitch_adaptive.py`, and resolve them relative to the manifest location in every consumer (`gemini_panel_boxes`/`yolo_panels`, `expand_boxes_to_gutters`, `panels_to_scenes`, `panels_materialize`). A small `resolve_rel(manifest_path, stored)` helper is added and used uniformly.

Each fix ships with a regression check (§9).

---

## 9. Testing

- **Catalog unit tests** (no I/O beyond a temp SQLite): every status transition, idempotent `fetch`/`run` no-ops, resume-from-`*_failed`, `UNIQUE` constraint behaviour, `next_actionable`.
- **Adapter contract conformance:** a parametrized test runs each adapter against a **recorded fixture** (saved HTML/JSON for one series + one chapter; gallery-dl invocation mocked) asserting `series_meta`, `list_chapters`, and that `download` writes `001.jpg…` in order. No live network in CI.
- **Normalizer tests:** mixed webp/png/jpg input with out-of-order filenames → correct `NNN.jpg` ordering and format conversion.
- **YOLO adapter test:** run `best.pt` on 1–2 committed sample chunks; assert `manifest.panels.json` matches the gemini schema keys and that boxes are normalized in `[ymin,xmin,ymax,xmax]` order and sorted. (Marked `@requires_ultralytics`; skipped if the dep is absent.)
- **Break-fix regression tests:**
  - B1: vision stage over a `p*.jpg` scenes dir finds all files.
  - B2: a synthetic `manifest.script.json` with a 2-paragraph group (`g0001_p00`, `g0001_p01`) + matching `tts_index.json` yields **2 distinct timeline items in `render.plan.json`, each carrying its own `segment_id` and `tts_audio`** (asserts the emit-`segment_id` deliverable, not just the re-keying).
  - B3: move the manifest dir, assert all consumers still resolve chunk paths.
- **One live smoke test (manual, not CI):** `studio add-series` + `fetch --chapters 1` + `run` for each of the three confirmed sources (Asura→Nano Machine, Webtoon→Omniscient Reader, Elftoon→Infinite Evolution From Zero), asserting a chapter reaches at least `scened`.

---

## 10. Error Handling & Edge Cases

- **Unsupported site:** `UnsupportedSource` → CLI prints which site + suggests writing a native adapter; chapter left `discovered`.
- **Partial download:** normalizer detects a gap in page indices → raises; status `downloaded`-fails; re-`fetch` resumes (gallery-dl skips existing files).
- **Domain moved / 404 / rate-limited (HTTP 429):** adapter retries with backoff (bounded), then fails the chapter with the HTTP context in `error`.
- **Missing YOLO weights / API creds:** detected before the relevant stage runs; the affected stage fails with an actionable message naming the specific missing credential — `[detect].yolo_weights` (YOLO), Vertex ADC + project (beats), `OPENAI_API_KEY` (script), `ELEVENLABS_API_KEY` (tts) — not a stack trace. The deterministic stages (`downloaded`→`visioned`) never require API creds.
- **Duplicate add-series:** `UNIQUE(source, series_url)` → upsert refreshes chapter list rather than erroring.
- **Legal/politeness:** fixed inter-request sleep and single-host serialization are non-optional defaults; the operator owns the decision to point adapters at a given site.

---

## 11. Build Sequence (for the implementation plan)

1. Verify gallery-dl support for the three sites (`--simulate`); record which need native `download`.
2. Catalog (`db`, `models`, `repo`) + tests.
3. `SourceAdapter` base + registry + gallery-dl backend + normalizer + tests.
4. Three adapters + recorded fixtures + conformance tests.
5. YOLO detection adapter + schema-compat test.
6. The three break-fixes + regression tests.
7. `pipeline.py` orchestration + `cli.py`.
8. Manual live smoke test on the three titles.

---

## 12. Success Criteria

- `studio add-series asura <nano-machine-url>` records the series and its chapters.
- `studio fetch <id> --chapters 1` produces `ongoing/<slug>/<label>/001.jpg…` and marks the chapter `downloaded`.
- `studio run <id> --chapters 1` advances the chapter through `detected → scened → visioned` deterministically with **no Vertex auth** (YOLO path), and — when `OPENAI_API_KEY`/Vertex/ElevenLabs creds are present — on to `planned` (`render.plan.json`).
- Re-running any command is idempotent; a failure is resumable from the failed stage.
- All three break-fix regression tests pass.
