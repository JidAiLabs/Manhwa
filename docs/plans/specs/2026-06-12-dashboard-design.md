# OriginPower Studio Dashboard — design spec (2026-06-12)

Approved via mockup `mockups/dashboard-mockup.html` (all data in it real).
User decisions: full control from day one (run buttons + live logs); one
manhwa at a time on a serial GPU queue; bundles = season or full series
(intro + chapter segments + outro, concat without re-render); AniList
discovery now, competitor YouTube scan v1.1; no external compute
dependencies (pipeline is fully local: Gemma beats optional flag, local
Qwen voice, local render; thumbnails already owned per series).

## Architecture

Two processes sharing `studio.db` (single source of truth):

- **`studio dashboard`** — FastAPI + Jinja2 + htmx (no JS build). Reads
  catalog/jobs/approvals; every action handler only INSERTS rows (jobs,
  approvals) — it never executes pipeline work in-request. Live updates =
  htmx polling (2s on queue page, 10s elsewhere). Localhost only.
- **`studio worker`** — queue executor. Claims the single next runnable
  job (serial = GPU safety), runs it via `studio/pipeline.py` stage
  functions / tools, streams output to `logs/jobs/<id>.log`, records
  per-stage durations into `stage_run`, marks job done/failed.

### Gate enforcement (worker-side, never UI-side)
- `render`/`concat`/`upload` job types require a matching `approval` row.
- `render` additionally requires the chapter's latest prep-QA scan to be
  ERROR-free (the scan is itself a recorded stage with its exit code).
- Queue policy: jobs ordered by (series priority, chapter number, stage
  order); only one job runs at any time.

## Data model (additive migrations)

- `job(id, type, series_id, chapter_id, bundle_id, payload_json, state
  queued|running|done|failed|cancelled, created_at, started_at,
  finished_at, log_path, error)`  — types: `chain` (run stages up to a
  target status), `qa_scan`, `render_segment`, `concat`, `refresh`.
- `stage_run(id, chapter_id, stage, started_at, duration_sec, ok,
  meta_json)` — every stage execution ever; feeds ETAs and the chapter
  timeline. Seeded with this week's measured medians (detect 1m, OCR 2m,
  beats 33m local / 6m API, voice 20m qwen / 3m kokoro, prep 2m, QA 2m,
  render 40m).
- `approval(id, gate render|concat|upload, series_id, chapter_id,
  bundle_id, created_at, note)`.
- `bundle(id, series_id, title, kind season|full|manual, season_no,
  state collecting|ready|approved|concatenated, output_path, meta_json)`
  + `bundle_chapter(bundle_id, chapter_id, position)`.
- `chapter.season` (INTEGER NULL; from source labels / AniList; editable).
- `discovery_title(id, anilist_id, title, trend_score, chapters,
  status candidate|tracked|in_production|ignored, fetched_at, meta_json)`.

## Bundles (intro/outro correctness)

Chapter segments render **without** per-chapter branding except: the
bundle's FIRST chapter renders with intro only, the LAST with outro only
(`render_prep` gains `--branding intro|outro|both|none`; default `both`
preserves today's single-chapter behavior). `concat` job = ffmpeg
stream-copy concat of ordered segment mp4s → bundle mp4 + export folder
(mp4 + series thumbnail + `youtube_meta.json`). Projected runtime = sum
of chapter plan durations + intro/outro; shown before approval.

## Pages (as mocked)

1. **Queue** — running job (state, %, elapsed/ETA, live log tail), queue
   table (reorder, cancel), blocked rows show which gate blocks them.
2. **Series board** — per-series: chapter progress bar by status, season
   map, new-chapter badges (refresh), QA rollup, cost so far (stats.usage
   sums), projected finish (ETA model), open → chapter list.
3. **Chapter detail** — stage timeline (done durations, active progress,
   pending ETAs, lock icons), QA badges + link to prep_qa.html, approve
   render button, narration-beside-panels group gallery (reuses prep_qa
   gallery data), audio preview links, cost.
4. **Videos** — bundle list (chapters, projected runtime, segments-ready
   bar, title/hooks from youtube_meta, state), edit-cut (move boundary,
   v1 = choose season/full + trim range), approve → enqueue concat.
5. **Discovery** — AniList trending manhwa/manhua cached locally
   (httpx, graceful offline), ranked vs catalog status; `track` marks
   candidate (actual add-series remains CLI: adapters need a source URL).
   Competitor-YouTube-scan column rendered as v1.1 stub.
6. **Health** — Ollama model presence, TTS venvs + narrator ref files,
   disk free, worker liveness (heartbeat row), external spend ($0 line).

## ETA model

Median of `stage_run.duration_sec` per (series, stage), falling back to
global medians, falling back to seeds. Chapter ETA = Σ remaining stages;
series ETA = remaining chapters × median chapter wall + queue position;
bundle ETA = segments remaining × render median.

## Out of scope (v1.1+)

YouTube OAuth upload; competitor YouTube scan; dashboard-driven
add-series; auth (localhost only); tiled-YOLO controls; multi-worker.

## Testing

Pure: queue claim/transitions, gate checks (approval+QA), ETA math,
bundle planning (season cuts, first/last branding flags), AniList parser
(fixture JSON). Routes: FastAPI TestClient smoke per page + action posts.
Worker: integration with a stub stage table. Migrations: fresh + existing
DB upgrade.
