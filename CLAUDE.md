# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **manhwa/webtoon → narrated video** pipeline for the YouTube channel **OriginPower Manhwa Recap**. It fetches manhwa chapters, slices them into panels/scenes (trained YOLO), extracts OCR (Google Vision), writes narrative beats (Gemini) + a recap script (OpenAI), voices it (ElevenLabs), plans a timeline, and renders in Blender VSE. The `tools/` scripts are the pipeline stages (manifest-in → manifest-out); the **`studio/` package is the orchestrated front-end** that drives them.

> **USE `studio/` — don't run `tools/` by hand.** There's a working CLI + SQLite catalog. See "studio/ — the front-end" below. Git repo on `main`. Tests: `.eval_venv/bin/python -m pytest -q` (170 passing). Use the existing venv `.eval_venv/` (Python 3.12 + torch/ultralytics/cv2/openai/google-genai/google-cloud-vision/gallery-dl).

## studio/ — the front-end (SP1, shipped)

`studio/` adds acquisition + a SQLite catalog + per-chapter pipeline orchestration on top of `tools/`. Run everything through its CLI (auto-loads `keys/creds.env`):

```bash
V=.eval_venv/bin/python
$V -m studio add-series <asura|webtoon|elftoon> <series_url>   # discover + track all chapters
$V -m studio fetch <series_id> --chapters 1                    # download → ongoing/<slug>/<label>/NNN.jpg
$V -m studio run   <series_id> --chapters 1                    # drive the pipeline (resumable)
$V -m studio qa    <series_id> --chapters 1                    # scene↔narration QA report (HTML)
$V -m studio status [series_id]                                # chapter status table
```

- **Catalog** (`studio/catalog/`, `studio.db`): per-chapter status state machine, resumable + idempotent:
  `discovered → downloaded → stitched → detected → scened → visioned → grouped → beated → scripted → voiced → planned`
- **Sources** (`studio/sources/`): `SourceAdapter` registry; gallery-dl backend + 3 adapters (webtoon=gallery-dl happy path; asura/elftoon=native httpx+selectolax). Base URLs are config in `studio.toml` (sites churn — design is disposable).
- **Detection** (`studio/detect/yolo_panels.py`): the trained YOLO at `/Users/anka/webtoon-ai/runs/detect/webtoon/yolo26_musgd_run/weights/best.pt` replaces Gemini panel detection (drop-in `manifest.panels.json`).
- **Pipeline** (`studio/pipeline.py`): stage table mapping status→tool invocation; cred-gated stages fail-soft + resumable.

### Credentials (`keys/creds.env`, gitignored — CLI auto-loads, creds.env is authoritative)

| Stage | Needs | Notes |
|-------|-------|-------|
| `visioned` (OCR) | `keys/gcp-vision.json` | repo key; auto-set |
| `beated` (Gemini beats) | same `keys/gcp-vision.json` SA key | **no gcloud needed** — pipeline uses the SA's OWN project (`gen-lang-client-…`), not `GOOGLE_CLOUD_PROJECT` |
| `scripted` (recap script) | `OPENAI_API_KEY` | in creds.env |
| `voiced` (TTS) | depends on `[tts].backend` | **`chatterbox`/`kokoro` = local, FREE, no key** (default chatterbox). `elevenlabs` needs `ELEVENLABS_API_KEY`+`ELEVENLABS_VOICE_ID` |

### Models & cost (configurable in `studio.toml`)
- `[models].beats_model` (Gemini, default `gemini-2.5-flash`; `gemini-2.5-flash-lite` ~5× cheaper) and `[models].script_model` (OpenAI, default **`gpt-5-nano`** — note `gpt-4.1-mini` API-retires **2026-10-14**).
- `[tts].backend` = `chatterbox` (local, MIT, expressive, maps mood tags→emotion; optional `voice_ref` for cloning) | `kokoro` (local, fast, flatter) | `elevenlabs` (cloud, paid). Install local: `pip install chatterbox-tts torchaudio`. Adapter `tools/local_tts_from_manifest.py` emits the same `clips/{segment_id}.wav` + `tts_index.json` contract.
- **Every paid run prints exact tokens + $** (`[cost]` line, also in manifest `stats.usage`) via `tools/usage_cost.py`; cached tokens billed at the lower rate (OpenAI auto-caches the static prompt). Measured Nano ch1: beats ~$0.085, script(gpt-4.1-mini) ~$0.065; gpt-5-nano+flash-lite+batch target ~$0.02/chapter. **Batch API (50% off, offline bulk mode) = TODO**, see SP2 spec.

### Test titles (live-verified)
Asura→**Nano Machine** (murim), Webtoon→**Omniscient Reader** (apocalypse), Elftoon→**Infinite Evolution From Zero**.

### Current state / next work
- **Nano Machine ch1 QA scorecard is CONFIDENT** (all green): 56 shown panels (2.33/page), 0 under 3.5s, 0 visible dups, 0 OCR-echoes, 0 silent groups. QA is an automated instrument now (`studio/qa_flags.py` — scores the *rendered* montage, not raw scenes; `studio/qa.py` renders scorecard + flag badges).
- **DONE this session (SP2 + cost):** QA confidence instrument; geometric sliver-merge (proved over-seg on dense manhwa is a *selection* problem, not geometry); **Gemini scene-selection** folded into the beats call (keep/redundant + bubble_mode + intensity, ~$0 extra) → timeline drops redundant-first; beats retry overhead cut (1.67→1.17 calls/group); script per-beat narration coverage (fixes silent groups) + type-aware anti-echo (keep short direct lines/titles, rephrase dialogue/monologue); exact token+$ cost logging w/ cache visibility; per-stage model config; **free local TTS adapter (chatterbox/kokoro)**.
- **Plans:** `docs/plans/2026-06-09-acquisition-catalog-spine.md` (SP1). **SP2: `docs/plans/specs/2026-06-09-scene-bubble-quality-design.md`** — montage/over-seg + dedup + pacing DONE; **remaining: #4 bubble inpaint (white+black), #5 bubble-mode→narration (data already in beats.scene_selection), Batch API bulk mode.** Lesson: **bubble/scene *understanding* belongs in the Gemini multimodal pass, not regex/YOLO.**
- To reach a rendered video: set `[tts].backend=chatterbox` (default), `pip install chatterbox-tts torchaudio`, run `studio run <id> --chapters N` → `render.plan.json` → Blender.

---

## tools/ — the pipeline stages (driven by studio/)

## Pipeline stage order

Each stage consumes the prior stage's manifest(s). Run in this order:

1. **Scrape** — `capture_chapter.py --url <U> --name <chapter>` (Playwright/Chromium) scrolls a reader page, tiles screenshots, stitches to one PNG in `out/raw/`. *Or* skip scraping and use pre-downloaded episode JPGs under `ongoing/<Series>/<Ep…>/`.
2. **Chunk-stitch** — `tools/chunk_stitch_adaptive.py` → `stitch_chunks/chunk_*.jpg` + `manifest.stitch.json`. Cuts only at safe gutter bands (white/black/flat fades) so panels/text are never bisected; overflows until a safe cut is found.
3. **Panel detection (LLM)** — `tools/gemini_panel_boxes.py` (reads `manifest.stitch.json`) → `manifest.panels.json` (normalized `[ymin,xmin,ymax,xmax]` boxes per chunk). Then `tools/expand_boxes_to_gutters.py` → `manifest.panels.expanded.json` (snaps boxes out to gutters).
4. **Materialize scenes** — `tools/panels_to_scenes.py` (stitch + panels.expanded) → scene JPGs + `manifest.scenes.json`. Splits merged crops on internal gutters before trimming. (`panels_materialize.py` is an alternate that flattens to `panels/panel_*.jpg` + `manifest.panels_flat.json`.)
5. **Vision** — `tools/vision_extract.py` (Google Cloud Vision) → `manifest.vision.json`: OCR words/blocks with normalized bboxes, faces, objects, text coverage, camera targets. Core input for almost every downstream stage.
6. **Group into shots** — `tools/scene_group_builder.py` (reads vision manifest) → `manifest.groups.json`. Deterministic, no LLM; merges only consecutive panels.
7. **Smart crop** — `tools/smart_cropper.py --vision-manifest … --out-dir …` → candidate shot crops + `manifest.smartcrop.json` (protected-span / narration-band aware).
8. **Shot selection (LLM)** — `tools/gemini_shot_selector.py` → `manifest.smartcrop.selected.json` (keep vs redundant + per-shot micro-narrative).
9. **Narrative beats (LLM)** — `tools/gemini_narrative_pass.py` → `manifest.beats.json`.
10. **Script expansion (LLM)** — `tools/script_expander.py --beats manifest.beats.json --vision manifest.vision.json --out manifest.script.json` (OpenAI). Genre/trope inference, ElevenLabs-v3 mood tags, SFX cues. Emits deterministic `segment_id = g####_p##` that TTS and timeline must match.
11. **TTS** — `tools/elevenlabs_tts_from_manifest.py` → per-paragraph `g####_p##.mp3` + `tts_index.json`. Clip filenames are keyed by `segment_id` — keep them aligned with the script.
12. **Timeline** — `tools/timeline_planner.py --out render.plan.json` (+ beats/vision/tts). Float-accurate timing from audio; emits `cuts[]` montage plan + camera/motion per shot.
13. **Render** — `blender --background --python tools/blender_vse_from_plan.py -- <render.plan.json …>` (Blender 5.0+ VSE). Consumes `render.plan.json` exactly: `start_sec/duration_sec`, `cuts[]`, `tts_audio`, `motion`/`camera_path`.

### Standalone alternative cropper

`manhwa-cropper/` is a separate installable package (a YOLO-based bubble detector + gutter splitter), independent of the `tools/` Gemini path. Run as `python -m manhwa_cropper.cli crop --input <dir> --output <dir> --trim`. Deps in `manhwa-cropper/requirements.txt` (ultralytics, opencv). Use it when you want detection-driven cropping without LLM calls; use the `tools/` chain for the full narrated-video pipeline.

## Conventions

- **Backup files:** `*-BAK.py`, `*XXX.py`, `*X.py` (e.g. `scene_splitX.py`) are frozen snapshots. The canonical script is the plain name (`smart_cropper.py`, not `smart_cropper-BAK.py`). **Edit the plain file; never the suffixed copies.**
- **Manifests are the API.** When changing a stage's output schema, update every downstream consumer that reads that manifest. `segment_id` (`g####_p##`) must stay byte-identical across script_expander → elevenlabs → timeline_planner or audio/timeline alignment silently breaks.
- **Resume + 429 safety:** the LLM stages (`gemini_*`, `script_expander`) support `--resume` and have exponential backoff with checkpoint writes. Prefer resuming over restarting on partial runs.
- Most scripts set `ImageFile.LOAD_TRUNCATED_IMAGES = True` — webtoon downloads are often truncated; preserve this.

## Auth / credentials

- **Google Cloud Vision** (`vision_extract.py`, `vision_anchors.py`): uses `vision.ImageAnnotatorClient()` — set `GOOGLE_APPLICATION_CREDENTIALS=keys/gcp-vision.json` (service-account key already in repo at `keys/gcp-vision.json`).
- **Gemini** (`gemini_*.py`): Vertex AI via `genai.Client(vertexai=True, project=…, location=…)` — auth with `gcloud auth application-default login`; pass `--project` / `--location`. Default model `gemini-2.5-flash`.
- **OpenAI** (`script_expander.py`): `OpenAI()` reads `OPENAI_API_KEY` from env. Default model `gpt-4.1-mini`.
- **ElevenLabs** (`elevenlabs_tts_from_manifest.py`): `ELEVENLABS_API_KEY` env var. Models `eleven_v3` / `eleven_multilingual_v2`.

## Layout

- `tools/` — the stage scripts (the pipeline). Also has `debug_clean/`, `shots_smart/` scratch output dirs.
- `manhwa-cropper/` — standalone YOLO cropper package.
- `ongoing/<Series>/<Ep…>/` — source episode page images (numbered JPGs).
- `out/raw/` — `capture_chapter.py` scrape output.
- `keys/` — GCP service-account credentials.
