# studio

Manhwa acquisition, catalog, and pipeline front-end.

`studio` is the CLI that bridges source sites and the `tools/` pipeline.
It fetches chapters from configurable web sources, stores every series and
chapter in a local SQLite catalog, and drives each chapter through the
multi-stage production pipeline from raw images to `render.plan.json`.

---

## Install

```bash
# From repo root — uses the shared eval virtualenv
.eval_venv/bin/pip install -e .

# Verify
.eval_venv/bin/studio --help
# or
.eval_venv/bin/python -m studio --help
```

---

## CLI commands

### `add-series`

Register a new series and discover all its chapters.

```bash
studio add-series webtoon \
  "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
# series_id=1 chapters=309
```

Calls the adapter's `series_meta` + `list_chapters`, upserts the series row
and all chapter rows as `discovered`.  Re-running is idempotent (upserts, no
duplicates).

---

### `list`

List all tracked series, or the chapters of one series.

```bash
# All series
studio list
#   [   1]  webtoon       Omniscient Reader

# Chapters of series 1
studio list --series 1
#   [   1] ch   1.0  Episode 1                       discovered
#   [   2] ch   2.0  Episode 2                       discovered
#   ...
```

---

### `fetch`

Download selected chapters into `ongoing/<slug>/<label>/001.jpg …` and mark
them `downloaded`.

```bash
# Single chapter
studio fetch 1 --chapters 1

# Range
studio fetch 1 --chapters 1-5

# All not yet downloaded
studio fetch 1 --chapters new

# Re-download even if already present
studio fetch 1 --chapters 1 --force
```

The episode directory is `ongoing/<series_slug>/<chapter_label>/`.
For ORV episode 1 this is `ongoing/omniscient-reader/Episode_1/`.

---

### `run`

Drive selected chapters through the pipeline stages, resuming from the last
failed stage if applicable.

```bash
studio run 1 --chapters 1
#   Running pipeline for ch1.0 (status=downloaded) …
#     → stitched
#   Running pipeline for ch1.0 (status=stitched) …
#     → detected
#   ...
```

Stages requiring API credentials (`beated`, `scripted`, `voiced`) will fail
with an actionable message if the relevant credential is absent; earlier
outputs are preserved and the run is resumable.

---

### `status`

Show a per-chapter status table.

```bash
# All series (summary)
studio status

# Chapters of series 1
studio status 1
#     ID       #  Label                           Status                Error
# --------------------------------------------------------------------------------
#      1     1.0  Episode 1                       downloaded
#      2     2.0  Episode 2                       discovered
```

---

## Source set

| ID        | Site               | Series (confirmed live)          | Backend              |
|-----------|--------------------|----------------------------------|----------------------|
| `webtoon` | webtoons.com       | Omniscient Reader (ch 1–309+)    | gallery-dl           |
| `asura`   | asurascans.com     | Nano Machine                     | httpx + selectolax   |
| `elftoon` | elftoon.com        | Infinite Evolution From Zero     | httpx + selectolax   |

Base URLs are config, never hardcoded — edit `studio.toml` if a domain
rotates or moves:

```toml
[sources.asura]
base_url = "https://asurascans.com"
```

---

## Adding a new source

1. **Subclass `SourceAdapter`** in `studio/sources/<name>.py`:

   ```python
   from studio.sources.base import (
       Capability, ChapterRef, SeriesMeta, SourceAdapter, register, slugify
   )

   @register
   class MyAdapter(SourceAdapter):
       id = "mysite"
       capabilities = Capability.DOWNLOAD | Capability.LIST_CHAPTERS | Capability.SERIES_META

       def series_meta(self, series_url: str) -> SeriesMeta: ...
       def list_chapters(self, series_url: str) -> list[ChapterRef]: ...
       def download(self, chapter: ChapterRef, dest_dir: Path) -> list[Path]: ...
   ```

2. **Import it** in `studio/sources/__init__.py` so `@register` fires on
   package load:

   ```python
   from . import mysite  # noqa: F401
   ```

3. **Add the base URL** to `studio.toml`:

   ```toml
   [sources.mysite]
   base_url = "https://mysite.example.com"
   ```

4. **Capture a fixture** — save the series-page HTML and one chapter-page
   HTML into `tests/sources/fixtures/mysite/` for offline testing.

5. **Write a conformance test** in `tests/sources/test_mysite.py` that
   monkeypatches the HTTP layer, calls `series_meta`, `list_chapters`, and
   `download`, and asserts the returned types match the contract.

---

## Chapter status state machine

```
discovered → downloaded → stitched → detected → scened → visioned
          → grouped → beated → scripted → voiced → planned
```

| Status      | Produced by stage                           | Output artifact                    |
|-------------|---------------------------------------------|------------------------------------|
| `discovered`| `add-series`                                | catalog row only                   |
| `downloaded`| `fetch` (`adapter.download`)                | `ongoing/<slug>/<label>/NNN.jpg`   |
| `stitched`  | `chunk_stitch_adaptive.py`                  | `manifest.stitch.json`             |
| `detected`  | YOLO / `expand_boxes_to_gutters.py`         | `manifest.panels.expanded.json`    |
| `scened`    | `panels_to_scenes.py`                       | `manifest.scenes.json` + scene JPGs|
| `visioned`  | `vision_extract.py` *(GCP Vision key)*      | `manifest.vision.json`             |
| `grouped`   | `scene_group_builder.py`                    | `manifest.groups.json`             |
| `beated`    | `gemini_narrative_pass.py` *(Vertex ADC)*   | `manifest.beats.json`              |
| `scripted`  | `script_expander.py` *(OPENAI_API_KEY)*     | `manifest.script.json`             |
| `voiced`    | `elevenlabs_tts_from_manifest.py` *(ElevenLabs)* | `tts/tts_index.json` + clips  |
| `planned`   | `timeline_planner.py`                       | `render.plan.json`                 |

**Credential requirements by stage:**

- `stitched` → `grouped`: deterministic, no API credentials required.
- `visioned`: requires `GOOGLE_APPLICATION_CREDENTIALS` pointing to a GCP
  Vision service-account key (or `keys/gcp-vision.json` at repo root).
- `beated`: requires Vertex AI ADC (`gcloud auth application-default login`)
  and `GOOGLE_CLOUD_PROJECT`.
- `scripted`: requires `OPENAI_API_KEY`.
- `voiced`: requires `ELEVENLABS_API_KEY` (and optionally
  `ELEVENLABS_VOICE_ID`).

Any stage failure sets `status = "<stage>_failed"` and records the error
message.  `studio run` resumes from the failed stage without re-running
earlier stages.
