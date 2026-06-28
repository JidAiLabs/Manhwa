# Teaser Planner — Implementation Spec

- **Date:** 2026-06-28
- **Status:** Approved design; ready for implementation plan
- **Design notes:** [`2026-06-26-arc-teaser-sequencing-design.md`](2026-06-26-arc-teaser-sequencing-design.md) (discovery),
  [`2026-06-26-reference-channel-gemma-baseline-design.md`](2026-06-26-reference-channel-gemma-baseline-design.md) (per-chapter contract)
- **Scope:** the arc teaser only. Flashback/memory inserts, the dashboard auto-batcher, and
  tail-handling are explicitly deferred (operator picks the bundle range manually via
  `bundles.create_bundle`).

## Goal

A published multi-chapter bundle should open with an **arc-level cold open** — a short
high-stakes window selected from *anywhere inside the selected bundle* — then bridge into the
chronological chapter bodies, matching the reference channel. The window is **discovered per
manhwa by score**, never a hard-coded chapter offset (Nano's ch5-8 was only an example; another
series scores 3-5, another 8-10).

This is the **sole intro mechanism**. The chapter-1 opening hook was removed (commit `e6d8951`),
so there is no compose/gate tension: a teaser is the bundle's cold open, period.

## Non-Goals

- No per-series chapter-number rules. No fixed word/second budget on the teaser narration
  (cost guards only). No re-rendering of chapter bodies. No new render or TTS code — the teaser
  reuses the existing per-chapter render path. No flashback inserts, auto-batcher, or
  tail-handling in this slice.

## Architecture

The teaser is a **synthetic episode**: a working dir that looks enough like an `ongoing/<series>/<ep>`
dir that the existing `scripted → voiced → planned → render_segment` stages run on it unchanged.
`teaser_planner.py` only does the novel work — **select the window** and **write the narration
manifests** into that dir. Everything downstream is reuse.

```text
selected bundle (chapters already prepared: understood/story/beats/scenes on disk)
  → tools/teaser_planner.py  (NEW)
      Stage 1  deterministic window scorer over cached understood.json  → top-N windows
      Stage 2  ONE model call: pick winner + write teaser panel narration + rewind_line
      writes:  dist/bundle_<id>/teaser/manifest.beats.json   (panel_narration, story-shaped)
               dist/bundle_<id>/teaser/manifest.cast.json    (merged from source chapters)
               dist/bundle_<id>/teaser/scenes/*.jpg          (symlinks to source scene files)
               dist/bundle_<id>/teaser/manifest.teaser.json  (the plan record, per design note)
  → existing stages on the teaser dir: scripted → voiced → planned → render_segment
      → dist/bundle_<id>/teaser/render/segment_none.mp4  → copied to dist/bundle_<id>/teaser.mp4
  → _h_concat prepends teaser.mp4 to the chapter segments
```

### Why synthetic-episode reuse
`_h_render_segment` already renders any dir's `render.plan.clean.json` to `segment_*.mp4` via
remotion; `script_expander`/`local_tts_from_manifest`/`timeline_planner` already turn
`manifest.beats.json` (+ scenes + cast) into a script, clips, and a plan. The teaser is just a
~6-12 panel "chapter" whose panels happen to come from several source chapters. Building the
beats manifest + symlinking the scenes is the entire integration; no stage is forked.

## Inputs — all cached, nothing recomputed

For each chapter in the bundle (`bundles.bundle_chapters`), read from its `ep_dir`:

- `manifest.panels.understood.json` — per-panel `description/action/setting/dialogue/panel_kind/intensity/subjects`
- `manifest.story.json` — `logline/premise/arc` (chapter through-line; **no `hook` field** since `e6d8951`)
- `manifest.beats.json` — per-panel grounded lines (optional reuse for the winning window)
- `manifest.cast.json` — for name consistency in teaser narration
- the scene image files referenced by the above

If a chapter is missing `understood.json`, it is **skipped for scoring** with a logged warning
(it can't contribute a window), but it still appears in the body. If *no* chapter in the bundle
has `understood.json`, the planner exits cleanly with "no teaser" (the bundle concats without one).

## tools/teaser_planner.py

Pure, testable functions + a thin `main()`. Module boundaries:

### Stage 1 — `score_windows(panels, *, cfg) -> list[Window]`  (pure, deterministic, $0)

`panels` is the bundle's flattened panel sequence in reading order, each carrying its source
`chapter_id`, `chapter_number`, `scene_file` (abs path), and its `understood` fields.

- **Window** = a contiguous run of `cfg.min_panels..cfg.max_hook_panels` panels (default 4-10),
  allowed to cross a chapter boundary.
- **Signals** (computed from `understood` text via regex/keyword sets, reusing the
  `recap_style` deterministic-signal pattern — keyword sets live in the module, no per-series config):
  - high-stakes rules: exam/test/rank/survival/execution/expulsion/contract
  - social pressure: humiliation/missing token-badge/clan-family rejection
  - power/status reveal: system window/rank jump/hidden skill/impossible result
  - antagonist pressure: named enemy/elder/clan heir/authority
  - visual variety: mix of `panel_kind` + a spread of `intensity` across the window
  - story clarity: penalize windows with too many distinct `subjects` (hard to grok in <20s)
  - intensity peak: max `intensity` in the window
- **Score** = weighted sum (weights are module constants, tunable; documented as the one
  calibration knob). Returns the top `cfg.shortlist_n` (default 4) non-overlapping windows.
- **Spoiler guard (deterministic, before scoring):** exclude any window that
  (a) overlaps the **last `cfg.payoff_tail_frac` (default 0.20)** of the bundle's panel
  sequence, or (b) contains a panel whose `understood` marks the bundle's single
  highest-intensity reveal. This keeps the teaser from spoiling the payoff.
- **Cost guards (caps, never creative rules):** `max_hook_scan_chapters` (limit how deep into
  the bundle we scan), `max_hook_panels`, `shortlist_n`, `max_teaser_seconds`. All in
  `studio.toml [teaser]`.

### Stage 2 — `select_and_write(windows, *, cfg, model_call) -> dict`  (ONE model call)

- Hand the model **only the shortlisted windows'** `understood` text (description/action/dialogue/
  subjects per panel) — not the whole bundle — plus the bundle's loglines for context.
- The model: picks the strongest window, writes **per-panel teaser narration** (rolling, under
  the **6 `recap_style` rules**), writes a one-line `rewind_line` bridging into the chronological
  body, and returns `reason` + `spoiler_boundary`. Prompt forbids naming any identity the source
  hasn't yet revealed and forbids referencing events past the window.
- The teaser's first line is a **strong cold-open hook by selection + framing** — uncapped, paced
  to content. It does **not** use the deleted `OPENING_HOOK_RULE` word window.
- Backend: the same `_call_model_with_backoff` infra `story_group`/`gemini_narrative_pass` use
  (ollama Gemma or Vertex), chosen by the bundle's series config.
- **Deterministic post-pass:** run `recap_style.neutralize_identity_reveal_leaks` and
  `recap_style.repair_spoken_fragments` over the teaser narration (spoiler + fragment safety),
  exactly as the chapter beated stage does.

### `manifest.teaser.json` schema (the plan record, per the design note)

```json
{
  "bundle_id": 12,
  "source_chapters": [5, 6, 7, 8],
  "window": {"chapter_panels": [{"chapter_number": 5, "scene_files": ["..."]}, ...]},
  "scene_files": ["<abs path in teaser order>", ...],
  "reason": "public test + missing badge + authority pressure",
  "rewind_line": "But to see how he ended up here, we have to go back to the day it started.",
  "spoiler_boundary": "may show the entrance test; must not reveal the seventh prince's identity",
  "panel_narration": [{"scene_file": "...", "line": "..."}, ...],
  "scores": {"chosen": 0.82, "shortlist": [0.82, 0.71, 0.69, 0.64]}
}
```

### `main()`
Args: `--bundle-id`, `--series-dir`, `--out-dir` (`dist/bundle_<id>/teaser`), `--backend`/`--model`/
`--project`/`--location` (mirroring `gemini_narrative_pass`). Reads the cached inputs, runs
Stage 1 → Stage 2, writes `manifest.teaser.json` + a `manifest.beats.json` (panel_narration
shape the scripted stage consumes) + merged `manifest.cast.json`, and symlinks the winning
`scene_files` into `teaser/scenes/`. Exits "no teaser" (rc 0, writes nothing prependable) when
the bundle is single-chapter or no `understood.json` exists.

## Stage 3 — materialize the segment (reuse, no new code)

The worker runs the existing stages against the teaser dir:
1. `script_expander.py` (gemini_verbatim) → `teaser/manifest.script.json`
2. `local_tts_from_manifest.py` → `teaser/tts/` clips (qwen-mlx, same as chapters)
3. `timeline_planner.py` / `render_prep` → `teaser/render.plan.json` + `render.plan.clean.json`
   (cuts reference the symlinked `teaser/scenes/*` exactly like a chapter)
4. `npx remotion render ... --props=teaser/render.plan.clean.json` → `teaser/render/segment_none.mp4`,
   copied to `dist/bundle_<id>/teaser.mp4`

No branding wrap on the teaser (it is itself the opening). The existing render seats every shot
at its absolute `start_sec`, so the teaser segment is A/V-aligned like any chapter (no drift).

## Integration

### Worker (`studio/worker.py`)
- New handler **`_h_teaser`** (job kind `"teaser"`): runs `teaser_planner.py`, then drives the
  scripted→voiced→planned→render_segment stages on the teaser dir, then writes
  `dist/bundle_<id>/teaser.mp4`. Records a `stage_run` like other jobs. Gated by bundle approval.
- **`_h_concat`** change: if `dist/bundle_<id>/teaser.mp4` exists **and** the teaser is approved,
  prepend it to `segs` before `wrap_with_branding`. One added line in the segment assembly; the
  ffmpeg stream-copy concat is format-compatible because the teaser is rendered by the same
  remotion composition as the chapter segments.

### Gates (`studio/dashboard/gates.py`)
- `concat_allowed` additionally requires: teaser **approved** OR teaser explicitly **declined**
  ("no teaser for this bundle"). A planned-but-unreviewed teaser blocks concat, mirroring the
  existing per-chapter approval discipline.

### Dashboard (`studio/dashboard/`)
- Bundle page: **"Plan teaser"** button → enqueues the `teaser` job. A **review card** shows the
  chosen window thumbnails, the teaser narration, `reason`, and `spoiler_boundary`, with
  **Approve** / **Decline** / **Re-plan** actions. (Worker/dashboard change → **daemon restart**
  on deploy, per the standing rule.)

### Config (`studio.toml [teaser]`)
```toml
[teaser]
enabled = true
shortlist_n = 4
min_panels = 4
max_hook_panels = 10
max_hook_scan_chapters = 12   # cost guard only
max_teaser_seconds = 90       # cost guard only
payoff_tail_frac = 0.20       # spoiler guard: never pull from the last 20%
```
Mirrored as fields in `studio/config.py` with `STUDIO_TEASER_*` env overrides, following the
existing config pattern.

### DB
The existing `bundle` / `bundle_chapter` / `approval` (with `bundle_id`) tables suffice. Teaser
approval is an `approval` row scoped to the bundle with a `kind="teaser"` marker (or a
`bundle.teaser_state` column if cleaner — decide in the plan). No schema redesign.

## Determinism + Tests

- `score_windows` is **pure** → unit tests: a high-stakes window outranks a calm one; a
  payoff-tail window is excluded; a >`max_hook_panels` window is never returned; crossing a
  chapter boundary is allowed; missing-`understood` chapters are skipped.
- `select_and_write` with a **stubbed `model_call`** (the `story_group` test pattern) → asserts
  `manifest.teaser.json` shape, that `panel_narration` covers the window, that the spoiler
  post-pass ran, and that no shortlisted-but-unchosen window leaks into the output.
- A small test that `_h_concat` prepends `teaser.mp4` when present+approved and does not when
  absent/declined (probe-injected paths, like `segments_ready`).
- Single-chapter bundle → planner emits "no teaser"; concat unchanged.

## Acceptance / Validation (per the design note's Nano proof)

1. On a Nano `1..10` bundle, the scorer surfaces the academy/test window **by score** (expected
   to land in the ch5-8 region) with **no chapter numbers in the code**.
2. `manifest.teaser.json` is spoiler-safe: no identity named that the source hasn't revealed; no
   panel from the last 20% of the bundle.
3. The rendered `teaser.mp4` concatenates cleanly (stream-copy, no re-encode) ahead of the
   chapter bodies; the final video opens on the hook, then `rewind_line` bridges to chapter 1.
4. Bodies remain panel-aligned and lose no panels (the teaser adds, never mutates chapters).
5. Human watches the opening: immediate rules/stakes + rapid visual movement, closer to the
   reference channel; the teaser doesn't ruin the payoff.

## Deferred (named so coverage isn't silently bounded)

- Flashback / story-memory inserts (`story_memory.index.json` + body memory_insert cuts).
- Dashboard auto-batcher (suggest next range) + tail-handling policy.
- Future-teasing opt-in (teaser windows outside the selected bundle).
