# Sub-Project 2 — Scene & Bubble Quality

**Date:** 2026-06-09
**Status:** Draft (captures QA findings from the first real ORV + Nano Machine runs)
**Depends on:** SP1 (working pipeline) — already shipped.

## Purpose

The SP1 pipeline produces a working recap (fetch → YOLO → scenes → OCR → Gemini beats → OpenAI script), and the narrative-quality overhaul (R2/R3/R4 + flashback + continuity) is in. But QA on real chapters surfaced **visual/scene defects** that separate "runs" from "looks professional." This sub-project fixes them, verified against the actual ORV/Nano examples.

## Findings & Tasks (each verified against real crops)

| # | Defect (real example) | Status | Fix |
|---|----------------------|--------|-----|
| 1 | Duplicate/overlapping crops (sub-region of same tall panel) | ✅ **done** | Geometric IoU/containment dedup (`--dedupe-overlap`), merged to main |
| 2 | **Over-segmentation** — 24 pages → ~116 scenes (~5/page); choppy, too many micro-crops | ⏳ | Tune panels_to_scenes gutter-split aggressiveness + min panel height; target ~1–3 meaningful panels/page. Acceptance: Nano ch1 yields a sane scene count (~40–60, not 116). |
| 3 | **Text-only bubble as its own scene** (p20 "PEASANT BLOOD…") | ⏳ | Detect bubble-only / text-dominated crops (high bubble-area fraction, low non-bubble content) and merge into the neighboring story panel rather than emitting standalone. |
| 4 | **Bubbles not removed** from displayed images (white **and black** backgrounds) | ⏳ | Wire `clean_panels_inpaint.py` as a `cleaned` stage → `clean_scenes/`; FIX its dark-bubble handling (audit: current bubble test requires bright bg, so dark/narration bubbles survive); timeline uses `--prefer-clean --clean-scene-dir`. OCR/vision still runs on raw (with bubbles) so narration keeps the text. |
| 5 | **Bubble TYPE → narration mode** (smooth=spoken, jagged=internal monologue e.g. p20, rectangle=narration) | ⏳ | YOLO can't subclassify by outline; the Gemini beats pass CAN. Enhance `gemini_narrative_pass` prompt to tag each line's mode (spoken / inner-thought / narration / shout) from the visual bubble style; pass mode to script so it narrates "he thinks…" vs "he says…". |
| 6 | **Split / diptych action panels** (Image #6: two images, opposing motion lines = one clash moment) split into 2 scenes | ⏳ | Detect adjacent crops that form one visual moment (side-by-side, motion-line cues, or Gemini判定) and keep them as one shot (fast intercut), not two unrelated scenes. |
| 7 | **Emotion-aware pacing** — narration/timing should track scene intensity (intense fight = punchy + fast cuts; quiet = longer holds). Currently even split. | ⏳ | Use beats' `mood_words`/intensity to modulate per-shot duration and `min_cut_sec` (intense → shorter holds/faster cuts within a min; emotional → longer). The narration-length-per-image should reflect emotion, not be uniform. |

## Architecture notes

- **Raw vs clean scenes:** keep `scenes/` (raw, with bubbles) for OCR/vision so narration knows the dialogue; produce `clean_scenes/` (bubble-inpainted) for the *video*. `timeline_planner` already supports `--prefer-clean --clean-scene-dir`.
- **Bubble understanding belongs in the Gemini pass**, not regex/YOLO — same lesson as the OCR-vs-Gemini discussion. Items #3, #5, #6 are best driven by enhancing the multimodal beats prompt to emit structured per-panel metadata (bubble type, is-text-only, is-split-pair, intensity).
- **Min-time-per-picture** (≥3.5s) is already enforced via `--min-cut-sec 3.5` (SP1 fix); #7 makes it emotion-adaptive rather than fixed.

## Build order (highest impact first)

1. **#2 over-segmentation** (biggest visual win — fewer, better crops; also shortens the choppy montage)
2. **#3 text-only bubble merge** (kills p20-type standalone bubbles)
3. **#4 bubble inpaint** (white+black) + `--prefer-clean` wiring
4. **#5 bubble-type → narration mode** (Gemini prompt; quality of narration)
5. **#6 split-panel grouping** + **#7 emotion-aware pacing** (polish)

## Acceptance

Re-run Nano Machine ch1 and ORV ch1 end-to-end; verify: scene count is sane (#2), no standalone text bubbles (#3), displayed images have bubbles cleaned on both bg colors (#4), narration distinguishes thought vs speech (#5), split-action panels stay paired (#6), and intense scenes pace faster than quiet ones (#7). Full test suite stays green.

## QA confidence instrument (added 2026-06-09, before SP2 fixes)

`studio/qa_flags.py` + the `qa.py` scorecard turn the eyeball-only report into a
measuring instrument so each SP2 fix is verifiable, not a judgment call. It
renders the **canonical, in-sync** scene set (warns on drift) and scores the
defect classes. **Baseline after the deduped re-derive of Nano ch1** (116 scenes,
24 pages, 36 groups) — the numbers SP2 must move:

| Metric | Baseline | Target |
|---|---|---|
| scenes / page | 4.83 | ≤3 |
| near-dup pairs (dHash ≤8) | 2 | 0 |
| short <3.5s pictures | 86 | ~0 |
| OCR-echoes | 4 | 0 |
| text-only bubbles | 3 | 0 |
| groups w/o narration | 3 (groups 4–6, p12–p16) | 0 |

## Review findings that refine the build (Nano ch1, 2026-06-09)

- **"short on screen" is a density symptom, not a per-image defect.** = a picture
  whose share (`shot narration_s ÷ #pictures`) is <3.5s. At render,
  `timeline_planner --min-cut-sec 3.5` keeps `floor(shot_s/3.5)` pictures and
  **drops the rest** (narration untouched). 86/116 flagged ⇒ ~half the montage
  would be discarded arbitrarily by the renderer. #2 fixes this at the source by
  not over-generating, so kept panels are chosen deliberately.
- **dHash dedup catches PIXEL near-dups only.** p63/p64 (h5), p84/p85 (h7) caught
  correctly; but **p74/p75 are a SEMANTIC duplicate (same moment) at hamming 20**
  — dHash can't see it, and raising the threshold adds false positives (coincidental
  dark panels at h11) without catching it. **Semantic redundancy must come from the
  Gemini shot-selector** (`tools/gemini_shot_selector.py` already exists — "keep vs
  redundant" — but is NOT wired into the studio pipeline). Wiring it is the real fix
  for the user's notion of "duplicate." → add to #1/#5 scope.
- **Tall "long shots" need render-side handling.** p79 aspect 2.64, p5 3.0 — too
  tall for 16:9 full-frame. The scener's gutter-`split` didn't fire (gutters not
  "safe"). Fix: tall-scene (aspect > ~1.8) → auto **vertical pan** via the
  `motion`/`camera_path` the timeline already emits, OR gutter-split. → fold into #6/#7.
- **Bubble inpaint (#4) confirmed NOT done** — `clean_scenes/` absent. Note: current
  `clean_panels_inpaint.py` bubble test assumes BRIGHT bubbles; p79 has black-outline
  bubbles over art, so dark/edge cases must be handled.

**Sequencing lesson:** #2 (over-segmentation) changes the scene set and forces a
downstream re-derive, so it MUST land before #3/#4 (which operate on the final
scenes) — otherwise inpaint/merge work is thrown away on re-scene.

### #2 update — geometry is the wrong lever for dense-real-panel manhwa (proven)

Implemented the agnostic geometric merge (`merge_small_bands` +
`median_page_height`, `--min-panel-page-frac`, page-fraction so it's
series-agnostic). **Real-data sweep on Nano ch1 disproved the geometric
hypothesis:**

| min-panel-page-frac | slivers merged | scenes | /page |
|---|---|---|---|
| 0.00 | 0 | 116 | 4.83 |
| 0.08 | 15 | 117 | 4.88 |
| 0.10 | 28 | 119 | 4.96 |
| 0.12 | 39 | 118 | 4.92 |

Merging up to 39 slivers leaves the count at ~116 — the scener runs
**merge → gutter-split**, and `split_crop_on_gutters` re-separates any merge that
crosses a real gutter. So Nano's 116 panels are **genuinely distinct,
gutter-separated panels**, not an over-detection artifact. Forcing the count down
geometrically would override real gutters = fuse distinct story panels (quality
damage). 

**Reframe:** over-segmentation on a dense manhwa is a **selection** problem, not a
geometry one. The agnostic fix = the **Gemini shot-selector** (`gemini_shot_selector.py`,
keep-vs-redundant) wired into studio — which also resolves the p74/p75 semantic
duplicates. The geometric merge stays (off by default) for series where YOLO
over-detects *contiguous* fragments (bands with no gutter between them); there it
sticks. **Next: wire the Gemini selector as the real #2 + semantic-dedup fix.**

### #2/dedup/pacing — DONE (2026-06-09)

Folded scene-understanding into the EXISTING beats Gemini call (no new API
stage, ~$0 extra): per-scene `keep`/`redundant` + `bubble_mode` + `intensity`.
`timeline_planner` drops `redundant` panels FIRST. Result on Nano ch1: shown 56
panels (2.33/page), 0 under 3.5s, 0 visible dups (all 3 incl. p74/p75 caught),
25 redundant marked. QA scorecard now judges the *rendered* montage and is GREEN.

## Cost reduction (2026-06-09) — partly done, batch is TODO

Measured per-chapter LLM cost (Nano ch1, exact via `tools/usage_cost.py`):
beats(Gemini Flash) ~$0.085, script(gpt-4.1-mini) ~$0.065. At 300 chapters ≈ $45/manhwa.

DONE: per-stage model config (`studio.toml [models]`), default `script_model=gpt-5-nano`
(~5× cheaper; gpt-4.1-mini API-retires 2026-10-14); exact token+$ logging with
cached-token visibility (OpenAI auto-caches the static system prompt). Free local
TTS (`chatterbox`/`kokoro`) eliminates the ElevenLabs per-character bill — the
dominant cost at scale.

**TODO — Batch API (50% off, the bulk-backlog mode):** This is NOT a flag; it
reshapes execution. The current pipeline runs one chapter synchronously
(submit→wait→advance status). Batch is async: build a JSONL of all requests
(beats: one line per group; script: one per section), submit to the provider's
batch endpoint, poll (up to 24h), retrieve, then map results back by
`custom_id`=segment/group key. Design:
- New `studio batch` subcommand operating over MANY discovered-but-unprocessed
  chapters (where batch pays off), separate from the interactive single-chapter run.
- Per-tool `--emit-batch <jsonl>` (build requests, no calls) and `--apply-batch
  <results>` (parse results into the manifest) modes, reusing the existing
  prompt/schema builders. Keep `custom_id` = `{chapter}:{group_id}` so results
  rejoin deterministically.
- Catalog: add `batch_submitted`/`batch_pending` substates so polling is resumable.
- Tradeoff: ≤24h latency, so batch is for overnight bulk runs, not previewing one
  chapter. Stacks with gpt-5-nano + flash-lite + caching → ~$0.02/chapter target.
