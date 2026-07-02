# Adaptive Flow Narration Implementation Plan

> **For agentic workers:** REQUIRED: Use subagent-driven-development (if subagents available) or executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Narration segments span 1–4 consecutive panels — flow passages voiced as ONE clip with panels paced under the voice; solo lines where a moment lands — replacing rigid per-panel 1:1.

**Architecture:** The beats writer emits `beats[].segments[] = [{"span": [scene_files…], "line": "…"}]` (the LLM chooses flow vs solo from panel understandings; a deterministic validator enforces exact cover, span cap, system-solo, word budget). All downstream consumers read segments via one shared helper; the verbatim script packer emits one paragraph + one shot (scene_files = span) per segment; the planner routes span>1 shots through the EXISTING `multi_cut` pacing. TTS stays one clip per segment (no stitching — per-group TTS was reverted for stutter/timbre drift; never reintroduce it).

**Tech Stack:** Python 3.12 (`.eval_venv`), pytest (baseline **1213 passed, 1 skipped**), ollama/gemma writer, qwen-mlx TTS, Remotion render (no renderer changes).

**Spec:** `docs/plans/specs/2026-07-02-adaptive-flow-narration-design.md` (user-approved 2026-07-02).

**Verified anchors (do not re-derive):**
- `segment_id = f"g{gid:04d}_p{i:02d}"` where `i` = paragraph index in section — `tools/script_expander.py:2180`. Fewer segments renumber naturally; contract byte-identical.
- Planner already reads per-shot `scene_files` (`tools/timeline_planner.py:1617`) and has `display_strategy` `multi_cut`/`single_hold` + `build_cuts` pacing (“pace the panels UNDER the voice”, `:1005`), per-panel floor `_floor_shot_dur` and BLOCKING `flash_cut` QA.
- `panel_narration` consumers (grep-verified, all must migrate to the helper): `tools/gemini_narrative_pass.py`, `tools/script_expander.py`, `tools/narration_punchup.py`, `tools/recap_style.py`, `tools/prep_qa.py`, `tools/teaser_planner.py`, `tools/narration_heal.py` (via corrections), `studio/dashboard/app.py`.
- 1:1 normalization to replace: `align_panel_narration` call at `tools/gemini_narrative_pass.py:1251-1258`; schema block `:797`; prompt “EVERY panel its own line” `:978`.
- Mood→exaggeration: script paragraph mood tag → `tools/local_tts_from_manifest.py:154-176`. Span mood = MAX intensity across span panels.
- Config: `studio/config.py` load pattern + `studio/pipeline.py:310-313` narration_source pass-through show how to plumb one new key.

---

## Chunk 1: beats writer — segments schema, prompt, validator, config

### Task 1.1: shared `beat_segments()` helper + config key

**Files:**
- Create: `tools/beats_segments.py`
- Modify: `studio/config.py` (new `[narration] segmentation` key, default `"adaptive"`, allowed `{"adaptive","per_panel"}`)
- Test: `tests/test_beats_segments.py`

- [ ] **Step 1: failing tests** — `beat_segments(beat)` returns `[{"span": [...], "line": str}]`:
  (a) beat with native `segments` → returned as-is (spans normalized to basenames);
  (b) legacy beat with only `panel_narration: [{scene_file, line}]` → singleton spans in order;
  (c) beat with neither → `[]`;
  (d) malformed entries (missing line/span) skipped.
  AND `write_segment_lines(beat, lines)` — the shape-aware WRITER (mutators must round-trip
  whichever shape the beat carries, or the teaser's `{"beats":[{"panel_narration": …}]}`
  round-trip silently loses repairs):
  (e) native-segments beat → `segments[].line` updated in order, `narration` join rebuilt;
  (f) legacy beat → `panel_narration[].line` updated in place, join rebuilt;
  (g) length mismatch → ValueError (a mutator may edit lines, never re-split).
- [ ] **Step 2: implement** (pure functions, no I/O; basenames via `os.path.basename`).
- [ ] **Step 3: config** — `segmentation: str = "adaptive"` on the Config object, parsed from `studio.toml` `[narration]` table; invalid value falls back to `"adaptive"` with a warning. Unit test: create `tests/test_config_narration.py` (mirror `tests/test_config_teaser.py`).
- [ ] **Step 4: run** `.eval_venv/bin/python -m pytest -q tests/test_beats_segments.py` → green; commit `feat(beats): beat_segments helper + narration.segmentation config`.

### Task 1.2: writer emits segments (schema + prompt + validator + one repair re-ask)

**Files:**
- Modify: `tools/gemini_narrative_pass.py`
- Test: `tests/test_narrative_segments.py`

**Contract (from spec §3.1):** in adaptive mode each beat's JSON must contain `segments`:
`[{"span": ["p000013.jpg", …], "line": "…"}]`, replacing the 1-line-per-panel normalization
(`align_panel_narration` stays ONLY as the per_panel/fallback path producing singleton spans).

Deterministic validator (`validate_segments(beat, scene_files, kinds, wpm)` — pure, unit-tested):
1. spans partition `scene_files` exactly (order-preserving, no skip/overlap/unknown file);
2. `len(span) <= SPAN_CAP` (code constant = 4; 4 × 6.0s = 24s max clip);
3. any `panel_kind == "system"` file must be in a singleton span;
4. word budget: `N*2.0s <= words / (WPM/60) <= N*6.0s` per segment (N = span size;
   `WPM = 135` code constant, matching script_expander's default) — reject too-thin AND
   too-fat lines;
5. every `line` non-empty, no bracket-mood prefix (added later by the packer as today).

**Load-bearing (spec §3.1):** `beat["narration"]` REMAINS the ordered join of segment lines
(gemini_narrative_pass.py:1262 today) — caption_unvoiced, narration_stale and
alignment_flags all key on it. The 1:1 one-line-per-panel count assert at
gemini_narrative_pass.py:1260 becomes the validator's COVER assert in adaptive mode.

On validation failure: ONE repair re-ask (same group, error list appended to the prompt);
still failing → fall back to `align_panel_narration` singleton spans for that beat (never
block the chapter; log `[segments] fallback beat gNNNN`).

Prompt changes (keep the existing persona/grounding/caption rules intact):
- replace “EVERY panel its own line” with flow/solo criteria: continuous action, traversal,
  montage progressions, caption-only runs → ONE connected passage over that span (clauses may
  lean across panels; end mid-momentum, not mid-word); emotional close-ups, reveals,
  punchlines, dialogue-heavy panels, system cards → solo line;
- word-budget guidance per span size (≈5–15 words/line solo; a 3-panel flow ≈ 25–45 words);
- forbid enumerating panels inside a passage (“in the next panel” banned — narrate the story).

- [ ] **Step 1: failing tests** — validator units (cover/overlap/cap/system-solo/budget cases + repair-fallback path with a stubbed model that returns bad-then-good, and bad-bad → singleton fallback). Writer output shape test with model stubbed.
- [ ] **Step 2: implement** schema + `--segmentation` CLI flag (tool default `adaptive`; env `STUDIO_NARR_SEGMENTATION` overrides; pipeline passes it explicitly; `per_panel` short-circuits to today’s path). **Existing narrative-pass tests whose model stubs return `panel_narration`-shaped output: pass `--segmentation per_panel` (or set the env) in those tests — they cover the legacy path; adaptive gets its own stubs.** New adaptive tests use segments-shaped stubs.
- [ ] **Step 3:** `.eval_venv/bin/python -m pytest -q tests/test_narrative_segments.py tests/test_beats_segments.py` green.
- [ ] **Step 4:** full suite → no regressions (existing narrative-pass tests keep passing: per_panel path must remain byte-compatible). Commit `feat(beats): adaptive flow segments — spans, validator, repair re-ask`.

### Task 1.3: pipeline plumb

**Files:**
- Modify: `studio/pipeline.py` (beated stage passes `--segmentation` from `cfg.segmentation`)
- Test: extend the existing pipeline stage-args test file

- [ ] Failing test asserting the flag is passed; implement; suite green; commit `feat(pipeline): pass narration segmentation to beats stage`.

## Chunk 2: consumers — packer, punchup, style, planner, teaser, dashboard

### Task 2.1: verbatim packer emits one paragraph/shot per segment

**Files:**
- Modify: `tools/script_expander.py` (`_build_verbatim_section` + shot rows)
- Test: `tests/test_script_expander_segments.py` (new; mirror existing verbatim tests)

- [ ] **Step 1: failing tests** — given a beat with a 3-panel span + a solo span: 2 paragraphs, 2 shots; `shots[k]["scene_files"]` == span (basenames, order kept); `segment_id`s enumerate paragraphs (`g####_p00`, `g####_p01`); mood tag derived from MAX panel intensity across the span (existing intensity source per panel; tie → existing behavior); caps/quote normalization unchanged (reuse existing helpers).
- [ ] **Step 2: implement** — replace the per-panel iteration with `beat_segments(beat)`; keep every existing text-normalization step per line. `primary_scene_file` = span[0] (planner fallback + narrated_files protection reads it — verify `narrated_files_from_plan` still protects the span head). **Retire `merge_short_panel_items`/`tts_merge_short` (script_expander.py:929-934) in adaptive mode** — flow spans supersede it; letting it also run would double-merge (skip when segmentation is adaptive; test asserts it's a no-op there).
- [ ] **Step 3:** targeted + full suite green. Commit `feat(script): pack narration segments — one paragraph/shot per span`.

### Task 2.2: punchup + recap_style iterate segments

**Files:**
- Modify: `tools/narration_punchup.py`, `tools/recap_style.py` (swap `panel_narration` iteration for `beat_segments`; write back into `segments[].line`; punchup rebuilds `beat["narration"]` as the join, as it does today at narration_punchup.py:650-651)
- Test: extend the existing punchup/style tests with a flow-span fixture

**CAUTION (spec reviewer, verified):** `recap_style` READS AND WRITES the old shape in ~7
places (re-grep `panel_narration` at implementation time — line anchors drift; last grep:
:236, :271, :290, :323, :504, :574, :613). All guard with `.get("panel_narration") or []`,
so a missed site SILENTLY NO-OPS the 6-rules enforcement — migrate every READ to
`beat_segments` and every WRITE-BACK to `write_segment_lines` (the Task 1.1 shape-aware
writer), and add a test that sauce_density counts flow-span lines.

**Teaser round-trip (plan reviewer, verified):** `teaser_planner.py:510-514` wraps its
narration as `{"beats":[{"panel_narration": …}]}`, has recap_style repair it IN PLACE, then
reads `panel_narration` back. Because the mutators now write via `write_segment_lines`
(shape-aware), that round-trip keeps working unchanged — add a test HERE (not 2.4): a
legacy-shaped teaser beat goes through `repair_spoken_fragments` +
`neutralize_identity_reveal_leaks` and the repaired lines are readable back from
`panel_narration`.

**Punchup budget guard (spec §3.1):** after a rewrite, re-check the span word budget
(`N*2.0s <= words/(135/60) <= N*6.0s`); a violating rewrite is REJECTED → keep the original
line (same pattern as the existing caption-preservation fallback). Test: stub a punchup that
returns a 5-word rewrite for a 3-panel span → original kept.

- [ ] Failing tests → implement → green. Punchup validates per line exactly as today (forbid_quotes fallback intact). Commit `feat(narration): punchup + style operate on segments; span budget guard`.

### Task 2.3: planner routes span shots through multi_cut

**Files:**
- Modify: `tools/timeline_planner.py`
- Test: `tests/test_timeline_planner_spans.py` (new)

- [ ] **Step 1: failing tests** — a shot with 3 scene_files + a 10.5s clip → `display_strategy == "multi_cut"`, 3 cuts in order, each ≥ 2.0s floor, sum == clip duration; a 1-file shot → today’s single path unchanged; protected/system files in-span still honored by `ensure_protected_shown`; **a 3-file span where 1 file was visually dropped upstream (filter_scene_files/clean-dir filtering) → 2 cuts, the full clip duration reallocated across survivors, no gap, no sub-2.0s cut** (spec §3.5 “visual drops inside a span shrink the cut list — narration untouched”; day-one scenario, `_heal_visual_drops` fires routinely).
- [ ] **Step 2: implement** — the routing condition (span>1 → multi_cut over the span) reusing `build_cuts`; do NOT touch floor/flash logic.
- [ ] **Step 3:** full suite green (planner tests are extensive — zero tolerance for behavior drift on single-file shots). Commit `feat(planner): pace flow-span panels under the voice via multi_cut`.

### Task 2.4: teaser_planner + dashboard read segments

**Files:**
- Modify: `tools/teaser_planner.py`, `studio/dashboard/app.py` (iterate `beat_segments`; dashboard shows one row per segment with its span thumbnails/count)
- Test: extend existing tests minimally (teaser scoring over lines; a dashboard route test in `tests/dashboard/` — the harness exists, e.g. `test_videos_teaser.py`)

- [ ] Failing → implement → green. Commit `feat(consumers): teaser + dashboard read narration segments`.
  **Deploy note:** `studio/dashboard/app.py` ⇒ dashboard daemon restart on Mini at rollout.

## Chunk 3: QA + heal span-awareness, e2e, docs

### Task 3.1: prep_qa cover checks + span-aware grounding/captions

**Files:**
- Modify: `tools/prep_qa.py`
- Test: `tests/test_prep_qa_spans.py` (new) + keep every existing prep_qa test green

- [ ] **Step 1: failing tests** —
  (a) `estimate_plan`/1:1 count assertions become COVER assertions: every shown story panel belongs to exactly one segment span (uncovered → ERROR `panel_uncovered`; double-covered → ERROR `panel_double_covered`); segment-count==panel-count checks removed;
  (b) grounding judges a segment against ALL its span panels (cache key = text_sha + sorted span);
  (c) `caption_unvoiced` searches the span’s combined OCR;
  (d) `spoken_fragment` unchanged per segment line;
  (e) `shot_description_flags` iterates `beat_segments` (prep_qa.py:743-763 currently iterates `panel_narration` — silently no-ops on segments-only beats; test: a flow-span line with camera language still flags).
- [ ] **Step 2: implement**; full suite green. Commit `feat(qa): span-aware cover, grounding, caption checks`.

### Task 3.2: heal regen is SPAN-PINNED

**Files:**
- Modify: `tools/gemini_narrative_pass.py` (the `--corrections` regen path)
- Modify (only if needed): `tools/narration_heal.py`
- Test: extend heal tests with a span fixture

**Rule (spec §3.5):** corrections stay group-scoped (`{group_id: note}` — interface
unchanged, narration_heal.py:78-102). But when regenerating a corrected group, the re-ask
passes the beat's EXISTING spans as FIXED — the writer rewrites LINES within those spans and
may never re-split them. A re-split would renumber sibling segment_ids → clip-cache churn +
audio_stale. Only a full beats re-run (no `--resume`) may change spans. Validator on the
regen result: same spans in, same spans out (else fall back to the previous lines).

- [ ] Failing test: regen with a stubbed model that returns a different segmentation → spans preserved via fallback; a compliant rewrite → new lines, same spans. Implement → green. Commit `feat(heal): span-pinned regen — lines may change, spans may not`.

### Task 3.3: e2e fixture + docs cascade

**Files:**
- Test: `tests/test_flow_e2e.py` — tiny synthetic episode: 1 group / 5 panels (1 system card), stubbed writer returns 1 flow-span(3) + 2 solos → assert: beats segments validate; script has 3 paragraphs w/ correct scene_files; planner plan gives every panel ≥2s; prep_qa cover check green; system card solo + shown.
- Modify: `CLAUDE.md` (short “adaptive flow narration” note under current-state), `.continue-here.md` refreshed at rollout.

- [ ] e2e green; full suite green (expect baseline 1213 + new tests, 0 failures). Commit `test(e2e): flow narration end-to-end fixture + docs`.

---

## Rollout (after merge — driver, not subagents)

1. FF-merge worktree branch → main; full suite on main; push.
2. Mini: `git pull` (tools+pipeline = subprocess-fresh) + **dashboard daemon restart** (`launchctl kickstart -k gui/$(id -u)/com.originpower.dashboard`) for Task 2.4. Worker restart NOT needed (worker.py untouched).
3. Reset ch1 (310) beats-onward ONLY (keep understanding/groups/vision caches): status → `grouped`; delete `manifest.beats.json manifest.cast.json manifest.script.json heal_corrections.json render.plan*.json prep_qa.{json,html} .narration_keepbase` + `tts/ render/ scenes_clean/` + clear `manual_drops.json` (panel set unchanged, drops re-derive) + `DELETE FROM approval WHERE chapter_id=310`.
4. Enqueue prepare (priority 1); verify: segments span mix present (not all singletons), QA green (cover checks pass, no flash_cut), fewer segments (~55–70 vs 112).
5. Enqueue voiceover + render explicitly (autopilot is OFF) → fresh `segment_both.mp4` + `voice_preview` for the user's listen. **Renders remain user-gated after this review copy.**
