# Per-Panel 1:1 Narration + Content-Driven Pacing + Intensity-True Delivery — Design

- **Date:** 2026-07-01
- **Status:** Approved design, pre-implementation
- **Goal:** Make every SHOWN story panel its own narration line → its own segment → its own TTS clip → its own image-aware duration, so panels stop being crammed under one short line ("flash montages"), and grade delivery (calm/tense/intense/explosive) into the production qwen-mlx voice.
- **Audit it builds on:** `docs/2026-06-30-manhwa-fresh-vs-current-audit.md`
- **Prior merged plan:** `docs/plans/specs/2026-06-30-narration-niche-modules-and-quality-fixes-design.md`
- **Lineage:** `docs/plans/specs/2026-06-15-recap-quality-root-cause-fix.md` (understanding-first redesign), `docs/plans/specs/2026-06-19-per-panel-rolling-narration-design.md` (per-panel rolling narration — this spec hardens it into a strict 1:1 invariant)

---

## 1. Architecture

### 1.1 The 1:1 principle

The pipeline already WRITES one `panel_narration` line per shown panel (the understanding-first redesign). The defect is that pacing/render still **collapses** real art panels: a story panel can lose its own line, `segment_id`, and TTS clip on the way to the timeline, then only reappear as a *silent injected cut* piled onto another segment. That pile-up is the flash.

The fix is a strict, end-to-end invariant:

> **One SHOWN story/art panel → one `panel_narration` line → one segment (`segment_id` `g####_p##`) → one TTS clip → one timeline item whose `duration_sec` comes from THAT line's own audio, image-adjusted.**

Grouping is **demoted to soft context only**: it still tags `segment` (present/flashback/dream), `arc_label`, and provides narration continuity and caption-coverage keys. It must **never collapse panels at render time**. A panel's visibility is decided ONCE, upstream, by its `panel_kind` — not re-decided by a time budget or a non-deterministic "redundant" verdict downstream.

### 1.2 Shown vs. folded (the gating that stays)

"Show every panel" means every **showable STORY/ART panel only**. The `panel_kind` enum decides:

- **`story`** → SHOWN, 1:1. One line, one segment, one clip.
- **`caption`** (text-only monologue/narration boxes) → NOT shown standalone; its words **fold** into the spoken narration over an adjacent art panel (`story_group.merge_caption_solos`, `tools/story_group.py:307`).
- **`chrome`** / **`empty`** → dropped as non-story (`story_group.nonstory_files`, `tools/story_group.py:200-211`).
- **`system`** (status/notification cards) → handled as today: a story beat, protected, never cleaned, shown.

Folding pure-text/caption panels is **not** the flash bug. The flash bug was collapsing real ART panels (multiple `story` panels onto one short segment). This spec stops the latter; it preserves the former exactly. State this explicitly in review: removing `inject_missing_protected` must not change `panel_kind` gating.

---

## 2. Components / data flow

Trace of one chapter, marking where each of the 4 changes lands:

| Stage | Tool / anchor | Per-panel role | Change |
|---|---|---|---|
| Understand | `tools/panel_understand.py` — `panel_kind` + `intensity` graded per panel (schema `:41-46`, prompt `:61-64`) | Stamps `panel_kind` (story/chrome/empty/caption/system) and `intensity` (calm/tense/intense/explosive) onto each panel | source for **C3** (intensity) + gating |
| Group | `tools/story_group.py` — spans + `segment`/`arc_label` (`:34`, `:55`); caption fold (`:307`); nonstory/effect filters (`:200-211`, `:262`) | SOFT context + flashback tag ONLY; must not drive render pacing | demote (C4 cap `DEFAULT_MAX_BEAT_LEN` `:42`) |
| Beats | `tools/gemini_narrative_pass.py` — writes `panel_narration[]` (one line/panel); `role: keep\|redundant` (`:1096`) | one line per shown panel | `redundant` no longer drops a story panel (**C1**) |
| Script | `tools/script_expander.py` — per-panel path (`:904-924`), `merge_short_panel_items` (`:792-844`, called `:924`), `segment_id` mint (`:2168`), mood escalate (`:746-756`) | one paragraph + one shot + one `segment_id` per panel | stop the merge collapse + keep segment (**C1**); per-panel mood (**C3**) |
| TTS | `tools/local_tts_from_manifest.py` — clip keyed by `segment_id` (`:814`, `:822`), mood→exag (`:154-162`), qwen-mlx synth (`:1388`) | one `clips/{segment_id}.wav` per segment | per-clip intensity → qwen-mlx (**C3**) |
| Timeline | `tools/timeline_planner.py` — `compute_duration_sec` (`:287-316`), `build_cuts` (`:937`), `inject_missing_protected` (`:1027-1060`), `dur = sum(cuts)` (`:1818`) | one item/segment; duration from its own audio | image-aware duration (**C2**); drop inject (**C1**); floor (**C4**) |
| Render | Remotion (production) seats each segment at its absolute `start_sec` | one shot/segment | unchanged |

**Hard contract (CLAUDE.md):** `segment_id` (`g####_p##`) must stay **byte-identical** across `script_expander` → TTS → `timeline_planner`. Verified end-to-end (obs 16077, 14774). Every change below must preserve this.

---

## 3. The four changes

### C1 — Stop dropping protected story panels upstream (the root fix)

**What:** Keep every SHOWN story panel's own `panel_narration` line + `segment_id` + TTS clip end-to-end. Make `inject_missing_protected` unnecessary, then remove it.

**Where the panel is lost today (cite all):**

1. **`script_expander.py:792-844` `merge_short_panel_items`** (invoked at `:924`, default-on via `tts_merge_short=True`, `:856`). It buckets consecutive short panel-lines into ONE clip (`≤ short_words`, `≤ max_panels`, `≤ max_words`). Its invariant preserves every `scene_file`, but it collapses N panels into ONE `segment_id`/clip — so the timeline must re-spread those files as cuts under one short segment. That is a primary flash source.
2. **The `redundant` scene_selection role** (written by the beats prompt, `gemini_narrative_pass.py:1096`) consumed by **`scene_selection.choose_kept_scenes` (`tools/scene_selection.py:73`, drop at `:80`/`:101-102`)** — a `story` panel the LLM (non-deterministically) tags `redundant` is **dropped from a *merged* segment's cuts**. Get the causal chain right: `script_expander`'s per-panel path (`:904-924`) does NOT consult the scene_selection role, so a `redundant` story panel DOES get its own paragraph, shot, and `segment_id`. It only loses its own segment because of the **merge** (`merge_short_panel_items`, point 1); the `redundant` role then drops it from the *merged* segment's cuts, after which it can only return as a silent injected cut. So disabling the merge yields 1:1, and the protected-set (C1-part-2 below) covers any residual.
3. **`timeline_planner.inject_missing_protected` (`:1027-1060`, helper `pick_protected_inject_segment` `:1002`)** — the band-aid: protected files that landed in NO segment are appended to ONE existing segment's pick list. That is the pile-up — many panels under one short line.

**Why:** Under 1:1, a shown story panel is *born* with a line + `segment_id` + clip and keeps them. There is nothing to inject because nothing was dropped. `merge_short_panel_items` collapse and the `redundant`-drop of `story` panels are the two upstream causes; remove both for story panels and the timeline injector has no work left.

**The fix shape (design intent, not code):**
- Treat the per-panel path (`script_expander.py:904-924`) as the canonical, non-collapsing path: one `items` entry → one paragraph → one shot → one `segment_id`. Disable/short-circuit `merge_short_panel_items` for story panels by gating off the `elif tts_merge_short:` guard at the call (`:924`). **KEEP the function** — do NOT delete it: it stays in place, gated off, and its 6 unit tests at `tests/test_verbatim_script.py:433-514` stay green (name the file so an implementer doesn't "clean up" the function out from under those tests).
- `choose_kept_scenes`/`redundant` must never drop a **`story`** panel. **C1-part-2 (implementation constraint):** neutralize the redundant-drop by passing every SHOWN story panel through the existing `protected` set at the **`build_cuts` call site (`timeline_planner.py:986`)** — story panels arrive pre-protected, so they are effectively always "keep". Do **NOT** change `choose_kept_scenes`'s core redundant logic: `tests/test_scene_selection.py:95-147` pins that `redundant` panels drop and `protected` overrides, so editing the function would break those tests and is unnecessary (it already force-keeps `protected` cards via the `protected` set, `:89-92`/`:107`; just widen what enters that set at the call site). Also note: for a true single-file (1:1) segment, `build_cuts` early-returns at `:963-964` before selection runs, so once `merge_short_panel_items` is gated off the redundant-drop is already moot for 1:1 segments and only matters for any rare multi-panel residual. Near-duplicate *merging* is an explicit NON-GOAL, §7.
- `inject_missing_protected` becomes a no-op and is removed once (1)+(2) hold.

**KEEP `inject_missing_protected` + `pick_protected_inject_segment`; under merge-off they only inject NARRATION-LESS protected/system/title cards (story panels keep their own per-panel segments → never dropped → never injected), so they no longer cause the flash and remain REQUIRED to show system/title cards (else `system_card_unshown`). The 5 unit tests at `tests/test_timeline_selection.py:218-263` STAY.**

**Safety constraint (critical):** The fix must be **UPSTREAM** — preserve each story panel's segment so its `segment_id` + clip survive into the timeline. You must **NOT** mint new `segment_id`s in `timeline_planner`. The mint lives at `script_expander.py:2168` (`f"g{gid:04d}_p{i:02d}"`, indexed by paragraph `i`); a timeline-minted segment has no matching `clips/{segment_id}.wav`, so it is silent and trips `missing_audio` ERROR (`prep_qa.py:1301`). The clip key is `it["segment_id"]` → `clips/{seg_id}.wav` (`local_tts_from_manifest.py:814`,`:822`). Keep the mint where it is; just stop collapsing the inputs to it.

> Note on `segment_id` indexing: the id is `g{group_id}_p{paragraph_index}`. Under 1:1 the paragraph count rises (one per shown story panel), so the `_pNN` index space grows within a group — that is expected and stays internally consistent as long as script → TTS → timeline read the same paragraph order. No change to the id *format*.

---

### C2 — Image-aware duration

**What:** A panel's on-screen time should respect its visual weight, not only its line length. Add a bounded, deterministic image floor.

**Where:** `timeline_planner.compute_duration_sec` (`:287-316`). The narrated branch (`:301-308`) today returns `max(base_min, audio_duration + pad)` — **narration only**.

**The change (design):**
```
duration = max(narration_audio + pad, image_min)
```
where `image_min` is a deterministic function of the panel's visual weight, computed from data ALREADY in the manifests — **no new model call**:
- **crop geometry** from `scene_dims` (width/height per shown file; prep_qa already reads `dims`/`scene_dims`) → normalized area and an "tallness"/aspect term. A full-width tall splash earns more dwell than a small reaction crop.
- **intensity** from the panel's understanding (`intensity` ∈ calm/tense/intense/explosive) → a small additive bump for a reveal/peak.

Concrete, simple heuristic to encode (tune in implementation, keep it monotonic + bounded):
```
visual_weight = norm(area) blended with aspect_tallness, plus a small intensity bump
image_min     = clamp(PANEL_FLOOR_SEC + k * visual_weight, PANEL_FLOOR_SEC, IMAGE_DWELL_CAP)
```
`IMAGE_DWELL_CAP` (≈ 3.5–4.0s) keeps a quiet splash from stalling the recap. Deterministic: same manifest in → same duration out (no RNG, no model).

**Why:** A splash/reveal panel under a short line currently flashes at the audio length; image-aware `image_min` gives it a beat to land. Under 1:1 this is per-panel and clean.

**Signature / backward-compatibility (implementation constraint):** `compute_duration_sec` (`:287-297`) does NOT currently receive `scene_dims` or `intensity`. The implementer must EITHER thread them in as **new OPTIONAL args** (defaulting to no image bump) OR compute `image_min` at the call site (`:1743`, where `beat`/`scene_dims` are already in scope) and pass it as **one new optional arg**. **Backward-compatibility is REQUIRED** so the 3 existing tests at `tests/test_timeline_selection.py:284-304` stay green: with defaults (no `scene_dims`/`intensity`/`image_min` supplied) → no image bump → `base_min` governs exactly as before.

**Safety constraint:** The new duration MUST route through `build_cuts` so the cuts still sum to `duration_sec`. `build_cuts` (`:937`) already extends via `_floor_shot_dur` (`:930-934`, floor `PANEL_FLOOR_SEC`), and the planner adopts the cuts' real total (`dur = sum(cuts)`, `:1818`) so `duration_sec`/`end_sec`/`time_cursor` stay byte-aligned. If `image_min` raised `dur`, pass that raised `dur` INTO `build_cuts` (call site `:1779-1785`) so the cut tiling and the item duration agree. Otherwise `cut_gap` (`prep_qa.py:1288-1295`, `|tile − item_dur| > 0.51`) and `total_drift` trip. Under strict 1:1 most segments are single-panel (one cut == the whole `dur`), so this is mostly a single-cut hold — the alignment is trivial but must still be honored.

---

### C3 — Intensity → MLX TTS delivery

**What:** A tense/intense/explosive panel must actually SOUND different on the production `qwen-mlx` backend.

**Where the signal exists today but dies:**
- Graded per panel: `panel_understand.py` `intensity` (schema `:41-46`, prompt `:61-64`).
- Turned into a mood/exag: `script_expander._escalate_tag_for_intensity` (`:746-756`, ranks `_INTENSITY_RANK` `:730`) writes a leading `[tag]` on the TTS paragraph; TTS parses it (`local_tts_from_manifest.py:816-821` `leading_tag`) → `mood_to_exaggeration` (`:154-162`, table `_EMOTION_BY_KEYWORD` `:90-96`) → flows as `exaggeration` into `run_guarded_synth` → `synth_fn(text, out, exaggeration)` (`:473`,`:505`).
- **Dropped at the backend:** `_make_qwen_mlx_synth` (`:1388`) IGNORES the per-clip `exaggeration` argument and applies a FIXED `exag` from `STUDIO_MLX_EXAG` (default 1.4, `:1408`) at the generate call (`:1428`). Docstring admits it: per-mood mapping is "a follow-up once the voice is approved" (`:1395-1397`). The voice is now approved (production backend, [[mlx-qwen-tts-evaluation]]).

**The change (design):** Wire the per-clip intensity into the MLX synth call. The `synth(text, out_path, exaggeration)` signature (`:1414`) already RECEIVES the per-clip `exaggeration`; use it instead of the fixed env `exag`. **Range caveat (flag in review):** the MLX expressiveness scale is centered ~1.4, while `mood_to_exaggeration` returns 0.25–0.95. A direct substitution would flatten everything — so map the 0..1 mood value onto the MLX expressiveness band (e.g. an affine remap around the approved 1.4 baseline, bounded), keeping `STUDIO_MLX_EXAG` as the neutral center / override. Optionally also feed `exaggeration_to_instruction` (`:165`) since Qwen is instruction-driven. Keep the `SynthFn` interface unchanged.

**Per-panel cleanliness (why 1:1 helps):** Today escalation uses `_intensity_rank_for_beat` — the **MAX** intensity across the beat's `scene_selection` (`:737-743`) applied to ALL of that beat's paragraphs ("beat-max bleed": one explosive panel makes the whole beat shout). Under 1:1 a segment == one panel, so each segment's tag is read from THAT panel's own `intensity`. No bleed. The wiring should read the per-panel intensity (not the beat max) when minting each paragraph's tag.

**Re-voice (cite the gate):** TTS reuses a clip only when `os.path.exists(audio) AND prior_sha == text_sha` (`local_tts_from_manifest.py:826-827`), `text_sha = narration_sha(source_text)` over the FULL paragraph **including the leading `[tag]`** (`:820`). So:
- If intensity changes the **tag** (text changes) → `text_sha` differs → cache miss → auto re-voice; `audio_stale` ERROR (`prep_qa.py:530`) would otherwise gate render.
- If only the **synth mapping** changes (same tag, different rendered exaggeration) → `text_sha` is unchanged → cache HIT → NOT re-voiced. Affected chapters must be **force re-voiced**: overwrite, or reset `planned/voiced → scripted` and clear `clips/` + `tts_index.json` (+ any align). (Matches [[per-group-tts-alignment-shipped]]: re-voicing a planned chapter skips `voiced` unless reset + cleared.)

---

### C4 — Backstop constants

These are RARE safety nets under 1:1, not the pacing mechanism.

- **`timeline_planner.PANEL_FLOOR_SEC` 1.2 → 2.0** (`:927`). Floors any per-panel cut so a dense segment can't flash. Under 1:1, single-panel segments take the full audio dur (≥ `base_min`), so the floor only bites the rare multi-panel residual.
- **`story_group.DEFAULT_MAX_BEAT_LEN` 8 → 6** (`:42`). Smaller soft span; tighter continuity context. Grouping no longer drives pacing, so this is a context-window knob, not a render knob.

**Test / gate updates required:**
- `tests/test_story_group.py` asserts `sg.DEFAULT_MAX_BEAT_LEN == 8` at **`:69`** and **`:384`** → bump both to 6. These are the ONLY `== 8` literals that need changing. Do NOT mislabel the nearby lines as "passes `max_beat_len` as an arg" — they are assertions and stay green on their own: `:75` references the constant **symbolically** (so it auto-adapts when the default becomes 6), and `:377`/`:379` pin a **literal 8** but remain correct.
- `prep_qa.py` flash_cut threshold `dur < 1.2` (`:1274`). The `PANEL_FLOOR_SEC` comment says "keep == prep_qa flash_cut threshold". **Decision:** keep the invariant — bump the gate to **2.0** so the floor and the QA gate stay coupled (the comment stays true; sub-2.0 cuts become a flagged defect, which under 1:1 should essentially never occur legitimately). *Alternative considered:* leave it at 1.2 as a looser, decoupled backstop (catches only true sub-1.2 flashes). Recommend coupling (bump to 2.0); flag in review that any legacy/edge cut in 1.2–2.0 will now flag.
- `tests/test_timeline_floor.py` passes the floor as a literal arg (`_floor_shot_dur(..., 1.2)`, `:6-9`) → **unaffected** by the constant change.

---

## 4. Bubbles / empty-bubble hard constraint (unchanged — preserve exactly)

The user explicitly required these stay intact. This refactor must not touch them:

- **Bubble TEXT is cleaned, bubble SHAPE is kept; NEVER inpaint/blur the bubble.** The cleaned shown art is produced by `render_prep.py` — per the user's directed approach the bubble (shape + outline) STAYS and only its text is blanked with the bubble's own flat color, **no inpainting / no smears** (`tools/render_prep.py:391-392`, `_bubble_text` `:383`). (The module header `:14-16` still describes the older ogkalu-mask→inpaint route; the live path is the flat-blank one — preserve the flat-blank behavior.)
- **System / notification cards are NEVER cleaned** and always survive to be shown (`render_prep.py:168-178`; `panel_kind == "system"`).
- **Empty bubbles are NEVER shown** — `empty_bubble_shown` stays a BLOCKING ERROR (`prep_qa.py:331`, via `rp.empty_bubble_panel`).
- **Text-only / bubble-only / empty panels are NOT shown 1:1.** Their dialogue folds into the spoken narration over adjacent art (`story_group.merge_caption_solos` `:307`; nonstory drop `:200-211`). The `panel_kind` gating (story = shown 1:1; caption/empty/chrome = folded or dropped; system = handled as today) is PRESERVED.
- **Folding pure-text panels is NOT the flash bug.** The flash bug was collapsing real ART (`story`) panels onto one short segment (C1). Folding caption panels into adjacent narration is correct and stays. Say this explicitly so the reviewer does not "fix" folding.

---

## 5. (reserved)

---

## 6. Acceptance criteria

1. **True 1:1:** emitted segment count == count of SHOWN `story` panels for the chapter (caption/empty/chrome folded or dropped, not counted). No story panel exists without its own `segment_id` + `clips/{segment_id}.wav`.
2. **Zero `flash_cut`** in `prep_qa` on a real chapter (at the chosen threshold).
3. **Zero `empty_bubble_shown`**, **zero `cut_gap`/`total_drift`**, **zero `missing_audio`** on the same chapter.
4. **`inject_missing_protected` no longer fires** (no protected file is missing from all segments) — verifiable before removal.
5. **Intensity audibly differs on qwen-mlx:** a calm vs. an intense vs. explosive segment render to perceptibly different expressiveness (A/B listen on the same chapter).
6. **Real-chapter listen pass:** a human listen confirms no flashes, panels dwell with their own line, delivery tracks the scene.
7. **Full pytest green** (`.eval_venv/bin/python -m pytest -q`), including the updated `test_story_group.py` assertions.

---

## 7. Non-goals (decided)

- **No near-duplicate merging.** Show every distinct STORY panel — fidelity over compression. (A slow-zoom run shows each frame for ≥ floor; accepted.)
- **Do not change grouping's tagging/context role** — only stop it driving render pacing.
- **No new external dependencies.**

---

## 8. Risks / blast radius

### Verified-SAFE (with citations)
- **Cast / character naming** — applied upstream of the timeline (cast built at the beated stage; `normalize_caps_for_tts` in `script_expander`), so 1:1 segmentation does not touch it. Unaffected.
- **Flashback / dream tags** — set per-beat in `story_group` (`segment` enum `:34`/`:55`), carried onto sub-beats on split (`:178`), and onto the timeline item (`timeline_planner.py:1826`, `gobj.get("segment")`). The renderer applies the flashback look from `segment != present`. Unaffected.
- **Caption coverage** — computed at the beats stage keyed by group. Grouping stays as context, so the keying is unchanged. Unaffected.
- **Heal loop** — per-group re-narration (`narration_heal`). Groups still exist as context; the heal contract is unchanged. Unaffected.

### Care items (must be handled)
- **`segment_id` upstream-only.** Never mint in the timeline (clipless = silent + `missing_audio`). Keep the mint at `script_expander.py:2168`; only stop collapsing its inputs. (CLAUDE.md hard rule: byte-identical across script → TTS → timeline.)
- **`cut_gap` via `build_cuts`.** Any image-aware duration must flow into `build_cuts` so `sum(cuts) == duration_sec` (`:1818`; gate `prep_qa.py:1288-1295`).
- **Re-voice required** for chapters whose intensity tag or synth mapping changed: reset `voiced/planned → scripted` + clear `clips/`/`tts_index.json` (the `text_sha`/`audio_stale` gates otherwise reuse stale clips).
- **Test updates:** `tests/test_story_group.py:69` and `:384` (DEFAULT 8→6); decide `prep_qa.py:1274` flash_cut threshold (recommend 2.0 to keep the documented invariant). **Removing `inject_missing_protected`/`pick_protected_inject_segment` breaks 5 tests in `tests/test_timeline_selection.py:218-263`** (`test_inject_missing_protected_adds_card_to_a_segment` `:218`, `..._noop_when_already_shown` `:231`, `..._ignores_non_protected_drops` `:238`, `..._only_injects_files_in_group_scene_files` `:250`, `test_pick_protected_inject_segment_prefers_smallest_then_latest` `:259`) — DELETE them with the injector OR repurpose into one regression test of the new invariant ("no shown story panel is ever missing from all segments under 1:1"). Acceptance #7 depends on this. **`merge_short_panel_items` is KEPT (gated off)**, so its 6 tests at `tests/test_verbatim_script.py:433-514` stay green; the 3 duration tests at `tests/test_timeline_selection.py:284-304` must stay green via the C2 backward-compatible signature; do NOT edit `choose_kept_scenes` (its `tests/test_scene_selection.py:95-147` invariants stand).
- **Regenerate derived manifests.** A prompt/understanding change requires deleting `understood.json` (per [[recap-quality-ordering-and-qa-gap]]); re-run grouped → beated → scripted → voiced → planned so the new 1:1 narration and durations propagate. Don't QA against stale manifests (the manifest-freshness guardrail will block, [[manifest-freshness-guardrail]]).
- **Deploy note:** changes touch `tools/` only (subprocesses → fresh on pull); if any worker/dashboard glue changes, a daemon restart is required ([[deploy-worker-needs-daemon-restart]]).

---

## 9. Phasing

The plan chunks into two subsystems:

- **Phase A — pacing / render (C1, C2, C4).** Stop the upstream drop/merge of story panels, add image-aware duration through `build_cuts`, bump the backstop constants + tests. This alone kills the flash and restores 1:1 video; voice is unchanged.
- **Phase B — TTS delivery (C3).** Wire per-panel intensity into the qwen-mlx synth call (+ range remap), then force re-voice affected chapters. Independent of A; can ship as a later phase of the same plan.

The implementation plan will chunk accordingly (writing-plans).
