# Per-Panel 1:1 Narration + Content-Driven Pacing + Intensity-True Delivery — Implementation Plan

> **For agentic workers:** REQUIRED: Use subagent-driven-development (if subagents available) or executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every SHOWN story panel its own narration line → its own segment (`segment_id`) → its own TTS clip → its own image-aware duration, and grade delivery (calm/tense/intense/explosive) into the production `qwen-mlx` voice — so panels stop being crammed under one short line ("flash montages").

**Architecture:** Four surgical changes (C1–C4) across the `tools/` pipeline stages. The pipeline already WRITES one `panel_narration` line per shown panel; the defect is that script/timeline still *collapse* real art panels (merge + redundant-drop + a silent "inject" band-aid). C1 removes the collapse so 1:1 is structural; C2 adds a bounded, deterministic image-dwell floor; C3 wires per-panel intensity into the MLX synth; C4 hardens the backstop constants. No new external dependencies. All changes are in `tools/` (subprocesses → fresh on `git pull`; no daemon restart needed).

**Tech Stack:** Python 3.12 (`.eval_venv`), pytest. Stages touched: `tools/script_expander.py` (beats→script + `segment_id` mint), `tools/timeline_planner.py` (durations + cuts + selection), `tools/local_tts_from_manifest.py` (TTS adapter, qwen-mlx backend), `tools/story_group.py` (grouping cap), `tools/prep_qa.py` (QA gates).

**Source of truth:** `docs/plans/specs/2026-07-01-perpanel-1to1-narration-design.md` (read it fully — it carries the *why*, the bubble hard-constraint §4, acceptance §6, non-goals §7, risks §8). Lineage: `docs/plans/specs/2026-06-15-recap-quality-root-cause-fix.md`, `docs/plans/specs/2026-06-19-per-panel-rolling-narration-design.md`. Builds on the merged niche feature (`3702f15`). Baseline suite on `main`: **1179 passed, 1 skipped**.

---

## Conventions (read before any task)

- **Run tests with the repo venv:**
  ```bash
  V=.eval_venv/bin/python
  $V -m pytest -q                                   # full suite (baseline: 1179 passed, 1 skipped)
  $V -m pytest tests/test_verbatim_script.py -q     # one file
  $V -m pytest tests/test_timeline_selection.py::test_one_panel_segment_yields_exactly_one_cut -q   # one test
  ```
- **`segment_id` (`g####_p##`) is a hard contract** — it must stay **byte-identical** across `script_expander` → TTS → `timeline_planner`. It is minted ONCE, upstream, at `tools/script_expander.py:2168` (`f"g{gid:04d}_p{i:02d}"`). **NEVER mint a `segment_id` in the timeline** — a timeline-minted segment has no `clips/{segment_id}.wav`, so it is silent and trips `missing_audio` ERROR (`prep_qa.py:1301`). Changes only stop *collapsing the inputs* to the mint; they never move it.
- **Edit the plain file, never the suffixed snapshots** (`*-BAK.py`, `*XXX.py`, `*X.py` are frozen). E.g. edit `smart_cropper.py`, not `smart_cropper-BAK.py`.
- **Commit after every task** with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- **Worktree note:** if you implement in a git worktree, symlink the venv into it — `tests/test_ocr_chrome.py:28` subprocesses a **relative** `.eval_venv/bin/python` (`Path(__file__).parent.parent / ".eval_venv"`), which won't exist in a fresh worktree:
  ```bash
  ln -s /Users/anka/repos/Manhwa/.eval_venv <worktree>/.eval_venv
  ```
- **TDD per task:** write the failing test → run it and SEE it fail with the expected message → implement the COMPLETE code → run it and SEE it pass → run the affected file (and at chunk end, the full suite) → commit.

---

## File Structure (decomposition)

No new source files. Changes are localized to existing stage modules; tests extend existing test files.

| File | Responsibility | Changes |
|---|---|---|
| `tools/script_expander.py` | beats → script paragraphs + shots + `segment_id` mint | C1: flip `tts_merge_short` default off (`:856`); C3: per-panel intensity tag (`:936-949`) |
| `tools/timeline_planner.py` | durations, cuts, panel selection, plan emit | C1: remove `inject_missing_protected`/`pick_protected_inject_segment` (`:1002-1060`) + call site (`:1665-1692`); C2: `compute_duration_sec` optional `image_min` (`:287-316`), new `compute_image_min` + `index_dims_by_file` helpers + call-site wiring (`:1470`, `:1743`); C4: `PANEL_FLOOR_SEC` 1.2→2.0 (`:927`) |
| `tools/local_tts_from_manifest.py` | TTS adapter (all backends), qwen-mlx synth | C3: new `mlx_exaggeration` affine remap (near `:154`); use per-clip exaggeration in `_make_qwen_mlx_synth` (`:1414-1429`) |
| `tools/story_group.py` | grouping into beats (soft context) | C4: `DEFAULT_MAX_BEAT_LEN` 8→6 (`:42`) |
| `tools/prep_qa.py` | QA gates | C4: flash_cut threshold `dur < 1.2` → `2.0` (`:1274`) |
| `tests/test_verbatim_script.py` | script/merge unit + integration | C1: convert `test_merge_integration_four_short_lines_fewer_shots` (`:533`) to 1:1; add N→N acceptance test; C3: per-panel intensity test |
| `tests/test_timeline_selection.py` | planner pure-function tests | C1: delete 5 inject tests (`:218-263`), add 2 invariant tests; C2: `compute_duration_sec` image_min tests + `compute_image_min`/`index_dims_by_file`/`_panel_intensity` tests |
| `tests/test_local_tts.py` | TTS pure-helper tests | C3: `mlx_exaggeration` remap tests |
| `tests/test_story_group.py` | grouping tests | C4: `== 8` → `== 6` at `:69` and `:384` ONLY |
| `tests/test_prep_qa.py` | QA gate tests | C4: new flash_cut threshold-boundary test |

---

## Recommended execution order: **Chunk 4 → Chunk 2 → Chunk 1 → Chunk 3**

The chunk headings below follow the spec's change IDs (Chunk 1 = C1, Chunk 2 = C2, Chunk 3 = C3, Chunk 4 = C4). **Execute them in this order:**

1. **Chunk 4 (C4 — constants).** Smallest blast radius (pure constants + a handful of test-literal updates). Lands the `PANEL_FLOOR_SEC` / `DEFAULT_MAX_BEAT_LEN` / flash_cut invariants first so later chunks build on the final thresholds.
2. **Chunk 2 (C2 — image-aware duration).** Low-risk and **backward-compatible** (new optional `image_min` defaults to a no-op). Safe before C1 because it routes through `build_cuts` and `base_min` (default 2.5s) governs as before until wired.
3. **Chunk 1 (C1 — upstream 1:1).** The big structural change (gate merge off, remove the inject band-aid, repurpose tests). Doing it after C2 means C1 preserves C2's call-site edits; doing it after C4 means the floor/flash invariants are already final.
4. **Chunk 3 (C3 — intensity → MLX).** Separable TTS subsystem + a forced re-voice. Cleanest last: under C1's 1:1 each segment is one panel, so per-panel intensity has no beat-max bleed.

**Dependencies (none hard-block):** C1's 1:1 makes C2's per-panel dwell and C3's per-panel intensity cleaner, but each chunk ends with a green full suite on its own. The only cross-chunk test coupling is noted inline (C3's per-panel test uses long lines so it is robust to merge state).

---

## Chunk 1: C1 — upstream 1:1 (stop dropping protected story panels)

> **Spec §3 C1.** The fix is UPSTREAM: keep every shown story panel's own line + `segment_id` + clip end-to-end, so the timeline injector has no work and is removed. Two upstream causes: (1) `merge_short_panel_items` collapses N short panel-lines into one clip; (2) the `redundant` role drops a story panel from a *merged* segment's cuts. Disable the merge → 1:1; the `protected` set already covers the residual.

> **Reconciliation note (verified against live code):** the spec's C1-part-2 ("pass every shown story panel through the `protected` set") is **already satisfied** — `protected_story_files` (`tools/timeline_planner.py:1130`) adds every `panel_kind == "story"` panel to `protected` (`protected = protected_cards | protected_story`, `:1457`), and that set is passed to `build_cuts` at `:1783`. The spec's `:986` anchor is the `choose_kept_scenes(...)` call **inside** `build_cuts` (the *consumer* of `protected`), not the build_cuts *call site* (which is `:1779-1785`). So C1-part-2 needs **no new widening code** — it becomes a regression test (Task 1.2) that pins the invariant. Do **NOT** edit `choose_kept_scenes` (its `tests/test_scene_selection.py:95-147` invariants stand). Also note: for a true single-file (1:1) segment, `build_cuts` early-returns at `:963-964` before selection runs, so the redundant-drop is already moot under 1:1.

### Task 1.1: Gate `merge_short_panel_items` OFF (keep the function)

The merge buckets consecutive short panel-lines into one `segment_id`/clip; the timeline then re-spreads those files as silent cuts under one short segment — the flash. **KEEP the function and its 6 unit tests** (`tests/test_verbatim_script.py:435-514`, which call it directly). Turn it off at the single call point by flipping the default of `tts_merge_short`. `main()` calls `_build_verbatim_section` without passing the kwarg (`tools/script_expander.py:2019-2027`), so flipping the default is the one-line off switch; the `elif tts_merge_short:` guard (`:919`, call `:924`) and the param stay (callers can still opt in).

**Files:**
- Modify: `tools/script_expander.py:856` (`tts_merge_short: bool = True` → `False`)
- Test: `tests/test_verbatim_script.py:533` (convert `test_merge_integration_four_short_lines_fewer_shots` to the 1:1 invariant)

- [ ] **Step 1: Convert the integration test to assert 1:1.** Replace the whole `test_merge_integration_four_short_lines_fewer_shots` function (`tests/test_verbatim_script.py:533-563`) with:

```python
def test_per_panel_no_merge_four_short_lines_four_shots():
    """C1: with the short-line merge gated OFF (default), four 1-word panel lines
    become FOUR shots (strict 1:1), not a merged few. The parallel-list contract
    len(script_paragraphs) == len(shots) == len(tts_paragraphs_v3) holds, and
    every input scene_file is its own shot."""
    panels = [
        {"scene_file": "p1.jpg", "line": "Run."},
        {"scene_file": "p2.jpg", "line": "Dodge."},
        {"scene_file": "p3.jpg", "line": "Strike."},
        {"scene_file": "p4.jpg", "line": "Fall."},
    ]
    chunk = [_panel_beat(1, panels)]
    payload = {"beats": [{"group_id": 1, "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"]}]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="action")

    shots = sec["shots"]
    paras = sec["script_paragraphs"]
    tts = sec["tts_paragraphs_v3"]
    assert len(paras) == len(shots) == len(tts), (
        f"contract broken: paras={len(paras)} shots={len(shots)} tts={len(tts)}"
    )
    assert len(shots) == 4, f"merge must be OFF: expected 4 shots, got {len(shots)}"
    assert [s.get("scene_files") for s in shots] == [
        ["p1.jpg"], ["p2.jpg"], ["p3.jpg"], ["p4.jpg"]]
```

- [ ] **Step 2: Run the converted test — see it FAIL.**

Run: `$V -m pytest tests/test_verbatim_script.py::test_per_panel_no_merge_four_short_lines_four_shots -q`
Expected: FAIL — current default `tts_merge_short=True` still merges the four short lines, so `len(shots)` is `< 4` (assertion `expected 4 shots, got 1`).

- [ ] **Step 3: Flip the default off.** In `tools/script_expander.py`, change the `_build_verbatim_section` signature line `:856`:

```python
    tts_merge_short: bool = False,
```

- [ ] **Step 4: Run the converted test + the full verbatim file — see them PASS.**

Run: `$V -m pytest tests/test_verbatim_script.py -q`
Expected: PASS. The 6 unit tests (`:435-514`, call the function directly) stay green; `test_merge_integration_long_lines_no_merge` (`:566`) stays green (long lines never merged); `test_merge_integration_disabled_by_flag` (`:587`, passes `tts_merge_short=False` explicitly) stays green.

- [ ] **Step 5: Commit.**

```bash
git add tools/script_expander.py tests/test_verbatim_script.py
git commit -m "fix(script): gate short-line TTS merge OFF — strict per-panel 1:1 (C1)"
```

### Task 1.2: Remove `inject_missing_protected` + `pick_protected_inject_segment`; repurpose 5 tests

> **CORRECTION (as-built): the injector was NOT removed — merge-off (Task 1.1) is what restores 1:1 and kills the flash; `inject_missing_protected` is retained to show narration-less system/title cards (avoids `system_card_unshown`), backstopped by `PANEL_FLOOR_SEC=2.0`. The 5 unit tests stay; the converted regression test `test_one_to_one_segments_show_every_story_panel` was added.**

The injector appends a still-missing protected file to ONE existing segment's pick list — that pile-up is the flash. Under 1:1 every shown story panel is born with its own segment + clip and is kept through `choose_kept_scenes` by the `protected` set, so nothing is ever missing and the injector has no work. Remove it and its helper, and replace the 5 tests that pinned it with 2 tests of the new invariant.

**Files:**
- Modify: `tools/timeline_planner.py` — delete `pick_protected_inject_segment` (`:1002-1024`) and `inject_missing_protected` (`:1027-1060`); rewire the call site (`:1665-1692`)
- Test: `tests/test_timeline_selection.py:218-263` (delete the 5 inject tests; add 2 invariant tests)

- [ ] **Step 1: Replace the 5 inject tests with the new invariant tests.** In `tests/test_timeline_selection.py`, delete the block from the comment at `:210` through `test_pick_protected_inject_segment_prefers_smallest_then_latest` (`:259-263`) — i.e. remove these 5 tests:
  - `test_inject_missing_protected_adds_card_to_a_segment` (`:218`)
  - `test_inject_missing_protected_noop_when_already_shown` (`:231`)
  - `test_inject_missing_protected_ignores_non_protected_drops` (`:238`)
  - `test_inject_missing_protected_only_injects_files_in_group_scene_files` (`:250`)
  - `test_pick_protected_inject_segment_prefers_smallest_then_latest` (`:259`)

  and replace them with:

```python
# ---- C1: 1:1 invariant replaces the inject_missing_protected band-aid --------
# The old injector piled a dropped protected card onto ONE segment (the flash).
# Under strict 1:1 every shown story panel is its OWN single-file segment, kept
# through choose_kept_scenes by the protected set, so no panel is ever missing
# from all segments — the injector is removed.

def test_one_to_one_segments_show_every_story_panel():
    # Each shown story panel is its own single-file segment; build_cuts surfaces
    # each panel exactly once across the segments — no panel missing, none extra.
    story_panels = ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"]
    shown = []
    for p in story_panels:
        cuts = tp.build_cuts([p], 4.0, min_cut_sec=3.0,
                             protected={p}, floor=tp.PANEL_FLOOR_SEC)
        assert len(cuts) == 1 and cuts[0]["file"] == p
        shown.append(cuts[0]["file"])
    assert sorted(shown) == sorted(story_panels)


def test_inject_helpers_removed():
    # The band-aid injector is gone — guard against accidental re-introduction.
    assert not hasattr(tp, "inject_missing_protected")
    assert not hasattr(tp, "pick_protected_inject_segment")
```

- [ ] **Step 2: Run the new tests — see `test_inject_helpers_removed` FAIL.**

Run: `$V -m pytest tests/test_timeline_selection.py::test_inject_helpers_removed tests/test_timeline_selection.py::test_one_to_one_segments_show_every_story_panel -q`
Expected: `test_one_to_one_segments_show_every_story_panel` PASSES already; `test_inject_helpers_removed` FAILS with `AssertionError` (the functions still exist on the module).

- [ ] **Step 3: Delete the two functions.** In `tools/timeline_planner.py`, remove `pick_protected_inject_segment` (entire def, `:1002-1024`) and `inject_missing_protected` (entire def, `:1027-1060`), including their docstrings.

- [ ] **Step 4: Rewire the call site.** In `tools/timeline_planner.py`, the FIRST-PASS block (`:1665-1692`) currently computes `_pre_picks` then calls `inject_missing_protected`. Replace the comment block + inject call + `injected_picks_by_sid` (`:1665-1692`) with the direct mapping (no inject), and rename `injected_picks_by_sid` to `picks_by_sid` at its one downstream use (`:1714`):

```python
        # Under strict 1:1 (C1) every shown story panel is born with its own
        # segment + clip and is kept through choose_kept_scenes by the protected
        # set (protected_story_files), so no protected file is ever missing from
        # all segments — the old inject_missing_protected band-aid (which piled a
        # dropped card onto one segment, the flash) has no work left and is gone.
        def _pick_for_segment(srow_: Any) -> List[str]:
            shot_sf = _scene_file_basenames((srow_ or {}).get("scene_files") or [])
            shot_fb = _scene_file_basenames((srow_ or {}).get("fallback_scene_files") or [])
            if not shot_sf:
                return list(scene_files)
            allowed = set(scene_files)
            picked = [f for f in shot_sf if f in allowed]
            if not picked:
                picked = [f for f in shot_fb if f in allowed]
            return picked or scene_files[:1]

        emit_sids = [sid for sid, srow in segments
                     if not (args.mode == "narrated"
                             and is_filler_narration(_safe_str(srow.get("paragraph")) if srow else ""))]
        emit_srows = {sid: srow for sid, srow in segments}
        picks_by_sid: Dict[str, List[str]] = {
            sid: _pick_for_segment(emit_srows.get(sid)) for sid in emit_sids}
```

  Then update the single downstream consumer (`:1714`):

```python
            segment_scene_files = picks_by_sid.get(segment_id)
```

- [ ] **Step 5: Run the timeline tests + full suite check.**

Run: `$V -m pytest tests/test_timeline_selection.py -q`
Expected: PASS — both new tests green; no `AttributeError` from leftover references.

- [ ] **Step 6: Commit.**

```bash
git add tools/timeline_planner.py tests/test_timeline_selection.py
git commit -m "fix(timeline): remove inject_missing_protected band-aid; 1:1 makes it dead (C1)"
```

### Task 1.3: Acceptance test — a beat of N story panels yields N segments (1:1)

Pin the headline invariant: N shown story panels → N paragraphs/shots (→ N `segment_id`s minted downstream), not 1. Use short lines (which the old merge WOULD have collapsed) to prove the merge is off.

**Files:**
- Test: `tests/test_verbatim_script.py` (new test, append after the converted Task 1.1 test)

- [ ] **Step 1: Add the acceptance test.**

```python
def test_n_story_panels_yield_n_segments_one_to_one():
    """C1 headline invariant: a beat of N short story-panel lines produces N
    shots/paragraphs (→ N segment_ids minted in main, one clip each), NOT one
    collapsed segment. segment_id is minted downstream from paragraph index, so
    one paragraph per shown panel == one segment per shown panel."""
    n = 6
    panels = [{"scene_file": f"p{i}.jpg", "line": f"Beat{i}."} for i in range(n)]
    chunk = [_panel_beat(9, panels)]
    payload = {"beats": [{"group_id": 9, "scene_files": [p["scene_file"] for p in panels]}]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="action")
    assert len(sec["shots"]) == n
    assert len(sec["script_paragraphs"]) == n
    assert len(sec["tts_paragraphs_v3"]) == n
    assert [s.get("scene_files") for s in sec["shots"]] == [[p["scene_file"]] for p in panels]
```

- [ ] **Step 2: Run it — see it PASS** (the merge default is already off from Task 1.1).

Run: `$V -m pytest tests/test_verbatim_script.py::test_n_story_panels_yield_n_segments_one_to_one -q`
Expected: PASS. (If run BEFORE Task 1.1 it would FAIL — short lines merge — which is why Task 1.1 lands first.)

- [ ] **Step 3: Commit.**

```bash
git add tests/test_verbatim_script.py
git commit -m "test(script): pin N story panels -> N segments 1:1 acceptance (C1)"
```

### Chunk 1 done — gate

- [ ] Run the full suite: `$V -m pytest -q`
- [ ] Expected: green (no failures). Baseline was 1179 passed / 1 skipped; net test count: −5 deleted inject tests, +2 invariant tests, +1 acceptance test, 1 integration test converted in place ⇒ **1177 passed, 1 skipped** (verify the number; the only acceptable deltas are the ones enumerated here).
- [ ] Confirm no leftover references: `grep -rn "inject_missing_protected\|pick_protected_inject_segment\|injected_picks_by_sid" tools/ tests/` returns nothing.

---

## Chunk 2: C2 — image-aware duration

> **Spec §3 C2.** A panel's on-screen time should respect its visual weight, not only its line length: `duration = max(narration_audio + pad, image_min)`, where `image_min` is a bounded, deterministic function of crop geometry + intensity, computed from data already in the manifests (no model call). Must route through `build_cuts` so `sum(cuts) == duration_sec` (`:1818`) — otherwise `cut_gap`/`total_drift` trip.

> **Reconciliation note (verified against live code):** the spec says geometry comes from `scene_dims` "already in scope at `:1743`". It is **not** — `scene_dims` is written by `tools/render_prep.py:2269-2285`, which runs *after* the planner; `grep -n "scene_dims\|intensity" tools/timeline_planner.py` returns nothing. The reachable sources at planner time are: **geometry** from the vision manifest items' `width`/`height` (the planner already loads `args.vision`; confirmed present, e.g. `1200×822`), and **intensity** from `beat.scene_selection[].intensity` (already read by `_intensity_rank_for_beat`; confirmed present in live beats manifests). This plan sources from those, NOT `scene_dims`.

### Task 2.1: Add a backward-compatible `image_min` floor to `compute_duration_sec`

**Files:**
- Modify: `tools/timeline_planner.py:287-316` (`compute_duration_sec` signature + narrated-audio branch `:301-308`)
- Test: `tests/test_timeline_selection.py` (append near the existing `compute_duration_sec` tests `:284-304`)

- [ ] **Step 1: Write the failing tests.**

```python
def test_compute_duration_image_min_floors_short_audio():
    # A visually heavy panel under a SHORT line: image_min raises the dwell above
    # the audio+pad / base_min floor.
    d = tp.compute_duration_sec(mode="narrated", tts_text="x", overlays=[],
                                base_min=2.5, max_sec=25.0, chars_per_sec=18.0,
                                audio_duration_sec=1.0, audio_pad_sec=0.2,
                                image_min=3.4)
    assert abs(d - 3.4) < 1e-6

def test_compute_duration_image_min_never_truncates_long_audio():
    # A long line still governs — image_min is a FLOOR, never a cap.
    d = tp.compute_duration_sec(mode="narrated", tts_text="x", overlays=[],
                                base_min=2.5, max_sec=25.0, chars_per_sec=18.0,
                                audio_duration_sec=30.0, audio_pad_sec=0.2,
                                image_min=3.4)
    assert abs(d - 30.2) < 1e-6

def test_compute_duration_image_min_default_is_noop():
    # No image_min supplied -> identical to before (backward-compatible).
    d = tp.compute_duration_sec(mode="narrated", tts_text="x", overlays=[],
                                base_min=2.5, max_sec=25.0, chars_per_sec=18.0,
                                audio_duration_sec=1.0, audio_pad_sec=0.2)
    assert abs(d - 2.5) < 1e-6
```

- [ ] **Step 2: Run — see them FAIL.**

Run: `$V -m pytest tests/test_timeline_selection.py -k image_min -q`
Expected: FAIL — `TypeError: compute_duration_sec() got an unexpected keyword argument 'image_min'`.

- [ ] **Step 3: Add the optional arg + floor.** In `tools/timeline_planner.py`, add `image_min` to the keyword-only signature (after `audio_pad_sec`, `:296`):

```python
    audio_pad_sec: float,
    image_min: float = 0.0,
) -> float:
```

  and change the narrated-with-audio branch (`:307-308`) to floor by `image_min`:

```python
        dur = float(audio_duration_sec) + float(audio_pad_sec)
        return float(max(base_min, dur, float(image_min)))
```

  Leave the no-audio/reading branch (`:310-316`) unchanged — `image_min` defaults to 0.0 so the existing `test_compute_duration_still_caps_silent_holds_at_max_sec` (`:298-304`) stays green.

- [ ] **Step 4: Run the new + existing duration tests — see them PASS.**

Run: `$V -m pytest tests/test_timeline_selection.py -k "image_min or compute_duration" -q`
Expected: PASS (the 3 new + the 3 existing at `:284-304`).

- [ ] **Step 5: Commit.**

```bash
git add tools/timeline_planner.py tests/test_timeline_selection.py
git commit -m "feat(timeline): compute_duration_sec image_min floor (backward-compatible, C2)"
```

### Task 2.2: `compute_image_min` deterministic heuristic + `IMAGE_DWELL_CAP`

A bounded, monotonic, RNG-free dwell floor from area + tallness + intensity. Returns the panel floor when geometry and intensity are both unknown (degrades gracefully on old manifests).

**Files:**
- Modify: `tools/timeline_planner.py` (add `IMAGE_DWELL_CAP` constant + `compute_image_min` near `PANEL_FLOOR_SEC`/`compute_duration_sec`, after `:316`)
- Test: `tests/test_timeline_selection.py` (append)

- [ ] **Step 1: Write the failing tests.**

```python
def test_image_min_unknown_geometry_and_intensity_is_floor():
    assert tp.compute_image_min(0, 0, "") == tp.PANEL_FLOOR_SEC

def test_image_min_large_tall_panel_exceeds_small_crop():
    small = tp.compute_image_min(400, 300, "calm")
    big = tp.compute_image_min(1200, 2000, "calm")
    assert big > small >= tp.PANEL_FLOOR_SEC

def test_image_min_intensity_adds_bump():
    calm = tp.compute_image_min(800, 800, "calm")
    explosive = tp.compute_image_min(800, 800, "explosive")
    assert explosive > calm

def test_image_min_is_bounded_by_cap():
    assert tp.compute_image_min(99999, 99999, "explosive") <= tp.IMAGE_DWELL_CAP

def test_image_min_is_deterministic():
    assert tp.compute_image_min(1200, 1600, "intense") == tp.compute_image_min(1200, 1600, "intense")
```

- [ ] **Step 2: Run — see them FAIL.**

Run: `$V -m pytest tests/test_timeline_selection.py -k image_min -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'compute_image_min'` (and `IMAGE_DWELL_CAP`).

- [ ] **Step 3: Implement the constant + helper.** In `tools/timeline_planner.py`, after `compute_duration_sec` (`:316`), add:

```python
# C2: image-aware dwell. A visually heavy panel earns more on-screen time than a
# small reaction crop, and a high-intensity reveal/peak earns a small bump.
# Deterministic (no RNG, no model): same manifest in -> same seconds out.
IMAGE_DWELL_CAP = 4.0   # a quiet splash never stalls the recap past this

_INTENSITY_DWELL_BUMP = {"calm": 0.0, "tense": 0.4, "intense": 0.8, "explosive": 1.2}


def compute_image_min(width: float, height: float, intensity: str,
                      *, floor: float = PANEL_FLOOR_SEC,
                      cap: float = IMAGE_DWELL_CAP) -> float:
    """Per-panel image dwell FLOOR in seconds, in [floor, cap].

    visual_weight blends normalized area (vs a ~1200x1600 reference panel) with a
    tallness term (a tall full-width strip reads as a splash). image_min =
    clamp(floor + (cap-floor)*visual_weight + intensity_bump, floor, cap).
    Returns `floor` when geometry AND intensity are both unknown -> a no-op upstream
    (compute_duration_sec image_min default), preserving old-manifest behavior.
    """
    w = float(width or 0.0)
    h = float(height or 0.0)
    if w > 0.0 and h > 0.0:
        area_term = min(1.0, (w * h) / (1200.0 * 1600.0))
        aspect_tall = max(0.0, min(1.0, (h / w) - 1.0))
        visual_weight = 0.65 * area_term + 0.35 * aspect_tall   # in [0, 1]
    else:
        visual_weight = 0.0
    inten_bump = _INTENSITY_DWELL_BUMP.get(str(intensity or "").strip().lower(), 0.0)
    raw = float(floor) + (float(cap) - float(floor)) * visual_weight + inten_bump
    return float(max(float(floor), min(float(cap), raw)))
```

- [ ] **Step 4: Run — see them PASS.**

Run: `$V -m pytest tests/test_timeline_selection.py -k image_min -q`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tools/timeline_planner.py tests/test_timeline_selection.py
git commit -m "feat(timeline): deterministic compute_image_min heuristic (area+tallness+intensity, C2)"
```

### Task 2.3: Wire dims + intensity into the planner call site (route through `build_cuts`)

Source geometry from the vision manifest and intensity from the beat's per-panel selection, compute `image_min` for the segment's primary panel, and pass it into `compute_duration_sec` (which sits BEFORE `build_cuts`, so the raised `dur` flows into the cuts and `dur = sum(cuts)` at `:1818` adopts the tiled total — alignment preserved).

**Files:**
- Modify: `tools/timeline_planner.py` — add `index_dims_by_file` (near `index_targets_by_file` `:1303`) and `_panel_intensity` helpers; build `dims_by_file` at `:1470`; compute + pass `image_min` at the duration call site (`:1743`)
- Test: `tests/test_timeline_selection.py` (unit tests for the two new helpers; the call-site wiring is exercised by the manual away-run)

- [ ] **Step 1: Write the failing helper tests.**

```python
def test_index_dims_by_file_reads_width_height(tmp_path):
    import json
    vp = tmp_path / "manifest.vision.json"
    vp.write_text(json.dumps({"items": [
        {"scene_file": "a/p1.jpg", "width": 1200, "height": 1600},
        {"scene_file": "p2.jpg", "width": 800, "height": 600},
        {"scene_file": "p3.jpg"},  # no dims -> skipped
    ]}))
    out = tp.index_dims_by_file(str(vp))
    assert out["p1.jpg"] == (1200, 1600)   # basename, ints
    assert out["p2.jpg"] == (800, 600)
    assert "p3.jpg" not in out

def test_panel_intensity_matches_scene_selection():
    beat = {"scene_selection": [
        {"scene_file": "x/p1.jpg", "intensity": "explosive"},
        {"scene_file": "p2.jpg", "intensity": "calm"},
    ]}
    assert tp._panel_intensity(beat, "p1.jpg") == "explosive"
    assert tp._panel_intensity(beat, "p2.jpg") == "calm"
    assert tp._panel_intensity(beat, "missing.jpg") == ""
```

- [ ] **Step 2: Run — see them FAIL.**

Run: `$V -m pytest tests/test_timeline_selection.py -k "index_dims or panel_intensity" -q`
Expected: FAIL — `AttributeError` (`index_dims_by_file`, `_panel_intensity` not defined).

- [ ] **Step 3: Add the helpers.** In `tools/timeline_planner.py`, after `index_targets_by_file` (`:1303-1325`), add:

```python
def index_dims_by_file(vision_path: str) -> Dict[str, "Tuple[int, int]"]:
    """Map scene_file basename -> (width, height) px from the vision manifest, for
    the C2 image-aware dwell floor. Degrades to {} on a missing/old manifest
    (no dims) -> compute_image_min then falls back to the panel floor."""
    out: Dict[str, Tuple[int, int]] = {}
    if not vision_path or not os.path.exists(vision_path):
        return out
    try:
        with open(vision_path, "r", encoding="utf-8") as fh:
            items = json.load(fh).get("items") or []
    except Exception:
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        f = os.path.basename(str(it.get("scene_file") or ""))
        w = it.get("width")
        h = it.get("height")
        if f and w and h:
            out[f] = (int(w), int(h))
    return out


def _panel_intensity(beat: Dict[str, Any], fname: str) -> str:
    """This panel's own understanding intensity (calm|tense|intense|explosive) from
    the beat's per-panel scene_selection. Under 1:1 the segment is one panel, so
    this is the segment's true intensity (no beat-max bleed)."""
    fb = os.path.basename(str(fname or ""))
    for e in (beat.get("scene_selection") or []):
        if isinstance(e, dict) and os.path.basename(str(e.get("scene_file") or "")) == fb:
            return str(e.get("intensity") or "")
    return ""
```

  Ensure `Tuple` is imported (it is used elsewhere; if the `typing` import line lacks it, add `Tuple`).

- [ ] **Step 4: Build `dims_by_file` once.** In `tools/timeline_planner.py`, next to `targets_by_file = index_targets_by_file(args.vision)` (`:1470`), add:

```python
    dims_by_file = index_dims_by_file(args.vision)
```

- [ ] **Step 5: Compute + pass `image_min` at the duration call site.** In `tools/timeline_planner.py`, immediately BEFORE the `dur = compute_duration_sec(` call (`:1743`), insert (using `segment_scene_files`, already computed at `:1714-1722`, and `beat`, in scope):

```python
            # C2: deterministic per-panel image dwell floor for this segment's
            # primary panel (dims from vision width/height; intensity from the
            # beat's own per-panel selection). Routed through compute_duration_sec
            # -> build_cuts so sum(cuts) == duration_sec (no cut_gap/total_drift).
            _seg_primary = segment_scene_files[0] if segment_scene_files else ""
            _seg_w, _seg_h = dims_by_file.get(_seg_primary, (0, 0))
            image_min = compute_image_min(_seg_w, _seg_h,
                                          _panel_intensity(beat, _seg_primary)) \
                if args.mode == "narrated" else 0.0
```

  and add `image_min=image_min,` to the `compute_duration_sec(...)` kwargs (after `audio_pad_sec=args.audio_pad_sec,`):

```python
                audio_pad_sec=args.audio_pad_sec,
                image_min=image_min,
            )
```

- [ ] **Step 6: Run helper tests + timeline file.**

Run: `$V -m pytest tests/test_timeline_selection.py -q`
Expected: PASS (helper tests green; no regression in existing planner tests).

- [ ] **Step 7: Commit.**

```bash
git add tools/timeline_planner.py tests/test_timeline_selection.py
git commit -m "feat(timeline): wire vision dims + per-panel intensity into image_min dwell (C2)"
```

### Chunk 2 done — gate

- [ ] Run the full suite: `$V -m pytest -q`
- [ ] Expected: green. Net vs end-of-Chunk-1: +3 duration tests, +5 `compute_image_min` tests, +2 helper tests (all additive).
- [ ] Note (not a code change): the call-site wiring (Steps 4–5) is exercised end-to-end only by the manual away-run (Plan complete §). The pure pieces (`compute_duration_sec`, `compute_image_min`, `index_dims_by_file`, `_panel_intensity`) are unit-tested here.

---

## Chunk 3: C3 — intensity → MLX TTS delivery

> **Spec §3 C3.** A tense/intense/explosive panel must SOUND different on the production `qwen-mlx` backend. The per-clip `exaggeration` already flows in (`mood_to_exaggeration(tag)` `:821` → `run_guarded_synth` → `synth(text, out, exaggeration)` `:835`/`:473`/`:505`), but `_make_qwen_mlx_synth` ignores it and applies a FIXED env `exag` (`:1408`,`:1428`). Wire the per-clip value in via an affine remap (the MLX scale is centered ~1.4; `mood_to_exaggeration` returns 0.25–0.95, neutral 0.5 — a direct substitution would flatten everything). Also make the upstream tag per-panel (not beat-max).

> **Re-voice (cite the gate):** TTS reuses a clip only when `os.path.exists(audio) AND prior_sha == text_sha` (`local_tts_from_manifest.py:826-827`), with `text_sha = narration_sha(source_text)` over the FULL paragraph **including the leading `[tag]`** (`:820`). So Task 3.3 (tag changes) auto-revoices the affected panels, but Task 3.2 (mapping-only, same tag) does NOT — affected chapters need a forced re-voice (Task 3.4).

### Task 3.1: Affine remap `mlx_exaggeration`

**Files:**
- Modify: `tools/local_tts_from_manifest.py` (add `mlx_exaggeration` near `mood_to_exaggeration` `:154-162`)
- Test: `tests/test_local_tts.py`

- [ ] **Step 1: Write the failing tests.** (Import the module as the file already does; check the top of `tests/test_local_tts.py` for the alias — use whatever it imports the module as, shown here as `ltm`.)

```python
def test_mlx_exaggeration_neutral_maps_to_baseline():
    assert abs(ltm.mlx_exaggeration(0.5) - 1.4) < 1e-6   # mood-neutral -> MLX baseline

def test_mlx_exaggeration_calm_below_baseline():
    assert ltm.mlx_exaggeration(0.30) < 1.4              # calm

def test_mlx_exaggeration_explosive_above_baseline():
    assert ltm.mlx_exaggeration(0.92) > 1.4             # explosive

def test_mlx_exaggeration_monotonic_and_bounded():
    vals = [ltm.mlx_exaggeration(x / 100.0) for x in range(0, 101)]
    assert vals == sorted(vals)                          # non-decreasing
    assert min(vals) >= 0.8 and max(vals) <= 2.0         # bounded
```

- [ ] **Step 2: Run — see them FAIL.**

Run: `$V -m pytest tests/test_local_tts.py -k mlx_exaggeration -q`
Expected: FAIL — `AttributeError: ... has no attribute 'mlx_exaggeration'`.

- [ ] **Step 3: Implement the remap.** In `tools/local_tts_from_manifest.py`, after `mood_to_exaggeration` (`:162`), add:

```python
def mlx_exaggeration(mood_exag: float, *, neutral: float = 1.4,
                     spread: float = 1.2, lo: float = 0.8, hi: float = 2.0) -> float:
    """Affine-remap the 0..1 mood exaggeration (mood_to_exaggeration, ~0.5 neutral)
    onto the MLX expressiveness band centered on `neutral` (STUDIO_MLX_EXAG). MLX
    is centered ~1.4 while the mood scale is ~0.5, so a direct substitution would
    flatten delivery; this keeps neutral lines at the approved baseline while calm
    dips and intense/explosive lifts, bounded to a safe range."""
    val = float(neutral) + (float(mood_exag) - 0.5) * float(spread)
    return float(max(float(lo), min(float(hi), val)))
```

- [ ] **Step 4: Run — see them PASS.**

Run: `$V -m pytest tests/test_local_tts.py -k mlx_exaggeration -q`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tools/local_tts_from_manifest.py tests/test_local_tts.py
git commit -m "feat(tts): mlx_exaggeration affine remap onto MLX expressiveness band (C3)"
```

### Task 3.2: Use the per-clip exaggeration in `_make_qwen_mlx_synth`

**Files:**
- Modify: `tools/local_tts_from_manifest.py:1414-1429` (the `synth` closure inside `_make_qwen_mlx_synth`)

> No new unit test: `_make_qwen_mlx_synth` imports `mlx_audio` at call time (`:1401-1404`), which is not installed in `.eval_venv` (it lives in the per-host `.mlx_venv`), so it can't be exercised in the test venv. Coverage = the Task 3.1 remap unit test + the manual A/B listen (Plan complete §, acceptance #5). Keep the `SynthFn` interface unchanged.

- [ ] **Step 1: Apply the change.** In `tools/local_tts_from_manifest.py`, inside the `synth(text, out_path, exaggeration)` body (before the `generate_audio(` call at `:1424`), replace the fixed `exaggeration=exag` with the per-clip remapped value (using the env `exag` as the neutral center):

```python
        clip_exag = mlx_exaggeration(float(exaggeration), neutral=exag)
        generate_audio(
            text=text, model=model,
            ref_audio=voice_ref, ref_text=(rtext or None),
            output_path=outdir, file_prefix=prefix, audio_format="wav",
            temperature=temp, exaggeration=clip_exag, verbose=False,
        )
```

  Also update the stale docstring note (`:1395-1397`) — the per-mood mapping is now live:

```python
    expressiveness; per-clip mood is remapped onto the MLX band via
    mlx_exaggeration (the voice is approved). mlx-audio writes ``{prefix}_NNN.wav``,
```

- [ ] **Step 2: Sanity-import (no MLX needed).** Confirm the module still imports cleanly in `.eval_venv`:

Run: `$V -c "import tools.local_tts_from_manifest as m; print(m.mlx_exaggeration(0.92))"`
Expected: prints a float > 1.4 (e.g. `1.904`), no ImportError.

- [ ] **Step 3: Commit.**

```bash
git add tools/local_tts_from_manifest.py
git commit -m "fix(tts): qwen-mlx synth uses per-clip exaggeration (remapped), not fixed env exag (C3)"
```

### Task 3.3: Per-panel intensity tag at minting (no beat-max bleed)

`_intensity_rank_for_beat` (`tools/script_expander.py:736-743`) uses the MAX intensity across the beat — one explosive panel makes the whole beat shout. Under 1:1 each paragraph is one panel; read THAT panel's own intensity (fall back to beat-max only when a paragraph has no single scene_file).

**Files:**
- Modify: `tools/script_expander.py` (add `_intensity_rank_for_panel` near `_intensity_rank_for_beat` `:736`; use it in the tag loop `:943-949`)
- Test: `tests/test_verbatim_script.py`

- [ ] **Step 1: Write the failing test.**

```python
def test_per_panel_intensity_no_beat_max_bleed():
    """C3: under 1:1, an explosive panel escalates its OWN tag while a calm panel
    in the same beat does NOT inherit the beat-max intensity."""
    panels = [
        {"scene_file": "p1.jpg", "line": "He waits in the quiet room alone tonight."},
        {"scene_file": "p2.jpg", "line": "The blast tears the entire tower apart now."},
    ]
    beat = _panel_beat(7, panels)
    beat["mood_words"] = []   # -> base tag 'serious' (escalatable) for both
    beat["scene_selection"] = [
        {"scene_file": "p1.jpg", "role": "keep", "intensity": "calm"},
        {"scene_file": "p2.jpg", "role": "keep", "intensity": "explosive"},
    ]
    sec = se._build_verbatim_section(
        section_index=0, chunk=[beat],
        payload={"beats": [{"group_id": 7, "scene_files": ["p1.jpg", "p2.jpg"]}]},
        word_target=120, genre_mode="action")
    tts = sec["tts_paragraphs_v3"]
    assert len(tts) == 2
    t0 = se._split_leading_bracket_tag(tts[0])[0]
    t1 = se._split_leading_bracket_tag(tts[1])[0]
    assert t1 == "excited"      # explosive panel escalates (rank 3)
    assert t0 != "excited"      # calm panel keeps 'serious' — no beat-max bleed
```

  (The lines are >6 words so the test is robust regardless of merge state. Base tag is `serious` because `mood_words=[]` falls through `_ensure_tts_tags_from_beats` to the default `serious`, which `_escalate_tag_for_intensity` escalates to `excited` only at rank ≥ 3.)

- [ ] **Step 2: Run — see it FAIL.**

Run: `$V -m pytest tests/test_verbatim_script.py::test_per_panel_intensity_no_beat_max_bleed -q`
Expected: FAIL — current code uses beat-MAX, so BOTH paragraphs get rank 3 and `t0` is also `excited` (assertion `t0 != "excited"` fails).

- [ ] **Step 3: Add the per-panel rank helper.** In `tools/script_expander.py`, after `_intensity_rank_for_beat` (`:743`), add:

```python
def _intensity_rank_for_panel(beat: Dict[str, Any], fname: str) -> int:
    """This panel's OWN scene_selection intensity as a rank 0..3 (no beat-max
    bleed). Missing/unknown ranks 0."""
    fb = os.path.basename(str(fname or ""))
    for e in beat.get("scene_selection") or []:
        if isinstance(e, dict) and os.path.basename(str(e.get("scene_file") or "")) == fb:
            return _INTENSITY_RANK.get(str(e.get("intensity") or "").strip().lower(), 0)
    return 0
```

  (Confirm `os` is imported at module top — it is, via `import os as _os`; use `_os.path.basename` if `os` is not directly bound. Check the existing import alias and match it.)

- [ ] **Step 4: Use per-panel rank in the tag loop.** In `tools/script_expander.py`, in the loop at `:943-949`, replace the `rank = _intensity_rank_for_beat(beat_for_para)` line with a per-panel read keyed by the paragraph's shot scene_file, falling back to beat-max when there is no single file:

```python
    for i, tp in enumerate(tagged):
        tag, _rest = _split_leading_bracket_tag(str(tp))
        beat_for_para = para_beats[i] if i < len(para_beats) else (chunk[i] if i < len(chunk) and isinstance(chunk[i], dict) else {})
        sf_for_para = (shots[i].get("scene_files") or [None])[0] if i < len(shots) else None
        rank = (_intensity_rank_for_panel(beat_for_para, sf_for_para)
                if sf_for_para else _intensity_rank_for_beat(beat_for_para))
        if shout_flags[i]:
            rank = max(rank, 2)
        tag = _escalate_tag_for_intensity(tag or "serious", rank)
```

  (`shots` is populated earlier in the same function, in the items loop at `:937`, so `shots[i]` is available here.)

- [ ] **Step 5: Run the test + full verbatim file — see them PASS.**

Run: `$V -m pytest tests/test_verbatim_script.py -q`
Expected: PASS (new test green; existing escalation/merge tests unaffected).

- [ ] **Step 6: Commit.**

```bash
git add tools/script_expander.py tests/test_verbatim_script.py
git commit -m "fix(script): per-panel intensity tag at minting, no beat-max bleed (C3)"
```

### Task 3.4: Force re-voice procedure (documented, not a code change)

A mapping-only change (Task 3.2) does NOT change `text_sha`, so cached clips are reused (stale). Task 3.3 changes the leading `[tag]` for some panels → `text_sha` differs → those auto-revoice, but a chapter already at `planned/voiced` must be reset for clips to regenerate.

- [ ] **Step 1: Document the reset in the chunk gate / handover** (no source edit): for each affected chapter, reset `voiced/planned → scripted` and clear `clips/` + `tts_index.json` (+ any `align`), then re-run voiced → planned. Matches [[per-group-tts-alignment-shipped]] (re-voicing a planned chapter skips `voiced` unless reset + cleared) and [[mlx-qwen-tts-evaluation]] (per-host `.mlx_venv`). This is part of the manual away-run (Plan complete §). No test.

### Chunk 3 done — gate

- [ ] Run the full suite: `$V -m pytest -q`
- [ ] Expected: green. Net vs end-of-Chunk-2: +4 `mlx_exaggeration` tests, +1 per-panel-intensity test (additive).
- [ ] Confirm import health: `$V -c "import tools.local_tts_from_manifest, tools.script_expander"` exits 0.

---

## Chunk 4: C4 — backstop constants

> **Spec §3 C4.** Rare safety nets under 1:1, not the pacing mechanism. `PANEL_FLOOR_SEC` 1.2→2.0; `DEFAULT_MAX_BEAT_LEN` 8→6; keep the documented invariant by bumping the flash_cut gate to 2.0. (Verified: `base_min_sec` default is `2.5` ≥ 2.0, so legit single-panel 1:1 cuts — which take the full audio dur ≥ base_min — never trip the bumped gate. The two existing flash_cut tests use 0.8s/0.3s cuts, both still `< 2.0`, so they stay green unchanged.)

### Task 4.1: `PANEL_FLOOR_SEC` 1.2 → 2.0

**Files:**
- Modify: `tools/timeline_planner.py:927`
- Test: `tests/test_timeline_selection.py` (add a constant-pin test)

- [ ] **Step 1: Write the failing test.**

```python
def test_panel_floor_is_two_seconds():
    # C4: the per-panel cut floor backstop is 2.0s (coupled to prep_qa flash_cut).
    assert tp.PANEL_FLOOR_SEC == 2.0
```

- [ ] **Step 2: Run — see it FAIL.**

Run: `$V -m pytest tests/test_timeline_selection.py::test_panel_floor_is_two_seconds -q`
Expected: FAIL — `assert 1.2 == 2.0`.

- [ ] **Step 3: Bump the constant.** In `tools/timeline_planner.py:927`:

```python
PANEL_FLOOR_SEC = 2.0   # keep == prep_qa flash_cut threshold
```

- [ ] **Step 4: Run the pin test + the floor test (literal arg, unaffected).**

Run: `$V -m pytest tests/test_timeline_selection.py::test_panel_floor_is_two_seconds tests/test_timeline_floor.py -q`
Expected: PASS (`test_timeline_floor.py` passes the floor as a literal `1.2` arg, so it is independent of the constant).

- [ ] **Step 5: Commit.**

```bash
git add tools/timeline_planner.py tests/test_timeline_selection.py
git commit -m "chore(timeline): PANEL_FLOOR_SEC 1.2 -> 2.0 backstop (C4)"
```

### Task 4.2: `DEFAULT_MAX_BEAT_LEN` 8 → 6

Grouping no longer drives pacing (it's soft context); a smaller span = tighter continuity. Update ONLY the two `== 8` assertions; `:75` references the constant symbolically (auto-adapts) and `:377`/`:379` pin a literal-8 `max_beat_len=8` arg (still correct).

**Files:**
- Modify: `tools/story_group.py:42`
- Test: `tests/test_story_group.py:69` and `:384` (`== 8` → `== 6`)

- [ ] **Step 1: Update the two assertions to the new default.** In `tests/test_story_group.py`, line `:69`:

```python
    assert sg.DEFAULT_MAX_BEAT_LEN == 6
```

  and line `:384`:

```python
    assert sg.DEFAULT_MAX_BEAT_LEN == 6
```

- [ ] **Step 2: Run — see them FAIL.**

Run: `$V -m pytest tests/test_story_group.py -k "default_max_beat_len or default_cap_is_tighter" -q`
Expected: FAIL — `assert 8 == 6` (constant still 8).

- [ ] **Step 3: Bump the constant.** In `tools/story_group.py:42`:

```python
DEFAULT_MAX_BEAT_LEN = 6
```

- [ ] **Step 4: Run the story_group file — see it PASS.**

Run: `$V -m pytest tests/test_story_group.py -q`
Expected: PASS — `:69`/`:384` now match; `:75` (`<= sg.DEFAULT_MAX_BEAT_LEN`, symbolic) auto-adapts to 6; `test_oversized_beat_splits_at_cap` (`:377`/`:379`, explicit `max_beat_len=8` arg + `<= 8`) stays correct and green.

- [ ] **Step 5: Commit.**

```bash
git add tools/story_group.py tests/test_story_group.py
git commit -m "chore(story_group): DEFAULT_MAX_BEAT_LEN 8 -> 6 (soft context window, C4)"
```

### Task 4.3: Couple the prep_qa flash_cut threshold to 2.0

Keep the `PANEL_FLOOR_SEC` comment's invariant ("keep == prep_qa flash_cut threshold") true: a sub-2.0 non-held cut becomes a flagged defect (under 1:1 it should essentially never occur legitimately).

**Files:**
- Modify: `tools/prep_qa.py:1274`
- Test: `tests/test_prep_qa.py` (add a boundary test)

- [ ] **Step 1: Write the failing boundary test.** Append near the existing flash_cut tests (`tests/test_prep_qa.py:1340-1361`):

```python
def test_flash_cut_threshold_is_coupled_to_two_seconds():
    # C4: a 1.5s non-held cut sits in the new 1.2-2.0 band -> must now flag
    # (it did NOT at the old 1.2 threshold), keeping floor == flash_cut threshold.
    plan = _plan([_item("g0001_p01", ["p.jpg"], dur=1.5)])
    flags = pq.plan_flags(plan, clean_files={"p.jpg"}, audio_exists=lambda p: True)
    assert any(f["code"] == "flash_cut" for f in flags)
```

- [ ] **Step 2: Run — see it FAIL.**

Run: `$V -m pytest tests/test_prep_qa.py::test_flash_cut_threshold_is_coupled_to_two_seconds -q`
Expected: FAIL — at threshold `1.2`, `1.5` is not `< 1.2`, so no flash_cut flag (`assert any(...)` fails).

- [ ] **Step 3: Bump the gate.** In `tools/prep_qa.py:1274`:

```python
            if dur < 2.0 and not c.get("held"):
```

- [ ] **Step 4: Run the prep_qa flash_cut tests — see them PASS.**

Run: `$V -m pytest tests/test_prep_qa.py -k flash_cut -q`
Expected: PASS — the new boundary test plus the existing `test_plan_flags_missing_file_dims_audio_and_flash_cut` (0.8s, still `< 2.0`), `test_flash_cut_is_blocking_error` (0.3s severity), and `test_flash_cut_held_card_is_not_flagged` (held, exempt) all green.

- [ ] **Step 5: Commit.**

```bash
git add tools/prep_qa.py tests/test_prep_qa.py
git commit -m "chore(prep_qa): flash_cut threshold 1.2 -> 2.0 coupled to PANEL_FLOOR_SEC (C4)"
```

### Chunk 4 done — gate

- [ ] Run the full suite: `$V -m pytest -q`
- [ ] Expected: green. Net vs baseline for this chunk: +1 floor-pin, +1 flash_cut boundary test; two `== 8` assertions edited in place.
- [ ] Confirm the invariant comment is honest: `grep -n "flash_cut threshold" tools/timeline_planner.py` and `grep -n "dur < 2.0" tools/prep_qa.py` both present.

---

## Plan complete

**Saved to** `docs/plans/2026-07-01-perpanel-1to1-narration.md`.

### Automated gate (must pass before any away-run)

- [ ] **Full suite green:** `$V -m pytest -q` — no failures, only the enumerated test deltas (C1: −5 inject tests, +2 invariant, +1 acceptance, 1 converted; C2: +10 additive; C3: +5 additive; C4: +2 additive, 2 edited-in-place).
- [ ] **No leftover references:** `grep -rn "inject_missing_protected\|pick_protected_inject_segment" tools/ tests/` returns nothing.
- [ ] **`segment_id` contract intact:** the mint stays at `tools/script_expander.py:2168`; `grep -rn "g{.*:04d}_p{" tools/timeline_planner.py` returns nothing (no timeline minting).

### Manual validation (deferred to the away-run on the Mini — spec §6)

Regenerate derived manifests first (a narration/duration change is downstream of grouping): delete `understood.json` and re-run grouped → beated → scripted → voiced → planned so the new 1:1 narration + durations + per-panel tags propagate (the manifest-freshness guardrail will otherwise block, [[manifest-freshness-guardrail]]). For C3, force re-voice affected chapters (Task 3.4: reset `voiced/planned → scripted`, clear `clips/` + `tts_index.json`). Then on a real chapter confirm:

- [ ] **True 1:1** — emitted segment count == count of SHOWN `story` panels (caption/empty/chrome folded or dropped, not counted); no story panel without its own `segment_id` + `clips/{segment_id}.wav`.
- [ ] **Zero `flash_cut`** in prep_qa (at the 2.0 threshold).
- [ ] **Zero `empty_bubble_shown`**, **zero `cut_gap`/`total_drift`**, **zero `missing_audio`** on the same chapter.
- [ ] **Niche A + C still applied** (bubble TEXT cleaned / shape kept, never inpainted; system cards never cleaned; empty bubbles never shown — spec §4, unchanged).
- [ ] **Intensity audibly differs on qwen-mlx** — A/B a calm vs intense vs explosive segment (run the per-host `.mlx_venv`; `STUDIO_MLX_EXAG` is the neutral center). 
- [ ] **Listen pass** — a human listen confirms no flashes, panels dwell with their own line, delivery tracks the scene.

### Execution handoff

Ready to execute? Use **subagent-driven-development** (fresh subagent per task + two-stage review). Recommended order: **Chunk 4 → Chunk 2 → Chunk 1 → Chunk 3** (justified above).

---

## Appendix: anchor reconciliation (live code vs spec, verified 2026-07-01)

All spec anchors were verified against `main` (HEAD `428f3b8`). Matches confirmed except:

1. **C1-part-2 protected set** — `protected_story_files` (`timeline_planner.py:1130`, wired into `protected` at `:1457`, passed to `build_cuts` at `:1783`) **already** adds every `panel_kind=="story"` panel to the protected set. The spec's `:986` anchor is the `choose_kept_scenes(...)` call *inside* `build_cuts` (the consumer), not the build_cuts *call site* (`:1779-1785`). ⇒ No new widening code; Task 1.2 ships a regression test instead.
2. **C1 merge gate-off breaks an unlisted test** — flipping `tts_merge_short` default off makes `test_merge_integration_four_short_lines_fewer_shots` (`tests/test_verbatim_script.py:533`) fail (it relied on default-on merge). The spec's "6 unit tests stay green" list (`:433-514`) did not cover the 3 integration tests (`:517-605`). Task 1.1 converts `:533` to the 1:1 invariant; `:566` and `:587` stay green. (`main()` at `:2019` does not pass the kwarg, so the default flip is the single off switch; the `elif tts_merge_short:` guard is at `:919`, call `:924`.)
3. **C2 `scene_dims` not in planner scope** — the spec says geometry is "already in scope at `:1743`" via `scene_dims`; it is not. `scene_dims` is written by `render_prep.py:2269-2285` *after* the planner (`grep "scene_dims\|intensity" tools/timeline_planner.py` → empty). Reachable instead: geometry from vision items' `width`/`height` (planner loads `args.vision`); intensity from `beat.scene_selection[].intensity`. Task 2.3 sources from those.
4. **Minor offsets** — `compute_duration_sec` signature is `:287-297`, narrated-audio branch `:301-308` (not the single `:287-297` line); `_intensity_rank_for_beat` is `:736-743`; qwen-mlx fixed `exag` is `:1408`, generate call `:1428`, synth def `:1414`. `base_min_sec` default = `2.5` (`timeline_planner.py:1372`) — confirms the C4 flash_cut bump is safe for legit 1:1 cuts. All consistent with the spec's intent.
