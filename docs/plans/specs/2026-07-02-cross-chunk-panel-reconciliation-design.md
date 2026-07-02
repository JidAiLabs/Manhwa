# Cross-Chunk Panel Seam Reconciliation — Design Spec

**Date:** 2026-07-02
**Status:** Design (no code)
**Goal:** Detect and re-assemble panels that a chunk seam bisected into two (or more) near-duplicate slices, upstream of vision/understanding/narration, so the `cross_dup` QA flag drops to ~0 and the same drawing never appears twice in a recap — without touching the narration pipeline.

---

## 1. Problem

### 1.1 Mechanism

The stitcher (`tools/chunk_stitch_adaptive.py`) glues the long webtoon strip into vertical **chunks** capped at `--max-chunk-height` (default `16000`, `chunk_stitch_adaptive.py:323`) with a hard ceiling of `max_chunk_height + --max-overflow-px` (default `+6000` → ~`22000`, `:334`, `:459-460`). When it cannot find a safe gutter band before the cap it forces a cut mid-content (`:493-501`). Panel detection then runs **per chunk** — `studio/detect/yolo_panels.py:detect_panels` (`:186`) feeds each `stitch_chunks/chunk_*.jpg` to YOLO independently and emits normalized boxes into `manifest.panels.json`.

A panel taller than the remaining space in chunk *N* is therefore cut by the forced chunk boundary. YOLO, seeing only one chunk at a time, detects:

- the **bottom slice** as the LAST panel of chunk *N* (box `ymax ≈ 1.0`, i.e. touching the forced bottom edge), and
- the **top slice** as the FIRST panel of chunk *N+1* (box `ymin ≈ 0.0`, touching the top edge).

These become two separate scene crops of one drawing → two cuts in the montage → the same art shown twice → the `cross_dup` ERROR (`tools/prep_qa.py:295`, `:308`).

### 1.2 Proven evidence (Nano ch1)

All 6 `cross_dup` pairs on Nano ch1 straddle a chunk seam and are **contiguous in global-y**, e.g.:

- `p000015` (chunk_0002) bottom at global-y ≈ **22050**, touching the ~**12520px** chunk bottom edge, and
- `p000016` (chunk_0003) top at global-y ≈ **22056**, box `ymin = 0`.

Δ ≈ 6px — the two boxes abut in stacked global-y. This is **not** an artist zoom/blow-up and **not** an aesthetic near-dup: it is one panel bisected by the chunk cut. A *clean* gutter cut lands the panel's `ymax` **before** the chunk edge, so it will never match the detector below; only genuinely-bisected panels do.

### 1.3 Two geometry facts the fix must honor (verified in source)

1. **Scene crops are materialized strictly per-chunk.** `tools/panels_to_scenes.py` crops with `chunk_im.crop(box_xyxy)` (`:914`) — a single chunk image. A merged cross-chunk panel spans **two** chunk JPGs and has **no single-chunk `panels_norm` representation**. (This contradicts the "merge boxes between detect and scened" assumption — see §4.)
2. **Consecutive chunks overlap.** Every flush carries a tail of `--overlap-px` (default `700`, `chunk_stitch_adaptive.py:331`) into the next chunk (`:428`, `:490`, `:498`, `:530`). So the bottom ~700px of chunk *N* and the top ~700px of chunk *N+1* are **the same source pixels**. The two slices of a bisected panel therefore share a ~700px duplicated band. Note the mechanism precisely: `cross_dup` fires via `tools/render_prep.multi_scale_contained` **template matching** (`prep_qa.py:306`), **not** `dhash` — `dhash64` drives the separate `panels_to_scenes --dedupe` path, not `cross_dup`. The shared band also pushes the two slices' `dhash64` toward a low Hamming distance, but since ~700px is only ~14% of a multi-thousand-px crop downscaled to a 9×8 hash, dhash here is **corroborative, not causal**. Reassembly must account for this band (§5).

Also note: `chunk_global_y0` is computed by **naive summation of full chunk heights** (`panels_to_scenes.py:779-792`, `global_y += h`), i.e. it does **not** subtract the overlap. This is internally consistent (every downstream reader uses the same naive stack), so the detector uses the same naive coordinate; it must not try to reconcile naive global-y against "true" source-y.

---

## 2. Architecture

Insert one **seam-reconciliation** step in the data flow, strictly upstream of vision:

```
stitch → detect (per-chunk YOLO) → expand-to-gutters → scenes (per-chunk crops)
        → [RECONCILE cross-chunk seams] ← NEW
        → vision → understanding → grouping → beats → narration → script → tts → plan → render
```

Because the step sits **before `visioned`**, everything from OCR onward simply receives the corrected, smaller panel set. The merged panel is one scene with one `panel_id`, so it flows through understanding → grouping → narration exactly like any other panel and receives its single per-panel narration line. The 1:1 panel→narration mechanism, niche selection, and intensity escalation are **unchanged** — they never learn a seam existed.

The step is a **new tool** (`tools/reconcile_seam_panels.py`, subprocess) that reads `manifest.scenes.json` + the slice JPGs and rewrites both in place. It runs after `panels_to_scenes` (which already computes `chunk_global_y0`, `chunk_h`, `box_px_xyxy`, `w`, `h`, `dhash64` and writes slice crops **aligned by construction**), so all inputs the detector and reassembler need are on disk.

---

## 3. The detector

Operate on the finished `scenes[]` records (schema confirmed against `ongoing/.../Chapter_1/manifest.scenes.json`): each carries `chunk_file`, `chunk_h`, `chunk_global_y0`, `panel_index_in_chunk`, `part_index`, `recovered`, `box_px_xyxy` (`[x0,y0,x1,y1]` in **chunk** pixels), `w`, `h`, `dhash64`, `out_file`, `split`.

Order chunks by their stitch position (`chunk_file` sort / `chunk_index`). A **seam-bisected pair** is:

Let `N` = a chunk, `N+1` = the next chunk in stitch order.
Let `A` = the scene in `N` with the **largest** `box_px_xyxy.y1` (its bottommost panel), `B` = the scene in `N+1` with the **smallest** `box_px_xyxy.y0` (its topmost panel).

Match when ALL hold:

1. **A touches the forced bottom edge:** `chunk_h[N] - A.y1 ≤ EDGE_TOL_PX`.
2. **B touches the top edge:** `B.y0 ≤ EDGE_TOL_PX`.
3. **Contiguous in stacked global-y:**
   `| (A.chunk_global_y0 + A.y1) - (B.chunk_global_y0 + B.y0) | ≤ SEAM_TOL_PX`.
   (Because global_y0 is naive, `B.chunk_global_y0 = A.chunk_global_y0 + chunk_h[N]`, so this reduces to conditions 1+2; keep it as the explicit robust conjunction and to reject accidental non-adjacent pairs.)
4. **Loose `dhash64` veto (safety, not a trigger):** REJECT an otherwise-matched pair only when `hamming64(A.dhash64, B.dhash64) > DHASH_VETO` (e.g. `> 20`). Because the two slices share the ~700px overlap band, a genuinely-bisected tall panel's halves sit **well under** 20 — only the rare case where the forced cut landed cleanly *between two distinct panels* (A and B are different drawings that merely abut the edge) exceeds it. This is a **veto-only** guard on a HIGH distance: it can *prevent* a false merge but never *trigger* one, and it must **not** block a tall panel whose halves merely differ moderately (do not veto at a low or medium distance). The geometric conjunction (cond. 1–3) is the sole trigger.

**Rationale for "touches the forced edge":** a clean gutter cut ends a panel *before* the chunk edge, so `A.y1` is well short of `chunk_h[N]` and condition 1 fails. Only a panel the cut ran *through* reaches the edge. This is the primary false-merge guard.

### 3.1 Chain handling (3+ chunks)

A very tall panel can span three or more chunks. The middle chunk(s) then contain a **single** panel whose box touches **both** the top edge (`y0 ≈ 0`) and the bottom edge (`y1 ≈ chunk_h`). Detection is transitive: build seam links pairwise (N↔N+1, N+1↔N+2, …) and take the **connected component**. A chunk qualifies as a pure "pass-through" middle link when its sole/relevant panel satisfies both edge conditions. Merge the entire chain into one panel in a single pass.

### 3.2 Tolerances (initial values, tune on Nano ch1 fixture)

- `EDGE_TOL_PX` ≈ `24` (matches the trim/gutter margins already used in `panels_to_scenes`); expressed in chunk pixels.
- `SEAM_TOL_PX` ≈ `EDGE_TOL_PX * 2` (Δ observed = 6px, so generous).
- `DHASH_VETO` ≈ `20` (veto only — reject a match ABOVE this Hamming distance; never used to trigger a merge).

Tolerances should be small **absolute** pixel values, not fractions — the seam is a hard geometric coincidence, and loose tolerances re-introduce false merges (§8).

---

## 4. Placement (and why it is zero-impact on narration)

**Recommended (lowest blast radius):** append a second `_run_tool("reconcile_seam_panels.py", …)` call **inside `_stage_scened`**, immediately after the existing `panels_to_scenes.py` invocation (`studio/pipeline.py:157-166`). The reconcile tool rewrites `manifest.scenes.json` and the `scenes/` dir in place. Because it is a **new tool run as a subprocess**, it is fresh on `git pull` — **no daemon restart** required (per the repo deploy note: tools + `pipeline.py` subprocess calls are fresh; only `studio/worker.py`/`studio/dashboard/**` need `launchctl kickstart -k`). Crucially, this adds **no new catalog status**, so `STATUS_ORDER` (`studio/catalog/models.py:3-4`), the worker, and the dashboard are untouched.

**Alternative (visible in the state machine):** insert a new status `"reconciled"` between `"scened"` and `"visioned"` in `STATUS_ORDER` (`models.py:3-4`) and a matching `_stage_reconcile` entry in `_STAGE_TABLE` between `("scened", …)` and `("visioned", …)` (`studio/pipeline.py:434-435`). Cleaner provenance/dashboard visibility, but it ripples into the state machine, worker, and dashboard → needs a daemon restart. **Not recommended** unless per-stage visibility is wanted.

**Why NOT "merge boxes between detect and scened":** the box-level manifests (`manifest.panels.json`, `manifest.panels.expanded.json`) are **per-chunk** (`{"chunks":[{"chunk_file","panels_norm":[[ymin,xmin,ymax,xmax],…]}]}`), and `panels_to_scenes` crops one chunk at a time (`panels_to_scenes.py:914`). A cross-chunk panel cannot be expressed there without teaching the cropper cross-chunk cropping. The post-materialization scene level is where the two aligned slice images and their global-y metadata already exist together — the natural seam.

**Why this is zero narration impact:** vision reads the scenes **directory** (`vision_extract.py --scenes-dir`, `studio/pipeline.py:169-175`), not the box manifests; understanding/grouping/beats/punchup/script all key off scene `out_file`/`panel_id`. Reconciliation finishes before any of them run, so they only ever see the corrected set. No narration code changes; the merged panel gets exactly one narration line by the normal 1:1 path.

---

## 5. The reassembly

Replace the N slice records with **one** merged panel:

- **Image:** rebuild ONE crop that spans the whole panel. Two equivalent methods (the tool should pick one; re-crop is more robust):
  - **Re-crop from chunk images (preferred):** open chunk *N* and *N+1* and re-crop the **exact global-y range** of the union straight from the chunk pixels — take `A`'s column down to the forced cut line `chunk_h[N]`, then continue from chunk *N+1* **below the duplicated overlap band**. Do **not** trim a flat `OVERLAP_PX` from `B`'s top: `B`'s box may begin at `B.y0 > 0` (up to `EDGE_TOL_PX`), so a flat trim leaves a ≤`EDGE_TOL_PX` (~24px) sliver of the repeated band — or gouges a gap. The exact seam offset is **`OVERLAP_PX − B.y0`** trimmed from `B`'s slice top (equivalently: having taken `A` down to `chunk_h[N]`, append `B` starting at source-y `OVERLAP_PX`, i.e. drop `OVERLAP_PX − B.y0` from the top of `B`'s crop). Read `OVERLAP_PX` from `manifest.stitch.json → adaptive.overlap_px` (default `700`). For a 3+ chunk chain, stack all slices trimming each interior seam by its own `OVERLAP_PX − y0` offset. The slices share the same width and source column, so they align by construction.
  - **Concat existing slice JPGs:** same as above but starting from the already-written `out_file` crops. Cheaper, but the slice JPGs were content-trimmed/split independently, so their heights may not line up cleanly at the seam; re-crop is safer.
- **New record fields:**
  - `box_px_xyxy` / `box_norm`: the union in the frame of the **top** chunk is not meaningful across chunks; store the merged crop's own dimensions and set `box_px_xyxy` to the reassembled rectangle relative to its origin chunk (document that a merged panel's box spans a seam). Keep `chunk_file`/`chunk_global_y0` = the **top** slice's (chunk *N*) so global ordering stays monotonic.
  - `w`, `h`: recompute from the reassembled image (two-slice case: `h` ≈ `A.h + (B.h − (OVERLAP_PX − B.y0))`, which simplifies to `A.h + B.h − OVERLAP_PX` when `B.y0 = 0`).
  - `dhash64`: recompute from the merged image (`panels_to_scenes.py:269 dhash64`).
  - `panel_index_in_chunk`: keep `A`'s; add a marker (`"reconciled_seam": true` and `"merged_from": [<A.panel_id>, <B.panel_id>, …]`) for provenance/QA. `reconciled_seam` is **load-bearing**: it is the flag `prep_qa` reads to exempt this (necessarily tall) panel from the `chunk_as_panel` height gate — see §5.1. It must survive into whatever record `prep_qa` inspects (scene record and/or plan `cut`).
  - `part_index`, `split`, `recovered`, `trim`: carry sensible values (`recovered=false`; `split.enabled` may be set false; provenance in the new marker).
- **File hygiene:** write the merged JPG into `scenes/` and **delete the orphan slice JPGs** for the merged members (vision globs `*.jpg`, so a stale slice would still be OCR'd and re-appear). Keep `out_file` naming consistent with the existing scheme so downstream references resolve.
- **Manifest:** update `count_scenes`, `stats`, and re-write `scenes[]` in stitch order.

`panel_id` sequencing: the simplest, least-surprising choice is to keep `A`'s `panel_id` for the merged panel and drop the others; downstream keys by `out_file`/`panel_id` and nothing depends on a contiguous sequence.

### 5.1 Downstream gate: `chunk_as_panel` exemption (required)

A correctly reassembled seam panel is **tall by definition** — it is exactly the panel the chunk cap was too short to hold. Merging two individually-under-8000px slices can therefore produce a merged scene with `h > 8000`, which trips `prep_qa.py`'s `chunk_as_panel` height check (`tools/prep_qa.py:187`, a **BLOCKING ERROR**) and would park the chapter — flagging *correct* reassembly as under-segmentation, the exact opposite of the truth. This makes the exemption a first-class part of the design, not an afterthought.

Two required parts:

1. **Reconcile stamps the marker.** Every merged scene MUST carry `reconciled_seam: true` (§5) and keep its true merged height (no height clamping to dodge the gate). The marker must survive into whatever record `prep_qa` inspects (scene record and/or the plan `cut`), so it is available at QA time.
2. **`prep_qa` exempts marked panels.** The `chunk_as_panel` height check (~`prep_qa.py:187`, `h > 8000`) MUST **exempt** any scene/cut carrying `reconciled_seam: true`. For those, tallness is expected and correct — a properly reassembled single panel — not the tall-strip under-detection / chunk-as-panel case the check exists to catch. Every non-reconciled panel is still subject to the check unchanged.

**Changeset impact:** this adds `tools/prep_qa.py` to the changeset alongside the new `tools/reconcile_seam_panels.py`. `prep_qa.py` is a **tool run as a subprocess**, so it is fresh on `git pull` — still **no daemon restart** required (only `studio/worker.py` / `studio/dashboard/**` need `launchctl kickstart -k`).

---

## 6. Acceptance

1. **Unit — seam detector.** Given a fixture of Nano ch1 `scenes[]` records (real `box_px_xyxy` + `chunk_h` + `chunk_global_y0`), the detector identifies exactly the **6** seam pairs and **no** others (in particular, it must not flag any panel that merely ends near a *gutter* short of the chunk edge). Include a negative fixture: a tall panel ending 300px before `chunk_h` (clean cut) → not matched.
2. **Unit — chain.** A synthetic 3-chunk panel (middle chunk box touches both edges) → one connected component of size 3 → single merged panel.
3. **Unit — reassembly.** Two aligned slice arrays with a known `OVERLAP_PX` band → one contiguous image of height `A.h + B.h - OVERLAP_PX`, no duplicated band, recomputed `w`/`h`/`dhash64`.
4. **Integration — reprocess Nano ch1.** Panel count drops by **(total seam slices − number of reconciled panels)** — i.e. the number of *extra* slices removed, **not** necessarily 6: a two-chunk seam collapses 2→1 (−1) while a 3-chunk chain collapses 3→1 (−2). For Nano ch1's 6 seam pairs this is *approximately* `112 → ~106`, but the exact figure depends on how many are chains. Regenerate the plan and run `prep_qa` and assert:
   - `cross_dup` → ~0 (the merges land), **and**
   - **no new `chunk_as_panel`** flag on the reconciled (tall) panels — i.e. the `reconciled_seam` exemption (§5.1) works, **and**
   - **no other new flags** (`montage_degenerate`, `blank_crop`, etc.).
   Spot-check the merged crops visually (contiguous single panel, no repeated overlap band, no seam sliver/gap).
5. **Full suite green:** `.eval_venv/bin/python -m pytest -q` (existing ~1198 tests) plus the new unit tests.

---

## 7. Non-goals

- **Do not** touch the stitch cut heuristics or the `max_chunk_height` / overflow cap (`chunk_stitch_adaptive.py`). Seams are inevitable on tall panels; the fix reconciles after the fact rather than trying to never cut mid-panel.
- **Do not** touch any narration-side code (understanding, grouping, beats, punchup, script, TTS, plan). The step is purely a scene-set correction upstream of vision.
- **Do not** repurpose this to dedup legitimately distinct panels (artist zoom/blow-up pairs, repeated establishing shots). Those do **not** touch a forced chunk edge and must be left to the existing perceptual dedup (`panels_to_scenes.py --dedupe`) and the `cross_dup` QA judgment. This step merges only the geometric seam case.

---

## 8. Risks & mitigations

- **False merge (biggest risk).** Guarded by the conjunction: *touches the forced edge* (§3 cond. 1+2) **AND** *contiguous stacked global-y* (cond. 3), with a *loose `dhash64` veto* (cond. 4) that rejects a match only on a **HIGH** Hamming distance (the rare forced-cut-between-two-distinct-panels case). A panel that merely ends near an internal gutter fails cond. 1 (`A.y1` short of `chunk_h`) and is never merged. Keep tolerances small and absolute.
- **New `chunk_as_panel` false-positive on the merged (tall) panel.** A reassembled seam panel is tall by design and would trip `prep_qa.py`'s `h > 8000` `chunk_as_panel` BLOCKING ERROR, parking the chapter. Mitigated by stamping `reconciled_seam: true` + the `prep_qa` exemption (§5.1); without *both*, correct reassembly reads as under-segmentation.
- **Overlap double-exposure.** Naive vertical concat would duplicate the shared ~700px band → visibly repeated art strip inside the merged crop. Mitigate by trimming `OVERLAP_PX` (read from `manifest.stitch.json → adaptive.overlap_px`) on each interior seam, or by re-cropping the true union (§5).
- **Chains.** A panel spanning 3+ chunks must merge as one connected component, not as two independent pairwise merges that leave a middle slice orphaned (§3.1).
- **Existing rendered chapters** were materialized before this step and won't benefit until reprocessed. Reprocessing from `scened` (delete `manifest.scenes.json` + `scenes/` and re-run, which also invalidates the downstream `understood.json`/`groups`/`beats`) is required to gain the fix. Document this in the run procedure; do not silently assume already-shipped chapters are clean.
- **Downstream count tolerance.** Consumers of `manifest.scenes.json` (`tools/render_prep.py`, `tools/prep_qa.py`, `tools/teaser_planner.py`, `tools/narration_punchup.py`, `tools/thumbnail_gen.py`, `studio/qa.py`, `studio/qa_flags.py`, `studio/worker.py`) key off `out_file`/`panel_id` and none hardcode a panel count, so a reduced set (fewer, correct panels) is safe by construction. Verify none assume `panel_index_in_chunk` is dense/contiguous after a merge.

---

## Key source anchors (verified 2026-07-02)

- Per-chunk detection + box schema: `studio/detect/yolo_panels.py:186` (`detect_panels`), `:26` (`boxes_to_panels_norm`, `[ymin,xmin,ymax,xmax]` normalized).
- Gutter expansion (per-chunk, same schema): `tools/expand_boxes_to_gutters.py:198`.
- Scene materialization + `chunk_global_y0` naive summation: `tools/panels_to_scenes.py:779-792`; per-chunk crop `:914`; scene record `:1004-1035`; `dhash64` `:269`.
- Chunk geometry / overlap / cap: `tools/chunk_stitch_adaptive.py:323` (`max_chunk_height=16000`), `:331` (`overlap_px=700`), `:334` (`max_overflow_px=6000`), `:414-424` (chunk record: `chunk_h`, `chunk_index`, `sources`), `:428`/`:490`/`:498`/`:530` (overlap tail), `:559-574` (`adaptive.overlap_px` in manifest).
- Pipeline stages: `studio/pipeline.py:141` (`_stage_detect`), `:155-166` (`_stage_scened` — **insertion point, after line 166**), `:169-175` (`_stage_visioned` reads scenes dir), `:431-441` (`_STAGE_TABLE`).
- State machine: `studio/catalog/models.py:3-4` (`STATUS_ORDER`, `scened`→`visioned`).
- QA target: `tools/prep_qa.py:295` (`cross_dup_flags`, ERROR `:308`), invoked `:1686` over `iter_shown_cuts(plan)` (`:90`), similarity via `render_prep.multi_scale_contained` (`prep_qa.py:306` call site; `tools/render_prep.py:115` definition) — **template matching, not `dhash`**.
- QA gate to exempt: `tools/prep_qa.py:187` (`chunk_as_panel` height check, `h > 8000`, BLOCKING ERROR) — must skip scenes/cuts carrying `reconciled_seam: true` (§5.1).
