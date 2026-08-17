# Art-only detector retrain + art-only display crops

**Owner goal (2026-07-23):** panels on screen should show **clean art only** — no
speech bubbles, no half-bubbles — with the narration carrying the dialogue.
This requires (a) **retraining the YOLO detector** so the `panel` box is art-only,
and (b) a **display-time crop** in `render_prep` that trims bubbles off the shown
frame. Owner will run training + testing next session; if confident, try a few
chapters.

---

## Root cause this fixes (why v3 doesn't already do it)

`webtoon_panels_v3.pt` was trained on a **merged dataset with two contradictory
panel conventions** (`/Users/anka/webtoon-ai/dataset_v3`, embedded in the ckpt's
`train_args.data`):

| group | files | mean panel height | convention |
|---|---|---|---|
| `long_*` / `panels_*` (3rd-party `manhwa-panel-detector`) | **3560** | 0.20 | bubble-**inclusive** |
| `ssys_sysbox_*` (owner's rich annotations) | **120** | 0.06 | **art-only** |
| bubble/text-only (no panel box) | 2731 | — | non-panel classes only |

The owner's art-only convention was **~3% of the panel signal, swamped ~30:1** by
the third-party bubble-inclusive panels. The "panel *excludes* its bubbles"
relationship was shown in only 120 co-occurrence images (of 3834 total bubbles,
only 195 co-occur with a panel). **Verified:** where the owner annotated bubbles,
100% sit *outside* the panel box (mean bubble-area-inside-panel = 0.00). So the
labels are correct; they were just drowned out. The model learned the majority
(bubble-inclusive) convention.

**Consequence:** even at panel scale the v3 `panel` box leans bubble-inclusive.
(Separately, the pipeline detects at 10k-px *chunk* scale where bubbles downscale
away, and `snap_panels_to_elements` + `expand_boxes_to_gutters` grow boxes to the
gutters — but the training dilution is the deeper cause.)

---

## Pipeline impact — OCR is NOT affected (owner's assumption confirmed)

Stage order (`studio/pipeline.py`): `stitch → detect → scened → **visioned (OCR)**
→ grouped (understanding) → beated → scripted → planned → voiced → … → render_prep`.

- **OCR (`_stage_visioned`, vision_extract.py) runs on the FULL scene** right after
  materialization, long before `render_prep`. Understanding (gemma, `_stage_grouped`)
  also runs on full scenes. Both see the bubbles + dialogue.
- The **art-only crop is a DISPLAY-time operation in `render_prep`** (the shown
  frame), which is downstream of OCR/understanding.
- **Therefore: art-only display loses no dialogue.** OCR + understanding + narration
  are unchanged; only the *rendered frame* is cropped tighter. "half bubbles" (a
  bubble bisected at a crop edge) also disappear.

**HARD CONSTRAINT for implementation:** the art-only crop MUST stay at `render_prep`
(display). Do **not** materialize art-only scenes at `_stage_scened` — that would
crop bubbles *before* OCR and lose the dialogue. Materialization stays full-panel.

This is the husk-mode GOAL (clean panels) achieved by **framing, not inpainting** —
no cv2 smear, no OCR loss. Effectively a new `--bubble-shown-mode art_only`
alongside `keep`/`husk`.

### Two honest limits (must handle)
1. **Over-art bubbles can't be cropped** — a bubble drawn *on* a character (not in a
   gutter) is inside the art region; a rectangle can't exclude it. Art-only removes
   only gutter/edge bubbles. Panels with over-art bubbles ship as-is (or fall back
   to `keep`). This is geometry, not a bug.
2. **Nameplate captions are WANTED** — `caption_box` intros (e.g. "CHEONMA CULT LORD
   — CHEON YOO JONG", "RIGHT GUARDIAN — SEOB MENG") are identity/story. The crop must
   **protect `caption_box` + `system`**, never trim them off. (Same protection
   `edge_recrop_window(protected=…)` already applies to `system`.)

---

## Dataset assessment — what's usable (owner added datasets to `dataset_v2/incoming/`)

Canonical 8-class target schema (keep v3's): `panel, speech_bubble, radio,
speech_background, sfx_text, system_ui, caption_box, free_text`.

### PANEL sources (for the art-only `panel` class)
| dataset | files | boxes | convention (spot-checked) | verdict |
|---|---|---|---|---|
| `dataset_v3` `ssys_sysbox_*` (owner) | 120 | art-only + full 8-class | **art-only, confirmed** (bubbles 100% outside) | **KEEP — gold** |
| `manhwa.v1-dataset.yolo26` (owner-added, class `painel`) | 196 | 1152 | art-only-leaning (tight to art, gutter bubbles excluded) | **KEEP (audit fully)** |
| `Manhwa-Base.v1i.yolo26` (owner-added, misnamed class) | 580 | 1336 | art-only-leaning (excludes captions) | **KEEP (audit fully)** |
| `Detect.v4-new-test.yolo26` (class `panel`) | 2932 | 15479 | tight-panel, excludes gutters/titles but **tolerates over-art bubbles** | **MAYBE — audit; large but not strictly bubble-excluding** |
| `Webtoon-Manhwa Panels.v1i` / `roboflow_panels` (7 panel-shape classes) | 95 | ~1000 | traditional comic panels, likely bubble-inclusive | **PROBABLY EXCLUDE** |
| `manhwa.v1i.yolo26` (class `manhwaa`, 84) / `Manhwa Panel Detector.v2i` (class `Image`, 265) | 75/254 | tiny, misnamed | **SKIP** |
| `dataset_v3` `long_*` / `panels_*` (3rd-party) | 3560 | bubble-inclusive | **the dilution culprit** | **EXCLUDE from panel class** |

### NON-PANEL sources (for bubble/sfx/system/caption classes)
| dataset | files | classes | verdict |
|---|---|---|---|
| `Manhwa detect.v8i.yolo26` | 1841 | background, logo, radio, screams, sfx, speech, speech_background, square, system, thought, watermark | **KEEP** (map speech→speech_bubble, screams/sfx→sfx_text, system→system_ui, …) |
| `mt1.v8i.yolo26` | 1141 | bubble, sfx, sub_text, text, watermark | **KEEP** (bubble→speech_bubble, sfx→sfx_text, text/sub_text→free_text) |
| `dataset_v3` rich (2731 no-panel + 120) | — | speech_bubble/sfx/system/caption | **KEEP** |

**CRITICAL: class-name unification.** Every incoming dataset uses different class
names/orders (`painel`, `manhwaa`, `Image`, `speech` vs `speech_bubble` vs `bubble`,
`screams`/`sfx`, `system` vs `system_ui`, `square`, misnamed single-class projects).
A mapping table from each dataset's class list → the canonical 8-class schema is
**prerequisite** to any merge, or the merge is garbage.

---

## Plan (next session — training + testing)

### Phase 1 — Dataset audit + clean merge (the load-bearing phase)
1. **Visual audit** every candidate PANEL dataset: draw boxes on ~10 samples each,
   classify **art-only vs bubble-inclusive** (spot-checks done: v1-dataset +
   Manhwa-Base look art-only; Detect.v4 tolerates over-art bubbles). Keep only
   art-only-convention panel labels.
2. **Class-name unification map** per dataset → canonical 8-class. Drop junk classes
   (logo/watermark/thought/square unless we want them).
3. **Build `dataset_v4`**: art-only panels (owner ssys_sysbox + v1-dataset +
   Manhwa-Base + Detect.v4-if-it-passes) + non-panel classes (v8i + mt1 + v3 rich).
   **Exclude** `dataset_v3` `long_*`/`panels_*`. Re-index labels to the canonical
   class ids. Deterministic train/val split; **no leakage** across the merge.
4. **Sanity metric** (reuse this session's probe): across the merged set, bubbles
   that co-occur with panels must be **≥ ~90% outside** the panel box, and the
   art-only : bubble-inclusive panel ratio must **invert** vs v3 (art-only should now
   dominate). If not, the merge is still diluted — fix before training.

### Phase 2 — Retrain
5. Train YOLO (ultralytics 8.4.62, `.eval_venv` py3.12, **imgsz 960**, single_cls
   False, 8 classes). Mirror v3 hyperparams (100 epochs, batch 4) as a start.
   **Decision — where:** data lives on the **Air** (`/Users/anka/webtoon-ai`); the
   **Mini** has more GPU but per `[[air-second-factory]]` is the E2E factory.
   Training ≠ E2E pipeline, so training on the Air (data-local) is acceptable, or
   copy `dataset_v4` to the Mini. **Owner to decide.** (~18 min/epoch on Air per
   `[[yolo-training-env]]`.)
6. **NEVER train on pipeline output** (`[[detector-rebuild-plan]]`).

### Phase 3 — Validate the model (before touching the pipeline)
7. Run the new weights vs `webtoon_panels_v3.pt` on the ch6 panels (p13/p16/p26/p30
   + a fight panel). Confirm the **`panel` box is now art-only** (excludes edge
   bubbles) and bubbles/sfx/system/caption still detected. Re-run this session's
   `bubble_probe.py`.
8. Owner review of art-only boxes on ~2 chapters' worth of panels before wiring in.

### Phase 4 — Pipeline: art-only DISPLAY crop
9. Add `--bubble-shown-mode art_only` in `render_prep` (alongside `keep`/`husk`).
   Reuse/extend `edge_recrop_window`: crop the shown frame to the **art-only panel
   box** (new detector) minus any `speech_bubble`, but **protect `caption_box` +
   `system_ui`** (nameplates/stat cards stay). Panels with **over-art** bubbles (no
   clean art band) **fall back to `keep`** — never cut art.
10. **Materialization + OCR + understanding UNCHANGED** (full panels). Only the shown
    frame changes. prep_qa's `visible_text`/bubble checks must respect the new mode
    (mirror how `63547d9` taught them `keep`).
11. Swap detector weights in `studio.toml [detect] yolo_weights` **only after** owner
    sign-off (same gate as the v3 swap, `0d04948`). Keep v3 as rollback.

### Phase 5 — Try a few chapters
12. Reset 1–2 chapters to `detected`, re-run prepare → QA → review the art-only
    frames. If good, widen.

---

## Open decisions for the owner
- **Training host** (Air data-local vs copy to Mini). See Phase 2.5.
- **Does art-only REPLACE keep-mode, or is it a per-series/global toggle?** Art-only
  hides on-screen dialogue (narration carries it) — a deliberate look change from the
  `keep` decision (`297c6a9`). Recommend a **mode flag**, default TBD by owner.
- **Include `Detect.v4` (15479 panels)?** Big recall boost but tolerates over-art
  bubbles — may re-introduce mild dilution. Decide after the visual audit.

## Risks
- **Merge dilution recurs** if a bubble-inclusive panel set sneaks in → Phase 1.4
  sanity metric is the gate.
- **Class-map errors** silently corrupt training → unify + spot-check before train.
- **Over-cropping art** on over-art-bubble panels → the `keep` fallback + owner
  review protect against it.
- **Too few art-only panels** (owner's 120 + ~776 new = ~900) may under-train recall
  → Detect.v4 (audited) or more annotation may be needed.

## Key artifacts this session produced
- Root-cause probes (dataset composition, bubble-containment): re-runnable, see the
  handover.
- gemma SFX validation (separate track, already shipped as `22c0275`): gemma reads
  Korean SFX + judges `strikes_or_weapons` — the impact_mismatch gate now uses it.
- Box-overlay audit images: `scratchpad/ds/*.jpg`.

---

# 2026-07-29 — Phases 1–2 EXECUTED (Air-local session; Mini off)

Tooling + evidence live next to the data: `dataset_v2/build/d8_audit_overlays.py`
(overlays in `dataset_v2/audit_v4/`), `build/d8_assemble_v4.py` (merge + gates),
`build/d8_compare_v3v4.py` (Phase-3 side-by-side). All run in `.eval_venv`.

## Audit verdicts (eyeball on GT overlays + geometry probes)

| dataset | verdict | evidence |
|---|---|---|
| `manhwa.v1-dataset` (196 img / 1152 boxes, native-res strips) | **KEEP** | art-only: frame-tight boxes; overflowing bubbles cut at frame; floating gutter narration unboxed |
| `Manhwa-Base.v1i` (580 / 1336, 640-stretch) | **KEEP** | art-only: bubbles below/between panels excluded; giant shout-bubble outside box (base_07) |
| `Webtoon-Manhwa Panels.v1i` (95 / 1015, 640-stretch) | **KEEP** — plan's "probably exclude" guess was wrong | colored webtoon pages, art-tight, gutter bubbles excluded; 7 shape classes → panel; 35 Outbound+Rectangle twin boxes deduped (IoU≥0.7, tighter wins) |
| `roboflow_panels` | ignore | byte-identical dupe of Webtoon-Manhwa Panels |
| `Detect.v4` (2932 / 15479) | **EXCLUDE** | export is grayscale (CRT phosphor) + 640-stretch + adaptive-equalize + 3× flip/90°-rotation augment, B&W-manga domain — poison regardless of box convention |
| `manhwa.v1i` (75 / **9 boxes**), `Manhwa Panel Detector.v2i` (254 / **11**) | **EXCLUDE** | near-empty labels; as v3's `mnh_`/`mpd_` they injected ~300 panel-ful images labeled panel-free |
| `salvage long/panels` (3464) | **EXCLUDE** | the 30:1 bubble-inclusive dilution culprit |
| `salvage cdi` (3) | EXCLUDE | unknown convention, irrelevant size |

## CORRECTION to this plan's premise — sysbox "owner gold panels" was wrong

`salvage/sysbox` **class-0 is not panels**: median 41 px tall, aspect 4.47, 87%
under 10% image height → **text lines** (title/stat rows inside system cards;
see `audit_v4/ssys_02/ssys_04`). The "bubbles 100% outside the panel box" probe
passed *trivially* against text-line-sized boxes. So v3 ingested **1161
text-line boxes as panel GT** — a second panel-class poison beside the 30:1
dilution. v4 drops ssys cls-0 and keeps sysbox's real gold: speech_bubble 1017,
system_ui 679, sfx 252 (cls-5 stays quarantined; sysbox has zero captions).

## dataset_v4 (built + gated; `dataset_v4/report.json`)

- **3564 train / 673 valid**; panel **2940/528 — 100% art-only trio**
  (base 1336 + v1ds 1152 + wmp 980), hard-asserted by source prefix;
  speech_bubble 3882/747, sfx 1400/275, system_ui 805/76, caption 379/60,
  free_text 264/56, speech_bg 212/40, radio 109/12.
- Plan §1.4's containment metric is **vacuous on the clean merge** — no kept
  source labels panel AND bubble on the same image (v3's sources were the same).
  Replaced by: source-purity assert, panel-geometry gate (median 181 px tall,
  aspect 1.15 — catches text-line poison), orphan check. Known-inherited noise:
  element-source images carry no panel labels and vice versa (cross-suppression
  v3 already trained through fine).

## Phase 2 — training LAUNCHED on the Air (data-local; Mini off → host decision resolved)

v3's exact recipe: `yolo26n.pt` init, MuSGD, batch 4, imgsz 960, patience 20,
seed 0, device mps. Run: `runs/detect/webtoon/v4_artonly_960` (+ `.pid`,
`.trainlog` siblings). ~891 batches/epoch; v3 ran ~19.5 min/epoch on 7.6k imgs,
v4 has 3.6k. Check: `tail runs/detect/webtoon/v4_artonly_960/results.csv`.

## Phase 3 — ready to run once weights exist

`build/d8_compare_v3v4.py --b runs/detect/webtoon/v4_artonly_960/weights/best.pt <imgs>`
at prod settings (conf .25, ckpt imgsz). Validation imagery ON the Air:
`dataset_v2/corpus/nano-machine/Chapter_1/{scenes,stitch_chunks}` (owner-approved
chapter; 114 scenes) + ch3/ch5 chunks + `omniscient-reader/Episode_2`. Scenes
judge the art-only convention; chunks judge recall. Owner sign-off gates the
weights swap (Phase 4 display crop needs the Mini for chapter E2E).

---

# 2026-08-17 — Phase 3 VALIDATED + Phase 4 BUILT (Air; Mini back online, idle)

## Training result (run `v4_artonly_960`, 100/100 epochs, finished 2026-08-05)
`best.pt` → committed as `assets/models/webtoon_panels_v4.pt` (v3 stays the rollback).
Own valid: panel P .867 R .928 AP50 .944 AP .852; overall mAP50 .861 / .767.

## Head-to-head — beware the contaminated split
`dataset_v4/valid` ∩ `dataset_v3/train` = 57 images (md5), incl. **30 of the 50
system_ui images** → v3's numbers on the naive split are memorized. Fair split =
`dataset_v4/valid_clean` (616 imgs v3 never trained on; `data_valid_clean.yaml`):

| class (valid_clean) | AP50 v3 → v4 | AP50-95 v3 → v4 | R v3 → v4 |
|---|---|---|---|
| panel | .301 → **.947** | .118 → .854 | .286 → .927 |
| speech_bubble | .941 → .989 | .858 → .905 | .944 → .975 |
| sfx_text | .776 → .841 | .596 → .645 | .665 → .810 |
| system_ui (31 inst) | .757 → .733 | .708 → .687 | .692 → .710 |
| caption_box | .931 → .950 | .928 → .931 | .917 → .933 |
| free_text | .859 → .901 | .754 → .806 | .750 → .893 |
| speech_background | .658 → .683 | .546 → .566 | .678 → .654 |
| **all** | **.777 → .880** | .688 → .799 | |

v4 ≥ v3 on every class; system_ui is a wash on 31 instances. JSONs in
`dataset_v2/audit_v4/compare_final/`.

## Corpus overlays (final weights; `compare_final/{ch1_scenes,chunks}/` + `summary.json`)
- **Scene crops: the `panel` class barely fires for EITHER model** (v3 17/112,
  v4 12/112 at conf .25) — a page-trained detector does not see an isolated crop as
  a panel. ⇒ the art_only shown frame must come from the DETECT-stage raw box, not
  a second pass on the crop. Elements (bubble/sfx/system/caption) fire fine on crops
  (v4 ≥ v3: bubbles 100 vs 95, system 7 vs 3, captions 3 vs 1).
- Chunks: panels A=438 B=377 over 47 chunks (v3's surplus = text-line poison boxes,
  under visual review).

## Seam dry-run — the risk was real, and is fixed upstream (`scratchpad/seam/run_seam.py`)
Real chain (`detect_panels → expand_boxes_to_gutters → panels_to_scenes`) on corpus
nano ch1, OCR words (748) mapped to chunk coords, "orphaned" = in NO produced scene:

| | v3 | v4 before fix | v4 after fix |
|---|---|---|---|
| raw → expanded → scenes | 111→107→106 (3 recovered) | 110→110→124 (**23** recovered) | 110→109→114 (10 recovered) |
| OCR words orphaned | 37 | **44** ("KILL HIM!", "...WHY? ME?", "KEUK...") | **33** |

Mechanism: art-only boxes leave gutter dialogue outside; `expand_boxes_to_gutters`
can't see a white bubble on a white gutter by row stats; the recovered gap span is
then dropped by `--skip-blank`. Root: **`elements_norm` was EMPTY on real chunks for
BOTH models** (a 10k-px strip at imgsz 960 resolves no bubbles) → `snap` never had
anything to snap on a real chapter.

**Fix (this session, `studio/detect/yolo_panels.py`):** `_tile_elements` (2400-px
windows → real chunk-space bubbles/captions/system/sfx in `elements_norm`) +
`snap_panels_to_elements(attach_gap=300px/img_h)` attaches hanging / near-floating
gutter dialogue over a panel's x-span to the NEAREST panel (edge-clamp so the
bubble is never bisected; corner touches / side floaters ignored) + `panels_norm_art`
(raw pre-snap art boxes) persisted per chunk (passes through the expander untouched).
**v3's output is byte-identical before/after** — inert on production weights until
the swap. Materialization thus stays "art + its dialogue" (OCR/understanding see
speaker+speech together, as with v3) while the raw art box drives the display crop.

## Phase 4 — `render_prep --bubble-shown-mode art_only` (built, tests green: 2091)
- `art_box_local(scene, art_boxes_chunk_px)`: raw art box → written-scene pixels
  (`box_px_xyxy` origin, `trim.left_px/top_px`), union across boxes.
- `art_only_window(img, art, bubbles, protected)`: art box grown to keep protected
  `system` + `caption` boxes (on-crop detector, `_element_boxes` now collects
  captions); a bubble STRADDLING a window edge pulls the edge out to take the whole
  bubble (chained) — no half-bubbles, never cut art (the plan's over-art → keep
  fallback); every discarded band verified by `band_is_chrome` (refactored out of
  `edge_recrop_window`, same test) — a band with uncovered art edges is refused on
  that side; slivers < 12px ignored; < 320px keep → full frame. Left/right too.
- No art box (recovered gap scenes, pre-v4 detections) → keep-mode edge trim.
- `prep_qa` gates bubble-interior checks for art_only like keep. Plan stamp unchanged.
- Config: `studio.toml [render] bubble_shown_mode = "keep"` (default) → `Config.
  bubble_shown_mode` → worker passes `--bubble-shown-mode` (worker.py touched →
  daemon restart on deploy). `STUDIO_BUBBLE_SHOWN_MODE` env override.
- **One weights authority:** `yolo_panels.default_weights()` = `[detect].yolo_weights`;
  `render_prep --panel-weights` and `panel_understand._DEFAULT_PANEL_WEIGHTS` now
  default to it (fail-soft to the v3 file). The swap = ONE config line.
- Offline demo on the v4 seam scenes (`scratchpad/seam/demo_art_only.py`): 114 scenes,
  art box for 101 (13 = recovered gap-scenes → keep), shown frame changes for 46
  (keep-mode: 23). p000028 "KILL HIM!" framed out; p000054 straddlers kept whole.

## Owner decisions still open (before Phase 5)
1. **Sign off v4 → swap** `[detect] yolo_weights = "assets/models/webtoon_panels_v4.pt"`
   on the Mini (+ `reset --to detected` for tried chapters — art boxes need a re-detect).
2. **Flip `[render] bubble_shown_mode = "art_only"`** globally or per trial.
3. **Straddling bubbles**: current rule keeps them WHOLE (= keep look on that side).
   Alternative = allow trimming ≤ N% into art to remove the remnant. Not built (YAGNI).
4. Small tech debt noted: keep/art_only `_cleaned` recrops don't offset `word_boxes`
   passed to `_write_part` (focal dead-regions slightly off) — pre-existing in keep.

## Visual review (8 vision reviewers over the 159 overlays) + ORV finding — 2026-08-17 late
- **Nano Machine (30 chunks): v4 missed 0 real panels**, spurious 22→12 (v3 fragments
  full-width panels into half-boxes, boxes title/credit cards and bubble-only strips;
  v4's few spurious = duplicate pairs → NMS/dedupe), art-only adherence yes/mostly on
  all; 9 mild art clips (axis-aligned boxes on parallelogram panels, one remnant).
- **ORV (17 chunks): v4 has NO box for 25 real panels** v3 found (black-background
  creature, clock/UI panels, tilted crash panel, the ~3900px full-bleed crowd splash
  gets 0 boxes at ANY conf/tiling). Knobs don't help: conf .15 → 84, .10 → 87, tiled
  6000px → 63 (v3 = 116, ~22 spurious). Real chain on Ep2: v4 raw 79 vs 116; scenes
  111 (36 recovered) vs 113 (7); coverage .77 vs .68; **OCR words orphaned 116 vs 84**.
  Cause: the art-only trio is bordered/white-gutter manhwa; dark, full-bleed, UI-screen
  pages are under-represented → training-domain gap, not a threshold.
- Scenes (112): 5 v4-worse bubble cases (large gutter/white bubbles v3 caught at .8–.9,
  v4 nothing: p000024/053/092/101/103); otherwise parity or v4 better; no system card
  or nameplate missed.
- **Owner decision 4 (added):** ship v4 globally now / per-series weights (v4 Asura-style,
  v3 Webtoon-style) / hold the swap for a v4.1 fine-tune on ~150–300 art-only-annotated
  dark+full-bleed+UI pages (annotate raw pages, never pipeline output).
- Sign-off report (artifact): https://claude.ai/code/artifact/68944a54-1f9e-4d8c-b435-d346fc36abaa
