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
