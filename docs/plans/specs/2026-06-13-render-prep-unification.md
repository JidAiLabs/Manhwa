# Render-prep unification — ending the panel-selection cycling

**Status:** Stage 0 shipped (allocation invariant + QA group-awareness). Stages 1–3 (classifier consolidation) planned, NOT yet done.
**Date:** 2026-06-13
**Why:** `tools/render_prep.py main()` runs 8+ scattered passes that each RE-DERIVE panel intent from pixels and disagree. Every per-defect geometric patch fixes one symptom and spawns another (missing cards ↔ duplications ↔ blanked docs ↔ QA false-positives). Root cause = no canonical, single-source classification of each panel.

## Evidence that drove this (reproduced 2026-06-13, all 3 test chapters)

- Most reported defects were **stale screenshots** — current code already keeps flat in-world system cards unblanked and no longer blanks the ORV "masterpiece" doc panel. The review surface (deployed dashboard) lagged the code.
- The **over-broad `doc`/`sys` tagging** is the disease: 52/93 (Nano) and 36/67 (IE) panels are tagged sys/doc. The exemption those tags grant was conflated as both "don't flag recurrence" AND "allow visible repeat", which is why protecting cards and suppressing dups kept fighting.
- The only genuinely-current duplications (IE: 4 ABA-dups) were **all** sys/doc-tagged panels reappearing 2 segments apart — several actually caption-over-art (p000084 "HER TIME FLEW BY", p000090 "AT SCHOOL WHENEVER…"), not true system cards.

## Stage 0 — SHIPPED (this session)

1. **QA group-awareness** (`prep_qa.py:semantic_alignment_flags`): the vision judge now evaluates the narration against EVERY cut in a multi_cut montage (`file` + split `file2`), early-exits on the first plausible match, and only flags `narration_mismatch` when the line fits NONE of the shown panels. Fixes the false WARN on g0001_p00 (3-cut montage judged against the primary landscape only). Tests: `test_semantic_judge_group_aware_*`.
2. **Single allocation invariant** (`render_prep.py:cap_repeats_with_holds`): NO panel — not even an exempt sys/doc card — is re-emitted as a fresh cut inside the radius-3 degenerate window; it HOLDS the previous panel instead. Exemption now relaxes only the GLOBAL cap (far-apart recurrence stays legal). This kills the IE ABA-dups **robustly — independent of classification accuracy**, which is the key property given the classifier is unreliable. Tests: `test_cap_repeats_holds_exempt_panel_on_nearby_repeat`, `test_cap_repeats_exempt_recurs_far_apart`.

Regression net: `/tmp/golden/` snapshots pre-refactor per-chapter invariants (panels shown, visible_dups, mandatory cards, never-empty); `/tmp/golden/check.py <nano|ie|orv> <plan>` asserts equal-or-better.

## Stages 1–3 — PLANNED (the structural consolidation; ~8–12h, staged, each golden-verified)

The target: replace the scattered passes with ONE classification authority + ONE selection authority.

### PASS 1 — `classify_panel(file) -> {CHROME, JUNK, SYSTEM_CARD, DOCUMENT, ART}` (classified once, immutable)

Ordered decision (reusing existing signal functions, computed once and cached):
1. **CHROME** if `is_chrome_scene(vision_item, series_title, midtone_frac)` — publisher/cover/counter/credits. Dropped.
2. **SYSTEM_CARD** if `bubble_coverage(_sys_boxes) >= 0.02` **AND** `file NOT in speech_files` (Gemini bubble_mode ∈ {spoken,shout,inner_thought}). The **speech-veto** is the fix: a dialogue bubble that YOLO mis-fires as a system_box is rejected; a narration-only styled UI/status card is accepted. Kept, text preserved, never cleaned.
3. **DOCUMENT** if `doc_like_v2` — the existing coverage/word gate **AND** `outside_bubble_words >= 8` (true app/stats/relationship pages carry substantial standalone text; a caption-over-art panel has 0–3 outside words → falls through to ART). Kept, text preserved, only floating speech bubbles blanked.
4. **JUNK** if `not panel_recoverable(cleaned)` or the AI `judge_cut_visuals` rejects the artwork. Dropped.
5. **ART** otherwise. Kept, speech bubbles blanked.

### PASS 2 — clean only kept panels per type (reuse `clean_scene_image`, `select_panel_crops`, dead-box recrop, split2). Emit `scene_dims` with `doc`/`sys`/`blanked` **derived from `panel_type`** (keep the booleans for backward-compat — see contracts).

### PASS 3 — `allocate_panels()` single authority (folds in `substitute_garbage_sole_cuts` + `drop_cross_segment_duplicate_cuts` + `cap_repeats_with_holds`). Invariants: no in-window visible repeat (Stage 0 already enforces this in `cap_repeats_with_holds`); global cap N; exempt(SYSTEM_CARD/DOCUMENT) bypasses only the cap; never-empty; no visually-identical neighbors.

### Contracts that MUST NOT break (audited)
- Timeline cut fields read by renderer/QA/dashboard: `file`, `file2`, `layout="split2"`, `held`, `start`, `dur`.
- `scene_dims` fields: `w`, `h`, `doc`, `sys`, `blanked` — read by `blender_vse_from_plan` (doc ⇒ contain-fit, never cover-crop), `prep_qa.montage_flags._protected` (sys|doc ⇒ exempt from degeneracy), `image_flags` (sys ⇒ skip husk/card checks). If `doc`/`sys` move to a single `panel_type`, update ALL these readers in the same change.
- Behaviors to preserve: chrome detection, system-box protection, DOCUMENT text preservation, dead-box recrop, split2 side-by-side, branding intro/outro, `manual_drops.json` operator bans, never-empty segments, holds-are-QA-exempt, `segment_id`=`g####_p##` byte-identical across script→tts→timeline.

### ⚠ Primary risk of Stages 1–3
Tightening `doc_like` can reclassify a TRUE document as ART → its text gets blanked (re-introduces the exact "blanked masterpiece" defect). **Before shipping the tightening, extend the golden net with a blanking-regression check**: for every panel classified DOCUMENT/SYSTEM today, assert its cleaned `std` stays ≈ raw `std`. Do Stage 1 (classify, additive) → Stage 2 (consult, behavior-preserving, golden IDENTICAL) → tightening (separate, blanking-verified) → Stage 3 (allocate). Roll back per stage on any golden FAIL.

## Verify-the-real-output discipline (process fix)
The cycling was amplified by reviewing stale renders. After ANY render_prep change: regenerate the affected chapters AND redeploy before reviewing the dashboard. A freshness/commit stamp on the QA report + gallery would make staleness visible.
