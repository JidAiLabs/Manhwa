# Recap Quality — Root-Cause Fix Plan

Status: **in progress.** Two deep investigations (image pipeline + narration) plus a
live per-panel audit of Omniscient Reader Ep0 found the real root causes behind
"broken images, missing images, flat narration, wrong grouping." This is the
tracked plan to fix them **for good** — no more one-off band-aids.

## The unifying flaw

The pipeline enforces quality by **dropping** panels and **exempting** others,
with each "protection" silently disabling several safety nets at once. The goal
("show every good panel, never show a broken one, rich cinematic+persona
story-aware narration") needs the inverse — **invariants**:
- never ship a broken crop → repair, or fall back to the original (don't drop),
- show every *keep* panel → stretch time to fit panels, don't truncate panels to
  fit a fixed audio duration,
- more beats + remove length throttles → richer narration,
- a chapter-level structure pass → flashback / arc awareness.

Exemption tags (sys/doc/title-card) must narrowly suppress only the *aesthetic
redundant* verdict — never the *is-this-a-real-image* check.

## Root causes (evidence: file:line)

### Images
- **RC-IMG-1 — no validity gate (DONE).** Nothing asserted a shown crop is a
  real image; all content checks were behind `if not doc and not sys`. An
  all-black panel passed QA. Fix shipped: `prep_qa.image_flags` blank_crop ERROR
  for every shown crop, no exemption (commit 9881b36).
- **RC-IMG-2 — over-inpaint whitens whole panels.** `render_prep._bubble_text`
  residue sweep (≈ lines 464-473) floods frame-sized "bubbles" (caption cards) to
  white. Fix: never flat-fill a box > ~60% of panel area; gate the residue sweep
  on blanked-area, not interior flatness.
- **RC-IMG-3 — title-card protection disables drop+judge+QA at once**
  (`render_prep.py:1373,1403,1504,1603`; commit 6ecf1e0). A whitened caption that
  trips `_is_title_card` becomes un-droppable, un-judged, un-QA'd. Fix: gate
  `_is_title_card` on the *original* (pre-clean) image + require intact glyph
  components; protection suppresses only the redundant/low-art drop, never the
  blank/husk check. (RC-IMG-1's gate now backstops QA regardless.)
- **RC-IMG-4 — broken crop is dropped, not repaired.** When cleaning blanks a
  panel, `panel_recoverable`/`judge_cut_visuals` drop it (and the judge is
  fail-soft + sys/doc-exempt). Fix: post-clean invariant — if the cleaned crop is
  near-uniform, **show the original uncleaned crop** instead of dropping.

### Coverage
- **RC-COV-1 — duration→panel cap truncates `keep` panels.**
  `timeline_planner.py:595` `kmax = floor(shot_dur / min_cut_sec)`;
  `scene_selection.choose_kept_scenes` then caps kept panels at kmax. A group's
  duration comes from its narration length, so short narration silently discards
  keep panels (ORV p7,p8). Fix: `shot_dur = max(narration_dur, n_keep*min_cut_sec)`
  — stretch time so every keeper is shown (extra panels = faster B-roll under the
  same audio). Makes "every keep panel shown" an invariant.
- **RC-COV-2 — shots with no beat get no selection + no narration.**
  `gemini_narrative_pass` `groups[:max_groups]` / stale-resume can leave tail
  shots beat-less; `timeline_planner.py:788` then emits an empty-beat fallback
  (arbitrary `files[:kmax]`, silent/filler). Fix: beats must cover ALL groups
  (default `max_groups=len(groups)`, resume backfills missing group_ids); a
  missing beat for a real shot is an error, never a silent empty fallback.
- **RC-COV-3 — real-art panels left ungrouped / over-dropped as redundant.**
  Live audit: p8,p9,p32 are real art but ungrouped/not-shown; redundant-drop rate
  is high. Fix: audit `scene_group_builder` so every non-chrome panel joins a
  group; loosen the redundant verdict toward coverage.

### Narration
- **RC-NAR-1 — chrome scrub emptied legit lines (DONE).** My band-aid pattern
  matched "Our adventure begins as…". Narrowed to format-chrome only (9881b36).
- **RC-NAR-2 — grouping collapse → too few beats.** The liked version had **11**
  beats; current grouping yields **6**, so multi-shot detail (dragon/tigers/sword)
  collapses into one flat line. NOT a writer/prompt regression (prompt + 2400
  token cap unchanged since Jun-10; per-line words went UP). Fix is RC-COV-3 /
  grouping: target beats-per-page as a tunable, regression-tested.
- **RC-NAR-3 — persona length throttle.** `narration_punchup.validate_line`
  caps CONNECTIVE at 1.5×grounded → richest persona expansions silently fall back
  to the grounded (persona-less) line; DRAMATIC/CONNECTIVE split leans cinematic
  (27/9 on Nano). Fix: raise CONNECTIVE max_ratio (~2.0-2.5); reconsider the lean
  (CINEMATIC_RULES already says "BOTH every line").
- **RC-NAR-4 — no story-awareness (flashback/arc).** Writer sees only the last 2
  beats; no chapter-level view. A flashback rule EXISTS but is dead code (only in
  `script_expander`, skipped under `gemini_verbatim`). Fix: a chapter-level
  structure pass (one cheap Gemma call over all groups' OCR + thumbnail strip,
  reuse `cast_builder` sampling) emitting per-group `{segment: present|flashback|
  dream, arc_label, callback_to[]}`, fed into `_pack_group_payload` + a writer
  rule that frames flashbacks/callbacks.

## Phased plan

1. **Never ship broken** — RC-IMG-1 (done) → RC-IMG-2, RC-IMG-3, RC-IMG-4.
2. **Full coverage** — RC-COV-1, RC-COV-2, RC-COV-3.
3. **Richer narration** — RC-NAR-1 (done) → RC-NAR-2 (falls out of #2), RC-NAR-3.
4. **Story-aware** — RC-NAR-4 (chapter structure pass).

Each phase: implement → unit tests → deploy → re-run ORV ch1 on the dashboard →
verify panel-by-panel before moving on. No voiceover until the visuals + narration
are right.
