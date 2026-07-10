# Foundation-First: Full Pipeline Audit + Plan (2026-07-11)

Owner's thesis, verified by three deep audits over code, git history, and 32+ real panels from
both series: **"with proper cuts and proper element tags, everything downstream becomes
deterministic; healing is a symptom."** The audits confirm it with numbers.

## The verdict in five facts

1. **~2/3 of ~50 downstream mechanisms exist to absorb three upstream defects**: (D1) cuts that
   produce merged/bisected/duplicate crops, (D2) OCR-guess cleaning that destroys or misses
   text, (D3) narration locked BEFORE the final shown-set exists. (Audit 3, compensation table.)
2. **~96% of live heal-target ERRORs (27/28) have deterministic upstream roots**; the heal loop
   converged on zero flag classes in all three logged runs. Healing is a retry-hammer applied to
   non-writer defects. (Audit 2, flag census + logs 14/17/21.)
3. **The trained YOLO already saw what got destroyed**: its `text` class boxed ORV p000006's
   phone screens (conf .52) — the pipeline discards classes text/sfx/speech_bubble/character at
   the source, and its emitted elements have zero consumers (and are empty at chunk scale).
   Cleaning errors: Nano 12.5% (0 catastrophic) vs ORV 44% (2 catastrophic). The only in-world
   screens that survived did so because of a TAG (panel_kind=system → keep-whole). (Audit 1.)
4. **Vertical gutters are unsplittable by construction** (panels_to_scenes is row-only); the
   p000006/p000060 merged-panel class cannot be fixed downstream. QA is blind to over-cleaning
   — content destruction ships green. (Audit 1.)
5. **Deterministic grouping failed in June because of its INPUTS** (text-density statistics),
   not because structure needs a model. Structure is already mostly code today (repair_to_shots,
   force-split, caption folding override the model); the model-critical surfaces are per-panel
   DESCRIPTIONS, the WRITER, and a thin chapter-level layer (flashback/arc/spine = 1-2 small
   calls). (Audit 2, history excavation d122833/7a15f86.)

Full audit reports: session scratchpad task outputs (parts 1-3); key artifacts copied to
scratchpad/audit1/ contact sheets. This doc is the synthesis + plan.

## The plan (all free, all local)

### F0 — Quick wins (days, before any retraining)
- **Vertical-gutter split** in panels_to_scenes (column-run symmetric to the existing row
  logic) + detector guard: 2+ panel boxes inside one emitted box → split. Kills the merged
  side-by-side class immediately.
- **Per-scene element detection** with the CURRENT model: run system_box + speech_bubble on
  scene crops (already done ad-hoc in 2 places — unify into ONE elements manifest at scened
  time); stop discarding the text class — emit it as advisory do-not-touch regions for the
  cleaner NOW (conf-gated).
- **Impact-detector precision**: stamps respect kinds (text cards/system/caption excluded once
  detector-backed); series-context gate so slice-of-life chapters don't grow cockroach impacts.
- **PROMPT_VERSION into freshness** (deps/freshness know understanding generations; kills the
  silent pu_v1 degradation).
- **Review-surface truth**: badge ken sub-cuts ("1 panel · N camera moves"), holds,
  substitutions; mood tag as chip not text; thumbnails from shown crops. Cheap, trust-critical.
- **Over-clean QA tripwire**: element boxes present in raw but blanked in clean → ERROR.

### F1 — The element-tag detector (the long pole, ~days, parallel with F0/F3)
Label top-up + local retrain of the existing 6-class YOLO on the webtoon-ai rig (Roboflow
leads in roboflow.rtf; ogkalu bootstraps bubble labels; OCR boxes + the current text class
bootstrap text labels; system_box already strong at .843). Target: panel / system_box /
speech_bubble / text(in-world+free) / sfx at usable precision ON SCENE CROPS. Acceptance: the
audit's 32-panel scorecard re-run — cuts ≥99%, zero catastrophic cleaning errors, elements
tagged on the 9/16 ORV panels that need them.

### F2 — Tag-driven cleaning (replaces the OCR-guess tree)
Erase ONLY inside speech_bubble boxes. text/system_box/sfx regions are do-not-touch. No
orphan-word pass. Delete the doc_like/_is_inworld_screen/speech_files/_is_title_card gate tree
(region-granular policy replaces file-granular gates). Deterministic by geometry.

### F3 — Geometry-first ordering (narrate LAST)
Freeze the final shown-set (files, crops, kinds, folds, protected cards — everything render_prep
decides today, none of which needs narration) BEFORE narration; derive spans over final files;
write lines against what will actually be seen; add the write-time seconds budget (line length
vs span visual weight — kills monster lines at the source). Deletes by construction:
narrated-panel protection, canonicalize-with-hold, garbage substitution, span pinning,
span_align (shrinks to tripwire), panel_uncovered/double_covered class, the visual-drop heal.

### F4 — Deterministic structure + thin semantics
Grouping boundaries from geometry+tags (panel_kind runs, setting/subject deltas); ONE
chapter-level model call for flashback/arc/spine (keep — live and good). Pass-1 understanding
KEEPS its narrative fields (descriptions are the writer's raw material — model-necessary) but
its classification half becomes detector-backed; delete the 5 corrective passes around
panel_kind. Cast layer stays (deterministic, grows). OCR name normalization into the cast noun
map (the Sanga/Sangah class).

### F5 — Net demolition (test-guarded, after F1-F3 land)
Per audit-3's table: dedup ladder → ~2 tripwires; husk/residue family → gone; pacing patches →
shrink; QA 76 codes → ~35 (plumbing + content-QA + tripwires); heal → single text-repair round
for genuine writer flakiness. Each deleted net's tests become the tripwire spec — the 1,779-test
harness is HOW this demolition is safe.

### Keep untouched (the July refactor survives entirely)
deps.py invalidation authority; manifest_io provenance; fail-closed QA verdict + blocking
discipline + env hatch; content-bound approvals + sha-pinned render; lease/orphan-reap;
Remotion absolute seating; cast_identity; beats_segments accessor; reconcile_seam_panels (the
fix-at-source template); the review loop and regression harness.

### Success criteria
- Scorecard: cuts ≥99% both series; catastrophic cleaning errors = 0; every in-world text
  region preserved.
- Heal fires <1 group per chapter; QA ERRORs on a fresh chapter ≤3, all content-class.
- Steady-state prepare <40 min (fewer model calls: judge shrinks, heal ~0, understanding cached).
- The vision-review loop (human-grade) returns ≤5 findings/chapter, none geometry-class.

### Sequencing
F0 immediately (each item independently shippable, test-gated, reviewed). F1 dataset+train in
parallel. F2 behind F1's detector (interim: current-model advisory tags). F3 as its own
reviewed refactor (the ordering change is the deepest cut — plan it like the July refactor:
tasks, reviews, gates). F4/F5 after F1+F3 prove out on the scorecard.
