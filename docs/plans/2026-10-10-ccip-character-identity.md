# Plan: image-based character identity — the protagonist is recognized from pictures, for every series, automatically

## Context

**The problem.** The narrator names the wrong person. Today "who is drawn in this panel" is decided by matching two
pieces of TEXT: gemma's per-panel description ("a young man with dark hair") against a per-chapter description the
model guessed for each cast member. Same-looking characters are indistinguishable that way. Measured: when the
identity rewrite changes who a line is about it is right ~1 time in 4 (54 graded, 2026-10-09). The owner,
2026-10-10: *"we can't recognize a character with a simple description like dark hair, white coat... it is too
vague... find a solution to this one first"*; and on curation: *"full automatic is better... how do you do it on new
manhwas? what happens when the character evolves?... normally the most visible character is the manhwa [protagonist]"*.

**What exists.** `tools/panel_identity.py` (2026-09-18) asks gemma "A / B / OTHER" with 2 owner-confirmed reference
panels. It runs only for ORV (233 chapters). Graded at scale (2026-10-10, 36 panels): the protagonist slot is right
13/15, the second slot 2/13 — "B" becomes a bin for "not A". 4.2 s per panel; needs curated references.

**The spike (2026-10-10, `~/idspike` on the Mini, throwaway; memory `ccip-character-identity-spike`).**
`dghs-imgutils` (anime head detector + CCIP character fingerprint; local, free, CoreML): a head in ~80% of panels
with people; the head with the most look-alikes IS the protagonist in both ORV and Tutorial Tower with zero curation;
6 automatic references per series span chapters 1→258 / 1→176 (outfit and style changes survive); precision by eye
≈88% (ORV) at the 0.20 cut and ≈90%+ (TT) at ~0.15 — the cut is per series; recall ≈ the gemma check (the MC on
~45-55% of panels with people); 0.26 s/panel on CPU. Chained cluster growth merges all dark-haired men: never.

**Intended outcome.** A stage that, for every series (new ones included) and with no owner input, discovers the
protagonist from the pictures, keeps a set of reference heads, and confirms per panel whether the protagonist is
drawn. Narration names only what the pictures or the page confirm; everyone else gets a neutral handle. Being vague
stays allowed; being wrong does not. Local and free (hard rule).

**Plumbing bugs found while mapping (fixed as part of this):** punchup (`narration_punchup.py:816-841`) and the heal
pass (`worker.py:706-723`) re-run the identity gate with word matching, undoing the image verdict (shown turning
"Our guy smirks" into "Namwoon Kim smirks"); prep_qa passes `{}` for a missing identity file (`prep_qa.py:3443`), which
silences `actor_mismatch` for every series without one.

## Design

### Principles
1. Pictures compared to pictures. Appearance text never decides identity again.
2. Only the protagonist is confirmed from the image. Everyone else is *unconfirmed* (not absent). No "B" slot.
3. Automatic per series: references found by density ("most visible"), spread over the series, refreshed with
   continuity checks. Registry `exemplars` become optional pins.
4. Fixed references, never chained growth. Per-series threshold, calibrated from the data and clamped tight.
5. Positive evidence only, and it is never strong enough to overrule a proper name the writer wrote. A confirmation
   may add the MC's name and may *flag*; absence of one never changes narration (recall ≈ 50%).

### 1. `tools/panel_identity_ccip.py` (new) — the stage tool
Own venv `.identity_venv` (`studio.toml [identity] python`, `STUDIO_IDENTITY_PYTHON`; the `[tts].python` pattern,
`studio/config.py:149-160`, `pipeline._run_tool(python_exe=)`). Lazy imports; `heads_fn=` / `embed_fn=` seams so
tests run in `.eval_venv` without onnx.

- **Index** `index_chapter(ep_dir)` → `ongoing/<slug>/.identity/<chapter>.npz` + `.json`: per detected head
  `{panel, box, score}` + float16 768-d feature; keyed on the understood manifest's sha and the scene files; rebuilt
  when either changes. Dot-dir: invisible to `ongoing/*/*/` globs, gitignored by `ongoing/*/`, never a chapter (chapter
  dirs come from the DB, `catalog/identity.py`).
- **Profile** `build_profile(series_dir)` → `.identity/profile.json` (written atomically via `manifest_io`, under a
  lock: the sweep and the worker may both write it). Ref embeddings stored **inline** with their source chapter sha
  (a `reset --to detected` invalidates that chapter's refs; they are dropped, never dangling).
  - sample ≤ 3,000 heads stratified by chapter; pairwise differences; seed = densest head within 0.20;
  - refs = seed + nearest heads with diff < 0.12 from **distinct chapters**, up to 8, spanning earliest→latest;
    registry `exemplars` (ORV) added as pins;
  - threshold = Otsu valley of min-diff-to-refs, **clamped to [0.13, 0.17]**, default 0.15 when bimodality is weak;
  - **FP proxy** (free): a panel where ≥ 2 heads both match the MC is a certain false positive; estimate the FP
    rate from multi-head panels and tighten the cut until < 5%; such panels are never named;
  - **active** only when: ≥ 5 chapters indexed, ≥ 150 matched heads, the seed's look-alikes appear in ≥ 60% of
    indexed chapters, and the seed is ≥ 1.5× denser than the densest non-matching head. Otherwise `provisional`:
    **no `manifest.identity.json` is written** (keyword path stands, thumbnails unaffected);
  - refresh: every chapter while < 10 are indexed, then every 10; new refs accepted only if they match the old refs
    (< threshold) — continuity; otherwise the old profile stays and the sweep raises `profile_flip`;
  - **limitation, stated:** a true redesign (new face) is not followed automatically — recall falls in that arc
    (vague, not wrong) and the sweep raises `profile_drift` (N consecutive chapters < 10% confirmed) for the owner.
- **Identify** `identify_chapter(ep_dir, profile)` → `manifest.identity.json`, same shape as today plus
  shape-additive fields: `panels[fn] = {"names": [<protagonist canonical_name from manifest.cast.json>] | [],
  "others": N, "heads": H, "diff": min_diff}`; confirmed iff a head's diff < threshold; `others` = unmatched heads +
  `max(0, person-subjects − heads)` using `subject_person_count`; headless panels with subjects are omitted (→
  all-unknown, as today). `_meta`: backend, profile version, threshold, refs' sources.
- CLI: `--episode-dir`, `--profile-only`, `--sheet` (contact sheet of 40 confirmed heads for grading a series).

### 2. Pipeline (`studio/pipeline.py` `_stage_beated`, after cast_builder, before ledger/writer)
`[identity] backend = "ccip" | "gemma" | "off"` (default after rollout: `ccip`; `gemma` = today's exemplar path).
ccip: index → refresh profile if due → identify (skipped while provisional). **Fail-soft** like story_pass
(`pipeline.py:382-387`): a detector/model failure logs and leaves the chapter on the keyword path; it never parks a
chapter. Staleness: identity.json ← understood (+ optional cast) in `deps.py`; the profile is **not** a deps edge and
beats do not depend on identity (`deps.py:108-110`), so a refresh never invalidates finished chapters; identity is
rebuilt only while preparing that chapter when its `_meta.profile_version` differs. `--identity` goes to the writer
(already), punchup and the heal pass.

### 3. Consumers — one authority
- `cast_identity._figures_from_identity` (645-670): confirmed MC (`evidence: image`); **page-name hits kept**: a cast
  member whose name token is printed in the panel's dialogue/description (today's `cast_identity.py:635-641` NAME
  path, never appearance) is listed with `evidence: page` — otherwise flipping the default silently strips every
  side-character name for every series. Unknowns as today.
- punchup + heal pass read `manifest.identity.json`; prep_qa passes `None` for a missing file.
- `publish_concept._lead_panels` (reads identity.json; needs names+others ≤ 2) works unchanged; provisional = no file.

### 4. `tools/identity_gate.py` — positive evidence, never a wrong name
- Rule 1 (protagonist handle → someone else) **removed**.
- Rule 2 fires only on a **solo-MC span** and only for **descriptive nouns** ("the guard"), never a proper name
  (KEEP stands — no override: 88% is not enough to overrule a name the writer saw). Solo-MC panel := `heads == 1`
  AND matched AND `subject_person_count` over person-like subjects == 1 AND kind ∉ {system, caption} AND no
  dialogue; a span is solo-MC iff every panel in prep_qa's `_covered_panels` window is solo-MC or person-free.
  Unknown-only and absent spans: hands off (no neutral rewrites). Dead-actor ledger rule unchanged.
- A proper name on a solo-MC span that is a different member → `actor_mismatch` WARN (report-only, exists).
- The 2026-10-08 `old/replace/shadow` modes are deleted.

### 5. Writer payload + prompt (precondition of the default flip)
`figures`: confirmed name; page names; `unconfirmed (<subject>)` for the rest. Prompt rule
(`gemini_narrative_pass.py:2865-2871`, "name ONLY from that list… unknown → neutral phrasing") becomes: a listed
name is confirmed; *unconfirmed means not confirmed, not absent* — the protagonist's handle stays allowed when story
and dialogue make it clear. **A/B before shipping** (owner rule): ≥ 20 random beats across ≥ 4 series, count
protagonist references, side-character names kept, wrong names; every flip looked at.

### 6. `tools/identity_sweep.py` (new; mirrors `identity_census.py` / `qa_rescan_sweep.py`)
Per series: index every chapter (CPU, ORV ≈ 1.5 h, fleet ≈ 4 h), build/refresh the profile, `--sheet` per series,
and the census computed **in memory** (identity.json is written only at prepare time, so finished chapters never mix
authorities): chapters whose shipped lines name a non-protagonist on solo-MC panels; `profile_flip` / `profile_drift`
alarms. `--apply` queues re-prepare jobs (dedupes); dry run by default.

### Out of scope (noted)
Gender-safe descriptors for unknowns; a second named character from images; gemma as verifier; following a true
redesign automatically; the cast builder filing the MC's real name as a separate member (per-series cast file, separate
task); `_figures_from_identity` evidence-by-position mis-indexing.

## Implementation steps (TDD; one commit each; full suite exit code read directly, never piped)

1. **Venv + config:** `.identity_venv` (python3.12: `dghs-imgutils`, `onnxruntime`), `requirements-identity.lock.txt`,
   `scripts/bootstrap_mac.sh`, `.gitignore` (+ `.mlx_venv`, `.mlx_vlm_venv`, `dist/`), `studio.toml [identity]`,
   `config.identity_python` / `identity_backend` + env overrides. Tests: resolution.
2. **`tools/panel_identity_ccip.py`** with seams. RED tests (`tests/test_panel_identity_ccip.py`): index shape/keying;
   profile — densest seed, distinct-chapter spanning refs, pins, clamp + weak-bimodality default, FP proxy tightening,
   every activation rule, continuity on refresh, dangling-ref drop; identify — shape incl. `heads`/`diff`, headless
   omission, `others` arithmetic, provisional → no file. **Real-data regression fixture:** the spike's ORV features
   (717 heads, float16 ≈ 1 MB) checked into `tests/fixtures/` with the owner-exemplar verdicts: assert the auto seed
   matches the exemplars (< 0.18), ≥ 6 refs from distinct chapters, threshold in clamp, FP proxy < 5%.
3. **Gate + consumers** (`tests/test_cast_identity.py`, `test_narration_punchup.py`, `test_prep_qa*.py`,
   `test_worker*.py`): Rule 1 gone; solo-MC matrix (each veto: dialogue, 2 heads, 2 persons, proper name → WARN not
   rewrite, absent panel, unknown-only); page-name figures; punchup/heal argv carry identity; prep_qa missing → None
   still flags. Existing ledger tests green.
4. **Pipeline wiring** (`tests/test_pipeline.py`): backend switch, order, fail-soft, refresh cadence, no deps edge,
   provisional skip, `deps.py` optional cast edge.
5. **Sweep tool** + tests (dry run, shard, apply dedupe, alarms).
6. **Real path on the Mini, before any default flip:** bootstrap `.identity_venv`; `identity_sweep` for ORV + TT
   with `--sheet`; grade both sheets (target ≥ 85% precision, FP proxy < 5%); compare verdicts with the spike; then
   `backend = ccip` for those two series only (per-series override in toml) and re-prepare ORV Ep6, TT ch102,
   TT ch151 end-to-end; read the narration: no wrong name on solo-MC spans, protagonist still referenced on
   unconfirmed panels, side characters still named where the page names them.
7. **Prompt wording A/B** (design §5); ship only if it passes.
8. **Flip the default** to `ccip`, pull on the Mini + `launchctl kickstart -k` (pipeline.py is imported by the
   worker), sweep every other series so profiles are active before their next prepare. Handover + memory.

## Verification
- Unit: `.eval_venv/bin/python -m pytest -q` (no onnx needed). Expect 2888 + new, exit 0.
- Real: `.identity_venv/bin/python tools/panel_identity_ccip.py --episode-dir ongoing/omniscient-reader/Episode_6
  --sheet`; grade; `python -m studio run 1 --chapters 6`; read `manifest.identity.json` + beats.
- Fleet: `tools/identity_sweep.py --series ongoing/<slug>` per series; census + alarms reviewed with the owner before
  any re-narration is queued.

## Implementation notes — where the data changed the plan (2026-10-10)

Measured on the spike's real features (now `tests/fixtures/ccip_spike_fixture.*`: 717 ORV + 489 TT heads),
before any constant was written:

- **CCIP difference == (1 − cos)/2** of the features (max error 0.00000 vs `ccip_batch_differences`, 300 heads),
  so fingerprints compare in numpy; only detection + embedding need onnxruntime.
- **Otsu dropped; the cut is FIXED at 0.15.** Otsu valley: ORV 0.175, TT 0.21 — and 0.145/0.150 with RANDOM
  references at the same η (0.68–0.71), so it measures nothing. By-eye grading is the only calibration we have
  (ORV ≈88% @0.20, TT ≈90%+ @0.15); 0.15 is the cut that holds for both.
- **FP proxy dropped.** "Two heads matching in one panel" is 46% (ORV) / 24% (TT) of matched multi-head panels at
  0.15; the sheet shows it is mostly the protagonist next to a look-alike talking to him (sometimes him drawn
  twice). Such a panel is never named; the rate is no calibration signal.
- **Dominance redefined:** seed density vs the densest head among heads ≥ 0.20 from every reference (the next
  group). ORV 8.8×, TT 5.5×; a deliberately wrong seed scores < 1. The "densest non-matching head" in the plan
  scored 1.17–1.56 because near-misses of the protagonist count as non-matching.
- **"Same character?" checks use the cut (0.15), not CCIP's 0.178.** ORV's and TT's leads are 0.165 apart. Owner
  exemplars sit 0.091/0.102 from ORV's automatic references; TT's lead 0.164 — so exemplar agreement is "every
  pin within the cut of a reference", and continuity on refresh is judged at the cut.
- **Cross-series control (new, free):** heads of another series are certainly not this protagonist. ORV's profile
  names 8% of TT heads; TT's names 27% of ORV heads (both leads are dark-haired young men). The sweep reports
  this per series; a high rate is the signal to tighten that series. TT's precision is graded on the real sheet
  before TT is switched on.
- **No deps edge identity ← cast:** it would mark existing identity files stale wherever a cast was rebuilt
  later, and stale manifests block QA. ccip re-runs on every narration build instead (cached heads: seconds).
- **Solo window = no written text at all** on any voiced panel (understood dialogue OR OCR): the understood
  `dialogue` field is filled on 1 of 71 ORV Ep6 story panels, so it cannot be the only veto.
- **Sweep `--apply` (queue re-prepares) not built:** backfills are the owner's decision; the sweep reports.
- Tool reproduces the spike exactly on the Mini (77/77 heads, fingerprint diff 0.0); 16 s/chapter uncached.
