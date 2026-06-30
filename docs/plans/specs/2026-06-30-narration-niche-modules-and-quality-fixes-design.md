# Narration Niche Modules + Recap-Quality Fixes — Design

- **Date:** 2026-06-30
- **Status:** Approved design, pre-implementation
- **Audit it builds on:** `docs/2026-06-30-manhwa-fresh-vs-current-audit.md`
- **Framework source:** the user's "Manhwa Fresh" reference (6 rules + a 3-prompt stack), obs #14283; reference-channel contract `docs/plans/specs/2026-06-26-reference-channel-gemma-baseline-design.md`
- **Memory:** [[abcd-modules-and-6-rules]], [[recap-quality-issues-2026-06-30]], [[narration-genre-persona-subjects-grounding]], [[register-off-default-persona-narration]]

## 1. Problem

A full audit (2026-06-30) of the current narration pipeline vs. the user's Manhwa Fresh
methodology established two things:

1. **The a/b/c/d "niche modules" (manhwa TYPES) are not implemented.** The code has only a
   coarse `murim/modern/system` *setting* axis in `narration_punchup.GENRE_ADDONS`, used solely to
   gate anachronism/game-framing vocabulary. There is no Isekai/Power-Fantasy, Romance/Drama,
   Dark-Action/Revenge, or Comedy/Slice-of-Life persona register.
2. **Four user-reported quality defects** on the fresh Nano ch1 render, all root-caused:
   - **#1 over-tense voice on minor events** — gemma grades ~57% of panels `intense`; intensity
     drives the TTS mood-tag/exag escalation, so a routine fall is delivered like a climax.
   - **#2 zero dialogue/quotes** — *not* a prompt contradiction (the current `_DIALOGUE_RULE`,
     `gemini_narrative_pass.py:155`, already invites ≤6-word quotes); the loss is downstream
     (gemma compliance + the OCR-garble/fragment scrub, which bans `'Ancestor...?'`-style trailing
     fragments by name — the user's own iconic-shout example).
   - **#3 big-group flash montage** — `g0010_p14` = 12 distinct panels in 2.5s. Direct consequence
     of the deliberate "keep every panel" architecture + over-large groups + **no planner
     min-per-panel floor**; `flash_cut` is WARN, never blocking.
   - **#4 flat / no persona** — `narration_punchup.CINEMATIC_RULES` makes persona "OCCASIONAL
     SEASONING, never the default" and `classify_beats` gates persona OFF on DRAMATIC
     (intensity ≥ `intense`) beats. With #1's 57%-intense grading, persona is starved everywhere.

A user-supplied reference recap (YouTube `y6IE2CIyqGE`) was analysed as the target voice: a
**persona-forward, funny/arrogant omniscient "knowing narrator"** whose voice is ON in *every*
line — even deaths ("And so Rick's dream was pitifully crushed") — using rotating casual epithets
("the little rascal", "our young hero"), dry understatement, expectation-gap humour, and
**liberally quoted** punchy dialogue. This is almost exactly the existing `BASE_PERSONA` that the
pipeline suppresses.

## 2. Locked design decisions

| # | Decision |
|---|---|
| D1 | **Keep every panel** (do NOT adopt Manhwa Fresh R6 "cut to 1/3"); fix #3 with pacing, not panel-dropping. Preserves the `637357a` panel-integrity work. |
| D2 | **Build the a/b/c/d niche modules** (the 4 manhwa-type persona registers) on top of the defect fixes. |
| D3 | **Niche detection = metadata-primary, automatic, per-series.** Scrape the source page's genre tags + synopsis at `add-series`; classify deterministically; cache on the series. No manual tagging, no per-chapter content inference (avoids gemma non-determinism and "needs N chapters" lag). Empty/unmappable tags → default voice (graceful), no content fallback. |
| D4 | **a/b/c/d = niche genres:** A = Isekai/Power-Fantasy, B = Romance/Drama, C = Dark-Action/Revenge, D = Comedy/Slice-of-Life. |
| D5 | **Persona model = one always-on base voice + niche temperature dials.** The funny/arrogant knowing-narrator is the channel DNA, present on every line; A/B/C/D only modulate temperature (hot/cold/funny/menacing). Beat-gravity drops the *jokes*, never the personality. This **inverts** today's persona-off-on-dramatic gate. |
| D6 | **Multi-niche = primary + secondary blend.** The classifier ranks niches; primary governs the dominant temperature, secondary flavors matching beats (Nano = C primary + A secondary). Combos are data (two register blocks injected with a "primary governs / secondary flavors" note), not hardcoded code. |
| D7 | The existing `murim/modern/system` **setting** axis stays — it only guards framing *vocabulary* (anachronism safety). Setting × niche compose. |

## 3. Architecture

Three layers; one new concept (the **niche**). Each unit is independently testable.

### 3.1 Niche detection (acquisition + catalog)

- **`studio/sources/base.py`** — extend the `SeriesMeta` value object with `genres: tuple[str, ...]`
  and `synopsis: str` (default empty). `Capability.SERIES_META` already exists.
- **`studio/sources/asura.py`, `webtoon.py`, `elftoon.py`** — each `series_meta()` additionally
  parses the series page's genre tags + synopsis. **Fail-soft**: any scrape miss returns empty
  (the adapters are explicitly disposable for site churn — a genre-scrape break must never break
  discovery/fetch).
- **`studio/catalog/db.py`** — schema migration: add `niche_primary TEXT`, `niche_secondary TEXT`,
  `genres TEXT`, `synopsis TEXT` to the `series` table (nullable; back-compat for existing rows).
- **`studio/catalog/models.py`** — add the four fields to the `Series` dataclass.
- **`studio/catalog/repo.py`** — `upsert_series` persists the new fields; `get_series`/`list_series`
  select them.
- **New `tools/niche_modules.py`** (single purpose):
  - `NICHE_REGISTERS: dict[str, str]` — the four temperature blocks (A/B/C/D), each a short prompt
    fragment describing how the always-on base voice should be modulated.
  - `classify_niche(genres, synopsis) -> list[tuple[str, float]]` — deterministic **weighted
    keyword map** over the genre tags (synopsis as tiebreak); returns `(niche, score)` ranked by
    score (`[]` if nothing maps). The caller takes `[0]` = primary; `[1]` = secondary **only when
    its score ≥ 0.5 × the primary's score** (ratio threshold; at most one secondary), else no
    secondary. For Nano, action/martial-arts → C and system/power → A both score high, yielding
    **C primary + A secondary**. No LLM call.
  - A runnable `__main__`/`demo()` self-check asserting representative tag-sets map to the right
    ranking (e.g. `["Action","Martial Arts","Revenge"]` → C primary; `["Action","System","Leveling"]`
    → A; `["Romance","Drama"]` → B; `["Comedy","Slice of Life"]` → D; murim-action → C+A).
- **`add-series` flow** (CLI / wherever `upsert_series` is called from discovery) — after fetching
  `SeriesMeta`, call `classify_niche(...)` and store `niche_primary`/`niche_secondary` (+ raw
  `genres`/`synopsis` for audit). Idempotent re-classify on re-add.

### 3.2 Persona engine (narration)

- **`tools/narration_punchup.py`** — the core change:
  - **Invert the gate.** `CINEMATIC_RULES` is rewritten so the base persona voice is the **default
    on every line**. `classify_beats` keeps returning DRAMATIC/CONNECTIVE/COMIC but the meaning
    changes: it sets the *temperature* (DRAMATIC = restrained, drops jokes, **keeps the voice**),
    never "pure cinematic / no persona." `narration_plain` (grounded) survival is unchanged; the
    persona line still lands in `panel_narration[].line`.
  - **Inject niche registers** from the chapter's `series.niche_primary`/`niche_secondary` (passed
    via `--niche`/`--niche-secondary`, see 3.4). Primary block governs; secondary block is added
    with a "flavor beats that match this register" instruction. Falls back to base-only when no
    niche is set.
  - The `murim/modern/system` setting axis (`genre_key`, `GENRE_ADDONS`, `infer_genre_from_content`)
    stays as-is, composed alongside the niche.
- **`tools/gemini_narrative_pass.py`** — pass the niche into the beats prompt so the *grounded*
  per-panel line is already lightly in-voice (the punchup then dials temperature). `_DIALOGUE_RULE`
  is retained (it already invites quotes).
- **`tools/recap_style.py`** — the 6 `RECAP_STYLE_RULES` stay wired (prompt + QA). Only the
  **fragment/quote handling** is relaxed so a short, iconic, attributable shout (the user's
  "Ancestor!" class) survives instead of being neutralized as a trailing fragment.

### 3.3 Calibration + pacing (the remaining defects)

- **#1 intensity** — two levers:
  1. `tools/panel_understand.py`: recalibrate the SYSTEM prompt's `intensity` enum guidance (`:61`)
     so `intense`/`explosive` are reserved for genuine peaks and most panels grade `calm`/`tense`.
  2. Soften the intensity → delivery escalation. The chain is two hops:
     `tools/script_expander.py` (`_intensity_rank_for_beat`/`_escalate_tag_for_intensity`,
     ~`:736-749`) maps intensity → a mood **tag**; `tools/local_tts_from_manifest.py`
     (`mood_to_exaggeration`, ~`:154`) maps that mood → the TTS **`exaggeration`** value. Primary
     lever = soften the `script_expander` mood-tag escalation (fewer climax tags), which propagates
     to exag. Secondary lever (only if the curve itself is still too hot) = flatten
     `local_tts_from_manifest.mood_to_exaggeration`. Net: an `intense` grade no longer forces a
     climax-level delivery.

- **#3 pacing** — responsibility is split between a **splitter** (makes the floor satisfiable) and
  an **enforcer** (guarantees it):
  - **`tools/story_group.py` — splitter (primary lever).** Bound beat size so a beat's panel count
    stays proportionate to its narration (lower/adaptive `DEFAULT_MAX_BEAT_LEN`, currently `15` at
    `:42`). Splitting 12 panels into several beats — each with its own line — grows total narration
    with the panel count (legitimate content, not padding: they are distinct panels), so in the
    common case the floor below is satisfiable **from audio alone**.
  - **`tools/timeline_planner.py` — enforcer (backstop).** Add a hard **minimum per-panel
    on-screen floor** (~1.2s): a segment's on-screen duration becomes `max(audio_len, n_panels ×
    floor)`. **This changes current behavior** — today `build_cuts`/`compute_duration_sec` pin a
    segment to `audio + pad` and split evenly (`per = shot_dur / k`, ~`:977`; the comment at
    ~`:1756` states "we never stretch into silence"). New rule: never sub-floor, never drop a panel.
    When the splitter has done its job this branch **rarely fires** (audio already covers the panels
    at floor); for the **rare residual** where `n_panels × floor > audio_len`, the overflow tail is
    **held on its panel with the renderer's existing slow Ken-Burns push** (a brief B-roll hold, not
    dead black) — **NOT** a new cross-segment rolling-audio mechanism. Audio placement stays
    per-segment, so the deterministic `text_sha` audio↔segment staleness gate, the `total_drift` QA,
    and the renderer are **untouched**. The old `total = audio` invariant is dropped for
    `total = max(audio, n_panels × floor)`. **Tradeoff (for user review):** this is a small, bounded
    relaxation of the "every panel strictly under voiceover / no silence" promise for the rare tail,
    chosen over building an audio-overlap subsystem (YAGNI). If strict no-silence is required, the
    alternative is the heavier cross-segment rolling-audio path (which must then list the text_sha
    gate + total_drift QA + renderer as change-sites).
  - **`tools/prep_qa.py`.** Promote `flash_cut` (dur < ~1.2s on a non-exempt panel; threshold at
    `:1274`, currently WARN at `:1275`) from WARN to **BLOCKING** so a flash montage can never ship.

### 3.4 Wiring

- **`studio/pipeline.py`** + **`studio/worker.py`** — read `series.niche_primary`/`niche_secondary`
  and pass `--niche`/`--niche-secondary` into the `grouped`/`beated` (gemini_narrative_pass) and the
  punchup invocation. (`worker.py`/`dashboard` changes need a daemon restart; tools + pipeline.py
  are subprocesses → fresh on `git pull`.)
- **`studio.toml`** — `punchup = "cinematic"` is kept but its semantics now mean "persona-forward
  base + temperature dials." No new required keys; an optional toggle may gate niche injection for
  A/B testing.

## 4. Data flow

```
add-series → adapter.series_meta() {genres, synopsis}
          → niche_modules.classify_niche() → [primary, secondary?]
          → series row (niche_primary, niche_secondary, genres, synopsis)
   ...per chapter...
grouped/beated → gemini_narrative_pass (niche-aware grounded per-panel line)
              → narration_punchup (always-on base voice + niche temperature; sets panel_narration[].line)
scripted → gemini_verbatim voices panel_narration[].line (softened intensity→exag)
planned → timeline_planner (min per-panel floor; every panel shown)
QA → prep_qa (flash_cut BLOCKING)
voiced → TTS
```

## 5. Testing (TDD against the existing 1147-test suite)

- `tools/niche_modules.py`: `classify_niche` keyword→niche, multi-niche ranking, empty→`[]`,
  ambiguous tie-break, and the **secondary 0.5×-primary margin** (Nano tags → C primary + A secondary;
  a single dominant tag → no secondary).
- `narration_punchup`: base persona present on a DRAMATIC beat (voice retained, jokes dropped);
  niche register text injected for a given primary/secondary; setting-axis framing still gated.
- `story_group`: an oversized group splits so each resulting beat's `n_panels × floor` is ≤ its
  narration audio in the common case (the floor is satisfiable without extension).
- `timeline_planner`: a short line over many panels yields each panel ≥ floor and no panel dropped;
  segment total = `max(audio, n_panels × floor)`; the rare audio-shorter-than-`panels × floor` case
  holds the tail panel (slow push) without dropping it; per-segment audio placement is unchanged
  (`text_sha` gate / `total_drift` QA untouched).
- `prep_qa`: `flash_cut` now BLOCKING; a sub-floor cut fails the gate.
- catalog: schema migration adds columns and is back-compat with pre-existing series rows;
  `upsert_series`/`get_series` round-trip the niche fields.
- recap_style: a short attributable shout survives the fragment/quote handling.

## 6. Acceptance criteria

1. A newly added series gets a non-empty `niche_primary` when its source page exposes mappable
   genre tags; Nano Machine resolves to **C primary + A secondary**.
2. On a re-prepared chapter, rendered narration carries the **funny/arrogant base voice on every
   line**, including grave beats (jokes absent, personality present) — verified by listening, not
   JSON.
3. Intensity distribution on a typical chapter is **not** majority `intense`; a routine fall is not
   delivered at climax intensity.
4. A short iconic quote can appear in the voiced output.
5. No timeline item shows a panel below the ~1.2s floor; `flash_cut` blocks if violated; every
   distinct panel is still shown (D1 preserved); a segment extended past its audio holds its tail
   panel (slow push, not dead black) and does **not** move audio across segments (`text_sha` gate
   untouched).
6. Full test suite green.

## 7. Out of scope / non-goals

- Adopting Manhwa Fresh R6 (cut-to-1/3 / fewer screenshots) — explicitly rejected (D1).
- Restoring the per-chapter 10-word opening hook (deleted `e6d8951`; the bundle teaser is the
  sole intro).
- Per-chapter content-based niche inference (D3 chose metadata-only with default fallback).
- The `system_card_unshown` blocker (tracked separately in [[system-card-override-and-outro-fix]]).

## 8. Rollout / deploy notes

- `worker.py`/`dashboard` changes → `launchctl kickstart -k` the daemons on the Mini; tools +
  pipeline.py are subprocesses (fresh on pull).
- Niche fields populate at `add-series`; **already-tracked series need a re-classify** (re-run
  `add-series`, or a one-off backfill that calls `series_meta()` + `classify_niche()` for existing
  rows). Persona/intensity changes take effect on a re-run that re-does `grouped`/`beated`
  (delete `manifest.panels.understood.json` / reset to `visioned`), per the idempotent-skip rule.

## 9. Risks

- **Genre-tag scraping is site-churn-prone.** Mitigated by fail-soft → default voice; the niche is
  an enhancement, never a hard dependency.
- **Inverting the persona gate could over-joke grave beats.** Mitigated by the DRAMATIC temperature
  (jokes off, voice on) and the `recap_style` restraint rules; validate on a dark chapter (Nano).
- **Intensity recalibration is a prompt change** (non-deterministic at the margin). Validate via a
  distribution check on a sample chapter, not a single assertion.

## 10. Implementation notes (from spec review)

- The held-tail floor is **not new machinery**: `compute_duration_sec` already returns
  `max(base_min, audio + pad)` (`timeline_planner.py:307-308`) and treats a silent-hold-with-motion
  tail as designed ("`max_sec` only caps SILENT holds," `:303-306`). The floor just changes that
  per-segment floor from `base_min` to `max(audio + pad, n_panels × floor)` — same shape, same
  per-segment audio placement, so the `text_sha` gate / `total_drift` QA / renderer are genuinely
  untouched.
- `compute_duration_sec` does not currently receive a panel count; honoring
  `max(audio + pad, n_panels × floor)` means feeding it (or the dur computation just before
  `build_cuts`, ~`:977`) the segment's panel count. Localized change at the sites already named.
- `n_panels` for the floor must be the **kept/distinct** count `build_cuts` actually shows (after
  its redundant-frame drop), not the pre-drop count — consistent with D1.
