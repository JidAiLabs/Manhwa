# Thumbnail cast contract — the seven unbuilt checks

*2026-09-05 · companion to the `manhwa-thumbnail` skill · owner-gated, nothing built yet*

> **2026-09-17 update (skill v2.0.0).** `triptych` is removed (owner decision; 0 of 27 example
> thumbnails use 3 panels), so item 5's three-position search is no longer needed:
> `before_after` gets its pre-climax, lead-first ref on both concept paths via
> `refs_for_style`. Also built that day: drawn-text refusal on the art
> (`thumbnail_build.art_text_words`, which covers item 2 for the art), both options + owner
> pick (`_h_series_thumbnail`, `/thumbnail/pick`), and `normalize_title`. New open items
> from the owner's examples: arrows that land on their labelled subject
> (`apple_vision.faces`), a second contrasting label, and number grounding for titles.
> The skill's "Not built yet" table is the current list.

The skill `manhwa-thumbnail` documents a contract the repo only partially implements:
**the hook decides the cast, the cast decides the reference panels.**
`publish_concept.select_before_ref` is the shipped proof it works — for 1 of 7 styles.

Seven checks are specified in the skill and marked `NOT BUILT`. Until they exist, the
skill instructs Claude to perform them by hand and say so. This is the build list.

Ordered by value per unit of work. Items 1-3 are cheap and close real holes; 4-5 make two
styles usable at all; 6 attacks the unresolved problem; 7 is bookkeeping.

---

> **2026-09-21 update (commit `3c8bfa0`).** Two items from the open list are BUILT: the SECOND
> LABEL (a hook design's headline is written to `concept.tags`, which the two-stage path never
> set) and the ARROW ONTO ITS SUBJECT — by construction, not detection: each entry of
> `thumbnail_styles.HOOK_DESIGNS` carries an `art_clause` that fixes where the lead stands and an
> `overlay.arrow_to` that aims there (face detection misses anime faces). The options are now the
> two hook designs the series' TEASER ranks first (`publish_concept.rank_designs`); `scene` +
> `before_after` are gone as options and before/after is a variant. §7 (licensed-text scanner) is
> PARTLY built: `_names_the_title` keeps the series name off the system-window card and out of
> the ranking; title, labels and description are still prompt-only. Still open and now the
> blocker for two more example designs (`contrast_pair`, `roll_call`): a COUNTERPART ref finder
> (§4 generalised from objects to a second person). `concept.json` gained `design`,
> `design_reason`, `claim_source`, `teaser_panels`, `card` — a start on §1's audit trail.

## 1. Ref audit trail in `concept.json`

**Why first:** diagnose mode cannot work without it, and every other item below is easier
to debug once the reasoning is recorded.

**Build:** extend the concept writer so each resolved slot records
`{slot, kind, panel, chapter, why_won, runners_up[]}`, plus the style's eligibility
verdict and the generation attempts with their running cost.

**Acceptance:** regenerate any series thumbnail; `concept.json` names a panel and a reason
for every slot in that style's contract. Diagnosing a deliberately-wrong `lead` ref from
the file alone identifies the slot without opening an image.

**Touches:** `tools/publish_concept.py` (`assemble_concept`, `build_bundle_concept`).

---

## 2. Style eligibility precheck

**Why:** the "LEVEL 999" class of failure at its source. A style whose slots the bundle
cannot fill must not be selectable — a hard-fail at ref time is a late, expensive way to
learn the style was wrong.

**Build:** a `style_is_eligible(style, ep_dirs, beats_objs) -> (bool, reason)` scan run
before `select_style` ranks candidates. `stat_callout` additionally requires a grounded
stat system, not merely a digit somewhere in the corpus.

**Acceptance:** a series with no monster panel never selects `vs_monster`. A series whose
largest narration number is 11 never selects `stat_callout`. Both report the reason.

**Touches:** `tools/thumbnail_styles.py`, `tools/publish_concept.py`.

---

## 3. Cost ceiling + attempt log

**Build:** an attempt counter and running dollar total in the job record; hard stop after
3 generations (~$0.39). Emit the same `[cost]` line shape the rest of the pipeline uses.

**Acceptance:** a run that hits the ceiling stops and escalates with the total spent, and
the number appears in `concept.json`.

**Touches:** `tools/thumbnail_gen.py`, `tools/thumbnail_build.py`, the worker job record.

---

## 4. Object-slot ref finder

**Why:** without it, `feat_object` cannot satisfy its own contract — and an absent object
reference is exactly when the model invents a weapon.

**Build:** find panels where a signature object is large and clearly drawn. The panel
understanding already carries subject and object language; prefer reading
`manifest.panels.understood.json` over adding a detector.

**Acceptance:** for a series with a signature object, the finder returns a panel a human
agrees shows it. For a series without one, it returns nothing and `feat_object` becomes
ineligible via item 2 — it must not guess.

**Touches:** `tools/publish_concept.py`.

---

## 5. Story-position refs, generalised — SUPERSEDED 2026-09-17 (triptych removed; see the update at the top)

**Why:** `triptych` needs the lead at three arc positions and has no implementation;
three refs from one chapter defeat the style's entire premise.

**Build:** generalise `select_before_ref` into
`select_lead_at(position, beats_objs, ep_dirs, climax_ci)` for `early | turn | realised |
weak | powered`. Keep its two hard-won properties: search by arc position, and require
`panel_shows_a_character`.

**Acceptance:** `triptych` returns three refs from three distinct chapters spanning the
arc. `before_after` returns byte-identical results to today's `select_before_ref` — this
is a generalisation, not a rewrite, and a regression here breaks a working style.

**Touches:** `tools/publish_concept.py`.

---

## 6. Lead-likeness check on the output

**Why:** nothing today ever compares the generated art to the reference it was given.
That absence is how four reference bugs stacked invisibly, and it is the only lever left
on the unresolved iconography problem that does not involve another negative clause.

**Build:** a perceptual comparison between the generated lead and its `lead` ref; flag
drift past a threshold. Calibrate the threshold on known-good and known-bad pairs from the
rejection corpus rather than picking a number — a wrong threshold here is a gate that
fires on its own blind spot, which this repo has paid for repeatedly.

**Acceptance:** the known-bad navy-polo thumbnail flags. Three known-good thumbnails do
not. Report the measured margin, not just the verdict.

**Touches:** new `tools/thumbnail_fidelity.py`.

---

## 7. Licensed-text scanner

**Why:** rule 1 is the one rule with legal weight, and it is currently enforced by asking
the model nicely. The art is text-free by design, so any glyph in the output is either
overlay text or a violation.

**Build:** OCR the finished jpg (on-device Apple Vision, already in the stack); hard-fail
if the series name or a chapter/episode number appears outside the known overlay regions.

**Acceptance:** a thumbnail with the series name burned into the art fails. A normal
thumbnail with its PIL overlay passes.

**Touches:** new check in `tools/thumbnail_build.py`.

---

## Not in scope

**Compositing real panels instead of generating art.** Considered and set aside: the owner
chose generation with mandatory references. It remains the documented fallback for the
case generation provably cannot handle, and item 6 is what would tell us we are in that
case.
