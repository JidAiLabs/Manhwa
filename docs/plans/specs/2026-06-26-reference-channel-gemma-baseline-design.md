# Reference Channel Baseline Before Model Bakeoff

- **Date:** 2026-06-26
- **Status:** Planned before the next Ch16 Gemma render
- **Reference:** Mamoru Manhwa Nano Machine recap, `gUCfdJdNYmU`
- **Purpose:** Define the production structure first. Gemma is the first
  implementation baseline; alternative local VLMs are evaluated only after
  this structure has a clean full render.

## Why The Order Matters

Changing the model before fixing the production contract would make a result
ambiguous. A faster model could appear better simply because it makes a
different kind of mistake.

The test order is:

1. Remove grouping as a creative/rendering requirement; keep it only as an
   internal aid if it helps the structure below.
2. Run the fixed pipeline on Ch16 with the existing `gemma4:26b` baseline.
3. Compare the rendered body against the reference-channel observations and
   the prior Ch16 render.
4. Only after Gemma passes the structural and visual gates, run the exact same
   benchmark with candidate local VLMs.

## Reference Findings

The checked reference opening is an arc-level cold open, not a chronological
first-panel opening. It uses a later high-stakes window from the Nano Machine
chapter 5-8 area, establishes rules/social pressure/conflict quickly, then
continues the recap.

In the first measured 180 seconds, it spoke about 690 words (about 230 wpm)
over 104 detected visual shots. The median shot was about 1.47 seconds. These
are observations, not quotas:

- It does not use a fixed words-per-panel rule.
- It can carry one thought across fast visual changes.
- It can hold a meaningful image while a larger thought develops.
- It selects visuals rapidly without treating every panel as an isolated,
  full-sentence narration unit.

The reference comparison therefore has two separate proofs:

1. **Ch16 body proof:** visual selection, rolling narration, persona, system
   panels, crop behavior, QA, TTS, and final render.
2. **Nano 5-8 teaser proof:** an optional generic arc teaser, followed by a
   chronological body. This is not mixed into the Ch16 body proof.

## Corrected Production Contract

### 1. Panel Is The Visual Atom

A surviving story panel has one canonical visual identity.

- A panel is shown at most once in the normal timeline.
- A reused source panel requires an explicit, machine-readable reason such as
  a deliberate hold with no alternative visual. It is otherwise a QA error.
- A full-screen panel never receives a zoom that crops away its readable
  composition. Motion must be fit/hold or a restrained pan inside safe bounds.
- Bubble-only, chrome, empty, and pure-effect panels are excluded before
  narration and render planning. Their text remains available as story context.
- In-world system/stat/quest/notification panels are never treated as generic
  text-only clutter. They are protected story panels and must appear in the
  rendered timeline when their information matters.

Every exclusion records one deterministic reason:

- `chrome`
- `empty_bubble`
- `pure_effect`
- `duplicate_source`
- `text_context_only`
- `other_explicit_reason`

No model is allowed to silently turn an excluded visual into a narrated visual
beat later in the pipeline.

### 2. Narration Is A Rolling Speech Track

Narration is not one mandatory complete sentence per panel and not one
mandatory sentence per group.

Each narration cue has:

- an `anchor_panel`;
- zero or more contiguous `continuation_panels`;
- a spoken line whose length follows the story beat, not a word target;
- a local grounding record showing which panels/dialogue/caption support it.

Examples:

- A sword clash may be one or two words over a one-second cut.
- A reaction can be silent while the preceding sentence continues across it.
- A reveal, thought, or rule explanation may hold one panel for 15-20 seconds
  and carry a longer line.

The renderer consumes the panel timeline plus narration cues. It does not use a
group boundary as a visual cut or duration boundary.

### 3. Context Spans Are Optional Internals

The current `round(len(story) / 16)` group target is removed. It creates a
global magic beat count and caused Ch16's five-panel bundles.

Context spans may remain useful, but only for model reasoning. They are not a
product requirement, and they can be replaced by another mechanism if it follows
the reference-channel behavior better:

- They provide previous/next story context, identity continuity, and dialogue
  attribution.
- They have no fixed target count, maximum panel count, visual duration, or
  sentence count.
- They are selected at semantic transitions and may overlap for context.
- The model can receive a batched rolling window and emit cues for individual
  panels without making one model call per panel.

This retains narrative coherence only when it helps. It must never force five
images under one generic paragraph, repeat one image across continuation groups,
or impose an invented structure just because the code calls something a group.

### 4. Persona Is A Grounded Layer

The factual visual read happens first. The persona pass can change phrasing,
rhythm, intimacy, and one grounded comic observation, but cannot add plot.

Comic/mockery/humiliation cues are evaluated at the cue level, not suppressed
because an adjacent panel is intense. A visual gag should land with a concise,
genre-appropriate recap observation. Serious injury, grief, danger, and
unresolved reveals stay restrained.

## Gemma Ch16 Acceptance Run

Run on the Mac Mini, using only `gemma4:26b`, after the structural changes.
The old Ch16 render remains an A/B artifact; the new run is isolated.

The baseline is accepted only when all of these are true:

1. Every shown visual has a source-panel identity and no unexplained repeated
   parent panel/crop.
2. Empty bubble husks and publication chrome are absent from the shown timeline.
3. Every in-world system/stat panel is either shown or has an explicit,
   reviewed `text_context_only` reason. A `system_card_unshown` error fails the
   run.
4. The final narration contains no raw OCR fragments, invented dialogue,
   invented character identity, or visible-only filler as measured by QA and
   reviewed against selected panels.
5. The bald-head/mockery moment receives one concise, grounded comic beat
   rather than trailer-serious narration.
6. The rendered camera never full-screen-zooms a complete panel into a worse
   crop.
7. The final TTS/render is listened to and watched, not inferred from JSON.
8. Runtime is recorded by stage, but no word count, group count, or duration
   target is used as a pass condition.

The run report compares old and new on:

- shown/excluded panel ledger;
- system/stat coverage;
- repeated source/crop count;
- grounding and style flags;
- cue-to-panel storyboard;
- wall-clock stage time;
- human review of the final rendered segment.

## Model Bakeoff After Gemma

Once the Gemma baseline is structurally clean, run candidate local VLMs against
the same frozen Ch16 inputs, prompts, context spans, and acceptance sheet.

A new model wins only if it is both:

- faster in measured wall-clock preparation time; and
- at least as good in panel grounding, system/text retention, persona-aware
  narration, and final rendered quality.

It is not enough to win generic VLM benchmarks or produce a faster JSON file.
