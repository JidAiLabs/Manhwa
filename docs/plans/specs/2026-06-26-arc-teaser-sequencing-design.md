# Arc Teaser Sequencing — Design Note

- **Date:** 2026-06-26
- **Status:** Discovery complete; implementation pending
- **Reference:** Mamoru Manhwa, `gUCfdJdNYmU`
- **Series checked:** Nano Machine, Asura chapters 5-8 candidate window

## Finding

The reference channel does not open with the first panel of the first covered
chapter. It builds an arc-level cold open from a later high-stakes window, then
uses narration to explain the situation before continuing the recap.

For the checked Nano Machine video, the opening audio starts with the Demonic
Academy entrance setup: entrance day, ten-year gate cycle, brutal test rules,
six clan heirs, the seventh prince missing his black badge, then the first test.
Visual matching against source pages shows the first hook window is assembled
from multiple adjacent chapters, with strong matches in Asura chapters 5-8 and
the test payoff continuing through chapters 7-8.

This is a different layer from per-panel narration:

- **Per-panel narration** fixes alignment, coverage, persona, and pacing inside
  the chosen sequence.
- **Arc teaser sequencing** chooses which chapter/window appears first, then
  rewinds or continues.

Do not solve teaser sequencing by hard-coding Nano chapter numbers or by
manually rewriting one chapter. It must be generic.

## Measured Reference Pace

First 180 seconds of the reference video:

- Approx. 690 transcribed words, about 230 wpm.
- 30-second windows ranged about 204-258 wpm.
- Scene-cut detector found 104 shots, average 1.73s, median 1.47s.
- No detected opening shot held longer than 4.53s.

Conclusion: the channel carries dense exposition over rapid visual sequencing.
It does not use a static words-per-panel rule. Long explanation can ride over
many short shots; a single important panel can still breathe when the story
needs it.

## Target Architecture

Add an optional `plan_teaser` stage at the publishable video/bundle layer. The
chapter pipeline still produces reusable chapter artifacts; the teaser is not
baked into every chapter segment.

```text
per-chapter understanding/story caches
  -> selected dashboard bundle/video batch
  -> teaser_planner.py over that bundle only
       outputs manifest.teaser.json
  -> compose video: teaser section once, then chronological chapter bodies
```

`manifest.teaser.json` should contain:

- `source_chapters`: source chapter ids used by the teaser.
- `scene_files`: selected panels in teaser order.
- `reason`: why this window is a hook.
- `rewind_line`: the transition back into the chronological recap.
- `spoiler_boundary`: what the teaser is allowed to reveal.

## Batch-Based Hook Planning

An arc-level hook cannot be planned from a single chapter. The planner must see
the batch of chapters the video is allowed to cover and choose the strongest
high-stakes window inside that batch.

Do not hard-code chapter counts like `N..N+8` into the creative logic:

- The input unit is the selected video batch or bundle, such as chapters `1..10`
  or `24..32`.
- The hook is selected by score from the batch, not by a fixed offset from the
  first chapter.
- Numeric limits are cost guards only: `max_hook_scan_chapters`,
  `max_hook_panels`, or `max_teaser_seconds`. They never decide what is
  narratively important.
- If the operator selects a bigger batch, the hook planner may use any chapter
  inside that batch, subject to spoiler rules.
- If the operator selects only one chapter, the planner can still make a cold
  open from that chapter, but it cannot claim to match the reference channel's
  multi-chapter arc hook style.

Default policy:

- For a single-chapter body proof such as Ch16, no reference-style teaser is
  evaluated. That test only proves panel/cue narration, QA, system panels, crop
  behavior, TTS, and render quality.
- For a reference-style opening episode, first create or select the intended
  batch. The planner then scores windows inside that batch.
- A strong hook window usually contains public rules/test, humiliation, missing
  token, authority pressure, power/system reveal, decisive action, or a vivid
  reversal. These are scoring signals, not chapter-number rules.
- The teaser may only use chapters inside the selected batch unless the operator
  explicitly enables future teasing.
- Cache per-chapter understanding/story summaries so larger batches do not
  re-pay the same visual analysis.

For the checked Nano reference, if the chosen video batch is chapters `1..10`,
the planner should naturally discover the chapter 5-8 academy/test hook. If the
batch is only Chapter 1 or only Chapter 16, that hook is outside the allowed
coverage and must not be used.

## Dashboard Workflow

The dashboard should keep two levels separate:

- **Chapter artifacts:** prepare, script, voice, QA, and render each chapter as
  reusable chronological material.
- **Published video/bundle:** choose an ordered set of chapters for one upload,
  plan one optional story teaser for that upload, then concatenate or compose
  the final video.

This maps onto the existing `bundle` / `bundle_chapter` model: a bundle is the
publishable video. The new teaser planner belongs beside bundle creation and
concat, not inside every chapter's normal render.

Recommended dashboard flow:

1. Auto-suggest the next video batch from available chapters using estimated
   runtime, chapter continuity, and arc-boundary signals.
2. Let the operator accept or edit the chapter range. The selected bundle is the
   source of truth.
3. Queue missing per-chapter prepare/voice/render work for the chapters in that
   bundle.
4. Run `plan_teaser` once for the bundle using only the selected chapters,
   unless future teasing is explicitly enabled.
5. Compose the output as `teaser -> rewind/bridge -> chronological bodies ->
   outro`.

The auto-batcher may have soft preferences such as target minutes or max scan
cost, but those are operational defaults. They must not become creative rules.
If the operator selects chapters `1..7`, the hook planner scores chapters
`1..7`. If the operator selects `1..12`, it scores `1..12`.

## Tail Handling

If only one or two chapters remain, the dashboard should not force a fake
multi-chapter arc hook. It should offer a tail policy:

- Merge the tail into the previous bundle if runtime and story continuity allow.
- Hold the tail until more chapters exist, for an ongoing series.
- Publish a shorter tail episode with a local hook drawn only from those
  chapters.
- Explicitly allow future teasing only if the operator accepts the spoiler
  tradeoff.

For completed backlogs, the default should be "merge tail into the previous
bundle unless that makes the video awkwardly long." For ongoing series, the
default should be "hold until enough new material exists" unless the operator
wants a short update.

## Teaser Scope

The reference-style cold open applies once per published video, not once per
chapter segment.

- If the channel publishes chapters `1..10` as one video, there is one teaser
  for that `1..10` video.
- If the channel publishes chapters `11..20` as the next video, there is one new
  teaser for `11..20`.
- If the channel publishes single chapters, each single chapter can have a local
  cold open, but it cannot use a later arc hook unless future teasing is
  explicitly enabled.

This is separate from the old branding intro. The channel-branding intro remains
disabled; the teaser is story content selected from the source panels.

## Flashback / Story-Memory Inserts

The reference channel also uses flashback images inside the body to keep a
bundle understandable without fully replaying every earlier setup scene. This
is not the same as the opening teaser.

Definitions:

- **Teaser:** a pre-body cold open that may jump to a later high-stakes moment
  inside the selected bundle, then rewinds or bridges into the chronological
  recap.
- **Flashback insert:** a short visual reminder used during the body while the
  narration explains a current beat. It should come from already-established
  story context, not from unearned future spoilers.

Allowed flashback sources should be computed generically:

- Earlier chapters already published in previous bundles for this series.
- Earlier chapters inside the current bundle, once the body has reached or
  established that context.
- A curated `story_memory` cache produced from chapter understanding: key
  betrayals, rules, items, system panels, relationships, vows, injuries,
  transformations, and public humiliations.

Disallowed by default:

- Later chapters outside the selected bundle.
- Later panels inside the current bundle that would spoil a reveal before the
  body reaches it, except for the explicit opening teaser.
- Title cards, licensed-series chrome, ads, credits, or empty speech-bubble
  panels.

Implementation shape:

```text
per-chapter understanding
  -> story_memory.index.json
  -> bundle body planner
       may attach memory_insert cuts to narration beats
       marks them as flashback/source_memory in the render plan
```

Flashback inserts should be sparse and purposeful. They are for context,
contrast, irony, or emotional payoff, not for filling time. The narration
should not announce them mechanically as "flashback"; it should make them feel
like natural memory or explanation.

## Generic Selection Signals

Score candidate windows across the next few available chapters, not just the
current chapter:

- high-stakes rules: exam, test, rank, survival, expulsion, execution, contract.
- social pressure: public humiliation, missing token/badge, clan/family rejection.
- power/status reveal: system window, rank jump, hidden skill, impossible result.
- antagonist pressure: named enemy, elder, clan heir, authority figure.
- visual variety: wide establishing shot, character lineup, crowd reaction,
  close-up, decisive action.
- story clarity: a viewer can understand the conflict in under 20 seconds.

The planner should prefer windows that create a question without ruining the
main payoff. It may use later panels as a hook, but it must preserve reveal
pacing and avoid naming hidden identities before the source itself resolves
them.

## Non-Goals

- No per-series chapter-number rules.
- No licensed-title or title-card narration.
- No fixed word budget.
- No fixed bundle size.
- No replacing per-panel narration; teaser sequencing feeds it.
- No inserting the teaser into every chapter artifact.
- No treating flashback inserts as permission to spoil future reveals.

## Validation

For a Nano proof, compare:

1. Chronological Chapter 1 opening.
2. Teaser opening selected from chapters 5-8, then rewind/body.

Pass criteria:

- The teaser is understandable without prior context.
- The body remains panel-aligned and does not lose panels.
- The opening feels closer to the reference channel: immediate rules/stakes,
  social pressure, and rapid visual movement.
- The teaser improves retention potential without making the recap confusing.
