# Narration blend (cinematic + persona) + deterministic audio↔narration gate

Date: 2026-06-14
Status: approved (brainstorm), implementing
Owner: render-quality

## Problem

1. **Voice quality regression.** The Jun-13 default `punchup = "full"` rewrites every
   beat into a flippant gamer-slang persona, *deleting* the cinematic atmosphere
   (verified by transcribing the Jun-9/10 audio: old = cinematic, new = "speedrunning
   an escape"). The user wants **both**: cinematic atmosphere AND channel persona,
   chosen **alternate-by-beat** (dramatic beats cinematic, connective beats persona).

2. **Silent audio↔narration drift.** Voiceover clips are voiced once, then the
   voiced stage is skipped whenever clip files exist (idempotency keyed on *file
   existence*). When beats/narration are regenerated, the audio no longer matches the
   plan's `tts_text` — yet it ships. There is no deterministic guard; `narration_stale`
   is heuristic and holds-exempt. The user wants a **deterministic checkpoint**: if the
   voiced audio's source text ≠ the current narration, re-voice (use the new one).

Both are manhwa-agnostic.

## Part 1 — Punch-up: alternate-by-beat

`tools/narration_punchup.py`, `studio/config.py`, `studio/pipeline.py`.

- New mode value: `punchup ∈ {off, light, full, cinematic}`. `cinematic` = the new
  alternate-by-beat blend. `full`/`light`/`off` unchanged.
- **Deterministic beat classification** (no extra LLM): each beat's intensity =
  the strongest `scene_selection[].intensity` among its scenes
  (`explosive` > `intense` > `tense` > `calm`/`unknown`). Map:
  - `explosive`, `intense` → **DRAMATIC** → cinematic rule
  - `tense`, `calm`, `unknown` → **CONNECTIVE** → persona rule
- One prompt, each line tagged `[DRAMATIC]` / `[CONNECTIVE]`:
  - DRAMATIC: preserve atmosphere & imagery, ≤1 subtle persona touch, *movie-trailer
    cinematic is allowed* (the "not a movie trailer" clause is dropped for these).
  - CONNECTIVE: full persona (current behavior).
- `validate_line`: raise the upper length bound for DRAMATIC lines (cinematic ≈ longer
  than grounded). Keep grounding contract (no invented facts, cast names verbatim,
  caption words preserved, mood-tag preserved, no chrome). Rejection still restores the
  grounded line.
- `build_prompt(..., humor, genre, classes)` gains a per-group class map. Backward
  compatible: when `classes` is empty it behaves as today (full/light).

### Interface
```
classify_beats(beats_obj) -> Dict[int group_id, "DRAMATIC"|"CONNECTIVE"]
build_prompt(lines, cast, humor, genre, classes)   # classes optional
```

## Part 2 — Deterministic audio↔narration consistency gate

`tools/local_tts_from_manifest.py`, `tools/elevenlabs_tts_from_manifest.py`
(contract parity), `tools/prep_qa.py`, `studio/pipeline.py`, `tools/timeline_planner.py`.

- **Fingerprint at voicing.** A shared helper:
  ```
  narration_sha(text) -> str   # sha256 of normalize(text)
  normalize(text): strip leading [mood]/[delivery] bracket tags, collapse
                   whitespace, casefold  -> compare audio vs plan apples-to-apples
  ```
  Each `tts_index.json` clip entry gains `text_sha`. (The item already carries `text`.)
- **Deterministic staleness check** (`$0`, no LLM): for each plan segment with audio,
  `stale = clip.text_sha is None or clip.text_sha != narration_sha(plan.tts_text)`.
  - Exposed as `audio_consistency(plan, tts_index) -> {fresh:[], stale:[], missing:[]}`.
- **Idempotency fix (the root cause).** `_stage_voiced` re-voices a segment when its
  `text_sha` changed (or clip/index missing), and **keeps** unchanged clips →
  incremental, $0-cheap, auto-uses the new narration. Orphan clips pruned.
- **Gates.**
  - `prep_qa`: `narration_stale` ERROR now fires deterministically from
    `audio_consistency` (per-segment sha mismatch), in addition to the existing
    coverage heuristic.
  - `render_allowed`: blocks render when any shown segment is stale/missing audio.
- **Policy:** on mismatch → auto-re-voice the changed segments (TTS is local/$0).
  Re-voice still honors the existing `voice` approval gate (won't voice unapproved
  chapters).

### Edge cases
- A clip with no `text_sha` (pre-upgrade index) → treated as stale → re-voiced once,
  which backfills the sha. Self-healing migration.
- Held/substituted cuts carry no narration of their own — keyed by segment, not cut, so
  holds are unaffected.
- `[delivery_tag]` prefix in plan tts_text is stripped by `normalize`, matching what TTS
  actually spoke.

## Testing (TDD, pure functions first)

Part 1:
- `classify_beats`: explosive/intense→DRAMATIC; tense/calm/unknown→CONNECTIVE; max wins.
- `build_prompt` includes per-line class tags only in `cinematic` mode.
- `validate_line` accepts a longer DRAMATIC cinematic rewrite that `full` would reject;
  still rejects fact-inflation / dropped cast names / chrome.

Part 2:
- `narration_sha` stable under tag/whitespace/case normalization; differs on real edits.
- `audio_consistency`: fresh when sha matches; stale on edit; missing when no clip.
- `_stage_voiced` incremental: only changed segments re-synthesized (inject a fake
  SynthFn, assert call set == changed segments; unchanged clips untouched).
- `prep_qa` emits `narration_stale` ERROR on a doctored mismatch; none when aligned.
- `render_allowed` False while stale.

## Rollout / verification
1. Land code + tests (full suite green).
2. Set `punchup = "cinematic"`; regenerate beats for the 4 green chapters (ollama, $0).
3. Incremental re-voice (only changed segments) → previews; prep_qa + freeze_qa green.
4. Full structural review + debugging pass; fix findings.
5. Re-run QA on the Mini and control the scorecard.

## Non-goals
- No change to scene selection, holds, or the freeze fix (already shipped).
- No cloud spend (ollama + local Qwen only; thumbnails unaffected).
