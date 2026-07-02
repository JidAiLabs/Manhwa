# Adaptive Flow Narration — Design

**Date:** 2026-07-02 · **Status:** DRAFT for user review · **Owner:** narration pipeline

## 1. Problem

The per-panel 1:1 refactor (one panel = one line = one segment = one clip) fixed the visual
bugs — no skipped panels, no flash cuts, deterministic audio alignment — but the user's ch1
review found it **kills the narration**:

1. **Text shape.** Every panel gets one self-contained sentence. Subjects restart every line
   ("He's… He's… The group… Our guy…"), lengths are uniform (8–15 words), nothing leans into
   the next line. It reads as a slideshow of captions, not a storyteller.
2. **Delivery.** 112 independently-synthesized Qwen clips, prosody resetting on every clip,
   plus a silence gap at every group boundary. No intonation ever arcs across panels.
3. (Related, observed) long caption runs force repeat-cap holds of one panel — fewer, longer
   segments reduce those runs structurally.

User verdict (2026-07-02, verbatim intent): *"keeping grouping approach while panels change
and narration continues was better. however that should depend on panels and scenes and what
they tell us"* → **adaptive**: flow where the story flows, single-panel where a moment lands.

## 2. Goal / Non-goals

**Goal:** narration written and voiced as *connected prose spanning several panels* where the
scene wants it, while keeping every 1:1-era guarantee: every panel shown, ≥2.0s on-screen
floor, no flash cuts, byte-identical `segment_id` contract, per-segment QA grounding + heal.

**Non-goals:**
- NO return of per-group TTS stitching (one long clip whisper-aligned back onto panels) — that
  was built and REVERTED (stutter, timbre drift). A flow segment here is ONE synthesis whose
  panels are paced under it by the planner; nothing is stitched or aligned after the fact.
- NO change to grouping itself (`panel_understand` → `story_group` stays the context unit).
- NO renderer changes (multi-cut items and per-cut motion already exist).

## 3. Design

### 3.1 The unit: narration segments with a panel span

The beats writer (`gemini_narrative_pass`, already writing per group with the full beat in
context) emits, per beat, an ordered list of **segments** instead of the strict 1:1 list:

```json
"segments": [
  {"span": ["p000012.jpg"],                     "line": "…"},                  // solo
  {"span": ["p000013.jpg","p000014.jpg","p000015.jpg"],
   "line": "He's plummeting down the ravine — every impact stacking like a debuff,
            until the bottom finally catches him and the pain catches up."}    // flow
]
```

- **The LLM decides flow vs solo** from the panel understandings, per the prompt criteria:
  continuous action / traversal / montage-like progressions / caption-only runs → *flow*;
  emotional close-ups, reveals, punchlines, heavy-dialogue panels → *solo*. This is the
  user's "depends on panels and scenes" — judgment stays in the multimodal pass
  (the realized lesson from the understanding-first redesign).
- **Deterministic guardrails OUTSIDE the LLM** (validator, auto-repair then re-ask once):
  1. spans cover the beat's panels **exactly** — every panel in exactly one span, in reading
     order (no skips, no overlaps; the panel-collapse regression stays impossible);
  2. span length ≤ 4 panels;
  3. stamped `panel_kind == "system"` panels are always solo (`inject_missing_protected`
     continues to cover narration-less cards);
  4. **duration-aware word budget**: a span of N panels must carry enough words that the
     clip runs ≥ N × 2.0s at the configured wpm (and ≤ N × 6.0s) — the old "narration length
     ≠ panel count" failure is prevented arithmetically, not hoped away.
- Prose rule in the prompt: a flow line is ONE connected passage (clauses may lean across
  panel boundaries); solo lines stay independently speakable (spoken_fragment QA unchanged,
  applied per segment).

### 3.2 segment_id and downstream contracts

`segment_id = g{group:04d}_p{first_panel_index:02d}` — keyed on the span's FIRST panel. The
`g####_p##` byte-identity contract through script → TTS → timeline → render is unchanged;
ids stay unique and ordered (spans are disjoint + ordered). `manifest.script.json` /
`tts_index.json` / `clips/{segment_id}.wav` shapes are untouched — there are simply fewer,
longer segments (ch1: ~112 → est. 55–70).

### 3.3 TTS

One clip per segment, exactly as today (`local_tts_from_manifest`, per-clip `text_sha`
cache, per-clip exaggeration from the segment's intensity). A flow passage is one synthesis
→ prosody arcs across its panels natively. Longer text per clip = fewer prosody resets and
fewer inter-clip gaps by construction.

### 3.4 Planner

Per segment:
- span == 1 → current single-cut behavior (unchanged).
- span > 1 → `display_strategy: "multi_cut"` — the EXISTING `build_cuts` path ("pace the
  panels UNDER the voice") allocates the clip's real duration across the span's panels,
  honoring the ≥2.0s per-panel floor (extend, never drop) and flash_cut stays BLOCKING.
The word-budget guardrail (3.1.4) guarantees the duration exists for the floor to hold
without stretching a clip.

### 3.5 QA + heal

- `estimate_plan`/1:1 count checks become **cover checks**: segments' spans partition the
  shown panels (no panel uncovered, none double-covered).
- Grounding: a segment is judged against its span's panels together (grounding cache key =
  text_sha + span file list). caption_unvoiced looks in the span's OCR, not one panel's.
- Heal: `narration_heal` corrections address a segment; regen rewrites that segment from its
  span panels (same resume mechanics). Visual drops inside a span shrink the span's cut
  list — narration untouched (the hold/substitute machinery just shipped handles display).
- held_repeat pressure drops structurally: caption runs become flow spans instead of
  repeat-cap holds of one panel.

### 3.6 Config

`[narration].segmentation = "adaptive" | "per_panel"` (default `adaptive`; `per_panel` is
the escape hatch to today's behavior for A/B listening). No other knobs.

## 4. What changes where

| component | change |
|---|---|
| `tools/gemini_narrative_pass.py` | prompt + output schema: `segments[]` with spans; validator + one auto-repair re-ask |
| `tools/narration_punchup.py` | operates per segment line (mechanically unchanged; prompt example refresh) |
| script/verbatim packer | pack segments (not panels) into `manifest.script.json` paragraphs |
| `tools/local_tts_from_manifest.py` | none (per-segment already) |
| `tools/timeline_planner.py` | route span>1 segments to the existing `multi_cut` path; floor/flash logic unchanged |
| `tools/prep_qa.py` | 1:1 checks → span-cover checks; grounding + caption checks span-aware |
| `tools/narration_heal.py` | corrections carry span (regen scope) |
| `studio/worker.py` | none expected (heal loop untouched) |
| tests | schema/validator/word-budget units; planner span pacing; QA cover checks; e2e fixture |

## 5. Rollout

1. Build + suite green (baseline 1213).
2. Deploy to Mini (tools-only → subprocess-fresh; no daemon restart unless worker.py moves).
3. Re-run **ch1 only** from `grouped` (understanding + groups + vision are cached; beats →
   script → voice → plan re-run). Renders stay held; user listens to the new voice preview /
   watches the review render.
4. User verdict on ch1 → then ch2/3, then ch4–10 (same gate as agreed).

## 6. Open questions (user)

1. **Persona density:** fold the "~1 texture touch per 4 eligible lines" rationing into the
   new flow prompt as-is (recap_style rule #4 unchanged), or revisit wording first?
   sauce_density measured 11% vs ~25% target on ch1 — flow passages give the writer more
   room per line, which may close part of the gap on its own.
2. **Inter-segment gap:** keep the current `_gap.wav` at group boundaries only, or also
   shorten it (flow reduces boundary count already)?
3. **Span cap:** 4 panels right (≈8–24s per clip)? Longer risks Qwen drift on very long
   passages.
