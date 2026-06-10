# Narration Grounding — A/B Design (Nano ch1)

**Date:** 2026-06-10
**Status:** SHIPPED 2026-06-10. Outcome: Variant **B (Gemini-only) won** the A/B → narration written by the image-seeing pass; OpenAI does not reword. **Stage A (cast) + B (dialogue-weaving) shipped**; **Stage C (closed-loop auto-correct) built but DISABLED** (over-flagged; degraded grp34 → "Elara" fantasy — needs a strictly-better-only safety guard). Follow-on prompt fixes shipped: dialogue-density (paraphrase + short quotes), action-beat energy, hard grounding ("no chandeliers"), non-empty narration retry, bubble-panel drop. Current state + next actions live in `/.continue-here.md`.
**Scope:** narration *prompting* only. Scene-dropping (timeline budget) and bubble-cleanup are separate threads tracked below, not in this change.

## Problem

Narration sometimes reads worse than the raw vision description, and sometimes better — the rule for "good line" looked non-obvious. Investigation resolved it:

- The grounded text the user preferred (ex1: "arms crossed, crackling with power") comes from `gemini_narrative_pass.py` → `beats[].what_happens` — the **multimodal stage that sees the panel**.
- The hallucinated text (ex1: "vanishing and reappearing behind them" — a teleport not in the art) comes from `script_expander.py` → `tts_paragraphs_v3` — the **OpenAI stage, blind to the image**, whose system prompt orders "feel like a movie trailer… fast-paced," pushing it to invent motion.

**Root cause:** the final narration is written by a model that cannot see the artwork and is told to dramatize. **Target:** cinematic prose, strictly grounded to the panel — write it where the art is visible.

## Approach — build BOTH variants, judge against images

Shared change (move the cinematic line into the image-seeing stage), then a toggle for whether OpenAI may reword. The two variants share ~90% of the work.

### Change 1 — `gemini_narrative_pass.py` (sees the art): emit `narration`
- Add `narration` (STRING) to `beat_schema.properties` and `required`.
- Prompt: "Write ONE line (1–2 sentences) of cinematic narrator prose for this beat. Ground it STRICTLY in what is visible. Flowing narration, NOT a caption: never 'is present', 'we see', 'the panel shows', 'reacts with'. Present tense, active voice, dramatic but invent NOTHING — no motion, teleport, before/after, or entity that isn't on the page."
- Additive manifest change (safe for downstream consumers). Cost: a little more flash-lite output (~pennies).

### Change 2 — `script_expander.py`: `--narration-source {legacy,gemini_verbatim,openai_polish}` (default `legacy` = unchanged)
- **Variant B `gemini_verbatim`:** use each beat's `narration` as the paragraph text verbatim. OpenAI/deterministic layer only assigns mood tags (`_ensure_tts_tags_from_beats` from beat intensity), SFX, shots, pronunciation. No rewording → **zero new hallucination by construction.**
- **Variant C `openai_polish`:** pass Gemini's `narration` as a GROUNDED DRAFT. Swap the "movie trailer / dramatize" mission for: "Tighten this grounded draft to the word budget and smooth cross-beat flow. Introduce NO new action, event, motion, sequence, or entity. Do not dramatize beyond the draft." Add the user's two A/B pairs as few-shot (grounded→good cinematic). OpenAI now edits good prose instead of inventing from a caption → small residual drift, best prose + clean word-budget control.

### Change 3 — `tools/narration_grounding_check.py` (new): quality check vs images
- Per beat: send the kept scene image(s) + the narration line to Gemini flash-lite (multimodal): "Does this line assert anything NOT visible in these panels? List ungrounded claims. Score fidelity 1–5 and prose 1–5."
- Emit `narration_compare.html`: per group → panel thumbnail | Variant B line + verdict | Variant C line + verdict. Reuse `studio/qa.py` HTML patterns.
- This is the objective instrument that decides B vs C (plus the user's eyeball).

## Run plan (Nano ch1)
1. Re-run beats (Gemini) → `manifest.beats.json` now carries `narration`.
2. Two scripts: `manifest.script.B.json` (gemini_verbatim), `manifest.script.C.json` (openai_polish).
3. Grounding check on both → `narration_compare.html`; open it; user picks B or C.
4. Then (separate steps): cache Modal weights → voice the chosen variant; fix dropping; fix bubbles.

Est. cost ch1: beats + 2 nano scripts + grounding check ≈ <$0.10.

## Regression safety
- `--narration-source` defaults to `legacy`; existing 170 tests and current behavior unchanged unless opted in.
- `narration` is an additive beat field; `timeline_planner`/`qa` ignore unknown fields.

---

## v2 — A/B OUTCOME + Cast-aware dialogue recap engine (approved 2026-06-10)

**A/B result (judged against the panels, `narration_compare.html`):** Variant **B (Gemini-only, image-grounded) won** — 23/36 head-to-head, fewer invented events (B 13 vs C 17 hallucination tags), mean fidelity B 2.78 vs C 2.67. C's OpenAI polish re-invented events (grp22 "first strike meets a crackling shield" — not on the page). **Decision: narration = Gemini's image-grounded `narration`; do NOT let OpenAI reword it.** The strict judge also flags emotional inference ("desperate resolve") as ungrounded — that tasteful color is wanted, so the gate must target invented *events*, not emotion.

**New requirement (from the example channel):** the recap must be **character-aware and dialogue-rich**, not anonymous panel description. Root causes found in Nano ch1 data (grp32–34):
1. Beats are narrated **per-group in isolation** — no chapter cast/continuity, so every scene says "a figure"/"a young man".
2. Dialogue is **deliberately stripped** (anti-OCR-echo) — even punchy story lines ("Hey, Ancestor-nim!", captured verbatim in OCR p105) get paraphrased away.
3. OCR names/address-terms discarded as noise.
Quoting real dialogue + real names is **more grounded**, so this aligns with the A/B result.

### A · Cast registry — `tools/cast_builder.py` (new)
One pass over all OCR + a sampling of panels → `manifest.cast.json`:
`{cast:[{id, canonical_name, aliases[], role, visual_description, is_protagonist}]}`. Names from dialogue/OCR; stable descriptive handles for the unnamed; the protagonist designated. Single Gemini multimodal call (cap ~24 images).

### B · Cast-aware + dialogue-weaving narration — modify `gemini_narrative_pass.py`
- New `--cast <path>`: thread the cast into every group payload; instruct: **name characters consistently** by matching appearance to the cast (canonical names / "the protagonist"); stop re-introducing "a figure".
- **Dialogue-weaving** driven by the `bubble_mode` already detected: `spoken`/`shout` + short punchy line → **quote it, attributed** ("'Hey, Ancestor-nim!' the young man calls"); `inner_thought` → internal monologue; long/expository → paraphrase. Never quote UI/SFX/chrome. (OCR is already in the payload; the image is the exact-wording source.)

### C · Closed-loop grounding gate — `tools/narration_refine.py` (new) + `--corrections` on beats
Orchestrates: run beats(cast) → judge each `narration` vs its panels (reuse `narration_grounding_check` judge) → for beats tagged **hallucination = invented event** (NOT emotional `minor_drift`), regenerate just those groups via `gemini_narrative_pass --resume --corrections <json>` (per-group note: "remove invented X; stay to what's visible + the dialogue") → repeat ≤N. Report residual.

### Run order (Nano ch1)
`cast_builder` → `gemini_narrative_pass --cast` → `narration_refine` (judge+regen loop) → `narration_compare.html` for review. Then voice the result (cache Modal first).

## Out of scope (separate threads, tracked)
- **Dropping:** group 22's `scene_selection` is all-"keep"; the cull is timeline/montage **budget** dropping non-redundant uniques. Fix = protect "keep" panels from budget culls (flex runtime instead).
- **Bubbles:** clean at the **crop stage** (art-boundary-aware crop + inpaint), not Blender. Extends SP2 #4 bubble-inpaint; outside-art bubble fragments get cropped/masked off.
- **Voice:** single Qwen persona ("deep, resonant male narrator"); nothing to pick — keep it. Cache Modal weights (`modal.Volume`) before voicing more.
