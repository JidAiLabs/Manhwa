# Per-Panel Rolling Narration — Design Spec

- **Date:** 2026-06-19
- **Status:** Approved (design); pending implementation plan
- **Owner:** Candas / Claude
- **Series under test:** Nano Machine, Chapter 1 (series 2, isolated on the Mini)
- **Supersedes:** the per-group + `narration_microbeats` materialization introduced in `b455a34`

## 1. Problem

Manhwa recap channels narrate continuously and cinematically while giving **every meaningful panel its own moment** — a quick action panel gets a punchy phrase, a pivotal panel gets a full beat, and the lines flow as one story. Our pipeline targets this but is built on the wrong unit.

Today the narration **unit is the group, not the panel**:

1. `story_group.py` groups consecutive panels into beats (shots).
2. `gemini_narrative_pass.py` writes **one `narration` string per beat** (`manifest.beats.json`). The "cover every panel" instruction is soft, so the model montages dense groups (observed: g0018 = 5 panels, ~3 narrated).
3. `script_expander.py` with `--microbeats` (`narration_microbeats`, default on in production) then **splits that one string by word-count** (`_split_recap_microbeats`, `tools/script_expander.py:1347`) into up to *distinct-panel-count* sub-beats, and assigns panels to text chunks **positionally** (`_scene_for_microbeat`, `tools/script_expander.py:1471`: `pos = index*len/total`).

The split is text-driven and the panel→chunk mapping is positional, so:

- **Under-coverage:** a dense group's narration covers fewer panels than it has → leftover panels are shown under an unrelated line (the falling figure + explosion play under the "blue hand" line).
- **Misalignment:** the line describing panel A can land on panel B because assignment is by position, not content.
- **Repeats (mostly fixed):** a sparse group split into more parts than panels re-cut the same frame — capped at distinct-panel count in `c32b079`, but mis-distribution remains.

**Root cause:** the group→one-narration→word-count-split chain cannot recover panel-to-line alignment, because alignment information was never produced. Microbeats is a band-aid for the wrong unit.

Key measurement that reframes the trade-off: **with `narration_microbeats` on, each sub-beat already becomes its own timeline segment and its own TTS clip** (`_build_microbeat_shot`, `tools/script_expander.py:1481`). So we already pay ≈ one clip per panel. Moving to per-panel narration **does not add TTS clips** (TTS is the throughput bottleneck — see `render-not-bottleneck-tts-is`); it only changes where each clip's text comes from. The real trade-off is narration quality/flow, not cost.

## 2. Goals / Non-Goals

**Goals**
- Every surviving story panel gets exactly **one** narration line and **one** cut, aligned by construction.
- Lines flow as one continuous, persona-driven story (rolling continuity), not disconnected captions.
- Line length is proportional to each panel's weight (quick hit vs full beat), model-chosen from content.
- In-world system/info cards are never dropped (fixes handover bug **b**, p000114).
- No increase in TTS clip count or render time vs the current microbeat config.

**Non-Goals**
- Changing grouping itself — grouping stays as the **continuity unit** (context for the rolling pass + pacing), it is no longer the narration unit.
- Changing the understanding pass (`panel_understand.py`), the cast builder, TTS backend, or render.
- Re-enabling `narration_register` (disabled in production — see `register-off-default-persona-narration`).
- Any batch run before Ch1 validates side-by-side.

## 3. Decisions (locked with user, 2026-06-19)

1. **Unit:** Per-panel narration in a single rolling pass. Grouping kept only as continuity context. Microbeats removed from the live path.
2. **Coverage:** Narrate every panel that survives the existing **chrome / pure-effect / duplicate** drops in `story_group.py`. System/info cards are kept and narrated.
3. **Length:** Model decides length from each panel's understanding (action / reveal / dialogue / establishing) + intensity. Soft "match length to weight" instruction, no hard word caps.
4. **Generation unit:** One LLM call **per story-group**, emitting one line per panel in that group, with the prior group's emitted lines threaded as the running spine. Reuses existing per-group checkpoint / resume / 429 / heal infrastructure.

## 4. Architecture & Data Flow

Stage names are unchanged. The change is what `beated` emits and what `scripted` consumes.

```
grouped:   panel_understand.py  → manifest.panels.understood.json   (every panel described — unchanged)
           story_group.py       → manifest.groups.json + manifest.story.json
                                   (grouping unchanged; ONE classification fix — §6)

beated:    cast_builder.py      → manifest.cast.json                (unchanged)
           gemini_narrative_pass.py → manifest.beats.json
                                   *** CORE CHANGE: each beat carries panel_narration[] ***
           narration_punchup.py → manifest.beats.json (in place)
                                   *** persona applied per panel-line ***

scripted:  script_expander.py   → manifest.script.json
                                   *** microbeat split REMOVED: one segment per panel ***
           narration_sanitize_pass.py → manifest.script.json (advertiser safety — unchanged)

voiced:    local_tts_from_manifest.py → clips/{segment_id}.wav      (unchanged contract)
planned:   timeline_planner.py  → render.plan.json                  (per-segment pick collapses to identity — §6)
```

**Continuity channels (both already exist):**
- **Chapter spine** — `manifest.story.json` (logline + premise + ordered arc), passed to `gemini_narrative_pass.py` via `--story` (`tools/gemini_narrative_pass.py:854`).
- **Rolling tail** — `previous_narration` threaded per call (`tools/gemini_narrative_pass.py:1135`). Under per-panel this carries the prior group's emitted **panel-lines** (a window), so each group continues from the last lines spoken.

## 5. Manifest Contract Change (the API)

Per CLAUDE.md, the manifest is the API; every downstream consumer of the changed field must be updated, and `segment_id` must stay byte-identical across `script_expander → tts → timeline_planner`.

### `manifest.beats.json` — each beat gains a per-panel array

Current beat keys: `group_id, scene_files, beat_title, what_happens, narration, emotional_turn, conflict_or_stakes, reveals_or_info, hook, mood_words, rendering_hints, scene_selection`.

**Add** (new source of truth):
```jsonc
"panel_narration": [
  { "scene_file": "p000113.jpg", "line": "Sky Corporation. The name looms like a verdict.", "tag": "[ominous]" },
  { "scene_file": "p000114.jpg", "line": "Seventh-generation nano machine — activation begins.", "tag": "[tense]" }
]
```
- One entry per file in the beat's `scene_files`, **same order**. Invariant: `len(panel_narration) == len(scene_files surviving drops)`.
- `tag` is the optional ElevenLabs/mood tag (today's leading-bracket tag), now per panel.

**Keep** `narration` (one string) = the panel lines joined with a space. This is the **back-compat bridge** for the **beat-`narration` readers we are NOT changing** — `narration_report.py` and `narration_accept_better.py` — so they keep working on the full text. New per-panel consumers read `panel_narration`.

Two categories need no bridge:
- `narration_punchup.py` reads beat `narration` today but **is being changed** to per-panel-line (§6), so it consumes `panel_narration` directly.
- The **script/plan per-segment readers** — `narration_sanitize_pass.py` (operates on `manifest.script.json` `script_paragraphs` / `tts_paragraphs_v3`) and `narration_consistency.py` (compares the render plan's per-segment lines to clips) — never read beat `narration` at all. Their per-segment arrays will already be one-line-per-panel and stay index-aligned, so no change is required.

### `manifest.script.json` — one segment per panel
- `script_expander.py` emits one segment per `panel_narration` entry. `segment_id = f"g{gid:04d}_p{i:02d}"` where `i` is the **panel index within the group** (`tools/script_expander.py:2274`). This is the same scheme microbeats used (`p##` was already the per-segment index), so TTS filenames and timeline lookups are unchanged in shape — only their text source moves.
- Each segment's `scene_files = [that one panel]`. No positional fan-out.

## 6. Component Changes

| File | Change |
|------|--------|
| `tools/gemini_narrative_pass.py` | Prompt + emission: for each group, return **one line per listed panel** (`panel_narration[]`), proportional length from understanding+intensity, flowing from `previous_narration` + spine. Thread the prior group's panel-lines as `previous_narration`. Write `narration` as the joined string for back-compat. |
| `tools/narration_punchup.py` | Apply the cinematic persona pass **per panel-line** (iterate `panel_narration`) instead of per beat; keep `narration_plain` semantics; rejoin `narration`. |
| `tools/script_expander.py` | **Delete the live microbeat path** (`_split_recap_microbeats`, `_scene_for_microbeat`, `_build_microbeat_shot` no longer on the default path). Build one segment per `panel_narration` entry, `scene_files=[panel]`. Keep functions behind the `--microbeats` flag for the A/B fallback (§9). |
| `tools/timeline_planner.py` | Each segment's `scene_files` is its one panel, so `_pick_for_segment` (`tools/timeline_planner.py:1351`) returns that panel — selection collapses to identity. **Keep** `inject_missing_protected` as a belt-and-suspenders net. Keep system-card protection. |
| `tools/story_group.py` | **System-card classification fix (bug b) — net-new heuristic gate:** a styled in-world system/info card whose understanding gave `panel_kind == "caption"` is reclassified **story** before it reaches `caption_files()` / `merge_caption_solos` (`tools/story_group.py:242`, `:251`). This gate does **not** exist today — it is new code: ALL-CAPS, 2–8 words, `text_coverage < 0.20`, not chrome. The card then survives into the per-panel set, gets a line, and renders. True external caption boxes still drop. (The understanding pass currently mislabels p000114 as `caption` upstream; this gate corrects it downstream without re-running understanding.) |
| `tools/prep_qa.py` | Coverage check becomes per-panel: every kept story panel must have a non-empty line and appear in exactly one cut. Drop microbeat-specific assertions. |
| `tools/narration_heal.py` | Heal re-narrates a **group's** `panel_narration` (the existing per-group unit) when QA flags it, never dropping a panel line. |
| `studio/config.py`, `studio/pipeline.py` | `narration_microbeats` retained as a **switchable fallback** flag through Ch1 validation (default flips to per-panel); removed after per-panel is confirmed. |

## 7. Coverage Rule

A panel gets a line **iff** it survives the existing `story_group.py` filters:
- **chrome** (logos / watermarks / page numbers) — dropped
- **pure-effect** (a glow / impact streak naming nothing concrete) — dropped
- **duplicate** (near-identical consecutive frame) — collapsed
- everything else, **including system/info cards** — kept, one line, one cut.

This matches "explain each meaningful image" while skipping content-free frames. It preserves the filter-quality-**before**-narration ordering (`recap-quality-ordering-and-qa-gap`).

## 8. Error Handling — coverage can never silently drop

If the model returns fewer lines than the group has surviving panels (or malformed JSON), a deterministic **repair-fill** runs — the same guarantee `repair_to_shots` gives grouping (`tools/story_group.py:102`):
- Pad missing panels with a minimal grounded line derived from that panel's understanding (`what_happens` / subjects), so every panel still has a line.
- Over-long returns are truncated to the panel list (extra lines merged into the nearest panel, never creating phantom panels).
- The invariant `len(panel_narration) == len(surviving scene_files)` is asserted before write; a violation fails the beated stage loudly rather than emitting a misaligned manifest.

## 9. Backward Compatibility & Rollout

- **`narration_microbeats` stays switchable** during validation. Per-panel is the new default; the old microbeat path remains reachable via the flag for the Ch1 A/B, then is removed.
- **Old manifests:** `timeline_planner.py` already falls back to a group-level item when no per-paragraph rows exist (`tools/timeline_planner.py:1338`); the joined `narration` string keeps legacy beats readable. No migration needed for already-rendered chapters.
- **Validation ladder** (per `confirm-upstream-before-expensive-downstream` and `qa-scan-before-rendering`):
  1. Implement behind tests (TDD).
  2. Run **Ch1** beated→scripted, render the plan, and produce a **side-by-side** vs the current group/microbeat output: every panel explained, flowing, cinematic, cards shown with text, no repeats, no misalignment.
  3. User reviews the side-by-side before any batch.
  4. Only then resume **ch1-20** (`/tmp/start_nano_20.py`) for the 20-chapter average, then the full 317-chapter run.

## 10. Testing (TDD)

Unit:
- **N-lines-per-group parse:** a mocked group call returning N lines yields N `panel_narration` entries aligned to `scene_files`.
- **Repair-fill:** fewer/more/blank lines → invariant holds, missing panels padded from understanding, no phantom panels.
- **segment_id alignment:** `script_expander` emits one segment per panel, `g####_p##` byte-identical to the panel index; TTS/timeline lookups resolve 1:1.
- **System-card survives:** a `panel_kind == "caption"` ALL-CAPS 2–8-word low-coverage card (p000114 fixture) is reclassified story by the new gate and appears in the final cuts; a true external caption box of the same kind still drops.
- **punchup per-line:** persona applied to each line; `narration_plain` preserved; rejoined `narration` matches.

Integration / E2E:
- Drive `gemini_narrative_pass.main()` (mock backend) → `script_expander.main()` → `timeline_planner.main()` on a small fixture; assert every kept panel has a distinct cut with its own line.
- Ch1 real run (Mini), side-by-side artifact for human review.

Regression:
- Full `pytest` suite green (≈762 at last run this session; re-confirm the exact count at commit time — CLAUDE.md's "170" is stale) before commit.

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Per-panel lines read choppy / lose flow | Rolling `previous_narration` window + chapter spine; proportional length; persona punchup per line. Judge on the Ch1 side-by-side before batch. |
| Model still under-returns lines for dense groups | Deterministic repair-fill guarantees coverage; invariant assert fails loud on misalignment. |
| Contract change breaks a downstream reader | `narration` joined-string kept for legacy readers; `segment_id` scheme unchanged; per-consumer audit in §6; full suite must stay green. |
| TTS time regresses | Clip count ≈ unchanged (already one clip/panel under microbeats-**on**, the production config in `studio.toml`). Pin the Ch1 A/B baseline to a microbeats-on prepare so the clip-count comparison is apples-to-apples; verify Ch1 timing before batch. |
| System-card reclassification over-captures true captions | Tight signal (ALL-CAPS, 2–8 words, `text_coverage<0.20`, not chrome); unit test asserts a true external caption still drops. |

## 12. Open Questions

None blocking. Resolved in §3 and §9 (microbeats kept switchable through validation, then removed; Ch1-first rollout).
