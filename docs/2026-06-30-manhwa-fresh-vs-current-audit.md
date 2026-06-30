# Manhwa Fresh methodology vs. current pipeline — full audit

- **Date:** 2026-06-30
- **Trigger:** user asked for a full audit of the persona rules/modules vs. the current
  approach, and flagged suspected contradictions. Corrected mid-audit: **a/b/c/d are manhwa
  TYPES (niche modules), not the panel/narration/persona reconstruction.**
- **Source of the framework:** the user's "Manhwa Fresh" share (obs #14283, Jun 24; reference =
  Mamoru Nano Machine recap `gUCfdJdNYmU`), codified in
  `docs/plans/specs/2026-06-26-reference-channel-gemma-baseline-design.md`, partially shipped in `d315621`.
- **Method:** read the actual prompts/code on `main` @ `942724d`. Every verdict has a file:line.

---

## 1. The framework (what the user is benchmarking against)

**Manhwa Fresh = 6 rules + a 3-prompt stack.**

6 rules: (1) **10-word trap** opening, (2) **cut description**, (3) **point don't paint** *(the
user's "most important")*, (4) **name-spam → stand-ins**, (5) **texture asides**, (6) **cut to
one-third + fewer screenshots**.

3-prompt stack: `Universal Compression Engine` → **`Niche Module (A/B/C/D)`** → optional
`Chapter-1 Hook Layer`.

**a/b/c/d = the Niche Modules = manhwa types:** **A** Isekai/Power-Fantasy · **B** Romance/Drama
· **C** Dark-Action/Revenge · **D** Comedy/Slice-of-Life.

---

## 2. Rule-by-rule: Manhwa Fresh vs. what the code does

| MF rule | Current implementation | Verdict |
|---|---|---|
| **R1** 10-word story-explaining hook (ch1) | `OPENING_HOOK_RULE` + `apply_opening_hook` + `is_opening_chapter_path` **DELETED** (`e6d8951`, Jun 28). Replaced by a **bundle-level teaser** (sole intro). | **REMOVED** — no per-chapter hook; season teaser substitutes. |
| **R2** cut description (no screen-reading) | `recap_style.py:14` Rule 1 "NO SCREEN READING" + shot/effect bans; QA `no_describe`. | **ALIGNED** ✅ |
| **R3** point, don't paint *(most important)* | `recap_style.py:28` Rule 2 "POINT, DON'T PAINT" (one familiar comparison). | **ALIGNED** ✅ (but soft prompt rule; no enforcement it actually fires) |
| **R4** ration the name | `recap_style.py:32` Rule 3 "RATION NAMES" + `cast_builder` "our protagonist" stand-in. | **ALIGNED** ✅ |
| **R5** texture asides | `recap_style.py:36` Rule 4 "ADD TEXTURE NOT JOKES" (~1 in 4 lines); QA `sauce_density`. | **ALIGNED** ✅ |
| **R6** cut to 1/3 + **fewer screenshots** | `recap_style.py:40` Rule 5 "COMPRESS DRAG" literally says **"do not target one-third of the panel count… keep every panel."** The whole render path (`637357a` "restore every distinct panel", `timeline_planner` "never truncate", `system_card_unshown` QA) **enforces showing every panel.** | **DELIBERATELY INVERTED** ❌ — this is the single biggest divergence. |
| — | `recap_style.py:46` Rule 6 "REVEAL PACING" (identity-reveal gating) | **PROJECT ADDITION** (not in MF). |

**Net:** 4 of 6 MF rules are faithfully implemented as recap_style rules 1–4. **R1 was deleted.
R6 was inverted.** `recap_style` is fully wired (`gemini_narrative_pass.py:1115` into the prompt,
`narration_punchup.py:258`, `prep_qa.py:56`) — the earlier "not wired" claim was false.

---

## 3. The a/b/c/d Niche Modules — NOT implemented

Grep for `isekai|power fantasy|romance|revenge|slice of life|niche module` across `tools/` +
`studio/` returns **nothing** (obs #15743). What exists instead is a coarse **setting** axis in
`narration_punchup.py`:

- `GENRE_ADDONS` (`:82`) = `murim` / `modern` / `system` — three buckets, and `genre_key` (`:231`)
  only ever returns `murim` or `modern`; `infer_genre_from_content` (`:547`) classifies off
  chapter content.
- Its **only job** is gating anachronism/game-framing (murim forbids RPG slang, system permits it).

So the user's 4 **emotional/genre registers** (Isekai power-fantasy hype, Romance/Drama intimacy,
Dark-Action/Revenge edge, Comedy/Slice-of-Life levity) **do not exist**. There is one persona
voice (`BASE_PERSONA` + `CINEMATIC_RULES`), tuned only by setting and by the intensity gate below.

---

## 4. The 4 user-reported ch1 defects, root-caused

| # | Defect | Root cause (verified) | Where |
|---|---|---|---|
| **1** | over-tense voice on minor events | gemma grades **57% of panels "intense"** (enum `calm/tense/intense/explosive`); intensity drives mood/exag escalation. | `panel_understand.py:42,61` |
| **4** | flat / no persona / "no fun" | **Persona is gated by intensity.** `classify_beats` tags any beat with intensity ≥ "intense" as **DRAMATIC**, and `CINEMATIC_RULES` says **DRAMATIC stays PURELY cinematic — no persona**; only CONNECTIVE/COMIC get persona. With 57% intense → persona suppressed almost everywhere. **#1 and #4 share this root.** | `narration_punchup.py:113,156,174,203,216` |
| **3** | big-group flash (12 panels in 2.5s) | Direct consequence of **R6 inversion**: "keep every panel" + short narration line + **no planner min-per-panel floor** → 12–15 panels crammed under one line. `flash_cut` QA is WARN not ERROR. | `story_group` + `timeline_planner`; grounded `g0010_p14`=12/2.5s |
| **2** | zero dialogue/quotes | **NOT a prompt contradiction** — `_DIALOGUE_RULE` (`gemini_narrative_pass.py:155`, b7da6b9) *invites* ≤6-word quotes. Likely gemma non-compliance + the OCR-garble/fragment scrub (`sfx_scrub.py`), and the rule bans `'Ancestor...?'` trailing fragments **by name** — exactly the user's iconic-shout example. | needs ch1 `beats.json` inspection |

Persona **does** reach TTS: `gemini_verbatim` voices `panel_narration[].line` (the punched-up
field, `script_expander.py:912`), grounded original kept as `line_plain`. The intensity gate, not
the plumbing, is the limiter.

---

## 5. Opposing-facts (the user's "flag contradictions") — resolved

- **A. REAL** — intensity 57% ⟷ persona only on CONNECTIVE/COMIC beats → persona starved. Root of #1+#4.
- **B. RESOLVED** — the "never quote" rule was already replaced by an "invite short quotes" rule (b7da6b9).
- **C. FALSE** — `recap_style` is wired and tracked.
- **D. RESOLVED** — `gemini_verbatim` voices the punched-up line; persona ships.

---

## 6. The decision this surfaces (the real story)

The pipeline diverges from Manhwa Fresh on its two most defining moves: **R1 (per-chapter hook)
deleted** and **R6 (cut to 1/3 / fewer screenshots) inverted to "show every panel."**

Critically, **R6 was inverted on purpose and recently** — the whole `637357a` "restore every
distinct panel" effort, `system_card_unshown`, and the render-protect machinery exist to *guarantee
every panel is shown*. That is the **opposite** of Manhwa Fresh's "fewer screenshots." So the fork
isn't "are we wrong vs. the reference" — it's **"which intent wins: the user's recent show-every-panel
priority, or Manhwa Fresh's cut-hard priority?"** They cannot both be true.

### Fix options per defect (independent of the fork unless noted)

- **#1 + #4 (highest impact, lowest risk — recommend first):** decouple persona from intensity.
  Raise the DRAMATIC threshold to *explosive-only* (so "intense" beats can still take persona)
  **and/or** recalibrate the gemma intensity prompt so a routine fall isn't "intense." One-knob
  change in `narration_punchup.classify_beats` + a prompt tweak in `panel_understand`.
- **#3:** add a planner **min-per-panel on-screen floor** (e.g. ≥1.2s, stretch/hold) + split
  over-large groups, **OR** (if the fork goes MF-ward) actually *drop* low-value panels.
- **#2:** strengthen the quote invite, relax the fragment ban for iconic short shouts, and inspect
  whether the scrub is eating real quotes.
- **a/b/c/d niche modules:** build 4 genre persona registers (replacing the single voice), gated by
  `infer_genre`/manhwa type. Bigger build; quality lever, not a defect fix.

**The fork:** (A) move toward Manhwa Fresh — let the recap drop panels + compress hard; or
(B) keep "show every panel" (the user's recent priority) and fix pacing/persona within it.
