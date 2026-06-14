# Publishing metadata: title + thumbnail + description (copyright-safe, coherent)

Date: 2026-06-14
Status: approved (brainstorm), ready to plan/build
Owner: publishing

## Goal
For each publishable unit (a **single** chapter at the ongoing edge, or a **bundle** /
season-pack / ladder range), produce a **coherent** YouTube package — title,
thumbnail, description, pinned comment — that matches the proven recap-channel
formula, is copyright-safe, and is **$0 except the one Nano Banana image**.

Replaces the disconnected v1 (`youtube_meta.py` + `thumbnail_gen.py` with no link,
single hardcoded BEFORE/AFTER style, crude ref pick, no match check).

## Hard rules
- **No real series name / chapter number in title, thumbnail, or description body.**
- Real name + creator + official read-link go in the **pinned comment ONLY**
  (user decision 2026-06-14). Title/thumbnail are trope-only always.
- All text rendered on the thumbnail is a **deterministic overlay we control**
  (never model-drawn) → guaranteed legible + guaranteed copyright-safe.

## Model map (free local + one paid image)
| Step | Model | Cost |
|---|---|---|
| Concept (title + hook word + style + speech + which moments + synopsis) | `gemma4:26b` (ollama, text) | $0 |
| Ref-panel pick | `gemma4:26b` vision (ollama; sends candidate panels) | $0 |
| Thumbnail ART (no text) | **Nano Banana Pro** `gemini-3-pro-image` | ~$0.13 |
| Match / legibility / copyright CHECK on rendered ART | `gemma4:26b` vision | $0 |
| Text/arrow/`!?`/speech overlay | deterministic (PIL) | $0 |

## Pipeline
`unit (chapter or range)` → **concept** → **ref-pick** → **Nano Banana ART** →
**vision check** → **deterministic overlay** → 1280×720 jpg (+ 2K master).
Description + pinned comment generated from the SAME concept object.

### Concept object (the single source of coherence)
```
{ title,                # 60-95 char trope clickbait, no real name
  style,                # one of the 6 modules below
  hook_word,            # the big overlay label (= title's core: GENIUS/SSS/9999999/80KG)
  speech,               # 0-2 short colored callouts ("HOW?!","EASY..")
  marks,                # reaction marks to place (!,?,???)
  moments,              # which beats/panels to feature (hero + reaction crowd)
  synopsis,             # 2-4 sentence trope teaser (emojis), no real name
  hashtags }            # #manhwa #manga + genre/theme
```
Title text and the overlay label both read `hook_word` → they cannot drift.

## Thumbnail style library (registered from 18 competitor refs in assets/thumbnail_refs/)
Dominant formula across ~16/18: **powered hero (aura) + reacting crowd + one big
yellow label-with-arrow**. Modules (concept picks one by content signal):
1. **power_reveal** — centered glowing hero + shocked crowd (`!?`) + label-arrow. DEFAULT.
2. **stat_callout** — system/regression genre or UI panel → floating "+LVL/+STR", "9999999", "876 LVL".
3. **feat_object** — impossible feat → label on an object ("80KG","6YR-OLD").
4. **humiliation** — opponents kneeling/fallen + contrast speech ("EASY.." / "HOW?!").
5. **vs_monster** — hero facing giant monster/rival, energy between, label-arrow ("GOD").
6. **before_after** — clear weak→strong transformation split (kept; periodic winner).

Selection signal = beats `hook`/`what_happens`/`scene_selection.intensity`/`bubble_mode`/genre.
Each module = (a) a Nano Banana ART prompt template (composition, aura, crowd,
lighting, grading; NO text) + (b) an overlay layout (label+arrow position, speech,
marks). Electric-blue grade default.

## Description template (from competitor examples)
- **Per-video (generated, $0):** synopsis teaser (emojis, trope-safe) + hashtags.
- **Static boilerplate (write once, channel const):** Patreon / business email /
  fair-use disclaimer / big keyword tag-dump (genre-adjusted slots).
- **Bundles:** auto "Parts" timestamps from the concatenated chapter offsets.
- **Pinned comment (generated):** `Manhwa: <real title> — read official: <link>`
  (the ONLY place the real name appears).

## Single vs bundle (chapter scope)
- **Single** → 1 chapter's beats.
- **Bundle** → ALL chapters in the range, **hierarchical**: per-chapter 1-line arc
  beat (cheap local roll-up) → concept over the N one-liners (fits context) →
  title + the climax chapter/panel for the thumbnail. A *good arc title* needs the
  range to contain a complete setup→payoff arc; never needs the series finale;
  re-title at each ladder step.

## Quality loop
Cheap checks first (confirm-upstream): validate concept/title before the $0.13
image. After generation, the vision check verifies match + legibility + no leaked
licensed text + characters match refs; on fail, regenerate with the
**closed-loop safety guard** (cap retries, accept only a strictly-passing result —
see [[closed-loop-regen-needs-safety-guard]]).

## Build phases
1. **Style library + overlay engine** — 6 module configs, the deterministic
   PIL text/arrow/marks overlay, branded font; unit-tested on a stub ART image.
2. **Concept tool** (`publish_concept.py`) — Gemma; single + bundle (hierarchical);
   emits the concept object; copyright ban-list enforced.
3. **Ref-pick** (Gemma vision) + **Nano Banana per-module ART prompts**.
4. **Vision check + closed-loop regen guard.**
5. **Description + pinned-comment generator** from the concept + boilerplate.
6. **Wire into dashboard/worker** as a `publish_meta` job per unit; bundle "Parts".

## Non-goals
- No automated YouTube upload (manual). Discovery/coverage check already exists.
- No change to the render/voice pipeline.
