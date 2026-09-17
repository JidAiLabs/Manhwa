# Image-based character identity

*2026-09-18 · owner-approved design · replaces word-matching identity where exemplars exist*

## The problem

`tools/cast_identity.py` decides who is in a panel by matching gemma's TEXT description of the
panel ("a young man with short dark hair") against a TEXT description of each character. It never
sees the image. In a cast where several characters are young dark-haired men it cannot tell them
apart, which produced:

- ORV Episode 6: Dokja narrated as "Namwoon Kim" 67 times, Namwoon as "our boy" (the chapter's cast
  was guessed pre-registry with the two looks swapped).
- 5 of 8 "clear shots of the MC" thumbnail suggestions showing other men.

Owner, 2026-09-17: *"characters are being confused"*, *"use the man, the white hair guy... just
dont mix it and dont drop a panel if we dont know"*, *"you use the names too much, you can also use
he, she"*.

## Measured (spike, 2026-09-17; full numbers in the memory note)

50 Episode 6 panels + 8 thumbnail tiles, labelled by eye. gemma4:26b, whole panels, no crops
(YOLO cannot crop manhwa characters: the legacy `character` class fired 2/60 at conf 0.01, COCO
person 5/60).

| candidates per call | tiles (3 real) | Ep6 Dokja precision / recall | Ep6 wrong names | s/panel |
|---|---|---|---|---|
| word matching (today) | 3 of 8 | 0.69 / 0.68 | 22 | 0 |
| 1 (A or OTHER) | 2 found, 3 false | 0.70 / 0.51 | 8 | 2.7 |
| **2 (A / B / OTHER)** | **3 found, 0 false** | **0.96 / 0.65** | **6** | 4.2 |
| 4 (A/B/C/D/OTHER) | 0 found | 0.75 / 0.24 | 3 (+12 claims about absent characters) | 6.7 |

**Two candidates, with an explicit OTHER, is the configuration.** One candidate forces look-alikes
onto A; four collapse recall and hallucinate the extras. The prompt must say to answer a letter only
when the FACE matches, since outfit/hair alone is what the old approach already got wrong.

## Design

1. **Exemplars live in the owner's series registry** `cast/<slug>.json`: a member may carry
   `exemplars: [<repo-relative panel paths>]` (2 recommended). ORV, owner-confirmed:
   Dokja = Episode_6 p000010, p000037; Namwoon Kim = Episode_6 p000011, p000097.
2. **`tools/panel_identity.py` (new)** — one gemma call per panel with ≤2 candidates:
   - `candidates(registry)` → the ≤2 members that carry exemplars (protagonist first).
   - `build_prompt(names)` → the measured forced-choice text.
   - `identify_panel(path, candidates, chat)` → `{"names": [...], "others": N}`; unparseable or
     failed call = no names (never a guess).
   - `identify_panels(...)` → `manifest.identity.json`:
     `{"_meta": ..., "panels": {"p000010.jpg": {"names": [...], "others": 1}}}`.
3. **Stage + freshness:** produced at `beated`, before the writer. `studio/deps.py` gains
   `manifest.identity.json` (inputs: `manifest.panels.understood.json`, optional
   `manifest.cast.json`); the registry file is its mtime edge, exactly like `manifest.cast.json`.
   A chapter whose series has no exemplars skips it and keeps today's behaviour.
4. **One authority stays one authority:** `cast_identity.resolve_figures_by_file(understood, cast,
   identity=None)`. When `identity` is present it WINS: a panel's figures are the confirmed names
   plus one `unknown` entry per unconfirmed person. No keyword fallback for that panel — a fallback
   is how a wrong name gets in. Absent identity manifest = today's keyword path (older chapters).
5. **The writer never receives an unconfirmed name.** `gemini_narrative_pass._pack_group_payload`
   passes confirmed names, and for everyone else a gender-safe descriptor built from the panel's own
   subject text ("the white-haired guy", "the figure in the green jacket"). Gender comes from the
   cast entry or the page's pronouns; unknown gender = attribute-only phrasing, never he/she.
6. **Fewer names, more pronouns** (`tools/recap_style.py` rule 6): name a character when they first
   appear in a beat or when the subject changes; after that a pronoun or handle. Today's rule says
   to name established characters on their own panels, which is why one chapter carried 67 "Namwoon
   Kim".
7. **A panel is never dropped for unknown identity.** Unknown = neutral wording only.
8. **Census (after the build):** run the same check over a series and report, per chapter, lines
   that name a character the panel does not contain. ORV ≈ 18.5k panels with people, 4.2 s each:
   ~21 h serial, ~5-6 h at ollama parallel 4. Output is the re-narration list.

## Cost

+4.2 s per panel with people ≈ +4 min on a ~50 min chapter prepare. Local gemma, free.

## Risks

- Recall ~0.65: a third of the MC's panels get a neutral handle instead of his name. Accepted
  (owner: don't mix, don't drop).
- Exemplars are owner-curated; a series without them keeps today's behaviour.
- Measured on one chapter + 8 tiles. The census is the scale check.
