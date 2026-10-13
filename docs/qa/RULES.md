# QA rulebook — what every finished chapter must satisfy

Owner, 2026-10-13: *"first you must have a quality check list for all these rules and
conditions and issues, and whenever you update the code you must never break a fixed
issue."* This is that list. One row per rule a **finished** chapter (a rendered video)
must satisfy.

## How a rule is kept

Each row names:

- **Decided at** — the ONE place in the code that makes the decision. Every other
  consumer reaches it through there; nobody re-implements it.
- **QA code** — what `tools/prep_qa.py` emits when the rule is broken.
- **Blocks** — `yes` when the code is in `studio/worker.py::_CRITICAL_QA_CODES`: the
  chapter cannot ship with it.
- **Test** — the regression test(s) that fail if the rule breaks (`file::function`).
- **Sweep** — the `tools/qa_rules_sweep.py` check that measures the rule over every
  finished chapter on disk.

**Status:**
- `BLOCKED` — the chapter cannot ship broken (heal first where the code is healable).
- `HEALED` — the heal loop re-narrates the group; a survivor ships, listed for review.
- `DROPPED` — the visual heal drops the panel and holds a neighbour.
- `REPORTED` — measured only; promoting it to blocking failed the precision bar.
- `FEATURE` — not a defect; the sweep measures how far it reaches.
- `OPEN` — known and not enforced yet. Fixing it is a planned task, never a silent change.

`tests/test_qa_rules_doc.py` enforces this table:
- every named test exists;
- every QA code is really emitted;
- `Blocks` matches the worker's blocking set;
- `HEALED` and `DROPPED` codes are handled by the heal;
- every BLOCKED row has a sweep check;
- every blocking code in the worker has a row here.

Deleting a test, demoting a code or adding a blocking code without updating this file
fails the suite.

## The contract for every change

1. **Before.** Read the rows whose *Decided at* the change touches, then run
   `python tools/qa_rules_sweep.py`. Keep its report; that is the BEFORE count.
2. **Root cause, whole class.** Fix the place that decides, not the place the symptom
   showed. Fix every sibling of the defect, not one in ten. Grep every consumer of the
   function before editing.
3. **Measure before deploy.** For a rule that drops, renames or rewrites, run it over
   the corpus on the Mini. Look at every flipped panel or line by eye, and include
   random controls (owner rule, 2026-10-08).
4. **After.** Deploy, then run the sweep on the chapters made since. It must show zero
   new violations. Record both counts in `.continue-here.md`.
5. **A new defect** gets an `OPEN` row here first, then a planned fix. A fix only
   reaches chapters made after it unless the sweep repairs the old ones. Say so, and
   say which.

## Rules

| ID | Rule (a finished chapter …) | Decided at | QA code | Blocks | Test | Sweep | Status |
|---|---|---|---|---|---|---|---|
| R01 | never shows the licensed series title/logo | `scene_chrome.is_licensed_title_scene` (Rule 0 of `is_chrome_scene`); `story_group.licensed_title_files` | licensed_title_shown | yes | tests/test_prep_qa.py::test_vision_flags_licensed_title_shown_blocks_even_for_a_system_kind, tests/test_story_group.py::test_licensed_title_panel_is_excluded_even_when_understood_as_system | licensed_title_shown | BLOCKED |
| R02 | never shows publication furniture: an ad, a recruitment card, a credits roster, a To-Be-Continued card | `scene_chrome.furniture_reason` (Rule 0b, above the label); `panel_understand.apply_furniture_overrides` (last override); `story_group` exclusion | publication_shown | yes | tests/test_publication_stamps.py::test_a_panel_that_IS_the_message_is_furniture_whatever_its_label, tests/test_publication_stamps.py::test_a_stamp_on_a_story_panel_never_makes_it_furniture, tests/test_story_group.py::test_publication_furniture_never_reaches_the_writer_whatever_its_label, tests/test_panel_understand_chrome.py::test_publication_furniture_becomes_chrome_after_every_override, tests/test_prep_qa.py::test_publication_furniture_shown_blocks_whatever_its_label | publication_shown | BLOCKED |
| R03 | never speaks publication text: a site or watermark, a release plug, a recruitment notice, a credit, "the story will continue" | `scene_chrome.scrub_stamps` at the source (`vision_extract.clean_ocr_fields`); `gemini_narrative_pass.clean_card_text` | publication_voiced | yes | tests/test_publication_stamps.py::test_a_stamp_is_scrubbed_and_the_story_text_kept, tests/test_publication_stamps.py::test_in_world_text_is_never_a_stamp, tests/test_publication_stamps.py::test_a_voiced_line_with_publication_text_is_caught, tests/test_system_card_fidelity.py::test_a_publication_stamp_is_never_spoken_as_card_text, tests/test_ocr_garbage_scrub.py::test_publication_stamps_leave_ocr_clean_and_are_recorded_beside_it, tests/test_prep_qa.py::test_publication_text_voiced_is_a_healable_blocking_error | publication_voiced | BLOCKED |
| R04 | never names anyone after a watermark | `cast_builder._page_words` (a stamp or a chrome panel never backs a name) | watermark_name | yes | tests/test_cast_tokens.py::test_a_watermark_never_backs_a_name, tests/test_publication_stamps.py::test_a_name_printed_only_in_watermarks_is_found, tests/test_prep_qa.py::test_a_name_printed_only_in_a_watermark_blocks | watermark_name | BLOCKED |
| R05 | shows every narrated panel exactly once | planner span cover + `render_prep` | panel_uncovered, panel_double_covered | yes | tests/test_prep_qa_spans.py::test_cover_uncovered_shown_panel_errors, tests/test_prep_qa_spans.py::test_cover_double_covered_panel_errors | blocking_now | BLOCKED |
| R06 | has no gap between cuts | `timeline_planner` | cut_gap | yes | tests/test_prep_qa.py::test_cut_gap_is_error_not_warn | blocking_now | BLOCKED |
| R07 | has no sub-1.2 s flash cut, no missing or stale shown file | `render_prep` floors + dims | flash_cut, missing_file, missing_dims, stale_dims | yes | tests/test_prep_qa.py::test_plan_flags_missing_file_dims_audio_and_flash_cut, tests/test_prep_qa.py::test_image_flags_stale_dims_mismatch | blocking_now | BLOCKED |
| R08 | has audio for every line | `local_tts_from_manifest` + `tts_index.json` | audio_failed, audio_missing, audio_stale, audio_index_missing, missing_audio | yes | tests/test_prep_qa.py::test_audio_failed_flag_on_tts_failed_clip, tests/test_prep_qa.py::test_audio_flags_missing_clip_for_voiced_chapter, tests/test_prep_qa.py::test_audio_flags_stale_when_narration_changed | blocking_now | BLOCKED |
| R09 | is built from fresh manifests | `studio/deps.py` + `manifest_freshness` | stale_manifest, missing_manifest | yes | tests/test_prep_qa.py::test_prep_qa_emits_stale_manifest_flag_when_verify_chapter_returns_stale | blocking_now | BLOCKED |
| R10 | is built from manifests that parse | `manifest_freshness` | corrupt_manifest | no | tests/test_manifest_freshness.py::test_corrupt_manifest_flagged | — | OPEN |
| R11 | never shows a whole stitch chunk as one panel | detection | chunk_as_panel | yes | tests/test_prep_qa.py::test_chunk_as_panel_blocks_a_whole_chunk | blocking_now | BLOCKED |
| R12 | never shows the same panel twice as twins | `render_prep` shown-twin invariant | dup_shown | yes | tests/test_prep_qa.py::test_dup_shown_blocking_flag | blocking_now | BLOCKED |
| R13 | never holds a stand-in panel past the hold cap | `render_prep` holds | long_hold | yes | tests/test_prep_qa.py::test_long_hold_blocks_substituted_only | blocking_now | BLOCKED |
| R14 | shows every in-world system card | `story_group` keeps system cards; render protections | system_card_unshown | yes | tests/test_prep_qa.py::test_story_flags_dropped_system_card | blocking_now | BLOCKED |
| R15 | reads system cards aloud | `gemini_narrative_pass.system_card_line` | system_card_unvoiced | no | tests/test_prep_qa.py::test_system_card_unvoiced_flags_a_card_voiced_under_half | — | HEALED |
| R16 | has no null, cut-off or garbled line | writer validators + heal | narration_null, truncated_line, garbled_line | yes | tests/test_prep_qa.py::test_narration_null_flags_catch_a_stringified_null_line, tests/test_prep_qa.py::test_truncated_line_flags_mid_sentence_stop_as_healable_error, tests/test_prep_qa.py::test_garbled_line_flags_a_missing_noun_mid_sentence | blocking_now | BLOCKED |
| R17 | narrates detector-verified impacts as impacts | writer `[IMPACT SFX]` marker | impact_mismatch | yes | tests/test_narration_heal.py::test_impact_mismatch_is_healable_error | blocking_now | BLOCKED |
| R18 | has no empty item, no degenerate montage, complete beats | planner + writer | empty_item, montage_degenerate, beats_incomplete | yes | tests/test_prep_qa.py::test_plan_flags_empty_item_and_clean_plan_passes, tests/test_prep_qa.py::test_montage_flags_degenerate_loop, tests/test_prep_qa.py::test_alignment_flags_beats_incomplete | blocking_now | BLOCKED |
| R19 | has no filler line ("the scene continues") | writer prompt | filler_narration | no | tests/test_prep_qa.py::test_story_flags_filler_and_empty_narration | — | HEALED |
| R20 | has no unspeakable card line | `gemini_narrative_pass.system_card_line` | unspeakable_line | no | tests/test_prep_qa.py::test_unspeakable_line_flags_a_card_line_made_of_the_cards_marks | — | HEALED |
| R21 | voices every caption | writer | caption_unvoiced | no | tests/test_prep_qa.py::test_caption_unvoiced_flags_fire_and_clear | — | HEALED |
| R22 | never speaks pipeline vocabulary (figure lists, mood tags, file names) | writer validators | figures_leak, mood_tag_leak, filename_in_narration | no | tests/test_prep_qa.py::test_figures_leak_flags_catches_the_unknown_wrapper_echo, tests/test_prep_qa.py::test_mood_tag_leak_flags_catches_the_real_round3_shapes, tests/test_prep_qa.py::test_filename_in_narration_flags_leak_as_healable_error | — | HEALED |
| R23 | never shows a blank crop | `render_prep` + visual heal | blank_crop | no | tests/test_prep_qa.py::test_blank_crop_black_void_errors_even_for_sys | — | DROPPED |
| R24 | shows the 2–3 panels one line covers side by side where they fit | `render_prep.assign_rows` (`[render] row_min_fit`) | row_file_unknown | no | tests/test_prep_qa.py::test_plan_flags_row_files_must_be_cuts_with_dims, tests/test_qa_rules_sweep.py::test_rows_backfill_is_only_queued_when_asked | rows_missing | FEATURE |
| R25 | never has a dead character act | ledger | dead_actor | no | tests/test_story_ledger.py::test_dead_actor_is_report_only | — | REPORTED |
| R26 | names the right person | picture identity (`panel_identity_ccip` + gemma check) | actor_mismatch | no | tests/test_cast_identity.py::test_actor_mismatch_is_report_only | — | REPORTED |

## Notes per rule

- **R01:** a line that speaks the title without its panel has no line-level check.
  The exclusion of the logo panel is what stopped the 20 known cases (2026-10-05).
  Line-level check: OPEN.
- **R02:**
  - Measured 2026-10-13 over 91,346 panels. The rule flips 70 panels the model had
    labelled story/system/caption, and all 70 were checked by eye:
    - 42 Asura recruitment cards;
    - 9 "READ AT … FASTEST RELEASES" cards;
    - DuskScans' card and 6 "This chapter is provided by" cards;
    - 12 To-Be-Continued cards.
  - Known limit: a crop that merges the last story panel with the end card goes as a
    whole (Wimp ch36 p000116, TT ch36 p000090).
  - A stamp alone never drops a panel. A dark door panel in Clan ch11 nearly went
    under an image-flatness rule, which was removed.
- **R03:**
  - 4,576 aggregator stamps and 1,684 site addresses were scrubbed from story text
    across the corpus. In-world look-alikes stay:
    - TT's "Venerable Asura" and "Asura Nullifying Fist";
    - ORV's "Architect of Discord" and an in-story e-mail address;
    - "read at a faster speed".
  - No space after the dot in a site address: "…LEAVE. COM" is dialogue beside a
    watermark fragment.
- **R04:** Boundless Necromancer's "Asura" (2 chapters, 19 lines). TT's in-world
  "Asura" is printed in its dialogue and stays a name.
- **R10:** the code is an ERROR but does not block, so a chapter whose manifest does
  not parse can ship. A planned fix is pending a decision.
- **R24:**
  - Rows shipped 2026-10-04 and work in every chapter made since (183/183 on
    2026-10-13). The 848 videos made before are a backfill: the owner decides after
    the test batch.
  - The sweep's `rows_missing` re-runs `assign_rows` on the shipped plan to know a
    video would gain rows.
- **R25, R26:** promoted only when a graded sample clears
  `tools/qa_gate_blast_radius.py`: precision ≥ 0.80 over ≥ 20 flags across ≥ 2 titles.
