# Tech debt

Known defects that were found while doing something else and deliberately NOT fixed inline.
One entry each: what, where, how it shows, why it waited. Delete an entry when it is fixed.

## Teaser

- **First-video creation queues a doomed `plan_teaser` job.** `studio/dashboard/app.py`
  (the create-first-video route, ~:1362) enqueues `plan_teaser` with `bundle_id` and a NULL
  `series_id`; `_h_teaser` reads `job["series_id"]`, finds no chapters and dies with
  `NonRetryableError("teaser: no processed chapters yet")`. Not gated by `[teaser].enabled`.
  Found 2026-09-21 while mapping the teaser for the thumbnail work; out of that scope.
- **A re-plan can resurrect a stale teaser.** `_h_teaser` clears `teaser_state` and `teaser.mp4`
  but never clears `dist/series_<id>/teaser/`. If the re-plan selects no montage, the previous
  run's `manifest.teaser.json` still exists, the existence gate passes, and the worker re-renders
  from stale manifests. The thumbnail now READS that manifest, so this matters more than it did.
- **`videos.html` reads the legacy `bundle.teaser_state`** while the gate, the Series page and
  the rest read `series.teaser_state`; the two can disagree after the bundle->series migration.
- **ORV's teaser on the Mini lives in `dist/series_1/teaser_src/`**, a name no code reads
  (code: `dist/series_<id>/teaser/`). The dashboard review card cannot see it and the thumbnail
  job falls back to the computed montage (same ranking, measured 2026-09-21). Rename on the
  Mini after checking the `scenes/` symlinks are absolute; back up first.

## Thumbnail

- **`printable_card_line` passes meaningless fragments** ("SMILE CASH." on the tower series):
  every token is a real word and it is one sentence. The owner sees the card before picking,
  so it waited. A fix wants evidence, not a longer word filter.
- **`select_style` is dead on the production path** (`publish_concept.main` computes it and the
  two-stage branch never uses it); `thumbnail_styles.beat_signals` has no other caller.
- **`/thumbnail/label` switches the hook only.** A `nametag_headline` option's `headlines[]`
  candidates are written to `concept.json` but cannot be switched from the Series page.
