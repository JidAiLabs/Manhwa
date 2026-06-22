# Per-Group TTS + Forced-Alignment — Design Spec

- **Date:** 2026-06-22
- **Status:** Approved (user authorized "run the optimization"); building
- **Goal:** Voice each group as ONE continuous Qwen clip (instead of one clip per panel), then forced-align to keep each panel's visual cut synced to the right moment in that clip. ~15% faster synthesis (measured), tighter pacing (no inter-clip padding), fewer onset "ah/uh" (continuous prosody).
- **Constraint:** single-machine code, runs identically on both boxes (no multi-machine assumptions). Uses faster-whisper already in `.qwen_venv`.

## Why
Today the `voiced` stage synthesizes one clip per panel-segment (~81 for Nano) purely so the timeline can read each panel's audio duration. Per-panel audio is an accident of the per-panel *narration* design — the **visuals** need per-panel timing, the **audio** does not. Measured: per-group synth is 15% faster (136s→115s on a 7-panel group) and the audio is ~10% shorter (no inter-clip lead/trail padding).

## Decisions (locked — technical choices, not user-facing)
1. **Synthesis granularity = per GROUP** (the story-group beats). Avg group ~24s audio, well under Qwen's ~2.5min/2048-token per-call cap. One `generate_voice_clone` call per group on the joined panel-lines.
2. **Alignment = faster-whisper word-timestamps.** Transcribe the group clip with `word_timestamps=True`, map the known panel-line word sequence onto the recognized words, take each panel line's last-word end-time as its boundary → per-panel `[start,end]` within the group clip.
3. **Fallback = proportional split.** If alignment confidence is low (poor word-match between intended and transcribed), split the group clip by each panel line's word-count fraction. Always produces boundaries; never desyncs the whole group. Flag which groups used the fallback.
4. **Escape hatch = the current per-panel path stays**, behind a config flag `tts_group_synth` (default **on**). Flip off → exact current behavior. Lets us A/B on Ch1 and revert instantly.

## Components & data flow
```
voiced:   local_tts_from_manifest.py
            per-group mode: one clip per group -> clips/g####.wav (joined panel-lines, Qwen clone + the gen params)
            writes tts_index.json with group clips
align:    NEW step (in the voiced stage or a thin tool) -> manifest.align.json
            per panel: { segment_id g####_p##, group_clip g####.wav, start_sec, end_sec, method: "asr"|"proportional" }
            faster-whisper word-timestamps; proportional fallback per group on low match
planned:  timeline_planner.py
            a panel cut = (group_clip, audio_start=start_sec, dur=end_sec-start_sec) from manifest.align.json
            audio track = the continuous group clips; visual cuts switch panels at the offsets
render:   render_prep / render — play group clips continuously; cut panels at the aligned offsets
```

## Contract
- `segment_id` stays `g####_p##` per panel (TTS/timeline/QA unchanged in shape).
- Clips become **per-group** (`g####.wav`) + a per-panel **offset** in `manifest.align.json`. The timeline reads offsets instead of per-clip durations.
- `tts_group_synth=False` → the old per-panel clips + per-clip-duration timeline (unchanged). Both paths coexist.

## Components (files)
| File | Change |
|------|--------|
| `tools/local_tts_from_manifest.py` | per-group synth mode (join a group's lines → 1 clip); keep per-panel mode behind the flag |
| `tools/tts_align.py` (NEW) | pure-ish: given group clip + ordered panel lines → per-panel `[start,end]` via faster-whisper word-timestamps; proportional fallback; injectable transcribe_fn for tests |
| `tools/timeline_planner.py` | when `manifest.align.json` exists, build cuts from (group_clip, offset, dur) instead of per-panel clip durations |
| `studio/pipeline.py` / `studio/config.py` | `tts_group_synth` flag (default on); wire the align step into voiced |
| `studio/worker.py` | voiceover path: run group synth + align before plan |

## Error handling
- Alignment word-match < threshold → proportional split for that group, `method:"proportional"`, logged + counted.
- A group clip that fails ASR entirely → proportional split (never blocks).
- `tts_group_synth=False` → full revert to per-panel (the shipped, known-good path).
- Invariant: every panel (`g####_p##`) gets exactly one `[start,end]` (asr or proportional) — no panel un-timed.

## Testing
- `tts_align`: unit tests with a stubbed transcribe_fn — clean alignment maps N lines→N boundaries in order; low-match → proportional fallback; boundaries monotonic + cover the clip; every panel timed.
- per-group synth: one clip per group, joined text, flag off → per-panel unchanged.
- timeline: a cut reads (group_clip, offset, dur) from align manifest; offsets monotonic per group.
- Ch1 real run (A/B vs per-panel): every panel cut lands on its line (spot-check a few), 15%-ish faster, tighter pacing, fewer onsets. Human judges quality.

## Rollout
TDD behind `tts_group_synth` (default on). Validate on Ch1 (A/B vs the shipped per-panel output) before any batch. Per-panel remains the instant fallback.
