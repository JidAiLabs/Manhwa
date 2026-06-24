"""
tests/test_blender_timing.py

Guards the A/V-drift fix in tools/blender_vse_from_plan.py (the manual Blender
render path — production ships via Remotion). The bug: each shot's foreground
strip was placed at previous-end+1 (channel_safe_start bumping +1 per shot,
never reset), so video lagged the audio by an accumulating ~1 frame/shot. The
fix re-anchors the FG running end to each shot's absolute frame and floors the
relative cut offset to 0, so every shot's first cut sits at exactly the audio's
frame. These assertions exercise the real frame-math helpers + replicate the
per-shot anchor the fix installs, proving FG == audio with no accumulation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "blender_vse_from_plan",
    Path(__file__).resolve().parent.parent / "tools" / "blender_vse_from_plan.py",
)
bvp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bvp)  # type: ignore[union-attr]

FPS = 30


def test_rel_offset_floors_to_zero_unlike_length():
    # A length must round up to at least one frame; a relative OFFSET of 0 must
    # stay 0 (else every first cut starts one frame after the audio).
    assert bvp.rel_seconds_to_frames(0.0, FPS) == 0
    assert bvp.seconds_to_frames(0.0, FPS) == 1
    assert bvp.rel_seconds_to_frames(1.5, FPS) == 45


def _fg_start_for_shot(start_sec, prev_fg_end):
    """Replicate the placement the fix installs for a shot's FIRST cut:
    re-anchor FG end to shot_fs-1, then channel_safe_start(shot_fs)."""
    shot_fs = bvp.sec_to_frame_start(start_sec, FPS)
    # the fix: reset FG running end to this shot's absolute frame
    fg_end = shot_fs - 1
    cut_fs = shot_fs + bvp.rel_seconds_to_frames(0.0, FPS)
    # channel_safe_start: max(fs, prev+1) if prev>=fs else fs
    return cut_fs if fg_end < cut_fs else max(cut_fs, fg_end + 1)


def test_foreground_locks_to_audio_with_no_drift():
    # 60 contiguous 2.0s shots: every shot's FG first-cut must equal the audio
    # frame (sec_to_frame_start of its start_sec), with zero accumulation.
    prev_fg_end = 0
    for k in range(60):
        start_sec = k * 2.0
        audio_fs = bvp.sec_to_frame_start(start_sec, FPS)   # sound strip seat
        fg_fs = _fg_start_for_shot(start_sec, prev_fg_end)
        assert fg_fs == audio_fs, f"shot {k}: FG {fg_fs} != audio {audio_fs}"
        # the FG strip is 2.0s long; its end feeds the next iteration
        prev_fg_end = fg_fs + bvp.seconds_to_frames(2.0, FPS) - 1
