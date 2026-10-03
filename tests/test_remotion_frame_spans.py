"""Frame tiling contract of the Remotion renderer (remotion/src/Shot.tsx + Root.tsx).

A Python mirror of the renderer's frame math (precedent: _effective_zoom_cap in
test_render_prep.py mirrors plan.ts). The ch141 case is the production black
frame: item g0007_p00 (start 115.583, dur 14.476) is ceil'd to 435 frames, but
its two cuts, each placed at round(start*30) with length ceil(dur*30), covered
only relative frames 0..433 — frame 434 (absolute 3901, 2:10.03) showed the
#000 background. The fix derives every cut's length from the NEXT cut's
snapped start (the last cut runs to the item's end) and ends the composition
where the last item ends, so tiling is contiguous by construction.

JS Math.round is half-up; Python round() is banker's — mirror with floor(x+.5).
"""
import math
import os
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30


def js_round(x: float) -> int:
    return math.floor(x + 0.5)


def to_start_frame(sec: float) -> int:          # plan.ts toStartFrame
    return js_round(sec * FPS)


def to_frames(sec: float) -> int:               # plan.ts toFrames
    return max(1, math.ceil(sec * FPS))


def legacy_cut_spans(item):
    """Pre-fix Shot.tsx: (from, len) per cut, each rounded on its own."""
    return [(to_start_frame(c["start"]), to_frames(c["dur"])) for c in item["cuts"]]


def cut_spans(item):
    """Fixed Shot.tsx: a cut ends where the next cut starts; the last cut ends
    at the item's end; a sub-frame cut yields no span (its neighbour covers it)."""
    cuts = item["cuts"]
    item_len = to_frames(item["duration_sec"])
    out = []
    for i, c in enumerate(cuts):
        f = to_start_frame(c["start"])
        end = to_start_frame(cuts[i + 1]["start"]) if i + 1 < len(cuts) else item_len
        if end - f > 0:
            out.append((f, end - f))
    return out


def composition_frames(timeline):
    """Fixed Root.tsx: the composition ends where the last item's Sequence ends."""
    last = timeline[-1]
    return to_start_frame(last["start_sec"]) + to_frames(last["duration_sec"])


def assert_tiles(spans, total):
    pos = 0
    for f, n in spans:
        assert f == pos, f"hole/overlap at relative frame {pos}: next span starts at {f}"
        assert n >= 1
        pos += n
    assert pos == total, f"spans end at {pos}, item is {total} frames"


CH141_ITEM = {
    "start_sec": 115.583, "duration_sec": 14.476, "end_sec": 130.059,
    "cuts": [{"file": "p000026.jpg", "start": 0.0, "dur": 6.5141},
             {"file": "p000026.jpg", "start": 6.5141, "dur": 7.9616}],
}


def test_ch141_legacy_math_leaves_the_last_item_frame_uncovered():
    item_len = to_frames(CH141_ITEM["duration_sec"])
    assert item_len == 435
    spans = legacy_cut_spans(CH141_ITEM)
    assert spans == [(0, 196), (195, 239)]            # overlap at 195, and …
    assert max(f + n for f, n in spans) == 434 < item_len   # … frame 434 is black


def test_ch141_item_frames_fully_covered():
    spans = cut_spans(CH141_ITEM)
    assert spans == [(0, 195), (195, 240)]       # the last cut absorbs the slack
    assert_tiles(spans, 435)


def test_sub_frame_cut_is_absorbed_by_neighbour():
    item = {"start_sec": 0.0, "duration_sec": 6.0, "end_sec": 6.0,
            "cuts": [{"file": "a", "start": 0.0, "dur": 3.0},
                     {"file": "b", "start": 3.0, "dur": 0.01},
                     {"file": "c", "start": 3.01, "dur": 2.99}]}
    spans = cut_spans(item)
    assert spans == [(0, 90), (90, 90)]
    assert_tiles(spans, 180)


def _planner_like_timeline(rng: random.Random, *, dp: int, jitter: bool):
    """Mimic timeline_planner.build_cuts (:1078-1085: per-cut starts/durs
    rounded independently, last cut absorbs) and the item cursor (:2045-2087)."""
    timeline, cursor = [], 0.0
    for _ in range(rng.randint(3, 40)):
        dur = rng.uniform(1.0, 15.0)
        k = rng.randint(1, 4)
        per, t, cuts = dur / k, 0.0, []
        for i in range(k):
            d = per if i < k - 1 else (dur - t)
            cuts.append({"file": f"p{i}", "start": round(t, dp), "dur": round(d, dp)})
            t += per
        if jitter:
            for c in cuts[1:]:
                c["start"] = round(c["start"] + rng.choice((-0.001, 0.001)), dp)
        item_dur = round(sum(c["dur"] for c in cuts), 3)
        timeline.append({"start_sec": round(cursor, 3), "duration_sec": item_dur,
                         "end_sec": round(cursor + item_dur, 3), "cuts": cuts})
        cursor += item_dur
    return timeline


def test_randomized_planner_tiling_is_contiguous():
    rng = random.Random(1234)
    for trial in range(300):
        dp = 4 if trial % 3 == 0 else 3
        tl = _planner_like_timeline(rng, dp=dp, jitter=(trial % 3 == 0))
        for item in tl:
            assert_tiles(cut_spans(item), to_frames(item["duration_sec"]))
        total = composition_frames(tl)
        assert all(to_start_frame(i["start_sec"]) + to_frames(i["duration_sec"]) <= total for i in tl)


def test_renderer_sources_use_boundary_derived_frame_spans():
    """Pins the TS to the mirror above: cut lengths come from the next cut's
    snapped start (not each cut's own ceil'd dur) and the composition ends at
    the last item's end (not ceil(total_duration_sec))."""
    shot = open(os.path.join(REPO, "remotion", "src", "Shot.tsx")).read()
    root = open(os.path.join(REPO, "remotion", "src", "Root.tsx")).read()
    assert "durationInFrames={toFrames(c.dur)}" not in shot
    assert "toStartFrame(cuts[i + 1].start)" in shot
    assert "Math.ceil(totalSec * FPS)" not in root
    assert "toStartFrame(last.start_sec) + toFrames(last.duration_sec)" in root
