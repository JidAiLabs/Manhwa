# tests/test_timeline_floor.py
from tools.timeline_planner import _floor_shot_dur


def test_floor_extends_when_audio_too_short():
    assert _floor_shot_dur(12, 2.5, 1.2) == 12 * 1.2   # 12 panels can't fit in 2.5s -> extend
    assert _floor_shot_dur(2, 10.0, 1.2) == 10.0       # ample audio -> unchanged
    assert _floor_shot_dur(0, 5.0, 1.2) == 5.0         # no panels -> unchanged
    assert _floor_shot_dur(3, 5.0, 0.0) == 5.0         # floor disabled -> unchanged
