"""yolo_panels re-tile guard: a tall chunk YOLO under-segments (a whole chunk
rendered as one panel) is re-detected on vertical sub-tiles so the panels come
back. Pure-logic + a fake-model retile (no ultralytics needed)."""
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "yolo_panels",
    Path(__file__).resolve().parent.parent / "studio" / "detect" / "yolo_panels.py")
yp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(yp)  # type: ignore[union-attr]


def test_under_segmented_trigger():
    assert not yp._under_segmented([(0, 0, 800, 2000)], 5000)        # short chunk: skip
    assert yp._under_segmented([], 16000)                            # tall, nothing found
    assert yp._under_segmented([(0, 0, 800, 15000)], 16000)          # one box spans it
    good = [(0, i * 1500, 800, i * 1500 + 1400) for i in range(10)]
    assert not yp._under_segmented(good, 16000)                      # well-segmented: skip


def test_dedup_iou_drops_overlaps():
    a, b, c = (0, 0, 100, 100), (4, 4, 104, 104), (0, 500, 100, 600)
    assert len(yp._dedup_iou([a, b, c])) == 2                        # a~b merged, c kept


class _Arr:
    def __init__(self, a): self._a = a
    def cpu(self): return self
    def numpy(self): return self._a


class _Boxes:
    def __init__(self, xyxy, cls):
        self._x = np.array(xyxy, dtype=float); self._c = np.array(cls, dtype=float)
    def __len__(self): return len(self._x)
    @property
    def xyxy(self): return _Arr(self._x)
    @property
    def cls(self): return _Arr(self._c)


class _FakeModel:
    """Returns one panel near the top of every tile it is handed."""
    def predict(self, source=None, **kw):
        return [type("R", (), {"boxes": _Boxes([[10, 10, 790, 1000]], [0])})()]


def test_retile_offsets_and_merges_windows(tmp_path):
    p = tmp_path / "chunk.jpg"
    Image.fromarray(np.zeros((12000, 800, 3), dtype=np.uint8)).save(str(p))
    boxes = yp._retile_panels(_FakeModel(), str(p), 800, 12000, 0.25, "cpu",
                              960, 0, win=5000, overlap=500)
    assert len(boxes) >= 2                       # multiple windows -> multiple panels
    assert max(b[3] for b in boxes) > 5000       # a box was offset into the lower chunk


def test_resolve_classes_both_generations():
    legacy = {0: "panel", 1: "system_box", 2: "speech_bubble", 3: "text",
              4: "sfx", 5: "character"}
    pid, els = yp.resolve_classes(legacy)
    assert pid == 0
    assert els == {1: "system_box", 2: "speech_bubble", 4: "sfx"}

    v3 = {0: "panel", 1: "speech_bubble", 2: "radio", 3: "speech_background",
          4: "sfx_text", 5: "system_ui", 6: "caption_box", 7: "free_text"}
    pid, els = yp.resolve_classes(v3)
    assert pid == 0
    # v3 names land on the manifest keys consumers already speak
    assert els[5] == "system_box" and els[4] == "sfx"
    assert els[1] == "speech_bubble" and els[2] == "radio"
    assert els[3] == "speech_background"
    assert els[6] == "caption_box" and els[7] == "free_text"


def test_dedup_iou_drops_contained_fragment():
    big = (0, 0, 800, 1000)
    frag = (100, 100, 300, 300)      # fully inside big -> double-cover, drop
    apart = (0, 1200, 800, 2000)
    kept = yp._dedup_iou([frag, big, apart])
    assert big in kept and apart in kept and frag not in kept
