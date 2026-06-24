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
        return [type("R", (), {"boxes": _Boxes([[10, 10, 790, 1000]], [yp._PANEL_CLASS_ID])})()]


def test_retile_offsets_and_merges_windows(tmp_path):
    p = tmp_path / "chunk.jpg"
    Image.fromarray(np.zeros((12000, 800, 3), dtype=np.uint8)).save(str(p))
    boxes = yp._retile_panels(_FakeModel(), str(p), 800, 12000, 0.25, "cpu",
                              win=5000, overlap=500)
    assert len(boxes) >= 2                       # multiple windows -> multiple panels
    assert max(b[3] for b in boxes) > 5000       # a box was offset into the lower chunk
