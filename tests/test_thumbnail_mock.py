"""tools/thumbnail_mock.py -- free mocks on the story's real panels (2026-09-29)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))
_SPEC = importlib.util.spec_from_file_location("thumbnail_mock", _TOOLS / "thumbnail_mock.py")
tm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tm)  # type: ignore[union-attr]


def _img(path, size, rgb):
    path = Path(path).with_suffix(".png")           # lossless: exact colours
    Image.new("RGB", size, rgb).save(path)
    return str(path)


def _near(a, b, tol=4):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_a_portrait_panel_gets_no_black_bars_and_no_stretch(tmp_path, monkeypatch):
    """ORV Ep6: 61 of 105 panels are portrait. The old overlay stretched them
    to 16:9 or padded black."""
    monkeypatch.setattr(tm, "_faces", lambda p: [])
    tall = _img(tmp_path / "t.jpg", (600, 1400), (200, 30, 30))
    im = tm.frame_panel(tall)
    assert im.size == (tm.W, tm.H)
    edge = im.getpixel((10, 360))
    assert edge != (0, 0, 0) and edge[0] < 150        # blurred, darkened, not black
    assert _near(im.getpixel((int(tm.W * 0.66), 360)), (200, 30, 30))  # the panel
    # full height, true aspect: 600x1400 -> ~309 wide at 720 high
    cols = [x for x in range(tm.W) if _near(im.getpixel((x, 360)), (200, 30, 30))]
    assert 290 <= len(cols) <= 320


def test_a_wide_panel_is_cover_cropped(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "_faces", lambda p: [])
    wide = _img(tmp_path / "w.jpg", (1600, 800), (30, 200, 30))
    im = tm.frame_panel(wide)
    assert im.size == (tm.W, tm.H) and _near(im.getpixel((5, 5)), (30, 200, 30))


def test_a_split_puts_low_left_and_high_right(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "_faces", lambda p: [])
    a = _img(tmp_path / "a.jpg", (700, 900), (10, 10, 200))
    b = _img(tmp_path / "b.jpg", (700, 900), (200, 200, 10))
    r = tm.render_mock({"layout": "split", "labels": ["F-RANK", "S-RANK"],
                        "panels": [a, b]}, str(tmp_path / "s.jpg"))
    im = Image.open(r["image"])
    left, right = im.getpixel((100, 100)), im.getpixel((tm.W - 100, 100))
    assert left[2] > 150 and right[0] > 150            # low left, high right
    assert r["arrow_box"] is None                      # a split never arrows


def test_an_arrow_is_drawn_only_toward_a_detected_subject(tmp_path, monkeypatch):
    """Owner 2026-09-27: a fixed arrow pointed at a chin or at air. The arrow
    goes to a box FOUND on the composed image, or is not drawn."""
    wide = _img(tmp_path / "w.jpg", (1600, 900), (90, 90, 90))
    spec = {"layout": "hero", "labels": ["NECROMANCER"], "panels": [wide],
            "arrow": "hero"}
    monkeypatch.setattr(tm, "_faces", lambda p: [])
    r = tm.render_mock(spec, str(tmp_path / "none.jpg"))
    assert r["arrow_box"] is None and "no face" in r["arrow_note"]
    monkeypatch.setattr(tm, "_faces", lambda p: [[0.60, 0.20, 0.80, 0.55]])
    r = tm.render_mock(spec, str(tmp_path / "yes.jpg"))
    assert r["arrow_box"] == [0.60, 0.20, 0.80, 0.55]
    # an "object" label has no detector yet: no arrow rather than a guess
    r = tm.render_mock(dict(spec, arrow="object"), str(tmp_path / "obj.jpg"))
    assert r["arrow_box"] is None


def test_claim_mocks_are_rendered_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "_faces", lambda p: [])
    p = _img(tmp_path / "p.jpg", (800, 1000), (50, 50, 50))
    claim = {"mocks": [{"name": "hero", "layout": "hero", "labels": ["OP"],
                        "panels": [p], "arrow": None},
                       {"name": "clean", "layout": "hero", "labels": [],
                        "panels": [str(tmp_path / "missing.jpg")], "arrow": None}]}
    done = tm.render_claim_mocks(claim, str(tmp_path / "mocks"))
    assert Path(done[0]["image"]).exists() and done[0]["image"].endswith("hero.jpg")
    assert done[1]["image"] == ""                      # nothing on disk: said so
