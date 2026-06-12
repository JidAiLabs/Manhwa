"""
M1 — YOLO blind-spot recovery in panels_to_scenes.

Black text-only narrative caption cards ("BACK THEN, I HAD NO IDEA.") are
not detected as panels, so they never became scenes — the Omniscient Reader
prologue lost its whole monologue spine. Uncovered vertical spans of a chunk
that still hold real content must be emitted as scenes, interleaved in
reading order; empty gutters must not.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "panels_to_scenes",
    Path(__file__).resolve().parent.parent / "tools" / "panels_to_scenes.py",
)
pts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pts)  # type: ignore[union-attr]


# ---- unit: uncovered_spans ---------------------------------------------------

def test_uncovered_spans_finds_middle_gap():
    covered = [[0, 0, 800, 400], [0, 800, 800, 1200]]
    gaps = pts.uncovered_spans(800, 1200, covered, min_h=90)
    assert gaps == [[0, 400, 800, 800]]


def test_uncovered_spans_ignores_small_gaps_and_merges_overlaps():
    covered = [[0, 0, 800, 500], [0, 490, 800, 960], [0, 1000, 800, 1200]]
    assert pts.uncovered_spans(800, 1200, covered, min_h=90) == []


def test_uncovered_spans_tail_gap():
    covered = [[0, 0, 800, 300]]
    assert pts.uncovered_spans(800, 1000, covered, min_h=90) == [
        [0, 300, 800, 1000]]


def test_uncovered_spans_no_boxes_means_whole_chunk():
    assert pts.uncovered_spans(800, 600, [], min_h=90) == [[0, 0, 800, 600]]


# ---- integration: caption card recovered, gutter not -------------------------

def _chunk_image(path: Path) -> None:
    """800x1500: art(0-500) / BLACK CAPTION CARD(500-900) / white gutter
    (900-1100) / art(1100-1500). YOLO 'found' only the two art blocks."""
    rng = np.random.default_rng(7)
    img = np.full((1500, 800, 3), 255, dtype=np.uint8)
    img[0:500] = rng.integers(40, 215, (500, 800, 3), dtype=np.uint8)
    img[500:900] = 8                                   # black card
    for i, y in enumerate(range(640, 760, 24)):        # white "text" strokes
        img[y:y + 10, 160 + (i % 3) * 40: 640 - (i % 2) * 60] = 245
    img[1100:1500] = rng.integers(40, 215, (400, 800, 3), dtype=np.uint8)
    Image.fromarray(img).save(path, "JPEG", quality=92)


def test_recovers_caption_card_in_reading_order(tmp_path):
    chunk = tmp_path / "chunk_0000.jpg"
    _chunk_image(chunk)
    stitch = {"chunks": [{"chunk_file": "chunk_0000.jpg",
                          "chunk_path": str(chunk)}]}
    panels = {"chunks": [{"chunk_file": "chunk_0000.jpg",
                          "panels_norm": [
                              [0.0, 0.0, 500 / 1500, 1.0],
                              [1100 / 1500, 0.0, 1.0, 1.0]]}]}
    sp = tmp_path / "stitch.json"
    pp = tmp_path / "panels.json"
    sp.write_text(json.dumps(stitch))
    pp.write_text(json.dumps(panels))
    out_dir = tmp_path / "scenes"
    out_manifest = tmp_path / "manifest.scenes.json"

    argv = ["panels_to_scenes.py",
            "--stitch-manifest", str(sp), "--panels-manifest", str(pp),
            "--out-dir", str(out_dir), "--out-manifest", str(out_manifest),
            "--panel-id-mode", "sequential"]
    old = sys.argv
    sys.argv = argv
    try:
        pts.main()
    finally:
        sys.argv = old

    m = json.loads(out_manifest.read_text())
    scenes = m["scenes"]
    assert len(scenes) == 3, [s["out_file"] for s in scenes]
    # reading order preserved: art, recovered card, art
    kinds = [bool(s.get("recovered")) for s in scenes]
    assert kinds == [False, True, False]
    ids = [s["out_file"] for s in scenes]
    assert ids == ["p000000.jpg", "p000001.jpg", "p000002.jpg"]
    card = scenes[1]
    y0, y1 = card["box_px_xyxy"][1], card["box_px_xyxy"][3]
    assert y0 >= 480 and y1 <= 1120     # the card span, not the gutter
    assert m.get("recovered_n") == 1
