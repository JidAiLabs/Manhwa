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


# ---- a recovered span taller than a panel is a COLUMN, split on its gutter ----

def _tall_column(path: Path) -> None:
    """800x9000 with NO YOLO boxes: art(0-4200) / white gutter / art(4700-9000).
    ORV Ep311 chunk_0020 (0 boxes in 9460px) and Death Knight ch33 (11218px):
    recovery kept the whole chunk as ONE scene and prep_qa blocked it as
    chunk_as_panel with nothing upstream able to clear it."""
    rng = np.random.default_rng(11)
    img = np.full((9000, 800, 3), 255, dtype=np.uint8)
    img[0:4200] = rng.integers(40, 215, (4200, 800, 3), dtype=np.uint8)
    img[4700:9000] = rng.integers(40, 215, (4300, 800, 3), dtype=np.uint8)
    Image.fromarray(img).save(path, "JPEG", quality=92)


def test_a_recovered_span_taller_than_a_panel_is_split_on_its_gutter(tmp_path):
    chunk = tmp_path / "chunk_0000.jpg"
    _tall_column(chunk)
    stitch = {"chunks": [{"chunk_file": "chunk_0000.jpg",
                          "chunk_path": str(chunk)}]}
    panels = {"chunks": [{"chunk_file": "chunk_0000.jpg", "panels_norm": []}]}
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
    scenes = json.loads(out_manifest.read_text())["scenes"]
    assert [bool(s.get("recovered")) for s in scenes] == [True, True], scenes
    assert all(s["h"] <= pts.CHUNK_AS_PANEL_MIN_H for s in scenes)
    # the cut lies inside the gutter band (the splitter cuts at its centre)
    assert 4200 <= scenes[0]["box_px_xyxy"][3] <= 4700
    assert 4200 <= scenes[1]["box_px_xyxy"][1] <= 4700


# ---- flat blank bands at a crop's edges are cut (ORV Ep311 p000008) -------

def _seam_tail(w=600, h=520):
    """Art on the top quarter touching the edge, a black stroke with a white
    rim hanging into white, then plain white gutter."""
    rng = np.random.default_rng(3)
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[0:130] = rng.integers(60, 180, (130, w, 3), dtype=np.uint8)
    img[130:190, 280:300] = 0                       # the sound-effect stroke
    img[130:190, 276:280] = 255
    return Image.fromarray(img)


def test_flat_edge_bands_are_cut_and_the_art_is_kept():
    out, info = pts.trim_blank_bands(_seam_tail())
    assert info["trimmed"] and info["mode"] == "blank_bands"
    assert out.size[1] < 220 and out.size[0] == 600      # art + stroke, no gutter
    assert info["top_px"] == 0 and info["bottom_px"] > 300


def test_line_art_on_white_is_never_cut():
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    img[:, ::9] = 0                                      # ink across every row
    img[::9, :] = 0
    out, info = pts.trim_blank_bands(Image.fromarray(img))
    assert not info["trimmed"] and out.size == (400, 400)


def test_a_sliver_is_left_alone():
    img = np.full((500, 500, 3), 255, dtype=np.uint8)
    img[0:40] = 120                                      # 40px of content only
    out, info = pts.trim_blank_bands(Image.fromarray(img))
    assert not info["trimmed"] and info["reason"] == "min_keep_guard"


def test_a_frame_under_half_blank_keeps_its_margins():
    img = np.full((500, 500, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(5)
    img[0:350] = rng.integers(60, 180, (350, 500, 3), dtype=np.uint8)  # 30% white
    out, info = pts.trim_blank_bands(Image.fromarray(img))
    assert not info["trimmed"] and info["reason"] == "below_threshold"
