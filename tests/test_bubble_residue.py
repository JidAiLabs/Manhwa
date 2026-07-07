"""Round-2 E2: bubble-clean residue net — stylized lettering Apple-OCR cannot
see stays readable in "cleaned" bubbles (Nano ch1 p000023 "JANG?", p000026
"END.", p000076, p000099 full dialogue; narration then double-exposes it).

The check is stroke density (Canny edge ratio, the impact_lettering gate
style) inside the bubble's interior component: dense strokes + empty OCR =
invisible text → clean_scene_image flattens the interior anyway (the existing
text-removal mechanism, without the tolerance bands the 240-254 ghost halos
slip through). Real fixture: tests/fixtures/sfx/p000099.jpg (downscaled from
the Mini's scenes/p000099.jpg); the genuinely-blank negative is the dedup
fixture p000095's bubble AFTER the existing flat-interior clean.

Boxes are the production bubble detector's real outputs on these fixtures,
frozen so the tests never load YOLO (hermetic + fast):
  p000099 → (17, 50, 218, 197), (74, 181, 265, 326)   conf .94 / .92
  p000095 → (155, 276, 389, 447)                       spiky shout bubble
Measured densities: p000099 raw interiors 0.060 / 0.079; cleaned (blank)
interiors 0.000 — BUBBLE_STROKE_DENSITY_MIN = 0.030 is a 2x margin.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import tools.render_prep as rp

_FIX = Path(__file__).resolve().parent / "fixtures"
P99 = _FIX / "sfx" / "p000099.jpg"
P95 = _FIX / "dedup" / "p000095.jpg"

BOXES_99 = [(17, 50, 218, 197), (74, 181, 265, 326)]
BOX_95 = (155, 276, 389, 447)


def _img(p: Path) -> np.ndarray:
    img = cv2.imread(str(p))
    assert img is not None, p
    return img


def test_density_fires_on_the_real_invisible_text_bubbles():
    img = _img(P99)
    for b in BOXES_99:
        d = rp.bubble_stroke_density(img, b)
        assert d >= rp.BUBBLE_STROKE_DENSITY_MIN, (b, d)


def test_density_silent_on_a_genuinely_empty_bubble():
    # p000095's flat white interior cleans fully via the existing contrast
    # path — the resulting blank bubble is the honest negative
    img = _img(P95)
    cleaned = rp.clean_scene_image(img.copy(), [BOX_95], text_boxes=[])
    d = rp.bubble_stroke_density(cleaned, BOX_95)
    assert d < rp.BUBBLE_STROKE_DENSITY_MIN, d


def test_density_failsoft_on_art_box_without_interior():
    # no white/black interior component -> 0.0, never a crash or a score
    img = _img(P99)
    assert rp.bubble_stroke_density(img, (300, 400, 460, 700)) == 0.0


def test_density_stays_under_floor_for_sparse_art_on_white():
    # A detector false-positive on artwork: a speech-shaped white region
    # DOES have a clean interior component (unlike the no-interior case
    # above), but only a couple of sparse, incidental strokes in it — not
    # dense stylized lettering. bubble_stroke_density is the SINGLE
    # authority the residue net's dense_invisible gate consults
    # (BUBBLE_STROKE_DENSITY_MIN), so this must score nonzero (it does have
    # strokes) yet stay well under the floor that real ghost text clears
    # (0.060/0.079 measured on p000099) — the floor's whole job is telling
    # sparse art apart from dense invisible text.
    img = np.full((300, 300, 3), 255, np.uint8)
    box = (50, 50, 200, 150)  # w=150 h=100, aspect 1.5 -> speech-shaped
    cv2.line(img, (70, 70), (120, 130), (20, 20, 20), 1)
    cv2.line(img, (150, 60), (170, 100), (20, 20, 20), 1)
    assert rp.speech_shaped_boxes([box], img.shape[1]) == [box]
    d = rp.bubble_stroke_density(img, box)
    assert 0.0 < d < rp.BUBBLE_STROKE_DENSITY_MIN, d


def test_residue_net_flattens_the_ghost_text():
    img = _img(P99)
    out = rp.clean_scene_image(img.copy(), BOXES_99, text_boxes=[],
                               residue_net=True)
    for b in BOXES_99:
        _t, fill, inside = rp._bubble_text(out, b)
        assert fill is not None and inside is not None
        g = out.mean(axis=2)
        nonflat = int(((np.abs(g - float(fill)) > 4) & (inside > 0)).sum())
        # interior is flat paper — no readable ghosts (was ~900 halo px)
        assert nonflat < 20, (b, nonflat)
        assert rp.bubble_stroke_density(out, b) < rp.BUBBLE_STROKE_DENSITY_MIN


def test_without_the_net_ghost_halos_survive():
    # regression proof the net is what does the work: the default clean
    # leaves the 240-254 anti-aliased halos readable
    img = _img(P99)
    out = rp.clean_scene_image(img.copy(), BOXES_99, text_boxes=[])
    worst = 0
    for b in BOXES_99:
        _t, fill, inside = rp._bubble_text(out, b)
        g = out.mean(axis=2)
        worst = max(worst, int(((np.abs(g - float(fill)) > 4)
                                & (inside > 0)).sum()))
    assert worst > 300, worst


def test_prep_qa_emits_bubble_text_residue_warn(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prep_qa", Path(__file__).resolve().parent.parent / "tools"
        / "prep_qa.py")
    pq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pq)  # type: ignore[union-attr]

    raw = _img(P99)   # simulates a production-cleaned file that kept its text
    flags = pq.image_flags("p000099.jpg", raw, BOXES_99, doc=False,
                           dims_entry=None, sys=False, segment_id="g0001",
                           vitem={"ocr_clean": ""})
    codes = [f["code"] for f in flags]
    assert "bubble_text_residue" in codes
    resid = [f for f in flags if f["code"] == "bubble_text_residue"][0]
    assert resid["severity"] == pq.WARN

    netted = rp.clean_scene_image(raw.copy(), BOXES_99, text_boxes=[],
                                  residue_net=True)
    flags2 = pq.image_flags("p000099.jpg", netted, BOXES_99, doc=False,
                            dims_entry=None, sys=False, segment_id="g0001",
                            vitem={"ocr_clean": ""})
    assert "bubble_text_residue" not in [f["code"] for f in flags2]
