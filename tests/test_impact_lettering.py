"""Qualification tests for tools/impact_lettering.py — the deterministic
impact-SFX detector (the manhwa violence signal the narration writer must see).

The bar (verified production panels, downscaled fixtures under
tests/fixtures/sfx/, max width 480 q70):
  * MUST fire on p000036 — the big red painted stab SFX + blood droplet;
  * MUST stay silent on p000000 — a calm desaturated night landscape;
  * MUST stay silent on tests/fixtures/dedup/p000055.jpg — a white speech
    bubble over a dark red-brown cloak (the false-positive trap: saturated
    reddish fabric next to dialogue, no SFX lettering).
p000034 (whitish/outline slash SFX) is deliberately NOT gated — v1 catches
saturated painted lettering only; see the ceiling comment in the module.
"""
import os
import sys

import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from impact_lettering import detect_impact_lettering  # noqa: E402

_FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(rel: str):
    path = os.path.join(_FIX, rel)
    img = cv2.imread(path)
    assert img is not None, f"fixture unreadable: {path}"
    return img


def test_fires_on_red_stab_sfx_panel():
    regions = detect_impact_lettering(_load("sfx/p000036.jpg"))
    assert regions, "must detect the big red stab SFX lettering on p000036"


def test_silent_on_calm_landscape():
    assert detect_impact_lettering(_load("sfx/p000000.jpg")) == []


def test_silent_on_dialogue_bubble_panel():
    # The trap: a white dialogue bubble + a saturated red-brown cloak.
    assert detect_impact_lettering(_load("dedup/p000055.jpg")) == []


def test_region_shape_contract():
    regions = detect_impact_lettering(_load("sfx/p000036.jpg"))
    h, w = _load("sfx/p000036.jpg").shape[:2]
    for r in regions:
        assert set(r) == {"bbox", "area_frac", "mean_hue_deg"}
        x, y, bw, bh = r["bbox"]
        assert 0 <= x < x + bw <= w and 0 <= y < y + bh <= h
        assert 0.0 < r["area_frac"] < 1.0
        assert 0.0 <= r["mean_hue_deg"] < 360.0


def test_empty_and_tiny_inputs_are_silent():
    import numpy as np
    assert detect_impact_lettering(None) == []
    assert detect_impact_lettering(np.zeros((8, 8, 3), dtype=np.uint8)) == []


def test_flat_red_wash_is_not_lettering():
    # A saturated solid red block WITHIN the area band has no internal strokes
    # — the edge-density gate must reject it (blood pools / red clothing, not
    # painted lettering). 120x120 of 400x400 = 9% area, inside the band.
    import numpy as np
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[140:260, 140:260] = (30, 20, 200)   # BGR flat red patch
    assert detect_impact_lettering(img) == []


@pytest.mark.parametrize("rel", ["sfx/p000036.jpg", "sfx/p000000.jpg",
                                 "sfx/p000034.jpg", "dedup/p000055.jpg"])
def test_detector_is_deterministic(rel):
    img = _load(rel)
    assert detect_impact_lettering(img) == detect_impact_lettering(img)
