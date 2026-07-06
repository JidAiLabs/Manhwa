#!/usr/bin/env python3
"""impact_lettering.py — deterministic CV detector for LARGE, HIGH-SATURATION
painted impact-SFX display lettering (the manhwa violence signal: the big red
푹 / 쾅 / 퍽 splashed over a stab/blow panel).

Why this exists (verified on production panels): Apple OCR captures ZERO
stylized Korean SFX — its detector never proposes regions for large painted
lettering — so the understanding model never learns a stab panel is a stab
panel ("peaceful stroll" written over 푹 + blood). When the signal IS supplied,
the model flips a misread "grabs the collar" to a piercing strike, and a calm
landscape control does NOT hallucinate violence. This module is that signal:
pure CV, stdlib + cv2/numpy, no model, no network — DETERMINISTIC, so its
verdict can be stamped on understanding records and gated on in prep_qa.

Recipe (tuned on tests/fixtures/sfx/ — see tests/test_impact_lettering.py):
  1. HSV mask of saturated, adequately-bright pixels (S>=110, V>=60);
  2. morphological close (5x5 ellipse) + connected components;
  3. keep components whose panel-area fraction is in a band — big enough to be
     display lettering, small enough to not be a red wash / clothing;
  4. AND whose bbox interior Canny edge density is stroke-like: painted
     lettering is full of stroke edges (measured 0.10-0.18 on the real 푹/!),
     while blood pools and flat fabric are smooth (measured 0.00-0.05);
  5. AND whose circular-mean hue falls in the red-orange band (wraps 330-360/
     0-25 deg) — gates 1-4 alone fire on ANY bold saturated lettering
     regardless of color (a blue "LEVEL", green "POISON", purple "BONUS", or
     yellow "SWOOSH" banner clears them just as easily as a real stab SFX;
     see tests/fixtures/sfx/synthetic_*). The domain signal IS red/orange.

ponytail: v1 ceiling — this catches SATURATED RED/ORANGE painted lettering
only. Other saturated bold lettering (blue/green/purple/yellow banners, UI
callouts, etc.) is rejected by the hue gate — that is domain-correct, not a
false negative to chase. A RED title card/stamp can still pass this detector;
tools/prep_qa.py's impact_mismatch gate is the SECOND layer (it excludes
panels whose understood panel_kind is system/chrome/caption from the trigger
set), so a stamped red banner that understanding correctly classifies as UI
chrome can never block on it. The whitish / dark-outline slash SFX (p000034's
스윽) has no saturation to mask on and needs a separate edge-cluster pass
later; do NOT loosen the saturation gate to chase it (that reopens the
red-cloak/blood false-positive door).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import cv2
import numpy as np

# HSV gate: high saturation + adequate value. Sweep-verified S in [100, 130]
# all qualify (fire on the p000036 stab SFX; silent on the p000000 landscape,
# the p000055 dialogue bubble + red-brown cloak, and the p000034 whitish
# slash); 110/60 is the center of that band.
SAT_MIN = 110
VAL_MIN = 60
# Component area as a fraction of the panel: the display-lettering band.
# Measured lettering blobs: 0.0021-0.0066 (fixture + full-res). Below the
# floor live blood droplets / speckle noise; above the cap live washes and
# saturated clothing regions.
AREA_FRAC_MIN = 0.0015
AREA_FRAC_MAX = 0.20
# Absolute pixel floor so a degenerate/tiny image can't promote a few pixels.
AREA_PX_MIN = 64
# Canny density inside the component bbox. Lettering strokes measure
# 0.10-0.18; a flat wash's only edges are its perimeter (~0.03 at band scale);
# blood pools measure ~0.00-0.05.
EDGE_DENSITY_MIN = 0.06
_CANNY_LO, _CANNY_HI = 60, 160
_CLOSE_KERNEL = 5
# Hue gate: impact SFX in THIS domain is specifically red/orange painted
# lettering. Without it, gates above fire on ANY bold saturated lettering —
# blue/green/purple/yellow banners all measured false-positive (see
# tests/fixtures/sfx/synthetic_*). Expressed in degrees (matching
# _circular_mean_hue_deg's output) rather than a single cv2 0-179 range
# because the band wraps the 0/360 seam. Real p000036 measured 348-357 deg.
HUE_MIN_DEG = 330.0
HUE_MAX_DEG = 25.0


def _hue_in_band(mean_hue_deg: float) -> bool:
    """True when a circular-mean hue (degrees) is red-orange, wrapping the
    0/360 seam (e.g. both 350 and 10 qualify; 90 (green) does not)."""
    return mean_hue_deg >= HUE_MIN_DEG or mean_hue_deg <= HUE_MAX_DEG


def _circular_mean_hue_deg(hue_u8: np.ndarray) -> float:
    """Circular mean of OpenCV hue (0-179) in degrees [0, 360). Red straddles
    the wrap (H~0 and H~179), so a naive mean would report cyan for red."""
    if hue_u8.size == 0:
        return 0.0
    ang = hue_u8.astype(np.float32) * (2.0 * math.pi / 180.0)
    deg = math.degrees(math.atan2(float(np.sin(ang).mean()),
                                  float(np.cos(ang).mean())))
    return deg % 360.0


def detect_impact_lettering(img_bgr: Any) -> List[Dict[str, Any]]:
    """Detect large high-saturation painted SFX lettering regions.

    Returns [{"bbox": [x, y, w, h], "area_frac": float, "mean_hue_deg": float}]
    sorted by area (largest first); [] = no signal. Fail-soft: any unusable
    input (None, wrong shape, tiny) returns []."""
    if img_bgr is None or getattr(img_bgr, "ndim", 0) != 3:
        return []
    h, w = img_bgr.shape[:2]
    if h < 16 or w < 16:
        return []
    panel_area = float(h * w)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] >= SAT_MIN) & (hsv[..., 2] >= VAL_MIN))
    mask = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_CLOSE_KERNEL, _CLOSE_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if n <= 1:
        return []

    edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                      _CANNY_LO, _CANNY_HI)
    regions: List[Dict[str, Any]] = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < AREA_PX_MIN:
            continue
        area_frac = area / panel_area
        if not (AREA_FRAC_MIN <= area_frac <= AREA_FRAC_MAX):
            continue
        box_edges = edges[y:y + bh, x:x + bw]
        edge_density = float((box_edges > 0).mean()) if box_edges.size else 0.0
        if edge_density < EDGE_DENSITY_MIN:
            continue
        mean_hue = _circular_mean_hue_deg(hsv[y:y + bh, x:x + bw, 0])
        if not _hue_in_band(mean_hue):
            continue
        regions.append({
            "bbox": [x, y, bw, bh],
            "area_frac": round(area_frac, 6),
            "mean_hue_deg": round(mean_hue, 1),
        })
    regions.sort(key=lambda r: (-r["area_frac"], r["bbox"][1], r["bbox"][0]))
    return regions


if __name__ == "__main__":                                   # pragma: no cover
    import json
    import sys
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        print(path, json.dumps(detect_impact_lettering(img)))
