"""edge_recrop_window: the tag-driven shown-frame trim (keep-mode).
Balloon stacks parked on a panel edge get cropped out of the SHOWN frame;
bubbles overlapping real art never trigger a cut."""

import numpy as np

from tools.render_prep import edge_recrop_window


def _img(h=1000, w=800, noise=False):
    if noise:
        return np.random.default_rng(2).integers(
            30, 230, size=(h, w, 3), dtype=np.uint8)
    return np.full((h, w, 3), 245, dtype=np.uint8)


def _paint_flat_top(img, y2):
    img[:y2] = 250                        # flat background behind balloons
    return img


def test_top_balloon_stack_is_cut():
    # p000023 shape: balloons occupy the top 45% over flat background
    img = _img(noise=True)
    _paint_flat_top(img, 450)
    bubbles = [(60, 10, 420, 250), (300, 200, 700, 450)]
    y0, y1 = edge_recrop_window(img, bubbles)
    assert y0 == 450 and y1 == 1000


def test_protected_system_box_blocks_cut():
    img = _img(noise=True)
    _paint_flat_top(img, 400)
    bubbles = [(60, 10, 700, 400)]
    y0, y1 = edge_recrop_window(img, bubbles,
                                protected=[(500, 100, 700, 300)])
    assert (y0, y1) == (0, 1000)


def test_mid_panel_overlap_bubble_never_cuts():
    img = _img(noise=True)
    bubbles = [(200, 400, 600, 600)]      # sits on art, mid-frame
    assert edge_recrop_window(img, bubbles) == (0, 1000)


def test_min_keep_guard_rejects_double_cut():
    img = _img()
    bubbles = [(0, 0, 800, 480), (0, 520, 800, 1000)]   # covers ~96%
    assert edge_recrop_window(img, bubbles) == (0, 1000)


def test_busy_art_band_with_sparse_bubble_not_cut():
    img = _img(noise=True)                # noisy everywhere: band NOT flat
    bubbles = [(0, 0, 200, 300)]          # covers 7.5% of the band
    assert edge_recrop_window(img, bubbles) == (0, 1000)


def test_bottom_balloon_is_cut():
    img = _img(noise=True)
    img[820:] = 250
    bubbles = [(100, 830, 700, 990)]
    y0, y1 = edge_recrop_window(img, bubbles)
    assert y0 == 0 and y1 == 830
