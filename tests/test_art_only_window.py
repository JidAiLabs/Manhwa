"""art_only shown frame (2026-08-17): the v4 art-only detector's RAW panel box
(panels_norm_art, chunk coords) mapped into the scene crop and used as the
shown-frame window — gutter/edge dialogue vanishes, art never gets cut.
Protected system/caption boxes stay in frame; a proposed cut band that carries
art (edges not covered by bubbles) is refused on that side."""

import numpy as np

from tools.render_prep import art_box_local, art_only_window


# ---- chunk-space art box -> scene-local -------------------------------------

def _scene(box, trim=None, w=None, h=None):
    x0, y0, x1, y1 = box
    return {"box_px_xyxy": list(box), "trim": trim or {"trimmed": False},
            "w": w or (x1 - x0), "h": h or (y1 - y0)}


def test_art_box_local_offsets_by_scene_origin():
    sc = _scene((0, 5000, 800, 6000))
    assert art_box_local(sc, [(0, 5300, 800, 6000)]) == (0, 300, 800, 1000)


def test_art_box_local_applies_trim_offsets():
    sc = _scene((0, 5000, 800, 6000),
                trim={"trimmed": True, "left_px": 10, "top_px": 40}, w=780, h=900)
    # art [0,5300,800,6000] in chunk -> local (0,300)-(800,1000) minus trim (10,40), clipped
    assert art_box_local(sc, [(0, 5300, 800, 6000)]) == (0, 260, 780, 900)


def test_art_box_local_unions_multiple_and_ignores_far_boxes():
    sc = _scene((0, 5000, 800, 6000))
    boxes = [(0, 5100, 800, 5400), (0, 5500, 800, 5900), (0, 9000, 800, 9500)]
    assert art_box_local(sc, boxes) == (0, 100, 800, 900)


def test_art_box_local_none_when_no_overlap():
    sc = _scene((0, 5000, 800, 6000))
    assert art_box_local(sc, [(0, 9000, 800, 9500)]) is None
    assert art_box_local(sc, []) is None


# ---- window ------------------------------------------------------------------

def _img(h=1000, w=800):
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    img[350:] = np.random.default_rng(1).integers(30, 230, size=(h - 350, w, 3), dtype=np.uint8)
    return img                                # flat top 350px (balloons live here), art below


def test_art_only_cuts_bubble_band_above_the_art():
    img = _img()
    win = art_only_window(img, (0, 350, 800, 1000), bubbles=[(100, 40, 500, 300)])
    assert win == (0, 350, 800, 1000)


def test_art_only_refuses_side_whose_band_carries_uncovered_art():
    img = _img()
    img[:350] = np.random.default_rng(3).integers(30, 230, size=(350, 800, 3), dtype=np.uint8)
    # detector says art starts at 350 but the band above is busy art with no bubble -> refuse
    assert art_only_window(img, (0, 350, 800, 1000), bubbles=[]) == (0, 0, 800, 1000)


def test_art_only_keeps_protected_caption_in_frame():
    img = _img()
    win = art_only_window(img, (0, 350, 800, 1000), bubbles=[(100, 40, 500, 180)],
                          protected=[(20, 200, 300, 330)])   # nameplate above the art
    assert win == (0, 200, 800, 1000)                       # bubble out, nameplate in


def test_art_only_min_keep_guard_falls_back_to_full():
    img = _img(h=600)
    assert art_only_window(img, (0, 500, 800, 600), bubbles=[(0, 0, 800, 480)]) == (0, 0, 800, 600)


def test_art_only_trims_left_right_gutter_bubbles_too():
    img = np.full((600, 1000, 3), 245, dtype=np.uint8)
    img[:, 200:800] = np.random.default_rng(4).integers(30, 230, size=(600, 600, 3), dtype=np.uint8)
    win = art_only_window(img, (200, 0, 800, 600), bubbles=[(10, 100, 180, 400), (820, 100, 990, 400)])
    assert win == (200, 0, 800, 600)


def test_art_only_straddling_bubble_pulls_edge_out_no_half_bubble():
    img = _img()                                        # flat top 350, art below
    # bubble hangs from the gutter INTO the art (350..420) -> "never cut art":
    # the edge moves up to the bubble's top instead of showing half a bubble
    win = art_only_window(img, (0, 350, 800, 1000), bubbles=[(100, 200, 500, 420)])
    assert win == (0, 200, 800, 1000)
    # a chained second bubble touching that new edge is pulled in as well
    win = art_only_window(img, (0, 350, 800, 1000),
                          bubbles=[(100, 200, 500, 420), (300, 60, 700, 210)])
    assert win == (0, 60, 800, 1000)
