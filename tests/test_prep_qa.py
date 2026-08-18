"""
tests/test_prep_qa.py

TDD for tools/prep_qa.py — the pre-render QA scanner (QA-first directive).
Scans render.plan.clean.json + scenes_clean/ per shown cut and flags every
known defect class BEFORE any render is started: husk leaks, dead blank-box
leaks, ghost/visible bubble text, chrome leakage (image + narration), doc/tall
consistency, plan integrity (missing files/dims/audio, flash cuts, cold open).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "prep_qa",
    Path(__file__).resolve().parent.parent / "tools" / "prep_qa.py",
)
pq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pq)  # type: ignore[union-attr]


# ---- helpers ----------------------------------------------------------------

def _art(h, w, tone=120):
    """Midtone art block with texture (passes art/midtone gates)."""
    img = np.full((h, w, 3), tone, dtype=np.uint8)
    ys, xs = np.mgrid[0:h, 0:w]
    img[((ys // 7) + (xs // 7)) % 2 == 0] = max(30, tone - 70)
    return img


def _plan(items):
    return {"timeline": items, "scenes_subdir": "scenes_clean",
            "total_duration_sec": sum(i.get("duration_sec", 0) for i in items),
            "scene_dims": {}}


def _item(seg, files, dur=8.0, **kw):
    cuts = [{"file": f, "start": i * dur / max(1, len(files)),
             "dur": dur / max(1, len(files))} for i, f in enumerate(files)]
    d = {"segment_id": seg, "cuts": cuts, "duration_sec": dur,
         "tts_text": kw.pop("tts_text", "A quiet morning passes."),
         "tts_audio": kw.pop("tts_audio", f"/tts/{seg}.wav"),
         "tts_audio_duration_sec": kw.pop("tts_audio_duration_sec", dur)}
    d.update(kw)
    return d


# ---- parent_scene / iter_shown_cuts ------------------------------------------

def test_parent_scene_maps_split_parts():
    assert pq.parent_scene("p000031_a.jpg") == "p000031.jpg"
    assert pq.parent_scene("p000031_b.jpg") == "p000031.jpg"
    assert pq.parent_scene("p000031.jpg") == "p000031.jpg"


def test_iter_shown_cuts_walks_cuts_split2_and_branding():
    items = [
        _item("g0001_p00", ["p000001.jpg"]),
        {"segment_id": "branding_intro", "branding": "intro", "duration_sec": 7.0,
         "cuts": [{"file": "p000002.jpg", "start": 0.0, "dur": 7.0}]},
        _item("g0002_p00", ["p000003_a.jpg"]),
    ]
    items[2]["cuts"][0]["file2"] = "p000003_b.jpg"
    items[2]["cuts"][0]["layout"] = "split2"
    cuts = pq.iter_shown_cuts(_plan(items))
    files = [(c["segment_id"], c["file"]) for c in cuts]
    assert ("g0001_p00", "p000001.jpg") in files
    assert ("branding_intro", "p000002.jpg") in files
    assert ("g0002_p00", "p000003_a.jpg") in files
    assert ("g0002_p00", "p000003_b.jpg") in files          # file2 included
    assert [c for c in cuts if c["file"] == "p000002.jpg"][0]["branding"]


# ---- box interior stats: blank voids, ghosts, visible text -------------------

def _bubble_panel(*, ghost=False, visible_text=False):
    """Art panel with one white bubble box; optionally ghost or crisp text."""
    img = _art(400, 300)
    img[60:200, 40:260] = 250                                  # blanked bubble
    if ghost:
        img[120:128, 70:230] = 215                             # faint remnant
    if visible_text:
        for y in range(90, 180, 18):
            for x in range(70, 220, 22):
                img[y:y + 8, x:x + 13] = 20                    # glyph blobs
    return img, (40, 60, 260, 200)


def test_box_interior_stats_blank_and_clean():
    img, box = _bubble_panel()
    st = pq.box_interior_stats(img, box)
    assert st["blank"] is True
    assert st["ghost_frac"] < 0.01 and st["ink_frac"] < 0.01


def test_box_interior_stats_detects_ghost():
    img, box = _bubble_panel(ghost=True)
    st = pq.box_interior_stats(img, box)
    assert st["blank"] is True and st["ghost_frac"] >= 0.02


def test_box_interior_stats_detects_visible_text():
    img, box = _bubble_panel(visible_text=True)
    st = pq.box_interior_stats(img, box)
    assert st["ink_frac"] >= 0.05
    assert st["ink_glyphs"] >= 6                   # many glyph-sized blobs
    assert st["blank"] is False                    # kept text != blank void


def test_box_interior_stats_art_stroke_not_glyphs():
    # a single thick art stroke inside a white-ish box is NOT text
    img = _art(400, 300)
    img[60:200, 40:260] = 250
    import cv2
    cv2.line(img, (60, 80), (240, 180), (20, 20, 20), 12)
    st = pq.box_interior_stats(img, (40, 60, 260, 200))
    assert st["ink_frac"] >= 0.05 and st["ink_glyphs"] < 6


# ---- image_flags --------------------------------------------------------------

def test_image_flags_husk_and_dead_box_leak():
    img = np.full((500, 800, 3), 250, dtype=np.uint8)          # near-empty
    img[0:80] = _art(80, 800)                                  # sliver of art
    img[100:480, 40:760] = 252                                 # giant blank box
    flags = pq.image_flags("p000010.jpg", img, [(40, 100, 760, 480)],
                           doc=False, dims_entry={"w": 800, "h": 500, "doc": False},
                           kept_bubbles=False)
    codes = {f["code"] for f in flags}
    assert "dead_box_leak" in codes


def test_image_flags_keep_mode_skips_bubble_interior_checks():
    # bubble_shown_mode=keep (default): bubbles ship AS DRAWN — readable
    # bubble text / blank interiors are design, never blanking misses.
    img = np.full((500, 800, 3), 250, dtype=np.uint8)
    img[0:80] = _art(80, 800)
    img[100:480, 40:760] = 252
    flags = pq.image_flags("p000010.jpg", img, [(40, 100, 760, 480)],
                           doc=False, dims_entry={"w": 800, "h": 500, "doc": False})
    codes = {f["code"] for f in flags}
    assert not codes & {"dead_box_leak", "ghost_text",
                        "visible_text", "bubble_text_residue"}


def test_image_flags_low_art_husk():
    img = np.full((400, 600, 3), 248, dtype=np.uint8)
    flags = pq.image_flags("p000011.jpg", img, [], doc=False,
                           dims_entry={"w": 600, "h": 400, "doc": False})
    assert any(f["code"] == "husk" and f["severity"] == "ERROR" for f in flags)


def test_image_flags_stale_dims_mismatch():
    img = _art(400, 600)
    flags = pq.image_flags("p000012.jpg", img, [], doc=False,
                           dims_entry={"w": 999, "h": 400, "doc": False})
    assert any(f["code"] == "stale_dims" for f in flags)


def test_blank_crop_black_void_errors_even_for_sys():
    # the gap the user caught: an all-black panel passed QA because content
    # checks were skipped for sys/doc. The validity gate has NO exemption.
    img = np.zeros((400, 600, 3), dtype=np.uint8)            # all black
    flags = pq.image_flags("p000023.jpg", img, [], doc=False, sys=True,
                           dims_entry={"w": 600, "h": 400})
    assert any(f["code"] == "blank_crop" and f["severity"] == "ERROR"
               for f in flags)


def test_blank_crop_white_void_errors_even_for_doc():
    img = np.full((400, 600, 3), 255, dtype=np.uint8)        # over-inpainted white
    flags = pq.image_flags("p000001.jpg", img, [], doc=True,
                           dims_entry={"w": 600, "h": 400})
    assert any(f["code"] == "blank_crop" and f["severity"] == "ERROR"
               for f in flags)


def test_blank_crop_keeps_text_on_white_card():
    # the dropped "7TH GEN NANO MACHINE, STARTING ACTIVATION" card: a HUD/system
    # reveal is mostly white BUT carries real OCR text -> REAL content, must NOT
    # be flagged blank_crop/husk. has_text(vitem) protects it.
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    for y in range(60, 120, 12):                 # dark "glyph" rows on white
        img[y:y + 6, 80:520] = 20
    vit = {"ocr_clean": "7TH GENERATION NANO MACHINE STARTING ACTIVATION",
           "n_words": 6, "text_coverage": 0.18}
    codes = {f["code"] for f in pq.image_flags(
        "p000114.jpg", img, [], doc=False, sys=False, vitem=vit,
        dims_entry={"w": 600, "h": 400})}
    assert "blank_crop" not in codes and "husk" not in codes
    # negative control: an identical white field with NO text still blocks
    blank = np.full((400, 600, 3), 255, dtype=np.uint8)
    bcodes = {f["code"] for f in pq.image_flags(
        "p000b.jpg", blank, [], doc=False, sys=False,
        vitem={"ocr_clean": "", "n_words": 0}, dims_entry={"w": 600, "h": 400})}
    assert "blank_crop" in bcodes


def test_chunk_as_panel_blocks_a_whole_chunk():
    # ch28/ch38: a ~9000px crop is a whole stitch chunk the detector failed to
    # segment -> BLOCKING ERROR (no legit panel is this tall; clean max ~5.2k).
    flags = pq.image_flags("p000005.jpg", _art(9000, 800), [], doc=True,
                           dims_entry={"w": 800, "h": 9000})
    assert any(f["code"] == "chunk_as_panel" and f["severity"] == "ERROR"
               for f in flags)


def test_tall_legit_panel_is_not_chunk_as_panel():
    # a 5000px full-height panel (under the 8k cap) must NOT trip the gate
    flags = pq.image_flags("p000006.jpg", _art(5000, 800), [], doc=True,
                           dims_entry={"w": 800, "h": 5000})
    assert not any(f["code"] == "chunk_as_panel" for f in flags)


def test_reconciled_tall_panel_is_exempt_from_chunk_as_panel():
    # a correctly reassembled seam panel is tall BY DESIGN (spec §5.1) -> the
    # reconciled marker exempts it from the h>8000 chunk_as_panel gate.
    flags = pq.image_flags("p000007.jpg", _art(9000, 800), [], doc=True,
                           reconciled=True, dims_entry={"w": 800, "h": 9000})
    assert not any(f["code"] == "chunk_as_panel" for f in flags)


def test_non_reconciled_tall_panel_still_blocks():
    # negative control: same tall crop with NO marker is still a BLOCKING ERROR.
    flags = pq.image_flags("p000008.jpg", _art(9000, 800), [], doc=True,
                           dims_entry={"w": 800, "h": 9000})
    assert any(f["code"] == "chunk_as_panel" and f["severity"] == "ERROR"
               for f in flags)


def test_valid_image_is_not_blank_crop():
    flags = pq.image_flags("p000005.jpg", _art(400, 600), [], doc=False,
                           dims_entry={"w": 600, "h": 400})
    assert not any(f["code"] == "blank_crop" for f in flags)


def test_image_flags_doc_panel_skips_husk_and_dead_box():
    img = np.full((400, 600, 3), 250, dtype=np.uint8)          # white doc page
    flags = pq.image_flags("p000013.jpg", img, [(10, 10, 590, 390)],
                           doc=True, dims_entry={"w": 600, "h": 400, "doc": True})
    codes = {f["code"] for f in flags}
    assert "husk" not in codes and "dead_box_leak" not in codes


def test_image_flags_extreme_tall_is_info():
    img = _art(3200, 400)
    flags = pq.image_flags("p000014.jpg", img, [], doc=False,
                           dims_entry={"w": 400, "h": 3200, "doc": False})
    assert any(f["code"] == "extreme_tall" and f["severity"] == "INFO"
               for f in flags)


def test_image_flags_sys_panel_exempt_from_text_and_card_checks():
    # system-message/status cards keep their
    # text BY DESIGN — no visible_text/ghost/binary_card/dead_box/husk flags
    img, box = _bubble_panel(visible_text=True, ghost=True)
    flags = pq.image_flags("p000114.jpg", img, [box], doc=False,
                           dims_entry={"w": 300, "h": 400, "doc": False},
                           sys=True)
    assert flags == []


def test_image_flags_binary_card_exempts_story_visual_panel():
    img = np.full((400, 300, 3), 250, dtype=np.uint8)
    img[40:100, 40:120] = 20        # enough structure to avoid blank_crop
    img[240:320, 170:240] = 20
    flags = pq.image_flags(
        "p000049.jpg", img, [], doc=False,
        dims_entry={"w": 300, "h": 400, "doc": False},
        vitem={"panel_kind": "story",
               "subjects": ["dark-haired character", "character with ponytail"],
               "ocr_clean": "CAN DOCTOR BAEK USE MARTIAL ARTS TOO?",
               "text_coverage": 0.07})
    assert not any(f["code"] == "binary_card" for f in flags)


def test_image_flags_visible_text_needs_glyph_look():
    # one thick art stroke in a white box: ink is high but it is NOT text
    img = _art(400, 300)
    img[60:200, 40:260] = 250
    import cv2
    cv2.line(img, (60, 80), (240, 180), (20, 20, 20), 12)
    flags = pq.image_flags("p000029.jpg", img, [(40, 60, 260, 200)], doc=False,
                           dims_entry={"w": 300, "h": 400, "doc": False})
    assert not any(f["code"] == "visible_text" for f in flags)


def test_image_flags_husk_borderline_is_warn():
    img = np.full((400, 600, 3), 248, dtype=np.uint8)
    img[0:40] = _art(40, 600)                      # a whisker of edges
    art = pq.rp.art_content_score(img, [])
    assert art > 0
    fl_warn = pq.image_flags("p1.jpg", img, [], doc=False, dims_entry=None,
                             min_art_score=art / 0.85)   # ratio 0.85 -> WARN
    fl_err = pq.image_flags("p1.jpg", img, [], doc=False, dims_entry=None,
                            min_art_score=art / 0.5)     # ratio 0.5 -> ERROR
    assert any(f["code"] == "husk" and f["severity"] == "WARN" for f in fl_warn)
    assert any(f["code"] == "husk" and f["severity"] == "ERROR" for f in fl_err)


# ---- narration flags ----------------------------------------------------------

def test_narration_flags_chrome_phrases():
    f1 = pq.narration_flags("g0001_p00", "Presented by Redice Studio.", [])
    f2 = pq.narration_flags("g0002_p00", "The view counter shows VIEWS: 1.", [])
    ok = pq.narration_flags("g0003_p00",
                            "Cheon flees through the fog-laced peaks.", [])
    assert any(f["code"] == "chrome_narration" for f in f1)
    assert any(f["code"] == "chrome_narration" for f in f2)
    assert not ok


def test_narration_flags_ocr_echo_only_when_text_visible():
    ocr = "I will never become a cyborg no matter what they do to me"
    narr = "He swears: I will never become a cyborg, he repeats."
    visible = pq.narration_flags(
        "g0004_p00", narr, [{"ocr": ocr, "visible": True}])
    blanked = pq.narration_flags(
        "g0004_p00", narr, [{"ocr": ocr, "visible": False}])
    assert any(f["code"] == "ocr_echo" for f in visible)
    # blanked bubbles: narration REPLACES the text — that is the design
    assert not any(f["code"] == "ocr_echo" for f in blanked)


# ---- vision consistency flags ---------------------------------------------------

def test_vision_flags_chrome_leak_via_title_dominance():
    vitem = {"ocr_clean": "OMNISCIENT READER", "text_only": False,
             "text_coverage": 0.05, "n_words": 2}
    fl = pq.vision_flags("p000029.jpg", vitem, dims_entry={"doc": False},
                         series_title="Omniscient Reader")
    assert any(f["code"] == "chrome_leak" and f["severity"] == "ERROR"
               for f in fl)


def test_vision_flags_empty_bubble_shown_errors():
    vitem = {"panel_kind": "empty", "subjects": ["speech bubble"],
             "ocr_clean": "DAMN IT,", "text_coverage": 0.0299}
    fl = pq.vision_flags("p000047.jpg", vitem,
                         dims_entry={"doc": False, "sys": False},
                         series_title=None)
    assert any(f["code"] == "empty_bubble_shown"
               and f["severity"] == "ERROR" for f in fl)


def test_vision_flags_doc_flag_missing_only_when_text_renders_unprotected():
    vitem = {"ocr_clean": "lots of ui text " * 10, "text_only": False,
             "text_coverage": 0.4, "n_words": 30}
    # wordy + shown with text NOT blanked and NOT protected -> defect
    fl = pq.vision_flags("p000003.jpg", vitem,
                         dims_entry={"doc": False, "sys": False, "blanked": False},
                         series_title=None)
    assert any(f["code"] == "doc_flag_missing" for f in fl)
    # blanked dialogue / doc-protected / sys panels: nothing to protect
    for d in ({"doc": False, "sys": False, "blanked": True},
              {"doc": True, "sys": False, "blanked": False},
              {"doc": False, "sys": True, "blanked": False}):
        ok = pq.vision_flags("p000003.jpg", vitem, dims_entry=d, series_title=None)
        assert not any(f["code"] == "doc_flag_missing" for f in ok), d


# ---- plan integrity flags --------------------------------------------------------

def test_plan_flags_no_cold_open_when_intro_first():
    items = [
        {"segment_id": "branding_intro", "branding": "intro", "duration_sec": 7.0,
         "cuts": [{"file": "p000001.jpg", "start": 0, "dur": 7.0}]},
        _item("g0001_p00", ["p000001.jpg"]),
    ]
    fl = pq.plan_flags(_plan(items), clean_files={"p000001.jpg"},
                       audio_exists=lambda p: True)
    assert any(f["code"] == "no_cold_open" for f in fl)


def test_plan_flags_missing_file_dims_audio_and_flash_cut():
    items = [_item("g0001_p00", ["p000001.jpg", "p000404.jpg"])]
    items[0]["cuts"][1]["dur"] = 0.8                       # flash cut
    plan = _plan(items)
    plan["source_tts_index"] = "/x/tts/tts_index.json"     # voiced plan
    plan["scene_dims"] = {"p000001.jpg": {"w": 100, "h": 100, "doc": False}}
    fl = pq.plan_flags(plan, clean_files={"p000001.jpg"},
                       audio_exists=lambda p: False)
    codes = [f["code"] for f in fl]
    assert "missing_file" in codes                          # p000404 not on disk
    assert "missing_dims" in codes                          # p000404 has no dims
    assert "missing_audio" in codes
    assert "flash_cut" in codes


def test_plan_flags_empty_item_and_clean_plan_passes():
    bad = _plan([dict(_item("g0001_p00", ["p000001.jpg"]), cuts=[])])
    fl = pq.plan_flags(bad, clean_files={"p000001.jpg"},
                       audio_exists=lambda p: True)
    assert any(f["code"] == "empty_item" and f["severity"] == "ERROR"
               for f in fl)

    # the outro is rendered by Remotion's own end-card — cuts=[] is BY DESIGN
    good_items = [
        _item("g0001_p00", ["p000001.jpg"]),
        {"segment_id": "branding_intro", "branding": "intro", "duration_sec": 7.0,
         "cuts": [{"file": "p000001.jpg", "start": 0, "dur": 7.0}]},
        {"segment_id": "branding_outro", "branding": "outro", "duration_sec": 5.0,
         "cuts": []},
    ]
    plan = _plan(good_items)
    plan["scene_dims"] = {"p000001.jpg": {"w": 100, "h": 100, "doc": False}}
    fl = pq.plan_flags(plan, clean_files={"p000001.jpg"},
                       audio_exists=lambda p: True)
    assert not [f for f in fl if f["severity"] == "ERROR"]


def test_plan_no_branding_items_yields_no_missing_branding():
    # Channel design (commit 3ea4271): a chapter ENDS on its last story panel and
    # carries NO outro (and no intro) — the channel watermark is a separate
    # always-on overlay (not a timeline item) and the arc intro is bundle-level,
    # prepended at concat. So a normal chapter has NO branding item, and the
    # absence of an outro must NOT raise the stale 'expected an outro' warn.
    plan = _plan([_item("g0001_p00", ["p000001.jpg"])])
    plan["scene_dims"] = {"p000001.jpg": {"w": 100, "h": 100, "doc": False}}
    fl = pq.plan_flags(plan, clean_files={"p000001.jpg"},
                       audio_exists=lambda p: True)
    assert not any(f["code"] == "missing_branding" for f in fl)


# ---- report assembly --------------------------------------------------------------

def test_build_report_counts_and_html_smoke():
    flags = [
        {"code": "husk", "severity": "ERROR", "scene": "p000011.jpg",
         "segment_id": "g0001_p00", "detail": "art_score=0.001"},
        {"code": "ghost_text", "severity": "WARN", "scene": "p000012.jpg",
         "segment_id": "g0002_p00", "detail": "ghost_frac=0.04"},
    ]
    rep = pq.build_report("Nano Machine — Chapter 1", flags, n_cuts=12)
    assert rep["counts"]["ERROR"] == 1 and rep["counts"]["WARN"] == 1
    assert rep["n_cuts"] == 12

    html = pq.render_html(rep, thumbs={"p000011.jpg": b"\xff\xd8fakejpg"})
    assert "husk" in html and "ghost_text" in html
    assert "data:image/jpeg;base64," in html


def test_render_html_segment_flag_uses_thumb_scene_fallback():
    # segment-level flags (ocr_echo) have no scene — they must still show
    # the panel that segment displays
    flags = [{"code": "ocr_echo", "severity": "WARN", "scene": "",
              "thumb_scene": "p000017.jpg", "segment_id": "g0011_p04",
              "detail": "narration repeats..."}]
    rep = pq.build_report("X", flags, n_cuts=1)
    html = pq.render_html(rep, thumbs={"p000017.jpg": b"\xff\xd8fakejpg"})
    assert html.count("data:image/jpeg;base64,") == 1


def test_render_html_gallery_groups_by_segment_with_narration():
    # gallery = one block per SEGMENT: its narration line above its cut
    # thumbs, in timeline order — the user reviews story + visuals together
    rep = pq.build_report("X", [], n_cuts=3)
    gallery = [
        {"segment_id": "g0001_p00",
         "narration": "Prince Cheon flees through the fog.",
         "files": ["p000001.jpg", "p000002.jpg"]},
        {"segment_id": "g0002_p01", "narration": "The assassins close in.",
         "files": ["p000003.jpg"]},
    ]
    html = pq.render_html(rep, thumbs={"p000001.jpg": b"\xff\xd8a",
                                       "p000002.jpg": b"\xff\xd8b",
                                       "p000003.jpg": b"\xff\xd8c"},
                          gallery=gallery)
    assert "All shown cuts" in html
    assert html.count("data:image/jpeg;base64,") == 3
    assert "Prince Cheon flees through the fog." in html
    assert "The assassins close in." in html
    assert html.index("g0001_p00") < html.index("g0002_p01")


def test_cross_dup_flag_for_consecutive_near_identical_cuts():
    import numpy as np
    import cv2
    big = np.full((600, 400, 3), 200, np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(40):
        x, y = int(rng.integers(10, 370)), int(rng.integers(10, 570))
        cv2.rectangle(big, (x, y), (x + 18, y + 12),
                      (int(rng.integers(0, 255)),) * 3, -1)
    zoom = cv2.resize(big[380:560, 100:340], (400, 300))
    other = np.full((600, 400, 3), 30, np.uint8)
    seq = [{"segment_id": "g1", "file": "a.jpg"},
           {"segment_id": "g2", "file": "b.jpg"},
           {"segment_id": "g3", "file": "c.jpg"}]
    imgs = {"a.jpg": big, "b.jpg": zoom, "c.jpg": other}
    fl = pq.cross_dup_flags(seq, lambda f: imgs.get(f))
    assert any(f["code"] == "cross_dup" and f["severity"] == "ERROR"
               and f["scene"] == "b.jpg" for f in fl)
    assert not any(f.get("scene") == "c.jpg" for f in fl)

    # a panel that OWNS its own narration line is a distinct beat — never a
    # cross_dup (else dropping it holds a neighbour and narrates an unshown
    # shot; real ch1 p043/p044 dhash 37 falsely matched by multi_scale_contained)
    fl2 = pq.cross_dup_flags(seq, lambda f: imgs.get(f), narrated={"b.jpg"})
    assert not any(f.get("scene") == "b.jpg" for f in fl2)


# ---- FIX 1 Root C: near_dup_residual tripwire (bubble-masked, WARN only) ------
# cross_dup keys on containment; this bubble-MASKS the perceptual hash so
# identical art under DIFFERENT dialogue (whose outlines survive text cleaning)
# is caught too. WARN only — never auto-dropped; render_prep is the real fix.

def _bubbled(grad="h", x1=30):
    """Identical-shape helper: flat-white top band + a low-freq gradient bottom,
    with a cleaned bubble (dark outline) on the band; returns (image, box)."""
    img = np.full((400, 300, 3), 255, np.uint8)
    if grad == "h":
        row = np.linspace(0, 220, 300).astype(np.uint8)
        img[200:400] = np.stack([np.tile(row, (200, 1))] * 3, axis=-1)
    else:
        col = np.linspace(0, 220, 200).astype(np.uint8)
        img[200:400] = np.stack([np.tile(col.reshape(-1, 1), (1, 300))] * 3, axis=-1)
    cv2.rectangle(img, (x1, 40), (x1 + 120, 160), (0, 0, 0), 4)
    return img.astype(np.uint8), (x1 - 2, 38, x1 + 122, 162)


def test_near_dup_residual_warns_on_masked_near_dup_pair():
    a, boxA = _bubbled("h", 30)
    b, boxB = _bubbled("h", 150)               # identical art, bubble elsewhere
    seq = [{"segment_id": "g1", "file": "a.jpg"},
           {"segment_id": "g2", "file": "b.jpg"}]
    imgs = {"a.jpg": a, "b.jpg": b}
    boxes = {"a.jpg": [boxA], "b.jpg": [boxB]}
    fl = pq.near_dup_residual_flags(seq, lambda f: imgs.get(f),
                                    lambda f: boxes.get(f, []))
    assert [f["code"] for f in fl] == ["near_dup_residual"]
    assert fl[0]["severity"] == "WARN" and fl[0]["scene"] == "b.jpg"


def test_near_dup_residual_silent_on_distinct_art():
    # a clean plan (distinct consecutive art) yields ZERO near_dup_residual
    a, boxA = _bubbled("h", 30)
    c, boxC = _bubbled("v", 30)                # different art
    seq = [{"segment_id": "g1", "file": "a.jpg"},
           {"segment_id": "g2", "file": "c.jpg"}]
    imgs = {"a.jpg": a, "c.jpg": c}
    boxes = {"a.jpg": [boxA], "c.jpg": [boxC]}
    fl = pq.near_dup_residual_flags(seq, lambda f: imgs.get(f),
                                    lambda f: boxes.get(f, []))
    assert fl == []


def test_near_dup_residual_exempts_system_and_doc_panels():
    # two system/doc panels sharing a UI frame but carrying different text must
    # NOT flag — a shared chrome is not a duplicate story panel
    a, boxA = _bubbled("h", 30)
    b, boxB = _bubbled("h", 150)
    seq = [{"segment_id": "g1", "file": "a.jpg"},
           {"segment_id": "g2", "file": "b.jpg"}]
    imgs = {"a.jpg": a, "b.jpg": b}
    boxes = {"a.jpg": [boxA], "b.jpg": [boxB]}
    fl = pq.near_dup_residual_flags(seq, lambda f: imgs.get(f),
                                    lambda f: boxes.get(f, []),
                                    is_exempt=lambda f: f == "b.jpg")
    assert fl == []


def test_near_dup_residual_silent_on_same_file_hold():
    # the SAME file held over two consecutive cuts is a deliberate continuous
    # shot (merge_consecutive_same_image_cuts), never a duplicate
    a, boxA = _bubbled("h", 30)
    seq = [{"segment_id": "g1", "file": "a.jpg"},
           {"segment_id": "g2", "file": "a.jpg"}]
    fl = pq.near_dup_residual_flags(seq, lambda f: {"a.jpg": a}.get(f),
                                    lambda f: {"a.jpg": [boxA]}.get(f, []))
    assert fl == []


def test_missing_audio_is_info_on_estimate_plans():
    # step-1 plans are built WITHOUT voiceover (duration estimates): audio
    # cannot exist yet — that's the designed state, not a defect
    items = [_item("g0001_p00", ["p000001.jpg"])]
    del items[0]["tts_audio"]
    plan = _plan(items)                      # no source_tts_index -> estimate
    plan["scene_dims"] = {"p000001.jpg": {"w": 9, "h": 9, "doc": False}}
    fl = pq.plan_flags(plan, clean_files={"p000001.jpg"},
                       audio_exists=lambda p: False)
    assert not [f for f in fl if f["code"] == "missing_audio"
                and f["severity"] == "ERROR"]
    # voiced plans still enforce hard
    plan["source_tts_index"] = "/x/tts/tts_index.json"
    fl2 = pq.plan_flags(plan, clean_files={"p000001.jpg"},
                        audio_exists=lambda p: False)
    assert any(f["code"] == "missing_audio" and f["severity"] == "ERROR"
               for f in fl2)


# ---- narration<->image alignment (stale-manifest class + semantic judge) ----

def _seg(seg_id, text, files):
    return {"segment_id": seg_id, "tts_text": text,
            "cuts": [{"file": f, "duration_sec": 4.0} for f in files]}


def test_alignment_clean_verbatim_passes():
    plan = {"timeline": [_seg("g0001_p00", "[excited] The FOG drifts!",
                              ["a.jpg"])]}
    beats = {"beats": [{"group_id": 1, "narration": "The fog drifts."}]}
    groups = {"shots": [{"group_id": 1}]}
    script = {"narration_source": "gemini_verbatim"}
    assert pq.alignment_flags(plan, beats, groups, script) == []


def test_alignment_flags_beats_incomplete():
    beats = {"beats": [{"group_id": 1, "narration": "x"}]}
    groups = {"shots": [{"group_id": 1}, {"group_id": 2}, {"group_id": 3}]}
    fl = pq.alignment_flags({"timeline": []}, beats, groups,
                            {"narration_source": "gemini_verbatim"})
    assert [f["code"] for f in fl] == ["beats_incomplete"]
    assert fl[0]["severity"] == pq.ERROR


def test_alignment_flags_narration_stale():
    plan = {"timeline": [_seg(
        "g0002_p01",
        "A screen displays the webnovel, episode after episode of it.",
        ["b.jpg"])]}
    beats = {"beats": [{"group_id": 2, "narration":
                        "The train rattles along; he stares at his phone."}]}
    groups = {"shots": [{"group_id": 2}]}
    fl = pq.alignment_flags(plan, beats, groups,
                            {"narration_source": "gemini_verbatim"})
    assert [f["code"] for f in fl] == ["narration_stale"]
    assert fl[0]["severity"] == pq.ERROR and fl[0]["segment_id"] == "g0002_p01"


def test_alignment_microbeats_compare_group_text():
    plan = {"timeline": [
        _seg("g0002_p00", "The train rattles along.", ["a.jpg"]),
        _seg("g0002_p01", "He stares at his phone.", ["b.jpg"]),
    ]}
    beats = {"beats": [{"group_id": 2, "narration":
                        "The train rattles along; he stares at his phone."}]}
    groups = {"shots": [{"group_id": 2}]}
    script = {"narration_source": "gemini_verbatim", "microbeats": True}
    assert pq.alignment_flags(plan, beats, groups, script) == []


def test_alignment_title_card_compares_against_story_hook():
    plan = {"timeline": [
        _seg("g0005_p00", "The truth is finally about to surface.", ["a.jpg"]),
    ]}
    beats = {"beats": [{
        "group_id": 5,
        "beat_title": "Chapter Title Card",
        "narration": "As the truth surfaces, we reach Chapter 7: The Trap.",
        "hook": "The truth is finally about to surface.",
    }]}
    groups = {"shots": [{"group_id": 5}]}
    script = {"narration_source": "gemini_verbatim", "microbeats": True}
    assert pq.alignment_flags(plan, beats, groups, script) == []


def test_alignment_skips_nonverbatim_script():
    plan = {"timeline": [_seg("g0001_p00", "totally different words",
                              ["a.jpg"])]}
    beats = {"beats": [{"group_id": 1, "narration": "the original line"}]}
    groups = {"shots": [{"group_id": 1}]}
    assert pq.alignment_flags(plan, beats, groups,
                              {"narration_source": "legacy"}) == []


def test_semantic_judge_flags_mismatch(monkeypatch, tmp_path):
    import sys, types
    fake = types.ModuleType("ollama")
    fake.chat = lambda **kw: {"message": {"content":
        '{"match": false, "confidence": 85, "reason": "image shows a dragon"}'}}
    monkeypatch.setitem(sys.modules, "ollama", fake)
    (tmp_path / "a.jpg").write_bytes(b"jpg")
    plan = {"timeline": [_seg("g0001_p00", "impressive statistics",
                              ["a.jpg"])]}
    fl = pq.semantic_alignment_flags(plan, str(tmp_path))
    assert [f["code"] for f in fl] == ["narration_mismatch"]
    assert fl[0]["severity"] == pq.WARN and "dragon" in fl[0]["detail"]


def test_semantic_judge_match_is_quiet(monkeypatch, tmp_path):
    import sys, types
    fake = types.ModuleType("ollama")
    fake.chat = lambda **kw: {"message": {"content":
        '{"match": true, "confidence": 95, "reason": "ok"}'}}
    monkeypatch.setitem(sys.modules, "ollama", fake)
    (tmp_path / "a.jpg").write_bytes(b"jpg")
    plan = {"timeline": [_seg("g0001_p00", "statistics", ["a.jpg"])]}
    assert pq.semantic_alignment_flags(plan, str(tmp_path)) == []


def test_semantic_judge_skips_without_ollama(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "ollama", None)
    plan = {"timeline": [_seg("g0001_p00", "x", ["a.jpg"])]}
    fl = pq.semantic_alignment_flags(plan, "/nonexistent")
    assert [f["code"] for f in fl] == ["semantic_skipped"]
    assert fl[0]["severity"] == pq.INFO


def _by_image(verdicts):
    """Mock ollama.chat returning a per-image verdict keyed on the basename of
    the image it was handed: {filename: (match_bool, confidence)}."""
    import os
    import json as _json

    def chat(**kw):
        img = os.path.basename(str(kw["messages"][0]["images"][0]))
        match, conf = verdicts.get(img, (True, 95))
        return {"message": {"content": _json.dumps(
            {"match": match, "confidence": conf, "reason": f"{img}"})}}
    return chat


def test_semantic_judge_group_aware_passes_when_any_cut_matches(monkeypatch,
                                                                tmp_path):
    """The montage, not the primary panel, is what the viewer sees. A
    multi_cut segment whose PRIMARY mismatches but whose later cut matches the
    narration must NOT be flagged — the narration belongs to the group."""
    import sys
    import types
    fake = types.ModuleType("ollama")
    fake.chat = _by_image({"landscape.jpg": (False, 90),   # peaceful, no blood
                           "prince.jpg": (True, 90)})       # bloodied prince
    monkeypatch.setitem(sys.modules, "ollama", fake)
    for f in ("landscape.jpg", "prince.jpg"):
        (tmp_path / f).write_bytes(b"jpg")
    plan = {"timeline": [_seg("g0001_p00",
                              "Prince Cheon flees, covered in blood.",
                              ["landscape.jpg", "prince.jpg"])]}
    assert pq.semantic_alignment_flags(plan, str(tmp_path)) == []


def test_semantic_judge_flags_only_when_no_cut_matches(monkeypatch, tmp_path):
    """If the narration fits NONE of the panels actually shown, it is still a
    real mismatch — flag once, citing the most confidently-rejected panel."""
    import sys
    import types
    fake = types.ModuleType("ollama")
    fake.chat = _by_image({"a.jpg": (False, 70),
                           "b.jpg": (False, 88)})
    monkeypatch.setitem(sys.modules, "ollama", fake)
    for f in ("a.jpg", "b.jpg"):
        (tmp_path / f).write_bytes(b"jpg")
    plan = {"timeline": [_seg("g0001_p00", "a dragon roars", ["a.jpg", "b.jpg"])]}
    fl = pq.semantic_alignment_flags(plan, str(tmp_path))
    assert [f["code"] for f in fl] == ["narration_mismatch"]
    assert fl[0]["scene"] == "b.jpg"   # highest-confidence rejection cited


def test_grounding_flags_parallel_matches_serial(monkeypatch, tmp_path):
    """The montage grounding judge is parallelized (STUDIO_QA_CONC>1) so the
    26B calls fill ollama's NUM_PARALLEL slots. Parallel MUST be byte-identical
    to serial: same beats flagged, same order, same issue text."""
    import sys
    import types
    import time
    import json as _json

    def chat(**kw):
        content = str(kw["messages"][0]["content"])
        weak = "WEAK" in content          # narration carrying the marker is weak
        time.sleep(0.02)                  # force threads to genuinely overlap
        return {"message": {"content": _json.dumps(
            {"ok": (not weak), "issue": ("invented thing" if weak else "")})}}

    fake = types.ModuleType("ollama")
    fake.chat = chat
    monkeypatch.setitem(sys.modules, "ollama", fake)
    files = [f"p{i:03d}.jpg" for i in range(8)]
    for f in files:
        (tmp_path / f).write_bytes(b"jpg")
    plan = {"timeline": [
        _seg(f"g{i:04d}_p00",
             ("WEAK narration" if i % 3 == 0 else "grounded narration"),
             [files[i]])
        for i in range(8)]}

    monkeypatch.setenv("STUDIO_QA_CONC", "1")
    serial = pq.grounding_flags(plan, str(tmp_path))
    monkeypatch.setenv("STUDIO_QA_CONC", "3")
    parallel = pq.grounding_flags(plan, str(tmp_path))

    assert serial == parallel                      # order + content preserved
    assert [f["segment_id"] for f in serial] == [
        "g0000_p00", "g0003_p00", "g0006_p00"]     # every 3rd beat, in order
    assert all(f["code"] == "grounding_weak" for f in serial)


def test_grounding_cache_reuses_verdicts(monkeypatch, tmp_path):
    """The verdict cache memoizes by (model, narration, panels): a second pass
    over unchanged beats makes ZERO new gemma calls and returns identical flags —
    this is what collapses the redundant voiceover-time grounding. A CHANGED
    narration re-judges only that beat."""
    import sys
    import types
    import os
    import json as _json

    calls = {"n": 0}

    def chat(**kw):
        calls["n"] += 1
        weak = "WEAK" in str(kw["messages"][0]["content"])
        return {"message": {"content": _json.dumps(
            {"ok": (not weak), "issue": ("x" if weak else "")})}}

    fake = types.ModuleType("ollama")
    fake.chat = chat
    monkeypatch.setitem(sys.modules, "ollama", fake)
    files = [f"p{i:03d}.jpg" for i in range(5)]
    for f in files:
        (tmp_path / f).write_bytes(b"jpg")
    plan = {"timeline": [
        _seg(f"g{i:04d}_p00", ("WEAK x" if i % 2 == 0 else "ok x"), [files[i]])
        for i in range(5)]}
    cache = str(tmp_path / ".gcache.json")

    first = pq.grounding_flags(plan, str(tmp_path), cache_path=cache)
    assert calls["n"] == 5                 # first pass judges every beat
    assert os.path.exists(cache)

    second = pq.grounding_flags(plan, str(tmp_path), cache_path=cache)
    assert calls["n"] == 5                 # second pass: ZERO new gemma calls
    assert first == second                 # identical flags

    plan["timeline"][1]["tts_text"] = "NEW WEAK x"   # one beat's narration changes
    third = pq.grounding_flags(plan, str(tmp_path), cache_path=cache)
    assert calls["n"] == 6                 # exactly ONE re-judge (the changed beat)
    assert "g0001_p00" in [f["segment_id"] for f in third]


def test_semantic_judge_skips_held_cuts(monkeypatch, tmp_path):
    """A held cut intentionally shows the PREVIOUS segment's panel while new
    narration plays — it is editorial coverage, not a narration match, so the
    judge must skip it (consistent with montage_flags). A segment whose only
    cut is held produces no narration_mismatch even when the judge would
    reject the held image."""
    import sys
    import types
    fake = types.ModuleType("ollama")
    fake.chat = _by_image({"prev.jpg": (False, 95)})   # held panel, mismatches
    monkeypatch.setitem(sys.modules, "ollama", fake)
    (tmp_path / "prev.jpg").write_bytes(b"jpg")
    plan = {"timeline": [{"segment_id": "g0002_p01", "tts_text": "new beat",
                          "cuts": [{"file": "prev.jpg", "held": True,
                                    "duration_sec": 4.0}]}]}
    assert pq.semantic_alignment_flags(plan, str(tmp_path)) == []


def test_semantic_judge_considers_split_half_file2(monkeypatch, tmp_path):
    """split2 cuts render file + file2 side-by-side; both are on screen and
    must be candidate matches for the narration."""
    import sys
    import types
    fake = types.ModuleType("ollama")
    fake.chat = _by_image({"left.jpg": (False, 90),
                           "right.jpg": (True, 90)})
    monkeypatch.setitem(sys.modules, "ollama", fake)
    for f in ("left.jpg", "right.jpg"):
        (tmp_path / f).write_bytes(b"jpg")
    plan = {"timeline": [{"segment_id": "g0001_p00", "tts_text": "the reveal",
                          "cuts": [{"file": "left.jpg", "file2": "right.jpg",
                                    "layout": "split2", "duration_sec": 4.0}]}]}
    assert pq.semantic_alignment_flags(plan, str(tmp_path)) == []


# ---- story-level QA: filler narration, substituted panels, dropped cards ----

def _beats(items):
    return {"beats": items}


def test_story_flags_filler_and_empty_narration():
    plan = {"timeline": [
        _item("g0001_p00", ["p000001.jpg"], tts_text="A real opening line."),
        _item("g0002_p01", ["p000002.jpg"], tts_text="The scene continues."),
        _item("g0003_p02", ["p000003.jpg"], tts_text="   "),
    ]}
    beats = _beats([
        {"group_id": 1, "narration": "A real opening line.", "scene_files": ["p000001.jpg"]},
        {"group_id": 2, "narration": "", "scene_files": ["p000002.jpg"]},
        {"group_id": 3, "narration": "", "scene_files": ["p000003.jpg"]},
    ])
    fl = pq.story_flags(plan, beats, {})
    filler = [f for f in fl if f["code"] == "filler_narration"]
    assert {f["segment_id"] for f in filler} == {"g0002_p01", "g0003_p02"}
    assert all(f["severity"] == pq.ERROR for f in filler)


def test_story_flags_substituted_panel_mismatch():
    # g0061: beat's intended panel p000094 was dropped; a stand-in is shown.
    plan = {"timeline": [
        _item("g0061_p00", ["p000089.jpg"], tts_text="The reason she's special is because..."),
        _item("g0062_p01", ["p000088.jpg"], tts_text="A faint aura, not human.", held=True),
    ]}
    # mark g0062's cut as held (stand-in)
    plan["timeline"][1]["cuts"][0]["held"] = True
    beats = _beats([
        {"group_id": 61, "narration": "...", "scene_files": ["p000094.jpg"]},
        {"group_id": 62, "narration": "...", "scene_files": ["p000095.jpg"]},
    ])
    fl = pq.story_flags(plan, beats, {})
    sub = {f["segment_id"]: f["severity"] for f in fl if f["code"] == "panel_substituted"}
    assert sub.get("g0061_p00") == pq.ERROR     # silent swap (not held)
    assert sub.get("g0062_p01") == pq.WARN      # held stand-in is softer


def test_story_flags_dropped_system_card():
    plan = {"timeline": [_item("g0001_p00", ["p000005.jpg"], tts_text="ok")]}
    vitems = {
        # clean flat-frame title card → flagged
        "p000113.jpg": {"ocr_clean": "SYSTEM ACTIVATION.", "text_only": False,
                        "text_coverage": 0.09, "flat_frac": 0.88},
        # publication/title chrome → intentionally absent from recap visuals
        "p000008.jpg": {"ocr_clean": "Nano Machine CHAPTER 7 그림 각색 원작",
                        "panel_kind": "chrome", "text_only": False,
                        "text_coverage": 0.12, "flat_frac": 0.91},
        # pure text/bubble context panels → narrated context, not system cards
        "p000047.jpg": {"ocr_clean": "DAMN IT,", "panel_kind": "empty",
                        "subjects": ["speech bubble"], "text_only": False,
                        "text_coverage": 0.03, "flat_frac": 0.82},
        "p000059.jpg": {"ocr_clean": "HE'LL HAVE NO PROBLEM WITH OPERATING FORMATION.",
                        "panel_kind": "story",
                        "subjects": ["speech bubble", "character's hair"],
                        "text_only": False, "text_coverage": 0.098,
                        "flat_frac": 0.78},
        # all-caps SFX on textured art (low flat_frac) → NOT a card, not flagged
        "p000099.jpg": {"ocr_clean": "ACK!!! KEUACK KKK!!!", "text_only": False,
                        "text_coverage": 0.04, "flat_frac": 0.12},
    }
    fl = pq.story_flags(plan, _beats([]), vitems)
    cards = [f for f in fl if f["code"] == "system_card_dropped"]
    assert [f["scene"] for f in cards] == ["p000113.jpg"]   # only the real card
    assert cards[0]["severity"] == pq.WARN   # WARN, not a hard-fail (cosmetic)


def test_story_flags_quiet_on_healthy_plan():
    plan = {"timeline": [_item("g0001_p00", ["p000001.jpg"], tts_text="A good line.")]}
    beats = _beats([{"group_id": 1, "narration": "A good line.",
                     "scene_files": ["p000001.jpg"]}])
    assert pq.story_flags(plan, beats, {}) == []


# ---- montage degeneracy (user screenshot: 6 segments cycling 2 crops) -------

def test_montage_flags_degenerate_loop():
    tl = []
    for i in range(6):
        f = "a.jpg" if i % 2 == 0 else "b.jpg"
        tl.append(_seg(f"g{i+1:04d}_p00", f"line {i}", [f]))
    fl = pq.montage_flags({"timeline": tl})
    codes = {f["code"] for f in fl}
    assert "visual_loop" in codes and "montage_degenerate" in codes
    assert all(f["severity"] == pq.ERROR for f in fl)


def test_montage_flags_quiet_on_healthy_plan():
    tl = [_seg(f"g{i+1:04d}_p00", "x", [f"p{i}.jpg", f"q{i}.jpg"])
          for i in range(6)]
    assert pq.montage_flags({"timeline": tl}) == []


def test_montage_flags_tolerates_single_reshow():
    tl = [_seg("g0001_p00", "x", ["a.jpg", "b.jpg"]),
          _seg("g0002_p00", "y", ["c.jpg"]),
          _seg("g0003_p00", "z", ["a.jpg", "d.jpg"])]   # one re-show is fine
    assert pq.montage_flags({"timeline": tl}) == []


# ---- caption voicing contract: showing optional, VOICING mandatory ----------

def test_caption_unvoiced_flags_fire_and_clear():
    beats = {"beats": [{"group_id": 5, "narration":
        "On the day he finished the web novel, everything changed.",
        "scene_files": ["c.jpg", "d.jpg"]}]}
    vitems = {"c.jpg": {"text_only": True,
                        "ocr_clean": "ON THE DAY I FINISHED THE WEB NOVEL..."},
              "d.jpg": {"recovered": True, "ocr_clean":
                        "I BECAME THE ONLY PERSON WHO KNEW HOW THE WORLD "
                        "WAS GOING TO END."}}
    fl = pq.caption_unvoiced_flags(beats, vitems)
    assert [f["code"] for f in fl] == ["caption_unvoiced"]
    assert fl[0]["scene"] == "d.jpg" and fl[0]["severity"] == pq.ERROR
    assert fl[0]["segment_id"] == "g0005"


def test_caption_unvoiced_ignores_art_panels_and_short_text():
    beats = {"beats": [{"group_id": 1, "narration": "x",
                        "scene_files": ["a.jpg", "b.jpg"]}]}
    vitems = {"a.jpg": {"text_only": False,
                        "ocr_clean": "WHO THE HELL ARE YOU TO SAY THAT"},
              "b.jpg": {"text_only": True, "ocr_clean": "THE END"}}
    assert pq.caption_unvoiced_flags(beats, vitems) == []


def test_caption_rule_in_writer_prompt():
    src = (Path(__file__).resolve().parent.parent / "tools"
           / "gemini_narrative_pass.py").read_text()
    assert "NARRATIVE CAPTIONS ARE NOT CHROME" in src
    assert "STORY'S VOICE" in src


def test_caption_unvoiced_skips_app_ui_screens():
    beats = {"beats": [{"group_id": 2, "narration":
        "He scrolls Three Ways to Survive the Apocalypse on his phone.",
        "scene_files": ["ui.jpg"]}]}
    vitems = {"ui.jpg": {"text_only": True, "ocr_clean":
        "THREE WAYS TO SURVIVE THE APOCALYPSE READ EPISODE 1389 "
        "COMMENTS : 1 VIEWS : 1 READ EP"}}
    assert pq.caption_unvoiced_flags(beats, vitems) == []


def test_continuity_context_in_writer():
    src = (Path(__file__).resolve().parent.parent / "tools"
           / "gemini_narrative_pass.py").read_text()
    assert "previous_narration" in src and "CONTINUITY" in src


def test_fragment_dangle_flags_trailing_stub():
    fl = pq.narration_flags(
        "g0009_p02",
        'Our protagonist is smirking, stuck on one realization: "And I..."',
        [])
    assert [f["code"] for f in fl] == ["fragment_dangle"]
    assert fl[0]["severity"] == pq.ERROR


def test_fragment_dangle_ignores_midline_and_long_quotes():
    ok1 = pq.narration_flags(
        "g0011_p04",
        "He reads about 'the ending...' and realizes what it means.", [])
    ok2 = pq.narration_flags(
        "g0002_p00",
        'She whispers: "I have waited ten years for this moment to come..."',
        [])
    assert [f["code"] for f in ok1] == []
    assert [f["code"] for f in ok2] == []


def test_montage_and_repeat_checks_exempt_held_cuts():
    held = {"file": "a.jpg", "start": 0.0, "dur": 4.0, "held": True}
    tl = [_seg("g0001_p00", "x", ["a.jpg"]),
          {"segment_id": "g0002_p00", "tts_text": "y", "cuts": [dict(held)]},
          {"segment_id": "g0003_p00", "tts_text": "z", "cuts": [dict(held)]},
          {"segment_id": "g0004_p00", "tts_text": "w", "cuts": [dict(held)]}]
    assert pq.montage_flags({"timeline": tl}) == []


def test_caption_check_skips_chrome_endcards():
    beats = {"beats": [{"group_id": 9, "narration": "The story ends.",
                        "scene_files": ["e.jpg"]}]}
    vitems = {"e.jpg": {"recovered": True, "ocr_clean":
              "THANKS FOR READING THIS CHAPTER ON OUR WEBSITE ELFTOON "
              ". com DON'T FORGET TO JOIN OUR DISCORD"}}
    assert pq.caption_unvoiced_flags(beats, vitems) == []


def test_caption_paraphrase_arbitration_downgrades_to_warn():
    beats = {"beats": [{"group_id": 3, "narration":
                        "He regards her as his very first friend here.",
                        "scene_files": ["c.jpg"]}]}
    vitems = {"c.jpg": {"text_only": True, "ocr_clean":
              "THIS GIRL IS MY FIRST FRIEND IN THIS WORLD, BUT AT THIS "
              "MOMENT, I HAVE NO CHOICE"}}
    fl = pq.caption_unvoiced_flags(beats, vitems,
                                   arbitrate=lambda cap, narr: True)
    assert [f["code"] for f in fl] == ["caption_paraphrased"]
    assert fl[0]["severity"] == pq.WARN
    fl2 = pq.caption_unvoiced_flags(beats, vitems,
                                    arbitrate=lambda cap, narr: False)
    assert [f["code"] for f in fl2] == ["caption_unvoiced"]


def test_montage_flags_exempt_sys_doc_recurrence():
    """IE g0006-g0009: alternating SYSTEM/DOCUMENT cards is legitimate —
    they're exempt from the repeat cap and must not read as degeneracy."""
    tl = []
    for i, f in enumerate(("s.jpg", "d.jpg", "s.jpg", "d.jpg")):
        tl.append(_seg(f"g{i+6:04d}_p00", "x", [f]))
    plan = {"timeline": tl, "scene_dims": {"s.jpg": {"sys": True},
                                           "d.jpg": {"doc": True}}}
    assert pq.montage_flags(plan) == []


# ---- audio <-> narration consistency gate --------------------------------

def _idx(*pairs):
    from narration_consistency import narration_sha
    return {"clips": [{"segment_id": s, "text_sha": narration_sha(t)}
                      for s, t in pairs]}


def test_audio_flags_fresh_when_audio_matches_narration():
    plan = {"source_tts_index": "tts/tts_index.json",      # voiced plan
            "timeline": [_seg("g0001_p00", "[tense] He runs.", ["a.jpg"])]}
    assert pq.audio_flags(plan, _idx(("g0001_p00", "He runs."))) == []


def test_audio_flags_stale_when_narration_changed():
    plan = {"source_tts_index": "tts/tts_index.json",      # voiced plan
            "timeline": [_seg("g0001_p00", "He sprints away now.", ["a.jpg"])]}
    out = pq.audio_flags(plan, _idx(("g0001_p00", "He runs.")))
    assert [f["code"] for f in out] == ["audio_stale"]
    assert out[0]["severity"] == "ERROR"


def test_audio_flags_empty_index_is_not_gated():
    plan = {"timeline": [_seg("g0002_p01", "Brand new beat.", ["a.jpg"])]}
    assert pq.audio_flags(plan, _idx()) == []        # not voiced yet


def test_audio_flags_estimate_plan_ignores_leftover_clips():
    """Re-preparing a chapter that was voiced before leaves the OLD clips on
    disk with stale text, but the fresh plan is a pre-voiceover ESTIMATE (no
    source_tts_index). Those clips get re-voiced after story approval, so QA
    must NOT ERROR on them — this was failing EVERY re-prepared chapter
    (ORV/Nano/IE) with 10+ bogus audio_stale errors."""
    plan = {"timeline": [_seg("g0001_p00", "Totally new narration.", ["a.jpg"])]}
    assert pq.audio_flags(plan, _idx(("g0001_p00", "Old stale line."))) == []


def test_audio_flags_voiced_plan_with_vanished_index_errors():
    plan = {"source_tts_index": "tts/tts_index.json",
            "timeline": [_seg("g0001_p00", "Has narration.", ["a.jpg"])]}
    out = pq.audio_flags(plan, {})                    # index missing/empty
    assert [f["code"] for f in out] == ["audio_index_missing"]
    assert out[0]["severity"] == "ERROR"


def test_narration_stale_tolerates_chrome_scrub_but_catches_real_drift():
    # the script stage scrubs chrome openers; the gate must scrub the beats side
    # too, else a legitimately-scrubbed plan reads as "stale" (the IE false pos).
    groups = {"shots": [{"group_id": 1}]}
    script = {"narration_source": "gemini_verbatim"}
    plan = {"timeline": [_seg("g0001_p00", "[serious] He wakes as a baby.", ["a.jpg"])]}
    chrome = {"beats": [{"group_id": 1,
                         "narration": "Welcome to the grind of Infinite Evolution From Zero."}]}
    assert "narration_stale" not in [f["code"] for f in
        pq.alignment_flags(plan, chrome, groups, script)]
    real = {"beats": [{"group_id": 1,
                       "narration": "An unrelated paragraph about distant dragons and war."}]}
    assert "narration_stale" in [f["code"] for f in
        pq.alignment_flags(plan, real, groups, script)]


def test_audio_flags_missing_clip_for_voiced_chapter():
    plan = {"source_tts_index": "tts/tts_index.json",      # voiced plan
            "timeline": [_seg("g0001_p00", "Has audio.", ["a.jpg"]),
                         _seg("g0002_p01", "No audio yet.", ["b.jpg"])]}
    out = pq.audio_flags(plan, _idx(("g0001_p00", "Has audio.")))
    assert [f["code"] for f in out] == ["audio_missing"]
    assert out[0]["segment_id"] == "g0002_p01"


def test_audio_failed_flag_on_tts_failed_clip():
    """A clip that exhausted TTS retries ships as a silence placeholder with
    tts_failed=true and a text_sha that MATCHES the narration (so the
    staleness gate alone can't catch it) — audio_flags must ERROR on the
    tts_failed marker directly, fail-closed, instead of letting the mute
    placeholder pass QA forever."""
    from narration_consistency import narration_sha
    plan = {"source_tts_index": "tts/tts_index.json",      # voiced plan
            "timeline": [_seg("g0001_p00", "He runs.", ["a.jpg"])]}
    idx = {"clips": [{"segment_id": "g0001_p00",
                      "text_sha": narration_sha("He runs."),
                      "audio_file": "clips/g0001_p00.wav",
                      "tts_failed": True}]}
    out = pq.audio_flags(plan, idx)
    assert [f["code"] for f in out] == ["audio_failed"]
    assert out[0]["severity"] == pq.ERROR
    assert out[0]["segment_id"] == "g0001_p00"
    assert "g0001_p00.wav" in out[0]["detail"]

    # a clean index (no tts_failed clips) must not trip the new flag
    assert pq.audio_flags(plan, _idx(("g0001_p00", "He runs."))) == []


def test_audio_failed_flag_group_mode_bare_segment_id():
    """Group-mode clips are keyed by the BARE group id (g0001, no _p## panel
    suffix) — audio_flags's tts_failed loop must not assume the per-panel
    g####_p## shape when flagging a synthesis failure."""
    from narration_consistency import narration_sha
    plan = {"source_tts_index": "tts/tts_index.json",      # voiced plan
            "timeline": [_seg("g0001", "He runs.", ["a.jpg"])]}
    idx = {"clips": [{"segment_id": "g0001",
                      "text_sha": narration_sha("He runs."),
                      "audio_file": "clips/g0001.wav",
                      "tts_failed": True}]}
    out = pq.audio_flags(plan, idx)
    assert [f["code"] for f in out] == ["audio_failed"]
    assert out[0]["segment_id"] == "g0001"
    assert "g0001.wav" in out[0]["detail"]


# ---- system_coverage_flags: stamped panel_kind="system" must be shown --------

def _beats_with_scene_files(items):
    return {"beats": items}


def test_system_coverage_flags_shown_system_panel_is_clean():
    # a panel with panel_kind="system" that IS in the shown cuts → no flag
    plan = _plan([_item("g0001_p00", ["sys.jpg"])])
    beats = _beats_with_scene_files([
        {"group_id": 1, "narration": "ok", "scene_files": ["sys.jpg"]}
    ])
    vitems = {"sys.jpg": {"panel_kind": "system"}}
    fl = pq.system_coverage_flags(beats, plan, vitems)
    assert fl == []


def test_system_coverage_flags_absent_system_panel_errors():
    # a panel with panel_kind="system" NOT in the shown cuts → ERROR
    plan = _plan([_item("g0001_p00", ["other.jpg"])])
    beats = _beats_with_scene_files([
        {"group_id": 1, "narration": "ok", "scene_files": ["sys.jpg", "other.jpg"]}
    ])
    vitems = {
        "sys.jpg": {"panel_kind": "system"},
        "other.jpg": {"panel_kind": "story"},
    }
    fl = pq.system_coverage_flags(beats, plan, vitems)
    assert len(fl) == 1
    assert fl[0]["code"] == "system_card_unshown"
    assert fl[0]["severity"] == pq.ERROR
    assert "sys.jpg" in fl[0]["scene"]


def test_system_coverage_flags_caption_panel_not_flagged():
    # a caption panel folded into its neighbor has panel_kind="caption" —
    # it is intentionally absent from the plan and must NOT be flagged
    plan = _plan([_item("g0001_p00", ["story.jpg"])])
    beats = _beats_with_scene_files([
        {"group_id": 1, "narration": "ok", "scene_files": ["cap.jpg", "story.jpg"]}
    ])
    vitems = {
        "cap.jpg": {"panel_kind": "caption"},
        "story.jpg": {"panel_kind": "story"},
    }
    fl = pq.system_coverage_flags(beats, plan, vitems)
    assert fl == []


def test_system_coverage_flags_split_half_shown_does_not_false_positive():
    # Regression: vitems key is the unsplit name p044.jpg, but the plan shows
    # the _a half (p044_a.jpg).  _base_scene normalises both to p044.jpg so
    # the panel IS considered shown — no system_card_unshown ERROR.
    plan = _plan([_item("g0001_p00", ["p044_a.jpg"])])
    beats = _beats_with_scene_files([
        {"group_id": 1, "narration": "ok", "scene_files": ["p044.jpg"]}
    ])
    vitems = {"p044.jpg": {"panel_kind": "system"}}
    fl = pq.system_coverage_flags(beats, plan, vitems)
    assert fl == [], f"unexpected flags: {fl}"


def test_system_coverage_flags_absent_system_also_heuristic_fires_error():
    # Documents accepted double-report behaviour: a panel_kind=="system" panel
    # that is absent from the plan AND trips the OCR title-card heuristic will
    # produce system_card_unshown (ERROR) from system_coverage_flags.
    # The system_card_dropped WARN from story_flags may also fire (intentional
    # belt-and-suspenders, slated for Ch7 removal); we only assert the ERROR.
    plan = _plan([_item("g0001_p00", ["other.jpg"])])
    beats = _beats_with_scene_files([
        {"group_id": 1, "narration": "ok", "scene_files": ["sys.jpg", "other.jpg"]}
    ])
    vitems = {
        "sys.jpg": {"panel_kind": "system",
                    "ocr_clean": "CHAPTER 3: THE AWAKENING"},
        "other.jpg": {"panel_kind": "story"},
    }
    fl = pq.system_coverage_flags(beats, plan, vitems)
    assert any(f["code"] == "system_card_unshown" and f["severity"] == pq.ERROR
               for f in fl), f"expected system_card_unshown ERROR, got: {fl}"


# ---- manifest freshness wiring into prep_qa ---------------------------------

def test_prep_qa_emits_stale_manifest_flag_when_verify_chapter_returns_stale(
        monkeypatch):
    """When verify_chapter returns a stale_manifest issue, prep_qa._pre_flags
    must include a stale_manifest ERROR flag using the same _flag() structure
    as every other prep_qa flag.

    We mock _verify_chapter_freshness (the alias bound at import time in
    prep_qa) to return a controlled stale issue, then assert the flag appears
    in _pre_flags with the right code and severity.
    """
    stale_issue = {
        "code": "stale_manifest",
        "severity": "ERROR",
        "file": "render.plan.clean.json",
        "detail": "render.plan.clean.json is older than manifest.beats.json",
    }

    # monkeypatch the function that prep_qa imported under its own namespace
    monkeypatch.setattr(pq, "_verify_chapter_freshness",
                        lambda ep, **kw: [stale_issue])

    # Build _pre_flags the same way main() does (without running the CLI)
    ep = "/fake/ep_dir"
    freshness_issues = pq._verify_chapter_freshness(ep)
    pre_flags = [
        pq._flag(iss["code"], pq.ERROR, iss["detail"],
                 scene=iss.get("file", ""))
        for iss in freshness_issues
    ]

    assert len(pre_flags) == 1
    assert pre_flags[0]["code"] == "stale_manifest"
    assert pre_flags[0]["severity"] == pq.ERROR
    assert "manifest.beats.json" in pre_flags[0]["detail"]
    assert pre_flags[0]["scene"] == "render.plan.clean.json"


def test_stale_video_emits_warn_not_error(monkeypatch):
    """A stale_video issue returned by verify_chapter must surface as WARN,
    not ERROR — a re-prepared-but-not-yet-rendered chapter is the normal state
    and must not block the pipeline."""
    stale_issue = {
        "code": "stale_video",
        "severity": "WARN",
        "file": "render/segment_both.mp4",
        "detail": "render/segment_both.mp4 is older than render.plan.clean.json"
                  " — re-voice + re-render to match the current narration",
    }

    monkeypatch.setattr(pq, "_verify_chapter_freshness",
                        lambda ep, **kw: [stale_issue])

    ep = "/fake/ep_dir"
    freshness_issues = pq._verify_chapter_freshness(ep)

    # stale_video issues must be surfaced at WARN — never promoted to ERROR
    warn_flags = [
        pq._flag(iss["code"], pq.WARN, iss["detail"],
                 scene=iss.get("file", ""))
        for iss in freshness_issues
        if iss["severity"] == "WARN"
    ]
    error_flags = [iss for iss in freshness_issues if iss["severity"] == pq.ERROR]

    assert len(warn_flags) == 1
    assert warn_flags[0]["code"] == "stale_video"
    assert warn_flags[0]["severity"] == pq.WARN
    assert error_flags == [], f"stale_video must not be ERROR, got: {error_flags}"


# ---- page_floor_flags (cross-chapter truncated-fetch net) -------------------

def _write_stitch(d: Path, n_pages: int):
    import json
    d.mkdir(parents=True, exist_ok=True)
    chunks = [{"sources": [f"{i:03d}.webp"]} for i in range(n_pages)]
    (d / "manifest.stitch.json").write_text(json.dumps({"chunks": chunks}))


def test_page_floor_flags_warns_on_truncated_chapter(tmp_path):
    series = tmp_path / "series"
    for k in range(6):                       # 6 healthy siblings, ~20 pages each
        _write_stitch(series / f"ch{k:03d}", 20 + (k % 3))
    ep = series / "ch_short"
    _write_stitch(ep, 4)                      # 4 pages << median 20 (floor 9)
    flags = pq.page_floor_flags(str(ep))
    assert [f["code"] for f in flags] == ["low_page_count"]
    assert flags[0]["severity"] == "WARN"     # advisory, never blocks


def test_page_floor_flags_silent_for_normal_chapter(tmp_path):
    series = tmp_path / "series"
    for k in range(6):
        _write_stitch(series / f"ch{k:03d}", 20)
    ep = series / "ch_ok"
    _write_stitch(ep, 16)                      # below median but above floor (9)
    assert pq.page_floor_flags(str(ep)) == []


def test_page_floor_flags_silent_without_stable_median(tmp_path):
    series = tmp_path / "series"
    for k in range(3):                         # only 3 siblings -> no stable median
        _write_stitch(series / f"ch{k:03d}", 20)
    ep = series / "ch_short"
    _write_stitch(ep, 2)
    assert pq.page_floor_flags(str(ep)) == []


# ---- SFX-voiced + held-repeat verifier checks --------------------------------

def test_sfx_voiced_flags_catches_scream_in_voiced_text():
    script = {"sections": [
        {"section_index": 0, "tts_paragraphs_v3": ['He fell, crying "EUAACK...!! ACK!!!" loudly.']},
        {"section_index": 1, "tts_paragraphs_v3": ['The order rang out: "Kill him!"']},
    ]}
    flags = pq.sfx_voiced_flags(script)
    assert [f["code"] for f in flags] == ["sfx_voiced"]      # only the scream
    assert flags[0]["severity"] == "ERROR"


def test_sfx_voiced_flags_clean_when_no_sfx():
    script = {"sections": [{"section_index": 0,
              "tts_paragraphs_v3": ["A cold moon hung over the mountains."]}]}
    assert pq.sfx_voiced_flags(script) == []


def test_held_repeat_flags_runs():
    def _plan(files):
        return {"timeline": [{"segment_id": f"g{i}", "cuts": [{"file": f}]}
                             for i, f in enumerate(files)]}
    # 3 in a row -> WARN
    f3 = pq.held_repeat_flags(_plan(["a.jpg", "x.jpg", "x.jpg", "x.jpg", "b.jpg"]))
    assert [f["code"] for f in f3] == ["held_repeat"] and f3[0]["severity"] == "WARN"
    # 5 in a row -> ERROR (panels lost upstream)
    f5 = pq.held_repeat_flags(_plan(["x.jpg"] * 5))
    assert f5[0]["severity"] == "ERROR"
    # all distinct -> none
    assert pq.held_repeat_flags(_plan(["a.jpg", "b.jpg", "c.jpg"])) == []


def test_raw_caps_voiced_flags_agnostic():
    # reading raw all-caps OCR aloud (any manhwa) -> ERROR; paraphrase -> clean
    dump = {"sections": [{"section_index": 0, "tts_paragraphs_v3":
            ['He demanded, "WHAT MORE DO YOU WANT FROM ME?" but they refused.']}]}
    assert any(f["code"] == "raw_caps_voiced" for f in pq.raw_caps_voiced_flags(dump))
    clean = {"sections": [{"section_index": 0, "tts_paragraphs_v3":
             ["He demanded to know why they wanted his life, but they refused."]}]}
    assert pq.raw_caps_voiced_flags(clean) == []
    ok = {"sections": [{"section_index": 0, "tts_paragraphs_v3":
          ["His HP hit zero and he collapsed."]}]}
    assert pq.raw_caps_voiced_flags(ok) == []


def test_clean_short_quote_does_not_trip_raw_caps():
    # D5: a clean sentence-case quote ("I can't move.") is fine to voice and must
    # NOT trip the raw-caps OCR-dump verifier.
    ok = {"sections": [{"section_index": 0, "tts_paragraphs_v3":
          ['He gasps, "I can\'t move."']}]}
    assert pq.raw_caps_voiced_flags(ok) == []


def test_shot_description_flags_camera_prose_per_group_as_error():
    # D4: the align pad copied a panel's camera-prose description verbatim. The
    # voiced narration naming the shot/camera is an ERROR, healable per group.
    beats = {"beats": [{
        "group_id": 5,
        "panel_narration": [
            {"scene_file": "p000034.jpg",
             "line": "A close-up shot shows his trembling hands."},
            {"scene_file": "p000035.jpg", "line": "He clenches his fists."},
        ],
    }]}
    fl = pq.shot_description_flags(beats)
    assert [f["code"] for f in fl] == ["shot_description"]
    assert fl[0]["severity"] == "ERROR"
    assert fl[0]["segment_id"] == "g0005"
    assert fl[0]["scene"] == "p000034.jpg"


def test_shot_description_flags_clean_when_story_lines():
    beats = {"beats": [{"group_id": 1, "panel_narration": [
        {"scene_file": "a.jpg", "line": "He draws the blade and lunges."},
        {"scene_file": "b.jpg", "line": "The scene shifts to the throne room."},
    ]}]}
    assert pq.shot_description_flags(beats) == []


def test_filename_in_narration_flags_leak_as_healable_error():
    # the prose-first writer receives scene_file names as sentence tags — the
    # real ch1 leak: a 4-panel run voiced as "It progresses through the series
    # to conclude at p000032.jpg." The grounding judge only said WARN; this
    # deterministic net makes it an ERROR the heal loop re-narrates.
    beats = {"beats": [{
        "group_id": 7,
        "segments": [
            {"span": ["p000027.jpg"],
             "line": "The killers close in from every side."},
            {"span": ["p000028.jpg", "p000030.jpg", "p000031.jpg",
                      "p000032.jpg"],
             "line": "It progresses through the series to conclude at "
                     "p000032.jpg."},
        ],
    }]}
    fl = pq.filename_in_narration_flags(beats)
    assert [f["code"] for f in fl] == ["filename_in_narration"]
    assert fl[0]["severity"] == "ERROR"
    assert fl[0]["segment_id"] == "g0007"
    assert fl[0]["scene"] == "p000028.jpg"


def test_filename_in_narration_flags_clean_on_story_lines():
    beats = {"beats": [{"group_id": 1, "segments": [
        {"span": ["a.jpg"], "line": "He draws the blade and lunges."},
        {"span": ["b.jpg"], "line": "Steel meets steel in the dark."},
    ]}]}
    assert pq.filename_in_narration_flags(beats) == []


def test_impact_marker_leak_flags_catches_the_bracket_echo():
    # the writer's payload carries "[IMPACT SFX on panel]" as a per-panel tag
    # (tools/gemini_narrative_pass.py's _pack_group_payload) — the SAME leak
    # channel as a scene_file tag: it can get echoed back verbatim instead of
    # being converted into narration of the strike itself.
    beats = {"beats": [{
        "group_id": 9,
        "segments": [
            {"span": ["p000010.jpg"],
             "line": "He steadies himself before the fight."},
            {"span": ["p000011.jpg"],
             "line": "[IMPACT SFX on panel] as he falls."},
        ],
    }]}
    fl = pq.impact_marker_leak_flags(beats)
    assert [f["code"] for f in fl] == ["impact_marker_leak"]
    assert fl[0]["severity"] == "ERROR"
    assert fl[0]["segment_id"] == "g0009"
    assert fl[0]["scene"] == "p000011.jpg"


def test_impact_marker_leak_flags_fires_regardless_of_impact_lexicon():
    # a leaked marker is unshippable bookkeeping whether or not the line ALSO
    # happens to carry an impact-class word — has_impact_lexeme must not be
    # the only thing standing between this leak and a green QA report.
    line = "[IMPACT SFX on panel] as the blade strikes true."
    assert pq.has_impact_lexeme(line)  # carries a lexeme too...
    beats = {"beats": [{"group_id": 2, "segments": [
        {"span": ["a.jpg"], "line": line}]}]}
    fl = pq.impact_marker_leak_flags(beats)
    assert [f["code"] for f in fl] == ["impact_marker_leak"]  # ...net still fires


def test_impact_marker_leak_flags_clean_on_story_lines():
    beats = {"beats": [{"group_id": 1, "segments": [
        {"span": ["a.jpg"], "line": "He draws the blade and lunges."},
        {"span": ["b.jpg"], "line": "Steel meets steel in the dark."},
    ]}]}
    assert pq.impact_marker_leak_flags(beats) == []


def test_figures_leak_flags_catches_the_unknown_wrapper_echo():
    # the writer's payload carries "unknown (<evidence>)" for an unresolved
    # cast_identity figure (tools/gemini_narrative_pass.py's
    # _pack_group_payload) — the SAME leak channel as the impact-SFX/
    # scene_file tags: it can get echoed back verbatim instead of being
    # converted into neutral phrasing.
    beats = {"beats": [{
        "group_id": 9,
        "segments": [
            {"span": ["p000010.jpg"],
             "line": "He steadies himself before the fight."},
            {"span": ["p000011.jpg"],
             "line": "unknown (a masked figure lurking) nears the gate."},
        ],
    }]}
    fl = pq.figures_leak_flags(beats)
    assert [f["code"] for f in fl] == ["figures_leak"]
    assert fl[0]["severity"] == "ERROR"
    assert fl[0]["segment_id"] == "g0009"
    assert fl[0]["scene"] == "p000011.jpg"


def test_figures_leak_flags_silent_on_resolved_cast_names():
    # FIGURES ARE GROUND TRUTH is the point of the feature: a resolved cast
    # name in narration is sanctioned, never a leak.
    beats = {"beats": [{"group_id": 1, "segments": [
        {"span": ["a.jpg"], "line": "Prince Cheon draws his hidden blade."},
        {"span": ["b.jpg"], "line": "The assassin leader closes in."},
    ]}]}
    assert pq.figures_leak_flags(beats) == []


@pytest.mark.parametrize("leaked_line", [
    # real round-3 Nano ch1 shapes (18 segments: 15 "Dramatic:", 3 "Comic:")
    "Dramatic: He’s tumbling down a massive cliff, screaming his lungs "
    "out while plummeting into the abyss.",
    "Comic: The masked guy grabs him by the throat and asks if that was "
    "his big attempt at revenge.",
    "Dramatic He's free-falling down a rocky cliff, screaming.",  # no colon
])
def test_mood_tag_leak_flags_catches_the_real_round3_shapes(leaked_line):
    # a VOICED line opening with a bare mood/tone word is pipeline/authoring
    # vocabulary read aloud — the SAME leak channel as impact_marker_leak /
    # figures_leak, just missing its brackets.
    beats = {"beats": [{
        "group_id": 9,
        "segments": [
            {"span": ["p000010.jpg"],
             "line": "He steadies himself before the fight."},
            {"span": ["p000011.jpg"], "line": leaked_line},
        ],
    }]}
    fl = pq.mood_tag_leak_flags(beats)
    assert [f["code"] for f in fl] == ["mood_tag_leak"]
    assert fl[0]["severity"] == "ERROR"
    assert fl[0]["segment_id"] == "g0009"
    assert fl[0]["scene"] == "p000011.jpg"


def test_mood_tag_leak_flags_silent_on_bracketed_and_ordinary_lines():
    beats = {"beats": [{"group_id": 1, "segments": [
        {"span": ["a.jpg"], "line": "[dramatic] He draws his hidden blade."},
        {"span": ["b.jpg"], "line": "Dramatic reveals stay restrained."},
        {"span": ["c.jpg"], "line": "The assassin leader closes in."},
    ]}]}
    assert pq.mood_tag_leak_flags(beats) == []


def test_cut_gap_is_error_not_warn():
    # D1-backstop: a residual render-plan time-hole (black screen) must BLOCK
    # autopilot, not ship silently as a WARN.
    items = [_item("g0001_p00", ["p000001.jpg"], dur=8.0)]
    items[0]["cuts"] = [{"file": "p000001.jpg", "start": 0.0, "dur": 3.0}]  # 5s hole
    plan = _plan(items)
    plan["scene_dims"] = {"p000001.jpg": {"w": 100, "h": 100, "doc": False}}
    fl = pq.plan_flags(plan, clean_files={"p000001.jpg"},
                       audio_exists=lambda p: True)
    gaps = [f for f in fl if f["code"] == "cut_gap"]
    assert len(gaps) == 1
    assert gaps[0]["severity"] == "ERROR"


def test_flash_cut_is_blocking_error():
    # Task 3.4: a sub-1.2s on-screen cut is the loud backstop for the per-panel
    # floor (Task 3.3) — it must BLOCK (ERROR), not merely WARN.
    # _item(seg, files, dur=...) builds one cut from `files`; dur=0.3 -> the lone
    # cut is 0.3s on screen, a flash cut.
    plan = _plan([_item("g0001_p01", ["p.jpg"], dur=0.3)])
    flags = pq.plan_flags(plan, clean_files={"p.jpg"},   # suppress missing_file noise
                          audio_exists=lambda p: True)     # audio_exists is a CALLABLE
    sev = {f["code"]: f["severity"] for f in flags}
    assert sev.get("flash_cut") == pq.ERROR


def test_flash_cut_held_card_is_not_flagged():
    # a legitimately-short HELD card (mirrors the repeat_cut held guard) must not
    # false-block: the held cut is the renderer's own pacing, not a flash.
    item = _item("g0001_p01", ["p.jpg"], dur=0.3)
    item["cuts"][0]["held"] = True
    plan = _plan([item])
    flags = pq.plan_flags(plan, clean_files={"p.jpg"},
                          audio_exists=lambda p: True)
    assert not any(f["code"] == "flash_cut" for f in flags)


def test_flash_cut_threshold_is_coupled_to_two_seconds():
    # C4: a 1.5s non-held cut sits in the new 1.2-2.0 band -> must now flag
    # (it did NOT at the old 1.2 threshold), keeping floor == flash_cut threshold.
    plan = _plan([_item("g0001_p01", ["p.jpg"], dur=1.5)])
    flags = pq.plan_flags(plan, clean_files={"p.jpg"}, audio_exists=lambda p: True)
    assert any(f["code"] == "flash_cut" for f in flags)


# ---- Tracks B+C: dup_shown blocking tripwire + long_hold cap ------------------
# dup_shown: the BLOCKING QA mirror of render_prep's shown-twin invariant —
# masked-RAW hashing (rp.twin_verdict, the SAME shared predicate) over shown
# cuts, so any FUTURE bypass of the dedup ladder trips a blocking flag instead
# of shipping a duplicate. long_hold: one file held continuously past the
# [render].max_same_image_hold_sec cap AND standing in for art it doesn't own
# (the p000090 24s eye) — a genuine own-panel long hold stays legal.

def _twin_ramp(base=0, w=300, h=400):
    """Two ramps at different brightness hash IDENTICAL under 8x8 dhash."""
    row = np.linspace(0, 200, w).astype(np.uint8)
    img = np.stack([np.tile(row, (h, 1))] * 3, axis=-1)
    return np.clip(img.astype(int) + base, 0, 255).astype(np.uint8)


def _vert_ramp(w=300, h=400):
    col = np.linspace(0, 200, h).astype(np.uint8)
    return np.stack([np.tile(col.reshape(-1, 1), (1, w))] * 3,
                    axis=-1).astype(np.uint8)


def test_dup_shown_blocking_flag():
    # hand-built plan with two shown masked-twins that (hypothetically) slipped
    # every render_prep pass -> prep_qa must emit a BLOCKING ERROR.
    plan = {"timeline": [
        {"segment_id": "g0001_p00", "tts_text": "a", "duration_sec": 4.0,
         "cuts": [{"file": "twinA.jpg", "start": 0.0, "dur": 4.0}]},
        {"segment_id": "g0002_p00", "tts_text": "b", "duration_sec": 4.0,
         "cuts": [{"file": "twinB.jpg", "start": 0.0, "dur": 4.0}]},
        {"segment_id": "g0003_p00", "tts_text": "c", "duration_sec": 4.0,
         "cuts": [{"file": "other.jpg", "start": 0.0, "dur": 4.0}]},
    ]}
    raws = {"twinA.jpg": _twin_ramp(0), "twinB.jpg": _twin_ramp(20),
            "other.jpg": _vert_ramp()}
    fl = pq.dup_shown_flags(pq.iter_shown_cuts(plan),
                            lambda f: raws.get(f),
                            lambda f: [], lambda f: "")
    dup = [f for f in fl if f["code"] == "dup_shown"]
    assert dup and all(f["severity"] == pq.ERROR for f in dup)
    assert any("twinB.jpg" == f["scene"] for f in dup)
    # the distinct panel never flags; a HEALTHY plan is quiet
    assert not any(f["scene"] == "other.jpg" for f in dup)
    healthy = {"timeline": [
        {"segment_id": "g1", "tts_text": "a", "duration_sec": 4.0,
         "cuts": [{"file": "twinA.jpg", "start": 0.0, "dur": 4.0}]},
        {"segment_id": "g2", "tts_text": "b", "duration_sec": 4.0,
         "cuts": [{"file": "other.jpg", "start": 0.0, "dur": 4.0}]},
    ]}
    assert pq.dup_shown_flags(pq.iter_shown_cuts(healthy),
                              lambda f: raws.get(f),
                              lambda f: [], lambda f: "") == []
    # a SAME-FILE pair (a deliberate hold / capped repeat) is never a dup here
    held = {"timeline": [
        {"segment_id": "g1", "tts_text": "a", "duration_sec": 4.0,
         "cuts": [{"file": "twinA.jpg", "start": 0.0, "dur": 4.0}]},
        {"segment_id": "g2", "tts_text": "b", "duration_sec": 4.0,
         "cuts": [{"file": "twinA.jpg", "start": 0.0, "dur": 4.0}]},
    ]}
    assert pq.dup_shown_flags(pq.iter_shown_cuts(held),
                              lambda f: raws.get(f),
                              lambda f: [], lambda f: "") == []
    # exempt (system/doc) panels are never compared
    assert pq.dup_shown_flags(pq.iter_shown_cuts(plan),
                              lambda f: raws.get(f),
                              lambda f: [], lambda f: "",
                              is_exempt=lambda f: True) == []
    # the worker BLOCKS on it — this is the tripwire, not a cosmetic nit
    from studio.worker import _CRITICAL_QA_CODES
    assert "dup_shown" in _CRITICAL_QA_CODES


def test_long_hold_blocks_substituted_only():
    # D4 shape: p000090 held ~24s across two segments, the second of which
    # intended DIFFERENT art (p000095 was canonicalized away) -> BLOCK.
    plan = {"timeline": [
        {"segment_id": "g0019_p00", "tts_text": "a", "duration_sec": 12.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 12.0}]},
        {"segment_id": "g0020_p01", "tts_text": "b", "duration_sec": 12.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 12.0}]},
    ]}
    beats_sub = _beats([
        {"group_id": 19, "scene_files": ["p000090.jpg"]},
        {"group_id": 20, "scene_files": ["p000095.jpg"]},
    ])
    fl = pq.long_hold_flags(plan, beats_sub, max_hold_sec=10.0)
    lh = [f for f in fl if f["code"] == "long_hold"]
    assert lh and all(f["severity"] == pq.ERROR for f in lh)
    assert lh[0]["scene"] == "p000090.jpg"
    # the SAME 24s span on a panel that genuinely owns both segments' narration
    # (no substitution): ownership clears the STAND-IN clause, but 24s with NO
    # ken variation (identical/absent motions) is past the unconditional
    # STATIC ceiling (V1, 2026-07 review — the 22.8s own-panel eye was
    # unwatchable), so it still fires...
    beats_own = _beats([
        {"group_id": 19, "scene_files": ["p000090.jpg"]},
        {"group_id": 20, "scene_files": ["p000090.jpg"]},
    ])
    own = pq.long_hold_flags(plan, beats_own, max_hold_sec=10.0)
    assert [f["code"] for f in own] == ["long_hold"]
    assert "STATIC" in own[0]["detail"] and "stand" not in own[0]["detail"]
    # ...while the production shape — the same 24s run carrying the merge
    # pass's continuous kenburns slices (VARIED motions) — stays legal:
    # content-driven pacing with a moving display.
    varied = {"timeline": [
        {"segment_id": "g0019_p00", "tts_text": "a", "duration_sec": 12.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 12.0,
                   "motion": {"ease": "ease_in", "zoom": {"start": 1.0}}}]},
        {"segment_id": "g0020_p01", "tts_text": "b", "duration_sec": 12.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 12.0,
                   "motion": {"ease": "ease_out", "zoom": {"start": 1.05}}}]},
    ]}
    assert pq.long_hold_flags(varied, beats_own, max_hold_sec=10.0) == []
    # under the cap: quiet even when substituted
    short = {"timeline": [
        {"segment_id": "g0019_p00", "tts_text": "a", "duration_sec": 4.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 4.0}]},
        {"segment_id": "g0020_p01", "tts_text": "b", "duration_sec": 4.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 4.0}]},
    ]}
    assert pq.long_hold_flags(short, beats_sub, max_hold_sec=10.0) == []
    # blocking: heal can't fix a hold, so the worker must gate on it
    from studio.worker import _CRITICAL_QA_CODES
    assert "long_hold" in _CRITICAL_QA_CODES


def test_long_hold_per_cut_standin_mixed_segment():
    # ONE multi-cut segment: cut 1 genuinely owns this beat's panel (in-group,
    # brief); cut 2 shows a DIFFERENT group's panel -- as a cross-group fold
    # would land it -- and is held long on its own. long_hold's stand-in test
    # is PER CUT, so cut 2 alone trips it even though its sibling cut 1 is
    # legitimate. panel_substituted (story_flags) is PER SEGMENT: it unions
    # every cut's file in the item, and cut 1 keeps that union intersecting
    # the beat's intended panels, so it stays quiet. This pins the two checks'
    # intentional difference in granularity -- not a bug in either.
    plan = {"timeline": [
        {"segment_id": "g0061_p00", "tts_text": "ok", "duration_sec": 13.0,
         "cuts": [{"file": "p000200.jpg", "start": 0.0, "dur": 1.0},
                  {"file": "p000201.jpg", "start": 1.0, "dur": 12.0}]},
    ]}
    beats = _beats([
        {"group_id": 61, "scene_files": ["p000200.jpg"]},
        {"group_id": 62, "scene_files": ["p000201.jpg"]},
    ])
    lh = [f for f in pq.long_hold_flags(plan, beats, max_hold_sec=10.0)
          if f["code"] == "long_hold"]
    assert lh and lh[0]["scene"] == "p000201.jpg"
    assert lh[0]["segment_id"] == "g0061_p00"
    # per-segment check sees cut 1's genuine art and stays quiet
    sub = [f for f in pq.story_flags(plan, beats, {})
           if f["code"] == "panel_substituted"]
    assert sub == []


def test_truncated_line_flags_mid_sentence_stop_as_healable_error():
    # 2026-07-06 review class C tripwire — the REAL g0011_p16 line verbatim
    beats = {"beats": [{"group_id": 11, "segments": [
        {"span": ["p000052.jpg"],
         "line": "But there is no mercy to be found, only the"},
        {"span": ["p000053.jpg"], "line": "The clearing goes silent."},
    ]}]}
    fl = pq.truncated_line_flags(beats)
    assert [f["code"] for f in fl] == ["truncated_line"]
    assert fl[0]["severity"] == "ERROR"
    assert fl[0]["segment_id"] == "g0011"
    assert fl[0]["scene"] == "p000052.jpg"


def test_truncated_line_flags_quiet_on_terminal_lines():
    beats = {"beats": [{"group_id": 1, "segments": [
        {"span": ["a.jpg"], "line": "Seriously, what even is that light?!"},
        {"span": ["b.jpg"], "line": "He can only mutter, 'Ancestor...?'"},
        {"span": ["c.jpg"], "line": "A pause hangs in the air…"},
    ]}]}
    assert pq.truncated_line_flags(beats) == []
    from tools.narration_heal import HEALABLE
    assert "truncated_line" in HEALABLE
    import studio.worker as worker
    assert "truncated_line" not in worker._CRITICAL_QA_CODES


def test_display_meta_and_camera_pov_fire_shot_description_flags():
    # 2026-07-06 review classes B + D reach the healable ERROR through the
    # same single-authority detector
    beats = {"beats": [{"group_id": 24, "segments": [
        {"span": ["p000109.jpg"],
         "line": "The text is displayed as a standalone caption."},
        {"span": ["p000111.jpg"],
         "line": "The text is displayed as a title or organizational name "
                 "card."},
    ]}, {"group_id": 18, "segments": [
        {"span": ["p000087.jpg"],
         "line": "An electrified hand reaches out toward the viewer."},
    ]}]}
    fl = pq.shot_description_flags(beats)
    assert [f["segment_id"] for f in fl] == ["g0024", "g0024", "g0018"]
    assert all(f["code"] == "shot_description" for f in fl)


# ---------------------------------------------------------------------------
# Wave-A minor #4 (2026-07-07): real-manifest sweep regression. The vendored
# 73-segment Nano ch1 beats manifest (tests/fixtures/nano_ch1_beats.json) is
# the exact evidence the 2026-07-06 review scored by hand. The display-meta/
# camera-POV detector (shot_description_flags) and the truncation detector
# (truncated_line_flags) must fire on EXACTLY the 5 reviewer-confirmed lines
# and nothing else -- zero collateral across the other 68 -- pinned by
# segment id so a future regression names the segment that broke instead of
# just a changed count.
# ---------------------------------------------------------------------------
_NANO_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nano_ch1_beats.json"
_NANO_EXPECTED_PINS = frozenset({
    "g0004_p01", "g0011_p03", "g0018_p03", "g0024_p03", "g0024_p05",
})


def test_nano_ch1_sweep_flags_exactly_five_pinned_segments():
    from tools.beats_segments import beat_segments
    beats = json.loads(_NANO_FIXTURE.read_text())

    pin_by_scene = {}
    total = 0
    for b in beats.get("beats") or []:
        gid = int(b.get("group_id") or 0)
        for i, s in enumerate(beat_segments(b)):
            total += 1
            head = str((s.get("span") or [""])[0])
            pin_by_scene[head] = f"g{gid:04d}_p{i:02d}"
    assert total == 73, "fixture must be the full 73-segment evidence manifest"

    hits = pq.shot_description_flags(beats) + pq.truncated_line_flags(beats)
    assert len(hits) == 5, "collateral or a missed hit -- see pinned ids"
    pins = {pin_by_scene.get(f["scene"]) for f in hits}
    assert pins == _NANO_EXPECTED_PINS


# ---- V1 tripwire (2026-07 review): long_hold unconditional STATIC ceiling ----
# Own-panel ownership stays exempt from the STAND-IN clause, but one file
# continuously STATIC (single cut / identical motions — no ken variation) past
# 1.5x the cap is unwatchable regardless (the 22.8s g0020_p01 eye). With
# render_prep.split_long_hold_cuts in place this should never fire — tripwire.

def test_long_hold_unconditional_static_ceiling():
    beats_own = _beats([{"group_id": 20, "scene_files": ["p000095.jpg"]}])
    static20 = {"timeline": [
        {"segment_id": "g0020_p01", "tts_text": "b", "duration_sec": 20.0,
         "cuts": [{"file": "p000095.jpg", "start": 0.0, "dur": 20.0,
                   "motion": {"mode": "tilt_down"}}]},
    ]}
    fl = pq.long_hold_flags(static20, beats_own, max_hold_sec=10.0)
    assert [f["code"] for f in fl] == ["long_hold"]
    assert fl[0]["severity"] == pq.ERROR
    assert fl[0]["scene"] == "p000095.jpg"
    assert "STATIC" in fl[0]["detail"]
    # same blocking membership as the stand-in clause (same code)
    from studio.worker import _CRITICAL_QA_CODES
    assert "long_hold" in _CRITICAL_QA_CODES
    # the ken-varied split render_prep emits for the same display: silent
    varied = {"timeline": [
        {"segment_id": "g0020_p01", "tts_text": "b", "duration_sec": 20.0,
         "cuts": [
             {"file": "p000095.jpg", "start": 0.0, "dur": 7.0,
              "motion": {"ken_region": "wide"}, "ken_variety": True},
             {"file": "p000095.jpg", "start": 7.0, "dur": 7.0,
              "motion": {"ken_region": "tight"}, "ken_variety": True},
             {"file": "p000095.jpg", "start": 14.0, "dur": 6.0,
              "motion": {"ken_region": "pull"}, "ken_variety": True},
         ]},
    ]}
    assert pq.long_hold_flags(varied, beats_own, max_hold_sec=10.0) == []
    # own-panel display over the cap but UNDER the ceiling: content-driven
    # pacing, still legal
    static12 = {"timeline": [
        {"segment_id": "g0020_p01", "tts_text": "b", "duration_sec": 12.0,
         "cuts": [{"file": "p000095.jpg", "start": 0.0, "dur": 12.0}]},
    ]}
    assert pq.long_hold_flags(static12, beats_own, max_hold_sec=10.0) == []
    # exempt files (wide/tall renderer drift, doc/system stillness) never fire
    assert pq.long_hold_flags(static20, beats_own, max_hold_sec=10.0,
                              is_exempt=lambda f: True) == []
    # the STAND-IN clause outranks the ceiling (one flag per run, stand-in
    # wording — heal routing depends on it)
    beats_sub = _beats([{"group_id": 20, "scene_files": ["pXXX.jpg"]}])
    fl2 = pq.long_hold_flags(static20, beats_sub, max_hold_sec=10.0)
    assert len(fl2) == 1 and "standing in" in fl2[0]["detail"]


def test_long_hold_ceiling_identical_motion_run_is_static():
    # a multi-cut same-file run with IDENTICAL motion dicts is still "no ken
    # variation" (a restarting loop); the merge pass's continuous slices
    # (different f0/f1 per cut) are variation and stay legal
    beats_own = _beats([
        {"group_id": 19, "scene_files": ["p000090.jpg"]},
        {"group_id": 20, "scene_files": ["p000090.jpg"]},
    ])
    same_motion = {"mode": "kenburns", "zoom": {"start": 1.0, "end": 1.1}}
    frozen = {"timeline": [
        {"segment_id": "g0019_p00", "tts_text": "a", "duration_sec": 8.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 8.0,
                   "motion": dict(same_motion)}]},
        {"segment_id": "g0020_p01", "tts_text": "b", "duration_sec": 8.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 8.0,
                   "motion": dict(same_motion)}]},
    ]}
    fl = pq.long_hold_flags(frozen, beats_own, max_hold_sec=10.0)
    assert [f["code"] for f in fl] == ["long_hold"]
    assert "STATIC" in fl[0]["detail"]
    sliced = {"timeline": [
        {"segment_id": "g0019_p00", "tts_text": "a", "duration_sec": 8.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 8.0,
                   "motion": {"ease": "ease_in",
                              "zoom": {"start": 1.0, "end": 1.05}}}]},
        {"segment_id": "g0020_p01", "tts_text": "b", "duration_sec": 8.0,
         "cuts": [{"file": "p000090.jpg", "start": 0.0, "dur": 8.0,
                   "motion": {"ease": "ease_out",
                              "zoom": {"start": 1.05, "end": 1.1}}}]},
    ]}
    assert pq.long_hold_flags(sliced, beats_own, max_hold_sec=10.0) == []


def test_held_repeat_and_repeat_cut_exempt_ken_variety_subcuts():
    # V1 sub-cuts deliberately repeat one file with DIFFERENT ken regions —
    # neither a frozen/looping repeat (held_repeat) nor an accidental
    # consecutive repeat (repeat_cut)
    sub = [{"file": "p000095.jpg", "start": 0.0, "dur": 7.0,
            "motion": {"ken_region": "wide"}, "ken_variety": True},
           {"file": "p000095.jpg", "start": 7.0, "dur": 7.0,
            "motion": {"ken_region": "tight"}, "ken_variety": True},
           {"file": "p000095.jpg", "start": 14.0, "dur": 6.0,
            "motion": {"ken_region": "pull"}, "ken_variety": True}]
    plan = {"timeline": [{"segment_id": "g0020_p01", "tts_text": "b",
                          "duration_sec": 20.0, "cuts": sub}],
            "scene_dims": {"p000095.jpg": {"w": 795, "h": 832,
                                           "doc": False}}}
    assert pq.held_repeat_flags(plan) == []
    fl = pq.plan_flags(plan, clean_files={"p000095.jpg"},
                       audio_exists=lambda p: True)
    assert "repeat_cut" not in [f["code"] for f in fl]
    # sanity contrast: the SAME shape without the marker still trips both
    bare = {"timeline": [{"segment_id": "g0020_p01", "tts_text": "b",
                          "duration_sec": 20.0,
                          "cuts": [{k: v for k, v in c.items()
                                    if k != "ken_variety"} for c in sub]}],
            "scene_dims": plan["scene_dims"]}
    assert [f["code"] for f in pq.held_repeat_flags(bare)] == ["held_repeat"]
    fl2 = pq.plan_flags(bare, clean_files={"p000095.jpg"},
                        audio_exists=lambda p: True)
    assert "repeat_cut" in [f["code"] for f in fl2]


# ---- V2 echo net (2026-07 review): perceptual_echo WARN, measure-first -------
# Shown-crop bubble-masked twins whose RAW panels are DISTINCT — the class the
# masked-RAW invariant (dup_shown) correctly keeps as separate panels but a
# viewer reads as the same picture twice. The p000090/p000095 eye-husk pair is
# the incident: p000095's SHIPPED crop hash-twinned p000090 (production dhash
# 3) while the raws measure masked ham 22. The fixture JPGs are downscaled q65
# re-encodes, so the shipped crop is reconstructed from p000090's art region;
# the RAW predicate runs on the real fixture panels.

_ECHO_FIX = Path(__file__).resolve().parent / "fixtures" / "dedup"
_P95_RAW_BOXES = [(155, 276, 389, 447)]     # p000095's blank cleaned bubble


def _echo_imgs():
    p90 = cv2.imread(str(_ECHO_FIX / "p000090.jpg"))
    p95 = cv2.imread(str(_ECHO_FIX / "p000095.jpg"))
    p54 = cv2.imread(str(_ECHO_FIX / "p000054.jpg"))
    assert p90 is not None and p95 is not None and p54 is not None
    return p90, p95, p54


def _cut(seg, f, dur=4.0, branding=False):
    return {"segment_id": seg, "file": f, "idx": 0, "dur": dur,
            "branding": branding}


def test_perceptual_echo_flags_p90_p95_pair_with_both_hams():
    p90, p95, p54 = _echo_imgs()
    crop95 = p90[5:225, 5:395]              # reconstructed shipped husk crop
    cuts = [_cut("g0019_p00", "p000090.jpg"),
            _cut("g0019_p01", "p000054.jpg"),     # distinct panel between
            _cut("g0020_p01", "p000095.jpg")]
    clean = {"p000090.jpg": p90, "p000054.jpg": p54, "p000095.jpg": crop95}
    raw = {"p000090.jpg": p90, "p000054.jpg": p54, "p000095.jpg": p95}
    rb = {"p000095.jpg": _P95_RAW_BOXES}
    fl = pq.perceptual_echo_flags(cuts, lambda f: clean.get(f),
                                  lambda f: [], lambda f: raw.get(f),
                                  lambda f: rb.get(f, []))
    assert [f["code"] for f in fl] == ["perceptual_echo"]
    assert fl[0]["severity"] == pq.WARN                    # measure-first
    assert fl[0]["scene"] == "p000095.jpg"
    assert fl[0]["segment_id"] == "g0020_p01"
    # both ham values in the detail (shown twin, raw distinct)
    assert "ham=" in fl[0]["detail"] and "raw masked ham=" in fl[0]["detail"]
    # genuinely different panels never flag (p54 pairs are silent)
    assert not any(f["scene"] == "p000054.jpg" for f in fl)
    # NOT blocking: never in the worker's critical gate
    from studio.worker import _CRITICAL_QA_CODES
    assert "perceptual_echo" not in _CRITICAL_QA_CODES


def test_perceptual_echo_skips_raw_twins_window_exempt_and_rawless():
    p90, p95, _p54 = _echo_imgs()
    crop95 = p90[5:225, 5:395]
    clean = {"p000090.jpg": p90, "p000095.jpg": crop95}
    raw = {"p000090.jpg": p90, "p000095.jpg": p95}
    rb = {"p000095.jpg": _P95_RAW_BOXES}
    pair = [_cut("g1", "p000090.jpg"), _cut("g2", "p000095.jpg")]
    # crop twins whose RAWS are ALSO twins: dup_shown's (blocking) domain
    raw_twin = {"p000090.jpg": p90, "p000095.jpg": p90}
    assert pq.perceptual_echo_flags(pair, lambda f: clean.get(f),
                                    lambda f: [],
                                    lambda f: raw_twin.get(f),
                                    lambda f: []) == []
    # out of the 3-cut window: a far-apart recurrence is out of scope
    far = ([pair[0]]
           + [_cut(f"x{i}", f"z{i}.jpg") for i in range(3)]
           + [pair[1]])
    zs = {f"z{i}.jpg": np.full((60, 60, 3), 10 * i + 5, np.uint8)
          for i in range(3)}
    assert pq.perceptual_echo_flags(far,
                                    lambda f: clean.get(f) if f in clean
                                    else zs.get(f),
                                    lambda f: [],
                                    lambda f: raw.get(f) if f in raw
                                    else zs.get(f),
                                    lambda f: rb.get(f, [])) == []
    # exempt (system/doc) files never compared
    assert pq.perceptual_echo_flags(pair, lambda f: clean.get(f),
                                    lambda f: [], lambda f: raw.get(f),
                                    lambda f: rb.get(f, []),
                                    is_exempt=lambda f: True) == []
    # the REAL wired exemption is STAMPED-only: scene_dims' pixel-level
    # sys:True (system-box YOLO overfire — all five incident evidence panels
    # carried it) must NOT self-exempt the pair; it STILL flags
    dims_sys = {"p000090.jpg": {"w": 795, "h": 832, "sys": True},
                "p000095.jpg": {"w": 795, "h": 832, "sys": True}}
    fl = pq.perceptual_echo_flags(pair, lambda f: clean.get(f),
                                  lambda f: [], lambda f: raw.get(f),
                                  lambda f: rb.get(f, []),
                                  is_exempt=pq.echo_exempt_fn(dims_sys, {}))
    assert [f["scene"] for f in fl] == ["p000095.jpg"]
    # a STAMPED panel_kind=='system' record DOES exempt (shared UI frames)
    vit_sys = {"p000095.jpg": {"panel_kind": "system"}}
    assert pq.perceptual_echo_flags(
        pair, lambda f: clean.get(f), lambda f: [], lambda f: raw.get(f),
        lambda f: rb.get(f, []),
        is_exempt=pq.echo_exempt_fn(dims_sys, vit_sys)) == []
    # no raw scene image (split halves): raw-distinctness unprovable -> skip
    assert pq.perceptual_echo_flags(pair, lambda f: clean.get(f),
                                    lambda f: [], lambda f: None,
                                    lambda f: []) == []
    # same-file pair (a hold / ken-variety sub-cuts): never an echo
    hold = [_cut("g1", "p000090.jpg"), _cut("g2", "p000090.jpg")]
    assert pq.perceptual_echo_flags(hold, lambda f: clean.get(f),
                                    lambda f: [], lambda f: raw.get(f),
                                    lambda f: rb.get(f, [])) == []
    # branding breaks nothing but is never a member
    br = [pair[0], _cut("intro", "intro.jpg", branding=True), pair[1]]
    fl = pq.perceptual_echo_flags(br, lambda f: clean.get(f),
                                  lambda f: [], lambda f: raw.get(f),
                                  lambda f: rb.get(f, []))
    assert [f["scene"] for f in fl] == ["p000095.jpg"]


# ---------------------------------------------------------------------------
# line_overlong (span word-budget net, 2026-07-16)
# ---------------------------------------------------------------------------
def test_line_overlong_flags_catch_the_escaped_fat_line():
    # the nano g0011 class: a ~55-word single-panel line that escaped the
    # writer validator via its fallback path -> a 21s hold -> triple ken split
    fat = " ".join(["word"] * 55)
    beats = {"beats": [{"group_id": 11, "segments": [
        {"span": ["p000031.jpg"], "line": fat}]}]}
    fl = pq.line_overlong_flags(beats)
    assert [f["code"] for f in fl] == ["line_overlong"]
    assert fl[0]["severity"] == "ERROR"
    assert fl[0]["segment_id"] == "g0011"
    assert fl[0]["scene"] == "p000031.jpg"
    assert "words" in fl[0]["detail"]


def test_line_overlong_budget_scales_with_span_size():
    # 55 words over a 2-panel span is ~24s against a ~30s cap — fine
    fat = " ".join(["word"] * 55)
    beats = {"beats": [{"group_id": 3, "segments": [
        {"span": ["a.jpg", "b.jpg"], "line": fat}]}]}
    assert pq.line_overlong_flags(beats) == []


def test_line_overlong_clean_on_normal_lines():
    beats = {"beats": [{"group_id": 1, "segments": [
        {"span": ["a.jpg"], "line": "He draws the blade and lunges."}]}]}
    assert pq.line_overlong_flags(beats) == []


# ---- 2026-08-18: caption coverage tolerates morphology + OCR mis-scans ------
# ORV Ep1 g0001: the caption IS voiced ("three ways to survive the apocalypse",
# "swipes through the pages", "the text fades") but literal token matching
# missed apocalypsh/apocalypse, swipe/swipes, fade/fades and pure mis-scans.

def test_caption_coverage_counts_morphology_and_ocr_misscans():
    beats = {"beats": [{"group_id": 1, "scene_files": ["c.jpg"],
                        "narration": "There are three ways to survive the apocalypse, "
                                     "and as he swipes the page the text fades away."}]}
    vitems = {"c.jpg": {"text_only": True,
                        "ocr_clean": "THERE ARE THREE WAYS TO SURVIVE THE APOCALYPSH "
                                     "SWIPE THE PAGE THE TEXT FADE"}}
    assert pq.caption_unvoiced_flags(beats, vitems) == []


def test_caption_unvoiced_still_fires_on_a_genuinely_dropped_caption():
    beats = {"beats": [{"group_id": 1, "scene_files": ["c.jpg"],
                        "narration": "He stares at the ruined skyline in silence."}]}
    vitems = {"c.jpg": {"text_only": True,
                        "ocr_clean": "MY MOTHER SOLD THE HOUSE AND MOVED TO BUSAN "
                                     "WITHOUT TELLING ANYONE THAT WINTER"}}
    out = pq.caption_unvoiced_flags(beats, vitems)
    assert len(out) == 1 and out[0]["code"] == "caption_unvoiced"
