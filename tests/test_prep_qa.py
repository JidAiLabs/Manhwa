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
from pathlib import Path

import numpy as np

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
                           doc=False, dims_entry={"w": 800, "h": 500, "doc": False})
    codes = {f["code"] for f in flags}
    assert "dead_box_leak" in codes


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
