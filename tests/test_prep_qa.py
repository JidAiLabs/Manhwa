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
    # system-message cards (Sky Corporation / activation captions) keep their
    # text BY DESIGN — no visible_text/ghost/binary_card/dead_box/husk flags
    img, box = _bubble_panel(visible_text=True, ghost=True)
    flags = pq.image_flags("p000114.jpg", img, [box], doc=False,
                           dims_entry={"w": 300, "h": 400, "doc": False},
                           sys=True)
    assert flags == []


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
