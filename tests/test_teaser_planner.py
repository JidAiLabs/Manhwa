"""
tests/test_teaser_planner.py

Unit tests for tools/teaser_planner.py — the bundle-level arc-teaser planner.

Chunk 2 scope: the deterministic, $0, pure Stage-1 scoring layer:
  - eligible_panels  (panel eligibility / flattening)
  - score_window     (signal scoring of one contiguous window)
  - score_windows    (spoiler guard + window enumeration + non-overlapping shortlist)

tools/ is not a package, so the module is loaded by path (the repo's standard
tool-test idiom).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "teaser_planner",
    Path(__file__).resolve().parent.parent / "tools" / "teaser_planner.py",
)
tp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tp)  # type: ignore[union-attr]


# ---------------------------------------------------------------- Task 3
def test_eligible_panels_skips_chrome_empty_error():
    panels = [
        {"scene_file": "a", "panel_kind": "story", "intensity": "calm"},
        {"scene_file": "b", "panel_kind": "chrome", "intensity": "calm"},
        {"scene_file": "c", "panel_kind": "empty", "intensity": "calm"},
        {"scene_file": "d", "panel_kind": "system", "intensity": "tense"},
        {"scene_file": "e", "panel_kind": "story", "error": "parse_failed"},
    ]
    out = tp.eligible_panels(panels)
    assert [p["scene_file"] for p in out] == ["a", "d"]


# ---------------------------------------------------------------- Task 4
def test_score_window_high_stakes_beats_calm():
    hot = [{"scene_file": f"h{i}", "panel_kind": "story", "intensity": "explosive",
            "description": "the entrance exam begins", "action": "a clan heir humiliates him",
            "dialogue": "you have no badge", "subjects": ["heir", "prince"]} for i in range(5)]
    calm = [{"scene_file": f"c{i}", "panel_kind": "story", "intensity": "calm",
             "description": "they eat lunch quietly", "action": "", "dialogue": "",
             "subjects": ["prince"]} for i in range(5)]
    assert tp.score_window(hot)["score"] > tp.score_window(calm)["score"]


def test_score_panel_power_reveal_outscores_calm_exposition():
    hot = {"scene_file": "h.jpg", "panel_kind": "story", "intensity": "explosive",
           "description": "the nano core activates, energy surging and aura glowing",
           "action": "", "dialogue": "", "subjects": []}
    calm = {"scene_file": "c.jpg", "panel_kind": "story", "intensity": "calm",
            "description": "they share a quiet meal and chat about the weather",
            "action": "", "dialogue": "", "subjects": []}
    sh, sc = tp.score_panel(hot), tp.score_panel(calm)
    # the transformation panel carries the power_reveal signal; calm exposition has none
    assert sh["power_reveal"] > sc["power_reveal"]
    assert sc["power_reveal"] == 0
    assert sh["score"] > sc["score"]


# ---------------------------------------------------------------- Task 5 (montage)
def _mk_panel(scene, ch, *, kind="story", intensity="tense", desc="", subjects=None):
    return {"scene_file": scene, "chapter_number": ch, "panel_kind": kind,
            "intensity": intensity, "description": desc, "action": "",
            "dialogue": "", "subjects": subjects or []}


def test_select_montage_climaxes_on_power_reveal_and_spans_chapters():
    descs = ["the entrance exam begins", "a clan heir mocks the outcast",
             "the survival trial turns deadly", "the duel escalates"]
    panels = []
    for ch in (1, 2):                                   # earlier setup chapters
        for i, d in enumerate(descs):
            panels.append(_mk_panel(f"ch{ch}_s{i}.jpg", ch, intensity="tense", desc=d))
    for i in range(3):                                  # calm filler in last chapter
        panels.append(_mk_panel(f"ch3_f{i}.jpg", 3, intensity="calm",
                                desc="a quiet aftermath in the courtyard"))
    # the genre-defining climax lives in the LAST chapter
    panels.append(_mk_panel("ch3_climax.jpg", 3, intensity="explosive",
                            desc="the nano core activates, energy surging and aura glowing"))

    montage = tp.select_montage(panels, max_panels=6, min_panels=4, payoff_tail_frac=0.0)
    assert montage is not None
    assert len(montage) <= 6
    # the power/transformation reveal is the LAST montage element
    assert montage[-1]["scene_file"] == "ch3_climax.jpg"
    assert montage[-1]["is_climax"] is True
    assert all(p["is_climax"] is False for p in montage[:-1])
    # the montage spans more than one chapter (an arc, not one chapter)
    assert len({p["chapter_number"] for p in montage}) > 1


def test_select_montage_calm_bundle_still_returns_something():
    panels = [_mk_panel(f"s{i}.jpg", (i // 3) + 1, intensity="calm",
                        desc="they share a quiet meal") for i in range(9)]
    montage = tp.select_montage(panels, max_panels=5, min_panels=4)
    assert montage is not None
    assert 4 <= len(montage) <= 5
    assert montage[-1]["is_climax"] is True


def test_select_montage_none_when_too_few_eligible():
    panels = [_mk_panel("s0.jpg", 1, intensity="calm", desc="x")]
    assert tp.select_montage(panels, max_panels=10, min_panels=4) is None


def test_select_montage_payoff_tail_trims_the_end():
    # a climax-worthy panel sits in the trimmed tail -> not pulled into the montage
    panels = [_mk_panel(f"s{i}.jpg", 1, intensity="tense", desc="the duel rages on")
              for i in range(8)]
    panels.append(_mk_panel("late.jpg", 2, intensity="explosive",
                            desc="the nano core activates, energy surging, aura glowing"))
    montage = tp.select_montage(panels, max_panels=4, min_panels=2, payoff_tail_frac=0.5)
    assert montage is not None
    assert all(p["scene_file"] != "late.jpg" for p in montage)


# ---------------------------------------------------------------- Task 6
def test_select_and_write_builds_teaser_manifest(tmp_path):
    montage = [{"chapter_number": 5, "scene_file": "/abs/ch5/scenes/scene_0007.jpg",
                "panel_kind": "story", "intensity": "explosive", "is_climax": True,
                "description": "the nano core activates", "action": "energy surges",
                "dialogue": "you have no badge", "subjects": ["heir", "prince"]}]

    def stub(payload):
        # the montage is already selected — payload carries panels + climax_index,
        # NOT a windows shortlist, and there is no chosen_index to return.
        assert "panels" in payload and "loglines" in payload
        assert payload["climax_index"] == 0
        assert payload["panels"][-1]["is_climax"] is True
        return {"panel_narration": [{"scene_file": "scene_0007.jpg", "line": "The power awakens."}],
                "rewind_line": "But to see how he got here, we go back to the start.",
                "reason": "the power reveal", "spoiler_boundary": "shows the awakening only"}

    out = tp.select_and_write(montage, loglines=["a hunted prince"], model_call=stub)
    assert out["rewind_line"].startswith("But to see")
    assert out["source_chapters"] == [5]
    # scene_file is the namespaced id; source path resolves per-panel
    assert out["panel_narration"][0]["scene_file"] == "ch5__scene_0007.jpg"
    assert out["scene_files"] == ["ch5__scene_0007.jpg"]
    assert out["panel_sources"]["ch5__scene_0007.jpg"].endswith(
        "ch5/scenes/scene_0007.jpg")


def test_select_and_write_preserves_climax_last_order():
    montage = [
        {"chapter_number": 5, "scene_file": "/abs/ch5/scenes/a.jpg",
         "panel_kind": "story", "intensity": "tense", "description": "the exam begins",
         "action": "", "dialogue": "", "subjects": []},
        {"chapter_number": 8, "scene_file": "/abs/ch8/scenes/z.jpg", "is_climax": True,
         "panel_kind": "story", "intensity": "explosive",
         "description": "the nano core activates", "action": "", "dialogue": "", "subjects": []},
    ]

    def stub(payload):
        return {"panel_narration": [{"scene_file": "a.jpg", "line": "L1"},
                                    {"scene_file": "z.jpg", "line": "L2"}],
                "rewind_line": "back to the start", "reason": "r", "spoiler_boundary": "s"}

    out = tp.select_and_write(montage, loglines=[], model_call=stub)
    # the climax panel (chapter 8) stays LAST; order is montage order, not chronology
    assert out["scene_files"] == ["ch5__a.jpg", "ch8__z.jpg"]
    assert out["source_chapters"] == [5, 8]
    assert out["panel_narration"][-1]["scene_file"] == "ch8__z.jpg"


def test_select_and_write_none_on_empty_montage():
    assert tp.select_and_write([], loglines=[], model_call=lambda p: {}) is None


# ---------------------------------------------------------------- Task 7
def test_materialize_teaser_dir(tmp_path):
    # a fake source chapter with one scene + a scenes manifest entry
    src = tmp_path / "ch5"
    (src / "scenes").mkdir(parents=True)
    (src / "scenes" / "scene_0007.jpg").write_bytes(b"\xff\xd8\xff")  # tiny jpg stub
    (src / "manifest.scenes.json").write_text(json.dumps(
        {"scenes": [{"out_file": "scene_0007.jpg", "box_px_xyxy": [0, 0, 100, 200],
                     "chunk_global_y0": 0, "w": 100, "h": 200}]}))
    ns = "ch5__scene_0007.jpg"
    src_abs = str(src / "scenes" / "scene_0007.jpg")
    teaser = {"source_chapters": [5], "scene_files": [ns],
              "panel_sources": {ns: src_abs},
              "panel_narration": [{"scene_file": ns, "line": "The exam begins."}],
              "rewind_line": "...", "reason": "...", "spoiler_boundary": "..."}
    out_dir = tmp_path / "teaser"
    tp.materialize_teaser_dir(teaser, out_dir, cast={"cast": []})
    assert (out_dir / "scenes" / ns).exists()                         # symlink/copy
    beats = json.loads((out_dir / "manifest.beats.json").read_text())
    assert beats["beats"][0]["panel_narration"][0]["line"] == "The exam begins."
    assert beats["beats"][0]["scene_files"] == [ns]
    groups = json.loads((out_dir / "manifest.groups.json").read_text())
    assert groups["shots"][0]["scene_files"] == [ns]
    scenes = json.loads((out_dir / "manifest.scenes.json").read_text())
    assert scenes["scenes"][0]["out_file"] == ns
    assert (out_dir / "manifest.cast.json").exists()


# ------------------------------------------------ regression: basename collision
def test_teaser_namespaces_scenes_by_chapter_no_basename_collision(tmp_path):
    """Two chapters share an identical scene basename (the chunk index restarts
    every chapter) but live in different ep_dirs with different bytes + different
    scenes manifests. The teaser must materialize the CHOSEN chapter's art — not
    whichever chapter happens to be last — by carrying chapter identity end-to-end
    via a namespaced scene id. Regression for the basename-collision bug.
    """
    base = "c0000_p0003_00.jpg"
    chA = tmp_path / "Ch_005"
    chB = tmp_path / "Ch_012"
    for ch, payload_bytes, mtag in ((chA, b"AAAA", "A"), (chB, b"BBBB", "B")):
        (ch / "scenes").mkdir(parents=True)
        (ch / "scenes" / base).write_bytes(payload_bytes)
        (ch / "manifest.scenes.json").write_text(json.dumps(
            {"scenes": [{"out_file": base, "box_px_xyxy": [0, 0, 100, 200],
                         "chunk_global_y0": 0, "w": 100, "h": 200, "tag": mtag}]}))

    # a montage built from chapter A's panel, exactly as load_bundle_panels emits:
    # abs scene_file + chapter_number, climax flagged.
    montage = [{"chapter_number": 5,
                "scene_file": str(chA / "scenes" / base), "is_climax": True,
                "panel_kind": "story", "intensity": "explosive",
                "description": "the nano core activates", "action": "energy surges",
                "dialogue": "", "subjects": []}]

    def stub(payload):
        # model echoes ONE line keyed by the (now colliding) basename; the planner
        # must align by ORDER, not by this basename.
        return {"panel_narration": [{"scene_file": base, "line": "The exam begins."}],
                "rewind_line": "But to see how he got here, we go back.",
                "reason": "r", "spoiler_boundary": "s"}

    teaser = tp.select_and_write(montage, loglines=["x"], model_call=stub)
    assert teaser is not None
    out_dir = tmp_path / "teaser"
    tp.materialize_teaser_dir(teaser, out_dir, cast={"cast": []})

    ns = f"ch5__{base}"
    materialized = out_dir / "scenes" / ns
    # the materialized scene must resolve to chapter A's bytes, NOT chapter B's
    assert materialized.exists()
    assert materialized.read_bytes() == b"AAAA"
    # the namespaced id appears consistently across EVERY manifest
    beats = json.loads((out_dir / "manifest.beats.json").read_text())
    assert beats["beats"][0]["scene_files"] == [ns]
    assert beats["beats"][0]["panel_narration"][0]["scene_file"] == ns
    groups = json.loads((out_dir / "manifest.groups.json").read_text())
    assert groups["shots"][0]["scene_files"] == [ns]
    scenes = json.loads((out_dir / "manifest.scenes.json").read_text())
    assert scenes["scenes"][0]["out_file"] == ns
    assert scenes["scenes"][0]["tag"] == "A"        # copied from chapter A's manifest
    teaser_man = json.loads((out_dir / "manifest.teaser.json").read_text())
    assert teaser_man["scene_files"] == [ns]
    assert teaser_man["source_chapters"] == [5]


# ---------------------------------------------------------------- Task 8
def test_build_arg_parser_required_flags():
    p = tp.build_arg_parser()
    args = p.parse_args(["--bundle-id", "12", "--chapter-dirs", "/a", "/b",
                         "--out-dir", "/o"])
    assert args.bundle_id == 12 and args.chapter_dirs == ["/a", "/b"]
    # ollama path uses --ollama-model (NOT --model); default gemma4:26b
    assert args.ollama_model == "gemma4:26b"
    # payoff tail is OFF by default — the power reveal is the hook, not a spoiler
    assert args.payoff_tail_frac == 0.0
    # --shortlist-n is still accepted (worker passes it) though the montage ignores it
    assert args.shortlist_n == 4


def test_load_bundle_panels_tags_chapter_and_abspath(tmp_path):
    ch = tmp_path / "ch5"
    ch.mkdir()
    (ch / "manifest.panels.understood.json").write_text(json.dumps(
        {"panels": [{"scene_file": "scene_0001.jpg", "panel_kind": "story",
                     "intensity": "tense", "description": "d", "action": "a",
                     "dialogue": "", "subjects": []}]}))
    panels = tp.load_bundle_panels([str(ch)])
    assert panels[0]["chapter_number"]  # derived from dir name or order
    assert panels[0]["scene_file"].endswith("ch5/scenes/scene_0001.jpg")
