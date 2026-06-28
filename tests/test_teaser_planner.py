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


# ---------------------------------------------------------------- Task 5
def test_payoff_tail_excluded():
    seq = [{"scene_file": f"p{i}", "panel_kind": "story", "intensity": "calm",
            "description": "x", "action": "", "dialogue": "", "subjects": []} for i in range(10)]
    wins = tp.score_windows(seq, min_panels=2, max_panels=3, payoff_tail_frac=0.2, shortlist_n=5)
    # last 20% (p8,p9) must not appear in any returned window
    assert all(p["scene_file"] not in ("p8", "p9") for w in wins for p in w["panels"])


def test_windows_respect_max_panels_and_nonoverlap():
    seq = [{"scene_file": f"p{i}", "panel_kind": "story", "intensity": "tense",
            "description": "exam", "action": "fight", "dialogue": "", "subjects": []} for i in range(12)]
    wins = tp.score_windows(seq, min_panels=4, max_panels=10, payoff_tail_frac=0.2, shortlist_n=3)
    assert wins and all(4 <= len(w["panels"]) <= 10 for w in wins)
    # non-overlapping shortlist
    spans = sorted((w["start"], w["end"]) for w in wins)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_no_windows_when_too_few_panels():
    seq = [{"scene_file": "p0", "panel_kind": "story", "intensity": "calm",
            "description": "x", "action": "", "dialogue": "", "subjects": []}]
    assert tp.score_windows(seq, min_panels=4, max_panels=10, payoff_tail_frac=0.2, shortlist_n=3) == []


# ---------------------------------------------------------------- Task 6
def test_select_and_write_builds_teaser_manifest(tmp_path):
    win = {"start": 0, "end": 4, "score": 9.0, "signals": {},
           "panels": [{"chapter_number": 5, "scene_file": "/abs/ch5/scenes/scene_0007.jpg",
                       "panel_kind": "story", "intensity": "tense",
                       "description": "exam begins", "action": "heir mocks him",
                       "dialogue": "you have no badge", "subjects": ["heir", "prince"]}]}

    def stub(payload):
        assert "windows" in payload and "loglines" in payload
        return {"chosen_index": 0,
                "panel_narration": [{"scene_file": "scene_0007.jpg", "line": "The exam begins."}],
                "rewind_line": "But to see how he got here, we go back to the start.",
                "reason": "public test + humiliation", "spoiler_boundary": "no identity reveal"}

    out = tp.select_and_write([win], loglines=["a hunted prince"], model_call=stub)
    assert out["rewind_line"].startswith("But to see")
    assert out["source_chapters"] == [5]
    assert out["panel_narration"][0]["scene_file"] == "scene_0007.jpg"


# ---------------------------------------------------------------- Task 7
def test_materialize_teaser_dir(tmp_path):
    # a fake source chapter with one scene + a scenes manifest entry
    src = tmp_path / "ch5"
    (src / "scenes").mkdir(parents=True)
    (src / "scenes" / "scene_0007.jpg").write_bytes(b"\xff\xd8\xff")  # tiny jpg stub
    (src / "manifest.scenes.json").write_text(json.dumps(
        {"scenes": [{"out_file": "scene_0007.jpg", "box_px_xyxy": [0, 0, 100, 200],
                     "chunk_global_y0": 0, "w": 100, "h": 200}]}))
    teaser = {"source_chapters": [5], "scene_files": ["scene_0007.jpg"],
              "panel_narration": [{"scene_file": "scene_0007.jpg", "line": "The exam begins."}],
              "rewind_line": "...", "reason": "...", "spoiler_boundary": "..."}
    # map each scene_file -> its source ep_dir
    src_of = {"scene_0007.jpg": str(src)}
    out_dir = tmp_path / "teaser"
    tp.materialize_teaser_dir(teaser, src_of, out_dir, cast={"cast": []})
    assert (out_dir / "scenes" / "scene_0007.jpg").exists()           # symlink/copy
    beats = json.loads((out_dir / "manifest.beats.json").read_text())
    assert beats["beats"][0]["panel_narration"][0]["line"] == "The exam begins."
    groups = json.loads((out_dir / "manifest.groups.json").read_text())
    assert groups["shots"][0]["scene_files"] == ["scene_0007.jpg"]
    scenes = json.loads((out_dir / "manifest.scenes.json").read_text())
    assert scenes["scenes"][0]["out_file"] == "scene_0007.jpg"
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

    # a window built from chapter A's panel, exactly as load_bundle_panels emits:
    # abs scene_file + chapter_number.
    win = {"start": 0, "end": 1, "score": 9.0, "signals": {},
           "panels": [{"chapter_number": 5,
                       "scene_file": str(chA / "scenes" / base),
                       "panel_kind": "story", "intensity": "tense",
                       "description": "exam begins", "action": "heir mocks him",
                       "dialogue": "", "subjects": []}]}

    def stub(payload):
        # model echoes ONE line keyed by the (now colliding) basename; the planner
        # must align by ORDER, not by this basename.
        return {"chosen_index": 0,
                "panel_narration": [{"scene_file": base, "line": "The exam begins."}],
                "rewind_line": "But to see how he got here, we go back.",
                "reason": "r", "spoiler_boundary": "s"}

    teaser = tp.select_and_write([win], loglines=["x"], model_call=stub)
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
