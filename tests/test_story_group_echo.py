"""Pre-narration zoom/echo-twin fold (2026-07-16 wave): compute_echo_pairs +
merge_echo_shots in story_group, and the writer-side span glue — a zoom
re-frame of the previous panel becomes ONE beat / ONE span / ONE voiced line
instead of two narrated events."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sg = _load("story_group")
gnp = _load("gemini_narrative_pass")


def _art(seed, h=200, w=100):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _panels(files):
    return [{"scene_file": f} for f in files]


def test_echo_pairs_identical_art_and_zoom_crop():
    base = _art(7, h=400)
    zoom = base[200:400].copy()               # the artist's re-framed lower half
    distinct = _art(99, h=400)
    imgs = {"a.jpg": base, "b.jpg": zoom, "c.jpg": distinct}
    vmap = {f: {"ocr_clean": ""} for f in imgs}
    pairs = sg.compute_echo_pairs(
        _panels(["a.jpg", "b.jpg", "c.jpg"]), vmap,
        get_img=lambda f: imgs[f], get_boxes=lambda img: ())
    assert ("a.jpg", "b.jpg") in pairs        # zoom branch (half-crop hash)
    assert all(b != "c.jpg" for _a, b in pairs)   # distinct art never pairs


def test_echo_pairs_respects_adjacency_window():
    base = _art(7, h=400)
    imgs = {"a.jpg": base, "x.jpg": _art(1, h=400), "y.jpg": _art(2, h=400),
            "b.jpg": base.copy()}
    vmap = {f: {"ocr_clean": ""} for f in imgs}
    pairs = sg.compute_echo_pairs(
        _panels(["a.jpg", "x.jpg", "y.jpg", "b.jpg"]), vmap,
        get_img=lambda f: imgs[f], get_boxes=lambda img: ())
    assert pairs == []                        # 3 apart > window=2, no pair


def test_merge_echo_shots_folds_straddling_beats_and_renumbers():
    shots = [
        {"shot_id": 1, "segment": "present", "scene_files": ["a.jpg"]},
        {"shot_id": 2, "segment": "present", "scene_files": ["b.jpg"]},
        {"shot_id": 3, "segment": "present", "scene_files": ["c.jpg"]},
    ]
    out = sg.merge_echo_shots(shots, [("a.jpg", "b.jpg")])
    assert [s["scene_files"] for s in out] == [["a.jpg", "b.jpg"], ["c.jpg"]]
    assert [s["shot_id"] for s in out] == [1, 2]
    # same-shot pair and empty pairs are no-ops
    assert sg.merge_echo_shots(shots, []) is shots
    same = sg.merge_echo_shots(out, [("a.jpg", "b.jpg")])
    assert [s["scene_files"] for s in same] == [["a.jpg", "b.jpg"], ["c.jpg"]]


def test_glue_echo_spans_moves_echo_into_original_span():
    surviving = ["a.jpg", "b.jpg", "c.jpg"]
    segs = [{"span": ["a.jpg"], "line": "The blade lands."},
            {"span": ["b.jpg"], "line": "Dust swirls."},
            {"span": ["c.jpg"], "line": "He staggers back."}]
    out = gnp.glue_echo_spans(segs, {"b.jpg": "a.jpg"}, surviving)
    assert out[0]["span"] == ["a.jpg", "b.jpg"]
    assert out[0]["line"] == "The blade lands. Dust swirls."   # words kept
    assert out[1]["span"] == ["c.jpg"]
    # non-adjacent echo is left alone (render ken restyle covers it)
    out2 = gnp.glue_echo_spans(segs, {"c.jpg": "a.jpg"}, surviving)
    assert [s["span"] for s in out2] == [["a.jpg"], ["b.jpg"], ["c.jpg"]]


def test_validate_segments_echo_belt_and_cap_exemption():
    surviving = ["a.jpg", "b.jpg"]
    kinds = {}
    split = [{"span": ["a.jpg"], "line": "One two three four five six seven."},
             {"span": ["b.jpg"], "line": "Eight nine ten eleven twelve came."}]
    errs = gnp.validate_segments(split, surviving, kinds,
                                 echo_of={"b.jpg": "a.jpg"})
    assert any("echo pair split" in e for e in errs)
    # a glued span over SPAN_CAP passes when the overflow is echo riders
    files = [f"p{i}.jpg" for i in range(gnp.SPAN_CAP)] + ["z.jpg"]
    seg = [{"span": files,
            "line": " ".join(["word"] * (3 * len(files)))}]
    errs = gnp.validate_segments(seg, files, {},
                                 echo_of={"z.jpg": files[-2]})
    assert not any("exceeds the cap" in e for e in errs)
    errs = gnp.validate_segments(seg, files, {})
    assert any("exceeds the cap" in e for e in errs)


def test_expand_index_ranges_tolerates_typoed_keys():
    # nano ch1 2026-07-16: the model wrote "fromindex" once; MLX at temp 0 is
    # deterministic, so every retry returned the identical typo — the parser
    # must normalize unambiguous key typos instead of failing the chapter.
    order = [f"p{i}.jpg" for i in range(8)]
    beats = [
        {"from_index": 0, "to_index": 3, "segment": "present",
         "arc_label": "a"},
        {"fromindex": 4, "to_index": 7, "segment": "present",
         "arc_label": "b"},
    ]
    expanded, issue = sg.expand_index_ranges(beats, order)
    assert issue == ""
    assert [b["scene_files"] for b in expanded] == [order[0:4], order[4:8]]
    # a genuinely missing key still fails loudly
    expanded, issue = sg.expand_index_ranges(
        [{"to_index": 3, "segment": "present"}], order)
    assert expanded == [] and "from_index" in issue


def test_expand_index_ranges_clamps_exclusive_end_fencepost():
    # jobs 50-52 2026-07-17: the model ended the LAST beat at to_index == N
    # (exclusive-end slip) for an N-panel chunk; deterministic backends repeat
    # it on every retry, so the parser clamps instead of failing the chapter.
    order = [f"p{i}.jpg" for i in range(5)]
    beats = [
        {"from_index": 0, "to_index": 3, "segment": "present", "arc_label": "a"},
        {"from_index": 4, "to_index": 5, "segment": "present", "arc_label": "b"},
    ]
    expanded, issue = sg.expand_index_ranges(beats, order)
    assert issue == ""
    assert expanded[-1]["scene_files"] == ["p4.jpg"]      # clamped to 4..4
    # a fully out-of-range beat still fails loudly
    expanded, issue = sg.expand_index_ranges(
        [{"from_index": 0, "to_index": 9, "segment": "present"}], order)
    assert expanded == [] and "out of bounds" in issue
