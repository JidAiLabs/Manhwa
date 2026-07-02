"""
tests/test_reconcile_seam_panels.py

TDD for tools/reconcile_seam_panels.py — cross-chunk seam detection + reassembly.
The detector operates purely on scenes[] records (no image I/O). The reassembler
is unit-tested with small synthetic PIL images in tmp_path.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "reconcile_seam_panels",
    Path(__file__).resolve().parent.parent / "tools" / "reconcile_seam_panels.py",
)
rsp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rsp)  # type: ignore[union-attr]


# ---- fixture builder --------------------------------------------------------

def _scene(panel_id, chunk_file, chunk_h, gy0, box, *, dhash=0, w=1200, h=None):
    """Minimal scene record carrying only the fields the detector reads."""
    x0, y0, x1, y1 = box
    return {
        "panel_id": panel_id,
        "chunk_file": chunk_file,
        "chunk_path": f"/fake/{chunk_file}",
        "chunk_h": chunk_h,
        "chunk_w": w,
        "chunk_global_y0": gy0,
        "box_px_xyxy": [x0, y0, x1, y1],
        "w": (x1 - x0) if w is None else w,
        "h": (y1 - y0) if h is None else h,
        "dhash64": dhash,
        "out_file": f"{panel_id}.jpg",
    }


def _ch1_like_scenes():
    """Real tutorial-tower ch1 geometry: chunk_0003→chunk_0004 is a true seam;
    chunk_0001→0002 and chunk_0002→0003 are clean gutter cuts (negatives)."""
    return [
        _scene("p01", "chunk_0001.jpg", 14457, 0,     [0, 12644, 1072, 13883]),  # gap 574
        _scene("p02", "chunk_0002.jpg", 15349, 14457, [0, 0, 1200, 1007]),
        _scene("p03", "chunk_0002.jpg", 15349, 14457, [309, 13352, 1134, 15120]),  # gap 229
        _scene("p04", "chunk_0003.jpg", 13398, 29806, [339, 10, 1159, 471]),
        _scene("p05", "chunk_0003.jpg", 13398, 29806, [0, 2115, 1200, 13398]),  # A: y1==chunk_h
        _scene("p06", "chunk_0004.jpg", 16026, 43204, [0, 0, 1200, 825]),        # B: y0==0
    ]


# ---- detector ---------------------------------------------------------------

def test_detects_the_true_seam_pair():
    chains = rsp.find_seam_chains(_ch1_like_scenes())
    # exactly one chain, the p05/p06 seam, in stitch order
    assert chains == [["p05", "p06"]]


def test_gutter_cut_pairs_are_not_merged():
    # remove the true seam so only the two gutter-cut adjacencies remain
    scenes = [s for s in _ch1_like_scenes() if s["panel_id"] not in ("p05", "p06")]
    assert rsp.find_seam_chains(scenes) == []


def test_high_dhash_pair_is_vetoed():
    scenes = _ch1_like_scenes()
    for s in scenes:
        if s["panel_id"] == "p05":
            s["dhash64"] = 0
        if s["panel_id"] == "p06":
            s["dhash64"] = (1 << 40) - 1  # popcount 40 > DHASH_VETO(20)
    assert rsp.find_seam_chains(scenes) == []


def test_three_chunk_chain_is_one_component():
    # a very tall panel across 3 chunks: the middle chunk's SOLE panel touches
    # BOTH edges (y0~0 AND y1~chunk_h) -> connected component of size 3.
    scenes = [
        _scene("a", "c1.jpg", 10000, 0,     [0, 3000, 1200, 10000]),  # bottom touches
        _scene("m", "c2.jpg", 10000, 10000, [0, 0, 1200, 10000]),     # touches BOTH
        _scene("b", "c3.jpg", 10000, 20000, [0, 0, 1200, 4000]),      # top touches
    ]
    assert rsp.find_seam_chains(scenes) == [["a", "m", "b"]]


# ---- reassembly -------------------------------------------------------------

def test_reassemble_trims_overlap_band_and_sums_height():
    OVERLAP = 30
    # A = 100px solid red. B = [30px green overlap band == A's tail] + 80px blue.
    a = Image.new("RGB", (40, 100), (255, 0, 0))
    b = Image.new("RGB", (40, 110), (0, 0, 255))
    for y in range(OVERLAP):                       # paint B's top band green
        for x in range(40):
            b.putpixel((x, y), (0, 255, 0))
    # B.y0 == 0 -> top_trim = OVERLAP - 0 = OVERLAP; A's top_trim = 0
    merged = rsp.reassemble_slices([a, b], [0, OVERLAP])
    assert merged.size == (40, 100 + 110 - OVERLAP)   # 180: A.h + B.h - overlap
    # the green overlap band must be gone (appears zero times)
    px = list(merged.getdata())
    assert (0, 255, 0) not in px
    # red on top, blue on the bottom row
    assert merged.getpixel((0, 0)) == (255, 0, 0)
    assert merged.getpixel((0, merged.height - 1)) == (0, 0, 255)


def test_reassemble_partial_edge_offset():
    # B.y0 = 5 (topmost box began 5px below the top edge, within EDGE_TOL) ->
    # top_trim = OVERLAP - B.y0 leaves NO sliver of the repeated band.
    OVERLAP = 30
    a = Image.new("RGB", (10, 50), (10, 10, 10))
    b = Image.new("RGB", (10, 60), (20, 20, 20))
    top_trim = OVERLAP - 5
    merged = rsp.reassemble_slices([a, b], [0, top_trim])
    assert merged.height == 50 + (60 - top_trim)


# ---- episode-level reconcile (image I/O in tmp_path) ------------------------

def _write_episode(tmp_path, overlap=30):
    """Two chunk images + a scenes manifest whose bottom-of-c1 / top-of-c2
    panels form a true seam. Returns (ep_dir, scenes_manifest_path)."""
    ep = tmp_path / "ep"
    (ep / "scenes").mkdir(parents=True)
    ch1 = Image.new("RGB", (40, 100), (200, 30, 30))   # chunk_h = 100
    ch2 = Image.new("RGB", (40, 100), (30, 30, 200))   # chunk_h = 100
    # make c2's top `overlap` band a copy of c1's bottom band (shared pixels)
    band = ch1.crop((0, 100 - overlap, 40, 100))
    ch2.paste(band, (0, 0))
    ch1.save(ep / "c1.jpg"); ch2.save(ep / "c2.jpg")

    def scene(pid, cf, gy0, box, chunk_path):
        x0, y0, x1, y1 = box
        crop = Image.open(chunk_path).crop((x0, y0, x1, y1))
        of = f"{pid}.jpg"
        crop.save(ep / "scenes" / of)
        return {"panel_id": pid, "chunk_file": cf, "chunk_path": str(chunk_path),
                "chunk_w": 40, "chunk_h": 100, "chunk_global_y0": gy0,
                "panel_index_in_chunk": 0, "recovered": False, "part_index": 0,
                "box_px_xyxy": [x0, y0, x1, y1],
                "box_norm": [y0 / 100, x0 / 40, y1 / 100, x1 / 40],
                "out_file": of, "out_path": str(ep / "scenes" / of),
                "w": x1 - x0, "h": y1 - y0, "blank_score": 0.0,
                "edge_density": 0.1, "trim": {"trimmed": False},
                "protected_spans_local": [], "dhash64": 0,
                "split": {"enabled": True}}

    scenes = [
        scene("p_top", "c1.jpg", 0,   [0, 5, 40, 40],  ep / "c1.jpg"),   # a normal panel
        scene("p_a",   "c1.jpg", 0,   [0, 40, 40, 100], ep / "c1.jpg"),  # A: y1==chunk_h
        scene("p_b",   "c2.jpg", 100, [0, 0, 40, 70],  ep / "c2.jpg"),   # B: y0==0
    ]
    sm = ep / "manifest.scenes.json"
    sm.write_text(json.dumps({"count_scenes": len(scenes), "scenes": scenes}))
    (ep / "manifest.stitch.json").write_text(
        json.dumps({"adaptive": {"overlap_px": overlap}}))
    return ep, sm


def test_reconcile_episode_merges_and_rewrites(tmp_path):
    import json
    ep, sm = _write_episode(tmp_path, overlap=30)
    n = rsp.reconcile_episode(str(sm), str(ep / "manifest.stitch.json"),
                              str(ep / "scenes"))
    assert n == 1  # one chain merged
    out = json.loads(sm.read_text())
    ids = [s["panel_id"] for s in out["scenes"]]
    assert "p_b" not in ids                     # orphan slice record dropped
    assert not (ep / "scenes" / "p_b.jpg").exists()   # orphan JPG deleted
    surv = next(s for s in out["scenes"] if s["panel_id"] == "p_a")
    assert surv["reconciled_seam"] is True
    assert surv["merged_from"] == ["p_a", "p_b"]
    # merged height = A(60) + B(70) - (overlap 30 - B.y0 0) = 100
    assert surv["h"] == 100
    assert Image.open(ep / "scenes" / surv["out_file"]).height == 100
    assert out["count_scenes"] == 2


def test_reconcile_episode_is_idempotent(tmp_path):
    import json
    ep, sm = _write_episode(tmp_path, overlap=30)
    assert rsp.reconcile_episode(str(sm), str(ep / "manifest.stitch.json"),
                                 str(ep / "scenes")) == 1
    # second pass: no seam pairs left -> 0 merges, manifest unchanged in shape
    assert rsp.reconcile_episode(str(sm), str(ep / "manifest.stitch.json"),
                                 str(ep / "scenes")) == 0
    assert json.loads(sm.read_text())["count_scenes"] == 2
