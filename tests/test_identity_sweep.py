"""tools/identity_sweep.py — index a whole series, build its profile, grade it.

Synthetic series on disk; the models are the heads_fn/embed_fn seams."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import identity_sweep as sw  # noqa: E402
import panel_identity_ccip as pic  # noqa: E402

DIM = 16
RNG = np.random.default_rng(1)
VEC = {k: RNG.normal(size=DIM) for k in ("mc", "x", "other_mc")}


def _series(root, name, plan, mc="mc"):
    """plan: [(n_mc_panels, n_x_panels)] per chapter. Returns series dir + the
    {path: who} map the fake embedder reads."""
    sd = root / name
    who = {}
    for c, (n_mc, n_x) in enumerate(plan, start=1):
        ep = sd / f"Chapter_{c}"
        (ep / "scenes").mkdir(parents=True)
        panels, items = [], []
        for k in range(n_mc + n_x):
            fn = f"p{k:06d}.jpg"
            Image.new("RGB", (8, 8), (c, k, 7)).save(ep / "scenes" / fn)
            kind = mc if k < n_mc else "x"
            who[str(ep / "scenes" / fn)] = kind
            panels.append({"scene_file": fn, "panel_kind": "story",
                           "subjects": ["a young man"]})
            items.append({"scene_file": fn, "ocr_clean": ""})
        (ep / "manifest.panels.understood.json").write_text(json.dumps({"panels": panels}))
        (ep / "manifest.vision.json").write_text(json.dumps({"items": items}))
        (ep / "manifest.cast.json").write_text(json.dumps({"cast": [
            {"canonical_name": "Kim Dokja", "is_protagonist": True,
             "visual_description": "dark hair"},
            {"canonical_name": "Namwoon Kim", "visual_description": "white hair"}]}))
        (ep / "manifest.beats.json").write_text(json.dumps({"beats": [
            {"group_id": 1, "segments": [
                {"span": ["p000000.jpg"], "line": "Namwoon smirks at the screen."},
                {"span": ["p000001.jpg"], "line": "Kim Dokja frowns."},
                {"span": [f"p{n_mc:06d}.jpg"], "line": "Namwoon steps back."}]}]}))
    return sd, who


def _seams(who):
    rng = np.random.default_rng(2)

    def heads(path):
        return [((0, 0, 8, 8), 0.9)]

    def embed(items):
        return np.asarray([VEC[who[p]] + rng.normal(scale=0.05, size=DIM)
                           for p, _b in items], dtype=np.float32)
    return heads, embed


def test_sweep_builds_an_active_profile_a_sheet_and_the_census(tmp_path):
    sd, who = _series(tmp_path, "s", [(30, 4)] * 6)
    heads, embed = _seams(who)
    sheet = tmp_path / "sheet.jpg"
    rep = sw.sweep_series(sd, heads_fn=heads, embed_fn=embed, sheet_path=sheet)
    assert rep["chapters"] == 6 and rep["profile"]["status"] == "active"
    assert pic.load_profile(sd)["status"] == "active"          # saved
    assert sheet.exists() and rep["sheet_heads"] > 0
    assert not rep["drift"]
    # census: a NON-protagonist named on a panel that shows only the
    # protagonist (p000000), per chapter; never on a panel showing someone else
    hits = rep["census"]
    assert {h["chapter"] for h in hits} == {f"Chapter_{c}" for c in range(1, 7)}
    assert all(h["span"] == ["p000000.jpg"] and "namwoon" in h["nouns"] for h in hits)
    # the sweep never writes a chapter's identity: that happens at prepare time
    assert not list(sd.glob("*/manifest.identity.json"))


def test_sweep_raises_drift_when_the_protagonist_vanishes(tmp_path):
    sd, who = _series(tmp_path, "s", [(20, 2)] * 9 + [(0, 10)] * 5)
    heads, embed = _seams(who)
    rep = sw.sweep_series(sd, heads_fn=heads, embed_fn=embed)
    assert rep["profile"]["status"] == "active"
    assert rep["drift"] == [["Chapter_10", "Chapter_14"]]


def test_cross_series_control(tmp_path):
    a, who_a = _series(tmp_path, "a", [(30, 4)] * 6)
    b, who_b = _series(tmp_path, "b", [(30, 4)] * 6, mc="other_mc")
    for sd, who in ((a, who_a), (b, who_b)):
        h, e = _seams(who)
        sw.sweep_series(sd, heads_fn=h, embed_fn=e)
    rates = sw.foreign_rates([a, b])
    assert rates["a"] < 0.05 and rates["b"] < 0.05
    # a third series whose lead IS a's lead: a's profile names its heads
    c, who_c = _series(tmp_path, "c", [(30, 4)] * 6)
    h, e = _seams(who_c)
    sw.sweep_series(c, heads_fn=h, embed_fn=e)
    assert sw.foreign_rates([a, c])["a"] > 0.5
