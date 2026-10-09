"""tools/panel_identity_ccip.py — who is drawn in a panel, from the picture.

Synthetic tests drive the logic through the heads_fn/embed_fn seams (no onnx in
.eval_venv); the real-data tests replay the 2026-10-10 spike's CCIP features
(tests/fixtures/ccip_spike_fixture.*: 717 ORV + 489 Tutorial Tower heads)."""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import panel_identity_ccip as pic  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
DIM = 32


# ---------------------------------------------------------------- synthetic
def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


RNG = np.random.default_rng(42)
CHARS = {name: _unit(RNG.normal(size=DIM)) for name in ("mc", "a", "b", "c", "d")}


def _face(who, rng, noise=0.05):
    return _unit(CHARS[who] + rng.normal(scale=noise, size=DIM))


def _indexes(n_chapters=6, mc_per=30, other_per=12, seed=0, mc="mc"):
    """Chapter indexes as index_chapter returns them: the MC dominates."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(1, n_chapters + 1):
        heads, feats = [], []
        for k in range(mc_per):
            heads.append({"panel": f"p{k:06d}.jpg", "box": [0, 0, 10, 10], "score": 0.9})
            feats.append(_face(mc, rng))
        for k in range(other_per):
            who = "abcd"[k % 4]
            heads.append({"panel": f"p{100 + k:06d}.jpg", "box": [0, 0, 10, 10], "score": 0.9})
            feats.append(_face(who, rng))
        out.append({"chapter": f"Chapter_{c}", "key": f"k{c}", "heads": heads,
                    "feats": np.asarray(feats, dtype=np.float32)})
    return out


def test_ccip_diff_is_half_one_minus_cosine():
    a = _unit([[1, 0, 0], [0, 1, 0], [-1, 0, 0]])
    d = pic.ccip_diff(a, a[:1])
    assert np.allclose(d[:, 0], [0.0, 0.5, 1.0], atol=1e-6)
    # scale-free: un-normalised features give the same difference
    assert np.allclose(pic.ccip_diff(a * 7, a[:1] * 3), d, atol=1e-6)


def test_chapter_order_is_natural():
    names = ["Chapter_10", "Chapter_2", "Chapter_1.5", "Episode_100", "Chapter_1"]
    assert sorted(names, key=pic.chapter_order) == [
        "Chapter_1", "Chapter_1.5", "Chapter_2", "Chapter_10", "Episode_100"]


# ---------------------------------------------------------------- profile
def test_profile_seed_is_the_densest_head_and_refs_span_distinct_chapters():
    prof = pic.build_profile(_indexes(n_chapters=12))
    assert prof["status"] == "active", prof["reasons"]
    refs = prof["refs"]
    assert 2 <= len(refs) <= pic.MAX_REFS
    chapters = [r["chapter"] for r in refs]
    assert len(set(chapters)) == len(chapters)            # distinct chapters
    assert "Chapter_1" in chapters and "Chapter_12" in chapters   # the span
    seed = np.asarray(prof["seed"]["feat"], dtype=np.float32)[None]
    assert pic.ccip_diff(seed, CHARS["mc"][None])[0, 0] < 0.05
    for r in refs:                                         # refs are the MC
        f = np.asarray(r["feat"], dtype=np.float32)[None]
        assert pic.ccip_diff(f, CHARS["mc"][None])[0, 0] < 0.05
    assert prof["cut"] == pic.CUT
    assert prof["stats"]["matched"] == 12 * 30


@pytest.mark.parametrize("kw,reason", [
    (dict(n_chapters=4), "chapters"),
    (dict(n_chapters=6, mc_per=20), "matched"),
])
def test_profile_stays_provisional_until_the_evidence_is_there(kw, reason):
    prof = pic.build_profile(_indexes(**kw))
    assert prof["status"] == "provisional"
    assert any(reason in r for r in prof["reasons"])


def test_profile_provisional_when_the_mc_is_missing_from_most_chapters():
    idx = _indexes(n_chapters=10, mc_per=40)
    for ix in idx[:5]:                       # the MC vanishes from 5 of 10
        keep = [i for i, h in enumerate(ix["heads"]) if h["panel"] >= "p000100"]
        ix["heads"] = [ix["heads"][i] for i in keep]
        ix["feats"] = ix["feats"][keep]
    prof = pic.build_profile(idx)
    assert prof["status"] == "provisional"
    assert any("spread" in r for r in prof["reasons"])


def test_profile_provisional_when_no_group_dominates():
    # two characters equally visible: "most visible" is a coin flip
    rng = np.random.default_rng(3)
    idx = []
    for c in range(1, 7):
        feats = [_face("mc", rng) for _ in range(30)] + [_face("a", rng) for _ in range(30)]
        idx.append({"chapter": f"Chapter_{c}", "key": f"k{c}",
                    "heads": [{"panel": f"p{k:06d}.jpg", "box": [0, 0, 1, 1], "score": 1}
                              for k in range(60)],
                    "feats": np.asarray(feats, dtype=np.float32)})
    prof = pic.build_profile(idx)
    assert prof["status"] == "provisional"
    assert any("dominan" in r for r in prof["reasons"])


def test_pins_join_the_refs_when_they_agree_and_block_when_they_do_not():
    rng = np.random.default_rng(9)
    good = np.asarray([_face("mc", rng)])
    prof = pic.build_profile(_indexes(), pins=good)
    assert prof["status"] == "active"
    assert sum(1 for r in prof["refs"] if r.get("pin")) == 1
    bad = np.asarray([_face("a", rng)])
    prof = pic.build_profile(_indexes(), pins=bad)
    assert prof["status"] == "provisional"
    assert any("exemplar" in r for r in prof["reasons"])


def test_refresh_keeps_continuity_and_versions():
    first = pic.build_profile(_indexes(seed=1))
    again = pic.build_profile(_indexes(n_chapters=8, seed=2), prior=first)
    assert again["status"] == "active" and again["version"] == first["version"] + 1
    assert not again["alarms"]


def test_refresh_that_flips_to_another_character_keeps_the_old_profile():
    first = pic.build_profile(_indexes(seed=1))
    flipped = pic.build_profile(_indexes(n_chapters=8, seed=2, mc="a"), prior=first)
    assert flipped["version"] == first["version"]
    assert [r["feat"] for r in flipped["refs"]] == [r["feat"] for r in first["refs"]]
    assert any(a.startswith("profile_flip") for a in flipped["alarms"])
    assert flipped["chapters_indexed"] == 8          # cadence still advances


def test_refs_from_a_reindexed_chapter_are_dropped_not_dangling():
    first = pic.build_profile(_indexes(seed=1))
    idx = _indexes(seed=1)
    gone = {r["chapter"] for r in first["refs"] if not r.get("pin")}
    for ix in idx:
        if ix["chapter"] in gone:
            ix["key"] = ix["key"] + "-redetected"
    kept = pic.live_refs(first, idx)
    assert all(r["chapter"] not in gone for r in kept)


def test_profile_due_cadence():
    assert pic.profile_due(None, 1)
    assert pic.profile_due({"chapters_indexed": 3}, 4)        # < 10: every chapter
    assert not pic.profile_due({"chapters_indexed": 4}, 4)
    assert not pic.profile_due({"chapters_indexed": 12}, 21)  # then every 10
    assert pic.profile_due({"chapters_indexed": 12}, 22)


def test_profile_round_trips_through_disk(tmp_path):
    prof = pic.build_profile(_indexes())
    pic.save_profile(tmp_path, prof)
    back = pic.load_profile(tmp_path)
    assert back["refs"] == prof["refs"] and back["status"] == "active"
    assert pic.load_profile(tmp_path / "nowhere") is None


# ---------------------------------------------------------------- index
def _scene(ep, fn, data=b"x"):
    (ep / "scenes").mkdir(parents=True, exist_ok=True)
    (ep / "scenes" / fn).write_bytes(data)


def _understood(ep, panels):
    ep.mkdir(parents=True, exist_ok=True)
    (ep / "manifest.panels.understood.json").write_text(json.dumps({"panels": panels}))


def test_index_chapter_detects_only_people_panels_and_caches(tmp_path):
    ep = tmp_path / "series" / "Chapter_3"
    for fn in ("p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"):
        _scene(ep, fn, fn.encode())
    _understood(ep, [
        {"scene_file": "scenes/p1.jpg", "panel_kind": "story", "subjects": ["a man"]},
        {"scene_file": "scenes/p2.jpg", "panel_kind": "system", "subjects": ["a man"]},
        {"scene_file": "scenes/p3.jpg", "panel_kind": "story", "subjects": []},
        {"scene_file": "p4.jpg", "panel_kind": "story", "subjects": ["two men"]},
    ])
    seen = []

    def heads(path):
        seen.append(os.path.basename(path))
        return [((1, 2, 3, 4), 0.9)] * (2 if path.endswith("p4.jpg") else 1)

    def embed(items):
        return np.ones((len(items), DIM), dtype=np.float32)

    ix = pic.index_chapter(ep, heads_fn=heads, embed_fn=embed)
    assert seen == ["p1.jpg", "p4.jpg"]
    assert [h["panel"] for h in ix["heads"]] == ["p1.jpg", "p4.jpg", "p4.jpg"]
    assert ix["feats"].shape == (3, DIM) and ix["chapter"] == "Chapter_3"
    assert (tmp_path / "series" / ".identity" / "Chapter_3.json").exists()

    def boom(*_a):
        raise AssertionError("cache must be reused")

    again = pic.index_chapter(ep, heads_fn=boom, embed_fn=boom)
    assert again["key"] == ix["key"] and again["feats"].shape == (3, DIM)
    # a re-detected chapter (new scene pixels) re-indexes
    _scene(ep, "p1.jpg", b"new pixels")
    seen.clear()
    pic.index_chapter(ep, heads_fn=heads, embed_fn=embed)
    assert seen == ["p1.jpg", "p4.jpg"]
    assert [ix["chapter"] for ix in pic.load_indexes(tmp_path / "series")] == ["Chapter_3"]


# ---------------------------------------------------------------- identify
def _ep_for_identify(tmp_path, cast=True):
    ep = tmp_path / "s" / "Chapter_9"
    _understood(ep, [
        {"scene_file": "scenes/solo.jpg", "panel_kind": "story", "subjects": ["a young man"]},
        {"scene_file": "scenes/other.jpg", "panel_kind": "story", "subjects": ["a guard"]},
        {"scene_file": "scenes/twice.jpg", "panel_kind": "story", "subjects": ["two men"]},
        {"scene_file": "scenes/crowd.jpg", "panel_kind": "story",
         "subjects": ["a young man", "a crowd of soldiers"]},
        {"scene_file": "scenes/headless.jpg", "panel_kind": "story", "subjects": ["a hand"]},
    ])
    if cast:
        (ep / "manifest.cast.json").write_text(json.dumps({"cast": [
            {"canonical_name": "Kim Dokja", "is_protagonist": True},
            {"canonical_name": "Yoo Joonghyuk", "is_protagonist": False}]}))
    rng = np.random.default_rng(5)
    heads = [("solo.jpg", "mc"), ("other.jpg", "a"), ("twice.jpg", "mc"),
             ("twice.jpg", "mc"), ("crowd.jpg", "mc")]
    index = {"chapter": "Chapter_9", "key": "k",
             "heads": [{"panel": p, "box": [0, 0, 1, 1], "score": 1} for p, _ in heads],
             "feats": np.asarray([_face(w, rng) for _, w in heads], dtype=np.float32)}
    return ep, index


def test_identify_chapter_shape_and_rules(tmp_path):
    ep, index = _ep_for_identify(tmp_path)
    prof = pic.build_profile(_indexes())
    got = pic.identify_chapter(ep, prof, index)
    P = got["panels"]
    assert P["solo.jpg"]["names"] == ["Kim Dokja"] and P["solo.jpg"]["mc"]
    assert P["solo.jpg"]["others"] == 0 and P["solo.jpg"]["heads"] == 1
    assert P["solo.jpg"]["diff"] < pic.CUT
    assert P["other.jpg"] == {"names": [], "others": 1, "mc": False, "heads": 1,
                              "diff": P["other.jpg"]["diff"]}
    assert P["other.jpg"]["diff"] >= pic.CUT
    # the MC is never drawn twice in practice: two matches = never named
    assert P["twice.jpg"]["names"] == [] and not P["twice.jpg"]["mc"]
    assert P["twice.jpg"]["others"] == 2
    # undetected people still count: 1 head (the MC) + a crowd (>1 person)
    assert P["crowd.jpg"]["names"] == ["Kim Dokja"] and P["crowd.jpg"]["others"] >= 2
    assert "headless.jpg" not in P                 # all-unknown, as today
    on_disk = json.loads((ep / "manifest.identity.json").read_text())
    assert on_disk["panels"] == P
    meta = on_disk["_meta"]
    assert meta["backend"] == "ccip" and meta["profile_version"] == prof["version"]
    assert meta["cut"] == pic.CUT and meta["protagonist"] == "Kim Dokja"


def test_identify_without_a_protagonist_in_the_cast_counts_the_mc_as_other(tmp_path):
    ep, index = _ep_for_identify(tmp_path, cast=False)
    got = pic.identify_chapter(ep, pic.build_profile(_indexes()), index)
    solo = got["panels"]["solo.jpg"]
    assert solo["names"] == [] and solo["mc"] and solo["others"] == 1


def test_provisional_profile_writes_nothing(tmp_path):
    ep, index = _ep_for_identify(tmp_path)
    prof = pic.build_profile(_indexes(n_chapters=3))
    assert prof["status"] == "provisional"
    assert pic.identify_chapter(ep, prof, index) is None
    assert pic.identify_chapter(ep, None, index) is None
    assert not (ep / "manifest.identity.json").exists()


# ---------------------------------------------------------------- real data
def _fixture(key):
    meta = json.loads((FIX / "ccip_spike_fixture.json").read_text())[key]
    arr = np.load(FIX / "ccip_spike_fixture.npz")
    feats = arr[f"{key}_feats"].astype(np.float32)
    by = defaultdict(list)
    for i, (ch, panel, *box) in enumerate(meta["recs"]):
        by[ch].append((i, panel, box))
    idx = [{"chapter": ch, "key": ch,
            "heads": [{"panel": p, "box": b, "score": 1.0} for _, p, b in rows],
            "feats": feats[[i for i, _, _ in rows]]} for ch, rows in by.items()]
    pins = arr.get(f"{key}_exemplar_feats")
    return idx, (None if pins is None else pins.astype(np.float32))


def test_real_orv_auto_profile_is_the_owners_protagonist():
    idx, exemplars = _fixture("orv")
    prof = pic.build_profile(idx, pins=exemplars)
    assert prof["status"] == "active", prof["reasons"]
    auto = [r for r in prof["refs"] if not r.get("pin")]
    assert len({r["chapter"] for r in auto}) >= 6
    # every owner exemplar (Dokja, Ep6) is one of the automatic references' character
    assert pic.ccip_diff(exemplars, pic._feats(auto)).min(axis=1).max() < pic.CUT
    assert prof["stats"]["dominance"] >= 3.0
    assert 330 <= prof["stats"]["matched"] <= 460              # spike: 394 @0.15


def test_real_tutorial_tower_auto_profile():
    idx, _ = _fixture("tt")
    prof = pic.build_profile(idx)
    assert prof["status"] == "active", prof["reasons"]
    assert len({r["chapter"] for r in prof["refs"]}) >= 5
    assert prof["stats"]["dominance"] >= 2.0
    assert 150 <= prof["stats"]["matched"] <= 220              # spike: 183 @0.15


def test_real_cross_series_control():
    """Heads of ANOTHER series are certainly not this series' protagonist.
    ORV's profile names few of them; Tutorial Tower's names many ORV heads
    (both leads are dark-haired young men, 0.165 apart) — the sweep reports
    this rate per series; a rise is the signal to tighten that series."""
    o, ex = _fixture("orv")
    t, _ = _fixture("tt")
    Ro = pic._feats(pic.build_profile(o, pins=ex)["refs"])
    Rt = pic._feats(pic.build_profile(t)["refs"])
    Fo = np.concatenate([ix["feats"] for ix in o])
    Ft = np.concatenate([ix["feats"] for ix in t])
    assert (pic.ccip_diff(Ft, Ro).min(axis=1) < pic.CUT).mean() <= 0.10   # 0.08
    assert (pic.ccip_diff(Fo, Rt).min(axis=1) < pic.CUT).mean() <= 0.30   # 0.27


def test_real_wrong_exemplars_keep_the_series_provisional():
    idx, _ = _fixture("orv")
    tt_idx, _ = _fixture("tt")
    tt_seed = pic.build_profile(tt_idx)["seed"]["feat"]
    prof = pic.build_profile(idx, pins=np.asarray([tt_seed], dtype=np.float32))
    assert prof["status"] == "provisional"


def test_run_while_provisional_removes_a_leftover_identity(tmp_path):
    """Provisional = the keyword identity stands; a file left by an earlier
    profile or another backend would silently keep naming from it."""
    ep = tmp_path / "series" / "Chapter_1"
    _scene(ep, "p1.jpg")
    _understood(ep, [{"scene_file": "p1.jpg", "panel_kind": "story",
                      "subjects": ["a man"]}])
    (ep / "manifest.identity.json").write_text('{"panels": {"p1.jpg": {"names": ["X"]}}}')
    got = pic.run(ep, heads_fn=lambda p: [((0, 0, 1, 1), 0.9)],
                  embed_fn=lambda items: np.ones((len(items), DIM), np.float32))
    assert got is None
    assert not (ep / "manifest.identity.json").exists()
    assert pic.load_profile(ep.parent)["status"] == "provisional"
