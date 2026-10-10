"""Automatic exemplar picker + profiles seeded from exemplars (2026-10-10).

The most-drawn face was the lead on 6 of 9 series; "most chapters" 7 of 9;
"3+ appearances in most chapters" 7-8 of 9 — each failing on a different
series. So the picker takes the candidates the three rules propose and, when
they disagree, asks gemma once which is the main character (with the story
synopsis). The profile is then SEEDED from the picked faces."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import identity_exemplars as ie  # noqa: E402
import panel_identity as pi  # noqa: E402
import panel_identity_ccip as pic  # noqa: E402

DIM = 16
RNG = np.random.default_rng(3)
VEC = {k: RNG.normal(size=DIM) for k in ("mc", "x", "y")}


def _at(diff, base="mc", seed=0):
    """A face exactly *diff* (CCIP difference) from VEC[base]."""
    u = VEC[base] / np.linalg.norm(VEC[base])
    w = np.random.default_rng(seed).normal(size=DIM)
    w -= w.dot(u) * u
    w /= np.linalg.norm(w)
    c = 1 - 2 * diff
    return 4 * (c * u + np.sqrt(1 - c * c) * w)


VEC["mc_odd"] = _at(0.06, seed=1)        # the lead from behind / in shadow
VEC["near"] = _at(0.22, seed=2)          # a look-alike drawn often, in 2 chapters only
VEC["rec"] = _at(0.28, seed=3)           # a look-alike in every chapter


def _series(root, plan):
    """plan: list of chapters, each a list of who per panel (one head each)."""
    sd = root / "s"
    who = {}
    for c, panels in enumerate(plan, start=1):
        ep = sd / f"Chapter_{c}"
        (ep / "scenes").mkdir(parents=True)
        und = []
        for k, w in enumerate(panels):
            fn = f"p{k:06d}.jpg"
            Image.new("RGB", (8, 8), (c, k % 250, 9)).save(ep / "scenes" / fn)
            who[str(ep / "scenes" / fn)] = w
            und.append({"scene_file": fn, "panel_kind": "story", "subjects": ["a man"]})
        (ep / "manifest.panels.understood.json").write_text(json.dumps({"panels": und}))
        (ep / "manifest.chapter_story.json").write_text(json.dumps(
            {"synopsis": f"Chapter {c}: the young hero trains."}))
    rng = np.random.default_rng(1)

    def heads(path):
        return [((0, 0, 8, 8), 0.9)]

    def embed(items):
        return np.asarray([VEC[who[p]] + rng.normal(scale=0.04, size=DIM) for p, _ in items],
                          dtype=np.float32)
    for ep in sorted(sd.iterdir()):
        pic.index_chapter(ep, heads_fn=heads, embed_fn=embed)
    return sd, who


# the lead (mc) is in EVERY chapter 20x; a side character (x) is drawn 70x but
# only in two chapters — "most drawn" picks x, "most chapters" picks mc
PLAN = [["mc"] * 20 + ["x"] * 70 + ["y"] * 4 if c < 2 else ["mc"] * 20 + ["y"] * 4
        for c in range(6)]


def _chat_picking(who, answer_for):
    seen = []

    def chat(model, think, options, messages):
        imgs = [i.decode() for i in messages[0]["images"]]
        seen.append(imgs)
        groups = [who[imgs[i]] for i in range(0, len(imgs), 2)]
        letter = "ABC"[groups.index(answer_for)]
        return {"message": {"content": json.dumps({"main": letter})}}
    chat.seen = seen
    return chat


def test_rules_disagree_and_gemma_decides(tmp_path):
    sd, who = _series(tmp_path, PLAN)
    chat = _chat_picking(who, "mc")
    out = tmp_path / "auto.json"
    obj = ie.auto_pick(sd, out, chat=chat, load=lambda p: str(p).encode())
    assert len(chat.seen) == 2                         # asked twice, order reversed
    assert [who[i] for i in chat.seen[0][::2]] == [who[i] for i in chat.seen[1][::2]][::-1]
    assert obj["seed"] == "exemplars" and obj["auto"]["chosen_by"] == "gemma"
    assert all(who[p] == "mc" for p in obj["protagonist"])
    assert len(obj["protagonist"]) == 2 and len(obj["decoy"]) == 2
    assert len({Path(p).parts[-3] for p in obj["protagonist"]}) == 2   # distinct chapters
    assert json.loads(out.read_text())["protagonist"] == obj["protagonist"]


def test_the_gemma_answer_is_what_decides(tmp_path):
    sd, who = _series(tmp_path, PLAN)
    obj = ie.auto_pick(sd, tmp_path / "a.json", chat=_chat_picking(who, "x"),
                       load=lambda p: str(p).encode())
    assert all(who[p] == "x" for p in obj["protagonist"])


def test_when_the_rules_agree_the_runner_up_is_still_offered(tmp_path):
    # Nano Machine 2026-10-10: all three rules chose the instructor
    sd, who = _series(tmp_path, [["mc"] * 10 + ["y"] * 3 for _ in range(6)])
    chat = _chat_picking(who, "y")
    obj = ie.auto_pick(sd, tmp_path / "a.json", chat=chat, load=lambda p: str(p).encode())
    assert obj["auto"]["rules"]["most_drawn"] == obj["auto"]["rules"]["most_present"]
    assert obj["auto"]["candidates"] == 2 and len(chat.seen) == 2
    assert all(who[p] == "y" for p in obj["protagonist"])


def test_an_answer_by_position_leaves_the_most_present_rules_lead(tmp_path):
    sd, who = _series(tmp_path, PLAN)
    chat = lambda model, think, options, messages: {"message": {"content": '{"main": "A"}'}}
    out = tmp_path / "a.json"
    obj = ie.auto_pick(sd, out, chat=chat, load=lambda p: str(p).encode())
    a = obj["auto"]
    assert a["answers"] == ["A", "A"] and a["chosen_by"] == "rules"
    assert a["chosen"] == a["rules"]["most_present"]
    assert all(who[p] == "mc" for p in obj["protagonist"]) and len(obj["candidates"]) >= 2
    assert json.loads(out.read_text())["candidates"] == obj["candidates"]
    for c in obj["candidates"]:                        # each: ONE lead, ONE look-alike
        assert 2 <= len(c["protagonist"]) <= 6 and 2 <= len(c["decoy"]) <= 4
        lead = {who[h["path"]] for h in c["protagonist"]}
        decoy = {who[h["path"]] for h in c["decoy"]}
        assert len(lead) == len(decoy) == 1 and lead != decoy


def test_typical_faces_beat_odd_ones_and_the_decoy_recurs(tmp_path):
    # the lead is drawn from behind in 30% of panels; "near" is a look-alike
    # drawn 15x in only 2 chapters, "rec" one in every chapter
    plan = [["mc"] * 14 + ["mc_odd"] * 6 + ["rec"] * 4 + (["near"] * 15 if c < 2 else [])
            for c in range(6)]
    sd, who = _series(tmp_path, plan)
    obj = ie.auto_pick(sd, tmp_path / "a.json", chat=_chat_picking(who, "mc"),
                       load=lambda p: str(p).encode())
    assert [who[p] for p in obj["protagonist"]] == ["mc", "mc"]
    assert [who[p] for p in obj["decoy"]] == ["rec", "rec"]


def test_a_profile_seeded_from_exemplars_follows_the_picked_lead(tmp_path):
    sd, who = _series(tmp_path, PLAN)
    idx = pic.load_indexes(sd)
    assert who_of(pic.build_profile(idx), who, sd) == "x"           # most drawn
    obj = ie.auto_pick(sd, tmp_path / "a.json", chat=_chat_picking(who, "mc"),
                       load=lambda p: str(p).encode())
    feats = pic.exemplar_feats(idx, obj)
    prof = pic.build_profile(idx, seed_feats=feats)
    assert prof["status"] == "active", prof["reasons"]                # no dominance rule
    assert who_of(prof, who, sd) == "mc"
    assert prof["seeded"] is True


def who_of(prof, who, sd):
    seed = np.asarray(prof["seed"]["feat"], dtype=np.float32)[None]
    best = min(VEC, key=lambda k: float(pic.ccip_diff(seed, VEC[k][None])[0, 0]))
    return best


def test_check_series_uses_the_seeded_profile_of_an_auto_file(tmp_path):
    sd, who = _series(tmp_path, PLAN)
    sw_prof = pic.build_profile(pic.load_indexes(sd))                 # saved: most drawn (x)
    pic.save_profile(sd, sw_prof)
    obj = ie.auto_pick(sd, tmp_path / "a.json", chat=_chat_picking(who, "mc"),
                       load=lambda p: str(p).encode())
    seen = []

    def chat(model, think, options, messages):
        seen.append(messages[0]["images"][-1].decode())
        return {"message": {"content": '{"people": ["A"]}'}}
    rep = pi.check_series(sd, tmp_path / "a.json", n=10, out_dir=tmp_path / "o", chat=chat,
                          load=lambda p: str(p).encode())
    assert rep["n"] == 10 and all(who[p] == "mc" for p in seen)


def test_production_run_grows_the_profile_from_confirmed_exemplars(tmp_path):
    sd, who = _series(tmp_path, PLAN)
    ep = sorted(sd.iterdir())[-1]
    run = lambda ex: pic.run(ep, exemplars=ex, heads_fn=lambda p: [((0, 0, 8, 8), 0.9)])
    run(None)
    assert who_of(pic.load_profile(sd), who, sd) == "x"              # most drawn
    mc = ie.auto_pick(sd, tmp_path / "mc.json", chat=_chat_picking(who, "mc"),
                      load=lambda p: str(p).encode())
    assert mc["protagonist"]
    run(tmp_path / "mc.json")
    prof = pic.load_profile(sd)
    assert prof["seeded"] and who_of(prof, who, sd) == "mc" and prof["status"] == "active"
    assert (ep / "manifest.identity.json").exists()
    ie.auto_pick(sd, tmp_path / "y.json", chat=_chat_picking(who, "y"),
                 load=lambda p: str(p).encode())
    run(tmp_path / "y.json")                                       # a new pick: rebuilt
    assert who_of(pic.load_profile(sd), who, sd) == "y"
    run(None)                                                       # removed: unseeded again
    assert not pic.load_profile(sd)["seeded"]


def test_two_seeds_on_one_person_are_one_candidate(tmp_path, monkeypatch):
    # Death Knight 2026-10-10: "most present" and "runner up" landed on the
    # same young man (close-ups 0.044 apart) and gemma answered by position
    sd, who = _series(tmp_path, PLAN)
    real = ie._seeds
    monkeypatch.setattr(ie, "_seeds", lambda F, chap: {**real(F, chap),
                                                        "runner_up": real(F, chap)["most_present"]})
    obj = ie.auto_pick(sd, tmp_path / "a.json", chat=_chat_picking(who, "mc"),
                       load=lambda p: str(p).encode())
    leads = [who[c["protagonist"][0]["path"]] for c in obj["candidates"]]
    assert len(leads) == len(set(leads))
    assert obj["auto"]["rules"]["runner_up"] == obj["auto"]["rules"]["most_present"]
