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
    assert len(chat.seen) == 1                         # one question for the series
    assert obj["seed"] == "exemplars" and obj["auto"]["asked_gemma"]
    assert all(who[p] == "mc" for p in obj["protagonist"])
    assert len(obj["protagonist"]) == 2 and len(obj["decoy"]) == 2
    assert len({Path(p).parts[-3] for p in obj["protagonist"]}) == 2   # distinct chapters
    assert json.loads(out.read_text())["protagonist"] == obj["protagonist"]


def test_the_gemma_answer_is_what_decides(tmp_path):
    sd, who = _series(tmp_path, PLAN)
    obj = ie.auto_pick(sd, tmp_path / "a.json", chat=_chat_picking(who, "x"),
                       load=lambda p: str(p).encode())
    assert all(who[p] == "x" for p in obj["protagonist"])


def test_when_the_rules_agree_no_question_is_asked(tmp_path):
    sd, who = _series(tmp_path, [["mc"] * 10 + ["y"] * 3 for _ in range(6)])
    chat = _chat_picking(who, "mc")
    obj = ie.auto_pick(sd, tmp_path / "a.json", chat=chat, load=lambda p: str(p).encode())
    assert chat.seen == [] and not obj["auto"]["asked_gemma"]
    assert all(who[p] == "mc" for p in obj["protagonist"])


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
