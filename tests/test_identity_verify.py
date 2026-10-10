"""gemma confirms the picture's candidates (owner decision 2026-10-10): the
protagonist is named only when the CCIP match (< 0.10) AND gemma's forced
choice against the series' exemplar panels agree."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import panel_identity as pi  # noqa: E402

EX = {"protagonist": ["ongoing/s/Chapter_1/scenes/a1.jpg", "ongoing/s/Chapter_2/scenes/a2.jpg"],
      "decoy": ["ongoing/s/Chapter_3/scenes/b1.jpg", "ongoing/s/Chapter_4/scenes/b2.jpg"]}


def _ep(tmp_path, panels, tool="panel_identity_ccip", verified=None):
    ep = tmp_path / "Chapter_9"
    (ep / "scenes").mkdir(parents=True)
    for fn in panels:
        (ep / "scenes" / fn).write_bytes(b"x")
    (ep / "manifest.identity.json").write_text(json.dumps({
        "panels": panels,
        "_meta": {"schema": 1, "tool": tool, "inputs": {"manifest.panels.understood.json": "abc"},
                  "backend": "ccip", "protagonist": "Kim Dokja", "verified_by": verified}}))
    ex = tmp_path / "s.exemplars.json"
    ex.write_text(json.dumps(EX))
    return ep, ex


def _chat(answers, calls):
    def chat(model, think, options, messages):
        calls.append(messages[0])
        name = "x"
        return {"message": {"content": answers.pop(0)}}
    return chat


REC = lambda cand, others=1: {"names": [], "others": others, "mc": False, "cand": cand,
                              "heads": 1, "diff": 0.05 if cand else 0.3}


def test_gemma_confirms_only_candidates_and_names_the_protagonist(tmp_path):
    ep, ex = _ep(tmp_path, {"p1.jpg": REC(True), "p2.jpg": REC(True), "p3.jpg": REC(False)})
    calls = []
    got = pi.verify_candidates(ep, ex, chat=_chat(['{"people": ["A"]}', '{"people": ["B"]}'], calls),
                               load=lambda p: b"img")
    P = got["panels"]
    assert len(calls) == 2                                 # p3 is no candidate
    assert len(calls[0]["images"]) == 5                    # 2 + 2 exemplars + the panel
    assert P["p1.jpg"]["names"] == ["Kim Dokja"] and P["p1.jpg"]["mc"]
    assert P["p1.jpg"]["others"] == 0 and P["p1.jpg"]["gemma"] == "A"
    assert P["p2.jpg"]["names"] == [] and not P["p2.jpg"]["mc"] and P["p2.jpg"]["gemma"] == "B"
    assert "gemma" not in P["p3.jpg"]
    disk = json.loads((ep / "manifest.identity.json").read_text())
    assert disk["panels"] == P
    meta = disk["_meta"]
    assert meta["verified_by"] == "gemma" and meta["exemplars"] == "s.exemplars.json"
    assert meta["inputs"] == {"manifest.panels.understood.json": "abc"}   # provenance kept
    assert meta["backend"] == "ccip" and meta["protagonist"] == "Kim Dokja"


def test_a_failed_or_empty_answer_confirms_nobody(tmp_path):
    """The 2026-10-10 GPU hang answered 200 with EMPTY content for 90 s."""
    ep, ex = _ep(tmp_path, {"p1.jpg": REC(True), "p2.jpg": REC(True)})

    def boom(**kw):
        raise TimeoutError("hung")
    got = pi.verify_candidates(ep, ex, chat=_chat(["", "nonsense"], []), load=lambda p: b"i")
    assert not any(r["mc"] for r in got["panels"].values())
    ep2, ex2 = _ep(tmp_path / "b", {"p1.jpg": REC(True)})
    got = pi.verify_candidates(ep2, ex2, chat=boom, load=lambda p: b"i")
    assert got["panels"]["p1.jpg"]["names"] == []


def test_verify_is_skipped_without_a_usable_setup(tmp_path):
    calls = []
    ch = _chat(['{"people": ["A"]}'] * 5, calls)
    # no exemplars file
    ep, _ = _ep(tmp_path, {"p1.jpg": REC(True)})
    assert pi.verify_candidates(ep, tmp_path / "missing.json", chat=ch) is None
    # exemplars must be exactly 2 + 2 (the measured prompt shows images 1-4)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"protagonist": EX["protagonist"][:1], "decoy": EX["decoy"]}))
    assert pi.verify_candidates(ep, bad, chat=ch) is None
    # not a picture-candidate identity (the gemma backend's own file)
    ep2, ex2 = _ep(tmp_path / "g", {"p1.jpg": REC(True)}, tool="panel_identity")
    assert pi.verify_candidates(ep2, ex2, chat=ch) is None
    # already verified: no second pass, no model calls
    ep3, ex3 = _ep(tmp_path / "v", {"p1.jpg": REC(True)}, verified="gemma")
    assert pi.verify_candidates(ep3, ex3, chat=ch) is None
    assert calls == []


def test_without_a_protagonist_name_nothing_is_named(tmp_path):
    ep, ex = _ep(tmp_path, {"p1.jpg": REC(True)})
    d = json.loads((ep / "manifest.identity.json").read_text())
    d["_meta"]["protagonist"] = None
    (ep / "manifest.identity.json").write_text(json.dumps(d))
    got = pi.verify_candidates(ep, ex, chat=_chat(['{"people": ["A"]}'], []), load=lambda p: b"i")
    assert got["panels"]["p1.jpg"]["names"] == [] and got["panels"]["p1.jpg"]["gemma"] == "A"


def test_check_series_samples_candidates_and_reports(tmp_path):
    """The per-series validation run (worker job identity_check): N random
    picture candidates across the series, gemma's verdict on each, a report
    and a grading sheet. Exemplar panels are never sampled."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_identity_sweep import _series, _seams
    import identity_sweep as sw
    sd, who = _series(tmp_path, "s", [(30, 4)] * 6)
    h, e = _seams(who)
    sw.sweep_series(sd, heads_fn=h, embed_fn=e)
    ps = sorted(who)
    ex = tmp_path / "s.exemplars.json"
    ex.write_text(json.dumps({"protagonist": ps[:2], "decoy": ps[-2:]}))
    seen = []

    def chat(model, think, options, messages):
        panel = messages[0]["images"][-1].decode()
        seen.append(panel)
        return {"message": {"content": '{"people": ["%s"]}' % ("A" if who[panel] == "mc" else "OTHER")}}

    rep = pi.check_series(sd, ex, n=12, out_dir=tmp_path / "out", chat=chat,
                          load=lambda p: str(p).encode())
    assert rep["n"] == 12 == len(seen) and rep["confirmed"] == 12
    assert not set(seen) & set(ps[:2] + ps[-2:])
    assert all(r["d"] < 0.10 for r in rep["results"])
    assert (tmp_path / "out" / "s.json").exists() and (tmp_path / "out" / "s_sheet.jpg").exists()
