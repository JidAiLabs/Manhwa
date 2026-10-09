"""The identity gate after image identity (2026-10-10): positive evidence only.

A descriptive actor-noun is rewritten to the protagonist only on a SOLO span:
the picture confirms him and nobody else is drawn, captioned or speaking on
every panel the line voices. Everything the word-matching gate used to rewrite
(protagonist handles, keyword figures, all-unknown spans) is left alone — it
was right ~1 time in 4 (54 graded, 2026-10-09)."""
import json
from pathlib import Path

import tools.cast_identity as ci
import tools.identity_gate as ig
import tools.narration_punchup as npu

MC = "Kim Dokja"
NM = {"guard": {"the guard"}, "assassin": {"unnamed assassin"},
      "namwoon": {"Namwoon Kim"}, "kim": {"Namwoon Kim", MC},
      "dokja": {MC}, "reader": {MC}}
PROT = {MC}


def _u(fn, subjects=("a young man with dark hair",), kind="story", dialogue=""):
    return {"scene_file": fn, "panel_kind": kind, "subjects": list(subjects),
            "dialogue": dialogue, "description": "", "action": ""}


def _rec(mc=True, heads=1, others=0):
    return {"names": [MC] if mc else [], "others": others, "mc": mc,
            "heads": heads, "diff": 0.08 if mc else 0.3}


def _gate(line, span, understood, identity, text=None, figures=None):
    ubf = {u["scene_file"]: u for u in understood}
    solo = ig.solo_mc_resolver({"panels": identity}, ubf, text or {})
    beat = {"group_id": 1, "segments": [{"span": list(span), "line": line}],
            "narration": line}
    rw = ig.enforce_actor_handles(beat, figures or {}, NM, PROT, solo_mc=solo)
    return beat["segments"][0]["line"], rw


SOLO = [_u("p1.jpg")]
SOLO_ID = {"p1.jpg": _rec()}


def test_descriptive_noun_on_a_solo_span_becomes_the_protagonist():
    line, rw = _gate("The guard draws his sword.", ["p1.jpg"], SOLO, SOLO_ID)
    assert line == "Kim Dokja draws his sword." and rw


def test_a_proper_name_is_never_rewritten():
    for text in ("Namwoon smirks at the screen.", "Namwoon Kim smirks."):
        assert _gate(text, ["p1.jpg"], SOLO, SOLO_ID)[0] == text


def test_each_veto_leaves_the_line_alone():
    line = "The guard draws his sword."
    vetoes = {
        "speech on the panel": (SOLO, SOLO_ID, {"p1.jpg": "HALT! WHO GOES THERE"}),
        "dialogue field": ([_u("p1.jpg", dialogue="Halt!")], SOLO_ID, {}),
        "two heads": (SOLO, {"p1.jpg": _rec(heads=2, others=1)}, {}),
        "two people": ([_u("p1.jpg", subjects=("a young man", "a guard"))], SOLO_ID, {}),
        "a crowd": ([_u("p1.jpg", subjects=("a crowd of soldiers",))], SOLO_ID, {}),
        "not confirmed": (SOLO, {"p1.jpg": _rec(mc=False)}, {}),
        "no identity record": (SOLO, {}, {}),
        "caption panel": ([_u("p1.jpg", kind="caption")], SOLO_ID, {}),
    }
    for why, (und, ident, text) in vetoes.items():
        assert _gate(line, ["p1.jpg"], und, ident, text)[0] == line, why


def test_a_person_free_panel_in_the_span_does_not_veto_but_someone_else_does():
    und = [_u("p1.jpg"), _u("p2.jpg", subjects=("a burning building",))]
    ident = {"p1.jpg": _rec()}
    assert _gate("The guard runs.", ["p1.jpg", "p2.jpg"], und, ident)[0] == \
        "Kim Dokja runs."
    und = [_u("p1.jpg"), _u("p2.jpg", subjects=("a guard in armor",))]
    ident = {"p1.jpg": _rec(), "p2.jpg": _rec(mc=False)}
    assert _gate("The guard runs.", ["p1.jpg", "p2.jpg"], und, ident)[0] == \
        "The guard runs."


def test_a_caption_folded_into_the_line_vetoes():
    """The line may voice a neighbouring caption the span does not list
    (prep_qa._covered_panels); its words are about whoever the caption says."""
    und = [_u("p0.jpg", subjects=(), kind="caption", dialogue="The guard approached."),
           _u("p1.jpg")]
    assert _gate("The guard approaches.", ["p1.jpg"], und, SOLO_ID)[0] == \
        "The guard approaches."


def test_the_protagonists_own_noun_and_his_handles_are_left_alone():
    assert _gate("The reader frowns.", ["p1.jpg"], SOLO, SOLO_ID)[0] == "The reader frowns."
    # Rule 1 is gone: a protagonist handle is never re-pointed by the gate
    figs = {"p2.jpg": [{"name": "unnamed assassin", "evidence": "a masked figure"}]}
    und = [_u("p2.jpg", subjects=("a masked figure",))]
    line, rw = _gate("Our guy slips through the smoke.", ["p2.jpg"], und,
                     {"p2.jpg": _rec(mc=False)}, figures=figs)
    assert line == "Our guy slips through the smoke." and rw == []


def test_without_image_identity_the_gate_rewrites_nothing_but_dead_actors():
    figs = {"p1.jpg": [{"name": MC, "evidence": "dark hair"}]}
    beat = {"group_id": 1, "segments": [{"span": ["p1.jpg"],
                                         "line": "The assassin's eyes burn."}]}
    assert ig.enforce_actor_handles(beat, figs, NM, PROT) == []
    assert beat["segments"][0]["line"] == "The assassin's eyes burn."


# ---- consumers -------------------------------------------------------------

CAST = {"cast": [
    {"canonical_name": MC, "is_protagonist": True,
     "visual_description": "a young man with dark hair"},
    {"canonical_name": "Namwoon Kim",
     "visual_description": "a young man with white hair"}]}


def test_page_names_survive_image_identity():
    """Without this, flipping to image identity silently strips every side
    character's name: only the protagonist is confirmed from pictures."""
    u = {"panels": [{"scene_file": "p1.jpg", "subjects": ["a man", "a man"],
                     "dialogue": "Namwoon, get down!"}]}
    ident = {"panels": {"p1.jpg": {"names": [MC], "others": 1}}}
    figs = ci.resolve_figures_by_file(u, CAST, identity=ident)["p1.jpg"]
    assert {"name": MC, "evidence": "image"} in figs
    assert any(f["name"] == "Namwoon Kim" and f["evidence"].startswith("page")
               for f in figs)
    # appearance never names anyone: no page token, no page figure
    u2 = {"panels": [{"scene_file": "p1.jpg", "subjects": ["a man with white hair"]}]}
    names = [f["name"] for f in ci.resolve_figures_by_file(
        u2, CAST, identity={"panels": {"p1.jpg": {"names": [], "others": 1}}})["p1.jpg"]]
    assert names == ["unknown"]


def test_a_missing_identity_manifest_is_not_an_empty_one():
    """prep_qa loads a missing manifest as {} — that read as 'the image pass
    confirmed nobody on any panel' and silenced actor_mismatch fleet-wide."""
    u = {"panels": [{"scene_file": "p1.jpg", "subjects": ["a young man with white hair"]}]}
    assert ci.resolve_figures_by_file(u, CAST, identity={}) == \
        ci.resolve_figures_by_file(u, CAST)


def test_punchup_backstop_reads_the_image_identity(tmp_path):
    ep = tmp_path / "Chapter_1"
    ep.mkdir()
    (ep / "manifest.identity.json").write_text(json.dumps(
        {"panels": {"p1.jpg": _rec()}}))
    assert npu._identity_obj(str(ep))["panels"]["p1.jpg"]["mc"]
    assert npu._identity_obj(str(tmp_path / "nowhere")) is None
    cast = {"cast": [{"canonical_name": MC, "is_protagonist": True,
                      "visual_description": "a young man with dark hair"},
                     {"canonical_name": "the guard", "visual_description": "armor"}]}
    out = {"beats": [{"group_id": 1, "segments": [
        {"span": ["p1.jpg"], "line": "The guard draws his sword."}],
        "narration": "The guard draws his sword."}]}
    stats = npu.apply_post_punchup_backstop(
        out, cast, {"p1.jpg": {"ocr_clean": ""}}, {"p1.jpg": _u("p1.jpg")},
        identity={"panels": {"p1.jpg": _rec()}})
    assert stats["actor_handles_rewritten"] == 1
    assert out["beats"][0]["segments"][0]["line"].startswith(MC)


def test_heal_writer_call_carries_the_identity(tmp_path, monkeypatch):
    import studio.worker as w
    ep = tmp_path / "Chapter_1"
    ep.mkdir()
    (ep / "manifest.beats.json").write_text(json.dumps({"beats": []}))
    corr = tmp_path / "corr.json"
    corr.write_text("[]")
    seen = []
    monkeypatch.setattr(w, "_stream", lambda cmd, log, **kw: seen.append(cmd) or 0)

    class Cfg:
        beats_model, punchup, semantic_heal = "m", "off", False

    try:
        w._regen_flagged(ep, Cfg(), str(corr), {}, open(tmp_path / "log", "w"))
    except Exception:
        pass                      # later steps need real manifests; argv is what we test
    writer = next(c for c in seen if any(str(x).endswith("gemini_narrative_pass.py") for x in c))
    i = writer.index("--identity")
    assert writer[i + 1] == str(ep / "manifest.identity.json")
