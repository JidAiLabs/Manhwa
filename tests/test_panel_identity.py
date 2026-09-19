"""panel_identity: who is in a panel, decided from the IMAGE.

Word matching ("a young man with short dark hair" vs a text look) cannot tell
same-looking characters apart: ORV Ep6 narrated Dokja as "Namwoon Kim" 67 times
and 5 of 8 "MC" thumbnail refs were other men. Spike numbers (2026-09-17, 50
Ep6 panels + 8 tiles): word matching precision 0.69, gemma with TWO exemplar
candidates and an explicit OTHER 0.96 — while ONE candidate (0.70) forces
look-alikes onto the lead and FOUR collapse recall to 0.24 and invent the
extras. Two candidates is therefore a hard cap, not a default.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "panel_identity",
    Path(__file__).resolve().parent.parent / "tools" / "panel_identity.py")
pi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pi)  # type: ignore[union-attr]

REG = {"cast": [
    {"canonical_name": "our protagonist", "is_protagonist": True,
     "aliases": ["Dokja"], "exemplars": ["a1.jpg", "a2.jpg"]},
    {"canonical_name": "Namwoon Kim", "exemplars": ["b1.jpg", "b2.jpg"]},
    {"canonical_name": "Michio Shoji"},                      # no exemplars
    {"canonical_name": "Han Sooyoung", "exemplars": ["c1.jpg"]},
]}


def test_candidates_are_capped_at_two_protagonist_first():
    c = pi.candidates(REG)
    assert [x["name"] for x in c] == ["our protagonist", "Namwoon Kim"]
    assert [x["exemplars"] for x in c] == [["a1.jpg", "a2.jpg"], ["b1.jpg", "b2.jpg"]]
    assert pi.candidates({"cast": [{"canonical_name": "X"}]}) == []     # none carry exemplars


def test_prompt_is_the_measured_forced_choice():
    p = pi.build_prompt(["our protagonist", "Namwoon Kim"])
    assert "OTHER" in p and "FACE" in p
    assert "A" in p and "B" in p
    # the letters are the answer space; no character NAME leaks in (a name is a
    # hint the model can pattern-match instead of looking)
    assert "our protagonist" not in p and "Namwoon" not in p


def test_parse_maps_letters_to_names_and_counts_the_rest():
    names = ["our protagonist", "Namwoon Kim"]
    assert pi.parse_reply('{"people": ["A", "OTHER", "B"]}', names) == \
        {"names": ["our protagonist", "Namwoon Kim"], "others": 1}
    assert pi.parse_reply('{"people": ["other", "OTHER"]}', names) == \
        {"names": [], "others": 2}
    assert pi.parse_reply('{"people": ["a", "a"]}', names) == \
        {"names": ["our protagonist"], "others": 0}       # deduped


def test_an_unusable_reply_confirms_nobody():
    names = ["our protagonist", "Namwoon Kim"]
    for raw in ("", "no json here", '{"people": "A"}', '{"people": ["C", "Z"]}'):
        assert pi.parse_reply(raw, names) == {"names": [], "others": 0}, raw


def _stub_panels(tmp_path, kinds):
    ep = tmp_path / "ch"; (ep / "scenes").mkdir(parents=True)
    panels = []
    for fn, kind, subs in kinds:
        (ep / "scenes" / fn).write_bytes(b"jpg")
        panels.append({"scene_file": "scenes/" + fn, "panel_kind": kind,
                       "subjects": subs})
    (ep / "manifest.panels.understood.json").write_text(
        json.dumps({"panels": panels}))
    return ep


def test_identify_panels_asks_only_about_panels_with_people(tmp_path):
    ep = _stub_panels(tmp_path, [
        ("p1.jpg", "story", ["a young man"]),
        ("p2.jpg", "system", ["a blue window"]),      # system screen
        ("p3.jpg", "story", []),                      # nobody recorded
    ])
    asked = []

    def chat(*, model, messages, options=None, think=False):
        asked.append(len(messages[0]["images"]))
        return {"message": {"content": '{"people": ["A", "OTHER"]}'}}

    out = pi.identify_panels(str(ep), REG, chat=chat, load=lambda p: b"IMG")
    assert asked == [5]                      # 4 exemplars + the panel, once
    assert out["panels"] == {"p1.jpg": {"names": ["our protagonist"], "others": 1}}
    assert out["_meta"]["candidates"] == ["our protagonist", "Namwoon Kim"]


def test_a_failed_call_leaves_the_panel_unconfirmed_not_guessed(tmp_path):
    ep = _stub_panels(tmp_path, [("p1.jpg", "story", ["a young man"]),
                                 ("p2.jpg", "story", ["a young man"])])
    calls = {"n": 0}

    def chat(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("ollama stalled")
        return {"message": {"content": '{"people": ["B"]}'}}

    out = pi.identify_panels(str(ep), REG, chat=chat, load=lambda p: b"IMG")
    assert out["panels"]["p1.jpg"] == {"names": [], "others": 0}   # no guess
    assert out["panels"]["p2.jpg"] == {"names": ["Namwoon Kim"], "others": 0}


def test_no_exemplars_means_no_manifest_at_all(tmp_path):
    ep = _stub_panels(tmp_path, [("p1.jpg", "story", ["a young man"])])

    def chat(**kw):                                    # must never be called
        raise AssertionError("asked the model without exemplars")

    assert pi.identify_panels(str(ep), {"cast": [{"canonical_name": "X"}]},
                              chat=chat, load=lambda p: b"IMG") is None


# ---- a name is only trusted when the CHAPTER contains that character --------
# ORV Ep128 (2026-09-19): its cast is Michio Shoji, Yuseung Shin, Izumi...  no
# Namwoon. The forced A/B/OTHER question still handed out "B" on 22 of 58
# panels, and the identity gate then rewrote the chapter's real names toward
# them ("Michio Shoji Michio Shoji", "Namwoon Michio Shoji"). The decoy stays in
# the PROMPT — that is what keeps look-alikes off the lead — but its name is not
# trusted in a chapter the character is absent from.

def _cast(*names):
    return {"cast": [{"canonical_name": n} for n in names]}


def test_a_confirmed_name_absent_from_the_chapter_cast_becomes_another_person(tmp_path):
    ep = _stub_panels(tmp_path, [("p1.jpg", "story", ["a young man"])])
    (ep / "manifest.cast.json").write_text(
        __import__("json").dumps(_cast("our protagonist", "Michio Shoji")))

    def chat(**kw):
        return {"message": {"content": '{"people": ["A", "B"]}'}}

    out = pi.identify_panels(str(ep), REG, chat=chat, load=lambda p: b"IMG")
    # A (the protagonist) is in this chapter's cast; B (Namwoon) is not
    assert out["panels"]["p1.jpg"] == {"names": ["our protagonist"], "others": 1}
    assert out["_meta"]["trusted"] == ["our protagonist"]


def test_without_a_chapter_cast_every_confirmed_name_still_counts(tmp_path):
    ep = _stub_panels(tmp_path, [("p1.jpg", "story", ["a young man"])])

    def chat(**kw):
        return {"message": {"content": '{"people": ["B"]}'}}

    out = pi.identify_panels(str(ep), REG, chat=chat, load=lambda p: b"IMG")
    assert out["panels"]["p1.jpg"] == {"names": ["Namwoon Kim"], "others": 0}


# ---- the second candidate comes from THIS chapter's cast -------------------
# Only two candidates fit in one call (measured), and the registry may carry
# many. ORV Ep128 is a Michio chapter: asking about Namwoon there wastes the
# slot and the answer is untrusted anyway, so the slot goes to Michio.

REG3 = {"cast": [
    {"canonical_name": "our protagonist", "is_protagonist": True,
     "exemplars": ["a1.jpg", "a2.jpg"]},
    {"canonical_name": "Namwoon Kim", "exemplars": ["b1.jpg", "b2.jpg"]},
    {"canonical_name": "Michio Shoji", "exemplars": ["c1.jpg", "c2.jpg"]},
]}


def test_the_second_candidate_is_a_character_the_chapter_contains():
    assert [c["name"] for c in pi.candidates(
        REG3, chapter=_cast("our protagonist", "Michio Shoji", "Izumi"))] == \
        ["our protagonist", "Michio Shoji"]
    assert [c["name"] for c in pi.candidates(
        REG3, chapter=_cast("our protagonist", "Namwoon Kim"))] == \
        ["our protagonist", "Namwoon Kim"]


def test_with_nobody_else_in_the_chapter_a_decoy_is_still_asked():
    """The decoy is what keeps look-alikes off the lead (one candidate alone:
    precision 0.70). Its name is untrusted, which the census guard handles."""
    c = pi.candidates(REG3, chapter=_cast("our protagonist", "Yuseung Shin"))
    assert [x["name"] for x in c] == ["our protagonist", "Namwoon Kim"]


def test_identify_panels_asks_about_the_chapter_s_own_second_character(tmp_path):
    import json as _j
    ep = _stub_panels(tmp_path, [("p1.jpg", "story", ["a young man"])])
    (ep / "manifest.cast.json").write_text(
        _j.dumps(_cast("our protagonist", "Michio Shoji")))

    def chat(**kw):
        return {"message": {"content": '{"people": ["B"]}'}}

    out = pi.identify_panels(str(ep), REG3, chat=chat, load=lambda p: b"IMG")
    assert out["_meta"]["candidates"] == ["our protagonist", "Michio Shoji"]
    assert out["panels"]["p1.jpg"] == {"names": ["Michio Shoji"], "others": 0}
