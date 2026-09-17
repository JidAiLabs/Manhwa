"""identity_census: which finished chapters NAME the wrong character?

The owner asked "how many chapters have this issue?". Comparing the names the
narration used against what the IMAGE pass confirms answers it without
re-narrating. The two counts stay apart on purpose: the pass recognises ~65% of
a lead's panels, so "nobody confirmed there" is mostly its own blind spot and
must never inflate the headline.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "identity_census",
    Path(__file__).resolve().parent.parent / "tools" / "identity_census.py")
ic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ic)  # type: ignore[union-attr]

REG = {"cast": [
    {"canonical_name": "our protagonist", "aliases": ["Dokja", "Dokja Kim"],
     "is_protagonist": True},
    {"canonical_name": "Namwoon Kim", "aliases": ["Namwoon"]},
]}


def _beats(*lines):
    return {"beats": [{"group_id": 1, "segments": [
        {"span": list(span), "line": line} for line, span in lines]}]}


def test_a_line_naming_someone_the_image_contradicts_is_counted():
    """ORV Ep6: 'Namwoon Kim winces...' over a panel showing Dokja."""
    beats = _beats(("Namwoon Kim winces, wondering if he should use that power.",
                    ["p000035.jpg"]))
    identity = {"panels": {"p000035.jpg": {"names": ["our protagonist"],
                                           "others": 0}}}
    got = ic.audit_chapter(beats, identity, REG)
    assert got["contradicted"] == 1 and got["unconfirmed"] == 0
    assert got["lines"][0]["named"] == "Namwoon Kim"
    assert got["lines"][0]["confirmed"] == ["our protagonist"]


def test_a_correct_name_and_an_alias_both_pass():
    beats = _beats(("Dokja steadies himself.", ["p1.jpg"]),
                   ("Namwoon lunges at him.", ["p2.jpg"]))
    identity = {"panels": {
        "p1.jpg": {"names": ["our protagonist"], "others": 0},
        "p2.jpg": {"names": ["Namwoon Kim"], "others": 1}}}
    assert ic.audit_chapter(beats, identity, REG) == {
        "contradicted": 0, "unconfirmed": 0, "lines": []}


def test_a_panel_the_pass_could_not_confirm_is_kept_apart():
    beats = _beats(("Dokja steadies himself.", ["p1.jpg"]))
    identity = {"panels": {"p1.jpg": {"names": [], "others": 2}}}
    got = ic.audit_chapter(beats, identity, REG)
    assert got["contradicted"] == 0 and got["unconfirmed"] == 1


def test_a_line_naming_nobody_is_not_counted_at_all():
    beats = _beats(("The white-haired guy swings at him.", ["p1.jpg"]))
    identity = {"panels": {"p1.jpg": {"names": ["our protagonist"], "others": 1}}}
    assert ic.audit_chapter(beats, identity, REG)["lines"] == []


def test_names_are_word_bounded():
    by = {"Namwoon Kim": ["Namwoon Kim", "Namwoon"]}
    assert ic.names_in_line("Namwoon lunges", by) == ["Namwoon Kim"]
    assert ic.names_in_line("namwoonish grin", by) == []


def test_a_line_spanning_several_panels_needs_the_name_in_none_of_them():
    """A span confirms across ALL its panels: the name must be absent from
    every one before the line counts as contradicted."""
    beats = _beats(("Dokja and the boy face off.", ["p1.jpg", "p2.jpg"]))
    identity = {"panels": {
        "p1.jpg": {"names": ["Namwoon Kim"], "others": 0},
        "p2.jpg": {"names": ["our protagonist"], "others": 0}}}
    assert ic.audit_chapter(beats, identity, REG)["contradicted"] == 0


def test_shards_split_the_chapters_without_overlap_or_gaps():
    """Four shards fit the Mini's ollama parallelism (~21 h serial -> ~5 h)."""
    eps = ["Episode_%d" % i for i in range(10)]
    shards = [eps[i - 1::4] for i in (1, 2, 3, 4)]
    assert sorted(sum(shards, [])) == sorted(eps)
    assert all(len(set(s)) == len(s) for s in shards)
    assert sum(len(s) for s in shards) == len(eps)
