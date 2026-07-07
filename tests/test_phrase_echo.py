"""Round-2 E5: PHRASE ECHO — a near-verbatim thought repeated across two
segments (real case g0020_p01 vs g0024_p12). prep_qa.phrase_echo_flags is a
cheap deterministic WARN net: two segments within 8 of each other sharing a
>= 6-word verbatim (case/punct-normalized) run. Heal-target at WARN severity
(narration_heal special-cases it like chrome_narration)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import tools.narration_heal as nh

_SPEC = importlib.util.spec_from_file_location(
    "prep_qa",
    Path(__file__).resolve().parent.parent / "tools" / "prep_qa.py",
)
pq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pq)  # type: ignore[union-attr]


def _beats(*group_lines):
    return {"beats": [
        {"group_id": gid, "segments": [
            {"span": [f"p{gid:06d}.jpg"], "line": line}]}
        for gid, line in group_lines]}


ECHO_A = "He wonders if this strange new power can really save him now."
ECHO_B = "Again he wonders if this strange new power can really save him."


def test_fires_on_near_verbatim_thought_within_window():
    flags = pq.phrase_echo_flags(_beats((20, ECHO_A), (21, "The dust "
                                 "settles over the courtyard."), (24, ECHO_B)))
    assert [f["code"] for f in flags] == ["phrase_echo"]
    f = flags[0]
    assert f["severity"] == pq.WARN
    assert f["segment_id"] == "g0024"          # the LATER segment heals
    assert "g0020" in f["detail"]


def test_case_and_punctuation_normalized():
    flags = pq.phrase_echo_flags(_beats(
        (1, "THE BLADE FINDS ITS MARK IN THE DARK tonight."),
        (2, "...the blade finds its mark, in the dark — again.")))
    assert [f["code"] for f in flags] == ["phrase_echo"]


def test_silent_on_short_overlaps_and_distinct_lines():
    # 5 shared words < 6 -> no flag; fully distinct lines -> no flag
    flags = pq.phrase_echo_flags(_beats(
        (1, "He grips the hidden blade tight."),
        (2, "He grips the hidden blade and waits."),
        (3, "Steel answers steel in the dark.")))
    assert flags == []


def test_silent_beyond_the_window():
    fillers = [
        "Steel rings against steel in the dark hall.",
        "A hooded shape drops from the rafters.",
        "Blood beads along the shallow cut.",
        "The prince staggers toward the broken door.",
        "Somewhere outside, a horn sounds twice.",
        "Dust curls through the shattered window.",
        "His grip tightens around the hidden hilt.",
        "The last candle gutters and dies.",
    ]
    rows = [(1, ECHO_A)] + [(i + 2, t) for i, t in enumerate(fillers)] \
        + [(10, ECHO_B)]                       # 9 segments apart > window 8
    assert pq.phrase_echo_flags(_beats(*rows)) == []


def test_one_flag_per_offending_segment():
    flags = pq.phrase_echo_flags(_beats((1, ECHO_A), (2, ECHO_A),
                                        (3, "The gate finally opens.")))
    assert len(flags) == 1


def test_phrase_echo_heals_at_warn_severity():
    corr = nh.corrections_from_qa({"flags": [
        {"code": "phrase_echo", "severity": "WARN", "segment_id": "g0024",
         "detail": "repeats g0020's phrase"}]})
    assert 24 in corr and "fresh wording" in corr[24].lower()
