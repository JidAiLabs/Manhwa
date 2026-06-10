"""
tests/test_verbatim_script.py

TDD for the `gemini_verbatim` narration source in tools/script_expander.py —
the beats.narration -> script materializer (handover gap #2) that lets the
image-grounded Gemini line (A/B Variant B winner) be voiced WITHOUT OpenAI.

Also covers the TTS caps-normalization pass (handover next-action #1): quoted
dialogue arrives as verbatim OCR shout-caps ("KILL HIM!"); TTS engines must get
sentence case, with the shout mapped to mood intensity instead of literal caps.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent / "tools"

_SPEC = importlib.util.spec_from_file_location("script_expander", _TOOLS / "script_expander.py")
se = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(se)  # type: ignore[union-attr]

_LT_SPEC = importlib.util.spec_from_file_location("local_tts", _TOOLS / "local_tts_from_manifest.py")
lt = importlib.util.module_from_spec(_LT_SPEC)
_LT_SPEC.loader.exec_module(lt)  # type: ignore[union-attr]


# ---- caps normalization (ALL-CAPS OCR dialogue -> sentence case) -----------

def test_caps_quoted_shout_sentence_cased():
    text = 'The Assassins sneer, "KILL HIM!"'
    out, had = se.normalize_caps_for_tts(text, {})
    assert out == 'The Assassins sneer, "Kill him!"'
    assert had is True


def test_caps_cast_proper_nouns_restored():
    proper = {"prince": "Prince", "cheon": "Cheon"}
    out, had = se.normalize_caps_for_tts('"YOU DIE HERE, PRINCE CHEON."', proper)
    assert out == '"You die here, Prince Cheon."'
    assert had is True


def test_caps_i_contractions_restored():
    out, had = se.normalize_caps_for_tts('"I CAN\'T MOVE… WHAT IS THIS?"', {})
    assert out == '"I can\'t move… what is this?"'
    assert had is True


def test_caps_keeps_short_acronyms_and_single_letters():
    out, had = se.normalize_caps_for_tts("THE AI FLICKERS. A blade gleams.", {})
    assert out == "The AI flickers. A blade gleams."
    assert had is True


def test_caps_untouched_text_unchanged():
    text = "Prince Cheon strides forward, calm and cold."
    out, had = se.normalize_caps_for_tts(text, {})
    assert out == text
    assert had is False


def test_caps_attribution_after_quote_not_capitalized():
    out, _ = se.normalize_caps_for_tts('"KILL HIM!" THE ASSASSINS LUNGE.', {})
    assert out == '"Kill him!" the assassins lunge.'


def test_caps_sentence_boundary_between_normalized_words():
    out, _ = se.normalize_caps_for_tts("HE FALLS. HE RISES.", {})
    assert out == "He falls. He rises."


# ---- shout-caps / scene intensity -> mood tag escalation -------------------

def test_intensity_rank_takes_max_of_scene_selection():
    beat = {"scene_selection": [
        {"scene_file": "a", "intensity": "tense"},
        {"scene_file": "b", "intensity": "explosive"},
    ]}
    assert se._intensity_rank_for_beat(beat) == 3


def test_intensity_rank_zero_when_missing():
    assert se._intensity_rank_for_beat({}) == 0
    assert se._intensity_rank_for_beat({"scene_selection": [
        {"scene_file": "a", "intensity": "weird"}]}) == 0


def test_escalate_neutral_tags_only():
    assert se._escalate_tag_for_intensity("serious", 3) == "excited"
    assert se._escalate_tag_for_intensity("calm", 2) == "tense"
    assert se._escalate_tag_for_intensity("serious", 1) == "serious"
    # deliberate moods are never overridden by intensity
    assert se._escalate_tag_for_intensity("sad", 3) == "sad"
    assert se._escalate_tag_for_intensity("angry", 3) == "angry"
    assert se._escalate_tag_for_intensity("whisper", 3) == "whisper"


# ---- gemini_verbatim section builder (beats.narration -> script section) ---

def _chunk_and_payload():
    chunk = [
        {
            "group_id": 7, "beat_id": 1,
            "beat_title": "Ambush",
            "narration": 'The Assassins close in. One sneers, "KILL HIM!"',
            "what_happens": "Assassins surround the prince.",
            "hook": "Blades rise.",
            "mood_words": [],
            "scene_selection": [{"scene_file": "a.jpg", "role": "keep",
                                 "bubble_mode": "shout", "intensity": "explosive"}],
        },
        {
            "group_id": 8, "beat_id": 2,
            "beat_title": "No way out",
            "narration": "",
            "what_happens": "Prince Cheon backs against the wall.",
            "hook": "He has one chance.",
            "mood_words": ["tense"],
        },
        {
            "group_id": 9, "beat_id": 3,
            "beat_title": "Beat",
            "what_happens": "Unable to parse model output.",
            "error": "parse_failed_after_retries",
            "mood_words": ["uncertain"],
        },
    ]
    payload = {"section_index": 0, "word_target": 120, "beats": [
        {"beat_id": 1, "group_id": 7, "allowed_scene_files": ["a.jpg"],
         "scene_files": ["a.jpg"], "ocr_snippets_by_scene_file": {}},
        {"beat_id": 2, "group_id": 8, "allowed_scene_files": ["b.jpg"],
         "scene_files": ["b.jpg"], "ocr_snippets_by_scene_file": {}},
        {"beat_id": 3, "group_id": 9, "allowed_scene_files": ["c.jpg"],
         "scene_files": ["c.jpg"], "ocr_snippets_by_scene_file": {}},
    ]}
    return chunk, payload


def test_verbatim_paragraphs_from_narration_with_fallbacks():
    chunk, payload = _chunk_and_payload()
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="unknown")
    paras = sec["script_paragraphs"]
    assert len(paras) == 3
    # narration verbatim, shout-caps normalized
    assert paras[0] == 'The Assassins close in. One sneers, "Kill him!"'
    # empty narration -> what_happens
    assert paras[1] == "Prince Cheon backs against the wall."
    # error beat must never voice the parse-failure placeholder
    assert paras[2] == "The scene continues."


def test_verbatim_tts_equals_script_text_with_valid_tag():
    chunk, payload = _chunk_and_payload()
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="unknown")
    for para, tts in zip(sec["script_paragraphs"], sec["tts_paragraphs_v3"]):
        tag, rest = se._split_leading_bracket_tag(tts)
        assert tag in se.V3_VALID_TAGS
        assert rest == para


def test_verbatim_explosive_beat_gets_escalated_tag():
    chunk, payload = _chunk_and_payload()
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="unknown")
    tag0, _ = se._split_leading_bracket_tag(sec["tts_paragraphs_v3"][0])
    # beat 1: neutral mood_words but explosive panel + shout-caps quote
    assert tag0 == "excited"


def test_verbatim_section_is_valid_with_one_shot_per_beat():
    chunk, payload = _chunk_and_payload()
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="unknown")
    assert se._validate_section_json(sec)
    shots = sec["shots"]
    assert [s["group_id"] for s in shots] == [7, 8, 9]
    assert shots[0]["scene_files"] == ["a.jpg"]
    assert sec["cliffhanger_line"]  # taken from the last beat's hook chain


# ---- CLI end-to-end: gemini_verbatim needs NO OpenAI key -------------------

def test_cli_verbatim_runs_without_openai_key(tmp_path):
    beats = {"beats": [
        {
            "group_id": 7,
            "scene_files": ["a.jpg"],
            "beat_title": "Ambush",
            "narration": 'One Assassin sneers, "YOU DIE HERE, PRINCE CHEON!"',
            "what_happens": "Assassins surround the prince.",
            "hook": "Blades rise.",
            "mood_words": [],
            "scene_selection": [{"scene_file": "a.jpg", "role": "keep",
                                 "bubble_mode": "shout", "intensity": "explosive"}],
        },
        {
            "group_id": 8,
            "scene_files": ["b.jpg"],
            "beat_title": "Cornered",
            "narration": "Prince Cheon backs against the cold stone wall.",
            "what_happens": "He is cornered.",
            "hook": "One chance remains.",
            "mood_words": ["tense"],
        },
    ]}
    cast = {"cast": [{"id": "protagonist", "canonical_name": "Prince Cheon",
                      "aliases": [], "role": "protagonist",
                      "visual_description": "young man", "is_protagonist": True}]}
    beats_p = tmp_path / "manifest.beats.json"
    cast_p = tmp_path / "manifest.cast.json"
    out_p = tmp_path / "manifest.script.json"
    beats_p.write_text(json.dumps(beats))
    cast_p.write_text(json.dumps(cast))

    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    r = subprocess.run(
        [sys.executable, str(_TOOLS / "script_expander.py"),
         "--beats", str(beats_p), "--out", str(out_p),
         "--narration-source", "gemini_verbatim", "--cast", str(cast_p)],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"expander failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"

    obj = json.loads(out_p.read_text())
    assert obj["narration_source"] == "gemini_verbatim"
    assert obj["stats"]["usage"]["calls"] == 0          # zero LLM calls
    assert obj["stats"]["usage"]["est_cost_usd"] == 0

    sec = obj["sections"][0]
    # cast-cased, shout-caps normalized, voiced verbatim
    assert sec["script_paragraphs"][0] == 'One Assassin sneers, "You die here, Prince Cheon!"'
    assert sec["script_paragraphs"][1] == "Prince Cheon backs against the cold stone wall."
    assert sec["shots"][0]["segment_id"] == "g0007_p00"
    assert sec["shots"][1]["segment_id"] == "g0008_p01"
    assert len(sec["tts_meta"]) == 2

    # the TTS adapters must see every paragraph under the same segment_ids
    items = lt.extract_items_from_manifest(obj, "tts_v3")
    assert [it["segment_id"] for it in items] == ["g0007_p00", "g0008_p01"]
    for it in items:
        assert "PRINCE CHEON" not in it["text"]         # caps never reach TTS
