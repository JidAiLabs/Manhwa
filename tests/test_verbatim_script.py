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
    assert se._escalate_tag_for_intensity("calm", 2) == "calm"
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


def test_verbatim_comic_beat_gets_energetic_tag():
    chunk = [{
        "group_id": 1,
        "beat_id": 1,
        "beat_title": "Bald joke",
        "narration": ("He loses it, pointing right at Heo Bong and howling, "
                      "'Where did all your hair go overnight?!'"),
        "mood_words": ["manic", "mocking", "humiliated", "intense"],
        "scene_files": ["a.jpg"],
        "scene_selection": [{"scene_file": "a.jpg", "role": "keep",
                             "intensity": "intense"}],
    }]
    payload = {"section_index": 0, "word_target": 120, "beats": [{
        "beat_id": 1,
        "group_id": 1,
        "allowed_scene_files": ["a.jpg"],
        "scene_files": ["a.jpg"],
    }]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="unknown")
    tag, _ = se._split_leading_bracket_tag(sec["tts_paragraphs_v3"][0])
    assert tag == "excited"


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


def test_verbatim_title_card_uses_story_hook_not_chapter_heading():
    chunk = [{
        "group_id": 5,
        "beat_id": 1,
        "beat_title": "Chapter Title Card",
        "narration": "As the truth surfaces, we reach Chapter 7: The Trap.",
        "what_happens": "The chapter title card appears.",
        "hook": "The truth is finally about to surface.",
        "mood_words": [],
        "scene_files": ["card.jpg"],
        "scene_selection": [{"scene_file": "card.jpg", "role": "redundant"}],
    }]
    payload = {"beats": [{
        "beat_id": 1,
        "group_id": 5,
        "allowed_scene_files": ["card.jpg"],
        "scene_files": ["card.jpg"],
        "ocr_snippets_by_scene_file": {},
    }]}

    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="unknown")

    assert sec["script_paragraphs"] == ["The truth is finally about to surface."]
    assert "Chapter" not in sec["tts_paragraphs_v3"][0]


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


# ---- Chunk 5: one segment per panel (panel_narration path) ------------------

def test_one_segment_per_panel_aligned():
    """Each panel_narration entry becomes its own paragraph + shot, aligned by
    construction (no positional guessing). Three panels with long-enough lines
    (> short_words=6) → three everything (merge does not fire)."""
    beats = [{"group_id": 7, "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg"],
              "panel_narration": [
                  {"scene_file": "p1.jpg", "line": "He steps into the arena and draws his sword."},
                  {"scene_file": "p2.jpg", "line": "The quest window flares bright inside his vision."},
                  {"scene_file": "p3.jpg", "line": "Numbers tick down toward the final threshold."}]}]
    payload = {"beats": [{"group_id": 7, "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg"]}]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=beats, payload=payload, word_target=120,
        genre_mode="action", proper_case=None, wpm=170)
    assert len(sec["shots"]) == 3
    assert [s.get("scene_files") for s in sec["shots"]] == [["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
    assert len(sec["script_paragraphs"]) == 3
    assert len(sec["tts_paragraphs_v3"]) == 3
    assert any("quest window" in p.lower() for p in sec["script_paragraphs"])


def test_cli_segment_ids_per_panel(tmp_path):
    """CLI: one group with 3 panel_narration entries using long lines (>6 words
    each) → segment_ids g0007_p00, g0007_p01, g0007_p02 (one per panel, no
    merge fires because each line is above the short_words=6 threshold)."""
    beats = {"beats": [
        {
            "group_id": 7,
            "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg"],
            "beat_title": "System Flash",
            "narration": "He steps into the arena. The quest window flares brightly. Numbers tick toward zero.",
            "what_happens": "System activates.",
            "hook": "The count begins.",
            "mood_words": ["serious"],
            "panel_narration": [
                {"scene_file": "p1.jpg", "line": "He steps into the arena and draws his blade."},
                {"scene_file": "p2.jpg", "line": "The quest window flares bright inside his vision."},
                {"scene_file": "p3.jpg", "line": "Numbers tick down toward the final threshold."},
            ],
        }
    ]}
    beats_p = tmp_path / "manifest.beats.json"
    out_p = tmp_path / "manifest.script.json"
    beats_p.write_text(json.dumps(beats))

    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    r = subprocess.run(
        [sys.executable, str(_TOOLS / "script_expander.py"),
         "--beats", str(beats_p), "--out", str(out_p),
         "--narration-source", "gemini_verbatim"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"expander failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"

    obj = json.loads(out_p.read_text())
    sec = obj["sections"][0]
    segment_ids = [s["segment_id"] for s in sec["shots"]]
    assert segment_ids == ["g0007_p00", "g0007_p01", "g0007_p02"]
    assert len(sec["script_paragraphs"]) == 3
    assert len(sec["tts_paragraphs_v3"]) == 3


def test_error_beat_with_valid_panel_narration_emits_one_shot_per_panel():
    """REGRESSION (panel-collapse): a group whose GROUP-level JSON parse failed
    still gets VALID per-panel lines backfilled (one per scene_file). The
    `error` flag must NOT discard them — every panel keeps its own shot, so the
    union of shown scene_files covers ALL panels (not collapsed to one
    'The scene continues.' stand-in showing only the first panel)."""
    beats = [{"group_id": 5, "error": "parse_failed_after_retries",
              "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg"],
              "panel_narration": [
                  {"scene_file": "p1.jpg", "line": "He steps onto the cracked arena floor."},
                  {"scene_file": "p2.jpg", "line": "The hidden quest window flares to life."},
                  {"scene_file": "p3.jpg", "line": "The countdown spirals toward the final number."}]}]
    payload = {"beats": [{"group_id": 5, "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg"]}]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=beats, payload=payload,
        word_target=120, genre_mode="action", proper_case=None, wpm=170,
        tts_merge_short=False)
    # one shot per panel — NOT collapsed to a single allowed[:1] stand-in
    assert len(sec["shots"]) == 3
    union = {f for s in sec["shots"] for f in (s.get("scene_files") or [])}
    assert union == {"p1.jpg", "p2.jpg", "p3.jpg"}
    joined = " ".join(sec["script_paragraphs"]).lower()
    assert "the scene continues" not in joined
    assert "quest window" in joined


def test_error_beat_with_only_placeholder_lines_folds_to_one_shot():
    """An error beat whose per-panel lines are empty/placeholder
    ('The scene continues.') has nothing real to voice — it still folds to a
    single fallback shot rather than emitting hollow per-panel cuts."""
    beats = [{"group_id": 6, "error": "parse_failed_after_retries",
              "scene_files": ["p1.jpg", "p2.jpg"],
              "panel_narration": [
                  {"scene_file": "p1.jpg", "line": "The scene continues."},
                  {"scene_file": "p2.jpg", "line": ""}]}]
    payload = {"beats": [{"group_id": 6, "scene_files": ["p1.jpg", "p2.jpg"]}]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=beats, payload=payload,
        word_target=120, genre_mode="action", proper_case=None, wpm=170)
    assert len(sec["shots"]) == 1
    joined = " ".join(sec["script_paragraphs"]).lower()
    assert "the scene continues" in joined


def test_legacy_path_unchanged_when_no_panel_narration():
    """Beats without panel_narration fall back to the legacy single-para-per-beat
    path. The existing _chunk_and_payload fixture has no panel_narration; confirm
    it still produces one shot per beat (groups 7, 8, 9)."""
    chunk, payload = _chunk_and_payload()
    # Confirm none of the test beats carry panel_narration
    assert all("panel_narration" not in b for b in chunk)
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="unknown")
    shots = sec["shots"]
    # Legacy: one shot per beat
    assert [s["group_id"] for s in shots] == [7, 8, 9]
    assert len(sec["script_paragraphs"]) == 3
    assert len(sec["tts_paragraphs_v3"]) == 3


# ---- Fix C: merge_short_panel_items unit tests -----------------------------

def test_merge_short_two_short_lines_become_one_bucket():
    """Two short lines (≤ short_words each) → one bucket with both scene_files
    and the joined line."""
    items = [
        ("He turns.", ["p1.jpg"]),
        ("Silence falls.", ["p2.jpg"]),
    ]
    result = se.merge_short_panel_items(items)
    assert len(result) == 1
    line, files = result[0]
    assert "He turns" in line
    assert "Silence falls" in line
    assert files == ["p1.jpg", "p2.jpg"]


def test_merge_short_long_line_stays_alone():
    """A 25-word line is above both short_words and max_words; stays solo."""
    long_line = "He draws his sword and faces the crowd as silence falls across the entire arena floor below him."
    items = [(long_line, ["p1.jpg"])]
    result = se.merge_short_panel_items(items)
    assert len(result) == 1
    assert result[0][0] == long_line


def test_merge_short_short_then_long_flushes():
    """Short line followed by long line: short is flushed alone when the long
    line would push the bucket over max_words."""
    short = "He waits."  # 2 words
    long = ("The assassins descend from the rooftop one by one, silent as death, "
            "blades already drawn and hungry for blood.")  # > 20 words
    items = [
        (short, ["p1.jpg"]),
        (long, ["p2.jpg"]),
    ]
    result = se.merge_short_panel_items(items)
    # short can't merge into long without exceeding max_words=20
    assert len(result) == 2
    assert result[0] == (short, ["p1.jpg"])
    assert result[1] == (long, ["p2.jpg"])


def test_merge_short_flatten_invariant():
    """Every input scene_file appears in exactly one output bucket (no drops,
    no duplicates, original order preserved)."""
    import itertools
    items = [
        ("Run.", ["a.jpg"]),
        ("Dodge.", ["b.jpg"]),
        ("Strike.", ["c.jpg"]),
        ("Fall.", ["d.jpg"]),
        ("Rise.", ["e.jpg"]),
    ]
    result = se.merge_short_panel_items(items)
    all_input_files = [f for _, files in items for f in files]
    all_output_files = [f for _, files in result for f in files]
    assert all_output_files == all_input_files, (
        f"panel order/coverage broken: {all_output_files} != {all_input_files}"
    )


def test_merge_short_max_panels_cap_respected():
    """Four short lines, max_panels=3 → the fourth cannot join the first bucket."""
    items = [
        ("Go.", ["p1.jpg"]),
        ("Run.", ["p2.jpg"]),
        ("Hide.", ["p3.jpg"]),
        ("Wait.", ["p4.jpg"]),   # would push to 4 panels → new bucket
    ]
    result = se.merge_short_panel_items(items, max_panels=3)
    # All 4 files must still be present
    all_files = [f for _, files in result for f in files]
    assert all_files == ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"]
    # The fourth panel must be in a separate bucket (not the first)
    assert len(result) >= 2
    first_bucket_files = result[0][1]
    assert "p4.jpg" not in first_bucket_files


def test_merge_short_empty_input():
    assert se.merge_short_panel_items([]) == []


# ---- Fix C: integration tests via _build_verbatim_section ------------------

def _panel_beat(gid, panels):
    """Helper: build a beat dict with panel_narration entries."""
    return {
        "group_id": gid,
        "scene_files": [p["scene_file"] for p in panels],
        "beat_title": "Beat",
        "narration": " ".join(p["line"] for p in panels),
        "what_happens": "Things happen.",
        "hook": "Something shifts.",
        "mood_words": [],
        "panel_narration": panels,
    }


def test_per_panel_no_merge_four_short_lines_four_shots():
    """C1: with the short-line merge gated OFF (default), four 1-word panel lines
    become FOUR shots (strict 1:1), not a merged few. The parallel-list contract
    len(script_paragraphs) == len(shots) == len(tts_paragraphs_v3) holds, and
    every input scene_file is its own shot."""
    panels = [
        {"scene_file": "p1.jpg", "line": "Run."},
        {"scene_file": "p2.jpg", "line": "Dodge."},
        {"scene_file": "p3.jpg", "line": "Strike."},
        {"scene_file": "p4.jpg", "line": "Fall."},
    ]
    chunk = [_panel_beat(1, panels)]
    payload = {"beats": [{"group_id": 1, "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"]}]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="action")

    shots = sec["shots"]
    paras = sec["script_paragraphs"]
    tts = sec["tts_paragraphs_v3"]
    assert len(paras) == len(shots) == len(tts), (
        f"contract broken: paras={len(paras)} shots={len(shots)} tts={len(tts)}"
    )
    assert len(shots) == 4, f"merge must be OFF: expected 4 shots, got {len(shots)}"
    assert [s.get("scene_files") for s in shots] == [
        ["p1.jpg"], ["p2.jpg"], ["p3.jpg"], ["p4.jpg"]]


def test_merge_integration_long_lines_no_merge():
    """Long panel lines (>6 words each) → one shot per panel (no over-merge)."""
    panels = [
        {"scene_file": "p1.jpg",
         "line": "He draws his sword and faces the assembled crowd below."},
        {"scene_file": "p2.jpg",
         "line": "The assassins fan out across the rooftop one by one."},
        {"scene_file": "p3.jpg",
         "line": "Prince Cheon stands alone against the entire inner circle."},
    ]
    chunk = [_panel_beat(2, panels)]
    payload = {"beats": [{"group_id": 2, "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg"]}]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="action")

    assert len(sec["shots"]) == 3, "long lines must not be over-merged"
    assert len(sec["script_paragraphs"]) == 3
    assert len(sec["tts_paragraphs_v3"]) == 3


def test_merge_integration_disabled_by_flag():
    """tts_merge_short=False → strict 1 shot per panel even for short lines."""
    panels = [
        {"scene_file": "p1.jpg", "line": "Run."},
        {"scene_file": "p2.jpg", "line": "Hide."},
        {"scene_file": "p3.jpg", "line": "Wait."},
    ]
    chunk = [_panel_beat(3, panels)]
    payload = {"beats": [{"group_id": 3, "scene_files": ["p1.jpg", "p2.jpg", "p3.jpg"]}]}
    sec = se._build_verbatim_section(
        section_index=0, chunk=chunk, payload=payload,
        word_target=120, genre_mode="action",
        tts_merge_short=False)

    assert len(sec["shots"]) == 3, "merge disabled → must be 1 shot per panel"
    assert len(sec["script_paragraphs"]) == 3
    assert [s.get("scene_files") for s in sec["shots"]] == [
        ["p1.jpg"], ["p2.jpg"], ["p3.jpg"]]
