"""gemini_narrative_pass: per-panel narration alignment + schema tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gemini_narrative_pass",
    Path(__file__).resolve().parent.parent / "tools" / "gemini_narrative_pass.py")
gnp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gnp)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# REGRESSION (panel-collapse): a parse-failed beat whose per-panel narration
# was backfilled must NOT keep the silencing `error` flag — it carries valid
# lines now, so the flag is renamed to `group_parse_error` (telemetry only).
# ---------------------------------------------------------------------------

def test_demote_backfilled_error_renames_flag_when_lines_present():
    beat = {"group_id": 3, "error": "parse_failed_after_retries",
            "panel_narration": [{"scene_file": "p1.jpg", "line": "He draws his blade."}]}
    out = gnp.demote_backfilled_error(beat)
    assert "error" not in out                       # no longer silences downstream
    assert out["group_parse_error"] == "parse_failed_after_retries"   # telemetry kept
    assert out["panel_narration"]                   # the real lines survive


def test_demote_backfilled_error_keeps_flag_without_lines():
    beat = {"group_id": 4, "error": "parse_failed_after_retries", "panel_narration": []}
    out = gnp.demote_backfilled_error(beat)
    assert out["error"] == "parse_failed_after_retries"   # nothing to honor -> stays errored
    assert "group_parse_error" not in out


def test_demote_backfilled_error_noop_on_healthy_beat():
    beat = {"group_id": 5, "panel_narration": [{"scene_file": "p1.jpg", "line": "x"}]}
    out = gnp.demote_backfilled_error(beat)
    assert "error" not in out and "group_parse_error" not in out


# ---------------------------------------------------------------------------
# Task 3-pre: build_arg_parser + --understood flag
# ---------------------------------------------------------------------------

def test_build_arg_parser_understood_flag():
    parser = gnp.build_arg_parser()
    args = parser.parse_args([
        "--groups-manifest", "g.json",
        "--vision-manifest", "v.json",
        "--out", "out.json",
        "--understood", "x.json",
    ])
    assert args.understood == "x.json"


def test_recap_rules_cover_density_name_ration_and_reveal_pacing():
    rules = gnp.RECAP_STYLE_RULES
    for phrase in ("NO SCREEN READING", "POINT, DON'T PAINT", "RATION NAMES",
                   "ADD TEXTURE", "COMPRESS DRAG", "REVEAL PACING"):
        assert phrase in rules


def test_dialogue_rule_allows_punchy_quote_forbids_onomatopoeia_and_fragments():
    rule = gnp._DIALOGUE_RULE.lower()
    assert "paraphrase" in rule
    # allows a short complete punchy quote
    assert "quote" in rule and ("punchy" in rule or "threat" in rule)
    # forbids onomatopoeia / sound effects and incomplete trailing-off fragments
    assert "onomatopoeia" in rule
    assert "fragment" in rule


def test_dedupe_consecutive_panel_lines_reexported():
    # Bug 2/3 narration-level dedup is available to the narrative pass.
    assert hasattr(gnp, "dedupe_consecutive_panel_lines")
    beats = {"beats": [{"group_id": 1, "scene_files": ["a.jpg", "b.jpg"],
                        "panel_narration": [
                            {"scene_file": "a.jpg", "line": "Same line."},
                            {"scene_file": "b.jpg", "line": "Same line."}]}]}
    assert gnp.dedupe_consecutive_panel_lines(beats) == 1


# ---------------------------------------------------------------------------
# Task 3a: align_panel_narration repair-fill helper
# ---------------------------------------------------------------------------

def test_align_pads_missing_panels_from_understanding():
    files = ["a.jpg", "b.jpg", "c.jpg"]
    model = [{"scene_file": "a.jpg", "line": "He draws the blade."},
             {"scene_file": "c.jpg", "line": "Silence falls."}]   # b missing
    u = {"b.jpg": {"description": "the beast lunges"}}
    out = gnp.align_panel_narration(files, model, u)
    assert [p["scene_file"] for p in out] == files
    assert out[1]["line"] == "the beast lunges"

def test_align_pad_never_emits_camera_prose_verbatim():
    # BUG D4: the understanding `description` is camera/shot framing ("A close-up
    # shot shows..."). The pad must NOT copy it verbatim — prefer the concrete
    # action/subjects, else a neutral bridge; never raw camera prose.
    files = ["a.jpg"]
    camera = "A close-up shot shows his trembling hands."
    out = gnp.align_panel_narration(files, [], {"a.jpg": {"description": camera}})
    assert out[0]["line"] != camera
    assert not gnp.is_shot_description(out[0]["line"])

    # action/subjects are preferred over a camera-prose description
    out2 = gnp.align_panel_narration(files, [], {"a.jpg": {
        "description": camera, "action": "He clenches his fists."}})
    assert out2[0]["line"] == "He clenches his fists."
    out3 = gnp.align_panel_narration(files, [], {"a.jpg": {
        "description": camera, "subjects": ["a wounded prince"]}})
    assert "wounded prince" in out3[0]["line"]


def test_align_is_positional_when_model_omits_scene_file():
    files = ["a.jpg", "b.jpg"]
    model = [{"line": "First."}, {"line": "Second."}]
    out = gnp.align_panel_narration(files, model, {})
    assert [p["line"] for p in out] == ["First.", "Second."]

def test_align_folds_overflow_into_last_panel_no_phantoms():
    files = ["a.jpg"]
    model = [{"scene_file": "a.jpg", "line": "One."}, {"scene_file": "zzz.jpg", "line": "Two."}]
    out = gnp.align_panel_narration(files, model, {})
    assert len(out) == 1 and out[0]["scene_file"] == "a.jpg"
    assert out[0]["line"] == "One. Two."

def test_align_invariant_length_matches_scene_files():
    files = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    out = gnp.align_panel_narration(files, [], {})
    assert len(out) == len(files)
    assert all(p["line"] for p in out)


# ---------------------------------------------------------------------------
# Task 3b: build_beat_schema + panel_narration field
# ---------------------------------------------------------------------------

def test_beat_schema_requires_panel_narration():
    schema = gnp.build_beat_schema()
    props = schema["properties"]
    assert "panel_narration" in props
    assert props["panel_narration"]["type"] == "ARRAY"
    item = props["panel_narration"]["items"]["properties"]
    assert set(item) >= {"scene_file", "line"}
    assert "panel_narration" in schema["required"]
    assert "narration" in props          # joined string kept for back-compat


def test_group_payload_threads_full_panel_understanding():
    group = {"shot_id": 1, "scene_files": ["a.jpg"]}
    vision = {"a.jpg": {
        "ocr_clean": "WHO ARE YOU",
        "subjects": ["fallback subject"],
        "vision": {"labels": [], "objects": []},
    }}
    understood = {"a.jpg": {
        "description": "A masked assassin questions an unfamiliar stranger.",
        "action": "The assassin raises his sword.",
        "setting": "forest clearing",
        "dialogue": "Who are you?",
        "panel_kind": "story",
        "intensity": "tense",
        "subjects": ["masked assassin", "unfamiliar stranger"],
    }}
    payload = gnp._pack_group_payload(group, vision, understood)
    scene = payload["scenes_signals"][0]
    assert scene["description"].startswith("A masked assassin")
    assert scene["action"] == "The assassin raises his sword."
    assert scene["dialogue"] == "Who are you?"
    assert scene["subjects"] == ["masked assassin", "unfamiliar stranger"]


def test_bumped_num_ctx_fits_oversized_beats_prompt():
    # the real ollama error from a 9358-token group hitting num_ctx 8192
    err = ('{"error":{"code":400,"message":"request (9358 tokens) exceeds the '
           'available context size (8192 tokens), try increasing it",'
           '"type":"exceed_context_size_error","n_prompt_tokens":9358,"n_ctx":8192}}')
    nb = gnp._bumped_num_ctx(err, cur_ctx=8192, num_predict=2048, ctx_max=16384)
    assert nb is not None and nb % 1024 == 0
    assert 9358 <= nb <= 16384 and nb > 8192      # fits the prompt, capped, bigger
    # a non-context error must NOT trigger a bump
    assert gnp._bumped_num_ctx("connection refused", 8192, 2048) is None
    # already large enough -> no bump
    assert gnp._bumped_num_ctx(err, cur_ctx=16384, num_predict=2048, ctx_max=16384) is None
