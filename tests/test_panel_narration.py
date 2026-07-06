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
# Task 3b: build_beat_schema + panel_narration field (legacy per_panel path;
# the adaptive default emits `segments` — see test_narrative_segments.py)
# ---------------------------------------------------------------------------

def test_beat_schema_requires_panel_narration():
    schema = gnp.build_beat_schema("per_panel")
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


# ---------------------------------------------------------------------------
# FIX 2 (narration quality): INPUT sanitization. The writer must never receive
# a camera/shot description (or a rendering-effect action) as a panel signal —
# it echoes them back as `shot_description` narration, and the skipped-panel
# grounded pad reuses the same poisoned text (so the heal never converges).
# `_non_camera_description` is the ONE non-camera ladder shared by
# `_pack_group_payload` (writer input) and `_grounded_pad_line` (pad).
# ---------------------------------------------------------------------------

def test_non_camera_description_prefers_action_over_shot_description():
    # description is camera prose, a clean action exists -> action wins, and the
    # returned signal is NOT a shot-description.
    u = {"description": "A close-up shot shows a man drawing his blade.",
         "action": "He draws his blade.",
         "subjects": ["a man", "a blade"]}
    sig = gnp._non_camera_description(u)
    assert sig == "He draws his blade."
    assert not gnp.is_shot_description(sig)


def test_non_camera_description_synthesizes_neutral_summary_from_subjects():
    # action AND description are BOTH camera prose (or empty) -> fall to a
    # neutral subjects+setting summary: not a shot-description, non-empty, and
    # returned WHOLE (the helper never truncates).
    u = {"description": "A wide establishing shot reveals the hall.",
         "action": "",
         "subjects": ["a hooded knight", "a shattered throne"],
         "setting": "the ruined great hall"}
    sig = gnp._non_camera_description(u)
    assert sig                                    # non-empty
    assert not gnp.is_shot_description(sig)
    assert sig == "a hooded knight, a shattered throne in the ruined great hall"


def test_non_camera_description_empty_when_nothing_usable():
    assert gnp._non_camera_description({}) == ""
    assert gnp._non_camera_description(None) == ""
    # everything usable is camera prose -> nothing survives the ladder
    assert gnp._non_camera_description(
        {"description": "A dramatic overhead shot captures the courtyard.",
         "action": "The panel shows a blur of motion."}) == ""


def test_pack_group_payload_sanitizes_shot_description_input():
    # ROOT CAUSE: a shot-description `description` must not reach the writer.
    group = {"shot_id": 7, "scene_files": ["a.jpg"]}
    vision = {"a.jpg": {"vision": {"labels": [], "objects": []}}}
    understood = {"a.jpg": {
        "description": "A close-up shot shows a man.",   # camera prose
        "action": "He steps into the light.",
        "subjects": ["a man"], "panel_kind": "story"}}
    scene = gnp._pack_group_payload(group, vision, understood)["scenes_signals"][0]
    assert not gnp.is_shot_description(scene["description"])
    assert scene["description"] == "He steps into the light."   # prefers action


def test_pack_group_payload_keeps_a_clean_description():
    # a NON-camera description threads through unchanged (no over-sanitizing).
    group = {"shot_id": 8, "scene_files": ["a.jpg"]}
    vision = {"a.jpg": {"vision": {"labels": [], "objects": []}}}
    understood = {"a.jpg": {
        "description": "The assassin corners the merchant in the alley.",
        "action": "He raises his blade.", "subjects": ["assassin", "merchant"]}}
    scene = gnp._pack_group_payload(group, vision, understood)["scenes_signals"][0]
    assert scene["description"] == "The assassin corners the merchant in the alley."


def test_pack_group_payload_sanitizes_a_camera_action():
    # action can ITSELF be a rendering/camera phrase (real Nano ch1: "A blade
    # swings through the air ...") -> it must not reach the writer as `action`.
    group = {"shot_id": 9, "scene_files": ["a.jpg"]}
    vision = {"a.jpg": {"vision": {"labels": [], "objects": []}}}
    understood = {"a.jpg": {
        "description": "The duel reaches its final exchange.",
        "action": "A blade swings through the air with lethal speed.",
        "subjects": ["a swordsman"]}}
    scene = gnp._pack_group_payload(group, vision, understood)["scenes_signals"][0]
    assert not gnp.is_shot_description(scene["action"])
    assert scene["action"] == ""                  # camera action dropped


def test_grounded_pad_line_uses_shared_non_camera_ladder():
    # _grounded_pad_line delegates to _non_camera_description (DRY): a skipped
    # panel with a camera description falls to its clean action, and the
    # original ultimate-fallback string is preserved when nothing is usable.
    u_by_file = {"p1.jpg": {
        "description": "A dramatic overhead shot captures the courtyard.",
        "action": "He kneels in the snow.", "subjects": ["a knight"]}}
    line = gnp._grounded_pad_line("p1.jpg", u_by_file)
    assert line == "He kneels in the snow."
    assert not gnp.is_shot_description(line)
    assert gnp._grounded_pad_line("x.jpg", {"x.jpg": {}}) == "The moment holds."


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


# ---------------------------------------------------------------------------
# Impact-SFX fusion (eyes wave): the detector-stamped signal must reach the
# narration writer's input — a terse per-panel marker the writer cannot miss.
# ---------------------------------------------------------------------------

def test_pack_group_payload_carries_impact_marker():
    group = {"shot_id": 9, "scene_files": ["a.jpg", "b.jpg"]}
    vision = {"a.jpg": {"vision": {"labels": [], "objects": []}},
              "b.jpg": {"vision": {"labels": [], "objects": []}}}
    understood = {
        "a.jpg": {"description": "A blade sinks into his side.",
                  "action": "He stabs the man.",
                  "impact_sfx": {"present": True, "regions": 2},
                  "strikes_or_weapons": "in_use"},
        "b.jpg": {"description": "They speak quietly.", "action": "He nods.",
                  "impact_sfx": {"present": False, "regions": 0},
                  "strikes_or_weapons": "none"},
    }
    scenes = gnp._pack_group_payload(group, vision, understood)["scenes_signals"]
    assert scenes[0]["impact_sfx"] == "[IMPACT SFX on panel]"
    assert scenes[0]["strikes_or_weapons"] == "in_use"
    # byte-compatible when there is no signal: the keys simply do not exist
    assert "impact_sfx" not in scenes[1]
    assert "strikes_or_weapons" not in scenes[1]


def test_pack_group_payload_weapon_marker_without_detector_signal():
    # a model-claimed visible weapon travels even when the detector is silent
    group = {"shot_id": 10, "scene_files": ["a.jpg"]}
    vision = {"a.jpg": {"vision": {"labels": [], "objects": []}}}
    understood = {"a.jpg": {"description": "He rests a hand on the hilt.",
                            "action": "He waits.",
                            "strikes_or_weapons": "visible"}}
    scene = gnp._pack_group_payload(group, vision, understood)["scenes_signals"][0]
    assert scene["strikes_or_weapons"] == "visible"
    assert "impact_sfx" not in scene


def test_pack_group_payload_legacy_understanding_unchanged():
    # a pu_v1-era record (no impact fields at all) packs exactly as before
    group = {"shot_id": 11, "scene_files": ["a.jpg"]}
    vision = {"a.jpg": {"vision": {"labels": [], "objects": []}}}
    understood = {"a.jpg": {"description": "The alley narrows.",
                            "action": "He runs."}}
    scene = gnp._pack_group_payload(group, vision, understood)["scenes_signals"][0]
    assert "impact_sfx" not in scene and "strikes_or_weapons" not in scene
