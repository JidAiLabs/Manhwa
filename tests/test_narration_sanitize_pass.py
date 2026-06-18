"""narration_sanitize_pass + narration_reframe: the advertiser-safety pass wired
before TTS. Tests the pure sanitize→reframe→re-sanitize logic with a STUBBED
call_fn (no live model / network), mirroring test_panel_understand.py's
importlib+stub style.

Covers the spec's required cases:
  - a flagged line gets reframed, then passes (clean)
  - a line that still BLOCKS after reframe is reported UNRESOLVED
  - a clean line is left untouched (and not sent to the reframe)
  - the deterministic safe SWAPS still apply (replace action)
plus: script-object pass mutates BOTH script_paragraphs and tts_paragraphs_v3
(preserving the leading mood tag), the marker round-trips, and reframe is
skipped when no model is wired.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / filename)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass introspection can resolve __module__
    # (dataclasses looks the module up in sys.modules during field typing).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


nrf = _load("narration_reframe", "narration_reframe.py")
nsp = _load("narration_sanitize_pass", "narration_sanitize_pass.py")


# --- reframe stubs ----------------------------------------------------------

def _stub_constant(reply: str):
    """call_fn that always returns {"narration": reply}, recording invocations."""
    calls = []

    def call_fn(system, user_payload, schema, max_tokens):
        calls.append({"system": system, "payload": user_payload, "max": max_tokens})
        return {"narration": reply}

    call_fn.calls = calls  # type: ignore[attr-defined]
    return call_fn


def _stub_echo():
    """call_fn that returns the SAME line it was given (model 'declines')."""
    calls = []

    def call_fn(system, user_payload, schema, max_tokens):
        calls.append(user_payload)
        return {"narration": user_payload["line"]}

    call_fn.calls = calls  # type: ignore[attr-defined]
    return call_fn


# --- narration_reframe.reframe_line -----------------------------------------

def test_reframe_returns_rewritten_line():
    call_fn = _stub_constant("He crossed a line he never should have.")
    # a block hit (sexual_violence) — note carries the implication-level guidance
    Hit = type("Hit", (), {})
    h = Hit(); h.category = "sexual"; h.note = "HARD BLOCK. Rewrite to implication only."
    out = nrf.reframe_line("he raped her", [h], call_fn)
    assert out == "He crossed a line he never should have."
    assert call_fn.calls and "implication" in call_fn.calls[0]["system"].lower()


def test_reframe_no_hits_is_noop_and_skips_model():
    call_fn = _stub_constant("SHOULD NOT BE USED")
    assert nrf.reframe_line("A calm morning in the city.", [], call_fn) == \
        "A calm morning in the city."
    assert call_fn.calls == []


def test_reframe_falls_back_to_original_on_model_failure():
    def boom(system, payload, schema, max_tokens):
        raise RuntimeError("backend down")
    Hit = type("Hit", (), {})
    h = Hit(); h.category = "violence"; h.note = "soften"
    # model error -> ORIGINAL line returned (so re-sanitize still gates it)
    assert nrf.reframe_line("the massacre", [h], boom) == "the massacre"


def test_reframe_falls_back_on_empty_reply():
    def empty(system, payload, schema, max_tokens):
        return {"narration": "   "}
    Hit = type("Hit", (), {})
    h = Hit(); h.category = "violence"; h.note = "soften"
    assert nrf.reframe_line("the massacre", [h], empty) == "the massacre"


def test_build_reframe_prompt_includes_notes_and_category_guidance():
    Hit = type("Hit", (), {})
    h1 = Hit(); h1.category = "self_harm_suicide"; h1.note = "never name a method"
    p = nrf.build_reframe_prompt("she slit her wrists", [h1])
    assert "never name a method" in p["system"]
    assert "method" in p["system"].lower()
    assert p["user_payload"]["line"] == "she slit her wrists"


# --- sanitize_script: flag -> reframe -> clean ------------------------------

def _script_with(*paragraphs):
    """One section, one shot per paragraph, tts mirrors script with a mood tag."""
    return {
        "sections": [{
            "section_index": 0,
            "shots": [{"group_id": 10 + i, "beat_id": i + 1} for i in range(len(paragraphs))],
            "script_paragraphs": list(paragraphs),
            "tts_paragraphs_v3": [f"[serious] {p}" for p in paragraphs],
        }]
    }


def test_flagged_line_is_reframed_then_passes_clean():
    # 'sex' is a FLAG (context-sensitive). Reframe to an implication that the
    # sanitizer no longer flags -> resolved, no unresolved blocks.
    obj = _script_with("They had sex that night.")
    call_fn = _stub_constant("They were intimate that night.")
    summ = nsp.sanitize_script(obj, seed="ch0001", call_fn=call_fn)
    sec = obj["sections"][0]
    assert sec["script_paragraphs"][0] == "They were intimate that night."
    # mood tag preserved on the voiced line
    assert sec["tts_paragraphs_v3"][0] == "[serious] They were intimate that night."
    assert summ.reframed == 1 and summ.changed == 1
    assert summ.unresolved_blocks == []


def test_block_still_present_after_reframe_is_unresolved():
    # A hard BLOCK (sexual_violence) where the model FAILS to soften it (echoes
    # the line back) -> re-sanitize still finds the block -> UNRESOLVED.
    obj = _script_with("He raped the guard.")
    call_fn = _stub_echo()
    summ = nsp.sanitize_script(obj, seed="ch0001", call_fn=call_fn)
    assert summ.reframed == 1
    assert summ.has_unresolved
    seg, matched = summ.unresolved_blocks[0]
    assert seg == "g0010_p00"
    assert matched.lower() == "raped"


def test_block_resolved_when_reframe_softens_to_implication():
    obj = _script_with("He raped the guard.")
    call_fn = _stub_constant("He crossed a line he never should have.")
    summ = nsp.sanitize_script(obj, seed="ch0001", call_fn=call_fn)
    sec = obj["sections"][0]
    assert sec["script_paragraphs"][0] == "He crossed a line he never should have."
    assert summ.unresolved_blocks == []


def test_clean_line_untouched_and_not_reframed():
    obj = _script_with("The hunter walks into the gate at dawn.")
    call_fn = _stub_constant("SHOULD NOT BE USED")
    summ = nsp.sanitize_script(obj, seed="ch0001", call_fn=call_fn)
    sec = obj["sections"][0]
    assert sec["script_paragraphs"][0] == "The hunter walks into the gate at dawn."
    assert sec["tts_paragraphs_v3"][0] == "[serious] The hunter walks into the gate at dawn."
    assert summ.changed == 0 and summ.reframed == 0
    assert call_fn.calls == []


def test_deterministic_swap_still_applies_without_model():
    # 'killed' is a REPLACE (deterministic safe swap) — applies with NO model
    # wired (call_fn=None), no reframe, no blocks.
    obj = _script_with("He killed the beast.")
    summ = nsp.sanitize_script(obj, seed="ch0001", call_fn=None)
    sec = obj["sections"][0]
    body = sec["script_paragraphs"][0]
    assert body != "He killed the beast."          # swapped
    assert "killed" not in body.lower()
    # voiced text carries the same swap, mood tag intact
    assert sec["tts_paragraphs_v3"][0] == f"[serious] {body}"
    assert summ.changed == 1 and summ.reframed == 0
    assert summ.unresolved_blocks == []


def test_swap_is_seeded_deterministic():
    a = _script_with("He killed one, then killed another.")
    b = _script_with("He killed one, then killed another.")
    nsp.sanitize_script(a, seed="chX", call_fn=None)
    nsp.sanitize_script(b, seed="chX", call_fn=None)
    assert a["sections"][0]["script_paragraphs"] == b["sections"][0]["script_paragraphs"]


def test_no_model_leaves_block_unresolved():
    # With no model, a block can't be softened -> it must still be reported so
    # the voiced gate fires (safety can't depend on an LLM being available).
    obj = _script_with("He raped the guard.")
    summ = nsp.sanitize_script(obj, seed="ch0001", call_fn=None)
    assert summ.reframed == 0
    assert summ.has_unresolved and summ.unresolved_blocks[0][0] == "g0010_p00"


def test_tts_tag_preserved_through_swap():
    obj = {
        "sections": [{
            "section_index": 0,
            "shots": [{"group_id": 21, "beat_id": 1}],
            "script_paragraphs": ["They murder the elders."],
            "tts_paragraphs_v3": ["[angry] They murder the elders."],
        }]
    }
    nsp.sanitize_script(obj, seed="ch0001", call_fn=None)
    tts = obj["sections"][0]["tts_paragraphs_v3"][0]
    assert tts.startswith("[angry] ")
    assert "murder" not in tts.lower()


def test_marker_roundtrip_and_unresolved_reader(tmp_path):
    obj = _script_with("He raped the guard.")
    summ = nsp.sanitize_script(obj, seed="ch0001", call_fn=None)
    marker = tmp_path / "manifest.sanitize.json"
    nsp.write_marker(marker, summ, seed="ch0001")
    data = json.loads(marker.read_text())
    assert data["ok"] is False
    assert data["seed"] == "ch0001"
    assert data["unresolved_blocks"][0]["segment_id"] == "g0010_p00"
    # the reader the voiced gate mirrors
    ub = nsp.read_unresolved_blocks(marker)
    assert ub and ub[0]["matched"].lower() == "raped"
    # a clean chapter writes ok=True and an empty list
    clean = _script_with("The hunter walks into the gate.")
    s2 = nsp.sanitize_script(clean, seed="ch0002", call_fn=None)
    m2 = tmp_path / "clean.sanitize.json"
    nsp.write_marker(m2, s2, seed="ch0002")
    assert json.loads(m2.read_text())["ok"] is True
    assert nsp.read_unresolved_blocks(m2) == []
    assert nsp.read_unresolved_blocks(tmp_path / "missing.json") == []


def test_missing_arrays_section_is_skipped():
    # a parse-failure section with no paragraphs must not crash the pass
    obj = {"sections": [{"section_index": 0, "error": "parse_failed"},
                        _script_with("They had sex.")["sections"][0]]}
    call_fn = _stub_constant("They were intimate.")
    summ = nsp.sanitize_script(obj, seed="ch0001", call_fn=call_fn)
    assert summ.reframed == 1
