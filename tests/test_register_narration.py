"""register-aware narration (gemini_narrative_pass --register-mode): with the
register path, a beat classified FAST gets the FAST gear prompt + short token
cap, and a DEEP beat gets the DEEP gear prompt + long cap. Uses a stubbed model
call (no Gemma/network), in the style of test_panel_understand.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gemini_narrative_pass",
    Path(__file__).resolve().parent.parent / "tools" / "gemini_narrative_pass.py")
gnp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gnp)  # type: ignore[union-attr]


def _make_stub(register_returns: str):
    """A stubbed _call_model_with_backoff that records every call's system
    prompt + token cap, classifies via the requested register, and returns a
    schema'd dict. No client/model/network is touched."""
    calls = []

    def stub(*, client, model, system_instruction, user_payload, image_paths,
             response_schema, max_output_tokens, temperature, backoff_max,
             backend="vertex"):
        calls.append({"system": system_instruction,
                      "max_output_tokens": max_output_tokens,
                      "temperature": temperature,
                      "schema": response_schema,
                      "images": list(image_paths),
                      "payload": user_payload})
        # classifier call -> returns {register: ...}; narration call -> {narration}
        if "register" in (response_schema.get("properties") or {}):
            return {"register": register_returns}, "raw", {"input": 5, "output": 1, "cached": 0}
        return {"narration": f"line for {register_returns}"}, "raw", {"input": 9, "output": 4, "cached": 0}

    return stub, calls


def _payload():
    return {"group_id": 1, "scene_files": ["p1.jpg"], "scenes_signals": [],
            "arc_label": "x", "segment": "present", "intensity": "tense"}


def test_classifier_returns_fast_or_deep_and_threads_usage(monkeypatch):
    stub, calls = _make_stub("DEEP")
    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    reg, usage = gnp._classify_register(
        client=None, model="m", payload=_payload(), image_paths=["/i/p1.jpg"],
        backoff_max=5.0, backend="ollama")
    assert reg == "DEEP"
    assert usage == {"input": 5, "output": 1, "cached": 0}
    # the classifier ran with the exact verified prompt, low cap, low temp, image
    assert len(calls) == 1
    assert calls[0]["system"] == gnp._REGISTER_CLASSIFIER_PROMPT
    assert calls[0]["max_output_tokens"] == 30 and calls[0]["temperature"] == 0.2
    assert calls[0]["images"] == ["/i/p1.jpg"]


def test_classifier_defaults_fast_on_garbage(monkeypatch):
    # an unrecognized register string falls back to FAST (the safe terse default)
    def stub(**kw):
        return {"register": "MAYBE?"}, "raw", {"input": 1, "output": 1, "cached": 0}
    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    reg, _ = gnp._classify_register(
        client=None, model="m", payload=_payload(), image_paths=[],
        backoff_max=5.0, backend="ollama")
    assert reg == "FAST"


def test_fast_beat_gets_fast_prompt_and_short_cap(monkeypatch):
    stub, calls = _make_stub("FAST")
    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    line, usage = gnp._register_narration(
        client=None, model="m", register="FAST", cast_block="", story_block="",
        is_first=False, payload=_payload(), image_paths=["/i/p1.jpg"],
        backoff_max=5.0, backend="ollama")
    assert line == "line for FAST"
    call = calls[0]
    # FAST gear prompt is present; DEEP prompt is NOT; cap is the enforced-short 70
    assert gnp._FAST_NARRATION_PROMPT in call["system"]
    assert gnp._DEEP_NARRATION_PROMPT not in call["system"]
    assert call["max_output_tokens"] == 70 and call["temperature"] == 0.3
    # advertiser-safety rules are appended; image grounding preserved
    assert gnp.SAFE_NARRATION_RULES in call["system"]
    assert call["images"] == ["/i/p1.jpg"]


def test_deep_beat_gets_deep_prompt_and_long_cap(monkeypatch):
    stub, calls = _make_stub("DEEP")
    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    line, _ = gnp._register_narration(
        client=None, model="m", register="DEEP", cast_block="", story_block="",
        is_first=False, payload=_payload(), image_paths=[],
        backoff_max=5.0, backend="ollama")
    assert line == "line for DEEP"
    call = calls[0]
    assert gnp._DEEP_NARRATION_PROMPT in call["system"]
    assert gnp._FAST_NARRATION_PROMPT not in call["system"]
    assert call["max_output_tokens"] == 350 and call["temperature"] == 0.4


def test_first_group_gets_cold_open_note_only_on_first():
    # SAFE_OPENING_NOTE rides the FIRST group's narration prompt, not later ones
    first = gnp._build_register_system("FAST", "", "", is_first=True)
    later = gnp._build_register_system("FAST", "", "", is_first=False)
    assert gnp.SAFE_OPENING_NOTE in first
    assert gnp.SAFE_OPENING_NOTE not in later
    # both still carry the gear prompt + advertiser-safety rules
    assert gnp._FAST_NARRATION_PROMPT in first and gnp.SAFE_NARRATION_RULES in first


def test_register_system_includes_cast_and_story_grounding():
    sysmsg = gnp._build_register_system(
        "DEEP", "CHAPTER CAST — alice", "CHAPTER STORY SPINE — arc", is_first=False)
    # grounding context (cast names + story spine) is preserved in the gear prompt
    assert "CHAPTER CAST — alice" in sysmsg
    assert "CHAPTER STORY SPINE — arc" in sysmsg
    assert gnp._DEEP_NARRATION_PROMPT in sysmsg


def test_register_system_carries_continuity_antiecho_rule():
    # the register override builds a FRESH system prompt; it MUST carry the
    # same previous_narration continuity/anti-echo rule the default call has,
    # or consecutive DEEP beats echo the same opener (the Ch20 bug).
    for reg in ("FAST", "DEEP"):
        sysmsg = gnp._build_register_system(reg, "", "", is_first=False)
        assert gnp._REGISTER_CONTINUITY_RULE in sysmsg
        # it must reference previous_narration and forbid reusing the opener
        assert "previous_narration" in gnp._REGISTER_CONTINUITY_RULE
        assert "opening words" in gnp._REGISTER_CONTINUITY_RULE


def test_register_narration_empty_on_parse_miss(monkeypatch):
    # a non-dict / missing-narration model response -> empty string, so the
    # caller keeps the default-call narration (never blanks the line)
    def stub(**kw):
        return None, "garbage", {"input": 0, "output": 0, "cached": 0}
    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    line, _ = gnp._register_narration(
        client=None, model="m", register="FAST", cast_block="", story_block="",
        is_first=False, payload=_payload(), image_paths=[],
        backoff_max=5.0, backend="ollama")
    assert line == ""
