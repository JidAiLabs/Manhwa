"""tests/test_meta_garbage_narration.py

The Ch20 g0014 bug: a panel's OCR was a long run of underscores; the narration
model, fed that corruption, returned VALID JSON whose narration was
META-COMMENTARY about parsing/JSON — and it got voiced. The beat's `error` was
None, so nothing caught it.

This drives `_is_meta_garbage` (a pure detector) and its wiring into the accept
loop: a meta-garbage narration is treated like an empty narration — RETRY the
full generation; on the last attempt fall back to a clean line (what_happens if
clean, else a neutral bridge). NEVER voice the meta-garbage.

Stubs the model (no Gemma/network), in the importlib style of
test_cast_tokens.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gemini_narrative_pass",
    Path(__file__).resolve().parent.parent / "tools" / "gemini_narrative_pass.py")
gnp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gnp)  # type: ignore[union-attr]


# The real corruption that leaked to TTS in Ch20 g0014.
_META = (
    "The system encountered a malformed JSON fragment containing excessive "
    "underscore characters. This process involves parsing the intended data "
    "structure and restoring the integrity of the `scene_files` array and the "
    "overall object schema."
)
_META_WHAT = (
    "The input was truncated and corrupted with a long string of underscores. "
    "The task is to reconstruct the valid JSON object..."
)


# ---------------------------------------------------------------------------
# _is_meta_garbage — pure detector
# ---------------------------------------------------------------------------
def test_detects_the_ch20_example():
    assert gnp._is_meta_garbage(_META) is True


def test_detects_the_ch20_what_happens():
    assert gnp._is_meta_garbage(_META_WHAT) is True


def test_real_narration_is_not_meta_garbage():
    real = ("Cheon Mu Geum unleashes the Butterfly Dance, and the murim hall "
            "falls deathly silent as petals scatter across the blood-slick floor.")
    assert gnp._is_meta_garbage(real) is False


def test_empty_is_not_meta_garbage():
    assert gnp._is_meta_garbage("") is False
    assert gnp._is_meta_garbage(None) is False  # type: ignore[arg-type]


def test_single_weak_signal_is_not_garbage():
    # "data structure" alone (no strong json/schema/scene_files/underscore signal)
    # must NOT trip the detector — real narration can mention structure.
    line = "The data structure of their alliance was finally clear to everyone."
    assert gnp._is_meta_garbage(line) is False


def test_strong_signal_required():
    # a strong signal (valid json) present -> garbage
    assert gnp._is_meta_garbage("Re-output a valid JSON object now.") is True


# ---------------------------------------------------------------------------
# fallback helper — clean line never meta-garbage
# ---------------------------------------------------------------------------
def test_fallback_uses_what_happens_when_clean():
    line = gnp._clean_fallback_narration("Beat title", "The duel begins at dawn.")
    assert line == "The duel begins at dawn."


def test_fallback_neutral_when_what_happens_is_garbage():
    line = gnp._clean_fallback_narration("Beat title", _META_WHAT)
    assert gnp._is_meta_garbage(line) is False
    assert line.strip() != ""


# ---------------------------------------------------------------------------
# accept-loop retry: meta-garbage narration -> retry -> clean line wins
# ---------------------------------------------------------------------------
def test_meta_garbage_then_clean_line_retries_to_clean(monkeypatch):
    """First model call returns meta-garbage narration (valid JSON, no error);
    the loop must reject it and RETRY; the second call's clean line is accepted.
    Mirrors the existing empty-narration retry guard."""
    calls = {"n": 0}

    def stub(*, client, model, system_instruction, user_payload, image_paths,
             response_schema, max_output_tokens, temperature, backoff_max,
             backend="vertex"):
        calls["n"] += 1
        if calls["n"] == 1:
            obj = {"beat_title": "Beat", "what_happens": "The duel begins.",
                   "narration": _META, "scene_selection": []}
        else:
            obj = {"beat_title": "Beat", "what_happens": "The duel begins.",
                   "narration": "Cheon Mu Geum steps onto the dueling ground.",
                   "scene_selection": []}
        return obj, "raw", {"input": 5, "output": 5, "cached": 0}

    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    beat = gnp._generate_beat_for_group(
        client=None, model="m", system_instruction="sys",
        payload={"scene_files": ["p1.jpg"]}, image_paths=[], beat_schema={},
        gid=14, retries=1, max_output_tokens=512, backoff_max=5.0, backend="ollama")
    assert calls["n"] == 2
    assert beat["narration"] == "Cheon Mu Geum steps onto the dueling ground."
    assert gnp._is_meta_garbage(beat["narration"]) is False


def test_meta_garbage_on_last_attempt_falls_back_to_clean(monkeypatch):
    """If EVERY attempt returns meta-garbage, the last attempt must fall back to
    a clean line (what_happens if clean, else a neutral bridge) — never voice
    the meta-garbage."""
    def stub(**kw):
        return ({"beat_title": "Beat", "what_happens": _META_WHAT,
                 "narration": _META, "scene_selection": []},
                "raw", {"input": 5, "output": 5, "cached": 0})

    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    beat = gnp._generate_beat_for_group(
        client=None, model="m", system_instruction="sys",
        payload={"scene_files": ["p1.jpg"]}, image_paths=[], beat_schema={},
        gid=14, retries=1, max_output_tokens=512, backoff_max=5.0, backend="ollama")
    # both what_happens AND narration were garbage -> neutral bridge, not garbage
    assert gnp._is_meta_garbage(beat["narration"]) is False
    assert beat["narration"].strip() != ""
