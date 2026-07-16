"""Local-LLM call hygiene (2026-07-16 owner goal): sampling/context settings
and the ollama `format` contract hold on EVERY backend — fence/array-tolerant
JSON extraction in the shared caller, and the MLX shim's format emulation
(schema instruction + canonical re-serialization + ONE bumped-temp self-repair
retry, the escape from deterministic-failure at temp 0)."""
import importlib.util
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gnp = _load("gemini_narrative_pass")
shim = _load("mlx_ollama_shim")
pu = _load("panel_understand")


# ---- shared extraction (backend-agnostic) ------------------------------------

_FENCED_ARRAY = ("```json\n[\n {\"from_index\": 0, \"to_index\": 3}\n]\n```")
_PROSE_OBJECT = "Sure! Here is the JSON:\n{\"ok\": true, \"issue\": \"\"}\nDone."
_THINK_OBJECT = "<think>hmm the panel shows…</think>{\"keep\": false}"


def test_extract_json_value_handles_fences_arrays_prose_think():
    assert gnp._extract_json_value(_FENCED_ARRAY) == [
        {"from_index": 0, "to_index": 3}]
    assert gnp._extract_json_value(_PROSE_OBJECT) == {"ok": True, "issue": ""}
    assert gnp._extract_json_value(_THINK_OBJECT) == {"keep": False}
    assert gnp._extract_json_value("no json here at all") is None
    assert gnp._extract_json_value("") is None
    # the job-45 corruption class (broken key fragment) stays a loud failure
    assert gnp._extract_json_value('[{ index": 12, "from_index": 12 }]') is None


def test_extract_json_object_stays_dict_only():
    assert gnp._extract_json_object(_PROSE_OBJECT) == {"ok": True, "issue": ""}
    assert gnp._extract_json_object(_FENCED_ARRAY) is None   # arrays -> value API


# ---- shim: ollama-semantics faithfulness -------------------------------------

def _client():
    from fastapi.testclient import TestClient
    return TestClient(shim.app)


def _post(client, **body):
    r = client.post("/api/chat", json={
        "model": "gemma4:26b",
        "messages": [{"role": "user", "content": "narrate"}], **body})
    assert r.status_code == 200, r.text
    return r.json()


def test_shim_format_schema_yields_canonical_json(monkeypatch):
    calls = []

    def fake_gen(prompt, imgs, opts, think, temperature):
        calls.append({"prompt": prompt, "temperature": temperature})
        return _FENCED_ARRAY, 10, 5, 0.1

    monkeypatch.setattr(shim, "_gen_once", fake_gen)
    out = _post(_client(), format={"type": "array"},
                options={"temperature": 0})
    # content is CANONICAL JSON (caller's json.loads succeeds first try)
    assert json.loads(out["message"]["content"]) == [
        {"from_index": 0, "to_index": 3}]
    assert len(calls) == 1                       # valid roll: no retry
    assert "JSON Schema EXACTLY" in calls[0]["prompt"]   # schema instruction
    assert calls[0]["temperature"] == 0.0        # explicit 0 honored, not 0.8


def test_shim_self_repair_retry_escapes_deterministic_malformation(monkeypatch):
    rolls = iter([('[{ index": 12, "from_index": 12 }]', 10, 5, 0.1),
                  ('[{"from_index": 12, "to_index": 14}]', 10, 5, 0.1)])
    temps = []

    def fake_gen(prompt, imgs, opts, think, temperature):
        temps.append(temperature)
        return next(rolls)

    monkeypatch.setattr(shim, "_gen_once", fake_gen)
    out = _post(_client(), format={"type": "array"},
                options={"temperature": 0})
    assert json.loads(out["message"]["content"]) == [
        {"from_index": 12, "to_index": 14}]
    assert temps == [0.0, 0.4]        # retry bumps temp: same roll can't repeat
    assert out["prompt_eval_count"] == 20        # both attempts accounted


def test_shim_default_temperature_matches_ollama(monkeypatch):
    temps = []

    def fake_gen(prompt, imgs, opts, think, temperature):
        temps.append(temperature)
        return "plain prose", 3, 2, 0.1

    monkeypatch.setattr(shim, "_gen_once", fake_gen)
    out = _post(_client())                       # no options, no format
    assert shim.DEFAULT_TEMPERATURE == 0.8       # the ollama daemon default
    assert temps == [0.8]
    assert out["message"]["content"] == "plain prose"   # no format: verbatim


def test_shim_strips_think_blocks_defensively():
    assert shim._extract_json_value(_THINK_OBJECT) == {"keep": False}
    assert shim._format_block("json").startswith("\n\nRespond with ONLY")
    assert "JSON Schema" in shim._format_block({"type": "object"})
    assert shim._format_block(None) == ""


# ---- analysis lanes are deterministic ----------------------------------------

def test_understanding_default_temperature_is_deterministic():
    import argparse
    for action in _understand_parser_actions():
        if action.dest == "temperature":
            assert action.default == 0.0
            return
    raise AssertionError("panel_understand lost its --temperature argument")


def _understand_parser_actions():
    # main() builds the parser inline; rebuild the same way argparse sees it
    import argparse
    import unittest.mock as um
    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def fake_parse(self, *a, **k):
        captured["parser"] = self
        raise SystemExit(0)

    with um.patch.object(argparse.ArgumentParser, "parse_args", fake_parse):
        try:
            pu.main()
        except SystemExit:
            pass
    return captured["parser"]._actions


def test_shim_format_retry_drops_montage_images_keeps_single(monkeypatch):
    # the retry tax: a FORMAT fix must not re-encode a 6-image montage
    # (79 of 200 measured minutes) — but a single-image call (understanding)
    # keeps its image.
    seen = []
    rolls = iter(['[{ index": 1 }]', '[{"a": 1}]',
                  '[{ index": 2 }]', '[{"b": 2}]'])

    def fake_gen(prompt, imgs, opts, think, temperature):
        seen.append(len(imgs))
        return next(rolls), 5, 5, 0.1

    monkeypatch.setattr(shim, "_gen_once", fake_gen)
    monkeypatch.setattr(shim, "_images", lambda msgs: ["i1", "i2", "i3"])
    _post(_client(), format={"type": "array"}, options={"temperature": 0})
    assert seen[:2] == [3, 0]                    # montage dropped on retry
    monkeypatch.setattr(shim, "_images", lambda msgs: ["only"])
    _post(_client(), format={"type": "array"}, options={"temperature": 0})
    assert seen[2:] == [1, 1]                    # single image kept
