"""tests/test_thumbnail_gen.py — pure pieces of the YouTube thumbnail tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "thumbnail_gen",
    Path(__file__).resolve().parent.parent / "tools" / "thumbnail_gen.py",
)
tg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tg)  # type: ignore[union-attr]


def _beats(sel):
    return {"beats": [{"scene_selection": [
        {"scene_file": f, "role": r, "intensity": i} for f, r, i in sel]}]}


def test_pick_reference_scenes_weak_then_climax():
    beats = _beats([
        ("p000001.jpg", "keep", "calm"),       # weak (earliest calm/tense)
        ("p000002.jpg", "redundant", "intense"),
        ("p000003.jpg", "keep", "intense"),
        ("p000009.jpg", "keep", "intense"),    # climax (last intense kept)
    ])
    refs = tg.pick_reference_scenes(beats)
    assert refs[0] == "p000001.jpg"
    assert "p000009.jpg" in refs
    assert "p000002.jpg" not in refs           # redundant never referenced
    assert len(refs) <= 3 == len(set(refs) | {refs[0]}) or len(refs) >= 2


def test_pick_reference_scenes_empty():
    assert tg.pick_reference_scenes({"beats": []}) == []


def test_build_prompt_default_before_after_no_titles():
    p = tg.build_prompt("")
    assert '"BEFORE"' in p and '"AFTER"' in p
    assert "no series name" in p               # licensed names never rendered
    assert "speech bubbles" in p and "16:9" in p


def test_build_prompt_hook_replaces_before_after():
    p = tg.build_prompt("Secret AI System")
    assert '"SECRET AI SYSTEM"' in p and "arrow" in p
    assert '"BEFORE"' not in p


def test_client_uses_only_the_gemini_api_key(monkeypatch, capsys):
    """Vertex is deliberately NOT used: a fallback chain hid which account
    paid AND hid failure (Vertex-refused and Vertex-worked produced the same
    image and exit code)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    made = []

    class _FakeClient:
        def __init__(self, **kw):
            made.append(kw)

    import sys as _sys
    import types as _types
    fake = _types.ModuleType("google.genai")
    fake.Client = _FakeClient
    pkg = _types.ModuleType("google")
    pkg.genai = fake
    monkeypatch.setitem(_sys.modules, "google", pkg)
    monkeypatch.setitem(_sys.modules, "google.genai", fake)

    attempts = tg._make_client()
    assert [k for k, _c in attempts] == ["api-key"]
    assert made == [{"api_key": "test-key"}]
    assert not any("vertexai" in kw for kw in made)


def test_missing_key_fails_loudly_with_no_fallback(monkeypatch, capsys):
    """No key in env AND none in the keychain -> a clear error naming both
    ways to supply one, never a silent fallback to another credential."""
    import subprocess as _sp
    import types as _t
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(_sp, "run", lambda *a, **k: _t.SimpleNamespace(
        returncode=1, stdout=""))
    monkeypatch.setattr(tg, "subprocess", _sp)
    assert tg._make_client() == []
    out = capsys.readouterr().out
    assert "no GEMINI_API_KEY" in out
    assert "keychain" in out          # tells you the Mini's storage location


def test_key_resolves_from_env_first(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert tg.resolve_api_key() == ("from-env", "env")


def test_key_falls_back_to_the_macos_keychain(monkeypatch):
    """The production Mini stores the key in the keychain, deliberately NOT
    in creds.env."""
    import subprocess as _sp
    import types as _t
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(tg.sys, "platform", "darwin")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _t.SimpleNamespace(returncode=0, stdout="from-keychain\n")

    monkeypatch.setattr(_sp, "run", fake_run)
    monkeypatch.setattr(tg, "subprocess", _sp)
    assert tg.resolve_api_key() == ("from-keychain", "keychain")
    assert seen["cmd"][:2] == ["security", "find-generic-password"]
    assert "GEMINI_API_KEY" in seen["cmd"] and "-w" in seen["cmd"]


def test_no_key_anywhere_returns_empty(monkeypatch):
    import subprocess as _sp
    import types as _t
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(_sp, "run", lambda *a, **k: _t.SimpleNamespace(
        returncode=1, stdout=""))
    monkeypatch.setattr(tg, "subprocess", _sp)
    assert tg.resolve_api_key() == ("", "")


def test_the_key_is_never_printed(monkeypatch, capsys):
    """A job log is served over the network by the dashboard — the secret
    must never reach it. Only the SOURCE is reported."""
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    import types as _t
    import sys as _sys
    fake = _t.ModuleType("google.genai")
    fake.Client = lambda **kw: object()
    pkg = _t.ModuleType("google")
    pkg.genai = fake
    monkeypatch.setitem(_sys.modules, "google", pkg)
    monkeypatch.setitem(_sys.modules, "google.genai", fake)
    tg._make_client()
    out = capsys.readouterr().out
    assert "super-secret-value" not in out
    assert "from env" in out
