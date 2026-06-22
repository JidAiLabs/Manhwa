"""tests/test_ollama_compat.py

Locks in the HARD wall-clock watchdog on ollama chat calls. The bug it guards
against (claw-mini, 2026-06-23): a gemma4 beats call stalled mid-generation and
the httpx client timeout never fired, hanging the gpu lane ~32 minutes. The
watchdog must abandon a stuck call after HARD_TIMEOUT_SEC regardless.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ollama_compat",
    Path(__file__).resolve().parent.parent / "tools" / "ollama_compat.py",
)
oc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oc)  # type: ignore[union-attr]


def _fake_ollama(monkeypatch, chat_impl):
    # bare module with .chat only (no Client) -> _raw_chat calls ollama.chat
    fake = types.ModuleType("ollama")
    fake.chat = chat_impl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", fake)


def test_chat_happy_path_returns_result(monkeypatch):
    _fake_ollama(monkeypatch, lambda **kw: {"message": {"content": "ok"}})
    assert oc.chat(model="m", messages=[])["message"]["content"] == "ok"


def test_chat_hard_timeout_abandons_a_stuck_call(monkeypatch):
    monkeypatch.setattr(oc, "HARD_TIMEOUT_SEC", 0.2)

    def _hang(**kw):
        time.sleep(5.0)                       # simulates the 32-min stall
        return {"message": {"content": "too late"}}

    _fake_ollama(monkeypatch, _hang)
    t0 = time.time()
    with pytest.raises(TimeoutError):
        oc.chat(model="m", messages=[])
    # abandoned at ~0.2s, NOT after the 5s sleep
    assert time.time() - t0 < 3.0


def test_chat_reraises_backend_error(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("backend down")

    _fake_ollama(monkeypatch, _boom)
    with pytest.raises(RuntimeError, match="backend down"):
        oc.chat(model="m", messages=[])
