"""
tests/test_qwen_genparams.py

TDD for qwen_clone_genkwargs() — the helper that provides official Qwen3-TTS
generation parameters to generate_voice_clone, env-overridable so live tuning
needs no redeploy.

Module-level spec-loader pattern (same as other tts tests) so qwen_tts itself
is never imported (it lives in .qwen_venv, not .eval_venv).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "local_tts",
    Path(__file__).resolve().parent.parent / "tools" / "local_tts_from_manifest.py",
)
lt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lt)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# qwen_clone_genkwargs — defaults
# ---------------------------------------------------------------------------


class TestQwenCloneGenkwargsDefaults:
    """qwen_clone_genkwargs() returns the documented Qwen3-TTS generation params."""

    def test_function_exists(self):
        assert callable(getattr(lt, "qwen_clone_genkwargs", None)), (
            "qwen_clone_genkwargs not found in local_tts_from_manifest"
        )

    def test_do_sample_default_true(self):
        kw = lt.qwen_clone_genkwargs()
        assert kw["do_sample"] is True

    def test_top_k_default_50(self):
        kw = lt.qwen_clone_genkwargs()
        assert kw["top_k"] == 50

    def test_top_p_default_1_0(self):
        kw = lt.qwen_clone_genkwargs()
        assert kw["top_p"] == pytest.approx(1.0)

    def test_temperature_default_0_9(self):
        kw = lt.qwen_clone_genkwargs()
        assert kw["temperature"] == pytest.approx(0.9)

    def test_repetition_penalty_firmer_than_official(self):
        """Default is 1.1 (slightly firmer than official 1.05) to push harder on stutter/filler."""
        kw = lt.qwen_clone_genkwargs()
        assert kw["repetition_penalty"] == pytest.approx(1.1)

    def test_subtalker_dosample_default_true(self):
        kw = lt.qwen_clone_genkwargs()
        assert kw["subtalker_dosample"] is True

    def test_subtalker_top_k_default_50(self):
        kw = lt.qwen_clone_genkwargs()
        assert kw["subtalker_top_k"] == 50

    def test_subtalker_top_p_default_1_0(self):
        kw = lt.qwen_clone_genkwargs()
        assert kw["subtalker_top_p"] == pytest.approx(1.0)

    def test_subtalker_temperature_default_0_9(self):
        kw = lt.qwen_clone_genkwargs()
        assert kw["subtalker_temperature"] == pytest.approx(0.9)

    def test_max_new_tokens_default_2048(self):
        kw = lt.qwen_clone_genkwargs()
        assert kw["max_new_tokens"] == 2048

    def test_non_streaming_mode_default_true(self):
        """non_streaming_mode=True feeds full text for cleaner offline synthesis."""
        kw = lt.qwen_clone_genkwargs()
        assert kw["non_streaming_mode"] is True

    def test_returns_plain_dict(self):
        kw = lt.qwen_clone_genkwargs()
        assert isinstance(kw, dict)

    def test_all_expected_keys_present(self):
        kw = lt.qwen_clone_genkwargs()
        expected = {
            "do_sample", "top_k", "top_p", "temperature", "repetition_penalty",
            "subtalker_dosample", "subtalker_top_k", "subtalker_top_p",
            "subtalker_temperature", "max_new_tokens", "non_streaming_mode",
        }
        missing = expected - set(kw)
        assert not missing, f"Missing keys: {missing}"


# ---------------------------------------------------------------------------
# env overrides
# ---------------------------------------------------------------------------


class TestQwenCloneGenkwargsEnvOverrides:
    """ENV vars override defaults without redeploying."""

    def test_rep_penalty_env_override(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_REP_PENALTY", "1.3")
        kw = lt.qwen_clone_genkwargs()
        assert kw["repetition_penalty"] == pytest.approx(1.3)

    def test_temperature_env_override(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_TEMPERATURE", "0.7")
        kw = lt.qwen_clone_genkwargs()
        assert kw["temperature"] == pytest.approx(0.7)

    def test_top_k_env_override(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_TOP_K", "100")
        kw = lt.qwen_clone_genkwargs()
        assert kw["top_k"] == 100

    def test_top_p_env_override(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_TOP_P", "0.95")
        kw = lt.qwen_clone_genkwargs()
        assert kw["top_p"] == pytest.approx(0.95)

    def test_do_sample_false_via_env(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_DO_SAMPLE", "false")
        kw = lt.qwen_clone_genkwargs()
        assert kw["do_sample"] is False

    def test_do_sample_true_via_1(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_DO_SAMPLE", "1")
        kw = lt.qwen_clone_genkwargs()
        assert kw["do_sample"] is True

    def test_non_streaming_false_via_env(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_NONSTREAM", "false")
        kw = lt.qwen_clone_genkwargs()
        assert kw["non_streaming_mode"] is False

    def test_sub_do_sample_override(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_SUB_DO_SAMPLE", "false")
        kw = lt.qwen_clone_genkwargs()
        assert kw["subtalker_dosample"] is False

    def test_sub_top_k_override(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_SUB_TOP_K", "25")
        kw = lt.qwen_clone_genkwargs()
        assert kw["subtalker_top_k"] == 25

    def test_sub_top_p_override(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_SUB_TOP_P", "0.85")
        kw = lt.qwen_clone_genkwargs()
        assert kw["subtalker_top_p"] == pytest.approx(0.85)

    def test_sub_temperature_override(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_SUB_TEMPERATURE", "0.75")
        kw = lt.qwen_clone_genkwargs()
        assert kw["subtalker_temperature"] == pytest.approx(0.75)

    def test_max_new_tokens_override(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_MAX_NEW_TOKENS", "1024")
        kw = lt.qwen_clone_genkwargs()
        assert kw["max_new_tokens"] == 1024


# ---------------------------------------------------------------------------
# parsing robustness
# ---------------------------------------------------------------------------


class TestEnvParsingRobustness:
    """Bool/float/int parsing handles varied string forms without crashing."""

    def test_bool_true_forms(self, monkeypatch):
        for val in ("true", "True", "TRUE", "1", "yes"):
            monkeypatch.setenv("STUDIO_QWEN_DO_SAMPLE", val)
            assert lt.qwen_clone_genkwargs()["do_sample"] is True, f"Failed for: {val!r}"

    def test_bool_false_forms(self, monkeypatch):
        for val in ("false", "False", "FALSE", "0", "no"):
            monkeypatch.setenv("STUDIO_QWEN_DO_SAMPLE", val)
            assert lt.qwen_clone_genkwargs()["do_sample"] is False, f"Failed for: {val!r}"

    def test_float_parsing(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_REP_PENALTY", "1.3")
        assert lt.qwen_clone_genkwargs()["repetition_penalty"] == pytest.approx(1.3)

    def test_int_parsing(self, monkeypatch):
        monkeypatch.setenv("STUDIO_QWEN_TOP_K", "75")
        assert lt.qwen_clone_genkwargs()["top_k"] == 75
        assert isinstance(lt.qwen_clone_genkwargs()["top_k"], int)


# ---------------------------------------------------------------------------
# clone call-site integration: kwargs forwarded to generate_voice_clone
# ---------------------------------------------------------------------------


class TestCloneSynthPassesGenkwargs:
    """
    The clone synth path in _make_qwen_synth must pass qwen_clone_genkwargs()
    to generate_voice_clone.  We stub the qwen_tts module at import time so
    no GPU/model dependency is needed.
    """

    def _make_stub_model(self):
        """Return a mock Qwen3TTSModel whose generate_voice_clone records kwargs."""
        import numpy as np

        model = MagicMock()
        dummy_wav = np.zeros(16000, dtype=np.float32)
        model.generate_voice_clone.return_value = ([dummy_wav], 24000)
        model.create_voice_clone_prompt.return_value = MagicMock()
        return model

    def _run_clone_synth(self, model_stub, tmp_path, monkeypatch):
        """
        Patch the lazy qwen_tts import so _make_qwen_synth uses our stub,
        then call the returned synth once with a short text.

        soundfile is in .qwen_venv, not .eval_venv — patch it at the module level
        so the synth's `sf.write` call never hits the real import.
        """
        import sys
        import numpy as np

        # Build a fake qwen_tts module
        fake_qwen_tts = MagicMock()
        fake_qwen_tts.Qwen3TTSModel.from_pretrained.return_value = model_stub
        monkeypatch.setitem(sys.modules, "qwen_tts", fake_qwen_tts)

        # Patch soundfile at the module level — it lives in .qwen_venv, not .eval_venv
        fake_sf = MagicMock()
        monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

        # Build a voice_ref path that exists (clone=True requires os.path.exists)
        voice_ref = str(tmp_path / "ref.wav")
        (tmp_path / "ref.wav").touch()
        # No .txt sidecar → x_vector_only_mode (fine for this test)

        # Patch torch to avoid MPS/CUDA detection on CI
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        # Patch the ASR transcribe so it returns a matching transcript (clean
        # first take → no re-roll, only one generate_voice_clone call).
        monkeypatch.setattr(lt, "_default_transcribe", lambda wav, sr: "hello world")
        monkeypatch.setattr(lt, "spectral_flatness", lambda wav: 0.1)

        synth = lt._make_qwen_synth(voice_ref=voice_ref)
        out = str(tmp_path / "out.wav")
        synth("Hello world", out, exaggeration=0.5)

        return model_stub.generate_voice_clone

    def test_generate_voice_clone_receives_gen_kwargs(self, monkeypatch, tmp_path):
        """generate_voice_clone is called with the keys from qwen_clone_genkwargs."""
        model = self._make_stub_model()
        gvc = self._run_clone_synth(model, tmp_path, monkeypatch)

        assert gvc.called, "generate_voice_clone was never called"
        _, kwargs = gvc.call_args
        expected_keys = {
            "do_sample", "top_k", "top_p", "temperature", "repetition_penalty",
            "subtalker_dosample", "subtalker_top_k", "subtalker_top_p",
            "subtalker_temperature", "max_new_tokens", "non_streaming_mode",
        }
        missing = expected_keys - set(kwargs)
        assert not missing, f"generate_voice_clone missing gen kwargs: {missing}"

    def test_repetition_penalty_value_passed(self, monkeypatch, tmp_path):
        model = self._make_stub_model()
        gvc = self._run_clone_synth(model, tmp_path, monkeypatch)
        _, kwargs = gvc.call_args
        assert kwargs["repetition_penalty"] == pytest.approx(1.1)

    def test_subtalker_top_k_value_passed(self, monkeypatch, tmp_path):
        model = self._make_stub_model()
        gvc = self._run_clone_synth(model, tmp_path, monkeypatch)
        _, kwargs = gvc.call_args
        assert kwargs["subtalker_top_k"] == 50

    def test_non_streaming_mode_passed(self, monkeypatch, tmp_path):
        model = self._make_stub_model()
        gvc = self._run_clone_synth(model, tmp_path, monkeypatch)
        _, kwargs = gvc.call_args
        assert kwargs["non_streaming_mode"] is True

    def test_env_override_reflected_in_call(self, monkeypatch, tmp_path):
        """When STUDIO_QWEN_REP_PENALTY=1.3, generate_voice_clone receives 1.3."""
        monkeypatch.setenv("STUDIO_QWEN_REP_PENALTY", "1.3")
        model = self._make_stub_model()
        gvc = self._run_clone_synth(model, tmp_path, monkeypatch)
        _, kwargs = gvc.call_args
        assert kwargs["repetition_penalty"] == pytest.approx(1.3)
