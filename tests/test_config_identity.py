"""[identity] config: the image-identity venv + backend switch (2026-10-10)."""
import pytest

from studio.config import REPO_ROOT, load


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("STUDIO_IDENTITY_PYTHON", "STUDIO_IDENTITY_BACKEND",
              "STUDIO_TTS_PYTHON"):
        monkeypatch.delenv(k, raising=False)


def test_identity_python_resolves_like_tts_python(tmp_path, monkeypatch):
    toml = tmp_path / "studio.toml"
    toml.write_text('[identity]\npython = ".identity_venv/bin/python"\n')
    assert load(toml).identity_python == str(REPO_ROOT / ".identity_venv/bin/python")
    # the TTS override never leaks into the identity interpreter
    monkeypatch.setenv("STUDIO_TTS_PYTHON", "/tts/py")
    assert load(toml).identity_python == str(REPO_ROOT / ".identity_venv/bin/python")
    monkeypatch.setenv("STUDIO_IDENTITY_PYTHON", "/custom/py")
    assert load(toml).identity_python == "/custom/py"
    # and the identity override never leaks into TTS
    assert load(toml).tts_python == "/tts/py"


def test_identity_backend_default_toml_env_and_invalid(tmp_path, monkeypatch, capsys):
    toml = tmp_path / "studio.toml"
    toml.write_text("")
    cfg = load(toml)
    assert cfg.identity_backend == "gemma" and cfg.identity_python == ""
    toml.write_text('[identity]\nbackend = "ccip"\n')
    assert load(toml).identity_backend == "ccip"
    monkeypatch.setenv("STUDIO_IDENTITY_BACKEND", "off")
    assert load(toml).identity_backend == "off"
    monkeypatch.setenv("STUDIO_IDENTITY_BACKEND", "faces")
    assert load(toml).identity_backend == "gemma"
    assert "identity.backend" in capsys.readouterr().err


def test_identity_backend_per_series_and_env_rollback(tmp_path, monkeypatch):
    toml = tmp_path / "studio.toml"
    toml.write_text('[identity]\nbackend = "gemma"\n'
                    '[identity.series]\n"omniscient-reader" = "ccip"\n'
                    '"bad-series" = "faces"\n')
    cfg = load(toml)
    assert cfg.identity_backend_for("omniscient-reader") == "ccip"
    assert cfg.identity_backend_for("some-other-series") == "gemma"
    assert cfg.identity_backend_for("bad-series") == "gemma"
    # the env lever is a fleet-wide rollback: it beats every per-series entry
    monkeypatch.setenv("STUDIO_IDENTITY_BACKEND", "off")
    assert load(toml).identity_backend_for("omniscient-reader") == "off"
