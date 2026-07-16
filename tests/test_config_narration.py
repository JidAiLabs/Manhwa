"""[narration] config table: segmentation = "adaptive" | "per_panel"."""
from studio.config import load, REPO_ROOT


def test_segmentation_default_is_adaptive(tmp_path, monkeypatch):
    monkeypatch.delenv("STUDIO_NARR_SEGMENTATION", raising=False)
    toml = tmp_path / "studio.toml"
    toml.write_text("")
    assert load(toml).segmentation == "adaptive"


def test_repo_toml_yields_a_valid_segmentation(monkeypatch):
    monkeypatch.delenv("STUDIO_NARR_SEGMENTATION", raising=False)
    cfg = load(REPO_ROOT / "studio.toml")
    assert cfg.segmentation in ("adaptive", "per_panel")


def test_narration_table_parsed(tmp_path, monkeypatch):
    monkeypatch.delenv("STUDIO_NARR_SEGMENTATION", raising=False)
    toml = tmp_path / "studio.toml"
    toml.write_text('[narration]\nsegmentation = "per_panel"\n')
    assert load(toml).segmentation == "per_panel"


def test_invalid_segmentation_falls_back_to_adaptive_with_warning(
        tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("STUDIO_NARR_SEGMENTATION", raising=False)
    toml = tmp_path / "studio.toml"
    toml.write_text('[narration]\nsegmentation = "bogus"\n')
    assert load(toml).segmentation == "adaptive"
    assert "segmentation" in capsys.readouterr().err


def test_env_overrides_toml(tmp_path, monkeypatch):
    # repo convention: STUDIO_* env wins over studio.toml (per-run toggle),
    # mirroring punchup/semantic_heal.
    monkeypatch.setenv("STUDIO_NARR_SEGMENTATION", "per_panel")
    toml = tmp_path / "studio.toml"
    toml.write_text('[narration]\nsegmentation = "adaptive"\n')
    assert load(toml).segmentation == "per_panel"


# ---- 2026-07-16 wave: semantic_heal ON by default, env is a TWO-WAY hatch ----

def test_semantic_heal_repo_default_is_on(monkeypatch):
    monkeypatch.delenv("STUDIO_SEMANTIC_HEAL", raising=False)
    assert load(REPO_ROOT / "studio.toml").semantic_heal is True


def test_semantic_heal_env_disables_and_enables(tmp_path, monkeypatch):
    toml = tmp_path / "studio.toml"
    toml.write_text("[models]\nsemantic_heal = true\n")
    monkeypatch.setenv("STUDIO_SEMANTIC_HEAL", "0")
    assert load(toml).semantic_heal is False     # the rollback lever
    monkeypatch.setenv("STUDIO_SEMANTIC_HEAL", "1")
    toml.write_text("[models]\nsemantic_heal = false\n")
    assert load(toml).semantic_heal is True
    monkeypatch.delenv("STUDIO_SEMANTIC_HEAL")
    assert load(toml).semantic_heal is False     # toml rules when env unset
