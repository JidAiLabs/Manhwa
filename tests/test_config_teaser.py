from studio.config import load, REPO_ROOT


def test_teaser_config_defaults_and_toml():
    cfg = load(REPO_ROOT / "studio.toml")
    assert cfg.teaser_enabled is True
    assert cfg.teaser_shortlist_n == 4
    assert cfg.teaser_min_panels == 4
    assert cfg.teaser_max_hook_panels == 10
    assert cfg.teaser_max_hook_scan_chapters == 12
    assert cfg.teaser_max_seconds == 90
    assert 0.0 < cfg.teaser_payoff_tail_frac < 1.0


def test_publish_auto_after_chapters_default_and_toml(tmp_path):
    cfg = load(REPO_ROOT / "studio.toml")
    assert cfg.publish_auto_after_chapters == 12         # shipped default

    # env override wins
    import os
    os.environ["STUDIO_PUBLISH_AUTO_AFTER"] = "5"
    try:
        assert load(REPO_ROOT / "studio.toml").publish_auto_after_chapters == 5
    finally:
        del os.environ["STUDIO_PUBLISH_AUTO_AFTER"]

    # a toml with no [publish] section still gets the default (back-compat)
    toml = tmp_path / "s.toml"
    toml.write_text('[teaser]\nenabled = false\n')
    assert load(toml).publish_auto_after_chapters == 12
