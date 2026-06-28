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
