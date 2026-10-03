from studio.config import load, REPO_ROOT


def test_teaser_config_defaults_and_toml():
    cfg = load(REPO_ROOT / "studio.toml")
    # OFF since 2026-09-05 (owner): this flag gates the AUTO paths — the
    # auto-debut bundle and the auto-intro. The debut proposer guarded on
    # "does this series have a bundle right now" rather than the thumbnail
    # proposer's "propose exactly ONCE, ever", so a deleted video came back on
    # the next prepare. The manual "Plan teaser" button is unconditional and
    # still works; videos are created by hand.
    assert cfg.teaser_enabled is False
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


def test_row_layout_config_defaults_and_off_switch(tmp_path):
    # Rows (2-3 panels one line covers, side by side) ship OFF: the owner signs
    # off one real chapter, then flips row_min_fit to 0.52 in studio.toml.
    cfg = load(REPO_ROOT / "studio.toml")
    assert cfg.row_min_fit == 0.0
    assert cfg.row_min_gap_items == 3
    toml = tmp_path / "s.toml"
    toml.write_text('[teaser]\nenabled = false\n')
    assert load(toml).row_min_fit == 0.52          # default without [render]
    assert load(toml).row_min_gap_items == 3
    # 0.0 is the OFF switch — the loader must not read it as "unset"
    toml.write_text('[render]\nrow_min_fit = 0.0\n')
    assert load(toml).row_min_fit == 0.0
