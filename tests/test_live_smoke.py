"""
tests/test_live_smoke.py

Live end-to-end smoke tests for the studio acquisition pipeline.

These tests HIT THE NETWORK and write real files.  They are NOT run in CI.
Run manually with:

    .eval_venv/bin/python -m pytest -m live tests/test_live_smoke.py -v

Every test is decorated ``@pytest.mark.live``.  The default ``pytest`` run
skips them because the ``live`` marker is not selected.

Confirmed live behaviour (2026-06-09):
  - webtoon add-series → 309 chapters discovered
  - webtoon fetch ch1 → 64 images downloaded, status=downloaded
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Happy-path: Webtoon / Omniscient Reader
# ---------------------------------------------------------------------------

_ORV_URL = "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"


@pytest.mark.live
def test_webtoon_add_and_fetch_chapter_1(tmp_path, monkeypatch):
    """Full add-series + fetch cycle against live webtoons.com.

    Uses a temp SQLite DB (monkeypatches ``studio.cli._db_path``) and a temp
    ongoing/ directory (monkeypatches ``studio.config.REPO_ROOT``).

    Assertions:
    - Chapter 1 status becomes ``downloaded``
    - At least one image file ``001.jpg`` exists in the episode directory
    - ``001.jpg`` opens as a valid JPEG via PIL (not a truncated/corrupt file)

    The test writes into *tmp_path* only; nothing persists after the test
    session ends.  It does NOT clean up ``tmp_path`` automatically — pytest
    retains the last three runs' tmp dirs for post-mortem inspection.
    """
    from PIL import Image

    import studio.cli as cli_mod
    from studio import config as studio_config
    from studio.catalog import db as catalog_db
    from studio.catalog import repo

    # --- Redirect DB to tmp ---
    db_file = tmp_path / "studio.db"
    monkeypatch.setattr(cli_mod, "_db_path", lambda: db_file)

    # --- Redirect ongoing/ to tmp ---
    monkeypatch.setattr(studio_config, "REPO_ROOT", tmp_path)

    # --- add-series ---
    cli_mod.main(["add-series", "webtoon", _ORV_URL])

    con = catalog_db.connect(db_file)
    series_list = repo.list_series(con)
    assert series_list, "add-series must create at least one series row"
    sid = series_list[0].id

    chapters = repo.list_chapters(con, sid)
    assert len(chapters) >= 300, (
        f"Expected 300+ chapters for ORV, got {len(chapters)}"
    )

    # --- fetch chapter 1 ---
    cli_mod.main(["fetch", str(sid), "--chapters", "1"])

    # Re-query to get updated status
    chapters = repo.list_chapters(con, sid)
    ch1 = next((c for c in chapters if c.number == 1.0), None)
    assert ch1 is not None, "Chapter 1 must exist in catalog"
    assert ch1.status == "downloaded", (
        f"Expected status 'downloaded', got '{ch1.status}'"
    )
    assert ch1.ep_dir is not None, "ep_dir must be set after fetch"

    ep_dir = Path(ch1.ep_dir)
    assert ep_dir.is_dir(), f"ep_dir must be a directory: {ep_dir}"

    first_image = ep_dir / "001.jpg"
    assert first_image.exists(), f"001.jpg must exist in {ep_dir}"
    assert first_image.stat().st_size > 0, "001.jpg must not be empty"

    # Verify it's a valid image PIL can decode
    img = Image.open(first_image)
    img.verify()  # raises on corrupt/truncated data
