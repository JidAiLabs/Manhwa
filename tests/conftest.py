"""Skip @pytest.mark.live tests unless explicitly selected with `-m live`.

Live tests hit the network (real source sites) and are slow, so the default
`pytest` run skips them. Run them on demand with:  pytest -m live
"""
import pytest


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr", "") or ""
    if "live" in markexpr:
        return  # caller explicitly asked for live tests
    skip_live = pytest.mark.skip(reason="live/network test; run with `-m live`")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
