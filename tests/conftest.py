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


# A handful of tests build a subprocess env from a *copy* of os.environ (e.g.
# tests/test_verbatim_script.py, tests/test_script_expander_segments.py,
# tests/test_flow_e2e.py) or exercise MissingCredential/backend-selection
# paths that key off these same names (tests/test_pipeline.py,
# tests/test_teaser_worker.py). None of them legitimately needs a REAL
# ambient value — every test that cares sets its own via monkeypatch — so the
# suite must pass identically whether or not the developer's shell happens to
# export real keys (creds.env, direnv, ...). Deleting them here, before every
# test, makes that true instead of accidentally true.
_AMBIENT_CRED_KEYS = (
    "OPENAI_API_KEY", "ELEVENLABS_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
)


@pytest.fixture(autouse=True)
def _no_ambient_llm_credentials(monkeypatch):
    for key in _AMBIENT_CRED_KEYS:
        monkeypatch.delenv(key, raising=False)
