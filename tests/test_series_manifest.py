# tests/test_series_manifest.py
import json
from studio.pipeline import write_series_manifest


def test_write_series_manifest_roundtrip(tmp_path):
    write_series_manifest(str(tmp_path), "C", "A")
    d = json.loads((tmp_path / "manifest.series.json").read_text())
    assert d["niche_primary"] == "C" and d["niche_secondary"] == "A"


def test_write_series_manifest_handles_empty(tmp_path):
    write_series_manifest(str(tmp_path), None, None)  # no niche -> still writes a file
    d = json.loads((tmp_path / "manifest.series.json").read_text())
    assert d["niche_primary"] in ("", None)
