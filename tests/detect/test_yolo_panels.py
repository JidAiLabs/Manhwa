"""
Integration tests for detect_panels — require ultralytics + trained weights.

Run with:  pytest tests/detect/test_yolo_panels.py -v -m requires_ultralytics
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

WEIGHTS = "/Users/anka/webtoon-ai/runs/detect/webtoon/yolo26_musgd_run/weights/best.pt"
FIXTURES = Path(__file__).parent / "fixtures"
STITCH_MANIFEST = str(FIXTURES / "manifest.stitch.json")


@pytest.mark.requires_ultralytics
def test_detect_panels_output_schema():
    """detect_panels returns schema-compatible output and writes valid JSON."""
    from studio.detect.yolo_panels import detect_panels

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = detect_panels(
            stitch_manifest_path=STITCH_MANIFEST,
            out_path=out_path,
            weights=WEIGHTS,
            conf=0.25,
        )

        # Top-level key
        assert "chunks" in result, f"missing 'chunks' key, got: {list(result.keys())}"

        # Written file matches returned dict
        with open(out_path, "r", encoding="utf-8") as f:
            written = json.load(f)
        assert written == result, "written JSON does not match returned dict"

        # Validate each chunk entry
        for chunk in result["chunks"]:
            assert "chunk_file" in chunk, f"missing 'chunk_file' in chunk: {chunk}"
            assert "panels_norm" in chunk, f"missing 'panels_norm' in chunk: {chunk}"
            assert isinstance(chunk["panels_norm"], list), "panels_norm must be a list"

            for panel in chunk["panels_norm"]:
                assert isinstance(panel, list), f"each panel must be a list, got: {type(panel)}"
                assert len(panel) == 4, f"each panel must have 4 values, got: {panel}"
                ymin, xmin, ymax, xmax = panel
                assert 0.0 <= ymin <= 1.0, f"ymin out of range: {ymin}"
                assert 0.0 <= xmin <= 1.0, f"xmin out of range: {xmin}"
                assert 0.0 <= ymax <= 1.0, f"ymax out of range: {ymax}"
                assert 0.0 <= xmax <= 1.0, f"xmax out of range: {xmax}"
                assert ymin < ymax, f"ymin ({ymin}) must be < ymax ({ymax})"

        # At least one chunk processed
        assert len(result["chunks"]) == 1, f"expected 1 chunk, got {len(result['chunks'])}"

        # chunk_file matches fixture basename
        assert result["chunks"][0]["chunk_file"] == "sample_chunk.jpg"

    finally:
        os.unlink(out_path)


@pytest.mark.requires_ultralytics
def test_detect_panels_sorted_top_to_bottom():
    """Panels within each chunk are sorted by ymin ascending."""
    from studio.detect.yolo_panels import detect_panels

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = detect_panels(
            stitch_manifest_path=STITCH_MANIFEST,
            out_path=out_path,
            weights=WEIGHTS,
            conf=0.25,
        )
        for chunk in result["chunks"]:
            ymins = [p[0] for p in chunk["panels_norm"]]
            assert ymins == sorted(ymins), f"panels not sorted by ymin: {ymins}"
    finally:
        os.unlink(out_path)
