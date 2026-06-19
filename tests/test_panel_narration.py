"""gemini_narrative_pass: per-panel narration alignment + schema tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gemini_narrative_pass",
    Path(__file__).resolve().parent.parent / "tools" / "gemini_narrative_pass.py")
gnp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gnp)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Task 3-pre: build_arg_parser + --understood flag
# ---------------------------------------------------------------------------

def test_build_arg_parser_understood_flag():
    parser = gnp.build_arg_parser()
    args = parser.parse_args([
        "--groups-manifest", "g.json",
        "--vision-manifest", "v.json",
        "--out", "out.json",
        "--understood", "x.json",
    ])
    assert args.understood == "x.json"
