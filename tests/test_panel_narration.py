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


# ---------------------------------------------------------------------------
# Task 3a: align_panel_narration repair-fill helper
# ---------------------------------------------------------------------------

def test_align_pads_missing_panels_from_understanding():
    files = ["a.jpg", "b.jpg", "c.jpg"]
    model = [{"scene_file": "a.jpg", "line": "He draws the blade."},
             {"scene_file": "c.jpg", "line": "Silence falls."}]   # b missing
    u = {"b.jpg": {"description": "the beast lunges"}}
    out = gnp.align_panel_narration(files, model, u)
    assert [p["scene_file"] for p in out] == files
    assert out[1]["line"] == "the beast lunges"

def test_align_is_positional_when_model_omits_scene_file():
    files = ["a.jpg", "b.jpg"]
    model = [{"line": "First."}, {"line": "Second."}]
    out = gnp.align_panel_narration(files, model, {})
    assert [p["line"] for p in out] == ["First.", "Second."]

def test_align_folds_overflow_into_last_panel_no_phantoms():
    files = ["a.jpg"]
    model = [{"scene_file": "a.jpg", "line": "One."}, {"scene_file": "zzz.jpg", "line": "Two."}]
    out = gnp.align_panel_narration(files, model, {})
    assert len(out) == 1 and out[0]["scene_file"] == "a.jpg"
    assert out[0]["line"] == "One. Two."

def test_align_invariant_length_matches_scene_files():
    files = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    out = gnp.align_panel_narration(files, [], {})
    assert len(out) == len(files)
    assert all(p["line"] for p in out)


# ---------------------------------------------------------------------------
# Task 3b: build_beat_schema + panel_narration field
# ---------------------------------------------------------------------------

def test_beat_schema_requires_panel_narration():
    schema = gnp.build_beat_schema()
    props = schema["properties"]
    assert "panel_narration" in props
    assert props["panel_narration"]["type"] == "ARRAY"
    item = props["panel_narration"]["items"]["properties"]
    assert set(item) >= {"scene_file", "line"}
    assert "panel_narration" in schema["required"]
    assert "narration" in props          # joined string kept for back-compat
