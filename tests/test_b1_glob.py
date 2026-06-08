"""
Test B1: vision_extract --glob default is *.jpg (not scene_*.jpg).

Because vision_extract.py imports google.cloud.vision (absent in test env),
we verify the default by reading the source via AST — this tests the real
current text of the file, not a mock, so any revert to scene_*.jpg fails.
"""

import ast
import pathlib


_VISION_EXTRACT = pathlib.Path(__file__).parent.parent / "tools" / "vision_extract.py"


def _find_glob_default(source: str) -> str | None:
    """Walk the AST and return the default value for the --glob argument."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # Looking for: ap.add_argument("--glob", default=<value>, ...)
        if not isinstance(node, ast.Call):
            continue
        # Must be a call whose first positional arg is the string "--glob"
        if not node.args:
            continue
        first_arg = node.args[0]
        if not (isinstance(first_arg, ast.Constant) and first_arg.value == "--glob"):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                if isinstance(kw.value, ast.Constant):
                    return kw.value.value
    return None


def test_glob_default_is_star_jpg():
    source = _VISION_EXTRACT.read_text(encoding="utf-8")
    default = _find_glob_default(source)
    assert default is not None, "--glob argument not found in vision_extract.py"
    assert default == "*.jpg", (
        f"Expected --glob default '*.jpg', got '{default}'. "
        "The old 'scene_*.jpg' default would silently skip panels not named scene_*."
    )


def test_old_scene_glob_gone():
    source = _VISION_EXTRACT.read_text(encoding="utf-8")
    default = _find_glob_default(source)
    assert default != "scene_*.jpg", (
        "--glob default is still 'scene_*.jpg'; fix was not applied."
    )
