import os
from pathlib import Path


def resolve_rel(manifest_path, stored: str) -> Path:
    """Resolve a stored path against the manifest's directory.

    Absolute stored paths pass through unchanged (back-compat).
    """
    if os.path.isabs(stored):
        return Path(stored)
    return Path(manifest_path).parent / stored
