"""chunk_stitch_adaptive: a chapter-tall GUTTERLESS strip (continuous art, no
white bands) must be sub-cut to a YOLO-processable height. Before the fix the
final remainder was only JPEG-capped (65k), so ch28's 61.6k-px second half
shipped as ONE chunk and the detector downscaled it to a single panel."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "chunk_stitch_adaptive.py"


def test_tall_gutterless_strip_subcut_to_cap(tmp_path):
    # pure noise = no white/flat gutter band anywhere -> the stitcher cannot find
    # a safe cut and must force-cut at the cap (exactly the ch28 monster case).
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(24000, 400, 3), dtype="uint8")
    Image.fromarray(arr).save(tmp_path / "001.jpg", quality=70)
    r = subprocess.run(
        [sys.executable, str(_TOOL), "--episode-dir", str(tmp_path),
         "--out-dir", str(tmp_path / "chunks"),
         "--max-chunk-height", "6000", "--max-overflow-px", "1000"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    man = json.loads((tmp_path / "manifest.stitch.json").read_text())
    hard_cap = man["max_chunk_height"] + man["adaptive"]["max_overflow_px"]
    heights = [c["chunk_h"] for c in man["chunks"]]
    assert max(heights) <= hard_cap, heights      # no monster chunk survives
    assert len(heights) >= 3                        # the 24k strip WAS subdivided
