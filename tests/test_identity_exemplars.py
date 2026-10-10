"""tools/identity_exemplars.py — close-up proposals and the 2 + 2 pick."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import identity_exemplars as ie  # noqa: E402


def test_proposals_are_protagonist_closeups_from_distinct_chapters(tmp_path):
    from test_identity_sweep import _series, _seams
    import identity_sweep as sw
    sd, who = _series(tmp_path, "s", [(30, 4)] * 6)
    h, e = _seams(who)
    sw.sweep_series(sd, heads_fn=h, embed_fn=e)
    props = ie.propose(sd)
    m = [v for k, v in props.items() if k.startswith("M")]
    assert len(m) == 6                                  # one per chapter, 6 chapters
    assert len({v["chapter"] for v in m}) == 6
    assert all(who[str(sd / v["chapter"] / "scenes" / v["panel"])] == "mc" for v in m)
    ie.sheet(sd, props, tmp_path / "sheet.jpg")
    assert (tmp_path / "sheet.jpg").exists()


def test_pick_writes_two_plus_two_from_one_group(tmp_path):
    sd = ROOT / "ongoing" / "s"
    props = {k: {"chapter": f"Chapter_{i}", "panel": f"p{i}.jpg"}
             for i, k in enumerate(["M1", "M2", "G1a", "G1b", "G2a"])}
    out = tmp_path / "s.exemplars.json"
    obj = ie.write_pick(sd, props, ["M1", "M2"], ["G1a", "G1b"], out)
    assert obj["protagonist"] == ["ongoing/s/Chapter_0/scenes/p0.jpg",
                                  "ongoing/s/Chapter_1/scenes/p1.jpg"]
    assert json.loads(out.read_text())["decoy"][0] == "ongoing/s/Chapter_2/scenes/p2.jpg"
    with pytest.raises(SystemExit):
        ie.write_pick(sd, props, ["M1"], ["G1a", "G1b"], out)        # not 2 + 2
    with pytest.raises(SystemExit):
        ie.write_pick(sd, props, ["M1", "M2"], ["G1a", "G2a"], out)  # two characters
