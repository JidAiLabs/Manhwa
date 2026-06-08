"""B2 regression: a group with multiple narration paragraphs must yield one
timeline item per paragraph in render.plan.json, each keyed by its own
segment_id with its own tts_audio. Before the fix, group-keyed indexing
collapsed the group to a single item and dropped every paragraph's audio but
the last."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLANNER = REPO / "tools" / "timeline_planner.py"


def test_multi_paragraph_group_yields_two_items(tmp_path):
    groups = {"shots": [{"group_id": 1, "shot_id": 1,
                         "scene_files": ["p001.jpg", "p002.jpg"]}]}
    script = {"sections": [{
        "section_index": 0,
        "script_paragraphs": [{"text": "Paragraph zero."}, {"text": "Paragraph one."}],
        "shots": [{"group_id": 1, "segment_id": "g0001_p00"},
                  {"group_id": 1, "segment_id": "g0001_p01"}],
    }]}
    tts = {"clips": [
        {"segment_id": "g0001_p00", "group_id": 1, "audio_file": "g0001_p00.mp3", "duration_sec": 3.0},
        {"segment_id": "g0001_p01", "group_id": 1, "audio_file": "g0001_p01.mp3", "duration_sec": 4.0},
    ]}
    (tmp_path / "groups.json").write_text(json.dumps(groups))
    (tmp_path / "script.json").write_text(json.dumps(script))
    (tmp_path / "tts.json").write_text(json.dumps(tts))
    out = tmp_path / "render.plan.json"

    r = subprocess.run(
        [sys.executable, str(PLANNER),
         "--groups", str(tmp_path / "groups.json"),
         "--script", str(tmp_path / "script.json"),
         "--tts-index", str(tmp_path / "tts.json"),
         "--out", str(out), "--mode", "narrated"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"planner failed:\n{r.stderr}"

    plan = json.loads(out.read_text())
    items = plan["timeline"]
    assert len(items) == 2, f"expected 2 timeline items, got {len(items)}"

    seg_ids = [it["segment_id"] for it in items]
    assert seg_ids == ["g0001_p00", "g0001_p01"], seg_ids

    audios = [it["tts_audio"] for it in items]
    assert audios[0].endswith("g0001_p00.mp3"), audios
    assert audios[1].endswith("g0001_p01.mp3"), audios
    assert audios[0] != audios[1]

    # sequential timing: paragraph two starts no earlier than paragraph one ends
    assert items[1]["start_sec"] >= items[0]["end_sec"] - 1e-3
