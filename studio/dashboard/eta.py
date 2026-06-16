"""ETA model. Medians from stage_run history (series-scoped first), seeded
with this week's measured wall times so day-one estimates are honest."""

from __future__ import annotations

import sqlite3
import statistics
from typing import List, Optional

# measured 2026-06-17 end-to-end on the M-series build machines (seconds).
# voiced/render_segment were previously SEEDED at the old "~40min render"
# guess (1200/2400); the real validated numbers are ~7min TTS, ~2.5-5min render.
SEED_SEC = {
    "fetched": 30, "stitched": 40, "detected": 60, "scened": 10,
    "visioned": 120, "grouped": 5, "beated": 1980, "scripted": 5,
    "voiced": 430, "planned": 10, "prepped": 80, "qa_scan": 200,
    "render_segment": 300, "concat": 60,
    # local-gemma prepare (fetch -> understand -> group -> narrate -> script) as
    # ONE recorded stage; ~3.5min measured median, refined by the live median
    "chain:scripted": 220,
}

# which worker lane each stage runs on, and the lane's parallel width — the
# build-time bottleneck is the SLOWEST lane, not the serial sum of stages.
# (mirrors studio/dashboard/jobs.py LANES / LANE_WIDTH.)
LANE_OF = {
    "chain:scripted": "gpu", "prepped": "gpu", "qa_scan": "gpu",
    "planned": "gpu", "voiced": "tts", "render_segment": "cpu",
}


def _median(con: sqlite3.Connection, stage: str,
            series_id: Optional[int]) -> Optional[float]:
    if series_id is not None:
        rows = con.execute(
            "SELECT duration_sec FROM stage_run WHERE stage=? AND ok=1 AND "
            "json_extract(meta_json, '$.series_id')=? AND duration_sec>0",
            (stage, series_id)).fetchall()
        if rows:
            return float(statistics.median(r[0] for r in rows))
    rows = con.execute(
        "SELECT duration_sec FROM stage_run WHERE stage=? AND ok=1 AND "
        "duration_sec>0", (stage,)).fetchall()
    if rows:
        return float(statistics.median(r[0] for r in rows))
    return None


def stage_eta(con: sqlite3.Connection, stage: str,
              series_id: Optional[int] = None) -> float:
    return _median(con, stage, series_id) or float(SEED_SEC.get(stage, 300))


def lane_bottleneck_sec(con: sqlite3.Connection, series_id: Optional[int],
                        target: str = "video") -> float:
    """Steady-state per-chapter BUILD time for a bulk run = the slowest worker
    lane. The lanes (gpu/tts/cpu) run in parallel, so across many chapters the
    throughput is bounded by the busiest lane — NOT the serial sum of every
    stage. target: 'qa' (prepare->QA only) | 'voice' (+TTS) | 'video' (+render).
    Measured reality: TTS (~7min/ch, width 1) is the bottleneck, so a 300-ep
    series is ~1.5-2 days, not the ~40min/ch the old serial-sum seed implied."""
    import os
    width = {"gpu": int(os.environ.get("STUDIO_GPU_WIDTH", "2")),
             "tts": int(os.environ.get("STUDIO_TTS_WIDTH", "1")),
             "cpu": int(os.environ.get("STUDIO_CPU_WIDTH", "1"))}
    stages = ["chain:scripted", "prepped", "qa_scan", "planned"]   # gpu: prepare->QA
    if target in ("voice", "video"):
        stages.append("voiced")                                    # tts lane
    if target == "video":
        stages.append("render_segment")                           # cpu lane
    lane_work: dict = {}
    for s in stages:
        lane = LANE_OF.get(s, "gpu")
        lane_work[lane] = lane_work.get(lane, 0.0) + stage_eta(con, s, series_id)
    if not lane_work:
        return 0.0
    return max(w / max(1, width.get(lane, 1)) for lane, w in lane_work.items())


def chapter_eta(con: sqlite3.Connection, chapter_id: int,
                remaining: List[str], series_id: Optional[int] = None) -> float:
    return sum(stage_eta(con, s, series_id) for s in remaining)


def chapter_wall_median(con: sqlite3.Connection,
                        series_id: Optional[int] = None) -> float:
    """Full-chapter wall estimate = sum of per-stage estimates."""
    stages = ["fetched", "stitched", "detected", "scened", "visioned",
              "grouped", "beated", "scripted", "voiced", "planned",
              "prepped", "qa_scan", "render_segment"]
    return sum(stage_eta(con, s, series_id) for s in stages)


def series_eta(con: sqlite3.Connection, series_id: int,
               chapters_remaining: int) -> float:
    return chapters_remaining * chapter_wall_median(con, series_id)


def fmt_eta(sec: float) -> str:
    sec = max(0, float(sec))
    if sec < 3600:
        return f"{int(sec // 60)}:{int(sec % 60):02d}"
    if sec < 86400:
        return f"{sec / 3600:.1f} h"
    return f"{int(round(sec / 86400))} days"
