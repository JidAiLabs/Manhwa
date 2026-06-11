"""ETA model. Medians from stage_run history (series-scoped first), seeded
with this week's measured wall times so day-one estimates are honest."""

from __future__ import annotations

import sqlite3
import statistics
from typing import List, Optional

# measured 2026-06 on the M-series build machine (seconds)
SEED_SEC = {
    "fetched": 30, "stitched": 40, "detected": 60, "scened": 10,
    "visioned": 120, "grouped": 5, "beated": 1980, "scripted": 5,
    "voiced": 1200, "planned": 10, "prepped": 130, "qa_scan": 120,
    "render_segment": 2400, "concat": 180,
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
