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


# A real pipeline stage never legitimately finishes in under ~5s — sub-second
# durations are NO-OP recordings (the resume-by-status runner re-ran a stage that
# was already complete, e.g. chain:scripted = 0.22s). Those would drag the median
# (and the on-screen ETA) absurdly low, so they're excluded; the seed is used
# until a real run lands.
_MIN_REAL_SEC = 5.0


def _median(con: sqlite3.Connection, stage: str,
            series_id: Optional[int]) -> Optional[float]:
    if series_id is not None:
        rows = con.execute(
            "SELECT duration_sec FROM stage_run WHERE stage=? AND ok=1 AND "
            "json_extract(meta_json, '$.series_id')=? AND duration_sec>=?",
            (stage, series_id, _MIN_REAL_SEC)).fetchall()
        if rows:
            return float(statistics.median(r[0] for r in rows))
    rows = con.execute(
        "SELECT duration_sec FROM stage_run WHERE stage=? AND ok=1 AND "
        "duration_sec>=?", (stage, _MIN_REAL_SEC)).fetchall()
    if rows:
        return float(statistics.median(r[0] for r in rows))
    return None


def stage_eta(con: sqlite3.Connection, stage: str,
              series_id: Optional[int] = None) -> float:
    return _median(con, stage, series_id) or float(SEED_SEC.get(stage, 300))


def job_eta(con: sqlite3.Connection, job_type: str,
            series_id: Optional[int] = None) -> Optional[float]:
    """Median WALL-CLOCK of finished jobs of this type (series-scoped first) — the
    honest running-job estimate. Summing per-stage medians under-counts: it misses
    heal cycles and the prep+QA that ride INSIDE a voiceover job (why a voiceover
    showed '~7:35' against a real ~20-30 min). The real job duration includes all
    of it. Returns None (caller falls back to the stage-sum seed) until history."""
    candidates: List[Optional[int]] = []
    if series_id is not None:
        candidates.append(series_id)
    candidates.append(None)
    for sid in candidates:
        q = ("SELECT (julianday(finished_at)-julianday(started_at))*86400.0 "
             "FROM job WHERE type=? AND state='done' AND started_at IS NOT NULL "
             "AND finished_at IS NOT NULL")
        params: List = [job_type]
        if sid is not None:
            q += " AND series_id=?"
            params.append(sid)
        durs = [r[0] for r in con.execute(q, params).fetchall()
                if r[0] is not None and r[0] >= _MIN_REAL_SEC]
        if durs:
            return float(statistics.median(durs))
    return None


# the four stages of a chapter's "prepare" — all Gemma-bound (understanding +
# narration + the semantic QA), so all on the ONE GPU.
_PREP_STAGES = ("chain:scripted", "prepped", "qa_scan", "planned")


def readiness_parts(con: sqlite3.Connection,
                    series_id: Optional[int]) -> tuple:
    """(prep, voice, render) MEASURED median seconds per chapter — the real
    averages behind the readiness estimate (seeds only until a real run lands)."""
    prep = sum(stage_eta(con, s, series_id) for s in _PREP_STAGES)
    return (prep, stage_eta(con, "voiced", series_id),
            stage_eta(con, "render_segment", series_id))


def lane_bottleneck_sec(con: sqlite3.Connection, series_id: Optional[int],
                        target: str = "video") -> float:
    """Real-median per-chapter wall time. The ONE GPU serializes BOTH Gemma (the
    prepare's understanding/narration + semantic QA) AND Qwen (the voiceover) —
    concurrent jobs SHARE it, so there is no 2x speedup; only the render (CPU /
    Remotion) overlaps. So per chapter the GPU does prep + voice back-to-back and
    the build is bounded by that. target: 'qa' | 'voice' | 'video'."""
    prep, voice, render = readiness_parts(con, series_id)
    gpu = prep + (voice if target in ("voice", "video") else 0.0)
    return max(gpu, render if target == "video" else 0.0)


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
    # build time is bottlenecked by the slowest PARALLEL lane, not the serial
    # sum of every stage (which double-counted the vestigial 'beated' seed and
    # read ~3x too high). Matches the bulk run-range estimate.
    return chapters_remaining * lane_bottleneck_sec(con, series_id, "video")


def fmt_eta(sec: float) -> str:
    sec = max(0, float(sec))
    if sec < 3600:
        return f"{int(sec // 60)}:{int(sec % 60):02d}"
    if sec < 86400:
        return f"{sec / 3600:.1f} h"
    return f"{int(round(sec / 86400))} days"
