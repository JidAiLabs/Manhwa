"""OriginPower Studio dashboard — FastAPI + Jinja + htmx.

UI handlers READ the catalog and INSERT job/approval rows. They never
execute pipeline work; `studio worker` consumes the queue and enforces the
gates. Visual contract: docs/plans/specs/mockups/dashboard-mockup.html.
"""

from __future__ import annotations

import json
import os
import secrets as _secrets
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from studio.catalog.db import connect
from studio.catalog.models import STATUS_ORDER
from studio.dashboard import bundles, discovery, eta, gates, jobs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# stages shown on the chapter timeline, in pipeline order
TIMELINE = ["fetched", "stitched", "detected", "scened", "visioned",
            "grouped", "beated", "scripted", "voiced", "planned",
            "qa_scan", "render_segment"]


def _status_idx(status: str) -> int:
    try:
        return STATUS_ORDER.index(status)
    except ValueError:
        return 0


def _chapter_costs(ep_dir: Optional[str]) -> float:
    total = 0.0
    if not ep_dir:
        return total
    for fn in ("manifest.beats.json", "manifest.cast.json",
               "manifest.script.json"):
        p = Path(ep_dir) / fn
        if p.exists():
            try:
                u = (json.loads(p.read_text()).get("stats") or {}).get(
                    "usage") or {}
                total += float(u.get("est_cost_usd") or 0.0)
            except Exception:
                pass
    return total


def _gallery(ep_dir: Optional[str]) -> List[Dict[str, Any]]:
    """Segment blocks (narration + panels). Prefers the prepped clean plan;
    before one exists, falls back to manifest.beats.json so the narration
    can be REVIEWED (and approved for voiceover) right after the script
    stage — confirm upstream before spending GPU time downstream."""
    if not ep_dir:
        return []
    p = Path(ep_dir) / "render.plan.clean.json"
    if p.exists():
        try:
            plan = json.loads(p.read_text())
        except Exception:
            return []
        out = []
        for item in plan.get("timeline") or []:
            if item.get("branding"):
                continue
            files = []
            for c in item.get("cuts") or []:
                for f in (c.get("file"), c.get("file2")):
                    if f:
                        files.append(str(f))
            out.append({"segment_id": item.get("segment_id"),
                        "narration": item.get("tts_text") or "",
                        "files": files, "src_dir": "scenes_clean",
                        "duration": item.get("duration_sec") or 0})
        return out

    b = Path(ep_dir) / "manifest.beats.json"
    if not b.exists():
        return []
    try:
        beats = json.loads(b.read_text()).get("beats") or []
    except Exception:
        return []
    return [{"segment_id": f"g{int(bt.get('group_id') or 0):04d}",
             "narration": bt.get("narration") or "",
             "files": [str(f) for f in (bt.get("scene_files") or [])[:4]],
             "src_dir": "scenes", "duration": 0}
            for bt in beats]


def _stage_timeline(con: sqlite3.Connection, ch: Dict[str, Any]) -> List[Dict[str, Any]]:
    done_idx = _status_idx(ch["status"])
    runs: Dict[str, float] = {}
    for stage, dur in con.execute(
            "SELECT stage, duration_sec FROM stage_run WHERE chapter_id=? "
            "AND ok=1 ORDER BY id", (ch["id"],)):
        runs[stage] = dur
    rows = []
    for s in TIMELINE:
        in_catalog = s in STATUS_ORDER
        is_done = (in_catalog and _status_idx(s) <= done_idx) or s in runs
        rows.append({
            "stage": s,
            "done": is_done,
            "dur": runs.get(s),
            "eta": None if is_done else eta.stage_eta(con, s, ch["series_id"]),
            "locked": s == "render_segment"
                      and not gates.render_allowed(con, ch["id"])[0],
        })
    return rows


def _series_rows(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = []
    for sid, title, source in con.execute(
            "SELECT id, title, source FROM series ORDER BY id"):
        chs = con.execute(
            "SELECT status, season FROM chapter WHERE series_id=?",
            (sid,)).fetchall()
        total = len(chs)
        done = sum(1 for s, _ in chs if _status_idx(s)
                   >= _status_idx("planned"))
        new = sum(1 for s, _ in chs if s == "discovered")
        seasons = sorted({sea for _, sea in chs if sea})
        cost = con.execute(
            "SELECT COALESCE(SUM(duration_sec),0) FROM stage_run sr JOIN "
            "chapter c ON c.id=sr.chapter_id WHERE c.series_id=?",
            (sid,)).fetchone()[0]
        remaining = max(0, total - done)
        rows.append({
            "id": sid, "title": title, "source": source, "total": total,
            "done": done, "new": new, "seasons": seasons,
            "pct": (100 * done // total) if total else 0,
            "eta": eta.fmt_eta(eta.series_eta(con, sid, remaining)),
            "wall_spent": eta.fmt_eta(cost),
        })
    return rows


def create_app(db_path: str = "studio.db") -> FastAPI:
    app = FastAPI(title="OriginPower Studio")

    # optional shared-secret gate for LAN/remote access: set
    # STUDIO_DASH_TOKEN on the host, then open /login?token=<value> once
    # per browser. Off when the env var is unset (localhost-only use).
    @app.middleware("http")
    async def _token_gate(request: Request, call_next):
        token = os.environ.get("STUDIO_DASH_TOKEN", "")
        if token:
            path = request.url.path
            if not (path.startswith("/static") or path.startswith("/login")):
                cookie = request.cookies.get("studio_token") or ""
                if not _secrets.compare_digest(cookie, token):
                    return PlainTextResponse(
                        "locked — open /login and enter the token",
                        status_code=401)
        return await call_next(request)

    @app.get("/login", response_class=HTMLResponse)
    def login_form():
        # token travels in a POST body, never in a URL (history/logs/referer)
        return HTMLResponse(
            '<form method="post" action="/login" '
            'style="margin:20vh auto;width:280px;font-family:sans-serif">'
            '<input type="password" name="token" placeholder="dashboard '
            'token" autofocus style="width:100%;padding:8px">'
            '<button style="margin-top:8px;width:100%;padding:8px">'
            'unlock</button></form>')

    @app.post("/login")
    def login(token: str = Form("")):
        resp = RedirectResponse("/", status_code=303)
        expected = os.environ.get("STUDIO_DASH_TOKEN", "")
        if token and expected and _secrets.compare_digest(token, expected):
            resp.set_cookie("studio_token", token, httponly=True,
                            samesite="strict",
                            max_age=60 * 60 * 24 * 90)
        return resp
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")),
              name="static")
    if (REPO / "ongoing").is_dir():
        app.mount("/media", StaticFiles(directory=str(REPO / "ongoing")),
                  name="media")

    def con() -> sqlite3.Connection:
        return connect(db_path)

    def page(name: str, request: Request, **ctx) -> HTMLResponse:
        ctx["fmt_eta"] = eta.fmt_eta
        return templates.TemplateResponse(request, name, ctx)

    # ---------------- pages ----------------

    @app.get("/", response_class=HTMLResponse)
    def queue_page(request: Request):
        c = con()
        return page("queue.html", request, jobs=jobs.queue_view(c),
                    series=_series_rows(c))

    @app.get("/partials/queue", response_class=HTMLResponse)
    def queue_partial(request: Request):
        return page("partials/queue_table.html", request,
                    jobs=jobs.queue_view(con()))

    @app.get("/partials/log/{job_id}", response_class=PlainTextResponse)
    def log_partial(job_id: int):
        c = con()
        r = c.execute("SELECT log_path FROM job WHERE id=?",
                      (job_id,)).fetchone()
        if not r or not r[0] or not os.path.exists(r[0]):
            return "(no log yet)"
        with open(r[0], "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            return f.read().decode("utf-8", "replace")

    @app.get("/series", response_class=HTMLResponse)
    def series_page(request: Request):
        return page("series.html", request, series=_series_rows(con()))

    @app.get("/series/{sid}", response_class=HTMLResponse)
    def series_detail(request: Request, sid: int):
        c = con()
        chs = [dict(zip(("id", "number", "label", "status", "season",
                         "ep_dir"), r))
               for r in c.execute(
                   "SELECT id, number, label, status, season, ep_dir FROM "
                   "chapter WHERE series_id=? ORDER BY number", (sid,))]
        title = (c.execute("SELECT title FROM series WHERE id=?",
                           (sid,)).fetchone() or ["?"])[0]
        return page("series_detail.html", request, sid=sid, title=title,
                    chapters=chs)

    @app.get("/chapter/{cid}", response_class=HTMLResponse)
    def chapter_page(request: Request, cid: int):
        c = con()
        r = c.execute("SELECT id, series_id, number, label, status, ep_dir "
                      "FROM chapter WHERE id=?", (cid,)).fetchone()
        if not r:
            return HTMLResponse("chapter not found", status_code=404)
        ch = dict(zip(("id", "series_id", "number", "label", "status",
                       "ep_dir"), r))
        title = (c.execute("SELECT title FROM series WHERE id=?",
                           (ch["series_id"],)).fetchone() or ["?"])[0]
        ep_rel = (Path(ch["ep_dir"]).resolve().relative_to(
            (REPO / "ongoing").resolve()) if ch["ep_dir"] else None)
        allowed, why = gates.render_allowed(c, cid)
        v_allowed, v_why = gates.voice_allowed(c, cid)
        has_preview = bool(ch["ep_dir"] and (
            Path(ch["ep_dir"]) / "render" / "voice_preview.mp3").exists())
        return page("chapter.html", request, ch=ch, series_title=title,
                    timeline=_stage_timeline(c, ch),
                    qa_ok=gates.latest_qa_ok(c, cid),
                    render_allowed=allowed, render_block_reason=why,
                    voice_allowed=v_allowed, voice_block_reason=v_why,
                    has_voice_preview=has_preview,
                    cost=_chapter_costs(ch["ep_dir"]),
                    gallery=_gallery(ch["ep_dir"]), ep_rel=ep_rel)

    @app.get("/videos", response_class=HTMLResponse)
    def videos_page(request: Request):
        c = con()
        rows = []
        for r in c.execute("SELECT id, series_id, title, kind, season_no, "
                           "state, output_path FROM bundle ORDER BY id"):
            b = dict(zip(("id", "series_id", "title", "kind", "season_no",
                          "state", "output_path"), r))

            def probe(cid: int) -> bool:
                row = c.execute("SELECT ep_dir FROM chapter WHERE id=?",
                                (cid,)).fetchone()
                if not row or not row[0]:
                    return False
                rd = Path(row[0]) / "render"
                return bool(list(rd.glob("*.mp4"))) if rd.is_dir() else False

            ready, total = bundles.segments_ready(c, b["id"], probe)

            def plan_dur(cid: int) -> Optional[float]:
                row = c.execute("SELECT ep_dir FROM chapter WHERE id=?",
                                (cid,)).fetchone()
                if not row or not row[0]:
                    return None
                p = Path(row[0]) / "render.plan.clean.json"
                try:
                    return float(json.loads(p.read_text())
                                 .get("total_duration_sec"))
                except Exception:
                    return None

            b.update(ready=ready, total=total,
                     runtime=eta.fmt_eta(bundles.projected_runtime_sec(
                         c, b["id"], plan_dur)),
                     approved=gates.concat_allowed(c, b["id"])[0])
            rows.append(b)
        series = [dict(zip(("id", "title"), r)) for r in
                  c.execute("SELECT id, title FROM series ORDER BY id")]
        return page("videos.html", request, bundles=rows, series=series)

    @app.get("/discovery", response_class=HTMLResponse)
    def discovery_page(request: Request, refresh: int = 0):
        c = con()
        if refresh:
            discovery.fetch_trending(c)
        return page("discovery.html", request, titles=discovery.listing(c))

    @app.get("/health", response_class=HTMLResponse)
    def health_page(request: Request):
        c = con()
        hb = c.execute("SELECT started_at FROM job WHERE type='heartbeat' "
                       "ORDER BY id DESC LIMIT 1").fetchone()
        ollama = ""
        try:
            import httpx
            tags = httpx.get("http://localhost:11434/api/tags",
                             timeout=2).json()
            ollama = ", ".join(m["name"] for m in tags.get("models", [])[:4])
        except Exception:
            ollama = "(not running)"
        checks = {
            "ollama models": ollama,
            "qwen venv": str((REPO / ".qwen_venv").is_dir()),
            "kokoro venv": str((REPO / ".kokoro_venv").is_dir()),
            "narrator ref": str((REPO / "assets/voice/narrator_ref.wav").exists()),
            "worker heartbeat": hb[0] if hb else "(no worker yet)",
            "disk free": f"{shutil.disk_usage(str(REPO)).free // 2**30} GB",
            "external spend": "$0/day (thumbnails owned; AniList read-only)",
        }
        return page("health.html", request, checks=checks)

    # ---------------- actions (insert-only) ----------------

    @app.post("/jobs")
    def post_job(type: str = Form(...), chapter_id: Optional[int] = Form(None),
                 series_id: Optional[int] = Form(None),
                 bundle_id: Optional[int] = Form(None),
                 target: str = Form(""), branding: str = Form("both")):
        c = con()
        if type == "prepare_series":
            # expand: one 'prepare' job per chapter that has no QA yet,
            # ordered by chapter number (the serial worker grinds the list)
            rows = c.execute(
                "SELECT id FROM chapter WHERE series_id=? AND id NOT IN "
                "(SELECT DISTINCT chapter_id FROM stage_run WHERE "
                "stage='qa_scan' AND ok=1) ORDER BY number", (series_id,)
            ).fetchall()
            for (cid,) in rows:
                jobs.enqueue(c, "prepare", chapter_id=cid,
                             series_id=series_id)
            return RedirectResponse("/", status_code=303)
        payload: Dict[str, Any] = {}
        if target:
            payload["target"] = target
        if type == "render_segment":
            payload["branding"] = branding
        jobs.enqueue(c, type, chapter_id=chapter_id, series_id=series_id,
                     bundle_id=bundle_id, payload=payload)
        return RedirectResponse("/", status_code=303)

    @app.post("/jobs/{job_id}/cancel")
    def post_cancel(job_id: int):
        jobs.cancel(con(), job_id)
        return RedirectResponse("/", status_code=303)

    @app.post("/jobs/{job_id}/up")
    def post_bump(job_id: int):
        jobs.bump(con(), job_id)
        return RedirectResponse("/", status_code=303)

    @app.post("/approve")
    def post_approve(gate: str = Form(...),
                     chapter_id: Optional[int] = Form(None),
                     bundle_id: Optional[int] = Form(None),
                     note: str = Form("")):
        c = con()
        gates.approve(c, gate, chapter_id=chapter_id,
                      bundle_id=bundle_id, note=note)
        # auto-advance: an approval IS the trigger for the next step
        # (the worker still re-checks every gate before doing anything)
        if gate == "voice" and chapter_id:
            jobs.enqueue(c, "voiceover", chapter_id=chapter_id)
        elif gate == "render" and chapter_id:
            jobs.enqueue(c, "render_segment", chapter_id=chapter_id,
                         payload={"branding": "both"})
        elif gate == "concat" and bundle_id:
            jobs.enqueue(c, "concat", bundle_id=bundle_id)
        back = f"/chapter/{chapter_id}" if chapter_id else "/videos"
        return RedirectResponse(back, status_code=303)

    @app.post("/bundles")
    def post_bundle(series_id: int = Form(...), kind: str = Form(...),
                    season_no: Optional[int] = Form(None),
                    title: str = Form("")):
        bundles.create_bundle(con(), series_id, kind, season_no=season_no,
                              title=title)
        return RedirectResponse("/videos", status_code=303)

    @app.post("/discovery/{anilist_id}/track")
    def post_track(anilist_id: int):
        discovery.mark(con(), anilist_id, "tracked")
        return RedirectResponse("/discovery", status_code=303)

    @app.post("/discovery/{anilist_id}/add")
    def post_discovery_add(anilist_id: int, source: str = Form(...),
                           url: str = Form(...)):
        c = con()
        discovery.mark(c, anilist_id, "in_production")
        jobs.enqueue(c, "add_series", payload={"source": source, "url": url})
        return RedirectResponse("/", status_code=303)

    return app
