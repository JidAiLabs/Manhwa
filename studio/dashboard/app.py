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
from fastapi.responses import (FileResponse, HTMLResponse, PlainTextResponse,
                               RedirectResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from studio.catalog.db import connect
from studio.catalog.models import STATUS_ORDER
from studio.dashboard import bundles, discovery, eta, gates, jobs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# manifest_freshness lives in tools/ — add to path once at import time
_TOOLS = str(REPO / "tools")
if _TOOLS not in __import__("sys").path:
    __import__("sys").path.insert(0, _TOOLS)
from manifest_freshness import verify_chapter as _verify_chapter_freshness  # noqa: E402

# stages shown on the chapter timeline, in pipeline order
TIMELINE = ["fetched", "stitched", "detected", "scened", "visioned",
            "grouped", "beated", "scripted", "voiced", "planned",
            "qa_scan", "render_segment"]


def _http_url(u: Optional[str]) -> Optional[str]:
    """Stored/scraped URLs are untrusted (sources are external sites) —
    only http(s) may ever reach an href; javascript:/data: etc. become None."""
    if isinstance(u, str) and u.lower().startswith(("http://", "https://")):
        return u
    return None


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
    for sid, title, source, surl, autopilot, new_pending in con.execute(
            "SELECT id, title, source, series_url, autopilot, "
            "COALESCE(new_pending, 0) FROM series ORDER BY id"):
        chs = con.execute(
            "SELECT status, season FROM chapter WHERE series_id=?",
            (sid,)).fetchall()
        total = len(chs)
        new = sum(1 for s, _ in chs if s == "discovered")
        seasons = sorted({sea for _, sea in chs if sea})
        cost = con.execute(
            "SELECT COALESCE(SUM(duration_sec),0) FROM stage_run sr JOIN "
            "chapter c ON c.id=sr.chapter_id WHERE c.series_id=?",
            (sid,)).fetchone()[0]

        # 3-segment readiness: prepared (QA-ok) >= voiced >= rendered. 'rendered'
        # is NOT in STATUS_ORDER, so count it directly; the stage_run table is the
        # source of truth for the earlier two.
        def _stage_ok(stage: str) -> int:
            return con.execute(
                "SELECT COUNT(DISTINCT chapter_id) FROM stage_run sr JOIN chapter "
                "c ON c.id=sr.chapter_id WHERE c.series_id=? AND sr.stage=? AND "
                "sr.ok=1", (sid, stage)).fetchone()[0]
        rendered = sum(1 for s, _ in chs if s == "rendered")
        voiced = max(_stage_ok("voiced"), rendered)
        prepared = max(_stage_ok("qa_scan"), voiced)
        remaining = max(0, total - rendered)
        # real measured averages behind the readiness estimate
        rp_prep, rp_voice, rp_render = eta.readiness_parts(con, sid)
        rows.append({
            "id": sid, "title": title, "source": source,
            "url": _http_url(surl), "autopilot": bool(autopilot),
            "total": total, "new": new, "seasons": seasons,
            "new_pending": int(new_pending or 0),
            "prepared": prepared, "voiced": voiced, "rendered": rendered,
            "done": rendered,
            "prepared_pct": (100 * prepared // total) if total else 0,
            "voiced_pct": (100 * voiced // total) if total else 0,
            "rendered_pct": (100 * rendered // total) if total else 0,
            "avg_prep": eta.fmt_eta(rp_prep),
            "avg_voice": eta.fmt_eta(rp_voice),
            "avg_render": eta.fmt_eta(rp_render),
            "eta": eta.fmt_eta(eta.series_eta(con, sid, remaining)),
            "wall_spent": eta.fmt_eta(cost),
        })
    return rows


def _series_delete_targets(slug: str, sid: int) -> List[Path]:
    """The on-disk roots created for a series. PATH-SAFE: the ongoing folder is
    only included when it resolves to a real child of REPO/ongoing (never the
    ongoing root itself, never an escaping slug like '..')."""
    targets: List[Path] = []
    ongoing_root = (REPO / "ongoing").resolve()
    if slug:
        d = (REPO / "ongoing" / slug).resolve()
        if d != ongoing_root and str(d).startswith(str(ongoing_root) + os.sep):
            targets.append(d)
    targets.append((REPO / "assets" / "branding" / "series" / str(int(sid))))
    targets.append((REPO / "dist" / f"series_{int(sid)}"))
    return targets


def _delete_series(con: sqlite3.Connection, sid: int) -> Dict[str, Any]:
    """Delete EVERY file + db row created for a series. Irreversible — callers
    must gate this behind the typed-name confirmation."""
    row = con.execute("SELECT slug, title FROM series WHERE id=?",
                      (sid,)).fetchone()
    if not row:
        return {"ok": False, "deleted_files": []}
    slug, title = row
    deleted_files = []
    for d in _series_delete_targets(slug, sid):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            deleted_files.append(str(d))
    chap_ids = [r[0] for r in
                con.execute("SELECT id FROM chapter WHERE series_id=?", (sid,))]
    if chap_ids:
        qs = ",".join("?" for _ in chap_ids)
        for tbl in ("stage_run", "approval", "bundle_chapter", "job"):
            con.execute(f"DELETE FROM {tbl} WHERE chapter_id IN ({qs})",
                        chap_ids)
    con.execute("DELETE FROM job WHERE series_id=?", (sid,))
    con.execute("DELETE FROM approval WHERE series_id=?", (sid,))
    con.execute("DELETE FROM bundle WHERE series_id=?", (sid,))
    con.execute("DELETE FROM chapter WHERE series_id=?", (sid,))
    con.execute("DELETE FROM series WHERE id=?", (sid,))
    con.commit()
    return {"ok": True, "title": title, "deleted_files": deleted_files,
            "deleted_chapters": len(chap_ids)}


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
                         "ep_dir", "url"), r))
               for r in c.execute(
                   "SELECT id, number, label, status, season, ep_dir, url "
                   "FROM chapter WHERE series_id=? ORDER BY number", (sid,))]
        for ch_row in chs:
            ch_row["url"] = _http_url(ch_row["url"])
        title, series_url, autopilot, style = (c.execute(
            "SELECT title, series_url, autopilot, narration_style "
            "FROM series WHERE id=?", (sid,)).fetchone() or ("?", "", 0, None))
        thumb = REPO / "dist" / f"series_{sid}" / "thumbnail_yt.jpg"
        thumb_exists = thumb.exists()
        thumb_ready = any(
            ch["ep_dir"] and (Path(ch["ep_dir"]) / "manifest.beats.json").exists()
            for ch in chs)
        return page("series_detail.html", request, sid=sid, title=title,
                    series_url=_http_url(series_url), chapters=chs,
                    autopilot=bool(autopilot),
                    narration_style=style or "default",
                    thumb_exists=thumb_exists, thumb_ready=thumb_ready,
                    thumb_v=int(thumb.stat().st_mtime) if thumb_exists else 0,
                    thumb_approved=gates.thumbnail_approved(c, sid))

    @app.get("/thumb/series/{sid}")
    def series_thumb(sid: int):
        p = REPO / "dist" / f"series_{sid}" / "thumbnail_yt.jpg"
        if not p.exists():
            return PlainTextResponse("no thumbnail yet", status_code=404)
        return FileResponse(str(p), media_type="image/jpeg")

    @app.get("/chapter/{cid}", response_class=HTMLResponse)
    def chapter_page(request: Request, cid: int):
        c = con()
        r = c.execute("SELECT id, series_id, number, label, status, ep_dir, "
                      "url FROM chapter WHERE id=?", (cid,)).fetchone()
        if not r:
            return HTMLResponse("chapter not found", status_code=404)
        ch = dict(zip(("id", "series_id", "number", "label", "status",
                       "ep_dir", "url"), r))
        ch["url"] = _http_url(ch["url"])
        title = (c.execute("SELECT title FROM series WHERE id=?",
                           (ch["series_id"],)).fetchone() or ["?"])[0]
        ep_rel = (Path(ch["ep_dir"]).resolve().relative_to(
            (REPO / "ongoing").resolve()) if ch["ep_dir"] else None)
        allowed, why = gates.render_allowed(c, cid)
        v_allowed, v_why = gates.voice_allowed(c, cid)
        has_preview = bool(ch["ep_dir"] and (
            Path(ch["ep_dir"]) / "render" / "voice_preview.mp3").exists())
        seg = (Path(ch["ep_dir"]) / "render" / "segment_both.mp4"
               if ch["ep_dir"] else None)
        render_url = (f"/media/{ep_rel}/render/segment_both.mp4"
                      f"?v={int(seg.stat().st_mtime)}"
                      if seg and seg.exists() and ep_rel is not None else None)
        qa_html = Path(ch["ep_dir"] or "") / "prep_qa.html"
        qa_v = int(qa_html.stat().st_mtime) if (
            ch["ep_dir"] and qa_html.exists()) else 0
        # manifest freshness — detect stale/missing plan files before render
        _freshness = (_verify_chapter_freshness(ch["ep_dir"])
                      if ch["ep_dir"] else [])
        plan_stale_issues = [i for i in _freshness
                             if i["code"] in ("stale_manifest", "missing_manifest")
                             and i["file"] in
                             ("render.plan.clean.json", "render.plan.json",
                              "manifest.beats.json", "manifest.script.json")]
        plan_stale = bool(plan_stale_issues)
        plan_stale_detail = [i["detail"] for i in plan_stale_issues]
        video_stale_issues = [i for i in _freshness if i["code"] == "stale_video"]
        video_stale = bool(video_stale_issues)
        video_stale_detail = next(
            (i["detail"] for i in video_stale_issues), "")
        return page("chapter.html", request, ch=ch, series_title=title,
                    timeline=_stage_timeline(c, ch),
                    qa_ok=gates.latest_qa_ok(c, cid),
                    render_allowed=allowed, render_block_reason=why,
                    voice_allowed=v_allowed, voice_block_reason=v_why,
                    has_voice_preview=has_preview, qa_v=qa_v,
                    render_url=render_url,
                    cost=_chapter_costs(ch["ep_dir"]),
                    gallery=_gallery(ch["ep_dir"]), ep_rel=ep_rel,
                    plan_stale=plan_stale,
                    plan_stale_detail=plan_stale_detail,
                    video_stale=video_stale,
                    video_stale_detail=video_stale_detail)

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

    @app.get("/partials/duration-estimate", response_class=HTMLResponse)
    def duration_estimate(series_id: int, num_from: float = 0.0,
                          num_to: float = 1e12, target: str = "qa"):
        """Two estimates for a selected range: ~processing time to build it, and
        the ~length of the final video. Rough (seed/median based)."""
        from studio.dashboard import eta as _eta
        seed = getattr(_eta, "SEED_SEC", {})
        c = con()
        rows = c.execute(
            "SELECT ep_dir FROM chapter WHERE series_id=? AND number BETWEEN ? "
            "AND ? ORDER BY number", (series_id, num_from, num_to)).fetchall()
        n_total = len(rows)
        # chapters NOT yet done at the target — what a bulk run ACTUALLY builds
        # (resume semantics; matches the prepare_range filter).
        done_stage = {"qa": "qa_scan", "voice": "voiced",
                      "video": "render_segment"}.get(target or "qa")
        n_missing = c.execute(
            "SELECT COUNT(*) FROM chapter WHERE series_id=? AND number BETWEEN ? "
            "AND ? AND id NOT IN (SELECT chapter_id FROM stage_run WHERE "
            "stage=? AND ok=1)",
            (series_id, num_from, num_to, done_stage)).fetchone()[0]
        vid = 0.0
        have = 0
        for (ep_dir,) in rows:
            p = Path(ep_dir or "") / "render.plan.clean.json"
            if p.exists():
                try:
                    vid += float(json.loads(p.read_text()).get(
                        "total_duration_sec") or 0)
                    have += 1
                except Exception:
                    pass
        if have and n_total > have:                 # extrapolate the un-built ones
            vid += (vid / have) * (n_total - have)
        elif not have:
            vid = n_total * float(seed.get("voiced", 600))
        # wall-clock build time = the SLOWEST worker lane × the chapters that
        # actually run (missing-only). The lanes (gpu/tts/cpu) pipeline, so it's
        # bounded by the busiest lane, not the serial sum of every stage.
        proc = _eta.lane_bottleneck_sec(c, series_id, target) * n_missing

        def fmt(s: float) -> str:
            s = int(s)
            h, m = divmod(s // 60, 60)
            return f"{h}h {m}m" if h else f"{m}m"

        return HTMLResponse(
            f'<span class="kv">{n_missing} of {n_total} chapters to build · '
            f'~<b>{fmt(proc)}</b> · ~<b>{fmt(vid)}</b> final video</span>')

    @app.post("/jobs")
    def post_job(type: str = Form(...), chapter_id: Optional[int] = Form(None),
                 series_id: Optional[int] = Form(None),
                 bundle_id: Optional[int] = Form(None),
                 target: str = Form(""), branding: str = Form("both"),
                 num_from: float = Form(0.0), num_to: float = Form(1e12)):
        c = con()
        if type == "prepare_range":
            # bulk: run chapters in [num_from..num_to] up to a target stage —
            # qa (prepare only) | voice (prepare→voiceover) | video (→render).
            # auto_to is carried on each prepare job; the worker advances past the
            # approval gates only as far as the target (QA must stay green).
            auto_to = {"voice": "voice", "video": "video"}.get(target or "qa")
            # RESUME semantics: only enqueue chapters NOT already done at the
            # target stage (and not already queued/running), so a 1..N bulk run
            # picks up "whatever is missing" instead of redoing finished chapters.
            done_stage = {"qa": "qa_scan", "voice": "voiced",
                          "video": "render_segment"}.get(target or "qa")
            rows = c.execute(
                "SELECT id FROM chapter WHERE series_id=? AND number BETWEEN ? "
                "AND ? AND id NOT IN (SELECT chapter_id FROM stage_run WHERE "
                "stage=? AND ok=1) AND id NOT IN (SELECT chapter_id FROM job "
                "WHERE type='prepare' AND state IN ('queued','running') AND "
                "chapter_id IS NOT NULL) ORDER BY number",
                (series_id, num_from, num_to, done_stage)).fetchall()
            for (cid,) in rows:
                jobs.enqueue(c, "prepare", chapter_id=cid, series_id=series_id,
                             payload={"auto_to": auto_to} if auto_to else {})
            c.execute("UPDATE series SET new_pending=0 WHERE id=?", (series_id,))
            c.commit()
            return RedirectResponse(f"/series/{series_id}", status_code=303)
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
            c.execute("UPDATE series SET new_pending=0 WHERE id=?", (series_id,))
            c.commit()
            return RedirectResponse("/", status_code=303)
        payload: Dict[str, Any] = {}
        if target:
            payload["target"] = target
        if type == "render_segment":
            payload["branding"] = branding
        jobs.enqueue(c, type, chapter_id=chapter_id, series_id=series_id,
                     bundle_id=bundle_id, payload=payload)
        return RedirectResponse("/", status_code=303)

    @app.post("/add-series-direct")
    def add_series_direct(source: str = Form(...), url: str = Form(...)):
        """Manually add a manhwa by source + URL (e.g. asura, webtoon, elftoon)
        — discovery runs in the background via an add_series job."""
        jobs.enqueue(con(), "add_series",
                     payload={"source": source.strip(), "url": url.strip()})
        return RedirectResponse("/series", status_code=303)

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
                     series_id: Optional[int] = Form(None),
                     note: str = Form("")):
        c = con()
        gates.approve(c, gate, series_id=series_id, chapter_id=chapter_id,
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
        # thumbnail approval is a record only (uploads are manual) — no job
        if gate == "thumbnail" and series_id:
            back = f"/series/{series_id}"
        else:
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

    @app.post("/chapter/{cid}/drop")
    def post_drop(cid: int, file: str = Form(...)):
        # the operator's button: ban this panel from the chapter and
        # re-prepare automatically — see a bad visual, click, done
        c = con()
        r = c.execute("SELECT ep_dir FROM chapter WHERE id=?",
                      (cid,)).fetchone()
        if not (r and r[0]):
            return PlainTextResponse("chapter has no episode dir",
                                     status_code=400)
        safe = os.path.basename(file)
        mdp = Path(r[0]) / "manual_drops.json"
        try:
            drops = json.loads(mdp.read_text()) if mdp.exists() else []
        except Exception:
            drops = []
        if safe not in drops:
            drops.append(safe)
            mdp.write_text(json.dumps(drops, indent=1))
        jobs.enqueue(c, "prepare", chapter_id=cid, priority=30)
        return RedirectResponse(f"/chapter/{cid}", status_code=303)

    @app.post("/chapter/{cid}/rebuild")
    def post_rebuild(cid: int):
        # force the chapter back through scene materialization so shipped
        # stage fixes actually apply (resume-by-status never re-runs them)
        c = con()
        c.execute("UPDATE chapter SET status='detected' WHERE id=?", (cid,))
        c.commit()
        jobs.enqueue(c, "prepare", chapter_id=cid, priority=50)
        return RedirectResponse(f"/chapter/{cid}", status_code=303)

    @app.post("/chapter/{cid}/revoice")
    def post_revoice(cid: int):
        # Re-synthesize the voiceover with the latest TTS (seeded + de-robotted)
        # and re-render — KEEPS the approved narration, only the audio + video
        # are rebuilt. Clearing the cached clips forces a full re-synth (the
        # text_sha cache would otherwise reuse them); rewinding to 'scripted'
        # makes the resume-by-status runner actually re-run the voiced stage.
        c = con()
        r = c.execute("SELECT ep_dir FROM chapter WHERE id=?", (cid,)).fetchone()
        if r and r[0]:
            tts_dir = Path(r[0]) / "tts"
            clips = tts_dir / "clips"
            if clips.exists():
                for w in clips.glob("*.wav"):
                    try:
                        w.unlink()
                    except OSError:
                        pass
            idx = tts_dir / "tts_index.json"
            if idx.exists():
                try:
                    idx.unlink()
                except OSError:
                    pass
        # narration stays approved; clear the render gate so auto_to=video
        # re-approves + re-renders (it no-ops when render is already approved)
        if not gates._has_approval(c, "voice", chapter_id=cid):
            gates.approve(c, "voice", chapter_id=cid, note="re-voice")
        c.execute("DELETE FROM approval WHERE gate='render' AND chapter_id=?",
                  (cid,))
        c.execute("UPDATE chapter SET status='scripted' WHERE id=?", (cid,))
        c.commit()
        jobs.enqueue(c, "voiceover", chapter_id=cid,
                     payload={"auto_to": "video"}, priority=50)
        return RedirectResponse(f"/chapter/{cid}", status_code=303)

    @app.post("/series/{sid}/style")
    def post_style(sid: int, style: str = Form(...)):
        if style not in ("default", "off", "light", "full"):
            return PlainTextResponse("invalid style", status_code=400)
        c = con()
        c.execute("UPDATE series SET narration_style=? WHERE id=?",
                  (None if style == "default" else style, sid))
        c.commit()
        return RedirectResponse(f"/series/{sid}", status_code=303)

    @app.post("/series/{sid}/seen")
    def post_seen(sid: int):
        # dismiss the new-chapter red alert without running the series
        c = con()
        c.execute("UPDATE series SET new_pending=0 WHERE id=?", (sid,))
        c.commit()
        return RedirectResponse("/series", status_code=303)

    def _series_running_jobs(c: sqlite3.Connection, sid: int) -> int:
        return c.execute(
            "SELECT COUNT(*) FROM job WHERE state='running' AND type!='heartbeat'"
            " AND (series_id=? OR chapter_id IN (SELECT id FROM chapter WHERE "
            "series_id=?))", (sid, sid)).fetchone()[0]

    @app.get("/series/{sid}/delete", response_class=HTMLResponse)
    def series_delete_confirm(request: Request, sid: int, error: str = ""):
        c = con()
        row = c.execute("SELECT slug, title FROM series WHERE id=?",
                        (sid,)).fetchone()
        if not row:
            return HTMLResponse("series not found", status_code=404)
        slug, title = row
        n_chapters = c.execute("SELECT COUNT(*) FROM chapter WHERE series_id=?",
                               (sid,)).fetchone()[0]
        size = ""
        d = (REPO / "ongoing" / slug) if slug else None
        if d and d.exists():
            try:
                import subprocess as _sp
                out = _sp.run(["du", "-sh", str(d)], capture_output=True,
                              text=True, timeout=20).stdout
                size = out.split("\t")[0].strip() if out else ""
            except Exception:
                size = ""
        return page("delete_series.html", request, sid=sid, title=title,
                    slug=slug, n_chapters=n_chapters, size=size,
                    running=bool(_series_running_jobs(c, sid)), error=error)

    @app.post("/series/{sid}/delete")
    def series_delete(sid: int, confirm: str = Form("")):
        c = con()
        row = c.execute("SELECT title FROM series WHERE id=?", (sid,)).fetchone()
        if not row:
            return RedirectResponse("/series", status_code=303)
        title = row[0]
        # guard 1 — the typed name must match EXACTLY (blocks accidental clicks)
        if (confirm or "").strip() != (title or "").strip():
            return RedirectResponse(
                f"/series/{sid}/delete?error=name+did+not+match,+nothing+deleted",
                status_code=303)
        # guard 2 — never delete files out from under a running job
        if _series_running_jobs(c, sid):
            return RedirectResponse(
                f"/series/{sid}/delete?error=a+job+is+running+for+this+series,+"
                "cancel+it+first", status_code=303)
        _delete_series(c, sid)
        return RedirectResponse("/series", status_code=303)

    @app.post("/series/{sid}/autopilot")
    def post_autopilot(sid: int):
        c = con()
        c.execute("UPDATE series SET autopilot = 1 - autopilot WHERE id=?",
                  (sid,))
        c.commit()
        return RedirectResponse(f"/series/{sid}", status_code=303)

    @app.post("/discovery/{anilist_id}/add")
    def post_discovery_add(anilist_id: int, source: str = Form(...),
                           url: str = Form(...)):
        if not _http_url(url):
            return PlainTextResponse("invalid url scheme", status_code=400)
        c = con()
        discovery.mark(c, anilist_id, "in_production")
        jobs.enqueue(c, "add_series", payload={"source": source, "url": url})
        return RedirectResponse("/", status_code=303)

    return app
