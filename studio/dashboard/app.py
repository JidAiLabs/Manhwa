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
from html import escape
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote_plus, urlparse

from fastapi import FastAPI, Form, Request
from fastapi.responses import (FileResponse, HTMLResponse, PlainTextResponse,
                               RedirectResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from studio import paths
from studio.catalog import reconcile, reset
from studio.catalog.db import connect
from studio.catalog.models import STATUS_ORDER, STATUS_RANK
from studio.dashboard import bundles, discovery, eta, gates, jobs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# the venv interpreter, NOT sys.executable: the dashboard may be launched by a
# system python (launchd, a bare uvicorn) that lacks the pipeline's deps, and
# the subprocess would then fail on import with a confusing traceback.
PY = str(REPO / ".eval_venv" / "bin" / "python")

# manifest_freshness / beats_segments live in tools/ — add to path once at import time
_TOOLS = str(REPO / "tools")
if _TOOLS not in __import__("sys").path:
    __import__("sys").path.insert(0, _TOOLS)
from beats_segments import beat_segments  # noqa: E402
from manifest_freshness import verify_chapter as _verify_chapter_freshness  # noqa: E402

# Stages shown on the chapter timeline, in pipeline order. EVERY entry must be
# either a STATUS_ORDER status (satisfied from chapter.status) or a stage name
# something actually writes to stage_run — an entry that is neither renders
# permanently red on every chapter. 'fetched' was exactly that: nothing has
# ever written it (the status is 'downloaded'), so step 1 was red forever.
# 'prepped' is the reverse case — written by _run_prep_and_qa but never shown.
TIMELINE = ["downloaded", "stitched", "detected", "scened", "visioned",
            "grouped", "beated", "scripted", "voiced", "planned",
            "prepped", "qa_scan", "render_segment"]


def _served_rel(p: Optional[Union[str, Path]]) -> Optional[str]:
    """A path's position under ongoing/, for building a /media URL — or None
    when it does not live there.

    Every caller used a bare `Path(...).relative_to(ongoing)`, which RAISES
    on a path outside the tree and took the whole chapter page down with a
    500. That is not a rare case: it is what every DB copied between the Air
    and the Mini looks like, since ep_dir is stored as an absolute path from
    whichever machine did the fetch. The page must degrade to "images
    unavailable", never to a stack trace."""
    if not p:
        return None
    try:
        return str(Path(p).resolve().relative_to((REPO / "ongoing").resolve()))
    except Exception:
        return None


def _safe_next(nxt: Optional[str]) -> str:
    """A caller-supplied post-login destination, reduced to a same-site path.

    An unvalidated `next` is a textbook open redirect: an attacker sends the
    operator a /login?next=<their site> link, the operator authenticates as
    normal, and the dashboard itself bounces them somewhere hostile with the
    credibility of a trusted origin.

    Rejecting only a leading '//' is NOT enough. Browsers normalise
    backslashes to forward slashes in the authority position, so '/\\evil'
    and '/\\/evil' are treated as protocol-relative URLs and escape a naive
    prefix check. Anything with a scheme, an authority, a backslash, or a
    control character is discarded outright and the user lands on '/'.
    """
    s = (nxt or "").strip()
    if not s.startswith("/") or s.startswith("//"):
        return "/"
    if "\\" in s or any(ord(c) < 0x20 or ord(c) == 0x7f for c in s):
        return "/"
    parsed = urlparse(s)
    if parsed.scheme or parsed.netloc:
        return "/"
    return s


def _http_url(u: Optional[str]) -> Optional[str]:
    """Stored/scraped URLs are untrusted (sources are external sites) —
    only http(s) may ever reach an href; javascript:/data: etc. become None."""
    if isinstance(u, str) and u.lower().startswith(("http://", "https://")):
        return u
    return None


def _status_idx(status: str) -> int:
    """Rank a chapter status for progress display. -1, NOT 0, for an unknown
    status: 0 is 'discovered' — a real rung — so a failure status like
    'voiced_failed' used to read as legitimate early progress. Ranks against
    STATUS_RANK so a 'rendered' chapter shows every pill done instead of
    none."""
    try:
        return STATUS_RANK.index(status)
    except ValueError:
        return -1


def _chapter_plan_dur(ep_dir: Optional[str]) -> Optional[float]:
    """A prepared chapter's final on-screen duration, from its render plan
    (seconds), or None if it hasn't been prepared yet."""
    if not ep_dir:
        return None
    p = Path(ep_dir) / "render.plan.clean.json"
    try:
        return float(json.loads(p.read_text()).get("total_duration_sec") or 0)
    except Exception:
        return None


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
                    if not f:
                        continue
                    # ken_variety sub-cuts = ONE panel, several camera moves;
                    # N identical thumbnails here read as the duplicate-panel
                    # bug (2026-07-16 owner report). Collapse consecutive
                    # repeats into one thumb with an xN motion badge.
                    if files and files[-1]["name"] == str(f):
                        files[-1]["n"] += 1
                    else:
                        files.append({"name": str(f), "n": 1})
            # honour the plan's OWN scenes_subdir, which is what prep_qa and
            # Remotion both read (prep_qa.py:2639). Hardcoding "scenes_clean"
            # here meant that any chapter whose plan pointed elsewhere showed
            # a gallery of broken images on the one page you use to review it.
            out.append({"segment_id": item.get("segment_id"),
                        "narration": item.get("tts_text") or "",
                        "files": files,
                        "src_dir": plan.get("scenes_subdir") or "scenes_clean",
                        "duration": item.get("duration_sec") or 0})
        return out

    b = Path(ep_dir) / "manifest.beats.json"
    if not b.exists():
        return []
    try:
        beats = json.loads(b.read_text()).get("beats") or []
    except Exception:
        return []
    # one row per narration SEGMENT with its span's panels grouped under the
    # line that voices them (a flow span = several panels, one clip; legacy
    # panel_narration = singleton spans). Beats with neither shape (old
    # per-beat manifests) keep the old one-row-per-beat fallback.
    out = []
    for bt in beats:
        gid = int(bt.get("group_id") or 0)
        segs = beat_segments(bt)
        if not segs:
            out.append({"segment_id": f"g{gid:04d}",
                        "narration": bt.get("narration") or "",
                        "files": [{"name": str(f), "n": 1}
                                  for f in (bt.get("scene_files") or [])[:4]],
                        "src_dir": "scenes", "duration": 0})
            continue
        for i, seg in enumerate(segs):
            out.append({"segment_id": f"g{gid:04d} · {i + 1}/{len(segs)}",
                        "narration": seg["line"],
                        "files": [{"name": str(f), "n": 1}
                                  for f in seg["span"]],
                        "src_dir": "scenes", "duration": 0})
    return out


def _teaser_newer_than(b: Dict[str, Any]) -> bool:
    """True when the manhwa's teaser is NEWER than this video's concatenated
    file — i.e. the video still carries an older cold open and needs a
    re-concat. Re-voicing bundle 1's teaser produced exactly this state, and
    with nothing surfacing it the Videos page looked unchanged."""
    out = b.get("output_path")
    t = REPO / "dist" / f"series_{b['series_id']}" / "teaser.mp4"
    if not out or not t.exists() or not Path(out).exists():
        return False
    return t.stat().st_mtime > Path(out).stat().st_mtime


def _teaser_card(sid: int) -> Optional[Dict[str, Any]]:
    """Review-card data for a PLANNED teaser: the model's reason/spoiler note +
    per-panel narration + thumbnails. The teaser keeps each scene as a symlink
    into the source chapter under ongoing/, so we resolve the link back and serve
    it via the existing /media mount. Returns None when no teaser was planned."""
    tdir = REPO / "dist" / f"series_{sid}" / "teaser"
    mf = tdir / "manifest.teaser.json"
    if not mf.exists():
        return None
    try:
        t = json.loads(mf.read_text())
    except Exception:
        return None
    lines: Dict[str, str] = {}
    for seg in beat_segments(t):        # legacy teaser shape adapts to segments
        for sf in seg["span"]:
            lines.setdefault(sf, seg["line"])
    panels = []
    for sf in (t.get("scene_files") or []):
        rel = _served_rel(tdir / "scenes" / sf)
        panels.append({"file": sf, "line": lines.get(sf, ""),
                       "img": f"/media/{rel}" if rel else None})
    return {"reason": t.get("reason") or "",
            "rewind_line": t.get("rewind_line") or "",
            "spoiler_boundary": t.get("spoiler_boundary") or "",
            "source_chapters": t.get("source_chapters") or [],
            "panels": panels}


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
                      and not gates.render_allowed(con, ch["id"], ch["ep_dir"])[0],
        })
    return rows


# reconcile WRITES (it repairs drifted statuses and prunes dead stage_run
# rows), and it ran on every single page load of a page listing every series —
# so idle browser tabs generated a steady stream of write transactions
# competing with the worker for the database. Throttled per series: the
# artifacts on disk do not change faster than the worker can change them.
_RECONCILE_THROTTLE_SEC = int(os.environ.get("STUDIO_RECONCILE_SEC", "60"))
_reconciled_at: Dict[int, float] = {}


def _reconcile_throttled(con: sqlite3.Connection, sid: int) -> None:
    now = time.time()
    if now - _reconciled_at.get(sid, 0.0) < _RECONCILE_THROTTLE_SEC:
        return
    _reconciled_at[sid] = now
    reconcile.reconcile_series(con, sid)


def _series_rows(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = []
    for sid, title, source, surl, autopilot, new_pending in con.execute(
            "SELECT id, title, source, series_url, autopilot, "
            "COALESCE(new_pending, 0) FROM series ORDER BY id"):
        # reconcile-on-load: prune dead stage_run rows + repair drifted statuses
        # so the readiness counts below reflect the artifacts on disk, not stale
        # rows (the "prep 2 · voice 1" lie). Skips chapters the worker is running.
        _reconcile_throttled(con, sid)
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
                    # THE FROZEN-DASHBOARD ROOT CAUSE. htmx does not swap a
                    # non-2xx response, so a bare 401 made the 2s queue poller
                    # fail silently: the page kept displaying whatever it last
                    # got and looked alive while showing state minutes or
                    # hours old. This happens routinely across machines —
                    # LAN, Tailscale and WireGuard are three different origins
                    # and therefore three separate cookie jars, so walking the
                    # Air from one network to another logs you out mid-view.
                    # Tell htmx to navigate instead of silently doing nothing.
                    if request.headers.get("HX-Request"):
                        return PlainTextResponse(
                            "", status_code=401,
                            headers={"HX-Redirect": "/login"})
                    if request.method == "GET":
                        nxt = quote_plus(str(request.url.path))
                        return RedirectResponse(f"/login?next={nxt}",
                                                status_code=303)
                    return PlainTextResponse(
                        "locked — open /login and enter the token",
                        status_code=401)
        return await call_next(request)

    @app.get("/login", response_class=HTMLResponse)
    def login_form(next: str = "/"):
        # token travels in a POST body, never in a URL (history/logs/referer)
        return HTMLResponse(
            '<form method="post" action="/login" '
            'style="margin:20vh auto;width:280px;font-family:sans-serif">'
            f'<input type="hidden" name="next" value="{escape(next)}">'
            '<input type="password" name="token" placeholder="dashboard '
            'token" autofocus style="width:100%;padding:8px">'
            '<button style="margin-top:8px;width:100%;padding:8px">'
            'unlock</button></form>')

    @app.post("/login")
    def login(token: str = Form(""), next: str = Form("/")):
        resp = RedirectResponse(_safe_next(next), status_code=303)
        expected = os.environ.get("STUDIO_DASH_TOKEN", "")
        if token and expected and _secrets.compare_digest(token, expected):
            resp.set_cookie("studio_token", token, httponly=True,
                            samesite="strict",
                            max_age=60 * 60 * 24 * 90)
        return resp
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")),
              name="static")
    # UNCONDITIONAL mounts. /media used to be mounted only if ongoing/ already
    # existed at startup — but the templates emit /media/... regardless, so on
    # a fresh checkout (or a host where the data lives elsewhere and the dir is
    # created later) every scene image and every video 404'd with no
    # explanation. mkdir first, then mount, so the route always exists.
    # /dist is the same story for finished bundle videos, which the Videos page
    # previously showed as an unclickable host path.
    for _url, _dir in (("/media", REPO / "ongoing"), ("/dist", REPO / "dist")):
        try:
            _dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue        # read-only checkout: skip rather than fail boot
        app.mount(_url, StaticFiles(directory=str(_dir)),
                  name=_url.lstrip("/"))

    def con() -> sqlite3.Connection:
        return connect(db_path)

    def rcon(series_id: Optional[int] = None) -> sqlite3.Connection:
        """A connection for a READ page, reconciled first.

        /series/{sid}, /videos and /runs never reconciled, while /series and
        /chapter/{id} always did — so the same chapter could show 'voiced' on
        one page and 'scripted' on another depending only on which you opened
        first. Reconcile repairs the DB against the artifacts actually on
        disk; doing it on one page and not its neighbour is what made the UI
        contradict itself. POST handlers deliberately keep plain con(): an
        action must not silently rewrite state on its way to doing something
        else.
        """
        c = connect(db_path)
        try:
            if series_id is None:
                for (sid,) in c.execute("SELECT id FROM series"):
                    _reconcile_throttled(c, sid)
            else:
                _reconcile_throttled(c, series_id)
        except Exception:
            pass        # a reconcile failure must never take a page down
        return c

    def page(name: str, request: Request, **ctx) -> HTMLResponse:
        ctx["fmt_eta"] = eta.fmt_eta
        return templates.TemplateResponse(request, name, ctx)

    # ---------------- pages ----------------

    @app.get("/", response_class=HTMLResponse)
    def queue_page(request: Request):
        c = con()
        return page("queue.html", request, series=_series_rows(c),
                    **_queue_ctx(c))

    def _queue_ctx(c: sqlite3.Connection) -> Dict[str, Any]:
        """Queue rows ANNOTATED with health, so a stuck job looks stuck.
        Without this the UI cannot distinguish 'running for 40 minutes
        because that is how long it takes' from 'running for 40 minutes
        because its process died 38 minutes ago'."""
        h = jobs.health(c)
        stale = {s["id"]: s for s in h["stale"]}
        overdue = {o["id"] for o in h["overdue"]}
        rows = jobs.queue_view(c)
        for j in rows:
            j["is_stale"] = j["id"] in stale
            j["is_overdue"] = j["id"] in overdue and j["id"] not in stale
        return {"jobs": rows, "health": h}

    @app.get("/partials/queue", response_class=HTMLResponse)
    def queue_partial(request: Request):
        return page("partials/queue_table.html", request, **_queue_ctx(con()))

    @app.get("/partials/health", response_class=HTMLResponse)
    def health_strip(request: Request):
        return page("partials/health_strip.html", request,
                    health=jobs.health(con()))

    def _log_tail(c: sqlite3.Connection, job_id: int) -> str:
        r = c.execute("SELECT log_path FROM job WHERE id=?",
                      (job_id,)).fetchone()
        if not r or not r[0]:
            return "(no log yet)"
        # rows written before log paths were absolutized are relative to the
        # WORKER's cwd (the repo root), not ours
        path = r[0]
        if not os.path.exists(path) and not os.path.isabs(path):
            path = str(REPO / path)
        if not os.path.exists(path):
            return "(no log yet)"
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            return f.read().decode("utf-8", "replace")

    @app.get("/partials/log/active", response_class=PlainTextResponse)
    def log_active():
        """The log of whatever is running right now. The pane used to say
        '(no job selected)' until you clicked something — so the default view
        of a busy machine showed no evidence it was doing anything."""
        c = con()
        r = c.execute(
            "SELECT id FROM job WHERE type!='heartbeat' AND state IN "
            "('running','cancelling') ORDER BY started_at DESC, id DESC "
            "LIMIT 1").fetchone()
        if not r:
            return "(nothing running)"
        return f"── job #{r[0]} ──\n" + _log_tail(c, r[0])

    @app.get("/partials/log/{job_id}", response_class=PlainTextResponse)
    def log_partial(job_id: int):
        return _log_tail(con(), job_id)

    @app.get("/runs", response_class=HTMLResponse)
    def runs_page(request: Request):
        return page("runs.html", request, jobs=jobs.runs_view(rcon()))

    def _source_ids() -> list:
        """Registered adapter ids, read from the REGISTRY so the picker can
        never drift from what the code can actually fetch.

        arenascan shipped registered-but-UNPICKABLE: the dropdown was a
        hardcoded 3-item list, so adding an arenascan URL silently fell back to
        the first option (asura), whose parser matches "/chapter/" and found 0
        of 168 chapters. The series looked added and was simply empty."""
        import studio.sources          # noqa: F401  (adapters self-register)
        from studio.sources.base import REGISTRY
        return sorted(REGISTRY)

    @app.get("/series", response_class=HTMLResponse)
    def series_page(request: Request, error: str = ""):
        return page("series.html", request, series=_series_rows(con()),
                    sources=_source_ids(), error=error)

    @app.get("/series/{sid}", response_class=HTMLResponse)
    def series_detail(request: Request, sid: int):
        c = rcon(sid)
        chs = [dict(zip(("id", "number", "label", "status", "season",
                         "ep_dir", "url"), r))
               for r in c.execute(
                   "SELECT id, number, label, status, season, ep_dir, url "
                   "FROM chapter WHERE series_id=? ORDER BY number", (sid,))]
        activity = jobs.chapter_activity(c, sid)
        for ch_row in chs:
            ch_row["url"] = _http_url(ch_row["url"])
            ch_row["job"] = activity.get(ch_row["id"])
        title, series_url, autopilot, style = (c.execute(
            "SELECT title, series_url, autopilot, narration_style "
            "FROM series WHERE id=?", (sid,)).fetchone() or ("?", "", 0, None))
        # ONE teaser per manhwa, made independently of any video — it used to
        # hang off a bundle, so deleting a video destroyed it.
        teaser = REPO / "dist" / f"series_{sid}" / "teaser.mp4"
        teaser_exists = teaser.exists()
        teaser_state = (c.execute("SELECT teaser_state FROM series WHERE id=?",
                                  (sid,)).fetchone() or ["none"])[0]
        thumb = REPO / "dist" / f"series_{sid}" / "thumbnail_yt.jpg"
        thumb_exists = thumb.exists()
        thumb_ready = any(
            ch["ep_dir"] and (Path(ch["ep_dir"]) / "manifest.beats.json").exists()
            for ch in chs)
        # the planner reads cached beats/understanding, so it needs prepared
        # chapters — the same readiness the thumbnail requires
        teaser_ready = thumb_ready
        # ONE option label for BOTH range selects. They were rendered by two
        # separate template loops with different rules — the "from" select
        # appended the label for chapter 0, the "to" select did not — so the
        # same chapter read differently depending on which dropdown you were
        # looking at. Both now iterate this.
        opts = [{"value": f"{ch['number']:g}",
                 "text": f"{ch['number']:g} · {ch['label']}"
                         if ch["label"] else f"{ch['number']:g}",
                 "undownloaded": ch["status"] == "discovered"}
                for ch in chs]
        return page("series_detail.html", request, sid=sid, title=title,
                    series_url=_http_url(series_url), chapters=chs,
                    chapter_options=opts,
                    failed=jobs.failed_chapters(c, sid),
                    autopilot=bool(autopilot),
                    narration_style=style or "default",
                    thumb_exists=thumb_exists, thumb_ready=thumb_ready,
                    thumb_v=int(thumb.stat().st_mtime) if thumb_exists else 0,
                    thumb_approved=gates.thumbnail_approved(c, sid),
                    teaser_card=_teaser_card(sid),
                    teaser_state=teaser_state, teaser_exists=teaser_exists,
                    teaser_ready=teaser_ready,
                    teaser_v=int(teaser.stat().st_mtime) if teaser_exists else 0)

    @app.get("/partials/series-chapters/{sid}", response_class=HTMLResponse)
    def series_chapters_partial(request: Request, sid: int):
        # Plain con() (NOT rcon): this polls every few seconds, and reconcile is
        # a WRITE — the page load already reconciled. Here we only refresh the
        # live job overlay + stage, both read straight from the DB.
        c = con()
        chs = [dict(zip(("id", "number", "label", "status", "season",
                         "ep_dir", "url"), r))
               for r in c.execute(
                   "SELECT id, number, label, status, season, ep_dir, url "
                   "FROM chapter WHERE series_id=? ORDER BY number", (sid,))]
        activity = jobs.chapter_activity(c, sid)
        for ch_row in chs:
            ch_row["url"] = _http_url(ch_row["url"])
            ch_row["job"] = activity.get(ch_row["id"])
        return page("partials/series_chapters.html", request, sid=sid,
                    chapters=chs)

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
        # reconcile-on-load: repair any drift between the DB and the files on disk
        # so this page can never show a state the artifacts contradict (stale
        # status, dead run-history rows, a render approval that outlived its video)
        _rec = reconcile.reconcile_chapter(c, ch)
        if _rec["status_to"]:
            ch["status"] = _rec["status_to"]
        title = (c.execute("SELECT title FROM series WHERE id=?",
                           (ch["series_id"],)).fetchone() or ["?"])[0]
        ep_rel = _served_rel(ch["ep_dir"])
        allowed, why = gates.render_allowed(c, cid, ch["ep_dir"])
        v_allowed, v_why = gates.voice_allowed(c, cid, ch["ep_dir"])
        # the chapter's live job, if any — approving a gate kicks off work the
        # operator could previously only find on the queue page or in the logs
        # 'cancelling' included: its child is still running, so from this
        # page's point of view the chapter IS busy. Leaving it out made a
        # cancelled-but-not-yet-reaped chapter look idle while a process was
        # still writing its manifests.
        aj = c.execute(
            "SELECT id, type, state FROM job WHERE chapter_id=? AND "
            "state IN ('running','cancelling','queued') ORDER BY id DESC "
            "LIMIT 1", (cid,)).fetchone()
        active_job = dict(zip(("id", "type", "state"), aj)) if aj else None
        if active_job:
            # the SAME stale/overdue verdict the queue page uses, from the
            # same jobs.health() call — two pages computing "is it stuck"
            # separately is how they end up disagreeing
            _h = jobs.health(c)
            active_job["is_stale"] = any(s["id"] == active_job["id"]
                                         for s in _h["stale"])
            active_job["is_overdue"] = any(o["id"] == active_job["id"]
                                           for o in _h["overdue"])
        # full per-phase run history with timings — the stage_run rows always
        # had this; there was just no per-chapter view exposing it
        run_history = [dict(zip(("stage", "started_at", "dur", "ok"), r))
                       for r in c.execute(
                           "SELECT stage, started_at, duration_sec, ok "
                           "FROM stage_run WHERE chapter_id=? "
                           "ORDER BY id DESC LIMIT 60", (cid,))]
        has_preview = bool(ch["ep_dir"] and (
            Path(ch["ep_dir"]) / "render" / "voice_preview.mp3").exists())
        seg = (Path(ch["ep_dir"]) / paths.SEGMENT_MP4
               if ch["ep_dir"] else None)
        render_url = (f"/media/{ep_rel}/{paths.SEGMENT_MP4}"
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
        # "never downloaded" and "the images are GONE" look identical on this
        # page otherwise — both just render an empty gallery. The second is an
        # emergency (a rename, a failed sync, a deleted directory) and has to
        # be unmissable.
        files_missing = ""
        if ch["ep_dir"] and not Path(ch["ep_dir"]).is_dir():
            files_missing = (
                f"the episode directory for this chapter is GONE: "
                f"{ch['ep_dir']} — the catalog says status {ch['status']!r}, "
                f"but nothing is on disk. Nothing here can be rebuilt until "
                f"it is re-fetched or the path is repaired (see Catalog).")
        elif not ch["ep_dir"]:
            files_missing = ""      # simply never downloaded — not an error
        return page("chapter.html", request, ch=ch, series_title=title,
                    timeline=_stage_timeline(c, ch),
                    active_job=active_job, run_history=run_history,
                    files_missing=files_missing,
                    error=request.query_params.get("error", ""),
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
        c = rcon()
        rows = []
        for r in c.execute("SELECT id, series_id, title, kind, season_no, "
                           "state, output_path, teaser_state FROM bundle "
                           "ORDER BY id"):
            b = dict(zip(("id", "series_id", "title", "kind", "season_no",
                          "state", "output_path", "teaser_state"), r))

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

            # prefer the AUTO-generated title (publish_meta) over the
            # provisional "Episodes X–Y" set at creation
            # the teaser is a real file but had no link anywhere: after a
            # re-voice there was no way to hear it from the dashboard at all,
            # and "open" plays the CONCATENATED video, which still carries the
            # teaser as it was when the concat ran.
            tf = REPO / "dist" / f"bundle_{b['id']}" / "teaser.mp4"
            b["teaser_mp4"] = (f"bundle_{b['id']}/teaser.mp4"
                               if tf.exists() else "")
            b["teaser_stale"] = bool(
                b["teaser_mp4"] and b.get("output_path")
                and Path(b["output_path"]).exists()
                and tf.stat().st_mtime > Path(b["output_path"]).stat().st_mtime)
            auto = ""
            mp = REPO / "dist" / f"bundle_{b['id']}" / "publish_meta.json"
            if mp.exists():
                try:
                    auto = str(json.loads(mp.read_text()).get("title") or "")
                except Exception:
                    auto = ""
            chs = bundles.bundle_chapters(c, b["id"])
            span = c.execute(
                "SELECT MIN(number), MAX(number) FROM chapter WHERE id IN "
                f"({','.join('?' for _ in chs)})", chs).fetchone() if chs \
                else (None, None)
            b.update(ready=ready, total=total,
                     title=auto or b["title"],
                     span=(f"{span[0]:g}–{span[1]:g}"
                           if span and span[0] is not None else ""),
                     runtime=eta.fmt_eta(bundles.projected_runtime_sec(
                         c, b["id"], plan_dur)),
                     approved=gates.concat_allowed(c, b["id"])[0],
                     teaser_stale=_teaser_newer_than(b),
                     carries_teaser=(b["id"] == c.execute(
                         "SELECT MIN(id) FROM bundle WHERE series_id=?",
                         (b["series_id"],)).fetchone()[0]))
            rows.append(b)
        # first series' form is rendered inline; changing the dropdown swaps it
        first_sid = c.execute("SELECT id FROM series ORDER BY id "
                              "LIMIT 1").fetchone()
        form = _bundle_form_ctx(c, first_sid[0]) if first_sid else {
            "series_id": None, "options": [], "first": True, "series": []}
        return page("videos.html", request, bundles=rows, form=form,
                    error=request.query_params.get("error", ""))

    @app.get("/discovery", response_class=HTMLResponse)
    def discovery_page(request: Request, refresh: int = 0):
        c = con()
        if refresh:
            discovery.fetch_trending(c)
        return page("discovery.html", request, titles=discovery.listing(c),
                    sources=_source_ids())

    @app.get("/catalog-health", response_class=HTMLResponse)
    def catalog_health_page(request: Request):
        """Read-only chapter-identity audit: where the catalog and the files
        on disk disagree about which episode is which. This is the page that
        answers "why is this showing the wrong episode number" with evidence
        instead of a guess."""
        from studio.catalog import identity
        return page("catalog_health.html", request,
                    audits=identity.audit_all(con()))

    @app.get("/health", response_class=HTMLResponse)
    def health_page(request: Request):
        c = con()
        hb = c.execute("SELECT started_at FROM job WHERE type='heartbeat' "
                       "ORDER BY id DESC LIMIT 1").fetchone()
        # Probe the endpoint the WORKER actually uses. This was hardcoded to
        # localhost:11434 while the worker runs against OLLAMA_HOST — on both
        # machines that is the MLX shim on :11500 — so the health page happily
        # reported "(not running)" for a perfectly working pipeline, and would
        # equally have reported green while the real backend was down. Label
        # the probe with the URL so the answer is never ambiguous again.
        host = (os.environ.get("OLLAMA_HOST") or "localhost:11434").strip()
        if not host.startswith(("http://", "https://")):
            host = "http://" + host
        ollama = ""
        try:
            import httpx
            tags = httpx.get(f"{host}/api/tags", timeout=2).json()
            ollama = ", ".join(m["name"] for m in tags.get("models", [])[:4])
        except Exception:
            ollama = "(not running)"
        # Thumbnail generation is the ONLY paid step, and its key lives in the
        # macOS keychain on the Mini. A keychain read needs an UNLOCKED login
        # keychain: a `gui/` launchd agent inherits one, a non-interactive ssh
        # session does not. Reporting the source here answers "can the daemon
        # actually see it?" from inside the daemon — the only context whose
        # answer matters. Never shows the key itself.
        try:
            from thumbnail_gen import resolve_api_key as _gemini_key
            _k, _src = _gemini_key()
            gemini = f"found (from {_src})" if _k else "NOT FOUND — thumbnails will fail"
        except Exception as e:
            gemini = f"(probe failed: {type(e).__name__})"
        checks = {
            f"ollama models @ {host}": ollama,
            "gemini key (thumbnails)": gemini,
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
    def duration_estimate(series_id: int, num_from: Optional[float] = None,
                          num_to: Optional[float] = None, target: str = "qa"):
        """Two estimates for a selected range: ~processing time to build it, and
        the ~length of the final video. Rough (seed/median based)."""
        from studio.dashboard import eta as _eta
        seed = getattr(_eta, "SEED_SEC", {})
        c = con()
        rows = jobs.range_total(c, series_id, num_from=num_from, num_to=num_to)
        n_total = len(rows)
        # EXACTLY the rows the button will enqueue — same helper, so the
        # estimate can no longer promise work the enqueue then skips.
        to_build = jobs.range_chapter_ids(c, series_id, num_from=num_from,
                                          num_to=num_to, target=target)
        n_missing = len(to_build)
        # chapters in range that still need downloading — a bulk run DOES
        # fetch them (the prepare handler fetches first), but they cost extra
        # wall-clock and the operator should not be surprised by it
        n_undownloaded = sum(1 for _, st in rows if st == "discovered")
        vid = 0.0
        have = 0
        for (ep_dir, _st) in rows:
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

        note = (f' · <b>{n_undownloaded}</b> need downloading first'
                if n_undownloaded else "")
        return HTMLResponse(
            f'<span class="kv">{n_missing} of {n_total} chapters to build · '
            f'~<b>{fmt(proc)}</b> · ~<b>{fmt(vid)}</b> final video{note}</span>')

    @app.post("/jobs")
    def post_job(type: str = Form(...), chapter_id: Optional[int] = Form(None),
                 series_id: Optional[int] = Form(None),
                 bundle_id: Optional[int] = Form(None),
                 target: str = Form(""), branding: str = Form("both"),
                 num_from: Optional[float] = Form(None),
                 num_to: Optional[float] = Form(None)):
        c = con()
        if type == "prepare_range":
            # bulk: run chapters in [num_from..num_to] up to a target stage —
            # qa (prepare only) | voice (prepare→voiceover) | video (→render).
            # auto_to is carried on each prepare job; the worker advances past the
            # approval gates only as far as the target (QA must stay green).
            auto_to = {"voice": "voice", "video": "video"}.get(target or "qa")
            # RESUME semantics live in jobs.range_chapter_ids — the SAME call
            # the estimate makes, so what the estimate promised is exactly
            # what gets enqueued.
            for cid in jobs.range_chapter_ids(c, series_id, num_from=num_from,
                                              num_to=num_to, target=target):
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

    @app.post("/jobs/{job_id}/requeue")
    def post_requeue(job_id: int):
        jobs.requeue(con(), job_id)
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
        ep_dir = gates.chapter_ep_dir(c, chapter_id) if chapter_id else None
        # Enforce the gate HERE, not only by hiding the button in the
        # template. Approving with red QA used to insert an approval row and
        # enqueue a job that failed a minute later inside the worker's own
        # gate check — a manufactured dead-letter, and an approval record for
        # something that was never approvable.
        ok, why = gates.approvable(c, gate, chapter_id=chapter_id,
                                   ep_dir=ep_dir)
        if not ok:
            back = f"/chapter/{chapter_id}" if chapter_id else "/videos"
            return RedirectResponse(f"{back}?error=" + quote_plus(why),
                                    status_code=303)
        gates.approve(c, gate, series_id=series_id, chapter_id=chapter_id,
                      bundle_id=bundle_id, note=note,
                      content_sha=gates.gate_sha(gate, ep_dir))
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

    def _bundle_form_ctx(c: sqlite3.Connection, series_id: int) -> Dict[str, Any]:
        """Options for the new-video form: EVERY rendered chapter of the series
        as a selectable from/to bound, with the ones already in a video marked
        rather than removed. Hiding them made the range un-free — a 12-episode
        debut left the form starting at Episode 12 with no way back. Chapter 0
        is a real option on a 0-based series."""
        chs = bundles.rendered_chapters(c, series_id)
        opts = []
        for ch in chs:
            text = (f"{ch['number']:g} · {ch['label']}"
                    if ch["label"] else f"{ch['number']:g}")
            if ch["bundled"]:
                text += "  ·  already in a video"
            opts.append({"value": f"{ch['number']:g}", "text": text})
        return {"series_id": series_id, "options": opts,
                "first": bundles.series_bundle_count(c, series_id) == 0,
                "series": [dict(zip(("id", "title"), r)) for r in c.execute(
                    "SELECT id, title FROM series ORDER BY id")]}

    def _range_duration(c: sqlite3.Connection, series_id: int,
                        num_from: Optional[float], num_to: Optional[float],
                        first: bool) -> Dict[str, Any]:
        ids = bundles.ids_in_range(c, series_id, num_from=num_from,
                                   num_to=num_to)
        total, known = 0.0, 0
        for cid in ids:
            r = c.execute("SELECT ep_dir FROM chapter WHERE id=?",
                          (cid,)).fetchone()
            d = _chapter_plan_dur(r[0] if r else None)
            if d:
                total += d
                known += 1
        # nominal cold-open + outro so the number is honest before render
        teaser = 30.0 if first else 0.0
        outro = 5.0
        return {"count": len(ids), "known": known,
                "seconds": total + (teaser if ids else 0) + (outro if ids else 0),
                "teaser": bool(first and ids)}

    @app.get("/partials/bundle-form", response_class=HTMLResponse)
    def bundle_form(request: Request, series_id: int):
        # status='rendered' is kept current by the /videos page-load reconcile
        # (rcon reconciles every series); this partial only re-renders the form
        return page("partials/bundle_form.html", request,
                    **_bundle_form_ctx(con(), series_id))

    @app.get("/partials/bundle-duration", response_class=HTMLResponse)
    def bundle_duration(series_id: int, num_from: Optional[float] = None,
                        num_to: Optional[float] = None):
        c = con()
        first = bundles.series_bundle_count(c, series_id) == 0
        d = _range_duration(c, series_id, num_from, num_to, first)
        if not d["count"]:
            return HTMLResponse('<span class="kv">no rendered chapters in '
                                'that range</span>')
        vid = eta.fmt_eta(d["seconds"])
        pend = (f' · <b>{d["count"] - d["known"]}</b> not rendered yet'
                if d["known"] < d["count"] else "")
        teaser = " · includes intro teaser" if d["teaser"] else ""
        return HTMLResponse(
            f'<span class="kv"><b>{d["count"]}</b> chapters · '
            f'~<b>{vid}</b> final video{teaser}{pend}</span>')

    @app.post("/bundles")
    def post_bundle(series_id: int = Form(...),
                    num_from: Optional[float] = Form(None),
                    num_to: Optional[float] = Form(None)):
        c = con()
        ids = bundles.ids_in_range(c, series_id, num_from=num_from,
                                   num_to=num_to)
        if not ids:
            return RedirectResponse(
                "/videos?error=" + quote_plus(
                    "no rendered chapters in that range — a video is a concat "
                    "of finished episode segments"), status_code=303)
        # Episodes may be re-used across videos deliberately now, so
        # "already bundled" can no longer be what stops an accidental
        # double-submit. Refuse only an IDENTICAL video; any other range,
        # overlapping or not, is allowed.
        dup = bundles.bundle_with_exact_chapters(c, series_id, ids)
        if dup is not None:
            return RedirectResponse(
                "/videos?error=" + quote_plus(
                    f"video #{dup} already covers exactly those episodes — "
                    "pick a different range, or delete it first"),
                status_code=303)
        first = bundles.series_bundle_count(c, series_id) == 0
        nums = [r[0] for r in c.execute(
            "SELECT number FROM chapter WHERE id IN "
            f"({','.join('?' for _ in ids)}) ORDER BY number", ids)]
        srow = c.execute("SELECT title FROM series WHERE id=?",
                         (series_id,)).fetchone()
        sname = str((srow[0] if srow else "") or "").strip()
        # name it after the manhwa AND the range picked, so a list of videos is
        # readable without opening them ("Debut — first 12" said neither)
        span = f"Episodes {nums[0]:g}–{nums[-1]:g}" if nums else "Video"
        title = f"{sname} — {span}" if sname else span
        bid = bundles.create_bundle(c, series_id, "manual", chapter_ids=ids,
                                    title=title)
        # auto title/description from the arc (gemma) — replaces the provisional
        jobs.enqueue(c, "publish_meta", bundle_id=bid)
        # the DEBUT video carries the channel's intro teaser (a proposal to
        # review); continuation videos don't re-intro.
        if first:
            jobs.enqueue(c, "plan_teaser", bundle_id=bid)
        return RedirectResponse("/videos", status_code=303)

    @app.post("/bundles/{bid}/delete")
    def post_bundle_delete(bid: int):
        """Delete a video (bundle). Removes ONLY the bundle's own rows and its
        dist/bundle_<id> artifacts (teaser, publish_meta) — the chapters it
        referenced are untouched and go back into the pool, so a stale or
        mis-ranged video can be removed and its episodes re-bundled. Closes the
        gap where the only way to clear a bundle was to delete the whole series
        (and a clean-slate reset left stale packs behind, locking their
        chapters out of the video creator)."""
        c = con()
        reset.delete_bundle(c, bid)      # shared with rewind's clear-on-reset
        c.commit()
        return RedirectResponse("/videos", status_code=303)

    @app.post("/series/{sid}/teaser/plan")
    def post_teaser_plan(sid: int):
        # ONE teaser per manhwa, made independently of any video: select a
        # high-stakes hook window across the series' first N processed chapters
        # and render the cold open. The worker does the model select + render.
        jobs.enqueue(con(), "plan_teaser", series_id=sid)
        return RedirectResponse(f"/series/{sid}", status_code=303)

    @app.post("/series/{sid}/teaser/approve")
    def post_teaser_approve(sid: int):
        # confirm-upstream-before-render: record the gate AND flip the state so
        # the concat gate (which reads it off the series) lets the FIRST video
        # proceed.
        c = con()
        gates.approve(c, "teaser", series_id=sid)
        c.execute("UPDATE series SET teaser_state='approved' WHERE id=?", (sid,))
        c.commit()
        return RedirectResponse(f"/series/{sid}", status_code=303)

    @app.post("/series/{sid}/teaser/decline")
    def post_teaser_decline(sid: int):
        # operator rejected the hook — unblock the concat without prepending it.
        c = con()
        c.execute("UPDATE series SET teaser_state='declined' WHERE id=?", (sid,))
        c.commit()
        return RedirectResponse(f"/series/{sid}", status_code=303)

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
        # stage fixes actually apply (resume-by-status never re-runs them).
        # rewind_chapter is the ONE atomic rewind: gen-1 audio/renders +
        # narration outputs cleared (beats backed up), voice+render approvals
        # dropped, and — the class the old inline version missed — the
        # chapter's stage_run history rewound so readiness counters stop
        # claiming voiced/rendered for artifacts that no longer exist.
        c = con()
        reset.rewind_chapter(c, cid, "detected")
        jobs.enqueue(c, "prepare", chapter_id=cid, priority=50)
        return RedirectResponse(f"/chapter/{cid}", status_code=303)

    @app.post("/chapter/{cid}/revoice")
    def post_revoice(cid: int):
        # Re-synthesize the voiceover with the latest TTS (seeded + de-robotted)
        # and re-render — KEEPS the approved narration, only the audio + video
        # are rebuilt. rewind_chapter('scripted') deletes the clip cache +
        # renders, drops the render gate, rewinds the voiced/render_segment
        # stage_run history and sets status='scripted' so the resume-by-status
        # runner actually re-runs the voiced stage.
        c = con()
        reset.rewind_chapter(c, cid, "scripted", clear_approvals=("render",))
        # narration stays approved; auto_to=video re-approves render after QA.
        # Clicking Re-voice IS consent for whatever narration is on disk now,
        # so (re-)stamp the voice gate unconditionally: ensure_approval no-ops
        # if the existing row is already valid for the CURRENT content, and
        # re-stamps if it's stale (a heal rewrote the script after approval)
        # or absent — otherwise the enqueue below would wedge on a sha
        # mismatch inside the voiceover handler's gate check.
        gates.ensure_approval(c, "voice", chapter_id=cid,
                              ep_dir=gates.chapter_ep_dir(c, cid), note="re-voice")
        # dedupe=False: operator intent carries a payload; the chapter lease
        # serializes the duplicate and resume-by-status makes it cheap.
        jobs.enqueue(c, "voiceover", chapter_id=cid,
                     payload={"auto_to": "video"}, priority=50, dedupe=False)
        return RedirectResponse(f"/chapter/{cid}", status_code=303)

    @app.post("/chapter/{cid}/reload")
    def post_reload(cid: int):
        # Manual reload of a dead-lettered chapter (auto-retry exhausted). Re-queue
        # its prepare at the FRONT with auto_to=video so it drives all the way to
        # render; prepare resumes by status, so it picks up wherever it died.
        c = con()
        r = c.execute("SELECT series_id FROM chapter WHERE id=?", (cid,)).fetchone()
        sid = r[0] if r else None
        front = int(os.environ.get("STUDIO_RETRY_PRIORITY", "1"))
        # dedupe=False: operator intent carries a payload; the chapter lease
        # serializes the duplicate and resume-by-status makes it cheap.
        jobs.enqueue(c, "prepare", chapter_id=cid, series_id=sid,
                     payload={"auto_to": "video"}, priority=front,
                     dedupe=False)
        return RedirectResponse(f"/series/{sid}" if sid else "/",
                                status_code=303)

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

    @app.post("/series/{sid}/refresh")
    def post_refresh(sid: int):
        # Re-discovery runs INLINE so the view updates immediately (an async
        # job left the total unmoved and the NEW badge uncleared, which is why
        # this was made synchronous). It reuses cmd_refresh — an idempotent
        # upsert, so only genuinely new chapters add rows.
        #
        # The return code is now CHECKED. Previously every failure mode was
        # swallowed by a bare `except Exception: pass` and the NEW badge was
        # cleared anyway — so a source site being down looked exactly like
        # "no new chapters", and the alert you needed was destroyed. On
        # failure the badge stays put and the error is surfaced.
        import subprocess
        c = con()
        try:
            r = subprocess.run([PY, "-m", "studio", "refresh",
                                "--series", str(sid)], cwd=str(REPO),
                               capture_output=True, text=True, timeout=90)
            ok, detail = r.returncode == 0, (r.stderr or "")[-300:]
        except subprocess.TimeoutExpired:
            ok, detail = False, "discovery timed out after 90s"
        except Exception as e:
            ok, detail = False, str(e)[:300]
        if ok:
            # a manual refresh means "I'm looking now"
            c.execute("UPDATE series SET new_pending=0 WHERE id=?", (sid,))
            c.commit()
            return RedirectResponse("/series", status_code=303)
        return RedirectResponse(
            f"/series?error=refresh+failed:+{quote_plus(detail)}",
            status_code=303)

    def _series_running_jobs(c: sqlite3.Connection, sid: int) -> int:
        # 'cancelling' counts as live: its child is still running until the
        # worker reaps it, so deleting the series out from under it would
        # leave a process writing into a directory we just removed.
        return c.execute(
            "SELECT COUNT(*) FROM job WHERE state IN ('running','cancelling') "
            "AND type!='heartbeat'"
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
