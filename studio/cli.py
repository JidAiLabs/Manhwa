"""
studio/cli.py

Command-line interface for manhwa-studio.

Subcommands:
  add-series  <source> <series_url>   -- discover + upsert series and all chapters
  list        [--series ID]            -- list tracked series or chapters of one series
  fetch       <series_id>              -- download selected chapters
  run         <series_id>              -- run pipeline on selected chapters
  status      [series_id]              -- show chapter status table

All timestamps are generated here and passed into repo/pipeline — pipeline.py
and repo functions never call datetime directly.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import studio.sources  # noqa: F401 — triggers adapter self-registration
from studio.catalog import db as catalog_db
from studio.catalog import repo
from studio.catalog.models import Chapter
from studio import config as studio_config
from studio.sources.base import get as get_adapter


# ---------------------------------------------------------------------------
# DB path (monkeypatchable for tests)
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    return studio_config.REPO_ROOT / "studio.db"


def _open_db():
    return catalog_db.connect(_db_path())


# ---------------------------------------------------------------------------
# Timestamp helper (always UTC ISO)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Chapter selector helper
# ---------------------------------------------------------------------------

def parse_chapter_selector(spec: str, chapters: list[Chapter]) -> list[Chapter]:
    """Return the subset of *chapters* matching *spec*.

    Supported forms:
      ``N``     -- single chapter by number (e.g. ``3``)
      ``N-M``   -- inclusive range by number (e.g. ``1-5``)
      ``new``   -- chapters whose status is ``discovered`` (not yet downloaded)
    """
    if spec == "new":
        return [c for c in chapters if c.status == "discovered"]

    range_match = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", spec)
    if range_match:
        lo = float(range_match.group(1))
        hi = float(range_match.group(2))
        return [c for c in chapters if lo <= c.number <= hi]

    single_match = re.fullmatch(r"(\d+(?:\.\d+)?)", spec)
    if single_match:
        n = float(single_match.group(1))
        return [c for c in chapters if c.number == n]

    raise ValueError(
        f"Invalid --chapters spec '{spec}'. "
        "Use 'N', 'N-M', or 'new'."
    )


# ---------------------------------------------------------------------------
# Label → filesystem-safe name
# ---------------------------------------------------------------------------

def _sanitize_label(label: str) -> str:
    """Convert a chapter label to a filesystem-safe directory name."""
    safe = re.sub(r"[^\w\-.]", "_", label)
    safe = re.sub(r"_+", "_", safe).strip("_.")
    return safe or "chapter"


# ---------------------------------------------------------------------------
# Subcommand: add-series
# ---------------------------------------------------------------------------

def cmd_add_series(args: argparse.Namespace) -> int:
    adapter = get_adapter(args.source)
    now = _now_iso()
    meta = adapter.series_meta(args.series_url)
    con = _open_db()
    sid = repo.upsert_series(
        con,
        meta.source,
        meta.series_url,
        meta.slug,
        meta.title,
        added_at=now,
    )
    chapters = adapter.list_chapters(args.series_url)
    for ch in chapters:
        repo.upsert_chapter(con, sid, ch.number, ch.label, ch.url, updated_at=now)
    print(f"series_id={sid} chapters={len(chapters)}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    con = _open_db()
    if args.series is not None:
        chapters = repo.list_chapters(con, args.series)
        if not chapters:
            print(f"No chapters found for series {args.series}")
        for ch in chapters:
            print(f"  [{ch.id:>4}] ch{ch.number:>6}  {ch.label:<30}  {ch.status}")
    else:
        series_list = repo.list_series(con)
        if not series_list:
            print("No series tracked.")
        for s in series_list:
            print(f"  [{s.id:>4}]  {s.source:<12}  {s.title}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: fetch
# ---------------------------------------------------------------------------

def cmd_fetch(args: argparse.Namespace) -> int:
    con = _open_db()
    series = repo.get_series(con, args.series_id)
    chapters = repo.list_chapters(con, args.series_id)
    selected = parse_chapter_selector(args.chapters, chapters)

    if not selected:
        print("No chapters match the selector.")
        return 0

    import studio.sources  # ensure adapters loaded
    adapter = get_adapter(series.source)

    from studio.sources.base import ChapterRef

    fetched = 0
    for ch in selected:
        if not args.force and ch.status != "discovered":
            print(f"  Skipping ch{ch.number} (status={ch.status}). Use --force to re-download.")
            continue

        ep_dir = (
            studio_config.REPO_ROOT
            / "ongoing"
            / series.slug
            / _sanitize_label(ch.label)
        )
        ep_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Fetching ch{ch.number} → {ep_dir} …")
        chapter_ref = ChapterRef(number=ch.number, label=ch.label, url=ch.url)
        adapter.download(chapter_ref, ep_dir)

        now = _now_iso()
        repo.set_chapter_status(
            con,
            ch.id,
            "downloaded",
            ep_dir=str(ep_dir),
            updated_at=now,
        )
        fetched += 1

    print(f"Fetched {fetched} chapter(s).")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    from studio import pipeline

    cfg = studio_config.load()
    con = _open_db()
    chapters = repo.list_chapters(con, args.series_id)
    selected = parse_chapter_selector(args.chapters, chapters)

    if not selected:
        print("No chapters match the selector.")
        return 0

    for ch in selected:
        from studio.catalog.models import STATUS_ORDER
        if ch.status == "discovered":
            print(f"  Skipping ch{ch.number} (not downloaded yet).")
            continue
        # Allow running on any downloaded+ status (including *_failed for resume)
        if not ch.status.endswith("_failed"):
            try:
                idx = STATUS_ORDER.index(ch.status)
            except ValueError:
                print(f"  Skipping ch{ch.number} (unknown status '{ch.status}').")
                continue
            if idx < STATUS_ORDER.index("downloaded"):
                print(f"  Skipping ch{ch.number} (status={ch.status}, not yet downloaded).")
                continue

        print(f"  Running pipeline for ch{ch.number} (status={ch.status}) …")
        pipeline.run_chapter(con, ch, cfg, now_fn=_now_iso)
        updated = repo.get_chapter(con, ch.id)
        print(f"    → {updated.status}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: qa
# ---------------------------------------------------------------------------

def cmd_qa(args: argparse.Namespace) -> int:
    from studio.qa import build_qa_report

    con = _open_db()
    chapters = repo.list_chapters(con, args.series_id)
    selected = parse_chapter_selector(args.chapters, chapters)

    if not selected:
        print("No chapters match the selector.")
        return 0

    for ch in selected:
        if not ch.ep_dir:
            print(f"  ch{ch.number}: no ep_dir recorded — run fetch first.")
            continue

        ep_dir = Path(ch.ep_dir)
        if args.out:
            out_html = Path(args.out)
        else:
            out_html = ep_dir / "qa_report.html"

        result = build_qa_report(ep_dir, out_html)
        print(str(result))

    return 0


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    con = _open_db()
    if args.series_id is not None:
        chapters = repo.list_chapters(con, args.series_id)
        if not chapters:
            print(f"No chapters for series {args.series_id}.")
            return 0
        print(f"{'ID':>6}  {'#':>6}  {'Label':<30}  {'Status':<20}  Error")
        print("-" * 80)
        for ch in chapters:
            err = (ch.error or "")[:40]
            print(f"{ch.id:>6}  {ch.number:>6}  {ch.label:<30}  {ch.status:<20}  {err}")
    else:
        series_list = repo.list_series(con)
        if not series_list:
            print("No series tracked.")
            return 0
        print(f"{'ID':>6}  {'Source':<12}  {'Title'}")
        print("-" * 50)
        for s in series_list:
            print(f"{s.id:>6}  {s.source:<12}  {s.title}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="studio",
        description="manhwa-studio pipeline CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # add-series
    p_add = sub.add_parser("add-series", help="Register a new series and discover chapters")
    p_add.add_argument("source", help="Source adapter id (e.g. webtoon, asura)")
    p_add.add_argument("series_url", help="URL of the series page")

    # list
    p_list = sub.add_parser("list", help="List tracked series or chapters")
    p_list.add_argument("--series", type=int, default=None, metavar="ID",
                        help="Show chapters of this series")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Download selected chapters")
    p_fetch.add_argument("series_id", type=int)
    p_fetch.add_argument("--chapters", required=True,
                         help="Chapter selector: N, N-M, or 'new'")
    p_fetch.add_argument("--force", action="store_true",
                         help="Re-download even if already downloaded")

    # run
    p_run = sub.add_parser("run", help="Run pipeline on selected chapters")
    p_run.add_argument("series_id", type=int)
    p_run.add_argument("--chapters", required=True,
                       help="Chapter selector: N, N-M, or 'new'")

    # status
    p_status = sub.add_parser("status", help="Show chapter status table")
    p_status.add_argument("series_id", type=int, nargs="?", default=None)

    # qa
    p_qa = sub.add_parser("qa", help="Generate scene↔narration QA report")
    p_qa.add_argument("series_id", type=int)
    p_qa.add_argument("--chapters", required=True,
                      help="Chapter selector: N, N-M, or 'new'")
    p_qa.add_argument("--out", default=None, metavar="PATH",
                      help="Output HTML path (default: <ep_dir>/qa_report.html)")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    """Entry point.  Parses *argv* (defaults to sys.argv[1:]) and dispatches.

    Calls sys.exit() only when invoked from a terminal context (i.e. argv is
    None, meaning real CLI invocation).  When called with an explicit argv list
    (e.g. from tests) it returns normally so callers can inspect results.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "add-series": cmd_add_series,
        "list":       cmd_list,
        "fetch":      cmd_fetch,
        "run":        cmd_run,
        "status":     cmd_status,
        "qa":         cmd_qa,
    }

    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        if argv is None:
            sys.exit(1)
        return

    rc = fn(args)
    if argv is None:
        sys.exit(rc)
