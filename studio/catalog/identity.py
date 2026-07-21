"""Chapter identity: what a chapter row claims vs what is actually on disk.

THE BUG THIS EXISTS FOR
-----------------------
A chapter's episode directory name is minted ONCE, from the label, at fetch
time (`studio.cli.cmd_fetch`). But `repo.upsert_chapter` keys on
`(series_id, number)` and freely OVERWRITES `label` and `url` every time the
daily refresh re-discovers the series. Nothing records what a directory
actually contains.

So the moment a source renumbers or retitles anything — a prologue promoted
to "Episode 1", a side story inserted, a scraper offset changed — the DB row
and the directory silently stop agreeing. The row says "Chapter 1", the
directory is named `Episode_2`, and the images inside it are whatever was
downloaded under the old numbering. Every downstream page then shows a
plausible, wrong episode: the exact "wrong episode numbers / wrong file
links" symptom.

This module only ANSWERS QUESTIONS. It never renames, never writes. The
repair tool (`studio.catalog.repair`) is separate, journaled and reversible,
because deciding which of two disagreeing sources is right requires looking
at the actual images — something no amount of code can do for you.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from studio import config as studio_config
from studio.sources.base import is_fetchable_url

# Issue codes, most severe first. Severity is about how badly a human is
# likely to be misled, not about how hard it is to fix.
SEVERITY = {
    "ep_dir_shared": "error",         # two chapters, one directory
    "url_unfetchable": "error",       # a placeholder link, not a chapter page
    "url_duplicate": "error",         # two rows claim the same source page
    "canonical_collision": "error",   # repairing would collide
    "ep_dir_outside_tree": "error",   # a DB copied from another machine
    "ep_dir_missing": "error",        # row says downloaded, nothing there
    "label_number_mismatch": "warn",  # 'Episode 5' filed under number 4
    "ep_dir_name_drift": "warn",      # directory named for a stale label
}


def canonical_dirname(label: str) -> str:
    """The filesystem-safe directory name for a chapter label.

    This is byte-for-byte the rule `studio.cli` has always used, kept here so
    the audit and the fetcher can never drift apart. Reproducing existing
    well-formed directories EXACTLY is the whole point: it means the audit's
    blast radius is only the rows that genuinely drifted, and a repair run on
    a healthy catalog is a no-op.
    """
    safe = re.sub(r"[^\w\-.]", "_", label)
    safe = re.sub(r"_+", "_", safe).strip("_.")
    return safe or "chapter"


def canonical_ep_dir(slug: str, label: str) -> Path:
    """Where a chapter's images belong: ongoing/<slug>/<canonical label>."""
    return (studio_config.REPO_ROOT / "ongoing" / slug
            / canonical_dirname(label))


_SEASON_LABEL_RE = re.compile(r"\bseason\s*\d+", re.IGNORECASE)

_NUM_RE = re.compile(
    r"(?:chapter|chap|ch|episode|ep|part|#)\s*\.?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE)


def label_number(label: str) -> Optional[float]:
    """The chapter number a label CLAIMS, or None when it does not clearly
    claim one. Returns None rather than guessing: a label like 'Prologue' or
    'Season 2 Finale' has no number, and inventing one would manufacture
    false mismatches — the audit must only report what it can prove."""
    m = _NUM_RE.search(label or "")
    if m:
        return float(m.group(1))
    # a bare number ('12', '12.5') is unambiguous; anything else is not
    s = (label or "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    return None


def _issue(code: str, chapter: Dict[str, Any], detail: str,
           **extra: Any) -> Dict[str, Any]:
    return {"code": code, "severity": SEVERITY.get(code, "warn"),
            "chapter_id": chapter.get("id"), "number": chapter.get("number"),
            "label": chapter.get("label"), "detail": detail, **extra}


def audit_series(con: sqlite3.Connection, series_id: int) -> Dict[str, Any]:
    """Every identity problem in one series. Read-only."""
    srow = con.execute("SELECT slug, title FROM series WHERE id=?",
                       (series_id,)).fetchone()
    if not srow:
        return {"series_id": series_id, "slug": None, "title": None,
                "chapters": 0, "issues": []}
    slug, title = srow
    rows = con.execute(
        "SELECT id, number, label, url, status, ep_dir FROM chapter "
        "WHERE series_id=? ORDER BY number", (series_id,)).fetchall()
    chapters = [dict(zip(("id", "number", "label", "url", "status", "ep_dir"),
                         r)) for r in rows]

    issues: List[Dict[str, Any]] = []
    offsets: List[tuple] = []
    by_dir: Dict[str, List[Dict[str, Any]]] = {}
    by_url: Dict[str, List[Dict[str, Any]]] = {}
    by_canon: Dict[str, List[Dict[str, Any]]] = {}
    ongoing = (studio_config.REPO_ROOT / "ongoing").resolve()

    for ch in chapters:
        canon = canonical_dirname(ch["label"] or "")
        ch["canonical"] = canon
        by_canon.setdefault(canon, []).append(ch)
        if ch["url"]:
            by_url.setdefault(ch["url"], []).append(ch)

        # A placeholder link stored as if it were a chapter page. This is how
        # elftoon's announced-but-unpublished chapters got in: "#" appended to
        # the origin yields a URL that returns HTTP 200 and serves the
        # HOMEPAGE, so nothing downstream can tell it apart from a real
        # chapter until a fetch fails deep inside image extraction — after the
        # row has already inflated the chapter count and joined every bulk-run
        # range. Checked here so ANY adapter regressing this way is caught.
        if ch["url"] and not is_fetchable_url(ch["url"]):
            issues.append(_issue(
                "url_unfetchable", ch,
                f"{ch['url']} is a placeholder, not a chapter page — the row "
                f"was created from an announced-but-unpublished entry and can "
                f"never be fetched. Delete it; a refresh will re-add the "
                f"chapter once the source publishes a real link."))

        # A label naming a different number than its row is only a DEFECT when
        # it disagrees with the rest of its series — see the post-pass below.
        #
        # A SEASON-RELATIVE label is excluded entirely: "Season 2 Episode 0"
        # restarts at 0 while `number` keeps counting globally, so the two are
        # not on the same scale and no offset between them means anything.
        # Tower of God's offsets run -1, -80, -81, -418 across its seasons;
        # comparing them would manufacture hundreds of false anomalies.
        claimed = label_number(ch["label"] or "")
        if (claimed is not None and ch["number"] is not None
                and not _SEASON_LABEL_RE.search(ch["label"] or "")):
            offsets.append((ch, claimed - float(ch["number"])))

        ep = ch["ep_dir"]
        if not ep:
            continue                    # never fetched: nothing to disagree
        p = Path(ep)
        by_dir.setdefault(str(p), []).append(ch)
        try:
            resolved = p.resolve()
            inside = resolved.is_relative_to(ongoing)
        except Exception:
            resolved, inside = p, False
        if not inside:
            issues.append(_issue(
                "ep_dir_outside_tree", ch,
                f"{ep} is not under ongoing/ — typically a catalog copied "
                f"from the other machine, where this absolute path was valid",
                ep_dir=ep))
            continue
        if not p.is_dir():
            issues.append(_issue(
                "ep_dir_missing", ch,
                f"status is {ch['status']!r} but {ep} does not exist",
                ep_dir=ep))
            continue
        if p.name != canon:
            issues.append(_issue(
                "ep_dir_name_drift", ch,
                f"directory is named {p.name!r} but the label is "
                f"{ch['label']!r} (canonically {canon!r}) — the label was "
                f"changed by a re-discovery after the images were fetched",
                ep_dir=ep, want=canon,
                want_path=str(p.parent / canon)))

    for d, chs in by_dir.items():
        if len(chs) > 1:
            for ch in chs:
                issues.append(_issue(
                    "ep_dir_shared", ch,
                    "shares its episode directory with chapter(s) "
                    + ", ".join(f"{c['label']}(#{c['id']})"
                                for c in chs if c["id"] != ch["id"])
                    + " — they cannot both be right", ep_dir=d))
    for u, chs in by_url.items():
        if len(chs) > 1:
            for ch in chs:
                issues.append(_issue(
                    "url_duplicate", ch,
                    "same source URL as "
                    + ", ".join(f"{c['label']}(#{c['id']})"
                                for c in chs if c["id"] != ch["id"])
                    + f" ({u}) — two rows were discovered from one page, so "
                      "at least one is numbered wrong"))
    # label-vs-number: flag DEVIATION FROM THE SERIES' OWN CONVENTION, not the
    # convention itself.
    #
    # `number` is not always the published episode number. For webtoon it is
    # gallery-dl's episode_no — a sequence index that is deliberately offset
    # from the label, because it is the UNIQUE key and the refresh upsert key
    # and therefore must not move (see sources/webtoon._label_from_slug).
    # A uniform offset across a series is a NUMBERING CONVENTION and is
    # silent; ORV would otherwise report 309 warnings and bury real findings.
    # A row that disagrees with its own series' convention is a genuine
    # anomaly — exactly the one-off drift this audit exists to catch.
    if offsets:
        counts: Dict[float, int] = {}
        for _ch, off in offsets:
            counts[off] = counts.get(off, 0) + 1
        convention = max(counts, key=lambda k: counts[k])
        for ch, off in offsets:
            if abs(off - convention) < 1e-9:
                continue
            issues.append(_issue(
                "label_number_mismatch", ch,
                f"row is number {ch['number']:g} and the label says "
                f"{ch['number'] + off:g} (offset {off:+g}), but the rest of "
                f"this series uses offset {convention:+g} — this row "
                f"disagrees with its own series, so one of the two is wrong"))

    for canon, chs in by_canon.items():
        if len(chs) > 1:
            for ch in chs:
                issues.append(_issue(
                    "canonical_collision", ch,
                    f"label resolves to the same directory name {canon!r} as "
                    + ", ".join(f"{c['label']}(#{c['id']})"
                                for c in chs if c["id"] != ch["id"])
                    + " — a rename repair cannot be applied here"))

    order = {"error": 0, "warn": 1}
    issues.sort(key=lambda i: (order.get(i["severity"], 2),
                               i["number"] if i["number"] is not None else 0))
    return {"series_id": series_id, "slug": slug, "title": title,
            "chapters": len(chapters), "issues": issues,
            "errors": sum(1 for i in issues if i["severity"] == "error"),
            "warnings": sum(1 for i in issues if i["severity"] == "warn")}


def audit_all(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    return [audit_series(con, sid) for (sid,) in
            con.execute("SELECT id FROM series ORDER BY id")]
