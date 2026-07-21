"""Journaled, reversible repair of chapter-identity drift.

Scope, deliberately narrow: this tool renames episode directories whose NAME
has drifted from their chapter's label, and updates `chapter.ep_dir` to
match. That is the one class of identity issue where the correct action is
mechanically derivable — the images are fine, only the folder name is stale.

It will NOT touch the classes that require looking at the pictures:
`url_duplicate`, `ep_dir_shared` and `canonical_collision` all mean two rows
disagree about which is which, and no rule can settle that from metadata
alone. When any of those are present the tool refuses to apply unless you
pass --content-verified, which is your assertion that you have opened the
directories and confirmed what is in them.

Every run writes its journal BEFORE mutating anything, so --undo can always
put the tree back. Renames are os.rename only (same filesystem, atomic);
nothing is ever copied or deleted.

    # look, change nothing (the default)
    python -m studio.catalog.repair --db studio.db
    # do it, keeping a journal
    python -m studio.catalog.repair --db studio.db --apply --journal r.json
    # put it back
    python -m studio.catalog.repair --db studio.db --undo r.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from studio.catalog import identity

# Issues that mean "two rows disagree about identity" — never auto-resolvable
_AMBIGUOUS = {"url_duplicate", "ep_dir_shared", "canonical_collision"}


def plan_series(con: sqlite3.Connection, series_id: int) -> Dict[str, Any]:
    """The rename actions for one series, plus everything blocking them."""
    audit = identity.audit_series(con, series_id)
    actions: List[Dict[str, Any]] = []
    blockers: List[str] = []

    ambiguous = [i for i in audit["issues"] if i["code"] in _AMBIGUOUS]
    drifted = [i for i in audit["issues"] if i["code"] == "ep_dir_name_drift"]

    # a chapter with a live job must not have its directory moved out from
    # under the process that is writing into it
    busy = {r[0] for r in con.execute(
        "SELECT chapter_id FROM job WHERE state IN ('running','cancelling') "
        "AND chapter_id IS NOT NULL")}

    for i in drifted:
        src, dst = Path(i["ep_dir"]), Path(i["want_path"])
        if i["chapter_id"] in busy:
            blockers.append(
                f"chapter {i['chapter_id']} ({i['label']}) has a live job — "
                f"refusing to move a directory a running process is using")
            continue
        if dst.exists():
            blockers.append(
                f"{dst} already exists — cannot rename {src.name} onto it "
                f"without deciding which content is correct")
            continue
        actions.append({"chapter_id": i["chapter_id"], "label": i["label"],
                        "number": i["number"], "from": str(src),
                        "to": str(dst)})

    # two actions must never target the same destination
    seen: Dict[str, int] = {}
    for a in actions:
        if a["to"] in seen:
            blockers.append(f"two chapters both want {a['to']}")
        seen[a["to"]] = a["chapter_id"]

    return {"series_id": series_id, "title": audit["title"],
            "actions": actions, "blockers": blockers,
            "ambiguous": ambiguous, "audit": audit}


def apply_plan(con: sqlite3.Connection, plans: List[Dict[str, Any]],
               journal_path: Path, db_path: str = "") -> Dict[str, Any]:
    """Rename + update the DB, journaling first. Returns a summary."""
    actions = [a for p in plans for a in p["actions"]]
    journal = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "db": db_path,
        "actions": actions,
    }
    # journal BEFORE the first mutation — a crash mid-run must still be
    # undoable, which means the record cannot be written at the end
    journal_path.write_text(json.dumps(journal, indent=1))
    os.sync() if hasattr(os, "sync") else None

    done: List[Dict[str, Any]] = []
    for a in actions:
        src, dst = Path(a["from"]), Path(a["to"])
        if not src.is_dir():
            print(f"  skip (gone): {src}")
            continue
        if dst.exists():
            print(f"  skip (target exists): {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)             # same fs, atomic; never a copy
        con.execute("UPDATE chapter SET ep_dir=? WHERE id=?",
                    (str(dst), a["chapter_id"]))
        done.append(a)
        print(f"  {src.name} -> {dst.name}  (chapter {a['chapter_id']}, "
              f"{a['label']})")
    con.commit()
    return {"renamed": len(done), "journal": str(journal_path)}


def undo(con: sqlite3.Connection, journal_path: Path) -> Dict[str, Any]:
    journal = json.loads(Path(journal_path).read_text())
    restored = 0
    for a in reversed(journal.get("actions", [])):
        src, dst = Path(a["from"]), Path(a["to"])
        if not dst.is_dir():
            print(f"  skip (not where the journal says): {dst}")
            continue
        if src.exists():
            print(f"  skip (original path reoccupied): {src}")
            continue
        os.rename(dst, src)
        con.execute("UPDATE chapter SET ep_dir=? WHERE id=?",
                    (str(src), a["chapter_id"]))
        restored += 1
        print(f"  {dst.name} -> {src.name}")
    con.commit()
    return {"restored": restored}


def _report(plans: List[Dict[str, Any]]) -> None:
    for p in plans:
        a = p["audit"]
        if not a["issues"]:
            continue
        print(f"\n[{p['series_id']}] {p['title']} — {a['chapters']} chapters, "
              f"{a['errors']} error / {a['warnings']} warning")
        for i in a["issues"]:
            mark = "!!" if i["severity"] == "error" else " ·"
            print(f"  {mark} {i['code']:22} #{i['number']:<7g} {i['label']!r}")
            print(f"       {i['detail']}")
        if p["actions"]:
            print(f"  -> {len(p['actions'])} directory rename(s) proposed:")
            for act in p["actions"]:
                print(f"       {Path(act['from']).name} -> "
                      f"{Path(act['to']).name}")
        for b in p["blockers"]:
            print(f"  BLOCKED: {b}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=None,
                    help="default: repo-anchored studio.db")
    ap.add_argument("--series", type=int, default=None,
                    help="only this series (default: all)")
    ap.add_argument("--apply", action="store_true",
                    help="actually rename (default is a dry run)")
    ap.add_argument("--journal", default=None,
                    help="where to write the undo journal (required with "
                         "--apply)")
    ap.add_argument("--undo", default=None, metavar="JOURNAL",
                    help="restore a previous --apply run")
    ap.add_argument("--content-verified", action="store_true",
                    help="assert you have LOOKED at the directories involved "
                         "in any ambiguous issue and confirmed their contents")
    args = ap.parse_args(argv)

    from studio import config as studio_config
    from studio.catalog.db import connect
    db = args.db or str(studio_config.REPO_ROOT / "studio.db")
    con = connect(db)

    if args.undo:
        print(f"undoing {args.undo}")
        r = undo(con, Path(args.undo))
        print(f"restored {r['restored']} directory(ies)")
        return 0

    sids = ([args.series] if args.series is not None
            else [r[0] for r in con.execute("SELECT id FROM series ORDER BY id")])
    plans = [plan_series(con, sid) for sid in sids]
    _report(plans)

    total = sum(len(p["actions"]) for p in plans)
    ambiguous = [i for p in plans for i in p["ambiguous"]]
    blockers = [b for p in plans for b in p["blockers"]]

    print(f"\n{total} rename(s) proposed, {len(ambiguous)} ambiguous issue(s), "
          f"{len(blockers)} blocker(s)")

    if not args.apply:
        if total:
            print("dry run — re-run with --apply --journal <path> to perform "
                  "the renames")
        return 0

    if ambiguous and not args.content_verified:
        print("\nREFUSING TO APPLY.")
        print("This catalog has issues that cannot be resolved from metadata:")
        for i in ambiguous[:10]:
            print(f"  - {i['code']} on #{i['number']:g} {i['label']!r}")
        print("\nTwo rows disagree about which chapter is which. Open the "
              "directories,\nlook at the actual pages, and decide. Then "
              "re-run with --content-verified\nto confirm you have done so.",
              file=sys.stderr)
        return 3
    if not args.journal:
        print("--apply requires --journal <path> so the run can be undone",
              file=sys.stderr)
        return 2
    if blockers:
        print("\nREFUSING TO APPLY — unresolved blockers listed above.",
              file=sys.stderr)
        return 4

    print(f"\napplying {total} rename(s), journal -> {args.journal}")
    r = apply_plan(con, plans, Path(args.journal), db_path=db)
    print(f"renamed {r['renamed']}; undo with: "
          f"python -m studio.catalog.repair --db {db} --undo {args.journal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
