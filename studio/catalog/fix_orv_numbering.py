"""One-off: realign Omniscient Reader's chapter numbers to episode_no.

WHY THIS EXISTS
---------------
Before the webtoon label fix, ORV's first rows were built inconsistently:

  id=1  number=0  "Episode 0 (Prologue)"  url=…/episode-0-prologue  <- prologue content
  id=2  number=1  "Episode 1"             url=…/episode-0-prologue  <- EPISODE 1 content
  id=3  number=2  "Episode 2"             url=…/episode-1           <- empty

id=2's url was overwritten by a later re-discovery, so it points at the
prologue while its directory holds episode 1. Meanwhile the adapter now emits
number=episode_no, where the prologue is number=1 and episode 1 is number=2 —
so nothing matches number=0, and a refresh would pair the prologue's LABEL
with episode 1's CONTENT.

THE FIX
-------
Rather than remap chapter_id across stage_run/approval/job/bundle_chapter
(error-prone, and it would detach 52 stage_run rows from their work), each
ROW simply takes the number it should have had. Row identity — and therefore
all of its history and artifacts — is untouched.

  1. delete id=3        (number 2; provably empty, frees the number)
  2. id=2 -> number 2   (holds episode-1 content)
  3. id=1 -> number 1   (holds prologue content)

Order matters: UNIQUE(series_id, number) means the target number must be free
before each update.

Afterwards `studio refresh --series <id>` relabels from the slug. The episode
directories were already renamed to Episode_0_Prologue / Episode_1, so both
rows land on their canonical directory with zero drift.

Dry run by default. Every precondition is asserted, not assumed: if the
catalog is not in the exact expected shape, nothing is written.

    python -m studio.catalog.fix_orv_numbering --series 1
    python -m studio.catalog.fix_orv_numbering --series 1 --apply
"""

from __future__ import annotations

import argparse
import datetime
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# (row id, current number, target number, expected ep_dir basename or None)
Plan = List[Tuple[int, float, Optional[float], Optional[str]]]

_CHILD_TABLES = ("stage_run", "approval", "job", "bundle_chapter")


def build_plan(con: sqlite3.Connection, series_id: int) -> Tuple[Plan, List[str]]:
    """Return (plan, problems). A non-empty problems list means DO NOT apply."""
    problems: List[str] = []
    rows = con.execute(
        "SELECT id, number, label, url, ep_dir FROM chapter "
        "WHERE series_id=? ORDER BY number LIMIT 3", (series_id,)).fetchall()

    # A chapter numbered 0 is the unambiguous signature of the pre-fix shape:
    # the adapter numbers from episode_no, which starts at 1, so number 0 can
    # only exist on a catalog written before the label fix. Keying on this
    # (rather than on row count) makes the refusal message honest whether the
    # migration already ran or the series simply isn't the expected one.
    numbers = [r[1] for r in rows]
    if 0.0 not in numbers:
        return [], [
            f"series {series_id} has no chapter numbered 0 (first numbers: "
            f"{', '.join(f'{n:g}' for n in numbers) or 'none'}). This "
            f"migration only applies to the pre-fix shape — it has already "
            f"been applied, or this is not the affected series."]
    if len(rows) < 3:
        return [], [f"series {series_id} has fewer than 3 chapters — cannot be "
                    f"the affected catalog"]

    (id0, n0, _l0, _u0, d0), (id1, n1, _l1, _u1, d1), (id2, n2, _l2, _u2, d2) = rows

    if (n0, n1, n2) != (0.0, 1.0, 2.0):
        problems.append(
            f"expected the first three chapters to be numbered 0,1,2 — found "
            f"{n0:g},{n1:g},{n2:g}; the catalog is not in the expected shape")
        return [], problems

    if not (d0 or "").endswith("Episode_0_Prologue"):
        problems.append(f"chapter {id0} (number 0) should hold the prologue "
                        f"directory Episode_0_Prologue, found {d0!r}")
    if not (d1 or "").endswith("Episode_1"):
        problems.append(f"chapter {id1} (number 1) should hold the episode-1 "
                        f"directory Episode_1, found {d1!r}")
    if d2 is not None:
        problems.append(f"chapter {id2} (number 2) is expected to be empty but "
                        f"has ep_dir={d2!r} — refusing to delete it")

    # number 2's row is deleted, so it must carry no work whatsoever
    for t in _CHILD_TABLES:
        n = con.execute(f"SELECT COUNT(*) FROM {t} WHERE chapter_id=?",
                        (id2,)).fetchone()[0]
        if n:
            problems.append(f"chapter {id2} (number 2) has {n} {t} row(s) — "
                            f"refusing to delete a chapter that has history")

    live = con.execute(
        "SELECT COUNT(*) FROM job WHERE state IN ('running','cancelling') "
        "AND type!='heartbeat'").fetchone()[0]
    if live:
        problems.append(f"{live} job(s) are live — run this with the worker idle")

    plan: Plan = [
        (id2, 2.0, None, None),                        # delete
        (id1, 1.0, 2.0, "Episode_1"),                  # episode 1
        (id0, 0.0, 1.0, "Episode_0_Prologue"),         # prologue
    ]
    return plan, problems


def apply_plan(con: sqlite3.Connection, plan: Plan) -> None:
    """Execute in the given order — each target number must be free first."""
    for cid, _cur, target, _dir in plan:
        if target is None:
            con.execute("DELETE FROM chapter WHERE id=?", (cid,))
        else:
            con.execute("UPDATE chapter SET number=? WHERE id=?", (target, cid))
    con.commit()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=None)
    ap.add_argument("--series", type=int, required=True)
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default is a dry run)")
    args = ap.parse_args(argv)

    from studio import config as studio_config
    from studio.catalog.db import connect
    db = args.db or str(studio_config.REPO_ROOT / "studio.db")
    con = connect(db)

    plan, problems = build_plan(con, args.series)
    print(f"catalog: {db}\n")
    for cid, cur, target, want_dir in plan:
        if target is None:
            print(f"  DELETE chapter id={cid} (number {cur:g}, empty)")
        else:
            print(f"  chapter id={cid}: number {cur:g} -> {target:g}"
                  f"   [{want_dir}]")
    for p in problems:
        print(f"  PROBLEM: {p}")

    if problems:
        print("\nREFUSING — the catalog is not in the expected shape.",
              file=sys.stderr)
        return 2
    if not args.apply:
        print("\ndry run — re-run with --apply to write")
        return 0

    backup = f"{db}.bak-orvnum-" \
             f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    con.execute("VACUUM INTO ?", (backup,))
    print(f"\nbackup: {backup}")
    apply_plan(con, plan)
    print("applied. Now run:  studio refresh --series "
          f"{args.series}   (relabels from the slug)")
    for r in con.execute(
            "SELECT id, number, label, ep_dir FROM chapter WHERE series_id=? "
            "ORDER BY number LIMIT 4", (args.series,)):
        base = Path(r[3]).name if r[3] else None
        print(f"  id={r[0]} number={r[1]:g} {r[2]!r} dir={base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
