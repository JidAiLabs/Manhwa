#!/usr/bin/env python3
"""
qa_gate_blast_radius.py — measure a QA gate against the real corpus BEFORE
making it blocking.

    python tools/qa_gate_blast_radius.py                  # every beats gate
    python tools/qa_gate_blast_radius.py --code garbled_line --show 5

A code that is an ERROR, in narration_heal.HEALABLE and in
worker._CRITICAL_QA_CODES makes the worker re-narrate the flagged groups up to
_HEAL_MAX cycles and then, if the flag survives, BLOCK the chapter. A FALSE
POSITIVE can never be healed away — the line was already correct — so it parks
that chapter forever and burns heal cycles getting there.

This has bitten three times: an 11-of-12 false-positive rate in the 2026-08-18
actor/caption/impact audit, and on 2026-09-06 a first cut of garbled_line that
flagged 60 lines across 135 chapters (43 chapters — 32% of the fleet), every
one ordinary English. Run this on the Mini, where the prepared chapters live,
and read the PARK column before promoting a code to _CRITICAL_QA_CODES.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_prep_qa():
    sys.path.insert(0, os.path.join(REPO, "tools"))
    spec = importlib.util.spec_from_file_location(
        "prep_qa", os.path.join(REPO, "tools", "prep_qa.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="", help="only this QA code")
    ap.add_argument("--show", type=int, default=3,
                    help="example lines to print per code (0 = none)")
    ap.add_argument("--glob", default="ongoing/*/*/manifest.beats.json")
    args = ap.parse_args()

    pq = _load_prep_qa()
    sys.path.insert(0, REPO)
    from tools.narration_heal import HEALABLE
    from studio.worker import _CRITICAL_QA_CODES

    # every beats-only detector prep_qa exposes (they take beats_obj alone)
    detectors = [getattr(pq, n) for n in dir(pq)
                 if n.endswith("_flags") and callable(getattr(pq, n))]

    chapters = sorted(glob.glob(os.path.join(REPO, args.glob)))
    per_code_lines = defaultdict(int)
    per_code_chaps = defaultdict(set)
    examples = defaultdict(list)
    scanned = 0
    for f in chapters:
        try:
            beats = json.load(open(f))
        except Exception:
            continue
        scanned += 1
        name = f.split(os.sep)[-2]
        for det in detectors:
            try:
                flags = det(beats)
            except Exception:
                continue                      # detector needs more than beats
            for fl in flags or []:
                code = fl.get("code", "?")
                if args.code and code != args.code:
                    continue
                per_code_lines[code] += 1
                per_code_chaps[code].add(name)
                if len(examples[code]) < args.show:
                    examples[code].append(f"{name} {fl.get('segment_id','')}: "
                                          f"{fl.get('detail','')[:110]}")
    if not scanned:
        print(f"no chapters matched {args.glob!r} — run this on the host that "
              "holds ongoing/ (the Mini)")
        return 1

    print(f"corpus: {scanned} chapters with beats\n")
    print(f"{'code':28s} {'lines':>6s} {'chapters':>9s} {'PARK%':>6s}  heal  block")
    for code in sorted(per_code_chaps, key=lambda c: -len(per_code_chaps[c])):
        n = len(per_code_chaps[code])
        print(f"{code:28s} {per_code_lines[code]:6d} {n:9d} "
              f"{n / scanned * 100:5.0f}%  "
              f"{'yes' if code in HEALABLE else ' - ':>4s}  "
              f"{'YES' if code in _CRITICAL_QA_CODES else ' - ':>5s}")
        for ex in examples[code]:
            print(f"    · {ex}")
    print("\nPARK% = share of the fleet this code would stop if it is BLOCKING "
          "and the heal cannot clear it. Confirm every hit is a real defect "
          "before adding a code to _CRITICAL_QA_CODES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
