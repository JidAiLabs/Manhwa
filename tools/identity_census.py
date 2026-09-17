#!/usr/bin/env python3
"""
identity_census.py — which finished chapters NAME the wrong character?

The narration names people; `manifest.identity.json` says who the IMAGE shows.
Comparing the two answers the owner's question ("how many chapters have this
issue?") without re-narrating anything.

Two counts per chapter, deliberately separate:

  contradicted  a line names X, and the image pass confirms someone ELSE on
                every panel that line covers, and never X. This is the real
                signal — ORV Ep6's "Namwoon Kim" over Dokja's panels.
  unconfirmed   a line names X and the image pass confirmed NOBODY there. The
                pass recognises ~65% of a lead's panels, so this is mostly its
                own blind spots, not evidence of a wrong name. Reported apart
                so it can never inflate the headline.

Runs the identity pass for a chapter that has none (that manifest is a normal
pipeline artifact, so the census also pre-computes what a later re-narration
would need). ~4.2 s per panel with people.

Usage:
  python tools/identity_census.py --series ongoing/omniscient-reader \
      --series-cast cast/omniscient-reader.json --out dist/identity_census.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)


def name_variants(member: Dict[str, Any]) -> List[str]:
    """Every string that NAMES this character in narration prose."""
    out = [str(member.get("canonical_name") or "").strip()]
    out += [str(a).strip() for a in (member.get("aliases") or [])]
    return [v for v in out if v]


def names_in_line(line: str, by_name: Dict[str, List[str]]) -> List[str]:
    """Canonical names the line mentions (word-bounded, case-insensitive)."""
    text = str(line or "")
    hits = []
    for canonical, variants in by_name.items():
        for v in variants:
            if re.search(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(v), text,
                         re.IGNORECASE):
                hits.append(canonical)
                break
    return hits


def audit_chapter(beats_obj: Any, identity: Dict[str, Any],
                  registry: Any) -> Dict[str, Any]:
    """{contradicted, unconfirmed, lines:[...]} for ONE chapter. Pure."""
    from beats_segments import beat_segments
    members = registry.get("cast") if isinstance(registry, dict) else registry
    by_name = {str(m.get("canonical_name") or "").strip(): name_variants(m)
               for m in (members or []) if isinstance(m, dict)
               and str(m.get("canonical_name") or "").strip()}
    panels = (identity or {}).get("panels") or identity or {}
    out: List[Dict[str, Any]] = []
    contradicted = unconfirmed = 0
    for beat in ((beats_obj or {}).get("beats") or []):
        for seg in beat_segments(beat):
            line = str(seg.get("line") or "")
            named = names_in_line(line, by_name)
            if not named:
                continue
            span = [os.path.basename(str(x)) for x in (seg.get("span") or [])]
            confirmed: set = set()
            for fn in span:
                confirmed |= set((panels.get(fn) or {}).get("names") or [])
            for n in named:
                if n in confirmed:
                    continue
                kind = "contradicted" if confirmed else "unconfirmed"
                if kind == "contradicted":
                    contradicted += 1
                else:
                    unconfirmed += 1
                out.append({"kind": kind, "named": n,
                            "confirmed": sorted(confirmed), "span": span,
                            "line": line[:160]})
    return {"contradicted": contradicted, "unconfirmed": unconfirmed,
            "lines": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, help="ongoing/<slug>")
    ap.add_argument("--series-cast", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N chapters")
    ap.add_argument("--shard", default="1/1",
                    help="i/N — this process takes every Nth chapter starting "
                         "at i. Four shards fit the Mini's ollama parallelism, "
                         "turning ~21 h serial into ~5 h.")
    ap.add_argument("--model", default="")
    args = ap.parse_args()
    import panel_identity as pi
    with open(args.series_cast, encoding="utf-8") as f:
        registry = json.load(f)
    if not pi.candidates(registry):
        print("[census] %s has no exemplars — nothing to check against"
              % args.series_cast)
        return 1
    eps = sorted(glob.glob(os.path.join(args.series, "*")),
                 key=lambda p: (len(p), p))
    eps = [e for e in eps
           if os.path.exists(os.path.join(e, "manifest.beats.json"))]
    if args.limit:
        eps = eps[:args.limit]
    try:
        si, sn = (int(x) for x in str(args.shard).split("/"))
    except ValueError:
        ap.error("--shard must look like 2/4")
    if not 1 <= si <= sn:
        ap.error("--shard i must be between 1 and N")
    eps = eps[si - 1::sn]
    report: Dict[str, Any] = {"series": args.series, "shard": args.shard,
                              "chapters": {}}
    for i, ep in enumerate(eps, 1):
        ident_path = os.path.join(ep, "manifest.identity.json")
        if not os.path.exists(ident_path):
            kw = {"model": args.model} if args.model else {}
            pi.identify_panels(ep, registry, **kw)
        try:
            with open(ident_path, encoding="utf-8") as f:
                identity = json.load(f)
            with open(os.path.join(ep, "manifest.beats.json"),
                      encoding="utf-8") as f:
                beats = json.load(f)
        except (OSError, ValueError) as e:
            print("[census] %s: %s" % (os.path.basename(ep), type(e).__name__))
            continue
        got = audit_chapter(beats, identity, registry)
        report["chapters"][os.path.basename(ep)] = got
        print("[%d/%d] %-24s contradicted %2d  unconfirmed %2d"
              % (i, len(eps), os.path.basename(ep), got["contradicted"],
                 got["unconfirmed"]), flush=True)
        if args.out:                      # written as it goes: resumable read
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)
    bad = [c for c, v in report["chapters"].items() if v["contradicted"]]
    print("\n== %d chapters checked | %d name a character the image pass "
          "contradicts" % (len(report["chapters"]), len(bad)))
    for c in bad[:25]:
        print("   %-24s %d" % (c, report["chapters"][c]["contradicted"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
