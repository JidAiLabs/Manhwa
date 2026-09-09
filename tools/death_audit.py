#!/usr/bin/env python3
"""tools/death_audit.py — does the fact record actually know who died, and where?

Measured on the Mini before the fix (207 chapters with a story + a ledger):
36 cast fates said "killed", the ledger anchored 7. The other 29 deaths never
reached anything — the anchor came from regex-reading the killing event's
English, and "is stabbed in the stomach", "Sends Namwoon to hell", "eradicate
the Beast Lord" are not in any verb list. dead_actor and role_stale were
mostly inert, and the two anchors that DID land sat on the caption that merely
announces the death.

Run it before and after a refresh-facts sweep. Read the anchored/killed ratio
and the anchor-source histogram:
  last_act — the chapter has the victim ACTING, and the death sits on their
             last such panel. This is the good case.
  named    — the chapter only ever mentions them (as someone else's target),
             so the anchor is that mention. Weaker; check it.
A killed fate with no anchor at all is listed too: the story never placed the
character on a panel, so the death deliberately does not propagate.

  python tools/death_audit.py                     # every title under ongoing/
  python tools/death_audit.py --series omniscient-reader --chapters 100-200
  python tools/death_audit.py --contradictions    # + what the SHIPPED
                                                  #   narration now violates
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from cast_identity import resolve_name  # noqa: E402
from story_ledger import entity_profiles, is_completed_death  # noqa: E402

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*$")


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def chapter_number(ep_dir: str) -> Optional[float]:
    m = _NUM_RE.search(os.path.basename(ep_dir).replace("-", "_"))
    return float(m.group(1)) if m else None


def audit_chapter(ep_dir: str, *, contradictions: bool = False) -> Dict[str, Any]:
    """One chapter's death record. {} when it has no story+ledger pair."""
    story = _load(os.path.join(ep_dir, "manifest.chapter_story.json"))
    led = _load(os.path.join(ep_dir, "manifest.ledger.json"))
    if not story or not led:
        return {}
    profiles = entity_profiles(led.get("entities") or [])
    deaths = {str(e.get("subject")): e for e in (led.get("events") or [])
              if e.get("type") == "death"}
    rows: List[Dict[str, Any]] = []
    for c in (story.get("cast") or []):
        if not isinstance(c, dict):
            continue
        fate = str(c.get("fate") or "")
        if not is_completed_death(fate):
            continue
        name = str(c.get("name") or "")
        who, _e = resolve_name(name, profiles)
        ev = deaths.get(who) if who != "unknown" else None
        rows.append({
            "name": name, "fate": fate, "resolved": who,
            "anchored_at": str(ev.get("scene_file")) if ev else "",
            "anchor_source": (str(ev.get("anchor_source") or "event")
                              if ev else ""),
        })
    out = {
        "ep_dir": ep_dir,
        "number": chapter_number(ep_dir),
        "prompt_version": str((story.get("_meta") or {}).get("prompt_version")
                              or story.get("prompt_version") or "?"),
        "killed": rows,
    }
    if contradictions:
        out["contradictions"] = _contradictions(ep_dir, led)
    return out


def _contradictions(ep_dir: str, led: Dict[str, Any]) -> List[str]:
    """What the chapter's CURRENT narration violates under these facts — the
    same gate the prepare runs, no model call. A non-empty list on a shipped
    chapter means the video says something the record now denies."""
    beats = _load(os.path.join(ep_dir, "manifest.beats.json"))
    cast = _load(os.path.join(ep_dir, "manifest.cast.json"))
    if not beats or not cast:
        return []
    from prep_qa import ledger_contradiction_flags
    return [str(f.get("code")) for f in
            ledger_contradiction_flags(beats, led, cast)]


def _fmt(rec: Dict[str, Any]) -> str:
    """One line per killed character that is NOT cleanly anchored. 'named'
    means the chapter never has them ACT — only mentions them — so the anchor
    is the weaker of the two signals."""
    lines = []
    for r in rec["killed"]:
        if r["anchor_source"] == "last_act":
            continue
        why = ("matches no entity" if r["resolved"] == "unknown"
               else "NOT anchored — the story never places them on a panel"
               if not r["anchored_at"]
               else f"anchored at {r['anchored_at']} by mention only")
        lines.append(f"    {r['name']!r}: {why} fate={r['fate'][:60]!r}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ongoing")
    ap.add_argument("--series", default="", help="slug filter (substring)")
    ap.add_argument("--chapters", default="",
                    help="number filter: N or N-M (by chapter number)")
    ap.add_argument("--contradictions", action="store_true",
                    help="also run prep_qa's dead_actor/role_stale gate over "
                         "each chapter's CURRENT narration (no model call)")
    ap.add_argument("--verbose", action="store_true",
                    help="list every unanchored death, not just the counts")
    args = ap.parse_args()

    lo = hi = None
    if args.chapters:
        parts = args.chapters.split("-")
        lo = float(parts[0])
        hi = float(parts[-1])

    by_title: Dict[str, Dict[str, Any]] = {}
    bad: List[Dict[str, Any]] = []
    for ep in sorted(glob.glob(os.path.join(args.root, "*", "*"))):
        if not os.path.isdir(ep):
            continue
        title = os.path.basename(os.path.dirname(ep))
        if args.series and args.series not in title:
            continue
        n = chapter_number(ep)
        if lo is not None and (n is None or not lo <= n <= hi):
            continue
        rec = audit_chapter(ep, contradictions=args.contradictions)
        if not rec:
            continue
        t = by_title.setdefault(title, {
            "chapters": 0, "killed": 0, "anchored": 0, "unresolved": 0,
            "sources": Counter(), "versions": Counter(),
            "contradicted": []})
        t["chapters"] += 1
        t["versions"][rec["prompt_version"]] += 1
        for r in rec["killed"]:
            t["killed"] += 1
            if r["resolved"] == "unknown":
                t["unresolved"] += 1
            if r["anchored_at"]:
                t["anchored"] += 1
                t["sources"][r["anchor_source"]] += 1
        if rec.get("contradictions"):
            t["contradicted"].append(
                (os.path.basename(ep), Counter(rec["contradictions"])))
        if any(r["anchor_source"] != "last_act" for r in rec["killed"]):
            bad.append(rec)

    for title, t in sorted(by_title.items()):
        ratio = (t["anchored"] / t["killed"]) if t["killed"] else 1.0
        print(f"{title}: {t['chapters']} chapter(s), {t['killed']} killed, "
              f"{t['anchored']} anchored ({ratio:.0%})")
        print(f"  anchor source: {dict(t['sources']) or '{}'}"
              f"  matches-no-entity={t['unresolved']}")
        print(f"  story prompt_version: {dict(t['versions'])}")
        for ep_name, codes in t["contradicted"]:
            print(f"  CONTRADICTED {ep_name}: {dict(codes)} — the shipped "
                  "narration violates these facts")
    if args.verbose:
        print("\n=== chapters with a death the ledger did not get from the "
              "story ===")
        for rec in bad:
            body = _fmt(rec)
            if body:
                print(f"  {rec['ep_dir']} [{rec['prompt_version']}]")
                print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
