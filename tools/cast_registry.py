#!/usr/bin/env python3
"""
cast_registry.py — seed a series' locked cast from its chapters' guesses.

    python tools/cast_registry.py ongoing/<slug>            # writes cast/<slug>.json (git-tracked)
    python tools/cast_registry.py ongoing/<slug> --stdout   # print, write nothing

cast_builder re-guesses every character's look per chapter, so across a
long series one character drifts through several descriptions and two
characters can end up with the same one (ORV Ep128: Dokja and Michio both
"dark hair, glasses, black shirt, dark jacket"). This aggregates every
`manifest.cast.json` under the series dir per character — most common
description, every description seen with counts, chapters — into a DRAFT
the owner edits by hand. Nothing here invents canon: `_seen_descriptions`
and `_chapters` are for the owner's eyes and never reach a chapter cast
(cast_builder.apply_series_cast copies only the known fields).

Owner rules for the file:
  * write looks POSITIVELY — "no glasses" would put `glasses` INTO the
    appearance evidence; exclusions go in `not: ["glasses"]`
  * the protagonist keeps canonical_name "our protagonist" (the narration's
    relaxed handle) with the real name in aliases
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def _members(cast):
    cast = cast.get("cast") if isinstance(cast, dict) else cast
    return [m for m in (cast or []) if isinstance(m, dict)]


def build_registry(series_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(series_dir, "*", "manifest.cast.json")))
    seen = defaultdict(lambda: {"aliases": set(), "descriptions": Counter(),
                                "chapters": [], "is_protagonist": False,
                                "names": Counter()})
    for f in files:
        chapter = os.path.basename(os.path.dirname(f))
        try:
            with open(f, encoding="utf-8") as fh:
                members = _members(json.load(fh))
        except (OSError, ValueError):
            continue
        for m in members:
            name = str(m.get("canonical_name") or "").strip()
            if not name:
                continue
            key = "__protagonist__" if m.get("is_protagonist") else name.lower()
            rec = seen[key]
            rec["names"][name] += 1
            rec["is_protagonist"] = rec["is_protagonist"] or bool(m.get("is_protagonist"))
            rec["aliases"].update(str(a) for a in (m.get("aliases") or []) if a)
            desc = str(m.get("visual_description") or "").strip()
            if desc:
                rec["descriptions"][desc] += 1
            rec["chapters"].append(chapter)
    # Fold name variants on CANONICAL-name tokens only ("Huiwon" ⊆ "Huiwon
    # Jeong", "Junghyeok" ⊆ "Junghyeok Yu"). Never on aliases: across ORV's
    # 130 chapters cast_builder hung OTHER characters' names on the
    # protagonist as aliases (Michio Shoji, Junghyeok Yu, Huiwon Jeong…), so
    # alias intersection would fold the whole cast into one entry.
    def _ntoks(name):
        return {t for t in name.lower().replace(".", " ").split() if len(t) > 1}
    recs = list(seen.values())
    merged = True
    while merged:
        merged = False
        for i in range(len(recs)):
            if recs[i]["is_protagonist"]:
                continue
            ti = _ntoks(recs[i]["names"].most_common(1)[0][0])
            for j in range(i + 1, len(recs)):
                if recs[j]["is_protagonist"]:
                    continue
                tj = _ntoks(recs[j]["names"].most_common(1)[0][0])
                if ti and tj and (ti <= tj or tj <= ti):
                    a, b = recs[i], recs.pop(j)
                    a["names"].update(b["names"])
                    a["aliases"] |= b["aliases"] | set(b["names"])
                    a["descriptions"].update(b["descriptions"])
                    a["chapters"] += b["chapters"]
                    merged = True
                    break
            if merged:
                break
    # aliases: keep name-like ones that are not some OTHER character's name
    # (same given name = same person in these romanisations: "Gilyeong Lee"
    # is Gilyeong's, not the protagonist's, however cast_builder filed it)
    canon = [r["names"].most_common(1)[0][0].lower() for r in recs]
    for rec in recs:
        me = rec["names"].most_common(1)[0][0].lower()
        keep = set()
        for a in rec["aliases"]:
            al = a.lower()
            first = (al.split() or [""])[0]
            others = [c for c in canon if c != me]
            if (not a[:1].isupper() or "(" in a or len(a.split()) > 3
                    or al == me or al in others
                    or any(first == (c.split() or [""])[0] for c in others)):
                continue
            keep.add(a)
        rec["aliases"] = keep
    cast = []
    for rec in sorted(recs, key=lambda r: (-len(r["chapters"]), r["names"].most_common(1)[0][0])):
        top = rec["descriptions"].most_common(5)
        name = rec["names"].most_common(1)[0][0]
        cast.append({
            "canonical_name": name,
            "aliases": sorted(a for a in rec["aliases"] if a.lower() != name.lower()),
            "visual_description": top[0][0] if top else "",
            "not": [],
            "is_protagonist": rec["is_protagonist"],
            "_seen_descriptions": [{"n": n, "text": t} for t, n in top],
            "_chapters": len(rec["chapters"]),
        })
    return {"series": os.path.basename(os.path.normpath(series_dir)),
            "_source_chapters": len(files), "cast": cast}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("series_dir", help="ongoing/<slug>")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    ap.add_argument("--min-chapters", type=int, default=2,
                    help="drop one-off characters seen in fewer chapters")
    args = ap.parse_args()
    reg = build_registry(args.series_dir)
    reg["cast"] = [c for c in reg["cast"] if c["_chapters"] >= args.min_chapters]
    text = json.dumps(reg, ensure_ascii=False, indent=2)
    if args.stdout:
        print(text)
        return 0
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    slug = os.path.basename(os.path.normpath(args.series_dir))
    out = os.path.join(repo, "cast", f"{slug}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out} (the owner edits this by hand)")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"[ok] draft written: {out} — {len(reg['cast'])} characters from "
          f"{reg['_source_chapters']} chapters; edit looks, fill `not`, then re-prepare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
