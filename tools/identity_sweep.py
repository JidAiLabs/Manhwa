#!/usr/bin/env python3
"""tools/identity_sweep.py — index every chapter of a series, build its image
identity profile, and report what it would change. Runs in .identity_venv.

Per series (ongoing/<slug>):
  * index every chapter (heads + CCIP fingerprints, cached; ~16 s/chapter on
    the Mini CPU, seconds when cached) and rebuild .identity/profile.json
    (continuity with the previous profile; owner exemplars from
    cast/<slug>.json pinned);
  * a grading sheet of random heads the profile names on one-head panels;
  * drift: DRIFT_RUN consecutive chapters where under DRIFT_FRAC of the panels
    with a head are the protagonist (a redesign the references don't follow:
    vague there, not wrong — the owner decides);
  * census, computed in memory (identity.json is written only at prepare
    time, so finished chapters never mix authorities): shipped lines that put
    someone ELSE in the subject of a panel the picture shows the protagonist
    alone on — the error class that called ORV's Dokja "Namwoon Kim".
Across series: the foreign-control rate — the share of OTHER series' heads a
profile names. They are certainly not its protagonist, so a high rate means
the profile matches a look (dark-haired male lead), not a person.

Report only. Re-narrating the census chapters is the owner's call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_identity_ccip as pic  # noqa: E402
from beats_segments import beat_segments  # noqa: E402
from cast_identity import actor_noun_map, subject_actor_nouns_ex  # noqa: E402
from identity_gate import solo_mc_resolver  # noqa: E402

DRIFT_RUN = 5
DRIFT_FRAC = 0.10
FOREIGN_SAMPLE = 2000


def _json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def chapter_dirs(series_dir) -> List[Path]:
    return sorted((p.parent for p in Path(series_dir).glob("*/manifest.panels.understood.json")),
                  key=lambda p: pic.chapter_order(p.name))


def census_chapter(ep: Path, identity: Dict[str, Any]) -> List[Dict[str, Any]]:
    understood = _json(ep / "manifest.panels.understood.json").get("panels") or []
    ubf = {str(u.get("scene_file")): u for u in understood if u.get("scene_file")}
    vision = _json(ep / "manifest.vision.json").get("items") or []
    text = {str(v.get("scene_file")): v.get("ocr_clean") or "" for v in vision
            if v.get("scene_file")}
    solo = solo_mc_resolver(identity, ubf, text)
    noun_map = actor_noun_map(_json(ep / "manifest.cast.json"))
    hits = []
    for beat in _json(ep / "manifest.beats.json").get("beats") or []:
        for seg in beat_segments(beat):
            name, line = solo(seg["span"]) if solo else None, seg["line"] or ""
            if not name or not line:
                continue
            nouns = [n for n, members, plural in subject_actor_nouns_ex(line, noun_map)
                     if members and not plural and name not in members]
            if nouns:
                hits.append({"chapter": ep.name, "group_id": beat.get("group_id"),
                             "span": list(seg["span"]), "line": line, "nouns": nouns})
    return hits


def sweep_series(series_dir, *, series_cast=None, heads_fn=None, embed_fn=None,
                 sheet_path=None) -> Dict[str, Any]:
    sd = Path(series_dir)
    eps = chapter_dirs(sd)
    indexes = {}
    for k, ep in enumerate(eps, 1):
        indexes[ep.name] = pic.index_chapter(ep, heads_fn=heads_fn, embed_fn=embed_fn)
        if k % 20 == 0:
            print(f"[sweep] {sd.name}: indexed {k}/{len(eps)}", flush=True)
    with pic.profile_lock(sd):
        pins = (pic.pin_features(series_cast, heads_fn=heads_fn, embed_fn=embed_fn)
                if series_cast else None)
        prof = pic.build_profile(pic.load_indexes(sd), pins=pins, prior=pic.load_profile(sd))
        pic.save_profile(sd, prof)
    rep: Dict[str, Any] = {
        "series": sd.name, "chapters": len(eps),
        "profile": {k: prof.get(k) for k in ("status", "reasons", "alarms", "version",
                                             "stats", "cut")},
        "refs": [f"{r['chapter']}/{r.get('panel', '')}" for r in prof.get("refs") or []],
        "confirmed_by_chapter": {}, "drift": [], "census": []}
    if prof.get("status") == "active":
        run: List[str] = []
        for ep in eps:
            ident = pic.identify_chapter(ep, prof, indexes[ep.name], write=False) or {}
            P = ident.get("panels") or {}
            frac = sum(1 for v in P.values() if v["mc"]) / len(P) if P else None
            rep["confirmed_by_chapter"][ep.name] = None if frac is None else round(frac, 3)
            if frac is not None and frac < DRIFT_FRAC:
                run.append(ep.name)
            else:
                if len(run) >= DRIFT_RUN:
                    rep["drift"].append([run[0], run[-1]])
                run = []
            rep["census"] += census_chapter(ep, ident)
        if len(run) >= DRIFT_RUN:
            rep["drift"].append([run[0], run[-1]])
    rep["census_chapters"] = len({h["chapter"] for h in rep["census"]})
    if sheet_path:
        rep["sheet_heads"] = pic.contact_sheet(sd, sheet_path)
    return rep


def foreign_rates(series_dirs) -> Dict[str, Optional[float]]:
    """Share of OTHER series' heads each active profile names (sampled)."""
    feats = {Path(s).name: [ix["feats"] for ix in pic.load_indexes(s) if len(ix["heads"])]
             for s in series_dirs}
    rng = np.random.default_rng(0)
    out: Dict[str, Optional[float]] = {}
    for s in series_dirs:
        name, prof = Path(s).name, pic.load_profile(s)
        other = [f for k, v in feats.items() if k != name for f in v]
        if not prof or prof.get("status") != "active" or not other:
            out[name] = None
            continue
        F = np.concatenate(other)
        if len(F) > FOREIGN_SAMPLE:
            F = F[rng.choice(len(F), FOREIGN_SAMPLE, replace=False)]
        d = pic.ccip_diff(F, pic._feats(prof["refs"])).min(axis=1)
        out[name] = round(float((d < prof["cut"]).mean()), 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--series", action="append", required=True,
                    help="ongoing/<slug> (repeatable)")
    ap.add_argument("--out", default="dist/identity_sweep")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for s in args.series:
        slug = Path(s).name
        reg = pic.REPO_ROOT / "cast" / f"{slug}.json"
        rep = sweep_series(s, series_cast=reg if reg.exists() else None,
                           sheet_path=out / f"{slug}_sheet.jpg")
        reports.append(rep)
        (out / f"{slug}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    rates = foreign_rates(args.series)
    for rep in reports:
        p = rep["profile"]
        print(f"[sweep] {rep['series']}: {rep['chapters']} chapters, profile v{p['version']} "
              f"{p['status']} {p.get('reasons') or ''}{p.get('alarms') or ''} | matched "
              f"{(p.get('stats') or {}).get('matched')} | foreign {rates.get(rep['series'])} | "
              f"drift {rep['drift']} | census {len(rep['census'])} lines in "
              f"{rep['census_chapters']} chapters | sheet {rep.get('sheet_heads')} heads")
    (out / "foreign_rates.json").write_text(json.dumps(rates, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
