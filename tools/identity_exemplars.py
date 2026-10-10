#!/usr/bin/env python3
"""tools/identity_exemplars.py — propose the exemplar panels gemma confirms the
protagonist against, then write the chosen 2 + 2 to cast/<slug>.exemplars.json.

The picture profile (panel_identity_ccip, .identity caches from the sweep)
proposes; a person picks (owner decision 2026-10-10 — auto-picked exemplars made
gemma answer "protagonist" 60/60 when one was a tiny far figure, and ~85% even
when they were right). Candidates are CLOSE-UPS only: the head is >= 6% of the
panel and the panel is not a tall strip, so the face survives gemma's 640 px.

  propose  M1..M12: the protagonist's closest close-ups, distinct chapters.
           G1..G3 a-d: the three densest groups of near-miss look-alikes
           (0.16-0.30 from every reference, tight groups < 0.08) — the decoy
           that keeps look-alikes off the protagonist must be ONE character.
  pick     --pick M1,M4 --decoy G1a,G1c -> cast/<slug>.exemplars.json

Runs in .eval_venv (numpy + PIL); no model calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_identity_ccip as pic  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NEAR = (0.16, 0.30)
GROUP_TIGHT = 0.08


def _closeup(series: Path, ch: str, head: Dict[str, Any]) -> bool:
    from PIL import Image
    x0, y0, x1, y1 = head["box"]
    with Image.open(series / ch / "scenes" / head["panel"]) as im:
        w, h = im.size
    return (x1 - x0) * (y1 - y0) / float(w * h) >= 0.06 and h / float(w) <= 2.5


def propose(series_dir) -> Dict[str, Dict[str, Any]]:
    sd = Path(series_dir)
    prof = pic.load_profile(sd)
    if not prof or prof.get("status") != "active":
        raise SystemExit(f"{sd.name}: no active profile (run tools/identity_sweep.py)")
    R = pic._feats(prof["refs"])
    rows, feats = [], []
    for ix in pic.load_indexes(sd):
        n = collections.Counter(h["panel"] for h in ix["heads"])
        for i, h in enumerate(ix["heads"]):
            if n[h["panel"]] == 1:                       # one-head panels only
                rows.append((ix["chapter"], h))
                feats.append(ix["feats"][i])
    F = np.asarray(feats, dtype=np.float32)
    d = pic.ccip_diff(F, R).min(axis=1)
    out: Dict[str, Dict[str, Any]] = {}
    used = set()
    for j in np.argsort(d):
        ch, h = rows[j]
        if ch not in used and _closeup(sd, ch, h):
            used.add(ch)
            out[f"M{len(used)}"] = {"chapter": ch, "panel": h["panel"], "box": h["box"],
                                   "d": round(float(d[j]), 3)}
        if len(used) == 12:
            break
    near = np.where((d >= NEAR[0]) & (d < NEAR[1]))[0]
    if len(near) > 3000:
        near = np.random.default_rng(0).choice(near, 3000, replace=False)
    D = pic.ccip_diff(F[near], F[near])
    np.fill_diagonal(D, np.inf)
    taken = np.zeros(len(near), bool)
    for g in range(1, 4 if len(near) else 1):        # no look-alikes: no groups
        dens = np.where(taken, -1, (D < 0.12).sum(axis=1))
        s = int(dens.argmax())
        if dens[s] <= 0:
            break
        group = [s] + [int(k) for k in np.argsort(D[s]) if D[s, k] < GROUP_TIGHT]
        taken[[k for k in np.argsort(D[s]) if D[s, k] < 0.12] + [s]] = True
        picks, chs = [], set()
        for k in group:
            ch, h = rows[near[k]]
            if ch not in chs and _closeup(sd, ch, h):
                chs.add(ch)
                picks.append((ch, h, float(d[near[k]])))
            if len(picks) == 4:
                break
        for letter, (ch, h, dd) in zip("abcd", picks):
            out[f"G{g}{letter}"] = {"chapter": ch, "panel": h["panel"], "box": h["box"],
                                    "d": round(dd, 3)}
    return out


def sheet(series_dir, props: Dict[str, Dict[str, Any]], out_path) -> None:
    from PIL import Image, ImageDraw
    sd = Path(series_dir)
    size, per_row = 200, 6
    keys = list(props)
    rows = (len(keys) + per_row - 1) // per_row
    cv = Image.new("RGB", (per_row * (size + 8) + 8, 30 + rows * (size + 22)), "white")
    dr = ImageDraw.Draw(cv)
    dr.text((6, 8), f"{sd.name}: M = protagonist candidates, G1-G3 = look-alike groups "
            "(pick 2 M + 2 from ONE G)", fill="blue")
    for k, key in enumerate(keys):
        p = props[key]
        x, y = 6 + (k % per_row) * (size + 8), 30 + (k // per_row) * (size + 22)
        im = pic._open(str(sd / p["chapter"] / "scenes" / p["panel"]))
        draw = ImageDraw.Draw(im)
        draw.rectangle(p["box"], outline="red", width=max(3, im.width // 150))
        im.thumbnail((size, size))
        cv.paste(im, (x, y))
        dr.text((x, y + size + 3), f"{key} {p['chapter'][-11:]} {p['panel'][1:7]} {p['d']}",
                fill="black")
    cv.save(out_path, quality=85)


def write_pick(series_dir, props: Dict[str, Dict[str, Any]], mc: List[str],
               decoy: List[str], out_path) -> Dict[str, Any]:
    if len(mc) != 2 or len(decoy) != 2:
        raise SystemExit("pick exactly 2 protagonist panels and 2 decoy panels")
    if len({k[:2] for k in decoy}) != 1 or not all(k.startswith("G") for k in decoy):
        raise SystemExit("both decoy panels must come from ONE look-alike group")
    sd = Path(series_dir)
    rel = lambda k: os.path.relpath(sd / props[k]["chapter"] / "scenes" / props[k]["panel"], REPO)
    obj = {"_readme": "Exemplar panels gemma confirms the protagonist against "
                      "(tools/panel_identity.py --verify-ccip). protagonist = 2 close-ups of "
                      "him; decoy = 2 close-ups of ONE look-alike. Proposed by "
                      "tools/identity_exemplars.py, picked by eye.",
           "series": sd.name,
           "protagonist": [rel(k) for k in mc],
           "decoy": [rel(k) for k in decoy],
           "picked": {"protagonist": mc, "decoy": decoy}}
    Path(out_path).write_text(json.dumps(obj, indent=2) + "\n")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--series", required=True, help="ongoing/<slug>")
    ap.add_argument("--sheet", default="", help="write the proposal sheet here")
    ap.add_argument("--pick", default="", help="M ids of the protagonist, e.g. M1,M4")
    ap.add_argument("--decoy", default="", help="ids from ONE group, e.g. G1a,G1c")
    args = ap.parse_args()
    sd = Path(args.series)
    saved = sd / ".identity" / "exemplar_proposals.json"
    if args.pick and saved.exists():          # the ids on the sheet that was graded
        props = json.loads(saved.read_text())
    else:
        props = propose(sd)
        saved.parent.mkdir(exist_ok=True)
        saved.write_text(json.dumps(props, indent=1))
    if args.sheet:
        sheet(sd, props, args.sheet)
        print(f"[ok] {len(props)} proposals -> {args.sheet}")
    if args.pick:
        out = REPO / "cast" / f"{sd.name}.exemplars.json"
        obj = write_pick(sd, props, args.pick.split(","), args.decoy.split(","), out)
        print(f"[ok] wrote {out}: {obj['protagonist']} vs {obj['decoy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
