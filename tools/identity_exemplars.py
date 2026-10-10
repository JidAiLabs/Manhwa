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
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------- automatic
LEAD_PROMPT = (
    "%s This is a manhwa. Its story so far: %s\n"
    "Which person is the MAIN CHARACTER — the protagonist the story follows? "
    "Use the story (age, gender, role) and the faces. Return ONLY JSON: {\"main\": %s}")


def _seeds(F: np.ndarray, chap: np.ndarray, seed: int = 0) -> Dict[str, int]:
    """Three automatic guesses at the lead (indices into F). Measured on 9
    series 2026-10-10: most-drawn right 6/9, most-chapters 7/9, most-present
    (chapters with 3+ appearances) 7-8/9, each failing on a different series."""
    samp = np.random.default_rng(seed).choice(len(F), min(3000, len(F)), replace=False)
    S = F[samp]
    D = pic.ccip_diff(S, S)
    np.fill_diagonal(D, np.inf)
    dens = (D < 0.20).sum(axis=1)
    Dall = pic.ccip_diff(S, F)
    spread = np.zeros(len(samp), int)
    depth = np.zeros(len(samp), int)
    for i in range(len(samp)):
        c = collections.Counter(chap[Dall[i] < 0.15].tolist())
        spread[i] = len(c)
        depth[i] = sum(1 for v in c.values() if v >= 3)
    return {"most_drawn": int(samp[dens.argmax()]),
            "most_chapters": int(samp[np.lexsort((dens, spread))[-1]]),
            "most_present": int(samp[np.lexsort((dens, depth))[-1]])}


def _closeups(sd: Path, rows, d: np.ndarray, k: int, lo: float = -1.0,
              hi: float = 9.0) -> List[int]:
    out, used = [], set()
    for j in np.argsort(d):
        if d[j] < lo:
            continue
        if d[j] >= hi:
            break
        ch, h, one = rows[j]
        if one and ch not in used and _closeup(sd, ch, h):
            used.add(ch)
            out.append(int(j))
        if len(out) == k:
            break
    return out


def _synopsis(sd: Path, n: int = 2, cap: int = 700) -> str:
    text = []
    for ep in sorted((p for p in sd.iterdir() if (p / "manifest.chapter_story.json").exists()),
                     key=lambda p: pic.chapter_order(p.name))[:n]:
        try:
            text.append(str(json.loads((ep / "manifest.chapter_story.json").read_text())
                            .get("synopsis") or ""))
        except (OSError, ValueError):
            pass
    return " ".join(text)[:cap]


def auto_pick(series_dir, out_path, *, chat=None, model: str = "gemma4:26b",
              load=None) -> Optional[Dict[str, Any]]:
    """The 2 + 2 exemplars with no person involved; written to *out_path* with
    seed: exemplars (the profile then grows from these faces). None (nothing
    written) when the lead cannot be decided."""
    sd = Path(series_dir)
    rows, feats = [], []
    for ix in pic.load_indexes(sd):
        n = collections.Counter(h["panel"] for h in ix["heads"])
        for i, h in enumerate(ix["heads"]):
            rows.append((ix["chapter"], h, n[h["panel"]] == 1))
            feats.append(ix["feats"][i])
    if len(rows) < 50:
        return None
    F = np.asarray(feats, dtype=np.float32)
    chap = np.asarray([r[0] for r in rows])
    seeds = _seeds(F, chap)
    groups: List[int] = []                       # distinct candidate leads
    for rule in ("most_present", "most_chapters", "most_drawn"):
        j = seeds[rule]
        if all(pic.ccip_diff(F[j:j + 1], F[g:g + 1])[0, 0] >= 0.05 for g in groups):
            groups.append(j)
    shots = {g: _closeups(sd, rows, pic.ccip_diff(F, F[g:g + 1])[:, 0], 2) for g in groups}
    groups = [g for g in groups if len(shots[g]) == 2]
    if not groups:
        return None
    path = lambda j: str(sd / rows[j][0] / "scenes" / rows[j][1]["panel"])
    answer, lead = None, groups[0]
    if len(groups) > 1:
        from panel_identity import _jpeg
        from ollama_compat import chat as _chat, first_json
        load = load or _jpeg
        letters = "ABC"[:len(groups)]
        shown = " ".join("Images %d and %d show PERSON %s." % (2 * i + 1, 2 * i + 2, L)
                         for i, L in enumerate(letters))
        prompt = LEAD_PROMPT % (shown, _synopsis(sd) or "(no synopsis)",
                                " or ".join('"%s"' % L for L in letters))
        try:
            resp = (chat or _chat)(model=model, think=False,
                                   options={"temperature": 0, "num_ctx": 8192, "num_predict": 40},
                                   messages=[{"role": "user", "content": prompt,
                                              "images": [load(path(j)) for g in groups
                                                         for j in shots[g]]}])
            answer = str((first_json(str((resp.get("message") or {}).get("content") or ""))
                          or {}).get("main") or "").strip().upper()
        except Exception as e:                           # noqa: BLE001
            print(f"[auto-pick] {sd.name}: lead question failed ({e}) -> abstain")
            return None
        if answer not in letters:
            print(f"[auto-pick] {sd.name}: unusable answer {answer!r} -> abstain")
            return None
        lead = groups[letters.index(answer)]
    prot = shots[lead]
    dl = pic.ccip_diff(F, F[prot]).min(axis=1)           # distance to the picked lead
    decoy = []
    for lo, hi in ((0.20, 0.35), (0.20, 9.0)):           # a look-alike, never the lead
        for j in _closeups(sd, rows, dl, 40, lo, hi):
            mate = [k for k in _closeups(sd, rows, pic.ccip_diff(F, F[j:j + 1])[:, 0], 2, hi=0.08)
                    if k != j and rows[k][0] != rows[j][0] and dl[k] >= 0.20]
            if mate:
                decoy = [j, mate[0]]
                break
        if decoy:
            break
    if len(decoy) != 2:
        print(f"[auto-pick] {sd.name}: no single-character look-alike with 2 close-ups -> abstain")
        return None
    rel = lambda j: (os.path.relpath(path(j), REPO)
                     if Path(path(j)).resolve().is_relative_to(REPO) else path(j))
    head = lambda j: {"chapter": rows[j][0], "panel": rows[j][1]["panel"], "box": rows[j][1]["box"]}
    obj = {"_readme": "Exemplars picked AUTOMATICALLY (tools/identity_exemplars.py --auto): the "
                      "protagonist's 2 clearest close-ups and 2 of one look-alike. seed: exemplars "
                      "= the profile grows from these faces, not the most-drawn face.",
           "series": sd.name, "seed": "exemplars",
           "protagonist": [rel(j) for j in prot], "decoy": [rel(j) for j in decoy],
           "heads": {"protagonist": [head(j) for j in prot], "decoy": [head(j) for j in decoy]},
           "auto": {"candidates": len(groups), "asked_gemma": len(groups) > 1, "answer": answer,
                    "rules": {r: next((L for L, g in zip("ABC", groups)
                                       if pic.ccip_diff(F[j:j + 1], F[g:g + 1])[0, 0] < 0.05), None)
                              for r, j in seeds.items()}}}
    Path(out_path).write_text(json.dumps(obj, indent=2) + "\n")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--series", required=True, help="ongoing/<slug>")
    ap.add_argument("--sheet", default="", help="write the proposal sheet here")
    ap.add_argument("--pick", default="", help="M ids of the protagonist, e.g. M1,M4")
    ap.add_argument("--decoy", default="", help="ids from ONE group, e.g. G1a,G1c")
    ap.add_argument("--auto", default="", metavar="OUT",
                    help="pick automatically (gemma decides the lead if the rules disagree); "
                         "write the exemplars to OUT")
    args = ap.parse_args()
    sd = Path(args.series)
    if args.auto:
        obj = auto_pick(sd, args.auto)
        print(f"[auto-pick] {sd.name}: " + (json.dumps(obj["auto"]) + f" -> {args.auto}"
                                           if obj else "abstained (nothing written)"))
        return 0
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
