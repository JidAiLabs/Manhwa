#!/usr/bin/env python3
"""tools/identity_exemplars.py — propose the exemplar panels gemma confirms the
protagonist against, then write the chosen 2 + 2 to cast/<slug>.exemplars.json.

The picture profile (panel_identity_ccip, .identity caches from the sweep)
proposes; a person confirms (owner decision 2026-10-10 — auto-picked exemplars made
gemma answer "protagonist" 60/60 when one was a tiny far figure, and ~85% even
when they were right). Candidates are CLOSE-UPS only: the head is >= 6% of the
panel and the panel is not a tall strip, so the face survives gemma's 640 px.

  --auto   the automatic proposal (worker job identity_propose): candidate
           leads, the "most present" rule's unless gemma gives one answer in
           both orders; confirmed or swapped with one click on the Series
           page (nothing switches on before that).

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
from typing import Any, Dict, List, Optional, Tuple

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
RECUR_CHAPTERS = 3      # a decoy is a RECURRING look-alike: its heads span this many chapters


def _seeds(F: np.ndarray, chap: np.ndarray, seed: int = 0) -> Dict[str, int]:
    """Automatic guesses at the lead (indices into F). Measured on 9 series
    2026-10-10: most-drawn right 6/9, most-chapters 7/9, most-present (chapters
    with 3+ appearances) 7-8/9, each failing on a different series; on Nano
    Machine all three chose the instructor. runner_up = the most-drawn face
    unlike all three, so the real lead is on offer even then."""
    samp = np.random.default_rng(seed).choice(len(F), min(3000, len(F)), replace=False)
    S = F[samp]
    D = pic.ccip_diff(S, S)
    np.fill_diagonal(D, np.inf)
    dens = (D < pic.SEED_RADIUS).sum(axis=1)
    Dall = pic.ccip_diff(S, F)
    spread = np.zeros(len(samp), int)
    depth = np.zeros(len(samp), int)
    for i in range(len(samp)):
        c = collections.Counter(chap[Dall[i] < 0.15].tolist())
        spread[i] = len(c)
        depth[i] = sum(1 for v in c.values() if v >= 3)
    out = {"most_drawn": int(dens.argmax()),
           "most_chapters": int(np.lexsort((dens, spread))[-1]),
           "most_present": int(np.lexsort((dens, depth))[-1])}
    unlike = D[:, list(out.values())].min(axis=1) >= pic.SEED_RADIUS
    unlike[list(out.values())] = False            # the diagonal is inf
    if unlike.any():
        out["runner_up"] = int(np.where(unlike, dens, -1).argmax())
    return {k: int(samp[v]) for k, v in out.items()}


def _typical(sd: Path, rows, F: np.ndarray, j: int, k: int, radius: float,
             ok: Optional[np.ndarray] = None, cap: int = 400) -> Tuple[List[int], int]:
    """Up to k close-ups of the character of head *j*, most typical first, and
    the character's medoid head. The seed is re-centred on the medoid of its
    look-alikes (< radius); candidates are one-head close-up panels from
    distinct chapters, ranked by their median difference to the character's
    heads: a clear face matches most of them, a back of the head or a face in
    shadow few (v1 took the nearest heads to one arbitrary seed head and
    showed gemma backs of heads)."""
    rng = np.random.default_rng(0)

    def near(center: int) -> np.ndarray:
        g = np.where(pic.ccip_diff(F, F[center:center + 1])[:, 0] < radius)[0]
        return rng.choice(g, cap, replace=False) if len(g) > cap else g
    g0 = near(j)
    medoid = int(g0[np.median(pic.ccip_diff(F[g0], F[g0]), axis=1).argmin()]) if len(g0) else j
    group = near(medoid)
    d = pic.ccip_diff(F, F[medoid:medoid + 1])[:, 0]
    cand = np.asarray([i for i in np.where(d < radius)[0]
                       if rows[i][2] and (ok is None or ok[i])], int)
    if not len(cand) or not len(group):
        return [], medoid
    typ = np.median(pic.ccip_diff(F[cand], F[group]), axis=1)
    out, used = [], set()
    for i in cand[np.argsort(typ, kind="stable")]:
        ch, h, _ = rows[i]
        if ch not in used and _closeup(sd, ch, h):
            used.add(ch)
            out.append(int(i))
        if len(out) == k:
            break
    return out, medoid


def _decoy(sd: Path, rows, F: np.ndarray, chap: np.ndarray, dl: np.ndarray,
           k: int = 4) -> List[int]:
    """Close-ups of ONE recurring look-alike of the lead: the densest face
    0.20-0.35 from every lead head whose look-alikes span RECUR_CHAPTERS
    chapters (v1 took the nearest look-alike with one mate; on Clan's Failure
    that pair was two different people)."""
    ok = dl >= pic.SEED_RADIUS                         # never the lead himself
    for lo, hi in ((pic.SEED_RADIUS, 0.35), (pic.SEED_RADIUS, 9.0)):
        band = np.where((dl >= lo) & (dl < hi))[0]
        if len(band) > 2000:
            band = np.random.default_rng(0).choice(band, 2000, replace=False)
        if len(band) < 2:
            continue
        D = pic.ccip_diff(F[band], F[band])
        np.fill_diagonal(D, np.inf)
        dens = (D < 0.10).sum(axis=1)
        tried = np.zeros(len(band), bool)
        for _ in range(10):                            # the 10 densest look-alikes
            s = int(np.where(tried, -1, dens).argmax())
            if tried[s] or dens[s] == 0:
                break
            mates = D[s] < 0.10
            tried |= mates
            tried[s] = True
            if len(set(chap[band[mates]].tolist()) | {chap[band[s]]}) < RECUR_CHAPTERS:
                continue
            picks, _ = _typical(sd, rows, F, int(band[s]), k, 0.10, ok=ok)
            if len(picks) >= 2:
                return picks
    return []


def _ask_lead(sd: Path, cands: List[Dict[str, Any]], path, chat, model: str,
              load) -> Tuple[Optional[int], List[str]]:
    """Which candidate is the lead — asked twice, the second time in reverse
    order; a model that answers by position (or not at all) gives no lead.
    v1 asked once and chose a purple-lit side figure on Death Knight."""
    from panel_identity import _jpeg
    from ollama_compat import chat as _chat, first_json
    load = load or _jpeg
    n = len(cands)
    letters = "ABCD"[:n]
    story = _synopsis(sd) or "(no synopsis)"
    picks, answers = [], []
    for order in (list(range(n)), list(range(n))[::-1]):
        shown = " ".join("Images %d and %d show PERSON %s." % (2 * i + 1, 2 * i + 2, L)
                         for i, L in enumerate(letters))
        prompt = LEAD_PROMPT % (shown, story, " or ".join('"%s"' % L for L in letters))
        try:
            resp = (chat or _chat)(model=model, think=False,
                                   options={"temperature": 0, "num_ctx": 8192, "num_predict": 40},
                                   messages=[{"role": "user", "content": prompt,
                                              "images": [load(path(j)) for c in order
                                                         for j in cands[c]["prot"][:2]]}])
            answer = str((first_json(str((resp.get("message") or {}).get("content") or ""))
                          or {}).get("main") or "").strip().upper()
        except Exception as e:                           # noqa: BLE001
            print(f"[auto-pick] {sd.name}: lead question failed ({e})")
            return None, answers
        answers.append(answer)
        if answer not in letters:
            return None, answers
        picks.append(order[letters.index(answer)])
    return (picks[0] if picks[0] == picks[1] else None), answers


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
    """A PROPOSAL for the series' exemplars, written to *out_path*: every
    candidate lead with up to 6 close-ups of him and up to 4 of one recurring
    look-alike, most typical first, and the chosen candidate — gemma's when it
    gives the same answer in both orders, else the "most present" rule's
    (chosen_by says which). The file's protagonist/decoy are the chosen
    candidate's first 2 + 2, so it works as an exemplars file (seed:
    exemplars); a person confirms or swaps it on the Series page. None
    (nothing written) when no candidate has 2 + 2."""
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
    cands: List[Dict[str, Any]] = []
    for rule in ("most_present", "most_chapters", "most_drawn", "runner_up"):
        g = seeds.get(rule)
        if g is None:
            continue
        prot, medoid = _typical(sd, rows, F, g, 6, pic.REF_TIGHT)
        if len(prot) < 2:
            continue
        # one person reached from two seeds (Death Knight 2026-10-10: two
        # candidates 0.044 apart, gemma then answered by position)
        same = next((c for c in cands if float(np.median(
            pic.ccip_diff(F[prot], F[c["prot"]]))) < pic.CUT), None)
        if same is not None:
            same["rules"].append(rule)
            continue
        dl = pic.ccip_diff(F, F[prot + [medoid]]).min(axis=1)
        dec = _decoy(sd, rows, F, chap, dl)
        if len(dec) >= 2:
            cands.append({"rules": [rule], "prot": prot, "decoy": dec})
    if not cands:
        print(f"[auto-pick] {sd.name}: no candidate lead with 2 close-ups and a recurring "
              "look-alike -> nothing proposed")
        return None
    path = lambda j: str(sd / rows[j][0] / "scenes" / rows[j][1]["panel"])
    # the default lead: "most present" (right on 7-8 of 9 series) else the
    # first candidate; gemma overrides it only with the SAME answer in both
    # orders — 2026-10-10 it answered "A" for whoever came first on 3 of 6
    chosen, by, answers = 0, "only candidate", []
    if len(cands) > 1:
        chosen = next((k for k, c in enumerate(cands) if "most_present" in c["rules"]), 0)
        said, answers = _ask_lead(sd, cands, path, chat, model, load)
        chosen, by = (said, "gemma") if said is not None else (chosen, "rules")
    rel = lambda j: (os.path.relpath(path(j), REPO)
                     if Path(path(j)).resolve().is_relative_to(REPO) else path(j))
    head = lambda j: {"path": rel(j), "chapter": rows[j][0], "panel": rows[j][1]["panel"],
                      "box": rows[j][1]["box"]}
    obj: Dict[str, Any] = {
        "_readme": "PROPOSED exemplars (tools/identity_exemplars.py --auto): each candidate "
                   "lead with close-ups of him and of one recurring look-alike, most typical "
                   "first; protagonist/decoy = gemma's choice. Nothing is switched on until "
                   "a person confirms it on the Series page.",
        "series": sd.name, "seed": "exemplars",
        "auto": {"version": 2, "candidates": len(cands), "asked_gemma": len(cands) > 1,
                 "answers": answers, "chosen": chosen, "chosen_by": by,
                 "rules": {r: next((k for k, c in enumerate(cands) if r in c["rules"]), None)
                           for r in seeds}},
        "candidates": [{"protagonist": [head(j) for j in c["prot"]],
                        "decoy": [head(j) for j in c["decoy"]]} for c in cands]}
    if chosen is not None:
        c = obj["candidates"][chosen]
        obj["protagonist"] = [h["path"] for h in c["protagonist"][:2]]
        obj["decoy"] = [h["path"] for h in c["decoy"][:2]]
        obj["heads"] = {"protagonist": c["protagonist"][:2], "decoy": c["decoy"][:2]}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(obj, indent=2) + "\n")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--series", required=True, help="ongoing/<slug>")
    ap.add_argument("--sheet", default="", help="write the proposal sheet here")
    ap.add_argument("--pick", default="", help="M ids of the protagonist, e.g. M1,M4")
    ap.add_argument("--decoy", default="", help="ids from ONE group, e.g. G1a,G1c")
    ap.add_argument("--auto", default="", metavar="OUT",
                    help="propose automatically (candidate leads, gemma's choice asked twice); "
                         "write the proposal to OUT")
    args = ap.parse_args()
    sd = Path(args.series)
    if args.auto:
        obj = auto_pick(sd, args.auto)
        print(f"[auto-pick] {sd.name}: " + (json.dumps(obj["auto"]) + f" -> {args.auto}"
                                           if obj else "nothing proposed"))
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
