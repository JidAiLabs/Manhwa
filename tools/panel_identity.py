#!/usr/bin/env python3
"""
panel_identity.py — WHO is in a panel, decided from the IMAGE.

`cast_identity` joins two TEXTS: gemma's description of the panel ("a young man
with short dark hair") and a text description of each character. In a cast where
several characters are young dark-haired men that join cannot separate them, and
it doesn't: ORV Episode 6 narrated the protagonist as "Namwoon Kim" 67 times
(the chapter's looks were swapped), and 5 of 8 "clear shots of the MC" thumbnail
suggestions were other men.

This module asks the local multimodal model to LOOK, against exemplar panels the
owner confirmed. Measured on 50 Ep6 panels + 8 cross-chapter tiles (2026-09-17):

  candidates/call   tiles (3 real)     Ep6 precision/recall   wrong names
  word matching     3 of 8 suggested   0.69 / 0.68            22
  1 (A or OTHER)    2 found, 3 FALSE   0.70 / 0.51             8
  2 (A/B/OTHER)     3 found, 0 false   0.96 / 0.65             6
  4 (A/B/C/D/OTHER) 0 found            0.75 / 0.24             3 + 12 claims
                                                                about characters
                                                                absent from every
                                                                panel

So TWO candidates is a hard cap, not a default: with one, look-alikes are forced
onto the lead; with four, it recognises nobody and invents the extras. Recall
~0.65 is the accepted cost — an unconfirmed figure keeps a neutral handle ("the
white-haired guy"), which is the owner's rule: never mix names, never drop a
panel.

Whole panels, no crops: neither YOLO model we have can find manhwa characters
(the legacy `character` class fired 2 times in 60 panels at conf 0.01, COCO
person 5 of 60).

Design: docs/plans/2026-09-18-image-based-character-identity.md
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)

MAX_CANDIDATES = 2          # measured ceiling, see the table above
MODEL = os.environ.get("STUDIO_IDENTITY_MODEL", "gemma4:26b")
# panels these kinds describe carry no drawn person to identify
_SKIP_KINDS = frozenset({"system", "caption", "empty"})
_LETTERS = "AB"


def _norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()


def _members(registry: Any) -> List[Dict[str, Any]]:
    if isinstance(registry, dict):
        registry = registry.get("cast") or registry.get("members") or []
    return [m for m in (registry or []) if isinstance(m, dict)]


def candidates(registry: Any, max_n: int = MAX_CANDIDATES,
               chapter: Any = None) -> List[Dict[str, Any]]:
    """The <=2 registry members that carry exemplars, protagonist first.

    Capped hard: a third candidate measurably destroys recall and invents
    appearances of the characters it was just shown. With more than two
    available, the second slot goes to a character THIS chapter contains
    (*chapter* = its manifest.cast.json): asking about someone absent wastes
    the slot and the answer would be untrusted anyway. When nobody else is in
    the chapter, any exemplar member still rides along as a DECOY — that is
    what keeps look-alikes off the lead (one candidate alone: precision 0.70).
    """
    in_chapter = {_norm(m.get("canonical_name") or m.get("id"))
                  for m in _members(chapter)} if chapter else set()
    out = []
    for m in _members(registry):
        ex = [str(p) for p in (m.get("exemplars") or []) if str(p).strip()]
        if not ex:
            continue
        out.append({"name": str(m.get("canonical_name") or m.get("id") or "").strip(),
                    "exemplars": ex,
                    "_prot": bool(m.get("is_protagonist"))})
    out.sort(key=lambda m: (0 if m["_prot"] else 1,
                            0 if _norm(m["name"]) in in_chapter else 1))
    return [{k: v for k, v in m.items() if k not in ("_prot",)} for m in out[:max_n]]


def build_prompt(names: List[str]) -> str:
    """The measured forced-choice question.

    Three things are load-bearing and were each measured: an explicit OTHER (a
    yes/no question makes the model say yes to any similar-looking man), the
    FACE instruction (outfit and hair colour are exactly what the old text
    matching already got wrong), and naming the letters only — a character NAME
    in the prompt is a hint to pattern-match on instead of looking.
    """
    letters = _LETTERS[:len(names)]
    shown = " ".join(
        "Images %d and %d show CHARACTER %s." % (2 * i + 1, 2 * i + 2, L)
        for i, L in enumerate(letters))
    n = 2 * len(names) + 1
    ab = " or ".join(letters)                       # "A" / "A or B"
    return (
        "%s%s Image %d is a panel from the same manhwa, which has MANY other "
        "characters who look similar: dark hair, suits, young men. For EACH "
        "person visible in image %d, answer %s or OTHER. Answer %s only when "
        "the FACE clearly matches (eyes, face shape, hairstyle); a similar "
        "outfit or hair colour alone is OTHER. Return ONLY JSON: "
        '{"people": [%s or "OTHER", ...]}'
        % (shown, " They are two different characters." if len(names) > 1 else "",
           n, n, ", ".join(letters), ab,
           " or ".join('"%s"' % L for L in letters)))


def parse_reply(raw: Any, names: List[str]) -> Dict[str, Any]:
    """{"names": [confirmed], "others": count}. Anything unusable confirms
    NOBODY — a guess is the failure this module exists to remove."""
    from ollama_compat import first_json
    got = first_json(str(raw or "")) or {}
    people = got.get("people")
    if not isinstance(people, list):
        return {"names": [], "others": 0}
    found: List[str] = []
    others = 0
    for p in people:
        tok = str(p or "").strip().upper()
        i = _LETTERS.find(tok) if len(tok) == 1 else -1
        if 0 <= i < len(names):
            if names[i] not in found:
                found.append(names[i])
        elif tok == "OTHER":
            others += 1
    return {"names": found, "others": others}


def _jpeg(path: str, side: int = 640) -> bytes:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def identify_panel(panel_path: str, cands: List[Dict[str, Any]], *,
                   chat: Optional[Callable] = None, model: str = MODEL,
                   exemplar_images: Optional[List[bytes]] = None,
                   load: Callable[[str], bytes] = _jpeg) -> Dict[str, Any]:
    """One call. A failed or unparseable call confirms nobody (never raises)."""
    names = [c["name"] for c in cands]
    if chat is None:
        from ollama_compat import chat as chat            # noqa: PLW0127
    imgs = list(exemplar_images if exemplar_images is not None
                else [load(p) for c in cands for p in c["exemplars"]])
    try:
        resp = chat(model=model, think=False,
                    options={"temperature": 0, "num_ctx": 8192, "num_predict": 80},
                    messages=[{"role": "user", "content": build_prompt(names),
                               "images": imgs + [load(panel_path)]}])
        return parse_reply((resp.get("message") or {}).get("content"), names)
    except Exception as e:                                # noqa: BLE001
        print("[identity] %s: %s -> nobody confirmed"
              % (os.path.basename(panel_path), type(e).__name__))
        return {"names": [], "others": 0}


def identify_panels(ep_dir: str, registry: Any, *,
                    chat: Optional[Callable] = None, model: str = MODEL,
                    out_path: str = "",
                    load: Callable[[str], bytes] = _jpeg
                    ) -> Optional[Dict[str, Any]]:
    """Identify every panel of *ep_dir* that shows people; write
    manifest.identity.json. None when the series has no exemplars (the chapter
    then keeps today's keyword behaviour rather than a half-trusted mix)."""
    try:
        with open(os.path.join(ep_dir, "manifest.cast.json"), encoding="utf-8") as f:
            chapter = json.load(f)
    except (OSError, ValueError):
        chapter = None
    cands = candidates(registry, chapter=chapter)
    if not cands:
        return None
    # A name is trusted only when THIS chapter's cast (built from its own
    # dialogue) contains that character. ORV Ep128 has no Namwoon, yet the
    # forced choice handed out "B" on 22 of 58 panels and the identity gate
    # rewrote the chapter's real names toward them. The decoy stays in the
    # PROMPT -- that is what keeps look-alikes off the lead (measured: one
    # candidate alone drops precision to 0.70) -- only its NAME is dropped
    # here, and the figure is counted as another person instead.
    trusted = [c["name"] for c in cands]
    in_chapter = {_norm(m.get("canonical_name") or m.get("id"))
                  for m in _members(chapter)} if chapter else set()
    if in_chapter:
        trusted = [n for n in trusted if _norm(n) in in_chapter]
    understood = os.path.join(ep_dir, "manifest.panels.understood.json")
    with open(understood, encoding="utf-8") as f:
        panels = (json.load(f) or {}).get("panels") or []
    exemplar_images = [load(p if os.path.isabs(p) else os.path.abspath(p))
                       for c in cands for p in c["exemplars"]]
    out: Dict[str, Any] = {}
    for p in panels:
        fn = os.path.basename(str(p.get("scene_file") or ""))
        if (not fn or not (p.get("subjects") or [])
                or str(p.get("panel_kind") or "").lower() in _SKIP_KINDS):
            continue
        path = os.path.join(ep_dir, "scenes", fn)
        if not os.path.exists(path):
            continue
        rec = identify_panel(path, cands, chat=chat, model=model,
                             exemplar_images=exemplar_images, load=load)
        kept = [n for n in rec["names"] if n in trusted]
        out[fn] = {"names": kept,
                   "others": rec["others"] + len(rec["names"]) - len(kept)}
    obj: Dict[str, Any] = {"panels": out}
    from manifest_io import write_manifest
    target = out_path or os.path.join(ep_dir, "manifest.identity.json")
    write_manifest(target, obj, inputs=[understood], tool="panel_identity",
                   extra_meta={"candidates": [c["name"] for c in cands],
                               "trusted": trusted, "model": model})
    return obj


_REPO = os.path.dirname(_TD)


def _exemplar_paths(path: Any) -> Optional[Dict[str, List[str]]]:
    """cast/<slug>.exemplars.json -> {protagonist: [2 paths], decoy: [2 paths]},
    repo-relative paths resolved; None unless it is exactly 2 + 2 (the
    measured prompt shows images 1-2 = A, 3-4 = B)."""
    try:
        with open(path, encoding="utf-8") as f:
            ex = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    out = {}
    for k in ("protagonist", "decoy"):
        ps = [str(x) for x in (ex.get(k) or []) if str(x).strip()]
        if len(ps) != 2:
            return None
        out[k] = [x if os.path.isabs(x) else os.path.join(_REPO, x) for x in ps]
    return out


def verify_candidates(ep_dir: Any, exemplars: Any, *, chat: Optional[Callable] = None,
                      model: str = MODEL, load: Callable[[str], bytes] = _jpeg
                      ) -> Optional[Dict[str, Any]]:
    """gemma confirms the picture's candidates (2026-10-10, owner decision).

    panel_identity_ccip proposes (`cand`: one head under its 0.10 cut) and names
    nobody. Here each candidate panel gets the measured forced choice — the
    series' protagonist (A) against its closest look-alike (B), two exemplar
    panels each — and only an "A" names the protagonist. Graded on ORV: the
    picture alone ~70-85% right, picture + gemma ~95%; every wrong gemma
    confirmation had a picture score >= 0.10, which is why the cut sits there.
    A failed call or an empty answer confirms nobody (the 2026-10-10 GPU hang
    answered 200 with empty content). None = skipped: no usable exemplars, not
    a picture-candidate file, or already verified (no second paid pass)."""
    ident_path = os.path.join(str(ep_dir), "manifest.identity.json")
    ex = _exemplar_paths(exemplars)
    if ex is None:
        return None
    try:
        with open(ident_path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    meta = obj.get("_meta") or {}
    if meta.get("tool") != "panel_identity_ccip" or meta.get("verified_by"):
        return None
    name = meta.get("protagonist")
    cands = [{"name": "A", "exemplars": ex["protagonist"]},
             {"name": "B", "exemplars": ex["decoy"]}]
    imgs = [load(x) for x in ex["protagonist"] + ex["decoy"]]
    for fn, rec in (obj.get("panels") or {}).items():
        if not rec.get("cand"):
            continue
        got = identify_panel(os.path.join(str(ep_dir), "scenes", fn), cands, chat=chat,
                             model=model, exemplar_images=imgs, load=load)
        rec["gemma"] = "A" if "A" in got["names"] else ("B" if "B" in got["names"] else "OTHER")
        if rec["gemma"] == "A":
            rec["mc"] = True
            if name:
                rec["names"] = [name]
                rec["others"] = max(0, int(rec.get("others") or 0) - 1)
    from manifest_io import write_manifest
    keep = {k: v for k, v in meta.items() if k not in ("schema", "written_at", "tool")}
    write_manifest(ident_path, obj, tool="panel_identity_ccip",
                   extra_meta=dict(keep, verified_by="gemma", model=model,
                                   exemplars=os.path.basename(str(exemplars))))
    return obj


def check_series(series_dir: Any, exemplars: Any, *, n: int = 40, out_dir: Any = "",
                 chat: Optional[Callable] = None, model: str = MODEL,
                 load: Callable[[str], bytes] = _jpeg, seed: int = 0) -> Dict[str, Any]:
    """Validation before a series is switched on (worker job identity_check):
    n random picture candidates across the series (one head under the CCIP
    cut, from the sweep's .identity caches), gemma's verdict on each, written to
    <out_dir>/<slug>.json + a grading sheet <slug>_sheet.jpg (confirmed row
    first). Exemplar panels are never sampled. Model calls only — no chapter
    manifest is touched."""
    import random
    from collections import defaultdict
    import panel_identity_ccip as pic
    sd = str(series_dir)
    slug = os.path.basename(os.path.normpath(sd))
    ex = _exemplar_paths(exemplars)
    prof = pic.load_profile(sd)
    if ex is None or not prof or prof.get("status") != "active":
        raise SystemExit(f"[identity] check {slug}: needs 2+2 exemplars and an active profile")
    R = pic._feats(prof["refs"])
    skip = {os.path.abspath(x) for x in ex["protagonist"] + ex["decoy"]}
    pool = []
    for ix in pic.load_indexes(sd):
        if not len(ix["heads"]):
            continue
        d = pic.ccip_diff(ix["feats"], R).min(axis=1)
        rows = defaultdict(list)
        for i, h in enumerate(ix["heads"]):
            rows[h["panel"]].append(i)
        for fn, r in rows.items():
            hit = [i for i in r if d[i] < pic.CUT]
            path = os.path.join(sd, ix["chapter"], "scenes", fn)
            if len(hit) == 1 and os.path.abspath(path) not in skip:
                i = hit[0]
                pool.append({"chapter": ix["chapter"], "panel": fn, "path": path,
                             "box": ix["heads"][i]["box"], "d": round(float(d[i]), 3)})
    random.Random(seed).shuffle(pool)
    cands = [{"name": "A", "exemplars": ex["protagonist"]},
             {"name": "B", "exemplars": ex["decoy"]}]
    imgs = [load(x) for x in ex["protagonist"] + ex["decoy"]]
    results = []
    for c in pool[:n]:
        got = identify_panel(c["path"], cands, chat=chat, model=model,
                             exemplar_images=imgs, load=load)
        c["gemma"] = "A" if "A" in got["names"] else ("B" if "B" in got["names"] else "OTHER")
        results.append(c)
    rep = {"series": slug, "n": len(results), "candidates": len(pool),
           "confirmed": sum(1 for r in results if r["gemma"] == "A"), "cut": pic.CUT,
           "exemplars": {k: [os.path.relpath(x, _REPO) for x in v] for k, v in ex.items()},
           "results": results}
    od = str(out_dir or os.path.join(_REPO, "dist", "identity_check"))
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"{slug}.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    _check_sheet(results, os.path.join(od, f"{slug}_sheet.jpg"), slug)
    return rep


def _check_sheet(results: List[Dict[str, Any]], out: str, slug: str) -> None:
    """Head crops, confirmed first then rejected, labelled chapter/panel/score —
    graded by eye before a series is switched on."""
    from PIL import Image, ImageDraw
    import panel_identity_ccip as pic
    size, per_row = 130, 12
    rows = [[r for r in results if r["gemma"] == "A"], [r for r in results if r["gemma"] != "A"]]
    lines = [rows[0][i:i + per_row] for i in range(0, len(rows[0]), per_row)] or [[]]
    lines += [rows[1][i:i + per_row] for i in range(0, len(rows[1]), per_row)] or [[]]
    sheet = Image.new("RGB", (per_row * (size + 6) + 6, 30 + len(lines) * (size + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 8), f"{slug}: gemma CONFIRMED rows first, then rejected (B/OTHER) — grade each", fill="blue")
    for li, line in enumerate(lines):
        for k, r in enumerate(line):
            x, y = 5 + k * (size + 6), 30 + li * (size + 22)
            try:
                c = pic._crop(pic._open(r["path"]), r["box"])
                c.thumbnail((size, size))
                sheet.paste(c, (x, y))
            except Exception:                              # noqa: BLE001
                pass
            draw.text((x, y + size + 3), f"{r['gemma']} {r['chapter'][-6:]} {r['panel'][1:7]} {r['d']}",
                      fill="black")
    sheet.save(out, quality=85)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", default="")
    ap.add_argument("--series-cast", default="",
                    help="cast/<slug>.json — the owner's exemplars live there")
    ap.add_argument("--verify-ccip", default="", metavar="EXEMPLARS",
                    help="cast/<slug>.exemplars.json: gemma confirms the picture's "
                         "candidates in manifest.identity.json (panel_identity_ccip)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="")
    ap.add_argument("--check-series", default="", metavar="SERIES_DIR",
                    help="validation: gemma on --n random picture candidates of the "
                         "series (needs --verify-ccip EXEMPLARS); writes dist/identity_check/")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    if args.check_series:
        rep = check_series(args.check_series, args.verify_ccip, n=args.n, model=args.model)
        print("[ok] identity check %s: %d of %d sampled candidates confirmed (%d candidates)"
              % (rep["series"], rep["confirmed"], rep["n"], rep["candidates"]))
        return 0
    if not args.episode_dir:
        ap.error("--episode-dir is required (except with --check-series)")
    if args.verify_ccip:
        got = verify_candidates(args.episode_dir, args.verify_ccip, model=args.model)
        if got is None:
            print("[identity] verify skipped (no usable exemplars, no picture "
                  "candidates, or already verified)")
        else:
            P = got["panels"]
            print("[ok] identity verified: %d candidates, %d confirmed the protagonist"
                  % (sum(1 for v in P.values() if v.get("cand")),
                     sum(1 for v in P.values() if v.get("mc"))))
        return 0
    if not args.series_cast:
        ap.error("--series-cast is required without --verify-ccip")
    try:
        with open(args.series_cast, encoding="utf-8") as f:
            registry = json.load(f)
    except OSError:
        print("[identity] no series cast %s — skipped" % args.series_cast)
        return 0
    got = identify_panels(args.episode_dir, registry, model=args.model,
                          out_path=args.out)
    if got is None:
        print("[identity] no exemplars in %s — skipped (keyword identity stands)"
              % args.series_cast)
        return 0
    named = sum(1 for v in got["panels"].values() if v["names"])
    print("[ok] identity: %d panels, %d with a confirmed character"
          % (len(got["panels"]), named))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
