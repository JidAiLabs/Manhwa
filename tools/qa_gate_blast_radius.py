#!/usr/bin/env python3
"""
qa_gate_blast_radius.py — measure a QA gate against the real corpus BEFORE
changing what it is allowed to do (block, park, heal).

    python tools/qa_gate_blast_radius.py                       # run every detector
    python tools/qa_gate_blast_radius.py --from-reports        # tally existing prep_qa.json
    python tools/qa_gate_blast_radius.py --code actor_mismatch --sample 20

Run it on the host that holds ongoing/ (the Mini).

A code that is ERROR, in narration_heal.HEALABLE and in
worker._CRITICAL_QA_CODES makes the worker re-narrate the flagged groups up to
_HEAL_MAX cycles and then, if the flag survives, BLOCK the chapter. A FALSE
POSITIVE can never be healed away — the line was already correct — so it parks
that chapter forever and burns heal cycles getting there. This has bitten
three times: an 11-of-12 false-positive rate in the 2026-08-18
actor/caption/impact audit, and on 2026-09-06 a first cut of garbled_line
that flagged 60 lines across 135 chapters (43 chapters — 32% of the fleet),
every one ordinary English.

Two lessons this file encodes (2026-09-07):

1. Bind detectors by PARAMETER NAME, never skip silently. The first version
   called det(beats) inside a bare try/except, so every multi-arg detector
   (actor_mismatch, actor_count_mismatch, impact_mismatch, narration_offset,
   dead_actor, role_stale, ...) was silently dropped and never measured —
   actor_mismatch shipped as a heal-target at 0/6 measured precision. Now a
   detector whose inputs are not on disk is printed under SKIPPED, with the
   names it needed.
2. One title is not a corpus. Counts are broken out per series, and the
   footer says so when only one title is present. 148 of the first 150
   prepared chapters were Omniscient Reader; a threshold that looks right
   there is title-tuned by construction.

THE BAR. A judgment-based code — one whose trigger is a resolver, a judge or
a heuristic over prose, not a structural fact of the plan — may enter
narration_heal.HEALABLE only when a graded --sample shows precision >= 0.80
over >= 20 flags spanning >= 2 titles with >= 5 flags each. It never enters
worker._CRITICAL_QA_CODES unless its trigger becomes true by construction,
and then only after the same bar is re-measured. 0.80 is a policy number
(one wasted heal cycle per false positive against a ~0.5 chance that a
re-roll beats the original), not a threshold tuned on any corpus.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import inspect
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRECISION_BAR = 0.80
MIN_GRADED = 20
MIN_TITLES = 2
MIN_PER_TITLE = 5

# detector parameter name -> file under the chapter dir. A detector whose
# parameter is not listed here (an image, a callable, a model handle) is not
# a chapter-level gate and is reported under SKIPPED with the name it needs.
INPUTS = {
    "beats_obj": "manifest.beats.json",
    "understood_obj": "manifest.panels.understood.json",
    "cast_obj": "manifest.cast.json",
    "ledger_obj": "manifest.ledger.json",
    "groups_obj": "manifest.groups.json",
    "script_obj": "manifest.script.json",
    "story_obj": "manifest.story.json",
    "plan": "render.plan.clean.json",
    "tts_index": os.path.join("tts", "tts_index.json"),
}


def _load_prep_qa():
    sys.path.insert(0, os.path.join(REPO, "tools"))
    spec = importlib.util.spec_from_file_location(
        "prep_qa", os.path.join(REPO, "tools", "prep_qa.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _vitems(ep: str) -> Dict[str, Dict[str, Any]]:
    """The vision items exactly as prep_qa.main builds them."""
    out: Dict[str, Dict[str, Any]] = {}
    v = _read_json(os.path.join(ep, "manifest.vision.json")) or {}
    for it in v.get("items") or []:
        out[str(it.get("scene_file") or "")] = {
            "ocr_clean": it.get("ocr_clean"),
            "text_only": it.get("text_only"),
            "text_coverage": it.get("text_coverage"),
            "subjects": it.get("subjects") or [],
            "n_words": len((it.get("vision") or {}).get("ocr_words") or []),
            "panel_kind": it.get("panel_kind"),
        }
    return out


def _load_chapter(ep: str) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {"ep": ep, "vitems": _vitems(ep)}
    for name, rel in INPUTS.items():
        obj = _read_json(os.path.join(ep, rel))
        if obj is not None:
            inputs[name] = obj
    return inputs


def _bind(det, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """kwargs for *det* from what is on disk, plus the names it needs and
    has no default for. Never guesses a positional."""
    kwargs: Dict[str, Any] = {}
    missing: List[str] = []
    for p in inspect.signature(det).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in inputs:
            kwargs[p.name] = inputs[p.name]
        elif p.default is p.empty:
            missing.append(p.name)
    return kwargs, missing


def _series(ep: str) -> str:
    return os.path.basename(os.path.dirname(ep.rstrip(os.sep)))


def _chapter(ep: str) -> str:
    return os.path.basename(ep.rstrip(os.sep))


class Tally:
    def __init__(self) -> None:
        self.lines: Dict[Tuple[str, str], int] = defaultdict(int)
        self.chaps: Dict[Tuple[str, str], set] = defaultdict(set)
        self.by_title: Dict[Tuple[str, str], Dict[str, set]] = defaultdict(
            lambda: defaultdict(set))
        self.flags: Dict[str, List[Tuple[str, str, str, Dict[str, Any]]]] = \
            defaultdict(list)                  # code -> (series, chapter, ep, flag)
        self.examples: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.scanned: Dict[str, int] = defaultdict(int)   # series -> chapters

    def add(self, ep: str, fl: Dict[str, Any], show: int) -> None:
        code = str(fl.get("code") or "?")
        sev = str(fl.get("severity") or "?")
        key = (code, sev)
        s, c = _series(ep), _chapter(ep)
        self.lines[key] += 1
        self.chaps[key].add(ep)
        self.by_title[key][s].add(ep)
        self.flags[code].append((s, c, ep, fl))
        if len(self.examples[key]) < show:
            self.examples[key].append(
                f"{s}/{c} {fl.get('segment_id', '')}: "
                f"{str(fl.get('detail', ''))[:110]}")


def run_detectors(pq, eps: List[str], code: str, show: int
                  ) -> Tuple[Tally, Dict[str, List[str]]]:
    detectors = [(n, getattr(pq, n)) for n in sorted(dir(pq))
                 if n.endswith("_flags") and not n.startswith("_")
                 and callable(getattr(pq, n))]
    tally = Tally()
    skipped: Dict[str, List[str]] = {}
    for ep in eps:
        inputs = _load_chapter(ep)
        if "beats_obj" not in inputs:
            continue
        tally.scanned[_series(ep)] += 1
        for name, det in detectors:
            kwargs, missing = _bind(det, inputs)
            if missing:
                skipped.setdefault(name, missing)
                continue
            try:
                flags = det(**kwargs) or []
            except Exception as exc:              # a detector crash is a finding
                skipped.setdefault(name, [f"raised {type(exc).__name__}"])
                continue
            for fl in flags:
                if code and fl.get("code") != code:
                    continue
                tally.add(ep, fl, show)
    return tally, skipped


def tally_reports(eps: List[str], code: str, show: int) -> Tally:
    tally = Tally()
    for ep in eps:
        rep = _read_json(os.path.join(ep, "prep_qa.json"))
        if not isinstance(rep, dict):
            continue
        tally.scanned[_series(ep)] += 1
        for fl in rep.get("flags") or []:
            if not isinstance(fl, dict):
                continue
            if code and fl.get("code") != code:
                continue
            tally.add(ep, fl, show)
    return tally


def print_table(tally: Tally, healable: set, critical: set) -> None:
    n_titles = len(tally.scanned)
    scanned = sum(tally.scanned.values())
    print(f"corpus: {scanned} chapters across {n_titles} title(s): "
          + ", ".join(f"{s} {n}" for s, n in sorted(tally.scanned.items())))
    print()
    print(f"{'code':26s} {'sev':5s} {'lines':>6s} {'chaps':>6s} {'PARK%':>6s} "
          f"{'titles':>7s}  heal  block")
    keys = sorted(tally.chaps, key=lambda k: (-len(tally.chaps[k]), k))
    for key in keys:
        code, sev = key
        n = len(tally.chaps[key])
        k = len(tally.by_title[key])
        print(f"{code:26s} {sev:5s} {tally.lines[key]:6d} {n:6d} "
              f"{n / scanned * 100:5.0f}% {k:3d}/{n_titles:<3d}  "
              f"{'yes' if code in healable else ' - ':>4s}  "
              f"{'YES' if code in critical else ' - ':>5s}")
        if n_titles > 1:
            for s in sorted(tally.by_title[key]):
                print(f"    · {s}: {len(tally.by_title[key][s])} chapter(s)")
        for ex in tally.examples[key]:
            print(f"    · {ex}")


def _segment_for(pq, inputs: Dict[str, Any], fl: Dict[str, Any]
                 ) -> Tuple[str, List[str]]:
    """(line, span) of the segment a beats-level flag points at."""
    from beats_segments import beat_segments
    m = re.match(r"g0*(\d+)", str(fl.get("segment_id") or ""))
    gid = int(m.group(1)) if m else -1
    scene = pq._base_scene(os.path.basename(str(fl.get("scene") or "")))
    fallback: Tuple[str, List[str]] = ("", [])
    for b in ((inputs.get("beats_obj") or {}).get("beats") or []):
        if int(b.get("group_id") or 0) != gid:
            continue
        for s in beat_segments(b):
            span = [str(x) for x in (s.get("span") or [])]
            if span and pq._base_scene(os.path.basename(span[0])) == scene:
                return str(s.get("line") or ""), span
            if not fallback[0]:
                fallback = (str(s.get("line") or ""), span)
    return fallback


def dump_sample(pq, tally: Tally, code: str, n: int, seed: int, out_path: str
                ) -> int:
    """Round-robin over titles, then per flag everything a grader needs on
    one screen: the FULL line, the covered panels' subjects/dialogue, what the
    oracle resolved there, the cast entries behind the flagged noun."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from cast_identity import resolve_figures_by_file
    by_title: Dict[str, list] = defaultdict(list)
    for item in tally.flags.get(code, []):
        by_title[item[0]].append(item)
    rng = random.Random(seed)
    for lst in by_title.values():
        rng.shuffle(lst)
    picked: list = []
    titles = sorted(by_title)
    while len(picked) < n and any(by_title[t] for t in titles):
        for t in titles:
            if by_title[t] and len(picked) < n:
                picked.append(by_title[t].pop())
    lines_out: List[str] = [f"# {code}: {len(picked)} flag(s) to grade "
                            f"(seed {seed}). Fill in `verdict:` as TP / FP.", ""]
    for s, c, ep, fl in picked:
        inputs = _load_chapter(ep)
        line, span = _segment_for(pq, inputs, fl)
        u_by_sf = {str(p.get("scene_file")): p
                   for p in ((inputs.get("understood_obj") or {}).get("panels") or [])
                   if isinstance(p, dict) and p.get("scene_file")}
        v_by_base = {pq._base_scene(os.path.basename(k)): k
                     for k in inputs.get("vitems", {})}
        claimed = set()
        try:
            from beats_segments import beat_segments
            for b in ((inputs.get("beats_obj") or {}).get("beats") or []):
                for sg in beat_segments(b):
                    for fn in sg.get("span") or []:
                        claimed.add(pq._base_scene(os.path.basename(str(fn))))
        except Exception:
            pass
        covered = pq._covered_panels(span, sorted(v_by_base), claimed) if span else []
        figures = resolve_figures_by_file(inputs.get("understood_obj"),
                                          inputs.get("cast_obj"))
        fig_by_base = {pq._base_scene(os.path.basename(k)): v
                       for k, v in figures.items()}
        u_by_base = {pq._base_scene(os.path.basename(k)): v
                     for k, v in u_by_sf.items()}
        lines_out += [f"## {s} / {c} / {fl.get('segment_id', '')} / "
                      f"{fl.get('scene', '')}",
                      f"flag:    {fl.get('detail', '')}",
                      f"line:    {line!r}",
                      f"span:    {span}   covered: {covered}"]
        for base in covered or [pq._base_scene(os.path.basename(x)) for x in span]:
            p = u_by_base.get(base) or {}
            lines_out.append(f"  panel {base}: subjects={p.get('subjects')}")
            if p.get("dialogue"):
                lines_out.append(f"      dialogue={str(p.get('dialogue'))[:160]!r}")
            lines_out.append(f"      resolved={[(f.get('name'), f.get('evidence'))
                                           for f in fig_by_base.get(base, [])]}")
        m = re.search(r"line names '([^']+)'", str(fl.get("detail") or ""))
        if m:
            noun = m.group(1).lower()
            for mem in ((inputs.get("cast_obj") or {}).get("cast") or []):
                names = [str(mem.get("canonical_name") or "")] + \
                    [str(a) for a in (mem.get("aliases") or [])]
                if any(noun in nm.lower() for nm in names):
                    lines_out.append(
                        f"  cast '{noun}': {mem.get('canonical_name')} "
                        f"aliases={mem.get('aliases')} "
                        f"look={str(mem.get('visual_description') or '')[:90]!r}")
        lines_out += ["verdict: ", ""]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    print(f"\nsample: {len(picked)} {code} flag(s) written to {out_path}")
    return len(picked)


def _footer(tally: Tally) -> None:
    print("\nPARK% = share of the fleet this code would stop if it is BLOCKING "
          "and the heal cannot clear it.")
    print(f"THE BAR: a judgment code enters HEALABLE only at precision >= "
          f"{PRECISION_BAR:.2f} over >= {MIN_GRADED} graded flags spanning "
          f">= {MIN_TITLES} titles with >= {MIN_PER_TITLE} each; "
          "_CRITICAL_QA_CODES only for a by-construction trigger.")
    if len(tally.scanned) < MIN_TITLES:
        print("WARNING: single-title corpus — no generalization claim is "
              "possible from these numbers; prepare chapters of another title "
              "before changing any gate on their strength.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--code", default="", help="only this QA code")
    ap.add_argument("--show", type=int, default=3,
                    help="example lines to print per code (0 = none)")
    ap.add_argument("--root", default=REPO, help="repo root holding ongoing/")
    ap.add_argument("--glob", default=os.path.join("ongoing", "*", "*"),
                    help="chapter dirs, relative to --root")
    ap.add_argument("--from-reports", action="store_true",
                    help="tally the flags already in each prep_qa.json instead "
                         "of running detectors (every code, any severity)")
    ap.add_argument("--sample", type=int, default=0,
                    help="write N flags of --code for a human to grade")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample-out", default="",
                    help="default: $TMPDIR/qa_sample_<code>.txt (never the repo)")
    args = ap.parse_args(argv)

    pq = _load_prep_qa()
    sys.path.insert(0, REPO)
    from tools.narration_heal import HEALABLE
    from studio.worker import _CRITICAL_QA_CODES

    eps = sorted(p for p in glob.glob(os.path.join(args.root, args.glob))
                 if os.path.isdir(p))
    if args.from_reports:
        tally = tally_reports(eps, args.code, args.show)
        skipped: Dict[str, List[str]] = {}
    else:
        tally, skipped = run_detectors(pq, eps, args.code, args.show)
    if not tally.scanned:
        print(f"no chapters matched {args.glob!r} under {args.root} — run this "
              "on the host that holds ongoing/ (the Mini)")
        return 1

    print_table(tally, set(HEALABLE), set(_CRITICAL_QA_CODES))
    if skipped:
        print("\nSKIPPED (needs inputs this tool does not load — an image, a "
              "callable, a model):")
        for name in sorted(skipped):
            print(f"  {name}: {', '.join(skipped[name])}")
    if args.sample:
        if not args.code:
            print("--sample needs --code")
            return 2
        out = args.sample_out or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"qa_sample_{args.code}.txt")
        dump_sample(pq, tally, args.code, args.sample, args.seed, out)
    _footer(tally)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
