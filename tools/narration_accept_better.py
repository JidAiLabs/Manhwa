#!/usr/bin/env python3
"""narration_accept_better.py — the strictly-better safeguard for auto-heal.

A judge→regenerate loop can DEGRADE a good line: the corrections prompt nudges
the narrator, it re-rolls (temp>0), and the new line may read worse while still
passing QA. This gate compares, per healed group, the OLD line vs the NEW line
AGAINST THE PANEL and keeps the new line ONLY when it is clearly better grounded
AND better written. Anything else (equivalent / worse / unsure) reverts to the
old line. So auto-heal can only improve or hold a beat — never make it worse.

Conservative by construction: the judge must return B_better to accept; every
other verdict keeps A. Used between the heal regen and the re-plan in
worker.py's heal loop.

CLI: narration_accept_better.py --old <beats_before.json> --new <beats_after.json>
       --vision-manifest <manifest.vision.json> --scenes-dir <scenes_clean>
       --out <accepted_beats.json> [--backend ollama --ollama-model gemma4:26b]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)

VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["A_better", "B_better", "equivalent"]},
        "reason": {"type": "STRING"},
    },
    "required": ["verdict"],
}

SYSTEM = (
    "You are a strict line editor for a manhwa recap channel. You are shown the "
    "panel(s) of one beat and two candidate narration lines:\n"
    "  A = the CURRENT line.\n"
    "  B = a PROPOSED replacement.\n"
    "Decide which line is better on BOTH of these, judged against what the panel "
    "actually shows:\n"
    "  1. GROUNDING — names the right subjects, invents nothing, mis-names nothing "
    "(e.g. calling beasts 'dogs' is mis-grounded).\n"
    "  2. WRITING — reads better as spoken recap narration (natural, vivid, not "
    "filler, not interface chatter).\n"
    "Be CONSERVATIVE. Choose 'B_better' ONLY if B is clearly better on both counts. "
    "If they are similar, or B fixes one thing but loses another, or you are unsure, "
    "choose 'equivalent'. NEVER choose B merely because it is different, longer, or "
    "more elaborate. Return {verdict, reason}."
)


def accept_new(verdict: str) -> bool:
    """The whole safety rule: keep the regenerated line ONLY when the judge says
    it is strictly better. Every other verdict (equivalent / A_better / unknown)
    keeps the original."""
    return str(verdict).strip() == "B_better"


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").split()).strip()


def changed_groups(old_beats: List[Dict[str, Any]],
                   new_beats: List[Dict[str, Any]]) -> List[int]:
    """Group ids whose narration the heal actually rewrote (old != new)."""
    old_by = {b.get("group_id"): b for b in old_beats}
    out: List[int] = []
    for nb in new_beats:
        gid = nb.get("group_id")
        ob = old_by.get(gid)
        if ob is None:
            continue
        if _norm(nb.get("narration")) != _norm(ob.get("narration")):
            out.append(gid)
    return out


def gate_beats(old_beats: List[Dict[str, Any]],
               new_beats: List[Dict[str, Any]],
               judge: Callable[[Dict[str, Any], Dict[str, Any]], str],
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (accepted_beats, decisions). For every group the heal rewrote, ask
    `judge(old_beat, new_beat) -> verdict`; keep the new beat only when
    accept_new(verdict), else revert to the old beat. Unchanged beats pass
    through untouched."""
    old_by = {b.get("group_id"): b for b in old_beats}
    accepted: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for nb in new_beats:
        gid = nb.get("group_id")
        ob = old_by.get(gid)
        if ob is None or _norm(nb.get("narration")) == _norm(ob.get("narration")):
            accepted.append(nb)
            continue
        verdict = judge(ob, nb)
        keep_new = accept_new(verdict)
        decisions.append({
            "group_id": gid, "verdict": verdict, "kept": "new" if keep_new else "old",
            "old": _norm(ob.get("narration"))[:120],
            "new": _norm(nb.get("narration"))[:120],
        })
        accepted.append(nb if keep_new else ob)
    return accepted, decisions


def _scene_paths(beat: Dict[str, Any], scenes_dir: str, limit: int = 2) -> List[str]:
    files = beat.get("scene_files") or beat.get("scenes") or []
    if not files and beat.get("primary_scene_file"):
        files = [beat["primary_scene_file"]]
    out = []
    for f in files[:limit]:
        p = os.path.join(scenes_dir, os.path.basename(str(f)))
        if os.path.exists(p):
            out.append(p)
    return out


def _make_judge(call_fn, scenes_dir: str):
    def judge(ob: Dict[str, Any], nb: Dict[str, Any]) -> str:
        imgs = _scene_paths(nb, scenes_dir)
        payload = {"A_current": _norm(ob.get("narration")),
                   "B_proposed": _norm(nb.get("narration"))}
        try:
            parsed = call_fn(payload, imgs)
            return str((parsed or {}).get("verdict") or "equivalent")
        except Exception:
            return "equivalent"   # fail-safe: keep the old line
    return judge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--vision-manifest", default="")
    ap.add_argument("--scenes-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vertex", "ollama"], default="ollama")
    ap.add_argument("--ollama-model", default="gemma4:26b")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    old_doc = json.load(open(args.old))
    new_doc = json.load(open(args.new))
    old_beats = old_doc.get("beats") or []
    new_beats = new_doc.get("beats") or []

    from gemini_narrative_pass import _call_model_with_backoff  # noqa: E402
    client = None
    model = args.ollama_model
    if args.backend == "vertex":
        from google import genai
        client = genai.Client(vertexai=True, project=args.project, location=args.location)
        model = args.model

    def call_fn(payload, image_paths):
        parsed, _raw, _u = _call_model_with_backoff(
            client=client, model=model, system_instruction=SYSTEM,
            user_payload=payload, image_paths=image_paths,
            response_schema=VERDICT_SCHEMA, max_output_tokens=200,
            temperature=args.temperature, backoff_max=60.0, backend=args.backend)
        return parsed

    judge = _make_judge(call_fn, args.scenes_dir)
    accepted, decisions = gate_beats(old_beats, new_beats, judge)
    out_doc = dict(new_doc)
    out_doc["beats"] = accepted
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    reverted = sum(1 for d in decisions if d["kept"] == "old")
    for d in decisions:
        print(f"  g{d['group_id']:>3} {d['verdict']:<11} kept={d['kept']}")
    print(f"[accept-better] judged={len(decisions)} kept_new={len(decisions)-reverted} "
          f"reverted={reverted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
