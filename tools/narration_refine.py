"""
tools/narration_refine.py — closed-loop grounding gate for recap narration.

Orchestrates the cast-aware narration and ENFORCES grounding:
  1. (optional) build the chapter cast               -> manifest.cast.json
  2. generate cast-aware beats narration             -> <out> (gemini_narrative_pass --cast)
  3. judge each beat's `narration` against its panels (flash-lite, multimodal)
  4. for lines that INVENT an event/entity (verdict=hallucination, NOT mere emotional
     drift), queue a correction and regenerate ONLY those groups
     (gemini_narrative_pass --resume --corrections)
  5. repeat until clean or --max-rounds reached; print a residual report.

Emotional inference ("determined", "tense") and dialogue that matches a real speech
bubble are ALLOWED — the gate only removes invented events (the teleport / off-panel clash).

  V=.eval_venv/bin/python
  $V tools/narration_refine.py \
      --groups-manifest ongoing/nano-machine/Chapter_1/manifest.groups.json \
      --vision-manifest ongoing/nano-machine/Chapter_1/manifest.vision.json \
      --out             ongoing/nano-machine/Chapter_1/manifest.beats.narr.json \
      --cast            ongoing/nano-machine/Chapter_1/manifest.cast.json \
      --project <proj> --location us-central1
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _vision_items(vision: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = vision.get("items") or vision.get("scenes") or []
    if isinstance(items, dict):
        items = list(items.values())
    return items


def _scene_paths(vision: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in _vision_items(vision):
        sf, sp = it.get("scene_file"), it.get("scene_path")
        if sf and sp:
            out[str(sf)] = str(sp)
    return out


def _kept_images(beat: Dict[str, Any], scene_paths: Dict[str, str], cap: int = 2) -> List[str]:
    sel = beat.get("scene_selection") or []
    kept = [s.get("scene_file") for s in sel if s.get("role") == "keep" and s.get("scene_file")]
    if not kept:
        kept = beat.get("scene_files") or []
    return [scene_paths[str(s)] for s in kept if str(s) in scene_paths][:cap]


JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "fidelity": {"type": "INTEGER"},
        "verdict": {"type": "STRING"},                       # grounded|minor_drift|hallucination
        "invented": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["fidelity", "verdict", "invented"],
}

JUDGE_SYSTEM = (
    "You see the ACTUAL manhwa panel(s) for one beat and ONE narration line. Flag ONLY "
    "INVENTED content: an action, motion, clash, outcome, or entity the line asserts that is "
    "NOT visible in these panels (e.g. 'vanishing and reappearing', 'the first strike meets a "
    "shield'). Do NOT flag: emotional/atmospheric inference ('determined', 'tense', 'ominous'), "
    "or a QUOTED line that plausibly matches a speech bubble in the panel — those are allowed.\n"
    "verdict: 'hallucination' if it invents a real event/entity; 'minor_drift' if it only adds "
    "emotional/atmospheric color; 'grounded' if fully faithful.\n"
    "fidelity 1-5 (5 = fully grounded). List invented phrases in 'invented' (empty unless "
    "verdict='hallucination'). Return ONLY JSON."
)


def _img_part(path: str) -> Optional[types.Part]:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return None
    try:
        return types.Part.from_bytes(data=data, mime_type="image/jpeg")
    except TypeError:
        return types.Part.from_bytes(bytes=data, mime_type="image/jpeg")


def _judge_line(client: genai.Client, model: str, images: List[str], line: str) -> Optional[Dict[str, Any]]:
    parts: List[types.Part] = [types.Part.from_text(text=JUDGE_SYSTEM)]
    for p in images:
        ip = _img_part(p)
        if ip is not None:
            parts.append(ip)
    parts.append(types.Part.from_text(text="NARRATION LINE:\n" + (line or "")))
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.0, response_mime_type="application/json", response_schema=JUDGE_SCHEMA),
        )
        return json.loads(resp.text)
    except Exception as e:
        print(f"  judge error: {e}", file=sys.stderr)
        return None


def _run_beats(args, extra: List[str]) -> None:
    cmd = [
        sys.executable, os.path.join(THIS_DIR, "gemini_narrative_pass.py"),
        "--groups-manifest", args.groups_manifest,
        "--vision-manifest", args.vision_manifest,
        "--out", args.out,
        "--project", args.project, "--location", args.location,
        "--model", args.model,
    ] + (["--cast", args.cast] if args.cast else []) + extra
    print("  $ " + " ".join(os.path.basename(c) if c.endswith(".py") else c for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-manifest", required=True)
    ap.add_argument("--vision-manifest", required=True)
    ap.add_argument("--out", required=True, help="beats output (manifest.beats.*.json)")
    ap.add_argument("--cast", default="", help="manifest.cast.json (built if missing)")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    ap.add_argument("--model", default="gemini-2.5-flash", help="beats/narration model")
    ap.add_argument("--judge-model", default="gemini-2.5-flash-lite")
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--max-images-per-group", type=int, default=2)
    args = ap.parse_args()

    if not args.project:
        keys = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if keys and os.path.exists(keys):
            args.project = json.loads(open(keys).read()).get("project_id", "")
    if not args.project:
        raise SystemExit("No --project and none derivable from GOOGLE_APPLICATION_CREDENTIALS")

    # 1) cast (build if absent)
    if args.cast and not os.path.exists(args.cast):
        print(f"[cast] building {args.cast} …")
        subprocess.run([
            sys.executable, os.path.join(THIS_DIR, "cast_builder.py"),
            "--groups-manifest", args.groups_manifest, "--vision-manifest", args.vision_manifest,
            "--out", args.cast, "--project", args.project, "--location", args.location,
        ], check=True)

    # 2) initial cast-aware narration
    print("[beats] generating cast-aware narration …")
    _run_beats(args, [])

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    vision = _load(args.vision_manifest)
    scene_paths = _scene_paths(vision)

    corrections_path = os.path.join(os.path.dirname(args.out), "_narration_corrections.json")
    target_gids: Optional[set] = None  # None on round 0 = judge all; later = only the regenerated

    for rnd in range(args.max_rounds + 1):
        beats = _load(args.out)
        by_gid = {int(b.get("group_id") or 0): b for b in beats.get("beats") or []}
        gids = sorted(by_gid) if target_gids is None else sorted(target_gids)
        print(f"\n[judge] round {rnd}: judging {len(gids)} line(s) …")

        corrections: Dict[str, str] = {}
        verdicts = {"grounded": 0, "minor_drift": 0, "hallucination": 0}
        for gid in gids:
            b = by_gid.get(gid) or {}
            line = (b.get("narration") or "").strip()
            imgs = _kept_images(b, scene_paths, args.max_images_per_group)
            if not line or not imgs:
                continue
            ev = _judge_line(client, args.judge_model, imgs, line)
            if not ev:
                continue
            v = (ev.get("verdict") or "").lower()
            verdicts[v] = verdicts.get(v, 0) + 1
            if v == "hallucination" and ev.get("invented"):
                corrections[str(gid)] = "; ".join(ev["invented"])[:300]
        print(f"  verdicts: {verdicts}")

        if not corrections:
            print(f"[done] no event-hallucinations left after round {rnd}.")
            break
        if rnd == args.max_rounds:
            print(f"[stop] {len(corrections)} still flagged at max-rounds; leaving as-is: {list(corrections)}")
            break

        with open(corrections_path, "w", encoding="utf-8") as f:
            json.dump(corrections, f, ensure_ascii=False, indent=2)
        print(f"[regen] round {rnd}: fixing groups {list(corrections)} …")
        _run_beats(args, ["--resume", "--corrections", corrections_path])
        target_gids = {int(g) for g in corrections}

    if os.path.exists(corrections_path):
        os.remove(corrections_path)
    print(f"\n[ok] grounded narration -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
