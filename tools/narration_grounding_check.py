"""
tools/narration_grounding_check.py — judge narration lines against the ACTUAL panels.

Given a chapter's beats (Variant B = image-grounded `narration`) and a polished
script (Variant C = openai_polish `tts_paragraphs_v3`), this asks a multimodal
model (Gemini flash-lite, which SEES the panels) whether each line asserts
anything NOT visible in the art, scores fidelity + prose, and renders a
side-by-side `narration_compare.html`.

This is the objective instrument for the narration A/B: hallucination shows up as
"ungrounded_claims" with a low fidelity score, regardless of how good the prose reads.

Auth mirrors gemini_narrative_pass: Vertex AI via the gcp-vision service-account
key (project = the key's own project_id), no gcloud needed.

  V=.eval_venv/bin/python
  $V tools/narration_grounding_check.py \
      --beats ongoing/nano-machine/Chapter_1/manifest.beats.narr.json \
      --script-c ongoing/nano-machine/Chapter_1/manifest.script.C.json \
      --vision ongoing/nano-machine/Chapter_1/manifest.vision.json \
      --out ongoing/nano-machine/Chapter_1/narration_compare.html
"""

import argparse
import html
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_adapter():
    """Reuse local_tts_from_manifest's tts_v3 extractor (segment_id -> group_id, text)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lt", os.path.join(TOOLS_DIR, "local_tts_from_manifest.py"))
    lt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lt)
    return lt


def _scene_path_map(vision: Dict[str, Any]) -> Dict[str, str]:
    """scene_file -> on-disk scene_path, from the vision manifest."""
    out: Dict[str, str] = {}
    items = vision.get("items") or vision.get("scenes") or []
    if isinstance(items, dict):
        items = list(items.values())
    for it in items:
        sf = it.get("scene_file") or it.get("file") or it.get("name")
        sp = it.get("scene_path") or it.get("path")
        if sf and sp:
            out[str(sf)] = str(sp)
    return out


def _b_lines_by_group(beats: Dict[str, Any]) -> Dict[int, str]:
    """Variant B: each beat's image-grounded `narration`, keyed by group_id."""
    out: Dict[int, str] = {}
    for bt in beats.get("beats") or []:
        gid = int(bt.get("group_id") or 0)
        out[gid] = (bt.get("narration") or "").strip()
    return out


def _kept_images_by_group(beats: Dict[str, Any], scene_paths: Dict[str, str],
                          max_images: int = 2) -> Dict[int, List[str]]:
    """Per group, the on-disk paths of panels the selector marked role=='keep'."""
    out: Dict[int, List[str]] = {}
    for bt in beats.get("beats") or []:
        gid = int(bt.get("group_id") or 0)
        sel = bt.get("scene_selection") or []
        kept = [s.get("scene_file") for s in sel if s.get("role") == "keep" and s.get("scene_file")]
        if not kept:
            kept = bt.get("scene_files") or []
        paths = [scene_paths[str(sf)] for sf in kept if str(sf) in scene_paths]
        out[gid] = paths[:max_images]
    return out


def _c_lines_by_group(script_c: Dict[str, Any], lt) -> Dict[int, str]:
    """Variant C: openai_polish tts paragraphs, joined per group, mood tag stripped."""
    out: Dict[int, List[str]] = {}
    for it in lt.extract_items_from_manifest(script_c, "tts_v3"):
        gid = int(it.get("group_id") or 0)
        out.setdefault(gid, []).append(lt.strip_bracket_tags(it.get("text") or ""))
    return {g: " ".join(v).strip() for g, v in out.items()}


JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},                 # "line_1" | "line_2"
                    "fidelity": {"type": "INTEGER"},           # 1..5 grounded in the art
                    "prose": {"type": "INTEGER"},              # 1..5 reads like good narration
                    "ungrounded_claims": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "verdict": {"type": "STRING"},             # grounded|minor_drift|hallucination
                },
                "required": ["id", "fidelity", "prose", "ungrounded_claims", "verdict"],
            },
        },
        "which_better": {"type": "STRING"},                    # line_1|line_2|tie
        "why": {"type": "STRING"},
    },
    "required": ["candidates", "which_better", "why"],
}

JUDGE_SYSTEM = (
    "You are a strict fact-checker for video narration. You are shown the ACTUAL manhwa "
    "panel image(s) for one story beat, plus two candidate narration lines. For EACH line:\n"
    "  - fidelity (1-5): is every action, motion, entity, and event it asserts actually "
    "VISIBLE in these panels? 5 = fully grounded; 1 = invents things not shown.\n"
    "  - List ungrounded_claims: specific phrases that assert something NOT visible (e.g. a "
    "teleport, a motion, a character, an outcome the panel doesn't show). Empty list if none.\n"
    "  - prose (1-5): does it read like flowing cinematic narration (5) or a flat caption / "
    "clunky description like 'is present', 'reacts with' (1)?\n"
    "  - verdict: 'grounded' (no invented content), 'minor_drift' (small unsupported flourish), "
    "or 'hallucination' (invents a real event/action/entity).\n"
    "Judge ONLY against what the image shows. Do not reward dramatic writing that isn't supported. "
    "Then pick which_better overall (fidelity first, prose as tiebreak). Return ONLY JSON."
)


def _img_part(path: str) -> Optional[types.Part]:
    try:
        with open(path, "rb") as f:
            return types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
    except Exception:
        return None


def _judge_group(client: genai.Client, model: str, images: List[str],
                 line_b: str, line_c: str) -> Optional[Dict[str, Any]]:
    parts: List[types.Part] = [types.Part.from_text(text=JUDGE_SYSTEM)]
    for p in images:
        ip = _img_part(p)
        if ip is not None:
            parts.append(ip)
    payload = {"line_1": line_b, "line_2": line_c}  # line_1=Variant B, line_2=Variant C
    parts.append(types.Part.from_text(text="CANDIDATES:\n" + json.dumps(payload, ensure_ascii=False)))
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=JUDGE_SCHEMA,
                ),
            )
            return json.loads(resp.text)
        except Exception as e:
            if attempt == 2:
                print(f"  judge failed: {e}", file=sys.stderr)
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def _verdict_badge(v: str) -> str:
    v = (v or "").lower()
    color = {"grounded": "#1a7f37", "minor_drift": "#9a6700", "hallucination": "#cf222e"}.get(v, "#57606a")
    return f'<span style="background:{color};color:#fff;padding:1px 7px;border-radius:10px;font-size:12px">{html.escape(v or "?")}</span>'


def _cell(label: str, line: str, ev: Optional[Dict[str, Any]]) -> str:
    if not ev:
        return f"<td><b>{label}</b><br><i>{html.escape(line or '(none)')}</i><br><small>no verdict</small></td>"
    claims = ev.get("ungrounded_claims") or []
    claims_html = "".join(f"<li>{html.escape(c)}</li>" for c in claims) or "<li><i>none</i></li>"
    return (
        f"<td><b>{label}</b> &nbsp; {_verdict_badge(ev.get('verdict'))}"
        f"<br>fidelity <b>{ev.get('fidelity','?')}/5</b> &nbsp; prose <b>{ev.get('prose','?')}/5</b>"
        f"<p style='margin:6px 0'>{html.escape(line or '(none)')}</p>"
        f"<small>ungrounded:</small><ul style='margin:2px 0'>{claims_html}</ul></td>"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", required=True, help="beats manifest WITH `narration` (Variant B)")
    ap.add_argument("--script-c", required=True, help="openai_polish script manifest (Variant C)")
    ap.add_argument("--vision", required=True, help="vision manifest (scene_path map)")
    ap.add_argument("--out", required=True, help="output HTML report")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    ap.add_argument("--max-images-per-group", type=int, default=2)
    ap.add_argument("--limit-groups", type=int, default=0, help="0 = all (debug: cap groups)")
    args = ap.parse_args()

    project = args.project
    if not project:
        keys = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if keys and os.path.exists(keys):
            project = json.loads(open(keys).read()).get("project_id", "")
    if not project:
        raise SystemExit("No --project and could not derive project_id from GOOGLE_APPLICATION_CREDENTIALS")

    lt = _load_adapter()
    beats = _load(args.beats)
    script_c = _load(args.script_c)
    vision = _load(args.vision)

    scene_paths = _scene_path_map(vision)
    b_lines = _b_lines_by_group(beats)
    c_lines = _c_lines_by_group(script_c, lt)
    images = _kept_images_by_group(beats, scene_paths, args.max_images_per_group)

    groups = sorted(set(b_lines) | set(c_lines))
    if args.limit_groups:
        groups = groups[: args.limit_groups]

    client = genai.Client(vertexai=True, project=project, location=args.location)

    rows: List[str] = []
    tally = {"line_1": 0, "line_2": 0, "tie": 0}
    fid = {"B": 0, "C": 0}
    n = 0
    for gid in groups:
        lb, lc = b_lines.get(gid, ""), c_lines.get(gid, "")
        if not lb and not lc:
            continue
        imgs = images.get(gid, [])
        print(f"[grp {gid}] {len(imgs)} img …")
        ev = _judge_group(client, args.model, imgs, lb, lc) if imgs else None
        by_id = {c.get("id"): c for c in (ev.get("candidates") if ev else []) or []}
        eb, ec = by_id.get("line_1"), by_id.get("line_2")
        if ev:
            tally[ev.get("which_better", "tie")] = tally.get(ev.get("which_better", "tie"), 0) + 1
            if eb:
                fid["B"] += int(eb.get("fidelity") or 0)
            if ec:
                fid["C"] += int(ec.get("fidelity") or 0)
            n += 1
        thumbs = "".join(
            f'<img src="{html.escape(os.path.relpath(p, os.path.dirname(args.out)))}" '
            f'style="height:120px;margin:2px;border:1px solid #ccc">' for p in imgs)
        rows.append(
            f"<tr><td valign=top><b>group {gid}</b><br>{thumbs or '<i>no img</i>'}</td>"
            f"{_cell('B · Gemini (grounded)', lb, eb)}{_cell('C · OpenAI polish', lc, ec)}"
            f"<td valign=top>{_verdict_badge(ev.get('which_better','?')) if ev else '?'}"
            f"<br><small>{html.escape((ev or {}).get('why',''))}</small></td></tr>"
        )

    avg_b = fid["B"] / n if n else 0
    avg_c = fid["C"] / n if n else 0
    summary = (
        f"<p><b>{n} groups judged.</b> Better line — "
        f"Variant B (Gemini): <b>{tally.get('line_1',0)}</b> · "
        f"Variant C (OpenAI polish): <b>{tally.get('line_2',0)}</b> · tie: {tally.get('tie',0)}."
        f"<br>Mean fidelity vs art — B: <b>{avg_b:.2f}/5</b> · C: <b>{avg_c:.2f}/5</b> "
        f"(higher = fewer invented events).</p>"
    )
    doc = (
        "<!doctype html><meta charset=utf-8><title>Narration grounding A/B</title>"
        "<style>body{font:14px/1.5 system-ui;margin:24px;max-width:1400px}"
        "table{border-collapse:collapse;width:100%}td{border:1px solid #ddd;padding:8px;vertical-align:top;width:30%}"
        "td:first-child{width:10%}h1{font-size:20px}</style>"
        f"<h1>Narration grounding A/B — judged against the actual panels</h1>{summary}"
        "<table><tr><th>panel</th><th>Variant B (Gemini, sees art)</th>"
        "<th>Variant C (OpenAI polish)</th><th>winner</th></tr>"
        + "".join(rows) + "</table>"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\n[ok] {args.out}")
    print(summary.replace("<b>", "").replace("</b>", "").replace("<br>", " ").replace("<p>", "").replace("</p>", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
