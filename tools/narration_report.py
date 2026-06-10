"""
tools/narration_report.py — view the final recap narration against its panels.

Per group: the kept panel thumbnails + the narration line that will be voiced.
No judging/enforcement here (the grounding gate is a separate tool) — this is the
"read it next to the art" QA view.

  V=.eval_venv/bin/python
  $V tools/narration_report.py \
      --beats  ongoing/nano-machine/Chapter_1/manifest.beats.narr.json \
      --vision ongoing/nano-machine/Chapter_1/manifest.vision.json \
      --out    ongoing/nano-machine/Chapter_1/narration_report.html
"""

import argparse
import html
import json
import os
from typing import Any, Dict, List


def _load(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _scene_paths(vision: Dict[str, Any]) -> Dict[str, str]:
    items = vision.get("items") or vision.get("scenes") or []
    if isinstance(items, dict):
        items = list(items.values())
    out: Dict[str, str] = {}
    for it in items:
        sf, sp = it.get("scene_file"), it.get("scene_path")
        if sf and sp:
            out[str(sf)] = str(sp)
    return out


def _kept(beat: Dict[str, Any]) -> List[str]:
    sel = beat.get("scene_selection") or []
    kept = [s.get("scene_file") for s in sel if s.get("role") == "keep" and s.get("scene_file")]
    return kept or (beat.get("scene_files") or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", required=True)
    ap.add_argument("--vision", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    beats = _load(args.beats)
    sp = _scene_paths(_load(args.vision))
    outdir = os.path.dirname(args.out)

    rows: List[str] = []
    nquote = nnamed = 0
    members = ("Prince Cheon", "Ancestor", "Assassin", "Nano")
    bs = sorted(beats.get("beats") or [], key=lambda x: int(x.get("group_id") or 0))
    for b in bs:
        gid = b.get("group_id")
        line = (b.get("narration") or "").strip()
        if ('"' in line) or ("'" in line):
            nquote += 1
        if any(m in line for m in members):
            nnamed += 1
        thumbs = ""
        for s in _kept(b):
            p = sp.get(str(s))
            if p:
                rel = os.path.relpath(p, outdir)
                thumbs += f'<img src="{html.escape(rel)}" style="height:150px;margin:2px;border:1px solid #ccc">'
        rows.append(
            f"<tr><td valign=top><b>grp {gid}</b><br>{html.escape(b.get('beat_title') or '')}"
            f"<br>{thumbs or '<i>no img</i>'}</td>"
            f"<td valign=top style='font-size:15px;line-height:1.6'>{html.escape(line) or '<i>(empty)</i>'}</td></tr>"
        )

    n = len(bs)
    doc = (
        "<!doctype html><meta charset=utf-8><title>Recap narration vs panels</title>"
        "<style>body{font:14px/1.5 system-ui;margin:24px;max-width:1200px}"
        "table{border-collapse:collapse;width:100%}td{border:1px solid #ddd;padding:10px;vertical-align:top}"
        "td:first-child{width:46%}h1{font-size:20px}</style>"
        f"<h1>Recap narration vs panels — {n} groups</h1>"
        f"<p>Cast-aware + dialogue-woven narration. Quoting dialogue: <b>{nquote}/{n}</b>; "
        f"naming cast: <b>{nnamed}/{n}</b>.</p>"
        "<table><tr><th>panels</th><th>narration (voiced line)</th></tr>"
        + "".join(rows) + "</table>"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[ok] {args.out} — {n} groups, {nquote} quote dialogue, {nnamed} name cast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
