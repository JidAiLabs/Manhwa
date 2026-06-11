#!/usr/bin/env python3
"""
ab_beats_report.py — side-by-side narration A/B (e.g. Gemini vs local Gemma).

For every group: panel thumbnails + narration A vs narration B + the
keep/redundant selection diff. One self-contained HTML for the quality
verdict before any backend transition.

Usage:
  python tools/ab_beats_report.py --episode-dir ongoing/<series>/<chapter> \
      --beats-a manifest.beats.json --label-a "Gemini 2.5 Flash" \
      --beats-b manifest.beats.gemma.json --label-b "Gemma 4 26B (local)" \
      [--out render/ab_gemma_report.html]
"""

from __future__ import annotations

import argparse
import base64
import html as _html
import json
import os
from typing import Any, Dict, List


def _beats_by_group(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return {int(b.get("group_id") or 0): b
                for b in json.load(f).get("beats") or []}


def _sel_summary(beat: Dict[str, Any]) -> str:
    sel = beat.get("scene_selection") or []
    keep = sum(1 for s in sel if str(s.get("role") or "keep") == "keep")
    return f"{keep}/{len(sel)} kept" if sel else "—"


def _thumb_tag(path: str, max_w: int = 150) -> str:
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        img = cv2.resize(img, (max_w, min(int(h * max_w / max(1, w)), 420)))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return (f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="max-width:{max_w}px;margin:2px">')
    except Exception:
        return ""


def render(episode_dir: str, beats_a: str, beats_b: str,
           label_a: str, label_b: str) -> str:
    A = _beats_by_group(os.path.join(episode_dir, beats_a)
                        if not os.path.isabs(beats_a) else beats_a)
    B = _beats_by_group(os.path.join(episode_dir, beats_b)
                        if not os.path.isabs(beats_b) else beats_b)
    rows: List[str] = []
    for gid in sorted(set(A) | set(B)):
        a, b = A.get(gid, {}), B.get(gid, {})
        files = (a.get("scene_files") or b.get("scene_files") or [])[:4]
        thumbs = "".join(_thumb_tag(os.path.join(episode_dir, "scenes", f))
                         for f in files)
        rows.append(f"""<tr>
<td style="white-space:nowrap"><b>g{gid:04d}</b><br>
<small>{_html.escape(_sel_summary(a))} | {_html.escape(_sel_summary(b))}</small></td>
<td>{thumbs}</td>
<td style="width:30%">{_html.escape(str(a.get('narration') or '—'))}</td>
<td style="width:30%">{_html.escape(str(b.get('narration') or '—'))}</td>
</tr>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>beats A/B — {_html.escape(os.path.basename(episode_dir))}</title>
<style>body{{font-family:-apple-system,Helvetica;margin:24px;background:#fafafa}}
table{{border-collapse:collapse;width:100%;background:#fff}}
td,th{{border:1px solid #ddd;padding:8px;vertical-align:top;text-align:left}}
th{{background:#263238;color:#fff;position:sticky;top:0}}</style></head><body>
<h1>Narration A/B — {_html.escape(label_a)} vs {_html.escape(label_b)}</h1>
<table><tr><th>group<br><small>kept A|B</small></th><th>panels</th>
<th>{_html.escape(label_a)}</th><th>{_html.escape(label_b)}</th></tr>
{''.join(rows)}</table></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--beats-a", default="manifest.beats.json")
    ap.add_argument("--beats-b", default="manifest.beats.gemma.json")
    ap.add_argument("--label-a", default="Gemini 2.5 Flash")
    ap.add_argument("--label-b", default="Gemma 4 26B (local)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = args.out or os.path.join(args.episode_dir, "render",
                                   "ab_gemma_report.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    html = render(args.episode_dir, args.beats_a, args.beats_b,
                  args.label_a, args.label_b)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
