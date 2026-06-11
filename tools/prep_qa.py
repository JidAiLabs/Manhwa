#!/usr/bin/env python3
"""
prep_qa.py — pre-render QA scanner (the QA-first instrument).

Scans the PREPPED artifacts — render.plan.clean.json + scenes_clean/ — i.e.
exactly what the renderer will show, and flags every known defect class
BEFORE any render is started:

  image:      husk (no art after cleaning), dead_box_leak (blank caption
              voids dominating the frame), ghost_text / visible_text inside
              blanked bubbles, binary_card (near-binary chrome cards),
              stale_dims (plan dims != file on disk), extreme_tall
  vision:     chrome_leak (publication chrome shown as story),
              doc_flag_missing (text-rich panel without doc protection)
  narration:  chrome_narration (credits/counters/markers narrated),
              ocr_echo (narration repeats on-page text)
  plan:       missing_file / missing_dims / missing_audio, empty_item,
              flash_cut, repeat_cut, cut_gap, no_cold_open, branding

Emits a console summary + JSON + self-contained HTML report (base64
thumbnails for every flagged scene). Exit code 1 when any ERROR-severity
flag is present, else 0.

Usage:
  python tools/prep_qa.py --episode-dir ongoing/<series>/<chapter> \
      --series-title "Nano Machine" [--no-detector] [--device mps]
"""

from __future__ import annotations

import argparse
import base64
import html as _html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
for _p in (_TOOLS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import render_prep as rp                      # art/bubble metrics, detector
from scene_chrome import is_chrome_scene      # chrome rules (single source)
from studio.qa_flags import longest_common_run

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"
_SEV_RANK = {ERROR: 0, WARN: 1, INFO: 2}

# narration that mentions publication chrome is narrating a cover/credits/
# counter panel — the beats prompt forbids it, this is the independent check
_CHROME_NARR_RE = re.compile(
    r"\b(redice|asura\s*(?:scans?|toon)?|elftoon|webtoons?|naver|kakao|"
    r"tapas|tappytoon|scanlat\w*|translat(?:or|ion|ed\s+by)\w*|proofread\w*|"
    r"typeset\w*|raw\s+provider|presented\s+by|patreon|discord|subscribe\w*|"
    r"views?\s*[:=]|likes?\s*[:=]|view\s+count\w*|"
    r"(?:chapter|episode)\s+\d+)\b",
    re.IGNORECASE)


def _flag(code: str, severity: str, detail: str, *,
          scene: str = "", segment_id: str = "") -> Dict[str, Any]:
    return {"code": code, "severity": severity, "detail": detail,
            "scene": scene, "segment_id": segment_id}


# ---------------------------------------------------------------------------
# plan walking
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"_(?:a|b)(?=\.[A-Za-z0-9]+$)")


def parent_scene(fname: str) -> str:
    """Map split2 parts (p000031_a.jpg) back to their source scene name."""
    return _SPLIT_RE.sub("", fname)


def iter_shown_cuts(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every (segment, file) actually displayed, split2 parts included."""
    out: List[Dict[str, Any]] = []
    for item in plan.get("timeline") or []:
        seg = str(item.get("segment_id") or "")
        branding = bool(item.get("branding"))
        for idx, c in enumerate(item.get("cuts") or []):
            for f in (c.get("file"), c.get("file2")):
                if f:
                    out.append({"segment_id": seg, "file": str(f), "idx": idx,
                                "dur": float(c.get("dur") or 0.0),
                                "branding": branding})
    return out


# ---------------------------------------------------------------------------
# image metrics
# ---------------------------------------------------------------------------

def _glyph_count(ink: np.ndarray) -> int:
    """Connected components that are glyph-sized — text is MANY small blobs,
    an art stroke crossing a white area is one big one."""
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    glyphs = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 8 <= area <= 600:
            glyphs += 1
    return glyphs


def box_interior_stats(img: np.ndarray,
                       box: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """What does the viewer see inside a detected bubble/caption box?

    blank      — interior is a near-uniform white (or black) VOID (no ink)
    ghost_frac — faint not-quite-background remnants (failed text blanking)
    ink_frac   — crisp glyph-strength pixels (text never blanked at all)
    ink_glyphs — glyph-sized ink components (distinguishes text from art)
    """
    gray = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    dx = max(4, int(0.12 * (x2 - x1)))
    dy = max(4, int(0.12 * (y2 - y1)))
    g = gray[max(0, y1 + dy):min(h, y2 - dy), max(0, x1 + dx):min(w, x2 - dx)]
    zero = {"blank": False, "white_frac": 0.0, "black_frac": 0.0,
            "ghost_frac": 0.0, "ink_frac": 0.0, "ink_glyphs": 0,
            "area_frac": 0.0}
    if g.size == 0:
        return zero
    white = float((g >= 235).mean())
    black = float((g <= 25).mean())
    st = dict(zero, white_frac=white, black_frac=black,
              area_frac=float((x2 - x1) * (y2 - y1)) / float(max(1, h * w)))
    if white >= black:
        ink = g <= 120
        st["ghost_frac"] = float(((g >= 140) & (g < 235)).mean())
        st["ink_frac"] = float(ink.mean())
        st["blank"] = white >= 0.70 and st["ink_frac"] < 0.03
    else:
        ink = g >= 180
        st["ghost_frac"] = float(((g > 25) & (g <= 120)).mean())
        st["ink_frac"] = float(ink.mean())
        st["blank"] = black >= 0.70 and st["ink_frac"] < 0.03
    st["ink_glyphs"] = _glyph_count(ink) if st["ink_frac"] >= 0.01 else 0
    return st


def image_flags(
    name: str,
    img: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    doc: bool,
    dims_entry: Optional[Dict[str, Any]],
    sys: bool = False,
    segment_id: str = "",
    min_art_score: float = 0.012,
) -> List[Dict[str, Any]]:
    """All image-level checks for one shown scenes_clean/ file.

    *doc* (document/UI) and *sys* (system-message) panels keep their text BY
    DESIGN — content checks (husk/card/void/text) do not apply to them."""
    flags: List[Dict[str, Any]] = []
    h, w = img.shape[:2]

    if dims_entry and (int(dims_entry.get("w", -1)) != w
                       or int(dims_entry.get("h", -1)) != h):
        flags.append(_flag(
            "stale_dims", ERROR,
            f"plan says {dims_entry.get('w')}x{dims_entry.get('h')}, file is "
            f"{w}x{h} — scenes_clean/ and plan are out of sync",
            scene=name, segment_id=segment_id))

    if h >= 6 * max(1, w):
        flags.append(_flag("extreme_tall", INFO,
                           f"aspect h/w={h / max(1, w):.1f} — scroll shot; "
                           "verify travel speed is watchable",
                           scene=name, segment_id=segment_id))

    if not doc and not sys:
        gray = img.mean(axis=2) if img.ndim == 3 else img
        art = rp.art_content_score(img, [])
        if art < min_art_score:
            sev = ERROR if art < 0.7 * min_art_score else WARN
            flags.append(_flag("husk", sev,
                               f"art_score={art:.4f} < {min_art_score} — "
                               + ("no art detail left after cleaning"
                                  if sev == ERROR else
                                  "borderline art detail, eyeball it"),
                               scene=name, segment_id=segment_id))
        midtone = float(((gray > 60) & (gray < 200)).mean())
        if midtone < 0.08:
            flags.append(_flag("binary_card", WARN,
                               f"midtone_frac={midtone:.3f} — near-binary "
                               "card (chrome-like), verify it is story",
                               scene=name, segment_id=segment_id))

        stats = [(b, box_interior_stats(img, b)) for b in boxes]
        blank_boxes = [b for b, st in stats if st["blank"]]
        blank_frac = rp.bubble_coverage((h, w), blank_boxes)
        if blank_frac >= 0.35:
            flags.append(_flag("dead_box_leak", ERROR,
                               f"blank_box_frac={blank_frac:.2f} — blanked "
                               "caption voids dominate the frame (should "
                               "have been recropped or dropped)",
                               scene=name, segment_id=segment_id))
        ghost = max([st["ghost_frac"] for _, st in stats
                     if st["blank"] and st["area_frac"] >= 0.02],
                    default=0.0)
        if ghost >= 0.03:
            flags.append(_flag("ghost_text", WARN,
                               f"ghost_frac={ghost:.3f} — faint text "
                               "remnants inside a blanked bubble",
                               scene=name, segment_id=segment_id))
        ink_hits = [st for _, st in stats
                    if st["white_frac"] >= 0.35 and st["area_frac"] >= 0.02
                    and st["ink_frac"] >= 0.05 and st["ink_glyphs"] >= 6]
        if ink_hits:
            top = max(ink_hits, key=lambda s: s["ink_frac"])
            flags.append(_flag("visible_text", ERROR,
                               f"ink_frac={top['ink_frac']:.3f} "
                               f"({top['ink_glyphs']} glyphs) — bubble text "
                               "still readable (blanking missed it)",
                               scene=name, segment_id=segment_id))
    return flags


# ---------------------------------------------------------------------------
# vision / narration / plan checks
# ---------------------------------------------------------------------------

def vision_flags(parent: str, vitem: Dict[str, Any], *,
                 dims_entry: Optional[Dict[str, Any]],
                 series_title: Optional[str],
                 segment_id: str = "") -> List[Dict[str, Any]]:
    d = dims_entry or {}
    flags: List[Dict[str, Any]] = []
    if is_chrome_scene(vitem, series_title=series_title):
        flags.append(_flag("chrome_leak", ERROR,
                           f"chrome per scene_chrome rules is SHOWN — "
                           f"ocr={str(vitem.get('ocr_clean'))[:80]!r}",
                           scene=parent, segment_id=segment_id))
    text_rich = (float(vitem.get("text_coverage") or 0.0) >= 0.22
                 or int(vitem.get("n_words") or 0) >= 15)
    unprotected = (not d.get("doc") and not d.get("sys")
                   and not d.get("blanked", False))
    if text_rich and unprotected:
        # wordy text that will RENDER (not blanked) without doc protection —
        # blanked dialogue panels have nothing left to protect
        flags.append(_flag("doc_flag_missing", WARN,
                           "text-rich panel lacks doc protection — renderer "
                           "may cover-crop or scroll its text",
                           scene=parent, segment_id=segment_id))
    return flags


def narration_flags(segment_id: str, narration: str,
                    panels: Sequence[Any]) -> List[Dict[str, Any]]:
    """*panels*: dicts {"ocr", "visible"} (bare strings mean visible=True).
    Echo is only a defect when the echoed text is STILL ON SCREEN — narration
    quoting a BLANKED bubble is the design (it replaces the text)."""
    flags: List[Dict[str, Any]] = []
    text = narration or ""
    m = _CHROME_NARR_RE.search(text)
    if m:
        flags.append(_flag("chrome_narration", WARN,
                           f"narration mentions chrome ({m.group(0)!r}): "
                           f"{text[:90]!r}",
                           segment_id=segment_id))
    for p in panels:
        if isinstance(p, str):
            ocr, visible = p, True
        else:
            ocr, visible = str(p.get("ocr") or ""), bool(p.get("visible"))
        if not visible:
            continue
        run = longest_common_run(text, ocr, min_words=4)
        if run:
            flags.append(_flag("ocr_echo", WARN,
                               f"narration repeats on-page VISIBLE text: "
                               f"{run!r}",
                               segment_id=segment_id))
            break
    return flags


def plan_flags(plan: Dict[str, Any], *, clean_files: set,
               audio_exists: Callable[[str], bool]) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    timeline = plan.get("timeline") or []
    dims = plan.get("scene_dims") or {}

    if timeline and timeline[0].get("branding"):
        flags.append(_flag("no_cold_open", WARN,
                           "video starts with the branding intro — no story "
                           "cold-open hook before it",
                           segment_id=str(timeline[0].get("segment_id"))))
    brandings = {str(i.get("branding")) for i in timeline if i.get("branding")}
    if not brandings:
        flags.append(_flag("no_branding", INFO,
                           "no intro/outro branding items in plan"))
    elif brandings != {"intro", "outro"}:
        flags.append(_flag("missing_branding", WARN,
                           f"branding items present: {sorted(brandings)} — "
                           "expected intro AND outro"))

    seen_parent_segments: Dict[str, set] = {}
    for item in timeline:
        seg = str(item.get("segment_id") or "")
        cuts = item.get("cuts") or []
        branding = bool(item.get("branding"))
        if not cuts:
            if item.get("branding") == "outro":
                continue  # the renderer draws its own end-card for the outro
            flags.append(_flag("empty_item", ERROR,
                               "timeline item has no cuts (nothing on "
                               "screen for its whole duration)",
                               segment_id=seg))
            continue

        prev_file = None
        for c in cuts:
            for f in (c.get("file"), c.get("file2")):
                if not f:
                    continue
                f = str(f)
                if f not in clean_files:
                    flags.append(_flag("missing_file", ERROR,
                                       "cut references a file missing from "
                                       "scenes_clean/",
                                       scene=f, segment_id=seg))
                if f not in dims:
                    flags.append(_flag("missing_dims", ERROR,
                                       "shown file absent from scene_dims — "
                                       "renderer cannot fit it",
                                       scene=f, segment_id=seg))
                if not branding:
                    seen_parent_segments.setdefault(
                        parent_scene(f), set()).add(seg)
            dur = float(c.get("dur") or 0.0)
            if dur < 1.2:
                flags.append(_flag("flash_cut", WARN,
                                   f"cut shows {c.get('file')} for only "
                                   f"{dur:.2f}s",
                                   scene=str(c.get("file") or ""),
                                   segment_id=seg))
            if c.get("file") == prev_file:
                flags.append(_flag("repeat_cut", WARN,
                                   "same file in consecutive cuts",
                                   scene=str(c.get("file")), segment_id=seg))
            prev_file = c.get("file")

        tile = sum(float(c.get("dur") or 0.0) for c in cuts)
        item_dur = float(item.get("duration_sec") or 0.0)
        if abs(tile - item_dur) > 0.51:
            flags.append(_flag("cut_gap", WARN,
                               f"cuts tile {tile:.2f}s of a {item_dur:.2f}s "
                               "item (gap or overlap on screen)",
                               segment_id=seg))

        if not branding:
            audio = item.get("tts_audio")
            if not audio or not audio_exists(str(audio)):
                flags.append(_flag("missing_audio", ERROR,
                                   f"tts_audio missing on disk: {audio}",
                                   segment_id=seg))

    for parent, segs in seen_parent_segments.items():
        if len(segs) > 1:
            flags.append(_flag("reshow", INFO,
                               f"scene shown in {len(segs)} segments: "
                               f"{sorted(segs)}",
                               scene=parent))

    total = float(plan.get("total_duration_sec") or 0.0)
    s = sum(float(i.get("duration_sec") or 0.0) for i in timeline)
    if abs(total - s) > 0.75:
        flags.append(_flag("total_drift", WARN,
                           f"total_duration_sec={total:.2f} but items sum "
                           f"to {s:.2f}"))
    return flags


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def build_report(title: str, flags: List[Dict[str, Any]], *,
                 n_cuts: int) -> Dict[str, Any]:
    counts = {ERROR: 0, WARN: 0, INFO: 0}
    for f in flags:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    ordered = sorted(flags, key=lambda f: (_SEV_RANK.get(f["severity"], 9),
                                           f.get("scene") or "~",
                                           f.get("segment_id") or "",
                                           f["code"]))
    return {"title": title, "n_cuts": n_cuts, "counts": counts,
            "flags": ordered}


_SEV_COLOR = {ERROR: "#c62828", WARN: "#ef6c00", INFO: "#546e7a"}


def _img_tag(thumbs: Dict[str, bytes], scene: str, max_w: int = 240) -> str:
    if scene not in thumbs:
        return ""
    b64 = base64.b64encode(thumbs[scene]).decode("ascii")
    return (f'<img src="data:image/jpeg;base64,{b64}" '
            f'style="max-width:{max_w}px;max-height:260px">')


def render_html(report: Dict[str, Any],
                thumbs: Optional[Dict[str, bytes]] = None,
                gallery: Optional[List[Dict[str, str]]] = None) -> str:
    thumbs = thumbs or {}
    c = report["counts"]
    rows: List[str] = []
    for f in report["flags"]:
        scene = f.get("scene") or ""
        img_tag = _img_tag(thumbs, scene or str(f.get("thumb_scene") or ""))
        color = _SEV_COLOR.get(f["severity"], "#000")
        rows.append(
            "<tr>"
            f'<td><b style="color:{color}">{f["severity"]}</b></td>'
            f"<td><code>{_html.escape(f['code'])}</code></td>"
            f"<td>{_html.escape(scene)}</td>"
            f"<td>{_html.escape(f.get('segment_id') or '')}</td>"
            f"<td>{_html.escape(f['detail'])}</td>"
            f"<td>{img_tag}</td></tr>")
    flags_html = (f"""<table><tr><th>sev</th><th>flag</th><th>scene</th>
<th>segment</th><th>detail</th><th>thumb</th></tr>{''.join(rows)}</table>"""
                  if rows else "<p><b>All clean — no flags.</b></p>")

    gallery_html = ""
    if gallery:
        cells = []
        for g in gallery:
            fn = str(g.get("file") or "")
            seg = str(g.get("segment_id") or "")
            cells.append(
                '<figure style="margin:4px;display:inline-block;'
                'text-align:center;background:#fff;border:1px solid #ddd;'
                'padding:4px">'
                f"{_img_tag(thumbs, fn, max_w=170)}"
                f'<figcaption style="font-size:11px;color:#444">'
                f"{_html.escape(seg)}<br>{_html.escape(fn)}</figcaption>"
                "</figure>")
        gallery_html = (f"<h2>All shown cuts ({len(gallery)}) — timeline "
                        f"order</h2><div>{''.join(cells)}</div>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>prep QA — {_html.escape(report['title'])}</title>
<style>
body{{font-family:-apple-system,Helvetica,sans-serif;margin:24px;background:#fafafa}}
table{{border-collapse:collapse;width:100%;background:#fff}}
td,th{{border:1px solid #ddd;padding:6px 10px;vertical-align:top;text-align:left}}
th{{background:#263238;color:#fff}}
.summary b{{margin-right:18px}}
</style></head><body>
<h1>prep QA — {_html.escape(report['title'])}</h1>
<p class="summary">
<b style="color:{_SEV_COLOR[ERROR]}">ERROR: {c.get(ERROR, 0)}</b>
<b style="color:{_SEV_COLOR[WARN]}">WARN: {c.get(WARN, 0)}</b>
<b style="color:{_SEV_COLOR[INFO]}">INFO: {c.get(INFO, 0)}</b>
<b>shown cuts: {report['n_cuts']}</b></p>
{flags_html}
{gallery_html}
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--plan", default="",
                    help="default: <episode>/render.plan.clean.json")
    ap.add_argument("--series-title", default="")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip the bubble detector (no ghost/visible/dead-box "
                         "checks)")
    ap.add_argument("--bubble-conf", type=float, default=0.20)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-html", default="")
    args = ap.parse_args()

    ep = args.episode_dir.rstrip("/")
    plan_path = args.plan or os.path.join(ep, "render.plan.clean.json")
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    clean_dir = os.path.join(ep, plan.get("scenes_subdir") or "scenes_clean")
    clean_files = set(os.listdir(clean_dir)) if os.path.isdir(clean_dir) else set()
    dims = plan.get("scene_dims") or {}

    # vision items by original scene file (+ word count for doc checks)
    vitems: Dict[str, Dict[str, Any]] = {}
    vp = os.path.join(ep, "manifest.vision.json")
    if os.path.exists(vp):
        with open(vp, "r", encoding="utf-8") as f:
            for it in json.load(f).get("items") or []:
                vitems[str(it.get("scene_file") or "")] = {
                    "ocr_clean": it.get("ocr_clean"),
                    "text_only": it.get("text_only"),
                    "text_coverage": it.get("text_coverage"),
                    "n_words": len((it.get("vision") or {}).get("ocr_words") or []),
                }

    flags: List[Dict[str, Any]] = plan_flags(
        plan, clean_files=clean_files, audio_exists=os.path.exists)

    detector = None
    if not args.no_detector:
        detector = rp._load_bubble_detector(args.device)

    cuts = iter_shown_cuts(plan)
    seg_by_file: Dict[str, str] = {}
    for c in cuts:
        seg_by_file.setdefault(c["file"], c["segment_id"])

    for fname in sorted(seg_by_file):
        path = os.path.join(clean_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue  # missing_file already flagged by plan_flags
        d = dims.get(fname) or {}
        doc = bool(d.get("doc"))
        sys_panel = bool(d.get("sys"))
        boxes: List[Tuple[int, int, int, int]] = []
        if detector is not None and not doc and not sys_panel:
            boxes = [(int(x1), int(y1), int(x2), int(y2))
                     for (x1, y1, x2, y2, _s) in detector.detect(
                         img, imgsz=1024, conf=args.bubble_conf)]
        flags.extend(image_flags(
            fname, img, boxes, doc=doc, dims_entry=d if d else None,
            sys=sys_panel, segment_id=seg_by_file[fname]))

    # vision-level checks once per shown parent scene
    seen_parents: set = set()
    for c in cuts:
        parent = parent_scene(c["file"])
        if parent in seen_parents or parent not in vitems:
            continue
        seen_parents.add(parent)
        flags.extend(vision_flags(
            parent, vitems[parent],
            dims_entry=dims.get(c["file"]),
            series_title=args.series_title or None,
            segment_id=c["segment_id"]))

    # narration checks per story item; a panel's text counts as VISIBLE when
    # it is shown with text kept (doc) or was never blanked. System panels
    # are EXCLUDED: reading the on-screen system message aloud is the design.
    def _text_visible(orig: str) -> bool:
        stem, ext = os.path.splitext(orig)
        for nm in (orig, f"{stem}_a{ext}", f"{stem}_b{ext}"):
            d = dims.get(nm)
            if d:
                if d.get("sys"):
                    return False
                return bool(d.get("doc") or not d.get("blanked", False))
        return False  # not shown at all -> nothing on screen to echo

    for item in plan.get("timeline") or []:
        if item.get("branding"):
            continue
        panels = [{"ocr": str((vitems.get(str(f)) or {}).get("ocr_clean") or ""),
                   "visible": _text_visible(str(f))}
                  for f in (item.get("scene_files") or [])]
        flags.extend(narration_flags(str(item.get("segment_id") or ""),
                                     str(item.get("tts_text") or ""), panels))

    # segment-level flags (no scene) still deserve a picture: the first cut
    # their segment actually shows
    first_cut_by_segment: Dict[str, str] = {}
    for c in cuts:
        first_cut_by_segment.setdefault(c["segment_id"], c["file"])
    for f in flags:
        if not f.get("scene") and f.get("segment_id") in first_cut_by_segment:
            f["thumb_scene"] = first_cut_by_segment[f["segment_id"]]

    title = args.series_title or os.path.basename(os.path.dirname(ep))
    title = f"{title} — {os.path.basename(ep).replace('_', ' ')}"
    report = build_report(title, flags, n_cuts=len(cuts))

    # gallery: every shown cut in timeline order (dedup, first appearance)
    gallery: List[Dict[str, str]] = []
    seen_gallery: set = set()
    for c in cuts:
        if c["file"] in seen_gallery:
            continue
        seen_gallery.add(c["file"])
        gallery.append({"file": c["file"], "segment_id": c["segment_id"]})

    thumbs: Dict[str, bytes] = {}
    want = ({str(f.get("scene") or f.get("thumb_scene") or "") for f in flags}
            | seen_gallery)
    for scene in sorted(want):
        if not scene or scene in thumbs:
            continue
        img = cv2.imread(os.path.join(clean_dir, scene))
        if img is None:
            continue
        h, w = img.shape[:2]
        tw = 240
        th = max(1, int(h * tw / max(1, w)))
        small = cv2.resize(img, (tw, min(th, 600)))
        ok, buf = cv2.imencode(".jpg", small,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            thumbs[scene] = buf.tobytes()

    out_json = args.out_json or os.path.join(ep, "prep_qa.json")
    out_html = args.out_html or os.path.join(ep, "prep_qa.html")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(render_html(report, thumbs, gallery=gallery))

    c = report["counts"]
    print(f"[prep-qa] {title}: cuts={len(cuts)} "
          f"ERROR={c[ERROR]} WARN={c[WARN]} INFO={c[INFO]}")
    for f in report["flags"]:
        if f["severity"] != INFO:
            loc = f.get("scene") or f.get("segment_id") or "-"
            print(f"  [{f['severity']}] {f['code']:<18} {loc:<18} {f['detail']}")
    print(f"[prep-qa] report: {out_html}")
    return 1 if c[ERROR] else 0


if __name__ == "__main__":
    raise SystemExit(main())
