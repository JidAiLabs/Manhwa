"""
studio/qa.py

QA report generator: renders each scene crop side-by-side with its narration
text so a human can verify "does the narration match the image".

Public API
----------
    build_qa_report(ep_dir: Path, out_html: Path) -> Path

The HTML is written *inside* ep_dir so that relative ``<img src="scenes/NNN.jpg">``
paths resolve correctly when opened from that directory.

Manifest shapes consumed
------------------------
REQUIRED
  manifest.groups.json  → {"shots": [{"shot_id": int, "scene_files": [basename, ...], ...}]}

OPTIONAL
  manifest.beats.json   → {"beats": [{"group_id": int, "hook": str, "beat_title": str,
                                       "what_happens": str, ...}]}
  manifest.script.json  → {"sections": [{"script_paragraphs": [str, ...],
                                          "shots": [{"group_id": int, "segment_id": str}, ...]}]}
  manifest.vision.json  → {"items": [{"scene_file": str, "ocr_clean": str, ...}]}

Key field mappings (from producers)
------------------------------------
  groups:  shot_id   → used as group_id throughout the pipeline
  beats:   group_id  → matches shot_id from groups
  script:  shots[i].group_id + segment_id paired with script_paragraphs[i] (positional)
  vision:  scene_file → ocr_clean
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

from studio import qa_flags


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_qa_report(ep_dir: Path, out_html: Path) -> Path:
    """Build a self-contained HTML QA report for *ep_dir* and write it to *out_html*.

    Parameters
    ----------
    ep_dir:
        Episode directory that contains the manifest JSON files and a ``scenes/``
        sub-directory with the crop images.
    out_html:
        Destination HTML file.  Typically inside *ep_dir* so that relative
        ``scenes/`` image paths resolve when the file is opened in a browser.

    Returns
    -------
    Path
        The *out_html* path (same object passed in).

    Raises
    ------
    FileNotFoundError
        If ``manifest.groups.json`` is missing from *ep_dir*.
    """
    ep_dir = Path(ep_dir)
    out_html = Path(out_html)

    # ------------------------------------------------------------------
    # 1. Load manifests
    # ------------------------------------------------------------------
    groups_path = ep_dir / "manifest.groups.json"
    if not groups_path.exists():
        raise FileNotFoundError(
            f"manifest.groups.json not found in {ep_dir}. "
            "This manifest is required to build the QA report."
        )

    groups_data: dict[str, Any] = json.loads(groups_path.read_text(encoding="utf-8"))
    beats_data: dict[str, Any] | None = _load_optional(ep_dir / "manifest.beats.json")
    script_data: dict[str, Any] | None = _load_optional(ep_dir / "manifest.script.json")
    vision_data: dict[str, Any] | None = _load_optional(ep_dir / "manifest.vision.json")
    scenes_data: dict[str, Any] | None = _load_optional(ep_dir / "manifest.scenes.json")

    found_manifests = ["groups"]
    if beats_data is not None:
        found_manifests.append("beats")
    if script_data is not None:
        found_manifests.append("script")
    if vision_data is not None:
        found_manifests.append("vision")
    if scenes_data is not None:
        found_manifests.append("scenes")

    # Optional TTS clips: map segment_id -> audio path (relative to the report,
    # which sits at ep_dir, so prefix the tts/ dir). Lets the report embed a
    # playable clip next to each narration paragraph.
    tts_data = _load_optional(ep_dir / "tts" / "tts_index.json")
    audio_by_seg: dict[str, str] = {}
    if tts_data:
        for clip in tts_data.get("clips") or []:
            sid = str(clip.get("segment_id") or "")
            af = str(clip.get("audio_file") or "")
            if sid and af:
                audio_by_seg[sid] = f"tts/{af}"
        if audio_by_seg:
            found_manifests.append(f"tts({tts_data.get('backend', '?')})")

    # Resolve the canonical scene image directory from the scenes manifest's
    # out_dir (its basename), defaulting to "scenes". This keeps the report
    # pointed at whichever scene set the pipeline actually produced rather than
    # a hardcoded path.
    scene_dir_name = "scenes"
    if scenes_data:
        out_dir = scenes_data.get("out_dir")
        if out_dir:
            scene_dir_name = Path(str(out_dir)).name

    # Compute the automated QA scorecard + per-scene/per-group flags. Source page
    # count = top-level page JPGs in the episode dir (the scraped/downloaded
    # pages), used for the over-segmentation density metric.
    source_pages = len(list(ep_dir.glob("*.jpg")))
    qa = qa_flags.compute_flags(
        scenes=scenes_data or {},
        vision_items=vision_data or {},
        groups=groups_data or {},
        script=script_data,
        beats=beats_data,
        source_page_count=source_pages,
    )
    scorecard = qa["scorecard"]
    scene_flags = qa["scene_flags"]
    group_flags = qa["group_flags"]

    # ------------------------------------------------------------------
    # 2. Build indexes
    # ------------------------------------------------------------------

    # groups: list of {"shot_id": int, "scene_files": [...]}
    # shot_id IS the group_id used downstream
    shots: list[dict[str, Any]] = groups_data.get("shots") or []

    # beats index: group_id → beat dict
    beats_by_gid: dict[int, dict[str, Any]] = {}
    if beats_data:
        for beat in beats_data.get("beats") or []:
            gid = int(beat.get("group_id") or 0)
            if gid:
                beats_by_gid[gid] = beat

    # script index: group_id → list of {"segment_id": str, "text": str}
    # script_paragraphs[i] pairs positionally with shots[i].group_id / shots[i].segment_id
    narration_by_gid: dict[int, list[dict[str, str]]] = {}
    if script_data:
        for section in script_data.get("sections") or []:
            paras: list[str] = section.get("script_paragraphs") or []
            shot_refs: list[dict[str, Any]] = section.get("shots") or []
            n = min(len(paras), len(shot_refs))
            for i in range(n):
                gid = int((shot_refs[i] or {}).get("group_id") or 0)
                seg_id = str((shot_refs[i] or {}).get("segment_id") or "")
                text = str(paras[i] or "")
                if gid:
                    narration_by_gid.setdefault(gid, []).append(
                        {"segment_id": seg_id, "text": text}
                    )

    # vision index: scene_file → ocr_clean
    ocr_by_file: dict[str, str] = {}
    if vision_data:
        for item in vision_data.get("items") or []:
            sf = item.get("scene_file")
            ocr = item.get("ocr_clean") or ""
            if sf:
                ocr_by_file[str(sf)] = str(ocr)

    # ------------------------------------------------------------------
    # 3. Render HTML
    # ------------------------------------------------------------------
    total_scenes = sum(len(s.get("scene_files") or []) for s in shots)
    html_parts: list[str] = [_HTML_HEAD]

    # Summary bar
    html_parts.append(_render_summary(ep_dir, len(shots), total_scenes, found_manifests))

    # Automated confidence scorecard
    html_parts.append(_render_scorecard(scorecard))

    # One card per group
    for shot in shots:
        group_id = int(shot.get("shot_id") or shot.get("group_id") or 0)
        scene_files: list[str] = shot.get("scene_files") or []
        beat = beats_by_gid.get(group_id)
        narrations = narration_by_gid.get(group_id) or []

        html_parts.append(
            _render_group_card(
                group_id=group_id,
                scene_files=scene_files,
                beat=beat,
                narrations=narrations,
                ocr_by_file=ocr_by_file,
                scene_dir_name=scene_dir_name,
                scene_flags=scene_flags,
                group_flags=group_flags.get(group_id) or [],
                audio_by_seg=audio_by_seg,
            )
        )

    html_parts.append(_HTML_FOOT)

    out_html.write_text("\n".join(html_parts), encoding="utf-8")
    return out_html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _e(text: str) -> str:
    """HTML-escape a string."""
    return _html.escape(str(text or ""), quote=True)


def _render_summary(
    ep_dir: Path,
    n_groups: int,
    n_scenes: int,
    found_manifests: list[str],
) -> str:
    manifests_str = ", ".join(found_manifests)
    return f"""\
<div class="summary">
  <h1>QA Report</h1>
  <table class="summary-table">
    <tr><th>Episode dir</th><td><code>{_e(str(ep_dir))}</code></td></tr>
    <tr><th>Groups</th><td>{n_groups}</td></tr>
    <tr><th>Scenes</th><td>{n_scenes}</td></tr>
    <tr><th>Manifests found</th><td>{_e(manifests_str)}</td></tr>
  </table>
</div>
"""


def _flag_badges(flags: list[dict[str, Any]]) -> str:
    """Render a list of {"kind","detail"} flags as colored badge spans."""
    if not flags:
        return ""
    spans = []
    for f in flags:
        kind = str(f.get("kind") or "")
        detail = str(f.get("detail") or "")
        spans.append(
            f'<span class="flag flag-{_e(kind)}" title="{_e(detail)}">'
            f'{_e(kind.replace("_", " "))}</span>'
        )
    return '<div class="flags">' + "".join(spans) + "</div>"


def _render_scorecard(sc: dict[str, Any]) -> str:
    """Render the automated confidence scorecard: counts + pass/fail chips."""
    def chip(label: str, ok: bool, value: Any) -> str:
        cls = "ok" if ok else "bad"
        return (
            f'<div class="metric metric-{cls}">'
            f'<div class="metric-value">{_e(value)}</div>'
            f'<div class="metric-label">{_e(label)}</div></div>'
        )

    overall = "ok" if sc.get("all_ok") else "bad"
    overall_txt = "CONFIDENT" if sc.get("all_ok") else "NEEDS WORK"

    chips = [
        chip("shown / page (≤3)", sc.get("density_ok", False),
             f'{sc.get("shown_per_page")}'),
        chip("shown panels", True, sc.get("shown_panels", 0)),
        chip("dropped (budget)", True, sc.get("dropped_panels", 0)),
        chip("visible dups (0)", sc.get("dup_ok", False), sc.get("visible_dup_pairs", 0)),
        chip("OCR-echoes (0)", sc.get("echo_ok", False), sc.get("ocr_echo")),
        chip("shown <3.5s (0)", sc.get("pacing_ok", False), sc.get("shown_under_min", 0)),
        chip("text-only bubbles", sc.get("text_dominated", 0) == 0, sc.get("text_dominated")),
        chip("groups w/o narration (0)", sc.get("narration_ok", False),
             sc.get("missing_narration_groups")),
        chip("redundant dropped", True, sc.get("redundant_marked", 0)),
        chip("scene-set in sync", sc.get("sync_ok", False),
             "yes" if sc.get("sync_ok") else "DRIFT"),
    ]

    drift_note = ""
    if sc.get("scene_set_drift"):
        miss = ", ".join(sc.get("drift_missing_files") or [])
        drift_note = (
            f'<p class="drift-note">⚠ groups reference scenes not in the current set: '
            f'<code>{_e(miss)}</code> — the report below may be stale. Re-run '
            f'<code>grouped→scripted</code> after re-scening.</p>'
        )

    return f"""\
<div class="scorecard scorecard-{overall}">
  <div class="scorecard-head">
    <span class="scorecard-verdict">{overall_txt}</span>
    <span class="scorecard-sub">{sc.get("total_scenes")} scenes ·
      {sc.get("source_pages")} pages · {sc.get("groups")} groups</span>
  </div>
  <div class="metrics">{''.join(chips)}</div>
  {drift_note}
</div>
"""


def _render_group_card(
    *,
    group_id: int,
    scene_files: list[str],
    beat: dict[str, Any] | None,
    narrations: list[dict[str, str]],
    ocr_by_file: dict[str, str],
    scene_dir_name: str = "scenes",
    scene_flags: dict[str, list] | None = None,
    group_flags: list[dict[str, Any]] | None = None,
    audio_by_seg: dict[str, str] | None = None,
) -> str:
    scene_flags = scene_flags or {}
    group_flags = group_flags or []
    audio_by_seg = audio_by_seg or {}
    parts: list[str] = []
    parts.append(f'<div class="group-card" id="group-{group_id}">')

    # ---- Left column: images ----
    parts.append('<div class="col-images">')
    parts.append(f'<div class="group-label">Group&nbsp;{group_id}</div>')
    for sf in scene_files:
        ocr = ocr_by_file.get(sf, "")
        parts.append('<div class="scene-block">')
        parts.append(
            f'<img src="{_e(scene_dir_name)}/{_e(sf)}" alt="{_e(sf)}" loading="lazy">'
        )
        parts.append(f'<div class="scene-name"><code>{_e(sf)}</code></div>')
        parts.append(_flag_badges(scene_flags.get(sf) or []))
        if ocr:
            parts.append(
                f'<details class="ocr-details"><summary>OCR</summary>'
                f'<pre class="ocr-text">{_e(ocr)}</pre></details>'
            )
        parts.append("</div>")  # .scene-block
    parts.append("</div>")  # .col-images

    # ---- Right column: beat + narration ----
    parts.append('<div class="col-text">')
    parts.append(_flag_badges(group_flags))

    # Beat info
    if beat:
        hook = beat.get("hook") or ""
        beat_title = beat.get("beat_title") or ""
        what_happens = beat.get("what_happens") or ""
        mood_words = beat.get("mood_words") or []

        if beat_title:
            parts.append(f'<h3 class="beat-title">{_e(beat_title)}</h3>')
        if what_happens:
            parts.append(f'<p class="what-happens">{_e(what_happens)}</p>')
        if mood_words:
            mood_str = ", ".join(str(m) for m in mood_words)
            parts.append(f'<p class="mood-words"><em>Mood:</em> {_e(mood_str)}</p>')
        if hook:
            parts.append(f'<blockquote class="hook">{_e(hook)}</blockquote>')

    # Narration paragraphs
    parts.append('<div class="narration-section">')
    parts.append('<h4>Narration</h4>')
    if narrations:
        for entry in narrations:
            seg_id = entry.get("segment_id", "")
            text = entry.get("text", "")
            parts.append('<div class="narration-entry">')
            if seg_id:
                parts.append(f'<div class="segment-id"><code>{_e(seg_id)}</code></div>')
            parts.append(f'<p class="narration-text">{_e(text)}</p>')
            audio_src = audio_by_seg.get(seg_id)
            if audio_src:
                parts.append(
                    f'<audio class="narration-audio" controls preload="none" '
                    f'src="{_e(audio_src)}"></audio>'
                )
            parts.append("</div>")  # .narration-entry
    else:
        parts.append(
            '<p class="narration-placeholder">'
            '&#8212; narration not generated yet &#8212;'
            "</p>"
        )
    parts.append("</div>")  # .narration-section

    parts.append("</div>")  # .col-text
    parts.append("</div>")  # .group-card

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML boilerplate (self-contained, no external assets)
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QA Report — Manhwa Scene&thinsp;↔&thinsp;Narration</title>
<style>
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.55;
  background: #f4f4f6;
  color: #1a1a2e;
}
h1 { font-size: 1.4rem; margin-bottom: .5rem; }
h3 { font-size: 1rem; margin-bottom: .25rem; }
h4 { font-size: .85rem; text-transform: uppercase; letter-spacing: .05em;
     color: #666; margin: .75rem 0 .35rem; }
code { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
       font-size: .82em; background: #eee; padding: 1px 4px; border-radius: 3px; }
pre  { white-space: pre-wrap; word-break: break-word; }
a    { color: #0057b7; }

/* ── Summary bar ── */
.summary {
  background: #fff;
  border-bottom: 2px solid #d0d0e0;
  padding: 1rem 1.5rem;
}
.summary-table { border-collapse: collapse; margin-top: .5rem; }
.summary-table th, .summary-table td { padding: .2rem .8rem .2rem 0; text-align: left; }
.summary-table th { color: #555; font-weight: 600; }

/* ── Scorecard ── */
.scorecard {
  margin: 1rem 1.5rem 0;
  padding: 1rem 1.2rem;
  border-radius: 8px;
  border: 2px solid;
}
.scorecard-ok  { background: #eefbf1; border-color: #34c759; }
.scorecard-bad { background: #fff4f3; border-color: #e0463a; }
.scorecard-head { display: flex; align-items: baseline; gap: .8rem; margin-bottom: .7rem; }
.scorecard-verdict { font-size: 1.05rem; font-weight: 800; letter-spacing: .04em; }
.scorecard-ok  .scorecard-verdict { color: #1a7f37; }
.scorecard-bad .scorecard-verdict { color: #b3261e; }
.scorecard-sub { color: #555; font-size: .85rem; }
.metrics { display: flex; flex-wrap: wrap; gap: .6rem; }
.metric {
  min-width: 84px; flex: 0 0 auto;
  padding: .45rem .7rem; border-radius: 6px; text-align: center;
  border: 1px solid #d8d8e8; background: #fff;
}
.metric-value { font-size: 1.15rem; font-weight: 700; line-height: 1.1; }
.metric-label { font-size: .68rem; color: #666; text-transform: uppercase; letter-spacing: .03em; }
.metric-ok  .metric-value { color: #1a7f37; }
.metric-bad { border-color: #e0463a; background: #fff4f3; }
.metric-bad .metric-value { color: #b3261e; }
.drift-note { margin-top: .7rem; color: #b3261e; font-size: .85rem; }

/* ── Flag badges ── */
.flags { display: flex; flex-wrap: wrap; gap: .25rem; margin: .2rem 0; }
.flag {
  font-size: .68rem; font-weight: 600; padding: 1px 6px; border-radius: 10px;
  background: #ffe8b3; color: #7a4f00; white-space: nowrap; cursor: help;
}
.flag-near_duplicate { background: #ffd6d6; color: #9a1b1b; }
.flag-text_dominated { background: #d9e4ff; color: #1f3d8a; }
.flag-short_on_screen { background: #ffe0b3; color: #8a4b00; }
.flag-ocr_echo { background: #f3d6ff; color: #6a1b8a; }
.flag-no_narration { background: #e0e0e0; color: #444; }
.flag-redundant { background: #d8d8d8; color: #555; text-decoration: line-through; }
.flag-dropped { background: #ececec; color: #888; }

/* ── Group card ── */
.group-card {
  display: flex;
  flex-direction: row;
  gap: 0;
  background: #fff;
  border: 1px solid #d8d8e8;
  border-radius: 6px;
  margin: 1rem 1.5rem;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}

/* ── Left: images ── */
.col-images {
  width: 42%;
  min-width: 220px;
  background: #1a1a2e;
  padding: .75rem;
  display: flex;
  flex-direction: column;
  gap: .6rem;
}
.group-label {
  color: #aab;
  font-size: .78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: .2rem;
}
.scene-block { display: flex; flex-direction: column; gap: .2rem; }
.scene-block img {
  width: 100%;
  height: auto;
  border-radius: 3px;
  display: block;
}
.scene-name code {
  background: transparent;
  color: #88aacc;
  font-size: .74em;
}

/* ── OCR collapsible ── */
.ocr-details summary {
  cursor: pointer;
  font-size: .76em;
  color: #88aacc;
  margin-top: .15rem;
}
.ocr-text {
  font-size: .76em;
  color: #ccd;
  background: #0d0d1a;
  border-radius: 3px;
  padding: .4rem .5rem;
  margin-top: .25rem;
  max-height: 140px;
  overflow-y: auto;
}

/* ── Right: text ── */
.col-text {
  flex: 1;
  padding: 1rem 1.2rem;
  display: flex;
  flex-direction: column;
  gap: .4rem;
  overflow-wrap: break-word;
}
.beat-title { color: #1a1a2e; }
.what-happens { color: #333; }
.mood-words { font-size: .82em; color: #666; }
.hook {
  border-left: 3px solid #0057b7;
  padding: .4rem .75rem;
  margin: .5rem 0;
  color: #003d85;
  font-style: italic;
  background: #f0f4ff;
  border-radius: 0 4px 4px 0;
}

/* ── Narration ── */
.narration-section { margin-top: .25rem; }
.narration-entry { margin-bottom: .6rem; }
.segment-id { margin-bottom: .15rem; }
.narration-text {
  font-size: .93em;
  line-height: 1.6;
  color: #222;
}
.narration-placeholder {
  color: #999;
  font-style: italic;
  font-size: .88em;
}
.narration-audio { width: 100%; height: 32px; margin-top: .3rem; }

/* ── Responsive ── */
@media (max-width: 640px) {
  .group-card { flex-direction: column; }
  .col-images { width: 100%; }
}
</style>
</head>
<body>
"""

_HTML_FOOT = """\
</body>
</html>
"""
