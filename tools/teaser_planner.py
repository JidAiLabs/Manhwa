#!/usr/bin/env python3
"""teaser_planner.py — bundle-level "arc teaser" planner.

Selects a single high-stakes window from the chapters in a bundle and (Stage 2,
later chunks) writes spoiler-safe narration + a synthetic episode dir so the
existing render/TTS tools can turn it into a short cold-open `teaser.mp4`
prepended to the bundle concat.

This module is split in two stages:

  Stage 1 (this chunk) — DETERMINISTIC, $0, pure functions. No LLM, no I/O.
    eligible_panels  — drop chrome/empty/error panels, keep story|caption|system.
    score_window     — score one contiguous window by keyword/intensity signals.
    score_windows    — spoiler guard (exclude the payoff tail + the single global
                       peak panel), enumerate min..max windows, greedily shortlist
                       the top non-overlapping windows.

  Stage 2 (later chunks) — one injectable model call to pick a window + write
    narration, then materialize the synthetic teaser dir.

The keyword sets and weights below are the ONLY calibration knobs and are
intentionally agnostic (no per-series config).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# tools/ is not a package; make sibling modules importable by name (the same
# idiom gemini_narrative_pass.py uses to reach recap_style). recap_style is
# lightweight (stdlib only) so a module-level import is safe; the heavier
# gemini_narrative_pass / google-genai imports stay lazy inside main()'s
# model-call builder so unit tests never pull them in.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
import recap_style  # noqa: E402

# Panel kinds that can carry the story forward (chrome/empty/etc. are dropped).
_ELIGIBLE_KINDS = {"story", "caption", "system"}

# Ordinal rank of an understood panel's intensity tag.
INTENSITY_RANK = {"calm": 0, "tense": 1, "intense": 2, "explosive": 3}

# --- Stage-1 scoring signals (agnostic; the ONLY calibration knobs) ---------- #
# Keyword families that mark a teaser-worthy moment. Word-boundary + greedy
# suffix (\w*) so "humiliat" catches humiliate/humiliated/humiliation, etc.
_STAKES_RE = re.compile(
    r"\b(exam|test|trial|rank|survival|expel|expuls|execut|contract|tournament|duel)\w*", re.I)
_SOCIAL_RE = re.compile(
    r"\b(humiliat|mock|laugh|badge|token|reject|outcast|disgrace|shame|peasant)\w*", re.I)
_POWER_RE = re.compile(
    r"\b(system|status window|skill|rank up|awaken|hidden|impossible|level|power|technique)\w*", re.I)
_ENEMY_RE = re.compile(
    r"\b(elder|heir|clan|authority|enemy|assassin|master|commander|villain)\w*", re.I)
# POWER / TRANSFORMATION reveal — the genre-defining turn the montage climaxes
# on ("what the protagonist becomes"): the power activates / awakens / transforms.
# Agnostic genre cues only (no per-series words).
_POWER_REVEAL_RE = re.compile(
    r"\b(nano|machine|activat|awaken|transform|unleash|surge|aura|energy|glow|"
    r"radiat|system (window|notification)|power|force|erupt|burst|shockwave|swirl)\w*",
    re.I)

# Per-signal keyword hits are capped so one keyword-stuffed window can't dominate.
_SIGNAL_CAP = 4
# How many distinct subjects a window may carry before it reads as cluttered.
_CLARITY_SUBJECT_LIMIT = 6

# Weights — the single calibration table. Keyword families, the intensity peak,
# visual variety, and a clarity penalty for too many distinct subjects.
_W_STAKES = 1.0
_W_SOCIAL = 1.0
_W_POWER = 0.8
_W_ENEMY = 0.6
_W_INTENSITY = 1.5
_W_VARIETY = 0.5
_W_CLARITY_PENALTY = 0.75
# The power/transformation reveal is the climax driver — weighted heavily so a
# panel where the power activates outranks ordinary exposition.
_W_POWER_REVEAL = 2.0


# --------------------------------------------------------------------------- #
# Task 3: panel eligibility + flattening
# --------------------------------------------------------------------------- #
def eligible_panels(panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only panels usable in a teaser, in reading order.

    A panel is eligible when its ``panel_kind`` is one of story|caption|system
    AND it has no ``error`` key (a parse/understanding failure). Chrome, empty
    fields, watermarks, etc. are dropped.
    """
    out: List[Dict[str, Any]] = []
    for p in panels:
        if "error" in p:
            continue
        if p.get("panel_kind") not in _ELIGIBLE_KINDS:
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Task 4: signal scoring of one window
# --------------------------------------------------------------------------- #
def _panel_text(panel: Dict[str, Any]) -> str:
    """The searchable text of one panel: description + action + dialogue."""
    return " ".join(
        str(panel.get(k, "") or "") for k in ("description", "action", "dialogue")
    )


def _signal_counts(text: str) -> Dict[str, int]:
    """Capped keyword-family hit counts over ``text`` (shared by panel/window).

    Each family is capped at ``_SIGNAL_CAP`` so one keyword-stuffed slice can't
    dominate. ``power_reveal`` is the POWER/TRANSFORMATION signal (the climax cue).
    """
    return {
        "stakes": min(len(_STAKES_RE.findall(text)), _SIGNAL_CAP),
        "social": min(len(_SOCIAL_RE.findall(text)), _SIGNAL_CAP),
        "power": min(len(_POWER_RE.findall(text)), _SIGNAL_CAP),
        "enemy": min(len(_ENEMY_RE.findall(text)), _SIGNAL_CAP),
        "power_reveal": min(len(_POWER_REVEAL_RE.findall(text)), _SIGNAL_CAP),
    }


def score_panel(panel: Dict[str, Any]) -> Dict[str, Any]:
    """Score ONE panel for the arc-montage selector.

    A weighted sum of the keyword families (each capped at ``_SIGNAL_CAP``) + the
    panel's intensity rank + the POWER/TRANSFORMATION ``power_reveal`` component —
    the signal that drives climax selection. Returns ``{"score", "power_reveal",
    "intensity_rank", "signals"}``; ``power_reveal``/``intensity_rank`` are lifted
    to the top level so ``select_montage`` can rank ``(power_reveal, intensity_rank,
    score)`` without re-parsing the panel.
    """
    c = _signal_counts(_panel_text(panel))
    intensity_rank = INTENSITY_RANK.get(panel.get("intensity"), 0)
    score = (
        _W_STAKES * c["stakes"]
        + _W_SOCIAL * c["social"]
        + _W_POWER * c["power"]
        + _W_ENEMY * c["enemy"]
        + _W_INTENSITY * intensity_rank
        + _W_POWER_REVEAL * c["power_reveal"]
    )
    return {
        "score": float(score),
        "power_reveal": c["power_reveal"],
        "intensity_rank": intensity_rank,
        "signals": {**c, "intensity_rank": intensity_rank},
    }


def score_window(panels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one contiguous window of panels for teaser-worthiness.

    The score is a weighted sum of:
      - stakes/social/power/enemy keyword hits (each capped at ``_SIGNAL_CAP``),
      - the intensity PEAK (max ``INTENSITY_RANK`` across the window),
      - visual variety (distinct panel_kinds + distinct intensities),
    minus a clarity penalty when the union of distinct ``subjects`` exceeds
    ``_CLARITY_SUBJECT_LIMIT``.

    Returns ``{"score": float, "signals": dict}``; an empty window scores 0.
    """
    if not panels:
        return {"score": 0.0, "signals": {}}

    text = " ".join(_panel_text(p) for p in panels)
    c = _signal_counts(text)
    stakes, social, power, enemy = c["stakes"], c["social"], c["power"], c["enemy"]

    intensity_peak = max(
        INTENSITY_RANK.get(p.get("intensity"), 0) for p in panels
    )

    distinct_kinds = {p.get("panel_kind") for p in panels}
    distinct_intensities = {p.get("intensity") for p in panels}
    variety = len(distinct_kinds) + len(distinct_intensities)

    subjects: set = set()
    for p in panels:
        for s in (p.get("subjects") or []):
            subjects.add(s)
    clarity_overflow = max(0, len(subjects) - _CLARITY_SUBJECT_LIMIT)

    score = (
        _W_STAKES * stakes
        + _W_SOCIAL * social
        + _W_POWER * power
        + _W_ENEMY * enemy
        + _W_INTENSITY * intensity_peak
        + _W_VARIETY * variety
        - _W_CLARITY_PENALTY * clarity_overflow
    )

    signals = {
        "stakes": stakes,
        "social": social,
        "power": power,
        "enemy": enemy,
        "intensity_peak": intensity_peak,
        "variety": variety,
        "distinct_subjects": len(subjects),
        "clarity_overflow": clarity_overflow,
    }
    return {"score": float(score), "signals": signals}


# --------------------------------------------------------------------------- #
# Task 5: spoiler guard + window enumeration + non-overlapping shortlist
# --------------------------------------------------------------------------- #
def score_windows(
    panels: List[Dict[str, Any]],
    *,
    min_panels: int,
    max_panels: int,
    payoff_tail_frac: float,
    shortlist_n: int,
) -> List[Dict[str, Any]]:
    """Enumerate, score, and shortlist teaser windows over the eligible pool.

    Steps:
      1. ``pool = eligible_panels(panels)``; bail (``[]``) if shorter than
         ``min_panels``.
      2. SPOILER GUARD — exclude indices in the last ``payoff_tail_frac`` of the
         pool plus the single global max-intensity panel (the likely payoff).
      3. Enumerate every contiguous window of size ``min_panels..max_panels``
         that contains no excluded index and score each with ``score_window``.
      4. Greedily pick the top-scoring NON-OVERLAPPING windows (score desc) until
         ``shortlist_n`` are chosen or candidates are exhausted.

    Window dicts use half-open spans: ``{"start": i, "end": j, "panels": [...],
    "score": float, "signals": dict}`` where the slice is ``pool[i:j]``.
    """
    pool = eligible_panels(panels)
    n = len(pool)
    if n < min_panels:
        return []

    # --- spoiler guard: exclude the payoff tail + the single global peak panel
    tail_count = int(math.ceil(n * payoff_tail_frac))
    excluded = set(range(n - tail_count, n))
    peak_idx = max(
        range(n), key=lambda i: INTENSITY_RANK.get(pool[i].get("intensity"), 0)
    )
    excluded.add(peak_idx)

    # --- enumerate candidate windows that dodge every excluded index
    candidates: List[Dict[str, Any]] = []
    for start in range(n):
        for size in range(min_panels, max_panels + 1):
            end = start + size
            if end > n:
                break
            if any(i in excluded for i in range(start, end)):
                continue
            slc = pool[start:end]
            scored = score_window(slc)
            candidates.append({
                "start": start,
                "end": end,
                "panels": slc,
                "score": scored["score"],
                "signals": scored["signals"],
            })

    # --- greedy non-overlapping shortlist (score desc, stable on ties)
    candidates.sort(key=lambda w: w["score"], reverse=True)
    picked: List[Dict[str, Any]] = []
    for w in candidates:
        if len(picked) >= shortlist_n:
            break
        if any(w["start"] < p["end"] and p["start"] < w["end"] for p in picked):
            continue
        picked.append(w)
    return picked


# --------------------------------------------------------------------------- #
# Arc-montage selector — builds to the power/transformation reveal
# --------------------------------------------------------------------------- #
def _chapter_key(panel: Dict[str, Any]) -> Any:
    """Sortable chapter key; ``None`` sorts LAST so unnumbered panels trail."""
    cn = panel.get("chapter_number")
    return cn if cn is not None else float("inf")


def _norm_text(panel: Dict[str, Any]) -> str:
    """Whitespace-collapsed, lowercased panel text — for near-duplicate checks."""
    return re.sub(r"\s+", " ", _panel_text(panel)).strip().lower()


def _scene_base(panel: Dict[str, Any]) -> str:
    return os.path.basename(str(panel.get("scene_file") or ""))


def select_montage(
    panels: List[Dict[str, Any]],
    *,
    max_panels: int,
    min_panels: int,
    payoff_tail_frac: float = 0.0,
) -> Optional[List[Dict[str, Any]]]:
    """Select an ARC MONTAGE that BUILDS to the power/transformation reveal.

    Unlike the old single-window selection, this spans the whole eligible pool and
    deliberately ends on the genre-defining hook ("what the protagonist becomes").

      1. ``pool = eligible_panels(panels)``; return ``None`` if shorter than
         ``min_panels``.
      2. ``payoff_tail_frac`` (default ``0.0`` — OFF) optionally trims a literal
         final-cliffhanger sliver off the END of the pool. The power reveal is the
         HOOK, not an outcome spoiler, so by default nothing is excluded.
      3. CLIMAX = the panel with the highest ``(power_reveal, intensity_rank,
         score)`` — the transformation/power reveal. It is ALWAYS the LAST montage
         panel; ties break to the later (closer-to-the-reveal) panel.
      4. SETUP = the next strongest panels by score, SPREAD across chapters (capped
         at ~``ceil(max_panels / n_chapters)+1`` per chapter, the climax counted),
         skipping near-duplicate adjacent panels, up to ``max_panels-1`` of them.
      5. ORDER = setup sorted CHRONOLOGICALLY by ``(chapter_number, reading
         index)``, then the climax appended LAST. The montage builds to the reveal;
         it is NOT required to be globally chronological.

    Returns the ordered montage (shallow copies with an ``is_climax`` flag on the
    last panel), or ``None`` when too few panels are eligible.
    """
    pool = eligible_panels(panels)
    n = len(pool)
    if n < min_panels:
        return None

    # optional literal-cliffhanger trim off the tail (default OFF)
    if payoff_tail_frac and payoff_tail_frac > 0.0:
        tail = int(math.ceil(n * payoff_tail_frac))
        if tail > 0:
            pool = pool[: max(0, n - tail)]
        if len(pool) < min_panels:
            return None

    # score every panel once, carrying its reading-order index
    scored = [(i, p, score_panel(p)) for i, p in enumerate(pool)]

    # climax = the power/transformation reveal (ties -> later panel)
    climax_i, climax_p, _csc = max(
        scored,
        key=lambda t: (t[2]["power_reveal"], t[2]["intensity_rank"],
                       t[2]["score"], t[0]),
    )

    n_chapters = len({_chapter_key(p) for _, p, _ in scored})
    per_cap = math.ceil(max_panels / max(1, n_chapters)) + 1

    # setup = strongest remaining panels, spread across chapters, no near-dupes
    per_chapter: Dict[Any, int] = {_chapter_key(climax_p): 1}   # climax counts
    seen_meta = [(climax_i, _norm_text(climax_p), _scene_base(climax_p))]
    setup: List[tuple] = []
    limit = max(0, max_panels - 1)
    candidates = sorted(
        (t for t in scored if t[0] != climax_i),
        key=lambda t: (-t[2]["score"], t[0]),
    )
    for idx, p, _sc in candidates:
        if len(setup) >= limit:
            break
        ck = _chapter_key(p)
        if per_chapter.get(ck, 0) >= per_cap:
            continue
        ntext, nbase = _norm_text(p), _scene_base(p)
        if any(abs(idx - j) <= 1 and (ntext == jt or (nbase and nbase == jb))
               for (j, jt, jb) in seen_meta):
            continue
        setup.append((idx, p))
        seen_meta.append((idx, ntext, nbase))
        per_chapter[ck] = per_chapter.get(ck, 0) + 1

    # order: setup chronological, climax LAST
    setup.sort(key=lambda ip: (_chapter_key(ip[1]), ip[0]))
    ordered = [p for _, p in setup] + [climax_p]
    out: List[Dict[str, Any]] = []
    for p in ordered:
        q = dict(p)
        q["is_climax"] = p is climax_p
        out.append(q)
    return out


# --------------------------------------------------------------------------- #
# Task 6: Stage-2 select + write narration (one injected model call)
# --------------------------------------------------------------------------- #
def _sanitize_chapter_number(ch_num: Any) -> str:
    """Render a chapter number as a filesystem-safe token for a namespaced id.

    Int-valued floats collapse to the bare int (``5.0`` -> ``"5"``); a fractional
    chapter keeps the fraction with the dot swapped for an underscore (``5.5`` ->
    ``"5_5"``); a stray trailing ``.0`` on a string is stripped. ``None`` becomes
    ``"x"`` so the id stays well-formed.
    """
    if ch_num is None:
        return "x"
    if isinstance(ch_num, bool):  # bool is an int subclass — treat as a label
        return str(ch_num)
    if isinstance(ch_num, float):
        return str(int(ch_num)) if ch_num.is_integer() else str(ch_num).replace(".", "_")
    if isinstance(ch_num, int):
        return str(ch_num)
    s = str(ch_num).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.replace(".", "_")


def _namespaced_scene_id(chapter_number: Any, scene_file: Any) -> str:
    """``ch{chapter}__{basename}`` — namespaces a scene by its chapter so the
    same per-chapter basename (the chunk index restarts every chapter) never
    collides across the bundle."""
    base = os.path.basename(str(scene_file or ""))
    return f"ch{_sanitize_chapter_number(chapter_number)}__{base}"


def _window_payload(window: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Understood text per panel (scene_file as a BASENAME) for the model call."""
    rows: List[Dict[str, Any]] = []
    for p in window.get("panels") or []:
        rows.append({
            "scene_file": os.path.basename(str(p.get("scene_file") or "")),
            "chapter_number": p.get("chapter_number"),
            "description": str(p.get("description", "") or ""),
            "action": str(p.get("action", "") or ""),
            "dialogue": str(p.get("dialogue", "") or ""),
            "panel_kind": p.get("panel_kind"),
            "intensity": p.get("intensity"),
            "subjects": list(p.get("subjects") or []),
        })
    return rows


def select_and_write(
    windows: List[Dict[str, Any]],
    *,
    loglines: List[str],
    model_call: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    cast_obj: Optional[Dict[str, Any]] = None,
    vision_by_file: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Ask the (injected) model to pick the strongest window + write narration.

    ``model_call(payload) -> dict|None`` follows the ``story_group`` call_fn
    pattern so tests never hit a real LLM. The payload carries the shortlisted
    ``windows`` (understood text per panel, basenames) plus the bundle
    ``loglines`` for context. The model returns ``chosen_index`` +
    spoiler-safe per-panel narration; we assemble the ``manifest.teaser.json``
    dict and run the shared spoiler/fragment post-pass on it.

    Identity is carried end-to-end: every panel becomes a NAMESPACED scene id
    ``ch{chapter}__{basename}`` (the chunk index restarts each chapter, so a bare
    basename collides across the bundle). The model's narration lines are aligned
    to the window panels BY ORDER (panel i <-> line i), padded/truncated to the
    panel count — never matched by the now-colliding basename. The returned dict
    carries ``panel_sources`` ({namespaced_id: source_abs_path}) so materialize
    can symlink the correct chapter's art.

    Returns the teaser dict, or ``None`` when the model abstains.
    """
    if not windows:
        return None

    payload = {
        "windows": [_window_payload(w) for w in windows],
        "loglines": list(loglines or []),
    }
    resp = model_call(payload)
    if not isinstance(resp, dict) or "chosen_index" not in resp:
        return None

    try:
        idx = int(resp["chosen_index"])
        chosen = windows[idx]
    except (ValueError, TypeError, IndexError):
        return None

    panels = chosen.get("panels") or []
    scene_files: List[str] = []
    panel_sources: Dict[str, str] = {}
    window_basenames: List[str] = []
    for p in panels:
        sf = str(p.get("scene_file") or "")
        ns_id = _namespaced_scene_id(p.get("chapter_number"), sf)
        scene_files.append(ns_id)
        window_basenames.append(os.path.basename(sf))
        if sf:
            panel_sources[ns_id] = os.path.abspath(sf)
    source_chapters = sorted({
        p.get("chapter_number") for p in panels
        if p.get("chapter_number") is not None
    })
    # Align model narration to the window panels BY ORDER (basename now collides),
    # padding/truncating to the panel count; scene_file is the namespaced id.
    lines = resp.get("panel_narration") or []
    panel_narration: List[Dict[str, Any]] = []
    for i, ns_id in enumerate(scene_files):
        line = ""
        if i < len(lines) and isinstance(lines[i], dict):
            line = str(lines[i].get("line") or "").strip()
        panel_narration.append({"scene_file": ns_id, "line": line})

    teaser = {
        "source_chapters": source_chapters,
        "scene_files": scene_files,
        "panel_sources": panel_sources,
        "window": window_basenames,
        "panel_narration": panel_narration,
        "reason": str(resp.get("reason") or ""),
        "rewind_line": str(resp.get("rewind_line") or ""),
        "spoiler_boundary": str(resp.get("spoiler_boundary") or ""),
        "scores": {
            "chosen": float(chosen.get("score", 0.0)),
            "shortlist": [float(w.get("score", 0.0)) for w in windows],
        },
    }

    # Spoiler/fragment post-pass — reuse the channel's shared writers. Wrap the
    # teaser as a single beat so the recap_style mutators (which operate on
    # {"beats":[{"panel_narration":[...]}]}) apply in-place; default cast/vision
    # to empty keeps it a safe no-op when no cast/OCR context is supplied.
    beats_obj = {"beats": [{"panel_narration": panel_narration}]}
    recap_style.neutralize_identity_reveal_leaks(
        beats_obj, cast_obj or {}, vision_by_file or {})
    recap_style.repair_spoken_fragments(beats_obj)
    teaser["panel_narration"] = beats_obj["beats"][0].get("panel_narration") or []
    return teaser


# --------------------------------------------------------------------------- #
# Task 7: synthetic-dir builder (manifests + scene symlinks)
# --------------------------------------------------------------------------- #
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def _scenes_entry(src_ep: str, scene_file: str,
                  _cache: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the source ``manifest.scenes.json`` entry whose out_file == basename."""
    man_path = os.path.join(src_ep, "manifest.scenes.json")
    if man_path not in _cache:
        try:
            _cache[man_path] = json.loads(Path(man_path).read_text())
        except Exception:
            _cache[man_path] = {}
    for entry in (_cache[man_path].get("scenes") or []):
        if entry.get("out_file") == scene_file:
            return entry
    return None


def materialize_teaser_dir(
    teaser: Dict[str, Any],
    out_dir: Any,
    cast: Dict[str, Any],
) -> Path:
    """Build the synthetic episode dir the render/TTS tools consume.

    Each scene is keyed by its NAMESPACED id (``ch{chapter}__{basename}``) and
    its source art is resolved PER-PANEL from ``teaser['panel_sources']``
    ({namespaced_id: source_abs_path}) — never by a bundle-wide basename map,
    which collides because the chunk index restarts every chapter. Each source is
    symlinked (fallback copy) to ``out_dir/scenes/{namespaced_id}`` and every
    manifest is written with the namespaced id as the scene key.

    A scene whose source ``manifest.scenes.json`` has no entry for its ORIGINAL
    basename (or that has no source path) is dropped — logged, not fatal — rather
    than emit a broken scenes manifest.
    """
    out_dir = Path(out_dir)
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    panel_sources: Dict[str, str] = teaser.get("panel_sources") or {}
    cache: Dict[str, Dict[str, Any]] = {}
    kept: List[str] = []
    scene_entries: List[Dict[str, Any]] = []
    for ns_id in teaser.get("scene_files") or []:
        src_abs = panel_sources.get(ns_id)
        if not src_abs:
            print(f"[teaser] WARN no source path for {ns_id}; dropping")
            continue
        # src_abs is <ep_dir>/scenes/<basename>; recover both halves.
        orig_base = os.path.basename(src_abs)
        src_ep = os.path.dirname(os.path.dirname(src_abs))
        entry = _scenes_entry(src_ep, orig_base, cache)
        if entry is None:
            print(f"[teaser] WARN {orig_base} not in "
                  f"{src_ep}/manifest.scenes.json; dropping")
            continue
        dst_img = scenes_dir / ns_id
        if not dst_img.exists():
            try:
                os.symlink(src_abs, dst_img)
            except OSError:
                shutil.copy2(src_abs, dst_img)
        entry = dict(entry)
        entry["out_file"] = ns_id          # render_prep/remotion resolve the symlink
        kept.append(ns_id)
        scene_entries.append(entry)

    kept_set = set(kept)
    panel_narration = [
        p for p in (teaser.get("panel_narration") or [])
        if str(p.get("scene_file") or "") in kept_set
    ]
    narration = " ".join(
        str(p.get("line") or "").strip() for p in panel_narration
        if str(p.get("line") or "").strip())

    _write_json(out_dir / "manifest.beats.json", {"beats": [{
        "group_id": 1,
        "scene_files": kept,
        "panel_narration": panel_narration,
        "narration": narration,
    }]})
    _write_json(out_dir / "manifest.groups.json", {"shots": [{
        "shot_id": 1,
        "scene_files": kept,
        "segment": "present",
        "arc_label": "teaser",
    }]})
    _write_json(out_dir / "manifest.scenes.json", {"scenes": scene_entries})
    _write_json(out_dir / "manifest.cast.json", cast)
    _write_json(out_dir / "manifest.teaser.json", teaser)
    return out_dir


# --------------------------------------------------------------------------- #
# Task 8: bundle loaders + arg parser + model-call builder + main()
# --------------------------------------------------------------------------- #
# Pull the chapter number out of a dir name (e.g. ".../Nano_Machine/Ch_012" or
# ".../012"). First integer run wins; fall back to reading order in main().
_CH_NUM_RE = re.compile(r"(\d+)")


def load_bundle_panels(
    chapter_dirs: List[str],
    *,
    max_scan_chapters: int = 0,
) -> List[Dict[str, Any]]:
    """Flatten each chapter's ``manifest.panels.understood.json`` in reading order.

    Tags every panel with its ``chapter_number`` (parsed from the dir name, else
    the 1-based reading position) and rewrites ``scene_file`` to the absolute
    ``<dir>/scenes/<basename>``. ``max_scan_chapters > 0`` caps how many chapters
    are scanned (wires the teaser cost guard). Dirs missing the understood
    manifest are skipped with a warning.
    """
    dirs = list(chapter_dirs)
    if max_scan_chapters and max_scan_chapters > 0:
        dirs = dirs[:max_scan_chapters]

    out: List[Dict[str, Any]] = []
    for idx, d in enumerate(dirs):
        d = str(d)
        man = os.path.join(d, "manifest.panels.understood.json")
        if not os.path.exists(man):
            print(f"[teaser] WARN missing understood manifest: {man}; skipping")
            continue
        try:
            data = json.loads(Path(man).read_text())
        except Exception as exc:  # noqa: BLE001 - tolerate a corrupt manifest
            print(f"[teaser] WARN unreadable {man}: {exc}; skipping")
            continue
        base = os.path.basename(os.path.normpath(d))
        m = _CH_NUM_RE.search(base)
        chapter_number = int(m.group(1)) if m else (idx + 1)
        for p in (data.get("panels") or []):
            q = dict(p)
            sf = os.path.basename(str(p.get("scene_file") or ""))
            q["chapter_number"] = chapter_number
            q["scene_file"] = os.path.abspath(os.path.join(d, "scenes", sf))
            out.append(q)
    return out


def load_loglines(chapter_dirs: List[str]) -> List[str]:
    """Collect each chapter's ``manifest.story.json`` logline (context for the model)."""
    out: List[str] = []
    for d in chapter_dirs:
        man = os.path.join(str(d), "manifest.story.json")
        if not os.path.exists(man):
            continue
        try:
            data = json.loads(Path(man).read_text())
        except Exception:  # noqa: BLE001
            continue
        logline = str(data.get("logline") or "").strip()
        if logline:
            out.append(logline)
    return out


def merge_cast(chapter_dirs: List[str]) -> Dict[str, Any]:
    """Union the bundle's ``manifest.cast.json`` members, deduped by canonical_name."""
    seen: set = set()
    members: List[Dict[str, Any]] = []
    for d in chapter_dirs:
        man = os.path.join(str(d), "manifest.cast.json")
        if not os.path.exists(man):
            continue
        try:
            data = json.loads(Path(man).read_text())
        except Exception:  # noqa: BLE001
            continue
        for member in (data.get("cast") or []):
            key = str(member.get("canonical_name") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            members.append(member)
    return {"cast": members}


# The teaser model contract: pick the strongest window by index, write rolling
# spoiler-safe per-panel narration under the recap rules, plus a rewind line.
TEASER_PROMPT = (
    "You are the story editor for a YouTube manhwa recap channel building a short "
    "ARC TEASER — a cold open that prepends a multi-chapter bundle to hook the "
    "viewer, then rewinds to the start.\n"
    "\n"
    "INPUT_JSON has `windows` (a shortlist of candidate panel windows; each panel "
    "carries description/action/dialogue/subjects/panel_kind/intensity, scene_file "
    "as a basename) and `loglines` (the bundle's chapters, for context only).\n"
    "\n"
    "DO:\n"
    "1. Pick the SINGLE strongest window — the most gripping, self-contained hook — "
    "and return its position as `chosen_index` (0-based into `windows`).\n"
    "2. For EVERY panel in the chosen window, in order, write ONE narration line in "
    "`panel_narration` as {scene_file, line}, echoing each panel's scene_file "
    "basename exactly. Make the lines FLOW as one continuous mini-story (rolling "
    "narration, not isolated captions). Match length to the panel: a punchy phrase "
    "for a quick beat, a fuller cinematic sentence for a pivotal one. The FIRST "
    "line is the cold-open hook — strong and uncapped.\n"
    "3. Write a `rewind_line`: one sentence that pivots from the hook back to the "
    "beginning (e.g. 'But to understand how it came to this, we have to go back.').\n"
    "4. Write a short `reason` (why this window) and a `spoiler_boundary` note.\n"
    "\n"
    "RECAP RULES (all six):\n"
    "  - GROUND every line strictly in what the panels show; invent NOTHING.\n"
    "  - Name the listed subjects in their own words; never rename or recount them.\n"
    "  - NEVER name an identity the art has not yet revealed — use a neutral handle.\n"
    "  - NEVER reference any event past the chosen window (no spoilers from later).\n"
    "  - Paraphrase dialogue; do not quote raw OCR.\n"
    "  - Keep it cinematic and propulsive; no meta, no 'panel'/'scene' talk.\n"
)

# genai response schema (UPPERCASE type enums; lowered for ollama's `format`).
_TEASER_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "chosen_index": {"type": "INTEGER"},
        "panel_narration": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "scene_file": {"type": "STRING"},
                    "line": {"type": "STRING"},
                },
                "required": ["scene_file", "line"],
            },
        },
        "rewind_line": {"type": "STRING"},
        "reason": {"type": "STRING"},
        "spoiler_boundary": {"type": "STRING"},
    },
    "required": ["chosen_index", "panel_narration", "rewind_line"],
}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="bundle-level arc teaser planner (window scorer + LLM pick + synthetic dir)")
    ap.add_argument("--bundle-id", type=int, required=True)
    ap.add_argument("--chapter-dirs", nargs="+", required=True,
                    help="ep_dirs of the bundle's chapters, in reading order")
    ap.add_argument("--out-dir", required=True,
                    help="synthetic teaser dir to materialize (dist/bundle_<id>/teaser)")
    # model backend (mirrors gemini_narrative_pass / story_group)
    ap.add_argument("--backend", choices=["vertex", "ollama"], default="vertex")
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="Vertex Gemini model id (ignored when --backend ollama)")
    ap.add_argument("--ollama-model", default="gemma4:26b",
                    help="ollama model id; the ollama path uses THIS, not --model")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="")
    # cost guards (defaults mirror studio.toml [teaser])
    ap.add_argument("--shortlist-n", type=int, default=4)
    ap.add_argument("--min-panels", type=int, default=4)
    ap.add_argument("--max-hook-panels", type=int, default=10)
    ap.add_argument("--payoff-tail-frac", type=float, default=0.20)
    ap.add_argument("--max-scan-chapters", type=int, default=0,
                    help="0 = scan all chapters in the bundle")
    ap.add_argument("--max-seconds", type=int, default=90,
                    help="reserved soft cap; narration stays uncapped (future duration trim)")
    return ap


def _build_model_call(args: argparse.Namespace):
    """Construct the real Vertex/ollama model_call (lazy heavy imports)."""
    from gemini_narrative_pass import _call_model_with_backoff  # noqa: E402

    if args.backend == "ollama":
        client = None
        model = args.ollama_model
    else:
        from google import genai  # noqa: E402
        if not args.project or not args.location:
            raise SystemExit("--project/--location are required for --backend vertex")
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)
        model = args.model

    def model_call(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        resp, _raw, _usage = _call_model_with_backoff(
            client=client,
            model=model,
            system_instruction=TEASER_PROMPT,
            user_payload=payload,
            image_paths=[],
            response_schema=_TEASER_SCHEMA,
            max_output_tokens=2400,
            temperature=0.3,
            backoff_max=60.0,
            backend=args.backend,
        )
        return resp

    return model_call


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    chapter_dirs = list(args.chapter_dirs)

    panels = load_bundle_panels(chapter_dirs, max_scan_chapters=args.max_scan_chapters)

    # No-teaser gates: fewer than 2 chapters, or too few eligible panels.
    if len(chapter_dirs) < 2 or len(eligible_panels(panels)) < args.min_panels:
        print("[teaser] no teaser")
        return 0

    windows = score_windows(
        panels,
        min_panels=args.min_panels,
        max_panels=args.max_hook_panels,
        payoff_tail_frac=args.payoff_tail_frac,
        shortlist_n=args.shortlist_n,
    )
    if not windows:
        print("[teaser] no teaser")
        return 0

    cast = merge_cast(chapter_dirs)
    model_call = _build_model_call(args)
    teaser = select_and_write(
        windows,
        loglines=load_loglines(chapter_dirs),
        model_call=model_call,
        cast_obj=cast,
        vision_by_file={},
    )
    if not teaser:
        print("[teaser] no teaser")
        return 0

    # Source art is resolved per-panel from teaser['panel_sources'] (namespaced
    # id -> abs path), so no bundle-wide basename map (it collided across chapters).
    materialize_teaser_dir(teaser, args.out_dir, cast=cast)
    print(f"[teaser] wrote {args.out_dir} "
          f"(chapters={teaser.get('source_chapters')}, "
          f"panels={len(teaser.get('scene_files') or [])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
