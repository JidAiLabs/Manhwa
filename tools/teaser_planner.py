#!/usr/bin/env python3
"""teaser_planner.py — bundle-level "arc teaser" planner.

Selects an ARC MONTAGE across the chapters in a bundle — a sequence that BUILDS
tension and CLIMAXES on the protagonist's power/transformation reveal ("what they
become", the genre-defining hook) — then writes spoiler-safe per-panel narration +
a synthetic episode dir so the existing render/TTS tools can turn it into a short
cold-open `teaser.mp4` prepended to the bundle concat.

This module is split in two stages:

  Stage 1 — DETERMINISTIC, $0, pure functions. No LLM, no I/O.
    eligible_panels  — drop chrome/empty/error panels, keep story|caption|system.
    score_panel      — score one panel (keyword families + intensity + the
                       two-tier power signal: a HIGH-weighted transformation cue
                       vs a LOW-weighted generic-combat cue).
    score_window     — score one contiguous window (legacy aggregate scorer).
    select_montage   — pick the climax (the LATEST strong transformation beat, so
                       a teased arc builds to it) as the LAST panel, then the
                       strongest setup panels SPREAD across chapters, ordered
                       chronologically before the climax.

  Stage 2 — one injectable model call that WRITES per-panel narration building to
    the (already selected) climax, then materializes the synthetic teaser dir.

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
from beats_segments import beat_segments  # noqa: E402

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
# POWER / TRANSFORMATION reveal — split into two tiers (agnostic cues only):
#   _TRANSFORM_RE = the GENRE-DEFINING turn the montage climaxes on ("what the
#     protagonist becomes"): the awakening / system-activation / power-up / fusion /
#     evolution / regression. Weighted HIGH — this is the climax driver.
#   _COMBAT_RE    = the broader, generic power/combat cues (a blow lands, energy
#     flares). Weighted LOW — a fight beat is teaser-worthy but is NOT the reveal.
# Tiering them lets a "nano core activates / system notification" panel outscore a
# "swings a blade with great force" panel, so the climax lands on the reveal.
_TRANSFORM_RE = re.compile(
    r"\b(awaken|activat|unlock|nano|machine|system (window|notification|message)?|"
    r"core|fuse|fused|transform|evolv|ascend|unleash|surge|awakening|"
    r"power[- ]?up|reincarnat|regress|level up|new power|reborn)\w*",
    re.I)
_COMBAT_RE = re.compile(
    r"\b(strike|blade|clash|force|energy|aura|glow|radiat|erupt|burst|"
    r"shockwave|swirl|slash|impact|blast)\w*",
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
# Two-tier power signal. The TRANSFORMATION cue is the climax driver — weighted so
# heavily that a single transform hit (>= _W_TRANSFORM) outranks a fully
# combat-stuffed panel (<= _SIGNAL_CAP * _W_COMBAT). A generic combat cue is
# teaser-worthy but secondary, so it carries a much smaller weight.
_W_TRANSFORM = 3.0
_W_COMBAT = 0.5


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
    dominate. ``transform`` is the GENRE-DEFINING reveal tier (the climax driver)
    and ``combat`` the lower, generic power/fight tier; ``score_panel`` blends them
    into ``power_reveal``.
    """
    return {
        "stakes": min(len(_STAKES_RE.findall(text)), _SIGNAL_CAP),
        "social": min(len(_SOCIAL_RE.findall(text)), _SIGNAL_CAP),
        "power": min(len(_POWER_RE.findall(text)), _SIGNAL_CAP),
        "enemy": min(len(_ENEMY_RE.findall(text)), _SIGNAL_CAP),
        "transform": min(len(_TRANSFORM_RE.findall(text)), _SIGNAL_CAP),
        "combat": min(len(_COMBAT_RE.findall(text)), _SIGNAL_CAP),
    }


def score_panel(panel: Dict[str, Any]) -> Dict[str, Any]:
    """Score ONE panel for the arc-montage selector.

    A weighted sum of the keyword families (each capped at ``_SIGNAL_CAP``) + the
    panel's intensity rank + the two-tier power signal. ``power_reveal`` is the
    transform-weighted power score ``_W_TRANSFORM * transform + _W_COMBAT *
    combat`` — the GENRE-DEFINING transformation cue dominates generic combat, so
    a "the nano core activates" panel outranks a "swings a blade with force" one.

    Returns ``{"score", "power_reveal", "transform_hits", "intensity_rank",
    "signals"}``; ``power_reveal``/``transform_hits``/``intensity_rank`` are lifted
    to the top level so ``select_montage`` can pick the climax (latest strong
    transformation beat) without re-parsing the panel.
    """
    c = _signal_counts(_panel_text(panel))
    intensity_rank = INTENSITY_RANK.get(panel.get("intensity"), 0)
    power_reveal = _W_TRANSFORM * c["transform"] + _W_COMBAT * c["combat"]
    score = (
        _W_STAKES * c["stakes"]
        + _W_SOCIAL * c["social"]
        + _W_POWER * c["power"]
        + _W_ENEMY * c["enemy"]
        + _W_INTENSITY * intensity_rank
        + power_reveal
    )
    return {
        "score": float(score),
        "power_reveal": float(power_reveal),
        "transform_hits": c["transform"],
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
      3. CLIMAX = the LATEST STRONG TRANSFORMATION beat (a teased arc BUILDS to
         the reveal, so it comes late). Take the top band of panels whose
         transform-weighted ``power_reveal`` is ``>= 0.8 * max`` AND that carry at
         least one transform cue, then pick the one LATEST in reading order
         ``(chapter_number, index)``. If NO panel has any transform cue, fall back
         to the single highest ``(power_reveal, intensity_rank, score)``. It is
         ALWAYS the LAST montage panel.
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

    # CLIMAX = the LATEST STRONG TRANSFORMATION beat. A teased arc builds TO the
    # genre-defining reveal, so among the panels that actually carry the reveal we
    # bias to the late-arc one rather than the global power argmax (which would
    # wrongly grab an earlier, more violent combat frame).
    if any(t[2]["transform_hits"] > 0 for t in scored):
        max_tp = max(t[2]["power_reveal"] for t in scored)
        band = [t for t in scored
                if t[2]["transform_hits"] > 0 and t[2]["power_reveal"] >= 0.8 * max_tp]
        # latest in reading order: max (chapter_number, index)
        climax_i, climax_p, _csc = max(band, key=lambda t: (_chapter_key(t[1]), t[0]))
    else:
        # no transformation cue anywhere — fall back to the single highest
        # (power_reveal, intensity, score) so calm/non-power bundles still work.
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


def _montage_payload(montage: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Understood text per panel (scene_file as a BASENAME) for the model call.

    Carries the ``is_climax`` flag so the model knows which panel the montage
    builds to (the LAST one, the power/transformation reveal)."""
    rows: List[Dict[str, Any]] = []
    for p in montage:
        rows.append({
            "scene_file": os.path.basename(str(p.get("scene_file") or "")),
            "chapter_number": p.get("chapter_number"),
            "description": str(p.get("description", "") or ""),
            "action": str(p.get("action", "") or ""),
            "dialogue": str(p.get("dialogue", "") or ""),
            "panel_kind": p.get("panel_kind"),
            "intensity": p.get("intensity"),
            "subjects": list(p.get("subjects") or []),
            "is_climax": bool(p.get("is_climax")),
        })
    return rows


def select_and_write(
    montage: List[Dict[str, Any]],
    *,
    loglines: List[str],
    model_call: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    cast_obj: Optional[Dict[str, Any]] = None,
    vision_by_file: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Write per-panel narration over an ALREADY-SELECTED arc montage.

    Selection happened upstream (``select_montage``): ``montage`` is the ordered
    panel list with the LAST panel flagged ``is_climax`` (the power/transformation
    reveal). There is no "pick a window" step anymore — this does ONE model call
    that WRITES spoiler-safe per-panel narration BUILDING to that climax.

    ``model_call(payload) -> dict|None`` follows the ``story_group`` call_fn
    pattern so tests never hit a real LLM. The payload carries the montage
    ``panels`` (understood text per panel, basenames, ``is_climax``), the
    ``climax_index``, and the bundle ``loglines`` for context. The model returns
    spoiler-safe per-panel narration + a rewind line; we assemble the
    ``manifest.teaser.json`` dict and run the shared spoiler/fragment post-pass.

    Identity is carried end-to-end: every panel becomes a NAMESPACED scene id
    ``ch{chapter}__{basename}`` (the chunk index restarts each chapter, so a bare
    basename collides across the bundle). The model's narration lines are aligned
    to the montage panels BY ORDER (panel i <-> line i), padded/truncated to the
    panel count — never matched by the now-colliding basename. The returned dict
    carries ``panel_sources`` ({namespaced_id: source_abs_path}) so materialize
    can symlink the correct chapter's art. Climax-last ORDER is preserved.

    Returns the teaser dict, or ``None`` when there is no montage / the model
    abstains.
    """
    if not montage:
        return None

    payload = {
        "panels": _montage_payload(montage),
        "climax_index": len(montage) - 1,
        "loglines": list(loglines or []),
    }
    resp = model_call(payload)
    if not isinstance(resp, dict):
        return None

    panels = list(montage)
    scene_files: List[str] = []
    panel_sources: Dict[str, str] = {}
    for p in panels:
        sf = str(p.get("scene_file") or "")
        ns_id = _namespaced_scene_id(p.get("chapter_number"), sf)
        scene_files.append(ns_id)
        if sf:
            panel_sources[ns_id] = os.path.abspath(sf)
    source_chapters = sorted({
        p.get("chapter_number") for p in panels
        if p.get("chapter_number") is not None
    })
    # Align model narration to the montage panels BY ORDER (basename now collides),
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
        "panel_narration": panel_narration,
        "reason": str(resp.get("reason") or ""),
        "rewind_line": str(resp.get("rewind_line") or ""),
        "spoiler_boundary": str(resp.get("spoiler_boundary") or ""),
        "scores": {
            "climax": float(score_panel(panels[-1])["score"]) if panels else 0.0,
            "montage": [float(score_panel(p)["score"]) for p in panels],
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
    # synthetic-beat construction: the manifest DELIBERATELY keeps the legacy
    # panel_narration shape (incl. narration-less panels, which the render's
    # protection machinery still shows); the narration JOIN is read through
    # the shared segments reader (skips empty-line entries).
    panel_narration = [
        p for p in (teaser.get("panel_narration") or [])
        if str(p.get("scene_file") or "") in kept_set
    ]
    narration = " ".join(
        seg["line"]
        for seg in beat_segments({"panel_narration": panel_narration}))

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


# The teaser model contract: the montage is ALREADY selected + ordered; the model
# WRITES rolling spoiler-safe per-panel narration that BUILDS to the climax (the
# last panel — the power/transformation reveal), plus a rewind line.
TEASER_PROMPT = (
    "You are the story editor for a YouTube manhwa recap channel building a short "
    "ARC TEASER — an INTENSE cold-open MONTAGE that prepends a multi-chapter "
    "bundle, then rewinds to the start. The montage builds tension across a few "
    "panels and CLIMAXES on the final panel: the protagonist's power / "
    "transformation reveal — the genre-defining hook of WHAT THEY BECOME.\n"
    "\n"
    "INPUT_JSON has `panels` (the montage, ALREADY selected and ORDERED — each "
    "panel carries description/action/dialogue/subjects/panel_kind/intensity, "
    "scene_file as a basename, and an `is_climax` flag set true on the LAST panel) "
    "and `loglines` (the bundle's chapters, for context only). `climax_index` is "
    "the 0-based position of the climax panel (always the last one).\n"
    "\n"
    "DO:\n"
    "1. For EVERY panel, in order, write ONE narration line in `panel_narration` "
    "as {scene_file, line}, echoing each panel's scene_file basename exactly. Make "
    "the lines FLOW as one continuous, ESCALATING mini-story (rolling narration, "
    "not isolated captions) that BUILDS tension toward the climax. Match length to "
    "the panel: a punchy phrase for a quick beat, a fuller cinematic sentence for a "
    "pivotal one. The FIRST line is the cold-open hook — strong and uncapped.\n"
    "2. The FINAL line LANDS the climax: hint at WHAT THE PROTAGONIST BECOMES (the "
    "power awakening / transformation) WITHOUT stating any later outcome.\n"
    "3. Write a `rewind_line`: one sentence that pivots from the hook back to the "
    "beginning (e.g. 'But to understand how it came to this, we have to go back.').\n"
    "4. Write a short `reason` (why this montage hooks) and a `spoiler_boundary` "
    "note.\n"
    "\n"
    "SPOILER BOUNDARY: you MAY show the power awakening / transformation — it is the "
    "HOOK, not an outcome spoiler — but you must NOT reveal who ultimately wins or "
    "dies, or any later twist, and you must NEVER name a concealed identity.\n"
    "\n"
    "RECAP RULES (all six):\n"
    "  - GROUND every line strictly in what the panels show; invent NOTHING.\n"
    "  - Name the listed subjects in their own words; never rename or recount them.\n"
    "  - NEVER name an identity the art has not yet revealed — use a neutral handle.\n"
    "  - Reveal only the power/transformation; no events beyond this montage.\n"
    "  - Paraphrase dialogue; do not quote raw OCR.\n"
    "  - Keep it cinematic and propulsive; no meta, no 'panel'/'scene' talk.\n"
)

# genai response schema (UPPERCASE type enums; lowered for ollama's `format`).
_TEASER_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
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
    "required": ["panel_narration", "rewind_line"],
}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="bundle-level arc teaser planner (montage selector + LLM narration + synthetic dir)")
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
    ap.add_argument("--shortlist-n", type=int, default=4,
                    help="DEPRECATED: the montage selector ignores this (kept for "
                         "backward-compatible invocation)")
    ap.add_argument("--min-panels", type=int, default=4)
    ap.add_argument("--max-hook-panels", type=int, default=10,
                    help="max panels in the montage (climax + setup)")
    ap.add_argument("--payoff-tail-frac", type=float, default=0.0,
                    help="0.0 = OFF (default): the power reveal is the hook, not a "
                         "spoiler. Set >0 to trim a literal final-cliffhanger sliver.")
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

    montage = select_montage(
        panels,
        max_panels=args.max_hook_panels,
        min_panels=args.min_panels,
        payoff_tail_frac=args.payoff_tail_frac,
    )
    if not montage:
        print("[teaser] no teaser")
        return 0

    cast = merge_cast(chapter_dirs)
    model_call = _build_model_call(args)
    teaser = select_and_write(
        montage,
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
