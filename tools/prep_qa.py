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
  cover:      panel_uncovered / panel_double_covered (segment spans must
              partition the shown story panels — adaptive flow narration)

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
from beats_segments import beat_segments
from render_prep import multi_scale_contained
from scene_chrome import is_chrome_scene, needs_image_stats
from studio.qa_flags import longest_common_run
from narration_consistency import audio_consistency, strip_chrome_opener
from manifest_freshness import verify_chapter as _verify_chapter_freshness
from manifest_io import read_manifest
from recap_style import (
    analyze_recap_style,
    ends_terminal,
    is_shot_description,
    mentions_figures_leak,
    mentions_image_file,
    mentions_impact_marker,
    mentions_mood_tag_leak,
)
from span_align import (  # single authority: lexicon + line<->span affinity
    _IMPACT_LEXEMES,          # noqa: F401  (re-export; tests use pq._IMPACT_LEXEMES)
    _IMPACT_LEXEME_RE,        # noqa: F401
    SPAN_ALIGN_MARGIN,
    has_impact_lexeme,
    offset_shift_candidate,
    window_affinities,
    window_score_pairs,
)

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
    vitem: Optional[Dict[str, Any]] = None,
    reconciled: bool = False,
    kept_bubbles: bool = True,
) -> List[Dict[str, Any]]:
    """All image-level checks for one shown scenes_clean/ file.

    *doc* (document/UI) and *sys* (system-message) panels keep their text BY
    DESIGN — content checks (husk/card/void/text) do not apply to them.
    *kept_bubbles* (plan bubble_shown_mode == "keep", the default): shown
    bubbles ship AS DRAWN, so the bubble-interior blanking checks
    (dead_box_leak / ghost_text / visible_text / bubble_text_residue) are
    meaningless and skipped — readable bubble text is design, not a miss."""
    flags: List[Dict[str, Any]] = []
    h, w = img.shape[:2]

    if dims_entry and (int(dims_entry.get("w", -1)) != w
                       or int(dims_entry.get("h", -1)) != h):
        flags.append(_flag(
            "stale_dims", ERROR,
            f"plan says {dims_entry.get('w')}x{dims_entry.get('h')}, file is "
            f"{w}x{h} — scenes_clean/ and plan are out of sync",
            scene=name, segment_id=segment_id))

    # A reconciled_seam panel is tall BY DESIGN (spec §5.1) — the seam-merge
    # re-assembled two chunk slices into one contiguous panel — so it is exempt;
    # every non-reconciled panel is still gated.
    if h > 8000 and not reconciled:
        # a "panel" taller than ~8k px is really a whole stitch chunk that the
        # detector failed to segment — a column of panels rendered as one thin
        # strip (ch28/ch38). No legit single panel is this tall (clean-corpus max
        # ~5.2k px), so this is a BLOCKING integrity failure, not a style note:
        # re-stitch + re-detect (the height-capped stitcher + re-tile guard).
        flags.append(_flag("chunk_as_panel", ERROR,
                           f"crop is {h}px tall (h/w={h / max(1, w):.1f}) — a whole "
                           "stitch chunk, not a panel; detection under-segmented "
                           "this region",
                           scene=name, segment_id=segment_id))
    elif h >= 6 * max(1, w):
        flags.append(_flag("extreme_tall", INFO,
                           f"aspect h/w={h / max(1, w):.1f} — scroll shot; "
                           "verify travel speed is watchable",
                           scene=name, segment_id=segment_id))

    # VALIDITY INVARIANT — runs for EVERY shown crop, including sys/doc/branding
    # (no exemption): a shown panel MUST be a real image, never a near-uniform
    # white/black void. A valid dark or bright scene still has structure (std
    # well above zero); a broken crop — an over-inpainted caption card or a
    # failed crop — is near-flat. This is the gap that let an all-black panel
    # pass QA. A title card's styled glyphs keep std high, so real cards survive.
    gray_full = img.mean(axis=2) if img.ndim == 3 else img
    std_full = float(gray_full.std())
    white_frac = float((gray_full > 244).mean())
    black_frac = float((gray_full < 12).mean())
    # empty_field: a crop is also a void when almost every pixel is paper or pure
    # ink (>=235 or <=20) with <=7% real content — catches the "white field + a
    # small dark blob/silhouette" husk that drives std HIGH and so slips the
    # uniform-void test below. (Does NOT catch a speed-line/SFX burst whose
    # anti-aliased edges read as content — that emphasis-husk needs a text-
    # coverage signal; see render_prep husk handling.)
    bg_frac = float(((gray_full >= 235) | (gray_full <= 20)).mean())
    empty_field = bg_frac >= 0.93
    # text-aware: a white/empty FIELD that carries real OCR glyphs (a HUD /
    # system / activation card like "7TH GEN NANO MACHINE, STARTING ACTIVATION")
    # is REAL content, not a void. Only the pure-flat test (std<6) still fires on
    # it (a truly flat frame has no glyphs anyway). This protects HUD/text-on-white
    # reveals the labeller didn't tag sys, the same way doc/sys cards are kept.
    _vt = vitem or {}
    _otxt = str(_vt.get("ocr_clean") or _vt.get("text") or "")
    has_text = (int(_vt.get("n_words") or 0) >= 3
                or float(_vt.get("text_coverage") or 0.0) >= 0.05
                or len(_otxt.split()) >= 3)
    if (std_full < 6.0
            or (((max(white_frac, black_frac) >= 0.97 and std_full < 25.0)
                 or empty_field) and not has_text)):
        kind = "white" if white_frac >= black_frac else "black"
        flags.append(_flag(
            "blank_crop", ERROR,
            f"shown crop is a near-empty {kind} void (std={std_full:.1f}, "
            f"bg={bg_frac:.2f}, white={white_frac:.2f}, black={black_frac:.2f}) — "
            "not a real image; recrop or drop this panel",
            scene=name, segment_id=segment_id))

    if not doc and not sys:
        gray = img.mean(axis=2) if img.ndim == 3 else img
        art = rp.art_content_score(img, [])
        if art < min_art_score and not has_text:
            sev = ERROR if art < 0.7 * min_art_score else WARN
            flags.append(_flag("husk", sev,
                               f"art_score={art:.4f} < {min_art_score} — "
                               + ("no art detail left after cleaning"
                                  if sev == ERROR else
                                  "borderline art detail, eyeball it"),
                               scene=name, segment_id=segment_id))
        midtone = float(((gray > 60) & (gray < 200)).mean())
        if midtone < 0.08 and not rp.story_visual_panel(vitem or {}):
            flags.append(_flag("binary_card", WARN,
                               f"midtone_frac={midtone:.3f} — near-binary "
                               "card (chrome-like), verify it is story",
                               scene=name, segment_id=segment_id))

        if kept_bubbles:
            return flags  # bubbles ship as drawn: no blanking to audit

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

        # Round-2 E2 measurement net (WARN): a speech-shaped bubble on the
        # CLEANED panel whose interior still shows dense strokes = stylized
        # text the cleaner's residue net missed (or a path that bypassed it).
        # Same single authority as the cleaner (rp.bubble_stroke_density) so
        # QA and render_prep can never disagree about "dense".
        dens_hits = [
            (b, d) for b in rp.speech_shaped_boxes(boxes, w)
            for d in (rp.bubble_stroke_density(img, b),)
            if d >= rp.BUBBLE_STROKE_DENSITY_MIN]
        if dens_hits:
            _b, top_d = max(dens_hits, key=lambda t: t[1])
            flags.append(_flag(
                "bubble_text_residue", WARN,
                f"stroke_density={top_d:.3f} >= "
                f"{rp.BUBBLE_STROKE_DENSITY_MIN} inside a cleaned "
                "speech bubble — likely stylized text OCR could not see "
                "(clean residue net missed or bypassed)",
                scene=name, segment_id=segment_id))
    return flags


# ---------------------------------------------------------------------------
# vision / narration / plan checks
# ---------------------------------------------------------------------------

def cross_dup_flags(seq: Sequence[Dict[str, Any]],
                    get_img,
                    narrated: Optional[set] = None) -> List[Dict[str, Any]]:
    """Consecutive shown cuts that are near-identical (or zoom pairs) — the
    on-screen duplicate class the user keeps catching by eye.

    A cut whose panel OWNS its own narration line (`narrated`, basenames) is a
    DISTINCT story beat the writer chose to describe and is NEVER flagged: the
    `multi_scale_contained` test over-fires on distinct action/face panels that
    merely share a composition (real ch1: p043 blade-swing vs p044 debris-strike
    read as a match though their dhash distance is 37; two face shots, 18), and
    dropping a narrated panel makes a neighbour HOLD 12-16s while the narrator
    describes a shot never shown. Truly redundant zoom/near-dups are merged
    upstream (understanding → story_group), before narration — the same
    no-drop-distinct guarantee render_prep already enforces on its dedup passes.
    """
    narrated = narrated or set()
    flags: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    for cur in seq:
        f = str(cur.get("file"))
        if (prev and str(prev.get("file")) != f
                and f not in narrated and parent_scene(f) not in narrated):
            ia, ib = get_img(str(prev.get("file"))), get_img(f)
            if ia is not None and ib is not None and (
                    multi_scale_contained(ib, ia)
                    or multi_scale_contained(ia, ib)):
                flags.append(_flag(
                    "cross_dup", ERROR,
                    f"near-duplicate of the previous cut "
                    f"({prev.get('file')} in {prev.get('segment_id')})",
                    scene=f, segment_id=str(cur.get("segment_id") or "")))
        prev = cur
    return flags


def near_dup_residual_flags(seq: Sequence[Dict[str, Any]],
                            get_img,
                            get_boxes,
                            *,
                            is_exempt=None,
                            ham_max: int = 8) -> List[Dict[str, Any]]:
    """TRIPWIRE for a near-duplicate panel that survived the render_prep dedup
    ladder — the residual the user keeps catching by eye. Cleaning removes only
    the text inside a bubble (the outline stays), so two identical drawings under
    different dialogue split a raw perceptual hash and slip the ladder; here we
    bubble-MASK the hash (`rp._mask_bubbles_for_hash`, via `rp._dhash8_bgr`'s
    boxes arg) over CONSECUTIVE shown cuts so identical art collapses to the same
    hash and the pair surfaces.

    A WARN, never an ERROR and never auto-dropped (kept out of _VISUAL_DROPPABLE):
    auto-dropping a residual would re-hit the sole-cut-empties-a-segment problem —
    render_prep (bubble-masked hashing + canonicalize) is the real fix, this is
    only the tripwire that turns a regression yellow instead of shipping silent.
    A same-file hold (prev == cur, a deliberate continuous shot) is never a dup;
    system/doc panels (a shared UI frame carrying different text) are exempt via
    *is_exempt*. *get_boxes*(f) supplies the per-file bubble boxes to mask."""
    exempt = is_exempt or (lambda f: False)
    flags: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    for cur in seq:
        if cur.get("branding"):
            prev = None                 # branding is never a dup nor a reference
            continue
        f = str(cur.get("file"))
        pf = str(prev.get("file")) if prev else ""
        if (prev and pf != f                     # distinct files (not a hold)
                and not exempt(f) and not exempt(pf)):
            ia, ib = get_img(pf), get_img(f)
            if ia is not None and ib is not None:
                ha = rp._dhash8_bgr(ia, get_boxes(pf))
                hb = rp._dhash8_bgr(ib, get_boxes(f))
                if (ha ^ hb).bit_count() <= ham_max:
                    flags.append(_flag(
                        "near_dup_residual", WARN,
                        f"near-identical to the previous cut after bubble-masking "
                        f"({pf} in {prev.get('segment_id')}) — a duplicate slipped "
                        f"the render_prep dedup ladder",
                        scene=f, segment_id=str(cur.get("segment_id") or "")))
        prev = cur
    return flags


def dup_shown_flags(seq: Sequence[Dict[str, Any]],
                    get_raw_img,
                    get_raw_boxes,
                    get_ocr,
                    *,
                    is_exempt=None,
                    window: int = 8,
                    ham_max: int = 8,
                    ham_max_contained: int = 14,
                    cap_kept_pairs=None) -> List[Dict[str, Any]]:
    """BLOCKING tripwire for the shown-twin INVARIANT: no two shown panels may
    be masked-raw twins. Uses the exact predicate render_prep enforces with
    (`rp.twin_verdict` over `rp._dhash8_bgr` of the RAW `scenes/` panels,
    bubbles masked) so enforcement and QA cannot drift — render_prep's final
    invariant pass folds every such pair, making this flag never fire in
    practice; when a FUTURE gate/bypass ships a twin pair anyway, this blocks
    the chapter instead of shipping an on-screen duplicate (the p054/p055 echo
    pair escaped every per-pass gate and shipped silently).

    Compares DISTINCT files among shown cuts within a sliding *window* of
    shown cuts (mirrors the invariant pass; a deliberate far-apart flashback
    callback is out of window). A same-file pair is never a dup here — holds
    and capped repeats are deliberate and governed by held_repeat/long_hold.
    *is_exempt* files (system/doc — shared UI frames under different text) and
    files with no raw scene image (split halves) are never compared. One flag
    per offending pair."""
    exempt = is_exempt or (lambda f: False)
    # render_prep's DOCUMENTED cap exception (plan `twin_cap_kept`,
    # 2026-07-20): a twin pair deliberately kept apart because folding it
    # would create an over-hold-cap stand-in is LEGAL (ken echo styles it) —
    # the same authority stamps it, so enforcement and QA cannot drift.
    cap_ok = {tuple(sorted((str(a), str(b))))
              for a, b in (cap_kept_pairs or [])}
    hashes: Dict[str, Optional[int]] = {}

    def _h(f: str) -> Optional[int]:
        if f not in hashes:
            img = get_raw_img(f)
            hashes[f] = (None if img is None
                         else rp._dhash8_bgr(img, get_raw_boxes(f)))
        return hashes[f]

    ent = [c for c in seq if not c.get("branding")]
    flags: List[Dict[str, Any]] = []
    seen_pairs: set = set()
    n = len(ent)
    for j in range(n):
        fj = str(ent[j].get("file") or "")
        if not fj or exempt(fj) or _h(fj) is None:
            continue
        for i in range(max(0, j - window), j):
            fi = str(ent[i].get("file") or "")
            if not fi or fi == fj or exempt(fi) or _h(fi) is None:
                continue
            ham = (_h(fi) ^ _h(fj)).bit_count()      # type: ignore[operator]
            if not rp.twin_verdict(ham, get_ocr(fi), get_ocr(fj),
                                   ham_max=ham_max,
                                   ham_max_contained=ham_max_contained):
                continue
            key = tuple(sorted((fi, fj)))
            if key in seen_pairs or key in cap_ok:
                continue
            seen_pairs.add(key)
            flags.append(_flag(
                "dup_shown", ERROR,
                f"shown panel is a masked-raw twin of {fi} "
                f"(in {ent[i].get('segment_id')}, masked ham={ham}) — a "
                "duplicate bypassed the render_prep dedup invariant",
                scene=fj, segment_id=str(ent[j].get("segment_id") or "")))
    return flags


def echo_exempt_fn(dims: Dict[str, Any], vitems: Dict[str, Any]):
    """STAMPED-only exemption for the perceptual_echo net — mirrors
    _static_ceiling_exempt minus the aspect clause: doc dims + stamped
    panel_kind=='system'. Deliberately NOT _qa_exempt: that one honors
    scene_dims' PIXEL-level 'sys' flag, which the system-box YOLO overfires
    on mere SFX/bubble text — the p000090/p000095 incident panels all
    carried sys:True and would self-exempt the exact evidence class this
    net exists to catch."""
    def _exempt(f: str) -> bool:
        d = dims.get(f) or {}
        if d.get("doc"):
            return True
        pk = str((vitems.get(parent_scene(f)) or vitems.get(f)
                  or {}).get("panel_kind") or "").strip().lower()
        return pk == "system"
    return _exempt


def perceptual_echo_flags(seq: Sequence[Dict[str, Any]],
                          get_clean_img,
                          get_clean_boxes,
                          get_raw_img,
                          get_raw_boxes,
                          *,
                          is_exempt=None,
                          window: int = 3,
                          ham_max: int = 8) -> List[Dict[str, Any]]:
    """V2 tripwire (WARN, measure-first — never blocking, never dropped): two
    nearby shown cuts whose SHOWN CROPS are bubble-masked dhash twins (ham <=
    *ham_max*) while their RAW panels are NOT (raw masked ham > *ham_max*).
    The masked-RAW invariant (dup_shown) correctly treats them as distinct
    panels — an artist zoom-echo (p000044 tight-re-crops p000043's lower
    half) or a husk re-crop (p000095's art band == p000090) — but the viewer
    reads the same picture twice. render_prep.ken_differentiate_echo_pairs is
    the fix (distinct ken regions); this measures what shipped.

    Pairs compared within a sliding *window* of shown cuts (|i-j| < window).
    Same-file pairs (holds/ken-variety sub-cuts) never flag; *is_exempt*
    files (system/doc — shared UI frames) and files without a raw scene image
    (split halves — raw-distinctness unprovable) are skipped. One flag per
    pair, carrying BOTH ham values."""
    exempt = is_exempt or (lambda f: False)
    # NOTE: deliberate divergence from render_prep.ken_differentiate_echo_pairs'
    # window arithmetic — QA walks the full SHOWN stream (split2 file2 halves
    # included via iter_shown_cuts); enforcement only walks modifiable cuts.
    ent = [c for c in seq if not c.get("branding")]
    ch: Dict[str, Optional[int]] = {}
    rh: Dict[str, Optional[int]] = {}

    def _ch(f: str) -> Optional[int]:
        if f not in ch:
            img = get_clean_img(f)
            ch[f] = (None if img is None
                     else rp._dhash8_bgr(img, get_clean_boxes(f)))
        return ch[f]

    def _rh(f: str) -> Optional[int]:
        if f not in rh:
            img = get_raw_img(f)
            rh[f] = (None if img is None
                     else rp._dhash8_bgr(img, get_raw_boxes(f)))
        return rh[f]

    flags: List[Dict[str, Any]] = []
    seen_pairs: set = set()
    for j in range(len(ent)):
        fj = str(ent[j].get("file") or "")
        if not fj or exempt(fj):
            continue
        for i in range(max(0, j - (window - 1)), j):
            fi = str(ent[i].get("file") or "")
            if not fi or fi == fj or exempt(fi):
                continue
            key = tuple(sorted((fi, fj)))
            if key in seen_pairs:
                continue
            if _ch(fi) is None or _ch(fj) is None:
                continue
            sham = (_ch(fi) ^ _ch(fj)).bit_count()   # type: ignore[operator]
            if sham > ham_max:
                continue
            if _rh(fi) is None or _rh(fj) is None:
                continue                # split half etc: raw unprovable
            rham = (_rh(fi) ^ _rh(fj)).bit_count()   # type: ignore[operator]
            if rham <= ham_max:
                continue                # raw twins — dup_shown's (blocking) job
            seen_pairs.add(key)
            flags.append(_flag(
                "perceptual_echo", WARN,
                f"shown crop reads as the same picture as {fi} "
                f"(in {ent[i].get('segment_id')}): shown-crop masked "
                f"ham={sham} <= {ham_max} while the RAW panels are distinct "
                f"(raw masked ham={rham}) — a zoom-echo/husk-crop echo; "
                "ken differentiation should vary the pair",
                scene=fj, segment_id=str(ent[j].get("segment_id") or "")))
    return flags


def vision_flags(parent: str, vitem: Dict[str, Any], *,
                 dims_entry: Optional[Dict[str, Any]],
                 series_title: Optional[str],
                 midtone_frac: Optional[float] = None,
                 segment_id: str = "") -> List[Dict[str, Any]]:
    d = dims_entry or {}
    flags: List[Dict[str, Any]] = []
    if is_chrome_scene(vitem, series_title=series_title,
                       midtone_frac=midtone_frac):
        flags.append(_flag("chrome_leak", ERROR,
                           f"chrome per scene_chrome rules is SHOWN — "
                           f"ocr={str(vitem.get('ocr_clean'))[:80]!r}",
                           scene=parent, segment_id=segment_id))
    # a stamped system card is INTENTIONALLY shown (its on-screen UI text IS the
    # story beat); the base 'empty/bubble-only' mark it carried BEFORE
    # apply_system_card_overrides reclassified it must not re-flag it once the
    # planner/render_prep (correctly) keep it. Same panel_kind=='system' predicate
    # the system_card_unshown gate + the render protections use.
    if (rp.empty_bubble_panel(vitem)
            and str(vitem.get("panel_kind") or "").strip().lower() != "system"):
        flags.append(_flag("empty_bubble_shown", ERROR,
                           "panel understanding marks this as empty / "
                           "speech-bubble-only, but it is still shown",
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
    dm = _DANGLING_QUOTE_RE.search(text)
    if dm and len(dm.group(1).replace("...", " ").split()) <= 3:
        flags.append(_flag(
            "fragment_dangle", ERROR,
            f"narration ENDS on a dangling quoted stub ({dm.group(1)!r}) — "
            "the thought must flow into the next line, not hang",
            segment_id=segment_id))
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


# ---------------------------------------------------------------------------
# narration <-> image alignment (stale-manifest class + semantic judge)
# ---------------------------------------------------------------------------

# narration ENDING on a short quoted stub trailing into '...' — half a
# sentence presented as a complete thought ("And I..." regression)
_DANGLING_QUOTE_RE = re.compile(
    r'[:,]?\s*["‘’“”\']([^"‘’“”\']'
    r'{1,40}\.\.\.)["‘’“”\']\s*$')

_MOOD_TAG_RE = re.compile(r"\[[a-z][a-z _-]{1,18}\]", re.I)
_NORM_NARR_RE = re.compile(r"[^a-z0-9]+")
_SEG_GROUP_RE = re.compile(r"g(\d{4})_p\d+$")
_CHAPTER_HEADING_RE = re.compile(r"\b(?:chapter|episode)\s+\d+\b", re.I)
_TITLE_CARD_RE = re.compile(r"\b(?:chapter|episode|title)\s+card\b", re.I)


def _norm_narr(s: str) -> str:
    return _NORM_NARR_RE.sub(" ", _MOOD_TAG_RE.sub(" ", s or "").lower()
                             ).strip()


def _alignment_beat_narration(beat: Dict[str, Any]) -> str:
    narr = strip_chrome_opener(str((beat or {}).get("narration") or ""))
    title = str((beat or {}).get("beat_title") or "")
    if _CHAPTER_HEADING_RE.search(narr) or _TITLE_CARD_RE.search(title):
        hook = strip_chrome_opener(str((beat or {}).get("hook") or ""))
        if hook and not _CHAPTER_HEADING_RE.search(hook):
            return hook
        return "The truth is about to surface."
    return narr


def alignment_flags(plan: Dict[str, Any], beats_obj: Dict[str, Any],
                    groups_obj: Dict[str, Any], script_obj: Dict[str, Any],
                    *, min_sim: float = 0.55) -> List[Dict[str, Any]]:
    """The stale-manifest failure class: beats that no longer cover every
    group (interrupted re-run), and verbatim plan text that diverged from the
    beat narration it was copied from (script.json older than beats.json).
    Both are mechanical staleness — the worker may self-heal by re-running
    the beated/scripted stages; prose is never rewritten by a judge."""
    flags: List[Dict[str, Any]] = []
    bn: Dict[int, str] = {}
    for b in (beats_obj or {}).get("beats") or []:
        try:
            bn[int(b.get("group_id"))] = _alignment_beat_narration(b)
        except (TypeError, ValueError):
            continue
    gids = set()
    for sh in (groups_obj or {}).get("shots") or []:
        try:
            gids.add(int(sh.get("group_id")))
        except (TypeError, ValueError):
            continue
    missing = sorted(g for g in gids if g not in bn)
    if missing:
        flags.append(_flag(
            "beats_incomplete", ERROR,
            f"beats cover {len(bn)}/{len(gids)} groups — missing group_ids "
            f"{missing[:8]} — re-run the beated stage (resume), then "
            "re-script"))
    if str((script_obj or {}).get("narration_source")) != "gemini_verbatim":
        return flags        # non-verbatim text legitimately diverges
    from difflib import SequenceMatcher
    plan_items = []
    for item in (plan or {}).get("timeline") or []:
        if item.get("branding"):
            continue
        seg = str(item.get("segment_id") or "")
        m = _SEG_GROUP_RE.match(seg)
        if not m:
            continue
        plan_items.append((int(m.group(1)), seg, str(item.get("tts_text") or "")))

    # Per-panel narration emits ONE plan segment per panel (g####_p##), but `bn`
    # is keyed per GROUP (the joined group narration). Always re-join a group's
    # per-panel plan text before comparing to its group narration — otherwise each
    # single panel line reads as "stale" against the full group text (false
    # narration_stale flood). (This used to be gated on the now-removed `microbeats`
    # flag; per-panel narration makes it the universal case.)
    grouped: Dict[int, List[str]] = {}
    first_seg: Dict[int, str] = {}
    for gid, seg, text in plan_items:
        grouped.setdefault(gid, []).append(text)
        first_seg.setdefault(gid, seg)
    compare_items = [
        (gid, first_seg.get(gid, f"g{gid:04d}_p00"), " ".join(texts))
        for gid, texts in grouped.items()
    ]

    from sfx_scrub import scrub_sfx_quotes  # mirror the script stage's scrub
    for gid, seg, text in compare_items:
        narr = bn.get(gid)
        # scrub series-intro/title-card chrome AND SFX/onomatopoeia/fragment quotes
        # from the beats side too, matching what the script stage actually voices —
        # otherwise a legitimately-scrubbed plan reads as "stale" against the raw,
        # un-scrubbed beats line (false-positive narration_stale flood).
        a, b = (_norm_narr(text),
                _norm_narr(scrub_sfx_quotes(strip_chrome_opener(narr or ""))))
        if not a or not b:
            continue
        sim = SequenceMatcher(None, a, b).ratio()
        if sim < min_sim:
            flags.append(_flag(
                "narration_stale", ERROR,
                f"plan text diverges from this group's beat narration "
                f"(sim {sim:.2f}) — script.json predates "
                "manifest.beats.json; re-run the scripted stage",
                segment_id=seg))
    return flags


def audio_flags(plan: Dict[str, Any],
                tts_index: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic audio↔narration gate: the voiced clips must have been
    voiced from the CURRENT narration. Each clip stores a text_sha; a mismatch
    means the beats/script were regenerated after voicing and the spoken audio
    is now stale (the bug the user caught by ear). $0, no LLM — re-voice the
    flagged segments (the voiced stage does this incrementally)."""
    # the staleness gate only means anything for a VOICED plan (built FROM the
    # clips). A pre-voiceover ESTIMATE plan (no source_tts_index) is timed from
    # word counts; the same signal the per-item loop uses to emit estimate_plan.
    voiced_plan = bool((plan or {}).get("source_tts_index"))
    if not (tts_index or {}).get("clips"):
        # a plan built voiced (source_tts_index set) but with no clip index is a
        # hard error — never silently pass it as "not voiced yet"
        if voiced_plan:
            return [_flag(
                "audio_index_missing", ERROR,
                "plan was built voiced (source_tts_index set) but "
                "tts/tts_index.json has no clips — run/repair the voiced stage")]
        return []                       # genuinely not voiced yet — nothing to check
    if not voiced_plan:
        # ESTIMATE phase with clips on disk = LEFTOVERS from a prior run. They
        # will be re-voiced after story approval (the voiced stage is
        # incremental, keyed on text_sha), so stale text here is expected and
        # harmless — NOT an error. The real audio<->narration gate runs once the
        # plan is rebuilt voiced. (Without this, re-preparing any chapter that
        # was voiced before fails QA on its own soon-to-be-replaced audio.)
        return []
    r = audio_consistency(plan, tts_index)
    flags: List[Dict[str, Any]] = []
    for seg in r["stale"]:
        flags.append(_flag(
            "audio_stale", ERROR,
            "voiceover audio was voiced from DIFFERENT text than the current "
            "narration — re-voice this segment (beats/script changed after "
            "voicing)", segment_id=seg))
    for seg in r["missing"]:
        flags.append(_flag(
            "audio_missing", ERROR,
            "narrated segment has no voiced clip — run the voiced stage",
            segment_id=seg))
    # fail-closed on TTS failures: a clip that exhausted retries ships as a
    # SILENCE placeholder whose text_sha MATCHES the narration, so the
    # staleness gate above can't see it — without this check the mute
    # segment passes QA, renders as dead air, and is cached forever.
    for clip in (tts_index or {}).get("clips") or []:
        if isinstance(clip, dict) and clip.get("tts_failed"):
            seg = str(clip.get("segment_id") or "")
            wav = str(clip.get("audio_file") or f"clips/{seg}.wav")
            flags.append(_flag(
                "audio_failed", ERROR,
                f"TTS synthesis failed for this segment — {wav} is a silence "
                "placeholder, not voiced audio; re-run the voiced stage",
                segment_id=seg))
    return flags


# the vision layer's own device/app chrome patterns, reused PER TOKEN
def _ui_noise_res():
    try:
        from vision_extract import VisionConfig
        pats = list(VisionConfig().ui_noise_patterns)
    except Exception:                                            # pragma: no cover
        pats = [r"\bLTE\b", r"\b5G\b", r"\bPM\b", r"\bAM\b",
                r"\b\d{1,3}%\b", r"\b\d{1,2}:\d{2}\b", r"\b\d+/\d+\b"]
    # generic scan/UI fragments a per-line strip cannot see: a token with a
    # digit glued to punctuation, or an all-caps stub of <=2 letters
    pats += [r"^\W*\d+\W+$", r"^\d+[)\]}]$", r"^[^a-z0-9]+$"]
    return tuple(re.compile(p, re.IGNORECASE) for p in pats)


_UI_NOISE_RES = _ui_noise_res()

# generic English function words short enough to survive the <=3-char filter
_SHORT_FUNCTION_WORDS = {
    "a", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is",
    "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
    "i", "am", "are", "was", "who", "why", "how", "you", "him", "her", "his",
    "our", "not", "now", "all", "any", "but", "for", "the", "and", "out",
}
_UI_TOKENS = {"read", "ep", "episode", "episodes", "comments", "comment",
              "views", "view", "likes", "like", "subscribe", "next", "prev",
              "previous", "tap", "menu", "notice", "unread"}


def _stem(w: str) -> str:
    """Cheap suffix strip so a caption's word matches the narration's inflected
    form (fade/fades, swipe/swipes/swiping). Generic English morphology."""
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _caption_word_covered(word: str, narration_words: set) -> bool:
    """True when the narration carries *word*: literally, by inflection, or as
    an OCR mis-scan of a word it does use ('APOCALYPSH' for 'apocalypse').
    Webtoon captions are painted text — OCR routinely drops or mangles a
    character, and a literal-membership test then reports a voiced caption as
    dropped (ORV Ep1 g0001: 47% literal coverage on a caption the narration
    quotes almost verbatim)."""
    if word in narration_words:
        return True
    st = _stem(word)
    if any(st == _stem(n) for n in narration_words):
        return True
    if len(word) >= 6:
        import difflib
        return any(difflib.SequenceMatcher(None, word, n).ratio() >= 0.85
                   for n in narration_words if abs(len(n) - len(word)) <= 3)
    return False


def caption_terms(ocr_clean: str,
                  understood_panel: Optional[Dict[str, Any]] = None) -> set:
    """The VOICEABLE words of a panel's OCR — the denominator of the caption
    coverage test. The raw OCR also carries painted SFX glyphs, device/app
    chrome and mis-scan fragments, i.e. exactly the text every writer stage is
    told NOT to voice; demanding literal coverage of those manufactured
    caption_unvoiced ERRORs (ORV Ep0 p000003: 12 of 18 'missing' tokens were
    noise). Filters, all generic: digits, the UI-token list, the model's OWN
    sfx_text transcription for THAT panel (so no glyph list is hardcoded), and
    <=3-character tokens that are not English function words."""
    words = [w for w in _norm_narr(ocr_clean or "").split()
             if not w.isdigit() and w not in _UI_TOKENS]
    sfx = set(_norm_narr(str((understood_panel or {}).get("sfx_text") or "")).split())
    keep = set()
    for w in words:
        if w in sfx:
            continue
        if len(w) <= 3 and w not in _SHORT_FUNCTION_WORDS:
            continue
        # device/app chrome: vision_extract strips these per LINE, so a token
        # sitting inside a mixed line ('SWIPE', '40)', '90%') survives into the
        # caption denominator — apply the same patterns per TOKEN
        if any(rx.search(w) for rx in _UI_NOISE_RES):
            continue
        keep.add(w)
    return keep


def caption_unvoiced_flags(beats_obj: Dict[str, Any],
                           vitems: Dict[str, Dict[str, Any]],
                           *, min_words: int = 4,
                           min_coverage: float = 0.5,
                           understood_by_file: Optional[Dict[str, Any]] = None,
                           arbitrate: Optional[Callable[[str, str], bool]]
                           = None) -> List[Dict[str, Any]]:
    """User contract: showing caption boxes is optional, VOICING them is
    mandatory — text-only/recovered panels carry the author's monologue
    ('ON THE DAY I FINISHED THE WEB NOVEL...') and their content must be
    woven into that group's narration."""
    flags: List[Dict[str, Any]] = []
    for b in (beats_obj or {}).get("beats") or []:
        nwords = set(_norm_narr(b.get("narration") or "").split())
        for sf in b.get("scene_files") or []:
            it = vitems.get(str(sf)) or {}
            if not (it.get("text_only") or it.get("recovered")):
                continue
            txt = str(it.get("ocr_clean") or "")
            try:
                import scene_chrome as _sc
                if _sc.is_chrome_scene({"ocr_clean": txt,
                                        "panel_kind": it.get("panel_kind")}):
                    continue   # resurrected end-cards/plugs are not captions
            except Exception:
                pass
            # app-UI screens are text_only too — their button/counter noise
            # ("READ EPISODE", "VIEWS: 1") is not monologue; don't demand it
            cwords = caption_terms(txt, (understood_by_file or {}).get(str(sf)))
            if len(cwords) < min_words:
                continue
            cov = sum(1 for w in cwords if _caption_word_covered(w, nwords)) \
                / max(1, len(cwords))
            if cov < min_coverage:
                narr = str(b.get("narration") or "")
                if arbitrate is not None and arbitrate(txt, narr):
                    flags.append(_flag(
                        "caption_paraphrased", WARN,
                        f"caption carried by PARAPHRASE (judge-accepted, "
                        f"{int(cov * 100)}% literal): {txt[:70]!r}",
                        scene=str(sf),
                        segment_id=f"g{int(b.get('group_id') or 0):04d}"))
                    continue
                flags.append(_flag(
                    "caption_unvoiced", ERROR,
                    f"caption text missing from narration "
                    f"({int(cov * 100)}% word coverage): {txt[:70]!r}",
                    scene=str(sf),
                    segment_id=f"g{int(b.get('group_id') or 0):04d}"))
    return flags


def system_coverage_flags(beats_obj: Dict[str, Any],
                          plan: Dict[str, Any],
                          vitems: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Authoritative check keyed on the stamped panel_kind=='system': every
    panel the understanding labelled 'system' (an in-world status screen,
    system message, etc.) MUST appear in at least one shown cut.  This is
    independent of the OCR-heuristic system_card_dropped WARN in story_flags
    — that check stays as a belt-and-suspenders signal; this one is the hard
    ERROR gate that defers entirely to the stamped kind (no regex)."""
    flags: List[Dict[str, Any]] = []
    shown = {_base_scene(str(c["file"])) for c in iter_shown_cuts(plan)}
    for b in (beats_obj or {}).get("beats") or []:
        for sf_raw in b.get("scene_files") or []:
            sf = str(sf_raw)
            vit = vitems.get(sf) or {}
            if str(vit.get("panel_kind") or "").lower() != "system":
                continue
            if _base_scene(sf) not in shown:
                flags.append(_flag(
                    "system_card_unshown", ERROR,
                    f"in-world system panel {sf!r} is not shown in any cut — "
                    "system cards are story beats and must appear on screen",
                    scene=sf))
    return flags


def _stitch_page_count(stitch_path: str) -> int:
    """Distinct source pages a chapter stitched == the pages it fetched."""
    try:
        with open(stitch_path) as f:
            m = json.load(f)
    except Exception:
        return 0
    srcs = set()
    for ch in m.get("chunks") or []:
        for s in ch.get("sources") or []:
            srcs.add(str(s))
    return len(srcs)


def page_floor_flags(ep: str) -> List[Dict[str, Any]]:
    """Cross-chapter integrity net for OPAQUE-name sources (asura hash/_pN),
    where the numeric-contiguity gate is blind because the filenames carry no
    sequence. A chapter that stitched FAR fewer pages than its series siblings is
    a likely truncated/partial fetch. WARN only, and the floor sits well below
    the median (0.45×) so a legitimately short chapter never trips it."""
    try:
        this_n = _stitch_page_count(os.path.join(ep, "manifest.stitch.json"))
        if this_n <= 0:
            return []
        series_dir = os.path.dirname(ep.rstrip("/"))
        counts: List[int] = []
        for name in os.listdir(series_dir):
            d = os.path.join(series_dir, name)
            if not os.path.isdir(d) or os.path.abspath(d) == os.path.abspath(ep):
                continue
            n = _stitch_page_count(os.path.join(d, "manifest.stitch.json"))
            if n > 0:
                counts.append(n)
        if len(counts) < 5:
            return []                      # too few siblings for a stable median
        counts.sort()
        median = counts[len(counts) // 2]
        floor = 0.45 * median
        if median >= 4 and this_n < floor:
            return [_flag(
                "low_page_count", WARN,
                f"stitched {this_n} pages vs series median {median} "
                f"(floor {floor:.0f}) — possible truncated/partial fetch; "
                f"re-fetch this chapter and compare")]
    except Exception:
        pass
    return []


def sfx_voiced_flags(script_obj: Any) -> List[Dict[str, Any]]:
    """The VOICED script text (post-scrub) still containing a sound-effect/scream
    quote ("EUAACK!! ACK!!!", "HUH... HUH?!", "Keuk...!") — i.e. the verbatim SFX
    scrub MISSED one. 0 = confirmed no SFX is read aloud."""
    from sfx_scrub import sfx_quotes
    flags: List[Dict[str, Any]] = []
    if not isinstance(script_obj, dict):
        return flags
    for si, sec in enumerate(script_obj.get("sections") or []):
        texts: List[str] = []
        for key in ("tts_paragraphs_v3", "script_paragraphs"):
            v = sec.get(key)
            if isinstance(v, list):
                texts += [x if isinstance(x, str)
                          else str((x or {}).get("text") or (x or {}).get("line") or "")
                          for x in v]
            elif isinstance(v, str):
                texts.append(v)
        for t in texts:
            for q in sfx_quotes(t):
                flags.append(_flag(
                    "sfx_voiced", ERROR,
                    f"voiced text contains a sound-effect/scream quote '{q[:30]}' — "
                    "the SFX scrub missed it; re-narrate as described action",
                    segment_id=str(sec.get("section_index", si))))
    return flags


def raw_caps_voiced_flags(script_obj: Any) -> List[Dict[str, Any]]:
    """AGNOSTIC OCR-dump check (no word list): voiced text reading a run of >=3
    consecutive ALL-CAPS words ('WHAT MORE DO YOU WANT FROM ME') is raw bubble OCR
    being read aloud, not story narration. Fires on any manhwa whose bubbles are
    capitalised (the universal webtoon case); paraphrased narration is sentence
    case and never trips it."""
    flags: List[Dict[str, Any]] = []
    if not isinstance(script_obj, dict):
        return flags
    for si, sec in enumerate(script_obj.get("sections") or []):
        texts: List[str] = []
        for key in ("tts_paragraphs_v3", "script_paragraphs"):
            v = sec.get(key)
            if isinstance(v, list):
                texts += [x if isinstance(x, str)
                          else str((x or {}).get("text") or (x or {}).get("line") or "")
                          for x in v]
            elif isinstance(v, str):
                texts.append(v)
        for t in texts:
            body = re.sub(r"^\s*\[[^\]]*\]\s*", "", t)      # drop a leading [mood] tag
            run = worst = 0
            for w in body.split():
                if (re.fullmatch(r"[A-Z][A-Z'’.!?,]*", w)
                        and sum(c.isalpha() for c in w) >= 2):
                    run += 1
                    worst = max(worst, run)
                else:
                    run = 0
            if worst >= 3:
                flags.append(_flag(
                    "raw_caps_voiced", ERROR,
                    f"voiced text reads {worst} consecutive ALL-CAPS words — raw "
                    "bubble OCR read aloud; paraphrase dialogue, don't read the page",
                    segment_id=str(sec.get("section_index", si))))
    return flags


def shot_description_flags(beats_obj: Any) -> List[Dict[str, Any]]:
    """A narration line that NAMES the shot/camera/panel/frame instead of
    narrating the story ('A close-up shot shows...', 'The panel focuses on...').
    This is camera-prose understanding `description` leaking verbatim into the
    voiced line — it must be re-narrated as story. Iterates `beat_segments`
    (native flow spans AND legacy panel_narration, adapted to singleton spans)
    so a flow-span line is screened too. ERROR + healable per group
    (segment_id g####); the span head carries the thumb."""
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict):
        return flags
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            line = s["line"]
            if line and is_shot_description(line):
                flags.append(_flag(
                    "shot_description", ERROR,
                    f"narration names the shot/camera, not the story: {line[:80]!r} "
                    "— re-narrate what HAPPENS in the panel",
                    scene=str((s["span"] or [""])[0]),
                    segment_id=seg))
    return flags


def truncated_line_flags(beats_obj: Any) -> List[Dict[str, Any]]:
    """A voiced line that stops mid-sentence ("But there is no mercy to be
    found, only the" — real Nano ch1 g0011_p16, 2026-07-06 review): the
    writer truncated its final passage sentence, and no guard fired because
    the line ended on a bare word, not a , ; or :. Writer-side nets (the
    sentence rejoin in segments_from_sentences + the fragment repair's
    amputation) stop it at source; this is the QA TRIPWIRE so any dangle
    that reaches a manifest is an ERROR the auto-heal re-narrates.
    Deterministic: terminal punctuation via the same recap_style authority
    the repair uses."""
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict):
        return flags
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            line = s["line"]
            if line and not ends_terminal(line):
                flags.append(_flag(
                    "truncated_line", ERROR,
                    f"narration stops mid-sentence: {line[:80]!r} — the "
                    "thought never ends; finish it or re-narrate the group",
                    scene=str((s["span"] or [""])[0]),
                    segment_id=seg))
    return flags


def filename_in_narration_flags(beats_obj: Any) -> List[Dict[str, Any]]:
    """A VOICED line that names an image file ("It progresses through the
    series to conclude at p000032.jpg.") is pipeline bookkeeping read aloud —
    the prose-first writer receives scene_file names as sentence tags, so a
    tag can leak into the passage. Writer-side gates (meta-garbage retry +
    validate_segments re-ask) stop it at source; this flag is the QA NET so a
    leak that ever reaches a manifest is an ERROR the auto-heal re-narrates
    (span-pinned). Deterministic — the semantic grounding judge scored the
    real ch1 leak only WARN 'vague filler', which deliberately does not heal."""
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict):
        return flags
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            line = s["line"]
            if line and mentions_image_file(line):
                flags.append(_flag(
                    "filename_in_narration", ERROR,
                    f"narration names an image file: {line[:80]!r} — file "
                    "names are pipeline bookkeeping, never story; re-narrate "
                    "what happens across these panels",
                    scene=str((s["span"] or [""])[0]),
                    segment_id=seg))
    return flags


def impact_marker_leak_flags(beats_obj: Any) -> List[Dict[str, Any]]:
    """A VOICED line that echoes the writer-input impact-SFX bracket marker
    ("[IMPACT SFX on panel]", stamped into the payload by
    tools/gemini_narrative_pass.py's _pack_group_payload) is pipeline
    bookkeeping read aloud — the SAME leak channel as
    filename_in_narration_flags (scene_file names are also fed to the writer
    as bracket/tag context that can echo back verbatim instead of being
    converted into prose). Deterministic substring match; fires regardless of
    whether the line ALSO happens to carry an impact lexeme — a leaked marker
    is unshippable either way, lexicon or not."""
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict):
        return flags
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            line = s["line"]
            if line and mentions_impact_marker(line):
                flags.append(_flag(
                    "impact_marker_leak", ERROR,
                    f"narration echoes the impact-SFX bracket marker "
                    f"verbatim: {line[:80]!r} — describe the strike/stab/"
                    "blow itself, never the bracket tag",
                    scene=str((s["span"] or [""])[0]),
                    segment_id=seg))
    return flags


def figures_leak_flags(beats_obj: Any) -> List[Dict[str, Any]]:
    """A VOICED line that echoes the writer-input FIGURES payload's
    unresolved-figure wrapper ("unknown (<evidence>)", stamped into the
    payload by tools/gemini_narrative_pass.py's _pack_group_payload from
    tools/cast_identity.py's resolve_figures) is pipeline bookkeeping read
    aloud — the SAME leak channel as filename_in_narration_flags /
    impact_marker_leak_flags (mirrors it exactly). A resolved cast NAME in
    narration is sanctioned (FIGURES ARE GROUND TRUTH is the whole point of
    the feature) and never fires this; only the raw 'unknown (' evidence
    format is a leak. Deterministic substring match."""
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict):
        return flags
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            line = s["line"]
            if line and mentions_figures_leak(line):
                flags.append(_flag(
                    "figures_leak", ERROR,
                    f"narration echoes the unresolved-figure payload "
                    f"wrapper verbatim: {line[:80]!r} — use neutral "
                    "phrasing (the masked figure, the man in the hood), "
                    "never the raw 'unknown (...)' evidence text",
                    scene=str((s["span"] or [""])[0]),
                    segment_id=seg))
    return flags


def mood_tag_leak_flags(beats_obj: Any) -> List[Dict[str, Any]]:
    """A VOICED line that OPENS with a bare (unbracketed) mood/tone word
    immediately followed by a fresh capitalized sentence ("Dramatic: He's
    tumbling…", "Comic: The masked guy…" — the round-3 Nano ch1 regression,
    18 segments) is pipeline/authoring vocabulary read aloud — the SAME leak
    channel as impact_marker_leak_flags / figures_leak_flags (mirrors them
    exactly), just missing its brackets. The sanctioned form is ALWAYS
    bracketed ("[dramatic] He's…"), added by the packer, never the writer.
    Deterministic pattern match; see recap_style.mentions_mood_tag_leak."""
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict):
        return flags
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            line = s["line"]
            if line and mentions_mood_tag_leak(line):
                flags.append(_flag(
                    "mood_tag_leak", ERROR,
                    f"narration opens with a bare mood/tone word verbatim: "
                    f"{line[:80]!r} — a mood tag is ALWAYS bracketed "
                    "([dramatic]) and added by the pipeline, never typed "
                    "into the story text; drop the label and start the "
                    "real sentence",
                    scene=str((s["span"] or [""])[0]),
                    segment_id=seg))
    return flags


# --- impact_mismatch (eyes wave) ---------------------------------------------
# Impact-class lexicon — a DATA constant, deliberately crude: full inflected
# word FORMS (not bare stems), matched at BOTH a leading and trailing word
# boundary, case-insensitive. It catches "peaceful vibes over a stab panel",
# not poetry. A leading-boundary-ONLY match (the prior version) let "stab"
# swallow "stable" and "cut" swallow "cutlery" — enumerating real inflections
# with \b on both ends fixes that without losing the "catch every form of the
# same verb" intent. Loose multi-word phrases that collide with mundane
# senses ("ran through the market" = ran on foot, not a blade) are NOT
# included — "run/runs/ran through" was removed for exactly that reason.
# Over-matching can only SUPPRESS a flag, never create one — the safe
# direction for a blocking gate. The trigger side is the model's OWN
# strikes_or_weapons=='in_use' read (it sees the art and reads the SFX in one
# pass); the old CV impact-lettering stamp drove false positives on ambient
# red glyphs (throne-boom == stab) and was retired from this gate 2026-07-23.
#
# The lexicon LIVES in span_align (the narration<->span affinity authority
# also needs it, and prep_qa imports span_align — this direction avoids the
# circular import). Re-exported here so every existing consumer/test keeps
# its prep_qa.has_impact_lexeme / pq._IMPACT_LEXEMES spelling.


# Panel kinds exempt from the impact trigger set (belt-and-suspenders): the
# model classifies panel_kind and strikes_or_weapons in the SAME pass, so it
# rarely calls a system/chrome/caption panel an 'in_use' strike — but if it
# ever does, a stat card / UI banner must never reach the blocking gate.
_IMPACT_EXEMPT_KINDS = frozenset({"system", "chrome", "caption"})


def impact_mismatch_flags(beats_obj: Any, understood_obj: Any
                          ) -> List[Dict[str, Any]]:
    """Narration-vs-art gate (ERROR, heal-then-block): a narrated segment
    whose span contains a panel the UNDERSTANDING marks as a strike in
    progress (strikes_or_weapons == 'in_use') must carry at least one
    impact-class lexeme. A stab panel narrated as a peaceful stroll is exactly
    the mismatch class the grounding judge scores too softly (WARN) to gate on.

    Trigger is gemma's OWN semantic read, not the CV impact-lettering detector
    (2026-07-23): the detector fires on any big red painted glyph and cannot
    tell a throne-boom (두둥) from a stab (푹), so it flooded every action
    chapter with false positives — while the model, which sees the art AND
    reads the SFX, already had the right answer in the same record (nano ch6:
    all 5 detector-flagged panels were strikes_or_weapons='none'). 'visible'
    (a weapon merely drawn, no blow) does NOT trigger; only a blow being
    delivered does. Panels whose understood panel_kind is system/chrome/
    caption stay exempt (belt-and-suspenders: the model rarely calls a stat
    card a strike). Healable: the heal loop re-narrates the group. Same
    _base_scene normalization as span_cover_flags so render-split halves trace
    back to the understood panel. Silent on legacy understanding (no
    strikes_or_weapons field)."""
    flags: List[Dict[str, Any]] = []
    impact_files = {
        _base_scene(os.path.basename(str(p.get("scene_file") or "")))
        for p in ((understood_obj or {}).get("panels") or [])
        if isinstance(p, dict)
        and str(p.get("strikes_or_weapons") or "").strip().lower() == "in_use"
        and str(p.get("panel_kind") or "").strip().lower()
        not in _IMPACT_EXEMPT_KINDS}
    if not impact_files or not isinstance(beats_obj, dict):
        return flags
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        segs = beat_segments(b)
        # A strike is ONE event even when the artist draws it across several
        # consecutive panels: scope the check to the maximal RUN of adjacent
        # strike panels, and let ANY segment covering that run voice it (a
        # sibling line carrying the blow is not a miss). A run of one behaves
        # exactly as before; non-adjacent strikes stay separate events.
        files = [str(f) for f in (b.get("scene_files") or [])]
        runs: List[List[str]] = []
        for f in files:
            if _base_scene(f) in impact_files:
                if runs and files.index(f) == files.index(runs[-1][-1]) + 1:
                    runs[-1].append(f)
                else:
                    runs.append([f])
        for run in runs:
            run_set = {_base_scene(f) for f in run}
            covering = [s for s in segs
                        if run_set & {_base_scene(f) for f in s["span"]}]
            if not covering:
                continue
            if any(has_impact_lexeme(s["line"]) for s in covering if s["line"]):
                continue                      # the blow IS voiced on this run
            first = next((s for s in covering if s["line"]), None)
            if first is None:
                continue
            flags.append(_flag(
                "impact_mismatch", ERROR,
                f"the panel understanding shows a strike in progress on "
                f"{run[0]} but the narration has no impact wording: "
                f"{first['line'][:80]!r} — re-narrate the strike/stab/blow "
                "explicitly",
                scene=str(run[0]), segment_id=seg))
    return flags


# Span word-budget arithmetic — tiny duplicate of narration_punchup.
# span_budget_ok / gemini_narrative_pass.validate_segments (same rationale as
# punchup's copy: importing either would pull their model deps into QA).
_BUDGET_WPM = 135.0                  # == gemini_narrative_pass.WPM
_BUDGET_MAX_SEC_PER_PANEL = 15.0     # == _SEG_MAX_SEC_PER_PANEL


def line_overlong_flags(beats_obj: Any) -> List[Dict[str, Any]]:
    """Deterministic length gate (ERROR, healable, NOT worker-blocking): a
    segment line past its span's word budget (N*15s at 135wpm ≈ 34 words per
    panel). The writer validator enforces this at authoring time, but its
    fallback path ships the model's lines VERBATIM when a re-ask also fails
    (gemini_narrative_pass segments fallback) — this is the choke-point net
    for any escaped line: a 55-word single-panel line = a 21s hold = a
    triple ken split on the dashboard (2026-07-16 nano g0011). Heal
    converges here — length is fully in the writer's control."""
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict):
        return flags
    for b in beats_obj.get("beats") or []:
        seg_id = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            n = max(1, len(s["span"]))
            words = len(str(s["line"] or "").split())
            sec = words / (_BUDGET_WPM / 60.0)
            cap = n * _BUDGET_MAX_SEC_PER_PANEL
            if sec > cap:
                max_words = int(cap * _BUDGET_WPM / 60.0)
                flags.append(_flag(
                    "line_overlong", ERROR,
                    f"segment line is {words} words (~{sec:.0f}s of voice) "
                    f"over a {n}-panel span (cap ~{cap:.0f}s / {max_words} "
                    f"words) — re-narrate tighter: {str(s['line'])[:80]!r}",
                    scene=str((s["span"] or [""])[0]), segment_id=seg_id))
    return flags


def narration_offset_flags(beats_obj: Any, understood_obj: Any
                           ) -> List[Dict[str, Any]]:
    """ONE-PANEL OFFSET tripwire (ERROR, heal-target, deliberately NOT in the
    worker blocking set — the first production run measures its precision):
    the dominant defect class of the 2026-07-06 Nano ch1 human vision review
    (~10/27 findings) was a line leading/lagging its span by one panel in an
    action run (the impact line voiced over the pre-impact panel; the
    "eyes widen" line one panel after the eyes). Fires when the segment
    line's affinity to a NEIGHBOR window (same size, shifted +-1 panel)
    beats its own span's by >= SPAN_ALIGN_MARGIN — the SAME shared scoring
    (span_align.window_affinities) the splitter's span_align_pass shifts on,
    so QA and the splitter can never disagree. Silent without understanding
    records (the affinity has nothing to score against)."""
    flags: List[Dict[str, Any]] = []
    u_by_file: Dict[str, Dict[str, Any]] = {}
    for p in ((understood_obj or {}).get("panels") or []):
        if isinstance(p, dict) and p.get("scene_file"):
            u_by_file[os.path.basename(str(p["scene_file"]))] = p
    if not u_by_file or not isinstance(beats_obj, dict):
        return flags
    for b in beats_obj.get("beats") or []:
        seg_tag = f"g{int(b.get('group_id') or 0):04d}"
        segs = beat_segments(b)
        files = [f for s in segs for f in s["span"]]
        if not files:
            continue
        # split render halves (p0098_a.jpg) trace back to the understood parent
        u_local = {f: (u_by_file.get(f) or u_by_file.get(_base_scene(f)) or {})
                   for f in files}
        kinds = {f: str(u_local[f].get("panel_kind") or "") for f in files}
        for i, s in enumerate(segs):
            if not s["line"]:
                continue
            direction = offset_shift_candidate(segs, i, files, kinds, u_local)
            if direction is None:
                continue
            own_p, minus_p, plus_p = window_score_pairs(segs, i, files, kinds,
                                                        u_local)
            best_p = plus_p if direction == "+1" else minus_p
            flags.append(_flag(
                "narration_offset", ERROR,
                f"line fits the {direction}-shifted panel window better "
                f"than its own span (affinity {own_p[1]:.2f} vs "
                f"{best_p[1]:.2f}, line-overlap {own_p[0]:.2f} vs "
                f"{best_p[0]:.2f}): {s['line'][:80]!r} — one-panel lead/lag; "
                "re-narrate this group so each line lands on the panels it "
                "describes", scene=str((s["span"] or [""])[0]),
                segment_id=seg_tag))
    return flags


# Reporting speech rather than acting: the speaker may be off-panel while the
# panel shows the listener reacting. Generic English verbs, no series content.
_SPEECH_ACT_RE = re.compile(
    r"\b(say|says|said|tell|tells|told|ask|asks|asked|shout|shouts|shouted|"
    r"yell|yells|yelled|sneer|sneers|sneered|mock|mocks|mocked|taunt|taunts|"
    r"taunted|jeer|jeers|jeered|warn|warns|warned|order|orders|ordered|"
    r"call|calls|called|whisper|whispers|whispered|explain|explains|"
    r"explained|declare|declares|declared|announce|announces|announced|"
    r"demand|demands|demanded|promise|promises|promised|laugh|laughs|"
    r"laughed|scoff|scoffs|scoffed|snarl|snarls|snarled|hiss|hisses|hissed|"
    r"spits|spat|mutter|mutters|muttered|reply|replies|replied)\b",
    re.IGNORECASE)


def _actor_noun_on_page(noun: str, span, vitems_by_base) -> bool:
    """True when the line's actor-noun is PRINTED on one of the span's panels
    — a chat handle, a nameplate, a signature, a caption byline.

    The actor gate catches a narrator who names someone the panel does not
    draw. A name the panel SPELLS OUT is a third case: not invented, not
    misattributed, just not a drawn body (ORV Ep1 p000087 — the protagonist
    reads his phone and the narration names the commenter 'TLS123' off the
    screen; healing it would force the WRONG actor onto the line). Matches a
    whole word or a word whose remainder is non-alphabetic ('tls' in
    'tls123') so a stylized handle still resolves, while 'ana' never matches
    'analysis'."""
    n = _norm_narr(str(noun or "")).strip()
    if len(n) < 3:
        return False
    pat = re.compile(r"\b" + re.escape(n) + r"(?![a-z])")
    for fn in span:
        item = vitems_by_base.get(_base_scene(os.path.basename(str(fn)))) or {}
        if pat.search(_norm_narr(str(item.get("ocr_clean") or ""))):
            return True
    return False


def actor_mismatch_flags(beats_obj: Any, understood_obj: Any,
                         cast_obj: Any,
                         vitems: Optional[Dict[str, Any]] = None,
                         ) -> List[Dict[str, Any]]:
    """CAST-GROUNDED actor gate (ERROR, heal-target, deliberately NOT in the
    worker blocking set — the first production run measures its precision):
    the round-2 vision review's dominant class (~6 findings) was identity
    misattribution — "the assassin draws his steel" over Prince Cheon's
    counter-draw (g0008_p06), the dying prince's eye narrated as "an
    assassin's eye" (g0019_p00), a departed assassin given the descendant's
    inner thoughts (g0020_p01).

    Fires when a line's SUBJECT-position actor-noun (noun map derived from
    manifest.cast.json — names/aliases/ids, no hardcoded series words) maps
    to cast members that are DISJOINT from the span's resolved figures
    (tools/cast_identity.py — the SAME deterministic resolution the writer
    payload's `figures` lines use, so QA and the writer can never disagree).
    Precision posture: subject-position-only nouns (late mentions are
    objects/off-panel references), spans with zero resolved figures are
    skipped (no ground truth), ties resolve to unknown upstream. Healable:
    the regenerated group's payload carries the figures lines the original
    roll lacked. Silent without cast or understanding."""
    from cast_identity import (actor_noun_map, group_member_names,
                               resolve_figures_by_file, shares_faction,
                               subject_actor_nouns, subject_person_count)
    flags: List[Dict[str, Any]] = []
    noun_map = actor_noun_map(cast_obj)
    group_names = group_member_names(cast_obj)
    figures = resolve_figures_by_file(understood_obj, cast_obj)
    if not noun_map or not figures or not isinstance(beats_obj, dict):
        return flags
    fig_by_base = {_base_scene(os.path.basename(f)): v
                   for f, v in figures.items()}
    v_by_base = {_base_scene(os.path.basename(str(k))): v
                 for k, v in (vitems or {}).items()}
    u_by_sf = {_base_scene(os.path.basename(str(p.get("scene_file") or ""))): p
               for p in ((understood_obj or {}).get("panels") or [])
               if isinstance(p, dict) and p.get("scene_file")}
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            line = s["line"]
            if not line:
                continue
            span_names = {f["name"]
                          for fn in s["span"]
                          for f in fig_by_base.get(_base_scene(fn), [])
                          if f.get("name") and f["name"] != "unknown"}
            if not span_names:
                continue
            # A panel whose text the narration is REPORTING (a taunt, an
            # order, a shout) can name a speaker who is not drawn in it —
            # a reaction shot of the listener is not a mismatch.
            span_has_dialogue = any(
                str((u_by_sf.get(_base_scene(fn)) or {}).get("dialogue")
                    or "").strip() for fn in s["span"])
            for noun, members in subject_actor_nouns(line, noun_map):
                if members & span_names:
                    continue
                # A GROUP handle over a panel drawn as a CROWD has no
                # per-individual ground truth to contradict: an appearance
                # oracle built for individuals never resolves a collective
                # subject string onto the group member (ch6 g0003/g0013).
                if members and group_names and members <= group_names and any(
                        subject_person_count(str(sub)) > 1
                        for fn in s["span"]
                        for sub in ((u_by_sf.get(_base_scene(fn)) or {})
                                    .get("subjects") or [])):
                    continue
                # SAME FACTION: the oracle resolves an ambiguous look-alike to
                # the least specific member, so 'leader' over a generic hooded
                # panel is not evidence of a wrong actor — we simply cannot
                # tell them apart. Shared authority with cast_identity.
                if shares_faction(members, span_names, cast_obj):
                    continue
                if span_has_dialogue and _SPEECH_ACT_RE.search(line):
                    continue
                if _actor_noun_on_page(noun, s["span"], v_by_base):
                    continue
                flags.append(_flag(
                    "actor_mismatch", ERROR,
                    f"line names '{noun}' as the actor but the span's "
                    f"resolved figures are {sorted(span_names)}: "
                    f"{line[:80]!r} — re-narrate naming the actor from the "
                    "panel's actual figures",
                    scene=str((s["span"] or [""])[0]), segment_id=seg))
    return flags


def ledger_contradiction_flags(beats_obj: Any, ledger_obj: Any,
                               cast_obj: Any) -> List[Dict[str, Any]]:
    """STORY-STATE gate (2026-07-20 wave) — narration vs the chapter's fact
    record (manifest.ledger.json, dialogue-arbitrated):
      dead_actor (ERROR, heal-THEN-block in the worker): a line's SUBJECT-
        position actor-noun maps ONLY to entities the ledger says are dead
        by this beat — a dead character cannot act (the nano ch1 'leader
        finishes the job' class);
      role_stale (ERROR, heal-only until precision is measured): a line uses
        a banned unique-role handle ('the leader' after the leader died — a
        surviving underling never inherits the title).
    Details carry the ledger's evidence quote so the heal note states the
    FACT, not just the violation. Silent without a ledger (old chapters)."""
    from cast_identity import actor_noun_map, subject_actor_nouns
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict) or not isinstance(ledger_obj, dict):
        return flags
    beat_facts = ledger_obj.get("beat_facts") or {}
    if not beat_facts:
        return flags
    noun_map = actor_noun_map(cast_obj)
    death_ev = {ev.get("subject"): ev
                for ev in (ledger_obj.get("events") or [])
                if ev.get("type") == "death"}

    def _quote(members) -> str:
        for m in members:
            ev = death_ev.get(m)
            if ev:
                q = str(ev.get("evidence_quote") or "").strip()
                sf = str(ev.get("scene_file") or "")
                return f" (killed at {sf}: \"{q}\")" if q else f" (killed at {sf})"
        return ""

    for b in beats_obj.get("beats") or []:
        gid = f"g{int(b.get('group_id') or 0):04d}"
        facts = beat_facts.get(gid) or {}
        dead = set(facts.get("dead_by_now") or [])
        banned = [str(h) for h in (facts.get("banned_handles") or []) if h]
        if not dead and not banned:
            continue
        for s in beat_segments(b):
            line = s["line"]
            if not line:
                continue
            fired_nouns: set = set()
            for noun, members in subject_actor_nouns(line, noun_map):
                if members and members <= dead:
                    fired_nouns.add(noun)
                    flags.append(_flag(
                        "dead_actor", ERROR,
                        f"line has '{noun}' acting but "
                        f"{sorted(members)} are dead by this beat"
                        f"{_quote(members)}: {line[:80]!r} — a dead "
                        "character cannot act; re-narrate from the living "
                        "actors the chapter record names",
                        scene=str((s["span"] or [""])[0]), segment_id=gid))
            for h in banned:
                if h.rsplit(" ", 1)[-1].lower() in fired_nouns:
                    continue                  # already flagged as dead_actor
                if re.search(r"\b" + re.escape(h) + r"\b", line,
                             re.IGNORECASE):
                    flags.append(_flag(
                        "role_stale", ERROR,
                        f"line says {h!r} but that role holder is dead by "
                        f"this beat{_quote(dead)}: {line[:80]!r} — nobody "
                        "inherits the title; name who is actually shown",
                        scene=str((s["span"] or [""])[0]), segment_id=gid))
    return flags


def cold_open_flags(beats_obj: Any) -> List[Dict[str, Any]]:
    """TRANSITION net (WARN, heal-target under semantic-heal): a beat's FIRST
    line re-establishes the scene coldly ('The scene shows…', 'In a dark
    ravine, a figure…') even though the narrator just spoke — the audible
    seam the 2026-07-16 transitions wave kills. Pattern authority lives in
    recap_style.is_cold_opener (shared with narration_punchup's bridge
    preservation). The detail carries the PREVIOUS line so the heal note is
    the exact (prev, this) bridge rewrite. First beat of the chapter is
    exempt — there is nothing to bridge from."""
    from recap_style import is_cold_opener
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict):
        return flags
    prev_line = ""
    for b in beats_obj.get("beats") or []:
        segs = beat_segments(b)
        if not segs:
            continue
        seg_id = f"g{int(b.get('group_id') or 0):04d}"
        first = str(segs[0].get("line") or "").strip()
        if prev_line and first and is_cold_opener(first):
            flags.append(_flag(
                "cold_open", WARN,
                f"beat opens cold ({first[:60]!r}) instead of bridging from "
                f"the narrator's previous line ({prev_line[-90:]!r})",
                scene=str((segs[0].get("span") or [""])[0]),
                segment_id=seg_id))
        last = str(segs[-1].get("line") or "").strip()
        prev_line = last or prev_line
    return flags


def actor_count_flags(beats_obj: Any, understood_obj: Any,
                      cast_obj: Any) -> List[Dict[str, Any]]:
    """PLURALITY gate (ERROR, heal-target, NOT worker-blocking — precision is
    measured first, same posture as actor_mismatch): a line that PLURALIZES a
    subject-position actor-noun ("our guy and his assassins go tumbling")
    while every panel in its span shows at most ONE person. Capacity is the
    max person count across the span's resolved figures (unknowns included —
    each is a person-ish subject); panels the analyst marked `uncertain`
    (pu_v4) contribute no ground truth and are skipped. Shares its pattern
    authority with the writer's identity gate (cast_identity)."""
    from cast_identity import (actor_noun_map, group_member_names,
                               subject_actor_nouns_ex, subject_person_count)
    flags: List[Dict[str, Any]] = []
    noun_map = actor_noun_map(cast_obj)
    group_names = group_member_names(cast_obj)
    if not noun_map or not isinstance(beats_obj, dict):
        return flags
    # capacity = person-ish SUBJECT count (resolve_figures dedupes same-cast
    # figures by name, so it under-counts a genuine two-assassin panel)
    cap_by_base: Dict[str, int] = {}
    unc_by_base: set = set()
    for p in ((understood_obj or {}).get("panels") or []):
        base = _base_scene(os.path.basename(str(p.get("scene_file") or "")))
        if not base:
            continue
        if p.get("uncertain"):
            unc_by_base.add(base)
        # a drawn CROWD is written as ONE subject string ("a group of people
        # with dark hair") — counting strings reported "every panel shows ONE
        # figure" over twenty drawn students (ch6 g0003)
        cap_by_base[base] = sum(
            subject_person_count(str(s)) for s in (p.get("subjects") or []))
    if not cap_by_base:
        return flags
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            line = s["line"]
            if not line:
                continue
            capacities = []
            for fn in s["span"]:
                base = _base_scene(os.path.basename(fn))
                if base in unc_by_base or base not in cap_by_base:
                    continue
                capacities.append(cap_by_base[base])
            if not capacities or max(capacities) != 1:
                continue          # multi-person span (or no ground truth)
            for noun, _members, plural in subject_actor_nouns_ex(line,
                                                                 noun_map):
                if not plural:
                    continue
                if _members and group_names and _members <= group_names:
                    continue      # a group handle is plural BY IDENTITY
                flags.append(_flag(
                    "actor_count_mismatch", ERROR,
                    f"line pluralizes '{noun}' but every panel in the span "
                    f"shows ONE figure: {line[:80]!r} — re-narrate with the "
                    "single actor shown, never invent companions",
                    scene=str((s["span"] or [""])[0]), segment_id=seg))
    return flags


def phrase_echo_flags(beats_obj: Any, *, window: int = 8,
                      min_words: int = 6) -> List[Dict[str, Any]]:
    """PHRASE ECHO (WARN, heal-target): two narrated segments within *window*
    of each other share a >= *min_words* verbatim word run (case/punct-
    normalized) — the round-2 g0020_p01/g0024_p12 near-verbatim repeated
    thought. Cheap + deterministic (the same longest_common_run authority the
    ocr_echo check uses). One flag per offending later segment; consecutive-
    duplicate LINES are a different class (dedupe_consecutive_panel_lines)."""
    flags: List[Dict[str, Any]] = []
    if not isinstance(beats_obj, dict):
        return flags
    rows: List[Tuple[str, str]] = []          # (segment_tag, line)
    for b in beats_obj.get("beats") or []:
        seg = f"g{int(b.get('group_id') or 0):04d}"
        for s in beat_segments(b):
            if s["line"]:
                rows.append((seg, s["line"]))
    flagged: set = set()
    for j in range(1, len(rows)):
        if j in flagged:
            continue
        for i in range(max(0, j - window), j):
            run = longest_common_run(rows[i][1], rows[j][1],
                                     min_words=min_words)
            if run:
                flags.append(_flag(
                    "phrase_echo", WARN,
                    f"repeats {rows[i][0]}'s phrase nearly verbatim "
                    f"({run[:60]!r}): {rows[j][1][:80]!r} — re-narrate "
                    "with fresh wording",
                    segment_id=rows[j][0]))
                flagged.add(j)
                break
    return flags


def span_cover_flags(plan: Dict[str, Any], beats_obj: Dict[str, Any],
                     vitems: Optional[Dict[str, Dict[str, Any]]] = None
                     ) -> List[Dict[str, Any]]:
    """Adaptive-flow COVER check — the replacement for every 1:1
    panel-count-shaped invariant: the beats' segment spans must PARTITION the
    shown story panels. A shown panel in NO span (`panel_uncovered`) has no
    narration carrying it on screen — the class the old per-panel count assert
    caught; a panel claimed by 2+ spans (`panel_double_covered`) would be paced
    under two different clips. Both ERROR (blocks autopilot spotless-advance).

    Judged on SHOWN panels only: a span panel visually dropped upstream is by
    design (spec 3.5 — drops shrink the cut list, narration untouched).
    Exempt: branding + held cuts (display machinery, not narration coverage)
    and protected system/doc cards (scene_dims sys/doc or stamped
    panel_kind=='system') — inject_missing_protected shows those narration-less
    BY DESIGN. Silent when no beat carries segments/panel_narration at all
    (pre-per-panel manifests have no spans to assert against)."""
    cover: Dict[str, List[str]] = {}
    any_segments = False
    for b in (beats_obj or {}).get("beats") or []:
        segs = beat_segments(b)
        if not segs:
            continue
        any_segments = True
        gid_tag = f"g{int(b.get('group_id') or 0):04d}"
        for s in segs:
            for f in s["span"]:
                cover.setdefault(_base_scene(f), []).append(gid_tag)
    if not any_segments:
        return []

    dims = (plan or {}).get("scene_dims") or {}
    vit = vitems or {}

    def _protected(fname: str, base: str) -> bool:
        for name in (fname, base):
            d = dims.get(name) or {}
            if d.get("sys") or d.get("doc"):
                return True
        kind = ((vit.get(base) or vit.get(fname)) or {}).get("panel_kind")
        return str(kind or "").lower() == "system"

    flags: List[Dict[str, Any]] = []
    seen: set = set()
    for item in (plan or {}).get("timeline") or []:
        if item.get("branding"):
            continue
        seg = str(item.get("segment_id") or "")
        for c in item.get("cuts") or []:
            if c.get("held"):
                continue
            for f in (c.get("file"), c.get("file2")):
                if not f:
                    continue
                f = str(f)
                base = _base_scene(f)
                if base in seen:
                    continue
                seen.add(base)
                owners = cover.get(base) or []
                if len(owners) > 1:
                    flags.append(_flag(
                        "panel_double_covered", ERROR,
                        f"panel is claimed by {len(owners)} narration segment "
                        f"spans ({', '.join(owners)}) — spans must partition "
                        "the panels; re-run the beated stage",
                        scene=base, segment_id=owners[0]))
                elif not owners and not _protected(f, base):
                    flags.append(_flag(
                        "panel_uncovered", ERROR,
                        "shown panel belongs to NO narration segment span — "
                        "no voiced line carries it on screen; re-run the "
                        "beated/scripted stages",
                        scene=base, segment_id=seg))
    return flags


def held_repeat_flags(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A single panel shown in >=3 consecutive cuts (a frozen/looping repeat with
    a restarting pan — the eye-panel-3x bug). >=4 = panels lost upstream (block);
    3 = editor coverage (warn)."""
    flags: List[Dict[str, Any]] = []
    seq: List[Tuple[str, str]] = []
    for it in (plan or {}).get("timeline") or []:
        if it.get("branding"):
            continue
        for c in it.get("cuts") or []:
            if c.get("ken_variety"):
                # V1 sub-cuts: deliberate DIFFERENT ken regions over one
                # panel (split_long_hold_cuts) — not a frozen/looping repeat;
                # the long_hold static ceiling governs that display instead
                continue
            f = str(c.get("file") or "")
            if f:
                seq.append((f, str(it.get("segment_id") or "")))
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1][0] == seq[i][0]:
            j += 1
        run = j - i + 1
        if run >= 3:
            # WARN for a normal hold (editor covering narration over one image);
            # ERROR only when excessive (>=5) which means panels were lost upstream.
            flags.append(_flag(
                "held_repeat", ERROR if run >= 5 else WARN,
                f"panel {seq[i][0]} shown in {run} consecutive cuts — must be ONE "
                "static hold (no restarting pan); >=5 means panels lost upstream",
                scene=seq[i][0], segment_id=seq[i][1]))
        i = j + 1
    return flags


def long_hold_flags(plan: Dict[str, Any], beats_obj: Dict[str, Any], *,
                    max_hold_sec: float = 10.0,
                    static_ceiling_factor: float = 1.5,
                    is_exempt=None) -> List[Dict[str, Any]]:
    """One FILE on screen continuously for > *max_hold_sec*
    ([render].max_same_image_hold_sec) while STANDING IN for art it does not
    own — the p000090 eye held ~24s because p000095 was canonicalized away and
    the narration for BOTH segments played over one image.

    "Stand-in" is evaluated PER CUT here — each cut's file is a stand-in when
    it is absent from its OWN segment's beat scene_files, independent of any
    other cut in that same segment. This is DELIBERATELY finer-grained than
    `panel_substituted` (story_flags), which is PER SEGMENT: it unions every
    cut's file across the item and flags only when NONE of them intersect the
    beat's intended panels. The per-segment view answers "did this beat's art
    land on screen at all" (one genuine cut is enough to say yes); it is
    structurally blind to a single foreign cut riding alongside a genuine one
    in a multi-cut segment — exactly what a cross-group fold from
    enforce_shown_twin_invariant/drop_cross_segment_near_identical_cuts can
    produce (a later beat's cut folds into an earlier group's near-identical
    art). That foreign cut hogging screen time past the cap is a real defect
    even though panel_substituted correctly stays quiet (the segment's own
    panel is ALSO present). Do not collapse this onto panel_substituted's
    per-segment result — that would blind long_hold to exactly this case.

    A long single-image span on a panel that GENUINELY owns that narration
    (its file is among the beat's scene_files, no substitution) is
    content-driven pacing by design — legal at any length WHEN the display
    varies. UNCONDITIONAL STATIC CEILING (V1 tripwire, 2026-07 review): one
    file continuously STATIC — a single cut, or a same-file run with NO ken
    variation (identical motion dicts) — past *static_ceiling_factor* x the
    cap fires ERROR regardless of stand-in/ownership (the 22.8s own-panel eye
    was unwatchable). render_prep's ken-variety split (split_long_hold_cuts)
    breaks any such display into varied sub-cuts, so this should never fire
    in practice. *is_exempt*(file) excuses panels whose renderer branch is
    never static (wide cover-drift / tall scroll — Cut.tsx animates those per
    cut) and text panels that need stillness (doc, stamped system cards).
    BLOCKING: heal re-writes narration, it cannot restore a swapped-away
    panel or re-cut a plan — the chapter must go back through prepare."""
    bfiles: Dict[int, set] = {}
    for b in (beats_obj or {}).get("beats") or []:
        try:
            gid = int(b.get("group_id"))
        except (TypeError, ValueError):
            continue
        bfiles[gid] = {str(f) for f in (b.get("scene_files") or [])}

    # flatten to (file, dur, segment, standin, motion_sig) per shown cut;
    # None breaks runs. motion_sig detects "no ken variation" for the ceiling.
    exempt = is_exempt or (lambda f: False)
    seq: List[Optional[Tuple[str, float, str, bool, str]]] = []
    for it in (plan or {}).get("timeline") or []:
        if it.get("branding"):
            seq.append(None)
            continue
        seg = str(it.get("segment_id") or "")
        m = _SEG_GROUP_RE.match(seg)
        intended = bfiles.get(int(m.group(1))) if m else None
        for c in it.get("cuts") or []:
            f = str(c.get("file") or "")
            if not f or c.get("file2") or c.get("layout"):
                seq.append(None)                 # split layouts break the run
                continue
            # per-CUT stand-in test (see docstring): a sibling cut in this
            # same segment being genuine does NOT clear this one.
            standin = bool(intended) and _base_scene(f) not in intended
            msig = json.dumps(c.get("motion"), sort_keys=True, default=str)
            seq.append((f, float(c.get("dur") or 0.0), seg, standin, msig))

    ceiling = static_ceiling_factor * max_hold_sec
    flags: List[Dict[str, Any]] = []
    i, n = 0, len(seq)
    while i < n:
        if seq[i] is None:
            i += 1
            continue
        j = i
        while (j + 1 < n and seq[j + 1] is not None
               and seq[j + 1][0] == seq[i][0]):
            j += 1
        run = [r for r in seq[i:j + 1] if r is not None]
        total = sum(r[1] for r in run)
        if total > max_hold_sec and any(r[3] for r in run):
            first_sub = next(r for r in run if r[3])
            flags.append(_flag(
                "long_hold", ERROR,
                f"panel {run[0][0]} held on screen {total:.1f}s (> "
                f"{max_hold_sec:.1f}s cap) while standing in for "
                f"{first_sub[2]}'s intended art — a swapped/held stand-in, "
                "not content-driven pacing",
                scene=run[0][0], segment_id=run[0][2]))
        elif (total > ceiling
              and (len(run) == 1 or len({r[4] for r in run}) == 1)
              and not exempt(run[0][0])):
            # unconditional STATIC ceiling: single cut / identical motions =
            # no ken variation; ownership does not excuse unwatchability
            flags.append(_flag(
                "long_hold", ERROR,
                f"panel {run[0][0]} shown {total:.1f}s continuously STATIC "
                f"(single cut, no ken variation) — past the unconditional "
                f"{ceiling:.1f}s ceiling ({static_ceiling_factor:.1f}x cap); "
                "render_prep's ken-variety split should have varied it "
                "(own-panel ownership does not exempt watchability)",
                scene=run[0][0], segment_id=run[0][2]))
        i = j + 1
    return flags


def montage_flags(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cross-segment visual degeneracy — the class the per-segment checks
    (and the per-segment LLM judge) cannot see: one panel carrying many
    segments, or a long stretch alternating between a tiny set of images.
    Regression source: Episode 2 showed 6 segments cycling 2 mangled crops
    after the phone panels were sliced and dropped upstream."""
    flags: List[Dict[str, Any]] = []
    dims = (plan or {}).get("scene_dims") or {}

    def _protected(f: str) -> bool:
        d = dims.get(f) or {}
        return bool(d.get("sys") or d.get("doc"))

    segs: List[Any] = []
    for it in (plan or {}).get("timeline") or []:
        if it.get("branding"):
            continue
        files = [str(c.get("file") or "") for c in it.get("cuts") or []
                 if c.get("file") and not c.get("held")
                 and not _protected(str(c.get("file")))]
        segs.append((str(it.get("segment_id") or ""), files))
    by_file: Dict[str, List[str]] = {}
    for sid, files in segs:
        for f in set(files):
            by_file.setdefault(f, []).append(sid)
    for f, sids in sorted(by_file.items()):
        if len(sids) >= 3:
            flags.append(_flag(
                "visual_loop", ERROR,
                f"same panel carries {len(sids)} segments "
                f"({', '.join(sids[:4])}…) — panels were lost upstream",
                scene=f))
    for i in range(len(segs) - 3):
        window = segs[i:i + 4]
        fresh = [files for _, files in window if files]
        if len(fresh) < 3:
            continue        # held stretches are intentional coverage
        uniq = {f for files in fresh for f in files}
        if uniq and len(uniq) <= 2:
            flags.append(_flag(
                "montage_degenerate", ERROR,
                f"segments {window[0][0]}…{window[-1][0]} draw on only "
                f"{len(uniq)} unique panels — the montage is starved; "
                "check dropped/missed panels upstream",
                segment_id=window[0][0]))
            break
    return flags


_SEM_PROMPT = """You are a QA judge for a manhwa recap video. The attached \
image is the panel shown on screen while the narrator reads this line:

NARRATION: {text}

Does the narration plausibly belong with this panel (same scene, characters, \
or on-screen content)? Narration may add story context, but it must not \
describe a clearly different panel.
Reply ONLY JSON: {{"match": true/false, "confidence": 0-100, \
"reason": "<short>"}}"""


def semantic_alignment_flags(plan: Dict[str, Any], clean_dir: str, *,
                             model: str = "gemma4:26b",
                             min_confidence: int = 60
                             ) -> List[Dict[str, Any]]:
    """Gemma vision-judge per shown segment: does the narration describe the
    panel? WARN-level by design — a judge flags for human review, it never
    blocks or rewrites prose (closed-loop regen degrades good lines)."""
    try:
        import ollama  # local + free; absent on boxes without the stack
    except ImportError:
        return [_flag("semantic_skipped", INFO,
                      "ollama not importable — semantic judge skipped")]
    from ollama_compat import chat as _ollama_chat

    def _judge(path: str, text: str) -> Optional[Dict[str, Any]]:
        resp = _ollama_chat(
            model=model, think=False,
            messages=[{"role": "user",
                       "content": _SEM_PROMPT.format(text=text[:400]),
                       "images": [path]}],
            # num_ctx: image tokens overflow ollama's default window and it
            # TRUNCATES SILENTLY — the judge then judges a partial context
            # (2026-07-16 audit; MLX ignores this, ollama needs it)
            options={"temperature": 0, "num_predict": 200,
                     "num_ctx": 8192})
        raw = str(resp["message"]["content"] or "")
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}

    flags: List[Dict[str, Any]] = []
    for item in (plan or {}).get("timeline") or []:
        if item.get("branding"):
            continue
        seg = str(item.get("segment_id") or "")
        text = (item.get("tts_text") or "").strip()
        cuts = item.get("cuts") or []
        if not text or not cuts:
            continue
        # the viewer sees the whole MONTAGE, not just the primary panel — the
        # narration belongs to the segment if it fits ANY panel actually shown
        # (every cut's file + its split2 file2). Judging primary-only is the
        # group-blindness bug: a multi_cut beat narrating cut #2 was wrongly
        # flagged against cut #1. Early-exit on the first plausible match keeps
        # single-cut segments at one judge call.
        files: List[str] = []
        for c in cuts:
            if c.get("held"):
                continue        # held = intentional coverage, not a match
            for f in (c.get("file"), c.get("file2")):
                f = str(f or "")
                if f and f not in files and os.path.exists(
                        os.path.join(clean_dir, f)):
                    files.append(f)
        if not files:
            continue
        rejected: List[Tuple[str, int, str]] = []
        matched = False
        for f in files:
            try:
                v = _judge(os.path.join(clean_dir, f), text)
            except Exception as e:                      # noqa: BLE001
                flags.append(_flag("semantic_error", INFO,
                                   f"judge failed on {f}: {e}",
                                   segment_id=seg))
                continue
            conf = int(v.get("confidence") or 0)
            if not (v.get("match") is False and conf >= min_confidence):
                matched = True          # plausible match (or judge unsure)
                break
            rejected.append((f, conf, str(v.get("reason") or "")))
        if not matched and rejected:
            f, _conf, reason = max(rejected, key=lambda r: r[1])
            flags.append(_flag(
                "narration_mismatch", WARN,
                f"judge: {reason[:160]}",
                scene=f, segment_id=seg))
    return flags


_GROUND_PROMPT = """You are a strict QA judge for a manhwa recap. The attached \
images are ALL the panels shown on screen while the narrator reads this line \
(a beat is a short montage of these panels, seen together):

NARRATION: {text}

Judge the narration against THESE panels TAKEN TOGETHER, on two things:
1. GROUNDING — does it INVENT or MIS-NAME something that appears in NONE of the \
panels? (e.g. calling beasts "dogs", inventing a character / crowd / quantity \
that does not appear anywhere). Naming something shown in ANY of the panels is \
fine — the line covers the whole montage, not one panel.
2. QUALITY — is it concrete, not vague filler ("something happens", "things \
change", "a moment passes") and not interface chatter?

Be conservative: flag ONLY a clear invention/mis-naming absent from EVERY panel, \
or genuine filler. If the line is grounded in any panel and not filler, it is ok.
Reply ONLY JSON: {{"ok": true/false, "issue": "<short — what is invented/mis- \
named or weak; empty if ok>"}}"""


def grounding_flags(plan: Dict[str, Any], clean_dir: str, *,
                    model: str = "gemma4:26b",
                    cache_path: Optional[str] = None,
                    uncertain_files: Optional[set] = None
                    ) -> List[Dict[str, Any]]:
    """Stronger 'eyes' than semantic_alignment_flags: per beat, judge whether the
    narration INVENTS or MIS-NAMES anything absent from every panel the beat
    shows, or is weak filler. Judged against the WHOLE montage (all the beat's
    panels at once) — not the primary panel — so a line grounded in a non-primary
    panel isn't falsely flagged. Emits a HEALABLE `grounding_weak` WARN; the
    auto-heal loop re-narrates it and the strictly-better safeguard reverts any
    non-improvement. Runs under --semantic (and --semantic-heal). `cache_path`
    memoizes verdicts by (model, narration, shown panels) so the voiceover-time
    re-scan reuses prepare's judgments instead of re-paying the 26B."""
    try:
        import ollama  # noqa: F401  (local + free; absent on bare boxes)
    except ImportError:
        return [_flag("grounding_skipped", INFO,
                      "ollama not importable — grounding judge skipped")]
    from ollama_compat import chat as _ollama_chat
    unc = {parent_scene(f) for f in (uncertain_files or set())} | set(
        uncertain_files or set())

    def _judge(paths: List[str], text: str, note: str = "") -> Dict[str, Any]:
        resp = _ollama_chat(
            model=model, think=False,
            messages=[{"role": "user",
                       "content": _GROUND_PROMPT.format(text=text[:400]) + note,
                       "images": paths}],
            # num_ctx: up to 6 montage images blow past ollama's default
            # window and it TRUNCATES SILENTLY — the judge was judging a
            # partial montage (2026-07-16 audit; MLX ignores this)
            options={"temperature": 0, "num_predict": 200,
                     "num_ctx": 8192})
        raw = str(resp["message"]["content"] or "")
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}

    # 1. collect the beats to judge, in timeline order (output stays deterministic)
    work: List[Tuple[str, str, List[str], str]] = []  # (segment_id, narration, files, note)
    for item in (plan or {}).get("timeline") or []:
        if item.get("branding"):
            continue
        seg = str(item.get("segment_id") or "")
        text = (item.get("tts_text") or "").strip()
        cuts = item.get("cuts") or []
        if not text or not cuts:
            continue
        # judge against EVERY panel the beat actually shows (the montage)
        files: List[str] = []
        for c in cuts:
            if c.get("held"):
                continue
            for f in (c.get("file"), c.get("file2")):
                f = str(f or "")
                if f and f not in files and os.path.exists(
                        os.path.join(clean_dir, f)):
                    files.append(f)
        if not files:
            continue
        # pu_v4: the analyst itself hedged on these panels — tell the judge so
        # a deliberately-vague line is CORRECT grounding, not weakness.
        hedged = sorted({f for f in files if f in unc or parent_scene(f) in unc})
        note = ""
        if hedged:
            note = ("\nNOTE: the visual analyst marked panel(s) "
                    + ", ".join(hedged[:3])
                    + " as visually AMBIGUOUS — hedged/vague wording about "
                    "them is correct grounding, never 'weak'.")
        work.append((seg, text, files, note))

    # 2. content-addressed verdict cache. A grounding verdict is a pure function
    #    of (model, narration, panels shown) — so the voiceover-time QA, which
    #    re-grounds narration ALREADY finalized at prepare time, hits the cache
    #    for every unchanged beat instead of re-paying the 26B. Collapses the
    #    redundant second pass (and heal re-runs) to ~0 gemma calls.
    import hashlib

    def _key(text: str, files: List[str], note: str = "") -> str:
        # the frame CONTENT is part of the key: scenes_clean/<f> is a derived
        # rendition (art_only crop, in-place push-in re-frame), so the same
        # name can hold a different picture between runs and a name-keyed
        # cache would reuse a judgment of an image that no longer exists.
        h = hashlib.sha1()
        digests = "\x00".join(
            rp.frame_digest(os.path.join(clean_dir, f)) for f in files[:6])
        for part in (model, text[:400], "\x00".join(files[:6]), digests, note):
            h.update(part.encode("utf-8", "replace"))
            h.update(b"\x00")
        return h.hexdigest()

    cache: Dict[str, Any] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:                                   # noqa: BLE001
            cache = {}
    keys = [_key(text, files, note) for (_, text, files, note) in work]
    miss = [i for i, k in enumerate(keys) if k not in cache]

    # judge only the MISSES, CONCURRENTLY so the 26B calls fill ollama's
    # OLLAMA_NUM_PARALLEL slots (the loop was serial — the dominant QA cost).
    # Each ollama_compat.chat builds its OWN Client + watchdog (no shared state),
    # so threading is safe; STUDIO_QA_CONC mirrors understanding's proven width.
    def _judge_one(i: int):
        _, text, files, note = work[i]
        try:
            return _judge([os.path.join(clean_dir, f) for f in files[:6]],
                          text, note)
        except Exception as e:                              # noqa: BLE001
            return e
    conc = max(1, int(os.environ.get("STUDIO_QA_CONC", "3")))
    if conc > 1 and len(miss) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=conc) as ex:
            fresh = dict(zip(miss, ex.map(_judge_one, miss)))
    else:
        fresh = {i: _judge_one(i) for i in miss}

    # resolve every beat to a verdict (cache hit or fresh) IN ORDER; persist the
    # fresh successes so the next pass reuses them. Failures aren't cached.
    verdicts: List[Any] = []
    dirty = False
    for i, k in enumerate(keys):
        if i in fresh:
            v = fresh[i]
            if not isinstance(v, Exception):
                cache[k] = v
                dirty = True
            verdicts.append(v)
        else:
            verdicts.append(cache[k])
    if cache_path and dirty:
        try:
            tmp = cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            os.replace(tmp, cache_path)
        except Exception:                                   # noqa: BLE001
            pass

    # 3. build flags in timeline order — identical to the serial output
    flags: List[Dict[str, Any]] = []
    for (seg, text, files, _note), v in zip(work, verdicts):
        if isinstance(v, Exception):
            flags.append(_flag("grounding_error", INFO,
                               f"judge failed on {seg}: {v}", segment_id=seg))
            continue
        if (v or {}).get("ok") is False:
            issue = str(v.get("issue") or "").strip()[:180]
            flags.append(_flag(
                "grounding_weak", WARN,
                f"weak/mis-grounded narration: {issue}",
                scene=files[0], segment_id=seg))
    return flags


# ---------------------------------------------------------------------------
# story-level QA: the checks the per-panel passes cannot see — does each
# segment tell a real beat (not filler), does the shown art belong to THIS
# beat (not a story-blind stand-in), and did a mandatory title/system card
# get dropped? These flag the failures the user caught the QA missing.
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"^\s*(the\s+(scene|story)\s+continues|to\s+be\s+continued|continues?)\.?\s*$",
    re.I)


def _is_title_card(ocr: str, vit: Dict[str, Any], *, ignore_chrome: bool = False) -> bool:
    """A styled title/system card (SYSTEM ACTIVATION., STARTING ACTIVATION.) —
    a short, mostly-uppercase phrase CENTERED ON A FLAT FRAME. The flat-frame
    test (*flat_frac*: fraction of near-white/near-black pixels, set by main()
    from the image) is what separates a real card from all-caps dialogue or a
    screamed SFX sitting on textured artwork — caps text alone cannot.

    ``ignore_chrome``: skip the chrome-stamp short-circuit. The story_group rescue
    uses this to recover an in-world SYSTEM card the LLM mislabeled 'chrome' — but
    ONLY after it has confirmed in-world system vocabulary, so genuine
    chapter-number / credits chrome (no such vocab) never reaches this path."""
    ocr = (ocr or "").strip()
    if not ignore_chrome and is_chrome_scene(vit):
        return False
    if rp.empty_bubble_panel(vit):
        return False
    # scanlation watermarks / URLs (ASURASCANS.COM, asura.gg, *.net) are SITE
    # CHROME, never a story title card — they must stay droppable, not "must
    # show". A real title/system card carries no domain.
    if re.search(r"[a-z0-9][\w-]*\.(com|net|org|gg|io|co|to|xyz|me|app|tv)\b",
                 ocr.lower()):
        return False
    # dialogue & SFX live on flat gutters too — they carry ~ ! ? or trailing
    # ellipses; a title/system card is a clean declarative name/phrase
    if "..." in ocr or any(ch in ocr for ch in "~!?"):
        return False
    words = [w for w in re.split(r"[^A-Za-z0-9']+", ocr) if any(c.isalpha() for c in w)]
    if not (2 <= len(words) <= 8):     # 1-word = SFX gibberish; long = a page
        return False
    letters = [c for c in ocr if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters)
    return (caps >= 0.8
            and float(vit.get("flat_frac") or 0.0) >= 0.6
            and float(vit.get("text_coverage") or 0.0) < 0.20
            and not vit.get("text_only"))


def _base_scene(f: str) -> str:
    """split halves (p044_a.jpg/p044_b.jpg) trace back to one source panel."""
    return re.sub(r"_[ab](\.[a-z]+)$", r"\1", str(f or ""))


def story_flags(plan: Dict[str, Any], beats_obj: Dict[str, Any],
                vitems: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    bn: Dict[int, str] = {}
    bfiles: Dict[int, set] = {}
    for b in (beats_obj or {}).get("beats") or []:
        try:
            gid = int(b.get("group_id"))
        except (TypeError, ValueError):
            continue
        bn[gid] = str(b.get("narration") or "")
        bfiles[gid] = {str(f) for f in (b.get("scene_files") or [])}

    shown_all: set = set()
    for item in (plan or {}).get("timeline") or []:
        if item.get("branding"):
            continue
        for c in item.get("cuts") or []:
            for f in (c.get("file"), c.get("file2")):
                if f:
                    shown_all.add(_base_scene(f))

    for item in (plan or {}).get("timeline") or []:
        if item.get("branding"):
            continue
        seg = str(item.get("segment_id") or "")
        m = _SEG_GROUP_RE.match(seg)
        gid = int(m.group(1)) if m else None
        text = (item.get("tts_text") or "").strip()
        cuts = item.get("cuts") or []

        # 1. filler / empty narration — the beat produced no real story line
        if not text or _FILLER_RE.match(text):
            flags.append(_flag(
                "filler_narration", ERROR,
                f"narration is empty/filler ({text[:40]!r}) — the beat carries "
                "no story; drop or re-roll the beat instead of voicing a "
                "placeholder", segment_id=seg))

        # 2. substituted/mismatched panel — none of the shown art belongs to
        # this beat (its real panel was dropped and a stand-in put on screen)
        intended = bfiles.get(gid) if gid is not None else None
        if intended:
            shown = {_base_scene(c.get("file")) for c in cuts if c.get("file")}
            if shown and not (shown & intended):
                held = any(c.get("held") for c in cuts)
                flags.append(_flag(
                    "panel_substituted", WARN if held else ERROR,
                    f"shown {sorted(shown)} is NONE of this beat's panels "
                    f"{sorted(intended)} — intended art dropped, "
                    + ("held stand-in" if held else "silent swap"),
                    segment_id=seg))

    # 3. dropped system/title card — these are story beats, never droppable.
    # A panel the understanding calls 'caption' is a narrative-voice MONOLOGUE
    # (its words ride the narration); it is SUPPOSED to be narrated, not shown,
    # so its absence from the montage is intended — never a system_card_dropped,
    # even when its short caps text looks like a title card to the heuristic.
    #
    # NOTE: the authoritative signal for in-world system panels is
    # system_card_unshown (ERROR, keyed on stamped panel_kind in
    # system_coverage_flags).  This OCR-heuristic WARN is retained as
    # belt-and-suspenders (it fires on title/cover/credit cards the
    # understanding may not stamp as "system") and is slated for removal
    # in the per-panel Ch7 cleanup.  An absent panel_kind=="system" that
    # also trips this heuristic will produce BOTH the ERROR and this WARN;
    # the ERROR is the actionable one.
    for f, vit in (vitems or {}).items():
        if str(vit.get("panel_kind") or "").lower() == "caption":
            continue
        if _base_scene(f) not in shown_all and _is_title_card(
                str(vit.get("ocr_clean") or ""), vit):
            flags.append(_flag(
                "system_card_dropped", WARN,
                f"title/system card {f} ({str(vit.get('ocr_clean') or '')[:30]!r}) "
                "was dropped before render — review if it's a real scene title "
                "(a cover / credit / watermark drop is fine and expected)",
                scene=str(f)))
    return flags


def plan_flags(plan: Dict[str, Any], *, clean_files: set,
               audio_exists: Callable[[str], bool]) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    timeline = plan.get("timeline") or []
    dims = plan.get("scene_dims") or {}
    # step-1 plans are built WITHOUT voiceover (timeline estimates durations
    # from word counts) — audio cannot exist yet and must not flag as ERROR
    voiced_plan = bool(plan.get("source_tts_index"))

    if timeline and timeline[0].get("branding"):
        flags.append(_flag("no_cold_open", WARN,
                           "video starts with the branding intro — no story "
                           "cold-open hook before it",
                           segment_id=str(timeline[0].get("segment_id"))))
    # Channel design (commit 3ea4271): per chapter there is NO intro and NO outro
    # — a chapter ENDS on its last story panel, the channel watermark is a
    # separate always-on overlay (not a timeline item), and the arc intro is
    # bundle-level (prepended at concat, reviewed separately). So a normal chapter
    # is EXPECTED to carry no branding item; the absence of an outro is BY DESIGN
    # and must never raise a flag. (no_cold_open above still warns IF an intro
    # ever leads a chapter timeline.)

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
                if not branding and not c.get("held"):
                    seen_parent_segments.setdefault(
                        parent_scene(f), set()).add(seg)
            dur = float(c.get("dur") or 0.0)
            if dur < 2.0 and not c.get("held"):
                flags.append(_flag("flash_cut", ERROR,
                                   f"cut shows {c.get('file')} for only "
                                   f"{dur:.2f}s",
                                   scene=str(c.get("file") or ""),
                                   segment_id=seg))
            if (c.get("file") == prev_file and not c.get("held")
                    and not c.get("ken_variety")):
                # ken_variety sub-cuts (V1) deliberately repeat the file with
                # DIFFERENT ken regions — not an accidental repeat
                flags.append(_flag("repeat_cut", WARN,
                                   "same file in consecutive cuts",
                                   scene=str(c.get("file")), segment_id=seg))
            prev_file = c.get("file")

        tile = sum(float(c.get("dur") or 0.0) for c in cuts)
        item_dur = float(item.get("duration_sec") or 0.0)
        if abs(tile - item_dur) > 0.51:
            # A residual time-hole = a black screen on the timeline; never ship
            # it silently. ERROR blocks autopilot's spotless-QA advance and parks
            # the chapter for review (the render_prep fix removes the cause).
            flags.append(_flag("cut_gap", ERROR,
                               f"cuts tile {tile:.2f}s of a {item_dur:.2f}s "
                               "item (gap or overlap on screen)",
                               segment_id=seg))

        if not branding:
            audio = item.get("tts_audio")
            if not audio or not audio_exists(str(audio)):
                if voiced_plan:
                    flags.append(_flag("missing_audio", ERROR,
                                       f"tts_audio missing on disk: {audio}",
                                       segment_id=seg))
                else:
                    flags.append(_flag("estimate_plan", INFO,
                                       "pre-voiceover plan: timing estimated, "
                                       "audio comes after story approval",
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
        blocks = []
        n_files = 0
        for g in gallery:
            seg = str(g.get("segment_id") or "")
            narration = str(g.get("narration") or "")
            figs = []
            for fn in g.get("files") or []:
                n_files += 1
                figs.append(
                    '<figure style="margin:4px;display:inline-block;'
                    'text-align:center;background:#fff;border:1px solid '
                    '#ddd;padding:4px">'
                    f"{_img_tag(thumbs, str(fn), max_w=170)}"
                    f'<figcaption style="font-size:11px;color:#444">'
                    f"{_html.escape(str(fn))}</figcaption></figure>")
            blocks.append(
                '<div style="background:#fff;border:1px solid #ddd;'
                'border-radius:6px;padding:8px 12px;margin:10px 0">'
                f'<div style="font-size:12px;color:#888"><code>'
                f"{_html.escape(seg)}</code></div>"
                + (f'<div style="font-size:14px;margin:4px 0 8px">'
                   f"{_html.escape(narration)}</div>" if narration else "")
                + "".join(figs) + "</div>")
        gallery_html = (f"<h2>All shown cuts ({n_files}) — timeline order, "
                        f"narration per segment</h2>{''.join(blocks)}")

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

# ---------------------------------------------------------------------------
# OCR grounding — suppress false narration_mismatch WARNs. The visual judge
# compares a line to the SHOWN panel; a number/name SPOKEN in another panel of
# the same beat (e.g. "THERE ARE MORE THAN THREE HUNDRED OF YOU") then reads as
# "invented" though it is grounded in the on-panel dialogue (OCR).
# ---------------------------------------------------------------------------

_GROUND_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "it", "its", "he", "she",
    "they", "you", "i", "his", "her", "their", "that", "this", "these", "those",
    "with", "for", "as", "by", "not", "no", "do", "does", "did", "can", "could",
    "will", "would", "should", "just", "all", "any", "has", "have", "had", "who",
    "what", "when", "where", "why", "how", "if", "so", "up", "out", "him", "them",
    "my", "me", "we", "our", "your", "from", "into", "than", "then", "there",
    "here", "over", "about", "only", "even", "still", "now", "also", "while",
    "which", "after", "before", "because", "since",
}


def _content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if len(w) >= 3 and w not in _GROUND_STOP}


def _ocr_grounds_narration(narration: str, ocr: str,
                           min_cov: float = 0.5, min_ocr_words: int = 5) -> bool:
    """True when the narration reproduces the group's on-panel DIALOGUE: at least
    *min_cov* of the OCR's distinctive words also appear in the narration. Needs
    enough OCR (*min_ocr_words*) to be a real signal — a textless action beat
    can't be OCR-grounded, so the visual judge still rules there."""
    ow = _content_words(ocr)
    if len(ow) < min_ocr_words:
        return False
    nw = _content_words(narration)
    if not nw:
        return False
    return len(ow & nw) / len(ow) >= min_cov


def _suppress_grounded_mismatches(
        flags: List[Dict[str, Any]], beats_obj: Dict[str, Any],
        vitems: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop narration_mismatch WARNs whose narration is supported by the group's
    OCR (on-panel dialogue). Conservative: only fires when the beat carries real
    dialogue AND the line reproduces most of it."""
    g_narr: Dict[int, str] = {}
    g_ocr: Dict[int, str] = {}
    for b in (beats_obj or {}).get("beats") or []:
        try:
            gid = int(b.get("group_id"))
        except (TypeError, ValueError):
            continue
        g_narr[gid] = str(b.get("narration") or "")
        g_ocr[gid] = " ".join(
            str((vitems.get(str(sf)) or {}).get("ocr_clean") or "")
            for sf in (b.get("scene_files") or []))
    out: List[Dict[str, Any]] = []
    dropped = 0
    for f in flags:
        if f.get("code") in ("narration_mismatch", "grounding_weak"):
            m = _SEG_GROUP_RE.match(str(f.get("segment_id") or ""))
            gid = int(m.group(1)) if m else None
            if gid is not None and _ocr_grounds_narration(
                    g_narr.get(gid, ""), g_ocr.get(gid, "")):
                dropped += 1
                continue   # grounded in the beat's dialogue — false positive
        out.append(f)
    if dropped:
        print(f"[qa] suppressed {dropped} OCR-grounded grounding WARN(s)")
    return out


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
    ap.add_argument("--semantic", action="store_true",
                    help="Gemma vision-judge: narration vs shown panel per "
                         "segment (WARN-level)")
    ap.add_argument("--semantic-model", default="gemma4:26b")
    ap.add_argument("--max-hold-sec", type=float, default=10.0,
                    help="long_hold cap: one file shown continuously past this "
                         "while standing in for another beat's art BLOCKS "
                         "([render].max_same_image_hold_sec)")
    ap.add_argument("--semantic-heal", action="store_true",
                    help="run the grounding 'eyes' (grounding_weak flags that "
                         "feed auto-heal); off by default — opt-in via "
                         "[heal].semantic. Pairs with the strictly-better "
                         "safeguard in the heal loop.")
    args = ap.parse_args()

    ep = args.episode_dir.rstrip("/")

    # Manifest completeness + staleness guard — runs before we open any file
    # so a missing or stale plan is flagged immediately rather than surfacing
    # as a confusing open() error or silent use of old cuts.
    _freshness_issues = _verify_chapter_freshness(ep)
    _pre_flags: List[Dict[str, Any]] = [
        _flag(iss["code"], ERROR, iss["detail"], scene=iss.get("file", ""))
        for iss in _freshness_issues
    ]

    plan_path = args.plan or os.path.join(ep, "render.plan.clean.json")
    # Hard-error on a missing/corrupt/keyless plan — a silent empty "timeline"
    # default used to let a corrupt plan pass through QA as an empty report.
    plan = read_manifest(plan_path, required_keys=("timeline",))
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
                    "subjects": it.get("subjects") or [],
                    "n_words": len((it.get("vision") or {}).get("ocr_words") or []),
                    # carry the understanding so is_chrome_scene defers to it (no
                    # false chrome_leak on a 'story' panel whose OCR is just '1')
                    "panel_kind": it.get("panel_kind"),
                }
    reconciled_files: set = set()
    sp_ = os.path.join(ep, "manifest.scenes.json")
    if os.path.exists(sp_):
        try:
            with open(sp_, "r", encoding="utf-8") as f:
                for sc in json.load(f).get("scenes") or []:
                    of = str(sc.get("out_file") or "")
                    if sc.get("recovered"):
                        vitems.setdefault(of, {})["recovered"] = True
                    if sc.get("reconciled_seam"):
                        reconciled_files.add(of)
        except Exception:
            pass

    flags: List[Dict[str, Any]] = _pre_flags + plan_flags(
        plan, clean_files=clean_files, audio_exists=os.path.exists)

    def _load_manifest(name: str) -> Dict[str, Any]:
        p = os.path.join(ep, name)
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # flat-frame fraction for the dropped-title-card detector — read the source
    # scene only for short-caps candidates (skips the full-image sweep)
    scenes_dir = os.path.join(ep, "scenes")
    for f, vit in vitems.items():
        ocr = str(vit.get("ocr_clean") or "")
        if 1 <= len(ocr.split()) <= 10 and not vit.get("text_only"):
            sp = os.path.join(scenes_dir, f)
            im = cv2.imread(sp) if os.path.exists(sp) else None
            if im is not None:
                g = im.mean(axis=2)
                vit["flat_frac"] = float(((g > 235) | (g < 25)).mean())

    beats_obj = _load_manifest("manifest.beats.json")
    groups_obj = _load_manifest("manifest.groups.json")
    script_obj = _load_manifest("manifest.script.json")
    story_obj = _load_manifest("manifest.story.json")
    cast_obj = _load_manifest("manifest.cast.json")
    understood_obj = _load_manifest("manifest.panels.understood.json")

    flags.extend(alignment_flags(plan, beats_obj, groups_obj, script_obj))
    flags.extend(audio_flags(plan, _load_manifest("tts/tts_index.json")))
    flags.extend(montage_flags(plan))
    flags.extend(page_floor_flags(ep))
    flags.extend(held_repeat_flags(plan))

    def _static_ceiling_exempt(f: str) -> bool:
        """Files the unconditional long_hold static ceiling excuses: renderer
        branches that are never static (Cut.tsx wide cover-drift w/h>=1.3,
        tall scroll h/w>=2.0 — parity with remotion/src/plan.ts) and text
        panels that NEED stillness to read (doc, stamped panel_kind=='system'
        — NOT scene_dims' pixel-level 'sys', which the system-box YOLO trips
        on mere SFX text: the 22.8s review holds all carried sys:True)."""
        d = dims.get(f) or {}
        if d.get("doc"):
            return True
        w, h = float(d.get("w") or 0.0), float(d.get("h") or 0.0)
        if h > 0 and w / h >= 1.3:      # WIDE_COVER_MIN_ASPECT
            return True
        if w > 0 and h / w >= 2.0:      # TALL_SCROLL_MIN_ASPECT
            return True
        pk = str((vitems.get(parent_scene(f)) or vitems.get(f)
                  or {}).get("panel_kind") or "").strip().lower()
        return pk == "system"

    flags.extend(long_hold_flags(plan, beats_obj,
                                 max_hold_sec=args.max_hold_sec,
                                 is_exempt=_static_ceiling_exempt))
    flags.extend(sfx_voiced_flags(script_obj))
    flags.extend(raw_caps_voiced_flags(script_obj))
    flags.extend(shot_description_flags(beats_obj))
    flags.extend(truncated_line_flags(beats_obj))
    flags.extend(filename_in_narration_flags(beats_obj))
    flags.extend(impact_marker_leak_flags(beats_obj))
    flags.extend(figures_leak_flags(beats_obj))
    flags.extend(mood_tag_leak_flags(beats_obj))
    flags.extend(story_flags(plan, beats_obj, vitems))
    flags.extend(system_coverage_flags(beats_obj, plan, vitems))
    flags.extend(span_cover_flags(plan, beats_obj, vitems))
    flags.extend(impact_mismatch_flags(beats_obj, understood_obj))
    flags.extend(line_overlong_flags(beats_obj))
    flags.extend(narration_offset_flags(beats_obj, understood_obj))
    flags.extend(actor_mismatch_flags(beats_obj, understood_obj, cast_obj,
                                      vitems))
    flags.extend(actor_count_flags(beats_obj, understood_obj, cast_obj))
    flags.extend(ledger_contradiction_flags(
        beats_obj, _load_manifest("manifest.ledger.json"), cast_obj))
    flags.extend(cold_open_flags(beats_obj))
    flags.extend(phrase_echo_flags(beats_obj))

    recap_style = analyze_recap_style(
        script_obj, beats_obj, story_obj, cast_obj, vitems)
    for issue in recap_style["issues"]:
        flags.append(_flag(
            str(issue.get("code") or "recap_style"),
            WARN,
            str(issue.get("detail") or "recap style rule missed"),
            scene=str(issue.get("scene") or ""),
            segment_id=str(issue.get("segment_id") or "")))

    def _judge_caption_carried(caption: str, narration: str) -> bool:
        try:
            from ollama_compat import chat as _chat
            resp = _chat(model=args.semantic_model, think=False, messages=[{
                "role": "user", "content":
                "CAPTION on the page: " + caption[:300] + "\n"
                "NARRATION spoken: " + narration[:400] + "\n"
                "Does the narration carry the caption's full meaning "
                "(paraphrase OK)? Reply ONLY JSON: {\"carried\": true/false}"}],
                options={"temperature": 0, "num_predict": 60,
                         "num_ctx": 8192})
            m = re.search(r"\{.*\}", str(resp["message"]["content"] or ""),
                          re.S)
            return bool(m and json.loads(m.group(0)).get("carried") is True)
        except Exception:
            return False

    flags.extend(caption_unvoiced_flags(
        beats_obj, vitems,
        understood_by_file={
            os.path.basename(str(p.get("scene_file") or "")): p
            for p in ((understood_obj or {}).get("panels") or [])
            if isinstance(p, dict)},
        arbitrate=_judge_caption_carried if args.semantic else None))
    if args.semantic or args.semantic_heal:
        # PER-BEAT montage grounding judge: ALL of a beat's panels go to Gemma in
        # ONE call (~1 call/group, ~23/chapter), vs the retired per-panel judge
        # that cost ~1 call PER SHOWN CUT (~61/chapter) for the same check — and
        # this one is montage-aware, so it has fewer false positives by design.
        flags.extend(grounding_flags(
            plan, clean_dir, model=args.semantic_model,
            cache_path=os.path.join(ep, ".grounding_cache.json"),
            uncertain_files={
                str(p.get("scene_file") or "")
                for p in ((understood_obj or {}).get("panels") or [])
                if p.get("uncertain")}))
        # a number/name SPOKEN in a non-shown panel is grounded in the dialogue —
        # drop the visual judge's false positive in that case
        flags = _suppress_grounded_mismatches(
            flags, beats_obj, vitems)

    detector = None
    if not args.no_detector:
        detector = rp._load_bubble_detector(args.device)

    # missing stamp = plan written before the stamp existed; every such plan in
    # the fleet was produced under keep-default render_prep, so default keep.
    # art_only frames dialogue OUT of the shown window (or falls back to keep)
    # — surviving bubbles still ship AS DRAWN, same gating as keep.
    kept_bubbles = plan.get("bubble_shown_mode", "keep") in ("keep", "art_only")

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
            sys=sys_panel, segment_id=seg_by_file[fname],
            vitem=vitems.get(parent_scene(fname)) or vitems.get(fname),
            reconciled=(parent_scene(fname) in reconciled_files
                        or fname in reconciled_files),
            kept_bubbles=kept_bubbles))

    # consecutive on-screen near-duplicates (zoom pairs included)
    _imc: Dict[str, Any] = {}

    def _clean_img(f: str):
        if f not in _imc:
            _imc[f] = cv2.imread(os.path.join(clean_dir, f))
        return _imc[f]

    # a panel owning its own narration line is a distinct beat, never a cross_dup
    flags.extend(cross_dup_flags(cuts, _clean_img,
                                 narrated=rp.narrated_files_from_plan(plan)))

    # residual near-dup tripwire: cross_dup keys on containment; this bubble-masks
    # the hash so identical art under DIFFERENT dialogue (whose outlines survive
    # cleaning) is caught too. WARN only — render_prep is the real fix, and
    # auto-dropping here would re-hit the sole-cut-empties-a-segment problem.
    _bxc: Dict[str, List[Tuple[int, int, int, int]]] = {}

    def _qa_boxes(f: str) -> List[Tuple[int, int, int, int]]:
        if f not in _bxc:
            img = _clean_img(f)
            if detector is None or img is None:
                _bxc[f] = []
            else:
                _bxc[f] = [(int(x1), int(y1), int(x2), int(y2))
                           for (x1, y1, x2, y2, _s) in detector.detect(
                               img, imgsz=1024, conf=args.bubble_conf)]
        return _bxc[f]

    def _qa_exempt(f: str) -> bool:
        d = dims.get(f) or {}
        if d.get("doc") or d.get("sys"):
            return True
        pk = str((vitems.get(parent_scene(f)) or vitems.get(f)
                  or {}).get("panel_kind") or "").strip().lower()
        return pk in ("system", "doc", "document")

    flags.extend(near_dup_residual_flags(cuts, _clean_img, _qa_boxes,
                                         is_exempt=_qa_exempt))

    # shown-twin INVARIANT tripwire (BLOCKING): masked-raw hashing over the
    # ORIGINAL scenes/ panels — the same rp.twin_verdict render_prep's final
    # invariant pass enforces, so this only fires when a future bypass ships a
    # twin pair the pass should have folded. Raw images/boxes (not the clean
    # crops): crop geometry must not be able to hide or manufacture a twin.
    _rawc: Dict[str, Any] = {}

    def _raw_img(f: str):
        if f not in _rawc:
            _rawc[f] = cv2.imread(os.path.join(scenes_dir, f))
        return _rawc[f]

    _rawbx: Dict[str, List[Tuple[int, int, int, int]]] = {}

    def _raw_boxes(f: str) -> List[Tuple[int, int, int, int]]:
        if f not in _rawbx:
            img = _raw_img(f)
            if detector is None or img is None:
                _rawbx[f] = []
            else:
                _rawbx[f] = [(int(x1), int(y1), int(x2), int(y2))
                             for (x1, y1, x2, y2, _s) in detector.detect(
                                 img, imgsz=1024, conf=args.bubble_conf)]
        return _rawbx[f]

    def _raw_ocr(f: str) -> str:
        vit = vitems.get(parent_scene(f)) or vitems.get(f) or {}
        return str(vit.get("ocr_clean") or "")

    flags.extend(dup_shown_flags(cuts, _raw_img, _raw_boxes, _raw_ocr,
                                 is_exempt=_qa_exempt,
                                 cap_kept_pairs=plan.get("twin_cap_kept")))

    # V2 echo net (WARN, measure-first): shown-crop twins whose RAWS are
    # distinct — the zoom-echo / husk-crop class dup_shown correctly ignores
    # but the viewer reads as a stutter. Same helpers as the checks above
    # (clean-crop masked hashing / raw masked hashing), so QA and render_prep's
    # ken differentiation measure the same thing. STAMPED-only exemption —
    # _qa_exempt's pixel-level sys flag would self-exempt the evidence class
    # (the p000090/p000095 incident panels all carried sys:True).
    flags.extend(perceptual_echo_flags(cuts, _clean_img, _qa_boxes,
                                       _raw_img, _raw_boxes,
                                       is_exempt=echo_exempt_fn(dims, vitems)))

    # vision-level checks once per shown parent scene
    seen_parents: set = set()
    for c in cuts:
        parent = parent_scene(c["file"])
        if parent in seen_parents or parent not in vitems:
            continue
        seen_parents.add(parent)
        vit = vitems[parent]
        mid = None
        if needs_image_stats(str(vit.get("ocr_clean") or "")):
            # same image-stat disambiguation the gate uses (watermark-on-art
            # vs cover; OCR-blind number cards)
            src = cv2.imread(os.path.join(ep, "scenes", parent))
            if src is not None:
                g = src.mean(axis=2)
                mid = float(((g > 60) & (g < 200)).mean())
        flags.extend(vision_flags(
            parent, vit,
            dims_entry=dims.get(c["file"]),
            series_title=args.series_title or None,
            midtone_frac=mid,
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
    report["recap_style"] = recap_style["metrics"]

    # gallery: one block per timeline item — narration + its cut thumbs
    gallery: List[Dict[str, Any]] = []
    seen_gallery: set = set()
    for item in plan.get("timeline") or []:
        files: List[str] = []
        for c in item.get("cuts") or []:
            for f in (c.get("file"), c.get("file2")):
                if f:
                    files.append(str(f))
                    seen_gallery.add(str(f))
        if not files and item.get("branding"):
            continue  # outro end-card draws itself
        narration = ("" if item.get("branding")
                     else str(item.get("tts_text") or ""))
        seg = str(item.get("segment_id") or "")
        gallery.append({"segment_id": seg, "narration": narration,
                        "files": files})

    thumbs: Dict[str, bytes] = {}
    want = ({str(f.get("scene") or f.get("thumb_scene") or "") for f in flags}
            | seen_gallery)
    for scene in sorted(want):
        if not scene or scene in thumbs:
            continue
        img = cv2.imread(os.path.join(clean_dir, scene))
        if img is None:  # parent-named flag for a split scene -> original
            img = cv2.imread(os.path.join(ep, "scenes", scene))
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
