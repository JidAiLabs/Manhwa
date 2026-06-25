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
from render_prep import multi_scale_contained
from scene_chrome import is_chrome_scene, needs_image_stats
from studio.qa_flags import longest_common_run
from narration_consistency import audio_consistency, strip_chrome_opener
from manifest_freshness import verify_chapter as _verify_chapter_freshness

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

    if h > 8000:
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

def cross_dup_flags(seq: Sequence[Dict[str, Any]],
                    get_img) -> List[Dict[str, Any]]:
    """Consecutive shown cuts that are near-identical (or zoom pairs) — the
    on-screen duplicate class the user keeps catching by eye."""
    flags: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    for cur in seq:
        f = str(cur.get("file"))
        if prev and str(prev.get("file")) != f:
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
    if rp.empty_bubble_panel(vitem):
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

    if bool((script_obj or {}).get("microbeats")):
        grouped: Dict[int, List[str]] = {}
        first_seg: Dict[int, str] = {}
        for gid, seg, text in plan_items:
            grouped.setdefault(gid, []).append(text)
            first_seg.setdefault(gid, seg)
        compare_items = [
            (gid, first_seg.get(gid, f"g{gid:04d}_p00"), " ".join(texts))
            for gid, texts in grouped.items()
        ]
    else:
        compare_items = plan_items

    for gid, seg, text in compare_items:
        narr = bn.get(gid)
        # scrub series-intro/title-card chrome from the beats side too, matching
        # what the script stage voices — otherwise a legitimately-scrubbed plan
        # reads as "stale" against an un-scrubbed beats line (false positive).
        a, b = _norm_narr(text), _norm_narr(strip_chrome_opener(narr or ""))
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
    return flags


_UI_TOKENS = {"read", "ep", "episode", "episodes", "comments", "comment",
              "views", "view", "likes", "like", "subscribe", "next", "prev",
              "previous", "tap", "menu", "notice", "unread"}


def caption_unvoiced_flags(beats_obj: Dict[str, Any],
                           vitems: Dict[str, Dict[str, Any]],
                           *, min_words: int = 4,
                           min_coverage: float = 0.5,
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
            cwords = {w for w in _norm_narr(txt).split()
                      if not w.isdigit() and w not in _UI_TOKENS}
            if len(cwords) < min_words:
                continue
            cov = len(cwords & nwords) / max(1, len(cwords))
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
            options={"temperature": 0, "num_predict": 200})
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
                    cache_path: Optional[str] = None) -> List[Dict[str, Any]]:
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

    def _judge(paths: List[str], text: str) -> Dict[str, Any]:
        resp = _ollama_chat(
            model=model, think=False,
            messages=[{"role": "user",
                       "content": _GROUND_PROMPT.format(text=text[:400]),
                       "images": paths}],
            options={"temperature": 0, "num_predict": 200})
        raw = str(resp["message"]["content"] or "")
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}

    # 1. collect the beats to judge, in timeline order (output stays deterministic)
    work: List[Tuple[str, str, List[str]]] = []   # (segment_id, narration, files)
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
        work.append((seg, text, files))

    # 2. content-addressed verdict cache. A grounding verdict is a pure function
    #    of (model, narration, panels shown) — so the voiceover-time QA, which
    #    re-grounds narration ALREADY finalized at prepare time, hits the cache
    #    for every unchanged beat instead of re-paying the 26B. Collapses the
    #    redundant second pass (and heal re-runs) to ~0 gemma calls.
    import hashlib

    def _key(text: str, files: List[str]) -> str:
        h = hashlib.sha1()
        for part in (model, text[:400], "\x00".join(files[:6])):
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
    keys = [_key(text, files) for (_, text, files) in work]
    miss = [i for i, k in enumerate(keys) if k not in cache]

    # judge only the MISSES, CONCURRENTLY so the 26B calls fill ollama's
    # OLLAMA_NUM_PARALLEL slots (the loop was serial — the dominant QA cost).
    # Each ollama_compat.chat builds its OWN Client + watchdog (no shared state),
    # so threading is safe; STUDIO_QA_CONC mirrors understanding's proven width.
    def _judge_one(i: int):
        _, text, files = work[i]
        try:
            return _judge([os.path.join(clean_dir, f) for f in files[:6]], text)
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
    for (seg, text, files), v in zip(work, verdicts):
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
    brandings = {str(i.get("branding")) for i in timeline if i.get("branding")}
    # channel design: NO intro on any video, outro only — so the ONLY branding
    # expectation is the outro. Warn only when the outro is missing.
    if "outro" not in brandings:
        flags.append(_flag("missing_branding", WARN,
                           f"branding items present: "
                           f"{sorted(brandings) or 'none'} — expected an outro"))

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
            if dur < 1.2:
                flags.append(_flag("flash_cut", WARN,
                                   f"cut shows {c.get('file')} for only "
                                   f"{dur:.2f}s",
                                   scene=str(c.get("file") or ""),
                                   segment_id=seg))
            if c.get("file") == prev_file and not c.get("held"):
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
                    "subjects": it.get("subjects") or [],
                    "n_words": len((it.get("vision") or {}).get("ocr_words") or []),
                    # carry the understanding so is_chrome_scene defers to it (no
                    # false chrome_leak on a 'story' panel whose OCR is just '1')
                    "panel_kind": it.get("panel_kind"),
                }
    sp_ = os.path.join(ep, "manifest.scenes.json")
    if os.path.exists(sp_):
        try:
            with open(sp_, "r", encoding="utf-8") as f:
                for sc in json.load(f).get("scenes") or []:
                    if sc.get("recovered"):
                        vitems.setdefault(str(sc.get("out_file") or ""),
                                          {})["recovered"] = True
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

    flags.extend(alignment_flags(plan, _load_manifest("manifest.beats.json"),
                                 _load_manifest("manifest.groups.json"),
                                 _load_manifest("manifest.script.json")))
    flags.extend(audio_flags(plan, _load_manifest("tts/tts_index.json")))
    flags.extend(montage_flags(plan))
    flags.extend(page_floor_flags(ep))
    flags.extend(held_repeat_flags(plan))
    flags.extend(sfx_voiced_flags(_load_manifest("manifest.script.json")))
    flags.extend(raw_caps_voiced_flags(_load_manifest("manifest.script.json")))
    flags.extend(story_flags(plan, _load_manifest("manifest.beats.json"), vitems))
    flags.extend(system_coverage_flags(
        _load_manifest("manifest.beats.json"), plan, vitems))

    def _judge_caption_carried(caption: str, narration: str) -> bool:
        try:
            from ollama_compat import chat as _chat
            resp = _chat(model=args.semantic_model, think=False, messages=[{
                "role": "user", "content":
                "CAPTION on the page: " + caption[:300] + "\n"
                "NARRATION spoken: " + narration[:400] + "\n"
                "Does the narration carry the caption's full meaning "
                "(paraphrase OK)? Reply ONLY JSON: {\"carried\": true/false}"}],
                options={"temperature": 0, "num_predict": 60})
            m = re.search(r"\{.*\}", str(resp["message"]["content"] or ""),
                          re.S)
            return bool(m and json.loads(m.group(0)).get("carried") is True)
        except Exception:
            return False

    flags.extend(caption_unvoiced_flags(
        _load_manifest("manifest.beats.json"), vitems,
        arbitrate=_judge_caption_carried if args.semantic else None))
    if args.semantic or args.semantic_heal:
        # PER-BEAT montage grounding judge: ALL of a beat's panels go to Gemma in
        # ONE call (~1 call/group, ~23/chapter), vs the retired per-panel judge
        # that cost ~1 call PER SHOWN CUT (~61/chapter) for the same check — and
        # this one is montage-aware, so it has fewer false positives by design.
        flags.extend(grounding_flags(plan, clean_dir, model=args.semantic_model,
                                     cache_path=os.path.join(ep, ".grounding_cache.json")))
        # a number/name SPOKEN in a non-shown panel is grounded in the dialogue —
        # drop the visual judge's false positive in that case
        flags = _suppress_grounded_mismatches(
            flags, _load_manifest("manifest.beats.json"), vitems)

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
            sys=sys_panel, segment_id=seg_by_file[fname],
            vitem=vitems.get(parent_scene(fname)) or vitems.get(fname)))

    # consecutive on-screen near-duplicates (zoom pairs included)
    _imc: Dict[str, Any] = {}

    def _clean_img(f: str):
        if f not in _imc:
            _imc[f] = cv2.imread(os.path.join(clean_dir, f))
        return _imc[f]

    flags.extend(cross_dup_flags(cuts, _clean_img))

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
