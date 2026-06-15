#!/usr/bin/env python3
"""
timeline_planner.py

Fixes:
- Narrated timing follows audio precisely (float seconds), no per-shot ceil.
- Emits cuts[] so Blender can montage multiple panels across the shot duration.
- Stronger filtering: blank + thin-strip content + bubble/SFX dominated frames.
"""

import argparse
import json
import os
import sys
import re
import math
import wave
from typing import Any, Dict, List, Tuple, Optional

# Shared keep/redundant selection logic (sibling tool module).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_selection import choose_kept_scenes  # noqa: E402

try:
    from PIL import Image, ImageStat, ImageFilter
except Exception:
    Image = None
    ImageStat = None
    ImageFilter = None


# -----------------------------
# IO helpers
# -----------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def text_len(x: Any) -> int:
    if not x:
        return 0
    if isinstance(x, str):
        return len(x.strip())
    if isinstance(x, list):
        return sum(len(str(s).strip()) for s in x if s is not None)
    return len(str(x).strip())


def _safe_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def _norm_words(words: Any) -> List[str]:
    if not isinstance(words, list):
        return []
    out: List[str] = []
    for w in words:
        if not w:
            continue
        s = str(w).strip().lower()
        if s:
            out.append(s)
    return out


def _has_any(hay: List[str], needles: List[str]) -> bool:
    hs = set([x.lower() for x in hay if x])
    for n in needles:
        if n.lower() in hs:
            return True
    return False


# -----------------------------
# Manifest indexing
# -----------------------------
def find_groups(groups_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(groups_obj.get("shots"), list):
        return groups_obj["shots"]
    if isinstance(groups_obj.get("groups"), list):
        return groups_obj["groups"]
    raise ValueError("No shots/groups array found in groups manifest")


def index_beats(beats_obj: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    beats = beats_obj.get("beats") or []
    out: Dict[int, Dict[str, Any]] = {}
    for b in beats:
        gid = b.get("group_id")
        if isinstance(gid, int):
            out[gid] = b
    return out


def index_script(script_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Key by segment_id (string).  Falls back to g{gid:04d}_p{i:02d} when
    segment_id is absent on the shot, matching script_expander's formula."""
    out: Dict[str, Dict[str, Any]] = {}
    sections = script_obj.get("sections") or []
    if not isinstance(sections, list):
        return out

    for sec in sections:
        sec_idx = int(sec.get("section_index") or 0)
        paras = sec.get("script_paragraphs") or []
        shots = sec.get("shots") or []
        if not isinstance(paras, list) or not isinstance(shots, list):
            continue

        n = min(len(paras), len(shots))
        for i in range(n):
            s = shots[i]
            if not isinstance(s, dict):
                continue
            gid = int(s.get("group_id") or 0)
            if gid <= 0:
                continue
            sid = _safe_str(s.get("segment_id")) or f"g{gid:04d}_p{i:02d}"

            p = paras[i]
            if isinstance(p, dict):
                paragraph_text = _safe_str(p.get("text"))
                delivery_tag = _safe_str(p.get("delivery_tag"))
            else:
                paragraph_text = _safe_str(p)
                delivery_tag = ""

            out[sid] = {
                "segment_id": sid,
                "group_id": gid,
                "section_index": sec_idx,
                "beat_id": int(s.get("beat_id") or i),
                "paragraph": paragraph_text,
                "delivery_tag": delivery_tag,
                "cliffhanger_line": _safe_str(sec.get("cliffhanger_line")),
            }

    return out


# -----------------------------
# TTS index
# -----------------------------
def _index_tts(tts_obj: Dict[str, Any], tts_index_path: str) -> Dict[str, Dict[str, Any]]:
    """Key by segment_id (string).  Falls back to group_id-based key when
    segment_id is absent on the clip item (legacy TTS outputs)."""
    out: Dict[str, Dict[str, Any]] = {}

    items = tts_obj.get("clips") or tts_obj.get("items") or tts_obj.get("sections") or []
    if not isinstance(items, list):
        return out

    base_dir = os.path.dirname(os.path.abspath(tts_index_path))
    for it in items:
        if not isinstance(it, dict):
            continue
        gid = int(it.get("group_id") or 0)
        sid = _safe_str(it.get("segment_id")) or (f"g{gid:04d}" if gid > 0 else "")
        if not sid:
            continue

        audio_path = _safe_str(it.get("audio_file") or it.get("audio_path") or it.get("path") or "")
        if audio_path and not os.path.isabs(audio_path):
            audio_path = os.path.normpath(os.path.join(base_dir, audio_path))

        dur = it.get("duration_sec")
        try:
            dur_f = float(dur) if dur is not None else 0.0
        except Exception:
            dur_f = 0.0

        out[sid] = {
            "segment_id": sid,
            "group_id": gid,
            "audio_path": audio_path,
            "duration_sec": dur_f,
            "text": _safe_str(it.get("sent_text") or it.get("text") or ""),
            "voice_id": _safe_str(tts_obj.get("voice_id") or ""),
            "model_id": _safe_str(tts_obj.get("model_id") or ""),
        }
    return out


def _wav_duration_sec(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return float(frames) / float(rate)


# -----------------------------
# Duration (FIXED: returns float)
# -----------------------------
def compute_duration_sec(
    *,
    mode: str,
    tts_text: str,
    overlays: List[Dict[str, Any]],
    base_min: float,
    max_sec: float,
    chars_per_sec: float,
    audio_duration_sec: float,
    audio_pad_sec: float,
) -> float:
    base_min = float(base_min)
    max_sec = float(max_sec)

    if mode == "narrated" and audio_duration_sec and audio_duration_sec > 0.0:
        dur = float(audio_duration_sec) + float(audio_pad_sec)
        dur = clamp(dur, base_min, max_sec)
        return float(dur)

    overlay_chars = sum(text_len(o.get("text")) for o in overlays if isinstance(o, dict))
    narr_chars = text_len(tts_text) if mode == "narrated" else 0
    total_chars = overlay_chars + narr_chars
    reading = total_chars / float(chars_per_sec) if chars_per_sec > 0 else 0.0
    dur = base_min + reading
    dur = clamp(dur, base_min, max_sec)
    return float(dur)


# -----------------------------
# Vision / camera (unchanged)
# -----------------------------
def _normalize_camera_motion_hint(beat: Dict[str, Any]) -> str:
    rh = beat.get("rendering_hints") or {}
    cm = _safe_str(rh.get("camera_motion")).lower()
    cm = re.sub(r"[^a-z_ ]+", "", cm).strip().replace(" ", "_")
    return cm


def _choose_motion_mode(beat: Dict[str, Any]) -> str:
    hint = _normalize_camera_motion_hint(beat)
    mood_words = _norm_words(beat.get("mood_words") or [])
    emotional_turn = _safe_str(beat.get("emotional_turn")).lower()

    direct_ok = {
        "static", "kenburns", "slow_pan", "pan", "zoom_in", "zoom_out",
        "tilt_up", "tilt_down", "slide_left", "slide_right", "push_in", "pull_out",
    }
    if hint in direct_ok:
        if hint in ("slow_pan", "pan"):
            return "kenburns"
        if hint == "push_in":
            return "zoom_in"
        if hint == "pull_out":
            return "zoom_out"
        return hint

    if _has_any(mood_words, ["action", "panic", "chase", "fight", "impact", "chaos"]):
        return "slide_left"
    if _has_any(mood_words, ["reveal", "introduction", "hero", "awe", "arrival"]):
        return "tilt_up"
    if _has_any(mood_words, ["sad", "regret", "defeat", "loss", "mourning"]):
        return "tilt_down"
    if _has_any(mood_words, ["tension", "mystery", "horror", "threat", "danger"]):
        return "zoom_in"
    if "calm" in emotional_turn or _has_any(mood_words, ["calm", "reflection", "peace"]):
        return "static"
    return "kenburns"


def _motion_params_for_mode(mode: str, dur: float, mood_words: List[str]) -> Dict[str, Any]:
    zoom_start = 1.05
    zoom_end = 1.12
    strength = 0.9 if dur <= 4 else 0.75

    start_bias = {"x": 0.0, "y": 0.0}
    end_bias = {"x": 0.0, "y": 0.0}

    if mode == "static":
        zoom_start, zoom_end = 1.0, 1.0
        strength = 0.0
    elif mode == "kenburns":
        start_bias = {"x": 0.35, "y": 0.20}
        end_bias = {"x": -0.35, "y": -0.20}
    elif mode == "zoom_in":
        start_bias = {"x": 0.10, "y": 0.05}
        end_bias = {"x": -0.10, "y": -0.05}
        zoom_start, zoom_end = 1.03, 1.16
        strength = 0.85 if dur <= 5 else 0.75
    elif mode == "zoom_out":
        zoom_start, zoom_end = 1.14, 1.03
        strength = 0.7
    elif mode == "tilt_up":
        start_bias = {"x": 0.0, "y": -0.70}
        end_bias = {"x": 0.0, "y": 0.70}
        zoom_start, zoom_end = 1.04, 1.14
    elif mode == "tilt_down":
        start_bias = {"x": 0.0, "y": 0.70}
        end_bias = {"x": 0.0, "y": -0.70}
        zoom_start, zoom_end = 1.04, 1.10
    elif mode == "slide_left":
        start_bias = {"x": 0.75, "y": 0.0}
        end_bias = {"x": -0.75, "y": 0.0}
        zoom_start, zoom_end = 1.03, 1.10
        strength = 0.95 if dur <= 4 else 0.85
    elif mode == "slide_right":
        start_bias = {"x": -0.75, "y": 0.0}
        end_bias = {"x": 0.75, "y": 0.0}
        zoom_start, zoom_end = 1.03, 1.10
        strength = 0.95 if dur <= 4 else 0.85

    blur_amount = 35
    dim = 0.18
    if _has_any(mood_words, ["tension", "mystery", "horror", "threat", "danger"]):
        blur_amount = 45
        dim = 0.28
    elif _has_any(mood_words, ["sad", "regret", "defeat", "loss"]):
        blur_amount = 40
        dim = 0.24
    elif _has_any(mood_words, ["action", "panic", "fight", "chaos"]):
        blur_amount = 28
        dim = 0.14
    elif _has_any(mood_words, ["calm", "reflection", "peace"]):
        blur_amount = 22
        dim = 0.10

    return {
        "mode": mode,
        "strength": round(float(clamp(strength, 0.0, 1.0)), 3),
        "ease": "ease_in_out",
        "start_bias": {"x": round(float(start_bias["x"]), 3), "y": round(float(start_bias["y"]), 3)},
        "end_bias": {"x": round(float(end_bias["x"]), 3), "y": round(float(end_bias["y"]), 3)},
        "zoom": {
            "start": round(float(clamp(zoom_start, 1.0, 2.5)), 3),
            "end": round(float(clamp(zoom_end, 1.0, 2.5)), 3),
        },
        "bg_fill": {
            "mode": "blur",
            "enabled": True,
            "amount": int(clamp(blur_amount, 0, 80)),
            "dim": round(float(clamp(dim, 0.0, 0.9)), 3),
            "scale": "cover",
        },
        "fg_fit": {
            "mode": "contain",
            "safe_inset_pct": 0.06,
            "target_fg_coverage": 0.60,
        },
        "transition": {
            "in": {"type": "none", "dur_sec": 0.0},
            "out": {"type": "none", "dur_sec": 0.0},
        },
    }


def _camera_compat_from_motion(motion: Dict[str, Any], avoid_text_zoom: bool) -> Dict[str, Any]:
    mode = (motion.get("mode") or "kenburns").lower()
    if mode == "static":
        style = "static"
        max_zoom = 1.0
        pan = "center"
    else:
        style = "gentle_kenburns"
        z = motion.get("zoom") or {}
        max_zoom = float(z.get("end") or z.get("start") or 1.06)
        pan = "auto_center"

    return {
        "style": style,
        "avoid_text_zoom": bool(avoid_text_zoom),
        "max_zoom": round(float(clamp(max_zoom, 1.0, 2.5)), 3),
        "pan": pan,
        "motion_mode": mode,
        "start_bias": motion.get("start_bias") or {"x": 0.0, "y": 0.0},
        "end_bias": motion.get("end_bias") or {"x": 0.0, "y": 0.0},
        "zoom": motion.get("zoom") or {"start": 1.0, "end": 1.0},
        "bg_fill": motion.get("bg_fill") or {},
        "fg_fit": motion.get("fg_fit") or {},
    }


# -----------------------------
# Image resolving
# -----------------------------
def _resolve_img(scene_dir: str, f: str) -> str:
    if not f:
        return ""
    if os.path.isabs(f) and os.path.exists(f):
        return f
    cand = os.path.join(scene_dir, f)
    if os.path.exists(cand):
        return cand
    root, _ = os.path.splitext(cand)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        c2 = root + ext
        if os.path.exists(c2):
            return c2
    return cand


# -----------------------------
# Filtering heuristics
# -----------------------------
def _img_metrics_gray_256(path: str) -> Optional[Dict[str, float]]:
    if not path or not os.path.exists(path):
        return None
    if Image is None or ImageStat is None:
        return None

    try:
        img = Image.open(path).convert("L").resize((256, 256))
        stat = ImageStat.Stat(img)
        std = float(stat.stddev[0])

        px = list(img.getdata())
        n = 256 * 256

        # thresholds in 0..255 space
        white_cut = 245  # near-white
        black_cut = 12   # near-black

        w = sum(1 for v in px if v >= white_cut)
        b = sum(1 for v in px if v <= black_cut)
        white_ratio = w / float(n)
        black_ratio = b / float(n)
        dom_ratio = max(white_ratio, black_ratio)

        # bbox of "non-white" pixels (captures thin strip content)
        nonwhite = [(i % 256, i // 256) for i, v in enumerate(px) if v < 235]
        if nonwhite:
            ys = [p[1] for p in nonwhite]
            miny, maxy = min(ys), max(ys)
            bbox_h_frac = (maxy - miny + 1) / 256.0
        else:
            bbox_h_frac = 0.0

        # edge energy (cheap)
        edge_mean = 0.0
        if ImageFilter is not None:
            edges = img.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            edge_mean = float(edge_stat.mean[0]) / 255.0  # normalize 0..1

        return {
            "white_ratio": float(white_ratio),
            "black_ratio": float(black_ratio),
            "dom_ratio": float(dom_ratio),
            "std": float(std),
            "bbox_h_frac": float(bbox_h_frac),
            "edge_mean": float(edge_mean),
        }
    except Exception:
        return None


def is_bad_panel(
    path: str,
    *,
    # classic blank thresholds (your old logic)
    blank_dom_ratio_thr: float,
    blank_std_thr: float,
    # thin-strip rule
    strip_white_ratio_thr: float,
    strip_bbox_h_frac_thr: float,
    # bubble/sfx rule
    bubble_dom_ratio_thr: float,
    bubble_edge_mean_thr: float,
    bubble_std_thr: float,
) -> Tuple[bool, str, Optional[Dict[str, float]]]:
    m = _img_metrics_gray_256(path)
    if m is None:
        return (False, "", None)

    dom = m["dom_ratio"]
    std = m["std"]
    bbox_h = m["bbox_h_frac"]
    edge = m["edge_mean"]

    # 1) classic blank: dominated by white/black + low variance
    if dom >= float(blank_dom_ratio_thr) and std <= float(blank_std_thr):
        return (True, "blank_dom_lowstd", m)

    # 2) thin strip content: mostly white, with content only in a small band
    if m["white_ratio"] >= float(strip_white_ratio_thr) and bbox_h <= float(strip_bbox_h_frac_thr):
        return (True, "thin_strip_content", m)

    # 3) bubble/SFX dominated: mostly flat (white/black), low edges, low variance
    if dom >= float(bubble_dom_ratio_thr) and edge <= float(bubble_edge_mean_thr) and std <= float(bubble_std_thr):
        return (True, "bubble_sfx_dominant", m)

    return (False, "", m)


def filter_scene_files(
    *,
    files: List[str],
    clean_dir: str,
    raw_dir: str,
    prefer_clean: bool,
    # thresholds
    blank_dom_ratio_thr: float,
    blank_std_thr: float,
    strip_white_ratio_thr: float,
    strip_bbox_h_frac_thr: float,
    bubble_dom_ratio_thr: float,
    bubble_edge_mean_thr: float,
    bubble_std_thr: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    dropped: List[Dict[str, Any]] = []
    kept: List[str] = []

    def check_one(fbase: str, base_dir: str) -> Tuple[bool, str, Optional[Dict[str, float]]]:
        p = _resolve_img(base_dir, fbase)
        bad, reason, metrics = is_bad_panel(
            p,
            blank_dom_ratio_thr=blank_dom_ratio_thr,
            blank_std_thr=blank_std_thr,
            strip_white_ratio_thr=strip_white_ratio_thr,
            strip_bbox_h_frac_thr=strip_bbox_h_frac_thr,
            bubble_dom_ratio_thr=bubble_dom_ratio_thr,
            bubble_edge_mean_thr=bubble_edge_mean_thr,
            bubble_std_thr=bubble_std_thr,
        )
        return (not bad), reason, metrics

    for f in files:
        fbase = os.path.basename(str(f))
        if not fbase:
            continue

        if prefer_clean and clean_dir:
            ok, reason, metrics = check_one(fbase, clean_dir)
            if ok:
                kept.append(fbase)
            else:
                dropped.append({"file": fbase, "reason": f"drop_clean:{reason}", "metrics": metrics or {}})
        else:
            kept.append(fbase)

    # fallback: if everything dropped and raw exists, take raw equivalents but still filter (so raw junk doesn't leak)
    if (not kept) and raw_dir:
        for f in files:
            fbase = os.path.basename(str(f))
            if not fbase:
                continue
            rp = _resolve_img(raw_dir, fbase)
            if not os.path.exists(rp):
                continue
            ok, reason, metrics = check_one(fbase, raw_dir)
            if ok:
                kept.append(fbase)
                dropped.append({"file": fbase, "reason": "fallback_to_raw:kept", "metrics": metrics or {}})
            else:
                dropped.append({"file": fbase, "reason": f"fallback_to_raw:dropped:{reason}", "metrics": metrics or {}})

    return kept, dropped


# -----------------------------
# Cuts generation (NEW)
# -----------------------------
def build_cuts(
    scene_files: List[str],
    shot_dur: float,
    *,
    min_cut_sec: float,
    selection: Optional[List[Dict[str, Any]]] = None,
    protected: Optional["set"] = None,
) -> List[Dict[str, Any]]:
    """
    Create deterministic montage plan across scene_files:
      - Use up to K panels where K <= floor(shot_dur / min_cut_sec)
      - When *selection* (beat.scene_selection) is given, drop 'redundant' panels
        FIRST so the panels that remain get their >=min_cut_sec (instead of an
        arbitrary files[:k] truncation).
      - Split duration evenly across the kept panels
      - Emit floats (start/dur)
    """
    files = [os.path.basename(str(x)) for x in (scene_files or []) if x]
    if not files:
        return []

    shot_dur = float(shot_dur)
    if shot_dur <= 0.0:
        return []

    if len(files) == 1:
        return [{"file": files[0], "start": 0.0, "dur": shot_dur}]

    # Show EVERY distinct panel within shot_dur: drop only near-duplicate
    # (role=redundant) frames, NEVER truncate distinct panels to fit a short
    # narration. With no background music we pace the panels UNDER the voice (a
    # faster montage when a beat is panel-dense) instead of stretching into
    # silence; one panel + a long line is simply a long hold. min_cut_sec is no
    # longer a hard cap — the story-grouper keeps beats small enough to watch.
    if selection:
        # selection scene_file values are basenames (from beats); match in kind.
        sel = [{**e, "scene_file": os.path.basename(str(e.get("scene_file") or ""))}
               for e in selection]
        # show EVERY keeper (cap = the number of keepers, not a time budget); if
        # the whole shot is redundant, choose_kept_scenes falls back to the first
        # one so two near-identical frames never both play.
        roles = {e["scene_file"]: str(e.get("role") or "keep") for e in sel}
        prot = protected or set()
        n_keep = sum(1 for f in files
                     if roles.get(f, "keep") != "redundant" or f in prot)
        files = choose_kept_scenes(files, sel, max(1, n_keep), protected=protected)
    k = len(files)
    if k == 0:
        return []

    per = shot_dur / float(k)
    cuts: List[Dict[str, Any]] = []
    t = 0.0
    for i, f in enumerate(files):
        dur = per if i < k - 1 else (shot_dur - t)  # last cut absorbs rounding
        cuts.append({"file": f, "start": round(t, 3), "dur": round(float(dur), 3)})
        t += per
    return cuts


_FILLER_NARRATION_RE = re.compile(
    r"^\s*(the\s+(scene|story)\s+continues|to\s+be\s+continued|continues?)\.?\s*$",
    re.I)


def is_filler_narration(text: str) -> bool:
    """A beat that yielded no real story line — empty, or the
    'The scene continues.' placeholder the script stage emits when the beat
    narration was empty. Such a segment must not be voiced over a stand-in
    panel; the beat is dropped from the timeline instead."""
    t = (text or "").strip()
    return (not t) or bool(_FILLER_NARRATION_RE.match(t))


def protected_card_files(vision_path: str, scene_dirs: List[str]) -> "set":
    """Title/system cards (SKY CORPORATION., STARTING ACTIVATION.) — short
    mostly-uppercase phrases centered on a flat (white/black) frame. They are
    story beats and must survive the LLM's non-deterministic 'redundant'
    verdict and the husk filter. Mirrors prep_qa._is_title_card; the flat-frame
    test separates a real card from caps dialogue/SFX on textured art."""
    out: set = set()
    if not vision_path or not os.path.exists(vision_path):
        return out
    try:
        with open(vision_path, "r", encoding="utf-8") as fh:
            items = json.load(fh).get("items") or []
    except Exception:
        return out
    # the pipeline doesn't pass the scene dirs to the planner — fall back to the
    # sibling scenes/ next to the vision manifest so the flat-frame test can run
    dirs = [d for d in (scene_dirs or []) if d]
    dirs.append(os.path.join(os.path.dirname(os.path.abspath(vision_path)), "scenes"))
    import re as _re
    import cv2  # lazy — keeps timeline_planner importable without the CV stack
    for it in items:
        f = os.path.basename(str(it.get("scene_file") or ""))
        ocr = str(it.get("ocr_clean") or "").strip()
        if not f or it.get("text_only") or "..." in ocr \
                or any(ch in ocr for ch in "~!?"):
            continue
        words = [w for w in _re.split(r"[^A-Za-z0-9']+", ocr)
                 if any(c.isalpha() for c in w)]
        letters = [c for c in ocr if c.isalpha()]
        if not (2 <= len(words) <= 8) or not letters:
            continue
        if sum(c.isupper() for c in letters) / len(letters) < 0.8:
            continue
        if float(it.get("text_coverage") or 0.0) >= 0.20:
            continue
        img = None
        for d in dirs:
            p = os.path.join(d, f)
            if os.path.exists(p):
                img = cv2.imread(p)
                break
        if img is None:
            continue
        g = img.mean(axis=2)
        if float(((g > 235) | (g < 25)).mean()) >= 0.6:
            out.add(f)
    return out


def protected_story_files(vision_path: str) -> "set":
    """Every panel the UNDERSTANDING calls real story content (panel_kind=='story',
    stamped on the vision manifest by panel_understand). These OUTRANK the beats
    LLM's per-panel 'redundant' verdict — which proved unreliable, dropping the
    very panel that named ORV's whole premise (the phone showing 'Three Ways to
    Survive the Apocalypse'). The understanding is authoritative: a real story
    panel is always SHOWN; only effect/empty/caption frames (already filtered at
    grouping) or true near-duplicates may be dropped. Degrades to the empty set
    (card-only behaviour) when panel_kind isn't stamped (an older manifest)."""
    out: set = set()
    if not vision_path or not os.path.exists(vision_path):
        return out
    try:
        with open(vision_path, "r", encoding="utf-8") as fh:
            items = json.load(fh).get("items") or []
    except Exception:
        return out
    for it in items:
        f = os.path.basename(str(it.get("scene_file") or ""))
        if f and str(it.get("panel_kind") or "").strip().lower() == "story":
            out.add(f)
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True, help="manifest.groups.json")
    ap.add_argument("--beats", default="", help="manifest.beats.json (optional)")
    ap.add_argument("--script", default="", help="manifest.script.json (preferred)")
    ap.add_argument("--vision", default="", help="manifest.vision.json (optional)")
    ap.add_argument("--tts-index", default="", help="tts_sections/tts_index.json (optional)")
    ap.add_argument("--out", required=True, help="render.plan.json")

    ap.add_argument("--mode", choices=["no_narration", "narrated"], default="narrated")

    ap.add_argument("--base-min-sec", type=float, default=2.5)
    ap.add_argument("--max-sec", type=float, default=25.0)
    ap.add_argument("--chars-per-sec", type=float, default=18.0)
    ap.add_argument("--audio-pad-sec", type=float, default=0.20)

    ap.add_argument("--default-display", choices=["auto", "single_hold", "multi_cut"], default="auto")

    ap.add_argument("--clean-scene-dir", default="", help="Directory of cleaned panels (bubble-removed)")
    ap.add_argument("--raw-scene-dir", default="", help="Optional fallback directory (raw panels)")
    ap.add_argument("--prefer-clean", action="store_true", help="Filter bad panels from clean dir and fallback to raw")

    # montage cut pacing
    ap.add_argument("--min-cut-sec", type=float, default=2.0, help="Minimum time per panel cut in montage")

    # Filtering thresholds (sane defaults for your observed failure cases)
    ap.add_argument("--blank-dom-ratio", type=float, default=0.975)
    ap.add_argument("--blank-std-thr", type=float, default=6.0)

    ap.add_argument("--strip-white-ratio", type=float, default=0.82)
    ap.add_argument("--strip-bbox-h-frac", type=float, default=0.25)

    ap.add_argument("--bubble-dom-ratio", type=float, default=0.88)
    ap.add_argument("--bubble-edge-mean", type=float, default=0.055)
    ap.add_argument("--bubble-std-thr", type=float, default=14.0)

    args = ap.parse_args()

    groups_obj = load_json(args.groups)
    groups = find_groups(groups_obj)

    beats_by_gid: Dict[int, Dict[str, Any]] = {}
    if args.beats:
        beats_by_gid = index_beats(load_json(args.beats))

    script_by_gid: Dict[str, Dict[str, Any]] = {}
    if args.script:
        script_by_gid = index_script(load_json(args.script))

    tts_by_gid: Dict[str, Dict[str, Any]] = {}
    if args.tts_index:
        tts_obj = load_json(args.tts_index)
        tts_by_gid = _index_tts(tts_obj, args.tts_index)

    # B2: group the per-paragraph script rows (keyed by segment_id g####_p##) by
    # group_id, in paragraph order, so the timeline emits ONE item per paragraph.
    # Without this, a group with multiple narration paragraphs collapses to a
    # single item and every paragraph's audio but the last is silently dropped.
    segments_by_group: Dict[int, List[Any]] = {}
    for _sid, _srow in script_by_gid.items():
        _gid = int(_srow.get("group_id") or 0)
        segments_by_group.setdefault(_gid, []).append((_sid, _srow))
    for _gid in segments_by_group:
        segments_by_group[_gid].sort(key=lambda t: t[0])  # g0001_p00 < g0001_p01

    timeline: List[Dict[str, Any]] = []
    time_cursor = 0.0
    dropped_summary: List[Dict[str, Any]] = []

    # title/system cards are mandatory story beats — protect them from the
    # husk filter and the LLM's 'redundant' verdict (which is non-deterministic;
    # SKY CORPORATION was kept on one host, dropped on another)
    protected_cards = protected_card_files(
        args.vision, [args.clean_scene_dir, args.raw_scene_dir])
    if protected_cards:
        print(f"[plan] protected title/system cards: {sorted(protected_cards)}")
    # the understanding's verdict OUTRANKS the beats LLM's 'redundant' tag: every
    # real story panel is shown (the LLM dropped the premise panel otherwise). Pure
    # effects/empties are already gone (filtered at grouping); captions stay
    # droppable (their words ride the narration). Union drives both filters below.
    protected_story = protected_story_files(args.vision)
    protected = protected_cards | protected_story
    if protected_story:
        print(f"[plan] protected {len(protected_story)} understood story panel(s) "
              f"from the redundant-drop")

    for gobj in groups:
        group_id = int(gobj.get("group_id") or gobj.get("shot_id") or 0)
        shot_id = int(gobj.get("shot_id") or group_id or 0)

        scene_files = gobj.get("scene_files") or []
        if not isinstance(scene_files, list):
            scene_files = []
        scene_files = [str(x) for x in scene_files if x]

        # Filter (clean preferred)
        if args.prefer_clean and args.clean_scene_dir and scene_files:
            kept, dropped = filter_scene_files(
                files=scene_files,
                clean_dir=args.clean_scene_dir,
                raw_dir=args.raw_scene_dir,
                prefer_clean=True,
                blank_dom_ratio_thr=float(args.blank_dom_ratio),
                blank_std_thr=float(args.blank_std_thr),
                strip_white_ratio_thr=float(args.strip_white_ratio),
                strip_bbox_h_frac_thr=float(args.strip_bbox_h_frac),
                bubble_dom_ratio_thr=float(args.bubble_dom_ratio),
                bubble_edge_mean_thr=float(args.bubble_edge_mean),
                bubble_std_thr=float(args.bubble_std_thr),
            )
            if dropped:
                dropped_summary.append({"group_id": group_id, "dropped": dropped})
            # never let the husk filter drop a mandatory card OR a real story panel
            keep_set = set(kept) | (protected & set(scene_files))
            scene_files = [f for f in scene_files if f in keep_set]  # orig order

        beat = beats_by_gid.get(group_id, {"group_id": group_id})
        mood_words = _norm_words(beat.get("mood_words") or [])
        rh = beat.get("rendering_hints") or {}
        avoid_text_zoom = bool(rh.get("avoid_text_zoom", False))

        # B2: a group may have several narration paragraphs (segment_id g####_p##).
        # Emit ONE timeline item per paragraph so each paragraph keeps its own
        # audio + timing. Fall back to a single group-level item when no
        # per-paragraph script rows exist (back-compat with old manifests).
        segments = segments_by_group.get(group_id) or []
        if not segments:
            fallback_sid = _safe_str(gobj.get("segment_id")) or f"g{group_id:04d}"
            fallback_srow = script_by_gid.get(fallback_sid) or script_by_gid.get(f"g{group_id:04d}")
            segments = [(fallback_sid, fallback_srow)]

        for segment_id, srow in segments:
            paragraph = _safe_str(srow.get("paragraph")) if srow else ""
            delivery_tag = _safe_str(srow.get("delivery_tag")) if srow else ""

            # a degenerate beat (empty narration → "The scene continues."
            # placeholder) is dropped: never voice filler over a stand-in panel
            if args.mode == "narrated" and is_filler_narration(paragraph):
                dropped_summary.append({"group_id": group_id,
                                        "dropped_filler_segment": segment_id})
                continue

            tts_text = paragraph if args.mode == "narrated" else ""
            if delivery_tag and tts_text:
                tts_text = f"[{delivery_tag}] {tts_text}"

            overlays: List[Dict[str, Any]] = []

            audio_duration = 0.0
            tts_audio_path = ""
            if args.mode == "narrated":
                tts_row = tts_by_gid.get(segment_id) or tts_by_gid.get(f"g{group_id:04d}")
                if tts_row:
                    audio_duration = float(tts_row.get("duration_sec") or 0.0)
                    tts_audio_path = _safe_str(tts_row.get("audio_path") or "")
                    if (audio_duration <= 0.0) and tts_audio_path.lower().endswith(".wav") and os.path.exists(tts_audio_path):
                        try:
                            audio_duration = _wav_duration_sec(tts_audio_path)
                        except Exception:
                            audio_duration = 0.0

            dur = compute_duration_sec(
                mode=args.mode,
                tts_text=tts_text,
                overlays=overlays,
                base_min=args.base_min_sec,
                max_sec=args.max_sec,
                chars_per_sec=args.chars_per_sec,
                audio_duration_sec=audio_duration,
                audio_pad_sec=args.audio_pad_sec,
            )

            motion_mode = _choose_motion_mode(beat)
            motion = _motion_params_for_mode(motion_mode, dur, mood_words)
            camera = _camera_compat_from_motion(motion, avoid_text_zoom=avoid_text_zoom)

            # display strategy
            if args.default_display == "single_hold":
                display_strategy = "single_hold"
            elif args.default_display == "multi_cut":
                display_strategy = "multi_cut"
            else:
                display_strategy = "multi_cut"

            # If no usable files: keep shot (audio still plays) but no cuts
            primary_scene_file = scene_files[0] if scene_files else ""

            # Cuts drive the montage. COVERAGE: build_cuts shows EVERY distinct
            # panel within `dur` (drops only near-duplicate frames) — no panel is
            # truncated to fit a short line, and with no music we never stretch
            # into silence. One panel + a long line = a long hold.
            cuts: List[Dict[str, Any]] = []
            if scene_files:
                if display_strategy == "single_hold":
                    cuts = build_cuts([primary_scene_file], dur,
                                      min_cut_sec=float(args.min_cut_sec))
                else:
                    cuts = build_cuts(
                        scene_files, dur,
                        min_cut_sec=float(args.min_cut_sec),
                        selection=beat.get("scene_selection"),
                        protected=protected,
                    )

            item: Dict[str, Any] = {
                "segment_id": segment_id,
                "group_id": group_id,
                "shot_id": shot_id,
                # story-structure tag from story_group (present|flashback|dream);
                # the renderer applies a flashback look when != present.
                "segment": str(gobj.get("segment") or "present"),
                "display_strategy": display_strategy,
                "primary_scene_file": primary_scene_file,
                "group_scene_files": scene_files,
                "scene_files": scene_files if display_strategy == "multi_cut" else ([primary_scene_file] if primary_scene_file else []),

                # NEW: montage plan for Blender
                "cuts": cuts,

                "start_sec": round(time_cursor, 3),
                "duration_sec": round(float(dur), 3),
                "end_sec": round(time_cursor + float(dur), 3),

                "camera": camera,
                "motion": motion,
                "overlays": overlays,
                "tts_text": tts_text.strip(),
                "tts_audio": tts_audio_path,
                "tts_audio_duration_sec": round(float(audio_duration), 3) if audio_duration else 0.0,
                "rendering_hints": {
                    "avoid_text_zoom": avoid_text_zoom,
                    "preferred_focus": rh.get("preferred_focus") or "",
                    "camera_motion": rh.get("camera_motion") or "",
                    "target_phrases": rh.get("target_phrases") if isinstance(rh.get("target_phrases"), list) else [],
                },
                "tags": {
                    "mood_words": beat.get("mood_words") or [],
                    "emotional_turn": beat.get("emotional_turn") or "",
                    "duration_source": "audio" if (args.mode == "narrated" and audio_duration > 0.0) else "estimate",
                    "filtered_blank_panels": True if args.prefer_clean else False,
                },
            }

            timeline.append(item)
            time_cursor += float(dur)

    out_obj = {
        "source_groups": os.path.abspath(args.groups),
        "source_beats": os.path.abspath(args.beats) if args.beats else "",
        "source_script": os.path.abspath(args.script) if args.script else "",
        "source_tts_index": os.path.abspath(args.tts_index) if args.tts_index else "",
        "mode": args.mode,
        "timing": {
            "base_min_sec": args.base_min_sec,
            "max_sec": args.max_sec,
            "chars_per_sec": args.chars_per_sec,
            "audio_pad_sec": args.audio_pad_sec,
            "default_display": args.default_display,
            "min_cut_sec": args.min_cut_sec,
        },
        "panel_filtering": {
            "prefer_clean": bool(args.prefer_clean),
            "clean_scene_dir": os.path.abspath(args.clean_scene_dir) if args.clean_scene_dir else "",
            "raw_scene_dir": os.path.abspath(args.raw_scene_dir) if args.raw_scene_dir else "",
            "blank_dom_ratio": args.blank_dom_ratio,
            "blank_std_thr": args.blank_std_thr,
            "strip_white_ratio": args.strip_white_ratio,
            "strip_bbox_h_frac": args.strip_bbox_h_frac,
            "bubble_dom_ratio": args.bubble_dom_ratio,
            "bubble_edge_mean": args.bubble_edge_mean,
            "bubble_std_thr": args.bubble_std_thr,
            "dropped": dropped_summary,
        },
        "total_duration_sec": round(float(time_cursor), 3),
        "timeline": timeline,
    }

    dump_json(args.out, out_obj)
    print(
        f"[ok] wrote={args.out} items={len(timeline)} total_sec={out_obj['total_duration_sec']} "
        f"dropped_groups={len(dropped_summary)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
