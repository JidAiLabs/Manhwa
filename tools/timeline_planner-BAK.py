#!/usr/bin/env python3
"""
timeline_planner.py (Audio drives length + montage cuts for Blender)

Key behavior:
- If --tts-index is provided and contains per-group audio durations, we set
  timeline item duration_sec from audio duration (+ pad).
- Correctly reads tts_index.json produced by elevenlabs_tts_from_manifest.py:
    { "clips": [ { "group_id", "audio_file", "duration_sec", ... } ] }
- Uses FLOAT durations (not int) for accurate A/V sync.
- Adds a Blender-friendly per-group montage plan:
    item["cuts"] = [{file, start, dur}, ...]
- Keeps your motion/camera/vision phrase targeting logic.
"""

import argparse
import json
import os
import re
import math
import wave
from typing import Any, Dict, List, Tuple, Optional


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


def index_script(script_obj: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Creates gid -> paragraph row.
    Uses pairing: section.script_paragraphs[i] with section.shots[i].
    """
    out: Dict[int, Dict[str, Any]] = {}
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

            p = paras[i]
            if isinstance(p, dict):
                paragraph_text = _safe_str(p.get("text"))
                delivery_tag = _safe_str(p.get("delivery_tag"))
            else:
                paragraph_text = _safe_str(p)
                delivery_tag = ""

            out[gid] = {
                "group_id": gid,
                "section_index": sec_idx,
                "beat_id": int(s.get("beat_id") or i),
                "paragraph": paragraph_text,
                "delivery_tag": delivery_tag,
                "cliffhanger_line": _safe_str(sec.get("cliffhanger_line")),
            }

    return out


# -----------------------------
# TTS index (Audio durations)
# -----------------------------
def _index_tts(tts_obj: Dict[str, Any], tts_index_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Supports elevenlabs_tts_from_manifest.py output:
      tts_index.json -> { "clips": [ { group_id, audio_file, duration_sec, ... } ] }
    Returns map: group_id -> {audio_path, duration_sec, text, ...}
    """
    out: Dict[int, Dict[str, Any]] = {}

    items = tts_obj.get("clips") or tts_obj.get("items") or []
    if not isinstance(items, list):
        return out

    base_dir = os.path.dirname(os.path.abspath(tts_index_path))

    for it in items:
        if not isinstance(it, dict):
            continue

        gid = int(it.get("group_id") or 0)
        if gid <= 0:
            continue

        audio_file = _safe_str(it.get("audio_file") or it.get("audio_path") or it.get("path") or "")
        audio_path = audio_file
        if audio_path and not os.path.isabs(audio_path):
            audio_path = os.path.normpath(os.path.join(base_dir, audio_path))

        dur = it.get("duration_sec")
        try:
            dur_f = float(dur) if dur is not None else 0.0
        except Exception:
            dur_f = 0.0

        out[gid] = {
            "group_id": gid,
            "audio_path": audio_path,
            "duration_sec": dur_f,
            "text": _safe_str(it.get("sent_text") or it.get("source_text") or it.get("text") or ""),
            "voice_id": _safe_str(tts_obj.get("voice_id") or ""),
            "model_id": _safe_str(tts_obj.get("model_id") or ""),
        }

    return out


def _wav_duration_sec(path: str) -> float:
    """Only for WAV. For mp3, prefer storing duration in the tts index."""
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return float(frames) / float(rate)


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
    """
    Priority:
      If audio_duration_sec > 0 => duration = audio_duration_sec + audio_pad_sec (clamped).
      Else fallback to char-based estimate.
    Returns FLOAT seconds (editor-grade).
    """
    base = float(base_min)

    if mode == "narrated" and audio_duration_sec and audio_duration_sec > 0.0:
        dur = float(audio_duration_sec) + float(audio_pad_sec)
        return float(clamp(dur, base_min, max_sec))

    overlay_chars = sum(text_len(o.get("text")) for o in overlays if isinstance(o, dict))
    narr_chars = text_len(tts_text) if mode == "narrated" else 0
    total_chars = overlay_chars + narr_chars

    reading = total_chars / float(chars_per_sec) if chars_per_sec > 0 else 0.0
    dur = base + reading
    return float(clamp(dur, base_min, max_sec))


# -----------------------------
# Montage cuts (Blender-friendly)
# -----------------------------
def build_cuts(scene_files: List[str], total_dur: float, display_strategy: str) -> List[Dict[str, Any]]:
    """
    Returns:
      [{ "file": "...", "start": 0.0, "dur": 2.345 }, ...]
    start is relative to the group's start (0..duration).
    """
    if not scene_files:
        return []

    total_dur = float(max(0.0, total_dur))

    if display_strategy == "single_hold" or len(scene_files) == 1:
        return [{"file": scene_files[0], "start": 0.0, "dur": round(total_dur, 3)}]

    n = len(scene_files)

    # Weighting: first/last slightly longer, middle evenly split
    if n == 2:
        weights = [0.55, 0.45]
    elif n == 3:
        weights = [0.40, 0.30, 0.30]
    else:
        mid = n - 2
        weights = [0.22] + ([0.56 / max(1, mid)] * mid) + [0.22]

    s = sum(weights) if weights else 1.0
    weights = [w / s for w in weights]

    cuts: List[Dict[str, Any]] = []
    t = 0.0
    for i, f in enumerate(scene_files):
        d = total_dur * weights[i]
        cuts.append({"file": f, "start": round(t, 3), "dur": round(d, 3)})
        t += d

    # Fix rounding drift
    if cuts:
        drift = total_dur - sum(c["dur"] for c in cuts)
        cuts[-1]["dur"] = round(cuts[-1]["dur"] + drift, 3)

    return cuts


# -----------------------------
# Motion selection (fallback)
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


def _motion_params_for_mode(mode: str, dur: float, mood_words: List[str], avoid_text_zoom: bool) -> Dict[str, Any]:
    """
    avoid_text_zoom influences zoom caps so stat/system screens remain readable.
    """
    dur = float(max(0.0, dur))

    zoom_start = 1.05
    zoom_end = 1.12
    strength = 0.9 if dur <= 4.0 else 0.75

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
        strength = 0.85 if dur <= 5.0 else 0.75
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
        strength = 0.95 if dur <= 4.0 else 0.85
    elif mode == "slide_right":
        start_bias = {"x": -0.75, "y": 0.0}
        end_bias = {"x": 0.75, "y": 0.0}
        zoom_start, zoom_end = 1.03, 1.10
        strength = 0.95 if dur <= 4.0 else 0.85

    # Text safety: clamp zoom if avoid_text_zoom
    if avoid_text_zoom:
        zoom_end = min(zoom_end, 1.06)
        zoom_start = min(zoom_start, 1.03)

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

    # extra text safety
    if avoid_text_zoom:
        max_zoom = min(max_zoom, 1.06)

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
# Vision phrase targeting
# -----------------------------
def _bbox_center(bb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bb
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_expand(bb: Tuple[float, float, float, float], pad: float = 0.02) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bb
    return (
        clamp(x0 - pad, 0.0, 1.0),
        clamp(y0 - pad, 0.0, 1.0),
        clamp(x1 + pad, 0.0, 1.0),
        clamp(y1 + pad, 0.0, 1.0),
    )


def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_scene_id_from_filename(path_or_name: str) -> int:
    base = os.path.basename(path_or_name or "")
    m = re.search(r"(\d+)", base)
    return int(m.group(1)) if m else 0


def find_phrase_bbox(vision_item: Dict[str, Any], phrase: str) -> Tuple[float, float, float, float]:
    phrase_n = _norm_text(phrase)

    v = vision_item.get("vision") or {}
    words = v.get("ocr_words") or []
    if not isinstance(words, list) or not words:
        for t in (vision_item.get("targets") or []):
            if t.get("type") == "text_block":
                bb = t.get("bbox")
                if isinstance(bb, list) and len(bb) == 4:
                    return tuple(map(float, bb))
        return (0.0, 0.0, 1.0, 1.0)

    toks: List[str] = []
    boxes: List[List[float]] = []
    for w in words:
        txt = _safe_str(w.get("t"))
        bb = w.get("bbox")
        if not txt or not (isinstance(bb, list) and len(bb) == 4):
            continue
        toks.append(_norm_text(txt))
        boxes.append([float(x) for x in bb])

    max_window = 8
    best: Optional[Tuple[float, float, float, float]] = None
    best_len = 0

    for i in range(len(toks)):
        acc = ""
        for j in range(i, min(len(toks), i + max_window)):
            acc = (acc + " " + toks[j]).strip()

            if acc == phrase_n:
                xs0, ys0, xs1, ys1 = 1.0, 1.0, 0.0, 0.0
                for k in range(i, j + 1):
                    x0, y0, x1, y1 = boxes[k]
                    xs0, ys0, xs1, ys1 = min(xs0, x0), min(ys0, y0), max(xs1, x1), max(ys1, y1)
                return (xs0, ys0, xs1, ys1)

            if phrase_n and phrase_n in acc and len(acc) > best_len:
                best_len = len(acc)
                xs0, ys0, xs1, ys1 = 1.0, 1.0, 0.0, 0.0
                for k in range(i, j + 1):
                    x0, y0, x1, y1 = boxes[k]
                    xs0, ys0, xs1, ys1 = min(xs0, x0), min(ys0, y0), max(xs1, x1), max(ys1, y1)
                best = (xs0, ys0, xs1, ys1)

    if best:
        return best

    for t in (vision_item.get("targets") or []):
        if t.get("type") == "text_block":
            bb = t.get("bbox")
            if isinstance(bb, list) and len(bb) == 4:
                return tuple(map(float, bb))

    return (0.0, 0.0, 1.0, 1.0)


def build_camera_path_from_phrases(vision_item: Dict[str, Any], phrases: List[str], avoid_text_zoom: bool) -> Dict[str, Any]:
    bbs: List[Tuple[float, float, float, float]] = []
    for ph in phrases:
        if not ph.strip():
            continue
        bb = find_phrase_bbox(vision_item, ph)
        bb = _bbox_expand(bb, pad=0.02)
        bbs.append(bb)

    uniq: List[Tuple[float, float, float, float]] = []
    for bb in bbs:
        cx, cy = _bbox_center(bb)
        if not uniq:
            uniq.append(bb)
            continue
        pcx, pcy = _bbox_center(uniq[-1])
        if abs(cx - pcx) + abs(cy - pcy) > 0.03:
            uniq.append(bb)

    if not uniq:
        return {}

    def zoom_for(bb: Tuple[float, float, float, float]) -> float:
        x0, y0, x1, y1 = bb
        bw = max(1e-6, x1 - x0)
        bh = max(1e-6, y1 - y0)
        z = 0.70 / max(bw, bh)
        if avoid_text_zoom:
            return float(clamp(z, 1.02, 1.08))
        return float(clamp(z, 1.02, 1.18))

    kfs: List[Dict[str, Any]] = []
    n = len(uniq)
    for i, bb in enumerate(uniq):
        cx, cy = _bbox_center(bb)
        t = 0.10 + (0.80 * (i / (max(1, n - 1))))
        z = zoom_for(bb)
        kfs.append({"t": round(t, 3), "cx": round(cx, 4), "cy": round(cy, 4), "zoom": round(z, 3)})
        kfs.append({"t": round(min(0.98, t + 0.12), 3), "cx": round(cx, 4), "cy": round(cy, 4), "zoom": round(z, 3)})

    kfs[0]["t"] = 0.0
    kfs[-1]["t"] = 1.0

    return {"space": "norm", "keyframes": kfs}


def extract_phrases_from_rendering_hints(rh: Dict[str, Any]) -> List[str]:
    tp = rh.get("target_phrases")
    if isinstance(tp, list):
        out = []
        for x in tp:
            s = str(x).strip()
            if s:
                out.append(s)
        return out[:3]

    cm = _safe_str(rh.get("camera_motion"))
    p1 = re.findall(r"'([^']+)'+", cm)
    if p1:
        return [x.strip() for x in p1 if x.strip()][:3]
    p2 = re.findall(r"\"([^\"]+)\"+", cm)
    return [x.strip() for x in p2 if x.strip()][:3]


def _is_text_heavy(shot: Dict[str, Any], beat: Dict[str, Any]) -> bool:
    for k in ("text_only", "is_text_only", "pure_text", "text_heavy"):
        if isinstance(shot.get(k), bool) and shot.get(k):
            return True
        if isinstance(beat.get(k), bool) and beat.get(k):
            return True

    rh = beat.get("rendering_hints") or {}
    hint = _safe_str(rh.get("panel_type")).lower()
    if hint in ("text_only", "bubble_only", "narration_only", "system_window", "stat_window"):
        return True

    mw = _norm_words(beat.get("mood_words") or [])
    if _has_any(mw, ["monologue", "exposition"]) and not _has_any(mw, ["action", "fight", "chase"]):
        return True

    return False


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True, help="manifest.groups.json")
    ap.add_argument("--beats", default="", help="manifest.beats.json (optional)")
    ap.add_argument("--script", default="", help="manifest.script.json (preferred)")
    ap.add_argument("--vision", default="", help="manifest.vision.json (optional; enables camera_path)")
    ap.add_argument("--tts-index", default="", help="tts_sections/tts_index.json (optional; enables audio durations)")
    ap.add_argument("--out", required=True, help="render.plan.json")

    ap.add_argument("--mode", choices=["no_narration", "narrated"], default="narrated")

    ap.add_argument("--base-min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=25.0)
    ap.add_argument("--chars-per-sec", type=float, default=18.0)
    ap.add_argument("--audio-pad-sec", type=float, default=0.20)

    ap.add_argument("--default-display", choices=["auto", "single_hold", "multi_cut"], default="auto")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-overlays", action="store_true")
    g.add_argument("--with-overlays", action="store_true")
    args = ap.parse_args()

    overlays_enabled = bool(args.with_overlays)

    groups_obj = load_json(args.groups)
    groups = find_groups(groups_obj)

    beats_by_gid: Dict[int, Dict[str, Any]] = {}
    if args.beats:
        beats_by_gid = index_beats(load_json(args.beats))

    script_by_gid: Dict[int, Dict[str, Any]] = {}
    if args.script:
        script_by_gid = index_script(load_json(args.script))

    vision_by_scene_id: Dict[int, Dict[str, Any]] = {}
    if args.vision:
        vobj = load_json(args.vision)
        for it in (vobj.get("items") or []):
            sid = int(it.get("scene_id") or 0)
            if sid > 0:
                vision_by_scene_id[sid] = it

    tts_by_gid: Dict[int, Dict[str, Any]] = {}
    if args.tts_index:
        tts_obj = load_json(args.tts_index)
        tts_by_gid = _index_tts(tts_obj, args.tts_index)

    timeline: List[Dict[str, Any]] = []
    time_cursor = 0.0

    for gobj in groups:
        group_id = int(gobj.get("group_id") or gobj.get("shot_id") or 0)
        shot_id = int(gobj.get("shot_id") or group_id or 0)

        scene_files = gobj.get("scene_files") or []
        if not isinstance(scene_files, list):
            scene_files = []
        scene_files = [str(x) for x in scene_files if x]

        beat = beats_by_gid.get(group_id, {"group_id": group_id})
        srow = script_by_gid.get(group_id)

        paragraph = _safe_str(srow.get("paragraph")) if srow else ""
        delivery_tag = _safe_str(srow.get("delivery_tag")) if srow else ""
        cliffhanger = _safe_str(srow.get("cliffhanger_line")) if srow else ""

        # TTS text (what voice was generated from)
        tts_text = paragraph if args.mode == "narrated" else ""
        if delivery_tag and tts_text:
            tts_text = f"[{delivery_tag}] {tts_text}"

        overlays: List[Dict[str, Any]] = []
        # (you can add overlays later; keep empty for now)

        # Audio duration lookup
        audio_duration = 0.0
        tts_audio_path = ""
        if args.mode == "narrated":
            tts_row = tts_by_gid.get(group_id)
            if tts_row:
                audio_duration = float(tts_row.get("duration_sec") or 0.0)
                tts_audio_path = _safe_str(tts_row.get("audio_path") or "")

                # WAV fallback duration if needed
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

        mood_words = _norm_words(beat.get("mood_words") or [])
        rh = beat.get("rendering_hints") or {}
        avoid_text_zoom = bool(rh.get("avoid_text_zoom", False))

        # Display strategy selection
        text_heavy = _is_text_heavy(gobj, beat)
        if args.default_display == "single_hold":
            display_strategy = "single_hold"
        elif args.default_display == "multi_cut":
            display_strategy = "multi_cut"
        else:
            # Auto: text-heavy => hold, else montage
            display_strategy = "single_hold" if text_heavy else "multi_cut"

        # Force text safety: text-heavy should avoid zoom
        if text_heavy:
            avoid_text_zoom = True

        primary_scene_file = scene_files[0] if scene_files else ""

        # Motion + camera
        motion_mode = "static" if text_heavy else _choose_motion_mode(beat)
        motion = _motion_params_for_mode(motion_mode, dur, mood_words, avoid_text_zoom=avoid_text_zoom)
        camera = _camera_compat_from_motion(motion, avoid_text_zoom=avoid_text_zoom)

        # Vision camera_path (optional phrase targeting)
        camera_path: Dict[str, Any] = {}
        if args.vision and primary_scene_file:
            sid = _parse_scene_id_from_filename(primary_scene_file)
            vitem = vision_by_scene_id.get(sid)
            phrases = extract_phrases_from_rendering_hints(rh)
            if vitem and phrases:
                camera_path = build_camera_path_from_phrases(vitem, phrases, avoid_text_zoom=avoid_text_zoom)

        # Montage cuts for Blender
        cuts = build_cuts(scene_files, float(dur), display_strategy)

        item: Dict[str, Any] = {
            "group_id": group_id,
            "shot_id": shot_id,

            "display_strategy": display_strategy,
            "primary_scene_file": primary_scene_file,
            "group_scene_files": scene_files,

            # Blender-friendly montage plan
            "cuts": cuts,

            # convenience list (same as cuts order)
            "scene_files": [c["file"] for c in cuts] if cuts else ([] if not primary_scene_file else [primary_scene_file]),

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
                "avoid_text_zoom": bool(avoid_text_zoom),
                "preferred_focus": rh.get("preferred_focus") or "",
                "camera_motion": rh.get("camera_motion") or "",
                "target_phrases": rh.get("target_phrases") if isinstance(rh.get("target_phrases"), list) else [],
            },

            "tags": {
                "text_heavy": bool(text_heavy),
                "mood_words": beat.get("mood_words") or [],
                "emotional_turn": beat.get("emotional_turn") or "",
                "camera_motion_hint": (beat.get("rendering_hints") or {}).get("camera_motion") or "",
                "duration_source": "audio" if (args.mode == "narrated" and audio_duration > 0.0) else "estimate",
                "cliffhanger_line": cliffhanger,
            },
        }

        if camera_path:
            item["camera_path"] = camera_path

        timeline.append(item)
        time_cursor += float(dur)

    out_obj = {
        "source_groups": os.path.abspath(args.groups),
        "source_beats": os.path.abspath(args.beats) if args.beats else "",
        "source_script": os.path.abspath(args.script) if args.script else "",
        "source_vision": os.path.abspath(args.vision) if args.vision else "",
        "source_tts_index": os.path.abspath(args.tts_index) if args.tts_index else "",
        "mode": args.mode,
        "timing": {
            "base_min_sec": args.base_min_sec,
            "max_sec": args.max_sec,
            "chars_per_sec": args.chars_per_sec,
            "audio_pad_sec": args.audio_pad_sec,
            "overlays_enabled": overlays_enabled,
            "default_display": args.default_display,
        },
        "total_duration_sec": round(float(time_cursor), 3),
        "timeline": timeline,
    }

    dump_json(args.out, out_obj)
    print(f"[ok] wrote={args.out} items={len(timeline)} total_sec={out_obj['total_duration_sec']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
