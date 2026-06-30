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
from scene_chrome import is_chrome_scene  # noqa: E402

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


def _scene_file_basenames(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    seen = set()
    for x in items:
        f = os.path.basename(str(x or "").strip())
        if f and f not in seen:
            out.append(f)
            seen.add(f)
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
                "scene_files": _scene_file_basenames(s.get("scene_files") or []),
                "fallback_scene_files": _scene_file_basenames(s.get("fallback_scene_files") or []),
                "avoid_text_zoom": bool(s.get("avoid_text_zoom", True)),
                "camera": _safe_str(s.get("camera")),
                "focus": _safe_str(s.get("focus")),
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
# Align manifest index
# -----------------------------
def _load_align_index(align_path: str, base_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load manifest.align.json and return a dict keyed by GROUP id (g####)
    whose value is the ordered list of per-panel align entries for that group.

    Each entry: {segment_id, group_clip (abs path), start_sec, end_sec, method}.
    Panels within a group are sorted by segment_id (g####_p00 < g####_p01).
    Returns {} when the file is absent or malformed.
    """
    if not align_path or not os.path.exists(align_path):
        return {}
    try:
        obj = load_json(align_path)
    except Exception:
        return {}
    segments = obj.get("segments") or []
    if not isinstance(segments, list):
        return {}

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for entry in segments:
        if not isinstance(entry, dict):
            continue
        sid = _safe_str(entry.get("segment_id"))
        if not sid:
            continue
        # group key: g#### (everything up to but not including _p##)
        m = re.match(r"^(g\d+)", sid)
        if not m:
            continue
        gkey = m.group(1)

        clip_rel = _safe_str(entry.get("group_clip") or "")
        if clip_rel and not os.path.isabs(clip_rel):
            clip_abs = os.path.normpath(os.path.join(base_dir, clip_rel))
        else:
            clip_abs = clip_rel

        try:
            start = float(entry.get("start_sec") or 0.0)
            end = float(entry.get("end_sec") or 0.0)
        except Exception:
            start, end = 0.0, 0.0

        by_group.setdefault(gkey, []).append({
            "segment_id": sid,
            "group_clip": clip_abs,
            "start_sec": start,
            "end_sec": end,
            "method": _safe_str(entry.get("method") or "proportional"),
        })

    # sort each group's panels by segment_id
    for gkey in by_group:
        by_group[gkey].sort(key=lambda e: e["segment_id"])

    return by_group


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
    image_min: float = 0.0,
) -> float:
    base_min = float(base_min)
    max_sec = float(max_sec)

    if mode == "narrated" and audio_duration_sec and audio_duration_sec > 0.0:
        # The panel must COVER the whole voiceover — NEVER truncate narration.
        # max_sec only caps SILENT holds (the no-audio branch below); when audio
        # is playing, the line's length governs (floored at base_min, no ceiling),
        # so a long beat just shows its panels longer instead of clipping the line
        # mid-sentence (the "...had absolutely no neigong at all" cut-off bug).
        # C2: a visually heavy panel also floors the dwell at image_min (a FLOOR,
        # never a cap — defaults to 0.0 so callers that don't pass it are unchanged).
        dur = float(audio_duration_sec) + float(audio_pad_sec)
        return float(max(base_min, dur, float(image_min)))

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


# Fallback rotation: a run of plain beats cycles through these four CLEAN
# directional slides so adjacent panels move in DIFFERENT directions. kenburns
# (a diagonal drift) used to be the fallback and landed on ~80% of cuts — a muddy,
# repetitive look. It is now reserved as an explicit accent only. static is no
# longer a fallback either: calm panels get a gentle slide, never a frozen frame.
_DIRECTIONAL_CYCLE = ("slide_left", "slide_right", "tilt_up", "tilt_down")


def _choose_motion_mode(beat: Dict[str, Any], ordinal: int = 0) -> str:
    """Pick a camera-motion MODE for a beat.

    `ordinal` (beat index) rotates the directional fallback so consecutive plain
    beats move in different directions. It is optional/back-compatible — callers
    that omit it get the first direction in the cycle.

    Priority: explicit `rendering_hints.camera_motion` > mood/emotional cues >
    a rotating directional-slide fallback. `static` is emitted ONLY when the hint
    explicitly asks to hold; "calm" maps to a gentle slide, not a freeze.
    """
    hint = _normalize_camera_motion_hint(beat)
    mood_words = _norm_words(beat.get("mood_words") or [])
    emotional_turn = _safe_str(beat.get("emotional_turn")).lower()

    # Explicit hints win. "hold" is an alias for an intentional static frame.
    direct_ok = {
        "static", "kenburns", "slow_pan", "pan", "zoom_in", "zoom_out",
        "tilt_up", "tilt_down", "slide_left", "slide_right", "push_in", "pull_out",
    }
    if hint in ("hold", "freeze"):
        return "static"
    if hint in direct_ok:
        if hint in ("slow_pan", "pan"):
            # a plain "pan" is a lateral slide, not a diagonal kenburns
            return _DIRECTIONAL_CYCLE[int(ordinal) % 2]  # slide_left / slide_right
        if hint == "push_in":
            return "zoom_in"
        if hint == "pull_out":
            return "zoom_out"
        return hint

    # Mood-driven directional intent.
    if _has_any(mood_words, ["action", "panic", "chase", "fight", "impact", "chaos"]):
        # alternate the lateral whip so back-to-back action beats don't repeat
        return "slide_left" if int(ordinal) % 2 == 0 else "slide_right"
    if _has_any(mood_words, ["reveal", "introduction", "hero", "awe", "arrival"]):
        return "tilt_up"
    if _has_any(mood_words, ["sad", "regret", "defeat", "loss", "mourning"]):
        return "tilt_down"
    if _has_any(mood_words, ["tension", "mystery", "horror", "threat", "danger"]):
        return "zoom_in"

    # Calm: a gentle slide, NOT static. (static is hint-only above.)
    # Fallback for everything else: rotate through clean directional slides so the
    # sequence reads varied instead of the old diagonal-on-every-panel kenburns.
    return _DIRECTIONAL_CYCLE[int(ordinal) % len(_DIRECTIONAL_CYCLE)]


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


def _bbox_center(bbox: Any) -> Optional[Tuple[float, float]]:
    """Center (cx, cy) of a normalized [x0,y0,x1,y1] target bbox, or None."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]),
                          float(bbox[2]), float(bbox[3]))
    except Exception:
        return None
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_area(bbox: Any) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        return max(0.0, float(bbox[2]) - float(bbox[0])) * \
               max(0.0, float(bbox[3]) - float(bbox[1]))
    except Exception:
        return 0.0


def pick_face_target(targets: Any) -> Optional[Dict[str, Any]]:
    """Choose the face the camera should END on: prefer the LARGEST face (the
    subject in focus); ties broken toward frame-center. NEVER a text_block — the
    bubble is inpainted blank, so ending there lands on an empty blob. Returns the
    chosen face target dict, or None when the panel has no usable face."""
    if not isinstance(targets, list):
        return None
    faces = []
    for t in targets:
        if not isinstance(t, dict) or t.get("type") != "face":
            continue
        c = _bbox_center(t.get("bbox"))
        if c is None:
            continue
        faces.append((t, c, _bbox_area(t.get("bbox"))))
    if not faces:
        return None
    # largest first; tie -> closest to frame center (0.5, 0.5)
    faces.sort(key=lambda fc: (-fc[2],
                               (fc[1][0] - 0.5) ** 2 + (fc[1][1] - 0.5) ** 2))
    return faces[0][0]


def _content_focus_y(targets: Any) -> float:
    """Vertical center (0..1) the TALL cover-crop window should frame on: a FACE if
    detected, else the middle of the LARGEST band of the panel NOT covered by a
    text_block (the bubble is inpainted blank, so we keep the window off it). This
    is what stops a tall close-up from framing the empty bubble. Defaults upper-
    middle (0.4) when there's nothing to go on (manhwa subjects sit high)."""
    face = pick_face_target(targets)
    if face is not None:
        c = _bbox_center(face.get("bbox"))
        if c is not None:
            return float(clamp(c[1], 0.0, 1.0))
    blanks = []
    for t in (targets or []):
        if isinstance(t, dict) and t.get("type") == "text_block":
            bb = t.get("bbox")
            if bb and len(bb) >= 4:
                blanks.append((float(bb[1]), float(bb[3])))
    if not blanks:
        return 0.4  # no face, no bubble to avoid -> upper-middle (manhwa subjects sit high)
    blanks.sort()
    bands, cur = [], 0.0
    for y0, y1 in blanks:
        if y0 > cur:
            bands.append((cur, y0))
        cur = max(cur, y1)
    if cur < 1.0:
        bands.append((cur, 1.0))
    if not bands:
        return 0.4
    by0, by1 = max(bands, key=lambda b: b[1] - b[0])
    return float(clamp((by0 + by1) / 2.0, 0.0, 1.0))


def face_end_bias(face_bbox: Any) -> Dict[str, float]:
    """End-of-move pan bias (normalized, in the engine's [-1..1] pan budget) that
    lands the FACE centered in frame at the end of the Ken Burns move.

    Convention (verified against remotion/src/Cut.tsx biasOffset + the translate):
    the bias drives a CSS/Blender translate of the foreground image. A face to the
    RIGHT of center (cx>0.5) needs the image to move LEFT, i.e. negative x bias, so
    +(cx-0.5) maps to a NEGATIVE x. For Y, Cut.tsx negates the y term
    (Blender offset_y is up-positive, CSS translateY is down-positive), so a face
    BELOW center (cy>0.5) needs a POSITIVE y bias. Magnitude = how far off-center
    the face is, normalized by the half-frame (0.5) and clamped to the pan budget;
    PAN_CAP downstream keeps the actual travel subtle."""
    c = _bbox_center(face_bbox)
    if c is None:
        return {"x": 0.0, "y": 0.0}
    cx, cy = c
    bx = clamp(-(cx - 0.5) / 0.5, -1.0, 1.0)
    by = clamp((cy - 0.5) / 0.5, -1.0, 1.0)
    return {"x": round(float(bx), 3), "y": round(float(by), 3)}


def face_aware_motion(base_motion: Dict[str, Any],
                      targets: Any) -> Dict[str, Any]:
    """Per-panel motion: if the panel has a FACE target, return a copy of the
    shot's motion whose pan ENDS centered on that face (start neutral so the move
    travels TOWARD it); otherwise return the base motion unchanged. Zoom, strength,
    bg_fill, fg_fit and ease are all preserved — only the pan focal point changes.
    A 'static' shot stays static (no pan to redirect)."""
    if not isinstance(base_motion, dict):
        return base_motion
    if (base_motion.get("mode") or "").lower() == "static":
        return base_motion
    face = pick_face_target(targets)
    if face is None:
        return base_motion
    end_bias = face_end_bias(face.get("bbox"))
    if end_bias == {"x": 0.0, "y": 0.0}:
        # face already dead-center: a pan toward it would be a no-op; keep the
        # shot's generic move rather than freezing the pan.
        return base_motion
    motion = dict(base_motion)
    # travel FROM neutral TO the face; nudge the start slightly opposite so the
    # move reads as a deliberate push toward the face instead of a static frame.
    motion["start_bias"] = {"x": round(-end_bias["x"] * 0.25, 3),
                            "y": round(-end_bias["y"] * 0.25, 3)}
    motion["end_bias"] = end_bias
    motion["focus"] = "face"  # debug/QA breadcrumb; renderers ignore unknown keys
    return motion


# Distinct camera moves cycled by GLOBAL cut index so neighbouring face-less panels
# never share the same move. The old table was MOSTLY diagonals (kenburns drift) ->
# "repetitive, same animation 3x in a row" + a muddy diagonal on every panel. It is
# now biased toward CLEAN directional slides — pure lateral (L/R) and pure vertical
# (U/D) — rotating L->R->U->D so adjacent cuts move in DIFFERENT directions. A single
# gentle diagonal remains as a rare accent. Applied ONLY to cuts with no face (face
# cuts keep their face-ending move). Tall scroll panels ignore bias in the renderer,
# so this is a harmless no-op there. Each entry: (start_bias, end_bias, zoom).
_SLIDE_TRAVEL = 0.75   # matches slide_left/right magnitude in _motion_params_for_mode
_TILT_TRAVEL  = 0.70   # matches tilt_up/down magnitude in _motion_params_for_mode
_MOTION_VARIANTS = [
    {"start": ( _SLIDE_TRAVEL, 0.0), "end": (-_SLIDE_TRAVEL, 0.0), "zoom": (1.03, 1.10)},  # slide: image L
    {"start": (0.0, -_TILT_TRAVEL),  "end": (0.0,  _TILT_TRAVEL),  "zoom": (1.04, 1.14)},  # tilt: image up
    {"start": (-_SLIDE_TRAVEL, 0.0), "end": ( _SLIDE_TRAVEL, 0.0), "zoom": (1.03, 1.10)},  # slide: image R
    {"start": (0.0,  _TILT_TRAVEL),  "end": (0.0, -_TILT_TRAVEL),  "zoom": (1.04, 1.10)},  # tilt: image down
    {"start": ( _SLIDE_TRAVEL, 0.0), "end": (-_SLIDE_TRAVEL, 0.0), "zoom": (1.05, 1.16)},  # slide L + push in
    {"start": (0.0, -_TILT_TRAVEL),  "end": (0.0,  _TILT_TRAVEL),  "zoom": (1.16, 1.05)},  # tilt up + pull out
    {"start": (-_SLIDE_TRAVEL, 0.0), "end": ( _SLIDE_TRAVEL, 0.0), "zoom": (1.05, 1.16)},  # slide R + push in
    {"start": (0.22, 0.14),          "end": (-0.22, -0.14),        "zoom": (1.05, 1.12)},  # rare gentle diagonal accent
]


def _vary_motion(base_motion: Dict[str, Any], ordinal: int) -> Dict[str, Any]:
    """Return a COPY of base_motion with its pan direction + zoom set to the
    ordinal-th variant, so consecutive face-less cuts never share a move. Strength,
    ease, bg_fill and fg_fit are preserved. A 'static' shot is returned unchanged."""
    if not isinstance(base_motion, dict):
        return base_motion
    if (base_motion.get("mode") or "").lower() == "static":
        return base_motion
    v = _MOTION_VARIANTS[int(ordinal) % len(_MOTION_VARIANTS)]
    m = dict(base_motion)
    m["start_bias"] = {"x": round(float(v["start"][0]), 3),
                       "y": round(float(v["start"][1]), 3)}
    m["end_bias"] = {"x": round(float(v["end"][0]), 3),
                     "y": round(float(v["end"][1]), 3)}
    m["zoom"] = {"start": round(float(v["zoom"][0]), 3),
                 "end": round(float(v["zoom"][1]), 3)}
    m["varied"] = True  # QA breadcrumb; renderers ignore unknown keys
    return m


# Duration thresholds for motion scaling. Cuts shorter than SHORT_CUT_SEC are
# almost imperceptible at normal strength — they need a bigger, faster move.
_SHORT_CUT_SEC = 4.0   # below this: boost strength + widen bias
_LONG_CUT_SEC  = 10.0  # above this: no boost needed (motion reads fine)

# Minimum bias travel magnitude to guarantee visible panning.
_MIN_BIAS_TRAVEL = 0.18   # |end - start| per axis at minimum


def motion_for_cut(dur: float, base_motion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a motion dict tuned for a cut of `dur` seconds.

    Two invariants are enforced:
    1. **Visible pan**: if start_bias == end_bias (pure zoom / no lateral move),
       inject a small diagonal drift so the cut is never fully static-looking.
    2. **Duration-aware strength**: short cuts (< SHORT_CUT_SEC) get a boosted
       effective strength so the move is perceptible in the narrow window;
       long cuts keep the base strength. Scale is linear, clamped to [0, 1].

    A 'static' shot is returned as-is (intentionally motionless).
    Face-aware pans (focus='face') have their direction preserved; only the
    magnitude is boosted where needed.

    The caller is responsible for ensuring base_motion already contains the
    correct pan direction (face-aware or variant-rotated). This helper only
    scales values — it never chooses a new direction.
    """
    # Build a working copy (never mutate the caller's dict)
    if base_motion is None:
        base_motion = _motion_params_for_mode("kenburns", dur, [])
    m = dict(base_motion)
    m["start_bias"] = dict(m.get("start_bias") or {"x": 0.0, "y": 0.0})
    m["end_bias"]   = dict(m.get("end_bias")   or {"x": 0.0, "y": 0.0})

    # Static shots: touch nothing
    if (m.get("mode") or "").lower() == "static":
        return m

    # ── 1. Guarantee perceptible pan ─────────────────────────────────────────
    dx = m["end_bias"]["x"] - m["start_bias"]["x"]
    dy = m["end_bias"]["y"] - m["start_bias"]["y"]
    travel = (dx ** 2 + dy ** 2) ** 0.5
    if travel < 1e-6:
        # Zero-pan (pure zoom): inject a gentle default diagonal drift.
        # Direction chosen to look natural on most manhwa panels (top-left → bottom-right).
        drift = _MIN_BIAS_TRAVEL
        m["start_bias"] = {"x": round( drift, 3), "y": round( drift * 0.6, 3)}
        m["end_bias"]   = {"x": round(-drift, 3), "y": round(-drift * 0.6, 3)}
        dx, dy = m["end_bias"]["x"] - m["start_bias"]["x"], m["end_bias"]["y"] - m["start_bias"]["y"]
        travel = (dx ** 2 + dy ** 2) ** 0.5

    # ── 2. Duration-aware strength boost ─────────────────────────────────────
    # Linear interpolation: at SHORT_CUT_SEC → factor 1.25×; at LONG_CUT_SEC → factor 1.0×.
    # Clamped so cuts shorter than SHORT get no more than the 1.25 cap.
    base_strength = float(m.get("strength") or 0.75)
    if dur <= _SHORT_CUT_SEC:
        factor = 1.25
    elif dur >= _LONG_CUT_SEC:
        factor = 1.0
    else:
        t = (dur - _SHORT_CUT_SEC) / (_LONG_CUT_SEC - _SHORT_CUT_SEC)
        factor = 1.25 - 0.25 * t   # 1.25 → 1.0 over the range

    new_strength = clamp(base_strength * factor, 0.0, 1.0)

    # Widen bias proportionally on short cuts so the pan travel also reads larger.
    if factor > 1.0:
        scale = min(factor, 1.0 / max(abs(m["start_bias"]["x"]), abs(m["start_bias"]["y"]),
                                       abs(m["end_bias"]["x"]), abs(m["end_bias"]["y"]), 1e-6))
        scale = min(scale, factor)   # never more than the factor itself
        # Scale the pan — but only up to the pan budget of ±1.0
        for key in ("start_bias", "end_bias"):
            bx = clamp(m[key]["x"] * scale, -1.0, 1.0)
            by = clamp(m[key]["y"] * scale, -1.0, 1.0)
            m[key] = {"x": round(float(bx), 3), "y": round(float(by), 3)}

    m["strength"] = round(float(new_strength), 3)
    return m


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
PANEL_FLOOR_SEC = 2.0   # keep == prep_qa flash_cut threshold


def _floor_shot_dur(n_kept: int, shot_dur: float, floor: float) -> float:
    """Extend a segment so each of n_kept panels gets >= floor seconds; never shrink."""
    if n_kept and floor and shot_dur / n_kept < floor:
        return float(n_kept) * float(floor)
    return float(shot_dur)


def build_cuts(
    scene_files: List[str],
    shot_dur: float,
    *,
    min_cut_sec: float,
    selection: Optional[List[Dict[str, Any]]] = None,
    protected: Optional["set"] = None,
    floor: float = 0.0,
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
    # EXCEPTION: when `floor` is set and the panels are TOO dense to each get
    # `floor` seconds (sub-second flashes), _floor_shot_dur (below) extends the
    # segment to k*floor so no panel is sub-floor — a small, deliberate stretch.
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

    shot_dur = _floor_shot_dur(k, shot_dur, floor)   # extend if too tight; never shrink
    per = shot_dur / float(k)
    cuts: List[Dict[str, Any]] = []
    t = 0.0
    for i, f in enumerate(files):
        dur = per if i < k - 1 else (shot_dur - t)  # last cut absorbs rounding
        cuts.append({"file": f, "start": round(t, 3), "dur": round(float(dur), 3)})
        t += per
    return cuts


def pick_protected_inject_segment(segment_picks: List[List[str]]) -> int:
    """Choose WHICH of a group's segments should carry a still-missing protected
    file (an in-world story/system card the per-shot LLM selection dropped, but
    which the group's protection guarantees must be SHOWN at least once).

    `segment_picks` is the ordered list of each emitted (non-filler) segment's
    chosen `segment_scene_files`. We prefer the LAST real segment so the card
    lands as a closing hold; among ties we prefer the segment whose pick list is
    SMALLEST so the injected card gets a fairer share of the hold. Returns the
    index into `segment_picks`, or -1 when there are no segments to inject into.
    """
    if not segment_picks:
        return -1
    n = len(segment_picks)
    # smallest pick list wins; ties broken toward the LATER segment (closing hold)
    best_idx = -1
    best_key = None
    for i in range(n):
        key = (len(segment_picks[i]), -i)  # fewer files first, then later index
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


def inject_missing_protected(
    segment_picks: List[List[str]],
    group_scene_files: List[str],
    protected: "set",
) -> List[List[str]]:
    """Make sure EVERY protected file in the group's `scene_files` is shown in at
    least one of the group's segments. The per-SHOT (microbeat) selection picks
    panels from the SCRIPT's per-shot list, which can EXCLUDE a protected card the
    LLM tagged 'redundant' — the group-level protection never propagates to it, so
    the card renders in NO segment. We compute the protected files that landed in
    NO segment and append each to ONE chosen segment (see
    pick_protected_inject_segment). Pure: returns a new list of pick lists,
    leaving non-protected selection untouched.
    """
    picks = [list(p) for p in segment_picks]
    prot = protected or set()
    if not prot or not picks:
        return picks
    in_group = [f for f in group_scene_files if f in prot]
    if not in_group:
        return picks
    shown: set = set()
    for p in picks:
        shown.update(p)
    missing = [f for f in in_group if f not in shown]
    if not missing:
        return picks
    idx = pick_protected_inject_segment(picks)
    if idx < 0:
        return picks
    for f in missing:
        if f not in picks[idx]:
            picks[idx].append(f)
    return picks


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
    """Title/system cards (SYSTEM ACTIVATION., STARTING ACTIVATION.) — short
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
        if (not isinstance(it, dict)
                or is_chrome_scene(it)
                or text_context_only_panel(it)):
            continue
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
        if not f or str(it.get("panel_kind") or "").strip().lower() != "story":
            continue
        # An in-world STYLED TEXT / SYSTEM / INFO card (SKY CORPORATION., 7TH
        # GENERATION NANO MACHINE, STARTING ACTIVATION.) is PLOT and must be
        # SHOWN. The detector sometimes mis-boxes such a styled card as a
        # "speech bubble" subject, which makes text_context_only_panel exclude
        # it. The title-card signal OUTRANKS that exclusion: a real system/info
        # card is always protected, even when it reads as text-only.
        if looks_like_system_card(it):
            out.add(f)
            continue
        if not text_context_only_panel(it):
            out.add(f)
    return out


_TEXT_CONTEXT_SUBJECT_TERMS = (
    "speech bubble",
    "bubble",
    "thought bubble",
    "text bubble",
    "caption",
    "text",
    "sfx",
    "sound effect",
    "onomatopoeia",
)

_MINOR_FRAGMENT_SUBJECT_TERMS = (
    "hair",
    "character's hair",
    "character hair",
)


def _looks_like_title_text(ocr: str, text_coverage: float) -> bool:
    """Short uppercase title/system cards are visual story beats, not context-only
    bubbles. This mirrors the text half of protected_card_files without requiring
    image IO."""
    ocr = str(ocr or "").strip()
    if not ocr or "..." in ocr or any(ch in ocr for ch in "~!?"):
        return False
    words = [w for w in re.split(r"[^A-Za-z0-9']+", ocr)
             if any(c.isalpha() for c in w)]
    letters = [c for c in ocr if c.isalpha()]
    if not (2 <= len(words) <= 8) or not letters:
        return False
    if sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return False
    return float(text_coverage or 0.0) < 0.20


def looks_like_system_card(vitem: Dict[str, Any]) -> bool:
    """Manifest-level signal for an in-world STYLED TEXT / SYSTEM / INFO card
    (SKY CORPORATION., 7TH GENERATION NANO MACHINE, STARTING ACTIVATION.).

    Mirrors render_prep._is_title_card / prep_qa._is_title_card so the stages
    agree, but uses only the vision MANIFEST fields (we have no image pixels
    here): a short MOSTLY-CAPS phrase (caps ratio >= 0.8, 2-8 words), with OCR
    present, low text_coverage (< 0.20), panel_kind story, and not chrome. The
    pixel flatness test (flat white/black frame) is render_prep's job; this
    manifest-level signal is enough to PROTECT the panel from being dropped.

    Deliberately conservative: a pure SPEECH bubble of conversational dialogue
    (lowercase, or caps SHOUT in a big high-coverage bubble) is NOT a card and
    stays excludable via text_context_only_panel. Only the styled-card case is
    rescued — never every text panel."""
    if not isinstance(vitem, dict):
        return False
    if str(vitem.get("panel_kind") or "").strip().lower() != "story":
        return False
    if is_chrome_scene(vitem):
        return False
    return _looks_like_title_text(
        vitem.get("ocr_clean"), float(vitem.get("text_coverage") or 0.0))


def text_context_only_panel(vitem: Dict[str, Any]) -> bool:
    """Panel whose only story signal is text/bubble content.

    Its words still feed narration context, but after bubble text is blanked it
    is not a useful visual cut. This catches the failure where understanding
    stamped a pure thought bubble as panel_kind=story, which then made the
    planner protect an empty bubble on screen.
    """
    kind = str(vitem.get("panel_kind") or "").strip().lower()
    ocr = str(vitem.get("ocr_clean") or "").strip()
    text_cov = float(vitem.get("text_coverage") or 0.0)
    if is_chrome_scene(vitem):
        return True
    subjects = [str(s or "").strip().lower()
                for s in (vitem.get("subjects") or []) if str(s or "").strip()]
    if kind == "caption":
        return True
    if kind == "empty":
        return True
    if not subjects:
        return False

    def is_text_subject(subj: str) -> bool:
        return any(term in subj for term in _TEXT_CONTEXT_SUBJECT_TERMS)

    def is_minor_fragment_subject(subj: str) -> bool:
        s = subj.strip().lower()
        return (s in _MINOR_FRAGMENT_SUBJECT_TERMS
                or s.endswith("'s hair")
                or s.endswith(" hair"))

    has_text_subject = any(is_text_subject(s) for s in subjects)
    has_real_subject = any(
        not is_text_subject(s) and not is_minor_fragment_subject(s)
        for s in subjects)
    has_text_signal = bool(ocr) or text_cov >= 0.02 or bool(vitem.get("text_only"))
    if has_text_subject and not has_real_subject and has_text_signal:
        # A clean flat system/title card is story content; a speech/thought
        # bubble plus only a sliver of hair is context once dialogue is blanked.
        has_bubble_subject = any("bubble" in s for s in subjects)
        return bool(has_bubble_subject or not _looks_like_title_text(ocr, text_cov))
    if _looks_like_title_text(ocr, text_cov):
        return False
    return False


def text_context_only_files(vision_path: str) -> "set":
    out: set = set()
    try:
        items = json.load(open(vision_path)).get("items") or []
    except Exception:
        return out
    for it in items:
        f = os.path.basename(str(it.get("scene_file") or ""))
        if f and text_context_only_panel(it):
            out.add(f)
    return out


def publication_chrome_files(vision_path: str) -> "set":
    """Publication/platform/credit/title pages. These are neither rendered nor
    voiced as standalone recap beats; in-world screens/stat cards remain
    panel_kind='story' and are preserved by the story/system-card paths."""
    out: set = set()
    try:
        items = json.load(open(vision_path)).get("items") or []
    except Exception:
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        f = os.path.basename(str(it.get("scene_file") or ""))
        if f and is_chrome_scene(it):
            out.add(f)
    return out


def index_targets_by_file(vision_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Map scene_file basename -> that panel's camera targets (from
    vision_extract.make_targets). Used to point each cut's pan at the panel's
    FACE. Degrades to {} when the manifest is missing/old (no targets)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not vision_path or not os.path.exists(vision_path):
        return out
    try:
        with open(vision_path, "r", encoding="utf-8") as fh:
            items = json.load(fh).get("items") or []
    except Exception:
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        f = os.path.basename(str(it.get("scene_file") or ""))
        tgts = it.get("targets")
        if f and isinstance(tgts, list):
            out[f] = tgts
    return out


def caption_files(vision_path: str) -> "set":
    """Panels the understanding calls a CAPTION — a bare narrative-voice / inner-
    monologue text card ('BACK THEN, I HAD NO IDEA.', 'AND I...'). Their WORDS are
    already woven into the narration, so the bare text card is not a shot; it's
    dropped from the montage. Distinct from a STORY panel (a real scene, or an
    in-world screen showing the character's own content), which is shown."""
    out: set = set()
    try:
        items = json.load(open(vision_path)).get("items") or []
    except Exception:
        return out
    for it in items:
        f = os.path.basename(str(it.get("scene_file") or ""))
        if f and str(it.get("panel_kind") or "").strip().lower() == "caption":
            out.add(f)
    return out


def drop_caption_cards(group_order: List[tuple], caption_set: "set") -> Dict[int, List[str]]:
    """Per beat, show its NON-caption panels (real scenes + in-world screens) and
    drop the bare caption cards — their words ride the narration. A beat left with
    nothing but captions shows NOTHING of its own (empty list): it must NOT hold a
    stand-in copy of an adjacent real panel — that manufactured a repeated static
    cut (the p097x3 panel-collapse symptom). Caption-only beats are already folded
    into a same-segment neighbour upstream (story_group.merge_caption_solos), so a
    bare card reaching here has no real panel of its own to show. group_order is an
    ordered list of (group_id, [panel basenames])."""
    if not caption_set:
        return {gid: list(files) for gid, files in group_order}
    return {gid: [f for f in files if f not in caption_set] for gid, files in group_order}


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
    ap.add_argument("--align", default="", help="manifest.align.json (per-group TTS mode; auto-detected if blank)")
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

    # GROUP MODE: load manifest.align.json when present (explicit --align or
    # auto-detected next to the tts-index).  Non-empty → one item per group;
    # empty → per-panel B2 path below (unchanged).
    _align_path = args.align
    if not _align_path and args.tts_index:
        _auto = os.path.join(os.path.dirname(os.path.abspath(args.tts_index)),
                             "manifest.align.json")
        if os.path.exists(_auto):
            _align_path = _auto
    _align_base = os.path.dirname(os.path.abspath(_align_path)) if _align_path else (
        os.path.dirname(os.path.abspath(args.tts_index)) if args.tts_index else ".")
    align_by_group: Dict[str, List[Dict[str, Any]]] = _load_align_index(_align_path, _align_base)
    if align_by_group:
        print(f"[plan] GROUP MODE: manifest.align.json detected — {len(align_by_group)} group(s)")

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
    # husk filter and the LLM's 'redundant' verdict, which is non-deterministic
    # across runs for flat in-world info cards.
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
    # bare caption/monologue cards are narrated, not shown: drop them from the
    # montage (the narration already carries their words), holding a real scene
    # for any beat that would otherwise be a blank card. In-world screens are
    # panel_kind 'story', not 'caption', so they stay.
    chrome_set = publication_chrome_files(args.vision)
    context_only_set = chrome_set | caption_files(args.vision) | text_context_only_files(args.vision)
    # per-panel camera targets (faces/objects/text_blocks) so each cut's pan can
    # END centered on the panel's FACE instead of a generic drift that may land on
    # an inpainted (blank) speech bubble.
    targets_by_file = index_targets_by_file(args.vision)
    montage = drop_caption_cards(
        [(int(g.get("group_id") or g.get("shot_id") or 0),
          [os.path.basename(str(x)) for x in (g.get("scene_files") or []) if x])
         for g in groups], context_only_set)
    if context_only_set:
        print(f"[plan] {len(context_only_set)} text/context-only panel(s) -> narrated, not shown")

    cut_ordinal = 0  # GLOBAL across all beats: drives per-cut motion variation
    beat_ordinal = 0  # GLOBAL across all beats: rotates the directional-slide fallback
    for gobj in groups:
        group_id = int(gobj.get("group_id") or gobj.get("shot_id") or 0)
        shot_id = int(gobj.get("shot_id") or group_id or 0)

        scene_files = gobj.get("scene_files") or []
        if not isinstance(scene_files, list):
            scene_files = []
        scene_files = [os.path.basename(str(x)) for x in scene_files if x]
        if scene_files and all(f in chrome_set for f in scene_files):
            dropped_summary.append({"group_id": group_id,
                                    "dropped_publication_chrome": scene_files})
            continue
        # caption cards out, real scenes in — words stay in narration. Respect an
        # EXPLICIT empty montage entry (a caption-only beat shows nothing of its
        # own; never hold a stand-in copy of a neighbour) — only fall back to the
        # group's own files when the beat is absent from the montage entirely.
        scene_files = montage.get(group_id, scene_files)

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
        # one ordinal per beat — rotates the directional-slide fallback so adjacent
        # beats move in different directions (slide_left -> slide_right -> tilt ...).
        this_beat_ordinal = beat_ordinal
        beat_ordinal += 1
        mood_words = _norm_words(beat.get("mood_words") or [])
        rh = beat.get("rendering_hints") or {}

        # ── GROUP MODE ────────────────────────────────────────────────────────
        # When manifest.align.json is present, emit ONE item per GROUP whose
        # tts_audio is the continuous group clip and whose cuts[] use the
        # per-panel aligned offsets (start/dur relative to the group clip).
        # This path is mutually exclusive with the per-segment B2 loop below.
        gkey = f"g{group_id:04d}"
        align_entries = align_by_group.get(gkey) if align_by_group else None
        if align_entries:
            # Build the scene_file lookup: segment_id -> basename from script
            seg_to_file: Dict[str, str] = {}
            for _sid, _srow in (script_by_gid.items() if script_by_gid else {}.items()):
                if int((_srow or {}).get("group_id") or 0) != group_id:
                    continue
                shot_sf = _scene_file_basenames((_srow or {}).get("scene_files") or [])
                if shot_sf:
                    seg_to_file[_sid] = shot_sf[0]

            # Group clip: first entry's group_clip (all entries share the same clip)
            group_clip_path = align_entries[0]["group_clip"]

            # Duration: read the WAV; fall back to max(end_sec)
            group_clip_dur = 0.0
            if group_clip_path and os.path.exists(group_clip_path):
                try:
                    group_clip_dur = _wav_duration_sec(group_clip_path)
                except Exception:
                    pass
            if group_clip_dur <= 0.0:
                group_clip_dur = max(e["end_sec"] for e in align_entries)

            # Joined narration text (for tts_text and tags)
            group_text_parts = []
            for _sid, _srow in sorted(
                [(sid, srow) for sid, srow in script_by_gid.items()
                 if int((srow or {}).get("group_id") or 0) == group_id],
                key=lambda t: t[0]
            ):
                p = _safe_str((_srow or {}).get("paragraph"))
                if p:
                    group_text_parts.append(p)
            group_tts_text = " ".join(group_text_parts)

            # Build cuts: one per aligned panel entry, offsets relative to group clip
            group_cuts: List[Dict[str, Any]] = []
            for entry in align_entries:
                sid_e = entry["segment_id"]
                # scene file: from script or groups manifest fallback
                fbase = seg_to_file.get(sid_e)
                if not fbase:
                    # fallback: use the panel index to pick from group's scene_files
                    m_pi = re.match(r"g\d+_p(\d+)$", sid_e)
                    pi = int(m_pi.group(1)) if m_pi else 0
                    fbase = scene_files[pi] if pi < len(scene_files) else (scene_files[0] if scene_files else "")
                if not fbase:
                    continue
                cut_start = float(entry["start_sec"])
                cut_dur = float(entry["end_sec"]) - cut_start

                # Per-cut motion — same logic as per-panel path
                motion_mode = _choose_motion_mode(beat, ordinal=this_beat_ordinal)
                base_motion = _motion_params_for_mode(motion_mode, cut_dur, mood_words)
                tf = targets_by_file.get(fbase)
                fm = face_aware_motion(base_motion, tf)
                if fm is not base_motion:
                    cm = fm
                else:
                    cm = _vary_motion(base_motion, cut_ordinal)
                    if cm is base_motion:
                        cm = dict(base_motion)
                cm = motion_for_cut(cut_dur, cm)
                cm["focus_y"] = round(_content_focus_y(tf), 3)
                cut_ordinal += 1

                group_cuts.append({
                    "file": fbase,
                    "start": round(cut_start, 3),
                    "dur": round(cut_dur, 3),
                    "motion": cm,
                })

            avoid_text_zoom = bool(rh.get("avoid_text_zoom", False))
            motion_mode = _choose_motion_mode(beat, ordinal=this_beat_ordinal)
            group_motion = _motion_params_for_mode(motion_mode, group_clip_dur, mood_words)
            group_camera = _camera_compat_from_motion(group_motion, avoid_text_zoom=avoid_text_zoom)

            primary_scene_file = group_cuts[0]["file"] if group_cuts else (scene_files[0] if scene_files else "")

            group_item: Dict[str, Any] = {
                "segment_id": gkey,
                "group_id": group_id,
                "shot_id": shot_id,
                "segment": str(gobj.get("segment") or "present"),
                "display_strategy": "multi_cut",
                "primary_scene_file": primary_scene_file,
                "group_scene_files": scene_files,
                "scene_files": [c["file"] for c in group_cuts],

                "cuts": group_cuts,

                "start_sec": round(time_cursor, 3),
                "duration_sec": round(float(group_clip_dur), 3),
                "end_sec": round(time_cursor + float(group_clip_dur), 3),

                "camera": group_camera,
                "motion": group_motion,
                "overlays": [],
                "tts_text": group_tts_text.strip(),
                "tts_audio": group_clip_path,
                "tts_audio_duration_sec": round(float(group_clip_dur), 3),
                "rendering_hints": {
                    "avoid_text_zoom": avoid_text_zoom,
                    "preferred_focus": rh.get("preferred_focus") or "",
                    "camera_motion": rh.get("camera_motion") or "",
                    "target_phrases": rh.get("target_phrases") if isinstance(rh.get("target_phrases"), list) else [],
                },
                "tags": {
                    "mood_words": beat.get("mood_words") or [],
                    "emotional_turn": beat.get("emotional_turn") or "",
                    "duration_source": "audio",
                    "filtered_blank_panels": True if args.prefer_clean else False,
                    "group_mode": True,
                },
            }
            timeline.append(group_item)
            time_cursor += float(group_clip_dur)
            continue  # skip the per-segment B2 loop for this group
        # ── END GROUP MODE ────────────────────────────────────────────────────

        # B2: a group may have several narration paragraphs (segment_id g####_p##).
        # Emit ONE timeline item per paragraph so each paragraph keeps its own
        # audio + timing. Fall back to a single group-level item when no
        # per-paragraph script rows exist (back-compat with old manifests).
        segments = segments_by_group.get(group_id) or []
        if not segments:
            fallback_sid = _safe_str(gobj.get("segment_id")) or f"g{group_id:04d}"
            fallback_srow = script_by_gid.get(fallback_sid) or script_by_gid.get(f"g{group_id:04d}")
            segments = [(fallback_sid, fallback_srow)]

        # FIRST PASS: compute each NON-FILLER segment's per-shot panel pick, then
        # guarantee every protected file in the GROUP's scene_files is shown in at
        # least one segment. The per-shot (microbeat) selection picks from the
        # SCRIPT's per-shot list, which can EXCLUDE a protected in-world story/
        # system card (the LLM tagged it 'redundant') — the group protection never
        # propagated to it, so the card rendered in NO segment. inject_missing_protected
        # appends any still-missing protected file to ONE segment (the smallest,
        # latest real one) so it always renders. Non-protected drops are untouched.
        def _pick_for_segment(srow_: Any) -> List[str]:
            shot_sf = _scene_file_basenames((srow_ or {}).get("scene_files") or [])
            shot_fb = _scene_file_basenames((srow_ or {}).get("fallback_scene_files") or [])
            if not shot_sf:
                return list(scene_files)
            allowed = set(scene_files)
            picked = [f for f in shot_sf if f in allowed]
            if not picked:
                picked = [f for f in shot_fb if f in allowed]
            return picked or scene_files[:1]

        emit_sids = [sid for sid, srow in segments
                     if not (args.mode == "narrated"
                             and is_filler_narration(_safe_str(srow.get("paragraph")) if srow else ""))]
        emit_srows = {sid: srow for sid, srow in segments}
        _pre_picks = [_pick_for_segment(emit_srows.get(sid)) for sid in emit_sids]
        _inj_picks = inject_missing_protected(
            _pre_picks, list(scene_files), protected & set(scene_files))
        injected_picks_by_sid: Dict[str, List[str]] = {
            sid: _inj_picks[i] for i, sid in enumerate(emit_sids)}

        for segment_id, srow in segments:
            paragraph = _safe_str(srow.get("paragraph")) if srow else ""
            delivery_tag = _safe_str(srow.get("delivery_tag")) if srow else ""
            avoid_text_zoom = bool((srow or {}).get(
                "avoid_text_zoom", rh.get("avoid_text_zoom", False)))
            shot_scene_files = _scene_file_basenames((srow or {}).get("scene_files") or [])
            shot_fallback_files = _scene_file_basenames((srow or {}).get("fallback_scene_files") or [])

            # a degenerate beat (empty narration → "The scene continues."
            # placeholder) is dropped: never voice filler over a stand-in panel
            if args.mode == "narrated" and is_filler_narration(paragraph):
                dropped_summary.append({"group_id": group_id,
                                        "dropped_filler_segment": segment_id})
                continue

            # Microbeat shots carry their own selected panel(s). Keep the story
            # group as context, but render the shot-level visual subset (computed
            # in the FIRST PASS above, with any group-protected card injected so it
            # always renders). If the subset was context-only/filtered away, fall
            # back to the nearest real scene chosen for the group.
            segment_scene_files = injected_picks_by_sid.get(segment_id)
            if segment_scene_files is None:
                segment_scene_files = scene_files
                if shot_scene_files:
                    allowed = set(scene_files)
                    picked = [f for f in shot_scene_files if f in allowed]
                    if not picked:
                        picked = [f for f in shot_fallback_files if f in allowed]
                    segment_scene_files = picked or scene_files[:1]

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

            motion_mode = _choose_motion_mode(beat, ordinal=this_beat_ordinal)
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
            primary_scene_file = segment_scene_files[0] if segment_scene_files else ""

            # Cuts drive the montage. COVERAGE: build_cuts shows EVERY distinct
            # panel within `dur` (drops only near-duplicate frames) — no panel is
            # truncated to fit a short line, and with no music we never stretch
            # into silence. One panel + a long line = a long hold.
            cuts: List[Dict[str, Any]] = []
            if segment_scene_files:
                if display_strategy == "single_hold":
                    cuts = build_cuts([primary_scene_file], dur,
                                      min_cut_sec=float(args.min_cut_sec))
                else:
                    cuts = build_cuts(
                        segment_scene_files, dur,
                        min_cut_sec=float(args.min_cut_sec),
                        selection=beat.get("scene_selection"),
                        protected=protected,
                        floor=PANEL_FLOOR_SEC,
                    )

            # PER-CUT motion: each cut is a DIFFERENT panel, so its pan must end on
            # ITS OWN face. The shot-level `motion` stays the default (and the
            # fallback for panels with no face / for held cuts that carry no own
            # motion); a face-bearing panel gets a copy whose end_bias lands the
            # face centered. The renderer prefers cut.motion over item.motion.
            for c in cuts:
                tf = targets_by_file.get(str(c.get("file") or ""))
                fm = face_aware_motion(motion, tf)
                if fm is not motion:
                    cm = fm                     # face panel: pan ends ON the face
                else:
                    # no face: rotate the move by GLOBAL cut index so consecutive
                    # face-less panels never share the same (previously identical)
                    # kenburns diagonal.
                    cm = _vary_motion(motion, cut_ordinal)
                    if cm is motion:            # static shot: still need a per-cut copy
                        cm = dict(motion)
                # Scale strength + guarantee perceptible pan for this cut's duration.
                cut_dur = float(c.get("duration_sec") or dur)
                cm = motion_for_cut(cut_dur, cm)
                # focus_y frames the TALL cover-crop window on the art (off the blank
                # bubble); the renderer only uses it on tall strips, a no-op elsewhere.
                cm["focus_y"] = round(_content_focus_y(tf), 3)
                c["motion"] = cm
                cut_ordinal += 1

            # The floor may have EXTENDED the cut tiling beyond the audio `dur`
            # (a panel-dense beat); adopt the cuts' real total so duration_sec /
            # end_sec / time_cursor below stay byte-aligned with the tiling and
            # audio placement (no cut_gap / total_drift). NEVER shrinks: cuts sum
            # to >= the original dur. Cutless segments keep the audio dur.
            dur = sum(float(c["dur"]) for c in cuts) if cuts else dur

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
                "scene_files": segment_scene_files if display_strategy == "multi_cut" else ([primary_scene_file] if primary_scene_file else []),

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
