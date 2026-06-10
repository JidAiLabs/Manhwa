#!/usr/bin/env python3
"""
blender_vse_from_plan.py (Blender 5.0+ VSE)

Builds a .blend from render.plan.json.
- Foreground image always "COVER"s 16:9 (cropping allowed) => no moving white borders.
- Optional blurred background filler (same image) to avoid black bars.
- Uses VSE strip.transform with PIXEL offsets (correct for Blender VSE).
- Supports callbacks (flashbacks) if present in plan.
- Renders to frames (PNG) reliably; encode to mp4 with ffmpeg afterward.
"""

import bpy
import json
import sys
import os
import argparse
import math
import random

# -----------------------
# Args
# -----------------------
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()

    p.add_argument("--plan", required=True)
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--out-blend", required=True)

    # frame render target (recommended)
    p.add_argument("--frames-dir", default="")
    p.add_argument("--frames-base", default="episode_")

    # render settings
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)

    # motion controls
    p.add_argument("--profile-strength", type=float, default=1.0)
    p.add_argument("--max-zoom-cap", type=float, default=1.35)
    p.add_argument("--pan-cap-frac", type=float, default=0.10)  # fraction of min(W,H) in pixels
    p.add_argument("--lock-center", action="store_true", help="Disable panning, only zoom/reveal/slide.")

    # multi-image handling
    p.add_argument("--split-multi-image", dest="split_multi_image", action="store_true", default=True)
    p.add_argument("--no-split-multi-image", dest="split_multi_image", action="store_false")

    # background filler
    p.add_argument("--bg-blur", action="store_true", default=True)
    p.add_argument("--no-bg-blur", dest="bg_blur", action="store_false")
    p.add_argument("--bg-blur-size", type=float, default=35.0)
    p.add_argument("--bg-dim", type=float, default=0.12)  # 0..1 via alpha over black (simple)

    # verbosity
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")

    return p.parse_args(argv)

# -----------------------
# Helpers
# -----------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path: str):
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path)) if os.path.splitext(path)[1] else os.path.abspath(path)
    os.makedirs(d, exist_ok=True)

def seconds_to_frames(sec: float, fps: int) -> int:
    return max(1, int(round(sec * fps)))

def resolve_path(scene_dir: str, f: str) -> str:
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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def get_profile(shot: dict) -> str:
    cam = shot.get("camera") or {}
    if isinstance(cam, dict):
        prof = cam.get("profile") or cam.get("style") or ""
        prof = str(prof).strip().lower()
        if prof:
            return prof
    return "default"

def mood_words(shot: dict):
    tags = shot.get("tags") or {}
    if isinstance(tags, dict):
        mw = tags.get("mood_words") or []
        if isinstance(mw, list):
            return [str(x).strip().lower() for x in mw if str(x).strip()]
    return []

def choose_motion_kind(profile: str, moods: list) -> str:
    m = set(moods)
    # You can extend this mapping easily
    if any(x in m for x in ("action", "fight", "chase", "explosion", "panic", "danger")):
        return "slide"
    if any(x in m for x in ("reveal", "intro", "entrance", "drama", "epic", "twist")):
        return "reveal_y"
    if any(x in m for x in ("calm", "sad", "reflection", "quiet")):
        return "pan_slow"
    if profile in ("static",):
        return "static"
    return "kenburns"

def pick_params(profile: str, moods: list, strength: float):
    presets = {
        "default":  {"z0": 1.05, "z1": 1.18, "pan": 0.07},
        "calm":     {"z0": 1.02, "z1": 1.06, "pan": 0.03},
        "drama":    {"z0": 1.06, "z1": 1.22, "pan": 0.08},
        "tension":  {"z0": 1.07, "z1": 1.26, "pan": 0.09},
        "action":   {"z0": 1.08, "z1": 1.30, "pan": 0.11},
        "horror":   {"z0": 1.09, "z1": 1.28, "pan": 0.08},
        "mystery":  {"z0": 1.06, "z1": 1.20, "pan": 0.07},
        "sad":      {"z0": 1.01, "z1": 1.05, "pan": 0.03},
        "epic":     {"z0": 1.05, "z1": 1.26, "pan": 0.08},
        "static":   {"z0": 1.00, "z1": 1.00, "pan": 0.00},
        "gentle_kenburns": {"z0": 1.05, "z1": 1.18, "pan": 0.07},
    }
    base = presets.get(profile, presets["default"]).copy()
    m = set(moods)

    if any(x in m for x in ("panic", "danger", "chase", "fight", "explosion")):
        base["z1"] += 0.04
        base["pan"] += 0.03
    if any(x in m for x in ("shock", "reveal", "twist", "betrayal")):
        base["z1"] += 0.03
        base["pan"] += 0.02
    if any(x in m for x in ("quiet", "reflection", "peace", "relief")):
        base["z1"] -= 0.04
        base["pan"] -= 0.02

    z0 = 1.0 + (base["z0"] - 1.0) * strength
    z1 = 1.0 + (base["z1"] - 1.0) * strength
    pan = base["pan"] * strength
    return z0, z1, pan

def load_image_size(filepath: str):
    img = bpy.data.images.load(filepath, check_existing=True)
    w, h = img.size[0], img.size[1]
    return max(1, int(w)), max(1, int(h))

def cover_base_scale(img_w: int, img_h: int, out_w: int, out_h: int) -> float:
    # scale so the image covers the entire output (cropping allowed)
    sx = out_w / float(img_w)
    sy = out_h / float(img_h)
    return max(sx, sy)

# -----------------------
# VSE strip creation
# -----------------------
def new_image_strip(scene, name, filepath, channel, frame_start, length):
    se = scene.sequence_editor
    st = se.strips.new_image(name=name, filepath=filepath, channel=channel, frame_start=frame_start)
    st.frame_final_duration = length
    return st

def new_effect(scene, name, effect_type, channel, frame_start, length, input1, input2=None):
    """
    Blender VSE effect API differs by build:
    - Some require keyword-only args (notably length=)
    - Some accept positional
    We'll try keyword form first, then fallback.
    """
    se = scene.sequence_editor
    try:
        return se.strips.new_effect(
            name=name,
            type=effect_type,
            channel=channel,
            frame_start=frame_start,
            length=length,
            input1=input1,
            input2=input2,
        )
    except TypeError:
        # fallback for builds that accept positional
        return se.strips.new_effect(name, effect_type, channel, frame_start, length, input1, input2)

def set_transform_key(strip, frame, scale, offx_px, offy_px):
    bpy.context.scene.frame_set(frame)
    strip.transform.scale_x = scale
    strip.transform.scale_y = scale
    strip.transform.offset_x = offx_px
    strip.transform.offset_y = offy_px
    strip.transform.keyframe_insert(data_path="scale_x", frame=frame)
    strip.transform.keyframe_insert(data_path="scale_y", frame=frame)
    strip.transform.keyframe_insert(data_path="offset_x", frame=frame)
    strip.transform.keyframe_insert(data_path="offset_y", frame=frame)

def motion_keyframes(kind: str, rng: random.Random, base_scale: float, z0: float, z1: float,
                     pan_cap_px: float, lock_center: bool):
    """
    Returns (s0, s1, ox0, oy0, ox1, oy1) in PIXELS.
    """
    # zoom factors applied on top of base cover scale
    zz0 = base_scale * z0
    zz1 = base_scale * z1

    if kind == "static":
        return zz0, zz0, 0.0, 0.0, 0.0, 0.0

    if lock_center:
        # allow reveal/slide only if you want; here fully lock offsets
        return zz0, zz1, 0.0, 0.0, 0.0, 0.0

    cap = float(pan_cap_px)

    if kind == "pan_slow" or kind == "kenburns":
        ang = rng.uniform(0, math.tau)
        dx = math.cos(ang)
        dy = math.sin(ang)
        mag = rng.uniform(0.35, 1.0) * cap
        ox0 = -dx * mag * 0.5
        oy0 = -dy * mag * 0.5
        ox1 = +dx * mag * 0.5
        oy1 = +dy * mag * 0.5
        return zz0, zz1, ox0, oy0, ox1, oy1

    if kind == "reveal_y":
        # vertical reveal: start lower, move up (foot -> face) or reverse
        direction = rng.choice([-1.0, 1.0])  # -1 = down->up (show lower first), +1 = up->down
        mag = rng.uniform(0.55, 1.0) * cap
        oy0 = direction * (mag * 0.5)
        oy1 = -direction * (mag * 0.5)
        ox0 = rng.uniform(-0.25, 0.25) * cap
        ox1 = -ox0
        return zz0, zz1, ox0, oy0, ox1, oy1

    if kind == "slide":
        # enter from left/right and settle; plus some zoom
        side = rng.choice([-1.0, 1.0])
        mag = rng.uniform(0.65, 1.0) * cap
        ox0 = side * (mag * 0.8)
        ox1 = 0.0
        oy0 = rng.uniform(-0.25, 0.25) * cap
        oy1 = 0.0
        return zz0, zz1, ox0, oy0, ox1, oy1

    return zz0, zz1, 0.0, 0.0, 0.0, 0.0

def add_blurred_bg(scene, fg_path, name_prefix, channel_base, fs, length, out_w, out_h,
                   blur_size: float, dim: float):
    """
    BG: same image on lower channel, scaled to cover (and slightly extra),
    then GAUSSIAN_BLUR effect on top.
    """
    bg_img_w, bg_img_h = load_image_size(fg_path)
    base = cover_base_scale(bg_img_w, bg_img_h, out_w, out_h)
    bg_scale = base * 1.10  # slightly larger so blur stays clean at edges

    bg = new_image_strip(scene, f"{name_prefix}_BG", fg_path, channel_base, fs, length)
    # keep bg centered (no pan)
    set_transform_key(bg, fs, bg_scale, 0.0, 0.0)
    set_transform_key(bg, fs + length, bg_scale, 0.0, 0.0)

    # blur effect strip
    bl = new_effect(scene, f"{name_prefix}_BLUR", "GAUSSIAN_BLUR", channel_base + 1, fs, length, bg, None)
    # blender exposes gaussian blur sizing differently across builds; try both
    if hasattr(bl, "size_x"):
        bl.size_x = blur_size
        bl.size_y = blur_size
    elif hasattr(bl, "factor"):
        bl.factor = 1.0

    # optional dim: overlay a semi-transparent color strip
    if dim and dim > 0:
        col = new_effect(scene, f"{name_prefix}_DIM", "COLOR", channel_base + 2, fs, length, None, None)
        # COLOR effect uses .color (RGBA)
        if hasattr(col, "color"):
            col.color = (0.0, 0.0, 0.0)
            col.blend_type = 'ALPHA_OVER'
            col.blend_alpha = clamp(dim, 0.0, 0.9)
        # place it over blur
        dimmed = new_effect(scene, f"{name_prefix}_ALPHA", "ALPHA_OVER", channel_base + 3, fs, length, col, bl)
        return dimmed  # final bg chain output
    return bl

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()

    fps = int(args.fps)
    out_w = int(args.width)
    out_h = int(args.height)

    # fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.resolution_x = out_w
    scene.render.resolution_y = out_h
    scene.render.use_sequencer = True

    if not scene.sequence_editor:
        scene.sequence_editor_create()

    plan = load_json(args.plan)
    timeline = plan.get("timeline") or []
    if not isinstance(timeline, list) or not timeline:
        raise SystemExit("Plan has no timeline items.")

    ensure_dir(args.out_blend)

    # frames output (recommended)
    if args.frames_dir:
        ensure_dir(args.frames_dir)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = os.path.join(args.frames_dir, args.frames_base)  # Blender will append ####.png

    # channel layout:
    # BG chain uses 1..4, FG starts at 10+, callbacks at 20+
    BG_CH = 1
    FG_CH = 10
    CB_CH = 20

    strength = float(args.profile_strength)
    max_zoom = float(args.max_zoom_cap)
    pan_cap_px = float(args.pan_cap_frac) * float(min(out_w, out_h))

    frame_cursor = 1
    strips_count = 0

    if args.verbose:
        print(f"[INFO] shots={len(timeline)} fps={fps} res={out_w}x{out_h} pan_cap_px={pan_cap_px:.1f} bg_blur={args.bg_blur}")

    def emit_shot_images(images, total_sec, profile, moods, is_callback=False):
        nonlocal frame_cursor, strips_count
        if not images:
            return

        per_sec = total_sec / max(1, len(images)) if args.split_multi_image else total_sec
        for i, img_path in enumerate(images):
            length = seconds_to_frames(per_sec, fps)
            fs = frame_cursor

            # BG
            bg_out = None
            if args.bg_blur:
                bg_out = add_blurred_bg(
                    scene, img_path,
                    name_prefix=("CB" if is_callback else "SH") + f"{fs:06d}",
                    channel_base=(BG_CH if not is_callback else BG_CH + 30),
                    fs=fs,
                    length=length,
                    out_w=out_w,
                    out_h=out_h,
                    blur_size=float(args.bg_blur_size),
                    dim=float(args.bg_dim),
                )

            # FG strip
            ch = CB_CH if is_callback else FG_CH
            fg = new_image_strip(scene, ("CB" if is_callback else "FG") + f"_{fs:06d}_{i:02d}", img_path, ch, fs, length)

            img_w, img_h = load_image_size(img_path)
            base_scale = cover_base_scale(img_w, img_h, out_w, out_h)

            z0, z1, pan = pick_params(profile, moods, strength)
            z0 = clamp(z0, 1.0, max_zoom)
            z1 = clamp(z1, 1.0, max_zoom)

            kind = "static" if (profile == "static") else choose_motion_kind(profile, moods)
            rng = random.Random((fs * 1000003) + 97)

            s0, s1, ox0, oy0, ox1, oy1 = motion_keyframes(kind, rng, base_scale, z0, z1, pan_cap_px * pan, args.lock_center)

            # keyframes
            set_transform_key(fg, fs, s0, ox0, oy0)
            set_transform_key(fg, fs + length, s1, ox1, oy1)

            # If we created a BG chain, ensure it is BELOW the FG by channels already.
            strips_count += 1
            if args.verbose:
                print(f"[ADD] {'CB' if is_callback else 'SH'} fs={fs} len={length} img={os.path.basename(img_path)} motion={kind} scale={s0:.3f}->{s1:.3f} off=({ox0:.1f},{oy0:.1f})->({ox1:.1f},{oy1:.1f})")

            frame_cursor += length

    for idx, shot in enumerate(timeline):
        dur_sec = float(shot.get("duration_sec") or 3.0)
        shot_id = int(shot.get("shot_id") or (idx + 1))

        # callbacks before main shot
        callbacks = shot.get("callbacks") or []
        if isinstance(callbacks, list) and callbacks:
            for cb in callbacks:
                cb_files = cb.get("scene_files") or []
                if not isinstance(cb_files, list) or not cb_files:
                    continue
                cb_dur = float(cb.get("duration_sec") or 0.7)
                resolved_cb = []
                for f in cb_files:
                    p = resolve_path(args.scene_dir, str(f))
                    if os.path.exists(p):
                        resolved_cb.append(p)
                emit_shot_images(resolved_cb, cb_dur, "static", ["flashback"], is_callback=True)

        # main shot images
        files = shot.get("scene_files") or []
        if not isinstance(files, list) or not files:
            continue
        resolved = []
        for f in files:
            p = resolve_path(args.scene_dir, str(f))
            if os.path.exists(p):
                resolved.append(p)

        if not resolved:
            continue

        profile = get_profile(shot)
        moods = mood_words(shot)
        emit_shot_images(resolved, dur_sec, profile, moods, is_callback=False)

        if args.verbose and (idx == 0 or (idx + 1) % 10 == 0):
            print(f"[INFO] progress {idx+1}/{len(timeline)} frame_cursor={frame_cursor}")

    scene.frame_start = 1
    scene.frame_end = max(1, frame_cursor - 1)

    bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)

    print(f"[OK] Saved blend: {args.out_blend}")
    if args.frames_dir:
        print(f"[OK] Render target (frames): {os.path.join(args.frames_dir, args.frames_base)}####.png")
    print(f"[OK] Timeline frames: {scene.frame_end}  seconds={scene.frame_end/fps:.2f}  strips={strips_count}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
