#!/usr/bin/env python3
"""
blender_vse_from_plan.py (Blender 5.0+ VSE) — aligned to render.plan.json (new pipeline)

What this version fixes / upgrades vs your current script:
- Uses render.plan.json fields exactly:
    - timeline[*].start_sec / duration_sec / end_sec (FLOAT-accurate)
    - timeline[*].cuts = [{file,start,dur}, ...] (montage plan from timeline_planner.py)
    - timeline[*].tts_audio (optional) -> adds SOUND strip aligned to shot start
    - timeline[*].motion / camera / camera_path (deterministic motion)
- Stops splitting images "evenly": montage is driven by cuts[] durations.
- Deterministic motion priority:
    1) camera_path (keyframes in normalized space, with zoom)
    2) motion (start_bias/end_bias + zoom.start/zoom.end + strength)
    3) fallback: gentle kenburns (still deterministic seed, no randomness)
- Honors text safety:
    - camera.avoid_text_zoom => tighter zoom cap
    - fg_fit.safe_inset_pct and fg_fit.mode ("contain" default) => readable framing
- Background fill is taken from motion.bg_fill (blur amount + dim)
- Places strips at absolute frames from start_sec (no drift).
"""

import bpy
import json
import sys
import os
import argparse
import math


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

    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)

    # Channels
    p.add_argument("--bg-channel", type=int, default=1)
    p.add_argument("--fg-channel", type=int, default=10)
    p.add_argument("--sfx-channel", type=int, default=25)  # sound strip channel

    # Safety / caps
    p.add_argument("--max-zoom-cap", type=float, default=1.35)      # global hard cap
    p.add_argument("--text-zoom-cap", type=float, default=1.06)     # tighter cap when avoid_text_zoom=True
    p.add_argument("--pan-cap-frac", type=float, default=0.10)      # fraction of min(W,H) in pixels

    # Defaults if plan omits them
    p.add_argument("--default-fg-fit", choices=["contain", "cover"], default="contain")
    p.add_argument("--default-safe-inset", type=float, default=0.06)

    # Background blur defaults (used only if plan doesn't specify bg_fill)
    p.add_argument("--bg-blur", action="store_true", default=True)
    p.add_argument("--no-bg-blur", dest="bg_blur", action="store_false")
    p.add_argument("--bg-blur-size", type=float, default=35.0)
    p.add_argument("--bg-dim", type=float, default=0.12)

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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def seconds_to_frames(sec: float, fps: int) -> int:
    return max(1, int(math.ceil(float(sec) * float(fps))))

def sec_to_frame_start(sec: float, fps: int) -> int:
    # Blender frames typically start at 1
    return max(1, 1 + int(round(float(sec) * float(fps))))

def resolve_path(scene_dir: str, f: str) -> str:
    # ALWAYS return an absolute path: relative strip paths are re-resolved by
    # Blender against the .blend file's own directory when the saved file is
    # reopened (e.g. by the render pass) — every strip silently goes missing
    # (magenta/black frames) while audio keeps working.
    if not f:
        return ""
    if os.path.isabs(f) and os.path.exists(f):
        return f
    cand = os.path.abspath(os.path.join(scene_dir, f))
    if os.path.exists(cand):
        return cand
    root, _ = os.path.splitext(cand)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        c2 = root + ext
        if os.path.exists(c2):
            return c2
    return cand

def load_image_size(filepath: str):
    img = bpy.data.images.load(filepath, check_existing=True)
    w, h = img.size[0], img.size[1]
    return max(1, int(w)), max(1, int(h))

def scale_contain(img_w: int, img_h: int, out_w: int, out_h: int, safe_inset_pct: float) -> float:
    """
    'contain' fit: entire image visible; safe inset shrinks effective output
    so text isn't cropped near edges.
    """
    safe = clamp(float(safe_inset_pct), 0.0, 0.25)
    eff_w = out_w * (1.0 - 2.0 * safe)
    eff_h = out_h * (1.0 - 2.0 * safe)
    sx = eff_w / float(img_w)
    sy = eff_h / float(img_h)
    return min(sx, sy)

def scale_cover(img_w: int, img_h: int, out_w: int, out_h: int) -> float:
    sx = out_w / float(img_w)
    sy = out_h / float(img_h)
    return max(sx, sy)

def get_nested(d: dict, path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def has_camera_path(shot: dict) -> bool:
    cp = shot.get("camera_path")
    return isinstance(cp, dict) and isinstance(cp.get("keyframes"), list) and len(cp["keyframes"]) > 0


# -----------------------
# VSE strip creation
# -----------------------
def new_image_strip(scene, name, filepath, channel, frame_start, length):
    se = scene.sequence_editor
    st = se.strips.new_image(name=name, filepath=filepath, channel=channel, frame_start=frame_start)
    st.frame_final_duration = length
    return st

def new_sound_strip(scene, name, filepath, channel, frame_start):
    se = scene.sequence_editor
    try:
        st = se.strips.new_sound(name=name, filepath=filepath, channel=channel, frame_start=frame_start)
    except Exception:
        # some builds use 'sound' not 'new_sound'
        st = se.strips.new_sound(name, filepath, channel, frame_start)
    return st

def new_effect(scene, name, effect_type, channel, frame_start, length, input1, input2=None):
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
        return se.strips.new_effect(name, effect_type, channel, frame_start, length, input1, input2)

def set_transform_key(strip, frame, scale, offx_px, offy_px):
    bpy.context.scene.frame_set(frame)
    strip.transform.scale_x = float(scale)
    strip.transform.scale_y = float(scale)
    strip.transform.offset_x = float(offx_px)
    strip.transform.offset_y = float(offy_px)
    strip.transform.keyframe_insert(data_path="scale_x", frame=frame)
    strip.transform.keyframe_insert(data_path="scale_y", frame=frame)
    strip.transform.keyframe_insert(data_path="offset_x", frame=frame)
    strip.transform.keyframe_insert(data_path="offset_y", frame=frame)


# -----------------------
# Background blur (from plan.motion.bg_fill)
# -----------------------
def add_blurred_bg(scene, fg_path, name_prefix, channel_base, fs, length, out_w, out_h,
                   blur_size: float, dim: float):
    bg_img_w, bg_img_h = load_image_size(fg_path)
    base = scale_cover(bg_img_w, bg_img_h, out_w, out_h)
    bg_scale = base * 1.10

    bg = new_image_strip(scene, f"{name_prefix}_BG", fg_path, channel_base, fs, length)
    set_transform_key(bg, fs, bg_scale, 0.0, 0.0)
    set_transform_key(bg, fs + length, bg_scale, 0.0, 0.0)

    bl = new_effect(scene, f"{name_prefix}_BLUR", "GAUSSIAN_BLUR", channel_base + 1, fs, length, bg, None)
    if hasattr(bl, "size_x"):
        bl.size_x = float(blur_size)
        bl.size_y = float(blur_size)
    elif hasattr(bl, "factor"):
        bl.factor = 1.0

    if dim and dim > 0:
        col = new_effect(scene, f"{name_prefix}_DIM", "COLOR", channel_base + 2, fs, length, None, None)
        if hasattr(col, "color"):
            col.color = (0.0, 0.0, 0.0)
            col.blend_type = 'ALPHA_OVER'
            col.blend_alpha = clamp(float(dim), 0.0, 0.9)
        dimmed = new_effect(scene, f"{name_prefix}_ALPHA", "ALPHA_OVER", channel_base + 3, fs, length, col, bl)
        return dimmed

    return bl


# -----------------------
# Motion application (camera_path / motion)
# -----------------------
def _plan_zoom_cap(shot: dict, global_cap: float, text_cap: float) -> float:
    cam = shot.get("camera") or {}
    avoid_text_zoom = bool(cam.get("avoid_text_zoom", False)) if isinstance(cam, dict) else False
    plan_max_zoom = float(cam.get("max_zoom", global_cap)) if isinstance(cam, dict) and cam.get("max_zoom") is not None else global_cap
    hard = min(float(global_cap), float(plan_max_zoom))
    return min(hard, float(text_cap)) if avoid_text_zoom else hard

def _base_scale_from_fit(img_w, img_h, out_w, out_h, shot: dict, default_fit: str, default_inset: float) -> float:
    fg_fit = get_nested(shot, "motion.fg_fit", {}) or {}
    if not isinstance(fg_fit, dict):
        fg_fit = {}
    mode = str(fg_fit.get("mode") or default_fit).strip().lower()
    inset = fg_fit.get("safe_inset_pct")
    inset = float(inset) if inset is not None else float(default_inset)

    if mode == "cover":
        return scale_cover(img_w, img_h, out_w, out_h)
    return scale_contain(img_w, img_h, out_w, out_h, inset)

def apply_motion_from_plan(strip, fs, length, img_w, img_h, base_scale, shot: dict, zoom_cap: float, pan_cap_px: float):
    """
    Deterministic motion based on shot.motion:
      zoom.start/end, start_bias/end_bias, strength
    Bias coordinates are in normalized space [-1..1] roughly (from planner).
    We interpret bias as "where to look" and convert to pixel offsets.
    """
    motion = shot.get("motion") or {}
    if not isinstance(motion, dict):
        return False

    strength = float(motion.get("strength", 0.75))
    z = motion.get("zoom") or {}
    if not isinstance(z, dict):
        z = {}

    z0 = float(z.get("start", 1.02))
    z1 = float(z.get("end", 1.10))
    z0 = clamp(z0, 1.0, zoom_cap)
    z1 = clamp(z1, 1.0, zoom_cap)

    sb = motion.get("start_bias") or {"x": 0.0, "y": 0.0}
    eb = motion.get("end_bias") or {"x": 0.0, "y": 0.0}
    if not isinstance(sb, dict):
        sb = {"x": 0.0, "y": 0.0}
    if not isinstance(eb, dict):
        eb = {"x": 0.0, "y": 0.0}

    # Convert bias [-1..1] into pixel offsets.
    # Scale offsets by strength and pan_cap_px.
    def bias_to_offset(b):
        bx = float(b.get("x", 0.0))
        by = float(b.get("y", 0.0))
        ox = clamp(bx * pan_cap_px * strength, -pan_cap_px, pan_cap_px)
        oy = clamp(by * pan_cap_px * strength, -pan_cap_px, pan_cap_px)
        return ox, oy

    ox0, oy0 = bias_to_offset(sb)
    ox1, oy1 = bias_to_offset(eb)

    s0 = base_scale * z0
    s1 = base_scale * z1

    set_transform_key(strip, fs, s0, ox0, oy0)
    set_transform_key(strip, fs + length, s1, ox1, oy1)
    return True

def apply_camera_path_keyframes(strip, fs, length, img_w, img_h, base_scale, cam_path, zoom_cap, pan_cap_px):
    kfs = cam_path.get("keyframes") or []
    if not isinstance(kfs, list) or not kfs:
        return False

    def _t(k):
        try:
            return float(k.get("t", 0.0))
        except Exception:
            return 0.0

    kfs = sorted(kfs, key=_t)

    for k in kfs:
        t = float(k.get("t", 0.0))
        cx = float(k.get("cx", 0.5))
        cy = float(k.get("cy", 0.5))
        z  = float(k.get("zoom", 1.05))
        z  = clamp(z, 1.0, zoom_cap)

        frame = fs + int(round(clamp(t, 0.0, 1.0) * length))
        scale = base_scale * z

        # Center on (cx,cy) in image space
        dx = (cx - 0.5) * (img_w * scale)
        dy = (cy - 0.5) * (img_h * scale)

        ox = -dx
        oy = +dy  # if vertical looks inverted in your renders, flip sign here

        ox = clamp(ox, -pan_cap_px, pan_cap_px)
        oy = clamp(oy, -pan_cap_px, pan_cap_px)

        set_transform_key(strip, frame, scale, ox, oy)

    return True

def apply_fallback_gentle(strip, fs, length, base_scale, zoom_cap):
    # Deterministic mild push-in
    z0 = clamp(1.03, 1.0, zoom_cap)
    z1 = clamp(1.10, 1.0, zoom_cap)
    set_transform_key(strip, fs, base_scale * z0, 0.0, 0.0)
    set_transform_key(strip, fs + length, base_scale * z1, 0.0, 0.0)
    return True


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()

    fps = int(args.fps)
    out_w = int(args.width)
    out_h = int(args.height)

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

    BG_CH = int(args.bg_channel)
    FG_CH = int(args.fg_channel)
    SFX_CH = int(args.sfx_channel)

    global_max_zoom = float(args.max_zoom_cap)
    text_zoom_cap = float(args.text_zoom_cap)
    pan_cap_px = float(args.pan_cap_frac) * float(min(out_w, out_h))

    if args.verbose:
        print(f"[INFO] items={len(timeline)} fps={fps} res={out_w}x{out_h} pan_cap_px={pan_cap_px:.1f}")

    max_frame_end = 1
    strips_count = 0
    sound_count = 0

    # Ensure we don't overlap strips by rounding collisions
    last_used_frame_end_by_channel = {}

    def channel_safe_start(ch: int, fs: int) -> int:
        prev = last_used_frame_end_by_channel.get(ch, 0)
        return max(fs, prev + 1) if prev >= fs else fs

    def mark_channel_end(ch: int, fe: int):
        last_used_frame_end_by_channel[ch] = max(last_used_frame_end_by_channel.get(ch, 0), fe)

    for idx, shot in enumerate(timeline):
        if not isinstance(shot, dict):
            continue

        start_sec = float(shot.get("start_sec") or 0.0)
        shot_dur = float(shot.get("duration_sec") or 3.0)
        if shot_dur <= 0.0:
            continue

        shot_fs = sec_to_frame_start(start_sec, fps)
        shot_len = seconds_to_frames(shot_dur, fps)

        # SOUND: align narration to shot start
        tts_audio = shot.get("tts_audio") or ""
        if tts_audio:
            tts_path = resolve_path(os.path.dirname(os.path.abspath(args.plan)), str(tts_audio))
            if os.path.exists(tts_path):
                sfs = channel_safe_start(SFX_CH, shot_fs)
                st = new_sound_strip(scene, f"SND_{shot_fs:06d}_{idx:04d}", tts_path, SFX_CH, sfs)
                sound_count += 1
                # we do not force duration; Blender will use audio length
                mark_channel_end(SFX_CH, st.frame_final_end)

        # Montage driven by cuts[]
        cuts = shot.get("cuts") or []
        if not isinstance(cuts, list) or not cuts:
            # fallback to scene_files if cuts absent
            files = shot.get("scene_files") or []
            if isinstance(files, list) and files:
                cuts = [{"file": files[0], "start": 0.0, "dur": shot_dur}]

        zoom_cap = _plan_zoom_cap(shot, global_max_zoom, text_zoom_cap)

        motion_bg = get_nested(shot, "motion.bg_fill", {}) or {}
        if not isinstance(motion_bg, dict):
            motion_bg = {}

        bg_enabled = bool(motion_bg.get("enabled", args.bg_blur)) if motion_bg else bool(args.bg_blur)
        bg_amount = float(motion_bg.get("amount", args.bg_blur_size)) if motion_bg else float(args.bg_blur_size)
        bg_dim = float(motion_bg.get("dim", args.bg_dim)) if motion_bg else float(args.bg_dim)

        # Render each cut at absolute frame = shot_fs + cut.start
        for ci, c in enumerate(cuts):
            if not isinstance(c, dict):
                continue

            rel_start = float(c.get("start") or 0.0)
            rel_dur = float(c.get("dur") or 0.0)
            if rel_dur <= 0.0:
                continue

            file_ref = str(c.get("file") or "")
            img_path = resolve_path(args.scene_dir, file_ref)
            if not img_path or not os.path.exists(img_path):
                continue

            cut_fs = shot_fs + seconds_to_frames(rel_start, fps)
            cut_len = seconds_to_frames(rel_dur, fps)

            # Channel collision safety
            cut_fs = channel_safe_start(FG_CH, cut_fs)

            # Background blur — channel_base must be the BG *channel*, never a
            # frame number: channel_safe_start() returns frames, and passing it
            # here once parked the whole blur+dim stack on channel 128 (the
            # clamp ceiling), ABOVE the foreground -> every frame rendered black.
            if bg_enabled:
                add_blurred_bg(
                    scene, img_path,
                    name_prefix=f"BG_{shot_fs:06d}_{idx:04d}_{ci:02d}",
                    channel_base=BG_CH,
                    fs=cut_fs,
                    length=cut_len,
                    out_w=out_w,
                    out_h=out_h,
                    blur_size=bg_amount,
                    dim=bg_dim,
                )
                # mark BG end (roughly; blur stack uses +3 channels)
                mark_channel_end(BG_CH, cut_fs + cut_len)
                mark_channel_end(BG_CH + 1, cut_fs + cut_len)
                mark_channel_end(BG_CH + 2, cut_fs + cut_len)
                mark_channel_end(BG_CH + 3, cut_fs + cut_len)

            fg = new_image_strip(scene, f"FG_{shot_fs:06d}_{idx:04d}_{ci:02d}", img_path, FG_CH, cut_fs, cut_len)

            img_w, img_h = load_image_size(img_path)
            base_scale = _base_scale_from_fit(
                img_w, img_h, out_w, out_h,
                shot=shot,
                default_fit=args.default_fg_fit,
                default_inset=args.default_safe_inset,
            )

            # Deterministic motion application order:
            used = False
            if has_camera_path(shot):
                used = apply_camera_path_keyframes(
                    fg, cut_fs, cut_len,
                    img_w, img_h,
                    base_scale,
                    shot["camera_path"],
                    zoom_cap=zoom_cap,
                    pan_cap_px=pan_cap_px,
                )

            # Per-cut motion (the panel's pan ends on ITS OWN face) overrides the
            # shot-level default; parity with remotion/src/Shot.tsx. A held or
            # substituted cut carries no own motion -> falls back to shot.motion.
            cut_motion = c.get("motion") if isinstance(c.get("motion"), dict) else None
            eff_shot = {**shot, "motion": cut_motion} if cut_motion else shot
            if (not used) and isinstance(eff_shot.get("motion"), dict):
                used = apply_motion_from_plan(
                    fg, cut_fs, cut_len,
                    img_w, img_h,
                    base_scale,
                    shot=eff_shot,
                    zoom_cap=zoom_cap,
                    pan_cap_px=pan_cap_px,
                )

            if not used:
                apply_fallback_gentle(fg, cut_fs, cut_len, base_scale, zoom_cap)

            strips_count += 1
            mark_channel_end(FG_CH, fg.frame_final_end)
            max_frame_end = max(max_frame_end, fg.frame_final_end)

            if args.verbose:
                motion_src = "camera_path" if has_camera_path(shot) else ("motion" if isinstance(shot.get("motion"), dict) else "fallback")
                print(f"[ADD] shot#{idx:04d} cut#{ci:02d} fs={cut_fs} len={cut_len} img={os.path.basename(img_path)} zoom_cap={zoom_cap:.2f} motion={motion_src}")

        if args.verbose and (idx == 0 or (idx + 1) % 10 == 0):
            print(f"[INFO] progress {idx+1}/{len(timeline)} max_frame_end={max_frame_end}")

    scene.frame_start = 1
    scene.frame_end = max(1, int(max_frame_end))

    bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)

    print(f"[OK] Saved blend: {args.out_blend}")
    print(f"[OK] Timeline frames: {scene.frame_end} seconds={scene.frame_end/fps:.2f} strips={strips_count} sounds={sound_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
