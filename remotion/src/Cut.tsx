import React from 'react';
import {AbsoluteFill, Easing, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {Camera, clamp, DEFAULT_SAFE_INSET, Motion, PAN_CAP_FRAC, SceneDims, TALL_SCROLL_MIN_ASPECT, WIDE_COVER_MIN_ASPECT, zoomCap} from './plan';

/**
 * One panel on screen with deterministic Ken Burns (zoom start→end, pan
 * start_bias→end_bias). Wide panels (aspect ≥ WIDE_COVER_MIN_ASPECT, known
 * from render_prep's scene_dims) render FULL-BLEED (cover, no margins);
 * everything else keeps the blurred-cover background + contained foreground —
 * the same semantics tools/blender_vse_from_plan.py applies to its strips.
 */
export const CutView: React.FC<{
  file: string;
  file2?: string;
  durationInFrames: number;
  motion?: Motion;
  camera?: Camera;
  scenesSubdir?: string;
  dims?: SceneDims;
}> = ({file, file2, durationInFrames, motion, camera, scenesSubdir = 'scenes', dims}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const src = staticFile(`${scenesSubdir}/${file}`);
  const doc = !!dims?.doc;
  const wide = !doc && !!dims && dims.h > 0 && dims.w / dims.h >= WIDE_COVER_MIN_ASPECT;
  const tall = !doc && !!dims && dims.w > 0 && dims.h / dims.w >= TALL_SCROLL_MIN_ASPECT;

  const cap = zoomCap(camera);
  const z0 = clamp(motion?.zoom?.start ?? 1.02, 1.0, cap);
  const z1 = clamp(motion?.zoom?.end ?? 1.1, 1.0, cap);
  const strength = motion?.strength ?? 0.75;
  const panCapPx = PAN_CAP_FRAC * Math.min(width, height);

  const t = interpolate(frame, [0, Math.max(1, durationInFrames - 1)], [0, 1], {
    easing: Easing.inOut(Easing.ease),
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const biasOffset = (b?: {x?: number; y?: number}) => ({
    x: clamp((b?.x ?? 0) * panCapPx * strength, -panCapPx, panCapPx),
    y: clamp((b?.y ?? 0) * panCapPx * strength, -panCapPx, panCapPx),
  });
  const o0 = biasOffset(motion?.start_bias);
  const o1 = biasOffset(motion?.end_bias);

  const zoom = z0 + (z1 - z0) * t;
  const ox = o0.x + (o1.x - o0.x) * t;
  // Blender's offset_y is up-positive; CSS translateY is down-positive.
  const oy = -(o0.y + (o1.y - o0.y) * t);

  const bg = motion?.bg_fill ?? {};
  const bgEnabled = bg.enabled ?? true;
  // Blender gaussian size ~35 ≈ a CSS blur of roughly half that radius.
  const bgBlurPx = (bg.amount ?? 35) * 0.5;
  const bgDim = bg.dim ?? 0.12;

  const inset = motion?.fg_fit?.safe_inset_pct ?? DEFAULT_SAFE_INSET;
  const boxPct = (1 - 2 * inset) * 100;

  if (tall && dims) {
    // Tall panels are complete cuts, not footage to crop into a 16:9 close-up.
    // Keep the whole panel visible and let the blurred fill carry the frame —
    // but DON'T sit still: drift the panel vertically (a gentle scroll shot)
    // from the top toward the planner's focus_y so the eye travels the strip.
    // The foreground is scaled up just enough to give vertical headroom for the
    // drift; the clip's overflow:hidden trims the off-frame sliver, and the
    // travel is slow + small so the whole panel reads over the cut's duration.
    const tallScale = 1.12;
    const fy = clamp(motion?.focus_y ?? 0.5, 0, 1);
    // Vertical headroom (px) created by the upscale, on a contain-fit panel that
    // already fills the frame height. Drift from the top of that headroom down
    // toward focus_y (a centered focus → a slow, even downward drift).
    const tallTravel = ((tallScale - 1) * height) / 2;
    const tallY0 = tallTravel; // start: top of the panel in view
    const tallY1 = tallTravel * (1 - 2 * fy); // end: biased toward focus_y
    const tallY = tallY0 + (tallY1 - tallY0) * t;
    return (
      <AbsoluteFill style={{backgroundColor: '#000', overflow: 'hidden'}}>
        <Img
          src={src}
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: 'scale(1.08)',
            filter: `blur(${bgBlurPx}px) brightness(${1 - bgDim})`,
          }}
        />
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', overflow: 'hidden'}}>
          <Img
            src={src}
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              transform: `translateY(${tallY}px) scale(${tallScale})`,
            }}
          />
        </AbsoluteFill>
      </AbsoluteFill>
    );
  }

  if (file2) {
    // split2: two halves of an over-merged crop, side by side, shared motion.
    const src2 = staticFile(`${scenesSubdir}/${file2}`);
    return (
      <AbsoluteFill style={{backgroundColor: '#000', overflow: 'hidden'}}>
        <Img
          src={src}
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: 'scale(1.1)',
            filter: `blur(${bgBlurPx}px) brightness(${1 - bgDim})`,
          }}
        />
        <AbsoluteFill
          style={{
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 24,
            padding: 36,
            transform: `scale(${zoom})`,
          }}
        >
          {[src, src2].map((s) => (
            <Img
              key={s}
              src={s}
              style={{
                maxWidth: '48%',
                maxHeight: '92%',
                objectFit: 'contain',
                borderRadius: 6,
                boxShadow: '0 10px 40px rgba(0,0,0,0.45)',
              }}
            />
          ))}
        </AbsoluteFill>
      </AbsoluteFill>
    );
  }

  if (wide) {
    // Wide/full-screen panels must stay readable as complete panels. Cover-crop
    // plus Ken Burns was cutting off the intended composition, so the wide path
    // uses a blurred fill behind a contained foreground — but with a GENTLE
    // horizontal drift (left→right) so the panel isn't motionless. The upscale
    // gives just enough horizontal headroom for the pan; overflow:hidden on the
    // clip trims the off-frame sliver, and the travel is small + slow so the
    // whole panel still reads across the cut's duration.
    const wideScale = 1.12;
    // Horizontal headroom (px) from the upscale on a contain-fit panel that
    // already fills the frame width. Drift from left edge toward the right.
    const wideTravel = ((wideScale - 1) * width) / 2;
    const wideX = interpolate(t, [0, 1], [wideTravel, -wideTravel]);
    return (
      <AbsoluteFill style={{backgroundColor: '#000', overflow: 'hidden'}}>
        <Img
          src={src}
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: 'scale(1.08)',
            filter: `blur(${bgBlurPx}px) brightness(${1 - bgDim})`,
          }}
        />
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', overflow: 'hidden'}}>
          <Img
            src={src}
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              transform: `translateX(${wideX}px) scale(${wideScale})`,
            }}
          />
        </AbsoluteFill>
      </AbsoluteFill>
    );
  }

  return (
    <AbsoluteFill style={{backgroundColor: '#000', overflow: 'hidden'}}>
      {bgEnabled ? (
        <Img
          src={src}
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: 'scale(1.1)',
            filter: `blur(${bgBlurPx}px) brightness(${1 - bgDim})`,
          }}
        />
      ) : null}
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <Img
          src={src}
          style={{
            maxWidth: `${boxPct}%`,
            maxHeight: `${boxPct}%`,
            objectFit: 'contain',
            transform: `translate(${ox}px, ${oy}px) scale(${zoom})`,
          }}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
