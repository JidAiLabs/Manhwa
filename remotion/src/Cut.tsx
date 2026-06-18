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
    // SCROLL SHOT: tall strips are unreadable contain-fitted — display at a
    // readable width and travel the camera vertically across the artwork,
    // easing in and HOLDING the final view for the last ~15% of the cut.
    // Default reads downward (webtoon order); tilt_up beats scroll upward.
    // readable width, but cap total travel at ~2.4 screen-heights so the
    // scroll speed stays watchable on very long strips
    const maxScaledH = height * 3.4;
    const dispW = Math.min(width * 0.92, maxScaledH * (dims.w / dims.h));
    const scaledH = dispW * (dims.h / dims.w);
    const travel = Math.max(0, scaledH - height);
    const prog = interpolate(frame, [0, Math.max(1, durationInFrames * 0.85)], [0, 1], {
      easing: Easing.inOut(Easing.ease),
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    // If this tall strip carries a FACE target, SETTLE the scroll so the face
    // lands vertically centered and HOLDS there — fixes the "starts on a face
    // then drifts down off it" complaint. Otherwise scroll in reading order
    // (down; a tilt_up beat scrolls up).
    const up = (motion?.mode ?? '') === 'tilt_up';
    let y: number;
    if (motion?.focus === 'face' && motion?.end_bias) {
      const faceCy = clamp(0.5 + (motion.end_bias.y ?? 0) * 0.5, 0, 1);
      const target = clamp(height / 2 - faceCy * scaledH, -travel, 0);
      y = target * prog; // ease from the top down to the face, then hold
    } else {
      y = up ? -travel * (1 - prog) : -travel * prog;
    }
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
        <AbsoluteFill style={{justifyContent: 'flex-start', alignItems: 'center'}}>
          <Img
            src={src}
            style={{
              width: dispW,
              transform: `translateY(${y}px)`,
              boxShadow: '0 0 50px rgba(0,0,0,0.5)',
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
    // Full-bleed: the panel IS the frame — no margins, no blur layer.
    return (
      <AbsoluteFill style={{backgroundColor: '#000', overflow: 'hidden'}}>
        <Img
          src={src}
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: `translate(${ox}px, ${oy}px) scale(${Math.max(zoom, 1.0)})`,
          }}
        />
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
