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
    // COVER-CROP ZOOM: a tall strip FILLS the frame with a window centred on the
    // CONTENT (motion.focus_y — the art, off the inpainted-blank bubble) and slowly
    // pushes in. NO scroll: the old top->down scroll drifted onto the blank bubble
    // and looked identical on every tall panel. focus_y defaults upper-middle.
    const focusY = clamp(motion?.focus_y ?? 0.4, 0, 1);
    const prog = interpolate(frame, [0, Math.max(1, durationInFrames)], [0, 1], {
      easing: Easing.inOut(Easing.ease),
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    const zoom = 1.04 + 0.09 * prog; // gentle Ken Burns push-in
    const dispW = width * zoom; // fill the frame width (cover)
    const dispH = dispW * (dims.h / dims.w); // taller than the frame
    // centre focusY vertically, clamped so the window never reveals empty edges
    const ty = clamp(height / 2 - focusY * dispH, height - dispH, 0);
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
        <AbsoluteFill style={{justifyContent: 'flex-start', alignItems: 'center', overflow: 'hidden'}}>
          <Img
            src={src}
            style={{
              position: 'absolute',
              top: 0,
              left: '50%',
              width: dispW,
              transform: `translateX(-50%) translateY(${ty}px)`,
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
