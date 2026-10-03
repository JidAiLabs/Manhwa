import React from 'react';
import {AbsoluteFill, Easing, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {SceneDims} from './plan';

/**
 * The 2-3 panels ONE narration line covers, side by side on one soft backdrop
 * (tools/render_prep.py assign_rows decides; cuts[] stays intact for QA).
 * Each panel fades into its FINAL slot with a short slide when the narration
 * reaches it (its first cut's start). Nothing swoops, grows or zooms, and a
 * panel already on screen never moves: the first prototype's travelling
 * panels and zooming backdrop read dizzy (owner, 2026-10-03). The row is
 * centred for the full count from the first frame, so the first panel sits
 * left until the others arrive, as in the reference channel.
 *
 * Geometry constants MUST mirror render_prep's _ROW_* (the fit rule that
 * decides which lines become rows measures against the same box).
 */
export const ROW_MX = 96;
export const ROW_MY = 54;
export const ROW_GAP = 28;
const ENTER_SEC = 0.5;
const SLIDE_PX = 70;
// The panel layer fades back to the backdrop over the last EXIT_FADE_SEC of
// the line (owner choice, 2026-10-03); 0 = hard cut as in the v2 prototype.
const EXIT_FADE_SEC = 0.4;

export const RowShot: React.FC<{
  row: {file: string; enter: number}[];
  durationInFrames: number;
  scenesSubdir: string;
  sceneDims: Record<string, SceneDims>;
}> = ({row, durationInFrames, scenesSubdir, sceneDims}) => {
  const frame = useCurrentFrame();
  const {width, height, fps} = useVideoConfig();
  const n = row.length;
  const ar = row.map((r) => sceneDims[r.file].w / sceneDims[r.file].h);
  const H = Math.min(
    height - 2 * ROW_MY,
    (width - 2 * ROW_MX - ROW_GAP * (n - 1)) / ar.reduce((s, a) => s + a, 0),
  );
  const ws = ar.map((a) => a * H);
  const rowW = ws.reduce((s, w) => s + w, 0) + ROW_GAP * (n - 1);
  const xs: number[] = [];
  let x = (width - rowW) / 2;
  ws.forEach((w) => {
    xs.push(x);
    x += w + ROW_GAP;
  });
  const top = (height - H) / 2;

  const exitFrames = Math.round(EXIT_FADE_SEC * fps);
  const exit =
    exitFrames > 0 && durationInFrames > exitFrames + 1
      ? interpolate(frame, [durationInFrames - exitFrames, durationInFrames - 1], [1, 0], {
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
        })
      : 1;

  return (
    <AbsoluteFill style={{backgroundColor: '#0b0d14', overflow: 'hidden'}}>
      <Img
        src={staticFile(`${scenesSubdir}/${row[0].file}`)}
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          transform: 'scale(1.3)',
          filter: 'blur(70px) brightness(0.78) saturate(1.15)',
        }}
      />
      <AbsoluteFill style={{opacity: exit}}>
        {row.map((r, i) => {
          const enter = Math.round(r.enter * fps);
          if (frame < enter) return null;
          const t = interpolate(frame, [enter, enter + Math.round(ENTER_SEC * fps)], [0, 1], {
            easing: Easing.out(Easing.cubic),
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          });
          return (
            <Img
              key={r.file}
              src={staticFile(`${scenesSubdir}/${r.file}`)}
              style={{
                position: 'absolute',
                left: xs[i] + (1 - t) * SLIDE_PX,
                top,
                width: ws[i],
                height: H,
                opacity: t,
                border: '3px solid #0a0a0a',
                boxShadow: '0 14px 44px rgba(0,0,0,0.55)',
              }}
            />
          );
        })}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
