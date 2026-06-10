import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  interpolate,
  Loop,
  OffthreadVideo,
  random,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import introWav from '../assets/intro.wav';
import logo from '../assets/logo.png';
import outroWav from '../assets/outro.wav';
import particles from '../assets/particles.mp4';
import watermark from '../assets/watermark.png';

/**
 * Pre-rendered particle overlay (tools/particle_overlay.py): gaussian bokeh
 * sprites with depth, sub-frame motion blur and integer-cycle turbulence,
 * rendered on black as a seamless 16s loop and SCREEN-blended here — the
 * standard technique behind cinematic dust/snow overlays.
 */
export const ParticleOverlay: React.FC<{opacity?: number}> = ({opacity = 0.8}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{pointerEvents: 'none', mixBlendMode: 'screen', opacity}}>
      <Loop durationInFrames={Math.round(16 * fps)}>
        <OffthreadVideo
          muted
          src={particles}
          style={{width: '100%', height: '100%', objectFit: 'cover'}}
        />
      </Loop>
    </AbsoluteFill>
  );
};

/** Semi-transparent channel watermark + minimal wordmark, bottom-right. */
export const Watermark: React.FC<{opacity?: number}> = ({opacity = 0.5}) => (
  <div
    style={{
      position: 'absolute',
      right: 28,
      bottom: 24,
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      opacity,
      pointerEvents: 'none',
    }}
  >
    <div
      style={{
        textAlign: 'right',
        fontFamily: 'Avenir Next, Helvetica, sans-serif',
        color: 'white',
        textShadow: '0 1px 6px rgba(0,0,0,0.55)',
        lineHeight: 1.25,
      }}
    >
      <div style={{fontSize: 19, fontWeight: 700, letterSpacing: 2.5}}>ORIGIN POWER</div>
      <div style={{fontSize: 13, fontWeight: 500, letterSpacing: 4.2, opacity: 0.85}}>
        MANHWA RECAP
      </div>
    </div>
    <Img src={watermark} style={{width: 64, height: 64}} />
  </div>
);

/**
 * Channel intro moment: the story panel stays on screen (rendered by Shot
 * underneath); this overlay adds a gentle dim, the logo and channel name
 * sliding in, and the narrator's welcome line — then the story resumes.
 */
export const IntroOverlay: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 200}});
  const fadeOut = interpolate(frame, [0, 8], [0, 1], {
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <Audio src={introWav} />
      <AbsoluteFill style={{backgroundColor: `rgba(10, 4, 20, ${0.35 * fadeOut})`}} />
      <div
        style={{
          position: 'absolute',
          bottom: 120,
          left: 0,
          right: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 28,
          transform: `translateY(${(1 - enter) * 60}px)`,
          opacity: enter,
        }}
      >
        <Img src={logo} style={{width: 110, height: 110, borderRadius: 24}} />
        <div style={{fontFamily: 'Avenir Next, Helvetica, sans-serif', color: 'white'}}>
          <div style={{fontSize: 52, fontWeight: 700, letterSpacing: 0.5}}>
            OriginPower Manhwa Recap
          </div>
          <div style={{fontSize: 26, opacity: 0.85, marginTop: 6}}>
            New chapters every week — enjoy the story
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

/**
 * End card: purple gradient (the channel's thumbnail tone), logo spring-in,
 * staggered Like / Subscribe / Comment prompts, and the narrator's outro line.
 * Fixes the "video just stops" ending.
 */
export const EndCard: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const logoIn = spring({frame, fps, config: {damping: 14, mass: 0.8}});
  const row = (i: number) =>
    spring({frame: frame - 12 - i * 10, fps, config: {damping: 200}});
  const prompts = ['👍  Drop a like', '🔔  Subscribe for the next chapter', '💬  Share your thoughts in the comments'];
  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(140deg, #1b0b2e 0%, #4a1d7a 55%, #7a2fb8 100%)',
        justifyContent: 'center',
        alignItems: 'center',
        fontFamily: 'Avenir Next, Helvetica, sans-serif',
      }}
    >
      <Audio src={outroWav} />
      <Img
        src={logo}
        style={{
          width: 190,
          height: 190,
          borderRadius: 38,
          transform: `scale(${logoIn})`,
          boxShadow: '0 18px 60px rgba(0,0,0,0.45)',
        }}
      />
      <div style={{color: 'white', fontSize: 56, fontWeight: 800, marginTop: 38}}>
        Thanks for watching!
      </div>
      <div style={{marginTop: 34, display: 'flex', flexDirection: 'column', gap: 18}}>
        {prompts.map((p, i) => (
          <div
            key={p}
            style={{
              color: 'white',
              fontSize: 34,
              opacity: row(i),
              transform: `translateX(${(1 - row(i)) * 40}px)`,
            }}
          >
            {p}
          </div>
        ))}
      </div>
      <div style={{color: 'rgba(255,255,255,0.75)', fontSize: 26, marginTop: 44}}>
        Next chapter coming soon…
      </div>
    </AbsoluteFill>
  );
};

/**
 * Ambient drift — soft luminous motes (snow/dust) in three parallax depth
 * layers. What makes it read premium instead of cheesy: radial-gradient
 * soft edges (never hard shapes), depth scaling (near = bigger, faster,
 * blurrier; far = tiny, slow, dim), per-mote twinkle, and wind built from
 * layered sines instead of linear motion. Deterministic via random(seed).
 */
type Layer = {n: number; size: number; speed: number; blur: number; alpha: number};
const LAYERS: Layer[] = [
  {n: 9, size: 3.5, speed: 0.55, blur: 0.5, alpha: 0.5}, // far
  {n: 7, size: 6.5, speed: 0.85, blur: 1.5, alpha: 0.75}, // mid
  {n: 5, size: 11, speed: 1.25, blur: 3.5, alpha: 1.0}, // near
];

export const AmbientDrift: React.FC<{seed?: string; intensity?: number}> = ({
  seed = 'motes',
  intensity = 0.55,
}) => {
  const frame = useCurrentFrame();
  const {width, height, fps} = useVideoConfig();
  const t = frame / fps;

  const motes: React.ReactNode[] = [];
  LAYERS.forEach((layer, li) => {
    for (let i = 0; i < layer.n; i++) {
      const k = `${seed}-${li}-${i}`;
      const rx = random(`${k}x`);
      const rv = random(`${k}v`);
      const rp = random(`${k}p`);
      const rz = random(`${k}z`);

      const fallSec = (30 - 12 * (layer.speed - 0.55)) * (0.8 + rv * 0.4);
      const prog = ((t / fallSec + rp) % 1 + 1) % 1;
      const y = prog * (height + 80) - 40;
      // wind: two incommensurate sines per mote = organic, non-looping drift
      const sway =
        Math.sin(t * (0.18 + rv * 0.22) + rp * 6.28) * 55 * layer.speed +
        Math.sin(t * (0.53 + rz * 0.31) + rx * 6.28) * 18;
      const x = rx * (width + 120) - 60 + sway + t * 6 * layer.speed; // gentle side wind
      const xw = ((x % (width + 120)) + (width + 120)) % (width + 120) - 60;

      const twinkle = 0.75 + 0.25 * Math.sin(t * (0.6 + rz * 0.9) + rp * 6.28);
      const size = layer.size * (0.8 + rz * 0.5);
      motes.push(
        <div
          key={k}
          style={{
            position: 'absolute',
            left: xw,
            top: y,
            width: size * 2.6,
            height: size * 2.6,
            background:
              'radial-gradient(circle, rgba(255,250,246,0.95) 0%, rgba(255,250,246,0.35) 45%, rgba(255,250,246,0) 72%)',
            filter: `blur(${layer.blur}px)`,
            opacity: intensity * layer.alpha * twinkle,
          }}
        />,
      );
    }
  });

  return <AbsoluteFill style={{pointerEvents: 'none'}}>{motes}</AbsoluteFill>;
};
