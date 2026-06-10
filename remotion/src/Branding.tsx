import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  interpolate,
  random,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import introWav from '../assets/intro.wav';
import logo from '../assets/logo.png';
import outroWav from '../assets/outro.wav';
import watermark from '../assets/watermark.png';

/** Semi-transparent channel watermark, bottom-right, above everything. */
export const Watermark: React.FC<{opacity?: number}> = ({opacity = 0.4}) => (
  <Img
    src={watermark}
    style={{
      position: 'absolute',
      right: 28,
      bottom: 24,
      width: 84,
      height: 84,
      opacity,
      pointerEvents: 'none',
    }}
  />
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
 * Very minimal ambient drift — a handful of pale petals/snowflakes falling
 * slowly with a gentle sway. Deterministic (remotion random(seed)) so renders
 * are reproducible. Keep it barely-there: it adds life to static art without
 * stealing attention.
 */
const COUNT = 14;

export const AmbientDrift: React.FC<{seed?: string; opacity?: number}> = ({
  seed = 'petals',
  opacity = 0.22,
}) => {
  const frame = useCurrentFrame();
  const {width, height, fps} = useVideoConfig();

  const petals = Array.from({length: COUNT}, (_, i) => {
    const rx = random(`${seed}-x-${i}`);
    const rs = random(`${seed}-s-${i}`);
    const rp = random(`${seed}-p-${i}`);
    const rz = random(`${seed}-z-${i}`);
    const fallFrames = (22 + rs * 18) * fps; // 22–40s per screen height
    const progress = ((frame + rp * fallFrames) % fallFrames) / fallFrames;
    const sway = Math.sin((frame / fps) * (0.3 + rs * 0.4) + rp * Math.PI * 2) * 42;
    return {
      x: rx * width + sway,
      y: progress * (height + 60) - 30,
      size: 5 + rz * 7,
      opacity: opacity * (0.55 + rz * 0.45),
      rot: frame * (0.2 + rs * 0.5) + rp * 360,
    };
  });

  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      {petals.map((p, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: p.x,
            top: p.y,
            width: p.size,
            height: p.size * 0.8,
            borderRadius: '60% 40% 55% 45%',
            background: 'rgb(252, 246, 250)',
            opacity: p.opacity,
            transform: `rotate(${p.rot}deg)`,
          }}
        />
      ))}
    </AbsoluteFill>
  );
};
