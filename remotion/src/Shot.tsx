import React from 'react';
import {AbsoluteFill, Audio, Sequence, staticFile} from 'remotion';
import {CutView} from './Cut';
import {publicRelAudio, SceneDims, TimelineItem, toFrames, toStartFrame} from './plan';

/**
 * One timeline item (= one narration group): its narration audio at the shot
 * start, and its cuts[] montage at their planner-given offsets/durations. A
 * beat tagged segment="flashback"/"dream" (by story_group) gets a faded sepia +
 * vignette look so the time-shift READS on screen, not just in the words.
 */
export const Shot: React.FC<{
  item: TimelineItem;
  scenesSubdir: string;
  sceneDims: Record<string, SceneDims>;
}> = ({item, scenesSubdir, sceneDims}) => {
  const cuts =
    item.cuts && item.cuts.length > 0
      ? item.cuts
      : item.scene_files && item.scene_files.length > 0
        ? [{file: item.scene_files[0], start: 0, dur: item.duration_sec}]
        : [];

  const flashback = item.segment === 'flashback' || item.segment === 'dream';

  const body = (
    <>
      {item.tts_audio ? <Audio src={staticFile(publicRelAudio(item.tts_audio))} /> : null}
      {cuts.map((c, i) => (
        <Sequence
          key={`${item.segment_id}_c${i}`}
          from={toStartFrame(c.start)}
          durationInFrames={toFrames(c.dur)}
        >
          <CutView
            file={c.file}
            file2={c.layout === 'split2' ? c.file2 : undefined}
            durationInFrames={toFrames(c.dur)}
            // Per-panel motion (its pan ends on THIS panel's face) when the
            // planner emitted one; else the shot-level default.
            motion={c.motion ?? item.motion}
            camera={item.camera}
            scenesSubdir={scenesSubdir}
            dims={sceneDims[c.file]}
          />
        </Sequence>
      ))}
    </>
  );

  if (!flashback) return body;

  // Flashback look: the panels tint to a faded sepia; a soft vignette darkens
  // the edges. Applied per-beat so the whole flashback run reads consistently.
  return (
    <AbsoluteFill style={{filter: 'sepia(0.5) saturate(0.62) brightness(0.9) contrast(0.96)'}}>
      {body}
      <AbsoluteFill
        style={{
          pointerEvents: 'none',
          boxShadow: 'inset 0 0 220px 48px rgba(24,14,6,0.7)',
        }}
      />
    </AbsoluteFill>
  );
};
