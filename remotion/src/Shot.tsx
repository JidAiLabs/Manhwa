import React from 'react';
import {Audio, Sequence, staticFile} from 'remotion';
import {CutView} from './Cut';
import {publicRelAudio, TimelineItem, toFrames, toStartFrame} from './plan';

/**
 * One timeline item (= one narration group): its narration audio at the shot
 * start, and its cuts[] montage at their planner-given offsets/durations.
 */
export const Shot: React.FC<{item: TimelineItem}> = ({item}) => {
  const cuts =
    item.cuts && item.cuts.length > 0
      ? item.cuts
      : item.scene_files && item.scene_files.length > 0
        ? [{file: item.scene_files[0], start: 0, dur: item.duration_sec}]
        : [];

  return (
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
            durationInFrames={toFrames(c.dur)}
            motion={item.motion}
            camera={item.camera}
          />
        </Sequence>
      ))}
    </>
  );
};
