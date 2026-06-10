import React from 'react';
import {AbsoluteFill, Sequence} from 'remotion';
import {RenderPlan, toFrames, toStartFrame} from './plan';
import {Shot} from './Shot';

export const RecapVideo: React.FC<RenderPlan> = ({timeline, scenes_subdir, scene_dims}) => {
  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      {(timeline ?? []).map((item) => (
        <Sequence
          key={item.segment_id}
          from={toStartFrame(item.start_sec)}
          durationInFrames={toFrames(item.duration_sec)}
        >
          <Shot
            item={item}
            scenesSubdir={scenes_subdir ?? 'scenes'}
            sceneDims={scene_dims ?? {}}
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
