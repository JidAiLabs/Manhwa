import React from 'react';
import {AbsoluteFill, Sequence} from 'remotion';
import {RenderPlan, toFrames, toStartFrame} from './plan';
import {Shot} from './Shot';

export const RecapVideo: React.FC<RenderPlan> = ({timeline}) => {
  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      {(timeline ?? []).map((item) => (
        <Sequence
          key={item.segment_id}
          from={toStartFrame(item.start_sec)}
          durationInFrames={toFrames(item.duration_sec)}
        >
          <Shot item={item} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
