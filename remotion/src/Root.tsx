import React from 'react';
import {Composition} from 'remotion';
import {FPS, HEIGHT, RenderPlan, WIDTH} from './plan';
import {RecapVideo} from './RecapVideo';

// Duration comes from the plan passed via --props=render.plan.json.
export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="RecapVideo"
      component={RecapVideo}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      durationInFrames={300}
      defaultProps={{timeline: [], total_duration_sec: 10} as RenderPlan}
      calculateMetadata={({props}) => {
        const last = props.timeline?.[props.timeline.length - 1];
        const totalSec = props.total_duration_sec ?? last?.end_sec ?? 10;
        return {durationInFrames: Math.max(1, Math.ceil(totalSec * FPS))};
      }}
    />
  );
};
