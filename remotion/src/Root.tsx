import React from 'react';
import {Composition} from 'remotion';
import {FPS, HEIGHT, RenderPlan, toFrames, toStartFrame, WIDTH} from './plan';
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
        // End exactly where the last item's Sequence ends. ceil(total_sec*FPS)
        // ran one frame past it in ~12% of chapters — a black frame at the
        // chapter's end, i.e. at every chapter seam of a bundle.
        const frames = last
          ? toStartFrame(last.start_sec) + toFrames(last.duration_sec)
          : toFrames(props.total_duration_sec ?? 10);
        return {durationInFrames: Math.max(1, frames)};
      }}
    />
  );
};
