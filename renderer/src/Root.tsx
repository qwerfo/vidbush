import React from "react";
import { Composition } from "remotion";
import { COMPONENTS } from "./compositions";
import { DEFAULT_SPEC, KINDS, Props } from "./types";

/** One composition per MotionKind; size, fps and length come from the MotionSpec input props. */
export const Root: React.FC = () => (
  <>
    {KINDS.map((kind) => (
      <Composition
        key={kind}
        id={kind.replace(/_/g, "-")}
        component={COMPONENTS[kind] as React.FC<Props>}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={90}
        defaultProps={{ spec: { ...DEFAULT_SPEC, kind } }}
        calculateMetadata={({ props }) => {
          const s = (props as Props).spec;
          return {
            width: s.width,
            height: s.height,
            fps: s.fps,
            durationInFrames: Math.max(1, Math.round(s.duration * s.fps)),
          };
        }}
      />
    ))}
  </>
);
