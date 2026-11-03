import React, { useEffect, useState } from "react";
import { AbsoluteFill, continueRender, delayRender, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { FONT, palette, px } from "../theme";
import { MotionSpec } from "../types";

const WEIGHTS: [string, number][] = [["Regular", 400], ["Medium", 500], ["SemiBold", 600], ["Bold", 700],
  ["ExtraBold", 800], ["Black", 900]];

/** Loads the bundled Inter weights once (blocks rendering until they are ready). */
const useFonts = (): void => {
  const [handle] = useState(() => delayRender("fonts"));
  useEffect(() => {
    Promise.all(WEIGHTS.map(([name, weight]) => {
      const f = new FontFace("Inter", `url(${staticFile(`fonts/Inter-${name}.otf`)})`, { weight: String(weight) });
      return f.load().then((loaded) => (document as any).fonts.add(loaded));
    })).finally(() => continueRender(handle));
  }, [handle]);
};

/** Background (gradient and vignette unless transparent), 0.3 s fade for full frame graphics, source note. */
export const Frame: React.FC<{ spec: MotionSpec; children: React.ReactNode; ownExit?: boolean }> = ({ spec, children, ownExit }) => {
  useFonts();
  const frame = useCurrentFrame();
  const { fps, durationInFrames, height } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const transparent = spec.background !== "solid";
  const f = Math.min(0.3 * fps, durationInFrames / 4);
  const fade = transparent && ownExit ? 1 : Math.min(
    interpolate(frame, [0, f], [0, 1], { extrapolateRight: "clamp" }),
    interpolate(frame, [durationInFrames - f, durationInFrames], [1, 0], { extrapolateLeft: "clamp" }));
  return (
    <AbsoluteFill style={{ background: transparent ? "transparent" : "black", fontFamily: FONT }}>
      <AbsoluteFill style={{ opacity: fade }}>
        {!transparent && (
          <AbsoluteFill style={{ background: `radial-gradient(ellipse at 50% 42%, ${pal.bg} 0%, ${pal.bg2} 100%)` }} />
        )}
        {!transparent && (
          <AbsoluteFill style={{ background: "radial-gradient(ellipse at center, rgba(0,0,0,0) 55%, rgba(0,0,0,0.35) 100%)" }} />
        )}
        {children}
        {spec.source_note && (
          <div style={{ position: "absolute", right: px(60, height), top: px(930 - 40, height), fontSize: px(21, height),
            fontWeight: 500, color: transparent ? "#EBEBEB" : pal.muted, letterSpacing: "0.02em",
            textShadow: transparent ? "0 1px 4px rgba(0,0,0,0.6)" : "none" }}>
            {spec.source_note}
          </div>
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
