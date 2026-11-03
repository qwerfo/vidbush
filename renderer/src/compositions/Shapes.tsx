import React from "react";
import { Easing, useCurrentFrame, useVideoConfig } from "remotion";
import { palette, prog, px } from "../theme";
import { Props } from "../types";
import { Frame } from "./Frame";

const exit = (frame: number, fps: number, end: number): number =>
  1 - prog(frame, fps, end - Math.min(0.45, end * 0.2), end, Easing.in(Easing.cubic));

export const HighlightBox: React.FC<Props> = ({ spec }) => {
  const frame = useCurrentFrame();
  const { fps, width: W, height: H, durationInFrames } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const p = spec.payload;
  const x = (p.x ?? 0.3) * W, y = (p.y ?? 0.3) * H, w = (p.w ?? 0.4) * W, h = (p.h ?? 0.3) * H;
  const perim = 2 * (w + h);
  const trace = prog(frame, fps, 0.15, 1.05, Easing.inOut(Easing.cubic));
  const dim = 0.38 * prog(frame, fps, 0, 0.6);
  return (
    <Frame spec={spec} ownExit>
      <svg width={W} height={H} style={{ position: "absolute", opacity: exit(frame, fps, durationInFrames / fps) }}>
        <defs><mask id="hole"><rect width={W} height={H} fill="white" /><rect x={x} y={y} width={w} height={h} fill="black" /></mask></defs>
        <rect width={W} height={H} fill="black" opacity={dim} mask="url(#hole)" />
        <rect x={x} y={y} width={w} height={h} fill="none" stroke={pal.accent} strokeWidth={px(5, H)}
          strokeDasharray={perim} strokeDashoffset={perim * (1 - trace)} style={{ filter: `drop-shadow(0 0 ${px(8, H)}px ${pal.accent})` }} />
        {p.label && (
          <g opacity={prog(frame, fps, 0.8, 1.3)}>
            <text x={x + px(12, H)} y={y - px(18, H)} fontSize={px(26, H)} fontWeight={600} fill={pal.accent}
              letterSpacing="0.1em" style={{ textTransform: "uppercase" }}>{String(p.label).toUpperCase()}</text>
          </g>
        )}
      </svg>
    </Frame>
  );
};

export const Arrow: React.FC<Props> = ({ spec }) => {
  const frame = useCurrentFrame();
  const { fps, width: W, height: H, durationInFrames } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const p = spec.payload;
  const sx = p.x * W, sy = p.y * H;
  const ex = (p.to_x ?? p.x + 0.18) * W, ey = (p.to_y ?? p.y + 0.12) * H;
  const len = Math.hypot(ex - sx, ey - sy) || 1;
  let nx = -(ey - sy) / len, ny = (ex - sx) / len;
  if (ny > 0) { nx = -nx; ny = -ny; }
  const cx = (sx + ex) / 2 + nx * len * 0.16, cy = (sy + ey) / 2 + ny * len * 0.16;
  const d = `M ${sx} ${sy} Q ${cx} ${cy} ${ex} ${ey}`;
  const draw = prog(frame, fps, 0.2, 1.1, Easing.inOut(Easing.cubic));
  const head = prog(frame, fps, 0.95, 1.3, Easing.out(Easing.back(2)));
  const ang = (Math.atan2(ey - cy, ex - cx) * 180) / Math.PI;
  const s = px(34, H) * head;
  return (
    <Frame spec={spec} ownExit>
      <svg width={W} height={H} style={{ position: "absolute", opacity: exit(frame, fps, durationInFrames / fps),
        filter: "drop-shadow(0 2px 6px rgba(0,0,0,0.5))" }}>
        <path d={d} fill="none" stroke={pal.accent} strokeWidth={px(7, H)} strokeLinecap="round"
          pathLength={1} strokeDasharray={1} strokeDashoffset={1 - draw} />
        <circle cx={sx} cy={sy} r={px(9, H) * prog(frame, fps, 0.05, 0.4)} fill={pal.accent} />
        {head > 0.01 && (
          <polygon points={`${s * 0.15},0 ${-s},${s * 0.55} ${-s},${-s * 0.55}`} fill={pal.accent}
            transform={`translate(${ex} ${ey}) rotate(${ang})`} />
        )}
        {p.label && (
          <text x={ex >= sx ? sx - px(24, H) : sx + px(24, H)} y={sy + px(10, H)} textAnchor={ex >= sx ? "end" : "start"}
            fontSize={px(30, H)} fontWeight={600} fill="#FFFFFF" opacity={prog(frame, fps, 0, 0.5)}>{p.label}</text>
        )}
      </svg>
    </Frame>
  );
};
