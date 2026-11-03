import React from "react";
import { Easing, useCurrentFrame, useVideoConfig } from "remotion";
import { palette, prog, px } from "../theme";
import { Props } from "../types";
import { Frame } from "./Frame";

export const RecapGrid: React.FC<Props> = ({ spec }) => {
  const frame = useCurrentFrame();
  const { fps, height: H } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const items: string[] = spec.payload.items ?? [];
  const n = items.length;
  const cols = n <= 4 ? n : Math.min(5, Math.ceil(n / 2));
  const countdown = spec.style.palette === "top10_bold" || String(spec.payload.title ?? "").toLowerCase().includes("top");
  return (
    <Frame spec={spec}>
      <div style={{ position: "absolute", left: px(110, H), right: px(110, H), top: px(180, H) }}>
        <div style={{ fontSize: px(54, H), fontWeight: 700, color: pal.text, letterSpacing: "0.08em",
          textTransform: "uppercase", opacity: prog(frame, fps, 0.05, 0.7) }}>{spec.payload.title}</div>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: px(22, H), marginTop: px(46, H) }}>
          {items.map((it, i) => {
            const a = 0.45 + (i * Math.min(spec.duration * 0.4, 0.16 * n)) / Math.max(1, n - 1);
            const k = prog(frame, fps, a, a + 0.5, Easing.out(Easing.back(1.4)));
            return (
              <div key={i} style={{ background: pal.panel, borderRadius: px(10, H), padding: px(22, H), minHeight: px(230, H),
                opacity: Math.min(1, k * 1.6), transform: `scale(${0.88 + 0.12 * k}) translateY(${px(20, H) * (1 - k)}px)`,
                boxShadow: "0 5px 14px rgba(0,0,0,0.35)" }}>
                <div style={{ fontSize: px(64, H), fontWeight: 900, color: pal.accent }}>{countdown ? `#${n - i}` : String(i + 1).padStart(2, "0")}</div>
                <div style={{ fontSize: px(30, H), fontWeight: 600, color: pal.panel_text, marginTop: px(6, H) }}>{it}</div>
              </div>
            );
          })}
        </div>
      </div>
    </Frame>
  );
};
