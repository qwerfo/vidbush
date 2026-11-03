import React from "react";
import { Easing, useCurrentFrame, useVideoConfig } from "remotion";
import { palette, prog, px } from "../theme";
import { Props } from "../types";
import { Frame } from "./Frame";

type Pos = "center" | "lower_left" | "upper_left" | "upper_right" | "lower_right";

/** CSS placement of a block at a TextPayload position inside the letterbox safe area (y 150..930 at 1080p). */
export const place = (pos: Pos, h: number): React.CSSProperties => {
  const m = px(110, h), top = px(180, h), bottom = h - px(930 - 30, h);
  switch (pos) {
    case "upper_left": return { position: "absolute", left: m, top };
    case "upper_right": return { position: "absolute", right: m, top, textAlign: "right" };
    case "lower_left": return { position: "absolute", left: m, bottom };
    case "lower_right": return { position: "absolute", right: m, bottom, textAlign: "right" };
    default: return { position: "absolute", left: 0, right: 0, top: 0, bottom: 0, display: "flex",
      flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center" };
  }
};

export const TitleCard: React.FC<Props> = ({ spec }) => {
  const frame = useCurrentFrame();
  const { fps, height, durationInFrames } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const p = spec.payload;
  const settle = prog(frame, fps, 0, Math.min(2.4, (durationInFrames / fps) * 0.6), Easing.out(Easing.exp));
  const rule = prog(frame, fps, 0.3, 1.4, Easing.inOut(Easing.cubic));
  const sub = prog(frame, fps, 0.8, 1.6);
  const color = spec.background === "solid" ? pal.text : "#FFFFFF";
  const ruleStyle = { width: px(520, height) * rule, height: Math.max(1, px(2, height)), background: pal.accent, margin: "0 auto" };
  return (
    <Frame spec={spec}>
      <div style={place("center", height)}>
        <div style={ruleStyle} />
        <div style={{ fontSize: px(120, height), fontWeight: 700, color, textTransform: "uppercase", maxWidth: "80%",
          letterSpacing: `${0.45 - 0.37 * settle}em`, opacity: prog(frame, fps, 0.05, 0.9), lineHeight: 1.04,
          margin: `${px(40, height)}px 0`, transform: `translateY(${px(16, height) * (1 - settle)}px)` }}>
          {p.title}
        </div>
        {p.subtitle && (
          <div style={{ fontSize: px(34, height), fontWeight: 500, color: pal.accent, letterSpacing: "0.26em",
            textTransform: "uppercase", opacity: sub, marginBottom: px(40, height) }}>{p.subtitle}</div>
        )}
        <div style={ruleStyle} />
      </div>
    </Frame>
  );
};

const NUM_RE = /^([^\d-]*?)(-?\d[\d,]*(?:\.\d+)?)(.*)$/;

export const NumberCard: React.FC<Props> = ({ spec }) => {
  const frame = useCurrentFrame();
  const { fps, height, durationInFrames } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const p = spec.payload;
  let text = String(p.number ?? "");
  const m = NUM_RE.exec(text);
  if (p.count_up && m && m[1].trim() !== "#") {
    const target = parseFloat(m[2].replace(/,/g, ""));
    const dec = m[2].includes(".") ? m[2].split(".")[1].length : 0;
    const year = !m[1] && !m[3].trim() && !m[2].includes(",") && dec === 0 && target >= 1000 && target <= 2100;
    const start = year ? target - 60 : 0;
    const k = prog(frame, fps, 0.2, Math.max(0.6, Math.min((durationInFrames / fps) * 0.6, 2.6)), Easing.out(Easing.exp));
    const v = start + (target - start) * k;
    const s = m[2].includes(",") ? v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec })
      : v.toFixed(dec);
    text = m[1] + s + m[3];
  }
  const pop = prog(frame, fps, 0, 0.7, Easing.out(Easing.back(1.2)));
  const color = pal === palette("top10_bold") ? pal.accent : spec.background === "solid" ? pal.text : "#FFFFFF";
  return (
    <Frame spec={spec}>
      <div style={place("center", height)}>
        <div style={{ fontSize: px(300, height), fontWeight: 900, color, fontVariantNumeric: "tabular-nums",
          transform: `scale(${0.9 + 0.1 * pop})`, opacity: prog(frame, fps, 0, 0.45), lineHeight: 1 }}>{text}</div>
        <div style={{ width: px(130, height) * prog(frame, fps, 0.35, 1.0), height: px(6, height), background: pal.accent,
          margin: `${px(30, height)}px auto` }} />
        {p.label && <div style={{ fontSize: px(52, height), fontWeight: 600, color: spec.background === "solid" ? pal.text : "#FFF",
          letterSpacing: "0.12em", textTransform: "uppercase", opacity: prog(frame, fps, 0.55, 1.15) }}>{p.label}</div>}
        {p.sublabel && <div style={{ fontSize: px(34, height), color: pal.muted, marginTop: px(20, height),
          opacity: prog(frame, fps, 0.75, 1.35) }}>{p.sublabel}</div>}
      </div>
    </Frame>
  );
};

/** Lower third and callout: accent bar, panel wiping open, text fading in; reverses at the end. */
const Panel: React.FC<Props & { lower: boolean }> = ({ spec, lower }) => {
  const frame = useCurrentFrame();
  const { fps, height, durationInFrames } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const p = spec.payload;
  const end = durationInFrames / fps;
  const open = prog(frame, fps, 0.12, 0.7) * (1 - prog(frame, fps, end - 0.55, end - 0.05, Easing.inOut(Easing.cubic)));
  const text = prog(frame, fps, 0.35, 0.8) * (1 - prog(frame, fps, end - 0.7, end - 0.4));
  const pos: Pos = lower ? (p.position === "center" ? "lower_left" : p.position) : (p.position ?? "upper_left");
  return (
    <Frame spec={spec} ownExit>
      <div style={{ ...place(pos, height), display: "flex", maxWidth: "45%" }}>
        <div style={{ width: px(8, height), background: pal.accent, transform: `scaleY(${Math.min(1, open * 1.6)})` }} />
        <div style={{ background: pal.panel, opacity: 0.93, padding: `${px(lower ? 18 : 28, height)}px ${px(lower ? 30 : 36, height)}px`,
          clipPath: lower ? `inset(0 ${100 - open * 100}% 0 0)` : `inset(0 0 ${100 - open * 100}% 0)`,
          boxShadow: "0 6px 18px rgba(0,0,0,0.35)" }}>
          <div style={{ fontSize: px(lower ? 50 : 56, height), fontWeight: lower ? 600 : 700, color: pal.panel_text,
            opacity: text, lineHeight: 1.12 }}>{p.title}</div>
          {p.subtitle && <div style={{ fontSize: px(lower ? 25 : 30, height), fontWeight: 500, opacity: text,
            color: lower ? pal.accent : pal.muted, textTransform: lower ? "uppercase" : "none",
            letterSpacing: lower ? "0.16em" : 0, marginTop: px(6, height) }}>{p.subtitle}</div>}
        </div>
      </div>
    </Frame>
  );
};

export const LowerThird: React.FC<Props> = ({ spec }) => <Panel spec={spec} lower />;
export const Callout: React.FC<Props> = ({ spec }) => <Panel spec={spec} lower={false} />;

export const TextOverlay: React.FC<Props> = ({ spec }) => {
  const frame = useCurrentFrame();
  const { fps, height, durationInFrames } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const p = spec.payload;
  const end = durationInFrames / fps;
  const words = String(p.title ?? "").split(/\s+/).filter(Boolean);
  const out = 1 - prog(frame, fps, end - 0.45, end, Easing.in(Easing.cubic));
  const solid = spec.background === "solid";
  return (
    <Frame spec={spec} ownExit>
      <div style={{ ...place(p.position ?? "center", height), opacity: out, maxWidth: solid ? "78%" : "50%",
        margin: solid ? "0 auto" : undefined }}>
        <div style={{ fontSize: px(solid ? 84 : 66, height), fontWeight: 700, color: solid ? pal.text : "#FFF",
          textShadow: solid ? "none" : "0 2px 8px rgba(0,0,0,0.65)", lineHeight: 1.14 }}>
          {words.map((w, i) => {
            const o = prog(frame, fps, 0.1 + i * 0.09, 0.55 + i * 0.09);
            return <span key={i} style={{ display: "inline-block", opacity: o, marginRight: "0.25em",
              transform: `translateY(${px(28, height) * (1 - o)}px)` }}>{w}</span>;
          })}
        </div>
        {p.subtitle && <div style={{ fontSize: px(28, height), fontWeight: 500, letterSpacing: "0.2em", marginTop: px(24, height),
          color: solid ? pal.accent : "#EBEBEB", textTransform: "uppercase",
          opacity: prog(frame, fps, 0.4 + words.length * 0.09, 0.9 + words.length * 0.09) }}>{p.subtitle}</div>}
      </div>
    </Frame>
  );
};
