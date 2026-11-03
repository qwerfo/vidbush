import React from "react";
import { Easing, useCurrentFrame, useVideoConfig } from "remotion";
import { palette, prog, px } from "../theme";
import { Props } from "../types";
import { Frame } from "./Frame";

const fmt = (v: number, units: string, dec: number): string => {
  const u = (units || "").toLowerCase().trim();
  const money = ["$", "usd", "dollars"].includes(u);
  const pct = ["%", "percent"].includes(u);
  const a = Math.abs(v);
  const s = a >= 1e9 ? `${+(v / 1e9).toFixed(1)}B` : a >= 1e6 ? `${+(v / 1e6).toFixed(1)}M`
    : v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
  return (money ? "$" : "") + s + (pct ? "%" : !money && units && units.length <= 8 ? ` ${units}` : "");
};

const Title: React.FC<{ text: string; color: string; accent: string; H: number; t: number }> = ({ text, color, accent, H, t }) =>
  text ? (
    <div style={{ position: "absolute", left: px(110, H), top: px(186, H), opacity: t }}>
      <div style={{ fontSize: px(54, H), fontWeight: 700, color }}>{text}</div>
      <div style={{ width: px(110, H) * t, height: px(5, H), background: accent, marginTop: px(12, H) }} />
    </div>
  ) : null;

export const BarChart: React.FC<Props> = ({ spec }) => {
  const frame = useCurrentFrame();
  const { fps, width: W, height: H } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const p = spec.payload;
  const labels: string[] = p.labels ?? [];
  const values: number[] = (p.series?.[0]?.values ?? []).slice(0, labels.length);
  const max = Math.max(1e-9, ...values);
  const dec = values.every((v) => Number.isInteger(v)) ? 0 : 1;
  const x0 = px(110, H), x1 = W - px(110, H), top = px(360, H), base = px(820, H);
  const slot = (x1 - x0) / Math.max(1, labels.length);
  const color = spec.background === "solid" ? pal.text : "#FFF";
  return (
    <Frame spec={spec}>
      <Title text={p.title} color={color} accent={pal.accent} H={H} t={prog(frame, fps, 0, 0.6)} />
      <svg width={W} height={H} style={{ position: "absolute" }}>
        <line x1={x0} x2={x1} y1={base} y2={base} stroke={pal.muted} strokeWidth={px(2, H)} />
        {labels.map((lb, i) => {
          const g = prog(frame, fps, 0.35 + (i * Math.min(1.4, spec.duration * 0.25)) / Math.max(1, labels.length - 1),
            0.35 + 1.0 + (i * Math.min(1.4, spec.duration * 0.25)) / Math.max(1, labels.length - 1));
          const v = values[i] ?? 0, h = ((base - top) * v * g) / max, bw = slot * 0.62, cx = x0 + slot * (i + 0.5);
          const hl = p.highlight_index == null ? true : p.highlight_index === i;
          return (
            <g key={i}>
              <rect x={cx - bw / 2} y={base - h} width={bw} height={h} fill={hl ? pal.accent : pal.muted} />
              {g > 0.02 && <text x={cx} y={base - h - px(12, H)} textAnchor="middle" fontSize={px(34, H)} fontWeight={600}
                fill={hl && p.highlight_index != null ? pal.accent : color} style={{ fontVariantNumeric: "tabular-nums" }}>
                {fmt(v * g, p.units, dec)}</text>}
              <text x={cx} y={base + px(46, H)} textAnchor="middle" fontSize={px(30, H)} fontWeight={500} fill={color}>{lb}</text>
            </g>
          );
        })}
      </svg>
    </Frame>
  );
};

export const LineChart: React.FC<Props> = ({ spec }) => {
  const frame = useCurrentFrame();
  const { fps, width: W, height: H } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const p = spec.payload;
  const labels: string[] = p.labels ?? [];
  const values: number[] = p.series?.[0]?.values ?? [];
  const lo = Math.min(0, ...values), hi = Math.max(...values) * 1.12 || 1;
  const x0 = px(220, H), x1 = W - px(150, H), y0 = px(360, H), y1 = px(820, H);
  const pts = values.map((v, i) => [x0 + ((x1 - x0) * i) / Math.max(1, values.length - 1), y1 - ((v - lo) / (hi - lo)) * (y1 - y0)]);
  const d = pts.map((q, i) => `${i ? "L" : "M"} ${q[0]} ${q[1]}`).join(" ");
  const draw = prog(frame, fps, 0.5, 0.5 + Math.max(0.8, Math.min(spec.duration * 0.55, 4.5)), Easing.inOut(Easing.cubic));
  const color = spec.background === "solid" ? pal.text : "#FFF";
  const cut = x0 + (x1 - x0) * draw;
  return (
    <Frame spec={spec}>
      <Title text={p.title} color={color} accent={pal.accent} H={H} t={prog(frame, fps, 0, 0.6)} />
      <svg width={W} height={H} style={{ position: "absolute" }}>
        <defs><clipPath id="reveal"><rect x={0} y={0} width={cut} height={H} /></clipPath></defs>
        {[0, 0.25, 0.5, 0.75, 1].map((k) => <line key={k} x1={x0} x2={x1} y1={y1 - (y1 - y0) * k} y2={y1 - (y1 - y0) * k}
          stroke={pal.grid} strokeWidth={px(1.5, H)} />)}
        <path d={d} fill="none" stroke={pal.accent} strokeWidth={px(5, H)} clipPath="url(#reveal)"
          style={{ filter: `drop-shadow(0 0 ${px(6, H)}px ${pal.accent})` }} />
        {labels.map((lb, i) => pts[i] && <text key={i} x={pts[i][0]} y={y1 + px(40, H)} textAnchor="middle"
          fontSize={px(24, H)} fill={color}>{lb}</text>)}
      </svg>
    </Frame>
  );
};
