// Same palettes as vidbush/mg/palettes.py.
import { interpolate, Easing } from "remotion";

export type Palette = Record<
  "bg" | "bg2" | "land" | "land_border" | "water" | "text" | "muted" | "accent" | "accent2" | "route" | "shade"
  | "panel" | "panel_text" | "grid", string> & { dark: boolean };

export const PALETTES: Record<string, Palette> = {
  documentary_warm: { bg: "#EAE0CB", bg2: "#CDBB98", land: "#E9DCBD", land_border: "#9B8763", water: "#A9BAB3",
    text: "#2B2118", muted: "#76664F", accent: "#C47A1C", accent2: "#8C3A2B", route: "#9E2F23", shade: "#D49533",
    panel: "#F2EADA", panel_text: "#2B2118", grid: "#B9A887", dark: false },
  documentary_cold: { bg: "#122030", bg2: "#070E17", land: "#2B3D51", land_border: "#5A7290", water: "#0C1724",
    text: "#EAF1F7", muted: "#8DA0B6", accent: "#3CCFE4", accent2: "#86AEDD", route: "#3CCFE4", shade: "#2C93A8",
    panel: "#0E1A27", panel_text: "#EAF1F7", grid: "#2A3B4F", dark: true },
  top10_bold: { bg: "#141418", bg2: "#050506", land: "#26262C", land_border: "#4A4A55", water: "#0D0D10",
    text: "#FFFFFF", muted: "#A2A2AD", accent: "#E5252C", accent2: "#F6B800", route: "#F6B800", shade: "#C81F26",
    panel: "#0E0E11", panel_text: "#FFFFFF", grid: "#2C2C33", dark: true },
};

export const palette = (name: string): Palette => PALETTES[name] ?? PALETTES.documentary_warm;

/** Eased 0..1 progress of frame through [a, b] seconds. */
export const prog = (frame: number, fps: number, a: number, b: number, ease = Easing.out(Easing.cubic)): number =>
  interpolate(frame / fps, [a, b], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: ease });

export const FONT = "Inter, 'Helvetica Neue', Arial, sans-serif";

/** Design pixels at 1080p scaled to the output height. */
export const px = (v: number, height: number): number => (v * height) / 1080;
