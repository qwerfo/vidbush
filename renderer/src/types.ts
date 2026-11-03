// Mirror of vidbush/models.py MotionSpec (the JSON contract). Payload is typed per kind.
export type MotionKind =
  | "map" | "bar_chart" | "line_chart" | "number_card" | "title_card" | "lower_third"
  | "callout" | "highlight_box" | "text_overlay" | "arrow" | "recap_grid";

export type MotionSpec = {
  kind: MotionKind;
  duration: number;
  width: number;
  height: number;
  fps: number;
  background: "transparent" | "solid" | "footage_behind";
  style: { font: string; palette: string };
  source_note: string;
  payload: Record<string, any>;
};

export type Props = { spec: MotionSpec };

export const KINDS: MotionKind[] = [
  "map", "bar_chart", "line_chart", "number_card", "title_card", "lower_third",
  "callout", "highlight_box", "text_overlay", "arrow", "recap_grid",
];

export const DEFAULT_SPEC: MotionSpec = {
  kind: "title_card", duration: 3, width: 1920, height: 1080, fps: 30, background: "solid",
  style: { font: "Inter", palette: "documentary_warm" }, source_note: "",
  payload: { title: "Vidbush", subtitle: "Remotion renderer" },
};
