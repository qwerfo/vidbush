import React from "react";
import { MotionKind, Props } from "../types";
import { BarChart, LineChart } from "./Charts";
import { MapComp } from "./Map";
import { RecapGrid } from "./Recap";
import { Arrow, HighlightBox } from "./Shapes";
import { Callout, LowerThird, NumberCard, TextOverlay, TitleCard } from "./Text";

export const COMPONENTS: Record<MotionKind, React.FC<Props>> = {
  map: MapComp, bar_chart: BarChart, line_chart: LineChart, number_card: NumberCard, title_card: TitleCard,
  lower_third: LowerThird, callout: Callout, highlight_box: HighlightBox, text_overlay: TextOverlay,
  arrow: Arrow, recap_grid: RecapGrid,
};
