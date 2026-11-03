import React, { useEffect, useMemo, useState } from "react";
import { continueRender, delayRender, Easing, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { palette, prog, px } from "../theme";
import { Props } from "../types";
import { Frame } from "./Frame";

type Geo = { features: { geometry: { type: string; coordinates: any } }[] };

/** Same local equirectangular projection as vidbush/mg/mapdraw.py (zoom 1 = 360 degrees across the width). */
const projector = (lat: number, lon: number, W: number, H: number, zoom: number) => {
  const cos = Math.max(0.3, Math.cos((lat * Math.PI) / 180));
  const s = (W * 2 ** (zoom - 1)) / 360;
  return (la: number, lo: number): [number, number] => [(lo - lon) * s + W / 2, (-la / cos + lat / cos) * s + H / 2];
};

const useGeo = (file: string): Geo | null => {
  const [geo, setGeo] = useState<Geo | null>(null);
  const [handle] = useState(() => delayRender(file));
  useEffect(() => {
    fetch(staticFile(`geo/${file}`)).then((r) => r.json()).then(setGeo).finally(() => continueRender(handle));
  }, [file, handle]);
  return geo;
};

export const MapComp: React.FC<Props> = ({ spec }) => {
  const frame = useCurrentFrame();
  const { fps, width: W, height: H, durationInFrames } = useVideoConfig();
  const pal = palette(spec.style.palette);
  const p = spec.payload;
  const [clat, clon] = p.center ?? [20, 0];
  const z0 = p.zoom ?? 3;
  const z1 = p.zoom_end ?? z0;
  const geo = useGeo(Math.max(z0, z1) >= 3 ? "ne_50m_admin_0_countries.geojson" : "ne_110m_admin_0_countries.geojson");
  const proj = useMemo(() => projector(clat, clon, W, H, z0), [clat, clon, W, H, z0]);
  const paths = useMemo(() => {
    if (!geo) return "";
    const ring = (r: number[][]) => r.map(([lo, la], i) => { const [x, y] = proj(la, lo); return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`; }).join("") + "Z";
    return geo.features.map((f) => {
      const g = f.geometry;
      if (!g) return "";
      const polys = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
      return polys.map((poly: number[][][]) => poly.map(ring).join("")).join("");
    }).join("");
  }, [geo, proj]);
  const zoom = interpolate(frame, [0, durationInFrames], [z0, z1], { easing: Easing.inOut(Easing.sin) });
  const scale = 2 ** (zoom - z0);
  const t = frame / fps;
  return (
    <Frame spec={spec}>
      <svg width={W} height={H} style={{ position: "absolute", background: pal.water }}>
        <g transform={`translate(${W / 2} ${H / 2}) scale(${scale}) translate(${-W / 2} ${-H / 2})`}>
          <path d={paths} fill={pal.land} stroke={pal.land_border} strokeWidth={px(1.3, H) / scale} fillRule="evenodd" />
          {(p.routes ?? []).map((r: any, i: number) => {
            const pts = r.points.map(([la, lo]: number[]) => proj(la, lo));
            const d = pts.map(([x, y]: number[], k: number) => `${k ? "L" : "M"}${x},${y}`).join("");
            const rev = prog(frame, fps, r.reveal_start, r.reveal_end, Easing.inOut(Easing.cubic));
            return <path key={i} d={d} fill="none" stroke={pal.route} strokeWidth={px(5, H) / scale} pathLength={1}
              strokeDasharray={1} strokeDashoffset={1 - rev} />;
          })}
          {(p.markers ?? []).map((m: any, i: number) => {
            const [x, y] = proj(m.lat, m.lon);
            const k = prog(frame, fps, m.appear_at, m.appear_at + 0.45, Easing.out(Easing.back(2.2)));
            return (
              <g key={i} transform={`translate(${x} ${y}) scale(${1 / scale})`} opacity={k > 0 ? 1 : 0}>
                <circle r={px(13, H) * k} fill={pal.panel} />
                <circle r={px(8, H) * k} fill={pal.accent} />
                {m.label && <text x={px(24, H)} y={px(9, H)} fontSize={px(28, H)} fontWeight={600} fill={pal.text}
                  opacity={prog(frame, fps, m.appear_at + 0.15, m.appear_at + 0.6)}
                  stroke={pal.panel} strokeWidth={px(5, H)} paintOrder="stroke">{m.label}</text>}
              </g>
            );
          })}
        </g>
      </svg>
      {p.title && <div style={{ position: "absolute", left: px(110, H), top: px(180, H), background: pal.panel,
        borderLeft: `${px(8, H)}px solid ${pal.accent}`, padding: `${px(14, H)}px ${px(24, H)}px`, fontSize: px(40, H),
        fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: pal.panel_text,
        opacity: prog(frame, fps, 0.2, 0.9), visibility: t >= 0 ? "visible" : "hidden" }}>{p.title}</div>}
    </Frame>
  );
};
