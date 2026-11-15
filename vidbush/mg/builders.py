"""MotionSpec builders with sensible defaults for every motion graphic kind.

Every function returns a MotionSpec whose payload validates through
spec.typed_payload(). Data-driven specs (maps, charts) always carry a
source_note. Safe area: important content stays inside y in [150, 930] px
(letterbox bars are 138 px).
"""
from __future__ import annotations

import math
from typing import Literal, Optional, Sequence

from ..models import (ChartPayload, ChartSeries, MapLabel, MapMarker, MapPayload, MapRoute, MotionKind,
                      MotionSpec, NumberCardPayload, RecapPayload, StyleTokens, TextPayload)
from ..util import clamp

Place = tuple[str, float, float]
Position = Literal["center", "lower_left", "upper_left", "upper_right", "lower_right"]
Background = Literal["transparent", "solid", "footage_behind"]

MAP_SOURCE = "Map data: Natural Earth"
DISPLAY_NAMES = {"United States of America": "United States", "United Kingdom": "Britain"}
_USABLE_H_PX = 780  # safe vertical band height at 1080p
_FRAME_W_PX = 1920


def _spec(kind: MotionKind, duration: float, payload: dict, palette: str, background: Background,
          source_note: str = "") -> MotionSpec:
    return MotionSpec(kind=kind, duration=round(max(0.5, duration), 3), background=background,
                      style=StyleTokens(palette=palette), source_note=source_note, payload=payload)


def source_note_for(title: str) -> str:
    """'Source: <title>' or 'Illustrative' when no source exists."""
    title = (title or "").strip()
    if not title:
        return "Illustrative"
    return title if title.lower().startswith(("source", "map data")) else f"Source: {title}"


def format_number(value: float, unit: str = "") -> str:
    """Compact display string: 22,000 / $375M / 1.5B / 40%."""
    u = unit.lower().strip()
    money = u in {"dollars", "usd", "$", "us dollars"}
    av = abs(value)
    if av >= 1e9:
        s = _short(value / 1e9) + "B"
    elif av >= 1e6 and money:
        s = _short(value / 1e6) + "M"
    elif av >= 1e6:
        s = _short(value / 1e6) + " million"
    elif float(value).is_integer():
        s = f"{int(value):,}"
    else:
        s = f"{value:,.1f}"
    if money:
        return "$" + s
    if u in {"percent", "%"}:
        return s + "%"
    return s


def _short(v: float) -> str:
    """Up to two decimals without trailing zeros: 3.25, 2.75, 1.5, 49."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- maps

def map_view(points: Sequence[tuple[float, float]], min_span: Optional[float] = None) -> tuple[list[float], float]:
    """Center [lat, lon] and zoom (1 = whole world, each step doubles scale) fitting all points."""
    if not points:
        return [20.0, 0.0], 1.0
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    if max(lons) - min(lons) > 180:
        lons = [lo + 360 if lo < 0 else lo for lo in lons]
    clat = (max(lats) + min(lats)) / 2
    clon = (max(lons) + min(lons)) / 2
    clon = (clon + 180) % 360 - 180
    lat_span = (max(lats) - min(lats)) / max(0.3, math.cos(math.radians(clat)))
    lon_span = max(lons) - min(lons)
    floor = min_span if min_span is not None else (10.0 if len(points) == 1 else 1.0)
    need = max(lon_span * 1.35, lat_span * 1.35 * _FRAME_W_PX / _USABLE_H_PX, floor)
    zoom = clamp(1 + math.log2(360.0 / need), 1.0, 11.0)
    return [round(clat, 4), round(clon, 4)], round(zoom, 2)


def in_view(center: Sequence[float], zoom: float, pt: Sequence[float], safe: bool = True) -> bool:
    """True when pt [lat, lon] lies inside the 1920x1080 frame of the view (inside the safe band when safe),
    with the projection of vidbush/mg/mapdraw.py."""
    s = _FRAME_W_PX * 2 ** (zoom - 1) / 360.0
    coslat = max(0.3, math.cos(math.radians(center[0])))
    x = abs((pt[1] - center[1] + 180.0) % 360.0 - 180.0) * s
    y = abs(pt[0] - center[0]) / coslat * s
    return x <= _FRAME_W_PX / 2 - (60 if safe else 0) and y <= (_USABLE_H_PX / 2 if safe else 540)


def trace_route(places: Sequence[Place], via: str) -> tuple[Optional[list[list[float]]], str]:
    """(geometry, mode): real geometry through the places in order (routes.route_points per leg, one of
    routes.modes(via) for the whole path), thinned for the payload; (None, via) when no mode has every leg."""
    from . import routes as R
    for mode in R.modes(via):
        out: list[list[float]] = []
        for a, b in zip(places, places[1:]):
            leg = R.route_points((a[1], a[2]), (b[1], b[2]), mode)
            if leg is None:
                break
            out += leg[1:] if out else leg
        else:
            if out:
                return R.thin(out), mode
    return None, via


def drawn_title(payload: dict) -> str:
    """A map's title rebuilt from what it draws, never free text: 'A to B' when exactly one route is drawn,
    else up to three marker names ('A, B and C'), else the shaded regions, else ''. A title is read as a claim
    about the picture, so it can only name places that are on it (run 2 titled a map 'Nome to Wrangell-St.
    Elias' with only Nome drawn); a distance claim lives on the checked route label."""
    routes = [r for r in payload.get("routes") or [] if isinstance(r, dict)]
    if len(routes) == 1 and routes[0].get("from_place") and routes[0].get("to_place"):
        return f"{routes[0]['from_place']} to {routes[0]['to_place']}"
    names = [str(m.get("label") or "").strip() for m in payload.get("markers") or [] if isinstance(m, dict)]
    names = list(dict.fromkeys(n for n in names if n))[:3] or [str(r) for r in payload.get("shaded_regions") or []][:3]
    names = [DISPLAY_NAMES.get(n, n) for n in names]  # the short name a viewer knows (United States)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def map_spec(places: Sequence[Place], duration: float = 8.0, route: bool = False, shaded: Sequence[str] = (),
             source_note: str = MAP_SOURCE, palette: str = "documentary_warm",
             push_in: bool = True, dashed: bool = False, route_label: str = "", via: str = "line") -> MotionSpec:
    """Map with staggered markers, an optional route through the places in order and shaded regions; the
    title is drawn_title of what it draws.

    The route is real geometry (trace_route with via); with none, the map keeps the markers only. shaded may
    name countries, states or provinces (shaded) or continents (a large label that frames the view).
    """
    from . import geo
    from . import routes as R
    line, via = trace_route(places, via) if route and len(places) >= 2 else (None, via)
    pts: list[tuple[float, float]] = [(p[1], p[2]) for p in places] + [(q[0], q[1]) for q in line or []]
    labels: list[MapLabel] = []
    countries: list[str] = []
    for name in shaded:
        region = geo.region_name(name, [(p[1], p[2]) for p in places])
        box = geo.region_bbox(region) if region else None
        hit = geo.lookup(name)
        if box:
            pts += [(box[0], box[1]), (box[2], box[3])]
            countries.append(region)
        elif hit:
            span = 12.0 if geo.place_kind(name) == "continent" else 4.0
            pts += [(hit.lat - span, hit.lon - span), (hit.lat + span, hit.lon + span)]
        if hit and not any(p[0] == hit.name for p in places):
            labels.append(MapLabel(lat=hit.lat, lon=hit.lon, text=hit.name.upper(), size="large"))
    center, zoom = map_view(pts)
    rs, re_ = _reveal_window(duration)
    n = len(places)
    markers = []
    for i, (name, lat, lon) in enumerate(places):
        at = rs + (re_ - rs) * i / max(1, n - 1) if line else 0.4 + i * min(0.7, duration * 0.45 / max(1, n))
        markers.append(MapMarker(lat=lat, lon=lon, label=name, appear_at=round(at, 2)))
    routes = []
    if line:
        routes.append(MapRoute(points=line, reveal_start=rs, reveal_end=re_, label=route_label,
                               dashed=dashed or via in R.ARC_VIAS, via=via, from_place=places[0][0],
                               to_place=places[-1][0]))
    payload = MapPayload(center=center, zoom=zoom, zoom_end=round(zoom + 0.35, 2) if push_in else None,
                         markers=markers, routes=routes, labels=labels,
                         shaded_regions=countries)
    payload.title = drawn_title(payload.model_dump())
    return _spec(MotionKind.map, duration, payload.model_dump(), palette, "solid", source_note or MAP_SOURCE)


def route_map_spec(paths: Sequence[Sequence[Place]], duration: float = 9.0, labels: Sequence[str] = (),
                   source_note: str = MAP_SOURCE, palette: str = "documentary_warm",
                   dashed_from: int = 1, via: str = "line") -> MotionSpec:
    """Several routes revealed one after another (e.g. old route vs new shortcut); later ones dashed. Each is
    real geometry (trace_route with via); a path with none keeps its endpoint markers only. The title is
    drawn_title of what it draws."""
    from . import routes as R
    traced = [trace_route(path, via) for path in paths]
    lines = [t[0] for t in traced]
    all_pts = [(p[1], p[2]) for path in paths for p in path] + [(q[0], q[1]) for ln in lines for q in ln or []]
    center, zoom = map_view(all_pts)
    rs, re_ = _reveal_window(duration)
    k = max(1, len(paths))
    step = (re_ - rs) / k
    routes, markers, seen = [], [], set()
    for i, path in enumerate(paths):
        a, b = rs + i * step, rs + (i + 1) * step
        if lines[i]:
            routes.append(MapRoute(points=lines[i], reveal_start=round(a, 2), reveal_end=round(b, 2),
                                   label=labels[i] if i < len(labels) else "",
                                   dashed=i >= dashed_from or traced[i][1] in R.ARC_VIAS, via=traced[i][1],
                                   from_place=path[0][0],
                                   to_place=path[-1][0]))
        for j in (0, len(path) - 1):
            name, lat, lon = path[j]
            if name and name not in seen:
                seen.add(name)
                markers.append(MapMarker(lat=lat, lon=lon, label=name, appear_at=round(a if j == 0 else b, 2)))
    payload = MapPayload(center=center, zoom=zoom, zoom_end=round(zoom + 0.25, 2), markers=markers,
                         routes=routes)
    payload.title = drawn_title(payload.model_dump())
    return _spec(MotionKind.map, duration, payload.model_dump(), palette, "solid", source_note or MAP_SOURCE)


def _reveal_window(duration: float) -> tuple[float, float]:
    rs = round(min(0.6, duration * 0.1), 2)
    re_ = round(clamp(duration * 0.6, rs + 0.5, max(rs + 0.5, duration - 1.0)), 2)
    return rs, re_


# ---------------------------------------------------------------- charts

def bar_chart_spec(labels: Sequence[str], values: Sequence[float], units: str = "", title: str = "",
                   source_note: str = "", highlight_index: Optional[int] = None, duration: float = 8.0,
                   palette: str = "documentary_warm") -> MotionSpec:
    """Growing bars; source_note defaults to 'Illustrative' when no source is given."""
    payload = ChartPayload(title=title, labels=list(labels), series=[ChartSeries(name=units, values=list(values))],
                           units=units, animate="grow", highlight_index=highlight_index)
    return _spec(MotionKind.bar_chart, duration, payload.model_dump(), palette, "solid", source_note_for(source_note))


def line_chart_spec(labels: Sequence[str], series: Sequence[Sequence[float]] | Sequence[float], units: str = "",
                    title: str = "", source_note: str = "", names: Sequence[str] = (),
                    highlight_index: Optional[int] = None, duration: float = 8.0,
                    palette: str = "documentary_warm") -> MotionSpec:
    """Line drawn left to right; series is one list of values or a list of lists."""
    rows = [list(series)] if series and not isinstance(series[0], (list, tuple)) else [list(s) for s in series]
    chart_series = [ChartSeries(name=names[i] if i < len(names) else units, values=[float(v) for v in r])
                    for i, r in enumerate(rows)]
    payload = ChartPayload(title=title, labels=list(labels), series=chart_series, units=units, animate="draw",
                           highlight_index=highlight_index)
    return _spec(MotionKind.line_chart, duration, payload.model_dump(), palette, "solid", source_note_for(source_note))


# ---------------------------------------------------------------- cards and text

def number_card_spec(number: str, label: str = "", sublabel: str = "", count_up: bool = False,
                     duration: float = 6.0, palette: str = "documentary_warm", source_note: str = "",
                     background: Background = "solid") -> MotionSpec:
    """Big number with a label; count_up animates numeric values."""
    payload = NumberCardPayload(number=str(number), label=label, sublabel=sublabel, count_up=count_up)
    return _spec(MotionKind.number_card, duration, payload.model_dump(), palette, background, source_note)


def title_card_spec(title: str, subtitle: str = "", duration: float = 6.0,
                    palette: str = "documentary_warm") -> MotionSpec:
    """Full-frame chapter or act title."""
    return _spec(MotionKind.title_card, duration, TextPayload(title=title, subtitle=subtitle).model_dump(),
                 palette, "solid")


def lower_third_spec(title: str, subtitle: str = "", duration: float = 5.0,
                     palette: str = "documentary_warm") -> MotionSpec:
    """Transparent lower third naming a person or place (sits above the caption band)."""
    payload = TextPayload(title=title, subtitle=subtitle, position="lower_left").model_dump()
    return _spec(MotionKind.lower_third, duration, payload, palette, "transparent")


def callout_spec(text: str, subtitle: str = "", duration: float = 5.0, position: Position = "upper_left",
                 palette: str = "documentary_warm", background: Background = "transparent") -> MotionSpec:
    """Short key phrase callout, transparent by default (overlay use)."""
    payload = TextPayload(title=text, subtitle=subtitle, position=position)
    return _spec(MotionKind.callout, duration, payload.model_dump(), palette, background)


def text_overlay_spec(text: str, position: Position = "center", subtitle: str = "", duration: float = 5.0,
                      palette: str = "documentary_warm", background: Background = "transparent") -> MotionSpec:
    """Date stamp, quote or statement. Use background='solid' for a full-frame quote card."""
    payload = TextPayload(title=text, subtitle=subtitle, position=position)
    return _spec(MotionKind.text_overlay, duration, payload.model_dump(), palette, background)


def recap_grid_spec(items: Sequence[str], title: str = "Recap", duration: float = 10.0,
                    palette: str = "top10_bold", numbers: Optional[Sequence[int]] = None) -> MotionSpec:
    """Grid listing items in the given order (countdown order for Top 10); numbers label each item."""
    nums = [int(n) for n in numbers] if numbers is not None and len(numbers) == len(items) else None
    payload = RecapPayload(title=title, items=list(items), numbers=nums)
    return _spec(MotionKind.recap_grid, duration, payload.model_dump(), palette, "solid")


# ---------------------------------------------------------------- retiming

def retime_spec(spec: MotionSpec, duration: float) -> MotionSpec:
    """Copy of spec with a new duration and every reveal timing scaled proportionally."""
    duration = round(max(0.5, duration), 3)
    old = spec.duration if spec.duration > 0 else duration
    f = duration / old
    payload = dict(spec.payload)
    if spec.kind == MotionKind.map:
        mp = MapPayload(**payload)
        last = max(0.0, duration - 0.3)
        for m in mp.markers:
            m.appear_at = round(min(m.appear_at * f, last), 3)
        for r in mp.routes:
            r.reveal_start = round(min(r.reveal_start * f, last), 3)
            r.reveal_end = round(clamp(r.reveal_end * f, r.reveal_start, last), 3)
        for a in mp.arrows:
            a.appear_at = round(min(a.appear_at * f, last), 3)
        payload = mp.model_dump()
    return spec.model_copy(update={"duration": duration, "payload": payload})


def with_palette(spec: MotionSpec, palette: str) -> MotionSpec:
    """Copy of spec using the given palette."""
    return spec.model_copy(update={"style": spec.style.model_copy(update={"palette": palette})})
