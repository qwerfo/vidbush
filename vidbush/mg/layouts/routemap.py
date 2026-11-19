"""route_map: places labelled as they are named, a route line drawn between the two words.

The map is the existing documentary map (vidbush/mg/mapdraw.MapComp, documentary_warm land and water) with
the house style on top: ink chips with paper type, an amber route and accent. Every place goes through the
same check as a director map (vidbush/shots/seat.verify_map: geocoded, a route is real geometry over the
bundled road, rail or river lines, a distance in its label within 25 percent of the drawn line). A place
that does not resolve or a route with no line makes the map incomplete (job.missing), so it is not shown at
all. A trail has no bundled line, so it is drawn as a dashed great circle with a note, and a
distance claimed for it is only checked to be no shorter than the straight line.

Timing: a place marker and chip settle on the place's cue (nothing before). A route draws from the from
place's word to the to place's word, or across the route element's own anchor when its places are not
separate elements. A distance chip sits on the line's middle and appears on its own cue.

View: fitted to the places and the line inside this comp's box (full frame or panel), with a slow push. A
registered painted base (the art lane's *_base.png with a sidecar json whose view matches, see
vidbush/art/hooks.painted_base) is used when one holds every place at about the fitted zoom; job.extra
["art_bases"] lists the dirs or files to look in. A panel whose shape is not the painting's 16:9 always uses
the plain base, since the painting would be stretched.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...models import LayoutKind, MapMarker, MapPayload, MapRoute, MotionKind, MotionSpec, StyleTokens
from .. import easing as ez
from ..house import MIN_LABEL_PX
from ..mapdraw import MapComp, Projection
from ..palettes import get_palette
from ..textdraw import Sprite, blit, fit_text, font, pill
from . import register
from .base import LayoutComp

PUSH = 0.3  # zoom steps of the slow push over the clip
FIT_X, FIT_Y = 0.70, 0.66  # share of the box half extent the places and line may use (room for chips)
ART_ZOOM_SLACK = 0.6  # a painting this many zoom steps wider than the fit still frames the places well
ART_ASPECT = 16 / 9
ROUTE_RE = re.compile(r"^\s*(?P<a>.+?)\s+to\s+(?P<b>.+?)(?:\s+by\s+(?P<via>[a-z]+))?\s*$", re.I)


def parse_route(text: str) -> Optional[tuple[str, str, str]]:
    """'Nenana to Nome by trail' -> ('Nenana', 'Nome', 'trail'); via '' when none is named."""
    m = ROUTE_RE.match(text or "")
    if not m:
        return None
    return m.group("a").strip(), m.group("b").strip(), (m.group("via") or "").lower()


def _same(a: str, b: str) -> bool:
    from ..geo import norm
    return norm(a) == norm(b)


def split_place(text: str) -> tuple[str, str]:
    """'Santa Monica, California' -> ('Santa Monica', 'California'): the name drawn, the qualifier geocoded."""
    name, _, qual = (text or "").partition(",")
    return name.strip(), qual.strip()


def resolve(name: str, qual: str = "", near: Any = ()) -> Any:
    """GeoHit of a place as verify_map resolves it (small places allowed, the map's other places breaking ties
    between homonyms, two homonyms equally near: none). A qualifier (state or country) picks among homonyms."""
    from .. import geo
    if qual:
        q = geo.norm(qual)
        got = [h for h in geo.candidates(name) if q in (geo.norm(h.state), geo.norm(h.country))
               or geo.norm(h.country) == geo.norm(geo.country_name(qual) or "-")]
        if got:
            return max(got, key=lambda h: h.pop)
    return geo.lookup(name, small=True) or (geo.lookup(name, near=near, small=True) if near else None)


def route_window(from_cue: Any, to_cue: Any, own_cue: Any) -> tuple[float, float]:
    """(reveal_start, reveal_end) of a route line: from the from place's word to the to place's word when
    both are separate spoken elements, else across the route element's own anchor span. At least 0.5 s."""
    if from_cue is not None and to_cue is not None and from_cue.found and to_cue.found:
        a, b = from_cue.word, to_cue.word
    else:
        a, b = own_cue.word, max(own_cue.word_end, own_cue.word)
    b = max(b, a + 0.5)
    return max(0.0, a), b


def fit_view(points: list[tuple[float, float]], W: int, H: int, box: tuple[float, float, float, float],
             fx: float = FIT_X, fy: float = FIT_Y) -> tuple[list[float], float]:
    """Center [lat, lon] and the deepest zoom at which every point lies inside box (frame pixels, projected
    about the frame centre as MapComp does) using fx, fy of its half extent."""
    lats = [p[0] for p in points]
    lons = list(np.unwrap(np.array([p[1] for p in points], np.float64), period=360))
    clat = (max(lats) + min(lats)) / 2
    coslat = max(0.3, math.cos(math.radians(clat)))
    du = (max(lons) - min(lons)) / 2
    dv = (max(lats) - min(lats)) / 2 / coslat
    hx = min(W / 2 - box[0], box[2] - W / 2) * fx
    hy = min(H / 2 - box[1], box[3] - H / 2) * fy
    s = min(hx / max(du, 1e-3), hy / max(dv, 1e-3), W / 0.8)  # 0.8 degrees across at most
    zoom = float(np.clip(1 + math.log2(s * 360 / W), 1.0, 11.0))
    clon = ((max(lons) + min(lons)) / 2 + 180) % 360 - 180
    return [round(clat, 4), round(clon, 4)], round(zoom, 2)


def inside(points: list[tuple[float, float]], center: list[float], zoom: float, W: int, H: int,
           box: tuple[float, float, float, float], fx: float = 0.85, fy: float = 0.8) -> bool:
    proj = Projection(center[0], center[1], W, H)
    q = proj.px(proj.uv(np.asarray(points, np.float64)), zoom)
    cx, cy = W / 2, H / 2
    hx, hy = (min(cx - box[0], box[2] - cx) * fx, min(cy - box[1], box[3] - cy) * fy)
    return bool(np.all(np.abs(q[:, 0] - cx) <= hx) and np.all(np.abs(q[:, 1] - cy) <= hy))


def painted_views(sources: Any) -> list[tuple[str, dict]]:
    """(png path, sidecar view) of every registered painted base (sidecar ok) under the given dirs or files."""
    out = []
    for src in sources or []:
        p = Path(src)
        files = sorted(p.glob("*_base.png")) if p.is_dir() else [p]
        for f in files:
            try:
                meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            v = meta.get("view") or {}
            if meta.get("ok") and {"lat", "lon", "zoom"} <= set(v):
                out.append((str(f), v))
    return out


class HouseMap(MapComp):
    """MapComp drawn in the house style inside a layout box: house chips at the type floor, amber route,
    no river names or free labels (route_map names only what is spoken), and a distance chip on its cue."""

    def __init__(self, spec: MotionSpec, payload: MapPayload, pal: Any, owner: "RouteMap") -> None:
        self.owner = owner
        self.ui_scale = owner.ui_scale
        super().__init__(spec, payload, pal, False)

    def _build_sprites(self) -> None:
        o, house = self.owner, self.owner.house
        x0, y0, x1, y1 = o.box
        self.safe_top, self.safe_bottom, self.margin = y0, y1, x0
        ink, paper, amber = house.rgb("ink"), house.rgb("paper"), house.rgb("accent")

        def chip(text: str, size: int, weight: str = "SemiBold") -> Sprite:
            f, ln = fit_text(text.upper(), weight, size, o.bw * 0.62, 1, o.fs(MIN_LABEL_PX), 0.08)
            return pill(ln[0], f, paper, ink, f.size * 0.45, f.size * 0.22, 0.08 * f.size, accent=amber, opacity=0.95)

        size = o.fs(36)
        self.marker_pills = [chip(m.label, size) if m.label else None for m, _ in self.markers]
        self.label_sprites = []
        self.route_pills = [chip(r.label, size) if r.label else None for r, _ in self.routes]
        self.arrow_pills = [None for _ in self.arrows]
        self.title = chip(self.p.title, o.fs(40), "Bold") if self.p.title else None
        self.locator = self._locator_sprite() if (min(self.z0, self.z1) >= 5.0 and not o.panel) else None
        self._note = o.credit
        self._layout()

    def _chip(self, uv: np.ndarray, spr: Optional[Sprite]) -> tuple[float, float]:
        # the line is the content: reserved before any chip (MapComp reserves it after the marker chips, so a
        # place chip could sit on the line it names)
        if not getattr(self, "_line_reserved", False):
            self._line_reserved = True
            rr = self.px(10)
            for _, ruv in self.routes:
                for pt in ruv[::max(1, len(ruv) // 48)]:
                    self._take([(x - rr, y - rr, x + rr, y + rr) for x, y in self._at(np.asarray([pt]))])
        return super()._chip(uv, spr)

    def _draw_routes(self, frame: np.ndarray, t: float, z: float) -> None:
        """Amber line on an ink casing, so it reads on pale land and on footage alike."""
        from ..mapdraw import dashes, polyline
        col, casing = self.owner.house.rgb("accent"), self.owner.house.rgb("ink")
        th = max(3, self.px(6))
        for k, pts, p in self._route_px(t, z):
            r = self.routes[k][0]
            pieces = dashes(pts, self.px(22), self.px(12)) if r.dashed else [pts]
            for piece in pieces:
                polyline(frame, piece, casing, th + max(3, self.px(5)))
            for piece in pieces:
                polyline(frame, piece, col, th)

    def _river_names(self) -> list:
        return []  # a river name under the type floor is unreadable, and at the floor it competes with places

    def _draw_route_heads(self, frame: np.ndarray, t: float, z: float) -> None:
        """Line head disc as MapComp draws it; the distance chip on its own cue instead of the line's half."""
        from ..mapdraw import disc, path_prefix
        col = self.pal.rgb("route")
        for k, (r, uv) in enumerate(self.routes):
            spr, cue = self.route_pills[k], self.owner.route_chip_cues[k]
            if spr is not None and cue is not None and t >= cue.appear:
                mid = self.proj.px(path_prefix(uv, 0.5), z)[-1]
                o = ez.ease_out_cubic(ez.progress(t, cue.appear, cue.appear + 0.45))
                dx, dy = self.route_offs[k]
                blit(frame, spr, mid[0] + dx, mid[1] + dy + self.px(12) * (1 - o), o)
        for k, pts, p in self._route_px(t, z):
            r = self.routes[k][0]
            fade_out = 1 - ez.progress(t, r.reveal_end, r.reveal_end + 0.6)
            if fade_out <= 0:
                continue
            x, y = pts[-1]
            rad = self.px(10) * fade_out
            disc(frame, x, y, rad * 1.4, col)
            disc(frame, x, y, rad * 0.7, (255, 255, 255))

    def _draw_title(self, frame: np.ndarray, t: float) -> None:
        o = ez.ease_out_cubic(ez.progress(t, self.owner.first_appear, self.owner.first_appear + 0.6))
        if self.locator is not None:
            blit(frame, self.locator, *self.locator_xy, o)
        if self.title is not None:
            x, y = self.title_xy
            blit(frame, self.title, x - self.px(30) * (1 - o), y, o)


@register(LayoutKind.route_map)
class RouteMap(LayoutComp):
    """Places, a route and a distance on a documentary map, full frame or as a panel."""

    panel_size = (0.34, 0.40)

    @classmethod
    def panel_share(cls, spec: Any) -> tuple[float, float]:
        """Share of the frame (w, h) as a panel: two places small, more places larger."""
        n = sum(1 for e in spec.elements if e.role == "place")
        return (0.34, 0.40) if n <= 2 else (0.40, 0.46) if n == 3 else (0.46, 0.52)

    def make_ground(self):  # noqa: ANN201
        return None

    def setup(self) -> None:
        job = self.job
        kr = min(self.W / 1920.0, self.H / 1080.0)
        self.ui_scale = max(1.0, 0.7 * self.OH / 1080.0 / kr)  # marker dots and lines readable in a panel
        self.first_appear = min((c.appear for _, _, c in job.els("place")), default=0.0)
        payload = self._verified()
        self._fit(payload)
        cf = font("Medium", self.fs(MIN_LABEL_PX))
        # the credit as a small ink chip: muted type straight on the map was lost over water (full frame only; a
        # panel has no room, the credit goes in the video's description)
        self.credit = None if self.panel else pill("Map data: Natural Earth", cf, self.house.rgb("muted"),
                                                   self.house.rgb("ink"), cf.size * 0.4, cf.size * 0.18, opacity=0.8)
        pal = get_palette("documentary_warm").model_copy(update={
            "route": self.house.accent, "accent": self.house.accent, "panel": self.house.ink,
            "panel_text": self.house.paper})
        mspec = MotionSpec(kind=MotionKind.map, duration=self.dur, width=self.W, height=self.H, fps=self.fps,
                           background="solid", style=StyleTokens(palette="documentary_warm"),
                           payload=payload.model_dump())
        self.map = HouseMap(mspec, payload, pal, self)
        painted = getattr(self.map.base, "art_overlay", None) is not None
        job.notes.append(f"route_map base: {'painted ' + Path(payload.art_base).name if painted else 'plain (drawn)'}"
                         + ("" if painted or not payload.art_base else ", painting did not match the view"))

    # ------------------------------------------------------------ checks (seat.verify_map's, through its functions)

    def _verified(self) -> MapPayload:
        """Places geocoded, routes traced on real lines, distances checked; what cannot be drawn goes in job.missing.

        The checks of seat.verify_map, made here because a layout knows each place's own cue and may qualify a
        name ('Santa Monica, California': geocoded with the qualifier, labelled 'Santa Monica')."""
        from ...shots import seat
        from .. import routes as R
        job = self.job
        mp = MapPayload(center=[0.0, 0.0], title=self.spec.title)  # never a drawn title: it names places unspoken
        hits: dict[str, Any] = {}
        first = {}
        for _, e, c in job.els("place"):
            name, qual = split_place(e.text)
            if name:
                first[e.text] = resolve(name, qual)
        near = [(h.lat, h.lon) for h in first.values() if h is not None]
        for _, e, c in job.els("place"):
            name, qual = split_place(e.text)
            if not name:
                continue
            hit = first.get(e.text) or resolve(name, qual, near)
            if hit is None:
                job.missing.append(f"place '{e.text}' did not resolve (unknown or two homonyms on this map)")
                continue
            hits[e.text] = (hit, c)
            mp.markers.append(MapMarker(lat=hit.lat, lon=hit.lon, label=name, appear_at=round(c.appear, 3)))
        dist = job.first("distance")
        self.route_chip_cues = []
        for _, e, c in job.els("route"):
            parsed = parse_route(e.text)
            if parsed is None:
                job.missing.append(f"route '{e.text}' is not '<from> to <to>'")
                continue
            a, b, via = parsed
            ends = []
            for x in (a, b):
                own = next((hits[k] for k in hits if _same(split_place(k)[0], split_place(x)[0])), None)
                ends.append(own or (resolve(*split_place(x), near), None))
            (ha, fc), (hb, tc) = ends
            if ha is None or hb is None or (abs(ha.lat - hb.lat) + abs(ha.lon - hb.lon)) < 0.01:
                job.missing.append(f"route '{e.text}': an end did not resolve, no line")
                continue
            rs, re_ = route_window(fc, tc, c)
            via = via if via in R.VIAS else R.infer_via(e.text)
            pts, mode = next(((q, v) for v in R.modes(via) if (q := R.route_points((ha.lat, ha.lon), (hb.lat, hb.lon), v))),
                             (None, via))
            label = dist[1].text.strip() if (dist is not None and not mp.routes) else ""
            if pts is not None:
                measured = R.path_miles(pts)
                kept = seat.checked_distance(label, measured)
                if label and not kept:
                    job.notes.append(f"distance '{label}' is not within 25 percent of the drawn {mode} line "
                                     f"({measured:.0f} mi): dropped")
                label, dashed, pts = kept, mode in R.ARC_VIAS, R.thin(pts)
            elif via in ("trail", "line"):
                pts, mode, dashed = R.great_circle((ha.lat, ha.lon), (hb.lat, hb.lon)), "trail", True
                straight = R.miles((ha.lat, ha.lon), (hb.lat, hb.lon))
                claim = seat.distance_claim(label)
                ok = claim is None or straight * 0.95 <= claim[0] <= straight * 2.5
                job.notes.append(f"route '{e.text}': no bundled {via} line, drawn as a dashed great circle "
                                 f"({straight:.0f} mi straight)" + ("" if ok else f"; distance '{label}' implausible, dropped"))
                label = label if ok else ""
            else:
                job.missing.append(f"route '{e.text}': no {via} line joins the two places")
                continue
            mp.routes.append(MapRoute(points=pts, reveal_start=round(rs, 3), reveal_end=round(re_, 3), label=label,
                                      dashed=dashed, via=mode, from_place=ha.name, to_place=hb.name))
            self.route_chip_cues.append(dist[2] if label else None)
            for h, at in ((ha, rs), (hb, re_)):  # an end that is not a place element shows on the route's timing
                if not any(abs(h.lat - m.lat) + abs(h.lon - m.lon) < 0.05 for m in mp.markers):
                    mp.markers.append(MapMarker(lat=h.lat, lon=h.lon, label=h.name, appear_at=round(max(0.0, at - 0.25), 3)))
        if dist is not None and not any(self.route_chip_cues):
            job.notes.append(f"distance '{dist[1].text}' has no checked line to sit on: not shown")
        return mp

    def _fit(self, mp: MapPayload) -> None:
        pts = [(m.lat, m.lon) for m in mp.markers] + [(q[0], q[1]) for r in mp.routes for q in r.points]
        if not pts:
            self.job.missing.append("route_map: nothing resolved, an empty map")
            mp.center, mp.zoom, mp.zoom_end = [20.0, 0.0], 1.0, 1.0 + PUSH
            return
        center, z_end = fit_view(pts, self.W, self.H, self.box)
        mp.center, mp.zoom, mp.zoom_end = center, round(z_end - PUSH, 2), z_end
        if abs(self.W / self.H - ART_ASPECT) > 0.03 * ART_ASPECT:
            if self.job.extra.get("art_bases"):
                self.job.notes.append("route_map: panel shape is not the painting's 16:9, plain base")
            return
        from ...art.hooks import _matches
        for path, v in painted_views(self.job.extra.get("art_bases")):
            c, z = [float(v["lat"]), float(v["lon"])], float(v["zoom"])
            if z < z_end - PUSH - ART_ZOOM_SLACK:
                continue  # far wider than the places need
            # the deepest push that keeps every place in the box; with none, the house drift is the push
            push = next((d for d in (PUSH, PUSH / 2, 0.0) if inside(pts, c, z + d, self.W, self.H, self.box, 0.9, 0.85)),
                        None)
            if push is None:
                continue
            trial = mp.model_copy(update={"center": c, "zoom": z, "zoom_end": round(z + push, 2), "art_base": path})
            if _matches(trial, path):
                mp.center, mp.zoom, mp.zoom_end, mp.art_base = trial.center, trial.zoom, trial.zoom_end, path
                if push < PUSH:
                    self.job.notes.append(f"route_map: painted view allows a push of {push:.2f} zoom steps only")
                return

    # ------------------------------------------------------------ frame

    def canvas(self, i: int) -> np.ndarray:
        return self.map.render(i)

    def draw(self, frame: np.ndarray, t: float) -> None:
        pass
