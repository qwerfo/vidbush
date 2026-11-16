"""Map composition over bundled Natural Earth data (never a tile server).

Projection: local equirectangular with a cos(center latitude) correction,
matching vidbush/mg/builders.map_view (zoom 1 = 360 degrees of longitude
across the frame width, each zoom step doubles the scale).

Performance: the static map (water, graticule, coast halo, land with texture,
borders, lakes, rivers, shaded regions) is drawn once into a base image at
the resolution needed for the deepest zoom of the push-in. Each frame is one
affine warp of that base plus the animated parts (routes, arrows, markers,
labels and river names, title, locator), vignette and grain.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..art import hooks as art_hooks
from ..config import REPO_ROOT
from . import easing as ez
from .palettes import RGB, mix
from .textdraw import (
    CompBase,
    _round_rect_mask,
    GlowLayer,
    Grain,
    Sprite,
    arrow_head,
    blit,
    coloured,
    disc,
    fill_poly,
    fit_text,
    font,
    path_prefix,
    pill,
    polyline,
    text_mask,
    texture,
    vignette_mask,
)

GEO_DIR = Path(REPO_ROOT) / "assets" / "geo"
ADMIN1 = "ne_50m_admin_1_states_provinces.geojson"
RIVERS = "ne_50m_rivers_lake_centerlines.geojson"
RIVER_RANK = 6  # only major rivers are drawn, at every zoom, so a minor one never reads as a route
RIVER_ZOOM = 3.5  # rivers and their names from this zoom
STATE_ZOOM = 3.5  # state and province borders from this zoom
CONTEXT_ZOOM = 5.0  # faint roads and railways from this (regional) zoom
LOCATOR_ZOOM = 5.0  # a view this close (under about 22 degrees of longitude) gets a locator map
LOCATOR_STEPS = 3.0  # the locator spans 2**3 = 8 times the view, so the view reads as a small box in context
Ring = np.ndarray  # (n, 2) lon, lat
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in frame pixels


# ---------------------------------------------------------------- data

@lru_cache(maxsize=16)
def load_layer(name: str) -> list[tuple[dict, list[Ring]]]:
    """(properties, rings) per feature of a bundled GeoJSON layer; rings are lon/lat float arrays."""
    data = json.loads((GEO_DIR / name).read_text(encoding="utf-8"))
    out = []
    for f in data["features"]:
        g = f.get("geometry") or {}
        t, c = g.get("type"), g.get("coordinates") or []
        if t == "Polygon":
            rings = c
        elif t == "MultiPolygon":
            rings = [r for poly in c for r in poly]
        elif t == "LineString":
            rings = [c]
        elif t == "MultiLineString":
            rings = c
        else:
            continue
        out.append((f.get("properties") or {}, [np.asarray(r, np.float64)[:, :2] for r in rings if len(r) > 1]))
    return out


def major_rivers() -> list[tuple[dict, list[Ring]]]:
    """River features drawn on a map: RIVER_RANK and above."""
    return [f for f in load_layer(RIVERS) if (f[0].get("scalerank") or 9) <= RIVER_RANK]


def _names(props: dict) -> set[str]:
    from .geo import norm
    keys = ("NAME", "NAME_LONG", "ADMIN", "SOVEREIGNT", "ISO_A3", "ADM0_A3", "BRK_NAME", "FORMAL_EN", "name")
    return {norm(str(props[k])) for k in keys if props.get(k) and props.get(k) != "-99"}


# ---------------------------------------------------------------- projection

class Projection:
    """World units: u = longitude (unwrapped near the center), v = -latitude / cos(center latitude)."""

    def __init__(self, lat: float, lon: float, w: int, h: int) -> None:
        self.coslat = max(0.3, math.cos(math.radians(lat)))
        self.cu, self.cv = lon, -lat / self.coslat
        self.w, self.h = w, h

    def scale(self, zoom: float) -> float:
        return self.w * 2 ** (zoom - 1) / 360.0

    def uv(self, latlon: np.ndarray) -> np.ndarray:
        """[[lat, lon], ...] to world units, longitudes unwrapped along the path and near the center."""
        lat, lon = latlon[:, 0], np.unwrap(latlon[:, 1], period=360)
        lon = lon + 360 * np.round((self.cu - lon[0]) / 360)
        return np.stack([lon, -lat / self.coslat], axis=1)

    def px(self, uv: np.ndarray, zoom: float) -> np.ndarray:
        s = self.scale(zoom)
        return np.stack([(uv[:, 0] - self.cu) * s + self.w / 2, (uv[:, 1] - self.cv) * s + self.h / 2], axis=1)

    def ring_px(self, ring: Ring, s: float, w: int, h: int, shift: float) -> np.ndarray:
        """Lon/lat ring to pixels of an image of size w x h at px-per-unit s, shifted by `shift` degrees."""
        x = (ring[:, 0] + shift - self.cu) * s + w / 2
        y = (-ring[:, 1] / self.coslat - self.cv) * s + h / 2
        return np.stack([x, y], axis=1)


def _shifts(ring: Ring, cu: float, half_span: float) -> list[float]:
    """Longitude shifts (multiples of 360) that bring the ring into the visible span."""
    lo, hi = ring[:, 0].min(), ring[:, 0].max()
    return [k for k in (-360.0, 0.0, 360.0) if hi + k >= cu - half_span and lo + k <= cu + half_span]


# ---------------------------------------------------------------- static base

class MapBase:
    """Pre-rendered static map for a view; `plain` and `shaded` images (uint8, C channels)."""

    def __init__(self, comp: "MapComp", z_lo: float, z_hi: float) -> None:
        self.c = comp
        self.S = float(np.clip(2 ** (z_hi - z_lo), 1.0, 2.5))
        self.bw, self.bh = int(comp.W * self.S), int(comp.H * self.S)
        self.z_lo = z_lo
        self.s = comp.proj.scale(z_lo) * self.S
        self.half_span = self.bw / 2 / self.s * 1.05
        self.lat_span = self.bh / 2 / self.s * comp.proj.coslat * 1.05
        detail = z_hi >= 3
        name = "ne_50m_admin_0_countries.geojson" if detail else "ne_110m_admin_0_countries.geojson"
        self.countries = load_layer(name)
        self.k = comp.k * self.S
        self.plain = self._render()
        self.plain = art_hooks.painted_base(self, self.plain)  # run 3 art layer: painted base of this view
        self.shaded = self._shade(self.plain) if comp.p.shaded_regions else None
        self.half = None if self.S < 1.6 else self._halves()

    def _halves(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        def h(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if img is None:
                return None
            return cv2.resize(img, (self.bw // 2, self.bh // 2), interpolation=cv2.INTER_AREA)
        return h(self.plain), h(self.shaded)

    def rings(self, layer: list[tuple[dict, list[Ring]]], keep=None) -> list[np.ndarray]:  # noqa: ANN001
        """Visible rings of a layer in base pixel coordinates."""
        proj, out = self.c.proj, []
        clat = -proj.cv * proj.coslat
        for props, rings in layer:
            if keep is not None and not keep(props):
                continue
            for r in rings:
                if r[:, 1].max() < clat - self.lat_span or r[:, 1].min() > clat + self.lat_span:
                    continue
                for sh in _shifts(r, proj.cu, self.half_span):
                    out.append(proj.ring_px(r, self.s, self.bw, self.bh, sh))
        return out

    def _render(self) -> np.ndarray:
        pal, k = self.c.pal, self.k
        water = np.array(pal.rgb("water"), np.float32)
        tex = texture(self.bw, self.bh, 11, 2.5)[..., None]
        img = np.empty((self.bh, self.bw, 3), np.float32)
        img[...] = water
        img *= 1.0 + tex * 0.05
        land_rings = self.rings(self.countries)
        mask = np.zeros((self.bh, self.bw), np.uint8)
        if land_rings:
            cv2.fillPoly(mask, [np.round(r * 16).astype(np.int32) for r in land_rings], 255, cv2.LINE_AA, 4)
        halo = cv2.GaussianBlur(mask, (0, 0), max(1.0, 16 * k)).astype(np.float32)[..., None] / 255.0
        light = np.array(mix(pal.rgb("water"), (255, 255, 255), 0.22 if not pal.dark else 0.1), np.float32)
        img = img * (1 - halo * 0.6) + light * halo * 0.6
        img = np.clip(img, 0, 255).astype(np.uint8)
        self._graticule(img)
        coast = mix(pal.rgb("water"), pal.rgb("text"), 0.28)
        for r in land_rings:
            polyline(img, r, coast, max(1.0, 2.2 * k))
        land = self._land_layer(mask)
        a = mask.astype(np.float32)[..., None] / 255.0
        img = (img.astype(np.float32) * (1 - a) + land * a)
        img = np.clip(img, 0, 255).astype(np.uint8)
        self._water_features(img)
        self._context_lines(img)
        for r in land_rings:
            polyline(img, r, pal.rgb("land_border"), max(1.0, 1.3 * k))
        grain = texture(self.bw, self.bh, 5, 40.0, 2)[..., None]
        img = np.clip(img.astype(np.float32) * (1.0 + grain * (0.035 if not pal.dark else 0.05)), 0, 255)
        return self._channels(img.astype(np.uint8))

    def _land_layer(self, mask: np.ndarray) -> np.ndarray:
        """Land colour with terrain texture and a soft inner shade toward the coast."""
        pal = self.c.pal
        land = np.empty((self.bh, self.bw, 3), np.float32)
        land[...] = pal.rgb("land")
        terrain = texture(self.bw, self.bh, 23, 6.0, 5)[..., None]
        land *= 1.0 + terrain * (0.06 if not pal.dark else 0.1)
        inner = cv2.GaussianBlur(mask, (0, 0), max(1.0, 10 * self.k)).astype(np.float32)[..., None] / 255.0
        land *= 0.9 + 0.1 * inner
        return land

    def _graticule(self, img: np.ndarray) -> None:
        z = self.z_lo
        step = 30 if z < 2 else 15 if z < 3 else 10 if z < 4 else 5 if z < 5 else 2 if z < 6.5 else 1
        col = mix(self.c.pal.rgb("water"), self.c.pal.rgb("grid"), 0.55)
        proj = self.c.proj
        lons = np.arange(-180, 180.001, 1.0)
        for lat in range(-80, 81, step):
            ring = np.stack([lons, np.full_like(lons, lat)], axis=1)
            for sh in (-360.0, 0.0, 360.0):
                polyline(img, proj.ring_px(ring, self.s, self.bw, self.bh, sh), col, max(1.0, self.k))
        lats = np.arange(-85, 85.001, 1.0)
        for lon in range(-180, 180, step):
            ring = np.stack([np.full_like(lats, lon), lats], axis=1)
            for sh in _shifts(ring, proj.cu, self.half_span):
                polyline(img, proj.ring_px(ring, self.s, self.bw, self.bh, sh), col, max(1.0, self.k))

    def _water_features(self, img: np.ndarray) -> None:
        pal, z = self.c.pal, self.z_lo
        water = pal.rgb("water")
        lakes = self.rings(load_layer("ne_50m_lakes.geojson"),
                           None if z >= 3.5 else (lambda p: (p.get("scalerank") or 9) <= 1))
        if lakes:
            fill_poly(img, lakes, mix(water, (255, 255, 255), 0.06))
            for r in lakes:
                polyline(img, r, mix(water, pal.rgb("text"), 0.25), max(1.0, 1.2 * self.k))
        if z >= RIVER_ZOOM:
            for r in self.rings(major_rivers()):
                polyline(img, r, mix(water, pal.rgb("land"), 0.15), max(1.0, 1.6 * self.k))

    def _context_lines(self, img: np.ndarray) -> None:
        """Faint state and province borders when zoomed in, faint roads and railways at regional zooms."""
        pal, z, k = self.c.pal, self.z_lo, self.k
        land = pal.rgb("land")
        if z >= STATE_ZOOM:
            for r in self.rings(load_layer(ADMIN1)):
                polyline(img, r, mix(land, pal.rgb("land_border"), 0.45), max(1.0, 0.9 * k))
        if z >= CONTEXT_ZOOM:
            rank = 99 if z >= CONTEXT_ZOOM + 2 else 6
            for name, f in (("ne_10m_roads.geojson", 0.12), ("ne_10m_railroads.geojson", 0.22)):
                for r in self.rings(load_layer(name), lambda p: (p.get("scalerank") or 9) <= rank):
                    polyline(img, r, mix(land, pal.rgb("text"), f), max(1.0, 0.9 * k))

    def _shade(self, plain: np.ndarray) -> np.ndarray:
        """Shaded regions: accent tinted fill, accent outline and a soft outer glow."""
        from .geo import admin1_key, norm
        pal = self.c.pal
        wanted = {norm(n) for n in self.c.p.shaded_regions if n}
        # a state is matched by geo.admin1_key, so 'Georgia' fills the country and never the US state too
        rings = (self.rings(self.countries, lambda p: bool(_names(p) & wanted))
                 + self.rings(load_layer(ADMIN1), lambda p: bool(p.get("name")) and norm(admin1_key(p)) in wanted))
        img = plain.copy()
        if not rings:
            return img
        mask = np.zeros((self.bh, self.bw), np.uint8)
        cv2.fillPoly(mask, [np.round(r * 16).astype(np.int32) for r in rings], 255, cv2.LINE_AA, 4)
        edge = np.zeros_like(mask)
        cv2.polylines(edge, [np.round(r * 16).astype(np.int32) for r in rings], True, 255,
                      max(1, int(3 * self.k)), cv2.LINE_AA, 4)
        glow = cv2.GaussianBlur(edge, (0, 0), max(1.0, 14 * self.k)).astype(np.float32) / 255.0
        a = mask.astype(np.float32) / 255.0
        f = img[..., :3].astype(np.float32)
        shade = np.array(pal.rgb("shade"), np.float32)
        accent = np.array(pal.rgb("accent"), np.float32)
        f = f * (1 - a[..., None] * 0.55) + shade * (a[..., None] * 0.55)
        g = np.clip(glow * 1.8, 0, 0.85)[..., None]
        f = f * (1 - g) + accent * g
        e = edge.astype(np.float32)[..., None] / 255.0
        f = f * (1 - e) + accent * e
        img[..., :3] = np.clip(f, 0, 255).astype(np.uint8)
        return img

    def _channels(self, img: np.ndarray) -> np.ndarray:
        if self.c.C == 3:
            return img
        return np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])

    def view(self, zoom: float, shade: float) -> np.ndarray:
        """Frame at zoom: one affine warp of the base (blended with the shaded base while it fades in)."""
        m = 2 ** (zoom - self.z_lo) / self.S
        plain, shaded, bw, bh = self.plain, self.shaded, self.bw, self.bh
        if self.half is not None and m < 0.62:
            plain, shaded = self.half
            m, bw, bh = m * 2, bw // 2, bh // 2
        W, H = self.c.W, self.c.H
        cw, ch = W / m, H / m
        x0, y0 = (bw - cw) / 2, (bh - ch) / 2
        ix0, iy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        ix1, iy1 = min(bw, int(np.ceil(x0 + cw)) + 1), min(bh, int(np.ceil(y0 + ch)) + 1)
        ox, oy = int(round((x0 - ix0) * m)), int(round((y0 - iy0) * m))
        size = (max(W + ox, int(round((ix1 - ix0) * m))), max(H + oy, int(round((iy1 - iy0) * m))))
        interp = cv2.INTER_AREA if m < 0.8 else cv2.INTER_LINEAR

        def warp(src: np.ndarray) -> np.ndarray:
            big = cv2.resize(src[iy0:iy1, ix0:ix1], size, interpolation=interp)
            return np.ascontiguousarray(big[oy:oy + H, ox:ox + W])
        if shaded is None or shade <= 0.001:
            return warp(plain)
        if shade >= 0.999:
            return warp(shaded)
        return cv2.addWeighted(warp(plain), 1 - shade, warp(shaded), shade, 0)


# ---------------------------------------------------------------- geometry helpers

def smooth_path(pts: np.ndarray, samples: int = 16) -> np.ndarray:
    """Two points: gentle arc. More: centripetal Catmull-Rom through every point."""
    if len(pts) < 2:
        return pts
    if len(pts) == 2:
        a, b = pts
        d = b - a
        length = float(np.hypot(*d)) or 1.0
        n = np.array([-d[1], d[0]]) / length
        if n[1] > 0:
            n = -n
        c = (a + b) / 2 + n * length * 0.14
        u = np.linspace(0, 1, 48)[:, None]
        return (1 - u) ** 2 * a + 2 * (1 - u) * u * c + u ** 2 * b
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        u = np.linspace(0, 1, samples, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(pts[-1:])
    return np.vstack(out)


def dashes(pts: np.ndarray, on: float, off: float) -> list[np.ndarray]:
    """Split a pixel polyline into dash pieces of length `on` separated by `off`."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out, s = [], 0.0
    while s < cum[-1]:
        e = min(cum[-1], s + on)
        ts = np.linspace(s, e, max(2, int((e - s) / 4) + 2))
        out.append(np.stack([np.interp(ts, cum, pts[:, 0]), np.interp(ts, cum, pts[:, 1])], axis=1))
        s = e + off
    return out


def halo_text(text: str, size: int, weight: str, color: RGB, halo: RGB, tracking_em: float, width: int) -> Sprite:
    """Map label: text with a soft halo in the map colour so it reads over borders."""
    f = font(weight, size)
    pad = width + 2
    m = text_mask(text, f, tracking_em * size, pad)
    k = 2 * width + 1
    h = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    h = cv2.GaussianBlur(h, (0, 0), max(0.6, width * 0.5))
    arr = coloured(h, halo, 0.85)
    txt = coloured(m, color)
    sa = txt[..., 3:4].astype(np.float32) / 255.0
    arr = (txt.astype(np.float32) + arr.astype(np.float32) * (1 - sa)).clip(0, 255).astype(np.uint8)
    return Sprite(arr, pad)


# ---------------------------------------------------------------- composition

class MapComp(CompBase):
    """Documentary map: slow eased push-in, shaded regions, staggered markers, revealed routes, arrows."""

    paints_background = True

    def __init__(self, spec, payload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        p = payload
        lat, lon = (p.center + [0.0, 0.0])[:2]
        self.proj = Projection(float(np.clip(lat, -85, 85)), float(lon), self.W, self.H)
        self.z0 = float(np.clip(p.zoom, 0.6, 12))
        self.z1 = float(np.clip(p.zoom_end, 0.6, 12)) if p.zoom_end is not None else self.z0
        self.base = MapBase(self, min(self.z0, self.z1), max(self.z0, self.z1))
        self.vig = cv2.merge([vignette_mask(self.W, self.H, 0.42 if pal.dark else 0.3)] * self.C)
        if self.C == 4:
            self.vig[..., 3] = 255
        self._grain = Grain(self.W, self.H, self.C, 4.0)
        self.glow = GlowLayer(self.W, self.H)
        self.routes = []
        for r in p.routes:
            if len(r.points) >= 2:
                uv = self.proj.uv(np.asarray(r.points, np.float64))
                self.routes.append((r, uv if r.via else smooth_path(uv)))  # traced geometry is drawn exactly
        self.arrows = [(a, smooth_path(self.proj.uv(np.asarray([a.start[:2], a.end[:2]], np.float64))))
                       for a in p.arrows if len(a.start) >= 2 and len(a.end) >= 2]
        self.markers = [(m, self.proj.uv(np.array([[m.lat, m.lon]], np.float64))) for m in p.markers]
        self._build_sprites()

    def canvas(self) -> np.ndarray:
        return np.zeros((self.H, self.W, self.C), np.uint8)

    def zoom_at(self, t: float) -> float:
        return ez.lerp(self.z0, self.z1, ez.ease_in_out_sine(ez.progress(t, 0.0, self.dur)))

    # ------------------------------------------------------------ sprites and placement

    def _build_sprites(self) -> None:
        pal = self.pal
        f = font("SemiBold", self.px(28))
        self.marker_pills = [pill(m.label, f, pal.rgb("panel_text"), pal.rgb("panel"), self.px(14), self.px(8),
                                  accent=pal.rgb("accent")) if m.label else None for m, _ in self.markers]
        halo = mix(pal.rgb("land"), pal.rgb("bg"), 0.3)
        sizes = {"small": (24, "Medium", 0.04), "medium": (32, "SemiBold", 0.08), "large": (46, "Bold", 0.22)}
        self.label_sprites = []
        for lb in self.p.labels:
            size, weight, tr = sizes.get(lb.size, sizes["medium"])
            text = lb.text.upper() if lb.size == "large" else lb.text
            self.label_sprites.append((lb, self.proj.uv(np.array([[lb.lat, lb.lon]], np.float64)),
                                       halo_text(text, self.px(size), weight, pal.rgb("text"), halo, tr,
                                                 max(2, self.px(4)))))
        rf = font("SemiBold", self.px(26))
        self.route_pills = [pill(r.label, rf, pal.rgb("panel_text"), pal.rgb("panel"), self.px(12), self.px(7))
                            if r.label else None for r, _ in self.routes]
        self.arrow_pills = [pill(a.label, rf, pal.rgb("panel_text"), pal.rgb("panel"), self.px(12), self.px(7))
                            if a.label else None for a, _ in self.arrows]
        self.title = self._title_sprite()
        self.locator = self._locator_sprite() if min(self.z0, self.z1) >= LOCATOR_ZOOM else None
        self._layout()

    def _title_sprite(self) -> Optional[Sprite]:
        if not self.p.title:
            return None
        f, ln = fit_text(self.p.title.upper(), "Bold", self.px(40), self.W * 0.55, 1, self.px(22), 0.08)
        return pill(ln[0], f, self.pal.rgb("panel_text"), self.pal.rgb("panel"), self.px(24), self.px(14),
                    0.08 * f.size, accent=self.pal.rgb("accent"))

    def _locator_sprite(self) -> Sprite:
        """Locator: the area LOCATOR_STEPS zoom steps wider than the final view, same centre, view outlined.

        A close-up map of plain land with a few dots says nothing about where on Earth it is; a small map of
        country and state outlines around it with the view drawn as a box does.
        """
        pal = self.pal
        lw, lh = self.px(300), self.px(190)
        zl = max(1.0, self.z1 - LOCATOR_STEPS)
        proj = Projection(-self.proj.cv * self.proj.coslat, self.proj.cu, lw, lh)
        s = proj.scale(zl)
        half = lw / 2 / s * 1.05

        def rings(layer: list[tuple[dict, list[Ring]]]) -> list[np.ndarray]:
            out = []
            for _, rs in layer:
                for r in rs:
                    for sh in _shifts(r, proj.cu, half):
                        q = proj.ring_px(r, s, lw, lh, sh)
                        if q[:, 1].max() >= 0 and q[:, 1].min() <= lh:
                            out.append(q)
            return out
        img = np.empty((lh, lw, 3), np.uint8)
        img[...] = pal.rgb("water")
        land = rings(load_layer("ne_110m_admin_0_countries.geojson"))
        fill_poly(img, land, pal.rgb("land"))
        if zl >= STATE_ZOOM - 1:
            for r in rings(load_layer(ADMIN1)):
                polyline(img, r, mix(pal.rgb("land"), pal.rgb("land_border"), 0.4), 1.0)
        for r in land:
            polyline(img, r, pal.rgb("land_border"), max(1.0, self.px(1.5)))
        vw = max(self.px(14), self.W / self.proj.scale(self.z1) * s)
        vh = max(self.px(9), self.H / self.proj.scale(self.z1) * s)
        box = np.array([[-vw, -vh], [vw, -vh], [vw, vh], [-vw, vh]]) / 2 + [lw / 2, lh / 2]
        polyline(img, box, pal.rgb("panel"), max(3, self.px(6)), closed=True)
        polyline(img, box, pal.rgb("accent"), max(2, self.px(3)), closed=True)
        mask = _round_rect_mask(lw, lh, max(2, self.px(10))).astype(np.float32) / 255.0
        edge = mask - cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=max(1, self.px(2)))
        panel = np.array(pal.rgb("panel"), np.float32)
        rgb = img.astype(np.float32) * (1 - edge[..., None]) + panel * edge[..., None]
        arr = np.empty((lh, lw, 4), np.uint8)
        arr[..., :3] = np.clip(rgb * mask[..., None], 0, 255).astype(np.uint8)
        arr[..., 3] = np.clip(mask * 255, 0, 255).astype(np.uint8)
        return Sprite(arr)

    # ------------------------------------------------------------ layout: one collision set

    def _layout(self) -> None:
        """Place every piece of text and chrome through one collision set, in priority order.

        No text may sit on top of other text, and a label that cannot be read is not drawn. Fixed reservations
        come first: the source credit, the marker dots, the route and arrow chips (above their point, else
        below), the marker chips and the route line, then the title chip and the locator, each in the first
        clear corner. Then free labels, region labels, river names. A label that overlaps anything already
        placed tries its next candidate position; free, region and river labels with no free candidate are
        skipped (a marker chip, route chip or arrow chip keeps its first side, since its name is the content).
        Each river name is drawn at most once. Map-anchored boxes are tested at the start, middle and end zoom of
        the push-in against what sits there at that zoom, so they stay clear for the whole shot.
        """
        self.zs = sorted({self.z0, (self.z0 + self.z1) / 2, self.z1})
        self.taken: list[list[Box]] = [[] for _ in self.zs]  # boxes placed so far, per tested zoom
        self.note_xy = (0.0, 0.0)
        if self._note is not None:
            self.note_xy = (self.W - self.margin * 0.55 - self._note.w, self.safe_bottom - self._note.h - self.px(14))
            self._take([_box(self.note_xy, self._note)] * len(self.zs))
        r = self.px(16)
        for _, uv in self.markers:
            self._take([(x - r, y - r, x + r, y + r) for x, y in self._at(uv)])
        # route and arrow chips go in with the marker chips, which then route around them: reserved after every
        # marker chip, a route chip covered the chip of a place near the route's midpoint
        self.route_offs = [self._chip(path_prefix(uv, 0.5)[-1:], spr)
                           for spr, (_, uv) in zip(self.route_pills, self.routes)]
        self.arrow_offs = [self._chip(uv[:1], spr) for spr, (_, uv) in zip(self.arrow_pills, self.arrows)]
        self.offsets = []
        r = self.px(22)
        for (_, uv), spr in zip(self.markers, self.marker_pills):
            if spr is None:
                self.offsets.append((0.0, 0.0))
                continue
            options = [(r, -spr.h / 2), (-r - spr.w, -spr.h / 2), (-spr.w / 2, -r - spr.h), (-spr.w / 2, r)]
            off = self._place(uv, spr, options)
            if off is None:
                off = options[0]
                self._take(self._boxes(uv, spr, *off))
            self.offsets.append(off)
        # the place names and the line they explain are the content, so the title and the locator take a corner
        # clear of both (a rebuild of run 2 put the locator over the Red Devil chip and the title on the route)
        rr = self.px(8)
        for _, uv in self.routes:
            for pt in uv[::max(1, len(uv) // 24)]:
                self._take([(x - rr, y - rr, x + rr, y + rr) for x, y in self._at(np.asarray([pt]))])
        self.title_xy = self._corner(self.title, self.px(30)) if self.title is not None else None
        if self.title_xy is None:
            # the title only repeats the names the map draws, so with no clear corner it is left out rather than
            # printed over a place name (a rebuild hid the Utqiagvik chip under the title)
            self.title = None
        self.locator_xy = self._corner(self.locator, self.px(20)) if self.locator is not None else None
        if self.locator_xy is None:
            self.locator = None  # no corner is clear of the credit, the markers, the route and the title
        self.labels: list[tuple[np.ndarray, Sprite, tuple[float, float]]] = []
        g = self.px(6)
        for _, uv, spr in sorted(self.label_sprites, key=lambda e: e[0].size == "large"):  # free before region
            off = self._place(uv, spr, [(-spr.w / 2, -spr.h / 2), (-spr.w / 2, -spr.h * 1.5 - g),
                                        (-spr.w / 2, spr.h / 2 + g)])
            if off is not None:
                self.labels.append((uv, spr, off))
        self.rivers = self._river_names()

    def _chip(self, uv: np.ndarray, spr: Optional[Sprite]) -> tuple[float, float]:
        """Offset of a route or arrow chip from its point: above, else below, else above anyway (reserved)."""
        if spr is None:
            return 0.0, 0.0
        options = [(-spr.w / 2, -spr.h - self.px(18)), (-spr.w / 2, self.px(18))]
        off = self._place(uv, spr, options)
        if off is None:
            off = options[0]
            self._take(self._boxes(uv, spr, *off))
        return off

    def _corner(self, spr: Sprite, inset: float) -> Optional[tuple[float, float]]:
        """First safe-area corner (upper left, upper right, lower left, lower right, above the credit) where spr
        overlaps nothing.

        None when no corner is clear: the caller then leaves the piece out."""
        left, right = self.margin, self.W - self.margin - spr.w
        top, bottom = self.safe_top + inset, self.safe_bottom - spr.h - inset
        above_note = self.note_xy[1] - spr.h - self.px(10) if self._note is not None else bottom
        for xy in ((left, top), (right, top), (left, bottom), (right, bottom), (right, above_note)):
            if self._fits([_box(xy, spr)] * len(self.zs)):
                self._take([_box(xy, spr)] * len(self.zs))
                return xy
        return None

    def _at(self, uv: np.ndarray) -> list[tuple[float, float]]:
        """Pixel position of a world point at each tested zoom."""
        return [tuple(self.proj.px(uv[:1], z)[0]) for z in self.zs]

    def _boxes(self, uv: np.ndarray, spr: Sprite, dx: float, dy: float) -> list[Box]:
        return [(x + dx, y + dy, x + dx + spr.w, y + dy + spr.h) for x, y in self._at(uv)]

    def _take(self, boxes: list[Box]) -> None:
        """Reserve one box per tested zoom."""
        for placed, b in zip(self.taken, boxes):
            placed.append(b)

    def _fits(self, boxes: list[Box]) -> bool:
        """Inside the safe area and clear of everything placed at the same zoom."""
        m = self.px(16)
        return all(b[0] >= m and b[2] <= self.W - m and b[1] >= self.safe_top and b[3] <= self.safe_bottom
                   and not any(_overlap(b, t) for t in self.taken[i]) for i, b in enumerate(boxes))

    def _place(self, uv: np.ndarray, spr: Sprite, options: list[tuple[float, float]]) -> Optional[tuple[float, float]]:
        """First offset whose boxes fit, reserved; None when no option fits."""
        for dx, dy in options:
            boxes = self._boxes(uv, spr, dx, dy)
            if self._fits(boxes):
                self._take(boxes)
                return dx, dy
        return None

    def _river_names(self) -> list[tuple[str, np.ndarray, Sprite, tuple[float, float]]]:
        """One name per drawn river, on its longest stretch in view: tried at the middle, then the thirds and
        quarters; a river whose name fits nowhere goes unnamed. Natural Earth splits a river into several
        features, so stretches are merged by name first: a name printed twice reads as two rivers."""
        if self.base.z_lo < RIVER_ZOOM:
            return []
        pal, proj, m = self.pal, self.proj, self.px(40)
        half = self.W / 2 / proj.scale(min(self.zs)) * 1.05
        stretches: dict[str, tuple[str, np.ndarray]] = {}
        for props, rings in major_rivers():
            name = str(props.get("name") or props.get("name_en") or "").strip()
            if not name:
                continue
            for r in rings:
                for sh in _shifts(r, proj.cu, half):
                    uv = np.stack([r[:, 0] + sh, -r[:, 1] / proj.coslat], axis=1)
                    ok = np.ones(len(uv), bool)
                    for z in self.zs:
                        q = proj.px(uv, z)
                        ok &= (q[:, 0] > m) & (q[:, 0] < self.W - m) & (q[:, 1] > m) & (q[:, 1] < self.H - m)
                    key = name.casefold()
                    if ok.sum() >= 4 and ok.sum() > len(stretches.get(key, ("", ()))[1]):
                        stretches[key] = (name, uv[ok])
        out = []
        col, halo, g = mix(pal.rgb("water"), pal.rgb("text"), 0.55), pal.rgb("land"), self.px(6)
        for name, pts in sorted(stretches.values(), key=lambda e: -len(e[1])):  # longest rivers claim space first
            spr = halo_text(name, max(8, self.px(20)), "Medium", col, halo, 0.04, max(2, self.px(3)))
            n = len(pts)
            for j in dict.fromkeys((n // 2, n // 3, 2 * n // 3, n // 4, 3 * n // 4)):
                if self._place(pts[j:j + 1], spr, [(-spr.w / 2, g)]) is not None:
                    out.append((name, pts[j:j + 1], spr, (-spr.w / 2, g)))
                    break
        return out

    # ------------------------------------------------------------ frame

    def render(self, i: int) -> np.ndarray:
        t = i / self.fps
        z = self.zoom_at(t)
        frame = self.base.view(z, ez.ease_in_out_cubic(ez.progress(t, 0.3, 1.4)))
        frame = art_hooks.moving_base(self, i, frame)  # run 3 art layer: checked clip of the painted base
        cv2.multiply(frame, self.vig, dst=frame, scale=1 / 255.0)
        self.glow.clear()
        self._draw_routes(frame, t, z)
        self._draw_arrows(frame, t, z)
        self.glow.composite(frame, sigma=2.2, gain=self.glow_gain)
        self._draw_labels(frame, t, z)
        self._draw_markers(frame, t, z)
        self._draw_route_heads(frame, t, z)
        self._draw_title(frame, t)
        self._grain.apply(frame, i)
        if self._note is not None:
            blit(frame, self._note, *self.note_xy, 0.9)
        return frame

    def _route_px(self, t: float, z: float) -> list[tuple[int, np.ndarray, float]]:
        out = []
        for k, (r, uv) in enumerate(self.routes):
            p = ez.ease_in_out_cubic(ez.progress(t, r.reveal_start, r.reveal_end))
            if p > 0:
                out.append((k, self.proj.px(path_prefix(uv, p), z), p))
        return out

    def _draw_routes(self, frame: np.ndarray, t: float, z: float) -> None:
        col = self.pal.rgb("route")
        casing = mix(self.pal.rgb("land"), (255, 255, 255), 0.55) if not self.pal.dark else (8, 10, 14)
        th = max(2, self.px(5))
        for k, pts, p in self._route_px(t, z):
            r = self.routes[k][0]
            polyline(self.glow.buf, pts / self.glow.f, col, max(1, th // 2 + 1))
            self.glow.dirty = True
            pieces = dashes(pts, self.px(20), self.px(12)) if r.dashed else [pts]
            for piece in pieces:
                polyline(frame, piece, casing, th + max(2, self.px(4)))
            for piece in pieces:
                polyline(frame, piece, col, th)

    def _draw_route_heads(self, frame: np.ndarray, t: float, z: float) -> None:
        col = self.pal.rgb("route")
        for k, pts, p in self._route_px(t, z):
            r = self.routes[k][0]
            spr = self.route_pills[k]
            if spr is not None and p > 0.5:
                mid = self.proj.px(path_prefix(self.routes[k][1], 0.5), z)[-1]
                o = ez.ease_out_cubic(ez.progress(p, 0.5, 0.8))
                dx, dy = self.route_offs[k]
                blit(frame, spr, mid[0] + dx, mid[1] + dy, o)
            fade_out = 1 - ez.progress(t, r.reveal_end, r.reveal_end + 0.6)
            if fade_out <= 0:
                continue
            x, y = pts[-1]
            rad = self.px(10) * fade_out
            disc(frame, x, y, rad * 2.0, mix(col, (255, 255, 255), 0.5) if self.pal.dark else col)
            disc(frame, x, y, rad * 1.4, col)
            disc(frame, x, y, rad * 0.7, (255, 255, 255))

    def _draw_arrows(self, frame: np.ndarray, t: float, z: float) -> None:
        col = self.pal.rgb("accent2") if self.pal.dark else self.pal.rgb("accent")
        th = max(2, self.px(6))
        for k, (a, uv) in enumerate(self.arrows):
            p = ez.ease_in_out_cubic(ez.progress(t, a.appear_at, a.appear_at + 0.9))
            if p <= 0:
                continue
            full = self.proj.px(uv, z)
            pts = self.proj.px(path_prefix(uv, p), z)
            polyline(self.glow.buf, pts / self.glow.f, (0, 0, 0), max(1, th // 2 + 1))
            self.glow.dirty = True
            polyline(frame, pts, col, th)
            head = ez.ease_out_back(ez.progress(t, a.appear_at + 0.8, a.appear_at + 1.1), 2.0)
            if head > 0.01:
                fill_poly(frame, [arrow_head(full, self.px(30) * head)], col)
            spr = self.arrow_pills[k]
            if spr is not None:
                o = ez.ease_out_cubic(ez.progress(t, a.appear_at, a.appear_at + 0.5))
                x, y = full[0]
                dx, dy = self.arrow_offs[k]
                blit(frame, spr, x + dx, y + dy, o)

    def _draw_markers(self, frame: np.ndarray, t: float, z: float) -> None:
        pal = self.pal
        acc = pal.rgb("accent")
        ring = (255, 255, 255) if pal.dark else pal.rgb("panel")
        for k, (m, uv) in enumerate(self.markers):
            pop = ez.ease_out_back(ez.progress(t, m.appear_at, m.appear_at + 0.45), 2.2)
            if pop <= 0.001:
                continue
            x, y = self.proj.px(uv, z)[0]
            pulse = ez.progress(t, m.appear_at + 0.1, m.appear_at + 1.4)
            if 0 < pulse < 1:
                disc(frame, x, y, self.px(14 + 34 * ez.ease_out_cubic(pulse)), mix(acc, ring, pulse),
                     thickness=max(1, int(self.px(3) * (1 - pulse)) + 1))
            disc(frame, x, y + self.px(2), self.px(14) * pop, mix(pal.rgb("text"), (0, 0, 0), 0.4))
            disc(frame, x, y, self.px(13) * pop, ring)
            disc(frame, x, y, self.px(8) * pop, acc)
            spr = self.marker_pills[k]
            if spr is not None:
                o = ez.ease_out_cubic(ez.progress(t, m.appear_at + 0.15, m.appear_at + 0.6))
                dx, dy = self.offsets[k]
                slide = self.px(14) * (1 - o) * (-1 if dx < 0 else 1)
                blit(frame, spr, x + dx + slide, y + dy, o)

    def _draw_labels(self, frame: np.ndarray, t: float, z: float) -> None:
        for _, uv, spr, (dx, dy) in self.rivers:
            x, y = self.proj.px(uv, z)[0]
            blit(frame, spr, x + dx, y + dy)
        for i, (uv, spr, (dx, dy)) in enumerate(self.labels):
            o = ez.ease_out_cubic(ez.progress(t, 0.5 + 0.15 * i, 1.2 + 0.15 * i))
            x, y = self.proj.px(uv, z)[0]
            blit(frame, spr, x + dx, y + dy, o * 0.95)

    def _draw_title(self, frame: np.ndarray, t: float) -> None:
        o = ez.ease_out_cubic(ez.progress(t, 0.2, 0.9))
        if self.locator is not None:
            blit(frame, self.locator, *self.locator_xy, o)
        if self.title is not None:
            x, y = self.title_xy
            blit(frame, self.title, x - self.px(30) * (1 - o), y, o)


def _box(xy: tuple[float, float], spr: Sprite) -> Box:
    return xy[0], xy[1], xy[0] + spr.w, xy[1] + spr.h


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
