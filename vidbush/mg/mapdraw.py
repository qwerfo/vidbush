"""Map composition over bundled Natural Earth data (never a tile server).

Projection: local equirectangular with a cos(center latitude) correction,
matching vidbush/mg/builders.map_view (zoom 1 = 360 degrees of longitude
across the frame width, each zoom step doubles the scale).

Performance: the static map (water, graticule, coast halo, land with texture,
borders, lakes, rivers, shaded regions) is drawn once into a base image at
the resolution needed for the deepest zoom of the push-in. Each frame is one
affine warp of that base plus the animated parts (routes, arrows, markers,
labels, title), vignette and grain.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import REPO_ROOT
from . import easing as ez
from .palettes import RGB, mix
from .textdraw import (
    CompBase,
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
Ring = np.ndarray  # (n, 2) lon, lat


# ---------------------------------------------------------------- data

@lru_cache(maxsize=4)
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


def _names(props: dict) -> set[str]:
    from .geo import norm
    keys = ("NAME", "NAME_LONG", "ADMIN", "SOVEREIGNT", "ISO_A3", "ADM0_A3", "BRK_NAME", "FORMAL_EN")
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
        if z >= 3.5:
            rank = 6 if z < 5 else 99
            rivers = self.rings(load_layer("ne_50m_rivers_lake_centerlines.geojson"),
                                lambda p: (p.get("scalerank") or 9) <= rank)
            for r in rivers:
                polyline(img, r, mix(water, pal.rgb("land"), 0.15), max(1.0, 1.6 * self.k))

    def _shade(self, plain: np.ndarray) -> np.ndarray:
        """Shaded regions: accent tinted fill, accent outline and a soft outer glow."""
        from .geo import norm
        pal = self.c.pal
        wanted = {norm(n) for n in self.c.p.shaded_regions if n}
        rings = self.rings(self.countries, lambda p: bool(_names(p) & wanted))
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
        self.routes = [(r, smooth_path(self.proj.uv(np.asarray(r.points, np.float64)))) for r in p.routes
                       if len(r.points) >= 2]
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
        self.offsets = self._place_labels()

    def _title_sprite(self) -> Optional[Sprite]:
        if not self.p.title:
            return None
        f, ln = fit_text(self.p.title.upper(), "Bold", self.px(40), self.W * 0.55, 1, self.px(22), 0.08)
        return pill(ln[0], f, self.pal.rgb("panel_text"), self.pal.rgb("panel"), self.px(24), self.px(14),
                    0.08 * f.size, accent=self.pal.rgb("accent"))

    def _place_labels(self) -> list[tuple[float, float]]:
        """Pick a side per marker label (right, left, above, below) avoiding overlaps at the final zoom."""
        placed: list[tuple[float, float, float, float]] = []
        r = self.px(22)
        offsets = []
        for (m, uv), spr in zip(self.markers, self.marker_pills):
            if spr is None:
                offsets.append((0.0, 0.0))
                continue
            x, y = self.proj.px(uv, self.z1)[0]
            options = [(r, -spr.h / 2), (-r - spr.w, -spr.h / 2), (-spr.w / 2, -r - spr.h), (-spr.w / 2, r)]
            best = options[0]
            for dx, dy in options:
                box = (x + dx, y + dy, x + dx + spr.w, y + dy + spr.h)
                inside = box[0] > self.px(20) and box[2] < self.W - self.px(20) and box[1] > self.safe_top \
                    and box[3] < self.safe_bottom
                if inside and not any(_overlap(box, b) for b in placed):
                    best = (dx, dy)
                    break
            placed.append((x + best[0], y + best[1], x + best[0] + spr.w, y + best[1] + spr.h))
            offsets.append(best)
        return offsets

    # ------------------------------------------------------------ frame

    def render(self, i: int) -> np.ndarray:
        t = i / self.fps
        z = self.zoom_at(t)
        frame = self.base.view(z, ez.ease_in_out_cubic(ez.progress(t, 0.3, 1.4)))
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
            blit(frame, self._note, self.W - self.margin * 0.55 - self._note.w,
                 self.safe_bottom - self._note.h - self.px(14), 0.9)
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
                blit(frame, spr, mid[0] - spr.w / 2, mid[1] - spr.h - self.px(18), o)
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
                blit(frame, spr, x - spr.w / 2, y + self.px(18), o)

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
        for i, (_, uv, spr) in enumerate(self.label_sprites):
            o = ez.ease_out_cubic(ez.progress(t, 0.5 + 0.15 * i, 1.2 + 0.15 * i))
            x, y = self.proj.px(uv, z)[0]
            blit(frame, spr, x - spr.w / 2, y - spr.h / 2, o * 0.95)

    def _draw_title(self, frame: np.ndarray, t: float) -> None:
        if self.title is None:
            return
        o = ez.ease_out_cubic(ez.progress(t, 0.2, 0.9))
        blit(frame, self.title, self.margin - self.px(30) * (1 - o), self.safe_top + self.px(30), o)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
