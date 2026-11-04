"""Bar and line chart compositions.

Bar charts grow with easing and a stagger while value labels count up with
units and thousands separators; vertical bars for a few short labels,
horizontal bars otherwise. Line charts draw left to right with a moving dot
and value tag, gridlines with nice ticks and a soft area fill. The
highlight_index category is drawn in the accent colour.
"""
from __future__ import annotations

import math

import numpy as np

from . import easing as ez
from .palettes import RGB, mix
from .textdraw import (
    CompBase,
    GlowLayer,
    Sprite,
    blit,
    disc,
    fit_text,
    font,
    mask_poly,
    on_colour,
    pill,
    polyline,
    rect,
    text_block,
    text_sprite,
    translucent,
)

TNUM = ["tnum"]


def unit_affixes(units: str) -> tuple[str, str]:
    """(prefix, suffix) for a unit string: '$' prefix for money, '%' suffix, short units as suffix."""
    u = (units or "").strip()
    low = u.lower()
    if low in {"$", "usd", "dollars", "us dollars", "us$"}:
        return "$", ""
    if low in {"%", "percent", "pct"}:
        return "", "%"
    return ("", " " + u) if 0 < len(u) <= 8 else ("", "")


def format_value(v: float, units: str = "", decimals: int = 0) -> str:
    """Display string with thousands separators, compact M/B above a million, and unit affixes."""
    pre, suf = unit_affixes(units)
    av = abs(v)
    if av >= 1e9:
        s = f"{v / 1e9:.1f}".rstrip("0").rstrip(".") + "B"
    elif av >= 1e6:
        s = f"{v / 1e6:.1f}".rstrip("0").rstrip(".") + "M"
    else:
        s = f"{v:,.{decimals}f}"
    sign = "-" if s.startswith("-") else ""
    return sign + pre + s.lstrip("-") + suf


def decimals_for(values: list[float]) -> int:
    """0 for integer data, else 1 (2 when values are small)."""
    if all(float(v).is_integer() for v in values):
        return 0
    return 2 if max(abs(v) for v in values) < 10 else 1


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """Round tick values covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks


class ChartBase(CompBase):
    """Title, unit note, plot rectangle and series colours shared by bar and line charts."""

    ticks_need_note = False  # line charts drop word units from tick labels and show them as a note

    def __init__(self, spec, payload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        p = payload
        self.series = [s for s in p.series if s.values] or []
        n = len(p.labels)
        for s in self.series:
            s.values = (list(s.values) + [0.0] * n)[:n] if n else list(s.values)
        self.labels = list(p.labels) or [str(i + 1) for i in range(len(self.series[0].values) if self.series else 0)]
        self.text_col = (255, 255, 255) if self.transparent else pal.rgb("text")
        self.muted = (215, 215, 215) if self.transparent else pal.rgb("muted")
        self.shadow = 0.6 if self.transparent else 0.0
        self.title = None
        if p.title:
            f, ln = fit_text(p.title, "Bold", self.px(54), self.W - 2 * self.margin, 1, self.px(30))
            self.title = text_sprite(ln[0], f, self.text_col, shadow=self.shadow)
        pre, suf = unit_affixes(p.units)
        self.unit_note = None
        if p.units and not pre and (not suf or self.ticks_need_note):
            f = font("Medium", self.px(24))
            self.unit_note = text_sprite(p.units.upper(), f, self.muted, 0.14 * f.size, self.shadow)
        top = self.safe_top + self.px(36) + (self.title.h + self.px(64) if self.title else 0)
        self.plot = (self.margin, top, self.W - self.margin, self.safe_bottom - self.px(90))
        self.vals = [v for s in self.series for v in s.values] or [0.0]
        self.dec = decimals_for(self.vals)

    def series_colour(self, k: int) -> RGB:
        pal = self.pal
        return [pal.rgb("accent"), pal.rgb("accent2"), mix(pal.rgb("muted"), pal.rgb("text"), 0.3),
                pal.rgb("route")][k % 4]

    def draw_header(self, frame: np.ndarray, t: float) -> None:
        if self.title is None:
            return
        o = ez.ease_out_cubic(ez.progress(t, 0.0, 0.6))
        x, y = self.margin, self.safe_top + self.px(36)
        blit(frame, self.title, x, y + self.px(12) * (1 - o), o)
        rw = self.px(110) * ez.ease_in_out_cubic(ez.progress(t, 0.2, 0.8))
        if rw > 1:
            blit(frame, rect(rw, self.px(5), self.pal.rgb("accent")), x, y + self.title.h + self.px(12))
        if self.unit_note is not None:
            blit(frame, self.unit_note, self.W - self.margin - self.unit_note.w, y + self.title.h - self.unit_note.h, o)

    def legend(self, frame: np.ndarray, t: float) -> None:
        """Swatches for multi-series charts (top right of the plot)."""
        named = [(k, s.name) for k, s in enumerate(self.series) if s.name]
        if len(self.series) < 2 or not named:
            return
        o = ez.ease_out_cubic(ez.progress(t, 0.4, 1.0))
        f = font("Medium", self.px(24))
        x = self.plot[2]
        y = self.plot[1] - self.px(44)
        for k, name in reversed(named):
            spr = text_sprite(name, f, self.muted, shadow=self.shadow)
            x -= spr.w
            blit(frame, spr, x, y, o)
            x -= self.px(26)
            blit(frame, rect(self.px(16), self.px(16), self.series_colour(k)), x, y + spr.h / 2 - self.px(8), o)
            x -= self.px(30)


class BarChart(ChartBase):
    """Bars grow with an eased stagger; value labels count up; highlight bar in accent."""

    def __init__(self, spec, payload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        n = len(self.labels)
        self.vertical = n <= 8 and max((len(lb) for lb in self.labels), default=0) <= 14
        self.hl = payload.highlight_index if payload.highlight_index is not None and 0 <= payload.highlight_index < n \
            else None
        self.lo = min(0.0, min(self.vals))
        self.hi = max(max(self.vals), self.lo + 1e-9)
        self.vf = font("SemiBold", self.px(34 if n <= 6 else 28))
        self._layout_labels()

    def _layout_labels(self) -> None:
        x0, y0, x1, y1 = self.plot
        n = max(1, len(self.labels))
        self.label_sprites: list[Sprite] = []
        if self.vertical:
            slot = (x1 - x0) / n
            for lb in self.labels:
                f, ln = fit_text(lb, "Medium", self.px(30), slot * 0.92, 2, self.px(16))
                self.label_sprites.append(text_block(ln, f, self.text_col, "center", 1.1, shadow=self.shadow))
            self.base_y = y1 - max(s.h for s in self.label_sprites) + self.px(30)
            self.bar_top = y0 + self.px(56)
        else:
            self.col_w = (x1 - x0) * 0.28
            row = (y1 - y0 + self.px(60)) / n
            for lb in self.labels:
                f, ln = fit_text(lb, "Medium", self.px(30), self.col_w - self.px(24), 2 if row > self.px(70) else 1,
                                 self.px(14))
                self.label_sprites.append(text_block(ln, f, self.text_col, "right", 1.05, shadow=self.shadow))
            self.row = row

    def _bar_colour(self, i: int, k: int) -> RGB:
        if len(self.series) > 1:
            return self.series_colour(k)
        if self.hl is None:
            return self.pal.rgb("accent")
        if i == self.hl:
            return self.pal.rgb("accent")
        return mix(self.pal.rgb("muted"), self.pal.rgb("bg"), 0.25) if not self.transparent else (200, 200, 200)

    def _grow(self, i: int, t: float) -> float:
        n = len(self.labels)
        a, b = ez.stagger(i, n, 0.35, min(1.4, self.dur * 0.25), min(1.1, self.dur * 0.35))
        return ez.ease_out_cubic(ez.progress(t, a, b))

    def draw(self, frame: np.ndarray, t: float) -> None:
        self.draw_header(frame, t)
        self.legend(frame, t)
        if self.vertical:
            self._draw_vertical(frame, t)
        else:
            self._draw_horizontal(frame, t)

    def _value_label(self, v: float, p: float, i: int) -> Sprite:
        bold = i == self.hl
        col = self.pal.rgb("accent") if bold else self.text_col
        f = font("Bold", self.vf.size) if bold else self.vf
        return text_sprite(format_value(v * p, self.spec.payload.get("units", ""), self.dec), f, col,
                           shadow=self.shadow, features=TNUM)

    def _draw_vertical(self, frame: np.ndarray, t: float) -> None:
        x0, _, x1, _ = self.plot
        n, m = len(self.labels), max(1, len(self.series))
        slot = (x1 - x0) / max(1, n)
        span = self.hi - self.lo
        h_full = self.base_y - self.bar_top
        zero_y = self.base_y - h_full * (0 - self.lo) / span
        axis_o = ez.ease_out_cubic(ez.progress(t, 0.1, 0.6))
        bw = slot * min(0.62, 0.8 / m) / (1 if m == 1 else 0.9)
        for i in range(n):
            p = self._grow(i, t)
            cx = x0 + slot * (i + 0.5)
            for k, s in enumerate(self.series):
                v = s.values[i]
                bx = cx - bw * m / 2 + k * bw + (bw * 0.06 if m > 1 else 0)
                w = bw * (0.88 if m > 1 else 1.0)
                top = zero_y - (v * p) / span * h_full
                y_a, y_b = sorted((top, zero_y))
                col = self._bar_colour(i, k)
                if y_b - y_a >= 1:
                    blit(frame, rect(w, y_b - y_a, col), bx, y_a)
                    blit(frame, rect(w, max(1, self.px(3)), mix(col, (255, 255, 255), 0.35)), bx, y_a, 0.6)
                if p > 0.02:
                    lab = self._value_label(v, p, i)
                    blit(frame, lab, bx + w / 2 - lab.w / 2, (y_a - lab.h - self.px(8)) if v >= 0 else y_b + self.px(6),
                         min(1.0, p * 2))
            ls = self.label_sprites[i]
            blit(frame, ls, cx - ls.w / 2, self.base_y + self.px(18), axis_o)
        lw = (x1 - x0) * axis_o
        blit(frame, rect(lw, max(1, self.px(2)), self.muted), x0, zero_y, 0.9)

    def _draw_horizontal(self, frame: np.ndarray, t: float) -> None:
        x0, y0, x1, _ = self.plot
        n, m = len(self.labels), max(1, len(self.series))
        bx0 = x0 + self.col_w
        if getattr(self, "_label_room", None) is None:
            self._label_room = max((self._value_label(v, 1.0, i).w for s in self.series
                                    for i, v in enumerate(s.values) if v >= 0), default=0) + self.px(16)
        room = self._label_room
        w_full = min((x1 - bx0) * 0.84, (x1 - bx0) - room)  # the widest value label always fits
        span = self.hi - self.lo
        zero_x = bx0 + w_full * (0 - self.lo) / span
        bh = self.row * 0.62 / m
        axis_o = ez.ease_out_cubic(ez.progress(t, 0.1, 0.6))
        for i in range(n):
            p = self._grow(i, t)
            cy = y0 + self.row * (i + 0.5)
            ls = self.label_sprites[i]
            blit(frame, ls, bx0 - self.px(24) - ls.w, cy - ls.h / 2, axis_o)
            for k, s in enumerate(self.series):
                v = s.values[i]
                by = cy - bh * m / 2 + k * bh
                end = zero_x + (v * p) / span * w_full
                xa, xb = sorted((zero_x, end))
                col = self._bar_colour(i, k)
                if xb - xa >= 1:
                    blit(frame, rect(xb - xa, bh * 0.9, col), xa, by)
                if p > 0.02:
                    lab = self._value_label(v, p, i)
                    lx = xb + self.px(12) if v >= 0 else xa - lab.w - self.px(12)
                    blit(frame, lab, lx, by + bh * 0.45 - lab.h / 2, min(1.0, p * 2))
        blit(frame, rect(max(1, self.px(2)), (self.row * n) * axis_o, self.muted), zero_x, y0, 0.9)


class LineChart(ChartBase):
    """Gridlines with nice ticks; each series draws left to right with a glowing dot and value tag."""

    ticks_need_note = True

    def __init__(self, spec, payload, pal, alpha: bool) -> None:  # noqa: ANN001
        super().__init__(spec, payload, pal, alpha)
        lo, hi = min(self.vals), max(self.vals)
        pad = (hi - lo) * 0.12 or abs(hi) * 0.1 or 1.0
        self.ticks = nice_ticks(lo - pad if lo < 0 or lo - pad > 0 else 0.0, hi + pad, 4)
        self.lo, self.hi = self.ticks[0], self.ticks[-1]
        pre, suf = unit_affixes(payload.units)
        units = payload.units if pre or suf.strip() == "%" else ""
        f = font("Medium", self.px(24))
        self.tick_sprites = [text_sprite(format_value(v, units, decimals_for(self.ticks)), f, self.muted,
                                         shadow=self.shadow, features=TNUM) for v in self.ticks]
        gutter = max(s.w for s in self.tick_sprites) + self.px(24)
        x0, y0, x1, y1 = self.plot
        self.area = (x0 + gutter, y0 + self.px(10), x1 - self.px(40), y1)
        self.x_labels = self._x_labels()
        self.paths = [self._points(s.values) for s in self.series]
        self.hl = payload.highlight_index
        self.glow = GlowLayer(self.W, self.H)
        ay0, ay1 = int(self.area[1]), int(self.area[3])
        w = np.zeros((self.H, 1), np.float32)
        w[ay0:ay1 + 1, 0] = np.linspace(1.0, 0.0, ay1 - ay0 + 1, dtype=np.float32) ** 1.4
        self.fill_weight = np.repeat(w, self.W, axis=1)

    def _points(self, values: list[float]) -> np.ndarray:
        x0, y0, x1, y1 = self.area
        n = len(values)
        xs = np.linspace(x0, x1, n) if n > 1 else np.array([(x0 + x1) / 2])
        ys = y1 - (np.array(values, np.float64) - self.lo) / (self.hi - self.lo) * (y1 - y0)
        return np.stack([xs, ys], axis=1)

    def _x_labels(self) -> list[tuple[Sprite, float]]:
        x0, _, x1, _ = self.area
        n = len(self.labels)
        every = max(1, math.ceil(n / 8))
        f = font("Medium", self.px(26 if n <= 8 else 22))
        out = []
        for i, lb in enumerate(self.labels):
            if i % every and i != n - 1:
                continue
            x = x0 + (x1 - x0) * (i / (n - 1) if n > 1 else 0.5)
            out.append((text_sprite(lb[:14], f, self.text_col, shadow=self.shadow), x))
        return out

    def _draw_grid(self, frame: np.ndarray, t: float) -> None:
        x0, y0, x1, y1 = self.area
        o = ez.ease_out_cubic(ez.progress(t, 0.0, 0.7))
        grid = self.pal.rgb("grid") if not self.transparent else (200, 200, 200)
        for v, spr in zip(self.ticks, self.tick_sprites):
            y = y1 - (v - self.lo) / (self.hi - self.lo) * (y1 - y0)
            w = (x1 - x0) * ez.ease_in_out_cubic(ez.progress(t, 0.0, 0.9))
            blit(frame, rect(w, max(1, self.px(1.5)), grid if v != self.lo else self.muted), x0, y, 0.7)
            blit(frame, spr, x0 - self.px(18) - spr.w, y - spr.h / 2, o)
        for spr, x in self.x_labels:
            blit(frame, spr, x - spr.w / 2, y1 + self.px(18), o)

    def draw(self, frame: np.ndarray, t: float) -> None:
        self.draw_header(frame, t)
        self.legend(frame, t)
        self._draw_grid(frame, t)
        x0, _, x1, y1 = self.area
        p = ez.ease_in_out_cubic(ez.progress(t, 0.5, 0.5 + max(0.8, min(self.dur * 0.55, 4.5))))
        cut = x0 + (x1 - x0) * p
        self.glow.clear()
        heads = []
        for k, pts in enumerate(self.paths):
            part = _clip_x(pts, cut)
            if len(part) < 1 or p <= 0:
                continue
            col = self.series_colour(k)
            if k == 0 and len(part) > 1:
                poly = np.vstack([part, [[part[-1][0], y1], [part[0][0], y1]]])
                translucent(frame, col, 0.28, lambda m, loc: mask_poly(m, [loc(poly)]),
                            (int(x0), int(self.area[1]), int(x1) + 2, int(y1) + 2), self.fill_weight)
            polyline(self.glow.buf, part / self.glow.f, col, 2)
            self.glow.dirty = True
            polyline(frame, part, col, max(2, self.px(5)))
            heads.append((part[-1], col, k))
        self.glow.composite(frame, 2.5, self.glow_gain)
        for (x, y), col, k in heads:
            disc(frame, x, y, self.px(13), col)
            disc(frame, x, y, self.px(6), (255, 255, 255))
            if k == 0 and p < 0.999:
                self._tag(frame, x, y, k)
        self._highlight(frame, p)

    def _value_at(self, k: int, x: float) -> float:
        pts, vals = self.paths[k], self.series[k].values
        return float(np.interp(x, pts[:, 0], vals)) if len(vals) > 1 else vals[0]

    def _tag(self, frame: np.ndarray, x: float, y: float, k: int) -> None:
        v = self._value_at(k, x)
        f = font("SemiBold", self.px(26))
        tag = pill(format_value(v, self.spec.payload.get("units", ""), self.dec), f,
                   self.pal.rgb("panel_text"), self.pal.rgb("panel"), self.px(14), self.px(8))
        blit(frame, tag, min(x + self.px(22), self.W - tag.w - self.margin * 0.3), y - tag.h - self.px(16))

    def _highlight(self, frame: np.ndarray, p: float) -> None:
        n = len(self.labels)
        if self.hl is None or not (0 <= self.hl < n) or not self.paths:
            return
        x, y = self.paths[0][self.hl]
        reach = (self.hl / (n - 1)) if n > 1 else 0.0
        if p + 1e-6 < reach:
            return
        o = ez.ease_out_back(ez.progress(p, reach - 0.001, min(1.0, reach + 0.12)), 1.6)
        acc = self.pal.rgb("accent")
        disc(frame, x, y, self.px(20) * o, acc, thickness=max(2, self.px(4)))
        f = font("Bold", self.px(28))
        tag = pill(format_value(self.series[0].values[self.hl], self.spec.payload.get("units", ""), self.dec), f,
                   on_colour(acc), acc, self.px(16), self.px(9))
        blit(frame, tag, x - tag.w / 2, y - tag.h - self.px(36), ez.clamp01(o))


def _clip_x(pts: np.ndarray, cut: float) -> np.ndarray:
    """Points with x <= cut plus the interpolated point at x == cut."""
    if len(pts) == 1:
        return pts if pts[0][0] <= cut + 1e-6 else pts[:0]
    keep = pts[pts[:, 0] <= cut]
    if len(keep) == len(pts) or len(keep) == 0:
        return keep
    a, b = pts[len(keep) - 1], pts[len(keep)]
    u = (cut - a[0]) / max(1e-9, b[0] - a[0])
    return np.vstack([keep, a + (b - a) * u])
