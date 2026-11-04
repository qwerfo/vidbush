"""Text, sprites and raster helpers shared by every motion graphics composition.

Frames are numpy uint8 arrays of shape (H, W, C): C = 3 for opaque output and
C = 4 (premultiplied RGBA) for alpha output. Sprites are premultiplied RGBA so
one "over" formula serves both. Text is rendered with Pillow (anti aliased,
kerned), vector shapes with OpenCV LINE_AA. `CompBase` holds what every
composition shares: background, grain, safe area, source note.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .palettes import RGB, Palette, font_path

Align = Literal["left", "center", "right"]
ELLIPSIS = "…"


# ---------------------------------------------------------------- fonts and measuring

@lru_cache(maxsize=256)
def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    """Cached Inter font at a pixel size."""
    return ImageFont.truetype(str(font_path(weight)), max(6, int(size)))


def text_width(text: str, f: ImageFont.FreeTypeFont, tracking: float = 0.0,
               features: Optional[list[str]] = None) -> float:
    """Advance width of a single line including letter spacing (px)."""
    if not text:
        return 0.0
    return f.getlength(text, features=features) + tracking * (len(text) - 1)


def wrap(text: str, f: ImageFont.FreeTypeFont, max_w: float, tracking: float = 0.0) -> list[str]:
    """Greedy word wrap. A single word wider than max_w stays on its own line."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and text_width(trial, f, tracking) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def _truncate(line: str, f: ImageFont.FreeTypeFont, max_w: float, tracking: float) -> str:
    if text_width(line, f, tracking) <= max_w:
        return line
    while line and text_width(line + ELLIPSIS, f, tracking) > max_w:
        line = line[:-1].rstrip()
    return line + ELLIPSIS


def fit_text(text: str, weight: str, size: float, max_w: float, max_lines: int = 1,
             min_size: Optional[float] = None, tracking_em: float = 0.0) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Largest font (<= size) whose wrapped text fits max_w in at most max_lines lines.

    Below min_size the text is truncated with an ellipsis, so it never overflows."""
    min_size = min_size or size * 0.4
    s = float(size)
    while True:
        f = font(weight, int(s))
        tr = tracking_em * s
        lines = wrap(text, f, max_w, tr)
        if len(lines) <= max_lines and all(text_width(ln, f, tr) <= max_w for ln in lines):
            return f, lines
        if s * 0.94 < min_size:
            lines = lines[:max_lines]
            if len(wrap(text, f, max_w, tr)) > max_lines:
                lines[-1] = lines[-1] + " " + ELLIPSIS
            return f, [_truncate(ln, f, max_w, tr) for ln in lines]
        s *= 0.94


# ---------------------------------------------------------------- sprites

@dataclass
class Sprite:
    """Premultiplied RGBA image. (pad, pad) is the content origin inside arr."""

    arr: np.ndarray
    pad: int = 0

    @property
    def w(self) -> int:
        return self.arr.shape[1] - 2 * self.pad

    @property
    def h(self) -> int:
        return self.arr.shape[0] - 2 * self.pad


def coloured(mask: np.ndarray, color: RGB, opacity: float = 1.0) -> np.ndarray:
    """Premultiplied RGBA from an alpha mask (uint8) and a colour."""
    a = mask.astype(np.float32) * opacity
    out = np.empty(mask.shape + (4,), np.uint8)
    for c in range(3):
        out[..., c] = np.rint(a * (color[c] / 255.0)).astype(np.uint8)
    out[..., 3] = np.rint(a).astype(np.uint8)
    return out


def text_mask(text: str, f: ImageFont.FreeTypeFont, tracking: float = 0.0, pad: int = 0,
              features: Optional[list[str]] = None) -> np.ndarray:
    """Alpha mask of one line; height is the font line box (ascent + descent) plus pad."""
    asc, desc = f.getmetrics()
    w = int(math.ceil(text_width(text, f, tracking, features))) + 2 * pad + 4
    img = Image.new("L", (max(1, w), asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(img)
    if tracking == 0.0:
        d.text((pad, pad), text, font=f, fill=255, features=features)
    else:
        for i, ch in enumerate(text):
            x = pad + f.getlength(text[:i], features=features) + tracking * i
            d.text((x, pad), ch, font=f, fill=255, features=features)
    return np.asarray(img)


def text_sprite(text: str, f: ImageFont.FreeTypeFont, color: RGB, tracking: float = 0.0,
                shadow: float = 0.0, pad: int = 0, features: Optional[list[str]] = None) -> Sprite:
    """One line of text; shadow > 0 adds a soft dark drop shadow of that opacity."""
    pad = max(pad, int(f.size * 0.25) if shadow > 0 else pad)
    mask = text_mask(text, f, tracking, pad, features)
    arr = coloured(mask, color)
    if shadow > 0:
        arr = with_shadow(arr, radius=max(2.0, f.size * 0.08), offset=max(1, int(f.size * 0.04)), opacity=shadow)
    return Sprite(arr, pad)


def text_block(lines: Sequence[str], f: ImageFont.FreeTypeFont, color: RGB, align: Align = "left",
               line_height: float = 1.12, tracking: float = 0.0, shadow: float = 0.0) -> Sprite:
    """Several lines stacked with the given alignment."""
    pad = int(f.size * 0.25) if shadow > 0 else 0
    masks = [text_mask(ln, f, tracking) for ln in lines]
    widths = [text_width(ln, f, tracking) for ln in lines]
    step = int(f.size * line_height)
    w = int(math.ceil(max(widths))) + 4
    asc, desc = f.getmetrics()
    h = step * (len(lines) - 1) + asc + desc
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    for i, (m, lw) in enumerate(zip(masks, widths)):
        x = pad + (0 if align == "left" else int((w - lw) / 2) if align == "center" else int(w - lw))
        y = pad + i * step
        mh, mw = m.shape
        mw = min(mw, canvas.shape[1] - x)
        canvas[y:y + mh, x:x + mw] = np.maximum(canvas[y:y + mh, x:x + mw], m[:, :mw])
    arr = coloured(canvas, color)
    if shadow > 0:
        arr = with_shadow(arr, radius=max(2.0, f.size * 0.08), offset=max(1, int(f.size * 0.04)), opacity=shadow)
    return Sprite(arr, pad)


def with_shadow(arr: np.ndarray, radius: float, offset: int, opacity: float, color: RGB = (0, 0, 0)) -> np.ndarray:
    """Composite arr over a blurred, offset copy of its alpha (soft drop shadow)."""
    a = arr[..., 3]
    sh = np.zeros_like(a)
    sh[offset:, offset:] = a[:-offset or None, :-offset or None] if offset else a
    sh = cv2.GaussianBlur(sh, (0, 0), radius)
    base = coloured(sh, color, opacity)
    return over(base, arr)


def over(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Premultiplied src over premultiplied dst, same shape (returns a new array)."""
    sa = src[..., 3:4].astype(np.float32) / 255.0
    return (src.astype(np.float32) + dst.astype(np.float32) * (1.0 - sa)).clip(0, 255).astype(np.uint8)


def blit(dst: np.ndarray, spr: Sprite | np.ndarray, x: float, y: float, opacity: float = 1.0) -> None:
    """Composite a premultiplied sprite onto frame dst in place; (x, y) is the content origin."""
    if opacity <= 0.003:
        return
    arr, pad = (spr.arr, spr.pad) if isinstance(spr, Sprite) else (spr, 0)
    x0, y0 = int(round(x)) - pad, int(round(y)) - pad
    h, w = arr.shape[:2]
    H, W = dst.shape[:2]
    dx0, dy0, dx1, dy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    s = arr[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0].astype(np.float32)
    if opacity < 1.0:
        s *= opacity
    inv = 1.0 - s[..., 3:4] / 255.0
    roi = dst[dy0:dy1, dx0:dx1]
    roi[...] = (roi.astype(np.float32) * inv + s[..., :roi.shape[2]]).clip(0, 255).astype(np.uint8)


def scaled(spr: Sprite, s: float) -> Sprite:
    """Sprite resized by factor s about its content origin."""
    if abs(s - 1.0) < 0.005:
        return spr
    h, w = spr.arr.shape[:2]
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return Sprite(cv2.resize(spr.arr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR),
                  int(round(spr.pad * s)))


@lru_cache(maxsize=128)
def _round_rect_mask(w: int, h: int, r: int) -> np.ndarray:
    ss = 3
    img = Image.new("L", (max(1, w) * ss, max(1, h) * ss), 0)
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r * ss, fill=255)
    return np.asarray(img.resize((max(1, w), max(1, h)), Image.Resampling.BOX))


def round_rect(w: int, h: int, r: int, color: RGB, opacity: float = 1.0) -> Sprite:
    """Anti aliased rounded rectangle sprite."""
    return Sprite(coloured(_round_rect_mask(int(w), int(h), int(min(r, w // 2, h // 2))), color, opacity))


def rect(w: int, h: int, color: RGB, opacity: float = 1.0) -> Sprite:
    return Sprite(coloured(np.full((max(1, int(h)), max(1, int(w))), 255, np.uint8), color, opacity))


def pill(text: str, f: ImageFont.FreeTypeFont, fg: RGB, bg: RGB, pad_x: float, pad_y: float,
         tracking: float = 0.0, accent: Optional[RGB] = None, opacity: float = 0.94) -> Sprite:
    """Label pill: rounded panel with text; optional accent bar on the left."""
    t = text_sprite(text, f, fg, tracking)
    asc, desc = f.getmetrics()
    bar = int(max(3, f.size * 0.18)) if accent else 0
    w = int(t.w + 2 * pad_x + bar)
    h = int(asc + desc * 0.4 + 2 * pad_y)
    shadow_pad = int(f.size * 0.6)
    base = np.zeros((h + 2 * shadow_pad, w + 2 * shadow_pad, 4), np.uint8)
    blit(base, round_rect(w, h, int(h * 0.22), bg, opacity), shadow_pad, shadow_pad)
    if accent:
        blit(base, rect(bar, h, accent), shadow_pad, shadow_pad)
    blit(base, t, shadow_pad + bar + pad_x, shadow_pad + pad_y)
    return Sprite(with_shadow(base, radius=f.size * 0.35, offset=int(f.size * 0.12), opacity=0.35), shadow_pad)


# ---------------------------------------------------------------- full frame helpers

def unpremultiply(rgba: np.ndarray) -> np.ndarray:
    """Premultiplied RGBA to straight RGBA (what ffmpeg overlay expects; avoids dark halos)."""
    a = rgba[..., 3]
    rgb = rgba[..., :3]
    a3 = cv2.merge([a, a, a])
    straight = cv2.divide(rgb, a3, scale=255.0)
    out = np.empty_like(rgba)
    out[..., :3] = straight
    out[..., 3] = a
    return out


def fade(frame: np.ndarray, f: float) -> np.ndarray:
    """Scale every channel by f (fade to black for opaque, to transparent for premultiplied)."""
    if f >= 0.999:
        return frame
    return cv2.convertScaleAbs(frame, alpha=max(0.0, f))


def vignette_mask(w: int, h: int, strength: float, power: float = 2.2) -> np.ndarray:
    """uint8 multiplier: 255 at center, darker toward the corners."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    d = np.sqrt(nx * nx * 0.85 + ny * ny * 1.1) / 1.18
    m = 1.0 - strength * np.clip(d, 0, 1.4) ** power
    return (np.clip(m, 0, 1) * 255).astype(np.uint8)


def texture(w: int, h: int, seed: int, scale: float, octaves: int = 4) -> np.ndarray:
    """Smooth value noise in [-1, 1] (paper blotches, terrain), float32."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = max(2, int(scale * 2 ** o))
        small = rng.standard_normal((max(2, int(cells * h / w)), cells)).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return np.clip(out / (np.abs(out).max() + 1e-6), -1, 1)


def background(pal: Palette, w: int, h: int, seed: int = 7) -> np.ndarray:
    """Never flat: radial gradient bg to bg2, low frequency texture and vignette (uint8 RGB)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w * 0.5) / w) ** 2 + ((yy - h * 0.42) / h) ** 2 * 0.8) / 0.62
    p = np.clip(d, 0, 1)[..., None] ** 1.6
    c0 = np.array(pal.rgb("bg"), np.float32)
    c1 = np.array(pal.rgb("bg2"), np.float32)
    img = c0 * (1 - p) + c1 * p
    tex = texture(w, h, seed, 3.0)[..., None]
    img *= 1.0 + tex * (0.05 if not pal.dark else 0.09)
    img *= (vignette_mask(w, h, 0.45 if pal.dark else 0.32)[..., None] / 255.0)
    return np.clip(img, 0, 255).astype(np.uint8)


class Grain:
    """Film grain applied per frame by slicing a pre-generated noise field at a moving offset."""

    def __init__(self, w: int, h: int, channels: int, strength: float, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        self.m = 48
        n = rng.normal(0, strength, (h + self.m, w + self.m)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 0.6)
        pos = np.clip(n, 0, 255).astype(np.uint8)
        neg = np.clip(-n, 0, 255).astype(np.uint8)
        self.pos = cv2.merge([pos] * channels)
        self.neg = cv2.merge([neg] * channels)
        self.w, self.h = w, h

    def apply(self, frame: np.ndarray, i: int) -> None:
        ox, oy = (i * 17) % self.m, (i * 29) % self.m
        c = frame.shape[2]
        cv2.add(frame, self.pos[oy:oy + self.h, ox:ox + self.w, :c], dst=frame)
        cv2.subtract(frame, self.neg[oy:oy + self.h, ox:ox + self.w, :c], dst=frame)


class GlowLayer:
    """Quarter resolution buffer for soft glows: draw strokes into .buf, then composite()."""

    def __init__(self, w: int, h: int, factor: int = 4) -> None:
        self.f = factor
        self.w, self.h = w, h
        self.buf = np.zeros((max(1, h // factor), max(1, w // factor), 4), np.uint8)
        self.dirty = False

    def clear(self) -> None:
        if self.dirty:
            self.buf[...] = 0
            self.dirty = False

    def pt(self, x: float, y: float) -> tuple[int, int]:
        return int(x / self.f * 16), int(y / self.f * 16)

    def composite(self, frame: np.ndarray, sigma: float, gain: float = 1.0) -> None:
        """Blur the buffer and lay it over frame (premultiplied over, so it works on light and dark)."""
        if not self.dirty:
            return
        g = cv2.GaussianBlur(self.buf, (0, 0), sigma)
        if gain != 1.0:
            g = cv2.convertScaleAbs(g, alpha=gain)
        bx, by, bw, bh = cv2.boundingRect(g[..., 3])
        if bw == 0 or bh == 0:
            return
        f = self.f
        X0, Y0 = bx * f, by * f
        X1, Y1 = min(self.w, (bx + bw) * f), min(self.h, (by + bh) * f)
        up = cv2.resize(g[by:by + bh, bx:bx + bw], (bw * f, bh * f), interpolation=cv2.INTER_LINEAR)
        up = up[:Y1 - Y0, :X1 - X0]
        roi = frame[Y0:Y1, X0:X1]
        c = frame.shape[2]
        inv_c = cv2.merge([255 - up[..., 3]] * c)
        cv2.multiply(roi, inv_c, dst=roi, scale=1 / 255.0)
        cv2.add(roi, up[..., :c], dst=roi)


def rgba(color: RGB, channels: int) -> tuple[int, ...]:
    """Opaque colour tuple for cv2 drawing on a frame with `channels` channels."""
    return tuple(int(c) for c in color) + ((255,) if channels == 4 else ())


# ---------------------------------------------------------------- composition base

class CompBase:
    """Shared state for one composition: size, palette, background, grain, safe area, source note.

    Subclasses implement draw(frame, t) and may override `own_exit` when they animate out themselves."""

    own_exit = False
    paints_background = False  # True when the composition draws its own full-frame background
    ui_scale = 1.0  # text and stroke scale on top of the frame size (insets render larger type)

    def __init__(self, spec, payload, pal: Palette, alpha: bool) -> None:  # noqa: ANN001
        self.spec, self.p, self.pal = spec, payload, pal
        self.W, self.H, self.fps = spec.width, spec.height, spec.fps
        self.k = min(self.W / 1920.0, self.H / 1080.0) * self.ui_scale
        self.transparent = spec.background in ("transparent", "footage_behind")
        self.C = 4 if alpha else 3
        self.safe_top = 150 * self.H / 1080
        self.safe_bottom = 930 * self.H / 1080
        self.margin = 110 * self.W / 1920
        self.dur = spec.duration
        self.glow_gain = 0.9 if pal.dark else 0.5
        plain = self.transparent or self.paints_background
        self._bg = None if plain else self.make_background()
        self._grain = None if plain else Grain(self.W, self.H, self.C, 5.0 if pal.dark else 4.0)
        self._note = self._note_sprite(spec.source_note)

    def make_background(self) -> np.ndarray:
        bg = background(self.pal, self.W, self.H)
        return bg if self.C == 3 else np.dstack([bg, np.full(bg.shape[:2], 255, np.uint8)])

    def canvas(self) -> np.ndarray:
        if self._bg is not None:
            return self._bg.copy()
        return np.zeros((self.H, self.W, self.C), np.uint8)

    def px(self, v: float) -> int:
        """Design pixels at 1080p to output pixels."""
        return int(round(v * self.k))

    def _note_sprite(self, note: str) -> Optional[Sprite]:
        note = (note or "").strip()
        if not note:
            return None
        f, lines = fit_text(note, "Medium", self.px(21), self.W * 0.45, 1, self.px(13))
        col = self.pal.rgb("muted") if not self.transparent else (235, 235, 235)
        return text_sprite(lines[0], f, col, tracking=f.size * 0.02, shadow=0.6 if self.transparent else 0.0)

    def draw(self, frame: np.ndarray, t: float) -> None:
        raise NotImplementedError

    def render(self, i: int) -> np.ndarray:
        """Frame i (premultiplied when 4 channels)."""
        t = i / self.fps
        frame = self.canvas()
        self.draw(frame, t)
        if self._grain is not None:
            self._grain.apply(frame, i)
        if self._note is not None:
            x = self.W - self.margin * 0.55 - self._note.w
            y = self.safe_bottom - self._note.h - self.px(14)
            blit(frame, self._note, x, y, 0.9 * self.presence(t))
        return frame

    def anchor(self, w: float, h: float, position: str, inset: float = 30.0) -> tuple[float, float]:
        """Top-left of a w x h block placed at a TextPayload position inside the safe area."""
        left, right = self.margin, self.W - self.margin - w
        top, bottom = self.safe_top + self.px(inset), self.safe_bottom - h - self.px(inset)
        return {
            "upper_left": (left, top), "upper_right": (right, top),
            "lower_left": (left, bottom), "lower_right": (right, bottom),
        }.get(position, ((self.W - w) / 2, (self.H - h) / 2))

    def presence(self, t: float) -> float:
        """Opacity envelope for secondary elements (source note) of transparent overlays."""
        if not self.transparent:
            return 1.0
        from .easing import fade_factor
        return fade_factor(t, self.dur, 0.4)


# ---------------------------------------------------------------- vector shapes (cv2, anti aliased, subpixel)

_SHIFT = 4
_ONE = 1 << _SHIFT


def _fixed(pts: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(pts, np.float64) * _ONE).astype(np.int32).reshape(-1, 1, 2)


def polyline(frame: np.ndarray, pts: np.ndarray, color: RGB, thickness: float, closed: bool = False) -> None:
    """Opaque anti aliased polyline in frame pixel coordinates (float)."""
    if len(pts) < 2:
        return
    cv2.polylines(frame, [_fixed(pts)], closed, rgba(color, frame.shape[2]), max(1, int(round(thickness))),
                  cv2.LINE_AA, _SHIFT)


def fill_poly(frame: np.ndarray, polys: list[np.ndarray], color: RGB) -> None:
    """Opaque anti aliased filled polygons."""
    if polys:
        cv2.fillPoly(frame, [_fixed(p) for p in polys], rgba(color, frame.shape[2]), cv2.LINE_AA, _SHIFT)


def disc(frame: np.ndarray, x: float, y: float, r: float, color: RGB, thickness: int = -1) -> None:
    """Opaque anti aliased circle (filled when thickness < 0)."""
    if r <= 0.3:
        return
    cv2.circle(frame, (int(x * _ONE), int(y * _ONE)), int(r * _ONE), rgba(color, frame.shape[2]),
               thickness, cv2.LINE_AA, _SHIFT)


def translucent(frame: np.ndarray, color: RGB, opacity: float, draw: Callable[[np.ndarray, Callable], None],
                box: tuple[int, int, int, int], weight: Optional[np.ndarray] = None) -> None:
    """Draw a translucent shape correctly: draw(mask, to_local) paints 255 into a uint8 mask covering
    box (x0, y0, x1, y1); the mask (times an optional frame-sized weight in 0..1) is blended over frame."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    if x1 <= x0 or y1 <= y0 or opacity <= 0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    draw(mask, lambda pts: np.asarray(pts, np.float64) - [x0, y0])
    a = mask.astype(np.float32) * (opacity / 255.0)
    if weight is not None:
        a *= weight[y0:y1, x0:x1]
    a = a[..., None]
    roi = frame[y0:y1, x0:x1]
    src = np.array(color + ((255,) if roi.shape[2] == 4 else ()), np.float32)
    roi[...] = (roi.astype(np.float32) * (1 - a) + src * a).clip(0, 255).astype(np.uint8)


def mask_poly(mask: np.ndarray, polys: list[np.ndarray]) -> None:
    """Fill polygons with 255 into a uint8 mask (anti aliased)."""
    cv2.fillPoly(mask, [_fixed(p) for p in polys], 255, cv2.LINE_AA, _SHIFT)


def mask_line(mask: np.ndarray, pts: np.ndarray, thickness: float, closed: bool = False) -> None:
    """Stroke a polyline with 255 into a uint8 mask (anti aliased)."""
    cv2.polylines(mask, [_fixed(pts)], closed, 255, max(1, int(round(thickness))), cv2.LINE_AA, _SHIFT)


def luminance(c: RGB) -> float:
    return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255.0


def on_colour(c: RGB) -> RGB:
    """Readable text colour on top of c."""
    return (20, 18, 16) if luminance(c) > 0.55 else (255, 255, 255)


def glow_sprite(spr: Sprite, color: RGB, sigma: float, opacity: float) -> Sprite:
    """Soft coloured glow from a sprite's alpha, padded so the blur is not clipped."""
    pad = int(sigma * 3)
    a = np.pad(spr.arr[..., 3], pad)
    small = cv2.resize(a, (max(1, a.shape[1] // 4), max(1, a.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / 4))
    big = cv2.resize(small, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return Sprite(coloured(big, color, opacity), spr.pad + pad)


def path_prefix(pts: np.ndarray, frac: float) -> np.ndarray:
    """Leading part of a polyline covering frac of its arc length (interpolated end point)."""
    if frac <= 0 or len(pts) < 2:
        return pts[:0]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = cum[-1] * min(1.0, frac)
    k = int(np.searchsorted(cum, target, side="right"))
    if k >= len(pts):
        return pts
    a, b = pts[k - 1], pts[k]
    u = (target - cum[k - 1]) / max(1e-9, seg[k - 1])
    return np.vstack([pts[:k], a + (b - a) * u])


def arrow_head(pts: np.ndarray, size: float) -> np.ndarray:
    """Arrow head triangle at the end of pts, pointing along the final tangent."""
    tip = pts[-1]
    d = tip - pts[-4]
    d = d / (np.hypot(*d) or 1.0)
    n = np.array([-d[1], d[0]])
    base = tip - d * size
    return np.array([tip + d * size * 0.15, base + n * size * 0.55, base - n * size * 0.55])
