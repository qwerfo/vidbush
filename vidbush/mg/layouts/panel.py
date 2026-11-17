"""The panel system: a route map, chart, table or timeline over the shot's picture.

Four presentations (LayoutSpec.presentation):
- full: the whole frame (rendered as an opaque full layout, not here).
- panel: a framed panel at the place chosen from the picture beneath (placement.py).
- full_then_dock: opens at about 75 percent of the frame so the viewer can take it in, then shrinks and
  settles as a panel while the picture continues underneath.
- panel_then_expand: starts as a panel, then grows to the full frame, and its stage 2 elements appear only
  once it is large enough to read them.

The content is laid out twice, once at the large size and once at the panel size, each with type at or
above the house floor; while the size changes (DOCK_S) the outgoing layout is scaled to the moving rectangle
and cross-fades into the incoming one, so text is never shrunk below its floor once the move has settled.
The overlay clip runs from the panel's first element to the moment its content stops being relevant; it
does not run to the cut.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .. import easing as ez
from ..textdraw import _round_rect_mask, blit, rect
from .base import LayoutComp, LayoutJob

DOCK_S = 0.8  # length of a size change
IN_S, OUT_S = 0.4, 0.35
LARGE = 0.75  # full_then_dock opening size, share of the visible frame
NEVER = 1e9

Rect = tuple[float, float, float, float]  # x, y, w, h


@dataclass
class PanelPlan:
    presentation: str
    small: Rect  # the panel's resting rectangle (frame pixels)
    large: Rect  # the opening (dock) or final (expand) rectangle
    change_at: float = NEVER  # when the size change starts (clip seconds)
    position: str = ""


def large_rect(job: LayoutJob, full: bool) -> Rect:
    x0, y0, x1, y1 = job.frame.visible(job.width, job.height)
    if full:
        return (x0, y0, x1 - x0, y1 - y0)
    w, h = (x1 - x0) * LARGE, (y1 - y0) * LARGE
    return (x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2, w, h)


def change_time(job: LayoutJob, presentation: str) -> float:
    """When a dock or expand starts.

    Dock: once the stage 1 content has been read (last stage 1 word plus 1.2 s, at least 2.4 s after the
    open), leaving at least 2 s docked. Expand: so the growth ends just as the first stage 2 element is due."""
    cues = job.cues
    if presentation == "full_then_dock":
        s1 = [c for c in cues if c.stage == 1]
        last = max((c.word_end for c in s1), default=0.0)
        return max(0.0, min(max(last + 1.2, 2.4), job.duration - DOCK_S - 2.0))
    if presentation == "panel_then_expand":
        s2 = [c.appear for c in cues if c.stage == 2]
        if not s2:
            return NEVER
        return max(1.5, min(s2) - DOCK_S)
    return NEVER


def _with_hidden_stage2(job: LayoutJob) -> LayoutJob:
    j = copy.copy(job)
    j.cues = [copy.copy(c) for c in job.cues]
    for c in j.cues:
        if c.stage == 2:
            c.appear = c.word = c.word_end = NEVER
    return j


def _with_stage2_after(job: LayoutJob, ready: float) -> LayoutJob:
    """Stage 2 elements wait until the panel is large enough (later than their word is allowed, earlier never)."""
    j = copy.copy(job)
    j.cues = [copy.copy(c) for c in job.cues]
    for c in j.cues:
        if c.stage == 2 and c.appear < ready:
            shift = ready - c.appear
            c.appear += shift
    return j


class PanelComp:
    """Renders premultiplied RGBA frames of a panel presentation over a transparent frame."""

    def __init__(self, job: LayoutJob, factory: Callable[[LayoutJob, int, int, bool], LayoutComp],
                 plan: PanelPlan) -> None:
        self.job, self.plan = job, plan
        self.W, self.H, self.fps, self.dur = job.width, job.height, job.fps, job.duration
        sw, sh = int(plan.small[2]) // 2 * 2, int(plan.small[3]) // 2 * 2
        lw, lh = int(plan.large[2]) // 2 * 2, int(plan.large[3]) // 2 * 2
        pres = plan.presentation
        if pres == "panel_then_expand":
            ready = plan.change_at + DOCK_S
            self.small = factory(_with_hidden_stage2(job), sw, sh, True)
            self.large = factory(_with_stage2_after(job, ready), lw, lh, lh < job.height * 0.95)
        elif pres == "full_then_dock":
            self.large = factory(job, lw, lh, True)
            self.small = factory(job, sw, sh, True)
        else:
            self.small = factory(_with_hidden_stage2(job), sw, sh, True)
            self.large = None
        self.k = job.height / 1080.0
        self.accent = job.house.rgb("accent")

    def phase(self, t: float) -> float:
        """0 = at the first rectangle, 1 = at the second (eased)."""
        return ez.ease_in_out_cubic(ez.progress(t, self.plan.change_at, self.plan.change_at + DOCK_S))

    def rect_at(self, t: float) -> Rect:
        p = self.plan
        if p.presentation == "panel":
            return p.small
        a, b = (p.large, p.small) if p.presentation == "full_then_dock" else (p.small, p.large)
        u = self.phase(t)
        return tuple(ez.lerp(x, y, u) for x, y in zip(a, b))  # type: ignore[return-value]

    def content(self, i: int, t: float, w: int, h: int) -> np.ndarray:
        p = self.plan
        if p.presentation == "panel":
            img = self.small.render(i)
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            return img
        first, second = (self.large, self.small) if p.presentation == "full_then_dock" else (self.small, self.large)
        u = self.phase(t)
        if u <= 0.0:
            img = first.render(i)
        elif u >= 1.0:
            img = second.render(i)
        else:
            a = cv2.resize(first.render(i), (w, h), interpolation=cv2.INTER_AREA)
            b = cv2.resize(second.render(i), (w, h), interpolation=cv2.INTER_AREA)
            mixu = ez.clamp01((u - 0.42) / 0.16)  # a short swap mid-move: no double text ghosting
            return cv2.addWeighted(a, 1 - mixu, b, mixu, 0)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    def opacity(self, t: float) -> tuple[float, float]:
        o_in = ez.ease_out_cubic(ez.progress(t, 0.0, IN_S))
        o_out = 1.0 - ez.ease_in_cubic(ez.progress(t, self.dur - OUT_S, self.dur))
        return min(o_in, o_out), o_in

    def render(self, i: int) -> np.ndarray:
        t = i / self.fps
        frame = np.zeros((self.H, self.W, 4), np.uint8)
        op, o_in = self.opacity(t)
        if op <= 0.003:
            return frame
        x, y, w, h = self.rect_at(t)
        s = 0.94 + 0.06 * o_in  # a short settle as it opens
        cx, cy = x + w / 2, y + h / 2
        w2, h2 = max(8, int(w * s)) // 2 * 2, max(8, int(h * s)) // 2 * 2
        x2, y2 = int(round(cx - w2 / 2)), int(round(cy - h2 / 2))
        full = w2 >= self.W - 2 and h2 >= self.H * 0.95
        img = self.content(i, t, w2, h2)
        keep3, ring_rgb, alpha, shadow = self._chrome(w2, h2, full)
        # premultiplied body in uint8: picture times the rounded mask (and the paper ring), alpha = mask
        body_rgb = cv2.add(cv2.multiply(img, keep3, scale=1 / 255.0), ring_rgb) if ring_rgb is not None else             cv2.multiply(img, keep3, scale=1 / 255.0)
        body = np.dstack([body_rgb, alpha])
        if op < 0.997:
            body = cv2.convertScaleAbs(body, alpha=op)
        if shadow is not None:
            pad = int(28 * self.k) + 2
            _paste(frame, shadow if op >= 0.997 else cv2.convertScaleAbs(shadow, alpha=op), x2 - pad, y2 - pad)
        _over(frame, body, x2, y2)
        if not full:  # amber rule grows along the top edge as the panel opens: the house signature
            rw = w2 * 0.28 * o_in
            if rw > 2:
                blit(frame, rect(rw, max(3, int(5 * self.k)), self.accent), x2 + int(14 * self.k), y2 - int(9 * self.k), op)
        return frame

    def _chrome(self, w: int, h: int, full: bool) -> tuple:
        """uint8 picture keep mask (3 channels), premultiplied paper ring, alpha and shadow for a panel size,
        built once per size (sizes repeat while the panel rests)."""
        key = (w, h, full)
        cache = self.__dict__.setdefault("_chrome_cache", {})
        if key not in cache:
            r = 0 if full else max(3, int(6 * self.k))
            m = _round_rect_mask(w, h, r).astype(np.float32) / 255.0 if r else np.ones((h, w), np.float32)
            ring_rgb = shadow = None
            keep = m
            if not full:
                edge = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(2 * self.k)))
                ring = (m - edge) * 0.85
                keep = m * (1 - ring)
                ring_rgb = cv2.merge([np.clip(ring * 238.0, 0, 255).astype(np.uint8)] * 3)
                shadow = self._shadow_sprite(w, h, m)
            keep3 = cv2.merge([np.clip(keep * 255.0, 0, 255).astype(np.uint8)] * 3)
            alpha = np.clip(m * 255.0, 0, 255).astype(np.uint8)
            if len(cache) > 64:
                cache.clear()
            cache[key] = (keep3, ring_rgb, alpha, shadow)
        return cache[key]

    def _shadow_sprite(self, w: int, h: int, m: np.ndarray) -> np.ndarray:
        pad = int(28 * self.k) + 2
        a = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        off = int(8 * self.k)
        a[pad + off:pad + off + h, pad + off:pad + off + w] = m
        a = cv2.GaussianBlur(a, (0, 0), 14 * self.k + 1) * 0.6
        sh = np.zeros(a.shape + (4,), np.uint8)
        sh[..., 3] = (a * 255).astype(np.uint8)
        return sh

    def close(self) -> None:
        for c in (self.small, self.large):
            if c is not None:
                c.close()


def _clip_roi(frame: np.ndarray, src: np.ndarray, x: int, y: int):  # noqa: ANN202
    H, W = frame.shape[:2]
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1], src[y0 - y:y1 - y, x0 - x:x1 - x]


def _paste(frame: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Copy src into an empty (zero) region of frame."""
    got = _clip_roi(frame, src, x, y)
    if got is not None:
        got[0][...] = got[1]


def _over(frame: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Premultiplied src over frame, uint8 throughout (cv2), in place."""
    got = _clip_roi(frame, src, x, y)
    if got is None:
        return
    roi, s = got
    inv = cv2.merge([255 - s[..., 3]] * 4)
    cv2.multiply(roi, inv, dst=roi, scale=1 / 255.0)
    cv2.add(roi, np.ascontiguousarray(s), dst=roi)
