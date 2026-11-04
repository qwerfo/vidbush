"""Inset panel: a map or chart composited over footage or a still as an animated overlay.

Reference documentaries keep the picture on footage and stills and move their
maps and charts over it, so a map or chart spec with a transparent background
renders as a framed panel in the upper right of the safe area instead of a
full frame. The inner composition is the normal MapComp or chart, rendered
opaque at INSET_FRACTION of the frame with larger type (ui_scale) so labels
stay readable, then placed with rounded corners, a thin border and a soft
shadow. The panel slides in and fades out by itself (own_exit).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..models import MotionKind, MotionSpec
from . import easing as ez
from .textdraw import CompBase, _round_rect_mask, blit

INSET_KINDS = {MotionKind.map, MotionKind.bar_chart, MotionKind.line_chart}
INSET_FRACTION = 0.45  # panel width as a share of the frame width
INSET_UI_SCALE = 1.6  # type scale inside the panel, so labels read at panel size
IN_S, OUT_S = 0.45, 0.35


def is_inset(spec: MotionSpec) -> bool:
    """A map or chart spec meant to ride over the picture (transparent background)."""
    return spec.kind in INSET_KINDS and spec.background in ("transparent", "footage_behind")


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


class InsetComp(CompBase):
    """Framed, animated panel around an opaque inner map or chart composition."""

    own_exit = True

    def __init__(self, spec: MotionSpec, payload: Any, pal: Any, alpha: bool, inner_cls: type) -> None:
        super().__init__(spec, payload, pal, alpha)
        self.pw, self.ph = _even(self.W * INSET_FRACTION), _even(self.H * INSET_FRACTION)
        inner_spec = spec.model_copy(update={"width": self.pw, "height": self.ph, "background": "solid"})
        scaled_cls = type(f"Inset{inner_cls.__name__}", (inner_cls,), {"ui_scale": INSET_UI_SCALE})
        self.inner = scaled_cls(inner_spec, payload, pal, False)
        self._note = None  # the inner composition draws the source note inside the panel
        r = max(4, self.px(14))
        self.mask = _round_rect_mask(self.pw, self.ph, r).astype(np.float32) / 255.0
        self.x0 = self.W - self.margin - self.pw
        self.y0 = self.safe_top + self.px(10)
        self.shadow = self._shadow()

    def _shadow(self) -> np.ndarray:
        pad = self.px(30)
        a = np.zeros((self.ph + 2 * pad, self.pw + 2 * pad), np.float32)
        a[pad + self.px(6):pad + self.px(6) + self.ph, pad + self.px(6):pad + self.px(6) + self.pw] = self.mask
        a = cv2.GaussianBlur(a, (0, 0), max(1.0, self.px(12))) * 0.55
        out = np.zeros(a.shape + (4,), np.uint8)
        out[..., 3] = (a * 255).astype(np.uint8)
        self._shadow_pad = pad
        return out

    def opacity(self, t: float) -> float:
        p_in = ez.ease_out_cubic(ez.progress(t, 0.0, IN_S))
        p_out = 1.0 - ez.progress(t, self.dur - OUT_S, self.dur)
        return max(0.0, min(p_in, p_out))

    def panel(self, i: int) -> np.ndarray:
        """Premultiplied RGBA panel: inner frame, rounded corners, thin light border."""
        rgb = self.inner.render(i).astype(np.float32)
        m = self.mask[..., None]
        edge = cv2.erode(self.mask, np.ones((3, 3), np.uint8), iterations=max(1, self.px(2)))
        border = (self.mask - edge)[..., None]
        rgb = rgb * (1 - border * 0.8) + 255.0 * border * 0.8
        out = np.empty((self.ph, self.pw, 4), np.uint8)
        out[..., :3] = np.clip(rgb * m, 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(self.mask * 255.0, 0, 255).astype(np.uint8)
        return out

    def render(self, i: int) -> np.ndarray:
        t = i / self.fps
        frame = np.zeros((self.H, self.W, self.C), np.uint8)
        op = self.opacity(t)
        if op <= 0.003:
            return frame
        slide = (1.0 - ez.ease_out_cubic(ez.progress(t, 0.0, IN_S))) * self.px(40)
        x, y = self.x0 + slide, self.y0
        pad = self._shadow_pad
        blit(frame, self.shadow, x - pad, y - pad, op)
        blit(frame, self.panel(i), x, y, op)
        return frame
