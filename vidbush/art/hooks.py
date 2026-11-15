"""The two single-line hooks the map renderer calls (vidbush/mg/mapdraw.py).

painted_base(base, plain): when the map payload names an art_base whose sidecar matches this view, the
painted image replaces the drawn land and water of MapBase.plain. True country borders and the major
river name labels are drawn back on top from the bundled data, so every line and word on the map is
the renderer's own; markers, routes, labels, title and the source note are drawn per frame as on a
plain map. Anything missing or mismatched returns the plain base unchanged: a map never shows a
painting of another view.

moving_base(comp, i, frame): when the payload also names an art_clip (a checked clip of that painting)
and the map has no shaded regions and a push-in of at most CLIP_MAX_PUSH, frame i of the clip replaces
the static base view, cropped to the frame's zoom exactly like MapBase.view crops the base.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

CLIP_MAX_PUSH = 1.5  # a clip covers the widest view at clip resolution; deeper push-ins would blur it
CLIP_MIN_S, CLIP_MAX_S = 3.0, 5.0  # the map holds its painting, then moves for this long at the end of the shot


def clip_allowed(p: Any) -> bool:
    """A painted map may play a clip as its base: no shaded region and a push-in of at most CLIP_MAX_PUSH."""
    z1 = p.zoom_end if p.zoom_end is not None else p.zoom
    return not p.shaded_regions and 2 ** abs(z1 - p.zoom) <= CLIP_MAX_PUSH + 1e-6


def sidecar(path: str | Path) -> Path:
    return Path(path).with_suffix(".json")


def _matches(p: Any, path: str) -> bool:
    from .mapref import view_key
    try:
        meta = json.loads(sidecar(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(meta.get("ok")) and meta.get("view") == view_key(p)


def _overlay(base: Any) -> np.ndarray:
    """Premultiplied RGBA layer of the true country borders and river name labels at the base size."""
    from ..mg import mapdraw as M
    from ..mg.palettes import mix
    pal = base.c.pal
    col = np.array(mix(pal.rgb("land_border"), (255, 255, 255), 0.35), np.float32)
    width = max(1.0, 1.3 * base.k)
    lines = np.zeros((base.bh, base.bw, 3), np.uint8)
    for r in base.rings(base.countries):
        M.polyline(lines, r, (255, 255, 255), width)
    a = lines[..., 0].astype(np.float32) / 255.0 * (170 / 255.0)
    layer = np.dstack([col[None, None, :] * a[..., None], a[..., None] * 255.0]).clip(0, 255).astype(np.uint8)
    if base.z_lo >= 3.5:
        rivers = [f for f in M.load_layer(M.RIVERS) if (f[0].get("scalerank") or 9) <= M.RIVER_RANK]
        base._river_labels(layer, rivers)  # blit composites premultiplied sprites, halo included
    return layer


def _composite(img: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    a = overlay[..., 3:4].astype(np.float32) / 255.0
    out = img[..., :3].astype(np.float32) * (1 - a) + overlay[..., :3].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def painted_base(base: Any, plain: np.ndarray) -> np.ndarray:
    """MapBase.plain with the painting of this view, or plain unchanged."""
    p = base.c.p
    path = getattr(p, "art_base", "") or ""
    if not path or not Path(path).exists() or not _matches(p, path):
        return plain
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return plain
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_AREA if img.shape[1] > base.bw else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (base.bw, base.bh), interpolation=interp)
    base.art_overlay = _overlay(base)
    return base._channels(_composite(img, base.art_overlay))


class _ClipReader:
    """Sequential frame reader with a one-frame cache (renderers ask for frames in order)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.cap: Optional[cv2.VideoCapture] = None
        self.idx = -1
        self.frame: Optional[np.ndarray] = None
        probe = cv2.VideoCapture(path)
        self.fps = probe.get(cv2.CAP_PROP_FPS) or 24.0
        self.count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        probe.release()

    def get(self, k: int) -> Optional[np.ndarray]:
        k = max(0, min(k, self.count - 1)) if self.count > 0 else max(0, k)
        if self.cap is None or k < self.idx:
            if self.cap is not None:
                self.cap.release()
            self.cap, self.idx = cv2.VideoCapture(self.path), -1
        while self.idx < k:
            ok, f = self.cap.read()
            if not ok:
                break
            self.idx, self.frame = self.idx + 1, f
        return self.frame


def _clip_usable(comp: Any) -> bool:
    p = comp.p
    clip = getattr(p, "art_clip", "") or ""
    if not clip or not Path(clip).exists() or getattr(comp.base, "art_overlay", None) is None:
        return False
    return clip_allowed(p)


def moving_base(comp: Any, i: int, frame: np.ndarray) -> np.ndarray:
    """Frame i of the painted map's clip at the frame's zoom, or frame unchanged."""
    if not _clip_usable(comp):
        return frame
    reader = getattr(comp, "_art_reader", None)
    if reader is None:
        reader = comp._art_reader = _ClipReader(comp.p.art_clip)
    # hold the painting, then move for CLIP_MIN_S to CLIP_MAX_S at the end (the clip's first frame is the painting)
    clip_s = reader.count / reader.fps if reader.count else CLIP_MAX_S
    length = min(max(comp.dur, CLIP_MIN_S), CLIP_MAX_S, clip_s)
    t_clip = i / comp.fps - max(0.0, comp.dur - length)
    if t_clip < 0:
        return frame
    raw = reader.get(int(t_clip * reader.fps))
    if raw is None:
        return frame
    base = comp.base
    if getattr(comp, "_art_size", None) != raw.shape[:2]:
        comp._art_size = raw.shape[:2]
        comp._art_over = cv2.resize(base.art_overlay, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
    rgb = _composite(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), comp._art_over)
    t = i / comp.fps
    m = 2 ** (comp.zoom_at(t) - base.z_lo)  # 1 at the widest view: the clip is that view
    h, w = rgb.shape[:2]
    cw, ch = w / m, h / m
    x0, y0 = (w - cw) / 2, (h - ch) / 2
    crop = rgb[int(round(y0)):int(round(y0 + ch)), int(round(x0)):int(round(x0 + cw))]
    out = cv2.resize(crop, (comp.W, comp.H), interpolation=cv2.INTER_LINEAR)
    if comp.C == 4:
        out = np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])
    return np.ascontiguousarray(out)
