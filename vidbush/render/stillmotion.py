"""Still motion with a destination.

Run 3 gave every still the same gentle zoom in a hashed direction. Here the move is chosen from the
picture (local analysis only: the prefilter's face cascade and a spectral residual saliency map) and the
shot, and has a reason a review can read:
- a planner labelled sequence (Shot.still_sequence) serves one high resolution photograph over
  consecutive shots: whole (the full picture, easing toward the subject), closer, detail (framed on the
  subject, never enlarged more than MAX_UPSCALE times its native pixels);
- a short shot (under HOLD_S) holds: a held picture reads faster than one in motion;
- a small subject (a face or a detail) in a shot long enough: start on the detail and reveal the whole;
- a larger subject: start on the whole picture and travel toward it;
- no clear subject (open landscape): a lateral drift across the long side;
- a narrow still that renders framed on the house ground: a slow push on the whole frame.
A move never cuts through a face: faces are found once per file on the uncropped picture, the cover crop
keeps every face whole (cover_box), and each end of a move is shifted, then widened, until its window holds
every face whole (keep_faces).
Parameters use the Ken Burns contract (zoom0, zoom1, x0, y0, x1, y1 in the prepared still's 0..1 space).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..util import stable_hash

HOLD_S = 2.6
MAX_UPSCALE = 1.5
SMALL_SUBJECT = 0.12  # subject box area share below which we reveal from the detail
SEQ_ZOOM = {"whole": (1.0, 1.05), "closer": (1.55, 1.62), "detail": (2.4, 2.5)}


def cover_box(path: str, crop: Optional[Sequence[float]], iw: int, ih: int,
              aspect: float) -> Optional[tuple[float, float, float, float]]:
    """The frame-aspect window (x, y, w, h, 0..1 of the iw x ih cropped picture) its cover crop shows: centred,
    moved to keep every face whole; None when no window of that shape can (the still is shown whole on its
    soft fill instead)."""
    from ..mg.layouts.placement import cover_window, faces_in_crop, faces_of
    return cover_window(faces_in_crop(faces_of(path), crop), iw / max(ih, 1), aspect)


@lru_cache(maxsize=512)
def _subject(path: str, mtime: float, crop: Optional[tuple[float, ...]],
             aspect: float) -> tuple[Optional[tuple[float, ...]], tuple[tuple[float, ...], ...]]:
    """((x0, y0, x1, y1, is_face) of the main subject or None, every face box), in the cover-cropped picture's
    0..1 space."""
    import cv2

    from ..mg.layouts.placement import _saliency, cover_window, faces_in_crop, faces_of
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None, ()
    h, w = img.shape[:2]
    if crop:
        x, y, cw, ch = crop
        img = img[int(y * h):int((y + ch) * h), int(x * w):int((x + cw) * w)]
        h, w = img.shape[:2]
    # the cover crop segment._prepare_still makes (centred unless a face needs it moved)
    win = cover_box(path, crop, w, h, aspect) or cover_window((), w / h, aspect)
    faces = faces_in_crop(faces_in_crop(faces_of(path), crop), win)
    x0, y0 = int(win[0] * w), int(win[1] * h)
    img = img[y0:y0 + max(1, int(win[3] * h)), x0:x0 + max(1, int(win[2] * w))]
    h, w = img.shape[:2]
    if faces:
        f = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return (*f, 1.0), faces
    s = 480 / w
    small = cv2.resize(img, (480, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    sal = _saliency(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    core = sal >= 0.6 * sal.max()
    if core.sum() < 0.002 * core.size or core.sum() > 0.5 * core.size:
        return None, ()
    ys, xs = np.nonzero(core)
    return (xs.min() / sw, ys.min() / sh, (xs.max() + 1) / sw, (ys.max() + 1) / sh, 0.0), ()


def keep_faces(kb: dict, faces: Sequence[Sequence[float]]) -> tuple[dict, bool]:
    """(kb, changed): each end of the move (its window is 1/zoom of the picture around x, y) shifted as little
    as it can, then widened, until the window holds every face box whole. Zoom 1 shows the whole cover crop,
    which already keeps every face whole, so every move ends up face safe."""
    if not faces:
        return kb, False
    from ..mg.layouts.placement import face_origin
    out, changed = dict(kb), False
    for zk, xk, yk in (("zoom0", "x0", "y0"), ("zoom1", "x1", "y1")):
        z = max(1.0, out[zk])
        while True:
            side = 1.0 / z
            o = face_origin(faces, side, side, (out[xk], out[yk]))
            if o is not None or z <= 1.0:
                break
            z = max(1.0, z - 0.05)
        o = o or (0.0, 0.0)
        new = (z, o[0] + side / 2, o[1] + side / 2)
        if any(abs(a - b) > 1e-3 for a, b in zip(new, (out[zk], out[xk], out[yk]))):
            changed = True
            out[zk], out[xk], out[yk] = round(new[0], 4), round(new[1], 4), round(new[2], 4)
    return out, changed


def zoom_cap(iw: int, ih: int, W: int, H: int) -> float:
    """Largest zoom that keeps the still under MAX_UPSCALE of its native pixels."""
    cover = max(W / max(iw, 1), H / max(ih, 1))
    return max(1.0, MAX_UPSCALE / cover)


def plan(shot: Any, cand: Any, W: int, H: int, intensity: float, framed: bool = False) -> tuple[Optional[dict], str]:
    """(Ken Burns params, reason), or (None, reason) to keep the default move. Every face stays whole."""
    p = Path(cand.local_path or "")
    if not p.is_file():
        return None, "picture missing"
    if framed:
        return {"zoom0": 1.0, "zoom1": 1.05, "x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.5}, \
            "narrow still framed on the house ground: slow push on the whole frame"
    crop = tuple(cand.crop) if cand.crop and len(cand.crop) == 4 else None
    sub, faces = _subject(str(p), p.stat().st_mtime, crop, W / H)
    kb, why = _move(shot, cand, p, crop, sub, W, H, intensity)
    kb, changed = keep_faces(kb, faces)
    return kb, why + ("; framed wider so every face stays whole" if changed else "")


def _move(shot: Any, cand: Any, p: Path, crop: Optional[tuple[float, ...]], sub: Optional[tuple[float, ...]],
          W: int, H: int, intensity: float) -> tuple[dict, str]:
    iw = int((cand.width or 0) * (crop[2] if crop else 1)) or W
    ih = int((cand.height or 0) * (crop[3] if crop else 1)) or H
    cap = zoom_cap(iw, ih, W, H)
    cx, cy = ((sub[0] + sub[2]) / 2, (sub[1] + sub[3]) / 2) if sub else (0.5, 0.5)
    seq = getattr(shot, "still_sequence", None)
    if seq is not None:
        z0, z1 = SEQ_ZOOM.get(seq.step, SEQ_ZOOM["whole"])
        if z1 > cap:
            z0, z1 = max(1.0, cap - (z1 - z0)), cap
        what = "the subject" if sub else "the centre"
        if seq.step == "whole":
            return {"zoom0": z0, "zoom1": z1, "x0": 0.5, "y0": 0.5, "x1": cx, "y1": cy}, \
                f"sequence {seq.id} whole view, easing toward {what}"
        return {"zoom0": z0, "zoom1": z1, "x0": cx, "y0": cy, "x1": cx, "y1": cy}, \
            f"sequence {seq.id} {seq.step} view on {what} at {z1:.2f}x (cap {cap:.2f}x)"
    if shot.duration < HOLD_S:
        return {"zoom0": 1.0, "zoom1": 1.015, "x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.5}, \
            f"short shot ({shot.duration:.1f} s): holds"
    reach = min(cap, 1.0 + max(0.12, intensity) * 2.2)
    if sub is None:
        wide = iw / max(ih, 1) >= W / H
        a, b = (0.5 - 0.12, 0.5 + 0.12) if (int(stable_hash("drift", p.name), 16) & 1) else (0.5 + 0.12, 0.5 - 0.12)
        z = min(cap, 1.12)
        if wide:
            return {"zoom0": z, "zoom1": z, "x0": a, "y0": 0.5, "x1": b, "y1": 0.5}, "no single subject: drifts across"
        return {"zoom0": z, "zoom1": z, "x0": 0.5, "y0": a, "x1": 0.5, "y1": b}, "no single subject: drifts down"
    area = (sub[2] - sub[0]) * (sub[3] - sub[1])
    kind = "face" if sub[4] else "detail"
    if area < SMALL_SUBJECT and shot.duration >= 4.0 and reach > 1.15:
        return {"zoom0": reach, "zoom1": 1.0, "x0": cx, "y0": cy, "x1": 0.5, "y1": 0.5}, \
            f"starts on the {kind} and reveals the whole picture"
    z = min(reach, 1.22)
    return {"zoom0": 1.0, "zoom1": z, "x0": 0.5, "y0": 0.5, "x1": cx, "y1": cy}, \
        f"starts on the whole picture and travels toward the {kind}"


def is_narrow(iw: int, ih: int, W: int, vis_h: int, max_scale: float = MAX_UPSCALE) -> bool:
    """True when the still, fitted to the picture height (and never enlarged past max_scale), is narrower
    than half the frame: it is framed on the house ground, never shown on blurred bars."""
    scale = min(vis_h / max(ih, 1), W / max(iw, 1), max_scale)
    return iw * scale < W / 2
