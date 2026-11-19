"""Where a panel sits, decided from the picture beneath it for the whole time it is visible.

The underlying clip or still is sampled across the panel's visible time (as it will be on screen, still
motion included). Each sample is read with local analysis only: the prefilter's face cascade, the
prefilter's OCR engine for text, and a detail measure (local gradient energy) plus a spectral residual
saliency map whose strongest blob stands for the main subject. Six positions are allowed: the four corners,
left centre and right centre, all inside the visible picture (letterbox bars and caption band excluded).

A position is ruled out when the panel would cover a face, text, or the core of the main subject in any
sample. Among the rest the calmest wins (least detail and saliency under the panel). Two panels in a row
do not take the same position unless it is the only clear one. When nothing is clear the caller shows the
content full frame (or on the next shot) instead. The reason is returned in words for the shot's notes.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional, Sequence

import cv2
import numpy as np

POSITIONS = ("upper_left", "upper_right", "lower_left", "lower_right", "left_centre", "right_centre")
WORK_W = 480
REPEAT_PENALTY = 0.12
SUBJECT_CORE = 0.6  # saliency share of the peak that counts as the subject's core
SUBJECT_MAX = 0.2  # a subject core wider than this share of the picture is not one subject
SUBJECT_OVERLAP = 0.05  # share of the subject's heart a panel may cover
HEART = 0.6  # a panel may cover at most this share of the subject core
MARGIN = 0.035  # gap between the panel and the picture edge, share of frame width
FACE_MIN = 0.02  # a face smaller than this share of the picture height is ignored by crops
FACE_PAD = 0.15  # each face box grows by this share of its size, so hair and chin stay inside a crop
FACE_W = 800  # width a picture is searched for faces at (a FACE_MIN face is still a dozen pixels)
_LOCAL = threading.local()


@dataclass
class Analysis:
    faces: list[tuple[float, float, float, float]] = field(default_factory=list)  # 0..1 boxes x0,y0,x1,y1
    text: list[tuple[float, float, float, float]] = field(default_factory=list)
    detail: Optional[np.ndarray] = None  # float map 0..1 at work size
    saliency: Optional[np.ndarray] = None
    protect: list[tuple[float, float, float, float]] = field(default_factory=list)  # boxes never to cover


@dataclass
class Placement:
    position: Optional[str]
    rect: Optional[tuple[float, float, float, float]]  # x, y, w, h in frame pixels
    reason: str
    scores: dict[str, float] = field(default_factory=dict)


def rect_for(position: str, size: tuple[float, float], visible: tuple[float, float, float, float],
             W: int) -> tuple[float, float, float, float]:
    """Panel rectangle (x, y, w, h) at a named position inside the visible area."""
    x0, y0, x1, y1 = visible
    m = MARGIN * W
    w, h = size
    left, right = x0 + m, x1 - m - w
    top, bottom, mid = y0 + m * 0.8, y1 - m * 0.8 - h, y0 + (y1 - y0 - h) / 2
    return {
        "upper_left": (left, top), "upper_right": (right, top), "lower_left": (left, bottom),
        "lower_right": (right, bottom), "left_centre": (left, mid), "right_centre": (right, mid),
    }[position] + (w, h)


# ---------------------------------------------------------------- analysis

def _profile_cascade() -> Any:
    """Profile face cascade, built once per thread (None when OpenCV has none)."""
    if not hasattr(_LOCAL, "profile"):
        c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        _LOCAL.profile = None if c.empty() else c
    return _LOCAL.profile


def _faces(grey: np.ndarray, min_share: float = 1 / 25) -> list[tuple[int, int, int, int]]:
    try:
        from ...judge.prefilter import face_cascade
        casc = face_cascade()
    except Exception:  # noqa: BLE001  no cascade: no face boxes
        return []
    side = max(12, int(grey.shape[0] * min_share))
    found = casc.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(side, side))
    out = [tuple(int(v) for v in b) for b in (found if len(found) else [])]
    try:  # profile faces too: a face turned sideways is still a face
        prof = _profile_cascade()
        if prof is not None:
            for flip in (False, True):
                g = cv2.flip(grey, 1) if flip else grey
                for x, y, w, hh in prof.detectMultiScale(g, 1.1, 6, minSize=(side, side)):
                    out.append((grey.shape[1] - x - w if flip else x, y, w, hh))
    except Exception:  # noqa: BLE001
        pass
    return out


Box = tuple[float, float, float, float]


def face_boxes(grey: np.ndarray) -> tuple[Box, ...]:
    """Every face of at least FACE_MIN of the height in a grey picture: padded 0..1 boxes (x0, y0, x1, y1)."""
    h, w = grey.shape[:2]
    if w != FACE_W:
        grey = cv2.resize(grey, (FACE_W, max(1, round(h * FACE_W / w))), interpolation=cv2.INTER_AREA)
        h, w = grey.shape[:2]
    out = []
    for x, y, fw, fh in _faces(grey, FACE_MIN):
        px, py = fw * FACE_PAD, fh * FACE_PAD
        out.append((max(0.0, (x - px) / w), max(0.0, (y - py) / h), min(1.0, (x + fw + px) / w),
                    min(1.0, (y + fh + py) / h)))
    return tuple(out)


@lru_cache(maxsize=2048)
def _file_faces(path: str, mtime: float) -> tuple[Box, ...]:
    from PIL import Image, ImageOps
    try:
        with Image.open(path) as im:
            im.draft("L", (2 * FACE_W, 2 * FACE_W))  # a JPEG decodes at a fraction of its size
            grey = np.asarray(ImageOps.exif_transpose(im).convert("L"))
    except Exception:  # noqa: BLE001  unreadable: no face boxes
        return ()
    return face_boxes(grey)


def faces_of(path: str) -> tuple[Box, ...]:
    """Faces of a picture file in the 0..1 space of the whole, uncropped picture; one detection per file."""
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return ()
    return _file_faces(str(path), mtime)


def faces_in_crop(faces: Sequence[Box], crop: Optional[Sequence[float]]) -> tuple[Box, ...]:
    """Face boxes in the 0..1 space of crop [x, y, w, h] of the picture; faces wholly outside it are left out."""
    if not crop:
        return tuple(faces)
    x, y, w, h = crop
    out = []
    for a, b, c, d in faces:
        box = ((a - x) / w, (b - y) / h, (c - x) / w, (d - y) / h)
        if box[2] > 0 and box[0] < 1 and box[3] > 0 and box[1] < 1:
            out.append(tuple(min(1.0, max(0.0, v)) for v in box))
    return tuple(out)  # type: ignore[return-value]


def face_origin(faces: Sequence[Box], cw: float, ch: float,
                focus: tuple[float, float] = (0.5, 0.5)) -> Optional[tuple[float, float]]:
    """Top left (x, y) of a cw x ch window (0..1 of the picture) as near to centring on focus as holds every
    face box whole, or None when no window of that size can."""
    out = []
    for size, f, lo_k, hi_k in ((cw, focus[0], 0, 2), (ch, focus[1], 1, 3)):
        lo, hi = 0.0, max(0.0, 1.0 - size)
        for b in faces:
            lo, hi = max(lo, b[hi_k] - size), min(hi, b[lo_k])
        if lo > hi + 1e-6:
            return None
        out.append(min(max(f - size / 2, lo), hi))
    return out[0], out[1]


def cover_window(faces: Sequence[Box], src_aspect: float, dst_aspect: float,
                 focus: tuple[float, float] = (0.5, 0.5)) -> Optional[Box]:
    """The dst_aspect window (x, y, w, h, 0..1) that covers a src_aspect picture and keeps every face whole,
    or None when no window of that shape can."""
    cw, ch = (dst_aspect / src_aspect, 1.0) if src_aspect > dst_aspect else (1.0, src_aspect / dst_aspect)
    o = face_origin(faces, cw, ch, focus)
    return None if o is None else (o[0], o[1], cw, ch)


def _text(rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
    try:
        from ...judge.prefilter import ocr_engine
        result, _ = ocr_engine()(np.ascontiguousarray(rgb[..., ::-1]))
    except Exception:  # noqa: BLE001  no OCR: no text boxes
        return []
    out = []
    for pts, _txt, conf in result or []:
        if float(conf) < 0.5:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


def _saliency(img: np.ndarray) -> np.ndarray:
    """Frequency tuned saliency (Achanta et al. 2009): distance of each softly blurred pixel from the
    picture's mean colour, 0..1. Works on grey (2D) or colour (3D, Lab) input."""
    f = img.astype(np.float32)
    if f.ndim == 3:
        f = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), max(1.0, img.shape[1] / 160))
    mean = f.reshape(-1, f.shape[-1]).mean(axis=0) if f.ndim == 3 else f.mean()
    d = np.sqrt(((blur - mean) ** 2).sum(axis=-1)) if f.ndim == 3 else np.abs(blur - mean)
    return d / (d.max() + 1e-6)


def analyse(frame_rgb: np.ndarray, ocr: bool = True) -> Analysis:
    h, w = frame_rgb.shape[:2]
    if w != WORK_W:
        frame_rgb = cv2.resize(frame_rgb, (WORK_W, max(1, round(h * WORK_W / w))), interpolation=cv2.INTER_AREA)
        h, w = frame_rgb.shape[:2]
    grey = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = [(x / w, y / h, (x + fw) / w, (y + fh) / h) for x, y, fw, fh in _faces(grey)]
    text = [(a / w, b / h, c / w, d / h) for a, b, c, d in (_text(frame_rgb) if ocr else [])]
    gx, gy = cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1)
    mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 6)
    detail = np.clip(mag / (np.percentile(mag, 98) + 1e-6), 0, 1)
    return Analysis(faces, text, detail, _saliency(frame_rgb))


# ---------------------------------------------------------------- choice

def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float = 0.0) -> float:
    """Share of box b covered by box a (both x0,y0,x1,y1), b grown by pad."""
    bx0, by0, bx1, by1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    ix = max(0.0, min(a[2], bx1) - max(a[0], bx0))
    iy = max(0.0, min(a[3], by1) - max(a[1], by0))
    area = max(1e-9, (bx1 - bx0) * (by1 - by0))
    return ix * iy / area


def subject_core(sal: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Mask of the main subject: the most salient pixels. A core spread over more than SUBJECT_MAX of the
    picture is tried at a stricter level; still that wide, it is texture or light (a bright sky, a glacier),
    not one subject, and only the detail cost applies."""
    if sal is None or sal.max() <= 0:
        return None
    h, w = sal.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # centre prior: a documentary subject is framed toward the middle more often than at the edges
    prior = np.exp(-(((xx - w / 2) / (0.45 * w)) ** 2 + ((yy - h * 0.55) / (0.5 * h)) ** 2))
    sal = sal * (0.4 + 0.6 * prior)
    for level in (SUBJECT_CORE, 0.8):
        core = (sal >= level * sal.max()).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        if n <= 1:
            return None
        # the main subject is the strongest single blob, not every salient pixel in the picture
        best = max(range(1, n), key=lambda k: float(sal[lab == k].sum()))
        blob = lab == best
        if blob.mean() <= SUBJECT_MAX:
            return blob
    return None


def subject_heart(sal: Optional[np.ndarray]) -> Optional[tuple[float, float, float, float]]:
    """The middle of the main subject (0..1 box): its blob's bounding box shrunk to HEART around the blob's
    centre of mass. A panel may sit over the edge of a wide subject (a train across the frame) but never over
    its heart."""
    core = subject_core(sal)
    if core is None:
        return None
    h, w = core.shape
    ys, xs = np.nonzero(core)
    cx, cy = xs.mean() / w, ys.mean() / h
    bw, bh = (xs.max() + 1 - xs.min()) / w * HEART, (ys.max() + 1 - ys.min()) / h * HEART
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def choose(samples: Sequence[Analysis], size: tuple[float, float], visible: tuple[float, float, float, float],
           W: int, H: int, previous: str = "", allowed: Sequence[str] = POSITIONS) -> Placement:
    """The calmest clear position for a panel of size (w, h) px, or position None with the reason."""
    blocked: dict[str, str] = {}
    scores: dict[str, float] = {}
    for pos in allowed:
        x, y, w, h = rect_for(pos, size, visible, W)
        box = (x / W, y / H, (x + w) / W, (y + h) / H)
        why = ""
        cost = 0.0
        for a in samples:
            if any(_overlap(box, f, 0.02) > 0.0 for f in a.faces):
                why = "covers a face"
                break
            if any(_overlap(box, tb, 0.01) > 0.15 for tb in a.text):
                why = "covers text"
                break
            if any(_overlap(box, pb) > 0.0 for pb in a.protect):
                why = "covers the framed picture"
                break
            sal = a.saliency
            heart = subject_heart(sal)
            sh, sw = a.detail.shape
            ys, xs = slice(int(box[1] * sh), int(box[3] * sh)), slice(int(box[0] * sw), int(box[2] * sw))
            if heart is not None and _overlap(box, heart) > SUBJECT_OVERLAP:
                why = "covers the main subject"
                break
            cost += float(a.detail[ys, xs].mean()) + 0.5 * float(sal[ys, xs].mean() if sal is not None else 0)
        if why:
            blocked[pos] = why
            continue
        scores[pos] = cost / max(1, len(samples))
    if not scores:
        reasons = ", ".join(f"{p} {r}" for p, r in blocked.items())
        return Placement(None, None, f"no clear position ({reasons})", {})
    ranked = dict(scores)
    if previous in ranked and len(ranked) > 1:
        ranked[previous] += REPEAT_PENALTY
    best = min(ranked, key=ranked.get)
    calm = "calmest" if len(scores) > 1 else "only clear"
    extra = f"; {', '.join(f'{p} {r}' for p, r in blocked.items())}" if blocked else ""
    rep = "; differs from the previous panel" if previous and best != previous else (
        "; same as the previous panel, the only clear place" if previous == best else "")
    reason = f"{best}: {calm} of {len(scores)} clear positions (detail {scores[best]:.2f}){extra}{rep}"
    return Placement(best, rect_for(best, size, visible, W), reason, scores)


def place_panel(frames: Sequence[np.ndarray], size: tuple[float, float], visible: tuple[float, float, float, float],
                W: int, H: int, previous: str = "", ocr: bool = True) -> Placement:
    """Analyse sampled frames (RGB, any size, frame aspect) and choose a position."""
    samples = [analyse(f, ocr) for f in frames]
    return choose(samples, size, visible, W, H, previous)


def debug_view(frame: np.ndarray, a: Analysis, p: Placement, W: int, H: int) -> np.ndarray:
    """The frame with faces (red), text (blue) and the chosen panel (amber) drawn, for proofs."""
    out = cv2.resize(frame, (W // 2, H // 2)).copy()
    sx, sy = out.shape[1], out.shape[0]
    for col, boxes in (((230, 40, 40), a.faces), ((40, 90, 230), a.text)):
        for b in boxes:
            cv2.rectangle(out, (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy)), col, 2)
    if p.rect:
        x, y, w, h = p.rect
        cv2.rectangle(out, (int(x / 2), int(y / 2)), (int((x + w) / 2), int((y + h) / 2)), (232, 163, 61), 3)
    return out


__all__ = ["POSITIONS", "Analysis", "Placement", "analyse", "choose", "place_panel", "rect_for", "debug_view",
           "faces_of", "face_boxes", "faces_in_crop", "face_origin", "cover_window"]
