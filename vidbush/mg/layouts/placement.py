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

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import numpy as np

POSITIONS = ("upper_left", "upper_right", "lower_left", "lower_right", "left_centre", "right_centre")
WORK_W = 480
REPEAT_PENALTY = 0.12
SUBJECT_CORE = 0.6  # saliency share of the peak that counts as the subject's core
SUBJECT_OVERLAP = 0.12  # a panel may cover at most this share of the subject core
MARGIN = 0.035  # gap between the panel and the picture edge, share of frame width


@dataclass
class Analysis:
    faces: list[tuple[float, float, float, float]] = field(default_factory=list)  # 0..1 boxes x0,y0,x1,y1
    text: list[tuple[float, float, float, float]] = field(default_factory=list)
    detail: Optional[np.ndarray] = None  # float map 0..1 at work size
    saliency: Optional[np.ndarray] = None


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

def _faces(grey: np.ndarray) -> list[tuple[int, int, int, int]]:
    try:
        from ...judge.prefilter import face_cascade
        casc = face_cascade()
    except Exception:  # noqa: BLE001  no cascade: no face boxes
        return []
    h = grey.shape[0]
    found = casc.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(max(12, h // 25),) * 2)
    out = [tuple(int(v) for v in b) for b in (found if len(found) else [])]
    try:  # profile faces too: a face turned sideways is still a face
        prof = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        if not prof.empty():
            for flip in (False, True):
                g = cv2.flip(grey, 1) if flip else grey
                for x, y, w, hh in prof.detectMultiScale(g, 1.1, 6, minSize=(max(12, h // 25),) * 2):
                    out.append((grey.shape[1] - x - w if flip else x, y, w, hh))
    except Exception:  # noqa: BLE001
        pass
    return out


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


def _saliency(grey: np.ndarray) -> np.ndarray:
    """Spectral residual saliency (Hou and Zhang 2007), 0..1."""
    small = cv2.resize(grey, (64, 64 * grey.shape[0] // max(1, grey.shape[1]) or 36)).astype(np.float32)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-6)
    resid = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(resid + 1j * np.angle(f)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2.5)
    sal = cv2.resize(sal, (grey.shape[1], grey.shape[0]))
    return sal / (sal.max() + 1e-6)


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
    return Analysis(faces, text, detail, _saliency(grey))


# ---------------------------------------------------------------- choice

def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float = 0.0) -> float:
    """Share of box b covered by box a (both x0,y0,x1,y1), b grown by pad."""
    bx0, by0, bx1, by1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    ix = max(0.0, min(a[2], bx1) - max(a[0], bx0))
    iy = max(0.0, min(a[3], by1) - max(a[1], by0))
    area = max(1e-9, (bx1 - bx0) * (by1 - by0))
    return ix * iy / area


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
            sal = a.saliency
            core = sal >= SUBJECT_CORE * sal.max() if sal is not None else None
            sh, sw = a.detail.shape
            ys, xs = slice(int(box[1] * sh), int(box[3] * sh)), slice(int(box[0] * sw), int(box[2] * sw))
            if core is not None and core.sum() > 0 and core[ys, xs].sum() / core.sum() > SUBJECT_OVERLAP:
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


__all__ = ["POSITIONS", "Analysis", "Placement", "analyse", "choose", "place_panel", "rect_for", "debug_view"]
