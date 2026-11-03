"""Frame and cut analysis for the style learner.

detect_shots(path) -> list[(start, end)] with PySceneDetect's ContentDetector.
sample_frames(path, times) -> list of RGB frames (numpy, downscaled).
detect_letterbox(frames) -> LetterboxResult: black bars are rows whose median
luminance is near black with low spread, consistent across most frames. A
row median ignores burned captions that sit inside the bar.
detect_captions(frames, bands) -> CaptionResult: text-like rows (dense, high
contrast horizontal edges with bright strokes) inside the bottom band (or
the bottom fifth when there is no letterbox) that recur in many frames.
duration_stats(values) -> DurationStats.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel

ANALYSIS_WIDTH = 1280
BAR_LUMA_MAX = 28.0  # row median luminance (0..255) counted as black
BAR_SPREAD_MAX = 14.0  # row luminance std for a black bar row (caption rows are excluded by the median)
BAR_FRAME_SHARE = 0.7  # a row must be a bar in this share of frames
COMMON_RATIOS = (2.76, 2.39, 2.35, 2.2, 2.0, 1.85)
TEXT_EDGE_DENSITY = 0.035  # share of a row's pixels that are strong edges for a text row
TEXT_FRAME_SHARE = 0.4  # captions must appear in this share of sampled frames
INTER_EM_PER_INK = 1.4  # Inter: median caption ink height (mixed case, outline included) is about 0.7 em


class DurationStats(BaseModel):
    count: int
    min: float
    max: float
    mean: float
    median: float
    cv: float  # coefficient of variation


class LetterboxResult(BaseModel):
    detected: bool
    top_px: int = 0
    bottom_px: int = 0
    ratio: Optional[float] = None  # active picture width / height, snapped to a common ratio when close
    frame_h: int = 0


class CaptionResult(BaseModel):
    detected: bool
    frame_share: float = 0.0
    center_y_px: float = 0.0  # in analysed frame pixels
    ink_height_px: float = 0.0
    band_height: float = 0.5  # vertical centre as a fraction of the bottom band from its top
    font_px_1080: int = 0  # approximate font size scaled to a 1080 px tall frame
    color: str = "#FFFFFF"


# ---------------------------------------------------------------- shots and frames

def probe(path: Path) -> tuple[int, int, float, float]:
    """(width, height, fps, duration_s) of a video."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return w, h, fps, frames / fps if fps else 0.0


def detect_shots(path: Path, threshold: float = 27.0, min_len_s: float = 0.5) -> list[tuple[float, float]]:
    """Shot spans in seconds. A video with no detected cut is one shot."""
    from scenedetect import ContentDetector, detect
    _, _, fps, duration = probe(path)
    scenes = detect(str(path), ContentDetector(threshold=threshold, min_scene_len=max(1, int(fps * min_len_s))))
    spans = [(round(float(s.seconds), 3), round(float(e.seconds), 3)) for s, e in scenes]
    return spans or [(0.0, round(duration, 3))]


def sample_frames(path: Path, times: list[float], width: int = ANALYSIS_WIDTH) -> list[np.ndarray]:
    """RGB frames at the given times, resized to `width` (aspect kept). Unreadable times are skipped."""
    cap = cv2.VideoCapture(str(path))
    out: list[np.ndarray] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            h = max(2, round(bgr.shape[0] * width / bgr.shape[1]))
            small = cv2.resize(bgr, (width, h), interpolation=cv2.INTER_AREA)
            out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return out


def duration_stats(values: list[float]) -> DurationStats:
    """Summary of shot lengths in seconds."""
    vals = [float(v) for v in values if v > 0] or [0.0]
    mean = statistics.fmean(vals)
    cv = statistics.pstdev(vals) / mean if mean > 0 else 0.0
    return DurationStats(count=len(vals), min=round(min(vals), 2), max=round(max(vals), 2), mean=round(mean, 2),
                         median=round(statistics.median(vals), 2), cv=round(cv, 3))


# ---------------------------------------------------------------- letterbox

def _luma(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)


def bar_rows(frames: list[np.ndarray]) -> np.ndarray:
    """Per row, the share of frames in which the row looks like a black bar."""
    votes = []
    for f in frames:
        g = _luma(f)
        med = np.median(g, axis=1)
        lo = np.percentile(g, 25, axis=1)
        spread = np.abs(med - lo) + np.minimum(g.std(axis=1), 3 * BAR_SPREAD_MAX) * 0.25
        votes.append((med <= BAR_LUMA_MAX) & (spread <= BAR_SPREAD_MAX))
    return np.mean(np.stack(votes), axis=0)


def _run_length(mask: np.ndarray) -> int:
    n = 0
    for v in mask:
        if not v:
            break
        n += 1
    return n


def snap_ratio(ratio: float, tolerance: float = 0.05) -> float:
    """Nearest common cinema ratio when within tolerance, else the measured ratio rounded."""
    best = min(COMMON_RATIOS, key=lambda r: abs(r - ratio))
    return best if abs(best - ratio) <= tolerance else round(ratio, 2)


def detect_letterbox(frames: list[np.ndarray]) -> LetterboxResult:
    """Symmetric black bars at top and bottom in most frames, each at least 3 percent of the height."""
    if not frames:
        return LetterboxResult(detected=False)
    h, w = frames[0].shape[:2]
    share = bar_rows(frames) >= BAR_FRAME_SHARE
    top, bottom = _run_length(share), _run_length(share[::-1])
    min_bar = 0.03 * h
    if top < min_bar or bottom < min_bar or abs(top - bottom) > 0.04 * h or top + bottom > 0.6 * h:
        return LetterboxResult(detected=False, top_px=top, bottom_px=bottom, frame_h=h)
    ratio = snap_ratio(w / max(1, h - top - bottom))
    return LetterboxResult(detected=True, top_px=top, bottom_px=bottom, ratio=ratio, frame_h=h)


# ---------------------------------------------------------------- captions

def caption_region(h: int, letterbox: LetterboxResult) -> tuple[int, int]:
    """(top, bottom) rows searched for captions: the bottom bar, else the bottom fifth."""
    if letterbox.detected and letterbox.bottom_px >= 0.05 * h:
        return h - letterbox.bottom_px, h
    return int(h * 0.8), h


def text_block(frame: np.ndarray, top: int, bottom: int) -> Optional[tuple[float, float, str]]:
    """(centre_y, ink_height, colour hex) of the main text-like row block in [top, bottom), or None."""
    g = _luma(frame[top:bottom])
    if g.shape[0] < 6:
        return None
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    strong = gx > 120.0
    bright = g > 170.0
    density = strong.mean(axis=1)
    rows = (density >= TEXT_EDGE_DENSITY) & (bright.mean(axis=1) >= 0.01)
    if rows.sum() < 4:
        return None
    idx = np.flatnonzero(rows)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
    run = max(runs, key=len)
    if len(run) < 4:
        return None
    cols = np.flatnonzero(strong[run[0]:run[-1] + 1].any(axis=0))
    w = g.shape[1]
    if cols.size == 0 or (cols[-1] - cols[0]) < 0.08 * w or (cols[-1] - cols[0]) > 0.98 * w:
        return None
    ink = frame[top + run[0]:top + run[-1] + 1][bright[run[0]:run[-1] + 1]]
    rgb = np.median(ink, axis=0) if ink.size else np.array([255, 255, 255])
    colour = "#{:02X}{:02X}{:02X}".format(*(int(v) for v in rgb[:3]))
    return top + (run[0] + run[-1]) / 2.0, float(run[-1] - run[0] + 1), colour


def _virtual_band(h: int, letterbox: LetterboxResult) -> tuple[float, float]:
    """(band_top, band_px) for band_height: the real bottom bar, or a virtual 2.39 bar."""
    if letterbox.detected and letterbox.bottom_px > 0:
        return float(h - letterbox.bottom_px), float(letterbox.bottom_px)
    band = h * 138.0 / 1080.0
    return h - band, band


def detect_captions(frames: list[np.ndarray], letterbox: LetterboxResult) -> CaptionResult:
    """Burned captions that recur in the caption region of many sampled frames."""
    if not frames:
        return CaptionResult(detected=False)
    h = frames[0].shape[0]
    top, bottom = caption_region(h, letterbox)
    hits = [b for b in (text_block(f, top, bottom) for f in frames) if b is not None]
    share = len(hits) / len(frames)
    if share < TEXT_FRAME_SHARE:
        return CaptionResult(detected=False, frame_share=round(share, 3))
    cy = float(np.median([b[0] for b in hits]))
    ink = float(np.median([b[1] for b in hits]))
    colours = [b[2] for b in hits]
    band_top, band_px = _virtual_band(h, letterbox)
    band_height = float(np.clip((cy - band_top) / band_px, 0.1, 0.9))
    font = int(round(ink * INTER_EM_PER_INK * 1080.0 / h))
    return CaptionResult(detected=True, frame_share=round(share, 3), center_y_px=round(cy, 1),
                         ink_height_px=round(ink, 1), band_height=round(band_height, 2),
                         font_px_1080=int(np.clip(font, 24, 120)), color=max(set(colours), key=colours.count))
