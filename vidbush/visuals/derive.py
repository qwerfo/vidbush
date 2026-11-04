"""Derived framings and varied in points for reused assets.

derive_crops: one approved still used by nearby shots (same asset within 3
shots) is shown as up to three framings, full, tight and detail. Crops are
16:9 windows [x, y, w, h] in 0..1 of the source, placed by a saliency map
(spectral residual plus edge density, weighted to the band that stays visible
under the letterbox) so tight and detail land on something interesting.
vary_repeated_video: a clip used by several shots gets a different in point
for each use.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..models import Candidate, Shot, ShotList, ShotType, SourceKind
from ..util import ToolError, media_duration

NEARBY = 3  # shots apart that still count as the same run
FRAME_ASPECT = 16 / 9
SCALES = {"tight": 0.68, "detail": 0.45}
MIN_CROP_PX = 960  # never crop a source below this many pixels wide
GRID = 32
VISIBLE_BAND = 803 / 1080  # share of crop height left visible by the 2.39:1 letterbox
CYCLE = ("full", "tight", "detail")
STILL_SHOT_TYPES = {ShotType.image, ShotType.image_overlay, ShotType.ai_image}

Crop = list[float]


# ---------------------------------------------------------------- saliency

def saliency_map(path: str | Path, width: int = 160) -> np.ndarray:
    """Saliency in 0..1 (rows x cols at `width` columns): spectral residual blended with edge density."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    h = max(8, round(img.shape[0] * width / img.shape[1]))
    a = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    spec = np.fft.fft2(a)
    log_amp = np.log(np.abs(spec) + 1e-6)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sr = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spec)))) ** 2
    sr = cv2.GaussianBlur(sr.astype(np.float32), (0, 0), width / 40)
    gx, gy = cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)
    edges = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), width / 32)
    return 0.6 * _norm(sr) + 0.4 * _norm(edges)


def _norm(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


def crop_size(img_w: int, img_h: int, scale: float) -> tuple[float, float]:
    """Normalised (w, h) of a 16:9 window at `scale` of the largest 16:9 window that fits the image."""
    aspect = img_w / max(img_h, 1)
    w_n, h_n = (FRAME_ASPECT / aspect, 1.0) if aspect >= FRAME_ASPECT else (1.0, aspect / FRAME_ASPECT)
    scale = max(scale, MIN_CROP_PX / max(img_w * w_n, 1))
    return min(1.0, w_n * scale), min(1.0, h_n * scale)


def ranked_crops(sal: np.ndarray, w: float, h: float, n: int = 3, max_iou: float = 0.35) -> list[Crop]:
    """Up to n distinct windows of size (w, h), best saliency first (inside the letterbox-visible band)."""
    rows, cols = sal.shape
    integral = cv2.integral(sal.astype(np.float64))
    cw, ch = max(1, round(w * cols)), max(1, round(h * rows))
    band = max(1, round(ch * VISIBLE_BAND))
    scored = []
    for x in np.linspace(0, 1 - w, GRID) if w < 1 else [0.0]:
        for y in np.linspace(0, 1 - h, GRID) if h < 1 else [0.0]:
            c0, r0 = round(x * cols), round(y * rows) + (ch - band) // 2
            r1, c1 = min(rows, r0 + band), min(cols, c0 + cw)
            total = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
            scored.append((total / max(1, (r1 - r0) * (c1 - c0)), [float(x), float(y), w, h]))
    scored.sort(key=lambda t: -t[0])
    out: list[Crop] = []
    for _, crop in scored:
        if all(iou(crop, o) <= max_iou for o in out):
            out.append([round(v, 4) for v in crop])
        if len(out) >= n:
            break
    return out


def iou(a: Crop, b: Crop) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def framing_crop(path: str, img_w: int, img_h: int, framing: str, rank: int = 0) -> Optional[Crop]:
    """Crop for one framing ("full" gives None); rank picks the next best distinct window for repeats."""
    if framing not in SCALES:
        return None
    w, h = crop_size(img_w, img_h, SCALES[framing])
    if w >= 0.95 and h >= 0.95:
        return None  # source too small to crop without visible softness
    crops = ranked_crops(saliency_map(path), w, h, n=rank + 1)
    return crops[min(rank, len(crops) - 1)] if crops else None


# ---------------------------------------------------------------- shot list passes

def _asset(c: Candidate) -> str:
    """Canonical asset identity (URL first: sourced files are copied into each shot's folder)."""
    return c.url or c.local_path or c.id


def _clip_key(c: Candidate) -> str:
    """Identity of one clip window: the same asset at the same in and out points."""
    return f"{_asset(c)}@{c.in_point:.2f}-{c.out_point:.2f}"


def _runs(shots: list[Shot], media: str, nearby: int = NEARBY) -> list[list[Shot]]:
    """Groups of shots selecting the same asset (same clip window for video), members within `nearby` shots."""
    by_asset: dict[str, list[Shot]] = defaultdict(list)
    for s in shots:
        c = s.selected_candidate()
        if c is not None and c.media == media and c.source != SourceKind.motion:
            by_asset[_clip_key(c) if media == "video" else _asset(c)].append(s)
    runs: list[list[Shot]] = []
    for members in by_asset.values():
        members.sort(key=lambda s: s.index)
        cur = [members[0]]
        for s in members[1:]:
            if s.index - cur[-1].index <= nearby:
                cur.append(s)
            else:
                runs.append(cur)
                cur = [s]
        runs.append(cur)
    return runs


def _replace_selected(shot: Shot, new: Candidate) -> None:
    shot.candidates = [new if c.id == shot.selected else c for c in shot.candidates]


def derive_crops(shotlist: ShotList, ctx: Any) -> int:
    """Set crops on reused stills (full, tight, detail) and on single stills framed tight or detail."""
    stills = [s for s in shotlist.shots if s.type in STILL_SHOT_TYPES]
    changed = 0
    for run in _runs(stills, "image"):
        cand = run[0].selected_candidate()
        if cand is None or not cand.local_path or not Path(cand.local_path).exists():
            continue
        planned = len(run) > 1 and all(s.framing == "full" for s in run)
        framings = [CYCLE[i % len(CYCLE)] if planned else s.framing for i, s in enumerate(run)]
        seen: dict[str, int] = defaultdict(int)
        img_w, img_h = _dims(cand)
        for shot, framing in zip(run, framings):
            try:
                crop = framing_crop(cand.local_path, img_w, img_h, framing, rank=seen[framing])
            except (ValueError, cv2.error) as e:
                ctx.log.warn("derived crop failed", shot=shot.id, error=str(e)[:200])
                crop = None
            seen[framing] += 1
            shot.framing = framing  # type: ignore[assignment]
            own = shot.selected_candidate()
            if own is not None and own.crop != crop:
                _replace_selected(shot, own.model_copy(update={"crop": crop}))
                changed += 1
    return changed


def _dims(c: Candidate) -> tuple[int, int]:
    if c.width and c.height:
        return c.width, c.height
    img = cv2.imread(c.local_path)
    return (img.shape[1], img.shape[0]) if img is not None else (1920, 1080)


def vary_repeated_video(shotlist: ShotList, ctx: Any) -> int:
    """Each later use of one clip window starts after the earlier uses, wrapping inside its own file.

    A shifted start that would repeat a segment of a neighbouring shot or of the
    shot's own scene (the judge's ban) is skipped for the next free one; with none
    free the shot keeps its selection.
    """
    changed = 0
    cache: dict[str, float] = {}
    for run in _runs(shotlist.shots, "video", nearby=len(shotlist.shots)):
        if len(run) < 2:
            continue
        first = run[0].selected_candidate()
        offset = (first.in_point if first else 0.0) + run[0].duration
        for shot in run[1:]:
            own = shot.selected_candidate()
            span = (_clip_length(own, cache) if own else 0.0) - shot.duration
            if own is None or span <= 0.5:
                continue
            moved = _free_shift(own, shot, offset, span, _segment_guard(shotlist.shots, shot))
            if moved is None:
                continue
            _replace_selected(shot, moved)
            offset = moved.in_point + shot.duration
            changed += 1
    return changed


def _wrap(offset: float, span: float) -> float:
    return round(offset if offset <= span + 1e-6 else offset % span, 3)


def _free_shift(own: Candidate, shot: Shot, offset: float, span: float,
                guard: list[Candidate]) -> Optional[Candidate]:
    """own moved to the first start from offset (stepping a shot length, wrapped) clear of every guard segment."""
    from ..judge.director_loop import SEGMENT_GAP_S, same_segment
    step = max(shot.duration, SEGMENT_GAP_S)
    for k in range(int(span // step) + 2):
        start = _wrap(offset + k * step, span)
        moved = own.model_copy(update={"in_point": start, "out_point": round(start + shot.duration, 3)})
        if not any(same_segment(moved, g) for g in guard):
            return moved
    return None


def _segment_guard(shots: list[Shot], shot: Shot) -> list[Candidate]:
    """Selections shot must not repeat: the shots directly before and after, and the rest of its scene."""
    picks = [(i, s) for i, s in enumerate(shots) if s.type != ShotType.motion_graphic and s.selected_candidate()]
    here = next(i for i, s in enumerate(shots) if s is shot)
    before = [s for i, s in picks if i < here][-1:]
    after = [s for i, s in picks if i > here][:1]
    same = [s for i, s in picks if i != here and s.scene_id == shot.scene_id]
    return [s.selected_candidate() for s in before + after + same]  # type: ignore[misc]


def _clip_length(c: Candidate, cache: dict[str, float]) -> float:
    """Length of the candidate's own file (probed once per path)."""
    key = c.local_path or _asset(c)
    if key not in cache:
        length = 0.0
        if c.local_path and Path(c.local_path).exists():
            try:
                length = media_duration(c.local_path)
            except (ToolError, OSError, ValueError):
                length = 0.0
        cache[key] = length or max(c.out_point, 0.0)
    return cache[key]
