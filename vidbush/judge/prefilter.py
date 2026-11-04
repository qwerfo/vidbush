"""Local pre-filter: cheap keyframe checks that hard reject a candidate before any vision call.

prefilter_candidate(cand, cfg) -> reject reason or None (pure, no side effects on cand).
apply_prefilter(cands, settings, log) -> rejected candidates; writes score 0.0, judge_reason
and reject_reason ("<label> (prefilter: ...)") so the director loop treats them as hard rejects
and never sends them to the vision judge.

Checks (each switchable under settings.judge.prefilter):
  text  burned-in subtitles or lower-third text read by OCR (rapidocr, weights ship in the wheel)
  logo  a textured, pixel-stable island in a corner while the rest of the clip moves (video only)
  face  a large centred frontal face on most keyframes, a talking head (video only; a still
        portrait is a normal documentary subject)
Every rule needs evidence repeated across keyframes, so a single still never trips text or logo.
Operator media (local library, uploads) and generated media are trusted and skipped.
A missing dependency skips its check with one warning per process, never an error.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..config import PrefilterConfig
from ..models import Candidate, SourceKind

WORK_WIDTH = 640  # frames are downscaled to this width before OCR and the face cascade
LOGO_WIDTH = 320  # the logo check works on smaller frames
OCR_MARGIN = 0.08  # OCR reads from this share of height above the band top, so straddling lines are seen
LOGO_EDGE_SHARE = 0.06  # a logo box must hold at least this share of edge pixels (flat bars and sky fail)
LOGO_CONTRAST = 12.0  # and this grey level spread
LOGO_STABLE_SHARE = 0.5  # at least this share of the box stays within logo_max_diff on every keyframe
LOGO_RING_MOVING = 0.5  # at least this share of the ring around the box must change between keyframes
STATIC_SHARE = 0.75  # when this share of the frame centre is stable the clip is static, no logo test
PREFILTER_WORKERS = 4
LABELS = ("burned_in_captions", "channel_logo", "talking_head")
# operator media (library folder, uploads) and generated media are not third party footage: never pre-filtered
TRUSTED_SOURCES = {SourceKind.local, SourceKind.upload, SourceKind.ai_image, SourceKind.ai_video, SourceKind.motion}

_LOCK = threading.Lock()
_OCR: list[Any] = []  # the lazily built engine; empty until first use
_LOCAL = threading.local()  # one cascade per thread: CascadeClassifier is not thread safe
_WARNED: set[str] = set()
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


# ---------------------------------------------------------------- dependencies (seams for tests)

def _cv2() -> Any:
    import cv2
    return cv2


def ocr_engine() -> Any:
    """RapidOCR built once per process behind a lock; the weights ship inside the wheel."""
    with _LOCK:
        if not _OCR:
            from rapidocr_onnxruntime import RapidOCR
            try:
                _OCR.append(RapidOCR(det_model_path=None, det_limit_side_len=WORK_WIDTH, det_limit_type="max",
                                     use_angle_cls=False))
            except (TypeError, KeyError):  # another wheel version with other option names
                _OCR.append(RapidOCR())
        return _OCR[0]


def face_cascade() -> Any:
    """Frontal face haar cascade from cv2.data, one per thread."""
    casc = getattr(_LOCAL, "face", None)
    if casc is None:
        cv2 = _cv2()
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if casc.empty():
            raise ImportError("haarcascade_frontalface_default.xml not found")
        _LOCAL.face = casc
    return casc


def _warn_once(log: Any, check: str, err: Exception) -> None:
    with _LOCK:
        if check in _WARNED:
            return
        _WARNED.add(check)
    if log is not None:
        log.warn("prefilter check skipped, dependency missing", check=check, error=str(err)[:200])


# ---------------------------------------------------------------- frames

def load_frames(paths: list[str], width: int = WORK_WIDTH) -> list[np.ndarray]:
    """BGR frames downscaled to width (never upscaled); unreadable files are dropped."""
    cv2 = _cv2()
    out = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR) if p and Path(p).exists() else None
        if im is None:
            continue
        h, w = im.shape[:2]
        if w > width:
            im = cv2.resize(im, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
        out.append(im)
    return out


# ---------------------------------------------------------------- text

def subtitle_lines(boxes: list[tuple[Box, float]], w: int, h: int, cfg: PrefilterConfig) -> list[Box]:
    """Lines of confident text in the band whose union spans text_min_width of the frame width."""
    band = [b for b, conf in boxes if conf >= cfg.text_min_confidence and (b[1] + b[3]) / 2 >= cfg.text_band_top * h]
    lines: list[list[float]] = []
    for x0, y0, x1, y1 in sorted(band, key=lambda b: (b[1] + b[3]) / 2):
        cy = (y0 + y1) / 2
        for ln in lines:
            if ln[1] <= cy <= ln[3]:  # same line: centre inside an existing line's vertical span
                ln[0], ln[2] = min(ln[0], x0), max(ln[2], x1)
                break
        else:
            lines.append([x0, y0, x1, y1])
    return [tuple(ln) for ln in lines if ln[2] - ln[0] >= cfg.text_min_width * w]  # type: ignore[misc]


def read_text(frame: np.ndarray, cfg: PrefilterConfig) -> list[tuple[Box, float]]:
    """OCR boxes (full frame pixel coordinates) and confidences for the lower part of the frame."""
    h = frame.shape[0]
    top = int(max(0.0, cfg.text_band_top - OCR_MARGIN) * h)
    result, _ = ocr_engine()(np.ascontiguousarray(frame[top:]))
    out = []
    for pts, _text, conf in result or []:
        xs, ys = [p[0] for p in pts], [p[1] + top for p in pts]
        out.append(((min(xs), min(ys), max(xs), max(ys)), float(conf)))
    return out


def text_reject(frames: list[np.ndarray], cfg: PrefilterConfig) -> Optional[str]:
    """Subtitle or lower-third text on text_min_keyframes or more keyframes."""
    need = max(1, cfg.text_min_keyframes)
    if len(frames) < need:
        return None
    hits = 0
    for i, f in enumerate(frames):
        if subtitle_lines(read_text(f, cfg), f.shape[1], f.shape[0], cfg):
            hits += 1
        elif hits + len(frames) - i - 1 < need:
            return None  # cannot reach the count any more, skip the remaining OCR calls
    if hits < need:
        return None
    return f"burned_in_captions (prefilter: subtitle text on {hits} of {len(frames)} keyframes)"


# ---------------------------------------------------------------- logo and watermark

def _stability(frames: list[np.ndarray]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(per pixel max abs grey difference to the first keyframe, median grey frame), None on size mismatch."""
    cv2 = _cv2()
    small = [cv2.resize(f, (LOGO_WIDTH, max(1, round(f.shape[0] * LOGO_WIDTH / f.shape[1]))),
                        interpolation=cv2.INTER_AREA) for f in frames]
    if len({s.shape for s in small}) != 1:
        return None
    greys = [cv2.GaussianBlur(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32) for s in small]
    diff = np.max([np.abs(g - greys[0]) for g in greys[1:]], axis=0)
    return diff, np.median(np.stack(greys), axis=0).astype(np.uint8)


def _island(box: tuple[int, int, int, int], stable: np.ndarray, edges: np.ndarray, grey: np.ndarray,
            cfg: PrefilterConfig) -> bool:
    """A logo box: right size, textured, mostly stable, and ringed by pixels that change."""
    x, y, bw, bh = box
    h, w = stable.shape
    if not cfg.logo_min_area <= (bw * bh) / float(w * h) <= cfg.logo_max_area:
        return False
    inner = (slice(y, y + bh), slice(x, x + bw))
    if edges[inner].mean() < LOGO_EDGE_SHARE or float(grey[inner].std()) < LOGO_CONTRAST:
        return False  # flat corners: black bars, sky, dark vignettes
    if float(stable[inner].mean()) < LOGO_STABLE_SHARE:
        return False
    m = max(3, round(0.25 * min(bw, bh)))
    ring = np.zeros_like(stable)
    ring[max(0, y - m):y + bh + m, max(0, x - m):x + bw + m] = True
    ring[inner] = False
    return bool(ring.any()) and float((~stable[ring]).mean()) >= LOGO_RING_MOVING


def _corner_boxes(stable: np.ndarray, near: np.ndarray, x0: int, y0: int, cw: int, ch: int) -> list[tuple[int, ...]]:
    """Boxes of stable textured blobs inside one corner that do not touch the sides facing the centre."""
    cv2 = _cv2()
    h, w = stable.shape
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y0 + ch, x0:x0 + cw] = (near & stable)[y0:y0 + ch, x0:x0 + cw]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for x, y, bw, bh, _area in stats[1:n]:
        touches_x = x + bw >= x0 + cw if x0 == 0 else x <= x0
        touches_y = y + bh >= y0 + ch if y0 == 0 else y <= y0
        if not (touches_x or touches_y):  # a blob reaching the inner sides is part of the scene
            out.append((int(x), int(y), int(bw), int(bh)))
    return out


def logo_reject(frames: list[np.ndarray], cfg: PrefilterConfig) -> Optional[str]:
    """A textured island in a corner that stays pixel identical on every keyframe while the clip moves."""
    got = _stability(frames) if len(frames) >= 2 else None
    if got is None:
        return None
    cv2 = _cv2()
    diff, grey = got
    stable = diff <= cfg.logo_max_diff
    h, w = stable.shape
    if stable[h // 4:3 * h // 4, w // 4:3 * w // 4].mean() >= STATIC_SHARE:
        return None  # a static clip: everything is stable, a corner says nothing
    edges = (cv2.Canny(grey, 50, 150) > 0) & stable
    near = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    cw, ch = max(4, int(cfg.logo_corner * w)), max(4, int(cfg.logo_corner * h))
    corners = (("top left", 0, 0), ("top right", w - cw, 0), ("bottom left", 0, h - ch),
               ("bottom right", w - cw, h - ch))
    for name, x0, y0 in corners:
        for box in _corner_boxes(stable, near, x0, y0, cw, ch):
            if _island(box, stable, edges, grey, cfg):
                share = 100.0 * box[2] * box[3] / (w * h)
                return f"channel_logo (prefilter: static {name} mark, {share:.1f}% of frame, on all keyframes)"
    return None


# ---------------------------------------------------------------- face to camera

def large_faces(frame: np.ndarray, cfg: PrefilterConfig) -> list[tuple[int, int, int, int]]:
    """Frontal face boxes (x, y, w, h); faces below face_min_area of the frame are not searched for."""
    cv2 = _cv2()
    h, w = frame.shape[:2]
    grey = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    side = max(24, int((cfg.face_min_area * w * h) ** 0.5))
    found = face_cascade().detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(side, side))
    return [(int(a), int(b), int(c), int(d)) for a, b, c, d in found]


def _centred(box: tuple[int, int, int, int], w: int, h: int, cfg: PrefilterConfig) -> bool:
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    big = bw * bh >= cfg.face_min_area * w * h
    return big and abs(cx - w / 2) <= cfg.face_centre * w / 2 and abs(cy - h / 2) <= cfg.face_centre * h / 2


def face_reject(frames: list[np.ndarray], cfg: PrefilterConfig) -> Optional[str]:
    """A large centred frontal face on more than face_min_share of the keyframes: a talking head."""
    if not frames:
        return None
    hits = sum(1 for f in frames if any(_centred(b, f.shape[1], f.shape[0], cfg) for b in large_faces(f, cfg)))
    if hits / len(frames) <= cfg.face_min_share:
        return None
    return f"talking_head (prefilter: large centred face on {hits} of {len(frames)} keyframes)"


# ---------------------------------------------------------------- entry points

Check = Callable[[list[np.ndarray], PrefilterConfig], Optional[str]]


def _checks(cand: Candidate, cfg: PrefilterConfig) -> list[tuple[str, Check]]:
    """Enabled checks for this candidate, cheapest first. Logo and face need a moving clip."""
    out: list[tuple[str, Check]] = []
    if cand.source in TRUSTED_SOURCES:
        return out
    video = cand.media == "video"
    if cfg.logo and video:
        out.append(("logo", logo_reject))
    if cfg.face and video:
        out.append(("face", face_reject))
    if cfg.text and len(cand.keyframes) >= max(1, cfg.text_min_keyframes):
        out.append(("text", text_reject))
    return out


def prefilter_candidate(cand: Candidate, cfg: PrefilterConfig, log: Any = None) -> Optional[str]:
    """First reject reason from the enabled checks, or None. A missing dependency skips its check."""
    checks = _checks(cand, cfg) if cfg.enabled else []
    try:
        frames = load_frames(cand.keyframes) if checks else []
    except ImportError as e:  # no OpenCV: every check needs it
        _warn_once(log, "frames", e)
        return None
    for name, check in checks if frames else []:
        try:
            reason = check(frames, cfg)
        except ImportError as e:
            _warn_once(log, name, e)
            continue
        if reason:
            return reason
    return None


def apply_prefilter(cands: list[Candidate], settings: Any, log: Any = None) -> list[Candidate]:
    """Pre-filter fresh candidates; a reject gets score 0.0, judge_reason and reject_reason. Returns rejects."""
    cfg = getattr(getattr(settings, "judge", None), "prefilter", None) or PrefilterConfig()
    todo = [c for c in cands if c.score is None and c.reject_reason is None]
    if not cfg.enabled or not todo:
        return []

    def one(c: Candidate) -> Optional[str]:
        try:
            return prefilter_candidate(c, cfg, log)
        except Exception as e:  # noqa: BLE001  a broken frame must never sink the stage; the judge still runs
            if log is not None:
                log.warn("prefilter failed for candidate", candidate=c.id, error=str(e)[:200])
            return None

    with ThreadPoolExecutor(max_workers=max(1, min(PREFILTER_WORKERS, len(todo)))) as pool:
        reasons = list(pool.map(one, todo))
    rejected = []
    for c, reason in zip(todo, reasons):
        if reason:
            c.score, c.reject_reason = 0.0, reason
            c.judge_reason = reason[reason.find("(") + 1:-1] if "(" in reason else reason
            rejected.append(c)
    if log is not None:
        counts = {lab: sum(1 for c in rejected if (c.reject_reason or "").startswith(lab)) for lab in LABELS}
        log.info("prefilter", checked=len(todo), rejected=len(rejected), **counts)
    return rejected
