"""What every designed layout shares: the job it renders, its pictures, the house ground and helpers.

A LayoutJob is everything one render needs: the spec, one Cue per element (times relative to the clip),
the resolved pictures, the clip length, the output frame and the frame profile (letterbox and captions,
which decide the area type may use). A LayoutComp draws one frame at a time at its own size: the whole
frame for a full layout, the panel's size for a panel (vidbush/mg/layouts/panel.py). Type sizes go through
fs(), which never returns less than the house floor in output pixels, so text in a small panel is as
readable as text full screen.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from ...models import GENERATED_SOURCES, LayoutSpec
from .. import easing as ez
from ..house import DEFAULT, MIN_LABEL_PX, SETTLE_S, House
from ..textdraw import Sprite, blit, coloured, fit_text, round_rect, text_block, text_sprite, with_shadow, _round_rect_mask
from .timing import Cue

LETTERBOX_RATIO = 2.39
CAPTION_BAND = 0.135  # share of frame height captions use at the bottom when there is no letterbox


@dataclass
class FrameProfile:
    """The output format a layout must fit: letterbox bars, burned captions."""

    letterbox: bool = True
    captions: bool = True
    ratio: float = LETTERBOX_RATIO

    def visible(self, W: int, H: int) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of the picture area not covered by bars or the caption band."""
        if self.letterbox:
            bar = round((H - W / self.ratio) / 2)
            return 0.0, float(bar), float(W), float(H - bar)
        bottom = H * (1 - CAPTION_BAND) if self.captions else float(H)
        return 0.0, 0.0, float(W), bottom


@dataclass
class PictureRef:
    """A real picture a layout shows: a still, or a clip read from in_point."""

    path: str
    media: str = "image"  # image | video
    in_point: float = 0.0
    source: str = ""
    title: str = ""
    width: int = 0
    height: int = 0

    @property
    def generated(self) -> bool:
        return self.source in {s.value for s in GENERATED_SOURCES}


@dataclass
class LayoutJob:
    spec: LayoutSpec
    cues: list[Cue]
    duration: float
    width: int = 1920
    height: int = 1080
    fps: int = 30
    frame: FrameProfile = field(default_factory=FrameProfile)
    house: House = DEFAULT
    pictures: dict[int, PictureRef] = field(default_factory=dict)  # element index -> picture
    extra: dict[str, Any] = field(default_factory=dict)  # kind specific inputs (map geometry, timeline history)
    notes: list[str] = field(default_factory=list)  # decisions a review can read
    missing: list[str] = field(default_factory=list)  # why it cannot be drawn whole: the stage then drops it

    def els(self, role: str) -> list[tuple[int, Any, Cue]]:
        """(index, element, cue) of every element with this role, in spoken order."""
        return [(i, e, self.cues[i]) for i, e in enumerate(self.spec.elements) if e.role == role]

    def first(self, role: str) -> Optional[tuple[int, Any, Cue]]:
        got = self.els(role)
        return got[0] if got else None


# ---------------------------------------------------------------- pictures

@lru_cache(maxsize=4)  # archival scans decode to hundreds of MB each
def _still(path: str) -> np.ndarray:
    from PIL import Image, ImageOps

    from ...render.segment import to_8bit_rgb
    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, 300_000_000)  # archival scans run to 140 megapixels
    im = Image.open(path)
    if max(im.size) > 16000:  # beyond any scan we read (OCR needs the full page)
        im.draft("RGB", (16000, 16000))
        im.thumbnail((16000, 16000))
    return np.asarray(to_8bit_rgb(ImageOps.exif_transpose(im)))


def picture_size(ref: PictureRef) -> tuple[int, int]:
    if ref.width and ref.height:
        return ref.width, ref.height
    if ref.media == "image":
        a = _still(ref.path)
        return a.shape[1], a.shape[0]
    return 1920, 1080


def cover(img: np.ndarray, w: int, h: int, focus: tuple[float, float] = (0.5, 0.5),
          faces: Sequence[tuple[float, float, float, float]] = ()) -> Optional[np.ndarray]:
    """img scaled to cover w x h and cropped around focus (0..1). With faces (0..1 boxes of img) the window
    keeps every face whole, and None is returned when no window of this shape can."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(math.ceil(iw * s))), max(h, int(math.ceil(ih * s)))
    x, y = focus[0] * nw - w / 2, focus[1] * nh - h / 2
    if faces:
        from .placement import face_origin
        o = face_origin(faces, w / nw, h / nh, focus)
        if o is None:
            return None
        x, y = o[0] * nw, o[1] * nh
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x, y = int(round(min(max(0, x), nw - w))), int(round(min(max(0, y), nh - h)))
    return r[y:y + h, x:x + w]


def contain(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """The whole picture fitted inside w x h on a blurred, darkened cover copy of itself: the framing of a
    picture that no cover window can show with every face whole."""
    out = darken(cover(img, w, h), 0.55, max(4.0, w / 60))
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    fw, fh = max(1, min(w, int(iw * s))), max(1, min(h, int(ih * s)))
    x, y = (w - fw) // 2, (h - fh) // 2
    out[y:y + fh, x:x + fw] = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return out


@lru_cache(maxsize=256)
def clip_faces(path: str, t: float) -> tuple[tuple[tuple[float, float, float, float], ...], float]:
    """(face boxes 0..1, width / height) of a clip's frame at t; ((), 0.0) when it cannot be read.
    shortcut: one frame stands for the clip, so a face that moves far later can still be cut; sample more
    frames if reviews show it."""
    from .placement import FACE_W, face_boxes
    try:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", path, "-frames:v", "1",
                            "-vf", f"scale={FACE_W}:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return (), 0.0
    grey = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_GRAYSCALE) if p.stdout else None
    if grey is None:
        return (), 0.0
    return face_boxes(grey), grey.shape[1] / grey.shape[0]


class Footage:
    """Frames of a picture at a fixed size over time: a still (with no motion of its own) or a clip read
    sequentially through ffmpeg from in_point (looped from in_point if it runs out). The cover crop keeps
    every face whole; when no crop of this shape can, the whole picture is shown (contained is True)."""

    def __init__(self, ref: PictureRef, w: int, h: int, fps: int) -> None:
        self.ref, self.w, self.h, self.fps = ref, int(w), int(h), fps
        self.proc: Optional[subprocess.Popen] = None
        self.last: Optional[np.ndarray] = None
        self.i = -1
        self.contained = False
        self.unreadable = False  # a clip that gave no frame: the house ground stands in, never black
        if ref.media == "image":
            from .placement import faces_of
            img = _still(ref.path)
            self.last = cover(img, self.w, self.h, faces=faces_of(ref.path))
            if self.last is None:
                self.contained, self.last = True, contain(img, self.w, self.h)
        else:
            self.vf = self._fill()

    def _fill(self) -> str:
        """ffmpeg filter giving w x h frames: the cover crop that keeps every face of the first frame whole,
        else the whole clip on the house ground."""
        from .placement import cover_window
        w, h = self.w, self.h
        faces, aspect = clip_faces(self.ref.path, self.ref.in_point)
        win = cover_window(faces, aspect, w / h) if aspect else (0.0, 0.0, 1.0, 1.0)
        if win is None:
            self.contained = True
            return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x{DEFAULT.ink.lstrip('#')}")
        x, y, cw, ch = win
        fx = x / (1 - cw) if cw < 0.999 else 0.5
        fy = y / (1 - ch) if ch < 0.999 else 0.5
        return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}:(iw-{w})*{fx:.4f}:(ih-{h})*{fy:.4f}"

    def _open(self) -> None:
        vf = f"{self.vf},fps={self.fps}"
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, self.ref.in_point):.3f}", "-i",
                                      self.ref.path, "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def frame(self, i: int) -> np.ndarray:
        if self.ref.media == "image":
            return self.last
        while self.i < i:
            if self.proc is None:
                self._open()
            n = self.w * self.h * 3
            buf = self.proc.stdout.read(n) if self.proc else b""
            if len(buf) < n:
                self.close()
                if self.last is None:
                    self.unreadable, self.last = True, ink_ground(self.w, self.h, DEFAULT).copy()
                self.i = i  # clip ran out: hold its last frame
                break
            self.last = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            self.i += 1
        return self.last

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdout.close()
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
            self.proc = None


def framed(img: np.ndarray, w: int, h: int, house: House, k: float, fit: bool = False) -> Sprite:
    """A picture in the house frame: cover (or fit) into w x h, thin paper border, rounded, soft shadow.
    Layout pictures arrive already cropped by Footage (faces kept whole), so this cover only sizes them."""
    w, h = max(8, int(w)), max(8, int(h))
    if fit:
        ih, iw = img.shape[:2]
        s = min(w / iw, h / ih)
        w, h = max(8, int(iw * s)), max(8, int(ih * s))
    pic = cover(img, w, h)
    b = max(2, int(round(house.frame_px * k)))
    r = max(2, int(round(house.radius_px * k)))
    W, H = w + 2 * b, h + 2 * b
    pad = int(30 * k) + 4
    out = np.zeros((H + 2 * pad, W + 2 * pad, 4), np.uint8)
    border = round_rect(W, H, r, house.rgb("paper"))
    blit(out, border, pad, pad)
    m = _round_rect_mask(w, h, max(1, r - b)).astype(np.float32)[..., None] / 255.0
    body = np.zeros((h, w, 4), np.uint8)
    body[..., :3] = (pic.astype(np.float32) * m).astype(np.uint8)
    body[..., 3] = (m[..., 0] * 255).astype(np.uint8)
    blit(out, body, pad + b, pad + b)
    out = with_shadow(out, radius=18 * k + 2, offset=max(1, int(8 * k)), opacity=0.55)
    return Sprite(out, pad)


# ---------------------------------------------------------------- composition base

class LayoutComp:
    """One layout drawn at W x H. Subclasses implement setup() and draw(frame, t)."""

    panel_size = (0.42, 0.42)  # (width, height) share of the frame when shown as a panel

    def __init__(self, job: LayoutJob, W: int, H: int, panel: bool = False) -> None:
        self.job, self.spec, self.house = job, job.spec, job.house
        self.W, self.H, self.panel = int(W), int(H), panel
        self.fps, self.dur = job.fps, job.duration
        self.OH = job.height  # output frame height: type floors are in output pixels
        self.k = min(self.W / 1920.0, self.H / 1080.0)  # design px (1080p full frame) to local px
        if panel:
            m = max(self.fs(26, 18), int(0.06 * min(self.W, self.H)))
            self.box = (float(m), float(m), float(self.W - m), float(self.H - m))
        else:
            x0, y0, x1, y1 = job.frame.visible(self.W, self.H)
            mx, my = 0.07 * self.W, 0.06 * (y1 - y0)
            self.box = (x0 + mx, y0 + my, x1 - mx, y1 - my)
        self._ground = self.make_ground()
        self.setup()

    # sizes and area
    def px(self, v: float) -> int:
        return int(round(v * self.k))

    def fs(self, base: float, floor: float = MIN_LABEL_PX) -> int:
        """Font size: base design px scaled to this comp, never under floor px of the output frame."""
        return int(round(max(base * self.k, floor * self.OH / 1080.0)))

    @property
    def bw(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def bh(self) -> float:
        return self.box[3] - self.box[1]

    # timing
    def cue(self, i: int) -> Cue:
        return self.job.cues[i]

    def settle(self, t: float, cue: Cue, length: float = SETTLE_S) -> float:
        """0..1 entry progress of an element (eased), starting at its cue."""
        return ez.ease_out_cubic(ez.progress(t, cue.appear, cue.appear + length))

    # ground
    def make_ground(self) -> Optional[np.ndarray]:
        """House ink ground: soft gradient, faint grain baked in. Subclasses that paint pictures return None."""
        return ink_ground(self.W, self.H, self.house)

    def canvas(self, i: int) -> np.ndarray:
        return self._ground.copy() if self._ground is not None else np.zeros((self.H, self.W, 3), np.uint8)

    # text
    def label(self, text: str, base: float, color: str = "paper", weight: str = "SemiBold", caps: bool = True,
              max_w: Optional[float] = None, lines: int = 1, floor: float = MIN_LABEL_PX,
              tracking_em: float = 0.12, shadow: float = 0.0, align: str = "left") -> Sprite:
        """A text sprite in house type that fits max_w (shrinking to the floor, then wrapping, then ellipsis)."""
        t = text.upper() if caps else text
        size = self.fs(base, floor)
        f, ln = fit_text(t, weight, size, max_w or self.bw, lines, self.fs(floor, floor), tracking_em if caps else 0.0)
        tr = (tracking_em if caps else 0.0) * f.size
        col = self.house.rgb(color)
        if len(ln) == 1:
            return text_sprite(ln[0], f, col, tr, shadow)
        return text_block(ln, f, col, align, 1.12, tr, shadow)

    def setup(self) -> None:
        pass

    def draw(self, frame: np.ndarray, t: float) -> None:
        raise NotImplementedError

    def render(self, i: int) -> np.ndarray:
        frame = self.canvas(i)
        self.draw(frame, i / self.fps)
        return frame

    def close(self) -> None:
        pass


@lru_cache(maxsize=8)
def _ground(w: int, h: int, ink: str, ink2: str, grain: float) -> np.ndarray:
    from ..palettes import hex_rgb
    from ..textdraw import texture, vignette_mask
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.clip(np.sqrt(((xx - w * 0.38) / w) ** 2 + ((yy - h * 0.35) / h) ** 2 * 0.9) / 0.75, 0, 1)[..., None]
    c0, c1 = np.array(hex_rgb(ink2), np.float32), np.array(hex_rgb(ink), np.float32)
    img = c0 * (1 - d) + c1 * d
    img *= 1.0 + texture(w, h, 11, 3.0)[..., None] * 0.06
    img += np.random.default_rng(5).normal(0, grain, (h, w, 1)).astype(np.float32)
    img *= vignette_mask(w, h, 0.35)[..., None] / 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def ink_ground(w: int, h: int, house: House) -> np.ndarray:
    return _ground(int(w), int(h), house.ink, house.ink2, float(house.grain))


def darken(img: np.ndarray, keep: float, blur: float = 0.0) -> np.ndarray:
    """Footage dimmed to `keep` brightness (and softly blurred) so type over it reads."""
    out = cv2.GaussianBlur(img, (0, 0), blur) if blur > 0 else img
    return cv2.convertScaleAbs(out, alpha=keep)


def drift(frame: np.ndarray, t: float, dur: float, amount: float) -> np.ndarray:
    """The slow push every full layout makes over its hold, so nothing on screen is static."""
    if amount <= 0:
        return frame
    p = ez.ease_in_out_sine(ez.clamp01(t / max(dur, 1e-3)))
    s = 1.0 + amount * p
    h, w = frame.shape[:2]
    m = np.array([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h * 0.48]], np.float64)
    return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def premul(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """RGB plus an alpha mask (float 0..1) as a premultiplied RGBA array."""
    a = np.clip(alpha, 0, 1)[..., None]
    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = (rgb.astype(np.float32) * a).astype(np.uint8)
    out[..., 3] = (a[..., 0] * 255).astype(np.uint8)
    return out


__all__ = ["FrameProfile", "PictureRef", "LayoutJob", "LayoutComp", "Footage", "framed", "cover", "contain", "darken",
           "drift", "ink_ground", "premul", "picture_size", "coloured"]
