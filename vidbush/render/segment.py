"""Render one shot to an H.264 segment with an exact frame count.

render_segment(shot, profile, caption_lines, settings, out_dir, frame_range)
renders frames [frame_range[0], frame_range[1]) of the timeline to
out_dir/<render_hash>.mp4 and returns the path (reused when it exists).

Layers, bottom to top: visual (trimmed and cover scaled video, or a still
with sub-pixel Ken Burns; portrait sources sit whole on a blurred fill, HDR
sources are tone mapped to SDR bt709, interlaced sources deinterlace to
fields, short clips use the rest of their file before a blended slowdown), colour LUT (not on motion graphics), alpha
overlay clip, letterbox bars, burned captions, fades through black.

Stills move with an OpenCV affine warp in float64 on a source prepared at
cover size times the largest zoom, so the warp scale stays between
1 / zoom and 1 and motion is smooth with no integer rounding jitter.
"""
from __future__ import annotations

import functools
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from ..config import Settings
from ..models import Candidate, Shot, SourceKind, StyleProfile
from ..util import JsonLogger, NullLogger, ToolError, ffprobe, stable_hash
from . import ffmpeg_cmds as fc
from .cache import (caption_dict, content_fingerprint, is_still, kenburns_for, lines_in_window, resolve_media,
                    segment_file, segment_hash)

MAX_SLOWDOWN = 1.25
EASE_SHARE = 0.35  # share of smoothstep blended into linear Ken Burns progress
SLATE = "0x0d0d0d"
HOLD_WARN_S = 0.25  # a held last frame longer than this is logged and flagged held_frame
PORTRAIT_ASPECT = 1.2  # display aspect below this gets a blurred fill instead of a cover crop
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


def render_segment(shot: Shot, profile: StyleProfile, caption_lines: list[Any], settings: Settings,
                   out_dir: Path, frame_range: tuple[int, int], fade_out: bool = False,
                   log: Optional[JsonLogger] = None) -> Path:
    """Render one shot segment (no audio) and return out_dir/<hash>.mp4."""
    log = log or NullLogger()
    r = settings.render
    n = int(frame_range[1]) - int(frame_range[0])
    if n <= 0:
        raise ValueError(f"shot {shot.id} has no frames in {frame_range}")
    t0, t1 = frame_range[0] / r.fps, frame_range[1] / r.fps
    window = lines_in_window(caption_lines, t0, t1)
    out = segment_file(out_dir, segment_hash(shot, profile, window, settings, tuple(frame_range), fade_out))
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".tmp_") as tmp:
        inputs, base, frames = _visual(shot, profile, settings, n, out.parent, log)
        graph, extra = _graph(shot, profile, settings, base, window, (t0, t1), n, fade_out, Path(tmp), log)
        part = Path(tmp) / "part.mp4"
        stdin_flag = ["-nostdin"] if frames is None else []
        cmd = ["ffmpeg", "-y", "-v", "error", *stdin_flag, *inputs, *extra, "-filter_complex", graph, "-map", "[vout]", "-frames:v", str(n),
               *fc.encoder_args(settings), str(part)]
        _run(cmd, frames)
        os.replace(part, out)
    return out


# ---------------------------------------------------------------- filter graph

def _graph(shot: Shot, profile: StyleProfile, settings: Settings, base: str, window: list[Any],
           span: tuple[float, float], n: int, fade_out: bool, tmp: Path,
           log: JsonLogger) -> tuple[str, list[str]]:
    """Full filter graph [0:v] (+ [1:v] overlay) -> [vout], plus extra input args."""
    r = settings.render
    cand = shot.selected_candidate()
    graded = cand is not None and cand.source != SourceKind.motion
    lut = profile.colour.lut_path
    lut_f = fc.lut(resolve_media(lut, settings)) if (lut and graded and resolve_media(lut, settings).exists()) else ""
    if lut and graded and not lut_f:
        log.warn("colour LUT missing, rendering ungraded", lut=lut)
    nodes = [f"[0:v]{fc.chain(base, lut_f, fc.TO_YUV)}[base]"]
    extra: list[str] = []
    top = "[base]"
    ov = resolve_media(shot.overlay_path, settings) if shot.overlay_path else None
    if ov is not None and ov.exists():
        extra = ["-i", str(ov)]
        nodes.append(f"[1:v]setpts=PTS-STARTPTS,fps={r.fps},scale={r.width}:{r.height},format=yuva420p[ov]")
        nodes.append("[base][ov]overlay=0:0:eof_action=pass:format=yuv420[comp]")
        top = "[comp]"
    elif ov is not None:
        log.warn("overlay clip missing, rendering without it", shot=shot.id, overlay=str(ov))
    dur = n / r.fps
    tail = fc.chain(
        fc.letterbox(r.width, r.height, profile.letterbox.ratio) if profile.letterbox.enabled else "",
        _captions(window, profile, settings, span, tmp, log),
        fc.fades(dur, *_fade_lengths(shot, profile, fade_out)),
        "format=yuv420p")
    nodes.append(f"{top}{tail}[vout]")
    return ";".join(nodes), extra


def _fade_lengths(shot: Shot, profile: StyleProfile, fade_out: bool) -> tuple[float, float]:
    """(fade in, fade out) seconds; a fade through black is split across the cut."""
    d = profile.transitions.crossfade_duration
    fade_in = 0.0
    if shot.transition_in == "fade":
        fade_in = d if shot.index == 0 else d / 2.0
    return fade_in, (d / 2.0 if fade_out else 0.0)


def _captions(window: list[Any], profile: StyleProfile, settings: Settings, span: tuple[float, float],
              tmp: Path, log: JsonLogger) -> str:
    """ass filter for the caption lines in this window, or ''."""
    if not profile.captions.enabled or not window:
        return ""
    try:
        from .captions import CaptionLine, write_ass_window
    except ImportError:
        log.warn("captions module unavailable, segment rendered without captions")
        return ""
    lines = [ln if isinstance(ln, CaptionLine) else CaptionLine(**caption_dict(ln)) for ln in window]
    r = settings.render
    ass = write_ass_window(lines, profile, span[0], span[1], tmp / "window.ass", r.width, r.height)
    return fc.subtitles(Path(ass), settings.path("assets/fonts"))


# ---------------------------------------------------------------- visual sources

def _visual(shot: Shot, profile: StyleProfile, settings: Settings, n: int, seg_dir: Path,
            log: JsonLogger) -> tuple[list[str], str, Optional[Iterator[bytes]]]:
    """(input args, filter chain for input 0, raw frame iterator for stills or None)."""
    r = settings.render
    cand = shot.selected_candidate()
    path = resolve_media(cand.local_path, settings) if cand else None
    if cand is None or path is None or not path.is_file():
        log.warn("shot has no usable visual, rendering a dark slate", shot=shot.id,
                 path=str(path) if path else "")
        return ["-f", "lavfi", "-i", f"color=c={SLATE}:s={r.width}x{r.height}:r={r.fps}"], "setsar=1", None
    vis_h = visible_height(profile, r.width, r.height)
    if is_still(shot):
        kb = kenburns_for(shot, profile)
        img = _prepare_still(path, cand.crop, kb, settings, seg_dir / "_stills", vis_h)
        args = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{r.width}x{r.height}", "-r", str(r.fps),
                "-i", "pipe:0"]
        chain = "setsar=1"
        return args, chain, still_frames(img, kb, n, r.width, r.height)
    plan = footage_plan(cand, path, n / r.fps, r.fps)
    if plan.hold > HOLD_WARN_S:
        log.warn("footage shorter than its shot, last frame held", shot=shot.id, hold_s=round(plan.hold, 2))
    args, chain = _video_input(cand, path, plan, settings, vis_h)
    return args, chain, None


def visible_height(profile: StyleProfile, width: int, height: int) -> int:
    """Picture height between the letterbox bars (the full height when letterbox is off)."""
    if not profile.letterbox.enabled:
        return height
    return height - 2 * fc.bar_height(width, height, profile.letterbox.ratio)


@dataclass
class FootagePlan:
    """How a clip fills a shot: read window, slowdown, held tail, colour transfer and shape."""

    start: float
    read: float
    slow: float
    hold: float
    transfer: str
    aspect: float  # display aspect ratio after the candidate crop
    interlaced: bool


@functools.lru_cache(maxsize=512)
def _probe(path: str, size: int, mtime_ns: int) -> dict[str, Any]:
    return ffprobe(path)


def probe_cached(path: Path) -> dict[str, Any]:
    """ffprobe memoized by (path, size, mtime_ns)."""
    st = Path(path).stat()
    return _probe(str(path), st.st_size, st.st_mtime_ns)


def _sar(vs: dict[str, Any]) -> float:
    try:
        a, b = str(vs.get("sample_aspect_ratio", "1:1")).split(":")
        return float(a) / float(b) if float(a) > 0 and float(b) > 0 else 1.0
    except ValueError:
        return 1.0


def footage_plan(cand: Candidate, path: Path, dur: float, fps: int) -> FootagePlan:
    """Read window for a clip: the selected span, extended into the rest of the file when it is too short.

    Only a shortfall left after that is slowed (at most MAX_SLOWDOWN) and then held.
    """
    info = probe_cached(path)
    vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    total = float(info.get("format", {}).get("duration") or vs.get("duration") or 0.0)
    start = max(0.0, cand.in_point)
    if total and start >= total - 0.05:
        start = max(0.0, total - dur)
    avail = (total - start) if total else dur
    if cand.out_point > start and cand.out_point - start >= dur:
        avail = min(avail, cand.out_point - start)
    if total and avail < dur:  # still short at the end of the file: start earlier
        start = max(0.0, total - dur)
        avail = total - start
    avail = max(avail, 1.0 / fps)
    slow = 1.0 if avail >= dur else min(dur / avail, MAX_SLOWDOWN)
    w, h = float(vs.get("width") or 16), float(vs.get("height") or 9)
    crop = cand.crop if cand.crop and len(cand.crop) == 4 and cand.crop[2] > 0.01 and cand.crop[3] > 0.01 else None
    aspect = w * _sar(vs) / max(h, 1.0) * ((crop[2] / crop[3]) if crop else 1.0)
    field = str(vs.get("field_order", "progressive"))
    return FootagePlan(start=start, read=dur + 0.1 if avail >= dur else avail, slow=slow,
                       hold=max(0.0, dur - avail * slow), transfer=str(vs.get("color_transfer", "")),
                       aspect=aspect, interlaced=field not in ("progressive", "unknown", ""))


def tonemap_chain(transfer: str) -> str:
    """PQ or HLG to SDR bt709 (zscale + hable), or '' for SDR sources or when zscale is missing."""
    if transfer not in HDR_TRANSFERS or not (fc.has_filter("zscale") and fc.has_filter("tonemap")):
        return ""
    return (f"zscale=tin={transfer}:min=bt2020nc:pin=bt2020:t=linear:npl=100,format=gbrpf32le,"
            "zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def _video_input(cand: Candidate, path: Path, plan: FootagePlan, settings: Settings,
                 vis_h: int) -> tuple[list[str], str]:
    """Trim, tone map HDR, deinterlace to fields, blend-slow a short clip, hold its end, fill the frame."""
    r = settings.render
    vs = next((s for s in probe_cached(path).get("streams", []) if s.get("codec_type") == "video"), {})
    motion = cand.source == SourceKind.motion
    deint = "" if motion else "bwdif=mode=send_field:deint=interlaced"
    sar = vs.get("sample_aspect_ratio", "1:1")
    anamorphic = "scale=trunc(iw*sar/2)*2:ih,setsar=1" if sar not in ("1:1", "0:1", "N/A", "") else ""
    if plan.slow > 1.0001:
        timing = fc.chain(f"setpts=(PTS-STARTPTS)*{plan.slow:.6f}",
                          f"framerate=fps={r.fps}:interp_start=0:interp_end=255:scene=100")
    else:
        timing = f"setpts=PTS-STARTPTS,fps={r.fps}"
    pre = fc.chain(deint, "" if motion else tonemap_chain(plan.transfer), timing, anamorphic,
                   fc.crop_fraction(cand.crop))
    hold = "tpad=stop_mode=clone:stop=-1"
    if not motion and plan.aspect < PORTRAIT_ASPECT:
        fill = fc.blur_fill(r.width, r.height, vis_h)
        chain = f"{pre},{fill},{hold}" if pre else f"{fill},{hold}"
    else:
        chain = fc.chain(pre, fc.cover_scale(r.width, r.height), hold)
    return ["-ss", f"{plan.start:.3f}", "-t", f"{plan.read:.3f}", "-i", str(path)], chain


def to_8bit_rgb(im: Any) -> Any:
    """Pillow image as RGB; 16-bit, 32-bit int and float greyscale are scaled to 0..255 first."""
    from PIL import Image
    if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        a = np.asarray(im, dtype=np.float32)
        top = 65535.0 if im.mode.startswith("I;16") else float(a.max()) if a.size and a.max() > 255 else 255.0
        lo = 0.0 if im.mode.startswith("I;16") else min(0.0, float(a.min()) if a.size else 0.0)
        a = np.clip((a - lo) / max(top - lo, 1e-6) * 255.0, 0, 255)
        im = Image.fromarray(a.astype(np.uint8))
    if im.mode in ("RGBA", "LA", "P", "PA"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def blur_fill_image(im: Any, w: int, h: int, vis_h: int) -> Any:
    """Portrait still on a w x h canvas: blurred, darkened cover copy behind the whole image fitted to vis_h."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    bg = ImageOps.fit(im, (w, h), method=Image.BILINEAR, centering=(0.5, 0.5))
    bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(max(8, w // 48)))).enhance(0.75)
    fh = max(2, min(h, vis_h))
    fw = max(2, min(w, int(round(im.width * fh / max(im.height, 1)))))
    fg = im.resize((fw, fh), Image.LANCZOS)
    bg.paste(fg, ((w - fw) // 2, (h - fh) // 2))
    return bg


def _prepare_still(path: Path, crop: Optional[list[float]], kb: dict[str, float], settings: Settings,
                   cache_dir: Path, vis_h: Optional[int] = None) -> np.ndarray:
    """RGB array of the still: EXIF upright, cropped, cover resized (or blur filled when portrait) to
    frame size times max zoom."""
    from PIL import Image, ImageOps
    r = settings.render
    zmax = max(kb["zoom0"], kb["zoom1"], 1.0)
    w, h = int(np.ceil(r.width * zmax)), int(np.ceil(r.height * zmax))
    vis = int(np.ceil((vis_h or r.height) * zmax))
    key = stable_hash("still-2", content_fingerprint(path), crop, w, h, vis)
    cached = cache_dir / f"{key}.png"
    if cached.exists():
        return np.asarray(Image.open(cached).convert("RGB"))
    im = to_8bit_rgb(ImageOps.exif_transpose(Image.open(path)))
    if crop and len(crop) == 4 and 0.01 < crop[2] and 0.01 < crop[3]:
        x, y, cw, ch = crop
        im = im.crop((int(x * im.width), int(y * im.height), int((x + cw) * im.width), int((y + ch) * im.height)))
    if im.width / max(im.height, 1) < PORTRAIT_ASPECT:
        im = blur_fill_image(im, w, h, vis)
    else:
        im = ImageOps.fit(im, (w, h), method=Image.LANCZOS, centering=(0.5, 0.5))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(f".{threading.get_ident()}.png")
    im.save(tmp, "PNG", compress_level=1)
    os.replace(tmp, cached)
    return np.asarray(im)


def kb_progress(i: int, n: int) -> float:
    """Ken Burns progress 0..1 for frame i of n: mostly linear with gently eased ends."""
    t = i / max(n - 1, 1)
    return (1 - EASE_SHARE) * t + EASE_SHARE * (t * t * (3 - 2 * t))


def still_frames(img: np.ndarray, kb: dict[str, float], n: int, width: int, height: int) -> Iterator[bytes]:
    """Yield n rgb24 frames of width x height panning and zooming over img (float64 affine warp)."""
    import cv2
    src_h, src_w = img.shape[:2]
    z0, z1 = kb["zoom0"], kb["zoom1"]
    for i in range(n):
        p = kb_progress(i, n)
        z = z0 * (z1 / z0) ** p  # exponential zoom reads as constant speed
        cx = (kb["x0"] + (kb["x1"] - kb["x0"]) * p) * src_w
        cy = (kb["y0"] + (kb["y1"] - kb["y0"]) * p) * src_h
        vw, vh = src_w / z, src_h / z
        sx, sy = width / vw, height / vh
        m = np.array([[sx, 0.0, -(cx - vw / 2) * sx], [0.0, sy, -(cy - vh / 2) * sy]], dtype=np.float64)
        frame = cv2.warpAffine(img, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        yield frame.tobytes()


def _run(cmd: list[str], frames: Optional[Iterator[bytes]]) -> None:
    """Run ffmpeg, feeding raw frames to stdin when given; raise ToolError on failure."""
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if frames is not None else subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=err)
        if frames is not None and proc.stdin is not None:
            try:
                for f in frames:
                    proc.stdin.write(f)
            except BrokenPipeError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
        code = proc.wait()
        if code != 0:
            err.seek(0)
            raise ToolError(f"ffmpeg segment render failed ({code}): {err.read().decode(errors='replace')[-1500:]}")
