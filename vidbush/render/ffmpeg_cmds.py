"""FFmpeg command and filter builders shared by segment rendering and assembly.

Every segment, intro and outro is encoded with encoder_args(), so the concat
demuxer can join them with stream copy. Filter helpers return filter strings
(no labels); callers join them with commas into a graph.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..config import Settings
from ..util import run

COLOUR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
AUDIO_RATE = 48000
TO_YUV = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"  # explicit bt709 RGB to YUV


def escape_filter_value(value: str) -> str:
    """Escape a path or text for use as a quoted filter option value."""
    v = str(value).replace("\\", "/")
    return "'" + v.replace("'", r"'\''").replace(":", r"\:") + "'"


def encoder_args(settings: Settings, threads: int = 2) -> list[str]:
    """H.264 high, yuv420p, constant fps, fixed GOP: identical for every concat member."""
    r = settings.render
    return ["-c:v", "libx264", "-preset", r.preset, "-crf", str(r.crf), "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-r", str(r.fps), "-g", str(r.fps * 2), "-keyint_min", str(r.fps),
            "-bf", "2", "-threads", str(threads), *COLOUR_TAGS,
            "-video_track_timescale", str(r.fps * 512), "-an"]


def bar_height(width: int, height: int, ratio: float) -> int:
    """Letterbox bar height in px for a picture of this aspect ratio, 0 when it is not wider."""
    if ratio <= 0:
        return 0
    return max(0, int(round((height - width / ratio) / 2)))


def cover_scale(width: int, height: int) -> str:
    """Scale to cover width x height without stretching, then centre crop; bt709 output."""
    return (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos"
            f":out_color_matrix=bt709:out_range=tv,crop={width}:{height},setsar=1")


def fit_pad(width: int, height: int) -> str:
    """Scale to fit inside width x height and pad with black (intros and outros keep their framing)."""
    return (f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos"
            f":out_color_matrix=bt709:out_range=tv,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")


def crop_fraction(crop: Optional[list[float]]) -> str:
    """Crop filter for a [x, y, w, h] crop in 0..1 of the source, or '' when no crop applies."""
    if not crop or len(crop) != 4:
        return ""
    x, y, w, h = (max(0.0, min(1.0, float(v))) for v in crop)
    if w <= 0.01 or h <= 0.01 or (w >= 0.999 and h >= 0.999):
        return ""
    return f"crop=w=iw*{w:.5f}:h=ih*{h:.5f}:x=iw*{x:.5f}:y=ih*{y:.5f}"


def letterbox(width: int, height: int, ratio: float) -> str:
    """Two solid black drawbox bars, or '' when the ratio leaves no bar."""
    bar = bar_height(width, height, ratio)
    if bar <= 0:
        return ""
    return (f"drawbox=x=0:y=0:w=iw:h={bar}:color=black@1:t=fill,"
            f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black@1:t=fill")


def fades(duration: float, fade_in: float, fade_out: float) -> str:
    """Fade from and to black, each clamped to half the segment."""
    parts = []
    half = duration / 2.0
    if fade_in > 0:
        parts.append(f"fade=t=in:st=0:d={min(fade_in, half):.4f}")
    if fade_out > 0:
        d = min(fade_out, half)
        parts.append(f"fade=t=out:st={duration - d:.4f}:d={d:.4f}")
    return ",".join(parts)


def subtitles(ass_path: Path, fonts_dir: Path) -> str:
    return f"ass=filename={escape_filter_value(str(ass_path))}:fontsdir={escape_filter_value(str(fonts_dir))}"


def lut(lut_path: Path) -> str:
    return f"lut3d=file={escape_filter_value(str(lut_path))}:interp=tetrahedral"


def chain(*parts: str) -> str:
    """Join non-empty filters with commas."""
    return ",".join(p for p in parts if p)


def write_concat_list(paths: list[Path], list_path: Path) -> Path:
    """Concat demuxer list file with absolute, quoted paths."""
    lines = ["ffconcat version 1.0"]
    for p in paths:
        safe = str(Path(p).resolve()).replace("'", r"'\''")
        lines.append(f"file '{safe}'")
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def loudnorm_measure(audio: Path, lufs: float, tp: float) -> dict[str, str]:
    """First loudnorm pass: measured input values as reported by ffmpeg."""
    cp = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio), "-af",
              f"loudnorm=I={lufs}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"], timeout=1800)
    text = cp.stderr.decode(errors="replace")
    blocks = re.findall(r"\{[^{}]*\}", text)
    if not blocks:
        raise RuntimeError("loudnorm produced no measurement")
    return json.loads(blocks[-1])


def loudnorm_apply(audio: Path, out: Path, lufs: float, tp: float, duration: float) -> Path:
    """Two pass loudnorm to lufs and true peak tp; output 48 kHz stereo wav of exactly duration seconds."""
    m = loudnorm_measure(audio, lufs, tp)
    af = (f"loudnorm=I={lufs}:TP={tp}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
          f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}"
          f":linear=true,aresample={AUDIO_RATE},{pad_trim(duration)}")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-af", af, "-ac", "2", "-ar", str(AUDIO_RATE),
         "-c:a", "pcm_s16le", str(out)], timeout=1800)
    return out


def pad_trim(duration: float) -> str:
    """Audio filter forcing an exact duration (pad with silence, then trim)."""
    return f"apad=whole_dur={duration:.6f},atrim=end={duration:.6f}"
