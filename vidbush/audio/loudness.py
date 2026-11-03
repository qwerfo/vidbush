"""Loudness measurement and two pass EBU R128 normalization, plus PCM IO helpers.

measure(path) returns integrated loudness (LUFS), true peak (dBTP) and LRA
(BS.1770 via the FFmpeg ebur128 filter, which is faster than a loudnorm
analysis pass and uses the same algorithm). normalize(...) is two pass: measure,
then a linear gain to the target, which is exactly what loudnorm does in its
linear mode. When that gain would push the true peak over the ceiling, a lookahead
limiter (delay compensated) shaves the peaks instead of loudnorm's dynamic mode,
so speech and music keep their dynamics. The result is verified and corrected
once if it missed by more than 0.3 LU.
"""
from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..util import run

SR = 48000


# ---------------------------------------------------------------- pcm io

def read_audio(path: Path | str, sr: int = SR, channels: int = 1) -> np.ndarray:
    """Decode any audio file to float32 samples of shape (n, channels) at sr."""
    cp = run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", str(channels), "-ar", str(sr),
              "-f", "f32le", "-"], timeout=1800)
    data = np.frombuffer(cp.stdout, dtype=np.float32)
    return data.reshape(-1, channels).copy()


def write_audio(path: Path | str, data: np.ndarray, sr: int = SR, codec: str = "pcm_s16le") -> Path:
    """Write float samples (n,) or (n, ch) to a file via FFmpeg (wav by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(arr.shape[1]), "-i", "-",
         "-c:a", codec, str(path)], timeout=1800, input_bytes=np.ascontiguousarray(arr).tobytes())
    return path


def db_to_amp(db: float) -> float:
    return float(10 ** (db / 20.0))


def rms_db(x: np.ndarray) -> float:
    """RMS level in dBFS of a sample block (very quiet blocks clamp at -120)."""
    v = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20 * math.log10(v) if v > 1e-6 else -120.0


# ---------------------------------------------------------------- measurement

def _summary_value(text: str, section: str, key: str, default: float) -> float:
    m = re.search(section + r":.*?" + key + r":\s+(-?inf|-?[\d.]+)", text, re.S)
    if not m or "inf" in m.group(1):
        return default
    return float(m.group(1))


def _parse_ebur128(stderr: str) -> dict[str, Any]:
    idx = stderr.rfind("Summary:")
    if idx < 0:
        raise RuntimeError("ebur128 produced no summary")
    text = stderr[idx:]
    lufs = _summary_value(text, "Integrated loudness", "I", -70.0)
    return {"lufs": lufs if lufs > -69.9 else -70.0,
            "threshold": _summary_value(text, "Integrated loudness", "Threshold", -70.0),
            "lra": _summary_value(text, "Loudness range", "LRA", 0.0),
            "true_peak": _summary_value(text, "True peak", "Peak", -70.0)}


def _file_graph(path: Path | str) -> tuple[list[str], str]:
    return ["-i", str(path)], "[0:a]anull[pre]"


def measure_graph(inputs: list[str], graph: str) -> dict[str, Any]:
    """Loudness of the [pre] output of a filter graph: {lufs, true_peak, lra, threshold}."""
    cp = run(["ffmpeg", "-hide_banner", "-nostats", *inputs, "-filter_complex",
              f"{graph};[pre]ebur128=peak=true:framelog=quiet[m]", "-map", "[m]", "-f", "null", "-"], timeout=7200)
    return _parse_ebur128(cp.stderr.decode(errors="replace"))


def measure(path: Path | str) -> dict[str, Any]:
    """Measure integrated LUFS, true peak (dBTP) and LRA of an audio file."""
    return measure_graph(*_file_graph(path))


# ---------------------------------------------------------------- normalization

def normalize(src: Path | str, dst: Path | str, lufs: float, tp: float = -1.0, sample_rate: int = SR,
              channels: int = 1, exact_samples: Optional[int] = None) -> dict[str, Any]:
    """Two pass normalization of an audio file into dst (PCM wav). Returns the measured loudness of dst."""
    inputs, graph = _file_graph(src)
    return normalize_graph(inputs, graph, dst, lufs, tp, sample_rate, channels, exact_samples)


def gain_chain(gain_db: float, peak_db: float, tp: float) -> str:
    """Linear gain, plus a delay compensated limiter when the gained true peak would pass tp."""
    chain = f"volume={gain_db:.2f}dB"
    if peak_db + gain_db > tp - 0.1:
        limit = max(db_to_amp(tp - 0.6), 0.0625)
        chain += f",alimiter=limit={limit:.4f}:attack=3:release=60:level=0:latency=1"
    return chain


def normalize_graph(inputs: list[str], graph: str, dst: Path | str, lufs: float, tp: float = -1.0,
                    sample_rate: int = SR, channels: int = 1, exact_samples: Optional[int] = None) -> dict[str, Any]:
    """Two pass normalization of a filter graph's [pre] output into dst, with no intermediate file.

    exact_samples pads or trims the output to exactly that many samples. Silent
    input is passed through resampled. Returns the measured loudness of dst.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    first = measure_graph(inputs, graph)
    tail = _length_filter(exact_samples)
    if first["lufs"] <= -69.0:
        _render(inputs, f"{graph};[pre]aresample={sample_rate}{tail}[out]", dst, sample_rate, channels)
        return measure(dst)
    chain = gain_chain(lufs - first["lufs"], first["true_peak"], tp)
    _render(inputs, f"{graph};[pre]{chain},aresample={sample_rate}{tail}[out]", dst, sample_rate, channels)
    got = measure(dst)
    err = lufs - got["lufs"]
    if abs(err) > 0.3 and got["lufs"] > -69.0:
        tmp = dst.with_name(dst.stem + "_pre" + dst.suffix)
        shutil.move(str(dst), str(tmp))
        _render(["-i", str(tmp)], f"[0:a]{gain_chain(err, got['true_peak'], tp)}{tail}[out]", dst,
                sample_rate, channels)
        tmp.unlink(missing_ok=True)
        got = measure(dst)
    return got


def _length_filter(exact_samples: Optional[int]) -> str:
    if not exact_samples:
        return ""
    return f",apad=whole_len={int(exact_samples)},atrim=end_sample={int(exact_samples)}"


def _render(inputs: list[str], graph: str, dst: Path, sample_rate: int, channels: int) -> None:
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]", "-ar", str(sample_rate),
         "-ac", str(channels), "-c:a", "pcm_s16le", str(dst)], timeout=7200)
