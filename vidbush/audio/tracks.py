"""Measure a music folder and write its tracks.json, so mood matching needs no hand tagging.

analyse_track(path) decodes a track at 22050 Hz mono and measures it:

- tempo: spectral flux onset strength, detrended, autocorrelated. Every candidate tempo in
  50..160 bpm is scored by the mean normalized autocorrelation at 1..8 beat periods (a comb),
  so a pulse that only sounds every 2 or 4 beats still lands on its beat inside the range
  (octave correction) and the long lags make the estimate precise to a fraction of a bpm.
  bpm_confidence is 3 * the best comb score, clipped to 0..1; below MIN_CONFIDENCE the track has
  no clear pulse and bpm, beat_s, bar_s, phrase_s are 0.
- first_downbeat_s: phase of the strongest onset when the flux is folded onto one bar.
- energy = 0.5 * loud + 0.5 * dense, loud = clip((lufs + 40) / 30), dense = clip(onsets_per_s / 4).
- brightness = clip(log2(centroid_hz / 250) / 4): 250 Hz is 0, 4000 Hz is 1.
- loopable: the last and first 3 s (about a bar, so a downbeat on one side does not count)
  have the same level (within 4 dB), are not faded (within 6 dB of the median 12 ms level)
  and have the same spectral shape (cosine of mean log spectra >= 0.9).
- mood: tempo nudges energy, e = energy + (bpm - 90) / 200 (no pulse: no nudge), then
  e < 0.5 and brightness < 0.5: mysterious; e < 0.5: calm;
  e >= 0.5 and brightness >= 0.5 and onsets_per_s < 3: uplifting; otherwise tense.

build_tracks_json(folder) merges the measurements into tracks.json: a field named in an entry's
"checked" list is Adam's and never overwritten; title, licence, notes and any other field that is
not measured are kept.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .loudness import measure, read_audio
from .music import AUDIO_EXT

SR = 22050


def file_fingerprint(path: Path | str) -> str:
    """Name, size and a hash of the first MB: unchanged by a checkout or a copy (mtime would be), so a fresh
    clone does not rewrite tracks.json."""
    p = Path(path)
    if not p.exists():
        return "missing"
    with p.open("rb") as f:
        head = hashlib.sha256(f.read(1 << 20)).hexdigest()[:16]
    return f"{p.name}:{p.stat().st_size}:{head}"


N_FFT = 1024
HOP = 256
FPS = SR / HOP
BPM_LO, BPM_HI = 50.0, 160.0
MIN_CONFIDENCE = 0.15
MEASURED_BY = "vidbush.audio.tracks"
# tracks.json field: analyse_track key (loop is what music.load_library reads)
MEASURED = {"mood": "mood", "energy": "energy", "bpm": "bpm", "bpm_confidence": "bpm_confidence",
            "beat_s": "beat_s", "bar_s": "bar_s", "phrase_s": "phrase_s", "first_downbeat_s": "first_downbeat_s",
            "duration_s": "duration_s", "lufs": "lufs", "lra": "lra", "brightness": "brightness",
            "centroid_hz": "centroid_hz", "onsets_per_s": "onsets_per_s", "loop": "loopable"}
HELP = ("Measured by vidbush.audio.tracks (python -m vidbush.audio.tracks <folder>). To correct a value, "
        "set it and add the field name to that track's \"checked\" list, e.g. \"checked\": [\"mood\"]. "
        "Checked fields are never overwritten on a re-run; other measured fields are refreshed when the "
        "file changes. title, licence and notes are never touched. mood is one of calm, tense, uplifting, "
        "mysterious; loop true means the end runs seamlessly into the start.")

_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _spectrum(x: np.ndarray) -> np.ndarray:
    """Magnitude STFT, frames centered on t * HOP: shape (frames, bins)."""
    x = np.pad(x, (N_FFT // 2, N_FFT // 2))
    if len(x) < N_FFT + HOP:
        x = np.pad(x, (0, N_FFT + HOP - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    return np.abs(np.fft.rfft(frames * np.hanning(N_FFT).astype(np.float32), axis=1)).astype(np.float32)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, min(int(n), len(x)))
    return np.convolve(x, np.ones(n) / n, mode="same")


_BAND = np.digitize(np.fft.rfftfreq(N_FFT, 1 / SR), np.geomspace(30, 11000, 41))


def _flux(mag: np.ndarray) -> np.ndarray:
    """Onset strength: 40 log spaced bands, log level rise over 2 frames, each band in units of its
    own std, counted only above 1.5 std, so steady noise and slow pad swells add almost nothing."""
    bands = np.zeros((mag.shape[0], 42), np.float32)
    np.add.at(bands.T, _BAND, mag.T)
    lm = np.log(bands[:, 1:41] + 1e-4 * float(bands.mean()) + 1e-12)
    d = lm[2:] - lm[:-2]
    d = d / (d.std(0) + 1e-9)
    return np.concatenate([[0.0, 0.0], np.clip(d - 1.5, 0, None).sum(1)])


def _tempo(flux: np.ndarray) -> tuple[float, float]:
    """(bpm, confidence) from the detrended onset strength."""
    x = flux - _smooth(flux, FPS * 1.5)
    n = len(x)
    if n < FPS * 4 or not np.any(x):
        return 0.0, 0.0
    spec = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    ac = ac / (n - np.arange(n))  # unbiased
    ac = ac / ac[0] if ac[0] > 0 else ac
    bpms = np.arange(BPM_LO, BPM_HI + 0.01, 0.1)
    periods = FPS * 60.0 / bpms
    ks = np.arange(1, 9)
    lags = periods[:, None] * ks[None, :]
    valid = lags < n * 0.5
    vals = np.where(valid, np.interp(lags, np.arange(n), ac), 0.0)
    score = vals.sum(1) / np.maximum(valid.sum(1), 1)
    i = int(np.argmax(score))
    return float(bpms[i]), float(np.clip(3.0 * score[i], 0.0, 1.0))  # a clean click track scores about 0.75


def _downbeat(flux: np.ndarray, bar_s: float) -> float:
    """Offset in seconds (0 <= t < bar_s) of the strongest onset phase within a bar."""
    pos = np.clip(flux - _smooth(flux, FPS * 1.5), 0, None)
    bar_f = bar_s * FPS
    nbins = int(round(bar_f))
    phase = np.arange(nbins) * bar_f / nbins
    starts = np.arange(0, len(pos) - bar_f, bar_f)
    if not len(starts) or nbins < 4:
        return 0.0
    folded = np.interp(starts[:, None] + phase[None, :], np.arange(len(pos)), pos).sum(0)
    folded = np.convolve(np.concatenate([folded[-2:], folded, folded[:2]]), np.ones(3) / 3, "same")[2:-2]
    return float(phase[int(np.argmax(folded))] / FPS)


def _onset_rate(flux: np.ndarray, duration: float) -> float:
    """Onsets per second: local flux maxima (+-50 ms) well above the local level."""
    if duration <= 0 or not np.any(flux):
        return 0.0
    base = _smooth(flux, FPS)
    z = flux - base
    thr = 1.5 * float(np.std(z)) + 0.05 * float(np.max(flux))
    w = max(1, int(0.05 * FPS))
    padded = np.pad(flux, w, constant_values=-np.inf)
    local_max = flux >= np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1).max(1)
    return float(np.sum(local_max & (z > thr)) / duration)


def _loopable(x: np.ndarray, mag: np.ndarray) -> bool:
    seg = int(3.0 * SR)
    if len(x) < 4 * seg:
        return False
    db = lambda v: 20 * math.log10(float(np.sqrt(np.mean(np.square(v, dtype=np.float64)))) + 1e-9)
    head, tail = db(x[:seg]), db(x[-seg:])
    frame_db = 20 * np.log10(np.sqrt(np.mean(np.square(x[: len(x) // HOP * HOP].reshape(-1, HOP)), 1)) + 1e-9)
    median = float(np.median(frame_db))
    nseg = int(3.0 * FPS)
    a, b = np.log1p(mag[:nseg].mean(0)), np.log1p(mag[-nseg:].mean(0))
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return abs(head - tail) <= 4.0 and min(head, tail) >= median - 6.0 and cos >= 0.9


def _mood(energy: float, brightness: float, bpm: float, onsets_per_s: float) -> str:
    e = energy + ((bpm - 90.0) / 200.0 if bpm else 0.0)
    if e < 0.5:
        return "mysterious" if brightness < 0.5 else "calm"
    return "uplifting" if brightness >= 0.5 and onsets_per_s < 3.0 else "tense"


def analyse_track(path: Path | str) -> dict[str, Any]:
    """Measure one track (see the module docstring for every formula). Cached per file fingerprint."""
    path = Path(path)
    key = (str(path.resolve()), file_fingerprint(path))
    if key in _CACHE:
        return dict(_CACHE[key])
    x = read_audio(path, SR, 1)[:, 0]
    duration = len(x) / SR
    mag = _spectrum(x)
    flux = _flux(mag)
    bpm, conf = _tempo(flux)
    if conf < MIN_CONFIDENCE:
        bpm = 0.0
    beat = 60.0 / bpm if bpm else 0.0
    loud = measure(path)
    lufs = float(loud.get("lufs", -70.0))
    weights = mag.sum(1)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    centroid = float((mag @ freqs).sum() / (weights.sum() + 1e-12))
    brightness = float(np.clip(math.log2(max(centroid, 1.0) / 250.0) / 4.0, 0, 1))
    rate = _onset_rate(flux, duration)
    energy = 0.5 * float(np.clip((lufs + 40) / 30, 0, 1)) + 0.5 * float(np.clip(rate / 4, 0, 1))
    out = {"duration_s": round(duration, 2), "lufs": round(lufs, 1), "lra": round(float(loud.get("lra", 0.0)), 1),
           "bpm": round(bpm, 1), "bpm_confidence": round(conf, 2), "beat_s": round(beat, 4),
           "bar_s": round(4 * beat, 4), "phrase_s": round(16 * beat, 3),
           "first_downbeat_s": round(_downbeat(flux, 4 * beat), 3) if bpm else 0.0,
           "energy": round(energy, 2), "brightness": round(brightness, 2), "centroid_hz": round(centroid),
           "onsets_per_s": round(rate, 2), "loopable": _loopable(x, mag),
           "mood": _mood(energy, brightness, bpm, rate)}
    _CACHE[key] = out
    return dict(out)


def _audio_files(folder: Path) -> list[Path]:
    return [f for f in sorted(folder.iterdir()) if f.is_file() and f.suffix.lower() in AUDIO_EXT]


def _read(folder: Path) -> dict[str, Any]:
    p = folder / "tracks.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _fresh(entry: Any, f: Path) -> bool:
    return isinstance(entry, dict) and entry.get("measured_by") == MEASURED_BY \
        and entry.get("fingerprint") == file_fingerprint(f)


def build_tracks_json(folder: Path | str, write: bool = True) -> dict[str, Any]:
    """Measure every audio file in folder and merge into tracks.json (checked fields survive)."""
    folder = Path(folder)
    old = _read(folder)
    out: dict[str, Any] = {"_help": HELP}
    for f in _audio_files(folder):
        prev = old.get(f.name) if isinstance(old.get(f.name), dict) else {}
        if _fresh(prev, f):
            out[f.name] = prev
            continue
        m = analyse_track(f)
        checked = list(prev.get("checked", []))
        entry: dict[str, Any] = {"title": prev.get("title", f.stem.replace("_", " ").title())}
        for field, src in MEASURED.items():
            entry[field] = prev[field] if field in checked and field in prev else m[src]
        entry.update({k: v for k, v in prev.items() if k not in entry})
        entry.update({"checked": checked, "fingerprint": file_fingerprint(f), "measured_by": MEASURED_BY,
                      "measured_on": datetime.date.today().strftime("%d/%m/%Y")})
        out[f.name] = entry
    if write:
        (folder / "tracks.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def needs_refresh(folder: Path | str) -> bool:
    """True when tracks.json is missing, or a file is new, unmeasured or changed, or an entry's file is gone."""
    folder = Path(folder)
    if not (folder / "tracks.json").exists():
        return True
    meta = _read(folder)
    files = _audio_files(folder)
    if any(not _fresh(meta.get(f.name), f) for f in files):
        return True
    return bool(set(k for k in meta if not k.startswith("_")) - {f.name for f in files})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vidbush.audio.tracks", description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--dry-run", action="store_true", help="measure and print, do not write tracks.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    data = build_tracks_json(args.folder, write=not args.dry_run)
    for name, e in data.items():
        if name.startswith("_"):
            continue
        print(f"{name}: bpm {e['bpm']} (conf {e['bpm_confidence']}) phrase {e['phrase_s']}s "
              f"downbeat {e['first_downbeat_s']}s energy {e['energy']} brightness {e['brightness']} "
              f"({e['centroid_hz']} Hz) onsets/s {e['onsets_per_s']} mood {e['mood']} loop {e['loop']} "
              f"lufs {e['lufs']} checked {e['checked']}")
    print(f"{len(data) - 1} tracks in {time.time() - t0:.1f}s{' (dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
