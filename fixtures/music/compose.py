"""Compose the bundled CC0 documentary underscore tracks with numpy.

Run: python fixtures/music/compose.py   (writes *.ogg, tracks.json, LICENSE.txt here)

Every layer is rendered circularly onto the loop length (notes that ring past
the end wrap to the start, noise is filtered with a circular FFT and the reverb
is a circular convolution), so each track loops seamlessly.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 44100
HERE = Path(__file__).resolve().parent


@dataclass
class Piece:
    name: str
    title: str
    mood: str
    energy: float
    bpm: float
    bars_per_chord: int
    cycles: int
    chords: list[list[int]]  # MIDI voicings, lowest note is the bass root
    pulse: list[float]  # pulse hits in beats within one bar (4/4)
    pluck: str  # none | sparse | arp | bells
    brightness: float  # pad harmonic rolloff
    noise_band: tuple[float, float]
    noise_gain: float
    tremolo: float = 0.0
    seed: int = 1
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def bar_s(self) -> float:
        return 4 * 60.0 / self.bpm

    @property
    def chord_s(self) -> float:
        return self.bar_s * self.bars_per_chord

    @property
    def length(self) -> int:
        return int(round(self.chord_s * len(self.chords) * self.cycles * SR))


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def add_wrapped(buf: np.ndarray, start: int, sig: np.ndarray) -> None:
    """Add a stereo signal into the loop buffer, wrapping past the end."""
    n, pos = len(buf), 0
    while pos < len(sig):
        a = (start + pos) % n
        take = min(n - a, len(sig) - pos)
        buf[a:a + take] += sig[pos:pos + take]
        pos += take


def env_pad(n: int, attack: int, release: int) -> np.ndarray:
    e = np.ones(n)
    a = min(attack, n)
    e[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    r = min(release, n)
    e[n - r:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r))
    return e


def pad_note(freq: float, n: int, bright: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Three detuned additive voices, harmonics rolled off by a time varying brightness."""
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for cents, pan in ((-6.0, 0.15), (0.0, 0.5), (6.0, 0.85)):
        f = freq * 2 ** (cents / 1200)
        voice = np.zeros(n)
        for k in range(1, 11):
            if f * k > 7000:
                break
            amp = np.exp(-(k - 1) / bright) / k
            voice += amp * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 2 * np.pi))
        out[:, 0] += voice * np.cos(pan * np.pi / 2)
        out[:, 1] += voice * np.sin(pan * np.pi / 2)
    return out / 3.0


def render_pads(p: Piece, rng: np.random.Generator) -> np.ndarray:
    buf = np.zeros((p.length, 2))
    total = p.length / SR
    seq = p.chords * p.cycles
    release = 2.6
    for i, chord in enumerate(seq):
        t0 = i * p.chord_s
        n = int((p.chord_s + release) * SR)
        tt = t0 + np.arange(n) / SR
        bright = p.brightness * (1.0 + 0.45 * np.sin(2 * np.pi * tt / total + 0.4) + 0.15 * np.sin(2 * np.pi * tt / 9.7))
        e = env_pad(n, int(1.6 * SR), int(release * SR))
        if p.tremolo:
            e = e * (1 - p.tremolo * 0.5 * (1 + np.sin(2 * np.pi * 5.2 * tt)))
        sig = sum(pad_note(hz(m), n, bright, rng) * (0.8 if j == 0 else 1.0) for j, m in enumerate(chord[1:]))
        add_wrapped(buf, int(t0 * SR), sig * e[:, None] / max(len(chord) - 1, 1))
    return buf


def render_pulse(p: Piece) -> np.ndarray:
    """Soft low sine pulse on the chord root (one octave down, kept above 36 Hz)."""
    buf = np.zeros((p.length, 2))
    beat = 60.0 / p.bpm
    n = int(1.4 * SR)
    t = np.arange(n) / SR
    env = (1 - np.exp(-t / 0.012)) * np.exp(-t / 0.42)
    for i, chord in enumerate(p.chords * p.cycles):
        root = chord[0] - 12
        while hz(root) < 36:
            root += 12
        f = hz(root)
        tone = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * env
        for bar in range(p.bars_per_chord):
            for b in p.pulse:
                start = i * p.chord_s + bar * p.bar_s + b * beat
                add_wrapped(buf, int(start * SR), np.stack([tone, tone], 1))
    return buf


def render_plucks(p: Piece, rng: np.random.Generator) -> np.ndarray:
    """Melodic top layer: sparse plucks, 8th note arpeggio, or inharmonic bells."""
    buf = np.zeros((p.length, 2))
    if p.pluck == "none":
        return buf
    beat = 60.0 / p.bpm
    step = beat / 2 if p.pluck == "arp" else beat
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    partials = [(1.0, 1.0), (2.76, 0.35), (5.4, 0.12)] if p.pluck == "bells" else [(1.0, 1.0), (2.0, 0.3), (3.0, 0.08)]
    decay = 1.8 if p.pluck == "bells" else 0.9
    seq = p.chords * p.cycles
    steps = int(round(p.chord_s / step))
    for i, chord in enumerate(seq):
        tones = sorted(set(m + 12 for m in chord[1:])) + [chord[2] + 24]
        for s in range(steps):
            if p.pluck == "arp":
                m = tones[(s * 2 + (s // 4)) % len(tones)]
                vel = 0.55 + 0.25 * (s % 4 == 0)
            elif rng.random() < (0.22 if p.pluck == "bells" else 0.3) and s % 2 == 0:
                m, vel = tones[rng.integers(len(tones))], rng.uniform(0.4, 0.8)
            else:
                continue
            f = hz(m)
            sig = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * r / decay) for r, a in partials)
            sig = sig * (1 - np.exp(-t / 0.004)) * vel
            pan = rng.uniform(0.25, 0.75)
            add_wrapped(buf, int((i * p.chord_s + s * step) * SR),
                        np.stack([sig * np.cos(pan * np.pi / 2), sig * np.sin(pan * np.pi / 2)], 1))
    return buf


def band_noise(p: Piece, rng: np.random.Generator) -> np.ndarray:
    """Circularly band filtered pink noise with a slow swell, decorrelated in stereo."""
    n = p.length
    out = np.zeros((n, 2))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    lo, hi = p.noise_band
    shape = np.where(freqs > 1, 1 / np.sqrt(np.maximum(freqs, 1)), 0)
    shape *= np.exp(-0.5 * (np.log(np.maximum(freqs, 1) / np.sqrt(lo * hi)) / np.log(hi / lo) * 2.2) ** 2)
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * np.arange(n) / n * 3 + 1.0)
    for ch in range(2):
        spec = np.fft.rfft(rng.standard_normal(n)) * shape
        sig = np.fft.irfft(spec, n)
        out[:, ch] = sig / (np.std(sig) + 1e-9) * swell
    return out


def reverb(x: np.ndarray, rng: np.random.Generator, seconds: float = 3.2, wet: float = 0.32) -> np.ndarray:
    """Circular convolution with a decaying, darkened noise impulse (one per channel)."""
    n = len(x)
    m = int(seconds * SR)
    t = np.arange(m) / SR
    out = np.zeros_like(x)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    dark = 1 / (1 + (freqs / 4500.0) ** 2)
    for ch in range(2):
        ir = np.zeros(n)
        ir[int(0.025 * SR): int(0.025 * SR) + m] = rng.standard_normal(m) * np.exp(-t * 6.9 / seconds)
        ir /= np.sqrt(np.sum(ir ** 2))
        wet_sig = np.fft.irfft(np.fft.rfft(x[:, ch]) * np.fft.rfft(ir) * dark, n)
        out[:, ch] = (1 - wet) * x[:, ch] + wet * wet_sig
    return out


def master(x: np.ndarray) -> np.ndarray:
    """High pass at 35 Hz, set RMS to about -20 dBFS, soft clip below -1 dBFS."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    hp = 1 / (1 + (35.0 / np.maximum(freqs, 1e-3)) ** 4)
    y = np.stack([np.fft.irfft(np.fft.rfft(x[:, c]) * hp, n) for c in range(2)], 1)
    y *= 10 ** (-20 / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9)
    ceiling = 10 ** (-1.5 / 20)
    return np.tanh(y / ceiling) * ceiling


def compose(p: Piece) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    pads = render_pads(p, rng)
    pulse = render_pulse(p)
    plucks = render_plucks(p, rng)
    noise = band_noise(p, rng)
    dry = (pads * 1.0 + pulse * p.extra.get("pulse_gain", 0.32) + plucks * p.extra.get("pluck_gain", 0.16)
           + noise * p.noise_gain)
    return master(reverb(dry, rng, wet=p.extra.get("wet", 0.32)))


PIECES = [
    Piece("harbor_light", "Harbor Light", "calm", 0.25, 68, 2, 4,
          [[38, 50, 57, 61, 64, 66], [35, 47, 54, 62, 64, 69], [31, 43, 50, 59, 66, 69], [33, 45, 52, 61, 62, 64]],
          [0.0, 2.0], "sparse", 2.2, (300, 2400), 0.018, seed=11),
    Piece("undertow", "Undertow", "tense", 0.6, 76, 2, 4,
          [[36, 48, 55, 62, 63, 67], [32, 44, 51, 60, 63, 67], [29, 41, 48, 56, 60, 67], [31, 43, 50, 55, 60, 62]],
          [0.0, 0.35, 2.0, 2.35], "none", 1.6, (120, 900), 0.03, tremolo=0.25, seed=23,
          extra={"pulse_gain": 0.45}),
    Piece("first_light", "First Light", "uplifting", 0.75, 84, 2, 6,
          [[31, 43, 55, 59, 62, 67], [30, 42, 54, 57, 62, 66], [28, 40, 52, 59, 62, 67], [24, 36, 55, 60, 62, 64]],
          [0.0, 1.0, 2.0, 3.0], "arp", 2.6, (800, 6000), 0.012, seed=37, extra={"pluck_gain": 0.12}),
    Piece("deep_archive", "Deep Archive", "mysterious", 0.4, 60, 2, 3,
          [[40, 52, 59, 66, 67], [36, 48, 55, 59, 66], [33, 45, 52, 59, 60], [35, 47, 54, 57, 64]],
          [0.0], "bells", 1.8, (200, 1600), 0.035, seed=53, extra={"wet": 0.4}),
]


def encode(x: np.ndarray, path: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-c:a", "libvorbis", "-q:a", "3", str(path)],
                   input=x.astype(np.float32).tobytes(), check=True)


def main() -> None:
    meta: dict[str, dict[str, object]] = {}
    for p in PIECES:
        audio = compose(p)
        out = HERE / f"{p.name}.ogg"
        encode(audio, out)
        meta[out.name] = {"title": p.title, "mood": p.mood, "energy": p.energy, "bpm": p.bpm,
                          "duration_s": round(len(audio) / SR, 2), "loop": True, "licence": "CC0-1.0"}
        print(f"{out.name}: {len(audio) / SR:.1f}s, {out.stat().st_size / 1e6:.2f} MB")
    (HERE / "tracks.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (HERE / "LICENSE.txt").write_text(
        "The audio tracks in this folder (*.ogg) were generated procedurally by fixtures/music/compose.py\n"
        "for the Vidbush project. No samples or third party recordings were used.\n\n"
        "They are dedicated to the public domain under CC0 1.0 Universal:\n"
        "https://creativecommons.org/publicdomain/zero/1.0/\n\n"
        "You may copy, modify, distribute and use them, including commercially and on monetized YouTube\n"
        "videos, without asking permission and without attribution.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
