"""Loudness measurement and two pass normalization."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vidbush.audio.loudness import measure, normalize, read_audio, write_audio


def _speechlike(seconds: float, sr: int = 48000, level: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(3)
    t = np.arange(int(seconds * sr)) / sr
    env = (np.sin(2 * np.pi * 3.1 * t) > -0.2).astype(np.float32)
    tone = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t) + 0.2 * rng.standard_normal(len(t))
    return (level * tone * env).astype(np.float32)


def test_measure_and_normalize(tmp_path: Path) -> None:
    src = write_audio(tmp_path / "quiet.wav", _speechlike(6.0))
    before = measure(src)
    assert before["lufs"] < -25
    out = tmp_path / "loud.wav"
    n = 6 * 48000 + 123
    got = normalize(src, out, -16.0, -1.0, 48000, 1, exact_samples=n)
    assert got["lufs"] == pytest.approx(-16.0, abs=0.6)
    assert got["true_peak"] <= -0.5
    assert len(read_audio(out)) == n


def test_normalize_silence_passthrough(tmp_path: Path) -> None:
    src = write_audio(tmp_path / "silent.wav", np.zeros(48000, np.float32))
    got = normalize(src, tmp_path / "out.wav", -16.0, exact_samples=48000)
    assert got["lufs"] <= -69
    assert len(read_audio(tmp_path / "out.wav")) == 48000
