"""tracks.json is measured from the audio: tempo, phrase, energy, brightness, loop, mood; hand fixes survive."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from vidbush.audio import music, tracks
from vidbush.audio.loudness import write_audio

SR = 22050
RNG = np.random.default_rng(5)


def _clicks(bpm: float, seconds: float, offset: float, gain: float = 0.8) -> np.ndarray:
    """Click track, the first beat of each 4/4 bar accented."""
    x = np.zeros(int(seconds * SR), np.float32)
    n = 300
    tone = np.sin(2 * np.pi * 1500 * np.arange(n) / SR) * np.exp(-np.arange(n) / 60)
    for i, t in enumerate(np.arange(offset, seconds - 0.1, 60.0 / bpm)):
        k = int(t * SR)
        x[k:k + n] += gain * (1.0 if i % 4 == 0 else 0.45) * tone
    return x + 0.003 * RNG.standard_normal(len(x)).astype(np.float32)


def _pad(seconds: float, freq: float = 110.0, amp: float = 0.03) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t) * (1 + 0.2 * np.sin(2 * np.pi * 0.2 * t))).astype(np.float32)


def _bright_noise(seconds: float) -> np.ndarray:
    return (0.1 * np.diff(RNG.standard_normal(int(seconds * SR) + 1))).astype(np.float32)


def test_click_track_tempo_phrase_and_downbeat(tmp_path: Path) -> None:
    m = tracks.analyse_track(write_audio(tmp_path / "click.wav", _clicks(90, 30, 0.5), SR))
    assert m["bpm"] == pytest.approx(90, abs=2)
    assert m["bpm_confidence"] > 0.5
    assert m["phrase_s"] == pytest.approx(16 * m["beat_s"], abs=0.01)
    assert m["bar_s"] == pytest.approx(4 * 60 / 90, abs=0.1)
    assert m["first_downbeat_s"] == pytest.approx(0.5, abs=0.05)


def test_half_time_pulse_is_octave_corrected(tmp_path: Path) -> None:
    # hits only every 2 beats at 90 bpm (45 per minute, outside 50..160): reported at 90
    m = tracks.analyse_track(write_audio(tmp_path / "half.wav", _clicks(45, 30, 0.2), SR))
    assert m["bpm"] == pytest.approx(90, abs=2)


def test_no_pulse_has_no_tempo(tmp_path: Path) -> None:
    m = tracks.analyse_track(write_audio(tmp_path / "pad.wav", _pad(30), SR))
    assert m["bpm"] == 0 and m["phrase_s"] == 0 and m["beat_s"] == 0


def test_brightness_and_energy_order(tmp_path: Path) -> None:
    pad = tracks.analyse_track(write_audio(tmp_path / "pad.wav", _pad(30), SR))
    noise = tracks.analyse_track(write_audio(tmp_path / "noise.wav", _bright_noise(30), SR))
    dense = tracks.analyse_track(write_audio(tmp_path / "dense.wav", _clicks(150, 30, 0.1) + 0.3 * _bright_noise(30)
                                             + _pad(30, 80, 0.2), SR))
    assert pad["brightness"] < noise["brightness"] and pad["centroid_hz"] < noise["centroid_hz"]
    assert dense["energy"] > pad["energy"] + 0.2
    assert pad["mood"] in ("calm", "mysterious")


def test_fade_out_tail_is_not_loopable(tmp_path: Path) -> None:
    x = _clicks(90, 30, 0.5) + 0.05 * _bright_noise(30)
    assert tracks.analyse_track(write_audio(tmp_path / "loop.wav", x, SR))["loopable"]
    fade = x * np.clip((30 - np.arange(len(x)) / SR) / 8, 0, 1).astype(np.float32)
    assert not tracks.analyse_track(write_audio(tmp_path / "fade.wav", fade, SR))["loopable"]


def test_checked_fields_survive_a_rebuild(tmp_path: Path) -> None:
    a = write_audio(tmp_path / "a.wav", _clicks(90, 25, 0.5), SR)
    b = write_audio(tmp_path / "b.wav", _pad(25), SR)
    assert tracks.needs_refresh(tmp_path)
    data = tracks.build_tracks_json(tmp_path)
    assert not tracks.needs_refresh(tmp_path)
    assert set(data) == {"_help", "a.wav", "b.wav"}
    assert data["a.wav"]["measured_by"] == "vidbush.audio.tracks" and data["a.wav"]["checked"] == []
    # Adam corrects a.wav by hand
    meta = json.loads((tmp_path / "tracks.json").read_text(encoding="utf-8"))
    meta["a.wav"].update({"mood": "uplifting", "bpm": 45, "energy": 0.123, "title": "Clicks", "licence": "CC0",
                          "notes": "fine", "checked": ["mood", "bpm"]})
    (tmp_path / "tracks.json").write_text(json.dumps(meta), encoding="utf-8")
    # a.wav changes on disk, b.wav is deleted
    write_audio(a, _clicks(90, 26, 0.5), SR)
    os.utime(a, (a.stat().st_atime, a.stat().st_mtime + 5))
    b.unlink()
    assert tracks.needs_refresh(tmp_path)
    again = tracks.build_tracks_json(tmp_path)
    e = again["a.wav"]
    assert set(again) == {"_help", "a.wav"}
    assert (e["mood"], e["bpm"], e["title"], e["licence"], e["notes"]) == ("uplifting", 45, "Clicks", "CC0", "fine")
    assert e["energy"] != 0.123 and e["duration_s"] == pytest.approx(26, abs=0.05)
    assert json.loads((tmp_path / "tracks.json").read_text(encoding="utf-8")) == again


def test_load_library_ignores_help_key(tmp_path: Path) -> None:
    write_audio(tmp_path / "a.wav", _clicks(90, 25, 0.5), SR)
    tracks.build_tracks_json(tmp_path)
    lib, has_meta = music.load_library(tmp_path)
    assert has_meta and [t.file for t in lib] == ["a.wav"] and lib[0].mood in ("calm", "tense", "uplifting",
                                                                                "mysterious")
    (tmp_path / "tracks.json").write_text(json.dumps({"_help": "only help"}), encoding="utf-8")
    assert music.load_library(tmp_path)[1] is False
