"""Music bed: planning, track choice, ducking depth, mix length and loudness."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vidbush.audio import music
from vidbush.audio.loudness import SR, rms_db, read_audio, write_audio
from vidbush.models import Beat, Brand, Narration, Scene, Script, ShotList, StyleProfile, Word
from vidbush.util import NullLogger, run


def _noise_track(path: Path, seconds: float, seed: int, level: float = 0.1) -> Path:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    spec = np.fft.rfft(x, axis=0)
    spec[int(n * 2000 / SR):] = 0  # dark steady noise: a stable bed level to measure
    y = np.fft.irfft(spec, n, axis=0)
    return write_audio(path, (y / np.std(y) * level).astype(np.float32))


@pytest.fixture
def library(tmp_path: Path) -> Path:
    folder = tmp_path / "music"
    folder.mkdir()
    _noise_track(folder / "calm.wav", 20.0, 1)
    _noise_track(folder / "tense.wav", 20.0, 2)
    (folder / "tracks.json").write_text(json.dumps({
        "calm.wav": {"mood": "calm", "energy": 0.25, "loop": True},
        "tense.wav": {"mood": "tense", "energy": 0.6, "loop": True}}), encoding="utf-8")
    return folder


def _speech(path: Path, spans: list[tuple[float, float]], total: float) -> Path:
    t = np.arange(int(total * SR)) / SR
    sig = np.zeros_like(t)
    for s, e in spans:
        m = (t >= s) & (t < e)
        sig[m] = 0.3 * np.sin(2 * np.pi * 200 * t[m]) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[m]))
    return write_audio(path, sig.astype(np.float32))


def _case(tmp_path: Path, total: float = 12.0) -> tuple[Narration, Script, ShotList]:
    spans = [(0.5, 3.0), (3.2, 5.0), (9.0, total - 1.0)]
    audio = _speech(tmp_path / "narration.wav", spans, total)
    words = [Word(text="w", start=s, end=e, beat_id="b1" if s < 7 else "b2") for s, e in spans]
    nar = Narration(audio_path=str(audio), duration=total, words=words, provider="test", aligned_by="silence",
                    beat_spans={"b1": [0.0, 7.0], "b2": [7.0, total]})
    script = Script(scenes=[Scene(id="s1", title="a", purpose="hook", beats=[Beat(id="b1", text="w w")]),
                            Scene(id="s2", title="b", purpose="turn", beats=[Beat(id="b2", text="w")])])
    return nar, script, ShotList(project_id="t", duration=total + 1.0, shots=[])


def test_plan_boundaries_changes_only_at_some_scenes() -> None:
    scenes = [(0.0, "hook"), (40.0, "setup"), (80.0, "act"), (120.0, "turn"), (150.0, "act"),
              (200.0, "climax"), (260.0, "resolution")]
    bounds = music.plan_boundaries(scenes, 300.0)
    times = [t for t, _ in bounds]
    assert times == [0.0, 40.0, 120.0, 200.0, 260.0]
    long_items = [(i * 40.0, "item") for i in range(20)]
    item_bounds = music.plan_boundaries(long_items, 800.0)
    assert 1 < len(item_bounds) < 6  # forced variety on long stretches, not every scene


def test_pick_track_by_purpose(library: Path) -> None:
    tracks, meta = music.load_library(library)
    assert meta
    rng = random.Random(1)
    assert music.pick_track("turn", tracks, None, {}, rng, True).file == "tense.wav"
    assert music.pick_track("setup", tracks, None, {}, rng, True).file == "calm.wav"
    assert music.pick_track("setup", tracks, "calm.wav", {}, rng, True).file == "tense.wav"


def test_speech_envelope_merges_short_gaps() -> None:
    env = music.speech_envelope([(0.5, 3.0), (3.2, 5.0), (9.0, 11.0)], 12.0, 0.8, 0.08)
    assert env == [(0.42, 5.0), (8.92, 11.0)]
    assert 20 * np.log10(music.key_threshold(12.0)) == pytest.approx(music.KEY_LEVEL_DB - 12.0 * 20 / 19)


def test_ducking_reduces_bed_under_speech(tmp_path: Path, library: Path) -> None:
    nar, script, sl = _case(tmp_path)
    profile = StyleProfile(id="p")
    tracks = {t.file: t for t in music.load_library(library)[0]}
    sections = [music.Section(track="calm.wav", start=0.0, end=sl.duration, purpose="hook")]
    env = music.speech_envelope([(0.5, 3.0), (3.2, 5.0), (9.0, 11.0)], sl.duration, 0.8, 0.08)
    key = music.write_key(env, sl.duration, tmp_path / "key.wav")
    n = int(sl.duration * SR)
    inputs, graph = music.duck_graph(music.bed_graph(sections, tracks, sl.duration, 1.5), key, profile, n)
    out = tmp_path / "duck.wav"
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[duck]", "-c:a", "pcm_f32le",
         str(out)])
    x = read_audio(out, SR, 2)[:, 0]

    def level(a: float, b: float) -> float:
        return rms_db(x[int(a * SR): int(b * SR)])

    under, gap_inside, open_ = level(1.0, 2.8), level(3.0, 3.3), level(6.5, 8.5)
    assert open_ - under == pytest.approx(12.0, abs=1.5), (open_, under)
    assert abs(gap_inside - under) < 1.5  # no pumping in short gaps


def test_build_mix_length_loudness_and_report(tmp_path: Path, library: Path, settings: Any) -> None:
    nar, script, sl = _case(tmp_path)
    brand = Brand(id="b", music_folder=str(library))
    work = tmp_path / "work"
    out = music.build_mix(nar, script, sl, brand, StyleProfile(id="p"), settings, work, NullLogger(), random.Random(3))
    assert out == work / "mix.wav"
    x = read_audio(out, SR, 2)
    assert len(x) == int(round(sl.duration * SR))
    info = json.loads((work / "mix.json").read_text())
    assert info["loudness"]["lufs"] == pytest.approx(-14.0, abs=1.0)
    assert info["loudness"]["true_peak"] <= -0.5
    assert info["selection"] == "tracks.json" and not info["narration_only"]
    assert info["tracks"][0]["track"] == "tense.wav"  # hook prefers energetic moods
    assert info["ducking"]["depth_db"] == 12.0 and info["ducking"]["speech_regions"] == 2


def test_build_mix_override_random_and_empty(tmp_path: Path, library: Path, settings: Any) -> None:
    nar, script, sl = _case(tmp_path, total=6.0)
    work = tmp_path / "work"
    work.mkdir()
    (work / "music_override.json").write_text(json.dumps({"track": "calm.wav"}))
    brand = Brand(id="b", music_folder=str(library))
    music.build_mix(nar, script, sl, brand, StyleProfile(id="p"), settings, work, NullLogger(), random.Random(1))
    info = json.loads((work / "mix.json").read_text())
    assert info["selection"] == "override" and {t["track"] for t in info["tracks"]} == {"calm.wav"}

    (library / "tracks.json").unlink()
    (work / "music_override.json").unlink()
    music.build_mix(nar, script, sl, brand, StyleProfile(id="p"), settings, work, NullLogger(), random.Random(1))
    assert json.loads((work / "mix.json").read_text())["selection"] == "random"

    empty = tmp_path / "empty"
    empty.mkdir()
    settings.music_folder = str(empty)
    out = music.build_mix(nar, script, sl, Brand(id="b", music_folder=str(empty)), StyleProfile(id="p"), settings,
                          work, NullLogger(), random.Random(1))
    info = json.loads((work / "mix.json").read_text())
    assert info["narration_only"] and info["tracks"] == []
    assert len(read_audio(out, SR, 2)) == int(round(sl.duration * SR))
    assert info["loudness"]["lufs"] == pytest.approx(-14.0, abs=1.0)


def test_bundled_library_metadata(settings: Any) -> None:
    folder = settings.path("fixtures/music")
    tracks, meta = music.load_library(folder)
    assert meta and len(tracks) >= 4
    assert {t.mood for t in tracks} >= {"tense", "calm", "uplifting", "mysterious"}
    assert all(90 <= t.duration <= 150 for t in tracks)
    assert sum(Path(t.path).stat().st_size for t in tracks) < 8_000_000
    assert (folder / "LICENSE.txt").read_text().count("CC0") >= 1
