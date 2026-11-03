"""Ducking key: one constant level through short pauses, slow linear ramps in long ones; sparse SFX."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from test_audio_music import _case, library  # noqa: F401  (fixture)
from vidbush.audio import music
from vidbush.audio.loudness import SR, read_audio, rms_db
from vidbush.mg import builders as b
from vidbush.models import Brand, Shot, ShotList, StyleProfile
from vidbush.util import NullLogger, run


def _ducked(tmp_path: Path, library: Path, regions: list[tuple[float, float]], total: float) -> np.ndarray:  # noqa: F811
    """Bed level after ducking with the production envelope settings."""
    profile = StyleProfile(id="p")
    tracks = {t.file: t for t in music.load_library(library)[0]}
    sections = [music.Section(track="calm.wav", start=0.0, end=total, purpose="hook")]
    env = music.speech_envelope(regions, total, music.KEY_MERGE_GAP_S, 0.08, music.KEY_HOLD_S)
    key = music.write_key(env, total, tmp_path / "key.wav", profile.music.ducking_depth_db)
    n = int(total * SR)
    inputs, graph = music.duck_graph(music.bed_graph(sections, tracks, total, 1.5), key, profile, n)
    out = tmp_path / "duck.wav"
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[duck]", "-c:a", "pcm_f32le",
         str(out)])
    return read_audio(out, SR, 2)[:, 0]


def _level(x: np.ndarray, a: float, b: float) -> float:
    return rms_db(x[int(a * SR): int(b * SR)])


def test_envelope_defaults_merge_scene_pauses_and_hold() -> None:
    env = music.speech_envelope([(0.5, 4.0), (5.0, 9.0), (12.0, 14.0)], 20.0, lead_s=0.08,
                                hold_s=music.KEY_HOLD_S)
    assert env == [(0.42, 9.4), (11.92, 14.4)]


def test_key_curve_ramps_linearly_in_db() -> None:
    db = music.key_db_curve([(2.0, 4.0)], 8.0, 12.0, ramp_in=0.5, ramp_out=2.0, rate=100)
    thr = 20 * np.log10(music.key_threshold(12.0))
    assert db[300] == pytest.approx(music.KEY_LEVEL_DB)
    assert db[500] == pytest.approx(thr + (music.KEY_LEVEL_DB - thr) * 0.5)
    assert np.isneginf(db[100]) and np.isneginf(db[700])


def test_one_second_gap_keeps_bed_ducked(tmp_path: Path, library: Path) -> None:  # noqa: F811
    x = _ducked(tmp_path, library, [(0.5, 4.0), (5.0, 9.0)], 10.0)
    under, gap = _level(x, 2.0, 3.8), _level(x, 4.2, 4.9)
    assert gap - under < 3.0, (gap, under)


def test_long_pause_opens_slowly(tmp_path: Path, library: Path) -> None:  # noqa: F811
    x = _ducked(tmp_path, library, [(0.5, 4.0), (10.0, 13.0)], 14.0)
    under, open_ = _level(x, 2.0, 3.8), _level(x, 7.0, 9.0)
    assert open_ - under == pytest.approx(12.0, abs=1.5), (open_, under)
    steps = [_level(x, t, t + 0.1) for t in np.arange(4.0, 7.0, 0.1)]
    assert max(np.diff(steps)) < 2.0, "no sudden swell after the last word"
    assert steps[3] - under < 3.0, "still ducked during the hold"


def _mg_shotlist() -> ShotList:
    specs = [b.title_card_spec("Chapter", "One"), b.number_card_spec("10", "Ten"), b.title_card_spec("Two", "x"),
             b.number_card_spec("9", "Nine")]
    shots = [Shot(id=f"s{i}", beat_id="b1", start=t, end=t + 4, duration=4, type="motion_graphic", intent="x",
                  motion=sp) for i, (t, sp) in enumerate(zip([0.5, 5.0, 25.0, 30.0], specs))]
    return ShotList(project_id="t", duration=34.0, shots=shots)


def test_sfx_events_off_by_default_and_capped(tmp_path: Path) -> None:
    sl = _mg_shotlist()
    assert music.sfx_events(sl, music.sfx_config(None, tmp_path)) == []
    cfg = music.SfxConfig(enabled=True, max_per_minute=3)
    ev = music.sfx_events(sl, cfg)
    assert [(e["shot_id"], e["sound"]) for e in ev] == [("s1", "hit"), ("s2", "whoosh")]
    assert music.synth_whoosh().shape[1] == 2 and np.abs(music.synth_hit()).max() == pytest.approx(1.0, abs=1e-3)


def test_build_mix_places_sfx_when_enabled(tmp_path: Path, library: Path, settings: Any) -> None:  # noqa: F811
    nar, script, _ = _case(tmp_path)
    sl = ShotList(project_id="t", duration=13.0, shots=[
        Shot(id="m1", beat_id="b1", start=6.0, end=9.0, duration=3.0, type="motion_graphic", intent="x",
             motion=b.number_card_spec("10", "Ten"))])
    work = tmp_path / "work"
    work.mkdir()
    (work / "sfx.json").write_text(json.dumps({"enabled": True, "gain_db": -12}), encoding="utf-8")
    brand = Brand(id="b", music_folder=str(library))
    music.build_mix(nar, script, sl, brand, StyleProfile(id="p"), settings, work, NullLogger(), random.Random(3))
    info = json.loads((work / "mix.json").read_text())
    assert info["sfx"]["enabled"] and [e["t"] for e in info["sfx"]["events"]] == [6.0]
    assert not (work / "mix_sfx.wav").exists()
    x = read_audio(work / "mix.wav", SR, 2)[:, 0]
    assert _level(x, 6.0, 6.3) > _level(x, 5.5, 5.9) + 3.0, "hit audible at the card start"
