"""Sound library: pick order, catalogue round trip, offline behaviour and loop seams. No network."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vidbush.audio import elevenlabs_sfx
from vidbush.audio.library import (APPROVED, REFUSED, UNREVIEWED, Sound, ensure, load_catalogue, loop_tile, pick,
                                   review_rank, save_catalogue)


def _sound(root: Path, sid: str, status: str, make_file: bool = True) -> Sound:
    rel = f"beds/road/{sid}.wav"
    if make_file:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")
    return Sound(id=sid, file=rel, category="bed", place="road", review_status=status)


def test_review_rank() -> None:
    assert review_rank(APPROVED) == 0
    assert review_rank("kept in the library by Adam, fault heard") == 0
    assert review_rank(UNREVIEWED) == 1
    assert review_rank(REFUSED) == 2


def test_pick_order(tmp_path: Path) -> None:
    sounds = [_sound(tmp_path, "u1", UNREVIEWED), _sound(tmp_path, "r1", REFUSED),
              _sound(tmp_path, "gone", APPROVED, make_file=False), _sound(tmp_path, "a1", APPROVED),
              _sound(tmp_path, "u2", UNREVIEWED)]
    save_catalogue(tmp_path, sounds)
    assert pick(tmp_path, "bed", "road").id == "a1"
    assert pick(tmp_path, "bed", "road", exclude=["a1"]).id == "u1"
    assert pick(tmp_path, "bed", "road", exclude=["a1", "u1"]).id == "u2"
    assert pick(tmp_path, "bed", "road", exclude=["a1", "u1", "u2"]) is None  # refused and missing never offered
    assert pick(tmp_path, "bed", "coast") is None


def test_catalogue_round_trip(tmp_path: Path) -> None:
    s = Sound(id="line_draw-1", file="effects/paper/line_draw_1.wav", category="effect", effect="line_draw",
              family="paper", name="Pencil", seconds=1.2, loudness_lufs=-30.5, review_words="fine")
    save_catalogue(tmp_path, [s])
    assert load_catalogue(tmp_path) == [s]
    assert load_catalogue(tmp_path / "nothing") == []


def test_ensure_offline_returns_none(settings, tmp_path: Path) -> None:
    settings.sound_library = str(tmp_path / "lib")
    assert ensure(settings, "bed", "coast") is None
    assert ensure(settings, "effect", "line_draw") is None
    assert load_catalogue(tmp_path / "lib") == []
    with pytest.raises(ValueError):
        ensure(settings, "bed", "moon")


def test_client_refuses_offline_and_without_key(settings) -> None:
    with pytest.raises(elevenlabs_sfx.SoundGenerationError, match="offline"):
        elevenlabs_sfx.generate("A soft tap", 1.0, settings=settings)
    settings.offline = False
    settings.env = {}
    with pytest.raises(elevenlabs_sfx.SoundGenerationError, match="ELEVENLABS_API_KEY"):
        elevenlabs_sfx.generate("A soft tap", 1.0, settings=settings)


def test_loop_tile_has_no_seam_step() -> None:
    sr = 48000
    t = np.arange(3 * sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 3.3 * t)).astype(np.float32)  # 3.3 Hz: the raw wrap jumps
    assert abs(x[-1] - x[0]) > 0.1
    y = loop_tile(x, copies=2)[:, 0]
    body = len(y) // 2
    typical = float(np.abs(np.diff(y)).max())
    assert abs(y[body] - y[body - 1]) <= typical * 1.5  # join between copies
    assert abs(y[0] - y[-1]) <= typical * 1.5  # wrap from end to start
    assert len(y) == 2 * (len(x) - int(0.2 * sr))
