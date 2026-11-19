"""The assembly sound gate (render/assemble.speech_sync) on synthetic signals, and the excerpt trim (work/trim.json)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vidbush.models import Shot, ShotList
from vidbush.jobs.stages import apply_trim
from vidbush.render.assemble import SOUND_SR, speech_sync
from vidbush.util import NullLogger

FPS = 30
SECONDS = 40.0


def _speech(seconds: float = SECONDS, seed: int = 1) -> np.ndarray:
    """Syllable-like bursts of a voice-band tone with pauses, mono at SOUND_SR."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SOUND_SR)
    gate, i, on = np.zeros(n), 0, True
    while i < n:
        step = int(rng.uniform(0.08, 0.4) * SOUND_SR)
        gate[i:i + step] = 1.0 if on else 0.0
        i, on = i + step, not on
    t = np.arange(n) / SOUND_SR
    return (0.2 * np.sin(2 * np.pi * 220 * t) * gate + 0.05 * np.sin(2 * np.pi * 1100 * t) * gate).astype(np.float32)


def _music(seconds: float = SECONDS) -> np.ndarray:
    t = np.arange(int(seconds * SOUND_SR)) / SOUND_SR
    return (0.05 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.3 * t))).astype(np.float32)


def _delay(x: np.ndarray, s: float) -> np.ndarray:
    return np.concatenate([np.zeros(int(s * SOUND_SR), np.float32), x])[:len(x)]


def test_narration_under_music_and_within_two_frames_passes() -> None:
    nar = _speech()
    res = speech_sync(nar, _delay(nar, 0.04) * 0.8 + _music(), SECONDS - 0.5, 2 / FPS)
    assert res["faults"] == []
    assert len(res["windows"]) == 5 and min(w["corr"] for w in res["windows"]) > 0.9


@pytest.mark.parametrize("final, words", [
    (lambda nar: _music(), "not in the final audio"),                     # music only
    (lambda nar: np.zeros_like(nar), "no sound on 100%"),                   # silence
    (lambda nar: _delay(nar, 0.2) + _music(), "+0.20 s off"),               # out of sync
    (lambda nar: _speech(seed=7), "not in the final audio"),                # another narration
    (lambda nar: nar[:int(20 * SOUND_SR)], "no sound on"),                  # cut short
])
def test_a_final_without_the_narration_in_sync_fails(final, words) -> None:
    nar = _speech()
    res = speech_sync(nar, final(nar), SECONDS - 0.5, 2 / FPS)
    assert res["faults"] and any(words in f for f in res["faults"]), res["faults"]


def test_a_dropout_inside_one_window_fails() -> None:
    nar = _speech()
    out = nar.copy()
    out[int(17 * SOUND_SR):int(19 * SOUND_SR)] = 0
    faults = speech_sync(nar, out, SECONDS - 0.5, 2 / FPS)["faults"]
    assert len(faults) == 1 and faults[0].startswith("16 to 24 s: no sound")


def test_a_silent_reference_is_a_fault() -> None:
    faults = speech_sync(np.zeros(int(SECONDS * SOUND_SR), np.float32), _music(), SECONDS, 2 / FPS)["faults"]
    assert faults == ["the narration has no speech energy in 0 to 40 s"]


# ---------------------------------------------------------------- excerpt trim

def _shotlist() -> ShotList:
    ends = [3.0, 7.5, 12.0, 20.0]
    shots = [Shot(id=f"s{i}", index=i, beat_id="b", scene_id="sc", start=s, end=e, duration=e - s, type="footage",
                  intent="x") for i, (s, e) in enumerate(zip([0.0] + ends[:-1], ends))]
    return ShotList(project_id="p", duration=20.0, shots=shots)


def _ctx(tmp_path, trim=None):
    if trim is not None:
        (tmp_path / "trim.json").write_text(json.dumps(trim))
    return SimpleNamespace(work=tmp_path, log=NullLogger())


def test_trim_ends_at_the_first_shot_end_at_or_after_until(tmp_path) -> None:
    out = apply_trim(_shotlist(), _ctx(tmp_path, {"until": 8.0}))
    assert [s.id for s in out.shots] == ["s0", "s1", "s2"] and out.duration == 12.0
    out = apply_trim(_shotlist(), _ctx(tmp_path, {"until": 7.5, "from": 0}))
    assert [s.id for s in out.shots] == ["s0", "s1"] and out.duration == 7.5
    assert apply_trim(_shotlist(), _ctx(tmp_path, {"until": 99})).duration == 20.0


def test_no_trim_file_changes_nothing_and_from_must_be_zero(tmp_path) -> None:
    sl = _shotlist()
    assert apply_trim(sl, _ctx(tmp_path)) is sl
    with pytest.raises(ValueError, match="from 0"):
        apply_trim(sl, _ctx(tmp_path, {"from": 5, "until": 10}))
    with pytest.raises(ValueError, match="number of seconds"):
        apply_trim(sl, _ctx(tmp_path, {"until": "10"}))
