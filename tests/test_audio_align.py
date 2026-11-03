"""Silence alignment, transcript matching and beat spans."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vidbush.audio import align
from vidbush.models import Word

ESPEAK = shutil.which("espeak-ng") or shutil.which("espeak")


def test_syllables_and_breaks() -> None:
    assert align.syllables("canal") == 2
    assert align.syllables("the") == 1
    assert align.syllables("1914.") == 4
    assert align.break_strength("forever.") == 1.0
    assert align.break_strength('"done!"') == 1.0
    assert align.break_strength("Horn,") == 0.6
    assert align.break_strength("canal") == 0.0


def test_regions_from_silences() -> None:
    regs = align.regions_from_silences([(0.0, 0.3), (2.0, 2.5), (4.8, 5.0)], 5.0)
    assert regs == [(0.3, 2.0), (2.5, 4.8)]


def test_silence_align_keeps_words_out_of_pauses() -> None:
    regions = [(0.2, 2.0), (2.8, 5.0)]
    toks = align.text_tokens("The canal opened in 1914. It changed world trade forever.", "b1")
    words = align.silence_align(toks, regions, 0.0, 5.2)
    assert [w.text for w in words] == [t.text for t in toks]
    by = {w.text: w for w in words}
    assert by["1914."].end <= 2.0 + 1e-6
    assert by["It"].start >= 2.8 - 1e-6
    for w in words:
        assert any(s - 1e-6 <= w.start and w.end <= e + 1e-6 for s, e in regions)
    assert all(a.start <= b.start for a, b in zip(words, words[1:]))


@pytest.mark.skipif(ESPEAK is None, reason="espeak-ng not installed")
def test_silence_align_on_espeak_clip(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    text = "Ships sailed around Cape Horn for centuries. Then, in 1914, everything changed."
    subprocess.run([ESPEAK, "-v", "en-us", "-s", "160", "-w", str(wav), text], check=True, capture_output=True)
    regions, dur = align.speech_regions(wav)
    assert len(regions) >= 2 and 2.0 < dur < 10.0
    words = align.silence_align(align.text_tokens(text, "b1"), regions, 0.0, dur)
    assert len(words) == len(text.split())
    centuries = next(w for w in words if w.text == "centuries.")
    then = next(w for w in words if w.text == "Then,")
    gap = [(e1, s2) for (_s1, e1), (s2, _e2) in zip(regions, regions[1:]) if s2 - e1 >= 0.25]
    assert gap, "espeak should pause at the sentence end"
    assert centuries.end <= gap[0][1] + 0.05 and then.start >= gap[0][0] - 0.05
    assert words[0].start >= regions[0][0] - 1e-6 and words[-1].end <= dur + 1e-6


def test_match_words_keeps_script_text_and_fills_missing() -> None:
    toks = align.text_tokens("The Panama Canal opened in 1914.", "b1") + align.text_tokens("It changed trade.", "b2")
    heard = [Word(text=t, start=i * 0.5, end=i * 0.5 + 0.4) for i, t in
             enumerate(["the", "panama", "canal", "opened", "in", "nineteen", "fourteen", "it", "trade"])]
    words = align.match_words(toks, heard, 0.0, 5.0)
    assert [w.text for w in words] == [t.text for t in toks]
    assert [w.beat_id for w in words] == [t.beat_id for t in toks]
    by = {w.text: w for w in words}
    assert by["Canal"].start == pytest.approx(1.0)
    assert 2.5 - 1e-6 <= by["1914."].start and by["1914."].end <= 3.4 + 1e-6
    assert by["It"].start == pytest.approx(3.5)
    assert 3.9 - 1e-6 <= by["changed"].start <= 4.0 + 1e-6
    assert all(a.start <= b.start for a, b in zip(words, words[1:]))


def test_beat_spans_contiguous_cover_duration() -> None:
    words = [Word(text="a", start=0.3, end=0.6, beat_id="b1"), Word(text="b", start=1.0, end=1.4, beat_id="b1"),
             Word(text="c", start=2.2, end=2.6, beat_id="b2"), Word(text="d", start=3.5, end=3.9, beat_id="b3")]
    spans = align.beat_spans(words, ["b1", "b2", "bx", "b3"], 4.5, fixed={"b3": 3.2})
    assert list(spans) == ["b1", "b2", "bx", "b3"]
    assert spans["b1"][0] == 0.0 and spans["b3"][1] == 4.5
    keys = list(spans)
    for a, b in zip(keys, keys[1:]):
        assert spans[a][1] == spans[b][0]
    assert spans["b2"][0] == pytest.approx(1.95)
    assert spans["bx"][0] == spans["bx"][1]
    assert spans["b3"][0] == 3.2
