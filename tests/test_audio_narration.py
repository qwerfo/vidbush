"""Narration: chunk planning, stitch timing math, alignment paths, beat spans, renarration."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from vidbush.audio import narration as nar
from vidbush.audio import transcribe
from vidbush.audio.loudness import measure, read_audio
from vidbush.models import Beat, Narration, Scene, Script, VoiceConfig, Word
from vidbush.providers import mock

ESPEAK = shutil.which("espeak-ng") or shutil.which("espeak")
pytestmark = pytest.mark.skipif(ESPEAK is None, reason="espeak-ng not installed")


def _script() -> Script:
    return Script(title="t", scenes=[
        Scene(id="s1", title="Hook", purpose="hook", beats=[
            Beat(id="b1", text="The Panama Canal opened in 1914."),
            Beat(id="b2", text="Ships no longer sailed around Cape Horn.")]),
        Scene(id="s2", title="Setup", purpose="setup", beats=[
            Beat(id="b3", text="But the story begins with a French dream.")]),
    ])


@pytest.fixture(autouse=True)
def no_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe, "transcribe_words", lambda audio, settings, log: None)


def _assert_spans(n: Narration, beat_ids: list[str]) -> None:
    assert list(n.beat_spans) == beat_ids
    assert n.beat_spans[beat_ids[0]][0] == 0.0
    assert n.beat_spans[beat_ids[-1]][1] == pytest.approx(n.duration, abs=1e-3)
    for a, b in zip(beat_ids, beat_ids[1:]):
        assert n.beat_spans[a][1] == n.beat_spans[b][0]
    for w in n.words:
        s, e = n.beat_spans[w.beat_id]
        assert s - 0.05 <= w.start <= e + 1e-3
    assert all(a.start <= b.start and a.end <= b.start + 1e-3 for a, b in zip(n.words, n.words[1:]))


def test_plan_chunks_per_beat_and_scene_split() -> None:
    sc = _script()
    per_beat = nar.plan_chunks(sc, per_beat=True)
    assert [c.beat_ids for c in per_beat] == [["b1"], ["b2"], ["b3"]]
    assert [c.scene_end for c in per_beat] == [False, True, True]
    per_scene = nar.plan_chunks(sc, per_beat=False)
    assert [c.beat_ids for c in per_scene] == [["b1", "b2"], ["b3"]]
    split = nar.plan_chunks(sc, per_beat=False, max_chars=45)
    assert [c.beat_ids for c in split] == [["b1"], ["b2"], ["b3"]]
    assert all(len(c.text) <= 45 for c in split)


def test_stitch_timing_math() -> None:
    sr = 1000
    parts = [np.ones(1000, np.float32), np.ones(2000, np.float32), np.ones(1500, np.float32)]
    out, offsets = nar.stitch(parts, [0.5, 0.0, 0.0], xfade_s=0.1, sr=sr)
    assert offsets == [0, 1400, 3300]
    assert len(out) == 4500 + 500 - 2 * 100
    assert np.allclose(out[1400:1500], np.linspace(0, 1, 100, endpoint=False), atol=1e-6)
    assert np.allclose(out[3300:3400], 1.0, atol=1e-6)
    assert np.allclose(out[1000:1400], 0.0)


def test_provider_words_ok() -> None:
    assert not nar.provider_words_ok(None, 5.0)
    assert nar.provider_words_ok([{"text": "a", "start": 0.1, "end": 4.9}], 5.0)
    assert not nar.provider_words_ok([{"text": "a", "start": 0.1, "end": 3.0}], 5.0)
    assert not nar.provider_words_ok([{"text": "a", "start": 0.1, "end": 6.0}], 5.0)


def test_narrate_mock_per_beat(make_ctx: Any) -> None:
    ctx = make_ctx(mode="script", project_id="nar1")
    n = nar.narrate(_script(), ctx.brand, ctx.seats, ctx.settings, ctx.work, ctx.log)
    assert Path(n.audio_path) == ctx.work / "narration" / "narration.wav"
    assert n.aligned_by == "silence" and n.provider.startswith("mock")
    assert len(n.words) == len(" ".join(b.text for b in _script().beats()).split())
    _assert_spans(n, ["b1", "b2", "b3"])
    assert len(read_audio(n.audio_path)) == pytest.approx(n.duration * 48000, abs=25)
    assert measure(n.audio_path)["lufs"] == pytest.approx(-16.0, abs=1.0)
    last_b2 = [w for w in n.words if w.beat_id == "b2"][-1]
    assert n.beat_spans["b3"][0] - last_b2.end >= 0.3  # scene pause


class FakeRealSeats:
    """Behaves like a real TTS provider: one call per request with word timings."""

    def __init__(self, scale: float = 1.0) -> None:
        self.calls: list[str] = []
        self.scale = scale

    def is_real(self, role: str) -> bool:
        return role == "tts"

    def tts(self, text: str, voice: VoiceConfig, out_path: Path, stage: str,
            **kw: Any) -> tuple[Path, Optional[list[dict]], bool, str]:
        self.calls.append(text)
        self.kwargs = kw
        res = mock.mock_tts(text, out_path)
        wav, words, _engine = res.value
        words = [{**w, "start": w["start"] * self.scale, "end": w["end"] * self.scale} for w in words]
        return Path(wav), words, False, "fake11"


def test_narrate_real_provider_per_scene(make_ctx: Any) -> None:
    ctx = make_ctx(mode="script", project_id="nar2")
    seats = FakeRealSeats()
    n = nar.narrate(_script(), ctx.brand, seats, ctx.settings, ctx.work, ctx.log)
    assert len(seats.calls) == 2  # one call per scene
    assert n.aligned_by == "provider" and n.provider == "fake11"
    _assert_spans(n, ["b1", "b2", "b3"])
    b3_words = [w for w in n.words if w.beat_id == "b3"]
    assert b3_words[0].start >= n.beat_spans["b3"][0] - 1e-3  # shifted by the scene offset


def test_narrate_drifting_provider_realigns(make_ctx: Any) -> None:
    ctx = make_ctx(mode="script", project_id="nar3")
    n = nar.narrate(_script(), ctx.brand, FakeRealSeats(scale=0.6), ctx.settings, ctx.work, ctx.log)
    assert n.aligned_by == "silence"
    _assert_spans(n, ["b1", "b2", "b3"])


def test_whisper_alignment_used_when_available(make_ctx: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = make_ctx(mode="script", project_id="nar4")
    heard: list[Word] = []

    def fake_transcribe(audio: Path, settings: Any, log: Any) -> list[Word]:
        from vidbush.audio import align
        regions, dur = align.speech_regions(audio)
        toks = align.script_tokens(_script())
        heard.extend(align.silence_align(toks, regions, 0.0, dur))
        return [Word(text=w.text.lower().strip(".,"), start=w.start, end=w.end) for w in heard]

    monkeypatch.setattr(transcribe, "transcribe_words", fake_transcribe)
    n = nar.narrate(_script(), ctx.brand, ctx.seats, ctx.settings, ctx.work, ctx.log)
    assert n.aligned_by == "whisper"
    assert [w.text for w in n.words] == [w.text for w in heard]
    _assert_spans(n, ["b1", "b2", "b3"])


def test_renarrate_beat_shifts_later_timings(make_ctx: Any) -> None:
    ctx = make_ctx(mode="script", project_id="nar5")
    sc = _script()
    n = nar.narrate(sc, ctx.brand, ctx.seats, ctx.settings, ctx.work, ctx.log)
    new_text = "Ships no longer had to sail all the way around the stormy tip of South America."
    n2 = nar.renarrate_beat(n, sc, "b2", new_text, ctx.brand, ctx.seats, ctx.settings, ctx.work, ctx.log)
    delta = n2.duration - n.duration
    assert delta > 1.0
    assert n2.beat_spans["b1"] == n.beat_spans["b1"] or n2.beat_spans["b1"][1] <= n.beat_spans["b1"][1]
    assert n2.beat_spans["b3"][0] == pytest.approx(n.beat_spans["b3"][0] + delta, abs=2e-3)
    old_b3 = [w for w in n.words if w.beat_id == "b3"]
    new_b3 = [w for w in n2.words if w.beat_id == "b3"]
    assert [w.start for w in new_b3] == pytest.approx([w.start + delta for w in old_b3], abs=2e-3)
    assert " ".join(w.text for w in n2.words if w.beat_id == "b2") == new_text
    _assert_spans(n2, ["b1", "b2", "b3"])
    assert len(read_audio(n2.audio_path)) == pytest.approx(n2.duration * 48000, abs=25)
    assert measure(n2.audio_path)["lufs"] == pytest.approx(-16.0, abs=1.0)


def test_narration_from_voiceover(make_ctx: Any, tmp_path: Path) -> None:
    ctx = make_ctx(mode="script", project_id="nar6")
    sc = _script()
    wav = tmp_path / "vo.wav"
    text = " ".join(b.text for b in sc.beats())
    subprocess.run([ESPEAK, "-v", "en-us", "-s", "150", "-w", str(wav), text], check=True, capture_output=True)
    mp3 = tmp_path / "vo.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)],
                   check=True)
    n = nar.narration_from_voiceover(mp3, sc, ctx.settings, ctx.work, ctx.log)
    assert n.provider == "voiceover" and n.aligned_by == "silence"
    assert len(n.words) == len(text.split())
    _assert_spans(n, ["b1", "b2", "b3"])
    assert measure(n.audio_path)["lufs"] == pytest.approx(-16.0, abs=1.0)


class FailingRealSeats(FakeRealSeats):
    """Real TTS seat whose second request fails over to the mock voice (allow_mock_fallback on the seat)."""

    def tts(self, text: str, voice: VoiceConfig, out_path: Path, stage: str,
            **kw: Any) -> tuple[Path, Optional[list[dict]], bool, str]:
        if self.calls:
            res = mock.mock_tts(text, out_path)
            return Path(res.value[0]), None, True, "mock_espeak"
        return super().tts(text, voice, out_path, stage, **kw)


def test_mock_audio_never_spliced_into_real_narration(make_ctx: Any) -> None:
    from vidbush.providers.base import ProviderError
    ctx = make_ctx(mode="script", project_id="nar7")
    ctx.settings.render.workers = 1
    with pytest.raises(ProviderError):
        nar.narrate(_script(), ctx.brand, FailingRealSeats(), ctx.settings, ctx.work, ctx.log)
    ctx = make_ctx(mode="script", project_id="nar7b")
    ctx.settings.render.workers = 1
    ctx.settings.allow_mock_fallback = True
    n = nar.narrate(_script(), ctx.brand, FailingRealSeats(), ctx.settings, ctx.work, ctx.log)
    assert "mock_espeak" in n.provider


def test_seat_error_propagates_and_language_is_passed(make_ctx: Any) -> None:
    from vidbush.providers.base import ProviderError

    class Down(FakeRealSeats):
        def tts(self, *a: Any, **kw: Any) -> Any:
            raise ProviderError("all providers failed for tts")

    ctx = make_ctx(mode="script", project_id="nar8")
    with pytest.raises(ProviderError):
        nar.narrate(_script(), ctx.brand, Down(), ctx.settings, ctx.work, ctx.log)
    seats = FakeRealSeats()
    ctx.brand.language = "de"
    nar.narrate(_script(), ctx.brand, seats, ctx.settings, ctx.work, ctx.log)
    assert seats.kwargs["language"] == "de"


class NoTimingSeats(FakeRealSeats):
    """A real TTS provider that returns audio only (Algrow style), with optional seat options."""

    def __init__(self, options: Optional[dict[str, Any]] = None, keep_words: bool = False) -> None:
        super().__init__()
        self.options = options or {}
        self.keep_words = keep_words
        self.order: list[str] = []

    def tts(self, text: str, voice: VoiceConfig, out_path: Path, stage: str,
            **kw: Any) -> tuple[Path, Optional[list[dict]], bool, str]:
        self.order.append(text)
        path, words, _mock, _engine = super().tts(text, voice, out_path, stage, **kw)
        return path, (words if self.keep_words else None), False, "algrow_tts"

    def tts_option(self, provider: str, key: str, default: Any = None) -> Any:
        return self.options.get(key, default) if provider == "algrow_tts" else default


def _fake_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_transcribe(audio: Path, settings: Any, log: Any) -> list[Word]:
        from vidbush.audio import align
        regions, dur = align.speech_regions(audio)
        return [Word(text=w.text.lower().strip(".,"), start=w.start, end=w.end)
                for w in align.silence_align(align.script_tokens(_script()), regions, 0.0, dur)]
    monkeypatch.setattr(transcribe, "transcribe_words", fake_transcribe)


def test_tts_without_timestamps_is_aligned_against_the_script(make_ctx: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_whisper(monkeypatch)
    ctx = make_ctx(mode="script", project_id="nar_al1")
    n = nar.narrate(_script(), ctx.brand, NoTimingSeats(), ctx.settings, ctx.work, ctx.log)
    assert n.aligned_by == "whisper" and n.provider == "algrow_tts"
    from vidbush.audio import align
    assert [w.text for w in n.words] == [t.text for t in align.script_tokens(_script())]
    _assert_spans(n, ["b1", "b2", "b3"])


def test_tts_with_timestamps_is_used_as_is(make_ctx: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_whisper(monkeypatch)
    ctx = make_ctx(mode="script", project_id="nar_al2")
    n = nar.narrate(_script(), ctx.brand, NoTimingSeats(keep_words=True), ctx.settings, ctx.work, ctx.log)
    assert n.aligned_by == "provider"


def test_force_align_ignores_provider_timestamps(make_ctx: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_whisper(monkeypatch)
    ctx = make_ctx(mode="script", project_id="nar_al3")
    seats = NoTimingSeats(options={"force_align": True}, keep_words=True)
    n = nar.narrate(_script(), ctx.brand, seats, ctx.settings, ctx.work, ctx.log)
    assert n.aligned_by == "whisper"
    _assert_spans(n, ["b1", "b2", "b3"])


def test_first_chunk_is_synthesized_before_the_rest(make_ctx: Any) -> None:
    ctx = make_ctx(mode="script", project_id="nar_al4")
    seats = NoTimingSeats()
    chunks = nar.plan_chunks(_script(), per_beat=False)
    nar.synthesize_all(chunks, ctx.brand, seats, ctx.settings, ctx.work / "narration", ctx.log)
    assert seats.order[0] == chunks[0].text and len(seats.order) == 2
