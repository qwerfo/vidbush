"""One video keeps one narrator: chunks, cached chunks and re-narrated beats all come from one tts provider.

Runs without espeak (the mock voice falls back to its tone bursts), unlike test_audio_narration.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from vidbush.audio import narration as nar
from vidbush.audio import transcribe
from vidbush.models import Beat, Scene, Script, VoiceConfig
from vidbush.providers import mock
from vidbush.providers.base import ProviderError


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


class SwitchingSeats:
    """Two real tts seats: the primary serves until it is marked down after `serve` calls, then the backup."""

    def __init__(self, serve: int = 99, refuse: str = "") -> None:
        self.calls: list[str] = []
        self.serve = serve
        self.refuse = refuse  # text the primary never speaks (one request falls back, the seat stays up)
        self.engines: list[str] = []

    def is_real(self, role: str) -> bool:
        return role == "tts"

    def tts_engine(self) -> str:
        return "primary" if len(self.calls) < self.serve else "backup"

    def tts(self, text: str, voice: VoiceConfig, out_path: Path, stage: str,
            **kw: Any) -> tuple[Path, Optional[list[dict]], bool, str]:
        engine = "backup" if text == self.refuse else self.tts_engine()
        self.calls.append(text)
        self.kwargs = kw
        wav, words, _engine = mock.mock_tts(text, out_path).value
        self.engines.append(engine)
        return Path(wav), words, False, engine


def test_seat_down_mid_run_keeps_one_narrator(make_ctx: Any) -> None:
    ctx = make_ctx(mode="script", project_id="nar_one1")
    ctx.settings.render.workers = 1
    seats = SwitchingSeats(serve=1)
    n = nar.narrate(_script(), ctx.brand, seats, ctx.settings, ctx.work, ctx.log)
    assert n.provider == "backup"
    assert seats.engines == ["primary", "backup", "backup"]  # the primary's chunk is made again
    # the primary is back: no backup chunk is reused from the cache, so the rerun is all primary again
    again = SwitchingSeats()
    assert nar.narrate(_script(), ctx.brand, again, ctx.settings, ctx.work, ctx.log).provider == "primary"
    assert again.engines == ["primary", "primary"]


def test_chunks_that_stay_mixed_raise(make_ctx: Any) -> None:
    ctx = make_ctx(mode="script", project_id="nar_one2")
    seats = SwitchingSeats(refuse=nar.plan_chunks(_script(), per_beat=False)[1].text)
    with pytest.raises(ProviderError, match="one narrator"):
        nar.narrate(_script(), ctx.brand, seats, ctx.settings, ctx.work, ctx.log)


def test_renarrate_passes_neighbours_and_keeps_the_narrator(make_ctx: Any) -> None:
    ctx = make_ctx(mode="script", project_id="nar_one3")
    sc = _script()
    seats = SwitchingSeats()
    n = nar.narrate(sc, ctx.brand, seats, ctx.settings, ctx.work, ctx.log)
    nar.renarrate_beat(n, sc, "b2", "Ships no longer had to sail around the Horn.", ctx.brand, seats, ctx.settings,
                       ctx.work, ctx.log)
    assert seats.kwargs["previous_text"] == sc.beats()[0].text and seats.kwargs["next_text"] == sc.beats()[2].text
    seats.serve = 0  # the narration's provider is down: a beat from the backup would be another voice
    with pytest.raises(ProviderError, match="one narrator"):
        nar.renarrate_beat(n, sc, "b2", "Ships sailed through instead.", ctx.brand, seats, ctx.settings,
                           ctx.work, ctx.log)
