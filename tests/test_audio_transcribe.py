"""Transcription wrapper: graceful absence of faster-whisper, paragraph building."""
from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from vidbush.audio import transcribe
from vidbush.models import Word
from vidbush.util import NullLogger


@pytest.fixture
def no_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("faster_whisper"):
            raise ImportError("faster_whisper not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(transcribe, "_MODELS", {})


def test_transcribe_words_returns_none_without_whisper(settings: Any, tmp_path: Path, no_faster_whisper: None) -> None:
    assert transcribe.transcribe_words(tmp_path / "x.wav", settings, NullLogger()) is None


def test_transcribe_to_text_raises_clear_error(settings: Any, tmp_path: Path, no_faster_whisper: None) -> None:
    with pytest.raises(RuntimeError, match="faster-whisper"):
        transcribe.transcribe_to_text(tmp_path / "x.wav", settings, NullLogger(), None)


def test_whisper_model_name_from_settings(settings: Any) -> None:
    assert transcribe._model_name(settings) in ("base", settings.key("WHISPER_MODEL"))
    settings.env["WHISPER_MODEL"] = "small.en"
    assert transcribe._model_name(settings) == "small.en"


def test_words_to_text_breaks_paragraphs_at_long_pauses() -> None:
    words = [Word(text="One.", start=0.0, end=0.4), Word(text="Two", start=0.5, end=0.8),
             Word(text="three.", start=0.9, end=1.2), Word(text="Next", start=3.0, end=3.3),
             Word(text="part.", start=3.4, end=3.8)]
    assert transcribe.words_to_text(words) == "One. Two three.\n\nNext part."
