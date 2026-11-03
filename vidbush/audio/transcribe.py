"""Speech to text with faster-whisper (optional dependency).

transcribe_words returns word timings or None when faster-whisper or its model
is unavailable. transcribe_to_text turns a voiceover into script text with a
paragraph break at every long pause, so the script parser can find scenes.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Word
from ..util import JsonLogger

_MODELS: dict[str, Any] = {}
_LOCK = threading.Lock()
PARAGRAPH_PAUSE_S = 1.1


def _model_name(settings: Settings) -> str:
    return settings.key("WHISPER_MODEL") or "base"


def _load_model(settings: Settings, log: JsonLogger) -> Optional[Any]:
    """Cached WhisperModel (cpu, int8). Offline settings use only the local model cache."""
    name = _model_name(settings)
    with _LOCK:
        if name in _MODELS:
            return _MODELS[name]
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(name, device="cpu", compute_type="int8", local_files_only=settings.offline)
        except Exception as e:  # noqa: BLE001 - any import, download or runtime failure means "not available"
            log.warn("faster-whisper unavailable, using fallback alignment", model=name, error=str(e)[:300])
            model = None
        _MODELS[name] = model
        return model


def transcribe_words(audio: Path, settings: Settings, log: JsonLogger) -> Optional[list[Word]]:
    """Word level transcript with timings, or None when transcription is not possible."""
    model = _load_model(settings, log)
    if model is None:
        return None
    try:
        segments, _info = model.transcribe(str(audio), word_timestamps=True, vad_filter=True,
                                           condition_on_previous_text=False)
        words = [Word(text=w.word.strip(), start=round(float(w.start), 3), end=round(float(w.end), 3))
                 for seg in segments for w in (seg.words or []) if w.word.strip()]
    except Exception as e:  # noqa: BLE001
        log.warn("whisper transcription failed", error=str(e)[:300])
        return None
    log.info("whisper transcription", words=len(words), model=_model_name(settings))
    return words or None


def words_to_text(words: list[Word], pause_s: float = PARAGRAPH_PAUSE_S) -> str:
    """Join words into paragraphs, breaking where the speaker paused for pause_s or more."""
    paras: list[list[str]] = [[]]
    prev_end: Optional[float] = None
    for w in words:
        if prev_end is not None and w.start - prev_end >= pause_s and paras[-1]:
            paras.append([])
        paras[-1].append(w.text)
        prev_end = w.end
    return "\n\n".join(" ".join(p) for p in paras if p).strip()


def transcribe_to_text(audio: Path, settings: Settings, log: JsonLogger, seats: Any = None) -> str:
    """Script text for a voiceover that came without a script. Raises RuntimeError if impossible.

    seats is accepted for the stage contract; transcription runs locally.
    """
    words = transcribe_words(audio, settings, log)
    if not words:
        raise RuntimeError(
            "Voiceover mode without a script needs speech recognition, but faster-whisper could not run "
            f"(model '{_model_name(settings)}' not available offline or not installed). Install faster-whisper "
            "and let it download the model once (or set WHISPER_MODEL to a local model path), or supply the "
            "script next to the voiceover with --script.")
    return words_to_text(words)
