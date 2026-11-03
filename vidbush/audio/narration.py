"""Narration: TTS per scene (or per beat in mock mode), stitching, alignment, loudness.

Real TTS gets one call per scene (scenes longer than MAX_CHUNK_CHARS split at
beat boundaries) so prosody flows across beats. Mock TTS synthesizes per beat so
beat spans are exact. Chunks are joined with 40 ms linear crossfades (the same
curve as FFmpeg acrossfade, computed sample exact in numpy so word timings can
be shifted precisely) and a 0.35 s pause after each scene. Provider word
timings are kept when they agree with the audio; otherwise words are re-aligned
with faster-whisper, else with silence based alignment. The result is
normalized to settings.render.narration_lufs into work/narration/narration.wav.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from ..config import Settings
from ..models import Brand, Narration, Script, Word
from ..providers.base import ProviderError
from ..util import JsonLogger, no_em_dash, read_json, stable_hash, write_json
from . import align
from .loudness import SR, normalize, read_audio, write_audio

CROSSFADE_S = 0.04
SCENE_PAUSE_S = 0.35
MAX_CHUNK_CHARS = 4500
DRIFT_TOLERANCE = 0.03


class Chunk(BaseModel):
    """One TTS request: a whole scene (real TTS) or one beat (mock TTS)."""

    scene_id: str
    beat_ids: list[str]
    texts: list[str]
    scene_end: bool = False
    path: str = ""
    words: Optional[list[dict[str, Any]]] = None  # provider words, seconds from chunk start
    is_mock: bool = True
    engine: str = ""

    @property
    def text(self) -> str:
        return " ".join(self.texts)

    def tokens(self) -> list[align.Token]:
        return [t for bid, txt in zip(self.beat_ids, self.texts) for t in align.text_tokens(txt, bid)]


# ---------------------------------------------------------------- planning and synthesis

def plan_chunks(script: Script, per_beat: bool, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split the script into TTS requests. Every chunk lies inside one scene."""
    chunks: list[Chunk] = []
    for scene in script.scenes:
        beats = [b for b in scene.beats if b.text.strip()]
        cur: Optional[Chunk] = None
        for b in beats:
            text = no_em_dash(b.text.strip())
            if cur is None or per_beat or len(cur.text) + len(text) + 1 > max_chars:
                cur = Chunk(scene_id=scene.id, beat_ids=[], texts=[])
                chunks.append(cur)
            cur.beat_ids.append(b.id)
            cur.texts.append(text)
        if cur is not None:
            cur.scene_end = True
    return chunks


def synthesize_chunk(chunk: Chunk, brand: Brand, seats: Any, nar_dir: Path, log: JsonLogger,
                     allow_mock_fallback: bool = False) -> Chunk:
    """TTS for one chunk, cached on disk by text, voice and language (a sidecar json keeps words).

    With a real TTS seat, a chunk that came back from the mock voice raises ProviderError
    unless allow_mock_fallback: mock audio is never spliced silently into real narration.
    """
    real = seats.is_real("tts")
    key = stable_hash(chunk.text, brand.voice, brand.language, "real" if real else "mock")
    sidecar = nar_dir / f"chunk_{key}.json"
    if sidecar.exists():
        cached = Chunk(**read_json(sidecar))
        if Path(cached.path).exists() and not (real and cached.is_mock):
            return chunk.model_copy(update={"path": cached.path, "words": cached.words,
                                            "is_mock": cached.is_mock, "engine": cached.engine})
    path, words, is_mock, engine = seats.tts(chunk.text, brand.voice, nar_dir / f"chunk_{key}.wav", "narration",
                                             language=brand.language)
    if real and is_mock:
        if not allow_mock_fallback:
            raise ProviderError(f"tts returned mock audio for scene {chunk.scene_id} with a real tts seat")
        log.warn("tts provider failed, chunk narrated by the mock voice", scene=chunk.scene_id, beats=chunk.beat_ids)
    out = chunk.model_copy(update={"path": str(path), "words": None if is_mock else words,
                                   "is_mock": is_mock, "engine": engine})
    write_json(sidecar, out)
    return out


def synthesize_all(chunks: list[Chunk], brand: Brand, seats: Any, settings: Settings, nar_dir: Path,
                   log: JsonLogger) -> list[Chunk]:
    workers = max(1, min(4, settings.render.workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(lambda c: synthesize_chunk(c, brand, seats, nar_dir, log,
                                                        settings.allow_mock_fallback), chunks))
    log.info("narration chunks synthesized", chunks=len(done), mock=sum(c.is_mock for c in done))
    return done


# ---------------------------------------------------------------- stitching

def stitch(parts: list[np.ndarray], pads_s: list[float], xfade_s: float = CROSSFADE_S,
           sr: int = SR) -> tuple[np.ndarray, list[int]]:
    """Join mono parts with linear crossfades; pads_s[i] of silence follows part i.

    Returns (audio, offsets) where offsets[i] is the sample where part i starts.
    Total length = sum(len(part) + pad) - xfade * (len(parts) - 1).
    """
    xf = int(round(xfade_s * sr))
    lengths = [max(len(p) + int(round(pad * sr)), 2 * xf) for p, pad in zip(parts, pads_s)]
    offsets = [0]
    for n in lengths[:-1]:
        offsets.append(offsets[-1] + n - xf)
    out = np.zeros(offsets[-1] + lengths[-1] if parts else 0, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, xf, endpoint=False, dtype=np.float32)
    for k, (off, part, n) in enumerate(zip(offsets, parts, lengths)):
        head = min(xf, len(part)) if k else 0
        if head:
            out[off:off + xf] *= 1.0 - ramp
            out[off:off + head] += part[:head] * ramp[:head]
        out[off + head:off + len(part)] += part[head:]
    return out, offsets


def provider_words_ok(words: Optional[list[dict[str, Any]]], duration: float) -> bool:
    """Provider timings are trusted when present, ordered and within 3 percent of the audio length."""
    if not words:
        return False
    last = float(words[-1].get("end") or 0.0)
    if last <= 0 or last > duration * (1 + DRIFT_TOLERANCE):
        return False
    trailing = duration - last
    return trailing <= max(duration * DRIFT_TOLERANCE, 0.6)


def provider_to_script(chunk: Chunk, offset: float) -> list[Word]:
    """Provider words mapped onto the chunk's script tokens, shifted to narration time."""
    heard = [Word(text=str(w["text"]), start=float(w["start"]) + offset, end=float(w["end"]) + offset)
             for w in chunk.words or []]
    tokens = chunk.tokens()
    if len(heard) == len(tokens):
        return [Word(text=t.text, beat_id=t.beat_id, start=round(h.start, 3), end=round(h.end, 3))
                for t, h in zip(tokens, heard)]
    return align.match_words(tokens, heard, offset, heard[-1].end if heard else offset)


# ---------------------------------------------------------------- alignment

class Aligner:
    """Re-aligns token runs against one audio file: whisper when available, else silences."""

    def __init__(self, audio: Path, settings: Settings, log: JsonLogger) -> None:
        self.audio, self.settings, self.log = audio, settings, log
        self.used: set[str] = set()
        self._heard: Optional[list[Word]] = None
        self._tried_whisper = False
        self._regions: Optional[list[align.Region]] = None

    def heard(self) -> Optional[list[Word]]:
        if not self._tried_whisper:
            from .transcribe import transcribe_words
            self._tried_whisper = True
            self._heard = transcribe_words(self.audio, self.settings, self.log)
        return self._heard

    def regions(self) -> list[align.Region]:
        if self._regions is None:
            self._regions, _ = align.speech_regions(self.audio)
        return self._regions

    def align(self, tokens: list[align.Token], t0: float, t1: float) -> list[Word]:
        heard = self.heard()
        if heard:
            self.used.add("whisper")
            subset = [w for w in heard if t0 - 0.05 <= w.start < t1]
            return align.match_words(tokens, subset, t0, t1)
        self.used.add("silence")
        return align.silence_align(tokens, self.regions(), t0, t1)

    def method(self) -> str:
        for m in ("silence", "whisper"):
            if m in self.used:
                return m
        return "provider"


def monotonic(words: list[Word], duration: float) -> list[Word]:
    """Clamp timings so words are ordered, inside [0, duration] and never overlap."""
    out: list[Word] = []
    for i, w in enumerate(words):
        start = max(w.start, out[-1].start + 0.01 if out else 0.0)
        start = min(start, max(duration - 0.02, 0.0))
        nxt = words[i + 1].start if i + 1 < len(words) else duration
        end = min(max(w.end, start + 0.02), max(nxt, start + 0.02), duration)
        out.append(w.model_copy(update={"start": round(start, 3), "end": round(max(end, start + 0.01), 3)}))
    return out


def _align_chunks(chunks: list[Chunk], parts: list[np.ndarray], offsets: list[int], aligner: Aligner) -> list[Word]:
    words: list[Word] = []
    for c, part, off in zip(chunks, parts, offsets):
        t0, dur = off / SR, len(part) / SR
        if not c.is_mock and provider_words_ok(c.words, dur):
            aligner.used.add("provider")
            words.extend(provider_to_script(c, t0))
        else:
            words.extend(aligner.align(c.tokens(), t0, t0 + dur))
    return words


# ---------------------------------------------------------------- entry points

def narrate(script: Script, brand: Brand, seats: Any, settings: Settings, work_dir: Path,
            log: JsonLogger) -> Narration:
    """Synthesize, stitch, align and normalize the narration for a script."""
    nar_dir = Path(work_dir) / "narration"
    nar_dir.mkdir(parents=True, exist_ok=True)
    chunks = plan_chunks(script, per_beat=not seats.is_real("tts"))
    if not chunks:
        raise ValueError("script has no narration text")
    chunks = synthesize_all(chunks, brand, seats, settings, nar_dir, log)
    parts = [read_audio(c.path)[:, 0] for c in chunks]
    pads = [SCENE_PAUSE_S if c.scene_end and i < len(chunks) - 1 else 0.0 for i, c in enumerate(chunks)]
    audio, offsets = stitch(parts, pads)
    stitched = write_audio(nar_dir / "stitched.wav", audio)
    aligner = Aligner(stitched, settings, log)
    duration = len(audio) / SR
    words = monotonic(_align_chunks(chunks, parts, offsets, aligner), duration)
    out = nar_dir / "narration.wav"
    loud = normalize(stitched, out, settings.render.narration_lufs, settings.render.true_peak, SR, 1,
                     exact_samples=len(audio))
    fixed = {c.beat_ids[0]: round(offsets[i] / SR, 3) for i, c in enumerate(chunks) if i > 0}
    spans = align.beat_spans(words, [b.id for b in script.beats()], duration, fixed)
    provider = "+".join(sorted({c.engine for c in chunks}))
    log.info("narration ready", duration=round(duration, 2), words=len(words), chunks=len(chunks),
             aligned_by=aligner.method(), provider=provider, lufs=loud.get("lufs"))
    return Narration(audio_path=str(out), duration=round(duration, 3), words=words, provider=provider,
                     aligned_by=aligner.method(), beat_spans=spans)


def narration_from_voiceover(audio: Path, script: Script, settings: Settings, work_dir: Path,
                             log: JsonLogger) -> Narration:
    """Use an operator voiceover: 48 kHz wav at narration loudness, words aligned to the script."""
    nar_dir = Path(work_dir) / "narration"
    nar_dir.mkdir(parents=True, exist_ok=True)
    samples = read_audio(audio)[:, 0]
    src = write_audio(nar_dir / "voiceover_48k.wav", samples)
    out = nar_dir / "narration.wav"
    loud = normalize(src, out, settings.render.narration_lufs, settings.render.true_peak, SR, 1,
                     exact_samples=len(samples))
    duration = len(samples) / SR
    aligner = Aligner(out, settings, log)
    words = monotonic(aligner.align(align.script_tokens(script), 0.0, duration), duration)
    spans = align.beat_spans(words, [b.id for b in script.beats()], duration)
    log.info("voiceover narration ready", duration=round(duration, 2), words=len(words),
             aligned_by=aligner.method(), lufs=loud.get("lufs"))
    return Narration(audio_path=str(out), duration=round(duration, 3), words=words, provider="voiceover",
                     aligned_by=aligner.method(), beat_spans=spans)


_RANK = {"provider": 0, "whisper": 1, "silence": 2, "estimate": 3}


def _trim_speech(samples: np.ndarray, path: Path) -> tuple[np.ndarray, float]:
    """Cut a chunk to its speech (small margins kept). Returns (speech, seconds trimmed at the start)."""
    regions, _ = align.speech_regions(path)
    if not regions:
        return samples, 0.0
    a = max(0, int((regions[0][0] - 0.04) * SR))
    b = min(len(samples), int((regions[-1][1] + 0.08) * SR))
    return (samples[a:b], a / SR) if b > a else (samples, 0.0)


def renarrate_beat(narration: Narration, script: Script, beat_id: str, new_text: str, brand: Brand, seats: Any,
                   settings: Settings, work_dir: Path, log: JsonLogger) -> Narration:
    """Re-synthesize one beat and splice it into narration.wav; later timings shift by the delta.

    The new speech keeps the old beat's leading and trailing pauses, joins use the
    same 40 ms crossfade, and the whole file is renormalized afterwards.
    """
    if beat_id not in narration.beat_spans:
        raise KeyError(f"beat {beat_id} has no span in the narration")
    nar_dir = Path(work_dir) / "narration"
    scene = next(sc for sc in script.scenes if any(b.id == beat_id for b in sc.beats))
    chunk = synthesize_chunk(Chunk(scene_id=scene.id, beat_ids=[beat_id], texts=[no_em_dash(new_text.strip())]),
                             brand, seats, nar_dir, log, settings.allow_mock_fallback)
    raw = read_audio(chunk.path)[:, 0]
    speech, trimmed = _trim_speech(raw, Path(chunk.path))
    s, e = narration.beat_spans[beat_id]
    old_words = [w for w in narration.words if w.beat_id == beat_id]
    lead = float(np.clip(old_words[0].start - s, 0.05, 0.6)) if old_words else 0.1
    trail = float(np.clip(e - old_words[-1].end, 0.1, 1.2)) if old_words else 0.3
    old = read_audio(narration.audio_path)[:, 0]
    sa, sb = int(round(s * SR)), int(round(e * SR))
    parts: list[np.ndarray] = [old[:sa]] if sa > 0 else []
    k = len(parts)
    parts.append(np.concatenate([np.zeros(int(lead * SR), np.float32), speech]))
    pads = [0.0] * k + [trail]
    if sb < len(old):
        parts.append(old[sb:])
        pads.append(0.0)
    audio, offsets = stitch(parts, pads)
    duration = len(audio) / SR
    new_start = offsets[k] / SR
    new_end = offsets[k + 1] / SR if k + 1 < len(parts) else duration
    delta = new_end - e
    stitched = write_audio(nar_dir / "stitched_edit.wav", audio)
    if not chunk.is_mock and provider_words_ok(chunk.words, len(raw) / SR):
        beat_words, method = provider_to_script(chunk, new_start + lead - trimmed), "provider"
    else:
        aligner = Aligner(stitched, settings, log)
        beat_words = aligner.align(chunk.tokens(), new_start, new_end)
        method = aligner.method()
    words = ([w for w in narration.words if w.beat_id != beat_id and w.end <= s + 1e-3] + beat_words +
             [w.model_copy(update={"start": round(w.start + delta, 3), "end": round(w.end + delta, 3)})
              for w in narration.words if w.beat_id != beat_id and w.start >= e - 1e-3])
    out = nar_dir / "narration.wav"
    normalize(stitched, out, settings.render.narration_lufs, settings.render.true_peak, SR, 1,
              exact_samples=len(audio))
    spans = _shift_spans(narration.beat_spans, beat_id, new_start, new_end, delta, duration)
    aligned_by = max(narration.aligned_by, method, key=lambda m: _RANK.get(m, 3))
    log.info("beat renarrated", beat_id=beat_id, delta=round(delta, 3), duration=round(duration, 2))
    return narration.model_copy(update={"duration": round(duration, 3), "words": monotonic(words, duration),
                                        "beat_spans": spans, "audio_path": str(out), "aligned_by": aligned_by})


def _shift_spans(spans: dict[str, list[float]], beat_id: str, new_start: float, new_end: float, delta: float,
                 duration: float) -> dict[str, list[float]]:
    """Spans after a splice: earlier beats end at new_start, later beats move by delta."""
    out: dict[str, list[float]] = {}
    after = False
    for bid, (s, e) in spans.items():
        if bid == beat_id:
            out[bid] = [round(new_start, 3), round(new_end, 3)]
            after = True
        elif after:
            out[bid] = [round(s + delta, 3), round(e + delta, 3)]
        else:
            out[bid] = [min(s, round(new_start, 3)), min(e, round(new_start, 3))]
    out[list(out)[-1]][1] = round(duration, 3)
    return out
