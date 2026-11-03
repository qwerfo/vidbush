"""Mock providers so the pipeline completes with no keys.

Every mock output is visibly a placeholder and every mock call is marked in
the cost report. Mock LLM returns canned fixtures by (fixture_key, task) or
None, in which case the caller uses its deterministic heuristic.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .base import CallResult, ProviderSpec

MOCK_SPEC = ProviderSpec(name="mock", type="mock", model="mock")
FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def _font(size: int, weight: str = "Bold") -> ImageFont.FreeTypeFont:
    p = FONT_DIR / f"Inter-{weight}.otf"
    try:
        return ImageFont.truetype(str(p), size)
    except OSError:
        return ImageFont.load_default()


# ---------------------------------------------------------------- llm

def mock_llm(fixtures_dir: Path, fixture_key: str, task: str) -> CallResult:
    """Look up fixtures/llm/<key>/<task>.json, then fixtures/llm/_default/<task>.json."""
    for key in (fixture_key, "_default"):
        p = fixtures_dir / "llm" / key / f"{task}.json"
        if key and p.exists():
            return CallResult(value=json.loads(p.read_text(encoding="utf-8")), provider=MOCK_SPEC, mock=True)
    return CallResult(value=None, provider=MOCK_SPEC, mock=True)


# ---------------------------------------------------------------- vision

def image_stats(path: str | Path) -> dict[str, float]:
    im = Image.open(path).convert("RGB").resize((160, 90))
    a = np.asarray(im).astype(np.float32) / 255.0
    lum = a.mean(axis=2)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    colourful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    edges = float(np.abs(np.diff(lum, axis=1)).mean() + np.abs(np.diff(lum, axis=0)).mean())
    return {"brightness": float(lum.mean()), "contrast": float(lum.std()), "colourful": colourful, "edges": edges}


def mock_vision_score(images: list[str], intent: str) -> CallResult:
    """Deterministic 0..100 score from image statistics plus a stable hash jitter."""
    if not images:
        return CallResult(value={"score": 0, "reason": "no keyframes", "rejects": []}, provider=MOCK_SPEC, mock=True)
    scores = []
    for p in images:
        st = image_stats(p)
        exposure = 1.0 - min(1.0, abs(st["brightness"] - 0.45) * 2.2)
        base = 52 + 18 * exposure + 40 * min(st["contrast"], 0.3) + 25 * min(st["colourful"], 0.4) + 60 * min(st["edges"], 0.15)
        scores.append(base)
    h = int(hashlib.md5((intent + "|" + "|".join(Path(p).name for p in images)).encode()).hexdigest()[:6], 16)
    score = max(0.0, min(99.0, sum(scores) / len(scores) + (h % 17) - 6))
    reason = f"mock score from image statistics (exposure, contrast, detail) for: {intent[:60]}"
    return CallResult(value={"score": round(score, 1), "reason": reason, "rejects": []}, provider=MOCK_SPEC, mock=True)


# ---------------------------------------------------------------- image

def mock_image(prompt: str, out_path: Path, width: int = 1920, height: int = 1080, label: str = "MOCK AI IMAGE") -> CallResult:
    """Gradient plus film grain plus the intent text. Clearly a placeholder."""
    seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    hue = rng.random(2)
    c1 = np.array(_hsv(hue[0], 0.55, 0.55))
    c2 = np.array(_hsv(hue[1], 0.65, 0.18))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    ang = rng.random() * math.pi
    t = ((xx / width) * math.cos(ang) + (yy / height) * math.sin(ang))
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    img = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
    vign = 1 - 0.55 * (((xx / width - 0.5) ** 2 + (yy / height - 0.5) ** 2) * 2.2)
    img = img * vign[..., None]
    grain = rng.normal(0, 0.035, (height, width, 1)).astype(np.float32)
    img = np.clip(img + grain, 0, 1)
    im = Image.fromarray((img * 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(0.6))
    _draw_caption(im, prompt, label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, "PNG")
    return CallResult(value=out_path, provider=MOCK_SPEC, mock=True, units=1, unit_kind="images")


def _draw_caption(im: Image.Image, prompt: str, label: str) -> None:
    d = ImageDraw.Draw(im)
    w, h = im.size
    d.text((60, 50), label, font=_font(34, "Black"), fill=(255, 255, 255, 200))
    words, lines, cur = prompt.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) > 46:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    f = _font(52, "SemiBold")
    y = h // 2 - len(lines[:6]) * 36
    for ln in lines[:6]:
        tw = d.textlength(ln, font=f)
        d.text(((w - tw) / 2, y), ln, font=f, fill=(245, 240, 230))
        y += 72


def _hsv(h: float, s: float, v: float) -> tuple[float, float, float]:
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


# ---------------------------------------------------------------- tts

def mock_tts(text: str, out_path: Path, wpm: float = 150.0, voice: str = "en-us") -> CallResult:
    """espeak-ng if available, else a tone-burst pattern. Words estimated, scaled to real length."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = out_path.with_suffix(".wav")
    engine = "tone"
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        cp = subprocess.run([exe, "-v", voice, "-s", str(int(wpm * 1.05)), "-w", str(wav), text],
                            capture_output=True, timeout=300)
        if cp.returncode == 0 and wav.exists() and wav.stat().st_size > 1000:
            engine = "espeak"
    if engine == "tone":
        _tone_bursts(text, wav, wpm)
    dur = _wav_duration(wav)
    words = estimate_words(text, 0.0, dur)
    return CallResult(value=(wav, words, engine), provider=MOCK_SPEC, mock=True, units=len(text), unit_kind="characters")


def _syll(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1 if re.search(r"\d", word) else 1
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1:
        n -= 1
    digits = len(re.findall(r"\d", word))
    return max(1, n + digits)


def estimate_words(text: str, start: float, end: float) -> list[dict[str, Any]]:
    """Distribute words across [start, end] weighted by syllables and punctuation pauses."""
    tokens = text.split()
    if not tokens:
        return []
    weights = []
    for t in tokens:
        w = _syll(t) + (1.2 if re.search(r"[.!?]$", t) else 0.5 if re.search(r"[,;:]$", t) else 0.0)
        weights.append(w)
    total = sum(weights)
    span = max(end - start, 0.01)
    out, t0 = [], start
    for tok, w in zip(tokens, weights):
        d = span * w / total
        pause = span * (1.2 if re.search(r"[.!?]$", tok) else 0.5 if re.search(r"[,;:]$", tok) else 0) / total
        out.append({"text": tok, "start": round(t0, 3), "end": round(t0 + d - pause * 0.8, 3)})
        t0 += d
    return out


def _tone_bursts(text: str, wav: Path, wpm: float) -> None:
    sr = 22050
    words = estimate_words(text, 0.15, 0.15 + len(text.split()) / wpm * 60.0)
    total = (words[-1]["end"] + 0.4) if words else 1.0
    sig = np.zeros(int(total * sr), dtype=np.float32)
    for i, w in enumerate(words):
        a, b = int(w["start"] * sr), int(w["end"] * sr)
        t = np.arange(b - a) / sr
        f = 180 + (i % 5) * 25
        env = np.minimum(1, np.minimum(t, t[::-1] if len(t) else t) * 40)
        sig[a:b] += 0.25 * np.sin(2 * np.pi * f * t) * env
    _write_wav(wav, sig, sr)


def _write_wav(path: Path, sig: np.ndarray, sr: int) -> None:
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def mock_search(query: str) -> CallResult:
    return CallResult(value=[], provider=MOCK_SPEC, mock=True)


def mock_video(prompt: str, still: Optional[Path], out_path: Path, seconds: float) -> CallResult:
    """No mock image-to-video: return None so the caller keeps the still with Ken Burns."""
    return CallResult(value=None, provider=MOCK_SPEC, mock=True)
