"""Live probes for the TTS seats, the Anthropic provider, streaming and the seat policy.

Run on a machine with keys in .env. Each probe sends at most one or two small requests and prints what
came back, so a person can check the adapters that were built offline:

    python -m vidbush.live_probe seats              # resolved provider and model per role (GET /models only)
    python -m vidbush.live_probe anthropic          # one streamed and one plain call on anthropic_direct
    python -m vidbush.live_probe stream closerouter # one streamed chat on a provider (wellflow_cheap, wellflow)
    python -m vidbush.live_probe algrow             # one short Algrow narration, then script alignment
    python -m vidbush.live_probe algrow --credits   # Algrow credit balance only, no audio made
    python -m vidbush.live_probe elevate            # one short Elevate narration, then script alignment
    python -m vidbush.live_probe elevate --credits  # Elevate tts lane health only, no audio made

Output goes to data/live_probe/. Nothing here is used by a pipeline run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, load_settings
from .providers.base import CostTracker, ProviderSpec
from .providers.seats import ROLES, Seats, chat_fn
from .util import JsonLogger

PROBE_TEXT = ("The Panama Canal opened in August 1914. It cut the voyage from New York to San Francisco by "
              "about eight thousand miles, and it changed how the world moved its goods.")


def _seats() -> tuple[Seats, Path]:
    settings = load_settings(REPO_ROOT)
    out = Path(settings.data_folder) / "live_probe"
    out.mkdir(parents=True, exist_ok=True)
    return Seats(settings, CostTracker("live_probe", out / "cost_report.json"),
                 JsonLogger("live_probe", out, echo=True)), out


def _spec(seats: Seats, role: str, provider: str) -> ProviderSpec:
    for p in seats.providers_for(role):
        if p.name == provider:
            if not p.configured:
                raise SystemExit(f"{provider} is not configured for {role} (key, base URL or model missing)")
            return p
    raise SystemExit(f"{provider} is not listed for role {role} in config/seats.yaml")


def probe_seats() -> None:
    seats, _ = _seats()
    for role in ROLES:
        rows = [f"{p.name}:{p.model or '-'}{'' if p.configured else ' (not configured)'}"
                for p in seats.providers_for(role)]
        print(f"{role:17s} " + "  >  ".join(rows or ["mock"]))


def probe_chat(provider: str, role: str = "director") -> None:
    seats, _ = _seats()
    spec = _spec(seats, role, provider)
    for stream in (True, False) if spec.type == "anthropic" else (True,):
        s = ProviderSpec(spec.name, spec.type, spec.model, spec.base_url, spec.api_key,
                         {**spec.options, "stream": stream})
        t0 = time.time()
        res = chat_fn(s)(s, "You answer in JSON.", 'Return {"ok": true, "model": "<your model name>"}',
                         want_json=True, timeout=180)
        print(json.dumps({"provider": provider, "model": spec.model, "stream": stream, "value": res.value,
                          "tokens": res.units, "cost_usd": round(res.cost_usd, 6),
                          "seconds": round(time.time() - t0, 1)}))


def probe_tts(provider: str, credits_only: bool) -> None:
    """One short narration on an audio only tts seat (algrow_tts, elevate_tts), then script alignment."""
    from .audio import align
    from .audio.narration import Aligner
    from .config import load_brand
    from .providers.seats import TTS_TYPES
    seats, out = _seats()
    spec = _spec(seats, "tts", provider)
    adapter = TTS_TYPES[spec.type]
    if credits_only:
        print(adapter.probe(spec))
        return
    voice = load_brand(seats.settings, "default").voice
    t0 = time.time()
    res = adapter.synthesize(spec, PROBE_TEXT, seats.voice_id(voice, spec), "auto", out / f"{provider}.mp3")
    path, words, model = res.value
    print(json.dumps({"audio": str(path), "bytes": path.stat().st_size, "words_returned": words is not None,
                      "model": model, "credits": res.credits, "seconds": round(time.time() - t0, 1)}))
    from .audio.loudness import read_audio, write_audio
    samples = read_audio(path)[:, 0]
    wav = write_audio(out / f"{provider}.wav", samples)
    aligner = Aligner(wav, seats.settings, seats.log)
    aligned = aligner.align(align.text_tokens(PROBE_TEXT, "b1"), 0.0, len(samples) / 48000)
    print(json.dumps({"aligned_by": aligner.method(), "first_words": [
        {"text": w.text, "start": round(w.start, 2), "end": round(w.end, 2)} for w in aligned[:8]]}))


def main(argv: Any = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vidbush.live_probe", description=__doc__.split("\n")[0])
    ap.add_argument("probe", choices=["seats", "anthropic", "stream", "algrow", "elevate"])
    ap.add_argument("provider", nargs="?", default="closerouter")
    ap.add_argument("--role", default="", help="role whose seat entry to use (default: director, or editor)")
    ap.add_argument("--credits", action="store_true",
                    help="algrow: only read the credit balance; elevate: only read the tts lane health")
    a = ap.parse_args(argv)
    if a.probe == "seats":
        probe_seats()
    elif a.probe == "anthropic":
        probe_chat("anthropic_direct", a.role or "director")
    elif a.probe == "stream":
        probe_chat(a.provider, a.role or ("editor" if a.provider in ("closerouter", "wellflow") else "director"))
    else:
        probe_tts(f"{a.probe}_tts", a.credits)


if __name__ == "__main__":
    main()
