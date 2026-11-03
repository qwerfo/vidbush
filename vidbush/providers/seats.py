"""Seat resolver: code asks for a role, config maps roles to ordered providers.

Fallback moves only the failed request. Image generation is sticky per
production: once a provider succeeds it stays first for the rest of the run.
When no real provider for a role is configured, the mock provider is used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import VoiceConfig
from ..util import JsonLogger, NullLogger
from . import image as image_mod
from . import llm as llm_mod
from . import mock
from . import search as search_mod
from . import tts as tts_mod
from .base import CallResult, CostTracker, ProviderClient, ProviderError, ProviderSpec

ROLES = ["researcher", "writer", "director", "judge", "editor", "cheap_classifier", "image", "video",
         "tts", "vision", "search"]
LLM_ROLES = {"researcher", "writer", "director", "editor"}
VISION_ROLES = {"judge", "vision", "cheap_classifier"}


class Seats:
    """Per-project facade over all providers."""

    def __init__(self, settings: Settings, tracker: CostTracker, logger: Optional[JsonLogger] = None,
                 fixture_key: str = "", force_mock: bool = False) -> None:
        self.settings = settings
        self.tracker = tracker
        self.log = logger or NullLogger()
        self.fixture_key = fixture_key
        self.force_mock = force_mock or settings.offline
        self.client = ProviderClient(tracker, self.log)
        self.fixtures_dir = Path(settings.repo_root) / "fixtures"
        self._sticky: dict[str, str] = {}
        self.mock_roles: set[str] = set()

    # ------------------------------------------------------------ resolution

    def providers_for(self, role: str) -> list[ProviderSpec]:
        """Ordered real providers for a role (configured or not). Mock is appended by callers."""
        if self.force_mock:
            return []
        roles = self.settings.seats.get("roles", {})
        provs = self.settings.seats.get("providers", {})
        out: list[ProviderSpec] = []
        for entry in roles.get(role, []) or []:
            pname = entry.get("provider", "")
            pdef = provs.get(pname, {})
            model = entry.get("model", "")
            if entry.get("model_env"):
                model = self.settings.key(entry["model_env"]) or model
            out.append(ProviderSpec(
                name=pname, type=pdef.get("type", "openai_compat"), model=model,
                base_url=self.settings.key(pdef.get("base_url_env", "")) or pdef.get("base_url", ""),
                api_key=self.settings.key(pdef.get("key_env", "")),
                options={**pdef.get("options", {}), **entry.get("options", {})},
            ))
        sticky = self._sticky.get(role)
        if sticky:
            out.sort(key=lambda s: 0 if s.name == sticky else 1)
        return out

    def is_real(self, role: str) -> bool:
        return any(p.configured for p in self.providers_for(role))

    def status(self) -> dict[str, Any]:
        """Which roles resolve to real providers and which fall back to mock."""
        out = {}
        for role in ROLES:
            provs = self.providers_for(role)
            out[role] = {
                "real": any(p.configured for p in provs),
                "providers": [{"name": p.name, "model": p.model, "type": p.type, "configured": p.configured}
                              for p in provs],
            }
        return out

    def _call(self, role: str, stage: str, fn: Any, mock_fn: Any, note: str = "") -> CallResult:
        provs = [p for p in self.providers_for(role) if p.configured]
        if provs:
            try:
                return self.client.call(role, stage, provs, fn, note)
            except ProviderError as e:
                self.log.warn("role fell back to mock", role=role, stage=stage, error=str(e)[:400])
        self.mock_roles.add(role)
        res = mock_fn()
        self.client._record(role, stage, mock.MOCK_SPEC, ok=True, latency=0.0, note=note or "mock", mock=True,
                            units=res.units, unit_kind=res.unit_kind)
        return res

    # ------------------------------------------------------------ llm

    def llm_json(self, role: str, task: str, system: str, user: str, stage: str,
                 temperature: float = 0.6) -> Optional[Any]:
        """JSON from an LLM role, or a canned fixture in mock mode, or None (caller uses heuristic)."""
        res = self._call(role, stage,
                         lambda spec: llm_mod.chat(spec, system, user, want_json=True, temperature=temperature),
                         lambda: mock.mock_llm(self.fixtures_dir, self.fixture_key, task), note=task)
        if not res.mock and res.cost_usd == 0 and res.units:
            res.cost_usd = res.units / 1000 * self.settings.cost.llm_per_1k_tokens_usd
        return res.value

    def llm_text(self, role: str, task: str, system: str, user: str, stage: str) -> Optional[str]:
        res = self._call(role, stage, lambda spec: llm_mod.chat(spec, system, user, want_json=False),
                         lambda: mock.mock_llm(self.fixtures_dir, self.fixture_key, task), note=task)
        v = res.value
        if isinstance(v, dict):
            return v.get("text")
        return v

    # ------------------------------------------------------------ vision

    def vision_json(self, role: str, task: str, system: str, prompt: str, images: list[str], stage: str,
                    intent: str = "") -> dict[str, Any]:
        """Score images with a vision role. Mock returns deterministic image statistics."""
        parts: list[Any] = [{"type": "text", "text": prompt}] + [llm_mod.image_part(p) for p in images]
        res = self._call(role, stage, lambda spec: llm_mod.chat(spec, system, parts, want_json=True, temperature=0.1),
                         lambda: mock.mock_vision_score(images, intent or prompt), note=task)
        val = res.value if isinstance(res.value, dict) else {}
        val["_mock"] = res.mock
        return val

    # ------------------------------------------------------------ image

    def image(self, prompt: str, out_path: Path, stage: str) -> tuple[Path, bool]:
        """Generate a 1920x1080 PNG. Sticky provider per production."""
        cost = self.settings.cost

        def fn(spec: ProviderSpec) -> CallResult:
            est = float(spec.options.get("cost_per_image", cost.image_per_image_usd))
            if est > cost.image_ceiling_usd:
                raise ProviderError(f"{spec.name} cost {est} above ceiling {cost.image_ceiling_usd}")
            if est > cost.image_warn_usd:
                self.log.warn("image cost above warn level", provider=spec.name, cost=est)
            r = image_mod.generate(spec, prompt, out_path, cost.image_per_image_usd)
            self._sticky["image"] = spec.name
            return r

        res = self._call("image", stage, fn, lambda: mock.mock_image(prompt, out_path), note="image")
        return Path(res.value), res.mock

    # ------------------------------------------------------------ tts

    def tts(self, text: str, voice: VoiceConfig, out_path: Path, stage: str) -> tuple[Path, Optional[list[dict]], bool, str]:
        """Synthesize narration. Returns (audio_path, words or None, is_mock, engine)."""
        def fn(spec: ProviderSpec) -> CallResult:
            settings = {"stability": voice.stability, "similarity_boost": voice.similarity}
            return tts_mod.synthesize(spec, text, voice.voice_id, voice.model, out_path.with_suffix(".mp3"),
                                      settings, self.settings.cost.tts_per_1k_chars_usd)

        res = self._call("tts", stage, fn, lambda: mock.mock_tts(text, out_path), note="tts")
        if res.mock:
            path, words, engine = res.value
            return Path(path), words, True, f"mock_{engine}"
        path, words = res.value
        return Path(path), words, False, res.provider.name

    # ------------------------------------------------------------ search

    def web_search(self, query: str, stage: str) -> list[dict[str, str]]:
        res = self._call("search", stage, lambda spec: search_mod.search(spec, query), lambda: mock.mock_search(query),
                         note="search")
        return res.value or []

    def video(self, prompt: str, still: Optional[Path], out_path: Path, seconds: float, stage: str) -> Optional[Path]:
        """Image to video seat. Off by default; returns None when unavailable."""
        def fn(spec: ProviderSpec) -> CallResult:
            from .video import generate_video
            return generate_video(spec, prompt, still, out_path, seconds, self.settings.cost.video_per_second_usd)

        res = self._call("video", stage, fn, lambda: mock.mock_video(prompt, still, out_path, seconds), note="video")
        return Path(res.value) if res.value else None
