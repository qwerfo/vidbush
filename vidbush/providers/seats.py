"""Seat resolver: code asks for a role, config maps roles to ordered providers.

Fallback moves only the failed request: the configured order is kept, and a
primary is skipped only while HEALTH has it marked down (its 10 minute probe
returns to it). Mock policy: a role with no configured real provider (or
offline mode) uses mock. A role with a configured real provider whose real
providers all fail a request raises ProviderError, so the stage fails and
resume can retry, unless settings.allow_mock_fallback is true. Every mock use
is recorded in mock_roles and the cost report; roles that were configured as
real but still used mock are also in fallback_roles.
"""
from __future__ import annotations

import threading
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
from .base import HEALTH, CallResult, CostTracker, ProviderClient, ProviderError, ProviderSpec, is_placeholder_model

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
        self._lock = threading.Lock()
        self.mock_roles: set[str] = set()
        self.fallback_roles: set[str] = set()

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
            if is_placeholder_model(model):
                model = ""
            out.append(ProviderSpec(
                name=pname, type=pdef.get("type", "openai_compat"), model=model,
                base_url=self.settings.key(pdef.get("base_url_env", "")) or pdef.get("base_url", ""),
                api_key=self.settings.key(pdef.get("key_env", "")),
                options={**pdef.get("options", {}), **entry.get("options", {})},
            ))
        return out

    def is_real(self, role: str) -> bool:
        return any(p.configured for p in self.providers_for(role))

    def status(self) -> dict[str, Any]:
        """Per role: configured real providers, HEALTH down time, and whether mock would be used."""
        down = HEALTH.snapshot()
        out = {}
        for role in ROLES:
            provs = self.providers_for(role)
            out[role] = {
                "real": any(p.configured for p in provs),
                "providers": [{"name": p.name, "model": p.model, "type": p.type, "configured": p.configured,
                               "down_since": down.get(f"{p.name}:{p.model}")} for p in provs],
            }
        return out

    def mock_policy(self) -> dict[str, Any]:
        """How this Seats decides to use mock, for reports and the Settings page."""
        return {"offline": self.force_mock, "allow_mock_fallback": self.settings.allow_mock_fallback,
                "rule": "mock only for roles with no configured provider, offline mode, or allow_mock_fallback"}

    def _call(self, role: str, stage: str, fn: Any, mock_fn: Any, note: str = "") -> CallResult:
        """Run fn on the role's configured real providers; mock only as the module docstring allows."""
        provs = [p for p in self.providers_for(role) if p.configured]
        if provs:
            try:
                return self.client.call(role, stage, provs, fn, note)
            except ProviderError as e:
                if not self.settings.allow_mock_fallback:
                    self.log.warn("role failed on every real provider", role=role, stage=stage, error=str(e)[:400])
                    raise
                self.log.warn("role fell back to mock", role=role, stage=stage, error=str(e)[:400])
                with self._lock:
                    self.fallback_roles.add(role)
        with self._lock:
            self.mock_roles.add(role)
        res = mock_fn()
        self.client._record(role, stage, mock.MOCK_SPEC, ok=True, latency=0.0, note=note or "mock", mock=True,
                            units=res.units, unit_kind=res.unit_kind)
        return res

    def used_mock_with_real_seats(self) -> bool:
        """True when some seat was real but some role used mock: the project should end needs_review."""
        return bool(self.mock_roles) and any(self.is_real(r) for r in ROLES)

    # ------------------------------------------------------------ llm

    def _llm_estimate(self, res: CallResult) -> CallResult:
        """Fill cost_usd from tokens when the provider did not return a cost."""
        if not res.mock and res.cost_usd == 0 and res.units:
            res.cost_usd = res.units / 1000 * self.settings.cost.llm_per_1k_tokens_usd
        return res

    def _vision_estimate(self, res: CallResult, n_images: int) -> CallResult:
        """Vision cost when missing: the larger of the per image and the token estimates."""
        if not res.mock and res.cost_usd == 0:
            per_img = n_images * self.settings.cost.vision_per_image_usd
            tokens = res.units / 1000 * self.settings.cost.llm_per_1k_tokens_usd
            res.cost_usd = max(per_img, tokens)
        return res

    def llm_json(self, role: str, task: str, system: str, user: str, stage: str,
                 temperature: float = 0.6, timeout: float = llm_mod.DEFAULT_TIMEOUT_S) -> Optional[Any]:
        """JSON from an LLM role, or a canned fixture in mock mode, or None (caller uses heuristic)."""
        res = self._call(role, stage,
                         lambda spec: self._llm_estimate(
                             llm_mod.chat(spec, system, user, want_json=True, temperature=temperature,
                                          timeout=timeout)),
                         lambda: mock.mock_llm(self.fixtures_dir, self.fixture_key, task), note=task)
        return res.value

    def llm_text(self, role: str, task: str, system: str, user: str, stage: str) -> Optional[str]:
        res = self._call(role, stage,
                         lambda spec: self._llm_estimate(llm_mod.chat(spec, system, user, want_json=False)),
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
        res = self._call(role, stage,
                         lambda spec: self._vision_estimate(
                             llm_mod.chat(spec, system, parts, want_json=True, temperature=0.1), len(images)),
                         lambda: mock.mock_vision_score(images, intent or prompt), note=task)
        val = res.value if isinstance(res.value, dict) else {}
        val["_mock"] = res.mock
        return val

    # ------------------------------------------------------------ image

    def image(self, prompt: str, out_path: Path, stage: str, negative_prompt: str = "") -> tuple[Path, bool]:
        """Generate a 1920x1080 PNG. The configured order is kept; only a failed request falls back.
        negative_prompt goes to the provider's own field, or into the prompt as prose when it has none."""
        cost = self.settings.cost

        def fn(spec: ProviderSpec) -> CallResult:
            est = image_mod.estimate_cost(spec, cost.image_per_image_usd)
            if est > cost.image_ceiling_usd:
                raise ProviderError(f"{spec.name} cost {est} above ceiling {cost.image_ceiling_usd}")
            if est > cost.image_warn_usd:
                self.log.warn("image cost above warn level", provider=spec.name, cost=est)
            r = image_mod.generate(spec, prompt, out_path, cost.image_per_image_usd, negative=negative_prompt)
            if r.cost_usd > cost.image_ceiling_usd:
                self.log.warn("image cost above ceiling after the call", provider=spec.name, cost=r.cost_usd)
            return r

        res = self._call("image", stage, fn, lambda: mock.mock_image(prompt, out_path), note="image")
        return Path(res.value), res.mock

    # ------------------------------------------------------------ tts

    def voice_id(self, voice: VoiceConfig) -> str:
        """Brand voice id, else ELEVENLABS_VOICE_ID, else the premade default voice."""
        return voice.voice_id or self.settings.key("ELEVENLABS_VOICE_ID") or tts_mod.DEFAULT_VOICE_ID

    def tts(self, text: str, voice: VoiceConfig, out_path: Path, stage: str, language: str = "",
            previous_text: str = "", next_text: str = "") -> tuple[Path, Optional[list[dict]], bool, str]:
        """Synthesize narration. Returns (audio_path, words or None, is_mock, engine)."""
        def fn(spec: ProviderSpec) -> CallResult:
            settings = {"stability": voice.stability, "similarity_boost": voice.similarity}
            return tts_mod.synthesize(spec, text, self.voice_id(voice), voice.model, out_path.with_suffix(".mp3"),
                                      settings, self.settings.cost.tts_per_1k_chars_usd, language=language,
                                      previous_text=previous_text, next_text=next_text)

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
