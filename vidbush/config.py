"""Configuration: loaded once at startup and passed down. Business logic never reads env."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from .models import Brand, StyleProfile

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. Existing process env wins."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class SourcesConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: ["local", "youtube", "wikimedia", "archive", "loc", "nasa", "pexels", "pixabay"]
    )
    allow_youtube: bool = True
    flag_nc_nd: bool = True
    blocklist: list[str] = Field(default_factory=list)
    max_candidates: int = 8
    youtube_max_height: int = 1080
    youtube_min_height: int = 720
    local_library: str = "fixtures/sample_assets"
    timeout_s: float = 20.0
    rate_limits: dict[str, list[float]] = Field(default_factory=dict)  # host -> [requests per second, burst]
    max_concurrency: int = 6
    query_budget: dict[str, int] = Field(default_factory=dict)  # source -> max searches per run


class JudgeConfig(BaseModel):
    threshold: float = 70.0
    max_attempts: int = 3
    post_render_every_s: float = 3.0


class CostConfig(BaseModel):
    image_ceiling_usd: float = 0.05
    image_warn_usd: float = 0.03
    # rough unit prices for estimates when the provider does not return cost
    llm_per_1k_tokens_usd: float = 0.002
    vision_per_image_usd: float = 0.002
    tts_per_1k_chars_usd: float = 0.18
    image_per_image_usd: float = 0.02
    video_per_second_usd: float = 0.05


class RenderConfig(BaseModel):
    renderer: str = "python"  # python | remotion
    preset: str = "veryfast"
    crf: int = 20
    workers: int = 3
    width: int = 1920
    height: int = 1080
    fps: int = 30
    target_lufs: float = -14.0
    true_peak: float = -1.0
    narration_lufs: float = -16.0


class Settings(BaseModel):
    """All runtime settings. Built once by load_settings()."""

    repo_root: str
    output_folder: str
    data_folder: str
    source_cache_folder: str
    music_folder: str
    config_folder: str
    env: dict[str, str] = Field(default_factory=dict)
    seats: dict[str, Any] = Field(default_factory=dict)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    offline: bool = False  # never touch the network (tests, sandbox)
    allow_mock_fallback: bool = False  # True: a role whose real providers all fail falls back to mock instead of failing the stage
    fail_at_stage: Optional[str] = None  # test hook: raise when this stage starts

    def key(self, name: str) -> str:
        return self.env.get(name, "")

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else Path(self.repo_root) / p


def _yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_settings(root: Optional[Path] = None, overrides: Optional[dict[str, Any]] = None) -> Settings:
    """Load .env, config/vidbush.yaml and config/seats.yaml into one Settings object."""
    root = Path(root or os.environ.get("VIDBUSH_ROOT") or REPO_ROOT)
    env = load_dotenv(root / ".env")
    env.update({k: v for k, v in os.environ.items() if k.isupper()})
    app = _yaml(root / "config" / "vidbush.yaml")
    seats = _yaml(root / "config" / "seats.yaml")
    s = Settings(
        repo_root=str(root),
        output_folder=env.get("OUTPUT_FOLDER") or app.get("output_folder", "output"),
        data_folder=app.get("data_folder", "data"),
        source_cache_folder=env.get("SOURCE_CACHE_FOLDER") or app.get("source_cache_folder", "data/source_cache"),
        music_folder=env.get("MUSIC_FOLDER") or app.get("music_folder", "fixtures/music"),
        config_folder=str(root / "config"),
        env=env,
        seats=seats,
        sources=SourcesConfig(**app.get("sources", {})),
        judge=JudgeConfig(**app.get("judge", {})),
        cost=CostConfig(**app.get("cost", {})),
        render=RenderConfig(**app.get("render", {})),
        offline=_truthy(env.get("VIDBUSH_OFFLINE", app.get("offline", False))),
        allow_mock_fallback=_truthy(env.get("VIDBUSH_ALLOW_MOCK_FALLBACK", app.get("allow_mock_fallback", False))),
        fail_at_stage=env.get("VIDBUSH_FAIL_AT") or None,
    )
    if env.get("YOUTUBE_ENABLED"):
        s.sources.allow_youtube = _truthy(env["YOUTUBE_ENABLED"])
    for k, v in (overrides or {}).items():
        setattr(s, k, v)
    return s


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def load_style_profile(settings: Settings, profile_id: str) -> StyleProfile:
    path = Path(settings.config_folder) / "style_profiles" / f"{profile_id}.yaml"
    if not path.exists():
        path = Path(settings.config_folder) / "style_profiles" / "documentary_default.yaml"
    data = _yaml(path)
    data.setdefault("id", profile_id)
    return StyleProfile(**data)


def load_brand(settings: Settings, brand_id: str) -> Brand:
    path = Path(settings.config_folder) / "brands" / f"{brand_id}.yaml"
    if not path.exists():
        path = Path(settings.config_folder) / "brands" / "default.yaml"
    data = _yaml(path)
    data.setdefault("id", brand_id)
    return Brand(**data)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def list_ids(settings: Settings, kind: str) -> list[str]:
    folder = Path(settings.config_folder) / kind
    return sorted(p.stem for p in folder.glob("*.yaml"))
