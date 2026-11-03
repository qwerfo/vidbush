"""StageContext: everything a stage needs, built once per project run."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from ..config import Settings, load_brand, load_style_profile
from ..models import Brand, Project, StyleProfile
from ..providers.base import CostTracker
from ..providers.seats import Seats
from ..util import JsonLogger, write_json
from .paths import read_rebased, stored_root

T = TypeVar("T", bound=BaseModel)

# file names inside output/<id>/work, one per stage output
WORK_FILES = {
    "research": "research.json",
    "outline": "outline.json",
    "script": "script.json",
    "narration": "narration.json",
    "plan": "director_plan.json",
    "shotlist": "shotlist.json",  # working copy, also copied to output root
    "captions": "captions.json",
    "mix": "mix.json",
}


@dataclass
class StageContext:
    settings: Settings
    project: Project
    brand: Brand
    profile: StyleProfile
    seats: Seats
    log: JsonLogger
    tracker: CostTracker
    stage: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    old_roots: list[str] = field(default_factory=list)  # earlier project roots, for rebasing moved paths

    # ------------------------------------------------------------ paths

    @property
    def root(self) -> Path:
        return Path(self.project.paths.root)

    @property
    def work(self) -> Path:
        return Path(self.project.paths.work)

    def work_path(self, key: str) -> Path:
        return self.work / WORK_FILES.get(key, key)

    def out_path(self, name: str) -> Path:
        return self.root / name

    # ------------------------------------------------------------ io

    def load(self, key: str, model: type[T]) -> T:
        """Read a work file; absolute paths from an earlier project root are rebased onto this one."""
        return read_rebased(self.work_path(key), model, self.project.id, self.root, self.old_roots)

    def maybe_load(self, key: str, model: type[T]) -> Optional[T]:
        p = self.work_path(key)
        return self.load(key, model) if p.exists() else None

    def save(self, key: str, obj: BaseModel) -> Path:
        return write_json(self.work_path(key), obj)

    def save_project(self) -> None:
        write_json(self.root / "project.json", self.project)

    def rng(self, salt: str = "") -> random.Random:
        """Deterministic per project and per salt."""
        return random.Random(f"{self.project.seed}:{salt}")


def build_context(settings: Settings, project: Project, echo: bool = True) -> StageContext:
    """Load brand, style profile, logger, cost tracker and seats for a project."""
    brand = load_brand(settings, project.brand_id)
    profile = load_style_profile(settings, project.style_profile_id or brand.style_profile_id)
    profile = apply_profile_override(profile, Path(project.paths.work) / "profile_override.json")
    log = JsonLogger(project.id, settings.path(settings.data_folder) / "logs", echo=echo)
    tracker = CostTracker(project.id, Path(project.paths.root) / "cost_report.json")
    seats = Seats(settings, tracker, log, fixture_key=fixture_key_for(project))
    old = stored_root(Path(project.paths.root) / "project.json")
    return StageContext(settings=settings, project=project, brand=brand, profile=profile, seats=seats,
                        log=log, tracker=tracker, old_roots=[old] if old and old != project.paths.root else [])


def apply_profile_override(profile: StyleProfile, path: Path) -> StyleProfile:
    """Merge a per-project override (written by edit ops such as set_caption_style)."""
    if not path.exists():
        return profile
    import json
    data = profile.model_dump(mode="json")
    _deep_merge(data, json.loads(path.read_text(encoding="utf-8")))
    return StyleProfile(**data)


def _deep_merge(base: dict, delta: dict) -> None:
    for k, v in delta.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def fixture_key_for(project: Project) -> str:
    """Mock LLM fixture folder for this project: fixtures/llm/<key>/."""
    from ..util import slug
    if project.brief is not None:
        return slug(project.brief.topic)
    if project.script_path:
        return Path(project.script_path).stem
    if project.voiceover_path:
        return Path(project.voiceover_path).stem
    return ""
