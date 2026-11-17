"""StageContext: everything a stage needs, built once per project run."""
from __future__ import annotations

import random
import hashlib
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

# Stages that write the shot list, in order. Each writes its own snapshot work/<stage>.shotlist.json beside
# the working copy and never another stage's; a stage reads the snapshot of the stage before it, so a rerun
# (resume, --force) starts from that stage's output, not from what later stages made of it.
SHOTLIST_STAGES = ("shotlist", "sourcing", "judge", "visuals", "motion", "assembly")
SNAPSHOT_INDEX = "snapshots.json"  # {"last": stage, "sha": sha of the working copy it wrote, "stages": {...}}


def snapshot_path(work: Path, stage: str) -> Path:
    """work/<stage>.shotlist.json."""
    return Path(work) / f"{stage}.shotlist.json"


def input_stage(stage: str) -> Optional[str]:
    """The shot list stage whose snapshot `stage` reads (None for the first, or a stage outside the chain)."""
    if stage not in SHOTLIST_STAGES or stage == SHOTLIST_STAGES[0]:
        return None
    return SHOTLIST_STAGES[SHOTLIST_STAGES.index(stage) - 1]


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


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
        """Read a work file; absolute paths from an earlier project root are rebased onto this one.

        The shot list inside the shot list chain comes from the snapshot of the stage before this one
        (shotlist_source), so a rerun never starts from a later stage's output."""
        path = self.shotlist_source() if key == "shotlist" else self.work_path(key)
        return read_rebased(path, model, self.project.id, self.root, self.old_roots)

    def shotlist_source(self) -> Path:
        """Which shot list file this stage reads.

        - The input stage's snapshot, when it exists.
        - The working copy when the working copy was changed after the last stage wrote it (an edit op or an
          operator) and this stage was not explicitly forced: the edit wins over the older snapshot.
        - The working copy for projects made before snapshots existed, and for stages outside the chain."""
        work_copy = self.work_path("shotlist")
        prev = input_stage(self.stage)
        if prev is None:
            return work_copy
        snap = snapshot_path(self.work, prev)
        if not snap.exists():
            return work_copy
        if self.edited_outside_stages():
            if self.stage not in self.extra.get("forced", ()):
                return work_copy
            self._keep_edit(work_copy)
        return snap

    def _keep_edit(self, work_copy: Path) -> None:
        """A forced stage starts from its input snapshot; keep the edited working copy beside it first."""
        import shutil
        import time
        if self.extra.get("edit_kept"):
            return
        keep = self.work / f"operator_edit_{int(time.time())}.shotlist.json"
        shutil.copy2(work_copy, keep)
        self.extra["edit_kept"] = str(keep)
        self.log.warn("forced stage starts from its input snapshot; the edited shot list was kept",
                      stage=self.stage, kept=keep.name)

    def edited_outside_stages(self) -> bool:
        """True when work/shotlist.json differs from what the last shot list stage wrote."""
        import json
        idx = self.work / SNAPSHOT_INDEX
        work_copy = self.work_path("shotlist")
        if not idx.exists() or not work_copy.exists():
            return False
        try:
            last = json.loads(idx.read_text(encoding="utf-8")).get("sha", "")
        except (OSError, ValueError):
            return True  # an unreadable index: do not risk ignoring an edit
        return bool(last) and _sha(work_copy) != last

    def snapshot_used(self) -> bool:
        """True when this stage's shot list came from a stage snapshot (not the working copy)."""
        return self.shotlist_source() != self.work_path("shotlist")

    def maybe_load(self, key: str, model: type[T]) -> Optional[T]:
        p = self.work_path(key)
        return self.load(key, model) if p.exists() else None

    def save(self, key: str, obj: BaseModel) -> Path:
        if key == "shotlist" and self.stage in SHOTLIST_STAGES:
            # snapshot and index first, working copy last: a crash in between never makes a stage's own
            # output look like an operator edit
            snap = write_json(snapshot_path(self.work, self.stage), obj)
            self._index(snap)
            import shutil
            work_copy = self.work_path(key)
            tmp = work_copy.with_suffix(".json.tmp")
            shutil.copy2(snap, tmp)
            tmp.replace(work_copy)
            return work_copy
        return write_json(self.work_path(key), obj)

    def _index(self, snap: Path) -> None:
        """Record this stage's snapshot hash in snapshots.json (the working copy gets the same bytes)."""
        import json
        import time
        idx = self.work / SNAPSHOT_INDEX
        try:
            data = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else {}
        except (OSError, ValueError):
            data = {}
        sha = _sha(snap)
        data.setdefault("stages", {})[self.stage] = {"sha": sha, "at": time.time(), "file": snap.name}
        data["last"], data["sha"] = self.stage, sha
        write_json(idx, data)

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
