"""Ladder step 1: re-source the clauses where nothing fits, once, with the picker's own queries.

The picker writes the queries from what the pool contained and what it lacked; this runs them through the
sources lane's shared-pool sourcing (registry.source_pool), so routing, screening, licence flags and the
download cache are those of a live run. The new candidates join the scene's pool.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from ..models import Candidate, Project, ProjectPaths, RealThing, Shot, ShotType


def shots_for(wants: dict[str, dict[str, Any]], scene_id: str) -> list[Shot]:
    """One sourcing shot per clause: its medium decides video or still; the real thing keys entity search."""
    out = []
    for cid, w in wants.items():
        qs = [q for q in w.get("queries", []) if q][:3]
        if not qs:
            continue
        rt = w.get("real_thing") or None
        typ = ShotType.image if w.get("medium") == "still" else ShotType.footage
        out.append(Shot(id=f"rs_{cid}", beat_id="", scene_id=scene_id, start=0.0, end=4.0, duration=4.0, type=typ,
                        intent=str(w.get("subject", "")), queries=qs,
                        real_thing=RealThing(**{k: str(rt.get(k, "")) for k in ("name", "kind", "date", "place")})
                        if isinstance(rt, dict) and rt.get("name") else None))
    return out


class Resourcer:
    """search(wants, scene_id) for the ladder; one project folder under out/resource holds the downloads."""

    def __init__(self, project_dir: Path, out: Path, settings: Any, log: Optional[Callable] = None) -> None:
        self.settings, self.log = settings, log
        src = json.loads((Path(project_dir) / "project.json").read_text(encoding="utf-8"))
        root = Path(out) / "resource"
        src["paths"] = ProjectPaths(root=str(root), work=str(root / "work"), sources=str(root / "sources"),
                                    segments=str(root / "work" / "segments"),
                                    thumbnails=str(root / "thumbnails")).model_dump()
        self.project = Project(**src)
        for p in (root / "work", root / "sources"):
            p.mkdir(parents=True, exist_ok=True)
        self._ctx = None

    def ctx(self):
        if self._ctx is None:
            from ..jobs.context import build_context
            self._ctx = build_context(self.settings, self.project, echo=False)
            self._ctx.stage = "sourcing"
        return self._ctx

    def search(self, wants: dict[str, dict[str, Any]], scene_id: str) -> list[Candidate]:
        from ..source.registry import source_pool
        shots = shots_for(wants, scene_id)
        if not shots:
            return []
        found = source_pool(shots, self.ctx())
        out = [c for cs in found.values() for c in cs]
        if self.log:
            self.log(f"{scene_id} re-sourced {len(shots)} clauses: {len(out)} new candidates")
        return out
