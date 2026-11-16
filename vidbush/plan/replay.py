"""Replay the clause-first planner on a stored project, with no sourcing and no render.

    python -m vidbush.plan.replay <project_dir> <out_dir> [--scenes s01,s08] [--env <dir with .env>]

Reads work/script.json, work/narration.json and the candidates stored in work/shotlist.json; writes
clauses.json, plan_v2.json, shotlist.json and metrics.json to out_dir. Model replies are cached in
out_dir/cache by a hash of the call, so a second replay with unchanged prompts costs nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from ..models import Narration, Script, ShotList


class CachedSeats:
    """Wraps Seats: identical llm_json / vision_json calls are answered from disk."""

    def __init__(self, seats, cache_dir: Path) -> None:
        self.seats = seats
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.calls = 0

    def _key(self, *parts: Any) -> Path:
        h = hashlib.sha256()
        for p in parts:
            if isinstance(p, list):
                for img in p:
                    h.update(Path(img).read_bytes() if Path(img).exists() else str(img).encode())
            else:
                h.update(str(p).encode("utf-8"))
        return self.dir / f"{h.hexdigest()[:24]}.json"

    def _cached(self, key: Path, fn):
        if key.exists():
            self.hits += 1
            return json.loads(key.read_text(encoding="utf-8"))
        self.calls += 1
        val = fn()
        if val is not None and not (isinstance(val, dict) and val.get("_mock")):
            key.write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
        return val

    def llm_json(self, role, task, system, user, stage, **kw):
        return self._cached(self._key(role, task, system, user),
                            lambda: self.seats.llm_json(role, task, system, user, stage, **kw))

    def vision_json(self, role, task, system, prompt, images, stage, **kw):
        return self._cached(self._key(role, task, system, prompt, list(images), kw.get("details")),
                            lambda: self.seats.vision_json(role, task, system, prompt, images, stage, **kw))


def build_seats(out: Path, env_dir: Optional[Path]):
    from ..config import load_dotenv, load_settings
    from ..providers.base import CostTracker
    from ..providers.seats import Seats
    from ..util import JsonLogger

    if env_dir:
        for name in (".env", ".env.local"):
            for k, v in load_dotenv(env_dir / name).items():
                os.environ.setdefault(k, v)
    settings = load_settings()
    tracker = CostTracker("plan_v2_replay", out / "cost_report.json")
    seats = Seats(settings, tracker, JsonLogger("plan_v2_replay", out / "logs", echo=False))
    return CachedSeats(seats, out / "cache"), tracker


def load_project(project: Path):
    w = project / "work"
    script = Script(**json.loads((w / "script.json").read_text(encoding="utf-8")))
    narration = Narration(**json.loads((w / "narration.json").read_text(encoding="utf-8")))
    sl_path = w / "shotlist.json" if (w / "shotlist.json").exists() else project / "shotlist.json"
    old = ShotList(**json.loads(sl_path.read_text(encoding="utf-8")))
    return script, narration, old


def main(argv: Optional[list[str]] = None) -> int:
    from . import clauses as C
    from . import direct as D

    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--env", default="")
    ap.add_argument("--stop-after", default="", choices=["", "clauses", "direct", "edit"])
    a = ap.parse_args(argv)
    project, out = Path(a.project), Path(a.out)
    (out / "work").mkdir(parents=True, exist_ok=True)
    script, narration, old = load_project(project)
    only = set(filter(None, a.scenes.split(","))) or None
    clauses = C.segment(script, narration)
    C.save(clauses, out / "work" / "clauses.json")
    print("clauses", C.stats(clauses))
    if a.stop_after == "clauses":
        return 0
    seats, tracker = build_seats(out, Path(a.env) if a.env else None)

    def log(m: str) -> None:
        print(m, f"(spent {tracker.total_usd:.3f} USD)", flush=True)

    scenes = D.direct_all(seats, script, clauses, narration, log=log, only=only)
    (out / "work" / "direct.json").write_text(json.dumps(scenes, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.stop_after == "direct":
        print("spent", round(tracker.total_usd, 4), "cache hits", seats.hits)
        return 0
    from . import edit as E
    from . import emit as M
    from . import metrics as X

    edits = E.edit_all(seats, scenes, clauses, narration, old, out / "sheets", log=log)
    plan, sl = M.emit(project.name, script, narration, clauses, scenes, edits)
    (out / "work" / "plan_v2.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "work" / "shotlist.json").write_text(sl.model_dump_json(indent=1), encoding="utf-8")
    m = X.report(clauses, plan, sl, narration, old, only, project=project)
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps(m, indent=1))
    print("spent", round(tracker.total_usd, 4), "cache hits", seats.hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
