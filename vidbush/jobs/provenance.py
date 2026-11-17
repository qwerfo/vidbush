"""Run provenance: every run records the git commit and the settings it ran with.

Principle: a result is only comparable with another when you know which code and which settings made it.
start_run() appends a RunRecord to project.runs; each stage attempt carries the commit too
(models.StageAttempt), so a project resumed on newer code shows which attempt ran on which commit.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ..models import RunRecord


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def code_version(root: Path) -> dict[str, Any]:
    """commit, branch and whether tracked files differ from the commit ("" when not a git checkout)."""
    commit = _git(root, "rev-parse", "HEAD")
    return {"commit": commit, "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(commit) and bool(_git(root, "status", "--porcelain", "--untracked-files=no"))}


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def settings_record(ctx: Any) -> dict[str, Any]:
    """What decides a run's output besides the code: seat models, config file hashes and switches.
    Never key values: only provider names, model ids and hashes of the config files."""
    s = ctx.settings
    cfg = Path(s.config_folder)
    from ..providers.seats import ROLES
    seats = {}
    for role in ROLES:
        try:
            seats[role] = [f"{p.name}:{p.model}" for p in ctx.seats.providers_for(role) if p.configured] or ["mock"]
        except Exception:  # noqa: BLE001  provenance never stops a run
            seats[role] = ["unknown"]
    return {
        "seats": seats,
        "config_sha": {name: _file_sha(cfg / name) for name in ("vidbush.yaml", "seats.yaml")},
        "style_profile": ctx.profile.id,
        "style_profile_sha": _file_sha(cfg / "style_profiles" / f"{ctx.profile.id}.yaml"),
        "brand": ctx.brand.id,
        "brand_sha": _file_sha(cfg / "brands" / f"{ctx.brand.id}.yaml"),
        "seed": ctx.project.seed,
        "offline": s.offline,
        "allow_mock_fallback": s.allow_mock_fallback,
        "reply_cache": ctx.seats.reply_cache.enabled,
        "judge_threshold": s.judge.threshold,
        "python": sys.version.split()[0],
    }


def start_run(ctx: Any, force: list[str], cost_cap: Optional[float]) -> RunRecord:
    """Append and return this run's record (saved with the project by the caller)."""
    ver = code_version(Path(ctx.settings.repo_root))
    rec = RunRecord(started_at=time.time(), commit=ver["commit"], branch=ver["branch"], dirty=ver["dirty"],
                    argv=list(sys.argv[1:]), force=force, cost_cap=cost_cap, settings=settings_record(ctx))
    ctx.project.runs.append(rec)
    ctx.log.info("run start", commit=rec.commit[:12], branch=rec.branch, dirty=rec.dirty, cost_cap=cost_cap)
    return rec


def finish_run(rec: RunRecord, status: str) -> None:
    rec.finished_at, rec.status = time.time(), status
