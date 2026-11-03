"""Project path rebasing, so a project survives being moved between machines.

project.json and work/*.json store absolute paths (project root, candidate
files, overlay clips, segments, narration audio). When the project folder is
opened under another root (host versus Docker, a renamed repo folder, another
machine), every stored absolute path inside the old project root is rewritten
onto the current root when it is loaded. Paths outside the project (a shared
source cache, operator files) are left alone.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar

from pydantic import BaseModel

from ..util import read_json

T = TypeVar("T", bound=BaseModel)

# first path parts below output/<id>/ that only a project folder has
_PROJECT_PARTS = ("work", "sources", "thumbnails", "inputs")
_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def _is_abs(s: str) -> bool:
    return s.startswith("/") or bool(_WIN_ABS.match(s)) or s.startswith("\\\\")


def _norm(s: str) -> str:
    return s.replace("\\", "/").rstrip("/")


def rebase_path(s: str, project_id: str, new_root: str, old_roots: Iterable[str] = ()) -> str:
    """s moved onto new_root when it lies inside an old project root; unchanged otherwise."""
    if not s or not _is_abs(s):
        return s
    norm, root = _norm(s), _norm(new_root)
    if norm == root or norm.startswith(root + "/"):
        return s
    for old in old_roots:
        o = _norm(old)
        if o and o != root and (norm == o or norm.startswith(o + "/")):
            return _join(new_root, norm[len(o):])
    marker = f"/{project_id}"
    pos = norm.rfind(marker)
    while pos >= 0:
        rest = norm[pos + len(marker):]
        if rest == "" or _looks_like_project_rest(rest):
            return _join(new_root, rest)
        pos = norm.rfind(marker, 0, pos)
    return s


def _looks_like_project_rest(rest: str) -> bool:
    """'/work/x.json', '/sources/..', '/final.mp4': something directly inside a project folder."""
    if not rest.startswith("/"):
        return False
    parts = rest[1:].split("/")
    if parts[0] in _PROJECT_PARTS:
        return True
    return len(parts) == 1 and "." in parts[0]


def _join(new_root: str, rest: str) -> str:
    rest = rest.lstrip("/")
    return str(Path(new_root) / rest) if rest else str(Path(new_root))


def rebase_tree(data: Any, project_id: str, new_root: str, old_roots: Iterable[str] = ()) -> Any:
    """Copy of a JSON tree with every string path rebased (see rebase_path)."""
    olds = [o for o in old_roots if o]
    if isinstance(data, str):
        return rebase_path(data, project_id, new_root, olds)
    if isinstance(data, list):
        return [rebase_tree(v, project_id, new_root, olds) for v in data]
    if isinstance(data, dict):
        return {k: rebase_tree(v, project_id, new_root, olds) for k, v in data.items()}
    return data


def read_rebased(path: Path | str, model: type[T], project_id: str, new_root: str | Path,
                 old_roots: Iterable[str] = ()) -> T:
    """Read a model from JSON with its project paths rebased onto new_root."""
    data = rebase_tree(read_json(path), project_id, str(new_root), old_roots)
    return model.model_validate(data)


def stored_root(project_json: Path) -> Optional[str]:
    """paths.root as written in project.json (the root it was created or last saved under)."""
    try:
        return str(read_json(project_json).get("paths", {}).get("root") or "") or None
    except (OSError, ValueError, AttributeError):
        return None
