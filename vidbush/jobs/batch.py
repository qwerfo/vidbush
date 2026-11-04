"""Batch queue: many briefs, scripts or voiceovers for one brand, run one after another.

State lives in data/batches/<batch_id>/state.json (BatchState); the report is written next to it
(batch_report.md and batch_report.json, see batch_report.py) every time an item changes.

- Discovery: a folder (non recursive, sorted by name) or a list file (one path per line, # comments,
  relative paths against the list file's folder). A .md or .txt next to an audio file with the same
  stem is that voiceover's sidecar (cli.detect_input picks it up), never a job of its own.
- Each item runs through the worker's pipeline path (api.jobs_handler.run_pipeline: create_project then
  run_project with the batch cost cap). A failed item is recorded and the batch moves on.
- Pause takes effect between projects: the running project finishes, then the runner stops.
  Resume continues from the first unfinished item; an interrupted item resumes its own project.
"""
from __future__ import annotations

import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from ..config import Settings
from ..util import read_model, write_json
from . import lock

ItemStatus = Literal["queued", "running", "done", "needs_review", "capped", "failed", "skipped"]
BatchStatus = Literal["queued", "running", "paused", "done", "interrupted"]
UNFINISHED = ("queued", "running")
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
TEXT_EXT = {".md", ".txt"}
_STATE_GUARD = threading.Lock()


class BatchItem(BaseModel):
    """One queued input and, once run, its report figures."""

    index: int
    input: str  # absolute path of the brief, script or voiceover
    mode: str = ""  # brief | script | voiceover ("" when the input could not be read)
    project_id: str = ""
    status: ItemStatus = "queued"
    note: str = ""  # error, cap reason or skip reason
    created: bool = False  # project.json written by this batch, so a rerun resumes instead of recreating
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_s: Optional[float] = None
    cost_usd: Optional[float] = None
    cost_per_min_usd: Optional[float] = None
    qa_pct: Optional[float] = None  # pct_above_threshold from qa_report.json


class BatchState(BaseModel):
    """data/batches/<id>/state.json: the queue, per item status and the paused flag."""

    id: str
    source: str
    brand: str = "default"
    profile: Optional[str] = None
    format: Optional[str] = None
    minutes: Optional[float] = None
    cost_cap_usd: Optional[float] = None  # None = no cap
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    paused: bool = False
    status: BatchStatus = "queued"
    items: list[BatchItem] = Field(default_factory=list)


# ---------------------------------------------------------------- paths and state io

def batches_dir(settings: Settings) -> Path:
    """data/batches/."""
    return settings.path(settings.data_folder) / "batches"


def batch_dir(settings: Settings, batch_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", batch_id or ""):
        raise ValueError(f"invalid batch id: {batch_id!r}")
    return batches_dir(settings) / batch_id


def load_batch(settings: Settings, batch_id: str) -> BatchState:
    """Read state.json. Raises FileNotFoundError for an unknown batch."""
    path = batch_dir(settings, batch_id) / "state.json"
    if not path.exists():
        raise FileNotFoundError(f"batch not found: {batch_id}")
    return read_model(path, BatchState)


def save_batch(settings: Settings, state: BatchState) -> None:
    """Write state.json and refresh the report next to it."""
    from .batch_report import write_report
    state.updated_at = time.time()
    write_json(batch_dir(settings, state.id) / "state.json", state)
    write_report(settings, state)


def update_batch(settings: Settings, batch_id: str, fn: Callable[[BatchState], None]) -> BatchState:
    """Read, change and write state.json in one step, so a pause written by another caller survives."""
    with _STATE_GUARD:
        state = load_batch(settings, batch_id)
        fn(state)
        save_batch(settings, state)
        return state


def list_batches(settings: Settings) -> list[BatchState]:
    """Every batch on disk, newest first."""
    out = []
    root = batches_dir(settings)
    for path in sorted(root.glob("*/state.json")) if root.exists() else []:
        try:
            out.append(read_model(path, BatchState))
        except ValueError:
            continue
    return sorted(out, key=lambda b: b.created_at, reverse=True)


def is_running(settings: Settings, batch_id: str) -> bool:
    """True while a live runner holds the batch lock (this or another process)."""
    try:
        lock.check_free(batch_dir(settings, batch_id), batch_id)
    except lock.ProjectBusy:
        return True
    return False


def effective_status(settings: Settings, state: BatchState) -> str:
    """The stored status, or interrupted when it says running but no runner holds the lock."""
    if state.status == "running" and not is_running(settings, state.id):
        return "interrupted"
    return state.status


# ---------------------------------------------------------------- discovery

def discover(source: Path) -> list[tuple[Path, str]]:
    """Inputs of a folder or a list file, in order, as (path, skip_note); skip_note "" means queue it."""
    if source.is_dir():
        paths = sorted((p for p in source.iterdir() if p.is_file() and not p.name.startswith(".")
                        and p.suffix.lower() in AUDIO_EXT | TEXT_EXT), key=lambda p: p.name)
        return [(p, "") for p in _drop_sidecars(paths)]
    out: list[tuple[Path, str]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line).expanduser()
        p = (p if p.is_absolute() else source.parent / p).resolve()
        if p.is_dir():
            out.extend(discover(p))
        elif not p.is_file():
            out.append((p, "file not found"))
        elif p.suffix.lower() not in AUDIO_EXT | TEXT_EXT:
            out.append((p, f"unsupported file type {p.suffix or '(none)'}"))
        else:
            out.append((p, ""))
    kept = set(_drop_sidecars([p for p, note in out if not note]))
    return [(p, note) for p, note in out if note or p in kept]


def sidecar_of(audio: Path) -> Optional[Path]:
    """The script that cli.detect_input pairs with a voiceover: <stem>.md, else <stem>.txt."""
    for ext in (".md", ".txt"):
        cand = audio.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def _drop_sidecars(paths: list[Path]) -> list[Path]:
    sidecars = {sidecar_of(p) for p in paths if p.suffix.lower() in AUDIO_EXT}
    return [p for p in paths if p not in sidecars]


def project_id_for(path: Path, prefix: str, taken: set[str]) -> str:
    """Id from the file stem the way cmd_run derives it, with an optional prefix, unique in `taken`."""
    base = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{prefix}{path.stem}")
    pid, n = base, 2
    while pid in taken:
        pid, n = f"{base}_{n}", n + 1
    taken.add(pid)
    return pid


def plan_items(source: Path, id_prefix: str = "") -> list[BatchItem]:
    """BatchItems for every discovered input, with the input mode read now (cli.detect_input)."""
    from ..cli import detect_input
    items: list[BatchItem] = []
    taken: set[str] = set()
    for i, (path, note) in enumerate(discover(source)):
        item = BatchItem(index=i, input=str(path), project_id=project_id_for(path, id_prefix, taken))
        if note:
            item.status, item.note = "skipped", note
        else:
            try:
                item.mode = detect_input(path, None)[0].value
            except Exception as e:  # noqa: BLE001 - unreadable or malformed input
                item.status, item.note = "skipped", f"cannot read input: {type(e).__name__}: {e}"[:300]
        items.append(item)
    return items


def create_batch(settings: Settings, source: str | Path, brand: str = "default", profile: Optional[str] = None,
                 fmt: Optional[str] = None, minutes: Optional[float] = None, cost_cap: Optional[float] = None,
                 id_prefix: str = "", batch_id: Optional[str] = None) -> BatchState:
    """Plan and save a new batch (nothing runs yet). cost_cap None = config batch.cost_cap_usd; <= 0 = no cap."""
    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"batch input not found: {src}")
    cap = settings.batch.cost_cap_usd if cost_cap is None else cost_cap
    bid = batch_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    if (batch_dir(settings, bid) / "state.json").exists():
        raise FileExistsError(f"batch {bid} already exists")
    state = BatchState(id=bid, source=str(src), brand=brand, profile=profile, format=fmt, minutes=minutes,
                       cost_cap_usd=cap if cap and cap > 0 else None, items=plan_items(src, id_prefix))
    with _STATE_GUARD:
        save_batch(settings, state)
    return state


# ---------------------------------------------------------------- running, pause and resume

def run_batch(settings: Settings, batch_id: str, echo: bool = True,
              on_item: Optional[Callable[[BatchState, BatchItem], None]] = None) -> BatchState:
    """Run queued items one by one until the queue is empty or the batch is paused.

    Holds data/batches/<id>/.lock so two runners never share a batch (raises lock.ProjectBusy).
    on_item(state, item) is called after each item (tests and progress output).
    """
    root = batch_dir(settings, batch_id)
    with lock.project_lock(root, batch_id, owner=f"run_batch {batch_id}"):
        update_batch(settings, batch_id, lambda s: setattr(s, "status", "running"))
        while True:
            state = load_batch(settings, batch_id)
            item = next(iter(i for i in state.items if i.status in UNFINISHED), None)
            if item is None:
                return update_batch(settings, batch_id, lambda s: setattr(s, "status", "done"))
            if state.paused:
                return update_batch(settings, batch_id, lambda s: setattr(s, "status", "paused"))
            lock.touch(root)
            item = run_item(settings, state, item, echo)
            if on_item:
                on_item(load_batch(settings, batch_id), item)


def run_item(settings: Settings, state: BatchState, item: BatchItem, echo: bool) -> BatchItem:
    """Create (first time) and run one item's project; record the outcome. Never raises on a failed run."""
    from ..api.jobs_handler import run_pipeline
    from .batch_report import project_metrics
    resume = item.created and _project_exists(settings, item.project_id)
    _set_item(settings, state.id, item.index, status="running", started_at=time.time(), finished_at=None,
              note="")
    try:
        if not resume:
            _create(settings, state, item)
            _set_item(settings, state.id, item.index, created=True)
        proj = run_pipeline(settings, item.project_id, {"cost_cap_usd": state.cost_cap_usd}, echo=echo)
        fields = outcome(settings, item.project_id, proj.status, proj.status_note)
    except Exception as e:  # noqa: BLE001 - one failed project never stops the batch
        fields = {"status": "failed", "note": f"{type(e).__name__}: {e}"[:500]}
        if echo:
            print(traceback.format_exc()[-2000:])
    fields.update(project_metrics(settings, item.project_id), finished_at=time.time())
    return _set_item(settings, state.id, item.index, **fields)


def outcome(settings: Settings, project_id: str, status: str, status_note: str = "") -> dict:
    """Item status and note from a finished run: capped beats needs_review when the cap stopped it."""
    from .costcap import read_cap
    from .pipeline import project_dir
    cap = read_cap(project_dir(settings, project_id) / "work")
    if cap:
        return {"status": "capped", "note": str(cap.get("reason", "cost cap reached"))}
    if status in ("done", "needs_review"):
        note = status_note or ("some seats used mock output; check qa_report.md"
                               if status == "needs_review" else "")
        return {"status": status, "note": note}
    return {"status": "failed", "note": status_note or f"project ended as {status}"}


def _create(settings: Settings, state: BatchState, item: BatchItem) -> None:
    """create_project for one input, the same call cmd_run makes."""
    from ..cli import detect_input
    from ..models import VideoFormat
    from .pipeline import create_project
    mode, kw = detect_input(Path(item.input), None)
    create_project(settings, item.project_id, mode, brand_id=state.brand,
                   fmt=VideoFormat(state.format) if state.format else None, target_minutes=state.minutes,
                   style_profile_id=state.profile, **kw)


def _project_exists(settings: Settings, project_id: str) -> bool:
    from .pipeline import project_dir
    return (project_dir(settings, project_id) / "project.json").exists()


def _set_item(settings: Settings, batch_id: str, index: int, **fields: object) -> BatchItem:
    def apply(state: BatchState) -> None:
        for k, v in fields.items():
            setattr(state.items[index], k, v)
    return update_batch(settings, batch_id, apply).items[index]


def pause_batch(settings: Settings, batch_id: str) -> BatchState:
    """Set the paused flag. A running project finishes first; the runner stops before the next one."""
    def apply(state: BatchState) -> None:
        state.paused = True
        if state.status != "done" and not is_running(settings, batch_id):
            state.status = "paused"
    return update_batch(settings, batch_id, apply)


def prepare_resume(settings: Settings, batch_id: str, retry_failed: bool = False,
                   cost_cap: Optional[float] = None) -> BatchState:
    """Clear the paused flag; optionally requeue failed and capped items and change the cap (<= 0 = none)."""
    def apply(state: BatchState) -> None:
        state.paused = False
        if cost_cap is not None:
            state.cost_cap_usd = cost_cap if cost_cap > 0 else None
        for item in state.items:
            if retry_failed and item.status in ("failed", "capped"):
                item.status, item.note = "queued", ""
        if state.status != "running" and any(i.status in UNFINISHED for i in state.items):
            state.status = "queued"
    return update_batch(settings, batch_id, apply)


def resume_batch(settings: Settings, batch_id: str, echo: bool = True, retry_failed: bool = False,
                 cost_cap: Optional[float] = None) -> BatchState:
    """Unpause and run from the first unfinished item (an interrupted item resumes its project)."""
    prepare_resume(settings, batch_id, retry_failed, cost_cap)
    return run_batch(settings, batch_id, echo=echo)
