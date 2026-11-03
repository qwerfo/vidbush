"""SQLite-backed job table and an in-process worker pool. No Redis, no Celery."""
from __future__ import annotations

import json
import threading
import time
import traceback
from typing import Any, Callable, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from ..config import Settings


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: str = Field(index=True)
    kind: str  # run | resume | rerender | revise | apply_ops
    status: str = "queued"  # queued | running | done | failed
    payload: str = "{}"
    error: str = ""
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class ProjectRow(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str = ""
    status: str = "created"
    current_stage: str = ""
    input_mode: str = ""
    format: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


_ENGINES: dict[str, Any] = {}


def engine_for(settings: Settings):
    path = settings.path(settings.data_folder) / "vidbush.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path)
    if key not in _ENGINES:
        eng = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(eng)
        _ENGINES[key] = eng
    return _ENGINES[key]


def upsert_project_row(settings: Settings, project: Any) -> None:
    with Session(engine_for(settings)) as s:
        row = s.get(ProjectRow, project.id) or ProjectRow(id=project.id)
        row.title = project.title or project.id
        row.status = project.status
        row.current_stage = project.current_stage or ""
        row.input_mode = project.input_mode.value
        row.format = project.format.value
        row.created_at = project.created_at
        row.updated_at = time.time()
        s.add(row)
        s.commit()


def enqueue(settings: Settings, project_id: str, kind: str, payload: Optional[dict[str, Any]] = None) -> int:
    with Session(engine_for(settings)) as s:
        job = Job(project_id=project_id, kind=kind, payload=json.dumps(payload or {}))
        s.add(job)
        s.commit()
        s.refresh(job)
        return int(job.id)


def list_jobs(settings: Settings, project_id: Optional[str] = None, limit: int = 50) -> list[Job]:
    with Session(engine_for(settings)) as s:
        q = select(Job).order_by(Job.id.desc()).limit(limit)
        if project_id:
            q = select(Job).where(Job.project_id == project_id).order_by(Job.id.desc()).limit(limit)
        return list(s.exec(q))


def _claim(settings: Settings, busy: set[str]) -> Optional[Job]:
    """Take the oldest queued job whose project is not already being worked on."""
    with Session(engine_for(settings)) as s:
        for job in s.exec(select(Job).where(Job.status == "queued").order_by(Job.id)):
            if job.project_id in busy:
                continue
            job.status, job.started_at = "running", time.time()
            s.add(job)
            s.commit()
            s.refresh(job)
            return job
    return None


def _finish(settings: Settings, job_id: int, ok: bool, error: str = "") -> None:
    with Session(engine_for(settings)) as s:
        job = s.get(Job, job_id)
        if job:
            job.status = "done" if ok else "failed"
            job.error = error[:4000]
            job.finished_at = time.time()
            s.add(job)
            s.commit()


class WorkerPool:
    """Threads pulling jobs from the table. One job per project at a time."""

    def __init__(self, settings: Settings, handler: Callable[[Settings, Job], None], workers: int = 2) -> None:
        self.settings = settings
        self.handler = handler
        self.workers = workers
        self._stop = threading.Event()
        self._busy: set[str] = set()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self._recover()
        for i in range(self.workers):
            t = threading.Thread(target=self._loop, name=f"vidbush-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()

    def _recover(self) -> None:
        """Jobs left running by a crash go back to the queue; the pipeline resumes them."""
        with Session(engine_for(self.settings)) as s:
            for job in s.exec(select(Job).where(Job.status == "running")):
                job.status = "queued"
                s.add(job)
            s.commit()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job = _claim(self.settings, self._busy)
                if job:
                    self._busy.add(job.project_id)
            if job is None:
                time.sleep(0.5)
                continue
            try:
                self.handler(self.settings, job)
                _finish(self.settings, int(job.id), True)
            except Exception as e:  # noqa: BLE001
                _finish(self.settings, int(job.id), False, f"{e}\n{traceback.format_exc()[-2000:]}")
            finally:
                with self._lock:
                    self._busy.discard(job.project_id)
