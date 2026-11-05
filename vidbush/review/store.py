"""output/<id>/review.json: one object per shot plus the project comment, with timestamps.

The operator watches final.mp4 and gives every shot a verdict, a free text comment (no length
limit) and a checklist of hard rejects. Each shot entry keeps a snapshot of what was on screen when
the operator last changed it (type, intent, selected source, judge score and reasons), so the export
and the calibration still describe the reviewed pick after a fix re-sources the shot.

Entries follow the current shot list order. A shot that left the shot list (merged, split) keeps its
entry with in_shotlist false; it is not counted in the progress total.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from pydantic import Field

from ..config import Settings
from ..models import SCHEMA_VERSION, Model, Shot, ShotList
from ..util import read_json, write_json

REVIEW_FILE = "review.json"
EXPORT_FILE = "review.md"

Verdict = Literal["good", "wrong_picture", "wrong_timing", "quality_problem", "other"]
VERDICTS: tuple[str, ...] = ("good", "wrong_picture", "wrong_timing", "quality_problem", "other")
VERDICT_LABELS = {"good": "good", "wrong_picture": "wrong picture", "wrong_timing": "wrong timing",
                  "quality_problem": "quality problem", "other": "other"}

# the review checklist; each maps onto the judge labels (or shot flags) that mean the same thing
REJECTS: tuple[str, ...] = ("burned_in_text", "watermark_logo", "talking_head", "anachronism", "repeated_clip")
REJECT_LABELS = {"burned_in_text": "burned-in text", "watermark_logo": "watermark or logo",
                 "talking_head": "talking head", "anachronism": "anachronism", "repeated_clip": "repeated clip"}
JUDGE_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "burned_in_text": ("burned_in_captions", "burned_in_text", "subtitle", "caption"),
    "watermark_logo": ("watermark", "channel_logo", "logo"),
    "talking_head": ("talking_head", "talking head"),
    "anachronism": ("anachronism", "anachronistic"),
    "repeated_clip": ("segment_repeat", "repeats a used segment", "repeated"),
}

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class RejectedCandidate(Model):
    id: str
    source: str
    reason: str


class ShotSnapshot(Model):
    """What the shot was when the operator last changed its review."""
    type: str
    start: float
    end: float
    duration: float
    intent: str = ""
    selected: Optional[str] = None
    source: str = ""  # source kind of the selected candidate (wikimedia, youtube, ai, motion ...)
    site: str = ""  # host of the selected candidate's url, else the source kind
    licence: str = ""
    title: str = ""
    url: str = ""
    judge_score: Optional[float] = None
    judge_reason: str = ""  # the selected candidate's judge reason
    judge_notes: str = ""  # the shot's judge notes
    selected_reject: Optional[str] = None  # reject reason on the pick itself (best available accepted)
    flags: list[str] = Field(default_factory=list)  # shot flags plus the pick's flags
    rejected: list[RejectedCandidate] = Field(default_factory=list)  # pre-filter and judge rejects


class FixRecord(Model):
    op: str
    job_id: Optional[int] = None
    applied_at: float


class ShotReview(Model):
    shot_id: str
    index: int = 0
    in_shotlist: bool = True
    verdict: Optional[Verdict] = None
    comment: str = ""  # no length limit, by the operator's rule
    rejects: list[str] = Field(default_factory=list)
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    snapshot: Optional[ShotSnapshot] = None
    fix: Optional[FixRecord] = None

    @property
    def reviewed(self) -> bool:
        return self.verdict is not None

    @property
    def touched(self) -> bool:
        return self.updated_at is not None


class Review(Model):
    schema_version: str = SCHEMA_VERSION
    project_id: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    project_comment: str = ""  # no length limit
    project_comment_updated_at: Optional[float] = None
    shots: list[ShotReview] = Field(default_factory=list)

    def entry(self, shot_id: str) -> ShotReview:
        for e in self.shots:
            if e.shot_id == shot_id:
                return e
        raise KeyError(shot_id)

    def current(self) -> list[ShotReview]:
        return [e for e in self.shots if e.in_shotlist]

    @property
    def total(self) -> int:
        return len(self.current())

    @property
    def reviewed(self) -> int:
        return sum(1 for e in self.current() if e.reviewed)


class ReviewError(ValueError):
    """Bad review input (unknown shot, verdict or reject); the API turns it into a 4xx."""


# ---------------------------------------------------------------- paths and loading

def project_root(settings: Settings, project_id: str) -> Path:
    from ..jobs.pipeline import project_dir
    return project_dir(settings, project_id)


def review_path(settings: Settings, project_id: str) -> Path:
    return project_root(settings, project_id) / REVIEW_FILE


def exists(settings: Settings, project_id: str) -> bool:
    return review_path(settings, project_id).exists()


def load_shotlist(settings: Settings, project_id: str) -> Optional[ShotList]:
    """work/shotlist.json, else the published shotlist.json, paths rebased onto the project root."""
    from ..jobs.paths import read_rebased, stored_root
    root = project_root(settings, project_id)
    old = stored_root(root / "project.json")
    for p in (root / "work" / "shotlist.json", root / "shotlist.json"):
        if p.exists():
            return read_rebased(p, ShotList, project_id, root, [old] if old else [])
    return None


def _lock(project_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(project_id, threading.Lock())


def snapshot(shot: Shot) -> ShotSnapshot:
    sel = shot.selected_candidate()
    url = sel.url if sel else ""
    host = (urlparse(url).hostname or "") if url else ""
    flags = list(dict.fromkeys([*shot.flags, *(sel.flags if sel else [])]))
    rejected = [RejectedCandidate(id=c.id, source=c.source.value, reason=c.reject_reason)
                for c in shot.candidates if c.reject_reason and (not sel or c.id != sel.id)]
    return ShotSnapshot(
        type=shot.type.value, start=shot.start, end=shot.end, duration=shot.duration, intent=shot.intent,
        selected=shot.selected, source=sel.source.value if sel else "", site=host or (sel.source.value if sel else ""),
        licence=sel.licence if sel else "", title=sel.title if sel else "", url=url,
        judge_score=shot.judge_score, judge_reason=sel.judge_reason if sel else "", judge_notes=shot.judge_notes,
        selected_reject=sel.reject_reason if sel else None, flags=flags, rejected=rejected)


def reconcile(review: Review, shotlist: Optional[ShotList]) -> Review:
    """Order entries by the shot list, add missing shots, mark entries whose shot is gone."""
    if shotlist is None:
        return review
    by_id = {e.shot_id: e for e in review.shots}
    ordered: list[ShotReview] = []
    for s in shotlist.shots:
        e = by_id.pop(s.id, None) or ShotReview(shot_id=s.id)
        e.index, e.in_shotlist = s.index, True
        ordered.append(e)
    for e in by_id.values():
        e.in_shotlist = False
        ordered.append(e)
    review.shots = ordered
    return review


def load(settings: Settings, project_id: str, shotlist: Optional[ShotList] = None) -> Review:
    """review.json reconciled with the current shot list (an empty review when there is no file yet)."""
    path = review_path(settings, project_id)
    if path.exists():
        review = Review.model_validate(read_json(path))
    else:
        review = Review(project_id=project_id)
    return reconcile(review, shotlist if shotlist is not None else load_shotlist(settings, project_id))


def save(settings: Settings, review: Review) -> Path:
    review.updated_at = time.time()
    return write_json(review_path(settings, review.project_id), review)


# ---------------------------------------------------------------- changes

_UNSET: Any = object()


def _clean_rejects(rejects: Any) -> list[str]:
    if rejects is None:
        return []
    if isinstance(rejects, str):
        rejects = [r for r in rejects.replace(";", ",").split(",")]
    out: list[str] = []
    for r in rejects:
        key = str(r).strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            continue
        if key not in REJECTS:
            raise ReviewError(f"unknown hard reject {r!r}; use one of {', '.join(REJECTS)}")
        if key not in out:
            out.append(key)
    return [r for r in REJECTS if r in out]


def _clean_verdict(verdict: Any) -> Optional[str]:
    if verdict is None or verdict == "":
        return None
    key = str(verdict).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in VERDICTS:
        raise ReviewError(f"unknown verdict {verdict!r}; use one of {', '.join(VERDICTS)}")
    return key


def update_shot(settings: Settings, project_id: str, shot_ref: Any, verdict: Any = _UNSET, comment: Any = _UNSET,
                rejects: Any = _UNSET) -> tuple[Review, ShotReview]:
    """Change one shot's review (fields left out stay as they are) and save. verdict None clears it.

    shot_ref is a shot id or the 1-based number shown to the operator.
    """
    with _lock(project_id):
        sl = load_shotlist(settings, project_id)
        if sl is None:
            raise ReviewError(f"project {project_id} has no shot list yet")
        shot = _resolve(sl, shot_ref)
        review = load(settings, project_id, sl)
        e = review.entry(shot.id)
        if verdict is not _UNSET:
            e.verdict = _clean_verdict(verdict)  # type: ignore[assignment]
        if comment is not _UNSET:
            e.comment = "" if comment is None else str(comment)
        if rejects is not _UNSET:
            e.rejects = _clean_rejects(rejects)
        now = time.time()
        e.created_at = e.created_at or now
        e.updated_at = now
        e.snapshot = snapshot(shot)
        save(settings, review)
        return review, e


def set_project_comment(settings: Settings, project_id: str, comment: str) -> Review:
    with _lock(project_id):
        review = load(settings, project_id)
        review.project_comment = "" if comment is None else str(comment)
        review.project_comment_updated_at = time.time()
        save(settings, review)
        return review


def mark_fixed(settings: Settings, project_id: str, ops: list[dict[str, Any]], job_id: Optional[int]) -> Review:
    """Record on each shot entry that a review fix op was queued for it."""
    with _lock(project_id):
        review = load(settings, project_id)
        now = time.time()
        for op in ops:
            sid = str((op.get("args") or {}).get("shot_id") or "")
            try:
                e = review.entry(sid)
            except KeyError:
                continue
            e.fix = FixRecord(op=str(op.get("op")), job_id=job_id, applied_at=now)
        save(settings, review)
        return review


def _resolve(sl: ShotList, ref: Any) -> Shot:
    from ..editor.retime import EditError, resolve_shot
    try:
        return resolve_shot(sl, ref)
    except EditError as e:
        raise ReviewError(str(e)) from e


def progress(settings: Settings, project_id: str) -> Optional[dict[str, int]]:
    """{"reviewed": n, "total": m} once review.json exists, else None."""
    if not exists(settings, project_id):
        return None
    review = load(settings, project_id)
    return {"reviewed": review.reviewed, "total": review.total}
