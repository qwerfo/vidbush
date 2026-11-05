"""Review verdicts to EditOps, only where the mapping is safe.

- wrong picture on footage or a still: change_type to the same type, which is the editor path that
  re-sources one shot (editor/ops.py _resource). The comment is added to the shot's must_avoid and
  queries, and the reviewed pick is rejected so the judge cannot pick it again.
- wrong picture on an AI image or AI video: regenerate_ai with the comment as an avoid note.
- quality problem with a hard reject ticked: the same re-source, with each ticked reject added to
  must_avoid instead of the comment.
- wrong timing: no op. The plan lists it so the UI can open the existing duration editor.
- everything else (good, other, quality problem with nothing ticked, any verdict on a motion graphic)
  stays a comment.

A shot whose fix was queued after its last review change is left out, so pressing Apply twice does
not re-source the same shot twice.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..config import Settings
from ..models import EditOp, EditOpName, Shot, ShotList, ShotType
from ..util import no_em_dash
from . import store
from .store import REJECT_LABELS, VERDICT_LABELS, Review, ShotReview

RESOURCE_TYPES = {ShotType.footage, ShotType.footage_overlay, ShotType.image, ShotType.image_overlay}
AI_TYPES = {ShotType.ai_image}  # regenerate_ai makes a still, so an ai_video shot stays a comment
QUERY_WORDS = 12  # a search query built from a comment keeps this many words
AVOID_PROMPT_CHARS = 300  # an image prompt avoid note keeps this many characters

REJECT_AVOID = {"burned_in_text": "burned-in text, subtitles or captions", "watermark_logo": "watermark or channel logo",
                "talking_head": "talking head or presenter to camera", "anachronism": "anachronism, modern objects",
                "repeated_clip": "the clip already used elsewhere in this video"}


def _flat(text: str) -> str:
    return " ".join(no_em_dash(text).split())


_WANT_RE = re.compile(r"\b(?:i want|we want|want|wanted|i need|we need|need|needs|should show|should be|show|use|"
                      r"instead(?: of that)?(?: use| show)?|prefer|looking for|replace (?:it|this) with)\b[:,]?\s+(.*)",
                      re.I)
_STOP_RE = re.compile(r"\b(?:no|not|without|but not|avoid|never)\b.*$", re.I)


def comment_query(comment: str) -> str:
    """A search query from what the comment asks for ("I want steam shovels, mud" -> "steam shovels, mud").

    Only the wanted part becomes a query: a complaint ("this shows a modern ship") would search for the
    very thing that was rejected, so a comment without a want phrase adds no query (it still goes to
    must_avoid). At most QUERY_WORDS words, no quote marks, negations cut off.
    """
    for sentence in re.split(r"(?<=[.!?;])\s+", _flat(comment)):
        m = _WANT_RE.search(sentence)
        if not m:
            continue
        text = _STOP_RE.sub("", re.sub(r"[\"'`]", "", m.group(1))).strip(" ,.!?;:")
        words = text.split()
        if words:
            return " ".join(words[:QUERY_WORDS])
    return ""


def _fix_failed(settings: Optional[Settings], project_id: str, e: ShotReview) -> bool:
    """True when the job that carried this entry's fix failed, or its edit plan lists an error for the shot."""
    if settings is None or e.fix is None or e.fix.job_id is None:
        return False
    try:
        from sqlmodel import Session

        from ..jobs import runner
        with Session(runner.engine_for(settings)) as s:
            job = s.get(runner.Job, e.fix.job_id)
        if job is not None and job.status == "failed":
            return True
        from ..editor.ops import read_plan
        plan_ = read_plan(settings, project_id)
        if plan_.get("job_id") == e.fix.job_id:
            return any(e.shot_id in str(err) for err in plan_.get("errors", []))
    except Exception:  # noqa: BLE001  unknown job state: trust the record
        return False
    return False


def _pending(e: ShotReview, settings: Optional[Settings] = None, project_id: str = "") -> bool:
    """True when this entry has no fix queued since its verdict or rejects last changed (or that fix failed)."""
    judged = e.judged_at or e.updated_at
    if not (e.fix and judged is not None and e.fix.applied_at >= judged):
        return True
    return _fix_failed(settings, project_id, e)


def _resource_op(shot: Shot, avoid: list[str], queries: list[str], reason: str, why: str) -> EditOp:
    args: dict[str, Any] = {"shot_id": shot.id, "type": shot.type.value, "add_must_avoid": avoid,
                            "reject_selected": reason}
    if queries:
        args["add_queries"] = queries
    return EditOp(op=EditOpName.change_type, args=args, reason=why)


def _ai_op(shot: Shot, avoid: str, why: str) -> EditOp:
    note = _flat(avoid)[:AVOID_PROMPT_CHARS].rstrip(" ,.")
    return EditOp(op=EditOpName.regenerate_ai, args={"shot_id": shot.id, "prompt_delta": f"Avoid: {note}"}, reason=why)


def op_for(shot: Shot, e: ShotReview) -> tuple[Optional[EditOp], str]:
    """(op or None, why). None means the review stays a comment (or, for wrong timing, an editor action)."""
    n = shot.index + 1
    comment = _flat(e.comment)
    if e.verdict == "wrong_picture":
        if shot.type in RESOURCE_TYPES:
            avoid = [comment] if comment else [f"the picture rejected in review ({shot.selected or 'none'})"]
            queries = [q for q in [comment_query(comment)] if q]
            why = f"review: shot {n} wrong picture" + (f": {comment}" if comment else "")
            return _resource_op(shot, avoid, queries, "operator review: wrong picture", why), why
        if shot.type in AI_TYPES:
            why = f"review: shot {n} wrong picture, new AI image"
            return _ai_op(shot, comment or "the previous image, it did not fit the narration", why), why
        if shot.type == ShotType.ai_video:
            return None, "AI video: regenerating would turn it into a still, edit by hand"
        return None, "motion graphic: a wrong picture needs a hand edit"
    if e.verdict == "quality_problem":
        if not e.rejects:
            return None, "quality problem with no hard reject ticked stays a comment"
        labels = [REJECT_LABELS[r] for r in e.rejects]
        avoid = [REJECT_AVOID[r] for r in e.rejects]
        why = f"review: shot {n} quality problem ({', '.join(labels)})"
        if shot.type in RESOURCE_TYPES:
            return _resource_op(shot, avoid, [], f"operator review: {', '.join(labels)}", why), why
        if shot.type in AI_TYPES:
            return _ai_op(shot, ", ".join(avoid), why), why
        if shot.type == ShotType.ai_video:
            return None, "AI video: regenerating would turn it into a still, edit by hand"
        return None, "motion graphic: a quality problem needs a hand edit"
    if e.verdict == "wrong_timing":
        return None, "wrong timing: open the duration editor"
    return None, "stays a comment"


def plan(settings: Settings, project_id: str, review: Optional[Review] = None,
         shotlist: Optional[ShotList] = None) -> dict[str, Any]:
    """{"ops": [EditOp json], "timing": [...], "comments": [...], "skipped": [...]} for the current review."""
    sl = shotlist or store.load_shotlist(settings, project_id)
    if sl is None:
        raise store.ReviewError(f"project {project_id} has no shot list yet")
    review = review or store.load(settings, project_id, sl)
    by_id = {s.id: s for s in sl.shots}
    ops: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    comments: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for e in review.current():
        shot = by_id.get(e.shot_id)
        if shot is None or not e.verdict or e.verdict == "good":
            continue
        row = {"shot_id": shot.id, "number": shot.index + 1, "start": shot.start, "duration": shot.duration,
               "type": shot.type.value, "verdict": e.verdict, "verdict_label": VERDICT_LABELS[e.verdict],
               "comment": e.comment, "rejects": list(e.rejects)}
        if not _pending(e, settings, project_id):
            skipped.append({**row, "why": f"fix already queued ({e.fix.op if e.fix else ''})"})
            continue
        op, why = op_for(shot, e)
        if op is not None:
            ops.append(op.model_dump(mode="json"))
        elif e.verdict == "wrong_timing":
            timing.append({**row, "why": why})
        else:
            comments.append({**row, "why": why})
    return {"project_id": project_id, "ops": ops, "timing": timing, "comments": comments, "skipped": skipped}


def plan_text(p: dict[str, Any]) -> str:
    """Plain text of a plan for the CLI."""
    lines = [f"Planned edit ops: {len(p['ops'])}"]
    for i, o in enumerate(p["ops"], 1):
        args = ", ".join(f"{k}={v}" for k, v in o["args"].items())
        lines.append(f"  {i}. {o['op']} {args}")
    if p["timing"]:
        lines.append("Wrong timing (open the duration editor, or: python -m vidbush duration <project> <shot> <seconds>):")
        lines += [f"  shot {t['number']} at {t['start']:.1f} s, now {t['duration']:.2f} s" for t in p["timing"]]
    if p["comments"]:
        lines.append(f"Stays a comment: {len(p['comments'])}")
        lines += [f"  shot {c['number']}: {c['verdict_label']} ({c['why']})" for c in p["comments"]]
    if p["skipped"]:
        lines.append(f"Already queued: {len(p['skipped'])}")
    return "\n".join(lines)
