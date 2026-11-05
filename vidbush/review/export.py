"""Review summary and the review.md export, written so an AI can read it cold and act on it."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..util import no_em_dash
from . import store
from .store import REJECT_LABELS, REJECTS, VERDICT_LABELS, VERDICTS, Review, ShotReview


def clock(sec: Optional[float]) -> str:
    """m:ss.s, the format the UI shows next to every shot."""
    if sec is None:
        return "-"
    s = round(max(0.0, float(sec)), 1)  # round first so 59.96 reads 1:00.0, not 0:60.0
    m = int(s // 60)
    r = s - m * 60
    return f"{m}:{r:04.1f}"


def _num(e: ShotReview) -> int:
    return e.index + 1


def _start(e: ShotReview) -> Optional[float]:
    return e.snapshot.start if e.snapshot else None


def summary(review: Review) -> dict[str, Any]:
    """Counts the export and the CLI print: progress, verdicts, hard rejects with timestamps, comments."""
    cur = review.current()
    counts = Counter(e.verdict for e in cur if e.verdict)
    flagged = [{"shot_id": e.shot_id, "number": _num(e), "start": _start(e), "time": clock(_start(e)),
                "rejects": list(e.rejects), "verdict": e.verdict}
               for e in cur if e.rejects]
    reject_counts = Counter(r for e in cur for r in e.rejects)
    return {
        "project_id": review.project_id,
        "reviewed": review.reviewed, "total": review.total, "unreviewed": review.total - review.reviewed,
        "verdicts": {v: counts.get(v, 0) for v in VERDICTS},
        "rejects": {r: reject_counts.get(r, 0) for r in REJECTS},
        "flagged": flagged,
        "comments": sum(1 for e in cur if e.comment.strip()),
        "has_project_comment": bool(review.project_comment.strip()),
        "updated_at": review.updated_at,
    }


def summary_text(review: Review) -> str:
    """Short plain text for `review-summary`."""
    s = summary(review)
    lines = [f"Review of {review.project_id}: reviewed {s['reviewed']} of {s['total']} shots"]
    lines.append("Verdicts: " + ", ".join(f"{VERDICT_LABELS[v]} {n}" for v, n in s["verdicts"].items())
                 + f", not reviewed {s['unreviewed']}")
    if s["flagged"]:
        lines.append(f"Hard rejects on {len(s['flagged'])} shot(s):")
        for f in s["flagged"]:
            lines.append(f"  shot {f['number']} at {f['time']}: " + ", ".join(REJECT_LABELS[r] for r in f["rejects"]))
    else:
        lines.append("Hard rejects: none ticked")
    lines.append(f"Shots with a comment: {s['comments']}; project comment: {'yes' if s['has_project_comment'] else 'no'}")
    return "\n".join(lines)


def _quote(text: str) -> list[str]:
    """Comment text as a markdown blockquote, every line kept, nothing cut."""
    return ["> " + ln if ln.strip() else ">" for ln in no_em_dash(text.strip()).splitlines()]


def _cell(text: Any) -> str:
    return no_em_dash(str(text)).replace("|", "/").replace("\n", " ").strip()


def _shot_block(e: ShotReview) -> list[str]:
    sn = e.snapshot
    verdict = VERDICT_LABELS.get(e.verdict or "", "not reviewed")
    head = f"### Shot {_num(e)} ({e.shot_id}) at {clock(_start(e))}: {verdict}"
    out = [head, ""]
    if sn:
        out.append(f"- Type: {sn.type}, {sn.duration:.1f} s ({clock(sn.start)} to {clock(sn.end)})")
        if sn.intent:
            out.append(f"- Intent: {_cell(sn.intent)}")
        if sn.selected:
            src = sn.site or sn.source
            out.append(f"- Selected: {sn.selected} from {src}, licence {_cell(sn.licence or 'unknown')}"
                       + (f", title {_cell(sn.title)}" if sn.title else ""))
        score = "none" if sn.judge_score is None else f"{sn.judge_score:.0f}"
        reason = sn.judge_reason or sn.judge_notes
        out.append(f"- Judge score: {score}" + (f", reason: {_cell(reason)}" if reason else ""))
        if sn.flags:
            out.append(f"- Flags: {', '.join(sn.flags)}")
    if e.rejects:
        out.append("- Hard rejects ticked: " + ", ".join(REJECT_LABELS[r] for r in e.rejects))
    if e.fix:
        out.append(f"- Fix already queued: {e.fix.op}" + (f" (job {e.fix.job_id})" if e.fix.job_id else ""))
    out += ["", "Comment:", ""] + _quote(e.comment) + [""]
    return out


def markdown(review: Review, title: str = "", threshold: Optional[float] = None) -> str:
    """The review.md text: summary table, hard rejects table, then every commented shot in order."""
    s = summary(review)
    name = title or review.project_id
    lines = [f"# Review: {_cell(name)}", "",
             f"Project id `{review.project_id}`. Reviewed {s['reviewed']} of {s['total']} shots."
             + (f" Judge threshold {threshold:.0f}." if threshold is not None else ""), "",
             "How to read this: the operator watched final.mp4 and marked each shot. Verdicts are good, "
             "wrong picture (the visual does not fit the narration), wrong timing (the cut is early, late, "
             "too long or too short), quality problem (the picture is fine in topic but bad in quality, see "
             "the ticked hard rejects) and other (read the comment). Shot numbers are 1-based as shown in "
             "the UI; times are m:ss.s into final.mp4. Comments are the operator's own words, unedited, and "
             "they outrank the judge score.", ""]
    lines += ["## Summary", "", "| Verdict | Shots |", "|---|---|"]
    lines += [f"| {VERDICT_LABELS[v]} | {n} |" for v, n in s["verdicts"].items()]
    lines += [f"| not reviewed | {s['unreviewed']} |", ""]
    lines += ["## Hard rejects", ""]
    if s["flagged"]:
        lines += ["| Shot | Time | Hard rejects | Verdict |", "|---|---|---|---|"]
        for f in s["flagged"]:
            lines.append(f"| {f['number']} ({f['shot_id']}) | {f['time']} | "
                         f"{', '.join(REJECT_LABELS[r] for r in f['rejects'])} | "
                         f"{VERDICT_LABELS.get(f['verdict'] or '', 'not reviewed')} |")
        totals = ", ".join(f"{REJECT_LABELS[r]} {n}" for r, n in s["rejects"].items() if n)
        lines += ["", f"Totals: {totals}.", ""]
    else:
        lines += ["None ticked.", ""]
    lines += ["## Project comment", ""]
    lines += (_quote(review.project_comment) if review.project_comment.strip() else ["(none)"]) + [""]
    commented = [e for e in review.current() if e.comment.strip()]
    lines += [f"## Shots with a comment ({len(commented)})", ""]
    if not commented:
        lines += ["(none)", ""]
    for e in commented:
        lines += _shot_block(e)
    silent = [e for e in review.current() if e.verdict and e.verdict != "good" and not e.comment.strip()]
    if silent:
        lines += [f"## Shots marked not good without a comment ({len(silent)})", "",
                  "| Shot | Time | Verdict | Type | Judge score | Hard rejects |", "|---|---|---|---|---|---|"]
        for e in silent:
            sn = e.snapshot
            score = "none" if not sn or sn.judge_score is None else f"{sn.judge_score:.0f}"
            lines.append(f"| {_num(e)} ({e.shot_id}) | {clock(_start(e))} | {VERDICT_LABELS[e.verdict or 'other']} | "
                         f"{sn.type if sn else '-'} | {score} | "
                         f"{', '.join(REJECT_LABELS[r] for r in e.rejects) or '-'} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_export(settings: Settings, project_id: str) -> Path:
    """Write output/<id>/review.md from review.json and return its path."""
    from ..jobs.pipeline import load_project
    review = store.load(settings, project_id)
    try:
        title = load_project(settings, project_id).title
    except Exception:  # noqa: BLE001  the export must not fail on an unreadable project.json
        title = ""
    path = store.project_root(settings, project_id) / store.EXPORT_FILE
    path.write_text(markdown(review, title, float(settings.judge.threshold)), encoding="utf-8")
    return path
