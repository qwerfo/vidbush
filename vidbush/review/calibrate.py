"""Judge calibration from a review: where the judge score and the operator's verdict disagree.

Only verdicts about the picture count: good against wrong picture and quality problem. Wrong timing
and other say nothing about the judge, and shots without a judge score (motion graphics) are skipped.
Scores come from each entry's snapshot, i.e. the pick the operator actually reviewed.

Suggested threshold: the whole number t in 0..100 with the fewest disagreements, where a disagreement
is a wrong shot scored at or above t or a good shot scored below t. Ties go to the t closest to the
current threshold, then to the stricter (higher) one. With nothing labelled it is the current one.

Missed rejects: a ticked hard reject the judge did not report on the reviewed pick (its reject
reason, judge reason or flags carry none of the equivalent judge labels).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from ..config import Settings
from . import store
from .store import JUDGE_EQUIVALENTS, REJECT_LABELS, REJECTS, Review, ShotReview

WRONG = ("wrong_picture", "quality_problem")
PREFILTER_HINT = {"burned_in_text": "judge.prefilter text (text_min_keyframes, text_min_width, text_min_confidence)",
                  "watermark_logo": "judge.prefilter logo (logo_min_area, logo_max_diff); semi-transparent marks need the vision judge",
                  "talking_head": "judge.prefilter face (face_min_area, face_min_share)",
                  "anachronism": "the judge prompt and the shot must_avoid (no local check exists)",
                  "repeated_clip": "selection dedupe (judge/director_loop.py segment rule)"}


def _scored(e: ShotReview) -> Optional[float]:
    return e.snapshot.judge_score if e.snapshot and e.snapshot.judge_score is not None else None


def _row(e: ShotReview) -> dict[str, Any]:
    sn = e.snapshot
    return {"shot_id": e.shot_id, "number": e.index + 1, "start": sn.start if sn else None,
            "verdict": e.verdict, "score": _scored(e), "type": sn.type if sn else "",
            "reason": (sn.selected_reject or sn.judge_reason or sn.judge_notes) if sn else "",
            "rejects": list(e.rejects), "comment": e.comment}


def errors_at(points: list[tuple[float, bool]], t: float) -> int:
    """Disagreements at threshold t; points are (score, operator said good)."""
    return sum(1 for s, good in points if (s >= t) != good)


def suggest_threshold(points: list[tuple[float, bool]], current: float) -> float:
    if not points:
        return float(current)
    best = min(range(0, 101), key=lambda t: (errors_at(points, t), abs(t - current), -t))
    return float(best)


def _caught(e: ShotReview, reject: str) -> bool:
    sn = e.snapshot
    if sn is None:
        return False
    text = " ".join([sn.selected_reject or "", sn.judge_reason, sn.judge_notes, " ".join(sn.flags)]).lower()
    return any(k.lower() in text for k in JUDGE_EQUIVALENTS[reject])


def calibrate(review: Review, threshold: float) -> dict[str, Any]:
    cur = [e for e in review.current() if e.verdict]
    labelled = [e for e in cur if e.verdict in WRONG + ("good",) and _scored(e) is not None]
    points = [(float(_scored(e)), e.verdict == "good") for e in labelled]  # type: ignore[arg-type]
    wrong_above = [_row(e) for e in labelled if e.verdict in WRONG and _scored(e) >= threshold]  # type: ignore[operator]
    good_below = [_row(e) for e in labelled if e.verdict == "good" and _scored(e) < threshold]  # type: ignore[operator]
    suggested = suggest_threshold(points, threshold)
    missed: Counter[str] = Counter()
    ticked: Counter[str] = Counter()
    missed_shots: dict[str, list[int]] = {r: [] for r in REJECTS}
    for e in cur:
        for r in e.rejects:
            ticked[r] += 1
            if not _caught(e, r):
                missed[r] += 1
                missed_shots[r].append(e.index + 1)
    goods = [p[0] for p in points if p[1]]
    wrongs = [p[0] for p in points if not p[1]]
    return {
        "project_id": review.project_id, "threshold": float(threshold), "suggested_threshold": suggested,
        "labelled": len(points), "good": len(goods), "wrong": len(wrongs),
        "skipped_unscored": sum(1 for e in cur if e.verdict in WRONG + ("good",) and _scored(e) is None),
        "errors_now": errors_at(points, threshold), "errors_suggested": errors_at(points, suggested),
        "mean_good": round(sum(goods) / len(goods), 1) if goods else None,
        "mean_wrong": round(sum(wrongs) / len(wrongs), 1) if wrongs else None,
        "wrong_above_threshold": sorted(wrong_above, key=lambda r: -r["score"]),
        "good_below_threshold": sorted(good_below, key=lambda r: r["score"]),
        "missed_rejects": [{"reject": r, "label": REJECT_LABELS[r], "missed": n, "ticked": ticked[r],
                            "shots": missed_shots[r], "tune": PREFILTER_HINT[r]}
                           for r, n in sorted(missed.items(), key=lambda kv: (-kv[1], REJECTS.index(kv[0])))],
    }


def calibrate_project(settings: Settings, project_id: str) -> dict[str, Any]:
    return calibrate(store.load(settings, project_id), float(settings.judge.threshold))


def calibration_text(c: dict[str, Any]) -> str:
    t = c["threshold"]
    lines = [f"Judge calibration for {c['project_id']} (judge.threshold {t:.0f})",
             f"Labelled shots with a judge score: {c['labelled']} ({c['good']} good, {c['wrong']} wrong picture or "
             f"quality problem); {c['skipped_unscored']} reviewed without a score skipped."]
    if c["mean_good"] is not None or c["mean_wrong"] is not None:
        lines.append(f"Mean score: good {c['mean_good'] if c['mean_good'] is not None else '-'}, "
                     f"wrong {c['mean_wrong'] if c['mean_wrong'] is not None else '-'}")
    lines.append("")
    lines.append(f"Marked wrong but scored at or above {t:.0f} ({len(c['wrong_above_threshold'])}):")
    for r in c["wrong_above_threshold"]:
        lines.append(f"  shot {r['number']} score {r['score']:.0f} ({r['verdict']}): {r['reason'] or 'no reason'}")
    if not c["wrong_above_threshold"]:
        lines.append("  none")
    lines.append(f"Marked good but scored below {t:.0f} ({len(c['good_below_threshold'])}):")
    for r in c["good_below_threshold"]:
        lines.append(f"  shot {r['number']} score {r['score']:.0f}: {r['reason'] or 'no reason'}")
    if not c["good_below_threshold"]:
        lines.append("  none")
    lines.append("")
    lines.append(f"Suggested judge.threshold: {c['suggested_threshold']:.0f} "
                 f"(disagreements {c['errors_now']} now, {c['errors_suggested']} at the suggestion)")
    lines.append("Hard rejects the judge missed:")
    for m in c["missed_rejects"]:
        lines.append(f"  {m['label']}: missed {m['missed']} of {m['ticked']} (shots "
                     f"{', '.join(str(n) for n in m['shots'])}); tune {m['tune']}")
    if not c["missed_rejects"]:
        lines.append("  none")
    return "\n".join(lines)
