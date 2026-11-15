"""Fixed-question check of an AI still, after the judge's overall score.

One overall score passed stills with the wrong subject, a split-panel collage, the wrong place or a fabricated
face of a named person. check_still asks one vision call a fixed set of yes/no questions about the band the
viewer sees (the still center-cropped to the letterbox ratio); any question answered not ok fails the still,
and its reasons become the next take's problem (context for the prompt rewrite, never prompt text).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..models import Candidate, Shot, SourceKind
from .judge import JUDGE_MOCK_FLAG, narration_by_beat

FAIL_FLAG = "ai_check_failed"
# Every question is phrased so yes (ok true) is good.
QUESTIONS = {
    "subject": "Is the acting subject the one the narration names, doing that action?",
    "count": "Is the number of the main subjects right, as the narration and must show state? "
             "(yes when no count is stated)",
    "setting": "Does the setting (terrain, vegetation, buildings, light, clothing, vehicles) fit the place and "
               "period of the brief, with nothing that contradicts them?",
    "one_image": "Is it one continuous photograph, with no split panels, collage, grid, diptych or inset frames?",
    "no_text": "Is it free of readable text, letters, numbers, logos and watermarks?",
    "no_named_face": "Is it free of a recognisable face presented as a named real person (the brief's people, "
                     "or anyone the narration names)?",
    "framing": "Is every head and the main subject fully inside this frame, with no obvious artifact (extra "
               "limbs, floating objects)?",
}
SYSTEM = ("You check one AI generated documentary still against its narration and scene brief. Answer every "
          "question with ok true (yes, the still passes) or ok false (no), and a one-line reason saying what "
          "you see. Return strict JSON "
          '{"answers": {' + ", ".join(f'"{q}": {{"ok": true, "reason": "..."}}' for q in QUESTIONS) + "}}.")


def check_prompt(shot: Shot, cand: Candidate, window: str, beat: str) -> str:
    b = cand.brief or {}
    lines = [f"Narration during this shot: {window or '(none)'}", f"Beat narration: {beat or '(none)'}"]
    lines += [f"Brief {k}: {b[k]}" for k in ("agent", "action", "object", "place", "era") if b.get(k)]
    if b.get("people"):
        lines.append(f"Brief people (named real persons, must not appear as a face): {', '.join(b['people'])}")
    if shot.must_show:
        lines.append(f"Must show: {'; '.join(shot.must_show)}")
    lines += ["Questions:"] + [f"{q}: {text}" for q, text in QUESTIONS.items()]
    return "\n".join(lines)


def band_image(path: str, ratio: Optional[float]) -> str:
    """The still center-cropped to the letterbox band (what the viewer sees) as <stem>_band.jpg next to it;
    the still itself when there is no letterbox."""
    if not ratio:
        return path
    out = Path(path).with_name(f"{Path(path).stem}_band.jpg")
    with Image.open(path) as im:
        w, h = im.size
        bh = min(h, round(w / ratio))
        top = (h - bh) // 2
        im.convert("RGB").crop((0, top, w, top + bh)).save(out, quality=90)
    return str(out)


def _ok(v: Any) -> Optional[bool]:
    """ok as a bool from true/false or a yes/no word ('Yes.', 'no, the sled is missing'); None when unreadable."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        words = v.strip().lower().replace(",", " ").replace(".", " ").replace("!", " ").split()
        if words and words[0] in ("yes", "true", "no", "false"):
            return words[0] in ("yes", "true")
    return None


def check_still(shot: Shot, cand: Candidate, ctx: Any) -> dict[str, Any]:
    """Ask the fixed questions about an AI still; a not ok answer sets reject_reason and the ai_check_failed
    flag. Skipped ({}) for other sources and mock replies. Unanswered questions never fail nor pass the still:
    they are recorded under "unanswered". A seat error keeps the judge's verdict."""
    if cand.source != SourceKind.ai_image or JUDGE_MOCK_FLAG in cand.flags or not cand.local_path:
        return {}
    from ..visuals.ai_images import letterbox_ratio, shot_window_text
    try:
        image = band_image(cand.local_path, letterbox_ratio(ctx))
        window, beat = shot_window_text(shot, ctx), narration_by_beat(ctx).get(shot.beat_id, "")
        raw = ctx.seats.vision_json("judge", "ai_check", SYSTEM, check_prompt(shot, cand, window, beat), [image],
                                    ctx.stage or "judge", intent=shot.intent, details=["high"])
    except Exception as e:  # noqa: BLE001  the still keeps the judge's verdict
        ctx.log.warn("ai still check failed", shot=shot.id, candidate=cand.id, error=str(e)[:200])
        return {}
    if not isinstance(raw, dict) or raw.get("_mock"):
        return {}
    given = raw.get("answers") if isinstance(raw.get("answers"), dict) else {}
    answers: dict[str, dict[str, Any]] = {}
    unanswered: list[str] = []
    for q in QUESTIONS:
        a = given.get(q)
        ok = _ok(a.get("ok")) if isinstance(a, dict) else None
        if ok is None:
            unanswered.append(q)
            continue
        answers[q] = {"ok": ok, "reason": " ".join(str(a.get("reason") or "").split())[:200]}
    if not answers:  # an empty or wrongly shaped reply says nothing about the still
        ctx.log.warn("ai still check reply unusable", shot=shot.id, candidate=cand.id)
        return {}
    if unanswered:
        ctx.log.warn("ai still check left questions unanswered", shot=shot.id, candidate=cand.id,
                     unanswered=unanswered)
    failed = [q for q, a in answers.items() if not a["ok"]]
    cand.checks = {"answers": answers, "failed": failed, "unanswered": unanswered}
    if failed:
        why = "ai_check: " + "; ".join(f"{q}: {answers[q]['reason'] or 'no'}" for q in failed)
        cand.reject_reason = f"{cand.reject_reason}; {why}" if cand.reject_reason else why
        if FAIL_FLAG not in cand.flags:
            cand.flags.append(FAIL_FLAG)
        ctx.log.info("ai still failed the fixed-question check", shot=shot.id, candidate=cand.id, failed=failed)
    return cand.checks
