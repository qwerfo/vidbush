"""Visual judge: score one candidate against its shot with the vision seat.

score_candidate(shot, cand, ctx) -> (score 0..100, one line reason, reject or None).
Local hard rejects (resolution, missing keyframes) never reach the model.
score_many scores (shot, candidate) pairs in a modest thread pool and writes
score, judge_reason, reject_reason and flags onto each candidate.
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..models import Candidate, Shot

JUDGE_WORKERS = 6
MIN_VIDEO_HEIGHT = 720
MIN_STILL_WIDTH = 1000
MAX_KEYFRAMES = 3
JUDGE_MOCK_FLAG = "judge_mock"
_VISION_SLOTS = threading.BoundedSemaphore(JUDGE_WORKERS)  # process wide cap on concurrent judge calls

HARD_REJECTS = ("burned_in_captions", "watermark", "channel_logo", "talking_head", "anachronism", "wrong_subject")
REJECT_RULES = "rejects must be [] when no hard reject applies; otherwise use only these labels: "

JUDGE_SYSTEM = (
    "You are the picture editor of a long-form documentary channel. You judge whether a candidate clip or "
    "still can cover one narration beat. House principles: good enough is the bar. Fail a candidate only "
    "when the difference is big enough to break viewer immersion. Small prop differences, minor crops and "
    "slight colour shifts pass. Judge in this order: first the place must make sense, then the subject must "
    "be right, then the action. Hard rejects: burned in captions or subtitles (burned_in_captions), a big "
    "watermark (watermark), a channel logo (channel_logo), a person speaking to camera (talking_head), a "
    "visible modern anachronism when the brief says to avoid it (anachronism), a clearly wrong subject "
    "(wrong_subject). " + REJECT_RULES + ", ".join(HARD_REJECTS) + ". Score 0 to 100 where 70 means "
    "usable. Be concise: the reason is one short line. Reply with strict JSON only: "
    '{"score": number, "reason": "one line", "rejects": []}'
)
_NONE_RE = re.compile(r"^\W*(none|n\s*/?\s*a|no|null|nil|nothing|-+|no hard rejects?|no rejects?)\W*$", re.I)
_LABEL_RULES: tuple[tuple[str, str], ...] = (
    (r"caption|subtitle|burn", "burned_in_captions"), (r"watermark", "watermark"),
    (r"logo|bug|channel", "channel_logo"), (r"talking|speak|to camera|presenter|interview", "talking_head"),
    (r"anachron|modern|out of era|wrong era", "anachronism"), (r"wrong|subject|unrelated|mismatch", "wrong_subject"),
    (r"black frame|blank", "black_frame"), (r"frozen|freeze|still frame", "frozen_frame"),
)

Result = tuple[float, str, Optional[str]]


def judge_prompt(shot: Shot, cand: Candidate) -> str:
    """User prompt for one candidate: intent, required and forbidden content, candidate metadata."""
    lines = [
        f"Beat intent: {shot.intent}",
        f"Must show: {', '.join(shot.must_show) or 'nothing specific'}",
        f"Must avoid: {', '.join(shot.must_avoid) or 'nothing specific'}",
        f"Shot type: {shot.type.value}, framing {shot.framing}, {shot.duration:.1f} seconds on screen",
        f"Candidate: {cand.media} from {cand.source.value}, title: {cand.title or 'untitled'}",
        f"The {len(cand.keyframes[:MAX_KEYFRAMES])} attached images are keyframes of this candidate.",
        "Return strict JSON with score, reason and rejects.",
    ]
    return "\n".join(lines)


def local_reject(cand: Candidate) -> Optional[str]:
    """Hard rejects decided without the model: missing keyframes or resolution below 720p."""
    frames = [k for k in cand.keyframes if k and Path(k).exists()]
    if not frames:
        return "missing keyframes"
    width, height = cand.width, cand.height
    if not (width and height):
        width, height = _image_size(cand.local_path if cand.media == "image" else frames[0])
    if cand.media == "video" and 0 < height < MIN_VIDEO_HEIGHT:
        return f"resolution below 720p ({height}p)"
    if cand.media == "image" and 0 < width < MIN_STILL_WIDTH:
        return f"still too small ({width} px wide)"
    return None


def _image_size(path: str) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, ValueError):
        return 0, 0


def normalize_rejects(raw: Any, allowed: tuple[str, ...] = HARD_REJECTS) -> tuple[list[str], list[str]]:
    """(hard reject labels, other notes): drops none-like entries, maps wording onto the allowed labels."""
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    labels: list[str] = []
    notes: list[str] = []
    for item in items:
        text = " ".join(str(item).split())
        if not text or _NONE_RE.match(text):
            continue
        key = text.lower().replace(" ", "_")
        label = key if key in allowed else next(
            (lab for pat, lab in _LABEL_RULES if lab in allowed and re.search(pat, text, re.I)), None)
        if label is None:
            notes.append(text)
        elif label not in labels:
            labels.append(label)
    return labels, notes


def parse_verdict(raw: Any, allowed: tuple[str, ...] = HARD_REJECTS) -> Optional[tuple[float, str, list[str]]]:
    """Normalise a model reply to (score 0..100, reason, rejects); None when the shape is unusable.

    A fractional score (0 < score < 1) is read as a 0..1 scale; anything else is taken as 0..100.
    Reject entries such as "none" are dropped; unknown labels move into the reason instead.
    """
    if not isinstance(raw, dict):
        return None
    try:
        score = float(raw.get("score"))
    except (TypeError, ValueError):
        return None
    if 0 < score < 1:
        score *= 100.0
    rejects, extra = normalize_rejects(raw.get("rejects") or [], allowed)
    reason = " ".join(str(raw.get("reason") or "").split())
    if extra:
        reason = "; ".join([reason] + extra) if reason else "; ".join(extra)
    return max(0.0, min(100.0, score)), reason[:240], rejects


def keyframe_details(n: int, settings: Any = None) -> list[str]:
    """Image detail per keyframe: judge.first_keyframe_detail for the first, keyframe_detail for the rest.

    The first keyframe carries the most context (it opens the clip), so it gets the sharper look;
    the rest only confirm motion and continuity. Seats.vision_json validates the values.
    """
    judge = getattr(settings, "judge", None)
    first = getattr(judge, "first_keyframe_detail", "high")
    rest = getattr(judge, "keyframe_detail", "low")
    return ([first] + [rest] * (n - 1)) if n > 0 else []


def score_candidate(shot: Shot, cand: Candidate, ctx: Any) -> Result:
    """Score one candidate. Returns (score, reason, reject or None); flags land on cand.flags."""
    rej = local_reject(cand)
    if rej:
        return 0.0, rej, rej
    frames = [k for k in cand.keyframes if Path(k).exists()][:MAX_KEYFRAMES]
    with _VISION_SLOTS:
        raw = ctx.seats.vision_json("judge", "judge", JUDGE_SYSTEM, judge_prompt(shot, cand), frames,
                                    ctx.stage or "judge", intent=shot.intent,
                                    details=keyframe_details(len(frames), getattr(ctx, "settings", None)))
    if raw.get("_mock") and JUDGE_MOCK_FLAG not in cand.flags:
        cand.flags.append(JUDGE_MOCK_FLAG)
    verdict = parse_verdict(raw)
    if verdict is None:
        ctx.log.warn("judge reply unusable, neutral score", shot=shot.id, candidate=cand.id)
        return 50.0, "judge reply unusable", None
    score, reason, rejects = verdict
    _flag_from_rejects(cand, rejects)
    score, reason = _short_clip_penalty(shot, cand, score, reason)
    reject = "; ".join(rejects) if rejects else None
    return round(score, 1), reason or "no reason given", reject


SHORT_CLIP_PENALTY = 15.0


def _short_clip_penalty(shot: Shot, cand: Candidate, score: float, reason: str) -> tuple[float, str]:
    """Video clips that cannot cover the shot even slowed to 1.25x lose points, so longer clips win."""
    span = cand.out_point - cand.in_point
    if cand.media == "video" and span > 0 and span * 1.25 < shot.duration:
        return max(0.0, score - SHORT_CLIP_PENALTY), f"{reason} (clip {span:.1f} s for a {shot.duration:.1f} s shot)"
    return score, reason


def _flag_from_rejects(cand: Candidate, rejects: list[str]) -> None:
    for r in rejects:
        if re.search(r"anachron|modern", r, re.I) and "anachronism" not in cand.flags:
            cand.flags.append("anachronism")


def apply_result(cand: Candidate, result: Result) -> None:
    """Write a judge result onto the candidate."""
    cand.score, cand.judge_reason, cand.reject_reason = result


def score_many(pairs: list[tuple[Shot, Candidate]], ctx: Any, workers: int = JUDGE_WORKERS) -> None:
    """Score every pair in parallel and store the results. Concurrent model calls never exceed JUDGE_WORKERS."""
    if not pairs:
        return

    def one(pair: tuple[Shot, Candidate]) -> Result:
        try:
            return score_candidate(pair[0], pair[1], ctx)
        except Exception as e:  # noqa: BLE001  one bad file must not sink the stage
            ctx.log.warn("judge failed for candidate", shot=pair[0].id, candidate=pair[1].id, error=str(e)[:300])
            return 0.0, f"judge error: {type(e).__name__}", "judge error"

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pairs)))) as pool:
        results = list(pool.map(one, pairs))
    for (_, cand), res in zip(pairs, results):
        apply_result(cand, res)
