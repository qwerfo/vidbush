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

JUDGE_SYSTEM = (
    "You are the picture editor of a long-form documentary channel. You judge whether a candidate clip or "
    "still can cover one narration beat. House principles: good enough is the bar. Fail a candidate only "
    "when the difference is big enough to break viewer immersion. Small prop differences, minor crops and "
    "slight colour shifts pass. Judge in this order: first the place must make sense, then the subject must "
    "be right, then the action. Hard rejects, listed in rejects: burned in captions or subtitles, a big "
    "watermark, a channel logo, a person speaking to camera, a visible modern anachronism when the brief "
    "says to avoid it, a clearly wrong subject. Score 0 to 100 where 70 means usable. Be concise: the "
    "reason is one short line. Reply with strict JSON only: "
    '{"score": number, "reason": "one line", "rejects": ["short reject label", ...]}'
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


def parse_verdict(raw: Any) -> Optional[tuple[float, str, list[str]]]:
    """Normalise a model reply to (score 0..100, reason, rejects); None when the shape is unusable."""
    if not isinstance(raw, dict):
        return None
    try:
        score = float(raw.get("score"))
    except (TypeError, ValueError):
        return None
    if 0 < score <= 1:
        score *= 100.0  # a 0..1 scale reply
    elif 0 < score <= 10:
        score *= 10.0  # a 0..10 scale reply (still below threshold when it was a poor 0..100 score)
    rejects = raw.get("rejects") or []
    if isinstance(rejects, str):
        rejects = [rejects]
    rejects = [str(r).strip() for r in rejects if str(r).strip()]
    reason = " ".join(str(raw.get("reason") or "").split())[:240]
    return max(0.0, min(100.0, score)), reason, rejects


def score_candidate(shot: Shot, cand: Candidate, ctx: Any) -> Result:
    """Score one candidate. Returns (score, reason, reject or None); flags land on cand.flags."""
    rej = local_reject(cand)
    if rej:
        return 0.0, rej, rej
    frames = [k for k in cand.keyframes if Path(k).exists()][:MAX_KEYFRAMES]
    with _VISION_SLOTS:
        raw = ctx.seats.vision_json("judge", "judge", JUDGE_SYSTEM, judge_prompt(shot, cand), frames,
                                    ctx.stage or "judge", intent=shot.intent)
    if raw.get("_mock") and JUDGE_MOCK_FLAG not in cand.flags:
        cand.flags.append(JUDGE_MOCK_FLAG)
    verdict = parse_verdict(raw)
    if verdict is None:
        ctx.log.warn("judge reply unusable, neutral score", shot=shot.id, candidate=cand.id)
        return 50.0, "judge reply unusable", None
    score, reason, rejects = verdict
    _flag_from_rejects(cand, rejects)
    reject = "; ".join(rejects) if rejects else None
    return round(score, 1), reason or "no reason given", reject


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
