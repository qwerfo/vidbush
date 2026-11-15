"""Visual judge: score one candidate against its shot with the vision seat.

score_candidate(shot, cand, ctx) -> (score 0..100, one line reason, reject or None).
Local hard rejects (resolution, missing keyframes) never reach the model.
score_many scores (shot, candidate) pairs in a modest thread pool and writes
score, judge_reason, reject_reason and flags onto each candidate.

A real candidate (footage or photograph) is judged first against the words spoken during its
shot window, the intent only a hint; a generated picture keeps the strict intent checks.
picture_kind(cand) is the one place that tells real pictures from generated ones.
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .. import models
from ..models import Candidate, Narration, Script, Shot, SourceKind, Word
from ..source.licence import is_public_domain, normalise_licence

JUDGE_WORKERS = 6
MIN_VIDEO_HEIGHT = 720
MIN_STILL_WIDTH = 1000
MIN_ARCHIVAL_WIDTH = 500  # public domain archival stills: the real photograph beats a generated one, framed when small
SPOKEN_MARGIN_S = 0.5  # words this close to the shot window still count as spoken over it
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
    "(wrong_subject). " + REJECT_RULES + ", ".join(HARD_REJECTS) + ". The narration is what the viewer "
    "hears over the picture: a picture that contradicts it is wrong_subject even when it fits the intent. For real "
    "footage or a real photograph, the words spoken during the shot come first and the intent is only a hint: a "
    "real picture that fits those words scores 70 or more even when it misses an intent detail no viewer would "
    "look for (an exact angle, count, season, prop or camera move); hard rejects still apply. For an "
    "AI generated still, a missing must show item, a wrong count or a broken relationship (one dog beside a "
    "person where the brief is a team pulling a sled) scores below 70. Score 0 to 100 where 70 means "
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

# ---------------------------------------------------------------- picture kinds

REAL_FOOTAGE, REAL_STILL, AI_PHOTO, DESIGNED, GRAPHIC = "real_footage", "real_still", "ai_photo", "designed", "graphic"
# the source sets live in models.py; compared by value so plain strings work too
AI_PHOTO_SOURCES = {k.value for k in models.PHOTO_AI_SOURCES}
DESIGNED_SOURCES = {k.value for k in models.DESIGNED_SOURCES}
GRAPHIC_SOURCES = {k.value for k in models.GRAPHIC_SOURCES}
GENERATED_KINDS = {AI_PHOTO, DESIGNED, GRAPHIC}
ARCHIVAL_SOURCES = {"loc", "archive", "nasa", "wikimedia", "nara"}  # by value: nara may not be a SourceKind yet


def source_value(cand: Candidate) -> str:
    return cand.source.value if isinstance(cand.source, SourceKind) else str(cand.source)


def picture_kind(cand: Candidate) -> str:
    """real_footage, real_still, ai_photo (photo-style AI still or clip), designed (designed animation) or graphic."""
    src = source_value(cand)
    if src in AI_PHOTO_SOURCES:
        return AI_PHOTO
    if src in DESIGNED_SOURCES:
        return DESIGNED
    if src in GRAPHIC_SOURCES:
        return GRAPHIC
    return REAL_FOOTAGE if cand.media == "video" else REAL_STILL


def is_real(cand: Candidate) -> bool:
    return picture_kind(cand) not in GENERATED_KINDS


def archival_still(cand: Candidate) -> bool:
    """A public domain still from an archive or agency source: accepted small, rendered framed."""
    return (cand.media == "image" and source_value(cand) in ARCHIVAL_SOURCES
            and (is_public_domain(normalise_licence(cand.licence)) or "no restrictions" in cand.licence.lower()))


# ---------------------------------------------------------------- branding notes

_BRAND_RE = re.compile(r"\bsubscrib(?:e|ing)\b|\bbell (?:icon|button)s?|notification bell|"
                       r"channel (?:name|branding|intro|outro)|\bbrand(?:ing|ed)\b|\bend[ -]?(?:card|screen)s?|"
                       r"\bwatermarks?\b|\blogos?\b", re.I)  # subscribe as a call to action, not "subscribers"
_CLAUSE_RE = re.compile(r"[.;,:!?()]|\b(?:but|though|although|while|yet|plus|and|however|except)\b", re.I)
_AFTER_END_RE = re.compile(r"[.;,!?()]|\b(?:but|though|although|while|yet|plus|and|however|except)\b", re.I)
_NEG_RE = re.compile(r"\b(?:no|not|without|free of|nor|lacks?|lacking|absent|zero|never|none)\b|n't\b", re.I)
# negated after the note: "logo-free", "watermark free", ": none", "... is absent", "are not visible"
_NEG_AFTER_RE = re.compile(r"^[\s-]*(?:free|less)\b|^\W*(?:none|absent)\b|"
                           r"\b(?:is|are)\s+(?:none|absent)\b|\bnot (?:visible|present|shown)\b", re.I)
_ITEM = r"\w[\w'-]*(?:\s+\w[\w'-]*){0,2}"  # a list item of up to three words
# a negated list: "no people, watermark or text" negates every item up to the last joiner
_NEG_LIST_RE = re.compile(rf"\b(?:no|without|free of)\s+{_ITEM}(?:\s*,\s*{_ITEM})*\s*,?\s+(?:or|and|nor)\s+{_ITEM}",
                          re.I)


def branding_note(text: str, brief: str = "") -> str:
    """The first subscribe, channel branding, end screen, watermark or logo note in a judge text, or "".

    A note negated in its own clause ("no watermark or logos") does not count, nor one the brief
    itself asks for (a railway logo as the subject).
    """
    text = text or ""
    lists = [(n.start(), n.end()) for n in _NEG_LIST_RE.finditer(text)]
    for m in _BRAND_RE.finditer(text):
        word = m.group(0).lower()
        if brief and re.search(r"\b" + re.escape(word.rstrip("s")), brief, re.I):
            continue
        cuts = [c.end() for c in _CLAUSE_RE.finditer(text, 0, m.start())]
        end = _AFTER_END_RE.search(text, m.end())
        after = text[m.end():end.start() if end else len(text)]
        if (_NEG_RE.search(text[cuts[-1] if cuts else 0:m.start()]) or _NEG_AFTER_RE.search(after)
                or any(a <= m.start() and m.end() <= b for a, b in lists)):
            continue
        return word
    return ""


def candidate_branding(cand: Candidate, shot: Optional[Shot] = None) -> str:
    """branding_note over the candidate's judge reason, reject reason and flags."""
    text = "; ".join([cand.judge_reason or "", cand.reject_reason or "", *(f.replace("_", " ") for f in cand.flags)])
    brief = " ".join([shot.intent, *shot.must_show]) if shot is not None else ""
    return branding_note(text, brief)


def judge_prompt(shot: Shot, cand: Candidate, narration: str = "", spoken: str = "") -> str:
    """User prompt for one candidate: narration, intent, required and forbidden content, candidate metadata.

    A real candidate is judged first on spoken (the words over its shot window; the beat narration when
    there are no timings), the intent and must show only hints. A generated one keeps the strict brief.
    """
    if is_real(cand) and (spoken or narration):
        lines = [f"Spoken during this shot: {spoken}"] if spoken else []
        if narration:
            lines.append(f"Beat narration (context): {narration}" if spoken else f"Narration: {narration}")
        lines += [f"Director's intent (a hint, not a checklist): {shot.intent}",
                  f"Hoped to show (hint): {', '.join(shot.must_show) or 'nothing specific'}"]
    else:
        lines = [f"Narration: {narration}"] if narration else []
        lines += [f"Beat intent: {shot.intent}", f"Must show: {', '.join(shot.must_show) or 'nothing specific'}"]
    lines += [
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
    small = 0 < width < (MIN_ARCHIVAL_WIDTH if archival_still(cand) else MIN_STILL_WIDTH)
    if cand.media == "image" and small:
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


def narration_by_beat(ctx: Any) -> dict[str, str]:
    """beat id -> narration text from the project's script; empty when there is none."""
    try:
        script = ctx.maybe_load("script", Script)
        return {b.id: b.text for b in script.beats()} if script else {}
    except Exception:  # noqa: BLE001  no script (or a test context): judge against the intent alone
        return {}


def narration_words(ctx: Any) -> list[Word]:
    """Word timings from the narration stage; empty when there are none."""
    try:
        nar = ctx.maybe_load("narration", Narration)
        return list(nar.words) if nar else []
    except Exception:  # noqa: BLE001  no narration (or a test context): judge against the beat narration
        return []


def spoken_in(words: list[Word], start: float, end: float, margin: float = SPOKEN_MARGIN_S) -> str:
    """Words spoken from start to end, a small margin either side."""
    return " ".join(w.text for w in words if w.end > start - margin and w.start < end + margin)


def score_candidate(shot: Shot, cand: Candidate, ctx: Any, narration: str = "", spoken: str = "") -> Result:
    """Score one candidate against its shot, the beat narration and the words spoken over the shot.
    Returns (score, reason, reject or None); flags land on cand.flags."""
    rej = local_reject(cand)
    if rej:
        return 0.0, rej, rej
    frames = [k for k in cand.keyframes if Path(k).exists()][:MAX_KEYFRAMES]
    with _VISION_SLOTS:
        raw = ctx.seats.vision_json("judge", "judge", JUDGE_SYSTEM, judge_prompt(shot, cand, narration, spoken), frames,
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
    """Video clips that cannot cover the shot even slowed to 1.25x lose points, so longer clips win.

    The render extends a short window into the rest of its file, so the span counts up to the file end.
    """
    span = cand.out_point - cand.in_point
    if cand.media == "video" and span > 0 and span * 1.25 < shot.duration:
        span = max(span, _file_seconds(cand) - cand.in_point)
    if cand.media == "video" and span > 0 and span * 1.25 < shot.duration:
        return max(0.0, score - SHORT_CLIP_PENALTY), f"{reason} (clip {span:.1f} s for a {shot.duration:.1f} s shot)"
    return score, reason


def _file_seconds(cand: Candidate) -> float:
    from ..util import media_duration
    try:
        return media_duration(cand.local_path) if cand.local_path and Path(cand.local_path).is_file() else 0.0
    except Exception:  # noqa: BLE001  an unreadable file keeps the window span
        return 0.0


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
    beats = narration_by_beat(ctx)
    words = narration_words(ctx)

    def one(pair: tuple[Shot, Candidate]) -> Result:
        try:
            return score_candidate(pair[0], pair[1], ctx, beats.get(pair[0].beat_id, ""),
                                   spoken_in(words, pair[0].start, pair[0].end))
        except Exception as e:  # noqa: BLE001  one bad file must not sink the stage
            ctx.log.warn("judge failed for candidate", shot=pair[0].id, candidate=pair[1].id, error=str(e)[:300])
            return 0.0, f"judge error: {type(e).__name__}", "judge error"

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pairs)))) as pool:
        results = list(pool.map(one, pairs))
    for (_, cand), res in zip(pairs, results):
        apply_result(cand, res)
