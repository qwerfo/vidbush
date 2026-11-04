"""Director loop: judge every candidate, select one per shot, never leave a shot empty.

judge_and_select(shotlist, ctx) -> ShotList.
  attempt 1  score the sourced candidates, pick the best at or above threshold
  attempt 2  rewrite the queries from the judge reasons and re-source
  attempt 3+ convert the shot to ai_image (overlay kept) and judge a generated still
After settings.judge.max_attempts the best available candidate is accepted
with flag no_match_accepted. motion_graphic shots are left to the motion stage.
"""
from __future__ import annotations

import inspect
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..models import Candidate, Script, Shot, ShotList, ShotType, SourceKind
from .judge import JUDGE_MOCK_FLAG, JUDGE_WORKERS, score_many
from .prefilter import apply_prefilter

SOURCED_TYPES = {ShotType.footage, ShotType.footage_overlay, ShotType.image, ShotType.image_overlay}
AI_TYPES = {ShotType.ai_image, ShotType.ai_video}
RESOLVE_WORKERS = 4
DIVERSITY_GAP = 5.0  # score points charged per earlier use of the same asset anywhere in the video
RECENT_S = 60.0
RECENT_PENALTY = 10.0  # extra when the earlier use is within RECENT_S
SEGMENT_GAP_S = 3.0  # a reused video must start at least this far from the earlier in point
SAME_SEGMENT_PENALTY = 15.0
SHOT_FLAGS = {"mock", "low_res", "nc_nd", "anachronism", "no_match_accepted"}
MAX_QUERIES = 6

REQUERY_SYSTEM = (
    "You are the director of a long-form documentary. Stock and archive searches for one shot found nothing "
    "usable. Using the judge's reasons, write new search queries that will find usable footage or stills. "
    "Prefer broader, concrete nouns: the place, the subject, the action. Drop styling words. Be concise. "
    'Reply with strict JSON only: {"queries": ["query", ...]} with 3 to 6 short queries.'
)
_STOP = {
    "archival", "archive", "historic", "historical", "vintage", "old", "ancient", "modern", "black", "white",
    "colour", "color", "aerial", "wide", "close", "closeup", "detail", "dramatic", "beautiful", "busy", "huge",
    "massive", "giant", "famous", "footage", "photo", "photograph", "image", "picture", "shot", "view",
    "establishing", "stock", "rare", "original", "sepia", "early", "late", "mid", "era", "period", "century",
    "cinematic", "of", "the", "a", "an", "in", "at", "on", "and", "with", "from", "during", "showing", "new",
}
_ERA = re.compile(r"^(1[0-9]\d\d|20\d\d)s?$|^\d+(st|nd|rd|th)$|^(19|20)th$", re.I)
_LEADING = re.compile(r"^.*?\bof\s+", re.I)


# ---------------------------------------------------------------- entry point

def judge_and_select(shotlist: ShotList, ctx: Any) -> ShotList:
    """Score, select, re-query, convert to AI; every non motion shot ends with a selected candidate."""
    threshold = float(ctx.settings.judge.threshold)
    active = [s for s in shotlist.shots if s.type != ShotType.motion_graphic]
    _seed_ai(active, ctx)
    _prefilter_and_score([(s, c) for s in active for c in s.candidates if c.score is None], ctx)
    for s in active:
        s.attempts = max(s.attempts, 1)
    failing = [s for s in active if not _passing(s, threshold)]
    with ThreadPoolExecutor(max_workers=RESOLVE_WORKERS) as pool:
        notes = dict(zip([s.id for s in failing], pool.map(lambda s: _resolve(s, ctx, threshold), failing)))
    _select_all(shotlist, ctx, threshold, notes)
    for s in active:
        if s.type == ShotType.ai_video:
            _finish_ai_video(s, ctx)
    ok = sum(1 for s in active if (s.judge_score or 0) >= threshold)
    ctx.log.info("judge done", shots=len(active), passed=ok, reworked=len(failing),
                 no_match=sum(1 for s in active if "no_match_accepted" in s.flags))
    return shotlist


# ---------------------------------------------------------------- helpers: scoring state

def _prefilter_and_score(pairs: list[tuple[Shot, Candidate]], ctx: Any, workers: int = JUDGE_WORKERS) -> None:
    """Local pre-filter first (hard rejects get score 0), then the vision judge on what is still unscored."""
    apply_prefilter([c for _, c in pairs], ctx.settings, ctx.log)
    score_many([(s, c) for s, c in pairs if c.score is None], ctx, workers=workers)


def _ok(c: Candidate, threshold: float) -> bool:
    return c.reject_reason is None and c.score is not None and c.score >= threshold


def _passing(shot: Shot, threshold: float) -> bool:
    return any(_ok(c, threshold) for c in shot.candidates)


def _ranked(shot: Shot) -> list[Candidate]:
    """Best first: candidates without a hard reject, then by score."""
    return sorted(shot.candidates, key=lambda c: (c.reject_reason is None, c.score or 0.0), reverse=True)


def asset_key(c: Candidate) -> str:
    """Identity of the underlying asset: canonical URL first, since files are copied per shot."""
    return c.url or c.local_path or c.id


# ---------------------------------------------------------------- AI shots

def _seed_ai(shots: list[Shot], ctx: Any) -> None:
    """Director planned ai_image / ai_video shots get a generated still when they have none."""
    todo = [s for s in shots if s.type in AI_TYPES and not any(c.source == SourceKind.ai_image for c in s.candidates)]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=RESOLVE_WORKERS) as pool:
        made = list(pool.map(lambda s: _generate(s, ctx), todo))
    for shot, cand in zip(todo, made):
        shot.candidates.append(cand)


def _generate(shot: Shot, ctx: Any, take: int = 1) -> Candidate:
    from ..visuals.ai_images import build_prompt, generate_ai_candidate
    prompt = None
    if take > 1:
        prompt = f"{build_prompt(shot)} Alternative take {take}: a different camera angle and composition."
    return generate_ai_candidate(shot, ctx, prompt=prompt)


def _add_ai(shot: Shot, ctx: Any) -> None:
    take = 1 + sum(1 for c in shot.candidates if c.source == SourceKind.ai_image)
    cand = _generate(shot, ctx, take)
    shot.candidates.append(cand)
    _prefilter_and_score([(shot, cand)], ctx, workers=1)


# ---------------------------------------------------------------- attempts 2 and 3+

def _resolve(shot: Shot, ctx: Any, threshold: float) -> list[str]:
    """Run further attempts on a failing shot. Returns notes for judge_notes."""
    notes: list[str] = []
    original = shot.type
    try:
        while shot.attempts < ctx.settings.judge.max_attempts and not _passing(shot, threshold):
            shot.attempts += 1
            if shot.attempts == 2 and shot.type in SOURCED_TYPES:
                notes.append(_requery(shot, ctx))
                continue
            if shot.type not in AI_TYPES:
                shot.type = ShotType.ai_image
                notes.append(f"converted to AI image at attempt {shot.attempts}")
            _add_ai(shot, ctx)
    except Exception as e:  # noqa: BLE001  keep what we have, selection still guarantees a pick
        ctx.log.warn("director loop attempt failed", shot=shot.id, error=str(e)[:300])
        notes.append(f"attempt {shot.attempts} failed: {type(e).__name__}")
    best = _ranked(shot)[0] if shot.candidates else None
    if shot.type != original and original in SOURCED_TYPES and best and best.source != SourceKind.ai_image:
        shot.type = original  # the AI still lost to a sourced candidate, keep the planned type
    return notes


def _requery(shot: Shot, ctx: Any) -> str:
    reasons = [c.reject_reason or c.judge_reason for c in _ranked(shot)[:4] if c.reject_reason or c.judge_reason]
    queries = requery_queries(shot, ctx, reasons)
    new = _source(shot, ctx, queries)
    added = add_candidates(shot, new)
    _prefilter_and_score([(shot, c) for c in added], ctx, workers=2)
    shot.queries = _dedupe(shot.queries + queries, 24)
    ctx.log.info("requery", shot=shot.id, queries=queries, new_candidates=len(added))
    return f"requery ({', '.join(queries[:3])}) found {len(added)} new"


def requery_queries(shot: Shot, ctx: Any, reasons: list[str]) -> list[str]:
    """Director seat rewrite of the queries; deterministic broadening when the seat gives nothing."""
    user = (f"Shot intent: {shot.intent}\nMust show: {', '.join(shot.must_show) or 'nothing specific'}\n"
            f"Must avoid: {', '.join(shot.must_avoid) or 'nothing specific'}\n"
            f"Queries tried: {'; '.join(shot.queries) or 'none'}\nJudge reasons: {'; '.join(reasons) or 'none'}")
    raw = ctx.seats.llm_json("director", "requery", REQUERY_SYSTEM, user, ctx.stage or "judge")
    got = raw.get("queries") if isinstance(raw, dict) else None
    if isinstance(got, list):
        clean = _dedupe([" ".join(str(q).split()) for q in got if isinstance(q, str)], MAX_QUERIES)
        clean = [q for q in clean if q.lower() not in {x.lower() for x in shot.queries}]
        if clean:
            return clean
    return heuristic_queries(shot, _scene_title(shot, ctx))


def heuristic_queries(shot: Shot, scene_title: str = "") -> list[str]:
    """Broaden: drop adjectives and era words, must_show terms alone, the scene subject."""
    out: list[str] = []
    for q in shot.queries:
        words = _content_words(q)
        out += [" ".join(words), " ".join(words[:2])]
    out += [t for t in shot.must_show]
    subject = _content_words(_LEADING.sub("", shot.intent, count=1))
    out += [" ".join(subject[:4]), " ".join(_content_words(scene_title)[:4])]
    tried = {q.lower() for q in shot.queries}
    return [q for q in _dedupe(out, MAX_QUERIES * 2) if q.lower() not in tried][:MAX_QUERIES]


def _content_words(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9'&]+", text)
    return [t for t in toks if t.lower() not in _STOP and not _ERA.match(t)]


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out = []
    for q in items:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:limit]


def _scene_title(shot: Shot, ctx: Any) -> str:
    try:
        script = ctx.maybe_load("script", Script)
    except Exception:  # noqa: BLE001  a broken script file only loses the scene hint
        return ""
    for scene in (script.scenes if script else []):
        if scene.id == shot.scene_id:
            return scene.title
    return ""


def _source(shot: Shot, ctx: Any, queries: list[str]) -> list[Candidate]:
    """Re-source through the registry (lazy import; another module owns it)."""
    try:
        from ..source.registry import source_shot
    except ImportError as e:
        ctx.log.warn("source registry unavailable for requery", error=str(e)[:200])
        return []
    kwargs: dict[str, Any] = {"queries": queries}
    if "exclude_ids" in inspect.signature(source_shot).parameters:
        kwargs["exclude_ids"] = tuple(c.id for c in shot.candidates)
    try:
        return list(source_shot(shot, ctx, **kwargs) or [])
    except Exception as e:  # noqa: BLE001  network or adapter failure: fall through to AI conversion
        ctx.log.warn("re-sourcing failed", shot=shot.id, error=str(e)[:300])
        return []


def add_candidates(shot: Shot, new: list[Candidate]) -> list[Candidate]:
    """Append candidates whose asset is new to the shot; renumber colliding ids. Returns those added."""
    have = {(asset_key(c), round(c.in_point, 2)) for c in shot.candidates}
    ids = {c.id for c in shot.candidates}
    added = []
    for c in new:
        key = (asset_key(c), round(c.in_point, 2))
        if key in have:
            continue
        if c.id in ids:
            n = 1
            while f"{shot.id}_{c.source.value}_{n}" in ids:
                n += 1
            c = c.model_copy(update={"id": f"{shot.id}_{c.source.value}_{n}"})
        have.add(key)
        ids.add(c.id)
        shot.candidates.append(c)
        added.append(c)
    return added


# ---------------------------------------------------------------- selection

class _Usage:
    """Assets already selected in this video: (shot start, asset key, in point) in timeline order."""

    def __init__(self) -> None:
        self.picks: list[tuple[float, str, float]] = []

    def penalty(self, c: Candidate, at: float) -> float:
        """Score points to subtract for reusing c at time at: any reuse, reuse within 60 s, same segment."""
        key, p = asset_key(c), 0.0
        for t, k, inp in self.picks:
            if k != key:
                continue
            p += DIVERSITY_GAP
            if abs(at - t) <= RECENT_S:
                p += RECENT_PENALTY
            if c.media == "video" and abs(c.in_point - inp) < SEGMENT_GAP_S:
                p += SAME_SEGMENT_PENALTY  # stills may repeat with a new framing; a video needs a new segment
        return p

    def add(self, c: Candidate, at: float) -> None:
        self.picks.append((at, asset_key(c), c.in_point))


def _select_all(shotlist: ShotList, ctx: Any, threshold: float, notes: dict[str, list[str]]) -> None:
    """Pick one candidate per shot, penalising reuse of an asset anywhere in the video (more within 60 s)."""
    usage = _Usage()
    for shot in shotlist.shots:
        if shot.type == ShotType.motion_graphic:
            continue
        _ensure_usable(shot, ctx)
        pinned = shot.selected_candidate()
        passing = [c for c in _ranked(shot) if _ok(c, threshold)]
        if pinned is not None and pinned.source == SourceKind.upload:
            pick, accepted = pinned, False
        elif passing:
            pick, accepted = _diverse(passing, usage, shot.start), False
        else:
            ranked = _ranked(shot)
            pool = [c for c in ranked if c.reject_reason is None] or ranked[:1]
            pick, accepted = _diverse(pool, usage, shot.start), True
        _apply(shot, pick, accepted, notes.get(shot.id, []))
        usage.add(pick, shot.start)


def _ensure_usable(shot: Shot, ctx: Any) -> None:
    """Never empty: generate an AI still when there is no candidate or only hard rejected ones."""
    if shot.candidates and any(c.reject_reason is None for c in shot.candidates):
        return
    if any(c.source == SourceKind.ai_image for c in shot.candidates):
        return  # an AI still was already tried; best available wins
    ctx.log.info("no usable candidate, generating AI image", shot=shot.id)
    _add_ai(shot, ctx)


def _diverse(pool: list[Candidate], usage: _Usage, at: float) -> Candidate:
    """Best of pool by judge score minus the reuse penalty; ties keep the judge order."""
    return max(pool, key=lambda c: (c.score or 0.0) - usage.penalty(c, at))


def _apply(shot: Shot, pick: Candidate, accepted: bool, notes: list[str]) -> None:
    shot.selected = pick.id
    shot.judge_score = pick.score
    shot.flags = [f for f in shot.flags if f not in SHOT_FLAGS]
    if "mock" in pick.flags or JUDGE_MOCK_FLAG in pick.flags:
        shot.flags.append("mock")
    shot.flags += [f for f in ("low_res", "nc_nd", "anachronism") if f in pick.flags]
    if accepted:
        shot.flags.append("no_match_accepted")
    reason = pick.reject_reason or pick.judge_reason or ""
    shot.judge_notes = " | ".join([reason] + notes + (["best available accepted"] if accepted else []))


def _finish_ai_video(shot: Shot, ctx: Any) -> None:
    """Animate the approved still when the video seat is real, else keep it as an ai_image shot."""
    from ..visuals.ai_video import animate_still
    still = shot.selected_candidate()
    clip = animate_still(shot, still, ctx) if still is not None and still.media == "image" else None
    if clip is None:
        if still is not None and still.media == "image":
            shot.type = ShotType.ai_image
            shot.judge_notes += " | ai_video unavailable, kept still"
        return
    shot.candidates.append(clip)
    shot.selected = clip.id
