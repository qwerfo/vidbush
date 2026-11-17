"""Director loop: judge every candidate, select one per shot, never leave a shot empty.

judge_and_select(shotlist, ctx) -> ShotList.
  attempt 1  score the sourced candidates, pick the best at or above threshold
  attempt 2  rewrite the queries from the judge reasons and re-source
  attempt 3+ convert the shot to ai_image (overlay kept) and judge a generated still; a failed still is
             regenerated with the judge's reason up to settings.judge.ai_retries times. A subject that is
             readable text (map, headline, document) is never converted: it is flagged needs_motion_graphic
After settings.judge.max_attempts the best available candidate is accepted
with flag no_match_accepted. motion_graphic shots are left to the motion stage.

Real first: a real candidate (footage or photograph, judge.picture_kind) at or above REAL_BAR beats
a photo-style AI still, and a shot that has one is never converted to an AI still.
Floor: a candidate below FLOOR, or whose judge notes carry subscribe, channel branding, end screen,
watermark or logo notes, is never shipped while anything else exists. Such a slot takes the best real
candidate of another shot in the same beat, then a generated still. When nothing meets the floor and
generation fails, the least bad candidate is kept (not branded first, then not hard rejected, then by
score), flagged below_floor and logged as a warning, so a shot is never empty.
After every selection sync_type leaves shot.type consistent with the pick.
"""
from __future__ import annotations

import inspect
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from ..providers.base import ProviderError
from ..models import OVERLAY_TYPES, Candidate, Script, Shot, ShotList, ShotType, SourceKind
from ..visuals.ai_images import needs_readable_text
from .judge import (AI_PHOTO, DESIGNED, GRAPHIC, JUDGE_MOCK_FLAG, JUDGE_WORKERS, candidate_branding, is_real,
                    picture_kind, score_many, source_value)
from .prefilter import apply_prefilter

SOURCED_TYPES = {ShotType.footage, ShotType.footage_overlay, ShotType.image, ShotType.image_overlay}
AI_TYPES = {ShotType.ai_image, ShotType.ai_video}
RESOLVE_WORKERS = 4
DIVERSITY_GAP = 5.0  # score points charged per earlier use of the same asset anywhere in the video
RECENT_S = 60.0
RECENT_PENALTY = 10.0  # extra when the earlier use is within RECENT_S
SEGMENT_GAP_S = 3.0  # video ranges that overlap or start closer than this are the same segment
CROSS_SCENE_SEGMENT_PENALTY = 40.0  # same segment again in another scene (banned within a scene and next shot)
NEEDS_MOTION_FLAG = "needs_motion_graphic"  # readable text subject that no sourced candidate covered
REAL_BAR = 55.0  # a real candidate at or above this beats a photo-style AI still
FLOOR = 50.0  # nothing below this ships while an alternative exists
BELOW_FLOOR_FLAG = "below_floor"  # nothing met the floor and generation failed: least bad candidate kept
BORROWED_FLAG = "borrowed_from_beat"  # a real candidate taken from another shot of the same beat
BORROW_TRIES = 3  # beat candidates tried for one floor slot (each may cost a judge call)
DESIGNED_FLAG = "designed_animation"  # marker on an ai_image shot whose pick is designed animation
SHOT_FLAGS = {"mock", "low_res", "small_archival", "nc_nd", "anachronism", "no_match_accepted",
              "standard_youtube_licence", NEEDS_MOTION_FLAG, BELOW_FLOOR_FLAG, BORROWED_FLAG, DESIGNED_FLAG}
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
        notes = dict(zip([s.id for s in failing], pool.map(
            lambda s: _resolve(s, ctx, threshold, _beat_siblings(shotlist.shots, s)), failing)))
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


def _clean(c: Candidate, shot: Shot) -> bool:
    """No hard reject and no branding note (subscribe, channel, end screen, watermark, logo)."""
    return c.reject_reason is None and not candidate_branding(c, shot)


def _ok(c: Candidate, shot: Shot, threshold: float) -> bool:
    return _clean(c, shot) and c.score is not None and c.score >= threshold


def _passing(shot: Shot, threshold: float) -> bool:
    return any(_ok(c, shot, threshold) for c in shot.candidates)


def _real_ok(shot: Shot) -> bool:
    """A clean real candidate at or above REAL_BAR: the shot keeps real material, no AI still."""
    return any(is_real(c) and _ok(c, shot, REAL_BAR) for c in shot.candidates)


def _ai_tried(shot: Shot) -> bool:
    return any(picture_kind(c) == AI_PHOTO for c in shot.candidates)


def _ranked(shot: Shot) -> list[Candidate]:
    """Best first: clean candidates, then those without a hard reject, then judged before unjudged, then by score."""
    return sorted(shot.candidates, key=lambda c: (_clean(c, shot), c.reject_reason is None, c.score is not None,
                                                  c.score or 0.0), reverse=True)


def asset_key(c: Candidate) -> str:
    """Identity of the underlying asset: canonical URL first, since files are copied per shot."""
    return c.url or c.local_path or c.id


# ---------------------------------------------------------------- AI shots

def _seed_ai(shots: list[Shot], ctx: Any) -> None:
    """Director planned ai_image / ai_video shots get a generated still when they have none."""
    todo = [s for s in shots if s.type in AI_TYPES and not _ai_tried(s)]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=RESOLVE_WORKERS) as pool:
        list(pool.map(lambda s: add_ai(s, ctx), todo))


def add_ai(shot: Shot, ctx: Any, prompt: Optional[str] = None) -> Candidate:
    """Generate an AI still, judge it, and regenerate a failed one with what it must show and the judge's
    reason as direction, up to settings.judge.ai_retries times. Every take is appended; the best is returned.
    No still is used unjudged. A failed retry keeps the takes made so far; a failed first take raises.
    A mock judge score is image statistics, not content, so it never buys a retry."""
    from ..visuals.ai_images import generate_ai_candidate
    threshold = float(ctx.settings.judge.threshold)
    retries = int(getattr(ctx.settings.judge, "ai_retries", 2))
    takes: list[Candidate] = []
    problem = ""
    for take in range(1 + max(0, retries)):
        try:
            cand = generate_ai_candidate(shot, ctx, prompt=prompt, problem=problem)
        except Exception as e:  # noqa: BLE001  a later take failing keeps the earlier ones
            if not takes:
                raise
            ctx.log.warn("AI still retry failed", shot=shot.id, take=take + 1, error=str(e)[:300])
            break
        shot.candidates.append(cand)
        _prefilter_and_score([(shot, cand)], ctx, workers=1)
        from .ai_checks import check_still; check_still(shot, cand, ctx)  # noqa: E702 fixed-question still check
        takes.append(cand)
        if _ok(cand, shot, threshold) or JUDGE_MOCK_FLAG in cand.flags:
            break
        problem = "; ".join(x for x in (cand.reject_reason, cand.judge_reason) if x)[:300] or "judged below threshold"
        ctx.log.info("AI still below threshold, regenerating", shot=shot.id, take=take + 1, score=cand.score)
    return max(takes, key=lambda c: (_clean(c, shot), c.score or 0.0))  # ties keep the earlier take


# ---------------------------------------------------------------- attempts 2 and 3+

def _resolve(shot: Shot, ctx: Any, threshold: float, siblings: Sequence[Shot] = ()) -> list[str]:
    """Run further attempts on a failing shot. Returns notes for judge_notes.

    A shot with a real candidate at or above REAL_BAR may requery but is never converted to an AI still:
    the real one wins selection anyway, so the still would only cost generation time and credits. Nor is
    a shot with nothing at the floor while another shot of its beat holds such a real candidate: selection
    borrows it (see _beat_pool), and still generates there when the borrow is refused (banned, recent, failed judge).
    """
    notes: list[str] = []
    try:
        while shot.attempts < ctx.settings.judge.max_attempts and not _passing(shot, threshold):
            shot.attempts += 1
            if shot.attempts == 2 and shot.type in SOURCED_TYPES:
                notes.append(_requery(shot, ctx))
                continue
            if _ai_tried(shot):
                break  # add_ai already spent the AI retries on this shot
            if _real_ok(shot):
                notes.append(f"real candidate at or above {REAL_BAR:g} kept, no AI still")
                break
            if not any(_shippable(c, shot) for c in shot.candidates) and _beat_real(shot, siblings):
                notes.append("a real candidate of the same beat will be borrowed, no AI still")
                break
            if shot.type not in AI_TYPES:
                if needs_readable_text(shot):
                    notes.append("subject is readable text, kept the best sourced candidate for a motion graphic")
                    break
                shot.type = ShotType.ai_image
                notes.append(f"converted to AI image at attempt {shot.attempts}")
            add_ai(shot, ctx)
    except Exception as e:  # noqa: BLE001  keep what we have, selection still guarantees a pick
        ctx.log.warn("director loop attempt failed", shot=shot.id, error=str(e)[:300])
        notes.append(f"attempt {shot.attempts} failed: {type(e).__name__}")
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
    """Director seat rewrite of the queries; deterministic broadening when the seat gives nothing usable
    or fails (a provider error or a bad reply), so a requery failure never ends the shot's resolve."""
    user = (f"Shot intent: {shot.intent}\nMust show: {', '.join(shot.must_show) or 'nothing specific'}\n"
            f"Must avoid: {', '.join(shot.must_avoid) or 'nothing specific'}\n"
            f"Queries tried: {'; '.join(shot.queries) or 'none'}\nJudge reasons: {'; '.join(reasons) or 'none'}")
    try:
        raw = ctx.seats.llm_json("director", "requery", REQUERY_SYSTEM, user, ctx.stage or "judge")
    except ProviderError as e:  # BadJSON and transport failures included
        ctx.log.warn("requery seat failed, heuristic queries", shot=shot.id, error=str(e)[:300])
        raw = None
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

@dataclass(frozen=True)
class _Pick:
    """One selection: shot start, shot index, scene and the picked candidate."""
    at: float
    index: int
    scene_id: str
    cand: Candidate


SEGMENT_REPEAT_FLAG = "segment_repeat"  # no allowed option was left, a banned segment was kept
_WINDOW_STEM = re.compile(r"^(.*)_(\d+)$")
WINDOWED_SOURCES = {SourceKind.youtube}  # adapters that download a window named <prefix>_<start s>


def _timeline(c: Candidate) -> tuple[str, float]:
    """(file family, offset): in_point + offset is time in the family's source.

    A windowed source names each downloaded window <prefix>_<start s>, so its
    files share one timeline. Any other file is its own family at offset 0,
    which also covers per shot copies of one file (same name).
    """
    stem = Path(c.local_path).stem if c.local_path else ""
    m = _WINDOW_STEM.match(stem) if c.source in WINDOWED_SOURCES else None
    return (m.group(1), float(m.group(2))) if m else (stem, 0.0)


def same_segment(a: Candidate, b: Candidate) -> bool:
    """Same video asset whose time ranges overlap or start within SEGMENT_GAP_S.

    Stills are never a segment (they repeat with a derived framing). Two files
    of one URL that share no window timeline cannot be proven apart, so they
    count as the same segment.
    """
    if a.media != "video" or b.media != "video" or asset_key(a) != asset_key(b):
        return False
    (fa, oa), (fb, ob) = _timeline(a), _timeline(b)
    if fa != fb:
        return True
    a0, b0 = a.in_point + oa, b.in_point + ob
    a1, b1 = max(a.out_point + oa, a0), max(b.out_point + ob, b0)
    return abs(a0 - b0) < SEGMENT_GAP_S or (a0 < b1 and b0 < a1)


class _Usage:
    """Selections in timeline order (motion graphics skipped), with scene and shot index.

    picks hold the shots before the one being selected; ahead holds already fixed
    shots after it (an editor re-source), nearest first.
    """

    def __init__(self) -> None:
        self.picks: list[_Pick] = []
        self.ahead: list[_Pick] = []

    def _all(self) -> list[_Pick]:
        return self.picks + self.ahead

    def neighbours(self) -> list[_Pick]:
        """The shot directly before and the fixed shot directly after."""
        return self.picks[-1:] + self.ahead[:1]

    def penalty(self, c: Candidate, at: float) -> float:
        """Score points to subtract for reusing c at time at: any reuse, reuse within 60 s, same segment."""
        key, p = asset_key(c), 0.0
        for pick in self._all():
            if asset_key(pick.cand) != key:
                continue
            p += DIVERSITY_GAP
            if abs(at - pick.at) <= RECENT_S:
                p += RECENT_PENALTY
            if same_segment(c, pick.cand):
                p += CROSS_SCENE_SEGMENT_PENALTY  # stills may repeat with a new framing; a video needs a new segment
        return p

    def banned(self, c: Candidate, scene_id: str) -> bool:
        """Same segment as a neighbouring shot, or as any other shot of this scene."""
        if any(same_segment(c, p.cand) for p in self.neighbours()):
            return True
        return any(p.scene_id == scene_id and same_segment(c, p.cand) for p in self._all())

    def add(self, c: Candidate, at: float, scene_id: str = "", index: int = -1) -> None:
        self.picks.append(_Pick(at, index, scene_id, c))


def _context_usage(context: ShotList, shot: Shot) -> _Usage:
    """Usage seeded with every other shot's current selection, split around shot's position."""
    usage, before = _Usage(), True
    for other in context.shots:
        if other.id == shot.id:
            before = False
            continue
        c = other.selected_candidate() if other.type != ShotType.motion_graphic else None
        if c is not None:
            (usage.picks if before else usage.ahead).append(_Pick(other.start, other.index, other.scene_id, c))
    return usage


def _beat_siblings(shots: list[Shot], shot: Shot) -> list[Shot]:
    return [o for o in shots if o.beat_id == shot.beat_id and o.id != shot.id and o.type != ShotType.motion_graphic]


def _select_all(shotlist: ShotList, ctx: Any, threshold: float, notes: dict[str, list[str]]) -> None:
    """Pick one candidate per shot: segment rule first, then reuse penalties anywhere in the video."""
    usage = _Usage()
    for shot in shotlist.shots:
        if shot.type == ShotType.motion_graphic:
            continue
        _ensure_usable(shot, ctx)
        pick, accepted, repeat, below = _choose(shot, usage, threshold, ctx, _beat_siblings(shotlist.shots, shot))
        _apply(shot, pick, accepted, notes.get(shot.id, []), repeat, below)
        usage.add(pick, shot.start, shot.scene_id, shot.index)


def enforce_segment_rule(context: ShotList, shot: Shot, ctx: Any) -> bool:
    """Re-select shot when its pick repeats a segment of a neighbour or of its scene in context.

    For a shot judged on its own (editor re-source), whose selection never saw the
    rest of the timeline. Returns True when the selection was redone.
    """
    pick = shot.selected_candidate()
    usage = _context_usage(context, shot)
    if pick is None or pick.source == SourceKind.upload or not usage.banned(pick, shot.scene_id):
        return False
    _ensure_usable(shot, ctx)
    new, accepted, repeat, below = _choose(shot, usage, float(ctx.settings.judge.threshold), ctx,
                                           _beat_siblings(context.shots, shot))
    _apply(shot, new, accepted, [], repeat, below)
    return True


def _choose(shot: Shot, usage: _Usage, threshold: float, ctx: Any,
            siblings: Sequence[Shot] = ()) -> tuple[Candidate, bool, bool, bool]:
    """(pick, accepted below threshold, repeats a banned segment, below floor), the segment ban and the
    floor applied before the threshold.

    Tiers: allowed passing, allowed at or above the floor (accepted), a real candidate borrowed from the
    same beat, a fresh AI still, and only then the least bad candidate (below floor). Among allowed
    candidates a real one at or above REAL_BAR removes photo-style AI stills from the pool.
    """
    pinned = shot.selected_candidate()
    if pinned is not None and pinned.source == SourceKind.upload:
        return pinned, False, False, False
    allowed = _allowed_usable(shot, usage) or _borrow(shot, usage, siblings, ctx)
    if not allowed and _try_ai(shot, ctx):
        allowed = _allowed_usable(shot, usage)
    if allowed:
        if any(is_real(c) and (c.score or 0.0) >= REAL_BAR for c in allowed):
            allowed = [c for c in allowed if picture_kind(c) != AI_PHOTO]
        passing = [c for c in allowed if _ok(c, shot, threshold)]
        return _diverse(passing or allowed, usage, shot, ctx), not passing, False, False
    ranked = _ranked(shot)
    tier = ([c for c in ranked if _clean(c, shot)] or [c for c in ranked if c.reject_reason is None] or ranked[:1])
    pick = _diverse(tier, usage, shot, ctx)
    below = not (_clean(pick, shot) and (pick.score or 0.0) >= FLOOR)
    if below:
        ctx.log.warn("no candidate meets the floor, least bad kept", shot=shot.id, pick=pick.id, score=pick.score,
                     note=(pick.reject_reason or candidate_branding(pick, shot) or "")[:120])
    return pick, True, usage.banned(pick, shot.scene_id), below


def _shippable(c: Candidate, shot: Shot) -> bool:
    return _ok(c, shot, FLOOR)


def _allowed_usable(shot: Shot, usage: _Usage) -> list[Candidate]:
    return [c for c in _ranked(shot) if _shippable(c, shot) and not usage.banned(c, shot.scene_id)]


def _beat_pool(shot: Shot, siblings: Sequence[Shot]) -> list[tuple[Shot, Candidate]]:
    """(donor, candidate) pairs a floor slot may borrow: real, clean, at or above REAL_BAR for the donor.

    Never a sibling's own pick or best candidate: taking it moves the problem to that sibling (a banned
    neighbour segment, or the same photo twice in a row). Never an asset the shot already holds: it was
    judged on this shot's words and failed.
    """
    taken = {asset_key(c) for o in siblings for c in [*_ranked(o)[:1], o.selected_candidate()] if c is not None}
    taken |= {asset_key(c) for c in list(shot.candidates)}
    return [(o, c) for o in siblings for c in list(o.candidates)
            if is_real(c) and _ok(c, o, REAL_BAR) and asset_key(c) not in taken]


def _beat_real(shot: Shot, siblings: Sequence[Shot]) -> bool:
    return bool(_beat_pool(shot, siblings))


def _avoids_covered(shot: Shot, donor: Shot) -> bool:
    """The donor's judge already checked every must_avoid item of shot."""
    def norm(items: list[str]) -> set[str]:
        return {x.strip().lower() for x in items}
    return norm(shot.must_avoid) <= norm(donor.must_avoid)


def _borrow(shot: Shot, usage: _Usage, siblings: Sequence[Shot], ctx: Any) -> list[Candidate]:
    """A real candidate of another shot in the same beat (see _beat_pool), not a banned segment here and not
    shown in the last minute, copied into shot, best first. Its score carries over when the donor's brief
    covered this shot's must_avoid; otherwise the copy is judged for this shot (one vision call, cheaper than
    a generated still) and must reach the floor."""
    pool = [(o, c) for o, c in _beat_pool(shot, siblings) if not usage.banned(c, shot.scene_id)
            and usage.penalty(c, shot.start) < DIVERSITY_GAP + RECENT_PENALTY]  # not shown in the last minute
    pool.sort(key=lambda oc: (oc[1].score or 0.0) - usage.penalty(oc[1], shot.start), reverse=True)
    for donor, best in pool[:BORROW_TRIES]:
        ids, n = {c.id for c in shot.candidates}, 0
        while f"{shot.id}_{source_value(best)}_b{n}" in ids:
            n += 1
        update: dict[str, Any] = {"id": f"{shot.id}_{source_value(best)}_b{n}",
                                  "flags": sorted(set(best.flags) | {BORROWED_FLAG})}
        rejudge = not _avoids_covered(shot, donor)
        if rejudge:
            update.update(score=None, judge_reason=None, reject_reason=None)
        added = add_candidates(shot, [best.model_copy(deep=True, update=update)])
        if not added:
            continue
        copy = added[0]
        if rejudge:
            _prefilter_and_score([(shot, copy)], ctx, workers=1)
            if not _shippable(copy, shot):
                continue
        ctx.log.info("no own candidate meets the floor, real candidate borrowed from the beat", shot=shot.id,
                     candidate=copy.id, origin=best.id, score=copy.score, rejudged=rejudge)
        return [copy]
    return []


def _try_ai(shot: Shot, ctx: Any) -> bool:
    """Generate an AI still (never a segment) unless one was already tried. True when one was added."""
    if _ai_tried(shot) or needs_readable_text(shot):
        return False
    try:
        add_ai(shot, ctx)
    except Exception as e:  # noqa: BLE001  generation failure falls back to the least penalised repeat
        ctx.log.warn("AI still for a repeated segment failed", shot=shot.id, error=str(e)[:300])
        return False
    ctx.log.info("every usable candidate repeats a used segment, AI image added", shot=shot.id)
    return True


def _ensure_usable(shot: Shot, ctx: Any) -> None:
    """Never empty: generate an AI still when there is no candidate at all (a failure raises).

    Candidates that only fail the floor are handled by _choose, which borrows from the beat first.
    """
    if shot.candidates:
        return
    ctx.log.info("no candidate, generating AI image", shot=shot.id)
    add_ai(shot, ctx)


def _diverse(pool: list[Candidate], usage: _Usage, shot: Shot, ctx: Any = None) -> Candidate:
    """Best of pool by judge score minus the reuse penalty, banned segments excluded; ties keep the judge order.

    When every candidate is banned the shot still gets one: a candidate that repeats
    neither neighbouring shot's segment first, then the least penalised.
    """
    def net(c: Candidate) -> float:
        return (c.score or 0.0) - usage.penalty(c, shot.start)

    allowed = [c for c in pool if not usage.banned(c, shot.scene_id)]
    if allowed:
        return max(allowed, key=net)
    near = [p.cand for p in usage.neighbours()]
    pick = max(pool, key=lambda c: (not any(same_segment(c, n) for n in near), net(c)))
    if ctx is not None:
        ctx.log.warn("every candidate repeats a used segment, least penalised kept", shot=shot.id, pick=pick.id)
    return pick


def _apply(shot: Shot, pick: Candidate, accepted: bool, notes: list[str], repeat: bool = False,
           below: bool = False) -> None:
    shot.selected = pick.id
    shot.judge_score = pick.score
    shot.flags = [f for f in shot.flags if f not in SHOT_FLAGS and f != SEGMENT_REPEAT_FLAG]
    if "mock" in pick.flags or JUDGE_MOCK_FLAG in pick.flags:
        shot.flags.append("mock")
    shot.flags += [f for f in ("low_res", "small_archival", "nc_nd", "anachronism", "standard_youtube_licence",
                               BORROWED_FLAG) if f in pick.flags]
    if accepted:
        shot.flags.append("no_match_accepted")
    if needs_readable_text(shot) and (accepted or picture_kind(pick) == AI_PHOTO):
        shot.flags.append(NEEDS_MOTION_FLAG)
    if repeat:
        shot.flags.append(SEGMENT_REPEAT_FLAG)
    if below:
        shot.flags.append(BELOW_FLOOR_FLAG)
    tail = (["best available accepted"] if accepted else []) + (["repeats a used segment"] if repeat else []) + \
        (["nothing met the floor"] if below else [])
    reason = pick.reject_reason or pick.judge_reason or ""
    shot.judge_notes = " | ".join([reason] + notes + tail)
    sync_type(shot)


def sync_type(shot: Shot) -> None:
    """Leave shot.type consistent with its pick, after any path that changes the pick.

    A photo-style AI pick makes the shot ai_image (ai_video for a clip, a planned ai_video keeps its type
    until it is animated); designed animation is ai_image plus the designed_animation marker until it
    has a type of its own; a real pick is footage or image by its media, the overlay variant kept when
    the shot has an overlay. The overlay spec is never touched. Motion graphics are left alone.
    """
    pick = shot.selected_candidate()
    if pick is None or shot.type == ShotType.motion_graphic:
        return
    kind = picture_kind(pick)
    shot.flags = [f for f in shot.flags if f != DESIGNED_FLAG]
    if kind == GRAPHIC:
        return
    if kind in (AI_PHOTO, DESIGNED):
        if pick.media == "video" and kind == AI_PHOTO:
            shot.type = ShotType.ai_video
        elif shot.type not in AI_TYPES:
            shot.type = ShotType.ai_image
        if kind == DESIGNED:
            shot.flags.append(DESIGNED_FLAG)
        return
    overlay = shot.overlay is not None or shot.type in OVERLAY_TYPES
    if pick.media == "video":
        shot.type = ShotType.footage_overlay if overlay else ShotType.footage
    else:
        shot.type = ShotType.image_overlay if overlay else ShotType.image


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
    sync_type(shot)
