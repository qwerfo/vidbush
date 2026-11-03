"""Timeline helpers for edit ops: shot references, splitting, merging, retiming.

Invariants kept by every helper (checked by check_timeline): shots cover
[0, total] in order with no gaps or overlaps, no shot is shorter than
MIN_SHOT seconds, the total stays at narration duration + TAIL, and every
motion spec lasts exactly as long as its shot.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional, Union

from ..models import (FOOTAGE_TYPES, Candidate, MotionSpec, Shot, ShotList, ShotType, SourceKind)

MIN_SHOT = 1.5
TAIL = 1.0
EPS = 1e-3
DERIVED_CROPS: list[list[float]] = [[0.16, 0.14, 0.68, 0.68], [0.06, 0.24, 0.6, 0.6], [0.34, 0.08, 0.6, 0.6],
                                    [0.27, 0.27, 0.46, 0.46]]
FRAMINGS = ("tight", "detail")
BASE_TYPE = {ShotType.footage_overlay: ShotType.footage, ShotType.image_overlay: ShotType.image}

ShotRef = Union[str, int, float]


class EditError(ValueError):
    """An op that cannot be applied; the message is shown to the operator."""


# ---------------------------------------------------------------- references

def resolve_shot(shotlist: ShotList, ref: Any) -> Shot:
    """Shot by id ("shot_007") or 1-based number as shown to the operator (7, "7", "#7", "shot 7")."""
    if isinstance(ref, bool) or ref is None:
        raise EditError(f"missing shot reference: {ref!r}")
    if isinstance(ref, str):
        text = ref.strip()
        for s in shotlist.shots:
            if s.id == text:
                return s
        m = re.fullmatch(r"(?:shot\s*)?#?\s*(\d+)", text, re.I)
        if not m:
            raise EditError(f"unknown shot {ref!r}")
        ref = int(m.group(1))
    if isinstance(ref, float) and not ref.is_integer():
        raise EditError(f"shot number must be whole: {ref}")
    n = int(ref)
    if not 1 <= n <= len(shotlist.shots):
        raise EditError(f"shot {n} is out of range (1 to {len(shotlist.shots)})")
    return shotlist.shots[n - 1]


def position(shotlist: ShotList, shot: Shot) -> int:
    """0-based position of a shot in the list."""
    for i, s in enumerate(shotlist.shots):
        if s.id == shot.id:
            return i
    raise EditError(f"shot {shot.id} is not in the shot list")


def new_shot_id(shotlist: ShotList, base: str) -> str:
    """Unique id derived from base: shot_007 -> shot_007b, shot_007c, ..."""
    taken = {s.id for s in shotlist.shots}
    m = re.match(r"^(.*?\d+)[a-z]?$", base)
    root = m.group(1) if m else base
    for ch in "bcdefghijklmnopqrstuvwxyz":
        if root + ch not in taken:
            return root + ch
    n = 2
    while f"{root}_{n}" in taken:
        n += 1
    return f"{root}_{n}"


# ---------------------------------------------------------------- invariants

def finalize(shotlist: ShotList, total: float) -> None:
    """Round durations, absorb rounding drift in the last shot, reindex, retime motion specs."""
    for s in shotlist.shots:
        s.duration = round(s.duration, 3)
    if shotlist.shots:
        rest = sum(s.duration for s in shotlist.shots[:-1])
        shotlist.shots[-1].duration = round(total - rest, 3)
    shotlist.reindex()
    for s in shotlist.shots:
        if s.motion is not None:
            s.motion = retime_motion(s.motion, s.duration)
        if s.overlay is not None:
            s.overlay = retime_motion(s.overlay, s.duration)


def retime_motion(spec: MotionSpec, duration: float) -> MotionSpec:
    """Spec with the new duration (reveal timings scaled by the mg builders when available)."""
    if abs(spec.duration - duration) < EPS:
        return spec
    try:
        from ..mg.builders import retime_spec
        return retime_spec(spec, duration)
    except ImportError:
        return spec.model_copy(update={"duration": round(duration, 3)})


def check_timeline(shotlist: ShotList, total: float) -> None:
    """Raise EditError when the shot list breaks a timeline invariant."""
    if not shotlist.shots:
        raise EditError("the shot list would be empty")
    t = 0.0
    for s in shotlist.shots:
        if abs(s.start - t) > 2 * EPS or abs(s.end - s.start - s.duration) > 2 * EPS:
            raise EditError(f"timeline gap or overlap at {s.id}")
        if s.duration < MIN_SHOT - EPS:
            raise EditError(f"{s.id} would be {s.duration:.2f} s, the minimum is {MIN_SHOT} s")
        t = s.end
    if abs(t - total) > 2 * EPS:
        raise EditError(f"timeline ends at {t:.3f} s instead of {total:.3f} s")


# ---------------------------------------------------------------- candidates for pieces

def _media_for(shot: Shot) -> str:
    sel = shot.selected_candidate()
    if sel is not None:
        return sel.media
    return "video" if shot.type in FOOTAGE_TYPES or shot.type == ShotType.ai_video else "image"


def alternates(shot: Shot, exclude: set[str]) -> list[Candidate]:
    """Usable candidates of the shot's media other than the excluded ids, best score first."""
    media = _media_for(shot)
    pool = [c for c in shot.candidates if c.id not in exclude and c.media == media and c.reject_reason is None
            and c.source != SourceKind.motion and c.local_path]
    return sorted(pool, key=lambda c: -(c.score if c.score is not None else -1.0))


def next_candidate_id(shot: Shot, source: str) -> str:
    """f"{shot.id}_{source}_{n}" with the first free n."""
    used = {c.id for c in shot.candidates}
    n = 1
    while f"{shot.id}_{source}_{n}" in used:
        n += 1
    return f"{shot.id}_{source}_{n}"


def derived_crop(cand: Candidate, rank: int) -> list[float]:
    """A tighter framing of a still or clip: saliency placed when possible, else a fixed punch-in window."""
    if cand.media == "image" and cand.width and cand.height:
        try:
            from ..visuals.derive import framing_crop
            crop = framing_crop(cand.local_path, cand.width, cand.height, FRAMINGS[rank % 2], rank // 2)
            if crop:
                return crop
        except (ImportError, ValueError, OSError):
            pass
    return list(DERIVED_CROPS[rank % len(DERIVED_CROPS)])


def derived_candidate(piece: Shot, base: Candidate, rank: int, offset: float) -> Candidate:
    """Copy of base with a tighter crop; video continues from offset seconds into the clip."""
    upd: dict[str, Any] = {"id": next_candidate_id(piece, base.source.value), "crop": derived_crop(base, rank),
                           "flags": sorted(set(base.flags) | {"derived"})}
    if base.media == "video":
        end = base.out_point if base.out_point > base.in_point else base.in_point + offset + piece.duration
        upd["in_point"] = round(min(base.in_point + offset, max(base.in_point, end - piece.duration)), 3)
    return base.model_copy(update=upd)


# ---------------------------------------------------------------- split and merge

def clone_for_piece(shotlist: ShotList, shot: Shot, duration: float) -> Shot:
    """New shot after `shot`: same beat and intent, no overlay, candidates renamed to the new id."""
    nid = new_shot_id(shotlist, shot.id)
    piece = shot.model_copy(deep=True)
    piece.id, piece.duration = nid, round(duration, 3)
    piece.overlay, piece.overlay_path, piece.kenburns = None, "", None
    piece.render_segment_path = piece.render_hash = ""
    piece.type = BASE_TYPE.get(piece.type, piece.type)
    piece.transition_in = "cut"
    for c in piece.candidates:
        if c.id.startswith(shot.id + "_"):
            c.id = nid + c.id[len(shot.id):]
    if shot.selected and shot.selected.startswith(shot.id + "_"):
        piece.selected = nid + shot.selected[len(shot.id):]
    return piece


def give_piece_visual(piece: Shot, used: set[str], rank: int, offset: float) -> None:
    """Select the best unused alternate on the piece, else a derived crop of its current pick."""
    if piece.type == ShotType.motion_graphic:
        return
    alts = alternates(piece, used)
    if alts:
        piece.selected = alts[0].id
    else:
        base = piece.selected_candidate()
        if base is None:
            return
        cand = derived_candidate(piece, base, rank, offset)
        piece.candidates.append(cand)
        piece.selected = cand.id
        piece.framing = FRAMINGS[rank % 2]
    used.add(piece.selected or "")
    pick = piece.selected_candidate()
    piece.judge_score = pick.score if pick and pick.score is not None else piece.judge_score


def _renamed(cid: Optional[str], src: Shot, dst: Shot) -> str:
    return dst.id + cid[len(src.id):] if cid and cid.startswith(src.id + "_") else (cid or "")


def split_into(shotlist: ShotList, shot: Shot, durations: list[float]) -> list[Shot]:
    """Replace shot by pieces with the given durations; pieces after the first get distinct visuals."""
    if len(durations) < 2:
        return [shot]
    if min(durations) < MIN_SHOT - EPS:
        raise EditError(f"split of {shot.id} would make a piece shorter than {MIN_SHOT} s")
    i = position(shotlist, shot)
    pieces, used, offset = [shot], {shot.selected or ""}, durations[0]
    shot.duration = round(durations[0], 3)
    for k, d in enumerate(durations[1:]):
        piece = clone_for_piece(shotlist, shot, d)
        shotlist.shots.insert(i + k + 1, piece)
        give_piece_visual(piece, {_renamed(u, shot, piece) for u in used}, k, offset)
        used.add(_renamed(piece.selected, piece, shot))
        offset += d
        pieces.append(piece)
    return pieces


def merge_pair(shotlist: ShotList, keep: Shot, drop: Shot) -> Shot:
    """keep absorbs drop's time; drop leaves the list. keep's visual and overlay stay."""
    keep.duration = round(keep.duration + drop.duration, 3)
    shotlist.shots = [s for s in shotlist.shots if s.id != drop.id]
    return keep


# ---------------------------------------------------------------- retime a section

def shots_in_range(shotlist: ShotList, start: float, end: float) -> list[Shot]:
    """Shots overlapping [start, end] by more than a few milliseconds."""
    return [s for s in shotlist.shots if s.end > start + 0.05 and s.start < end - 0.05]


def scene_range(shotlist: ShotList, scene_id: str) -> tuple[float, float]:
    shots = [s for s in shotlist.shots if s.scene_id == scene_id]
    if not shots:
        raise EditError(f"no shots in scene {scene_id!r}")
    return min(s.start for s in shots), max(s.end for s in shots)


def plan_pieces(shots: list[Shot], factor: float) -> dict[str, int]:
    """Pieces per shot so the mean shot length becomes about factor times the old mean.

    Extra cuts go to the shot with the longest current piece that can still be cut without a
    piece under MIN_SHOT. Motion graphics are never cut (they need reading time).
    """
    pieces = {s.id: 1 for s in shots}
    cuttable = [s for s in shots if s.type != ShotType.motion_graphic]
    target = math.ceil(len(shots) / factor - 1e-9)
    while sum(pieces.values()) < target:
        options = [s for s in cuttable if s.duration / (pieces[s.id] + 1) >= MIN_SHOT]
        if not options:
            break
        best = max(options, key=lambda s: s.duration / pieces[s.id])
        pieces[best.id] += 1
    return pieces


def faster(shotlist: ShotList, shots: list[Shot], factor: float) -> list[str]:
    """Split shots in the section; returns ids of every new or changed shot."""
    changed: list[str] = []
    plan = plan_pieces(shots, factor)
    for s in shots:
        n = plan[s.id]
        if n > 1:
            changed += [p.id for p in split_into(shotlist, s, even_split(s.duration, n))]
    return changed


def even_split(duration: float, n: int) -> list[float]:
    """n near-equal durations (3 decimals) summing exactly to duration."""
    base = round(duration / n, 3)
    return [base] * (n - 1) + [round(duration - base * (n - 1), 3)]


def slower(shotlist: ShotList, shots: list[Shot], factor: float) -> list[str]:
    """Merge neighbours in the section until the count is about len/factor; the longer shot keeps its visual."""
    target = max(1, round(len(shots) / factor))
    live = list(shots)
    changed: set[str] = set()
    while len(live) > target:
        a, b = min(zip(live, live[1:]), key=lambda p: _merge_cost(p[0], p[1]))
        keep, drop = (a, b) if a.duration >= b.duration else (b, a)
        if keep is b:
            b.transition_in = a.transition_in
        merge_pair(shotlist, keep, drop)
        live = [s for s in live if s.id != drop.id]
        changed.add(keep.id)
    return sorted(changed)


def _merge_cost(a: Shot, b: Shot) -> float:
    cost = a.duration + b.duration
    if a.scene_id != b.scene_id:
        cost += 100.0
    if ShotType.motion_graphic in (a.type, b.type):
        cost += 50.0
    return cost
