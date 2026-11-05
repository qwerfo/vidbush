"""Structured edit operations on a finished (or partly finished) project.

apply_ops(settings, project_id, ops) -> EditResult loads work/shotlist.json
(plus script.json and narration.json when an op needs them), applies each op
in order on a copy of the shot list, keeps it only when the timeline
invariants hold (see retime.py), saves the shot list, appends every op to
work/edits.jsonl and writes work/edit_plan.json {"force": [stages]}, the
stages that must rerun (also marked pending in project.json, so a failed
render followed by resume still reruns them). apply_and_render runs the
pipeline with that force set; the segment cache means only changed segments
re-render. Both hold the project lock (jobs/lock.py). Ops that touch files
besides the shot list (rewrite_beat) commit them only after the timeline
checks pass.

Shot references accept an id ("shot_007") or the 1-based number shown to the
operator (7 or "7"). Force sets: candidate or upload changes rerun assembly
and packaging; overlay, motion and timing changes add motion; music and
caption style add mix; rewrite_beat reruns motion, mix, assembly, packaging.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError

from ..config import Settings
from ..models import (STAGES, CaptionStyle, Candidate, EditOp, EditOpName, EditResult, MotionKind, MotionSpec,
                      Narration, Project, Script, Shot, ShotList, ShotType, SourceKind, StyleTokens, TextPayload)
from ..util import media_duration, no_em_dash, read_json, write_json
from . import retime
from .retime import MIN_SHOT, TAIL, EditError, resolve_shot

FORCE_VISUAL = ("assembly", "packaging")
FORCE_MOTION = ("motion", "assembly", "packaging")
FORCE_MIX = ("mix", "assembly", "packaging")
FORCE_REWRITE = ("motion", "mix", "assembly", "packaging")
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
COLOR_NAMES = {"white": "#FFFFFF", "yellow": "#FFD84A", "gold": "#F2C14E", "orange": "#FF9F1C", "red": "#E63946",
               "green": "#52B788", "blue": "#4EA8DE", "cyan": "#48CAE4", "pink": "#FF70A6", "purple": "#9D4EDD",
               "black": "#000000", "grey": "#B0B0B0", "gray": "#B0B0B0"}
_LEADING = re.compile(r"^(archival |stock |wide |aerial |historical )*(footage|photograph|photo|image|shot|clip)"
                      r"\s+of\s+", re.I)


@dataclass
class Outcome:
    """What one op did: a short message for the operator and the stages it dirties.

    total is the new timeline length when the op changed the narration; commit runs once the
    shot list passes the timeline checks, rollback when it does not.
    """
    message: str
    force: tuple[str, ...] = ()
    total: Optional[float] = None
    commit: Optional[Callable[[], None]] = None
    rollback: Optional[Callable[[], None]] = None


@dataclass
class EditSession:
    """Project state shared by the ops of one apply_ops call."""
    settings: Settings
    project: Project
    shotlist: ShotList
    total: float
    _ctx: Any = None
    _script: Optional[Script] = None
    _narration: Optional[Narration] = None
    force: set[str] = field(default_factory=set)
    _lengths: dict[str, float] = field(default_factory=dict)

    @property
    def work(self) -> Path:
        return Path(self.project.paths.work)

    def ctx(self) -> Any:
        """StageContext for seats, brand, profile and logging (built on first use)."""
        if self._ctx is None:
            from ..jobs.context import build_context
            self._ctx = build_context(self.settings, self.project, echo=False)
            self._ctx.stage = "edit"
        return self._ctx

    def script(self) -> Script:
        if self._script is None:
            self._script = self.read("script.json", Script)
        return self._script

    def narration(self) -> Narration:
        if self._narration is None:
            self._narration = self.read("narration.json", Narration)
        return self._narration

    def read(self, name: str, model: type) -> Any:
        """A work file with project paths rebased onto the current root (moved projects)."""
        from ..jobs.pipeline import read_work_model
        return read_work_model(self.settings, self.project.id, name, model)

    def probe(self, path: str) -> float:
        """Media file length in seconds, cached for this session."""
        if path not in self._lengths:
            self._lengths[path] = media_duration(path)
        return self._lengths[path]

    def log(self, msg: str, **fields: Any) -> None:
        if self._ctx is not None:
            self._ctx.log.info(msg, **fields)


def open_session(settings: Settings, project_id: str) -> EditSession:
    """Load project.json and work/shotlist.json; the total is narration duration + tail when known."""
    from ..jobs.pipeline import load_project
    project = load_project(settings, project_id)
    work = Path(project.paths.work)
    sl_path = work / "shotlist.json"
    if not sl_path.exists():
        raise EditError(f"project {project_id} has no shot list yet: run it up to the shotlist stage first")
    sess = EditSession(settings=settings, project=project, shotlist=ShotList.model_construct(), total=0.0)
    sess.shotlist = sess.read("shotlist.json", ShotList)
    sess.total = sess.shotlist.duration
    if (work / "narration.json").exists():
        sess.total = round(sess.narration().duration + TAIL, 3)
    return sess


# ---------------------------------------------------------------- shared helpers

def _arg(args: dict[str, Any], *names: str, default: Any = None) -> Any:
    for n in names:
        if n in args and args[n] is not None and args[n] != "":
            return args[n]
    return default


def _need(args: dict[str, Any], *names: str) -> Any:
    v = _arg(args, *names)
    if v is None:
        raise EditError(f"missing argument {names[0]!r}")
    return v


def _float(v: Any, name: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise EditError(f"{name} must be a number, got {v!r}") from e


def type_for(shot: Shot, cand: Candidate) -> ShotType:
    """Shot type that fits a newly selected candidate, keeping the overlay when there is one."""
    if cand.source == SourceKind.motion:
        return ShotType.motion_graphic
    if cand.source == SourceKind.ai_image:
        return ShotType.ai_image
    if cand.source == SourceKind.ai_video:
        return ShotType.ai_video
    has_overlay = shot.overlay is not None
    if cand.media == "video":
        return ShotType.footage_overlay if has_overlay else ShotType.footage
    return ShotType.image_overlay if has_overlay else ShotType.image


def select(shot: Shot, cand: Candidate, flag: str) -> None:
    """Make cand the shot's pick; leaving motion_graphic drops the full-frame spec."""
    if all(c.id != cand.id for c in shot.candidates):
        shot.candidates.append(cand)
    shot.type = type_for(shot, cand)
    if shot.type != ShotType.motion_graphic:
        shot.motion = None
    shot.selected, shot.judge_score = cand.id, cand.score
    shot.flags = sorted((set(shot.flags) - {"no_match_accepted"}) | {flag})


def beat_text(sess: EditSession, beat_id: str) -> str:
    try:
        return next((b.text for b in sess.script().beats() if b.id == beat_id), "")
    except FileNotFoundError:
        return ""


def _subject(intent: str) -> str:
    s = _LEADING.sub("", intent.strip().rstrip("."))
    return s[:1].upper() + s[1:] if s else "Documentary scene"


# ---------------------------------------------------------------- candidate ops

def op_swap_shot(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    shot = resolve_shot(sl, _need(args, "shot_id", "shot"))
    cid = str(_arg(args, "candidate_id", "candidate", default="next"))
    if cid in ("next", "next_best", "auto"):
        alts = retime.alternates(shot, {shot.selected or ""}) or [
            c for c in shot.candidates if c.id != shot.selected and c.source != SourceKind.motion and c.local_path]
        if not alts:
            raise EditError(f"{shot.id} has no other candidate; try change_type to re-source it")
        cand = alts[0]
    else:
        cand = next((c for c in shot.candidates if c.id in (cid, f"{shot.id}_{cid}")), None)
        if cand is None:
            raise EditError(f"{shot.id} has no candidate {cid!r}")
    if cand.source == SourceKind.motion and shot.motion is None:
        raise EditError(f"{cand.id} is a rendered motion graphic without a spec on this shot")
    select(shot, cand, "operator_pick")
    note = f" (judge had rejected it: {cand.reject_reason})" if cand.reject_reason else ""
    return Outcome(f"{shot.id} now uses {cand.id}{note}", FORCE_MOTION if shot.overlay else FORCE_VISUAL)


def _keyframes(cand: Candidate, out_dir: Path) -> list[str]:
    """Keyframes through the source keyframe extractor (ffmpeg for video, the still itself for images)."""
    from ..source.keyframes import extract_keyframes
    return extract_keyframes(cand, out_dir)


def op_upload_shot(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    shot = resolve_shot(sl, _need(args, "shot_id", "shot"))
    src = Path(str(_need(args, "path", "file"))).expanduser()
    if not src.is_file():
        raise EditError(f"file not found: {src}")
    ext = src.suffix.lower()
    if ext not in VIDEO_EXT | IMAGE_EXT:
        raise EditError(f"unsupported file type {ext or '(none)'}: use an image or a video")
    folder = Path(sess.project.paths.sources) / shot.id
    folder.mkdir(parents=True, exist_ok=True)
    cid = retime.next_candidate_id(shot, "upload")
    dest = folder / f"{cid}{ext}"
    shutil.copy2(src, dest)
    media = "video" if ext in VIDEO_EXT else "image"
    cand = Candidate(id=cid, source=SourceKind.upload, local_path=str(dest), media=media, licence="Operator supplied",
                     attribution="Operator supplied", title=src.name, flags=["operator_supplied"])
    try:
        cand.keyframes = _keyframes(cand, folder / "keyframes")
    except Exception as e:  # unreadable media: keep the project as it was
        dest.unlink(missing_ok=True)
        raise EditError(f"could not read {src.name}: {str(e)[:200]}") from e
    write_json(dest.with_name(dest.name + ".json"), {"source": "upload", "licence": cand.licence, "original": str(src),
                                                    "title": src.name, "added_at": time.time()})
    select(shot, cand, "operator_supplied")
    return Outcome(f"{shot.id} now uses your file {src.name}", FORCE_MOTION if shot.overlay else FORCE_VISUAL)


def _ai_candidate(sess: EditSession, shot: Shot, prompt: Optional[str]) -> Candidate:
    from ..visuals.ai_images import generate_ai_candidate
    return generate_ai_candidate(shot, sess.ctx(), prompt=prompt)


def op_regenerate_ai(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    shot = resolve_shot(sl, _need(args, "shot_id", "shot"))
    delta = no_em_dash(str(_arg(args, "prompt_delta", "prompt", default="")).strip())
    prompt = None
    if delta:
        core = shot.ai_prompt.strip() or _subject(shot.intent)
        prompt = f"{core.rstrip('. ')}. {delta.rstrip('.')}."
        shot.ai_prompt = prompt
    cand = _ai_candidate(sess, shot, prompt)
    select(shot, cand, "operator_ai")
    return Outcome(f"{shot.id} has a new AI image {cand.id}", FORCE_MOTION)


# ---------------------------------------------------------------- change_type

def op_change_type(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    shot = resolve_shot(sl, _need(args, "shot_id", "shot"))
    try:
        new = ShotType(str(_need(args, "type")))
    except ValueError as e:
        raise EditError(f"unknown shot type {args.get('type')!r}") from e
    _review_notes(shot, args)
    if new == ShotType.motion_graphic:
        spec = build_motion_spec(sess, shot, args)
        shot.type, shot.motion, shot.overlay, shot.overlay_path = new, spec, None, ""
        shot.selected, shot.judge_score = None, None
        return Outcome(f"{shot.id} is now a {spec.kind.value} motion graphic", FORCE_MOTION)
    if new in (ShotType.ai_image, ShotType.ai_video):
        return _to_ai(sess, shot, new)
    return _resource(sess, sl, shot, new)


def _str_list(value: Any) -> list[str]:
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    return [no_em_dash(" ".join(str(v).split())) for v in items if str(v).strip()]


def _review_notes(shot: Shot, args: dict[str, Any]) -> None:
    """Optional change_type args from the review sheet (review/fixes.py), applied before a re-source.

    add_must_avoid and add_queries extend the shot's lists (no duplicates). reject_selected marks the
    current pick rejected with that reason so the judge cannot choose it again; when must_avoid grew,
    the other judged candidates lose their score so the judge sees the new must_avoid.
    """
    avoid = [a for a in _str_list(args.get("add_must_avoid")) if a not in shot.must_avoid]
    shot.must_avoid += avoid
    shot.queries += [q for q in _str_list(args.get("add_queries")) if q not in shot.queries]
    reason = str(args.get("reject_selected") or "").strip()
    pick = shot.selected_candidate()
    if reason and pick is not None and pick.source != SourceKind.motion:
        pick.reject_reason, pick.score = no_em_dash(reason), 0.0
    if avoid:
        for c in shot.candidates:
            if c.reject_reason is None:
                c.score, c.judge_reason = None, ""


def _to_ai(sess: EditSession, shot: Shot, new: ShotType) -> Outcome:
    cand = _ai_candidate(sess, shot, None)
    select(shot, cand, "operator_ai")
    if new == ShotType.ai_video:
        from ..visuals.ai_video import animate_still
        clip = animate_still(shot, cand, sess.ctx())
        if clip is None:
            return Outcome(f"{shot.id} is now an AI image {cand.id} (no image to video seat available)", FORCE_MOTION)
        select(shot, clip, "operator_ai")
    return Outcome(f"{shot.id} is now {shot.type.value} {shot.selected}", FORCE_MOTION)


def _resource(sess: EditSession, sl: ShotList, shot: Shot, new: ShotType) -> Outcome:
    """Footage or still: re-source this shot, judge it on a one-shot list, then apply the segment rule in context."""
    from ..judge.director_loop import enforce_segment_rule, judge_and_select
    from ..source.registry import source_shot
    shot.type, shot.motion = new, None
    if new not in (ShotType.footage_overlay, ShotType.image_overlay):
        shot.overlay, shot.overlay_path = None, ""
    shot.candidates = [c for c in shot.candidates if c.source != SourceKind.motion]
    ctx = sess.ctx()
    known = {c.id for c in shot.candidates}
    shot.candidates += [c for c in source_shot(shot, ctx) if c.id not in known]
    shot.selected, shot.judge_score = None, None
    one = ShotList(project_id=sl.project_id, fps=sl.fps, width=sl.width, height=sl.height, duration=shot.duration,
                   shots=[shot])
    judged = judge_and_select(one, ctx).shots[0]
    sl.shots[retime.position(sl, shot)] = judged
    if enforce_segment_rule(sl, judged, ctx):
        ctx.log.info("re-sourced pick repeated a neighbouring segment, reselected", shot=judged.id)
    pick = judged.selected_candidate()
    if pick is None:
        raise EditError(f"no usable candidate found for {shot.id} as {new.value}")
    return Outcome(f"{judged.id} re-sourced as {judged.type.value}, picked {pick.id}", FORCE_MOTION)


# ---------------------------------------------------------------- motion specs

_NUMBER_RE = re.compile(r"(\$?\d(?:[\d,]*\d)?(?:\.\d+)?(?:\s*(?:%|percent\b|million\b|billion\b|thousand\b))?)"
                        r"[ \t]*((?:[A-Za-z][A-Za-z\-']*[ \t]*){0,3})")
_YEAR_RE = re.compile(r"1[5-9]\d\d|20\d\d")


def headline(text: str, max_words: int = 8) -> str:
    """First sentence of text, at most max_words words, without trailing punctuation."""
    first = re.split(r"(?<=[.!?])\s+", text.strip())[0] if text.strip() else ""
    words = first.split()
    out = " ".join(words[:max_words]).rstrip(".,;:!?")
    return no_em_dash(out)


def _number(text: str) -> Optional[tuple[str, str]]:
    """(number, label) for the first number of two or more digits; the label is the words after it or its
    sentence."""
    for m in _NUMBER_RE.finditer(text):
        num = m.group(1).strip()
        if re.fullmatch(r"\d", num):
            continue
        label = m.group(2).strip()
        if not label:
            sentence = next((s for s in re.split(r"(?<=[.!?])\s+", text) if num in s), "")
            rest = re.sub(rf"\b(?:in|by|since|from|around|about|of|until)?\s*{re.escape(num)}\s*,?\s*", " ",
                          sentence, count=1, flags=re.I)
            label = headline(rest.strip(" ,"), 6)
        return num, label
    return None


def _validated_spec(data: Any, duration: float, background: str) -> MotionSpec:
    if not isinstance(data, dict):
        raise EditError("motion_spec must be an object with kind and payload")
    try:
        spec = MotionSpec.model_validate({**data, "duration": duration})
        spec.typed_payload()
    except (ValidationError, TypeError) as e:
        raise EditError(f"invalid motion_spec: {str(e)[:300]}") from e
    return spec.model_copy(update={"background": background})


def _motion_kind(kind: str, text: str, context: str) -> tuple[str, Any]:
    """(kind, data): a map with places or a number card with (number, label); the request text decides first."""
    if kind == "map":
        return "map", _places(context)
    if kind == "number_card":
        return "number_card", _number(text) or _number(context)
    if kind:
        return kind, None
    for source in (text, context):
        if places := _places(source):
            return "map", places
        if num := _number(source):
            return "number_card", num
    return "text_overlay", None


def build_motion_spec(sess: EditSession, shot: Shot, args: dict[str, Any]) -> MotionSpec:
    """Full-frame spec from args.motion_spec, else a map, number card or text card from the intent and beat."""
    if _arg(args, "motion_spec", "spec") is not None:
        return _validated_spec(_arg(args, "motion_spec", "spec"), shot.duration, "solid")
    text = str(_arg(args, "intent", default=""))
    context = " ".join([shot.intent, beat_text(sess, shot.beat_id)])
    kind, data = _motion_kind(str(_arg(args, "kind", default="")).lower(), text, context)
    if kind in ("map", "number_card") and not data:
        raise EditError(f"no recognisable {'place' if kind == 'map' else 'number'} for a {kind} on {shot.id}; "
                        "name it in the request")
    palette = sess.ctx().profile.palette
    title = str(_arg(args, "text", default="")) or headline(beat_text(sess, shot.beat_id)) or _subject(shot.intent)
    try:
        from ..mg import builders
    except ImportError:
        return _text_card(shot, title, palette)
    if kind == "map":
        return builders.map_spec(data[:4], duration=shot.duration, palette=palette, route=len(data) > 1)
    if kind == "number_card":
        return builders.number_card_spec(data[0], label=data[1], duration=shot.duration, palette=palette,
                                         count_up=not _YEAR_RE.fullmatch(data[0]))
    return builders.text_overlay_spec(title, duration=shot.duration, palette=palette, background="solid")


def _places(text: str) -> list[tuple[str, float, float]]:
    try:
        from ..mg.geo import find_places
    except ImportError:
        return []
    return list(find_places(text))


def _text_card(shot: Shot, title: str, palette: str) -> MotionSpec:
    """Builder-free fallback: a solid text_overlay card."""
    return MotionSpec(kind=MotionKind.text_overlay, duration=shot.duration, background="solid",
                      style=StyleTokens(palette=palette), payload=TextPayload(title=title).model_dump())


def op_set_overlay(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    shot = resolve_shot(sl, _need(args, "shot_id", "shot"))
    if shot.type == ShotType.motion_graphic:
        raise EditError(f"{shot.id} is a full-frame motion graphic; change its type before adding an overlay")
    shot.overlay = _validated_spec(_need(args, "motion_spec", "spec"), shot.duration, "transparent")
    shot.overlay_path = ""
    shot.type = {ShotType.footage: ShotType.footage_overlay, ShotType.image: ShotType.image_overlay}.get(shot.type,
                                                                                                         shot.type)
    return Outcome(f"{shot.id} has a {shot.overlay.kind.value} overlay", FORCE_MOTION)


def op_remove_overlay(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    shot = resolve_shot(sl, _need(args, "shot_id", "shot"))
    if shot.overlay is None and not shot.overlay_path:
        raise EditError(f"{shot.id} has no overlay")
    shot.overlay, shot.overlay_path = None, ""
    shot.type = retime.BASE_TYPE.get(shot.type, shot.type)
    return Outcome(f"overlay removed from {shot.id}", FORCE_MOTION)


# ---------------------------------------------------------------- timing ops

def op_set_duration(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    """Take or give the time to the next shot (the previous one for the last shot); total unchanged."""
    shot = resolve_shot(sl, _need(args, "shot_id", "shot"))
    want = _float(_need(args, "seconds", "duration"), "seconds")
    if want < MIN_SHOT:
        raise EditError(f"a shot must last at least {MIN_SHOT} s")
    i = retime.position(sl, shot)
    if len(sl.shots) < 2:
        raise EditError("the only shot always lasts the whole video")
    other = sl.shots[i + 1] if i + 1 < len(sl.shots) else sl.shots[i - 1]
    pool = shot.duration + other.duration
    got = round(min(want, pool - MIN_SHOT), 3)
    other.duration = round(pool - got, 3)
    shot.duration = got
    note = "" if abs(got - want) < 1e-3 else f" (limited so {other.id} keeps {MIN_SHOT} s)"
    return Outcome(f"{shot.id} lasts {got:.2f} s, {other.id} {other.duration:.2f} s{note}", FORCE_MOTION)


def op_split_shot(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    shot = resolve_shot(sl, _need(args, "shot_id", "shot"))
    if shot.type == ShotType.motion_graphic:
        raise EditError(f"{shot.id} is a motion graphic; splitting it would play its animation twice. "
                        "Change its type first, or set its duration instead")
    at = _float(_arg(args, "at", "seconds", default=shot.duration / 2), "at")
    if at < MIN_SHOT - 1e-3 or shot.duration - at < MIN_SHOT - 1e-3:
        raise EditError(f"split point {at:.2f} s leaves a piece under {MIN_SHOT} s ({shot.id} lasts "
                        f"{shot.duration:.2f} s)")
    pieces = retime.split_into(sl, shot, [round(at, 3), round(shot.duration - at, 3)])
    return Outcome(f"{shot.id} split at {at:.2f} s into {', '.join(p.id for p in pieces)}", FORCE_MOTION)


def op_merge_shots(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    a = resolve_shot(sl, _need(args, "a", "shot_a", "first"))
    b = resolve_shot(sl, _need(args, "b", "shot_b", "second"))
    ia, ib = retime.position(sl, a), retime.position(sl, b)
    if abs(ia - ib) != 1:
        raise EditError(f"{a.id} and {b.id} are not adjacent")
    first, second = (a, b) if ia < ib else (b, a)
    retime.merge_pair(sl, first, second)
    return Outcome(f"{second.id} merged into {first.id} ({first.duration:.2f} s)", FORCE_MOTION)


def op_reorder(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    """Move a shot to new_index (0-based) or position (1-based); durations move with the shots."""
    shot = resolve_shot(sl, _need(args, "shot_id", "shot"))
    if _arg(args, "new_index") is not None:
        idx = int(_float(args["new_index"], "new_index"))
    else:
        idx = int(_float(_need(args, "position"), "position")) - 1
    if not 0 <= idx < len(sl.shots):
        raise EditError(f"new index {idx} is out of range (0 to {len(sl.shots) - 1})")
    sl.shots.remove(shot)
    sl.shots.insert(idx, shot)
    return Outcome(f"{shot.id} moved to position {idx + 1}", FORCE_MOTION)


def op_retime_section(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    factor = _float(_need(args, "factor"), "factor")
    if not 0.2 <= factor <= 5.0 or abs(factor - 1.0) < 1e-3:
        raise EditError("factor must be between 0.2 and 5 and not 1 (below 1 cuts faster, above 1 slower)")
    if _arg(args, "scene_id") is not None:
        start, end = retime.scene_range(sl, str(args["scene_id"]))
    else:
        rng = _need(args, "range")
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise EditError("range must be [start_s, end_s]")
        start, end = sorted((_float(rng[0], "range start"), _float(rng[1], "range end")))
    shots = retime.shots_in_range(sl, start, end)
    if not shots:
        raise EditError(f"no shots between {start:.1f} s and {end:.1f} s")
    before = len(sl.shots)
    changed = retime.faster(sl, shots, factor) if factor < 1 else retime.slower(sl, shots, factor)
    if not changed:
        raise EditError("the section cannot be retimed further without shots under 1.5 s")
    verb = "faster" if factor < 1 else "slower"
    return Outcome(f"{start:.0f} to {end:.0f} s cut {verb}: {len(shots)} shots became "
                   f"{len(shots) + len(sl.shots) - before}", FORCE_MOTION)


# ---------------------------------------------------------------- music and captions

def music_tracks(settings: Settings, brand: Any) -> dict[str, dict[str, Any]]:
    """Tracks in the brand music folder: file name -> tracks.json metadata (empty dict without one)."""
    from ..audio.music import AUDIO_EXT, music_folder
    folder = music_folder(brand, settings)
    if folder is None:
        return {}
    meta = read_json(folder / "tracks.json") if (folder / "tracks.json").exists() else {}
    files = {p.name for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT}
    return {f: (meta.get(f) or {}) for f in sorted(files)}


def op_change_music(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    track = str(_arg(args, "track", default="")).strip()
    path = sess.work / "music_override.json"
    if track.lower() in ("", "auto", "default", "none"):
        path.unlink(missing_ok=True)
        return Outcome("music back to automatic selection", FORCE_MIX)
    tracks = music_tracks(sess.settings, sess.ctx().brand)
    key = track.lower()
    hit = next((f for f in tracks if key in (f.lower(), Path(f).stem.lower())), None) or next(
        (f for f, m in tracks.items() if key == str(m.get("title", "")).lower()), None) or next(
        (f for f, m in tracks.items() if key == str(m.get("mood", "")).lower()), None)
    if hit is None:
        raise EditError(f"no track {track!r} in the music folder (have: {', '.join(tracks) or 'none'})")
    write_json(path, {"track": hit})
    return Outcome(f"music set to {hit}", FORCE_MIX)


def _caption_value(key: str, value: Any) -> Any:
    if key in ("color", "outline_color", "highlight_color"):
        v = COLOR_NAMES.get(str(value).strip().lower(), str(value).strip())
        if not HEX_RE.match(v):
            raise EditError(f"{key} must be a colour name or #RRGGBB, got {value!r}")
        return v.upper()
    if key == "size":
        return int(min(120, max(24, round(_float(value, "size")))))
    return value


def op_set_caption_style(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    delta = _arg(args, "delta", default=None)
    if delta is None:
        delta = {k: v for k, v in args.items() if k in CaptionStyle.model_fields}
    if not isinstance(delta, dict) or not delta:
        raise EditError("set_caption_style needs a delta of CaptionStyle fields")
    unknown = sorted(set(delta) - set(CaptionStyle.model_fields))
    if unknown:
        raise EditError(f"unknown caption fields {unknown}; allowed: {sorted(CaptionStyle.model_fields)}")
    clean = {k: _caption_value(k, v) for k, v in delta.items()}
    path = sess.work / "profile_override.json"
    override = read_json(path) if path.exists() else {}
    merged = {**(override.get("captions") or {}), **clean}
    try:
        CaptionStyle(**{**sess.ctx().profile.captions.model_dump(), **merged})
    except ValidationError as e:
        raise EditError(f"invalid caption style: {str(e)[:300]}") from e
    write_json(path, {**override, "captions": merged})
    return Outcome("captions: " + ", ".join(f"{k}={v}" for k, v in clean.items()), FORCE_MIX)


# ---------------------------------------------------------------- rewrite_beat

FILLER = (r"\bit is worth noting that\b", r"\bit's worth noting that\b", r"\bas a matter of fact,?", r"\bin fact,?",
          r"\bof course,?", r"\bbasically\b", r"\bactually\b", r"\breally\b", r"\bvery\b", r"\bquite\b",
          r"\bsimply\b", r"\bsomewhat\b", r"\btruly\b", r"\bliterally\b", r"\bessentially\b", r"\bindeed,?")


def _tidy(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s{2,}", " ", text)).strip(" ,;")
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text if re.search(r"[.!?]$", text) else text + "."


def _defill(text: str) -> str:
    for pat in FILLER:
        text = re.sub(pat, "", text, flags=re.I)
    return re.sub(r"\bin order to\b", "to", text, flags=re.I)


def heuristic_rewrite(text: str, instruction: str) -> str:
    """Shorter drops the last sentence or the last clause; punchier cuts filler and splits run-ons."""
    instr = instruction.lower()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    out = text
    if re.search(r"short|trim|tight|concise|brief|cut|condense", instr):
        if len(sentences) >= 2:
            out = " ".join(sentences[:-1])
        else:
            head, _, tail = text.rpartition(",")
            out = head if head and len(head.split()) >= 4 else _defill(text)
    if re.search(r"punch|snapp|dramatic|strong|energ|bold|hook|exciting|crisp", instr):
        out = re.sub(r",\s+and\s+", ". ", _defill(out))
        out = re.sub(r";\s+", ". ", out)
    return no_em_dash(_tidy(out)) if out != text else text


REWRITE_SYSTEM = ("You rewrite one beat of documentary voiceover. Follow the instruction, keep every fact, name, "
                  "date and number that stays, write plain spoken English in one to three sentences, no lists, no "
                  "em dashes. Reply with strict JSON only: {\"text\": \"<new beat text>\"}.")


def rewrite_text(sess: EditSession, beat_id: str, instruction: str) -> str:
    """New beat text from the editor seat (task rewrite_beat), else the heuristic."""
    script = sess.script()
    beats = script.beats()
    i = next(k for k, b in enumerate(beats) if b.id == beat_id)
    prev = beats[i - 1].text if i else ""
    nxt = beats[i + 1].text if i + 1 < len(beats) else ""
    user = (f"Instruction: {instruction}\nPrevious beat: {prev}\nBeat to rewrite: {beats[i].text}\n"
            f"Next beat: {nxt}\nReturn concise JSON.")
    res = sess.ctx().seats.llm_json("editor", "rewrite_beat", REWRITE_SYSTEM, user, stage="edit")
    if isinstance(res, dict) and isinstance(res.get("text"), str) and 2 <= len(res["text"].split()) <= 120:
        return no_em_dash(" ".join(res["text"].split()))
    return heuristic_rewrite(beats[i].text, instruction)


def resolve_beat(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> str:
    """Beat id from beat_id (id or 1-based number) or from a shot reference."""
    beats = sess.script().beats()
    ref = _arg(args, "beat_id", "beat")
    if ref is None:
        return resolve_shot(sl, _need(args, "shot_id", "shot")).beat_id
    ids = [b.id for b in beats]
    if str(ref) in ids:
        return str(ref)
    if re.fullmatch(r"\d+", str(ref)) and 1 <= int(ref) <= len(beats):
        return beats[int(ref) - 1].id
    raise EditError(f"unknown beat {ref!r}")


def fit_beat_shots(sl: ShotList, beat_id: str, span: list[float], delta: float) -> list[str]:
    """Stretch or shrink the shots covering a beat by delta seconds, merging when pieces would be too short."""
    cover = [s for s in sl.shots if s.beat_id == beat_id] or retime.shots_in_range(sl, span[0], span[1])
    if not cover:
        cover = [min(sl.shots, key=lambda s: abs(s.start - span[0]))]
    target = sum(s.duration for s in cover) + delta
    while len(cover) > 1 and target < MIN_SHOT * len(cover):
        drop = min(cover[1:], key=lambda s: s.duration)
        keep = cover[cover.index(drop) - 1]
        retime.merge_pair(sl, keep, drop)
        cover.remove(drop)
    if target < MIN_SHOT:
        i = retime.position(sl, cover[-1])
        donor = sl.shots[i + 1] if i + 1 < len(sl.shots) else sl.shots[i - 1]
        take = MIN_SHOT - target
        if donor.duration - take < MIN_SHOT:
            raise EditError("the rewritten beat is too short to keep its shot")
        donor.duration = round(donor.duration - take, 3)
        target = MIN_SHOT
    current = sum(s.duration for s in cover)
    slack = sum(s.duration - MIN_SHOT for s in cover)
    for s in cover:  # growth is proportional; shrinking takes from the time each shot has above MIN_SHOT
        share = s.duration / current if target >= current else (s.duration - MIN_SHOT) / max(slack, 1e-9)
        s.duration = round(s.duration + (target - current) * share, 3)
    return [s.id for s in cover]


def op_rewrite_beat(sess: EditSession, sl: ShotList, args: dict[str, Any]) -> Outcome:
    """Re-narrate one beat and refit its shots. Atomic: nothing on disk changes unless the op commits."""
    beat_id = resolve_beat(sess, sl, args)
    instruction = str(_arg(args, "instruction", default="tighten")).strip()
    script, nar = sess.script(), sess.narration()
    beat = next(b for b in script.beats() if b.id == beat_id)
    new_text = rewrite_text(sess, beat_id, instruction)
    if new_text.strip() == beat.text.strip():
        return Outcome(f"beat {beat_id} unchanged (nothing to {instruction})")
    wav = sess.work / "narration" / "narration.wav"
    backup = _backup(wav)
    try:
        new_nar, total = _renarrate_and_fit(sess, sl, beat_id, new_text)
    except Exception:
        _restore(backup, wav)
        raise
    new_script = script.model_copy(deep=True)
    next(b for b in new_script.beats() if b.id == beat_id).text = new_text

    def commit() -> None:
        write_json(sess.work / "script.json", new_script)
        write_json(sess.work / "narration.json", new_nar)
        sess._script, sess._narration = new_script, new_nar
        if backup:
            backup.unlink(missing_ok=True)

    delta = new_nar.duration - nar.duration
    return Outcome(f"beat {beat_id} rewritten ({delta:+.2f} s): {new_text}", FORCE_REWRITE, total=total,
                   commit=commit, rollback=lambda: _restore(backup, wav))


def _renarrate_and_fit(sess: EditSession, sl: ShotList, beat_id: str, new_text: str) -> tuple[Narration, float]:
    """New narration for the beat (narration.wav rewritten) and the shot list refit to it."""
    from ..audio.narration import renarrate_beat
    ctx = sess.ctx()
    nar = sess.narration()
    span = list(nar.beat_spans.get(beat_id) or [0.0, 0.0])
    new_nar = renarrate_beat(nar, sess.script(), beat_id, new_text, ctx.brand, ctx.seats, sess.settings, sess.work,
                             ctx.log)
    total = round(new_nar.duration + TAIL, 3)
    fit_beat_shots(sl, beat_id, span, new_nar.duration - nar.duration)
    retime.finalize(sl, total)
    retime.check_timeline(sl, total)
    return new_nar, total


def _backup(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    bak = path.with_name(path.name + ".edit_bak")
    shutil.copy2(path, bak)
    return bak


def _restore(backup: Optional[Path], path: Path) -> None:
    """Put the backed up file back (a failed rewrite_beat must not leave new narration on disk)."""
    if backup and backup.exists():
        backup.replace(path)


# ---------------------------------------------------------------- driver

Handler = Callable[[EditSession, ShotList, dict[str, Any]], Outcome]
HANDLERS: dict[EditOpName, Handler] = {
    EditOpName.swap_shot: op_swap_shot, EditOpName.upload_shot: op_upload_shot,
    EditOpName.regenerate_ai: op_regenerate_ai, EditOpName.set_duration: op_set_duration,
    EditOpName.split_shot: op_split_shot, EditOpName.merge_shots: op_merge_shots,
    EditOpName.change_type: op_change_type, EditOpName.set_overlay: op_set_overlay,
    EditOpName.remove_overlay: op_remove_overlay, EditOpName.reorder: op_reorder,
    EditOpName.retime_section: op_retime_section, EditOpName.change_music: op_change_music,
    EditOpName.set_caption_style: op_set_caption_style, EditOpName.rewrite_beat: op_rewrite_beat,
}


def _fingerprints(sl: ShotList) -> dict[str, str]:
    skip = {"index", "start", "end", "render_segment_path", "render_hash"}
    return {s.id: json.dumps(s.model_dump(mode="json", exclude=skip), sort_keys=True) for s in sl.shots}


def _apply_one(sess: EditSession, op: EditOp) -> Outcome:
    """Apply one op to a copy of the shot list; commit only when the timeline invariants hold."""
    work = sess.shotlist.model_copy(deep=True)
    before = {s.id: s.duration for s in work.shots}
    outcome = HANDLERS[op.op](sess, work, dict(op.args))
    total = outcome.total if outcome.total is not None else sess.total
    try:
        retime.finalize(work, total)
        notes = retime.extend_video_spans(work, before, sess.probe)
        retime.check_timeline(work, total)
    except Exception:
        if outcome.rollback:
            outcome.rollback()
        raise
    if outcome.commit:
        outcome.commit()
    if notes:
        outcome.message += " (" + "; ".join(notes) + ")"
    sess.shotlist, sess.total = work, total
    sess.force.update(outcome.force)
    return outcome


def _record(sess: EditSession, op: EditOp, status: str, message: str) -> None:
    line = json.dumps({"ts": round(time.time(), 3), "op": op.op.value, "args": op.args, "reason": op.reason,
                       "status": status, "message": message}, default=str, ensure_ascii=False)
    with (sess.work / "edits.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def apply_ops(settings: Settings, project_id: str, ops: list[EditOp], job_id: Optional[int] = None) -> EditResult:
    """Apply ops in order under the project lock; failed ops are reported in errors and change nothing.

    The forced stages are marked pending in project.json, so a render that fails part way and is
    resumed later still reruns them. job_id (an apply_ops job) is recorded in work/edit_plan.json
    right after shotlist.json, so a recovered job can tell its ops were already applied.
    """
    from ..jobs.lock import project_lock
    from ..jobs.pipeline import mark_pending, project_dir
    with project_lock(project_dir(settings, project_id), project_id, owner=f"apply_ops {project_id}"):
        sess = open_session(settings, project_id)
        result = _apply_all(sess, ops)
        force = [s for s in STAGES if s in sess.force] if result.applied else []
        result.force = force
        if result.applied:
            write_json(sess.work / "shotlist.json", sess.shotlist)
        write_json(sess.work / "edit_plan.json", {"force": force, "applied": result.applied, "errors": result.errors,
                                                  "dirty_shots": result.dirty_shots, "job_id": job_id,
                                                  "ts": round(time.time(), 3)})
        if force:
            mark_pending(settings, project_id, force)
    return result


def _apply_all(sess: EditSession, ops: list[EditOp]) -> EditResult:
    """Run every op on the session; fills applied, errors and dirty_shots."""
    before = _fingerprints(sess.shotlist)
    result = EditResult(ops=list(ops))
    for op in ops:
        try:
            out = _apply_one(sess, op)
        except Exception as e:  # one bad op must not lose the others; the message reaches the operator
            msg = str(e) if isinstance(e, EditError) else f"{type(e).__name__}: {str(e)[:300]}"
            result.errors.append(f"{op.op.value}: {msg}")
            _record(sess, op, "error", msg)
            sess.log("edit op failed", op=op.op.value, error=msg)
            continue
        result.applied.append(f"{op.op.value}: {out.message}")
        result.needs_renarration |= op.op == EditOpName.rewrite_beat and bool(out.force)
        _record(sess, op, "applied", out.message)
        sess.log("edit op applied", op=op.op.value, message=out.message)
    # no refit here: _apply_one finalizes every committed shot list to sess.total and checks it, so a
    # failed op can never leave the timeline and the narration out of step
    after = _fingerprints(sess.shotlist)
    result.dirty_shots = [sid for sid, fp in after.items() if before.get(sid) != fp]
    return result


def read_plan(settings: Settings, project_id: str) -> dict[str, Any]:
    """work/edit_plan.json written by the last apply_ops ({} when there is none)."""
    from ..jobs.pipeline import project_dir
    path = project_dir(settings, project_id) / "work" / "edit_plan.json"
    return read_json(path) if path.exists() else {}


def apply_and_render(settings: Settings, project_id: str, ops: list[EditOp], echo: bool = True,
                     job_id: Optional[int] = None) -> EditResult:
    """apply_ops, then rerun the forced stages (cached segments are reused), all under the project lock.

    When job_id is the job recorded by the last apply_ops (a job recovered after a crash), the ops
    are not applied again: only the render runs, with the saved force set.
    """
    from ..jobs.lock import project_lock
    from ..jobs.pipeline import project_dir, run_project
    with project_lock(project_dir(settings, project_id), project_id, owner=f"edit {project_id}"):
        plan = read_plan(settings, project_id)
        if job_id is not None and plan.get("job_id") == job_id:
            result = EditResult(ops=list(ops), applied=list(plan.get("applied", [])),
                                errors=list(plan.get("errors", [])), dirty_shots=list(plan.get("dirty_shots", [])),
                                force=list(plan.get("force", [])))
            result.applied.append("recovered: ops were already applied, render only")
        else:
            result = apply_ops(settings, project_id, ops, job_id=job_id)
        if result.force:
            run_project(settings, project_id, force=result.force, echo=echo)
    return result
