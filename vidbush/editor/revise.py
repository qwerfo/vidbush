"""Map a plain-language revision message to EditOps.

map_message(settings, project_id, message) -> list[EditOp]. The editor seat
(task "revise", stage "revise") gets the message, the op vocabulary with
argument schemas, a compact shot list and the scene list, and answers strict
JSON {"ops": [...]}. Ops are validated with pydantic and their shot
references are resolved to ids. When the seat returns nothing usable, a rule
based parser handles common phrasings, including several requests in one
message joined by "and", commas or "then".

Shot numbers in the message are the 1-based positions BEFORE any op runs;
they are resolved to ids here, so later ops still hit the intended shot even
when an earlier op splits or merges shots.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError

from ..config import Settings
from ..models import CaptionStyle, EditOp, EditOpName, Script, ShotList, ShotType
from ..util import no_em_dash, read_json, read_model
from . import retime
from .retime import EditError

# ---------------------------------------------------------------- vocabulary for the seat

VOCABULARY = """Operations (args):
swap_shot {shot_id, candidate_id or "next"}: use another sourced candidate for a shot
upload_shot {shot_id, path}: use a local file
regenerate_ai {shot_id, prompt_delta}: new AI image, prompt_delta is extra direction
set_duration {shot_id, seconds}: set one shot length, the neighbour absorbs the change
split_shot {shot_id, at}: split at seconds into the shot
merge_shots {a, b}: merge two adjacent shots
change_type {shot_id, type, kind?, intent?}: type is one of footage, footage_overlay, image, image_overlay, ai_image, ai_video, motion_graphic; for motion_graphic kind may be map, number_card or text_overlay and intent describes it
set_overlay {shot_id, motion_spec}: motion_spec is {kind, payload}, kinds lower_third, callout, text_overlay with payload {title, subtitle}
remove_overlay {shot_id}
reorder {shot_id, new_index}: new_index is 0-based
retime_section {range: [start_s, end_s] or scene_id, factor}: factor below 1 cuts faster (0.7 typical), above 1 slower (1.4 typical)
change_music {track}: a file name from the music list
set_caption_style {delta}: delta keys from CaptionStyle: size (px at 1080p), color, highlight_color, outline_color, outline, weight, uppercase, enabled, band_height
rewrite_beat {beat_id, instruction}: rewrite and re-narrate one beat"""

SYSTEM = ("You turn a video editor's revision request into edit operations for a long-form documentary. Use only "
          "the listed operations and argument names. Shot ids come from the shot list; shot numbers in the "
          "request are the 1-based numbers in that list. Keep the order of the requests. Reply with strict JSON "
          "only, no prose: {\"ops\": [{\"op\": \"...\", \"args\": {...}, \"reason\": \"<few words>\"}]}.")

REQUIRED: dict[EditOpName, tuple[str, ...]] = {
    EditOpName.swap_shot: ("shot_id",), EditOpName.upload_shot: ("shot_id", "path"),
    EditOpName.regenerate_ai: ("shot_id",), EditOpName.set_duration: ("shot_id", "seconds"),
    EditOpName.split_shot: ("shot_id",), EditOpName.merge_shots: ("a", "b"),
    EditOpName.change_type: ("shot_id", "type"), EditOpName.set_overlay: ("shot_id", "motion_spec"),
    EditOpName.remove_overlay: ("shot_id",), EditOpName.reorder: ("shot_id",),
    EditOpName.retime_section: ("factor",), EditOpName.change_music: ("track",),
    EditOpName.set_caption_style: ("delta",), EditOpName.rewrite_beat: ("instruction",),
}
SHOT_KEYS = ("shot_id", "a", "b")


@dataclass
class RuleContext:
    """Project facts the rules need: shots, beats, scenes, caption size and music tracks."""
    shotlist: ShotList
    script: Optional[Script]
    caption_size: int = 64
    tracks: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_track: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.shotlist.duration


# ---------------------------------------------------------------- entry point

def map_message(settings: Settings, project_id: str, message: str) -> list[EditOp]:
    """EditOps for a revision message: editor seat first, rule based parser as the fallback."""
    from ..jobs.context import build_context
    from ..jobs.pipeline import load_project
    project = load_project(settings, project_id)
    work = Path(project.paths.work)
    shotlist = read_model(work / "shotlist.json", ShotList)
    script = read_model(work / "script.json", Script) if (work / "script.json").exists() else None
    ctx = build_context(settings, project, echo=False)
    ctx.stage = "revise"
    rc = RuleContext(shotlist=shotlist, script=script, caption_size=ctx.profile.captions.size,
                     tracks=_tracks(settings, ctx.brand), current_track=_override(work))
    user = (f"Request: {message}\n\n{VOCABULARY}\n\nShots (number | id | type | start-end s | intent | overlay):\n"
            f"{shot_summary(shotlist)}\n\nScenes:\n{scene_summary(script, shotlist)}\n\n"
            f"Music tracks: {', '.join(rc.tracks) or 'none'}\nReturn concise JSON.")
    raw = ctx.seats.llm_json("editor", "revise", SYSTEM, user, stage="revise", temperature=0.1)
    ops = validate_ops(raw, rc)
    source = "seat"
    if not ops:
        ops, source = rule_ops(message, rc), "rules"
    ctx.log.info("revision mapped", source=source, ops=[o.op.value for o in ops], message=message[:300])
    return ops


def _tracks(settings: Settings, brand: Any) -> dict[str, dict[str, Any]]:
    from .ops import music_tracks
    try:
        return music_tracks(settings, brand)
    except (OSError, ValueError):
        return {}


def _override(work: Path) -> Optional[str]:
    p = work / "music_override.json"
    return str(read_json(p).get("track") or "") or None if p.exists() else None


def shot_summary(shotlist: ShotList) -> str:
    """One compact line per shot for the seat."""
    lines = []
    for i, s in enumerate(shotlist.shots, 1):
        ov = s.overlay.kind.value if s.overlay else "-"
        lines.append(f"{i} | {s.id} | {s.type.value} | {s.start:.1f}-{s.end:.1f} | {s.intent[:70]} | {ov}")
    return "\n".join(lines)


def scene_summary(script: Optional[Script], shotlist: ShotList) -> str:
    """Scene id, title, time span and beat ids."""
    if script is None:
        return "(no script)"
    out = []
    for n, sc in enumerate(script.scenes, 1):
        try:
            a, b = retime.scene_range(shotlist, sc.id)
            span = f"{a:.1f}-{b:.1f}"
        except EditError:
            span = "?"
        out.append(f"{n} | {sc.id} | {sc.title[:50]} | {span} | beats {', '.join(b.id for b in sc.beats)}")
    return "\n".join(out)


# ---------------------------------------------------------------- seat output validation

def validate_ops(raw: Any, rc: RuleContext) -> list[EditOp]:
    """Valid ops from seat JSON with shot references resolved to ids; invalid entries are dropped."""
    items = raw.get("ops") if isinstance(raw, dict) else raw if isinstance(raw, list) else None
    if not isinstance(items, list):
        return []
    out: list[EditOp] = []
    for item in items:
        try:
            op = EditOp.model_validate(item)
            out.append(normalize(op, rc))
        except (ValidationError, EditError, TypeError, ValueError):
            continue
    return out


def normalize(op: EditOp, rc: RuleContext) -> EditOp:
    """Check required args and turn shot numbers into ids; raises EditError when the op is unusable."""
    args = dict(op.args)
    missing = [k for k in REQUIRED[op.op] if args.get(k) in (None, "")]
    if op.op == EditOpName.rewrite_beat and not (args.get("beat_id") or args.get("shot_id")):
        missing.append("beat_id")
    if op.op == EditOpName.retime_section and not (args.get("range") or args.get("scene_id")):
        missing.append("range")
    if op.op == EditOpName.reorder and args.get("new_index") is None and args.get("position") is None:
        missing.append("new_index")
    if missing:
        raise EditError(f"{op.op.value} missing {missing}")
    for key in SHOT_KEYS:
        if key in args:
            args[key] = retime.resolve_shot(rc.shotlist, args[key]).id
    if op.op == EditOpName.change_type:
        ShotType(str(args["type"]))
    if op.op == EditOpName.set_caption_style and not set(args["delta"]) <= set(CaptionStyle.model_fields):
        raise EditError("unknown caption fields")
    return op.model_copy(update={"args": args})


# ---------------------------------------------------------------- clause splitting

_SPLIT = re.compile(r"\s*(?:[;,]|\.\s)\s*(?:and\s+)?(?:then\s+|also\s+)?|\s+(?:and|then|also|plus)\s+(?:then\s+)?",
                    re.I)


def split_clauses(message: str) -> list[str]:
    """Fragments of the message split at conjunctions and punctuation."""
    text = no_em_dash(message.strip().rstrip(".!"))
    return [p.strip() for p in _SPLIT.split(text) if p and p.strip()]


def rule_ops(message: str, rc: RuleContext) -> list[EditOp]:
    """Rule based mapping. A fragment that means nothing alone is joined to the one before it
    ("merge shots 4" + "5", "captions in yellow" + "bigger")."""
    clauses: list[tuple[str, list[EditOp]]] = []
    for frag in split_clauses(message):
        alone = parse_clause(frag, rc)
        if clauses and (not clauses[-1][1] or not alone):
            joined = f"{clauses[-1][0]} and {frag}"
            both = parse_clause(joined, rc)
            if both or not clauses[-1][1]:
                clauses[-1] = (joined, both)
                continue
        clauses.append((frag, alone))
    return [op for _, ops in clauses for op in ops]


def parse_clause(clause: str, rc: RuleContext) -> list[EditOp]:
    """First rule that understands the clause wins."""
    text = clause.lower().strip()
    for rule in RULES:
        try:
            ops = rule(text, clause, rc)
        except EditError:
            return []
        if ops:
            return ops
    return []


# ---------------------------------------------------------------- references and numbers

_REF_HEAD = re.compile(r"\bshots?\b\s*(?:number\s*|no\.?\s*)?#?\s*(\d+)\b")
_REF_MORE = re.compile(r"\s*(?:,|&|\band\b|\bwith\b|\binto\b)\s*(?:the\s+)?(?:shots?\s*)?#?\s*(\d+)\b(?!\s*(?:s|sec|secs|"
                       r"seconds?|minutes?|mins?|x|%)\b)")
_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10, "opening": 1}
_WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
             "sixty": 60, "ninety": 90, "half a": 0.5}
_NUM = r"(\d+(?:\.\d+)?|half a|an|a|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty|fifty|sixty|ninety)"
_SECONDS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)\b")


def _num(token: str) -> float:
    token = token.strip().lower()
    return _WORD_NUM[token] if token in _WORD_NUM else float(token)


def shot_refs(text: str, rc: RuleContext) -> list[str]:
    """Shot ids mentioned in the clause, in order (numbers are 1-based positions now)."""
    n = len(rc.shotlist.shots)
    nums: list[tuple[int, int]] = []
    for m in _REF_HEAD.finditer(text):
        nums.append((m.start(), int(m.group(1))))
        pos = m.end()
        while (mm := _REF_MORE.match(text, pos)) is not None:
            nums.append((mm.start(1), int(mm.group(1))))
            pos = mm.end()
    for m in re.finditer(r"\b(\d+)(?:st|nd|rd|th)\s+shot\b", text):
        nums.append((m.start(), int(m.group(1))))
    for m in re.finditer(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|opening|last|"
                         r"final|closing)\s+shot\b", text):
        word = m.group(1)
        nums.append((m.start(), n if word in ("last", "final", "closing") else _ORDINAL[word]))
    for m in re.finditer(r"\bshot_\d+[a-z]?\b", text):
        ids = [s.id for s in rc.shotlist.shots]
        if m.group(0) in ids:
            nums.append((m.start(), ids.index(m.group(0)) + 1))
    for m in re.finditer(r"\bshot\s+at\s+(\d+):(\d{2})\b", text):
        nums.append((m.start(), _shot_at(rc, int(m.group(1)) * 60 + int(m.group(2)))))
    out: list[str] = []
    for _, k in sorted(nums):
        if not 1 <= k <= n:
            raise EditError(f"shot {k} is out of range")
        sid = rc.shotlist.shots[k - 1].id
        if sid not in out:
            out.append(sid)
    return out


def _shot_at(rc: RuleContext, t: float) -> int:
    for i, s in enumerate(rc.shotlist.shots, 1):
        if s.start <= t < s.end:
            return i
    return len(rc.shotlist.shots)


def _shot(rc: RuleContext, sid: str):
    return rc.shotlist.by_id(sid)


def _op(name: EditOpName, reason: str, **args: Any) -> EditOp:
    return EditOp(op=name, args=args, reason=reason[:200])


# ---------------------------------------------------------------- rules: captions and music

_COLOURS = ("white", "yellow", "gold", "orange", "red", "green", "blue", "cyan", "pink", "purple", "black", "grey",
            "gray")
_COLOUR_RE = re.compile(r"(#[0-9a-f]{6}\b|\b(?:" + "|".join(_COLOURS) + r")\b)")


def _scale_words(text: str, up: float, down: float) -> float:
    if re.search(r"\b(much|a lot|way|lot|really|huge|massive)\b", text):
        return up
    if re.search(r"\b(slightly|a bit|a little|bit|little|tad|somewhat)\b", text):
        return down
    return 0.0


def rule_captions(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not re.search(r"\b(captions?|subtitles?|subs)\b", text):
        return []
    delta: dict[str, Any] = {}
    if m := re.search(r"\b(?:size\s*)?(\d{2,3})\s*(?:px|pt|pixels?|points?)\b|\bsize\s+(\d{2,3})\b", text):
        delta["size"] = int(m.group(1) or m.group(2))
    elif re.search(r"\b(bigger|larger|increase|huge|enlarge|large|big)\b", text):
        delta["size"] = round(rc.caption_size * (_scale_words(text, 1.35, 1.1) or 1.2))
    elif re.search(r"\b(smaller|reduce|tiny|shrink|small)\b", text):
        delta["size"] = round(rc.caption_size * (_scale_words(text, 0.72, 0.92) or 0.85))
    for m in _COLOUR_RE.finditer(text):
        before = text[max(0, m.start() - 22):m.start()]
        key = ("highlight_color" if "highlight" in before else "outline_color" if re.search(r"outline|border|stroke",
                                                                                           before) else "color")
        delta.setdefault(key, m.group(1))
    if re.search(r"\ball caps\b|\bupper ?case\b|\bcapitals?\b|\bcapitali[sz]ed\b", text):
        delta["uppercase"] = True
    elif re.search(r"\blower ?case\b|\bsentence case\b|\bnormal case\b|\bno caps\b", text):
        delta["uppercase"] = False
    if re.search(r"\b(bolder|heavier|thicker font|more bold)\b", text):
        delta["weight"] = "Black"
    elif re.search(r"\b(thinner|less bold|lighter weight|lighter font)\b", text):
        delta["weight"] = "SemiBold"
    if re.search(r"\b(higher|raise|move (?:them |it )?up)\b", text):
        delta["band_height"] = 0.35
    elif re.search(r"\b(move (?:them |it )?down|lower (?:them|it|the captions?|down)|lower in)\b", text):
        delta["band_height"] = 0.65
    if re.search(r"\b(turn off|disable|remove|hide|no captions|without captions|get rid of|drop)\b", text):
        delta = {"enabled": False}
    elif not delta and re.search(r"\b(turn on|enable|add|show|bring back)\b", text):
        delta = {"enabled": True}
    return [_op(EditOpName.set_caption_style, clause, delta=delta)] if delta else []


_MOODS = {"calm": r"calm|relax|gentle|soft|quiet|peaceful|chill|mellow|ambient|softer|quieter",
          "tense": r"tense|dark|dramatic|suspense|ominous|intense|serious|heavy|tension",
          "uplifting": r"uplift|upbeat|happy|happier|positive|hopeful|bright|energetic|epic|inspir|cheer|lively",
          "mysterious": r"myster|eerie|moody|haunting|curious|strange"}


def rule_music(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not re.search(r"\b(music|soundtrack|song|track|background score|music bed)\b", text) or not rc.tracks:
        return []
    for f, meta in rc.tracks.items():
        names = {Path(f).stem.lower().replace("_", " "), str(meta.get("title", "")).lower(), f.lower()}
        if any(n and n in text for n in names):
            return [_op(EditOpName.change_music, clause, track=f)]
    if re.search(r"\b(automatic|default|auto)\b", text):
        return [_op(EditOpName.change_music, clause, track="auto")]
    for mood, pat in _MOODS.items():
        if re.search(rf"\b(?:{pat})", text):
            return [_op(EditOpName.change_music, clause, track=_track_for(rc, mood))]
    if re.search(r"\b(change|different|another|swap|switch|new|replace|other)\b", text):
        names = sorted(rc.tracks)
        cur = rc.current_track if rc.current_track in names else None
        nxt = names[(names.index(cur) + 1) % len(names)] if cur else names[0]
        return [_op(EditOpName.change_music, clause, track=nxt)]
    return []


def _track_for(rc: RuleContext, mood: str) -> str:
    """Track tagged with the mood, else the lowest (calm) or highest (other moods) energy track."""
    tagged = [f for f, m in rc.tracks.items() if str(m.get("mood", "")).lower() == mood]
    if tagged:
        return sorted(tagged, key=lambda f: float(rc.tracks[f].get("energy", 0.5)))[0]
    pick = min if mood == "calm" else max
    return pick(rc.tracks, key=lambda f: float(rc.tracks[f].get("energy", 0.5)))


# ---------------------------------------------------------------- rules: narration

_REWRITE = re.compile(r"\b(punchier|punchy|punch (?:it )?up|snappier|more dramatic|more exciting|more engaging|"
                      r"rewrite|reword|rephrase|tighten|tighter|shorten|shorter|more concise|concise|trim|condense|"
                      r"simplify|simpler|stronger)\b")
_NARRATION = re.compile(r"\b(intro|introduction|opening|hook|outro|ending|conclusion|narration|voiceover|"
                        r"voice over|line|lines|script|sentence|wording|beat|paragraph|first beat|last beat)\b")


def _beat_targets(text: str, rc: RuleContext) -> list[str]:
    beats = rc.script.beats() if rc.script else []
    if not beats:
        return []
    if m := re.search(r"\bbeat\s+(b\d+|\d+)\b", text):
        ref = m.group(1)
        if ref in {b.id for b in beats}:
            return [ref]
        return [beats[int(ref) - 1].id] if ref.isdigit() and 1 <= int(ref) <= len(beats) else []
    if re.search(r"\b(outro|ending|conclusion|last beat|final line|closing)\b", text):
        return [beats[-1].id]
    refs = shot_refs(text, rc)
    if refs:
        return [_shot(rc, refs[0]).beat_id]
    if m := re.search(r"\b(?:scene|chapter|section)\s+(\d+)\b", text):
        k = int(m.group(1))
        return [b.id for b in rc.script.scenes[k - 1].beats[:4]] if 1 <= k <= len(rc.script.scenes) else []
    return [beats[0].id]


def rule_rewrite(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not (_REWRITE.search(text) and _NARRATION.search(text)):
        return []
    return [_op(EditOpName.rewrite_beat, clause, beat_id=b, instruction=clause.strip())
            for b in _beat_targets(text, rc)]


# ---------------------------------------------------------------- rules: structure

def rule_remove_overlay(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not re.search(r"\b(remove|delete|drop|get rid of|take off|take out|hide|kill|lose|no|without|clear)\b", text):
        return []
    if not re.search(r"\b(overlays?|lower thirds?|callouts?|on-screen text|text on|graphic on|title on|labels?)\b",
                     text):
        return []
    if re.search(r"\b(all|every)\b.*\boverlays?\b", text):
        return [_op(EditOpName.remove_overlay, clause, shot_id=s.id) for s in rc.shotlist.shots if s.overlay]
    return [_op(EditOpName.remove_overlay, clause, shot_id=sid) for sid in shot_refs(text, rc)]


def _neighbour(rc: RuleContext, sid: str, step: int) -> Optional[str]:
    ids = [s.id for s in rc.shotlist.shots]
    k = ids.index(sid) + step
    return ids[k] if 0 <= k < len(ids) else None


def rule_merge(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not re.search(r"\b(merge|combine|join)\b", text):
        return []
    refs = shot_refs(text, rc)
    if len(refs) == 1:
        other = _neighbour(rc, refs[0], -1 if re.search(r"\b(previous|prior|before|one before)\b", text) else 1)
        refs = refs + [other] if other else refs
    return [_op(EditOpName.merge_shots, clause, a=refs[0], b=refs[1])] if len(refs) >= 2 else []


def rule_split(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not re.search(r"\bsplit\b|\bcut\b.*\bin (?:two|half)\b|\bbreak\b.*\b(?:up|in two|in half)\b", text):
        return []
    refs = shot_refs(text, rc)
    if not refs:
        return []
    args: dict[str, Any] = {"shot_id": refs[0]}
    if m := re.search(r"\bat\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?\b", text):
        args["at"] = float(m.group(1))
    return [_op(EditOpName.split_shot, clause, **args)]


def rule_reorder(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not re.search(r"\b(move|put|place|reorder|bring|shift)\b", text) or re.search(r"\b(music|captions?)\b", text):
        return []
    refs = shot_refs(text, rc)
    if not refs:
        return []
    ids = [s.id for s in rc.shotlist.shots]
    cur = ids.index(refs[0])
    if m := re.search(r"\bto\s+(?:the\s+)?(?:position|slot|spot|place|number)?\s*#?\s*(\d+)\b(?!\s*(?:s|sec|seconds?))",
                      text):
        idx = int(m.group(1)) - 1
    elif re.search(r"\b(after|before)\b", text) and len(refs) >= 2:
        tgt = ids.index(refs[1])
        after = re.search(r"\bafter\b", text) is not None
        idx = (tgt if cur < tgt else tgt + 1) if after else (tgt - 1 if cur < tgt else tgt)
    elif re.search(r"\b(to the end|to the last|last position|to the back)\b", text):
        idx = len(ids) - 1
    elif re.search(r"\b(to the start|to the beginning|to the front|first position|to the top)\b", text):
        idx = 0
    else:
        return []
    if not 0 <= idx < len(ids):
        raise EditError(f"position {idx + 1} is out of range")
    return [_op(EditOpName.reorder, clause, shot_id=refs[0], new_index=idx)]


def rule_motion(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    """Maps, number cards and full-frame graphics become motion_graphic shots."""
    refs = shot_refs(text, rc)
    if not refs:
        return []
    if re.search(r"\bmaps?\b", text):
        kind = "map"
    elif re.search(r"\b(number card|stat card|statistic|big number)\b", text):
        kind = "number_card"
    elif re.search(r"\b(motion graphics?|graphic|animation|title card|text card|quote card)\b", text) and \
            re.search(r"\b(make|turn|change|convert|replace|use|into|switch)\b", text):
        kind = ""
    else:
        return []
    return [_op(EditOpName.change_type, clause, shot_id=sid, type="motion_graphic", kind=kind, intent=clause)
            for sid in refs]


_QUOTED = re.compile(r"[\"“]([^\"”]{2,80})[\"”]|'([^']{2,80})'")
_SAYING = re.compile(r"\b(?:saying|that says|reading|which says|with the text)\s+(.+?)(?:\s+(?:to|on|over|for)\s+(?:the\s+)?"
                     r"(?:shot|\d+(?:st|nd|rd|th)|first|last)\b.*)?$", re.I)


def rule_add_text(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not re.search(r"\b(add|put|show|place|display|overlay)\b", text) or re.search(r"\bcaptions?\b", text):
        return []
    if not re.search(r"\b(text|title|label|lower third|callout|quote|date|name|words?)\b", text):
        return []
    refs = shot_refs(text, rc)
    if not refs:
        return []
    shot = _shot(rc, refs[0])
    q, say = _QUOTED.search(clause), _SAYING.search(clause)
    words = ((q.group(1) or q.group(2)) if q else say.group(1) if say else "").strip()
    words = words or _beat_headline(rc, shot.beat_id) or shot.intent[:60]
    kind = "lower_third" if re.search(r"\b(lower third|name|label)\b", text) else (
        "callout" if re.search(r"\bcallout\b", text) else "text_overlay")
    position = "lower_left" if kind == "lower_third" else "upper_left" if kind == "callout" else "center"
    spec = {"kind": kind, "background": "transparent", "payload": {"title": no_em_dash(words), "position": position}}
    return [_op(EditOpName.set_overlay, clause, shot_id=refs[0], motion_spec=spec)]


def _beat_headline(rc: RuleContext, beat_id: str) -> str:
    from .ops import headline
    beat = next((b for b in (rc.script.beats() if rc.script else []) if b.id == beat_id), None)
    return headline(beat.text, 6) if beat else ""


_DELTA = re.compile(r"(?:\b(?:showing|depicting|that shows|where|but|make it|with|so that)|:)\s+(.+)$")


def rule_ai(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    ai = re.search(r"\b(ai|a\.i\.|generated|generate|regenerate|redo the image|re-generate)\b", text)
    if not ai:
        return []
    refs = shot_refs(text, rc)
    out = []
    for sid in refs:
        shot = _shot(rc, sid)
        again = re.search(r"\b(another|new|different|regenerate|re-generate|redo|again|retry|try)\b", text)
        if re.search(r"\bregenerate\b|\bre-generate\b", text) or (shot.type == ShotType.ai_image and again):
            m = _DELTA.search(text) or re.search(r"\b(more \w+(?: \w+)?|less \w+(?: \w+)?)\b", text)
            out.append(_op(EditOpName.regenerate_ai, clause, shot_id=sid,
                           prompt_delta=(m.group(1).strip() if m else "")))
        else:
            kind = "ai_video" if re.search(r"\b(ai video|animate|animated|moving)\b", text) else "ai_image"
            out.append(_op(EditOpName.change_type, clause, shot_id=sid, type=kind))
    return out


_TYPE_WORDS = [(r"\b(footage|video|clip|live action|moving)\b", "footage"),
               (r"\b(still|photo|photograph|picture|image|archival photo)\b", "image")]


def rule_type(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not re.search(r"\b(use|make|turn|change|switch|replace|convert|should be|into|instead)\b", text):
        return []
    refs = shot_refs(text, rc)
    if not refs:
        return []
    for pat, base in _TYPE_WORDS:
        if re.search(pat, text):
            out = []
            for sid in refs:
                shot = _shot(rc, sid)
                t = f"{base}_overlay" if shot.overlay is not None else base
                out.append(_op(EditOpName.change_type, clause, shot_id=sid, type=t))
            return out
    return []


def rule_duration(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    refs = shot_refs(text, rc)
    if len(refs) != 1 or re.search(r"\bsplit\b|\bat\s+\d", text):
        return []
    shot = _shot(rc, refs[0])
    rel = re.search(rf"\b{_NUM}\s*(?:s|sec|secs|seconds?)\s+(longer|shorter)\b", text)
    if rel:
        sign = 1 if rel.group(2) == "longer" else -1
        return [_op(EditOpName.set_duration, clause, shot_id=refs[0],
                    seconds=round(max(retime.MIN_SHOT, shot.duration + sign * _num(rel.group(1))), 2))]
    m = _SECONDS.search(text) or re.search(r"\bduration (?:to |of )?(\d+(?:\.\d+)?)\b", text)
    if m:
        return [_op(EditOpName.set_duration, clause, shot_id=refs[0], seconds=float(m.group(1)))]
    if re.search(r"\b(longer|extend|lengthen|linger|hold)\b", text):
        return [_op(EditOpName.set_duration, clause, shot_id=refs[0], seconds=round(min(shot.duration * 1.5,
                                                                                       shot.duration + 3), 2))]
    if re.search(r"\b(shorter|shorten|trim|quicker|briefer)\b", text):
        return [_op(EditOpName.set_duration, clause, shot_id=refs[0],
                    seconds=round(max(retime.MIN_SHOT, shot.duration * 0.67), 2))]
    return []


_CAND = re.compile(r"\b((?:shot_\d+[a-z]?_)?(?:youtube|wikimedia|archive|loc|nasa|pexels|pixabay|local|ai_image|"
                   r"ai_video|upload)_\d+)\b")


def rule_swap(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    if not re.search(r"\b(swap|replace|change|different|another|other|next|alternative|new|better|option|"
                      r"bad|wrong|ugly|boring|doesn'?t fit|does not fit|weird|off)\b", text):
        return []
    refs = shot_refs(text, rc)
    if len(refs) == 2 and re.search(r"\b(swap|switch|exchange|trade)\b", text):
        ids = [s.id for s in rc.shotlist.shots]
        ia, ib = ids.index(refs[0]), ids.index(refs[1])
        return [_op(EditOpName.reorder, clause, shot_id=refs[0], new_index=ib),
                _op(EditOpName.reorder, clause, shot_id=refs[1], new_index=ia)]
    out = []
    for sid in refs:
        shot = _shot(rc, sid)
        if m := _CAND.search(text):
            out.append(_op(EditOpName.swap_shot, clause, shot_id=sid, candidate_id=m.group(1)))
            continue
        alts = retime.alternates(shot, {shot.selected or ""})
        if alts:
            out.append(_op(EditOpName.swap_shot, clause, shot_id=sid, candidate_id=alts[0].id))
        elif shot.type != ShotType.motion_graphic:
            out.append(_op(EditOpName.change_type, clause, shot_id=sid, type=shot.type.value))
    return out


# ---------------------------------------------------------------- rules: pacing

_FASTER = re.compile(r"\b(faster|quicker|speed (?:it |this )?up|sped up|pace up|pick up the pace|more cuts|cut faster|"
                     r"snappier|tighten|tighter|too slow|drags?|dragging|punchier pacing|more dynamic|quicken)\b")
_SLOWER = re.compile(r"\b(slower|slow (?:it |this )?down|calmer pacing|more breathing room|fewer cuts|less cuts|"
                     r"too fast|too quick|too busy|relax the pace|let it breathe|breathe|longer shots|less frantic)\b")
_TIME = r"(\d+:\d{2}|\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds?|m|min|mins|minutes?)?)"


def _seconds(token: str) -> float:
    token = token.strip()
    if ":" in token:
        m, s = token.split(":")
        return int(m) * 60 + int(s)
    val = float(re.match(r"\d+(?:\.\d+)?", token).group(0))
    return val * 60 if re.search(r"m(in|ins|inutes?)?$", token) else val


def section(text: str, rc: RuleContext) -> dict[str, Any]:
    """retime_section target: {"range": [a, b]} or {"scene_id": id}; the whole video by default."""
    d = rc.duration
    unit = r"(seconds?|secs?|s|minutes?|mins?|m)\b"
    if m := re.search(rf"\b(first|opening|last|final|closing)\s+(?:{_NUM}\s*)?{unit}", text):
        n = _num(m.group(2)) if m.group(2) else 1.0
        span = n * 60 if m.group(3).startswith("m") else n
        return {"range": [0.0, round(min(span, d), 3)]} if m.group(1) in ("first", "opening") else \
            {"range": [round(max(0.0, d - span), 3), round(d, 3)]}
    if m := re.search(rf"\b(?:from|between)\s+{_TIME}\s+(?:to|and|until|-)\s+{_TIME}", text):
        a, b = sorted((_seconds(m.group(1)), _seconds(m.group(2))))
        return {"range": [a, min(b, d)]}
    scenes = rc.script.scenes if rc.script else []
    if m := re.search(r"\b(?:scene|chapter|section|act|part)\s+(\d+)\b", text):
        k = int(m.group(1))
        if 1 <= k <= len(scenes):
            return {"scene_id": scenes[k - 1].id}
    if re.search(r"\b(intro|introduction|opening|hook|beginning|start)\b", text):
        return {"scene_id": scenes[0].id} if scenes else {"range": [0.0, min(60.0, d)]}
    if re.search(r"\b(outro|ending|end|conclusion|finale)\b", text):
        return {"scene_id": scenes[-1].id} if scenes else {"range": [max(0.0, d - 60.0), d]}
    return {"range": [0.0, round(d, 3)]}


def _factor(text: str, faster: bool) -> float:
    if m := re.search(r"(\d+(?:\.\d+)?)\s*x\b", text):
        x = max(1.05, float(m.group(1)))
        return round(1 / x if faster else x, 2)
    if m := re.search(r"(\d+)\s*(?:%|percent)", text):
        p = 1 + int(m.group(1)) / 100
        return round(1 / p if faster else p, 2)
    if faster:
        return _scale_words(text, 0.55, 0.85) or 0.7
    return _scale_words(text, 1.8, 1.2) or 1.4


def rule_retime(text: str, clause: str, rc: RuleContext) -> list[EditOp]:
    fast, slow = _FASTER.search(text), _SLOWER.search(text)
    if not fast and not slow:
        return []
    return [_op(EditOpName.retime_section, clause, factor=_factor(text, bool(fast)), **section(text, rc))]


Rule = Callable[[str, str, RuleContext], list[EditOp]]
RULES: list[Rule] = [rule_captions, rule_music, rule_rewrite, rule_remove_overlay, rule_merge, rule_split,
                     rule_reorder, rule_motion, rule_add_text, rule_ai, rule_type, rule_duration, rule_swap,
                     rule_retime]
