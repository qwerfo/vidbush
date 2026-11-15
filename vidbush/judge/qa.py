"""Post-render QA: build_qa(shotlist, ctx) -> [qa_report.json, qa_report.md].

Every shot is judged on frames from its own middle in final.mp4 (after any intro), never on a fixed
sampling grid, against the narration spoken while it is on screen, what kind of picture it really is
and every piece of text the edit draws over it. The vision seat answers fixed checks (picture matches the
words, text is true, clean, one image, no fake face of a named person, quality) plus a score; any failed
check or a score below threshold fails the shot. The report also states screen time by what is really on
screen (real footage, real still, generated still, generated video, graphic), checks captions against the
narration, every on-screen text item (overlays, cards, lower thirds, map and chart titles and labels)
against the narration around it, and reports mix, pacing, judge pass rate, flags, attribution, NC/ND and
Standard YouTube Licence assets, mock seats and the sourcing search usage per run (work/source_notes.json,
written by source/registry.py). qa_report.md opens with a five-line summary.
"""
from __future__ import annotations

import re
import shutil
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from ..models import (DESIGNED_SOURCES, GENERATED_SOURCES, SHOT_TYPES, STATIC_CARD_KINDS, FallbackNote, MotionKind, MotionSpec, Narration, QAReport, QAShot,
                      Script, Shot, ShotList, ShotType, SourceKind, Word)
from ..shots import ideas as I
from ..util import ToolError, media_duration, read_json, run, write_json
from .judge import JUDGE_MOCK_FLAG, JUDGE_WORKERS, REJECT_RULES, narration_by_beat, parse_verdict

QA_FLAGS = ("mock", "nc_nd", "standard_youtube_licence", "anachronism", "low_res", "no_match_accepted",
            "held_frame", "missing_media", "segment_repeat", "needs_motion_graphic")
SOURCE_NOTES = "source_notes.json"
SOURCE_LABELS = {"youtube": "YouTube", "pexels": "Pexels", "pixabay": "Pixabay", "wikimedia": "Wikimedia Commons",
                 "archive": "Internet Archive", "loc": "Library of Congress", "nasa": "NASA",
                 "nara": "US National Archives"}
NO_MATCH_TARGET_PCT = 1.0
OVERLAY_TIMING_MAX_S = 0.5  # median gap between a text overlay and its anchor word above this reads as off
MIX_TOLERANCE_PTS = 5.0
OVERLAY_SIBLING = {"footage": "footage_overlay", "image": "image_overlay"}
NO_ATTRIBUTION = GENERATED_SOURCES


def build_qa(shotlist: ShotList, ctx: Any) -> list[Path]:
    """Write qa_report.json (models.QAReport) and qa_report.md in the project root."""
    threshold = float(ctx.settings.judge.threshold)
    offset = intro_offset(ctx)
    notes: list[str] = []
    final = ctx.root / "final.mp4"
    frames: dict[str, list[str]] = {}
    if final.exists():
        frames = sample_frames(final, shotlist, ctx.work / "qa_frames", offset=offset)
    else:
        notes.append("final.mp4 not found, post-render pass skipped")
    nar = ctx.maybe_load("narration", Narration)
    words = nar.words if nar else []
    post = judge_rendered(shotlist, frames, ctx, words)
    coverage, match = caption_check(ctx)
    report = make_report(shotlist, ctx, threshold, post, coverage, match, notes, words)
    report.summary = summary_lines(report, offset)  # final.mp4 time, as in the markdown
    json_path = write_json(ctx.root / "qa_report.json", report)
    md_path = ctx.root / "qa_report.md"
    md_path.write_text(render_markdown(report, offset), encoding="utf-8")
    ctx.log.info("qa report written", above=report.pct_above_threshold, no_match=report.pct_no_match_accepted,
                 fail_rendered=report.pct_fail_rendered, screen_time=report.screen_time,
                 caption_coverage=report.caption_coverage)
    return [json_path, md_path]


# ---------------------------------------------------------------- frames

def intro_offset(ctx: Any) -> float:
    """Seconds of intro before shot time 0 in final.mp4 (0 without an intro)."""
    if not ctx.brand.intro_path:
        return 0.0
    asm = ctx.work / "assembly.json"
    if asm.exists():
        try:
            return float(read_json(asm).get("intro_duration", 0.0) or 0.0)
        except (ValueError, TypeError, AttributeError):
            pass
    intro = Path(ctx.brand.intro_path)
    intro = intro if intro.is_absolute() else ctx.settings.path(str(intro))
    try:
        return media_duration(intro) if intro.exists() else 0.0
    except ToolError:
        return 0.0


LONG_SHOT_S = 4.0  # shots this long get frames at a quarter and three quarters too
EDGE_S = 0.3  # never sample this close to a cut, where a fade or the neighbour can show
FRAME_WIDTH = 768  # wide enough to read lower thirds, map titles and faces
GRAB_WORKERS = 4
FRAMES_PER_SHOT = 3


def shot_times(shot: Shot) -> list[float]:
    """Shot-time seconds to judge the shot at: its middle, plus its quarters on long shots and the middle
    of the part after its overlay text appears, each kept EDGE_S inside the shot. A frame from a fixed grid
    lands late or in the next shot on short shots, so every shot is sampled inside its own window."""
    d = max(shot.end - shot.start, 0.0)
    fracs = [0.5] + ([0.25, 0.75] if d >= LONG_SHOT_S else [])
    times = [shot.start + f * d for f in fracs]
    spec = shot.overlay if shot.type != ShotType.motion_graphic else None
    if spec is not None and spec.start_at > 0 and shot.start + spec.start_at > times[0] - EDGE_S:
        times.append((shot.start + spec.start_at + shot.end) / 2)  # a frame where the text is on screen
    margin = min(EDGE_S, d / 4)
    out: list[float] = []
    for t in sorted(times):
        t = round(min(max(t, shot.start + margin), shot.end - margin), 3)
        if not out or t - out[-1] > 0.2:
            out.append(t)
    return out


def sample_frames(video: Path, shotlist: ShotList, out_dir: Path, every: float = 0.0,
                  offset: float = 0.0) -> dict[str, list[str]]:
    """Frames of each shot from its own window (shot_times), shifted by the intro offset: shot id -> paths,
    the middle frame first, then one where the overlay text shows. every is kept for callers and ignored: a
    fixed grid sampled short shots late or in their neighbour."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    jobs = []
    for s in shotlist.shots:
        times = shot_times(s)
        mid = min(times, key=lambda t: abs(t - (s.start + s.end) / 2))
        spec = s.overlay if s.type != ShotType.motion_graphic else None
        shown = s.start + spec.start_at if spec is not None and spec.start_at > 0 else None
        text = next((t for t in reversed(times) if shown is not None and t >= shown and t != mid), None)
        # the middle first, then a frame with the overlay text on screen, so both survive FRAMES_PER_SHOT
        order = [mid] + ([text] if text is not None else []) + [t for t in times if t not in (mid, text)]
        jobs += [(s.id, k, t) for k, t in enumerate(order)]

    def one(job: tuple[str, int, float]) -> tuple[str, Optional[str]]:
        sid, k, t = job
        return sid, _grab(video, offset + t, out_dir / f"f_{sid}_{k}.jpg")

    by_shot: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=GRAB_WORKERS) as pool:
        for sid, path in pool.map(one, jobs):
            if path:
                by_shot.setdefault(sid, []).append(path)
    return by_shot


def shot_at(shotlist: ShotList, t: float) -> Optional[Shot]:
    for s in shotlist.shots:
        if s.start <= t < s.end:
            return s
    return None


def _grab(video: Path, t: float, out: Path, width: int = FRAME_WIDTH) -> Optional[str]:
    try:  # -ss before -i still lands on the frame shown at t: ffmpeg decodes from the keyframe before it
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(t, 0.0):.3f}", "-i", str(video),
             "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(out)], timeout=60)
    except ToolError:
        return None
    return str(out) if out.exists() else None


# ---------------------------------------------------------------- what is on screen

# Five kinds, counted apart everywhere: real footage, real stills, photo-style AI (stills and clips that stand
# in for real footage, kept rare), designed animation (illustrated maps, declared-style illustrations and clips
# made from them, openly non-photographic) and graphics (cards, charts, drawn maps).
ON_SCREEN = ("real_footage", "real_still", "generated_still", "generated_video", "designed", "graphic")
ON_SCREEN_LABELS = {"real_footage": "real footage", "real_still": "real stills", "generated_still": "AI stills",
                    "generated_video": "AI video", "designed": "designed animation", "graphic": "graphics"}
PHOTO_AI_SCREEN = ("generated_still", "generated_video")
DESIGNED_TYPE = "designed_animation"  # mix table row for shots whose pick is designed animation
ON_SCREEN_PROMPT = {"real_footage": "real footage (a sourced video clip)",
                    "real_still": "a real photograph or scan (a sourced still)",
                    "generated_still": "an AI generated still", "generated_video": "an AI generated video clip",
                    "designed": "designed animation (an illustrated map, an illustration in a declared style, or a "
                                "clip made from one), openly not a photograph",
                    "graphic": "a motion graphic made by the edit (map, chart or card)"}


def is_designed(shot: Shot) -> bool:
    """The pick is designed animation (models.DESIGNED_SOURCES), or a drawn map painted by the art layer."""
    c = shot.selected_candidate()
    if c is not None and c.source in DESIGNED_SOURCES:
        return True
    p = (shot.motion.payload if shot.motion is not None else None) or {}
    return bool(isinstance(p, dict) and (p.get("art_base") or p.get("art_clip")))


def on_screen(shot: Shot) -> str:
    """What the viewer really sees, decided by the selected candidate and never by the planned shot type,
    because a footage or image shot converted to an AI still keeps its old type."""
    c = shot.selected_candidate()
    if is_designed(shot):
        return "designed"
    if shot.type == ShotType.motion_graphic or (c is not None and c.source == SourceKind.motion):
        return "graphic"
    if c is not None:
        if c.source == SourceKind.ai_image:
            return "generated_still"
        if c.source == SourceKind.ai_video:
            return "generated_video"
        if "depthflow" in c.flags:  # a still turned into a parallax clip is still a still
            return "real_still"
        return "real_footage" if c.media == "video" else "real_still"
    return {ShotType.ai_image: "generated_still", ShotType.ai_video: "generated_video"}.get(
        shot.type, "real_footage" if shot.type.value.startswith("footage") else "real_still")


def effective_type(shot: Shot) -> str:
    """Shot type as rendered: a shot whose pick is an AI still or AI clip counts as ai_image or ai_video."""
    seen = on_screen(shot)
    if seen == "designed" and shot.type != ShotType.motion_graphic:
        return DESIGNED_TYPE
    if seen == "generated_still" and shot.type != ShotType.motion_graphic:
        return ShotType.ai_image.value
    if seen == "generated_video" and shot.type != ShotType.motion_graphic:
        return ShotType.ai_video.value
    return shot.type.value


def screen_time(shotlist: ShotList) -> tuple[dict[str, float], dict[str, float]]:
    """(percent of the runtime, seconds) per ON_SCREEN class."""
    secs = {k: 0.0 for k in ON_SCREEN}
    for s in shotlist.shots:
        secs[on_screen(s)] += max(s.end - s.start, 0.0)
    total = sum(secs.values()) or 1.0
    return ({k: round(100.0 * v / total, 1) for k, v in secs.items()}, {k: round(v, 1) for k, v in secs.items()})


# ---------------------------------------------------------------- on-screen text

def spec_texts(spec: Optional[MotionSpec]) -> list[tuple[str, str]]:
    """(role, text) of every piece of text a motion spec draws: titles, subtitles, box and arrow labels,
    number cards, map titles, marker, label and route names, chart titles and axis labels, recap items."""
    if spec is None:
        return []
    p = spec.payload or {}
    k = spec.kind
    out: list[tuple[str, str]] = []

    def add(role: str, text: Any) -> None:
        t = " ".join(str(text or "").split())
        if t:
            out.append((role, t))

    if k in (MotionKind.title_card, MotionKind.lower_third, MotionKind.callout, MotionKind.text_overlay):
        add("title", p.get("title"))
        add("subtitle", p.get("subtitle"))
    elif k in (MotionKind.highlight_box, MotionKind.arrow):
        add("label", p.get("label"))
    elif k == MotionKind.number_card:
        add("number", " ".join(str(p.get(x) or "") for x in ("number", "label", "sublabel")))
    elif k == MotionKind.map:
        add("title", p.get("title"))
        for group in ("markers", "labels", "routes", "arrows"):
            for item in p.get(group) or []:
                if isinstance(item, dict):
                    add("place", item.get("label") or item.get("text"))
    elif k in (MotionKind.bar_chart, MotionKind.line_chart):
        add("title", p.get("title"))
        for lab in p.get("labels") or []:
            add("label", lab)
    elif k == MotionKind.recap_grid:
        add("title", p.get("title"))
        for item in p.get("items") or []:
            add("label", item)
    return out


def screen_texts(shot: Shot) -> list[dict[str, Any]]:
    """Every text item the edit puts on screen in this shot: {kind, role, text, at (shot timeline time it
    appears), until (time it leaves), anchor, card}. A full screen card's text shows from the shot start."""
    items: list[dict[str, Any]] = []
    card = shot.type == ShotType.motion_graphic
    spec = shot.motion if card else shot.overlay
    if spec is None:
        return items
    at = shot.start + (0.0 if card else max(spec.start_at, 0.0))
    for role, text in spec_texts(spec):
        items.append({"kind": spec.kind.value, "role": role, "text": text, "at": round(at, 2),
                      "until": round(shot.end, 2), "anchor": spec.anchor, "card": card})
    return items


def window_words(words: list[Word], start: float, end: float) -> str:
    """Narration spoken while [start, end] is on screen: every word that overlaps the window."""
    return " ".join(w.text for w in words if w.end > start and w.start < end)


# ---------------------------------------------------------------- frame checks

QA_REJECTS = ("burned_in_captions", "watermark", "channel_logo", "talking_head", "black_frame", "frozen_frame")
QA_CHECKS = ("matches_words", "text_true", "clean", "one_image", "no_fake_face", "quality")
LOCAL_CHECKS = ("text_timing", "text_repeat", "text_filler", "picture_repeat")
SCORE_FLOOR = 40.0  # a score this low fails a shot on its own; above it only a failed check does (calibration)
QA_SYSTEM = (
    "You check one shot of a finished documentary video from frames taken inside the shot. You get the narration "
    "spoken during this shot (what the viewer hears while these frames are on screen), the lines just before and "
    "after it, the video topic, what kind of picture it is, the director's intent and every text the edit draws "
    "over it. Ignore the black letterbox bars and the single caption line in the bottom band: they are always "
    "there. First describe literally what the frames show (who or what, how many, doing what, where, the light "
    "and period, any text). Then answer each check true when the shot passes it, false when it fails. Be "
    "strict: a careful viewer who knows the subject notices these.\n"
    "matches_words: the picture shows what the narration during this shot says: its place, subject or action, or "
    "a fair illustration of it. Fails: a picture that illustrates the line before or after instead, a wrong "
    "place or a setting that contradicts the named place (mountains, forest or a European village for a flat "
    "treeless Arctic town), a wrong period, daylight when the words say dark, a wrong agent (who does the "
    "action: men carrying what the words say a dog team carried), a wrong count, or a generic picture that "
    "ignores a specific person, object or event the words name.\n"
    "text_true: every text the edit adds (lower third, label, box, arrow, map title or place, number, card) is "
    "true of the picture and of what is said now. Fails: a label naming something the picture does not show (a "
    "town name over empty wilderness, a ship name over another ship), a box or arrow around nothing it names, "
    "a map of a place not spoken during the shot, a title repeating the narration word for word, a subtitle "
    "that only adds a country or a decade, or unreadable overlapping labels. True when the edit adds no text.\n"
    "clean: no burned in subtitles from the source, no watermark, channel logo, subscribe button, browser or "
    "app screen, no person speaking to camera, no black or frozen frame.\n"
    "one_image: one continuous picture, not a split panel, collage or grid, unless the words compare several "
    "places or things side by side.\n"
    "no_fake_face: no AI generated face presented as a named real person.\n"
    "quality: heads and the main subject are not cut off by the frame or the bars, the picture is not soft, "
    "blurred or low resolution, and an AI still has no obvious artifact (floating objects, extra limbs, garbled "
    "shapes).\n"
    "Then score 0 to 100 how well the shot serves the viewer at this moment (70 is usable). " + REJECT_RULES
    + ", ".join(QA_REJECTS) + ". Reply with strict JSON only: "
    '{"seen": "what the frames show", "checks": {"matches_words": true, "text_true": true, "clean": true, '
    '"one_image": true, "no_fake_face": true, "quality": true}, "fails": {"<check that is false>": "short '
    'reason"}, "score": 0-100, "reason": "one line", "rejects": []}'
)


NEIGHBOUR_S = 6.0  # narration this long before and after the shot, so a picture of the next line is recognised


def qa_prompt(shot: Shot, heard: str, beat: str, words: list[Word], topic: str = "") -> str:
    """User prompt for one rendered shot: the words spoken during it first, then context."""
    lines = [f"Narration during this shot ({shot.end - shot.start:.1f} s): {heard or '(no narration)'}"]
    before = window_words(words, shot.start - NEIGHBOUR_S, shot.start) if words else ""
    after = window_words(words, shot.end, shot.end + NEIGHBOUR_S) if words else ""
    lines += ([f"Just before this shot: {before}"] if before else []) + ([f"Just after: {after}"] if after else [])
    if beat and " ".join(beat.split()) != " ".join(heard.split()) and not (before or after):
        lines.append(f"Narration of the whole beat, for context only: {beat}")
    lines += ([f"Video topic: {topic}"] if topic else []) + [
        f"Picture: {ON_SCREEN_PROMPT[on_screen(shot)]}", f"Director intent (secondary): {shot.intent}"]
    texts = screen_texts(shot)
    if texts:
        parts = []
        for t in texts[:12]:
            part = f"{t['kind'].replace('_', ' ')} {t['role']} \"{t['text']}\""
            if t["at"] > shot.start + 0.05:
                said = window_words(words, t["at"], t["until"])
                part += f" from {t['at'] - shot.start:.1f} s" + (f" (spoken while it shows: \"{said[:160]}\")"
                                                                  if said else "")
            parts.append(part)
        lines.append("On-screen text the edit adds: " + "; ".join(parts))
    else:
        lines.append("On-screen text the edit adds: none")
    lines += [f"Must avoid: {', '.join(shot.must_avoid) or 'nothing specific'}", "Return strict JSON."]
    return "\n".join(lines)


_FALSE = ("false", "no", "fail", "failed")


def parse_checks(raw: Any) -> tuple[list[str], dict[str, str]]:
    """(failed check names in QA_CHECKS order, reason per failed check) from a QA reply. A check the model
    did not answer is not failed; only an explicit false (or "no", "fail") is."""
    checks = raw.get("checks") if isinstance(raw, dict) else None
    fails = raw.get("fails") if isinstance(raw, dict) else None
    reasons = dict(fails) if isinstance(fails, dict) else {}
    failed: list[str] = []
    if isinstance(checks, dict):
        for name in QA_CHECKS:
            v = checks.get(name)
            if isinstance(v, dict):
                if v.get("ok") is False or v.get("pass") is False:
                    failed.append(name)
                    reasons.setdefault(name, v.get("reason") or "")
            elif v is False or (isinstance(v, str) and v.strip().lower() in _FALSE):
                failed.append(name)
    return failed, {n: " ".join(str(reasons.get(n) or "").split())[:200] for n in failed}


LABEL_REPEAT_S = 60.0  # the same on-screen label again within this long tells the viewer nothing new
PICTURE_REPEAT_S = 60.0  # the same picture again within this long reads as a mistake
PICTURE_HASH_MAX = 5  # bits of 64 two middle frames may differ by and still be the same picture
DECADE_RE = re.compile(r"^(?:the\s+)?\d{4}s$", re.I)


def _frame_hash(path: str) -> Optional[int]:
    """64-bit difference hash of the picture area of a rendered frame: the letterbox bars and the caption band
    are cut away first, so the same picture under other captions or overlays hashes alike."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            g = im.convert("L").crop((int(w * 0.1), int(h * 0.2), int(w * 0.9), int(h * 0.72))).resize((9, 8))
            px = list(g.tobytes())  # mode L: one byte per pixel
    except (OSError, ValueError):
        return None
    if max(px) - min(px) < 12:  # a flat frame (black, fade) says nothing about the picture
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def _picture_key(shot: Shot) -> str:
    """What makes two shots the same picture: one source asset (the same clip may play twice only as two
    different segments), or AI stills drawn from the same words."""
    c = shot.selected_candidate()
    if c is None:
        return ""
    if c.source == SourceKind.ai_image:
        return "ai:" + " ".join((c.prompt or shot.ai_prompt or shot.intent).lower().split())
    if c.media == "video":
        return f"{c.url or c.local_path}@{c.in_point:.0f}" if (c.url or c.local_path) else ""
    return c.url or c.local_path


def _filler(text: str) -> bool:
    """A subtitle that adds nothing: a bare decade or the name of a country."""
    from ..mg import geo
    t = " ".join(text.split()).strip(" .")
    return bool(DECADE_RE.match(t)) or geo.is_country(t)


def local_checks(shotlist: ShotList, words: list[Word], frames: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """Checks decided without the model, shot id -> {check: reason}. text_timing: an on-screen text item shows
    more than EARLY_S before or LATE_S after its words, far from any of them, repeats the narration, or leaves
    before it can be read. text_repeat: the same label was on screen within LABEL_REPEAT_S. text_filler: a
    lower third subtitle that is a bare decade or a country name. picture_repeat: the middle frame shows the
    same picture as a shot within PICTURE_REPEAT_S. The model sees one shot at a time and cannot know these."""
    out: dict[str, dict[str, str]] = {}
    seen_labels: list[tuple[float, str, str]] = []
    seen_pics: list[tuple[float, Optional[int], str, str]] = []
    for s in shotlist.shots:
        found: dict[str, str] = {}
        r = text_item_report(s, words)
        if r is not None:
            what = f"{r['kind'].replace('_', ' ')} \"{r['texts'][0][:40]}\""
            if r["early"]:
                found["text_timing"] = (f"{what} shows {-r['gap']:.1f} s before its words" if r["gap"] is not None
                                        else f"{what} shows long before its words are spoken")
            elif r["late"]:
                found["text_timing"] = f"{what} shows after its words were spoken"
            elif r["unanchored"] and not r["card"]:
                found["text_timing"] = f"{what} is not spoken near the shot"
            elif r["verbatim"]:
                found["text_timing"] = f"{what} repeats the narration"
            elif r["short"]:
                found["text_timing"] = f"{what} leaves before it can be read"
        for t in screen_texts(s):
            if t["role"] not in ("title", "label", "number"):
                continue
            key = " ".join(t["text"].lower().split())
            prior = next((sid for at, k, sid in seen_labels if k == key and t["at"] - at < LABEL_REPEAT_S), None)
            if prior and "text_repeat" not in found:
                found["text_repeat"] = f"\"{t['text'][:40]}\" was on screen again in {prior}"
            seen_labels.append((t["at"], key, s.id))
        for t in screen_texts(s):
            if t["kind"] == MotionKind.lower_third.value and t["role"] == "subtitle" and _filler(t["text"]):
                found["text_filler"] = f"subtitle \"{t['text']}\" adds nothing"
        if on_screen(s) != "graphic":
            key = _picture_key(s)
            h = _frame_hash(frames[s.id][0]) if frames.get(s.id) else None
            twin = next((sid for at, hh, kk, sid in seen_pics if s.start - at < PICTURE_REPEAT_S and (
                (kk and kk == key) or (h is not None and hh is not None
                                       and bin(h ^ hh).count("1") <= PICTURE_HASH_MAX))), None)
            if twin:
                found["picture_repeat"] = f"same picture as {twin}"
            seen_pics.append((s.start, h, key, s.id))
        if found:
            out[s.id] = found
    return out


def judge_rendered(shotlist: ShotList, frames: dict[str, list[str]], ctx: Any,
                   words: Optional[list[Word]] = None) -> dict[str, dict[str, Any]]:
    """Frame check per shot: {score, reason, rejects, failed (check names), window, seen, fail}. A shot fails
    on any failed check (the model's QA_CHECKS and the LOCAL_CHECKS), a hard reject, or a score below
    SCORE_FLOOR: calibration on run 2 showed the score alone sits near 70 for good and bad shots alike, while
    the named checks separate them."""
    threshold = float(ctx.settings.judge.threshold)
    if words is None:
        load = getattr(ctx, "maybe_load", None)
        nar = load("narration", Narration) if load else None
        words = nar.words if nar else []
    beats = narration_by_beat(ctx)
    topic = _topic(ctx)
    local = local_checks(shotlist, words, frames)
    todo = [s for s in shotlist.shots if frames.get(s.id)]

    def one(shot: Shot) -> Optional[dict[str, Any]]:
        imgs = frames[shot.id][:FRAMES_PER_SHOT]
        heard = window_words(words, shot.start, shot.end)
        prompt = qa_prompt(shot, heard, beats.get(shot.beat_id, ""), words, topic)
        try:  # the middle frame is looked at closely, the others confirm
            raw = ctx.seats.vision_json("vision", "qa", QA_SYSTEM, prompt, imgs, ctx.stage or "packaging",
                                        intent=shot.intent, details=["high"] + ["low"] * (len(imgs) - 1))
        except Exception as e:  # noqa: BLE001  QA never fails the stage
            ctx.log.warn("qa scoring failed", shot=shot.id, error=str(e)[:300])
            return None
        verdict = parse_verdict(raw, QA_REJECTS)
        if verdict is None:
            return None
        score, reason, rejects = verdict
        failed, why = ([], {}) if raw.get("_mock") else parse_checks(raw)
        for name, note in local.get(shot.id, {}).items():
            failed.append(name)
            why[name] = note
        notes = [f"{n}: {why[n]}" if why.get(n) else n for n in failed]
        text = "; ".join([reason] + notes + rejects) if (notes or rejects) else reason
        return {"score": score, "reason": text, "rejects": rejects, "failed": failed, "window": heard,
                "seen": " ".join(str(raw.get("seen") or "").split())[:300],
                "fail": bool(failed or rejects or score < min(threshold, SCORE_FLOOR))}

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        results = list(pool.map(one, todo))
    return {s.id: r for s, r in zip(todo, results) if r is not None}


def _topic(ctx: Any) -> str:
    """The video's title, so the checker knows which place and period the words are about."""
    load = getattr(ctx, "maybe_load", None)
    try:
        script = load("script", Script) if load else None
    except Exception:  # noqa: BLE001  no script: judge without the topic
        script = None
    return str(getattr(script, "title", "") or "")


def score_post_render(shotlist: ShotList, frames: dict[str, list[str]], ctx: Any) -> dict[str, tuple[float, str]]:
    """(score, reason) per shot from its rendered frames; judge_rendered returns the full verdicts."""
    return {sid: (v["score"], v["reason"]) for sid, v in judge_rendered(shotlist, frames, ctx).items()}


# ---------------------------------------------------------------- captions

def _tokens(text: str) -> list[str]:
    return [t for t in (re.sub(r"[^\w']", "", w.lower()).strip("'") for w in text.split()) if t]


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def caption_check(ctx: Any) -> tuple[float, float]:
    """(coverage, word match): share of narration word time inside caption lines, multiset word overlap."""
    nar = ctx.maybe_load("narration", Narration)
    path = ctx.work / "captions.json"
    if nar is None or not nar.words or not path.exists():
        return 0.0, 0.0
    data = read_json(path)
    lines = data.get("lines", []) if isinstance(data, dict) else data
    spans = _merge(sorted((float(ln.get("start", 0)), float(ln.get("end", 0))) for ln in lines))
    nar_tokens = Counter(t for w in nar.words for t in _tokens(w.text))
    cap_tokens = Counter(t for ln in lines for t in _tokens(str(ln.get("text", ""))))
    total = sum(nar_tokens.values())
    match = sum(min(n, cap_tokens[t]) for t, n in nar_tokens.items()) / total if total else 0.0
    covered = weight = 0.0
    for w in nar.words:
        dur = max(w.end - w.start, 0.0)
        if dur == 0:
            hit = any(a <= w.start <= b for a, b in spans)
            covered, weight = covered + (1.0 if hit else 0.0) * 0.05, weight + 0.05
            continue
        weight += dur
        covered += sum(max(0.0, min(b, w.end) - max(a, w.start)) for a, b in spans if b > w.start and a < w.end)
    coverage = min(1.0, covered / weight) if weight else 0.0
    return round(coverage, 4), round(match, 4)


# ---------------------------------------------------------------- report

def _shot_flags(shot: Shot, post: Optional[float], threshold: float) -> list[str]:
    cand = shot.selected_candidate()
    pool = set(shot.flags) | set(cand.flags if cand else [])
    if JUDGE_MOCK_FLAG in pool:
        pool.add("mock")
    flags = [f for f in QA_FLAGS if f in pool]
    if post is not None and post < threshold:
        flags.append("post_render_low")
    return flags


def mix_and_pacing(shotlist: ShotList) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Share of shot count per type as rendered (percent, the headline mix) and per type pacing stats. A shot
    whose pick is an AI still counts as ai_image whatever type it was planned as.

    Pacing rows carry count_pct (same as the mix) and time_pct, the screen time share, as secondary info.
    """
    total_t = sum(s.duration for s in shotlist.shots) or 1.0
    total_n = len(shotlist.shots) or 1
    mix: dict[str, float] = {}
    pacing: dict[str, dict[str, float]] = {}
    for t in SHOT_TYPES + [DESIGNED_TYPE]:
        durs = [s.duration for s in shotlist.shots if effective_type(s) == t]
        mix[t] = round(100.0 * len(durs) / total_n, 1)
        if durs:
            pacing[t] = {"count": float(len(durs)), "min": round(min(durs), 2), "max": round(max(durs), 2),
                         "mean": round(sum(durs) / len(durs), 2), "count_pct": mix[t],
                         "time_pct": round(100.0 * sum(durs) / total_t, 1)}
    return mix, pacing


def _target(ctx: Any) -> dict[str, float]:
    """Target share of shot count per type: full screen cards as motion_graphic, animated overlays added to
    the overlay types they decorate (StyleProfile.shot_mix)."""
    mix = ctx.profile.shot_mix()
    return {t: round(float(mix.get(t, 0.0)), 1) for t in SHOT_TYPES}


INSET_KINDS = ("map", "bar_chart", "line_chart")


def animation_share(shotlist: ShotList) -> dict[str, float]:
    """Achieved animation in percent of shots: full screen cards, shots carrying an overlay, and the
    subset of overlays that are map or chart insets. Also what the viewer feels: full screen card time in
    percent of the runtime, and the longest hold of a static text or number card in seconds."""
    n = len(shotlist.shots) or 1
    cards = [s for s in shotlist.shots if s.type == ShotType.motion_graphic]
    over = [s for s in shotlist.shots if s.overlay is not None and s.type != ShotType.motion_graphic]
    insets = sum(1 for s in over if s.overlay.kind.value in INSET_KINDS)
    holds = [s.duration for s in cards if s.motion is not None and s.motion.kind in STATIC_CARD_KINDS]
    return {"full_screen_cards": round(100.0 * len(cards) / n, 1), "overlay_shots": round(100.0 * len(over) / n, 1),
            "inset_overlays": round(100.0 * insets / n, 1),
            "full_screen_card_time": round(100.0 * sum(s.duration for s in cards) / (shotlist.duration or 1.0), 1),
            "max_card_hold_s": round(max(holds, default=0.0), 2)}


EARLY_S = 1.0  # a text overlay shown more than this before its words are spoken reads as early
LATE_S = 1.5  # text shown more than this after its words reads as late (a card's cue window)
GRAPHIC_LEAD_S = 1.5  # a card, map, chart or number may lead its word by the card cue window
SHORT_S = 2.5  # text on screen for less than this, or less than its reading time, cannot be found and read
GRAPHIC_KINDS = {"map", "bar_chart", "line_chart", "number_card", "title_card", "recap_grid"}


def text_item_report(shot: Shot, words: list[Word]) -> Optional[dict[str, Any]]:
    """One on-screen text item (the overlay or full screen card of a shot, with all its texts) against the
    narration: {kind, texts, at, gap (shown minus spoken, None when its words are not spoken), verbatim,
    early, late, short, unanchored, card}. Its words are the anchor, else the title, else the first place,
    label or number it shows; they are looked up first where the pipeline anchored them (from the shot
    start to the moment the text shows), then within REPEAT_WINDOW_S, then anywhere in the narration."""
    texts = screen_texts(shot)
    if not texts:
        return None
    first = texts[0]
    card, at, until = first["card"], first["at"], first["until"]
    anchor = first["anchor"] or next((t["text"] for t in texts if t["role"] == "title"), first["text"])
    near = I.spoken_near(words, at)
    name_kind = first["kind"] == "lower_third"
    # a map title or place list made of names ('Atlantic Ocean and Pacific Ocean') names places, it does not
    # repeat the narration
    verbatim = any(I.repeats_narration(t["text"], near, name_kind or t["role"] in ("place", "label", "number"))
                   for t in texts if not (first["kind"] == "map" and I.reads_as_name(t["text"])))
    spoken = I.anchor_time(anchor, words, shot.start - I.ANCHOR_LEAD_S, at)
    if spoken is None:
        spoken = I.anchor_time(anchor, words, at - I.REPEAT_WINDOW_S, at + I.REPEAT_WINDOW_S)
    far = None
    if spoken is None and words:
        far = I.anchor_time(anchor, words, words[0].start, words[-1].end + 1.0)
        verbatim = verbatim or (card and far is not None and I.repeats_narration(
            anchor, I.spoken_near(words, far), False))
    gap = None if spoken is None else round(at - spoken, 2)
    read = max(I.READ_MIN_S, I.READ_PER_WORD_S * len(" ".join(t["text"] for t in texts[:2]).split()))
    lead = GRAPHIC_LEAD_S if card or first["kind"] in GRAPHIC_KINDS else EARLY_S
    return {"kind": first["kind"], "texts": [t["text"] for t in texts], "at": at, "gap": gap, "card": card,
            "verbatim": bool(verbatim),
            "early": (gap is not None and gap < -lead) or (far is not None and far > at + I.REPEAT_WINDOW_S),
            "late": (gap is not None and gap > LATE_S) or (far is not None and far < at - I.REPEAT_WINDOW_S),
            "short": until - at < max(SHORT_S, read) - 0.05,
            # its words are not spoken near it (none at all, or only long after): the legacy meaning
            "unanchored": bool(words) and spoken is None and (far is None or far > at + I.REPEAT_WINDOW_S)}


LEGACY_TEXT_KINDS = {k.value for k in I.TEXT_OVERLAY_KINDS}
LEGACY_CARD_KINDS = {MotionKind.callout.value, MotionKind.text_overlay.value}


def _text_stats(reports: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """(items, verbatim percent, median timing gap of overlays, overlays far from their words)."""
    gaps = [abs(r["gap"]) for r in reports if not r["card"] and r["gap"] is not None]
    n = len(reports)
    return (float(n), round(100.0 * sum(r["verbatim"] for r in reports) / n, 1) if n else 0.0,
            round(statistics.median(gaps), 2) if gaps else 0.0,
            float(sum(1 for r in reports if r["unanchored"] and not r["card"])))


def overlay_text_stats(shotlist: ShotList, words: list[Word]) -> dict[str, float]:
    """On-screen text against the narration, one item per overlay or full screen card.

    text_items, verbatim_pct, timing_error_s and unanchored cover the text overlays (callouts, text overlays,
    lower thirds, boxes, arrows) and full screen quote and callout cards, the set the overlay rules place.
    The all_ keys cover every piece of on-screen text, including map and chart titles and labels, number
    overlays and every full screen card, because the viewer reads all of it. verbatim: the text repeats the
    narration spoken around it (a card also when it repeats a line spoken long before); timing: median absolute
    gap between an overlay's first frame and the narration word it belongs to (full screen cards follow their
    own cue rule and count in early and late only); unanchored: overlays whose words are not spoken near them.
    early, late, short count items shown more than EARLY_S before or LATE_S after their words, or on screen
    too briefly to read; problem_items holds any of these problems."""
    reports = [r for r in (text_item_report(s, words) for s in shotlist.shots) if r is not None]
    legacy = [r for r in reports if r["kind"] in (LEGACY_CARD_KINDS if r["card"] else LEGACY_TEXT_KINDS)]
    items, verbatim, timing, unanchored = _text_stats(legacy)
    all_items, all_verbatim, all_timing, all_unanchored = _text_stats(reports)
    problems = sum(1 for r in reports if r["verbatim"] or r["early"] or r["late"] or r["short"]
                   or (r["unanchored"] and not r["card"]))
    return {"text_items": items, "timing_error_s": timing, "unanchored": unanchored, "verbatim_pct": verbatim,
            "all_items": all_items, "all_verbatim_pct": all_verbatim, "all_timing_error_s": all_timing,
            "all_unanchored": all_unanchored, "early": float(sum(r["early"] for r in reports)),
            "late": float(sum(r["late"] for r in reports)), "short": float(sum(r["short"] for r in reports)),
            "problem_items": float(problems)}


def heuristic_fallbacks(ctx: Any) -> list[FallbackNote]:
    """Writer sections and director chunks that fell back to heuristics with a real provider configured
    (work/fallbacks.json, written by the script and shotlist stages)."""
    work = getattr(ctx, "work", None)
    path = Path(work) / "fallbacks.json" if work else None
    if path is None or not path.exists():
        return []
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return []
    out: list[FallbackNote] = []
    for stage in ("script", "shotlist"):
        for item in data.get(stage) or []:
            try:
                out.append(FallbackNote(**item))
            except (TypeError, ValueError):
                continue
    return out


def _nc_nd(shotlist: ShotList, flag: str = "nc_nd") -> list[str]:
    """One line per selected asset carrying flag (title, licence, url, shots using it)."""
    seen: dict[str, list[str]] = {}
    label: dict[str, str] = {}
    for s in shotlist.shots:
        c = s.selected_candidate()
        if c is not None and flag in c.flags:
            key = c.url or c.local_path or c.id
            seen.setdefault(key, []).append(s.id)
            label[key] = f"{c.title or key} [{c.licence}] {c.url}".strip()
    return [f"{label[k]} (shots {', '.join(v)})" for k, v in seen.items()]


def _mock_roles(ctx: Any) -> set[str]:
    return set(getattr(ctx.seats, "mock_roles", set())) | {e.role for e in ctx.tracker.report.entries if e.mock}


def refresh_mock_used(ctx: Any) -> None:
    """Add the roles that fell back to mock after build_qa (packaging: titles, thumbnails) to qa_report, so
    final_status still sees them. Only mock_used and its markdown line change."""
    path = ctx.root / "qa_report.json"
    if not path.exists():
        return
    report = QAReport.model_validate(read_json(path))
    mocks = sorted(set(report.mock_used) | _mock_roles(ctx))
    if mocks == report.mock_used:
        return
    report.mock_used = mocks
    write_json(path, report)
    (ctx.root / "qa_report.md").write_text(render_markdown(report, intro_offset(ctx)), encoding="utf-8")


def _verdict(p: Any) -> Optional[dict[str, Any]]:
    """A post-render entry as a verdict dict; older callers pass (score, reason)."""
    if p is None:
        return None
    if isinstance(p, dict):
        return p
    score, reason = p[0], p[1]
    return {"score": score, "reason": reason, "rejects": [], "failed": [], "window": "", "fail": None}


CHECK_LABELS = {"matches_words": "picture does not match the words", "text_true": "on-screen text not true",
                "text_timing": "text off its words", "text_repeat": "label repeated within a minute",
                "text_filler": "filler subtitle", "picture_repeat": "picture repeated within a minute",
                "clean": "source graphics, logo or talking head", "one_image": "split panel or collage",
                "no_fake_face": "fake face of a named person", "quality": "framing, blur or artifact",
                "score": f"scored below {SCORE_FLOOR:g}", "reject": "hard reject"}


def make_report(shotlist: ShotList, ctx: Any, threshold: float, post: dict[str, Any],
                coverage: float, match: float, notes: list[str], words: Optional[list[Word]] = None) -> QAReport:
    """Assemble the QAReport from the shot list, the rendered frame verdicts (judge_rendered, or older
    (score, reason) pairs) and the caption check."""
    words = words or []
    shots = []
    for s in shotlist.shots:
        v = _verdict(post.get(s.id))
        fail = None if v is None else (v["fail"] if v["fail"] is not None else v["score"] < threshold)
        shots.append(QAShot(shot_id=s.id, index=s.index, start=s.start, end=s.end, type=effective_type(s),
                            score=s.judge_score, reason=(v["reason"] if v else s.judge_notes)[:300],
                            attempts=s.attempts, flags=_shot_flags(s, v["score"] if v else None, threshold),
                            post_render_score=round(v["score"], 1) if v else None, on_screen=on_screen(s),
                            window_text=(v["window"] if v and v.get("window") else
                                         window_words(words, s.start, s.end))[:400],
                            checks_failed=list(v["failed"]) if v else [], failed=bool(fail)))
        if fail and "rendered_fail" not in shots[-1].flags:
            shots[-1].flags.append("rendered_fail")
    judged = [s for s in shotlist.shots if s.type != ShotType.motion_graphic]
    above = sum(1 for s in judged if s.judge_score is not None and s.judge_score >= threshold)
    no_match = sum(1 for s in judged if "no_match_accepted" in s.flags)
    attributed = 0
    for s in shotlist.shots:
        c = s.selected_candidate()
        attributed += int(c is not None and c.source not in NO_ATTRIBUTION and bool(c.attribution.strip()))
    mix, pacing = mix_and_pacing(shotlist)
    mocks = _mock_roles(ctx)
    pct, secs = screen_time(shotlist)
    rendered = [q for q in shots if q.post_render_score is not None]
    report = QAReport(project_id=shotlist.project_id, threshold=threshold, shots=shots,
                      pct_above_threshold=round(100.0 * above / len(judged), 1) if judged else 100.0,
                      pct_no_match_accepted=round(100.0 * no_match / len(judged), 1) if judged else 0.0,
                      mix_target=_target(ctx), mix_achieved=mix, pacing=pacing, caption_coverage=coverage,
                      animation_target={k: round(v, 1) for k, v in ctx.profile.animation_target().items()},
                      animation_achieved=animation_share(shotlist),
                      caption_word_match=match, attribution_count=attributed, mock_used=sorted(mocks),
                      heuristic_fallbacks=heuristic_fallbacks(ctx),
                      nc_nd_assets=_nc_nd(shotlist), notes=list(notes),
                      standard_youtube_assets=_nc_nd(shotlist, "standard_youtube_licence"),
                      overlay_text=overlay_text_stats(shotlist, words), screen_time=pct, screen_time_s=secs,
                      pct_fail_rendered=round(100.0 * sum(q.failed for q in rendered) / len(rendered), 1)
                      if rendered else 0.0)
    report.worst = worst_moments(report)
    report.summary = summary_lines(report)
    report.notes += review_notes(report, ctx) + source_usage_notes(ctx)
    return report


WORST_N = 5


def _fail_reasons(q: QAShot, threshold: float) -> list[str]:
    out = list(q.checks_failed)
    if q.post_render_score is not None and q.post_render_score < min(threshold, SCORE_FLOOR):
        out.append("score")
    return out or (["reject"] if q.failed else [])


def worst_moments(r: QAReport, n: int = WORST_N) -> list[str]:
    """The n rendered shots that fail worst: most failed checks first, then the lowest score, then the
    longest on screen. One line each: time, shot, what is on screen, why."""
    bad = [q for q in r.shots if q.failed]
    bad.sort(key=lambda q: (-len(q.checks_failed), q.post_render_score if q.post_render_score is not None
                            else 100.0, -(q.end - q.start)))
    out = []
    for q in bad[:n]:
        what = ON_SCREEN_LABELS.get(q.on_screen, q.on_screen).rstrip("s") if q.on_screen != "graphic" else "graphic"
        out.append(f"{mmss(q.start)} {q.shot_id} ({what}): {_cell(q.reason)[:110]}")
    return out


def summary_lines(r: QAReport, offset: float = 0.0) -> list[str]:
    """Five lines an operator can trust: screen time by what is really on screen, rendered shots that fail
    and why, the worst five moments, on-screen text problems, and the pipeline judge against the frames."""
    st, secs = r.screen_time, r.screen_time_s
    real = round(st.get("real_footage", 0.0) + st.get("real_still", 0.0), 1)
    gen = round(sum(st.get(k, 0.0) for k in PHOTO_AI_SCREEN), 1)
    parts = ", ".join(f"{ON_SCREEN_LABELS[k]} {st.get(k, 0):g}% ({secs.get(k, 0):.0f} s)" for k in ON_SCREEN
                      if st.get(k, 0) or k in ("real_footage", "real_still", "generated_still"))
    lines = [f"Screen time: real {real:g}%, photo-style AI {gen:g}%, designed animation {st.get('designed', 0):g}%, "
             f"graphics {st.get('graphic', 0):g}% ({parts})"]
    rendered = [q for q in r.shots if q.post_render_score is not None]
    if rendered:
        fails = [q for q in rendered if q.failed]
        why = Counter(x for q in fails for x in _fail_reasons(q, r.threshold))
        top = ", ".join(f"{n} {CHECK_LABELS.get(k, k)}" for k, n in why.most_common(4))
        lines.append(f"Rendered shots that fail their frame check: {len(fails)} of {len(rendered)} "
                     f"({r.pct_fail_rendered:g}%)" + (f": {top}" if top else ""))
    else:
        lines.append("Rendered shots that fail their frame check: not measured (no final.mp4 frames)")
    worst = [w if not offset else f"{mmss(_secs(w) + offset)}{w[5:]}" for w in r.worst]
    lines.append("Worst five: " + ("; ".join(worst) if worst else "none"))
    ot = r.overlay_text
    lines.append(f"On-screen text (every overlay, card, map and chart): {ot.get('all_items', 0):g} items, "
                 f"{ot.get('problem_items', 0):g} with a timing or repeat problem ({ot.get('all_verbatim_pct', 0):g}% "
                 f"repeat the narration, {ot.get('early', 0):g} early, {ot.get('late', 0):g} late, "
                 f"{ot.get('short', 0):g} too short to read, {ot.get('all_unanchored', 0):g} far from their words)")
    pass_rendered = round(100.0 - r.pct_fail_rendered, 1)
    lines.append(f"Pipeline judge: {r.pct_above_threshold:g}% of shots at or above {r.threshold:g}; "
                 + (f"the rendered frames pass {pass_rendered:g}%" if rendered else "rendered frames not checked"))
    return lines


def _secs(line: str) -> float:
    m, s = line[:5].split(":")
    return int(m) * 60 + int(s)


def source_usage_notes(ctx: Any) -> list[str]:
    """Searches used out of the run budget per source and sourcing run, with the shots that shared it.

    Every source with a run budget gets a line, like YouTube always had; a spent budget is a warning.
    Searches a shot skipped because it had used its fair share are counted as deferred."""
    path = ctx.work / SOURCE_NOTES
    try:
        runs = (read_json(path).get("runs") or {}) if path.exists() else {}
    except (OSError, ValueError, AttributeError):
        return []
    out: list[str] = []
    ordered = sorted(runs.values(), key=lambda r: float(r.get("ts", 0)) if isinstance(r, dict) else 0.0)
    for k, run_ in enumerate(ordered):
        tag = f" (run {k + 1} of {len(ordered)})" if len(ordered) > 1 else ""
        for name, u in sorted((run_.get("usage") or {}).items()):
            used, budget, refused = int(u.get("used", 0)), int(u.get("budget", 0)), bool(u.get("refused"))
            label = SOURCE_LABELS.get(name, name)
            shots, deferred = int(u.get("shots", 0) or 0), int(u.get("deferred", 0) or 0)
            spread = f" across {shots} shots" if shots else ""
            fair = f", {deferred} deferred by fair shares" if deferred else ""
            if refused or (budget and used >= budget):
                out.append(f"warning: {label} search budget spent ({used} of {budget} searches{spread}){tag}: "
                           "later queries were skipped, raise settings.sources.query_budget or re-run to use "
                           "the cache")
            else:
                out.append(f"{label} searches used: {used} of {budget}{spread}{fair}{tag}")
    return out


def review_notes(r: QAReport, ctx: Any) -> list[str]:
    """Plain language findings for the operator."""
    out: list[str] = []
    if r.pct_no_match_accepted > NO_MATCH_TARGET_PCT:
        out.append(f"{r.pct_no_match_accepted}% of shots accepted without a match, above the {NO_MATCH_TARGET_PCT}% "
                   "target: review the flagged shots in the editor")
    for t, target in r.mix_target.items():
        got = r.mix_achieved.get(t, 0.0)
        if t in OVERLAY_SIBLING.values():  # a ceiling: a shot carries text only when it adds to the narration
            if got - target > MIX_TOLERANCE_PTS:
                out.append(f"{t} share of shots {got}% above its ceiling {target}%")
            continue
        name = t
        if t in OVERLAY_SIBLING:  # plain shots take the overlay slots that had nothing honest to show
            sib = OVERLAY_SIBLING[t]
            got, target = round(got + r.mix_achieved.get(sib, 0.0), 1), round(target + r.mix_target.get(sib, 0.0), 1)
            name = f"{t} (with or without overlay)"
        if abs(got - target) > MIX_TOLERANCE_PTS:
            out.append(f"{name} share of shots {got}% versus target {target}%")
    if ctx.profile.captions.enabled and (r.caption_coverage < 0.95 or r.caption_word_match < 0.9):
        out.append(f"captions cover {r.caption_coverage:.0%} of narration time and match {r.caption_word_match:.0%} "
                   "of narration words")
    ot = r.overlay_text
    if ot.get("verbatim_pct", 0.0) > 0:
        out.append(f"{ot['verbatim_pct']:g}% of on-screen text repeats the narration the captions already show")
    if ot.get("timing_error_s", 0.0) > OVERLAY_TIMING_MAX_S or ot.get("unanchored", 0.0) > 0:
        out.append(f"text overlays sit {ot['timing_error_s']:g} s (median) from the words they belong to, "
                   f"{ot['unanchored']:g} far from any of their words")
    low = sum(1 for s in r.shots if "post_render_low" in s.flags)
    if low:
        out.append(f"{low} shots scored below threshold on the rendered video")
    if r.nc_nd_assets:
        out.append(f"{len(r.nc_nd_assets)} NC or ND licensed assets: confirm the licence before a monetised upload")
    if r.standard_youtube_assets:
        out.append(f"{len(r.standard_youtube_assets)} Standard YouTube Licence assets: confirm reuse rights "
                   "(fair use or permission) before upload")
    if r.mock_used:
        out.append(f"mock seats used ({', '.join(r.mock_used)}): this render is a placeholder preview")
    if r.heuristic_fallbacks:
        where = ", ".join(f"{n.role} {n.scene_id}" for n in r.heuristic_fallbacks)
        out.append(f"{len(r.heuristic_fallbacks)} parts fell back to heuristics while a real provider is configured "
                   f"({where}): review those scenes, then rerun the stage with resume --force")
    return out


# ---------------------------------------------------------------- markdown

def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60:02d}:{t % 60:02d}"


def _cell(text: str) -> str:
    return " ".join(str(text).replace("|", "/").split())[:120]


def render_markdown(r: QAReport, offset: float = 0.0) -> str:
    """The five-line summary first, then headline numbers, screen time, mix, pacing and flagged shots with
    final.mp4 timestamps."""
    md = [f"# QA report: {r.project_id}", ""] + summary_lines(r, offset) + ["", "## Details", "",
          f"- Shots at or above threshold ({r.threshold:g}): **{r.pct_above_threshold}%** (pipeline judge); "
          f"rendered shots failing their frame check: **{r.pct_fail_rendered}%**",
          f"- Accepted without a match: **{r.pct_no_match_accepted}%** (target at most {NO_MATCH_TARGET_PCT:g}%)",
          f"- Caption coverage {r.caption_coverage:.1%}, caption word match {r.caption_word_match:.1%}",
          f"- Attributed assets: {r.attribution_count}; NC or ND assets: {len(r.nc_nd_assets)}; "
          f"Standard YouTube Licence assets: {len(r.standard_youtube_assets)}",
          f"- Mock seats: {', '.join(r.mock_used) or 'none'}",
          f"- Heuristic fallbacks with a real provider: {len(r.heuristic_fallbacks)}",
          f"- On-screen text: {r.overlay_text.get('text_items', 0):g} items, "
          f"{r.overlay_text.get('verbatim_pct', 0):g}% repeat the narration, text overlays "
          f"{r.overlay_text.get('timing_error_s', 0):g} s (median) from their words, "
          f"{r.overlay_text.get('unanchored', 0):g} unanchored", ""]
    if r.screen_time:
        md += ["## Screen time by what is on screen", "", "| On screen | Percent of runtime | Seconds |", "|---|---|---|"]
        md += [f"| {ON_SCREEN_LABELS[k]} | {r.screen_time.get(k, 0):g} | {r.screen_time_s.get(k, 0):g} |"
               for k in ON_SCREEN] + [""]
    if r.notes:
        md += ["## Notes", ""] + [f"- {n}" for n in r.notes] + [""]
    if r.heuristic_fallbacks:
        md += ["## Heuristic fallbacks (needs review)", "", "| Scene | Role | Stage | Reason |", "|---|---|---|---|"]
        md += [f"| {n.scene_id} | {n.role} | {n.stage} | {_cell(n.reason)} |" for n in r.heuristic_fallbacks] + [""]
    for title, items in (("NC or ND assets", r.nc_nd_assets), ("Standard YouTube Licence assets",
                                                                r.standard_youtube_assets)):
        if items:
            md += [f"## {title} ({len(items)})", ""] + [f"- {' '.join(a.split())}" for a in items] + [""]
    md += ["## Visual mix (share of shot count)", "",
           "| Type | Target % | Shots % | Screen time % | Count | Min s | Max s | Mean s |",
           "|---|---|---|---|---|---|---|---|"]
    for t in SHOT_TYPES + ([DESIGNED_TYPE] if DESIGNED_TYPE in r.pacing else []):
        p = r.pacing.get(t, {})
        md.append(f"| {t} | {r.mix_target.get(t, 0):g} | {r.mix_achieved.get(t, 0):g} | {p.get('time_pct', 0):g} | "
                  f"{int(p.get('count', 0))} | {p.get('min', 0):g} | {p.get('max', 0):g} | {p.get('mean', 0):g} |")
    if r.animation_achieved:
        md += ["", "## Animation (share of shots)", "",
               "Full screen cards are motion_graphic shots. Animated overlays ride on footage and image shots, so",
               "they are counted on top of the shot they decorate.", "",
               "| Measure | Target % | Achieved % |", "|---|---|---|"]
        for key, label in (("full_screen_cards", "Full screen cards"), ("overlay_shots", "Shots with an overlay"),
                           ("inset_overlays", "Map and chart insets over the picture"),
                           ("full_screen_card_time", "Full screen card time (percent of runtime)"),
                           ("max_card_hold_s", "Longest text or number card hold (seconds, not percent)")):
            target = r.animation_target.get(key)
            md.append(f"| {label} | {'n/a' if target is None else f'{target:g}'} | "
                      f"{r.animation_achieved.get(key, 0):g} |")
    flagged = [s for s in r.shots if s.flags or s.failed or (s.score is not None and s.score < r.threshold)]
    md += ["", f"## Flagged shots ({len(flagged)})", ""]
    if not flagged:
        return "\n".join(md + ["None.", ""])
    md += ["| Time | Shot | Type | On screen | Judge | Rendered | Failed checks | Attempts | Flags | Note |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for s in flagged:
        judge = f"{s.score:.0f}" if s.score is not None else "n/a"
        rendered = f"{s.post_render_score:.0f}" if s.post_render_score is not None else "n/a"
        md.append(f"| {mmss(s.start + offset)} | {s.shot_id} | {s.type} | {ON_SCREEN_LABELS.get(s.on_screen, '')} | "
                  f"{judge} | {rendered} | {', '.join(s.checks_failed) or 'none'} | {s.attempts} | "
                  f"{', '.join(s.flags) or 'none'} | {_cell(s.reason)} |")
    return "\n".join(md + [""])
