"""Post-render QA: build_qa(shotlist, ctx) -> [qa_report.json, qa_report.md].

Samples one frame every settings.judge.post_render_every_s seconds of
final.mp4 (after any intro), maps frames to shots, rescores each shot against
its intent with the vision seat (task "qa"), checks captions against the
narration, and reports mix, pacing, judge pass rate, flags, attribution,
on-screen text timing and repeats of the narration,
NC/ND and Standard YouTube Licence assets, mock seats and the sourcing
search usage per run (work/source_notes.json, written by source/registry.py)
with a warning when a search budget ran out.
"""
from __future__ import annotations

import re
import shutil
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from ..models import (SHOT_TYPES, STATIC_CARD_KINDS, FallbackNote, MotionKind, Narration, QAReport, QAShot, Shot,
                      ShotList, ShotType, SourceKind, Word)
from ..shots import ideas as I
from ..util import ToolError, media_duration, read_json, run, write_json
from .judge import JUDGE_MOCK_FLAG, JUDGE_WORKERS, REJECT_RULES, narration_by_beat, parse_verdict

QA_FLAGS = ("mock", "nc_nd", "standard_youtube_licence", "anachronism", "low_res", "no_match_accepted",
            "held_frame", "missing_media", "segment_repeat", "needs_motion_graphic")
SOURCE_NOTES = "source_notes.json"
SOURCE_LABELS = {"youtube": "YouTube", "pexels": "Pexels", "pixabay": "Pixabay", "wikimedia": "Wikimedia Commons",
                 "archive": "Internet Archive", "loc": "Library of Congress", "nasa": "NASA"}
NO_MATCH_TARGET_PCT = 1.0
OVERLAY_TIMING_MAX_S = 0.5  # median gap between a text overlay and its anchor word above this reads as off
MIX_TOLERANCE_PTS = 5.0
OVERLAY_SIBLING = {"footage": "footage_overlay", "image": "image_overlay"}
FRAMES_PER_SHOT = 3
NO_ATTRIBUTION = {SourceKind.motion, SourceKind.ai_image, SourceKind.ai_video}

QA_REJECTS = ("burned_in_captions", "watermark", "channel_logo", "talking_head", "black_frame", "frozen_frame")
QA_SYSTEM = (
    "You check frames of a finished documentary video. Each request shows frames from one shot and the "
    "narration intent it should cover. Ignore the black letterbox bars, the single caption line in the "
    "bottom band and branded graphics such as maps, charts, number cards and lower thirds: they are part of "
    "the edit. The narration is what the viewer hears over the shot: a picture that contradicts it scores "
    "below 70 even when it fits the intent. Good enough is the bar: fail only what would break viewer "
    "immersion. Check that the place "
    "makes sense, then the subject, then the action. Hard problems: foreign burned in subtitles "
    "(burned_in_captions), a big watermark (watermark), a channel logo (channel_logo), a person speaking to "
    "camera (talking_head), a black frame (black_frame), a frozen frame (frozen_frame). " + REJECT_RULES
    + ", ".join(QA_REJECTS) + '. Be concise. Reply with strict JSON only: '
    '{"score": 0-100, "reason": "one line", "rejects": []}'
)


def build_qa(shotlist: ShotList, ctx: Any) -> list[Path]:
    """Write qa_report.json (models.QAReport) and qa_report.md in the project root."""
    threshold = float(ctx.settings.judge.threshold)
    offset = intro_offset(ctx)
    notes: list[str] = []
    final = ctx.root / "final.mp4"
    frames: dict[str, list[str]] = {}
    if final.exists():
        frames = sample_frames(final, shotlist, ctx.work / "qa_frames", float(ctx.settings.judge.post_render_every_s),
                               offset)
    else:
        notes.append("final.mp4 not found, post-render pass skipped")
    post = score_post_render(shotlist, frames, ctx)
    coverage, match = caption_check(ctx)
    nar = ctx.maybe_load("narration", Narration)
    report = make_report(shotlist, ctx, threshold, post, coverage, match, notes, nar.words if nar else [])
    json_path = write_json(ctx.root / "qa_report.json", report)
    md_path = ctx.root / "qa_report.md"
    md_path.write_text(render_markdown(report, offset), encoding="utf-8")
    ctx.log.info("qa report written", above=report.pct_above_threshold, no_match=report.pct_no_match_accepted,
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


def sample_frames(video: Path, shotlist: ShotList, out_dir: Path, every: float, offset: float) -> dict[str, list[str]]:
    """Frames at every/2 + k*every of shot time, mapped to shot ids; shots without one get a mid frame."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    every = max(every, 0.5)
    try:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{offset + every / 2:.3f}", "-i", str(video),
             "-t", f"{max(shotlist.duration - every / 2, 0.1):.3f}", "-vf", f"fps=1/{every},scale=480:-2",
             "-q:v", "6", str(out_dir / "f_%05d.jpg")], timeout=max(300.0, shotlist.duration * 2))
    except ToolError:
        pass
    by_shot: dict[str, list[str]] = {}
    for k, path in enumerate(sorted(out_dir.glob("f_*.jpg"))):
        shot = shot_at(shotlist, every / 2 + k * every)
        if shot is not None:
            by_shot.setdefault(shot.id, []).append(str(path))
    for shot in shotlist.shots:
        if shot.id not in by_shot:
            mid = _grab(video, offset + (shot.start + shot.end) / 2, out_dir / f"mid_{shot.id}.jpg")
            if mid:
                by_shot[shot.id] = [mid]
    return by_shot


def shot_at(shotlist: ShotList, t: float) -> Optional[Shot]:
    for s in shotlist.shots:
        if s.start <= t < s.end:
            return s
    return None


def _grab(video: Path, t: float, out: Path) -> Optional[str]:
    try:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
             "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "6", str(out)], timeout=60)
    except ToolError:
        return None
    return str(out) if out.exists() else None


def score_post_render(shotlist: ShotList, frames: dict[str, list[str]], ctx: Any) -> dict[str, tuple[float, str]]:
    """Vision seat score per shot from its rendered frames (task qa)."""
    todo = [s for s in shotlist.shots if frames.get(s.id)]
    beats = narration_by_beat(ctx)

    def one(shot: Shot) -> Optional[tuple[float, str]]:
        imgs = _spread(frames[shot.id], FRAMES_PER_SHOT)
        heard = beats.get(shot.beat_id, "")
        prompt = ((f"Narration: {heard}\n" if heard else "") + f"Narration intent: {shot.intent}\n"
                  f"Shot type: {shot.type.value}\n"
                  f"Must avoid: {', '.join(shot.must_avoid) or 'nothing specific'}\nReturn strict JSON.")
        try:
            raw = ctx.seats.vision_json("vision", "qa", QA_SYSTEM, prompt, imgs, ctx.stage or "packaging",
                                        intent=shot.intent, details=["low"] * len(imgs))  # QA stays low detail
        except Exception as e:  # noqa: BLE001  QA never fails the stage
            ctx.log.warn("qa scoring failed", shot=shot.id, error=str(e)[:300])
            return None
        verdict = parse_verdict(raw, QA_REJECTS)
        if verdict is None:
            return None
        score, reason, rejects = verdict
        return score, "; ".join([reason] + rejects) if rejects else reason

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        results = list(pool.map(one, todo))
    return {s.id: r for s, r in zip(todo, results) if r is not None}


def _spread(items: list[str], n: int) -> list[str]:
    if len(items) <= n:
        return items
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


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
    """Share of shot count per type (percent, the headline mix) and per type pacing stats.

    Pacing rows carry count_pct (same as the mix) and time_pct, the screen time share, as secondary info.
    """
    total_t = sum(s.duration for s in shotlist.shots) or 1.0
    total_n = len(shotlist.shots) or 1
    mix: dict[str, float] = {}
    pacing: dict[str, dict[str, float]] = {}
    for t in SHOT_TYPES:
        durs = [s.duration for s in shotlist.shots if s.type.value == t]
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


def overlay_text_stats(shotlist: ShotList, words: list[Word]) -> dict[str, float]:
    """On-screen text against the narration. text_items: text overlays on the picture plus full screen quote
    and callout cards. timing_error_s: median absolute gap between a text overlay's first frame and the
    narration word where its anchor (else its title) is spoken, in its shot before it shows (where the pipeline
    anchored it) else within REPEAT_WINDOW_S; unanchored: text
    overlays whose anchor is not spoken that close. verbatim_pct: share of text items that repeat the
    narration spoken around them."""
    gaps: list[float] = []
    items = repeats = unanchored = 0
    for s in shotlist.shots:
        card = s.type == ShotType.motion_graphic and s.motion is not None and s.motion.kind in (
            MotionKind.callout, MotionKind.text_overlay)
        spec = s.motion if card else s.overlay if s.type != ShotType.motion_graphic else None
        if spec is None or spec.kind not in I.TEXT_OVERLAY_KINDS:
            continue
        shown = s.start + spec.start_at
        items += 1
        repeats += I.spec_repeats_narration(spec, I.spoken_near(words, shown))
        if card:
            continue
        anchor = spec.anchor or I.overlay_title(spec)
        # the moment the pipeline anchored to (overlay_start's window) first, so a word repeated earlier is not it
        t = I.anchor_time(anchor, words, s.start - I.ANCHOR_LEAD_S, shown)
        if t is None:
            t = I.anchor_time(anchor, words, shown - I.REPEAT_WINDOW_S, shown + I.REPEAT_WINDOW_S)
        if t is None:
            unanchored += 1
        else:
            gaps.append(abs(shown - t))
    return {"text_items": float(items), "timing_error_s": round(statistics.median(gaps), 2) if gaps else 0.0,
            "unanchored": float(unanchored), "verbatim_pct": round(100.0 * repeats / items, 1) if items else 0.0}


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


def make_report(shotlist: ShotList, ctx: Any, threshold: float, post: dict[str, tuple[float, str]],
                coverage: float, match: float, notes: list[str], words: Optional[list[Word]] = None) -> QAReport:
    """Assemble the QAReport from the shot list, post-render scores and caption check."""
    shots = []
    for s in shotlist.shots:
        p = post.get(s.id)
        shots.append(QAShot(shot_id=s.id, index=s.index, start=s.start, end=s.end, type=s.type.value,
                            score=s.judge_score, reason=(p[1] if p else s.judge_notes)[:300], attempts=s.attempts,
                            flags=_shot_flags(s, p[0] if p else None, threshold),
                            post_render_score=round(p[0], 1) if p else None))
    judged = [s for s in shotlist.shots if s.type != ShotType.motion_graphic]
    above = sum(1 for s in judged if s.judge_score is not None and s.judge_score >= threshold)
    no_match = sum(1 for s in judged if "no_match_accepted" in s.flags)
    attributed = 0
    for s in shotlist.shots:
        c = s.selected_candidate()
        attributed += int(c is not None and c.source not in NO_ATTRIBUTION and bool(c.attribution.strip()))
    mix, pacing = mix_and_pacing(shotlist)
    mocks = _mock_roles(ctx)
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
                      overlay_text=overlay_text_stats(shotlist, words or []))
    report.notes += review_notes(report, ctx) + source_usage_notes(ctx)
    return report


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
    """Readable summary: headline numbers, mix, pacing, flagged shots with final.mp4 timestamps."""
    md = [f"# QA report: {r.project_id}", "",
          f"- Shots at or above threshold ({r.threshold:g}): **{r.pct_above_threshold}%**",
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
    for t in SHOT_TYPES:
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
    flagged = [s for s in r.shots if s.flags or (s.score is not None and s.score < r.threshold)]
    md += ["", f"## Flagged shots ({len(flagged)})", ""]
    if not flagged:
        return "\n".join(md + ["None.", ""])
    md += ["| Time | Shot | Type | Judge | Rendered | Attempts | Flags | Note |", "|---|---|---|---|---|---|---|---|"]
    for s in flagged:
        judge = f"{s.score:.0f}" if s.score is not None else "n/a"
        rendered = f"{s.post_render_score:.0f}" if s.post_render_score is not None else "n/a"
        md.append(f"| {mmss(s.start + offset)} | {s.shot_id} | {s.type} | {judge} | {rendered} | {s.attempts} | "
                  f"{', '.join(s.flags) or 'none'} | {_cell(s.reason)} |")
    return "\n".join(md + [""])
