"""Post-render QA: build_qa(shotlist, ctx) -> [qa_report.json, qa_report.md].

Samples one frame every settings.judge.post_render_every_s seconds of
final.mp4 (after any intro), maps frames to shots, rescores each shot against
its intent with the vision seat (task "qa"), checks captions against the
narration, and reports mix, pacing, judge pass rate, flags, attribution,
NC/ND assets and mock seats.
"""
from __future__ import annotations

import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from ..models import SHOT_TYPES, Narration, QAReport, QAShot, Shot, ShotList, ShotType, SourceKind
from ..util import ToolError, media_duration, read_json, run, write_json
from .judge import JUDGE_MOCK_FLAG, JUDGE_WORKERS, parse_verdict

QA_FLAGS = ("mock", "nc_nd", "anachronism", "low_res", "no_match_accepted")
NO_MATCH_TARGET_PCT = 1.0
MIX_TOLERANCE_PTS = 5.0
FRAMES_PER_SHOT = 3
NO_ATTRIBUTION = {SourceKind.motion, SourceKind.ai_image, SourceKind.ai_video}

QA_SYSTEM = (
    "You check frames of a finished documentary video. Each request shows frames from one shot and the "
    "narration intent it should cover. Ignore the black letterbox bars, the single caption line in the "
    "bottom band and branded graphics such as maps, charts, number cards and lower thirds: they are part of "
    "the edit. Good enough is the bar: fail only what would break viewer immersion. Check that the place "
    "makes sense, then the subject, then the action. List hard problems in rejects: foreign burned in "
    "subtitles, a big watermark, a channel logo, a person speaking to camera, a black or frozen frame. "
    'Be concise. Reply with strict JSON only: {"score": 0-100, "reason": "one line", "rejects": [...]}'
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
    report = make_report(shotlist, ctx, threshold, post, coverage, match, notes)
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

    def one(shot: Shot) -> Optional[tuple[float, str]]:
        imgs = _spread(frames[shot.id], FRAMES_PER_SHOT)
        prompt = (f"Narration intent: {shot.intent}\nShot type: {shot.type.value}\n"
                  f"Must avoid: {', '.join(shot.must_avoid) or 'nothing specific'}\nReturn strict JSON.")
        try:
            raw = ctx.seats.vision_json("vision", "qa", QA_SYSTEM, prompt, imgs, ctx.stage or "packaging",
                                        intent=shot.intent)
        except Exception as e:  # noqa: BLE001  QA never fails the stage
            ctx.log.warn("qa scoring failed", shot=shot.id, error=str(e)[:300])
            return None
        verdict = parse_verdict(raw)
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
    """Screen time share per type (percent) and per type pacing stats with count and time shares."""
    total_t = sum(s.duration for s in shotlist.shots) or 1.0
    total_n = len(shotlist.shots) or 1
    mix: dict[str, float] = {}
    pacing: dict[str, dict[str, float]] = {}
    for t in SHOT_TYPES:
        durs = [s.duration for s in shotlist.shots if s.type.value == t]
        mix[t] = round(100.0 * sum(durs) / total_t, 1)
        if durs:
            pacing[t] = {"count": float(len(durs)), "min": round(min(durs), 2), "max": round(max(durs), 2),
                         "mean": round(sum(durs) / len(durs), 2), "count_pct": round(100.0 * len(durs) / total_n, 1),
                         "time_pct": mix[t]}
    return mix, pacing


def _target(ctx: Any) -> dict[str, float]:
    raw = {t: float(ctx.profile.visual_mix.get(t, 0) or 0) for t in SHOT_TYPES}
    total = sum(raw.values()) or 1.0
    return {t: round(100.0 * v / total, 1) for t, v in raw.items()}


def _nc_nd(shotlist: ShotList) -> list[str]:
    seen: dict[str, list[str]] = {}
    label: dict[str, str] = {}
    for s in shotlist.shots:
        c = s.selected_candidate()
        if c is not None and "nc_nd" in c.flags:
            key = c.url or c.local_path or c.id
            seen.setdefault(key, []).append(s.id)
            label[key] = f"{c.title or key} [{c.licence}] {c.url}".strip()
    return [f"{label[k]} (shots {', '.join(v)})" for k, v in seen.items()]


def make_report(shotlist: ShotList, ctx: Any, threshold: float, post: dict[str, tuple[float, str]],
                coverage: float, match: float, notes: list[str]) -> QAReport:
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
    mocks = set(getattr(ctx.seats, "mock_roles", set())) | {e.role for e in ctx.tracker.report.entries if e.mock}
    report = QAReport(project_id=shotlist.project_id, threshold=threshold, shots=shots,
                      pct_above_threshold=round(100.0 * above / len(judged), 1) if judged else 100.0,
                      pct_no_match_accepted=round(100.0 * no_match / len(judged), 1) if judged else 0.0,
                      mix_target=_target(ctx), mix_achieved=mix, pacing=pacing, caption_coverage=coverage,
                      caption_word_match=match, attribution_count=attributed, mock_used=sorted(mocks),
                      nc_nd_assets=_nc_nd(shotlist), notes=list(notes))
    report.notes += review_notes(report, ctx)
    return report


def review_notes(r: QAReport, ctx: Any) -> list[str]:
    """Plain language findings for the operator."""
    out: list[str] = []
    if r.pct_no_match_accepted > NO_MATCH_TARGET_PCT:
        out.append(f"{r.pct_no_match_accepted}% of shots accepted without a match, above the {NO_MATCH_TARGET_PCT}% "
                   "target: review the flagged shots in the editor")
    for t, target in r.mix_target.items():
        got = r.mix_achieved.get(t, 0.0)
        if abs(got - target) > MIX_TOLERANCE_PTS:
            out.append(f"{t} screen time {got}% versus target {target}%")
    if ctx.profile.captions.enabled and (r.caption_coverage < 0.95 or r.caption_word_match < 0.9):
        out.append(f"captions cover {r.caption_coverage:.0%} of narration time and match {r.caption_word_match:.0%} "
                   "of narration words")
    low = sum(1 for s in r.shots if "post_render_low" in s.flags)
    if low:
        out.append(f"{low} shots scored below threshold on the rendered video")
    if r.nc_nd_assets:
        out.append(f"{len(r.nc_nd_assets)} NC or ND licensed assets: confirm the licence before a monetised upload")
    if r.mock_used:
        out.append(f"mock seats used ({', '.join(r.mock_used)}): this render is a placeholder preview")
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
          f"- Attributed assets: {r.attribution_count}; NC or ND assets: {len(r.nc_nd_assets)}",
          f"- Mock seats: {', '.join(r.mock_used) or 'none'}", ""]
    if r.notes:
        md += ["## Notes", ""] + [f"- {n}" for n in r.notes] + [""]
    md += ["## Visual mix", "", "| Type | Target % | Screen time % | Shots % | Count | Min s | Max s | Mean s |",
           "|---|---|---|---|---|---|---|---|"]
    for t in SHOT_TYPES:
        p = r.pacing.get(t, {})
        md.append(f"| {t} | {r.mix_target.get(t, 0):g} | {r.mix_achieved.get(t, 0):g} | {p.get('count_pct', 0):g} | "
                  f"{int(p.get('count', 0))} | {p.get('min', 0):g} | {p.get('max', 0):g} | {p.get('mean', 0):g} |")
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
