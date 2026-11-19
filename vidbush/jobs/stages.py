"""Stage functions: thin glue from disk inputs to module calls to disk outputs.

Each returns the list of output paths it wrote. Module functions called here
are the integration contracts (see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..models import (DirectorPlan, InputMode, Narration, Research, Script, ShotList, VideoFormat)
from ..util import read_json, write_json
from .context import StageContext

FALLBACKS_FILE = "fallbacks.json"


def record_fallbacks(ctx: StageContext, stage: str) -> Path:
    """Write the stage's heuristic fallback notes to work/fallbacks.json, replacing that stage's old ones,
    so a re-run that succeeds clears them and QA always reads the current state."""
    path = ctx.work / FALLBACKS_FILE
    try:
        data = read_json(path) if path.exists() else {}
    except (OSError, ValueError):
        data = {}
    take = getattr(ctx.seats, "take_fallbacks", None)
    data[stage] = take(stage) if callable(take) else []
    write_json(path, data)
    return path


def stage_intake(ctx: StageContext) -> list[str]:
    p = ctx.project
    if p.input_mode == InputMode.brief and p.brief is None:
        raise ValueError("brief mode needs a brief")
    if p.input_mode == InputMode.script and not p.script_path:
        raise ValueError("script mode needs a script file")
    if p.input_mode == InputMode.voiceover and not p.voiceover_path:
        raise ValueError("voiceover mode needs an audio file")
    if ctx.profile.top10 is False and p.format == VideoFormat.top10:
        use_top10_profile(ctx)
    snap = _write_snapshot(ctx)
    ctx.save_project()
    return [str(ctx.root / "project.json"), str(snap)]


def _write_snapshot(ctx: StageContext) -> Path:
    return write_json(ctx.work_path("profile_snapshot.json"), {"brand": ctx.brand, "profile": ctx.profile,
                                                                "seats": ctx.seats.status()})


def use_top10_profile(ctx: StageContext) -> None:
    """Switch the context and project to top10_default (with any per-project override applied)."""
    from ..config import load_style_profile
    from .context import apply_profile_override
    profile = load_style_profile(ctx.settings, "top10_default")
    ctx.profile = apply_profile_override(profile, ctx.work / "profile_override.json")
    ctx.project.style_profile_id = profile.id
    ctx.log.info("style profile switched for a top10 script", profile=profile.id)


def stage_research(ctx: StageContext) -> list[str]:
    from ..research.researcher import run_research
    research = run_research(ctx.project.brief, ctx.brand, ctx.seats, ctx.settings, ctx.log)
    return [str(ctx.save("research", research))]


def stage_outline(ctx: StageContext) -> list[str]:
    from ..script.outline import make_outline
    outline = make_outline(ctx.project.brief, ctx.load("research", Research), ctx.brand, ctx.seats)
    return [str(ctx.save("outline", outline))]


def stage_script(ctx: StageContext) -> list[str]:
    from ..models import Outline
    from ..script.parser import parse_script
    from ..script.writer import write_script
    p = ctx.project
    if p.input_mode == InputMode.brief:
        script = write_script(ctx.load("outline", Outline), ctx.load("research", Research), p.brief,
                              ctx.brand, ctx.seats)
    elif p.script_path:
        script = parse_script(Path(p.script_path).read_text(encoding="utf-8"), p.format)
    else:
        from ..audio.transcribe import transcribe_to_text
        text = transcribe_to_text(Path(p.voiceover_path), ctx.settings, ctx.log, ctx.seats)
        script = parse_script(text, p.format)
    if script.format != p.format and p.input_mode != InputMode.brief:
        p.format = script.format
    if script.format == VideoFormat.top10 and not ctx.profile.top10 and not p.style_profile_id:
        use_top10_profile(ctx)  # detected from the script; an operator chosen profile is kept
        _write_snapshot(ctx)
    if not p.title or p.title == Path(p.script_path or p.voiceover_path or "").stem:
        p.title = script.title or p.title
    ctx.save_project()
    record_fallbacks(ctx, "script")
    return [str(ctx.save("script", script))]


def stage_narration(ctx: StageContext) -> list[str]:
    from ..audio.narration import narrate, narration_from_voiceover
    script = ctx.load("script", Script)
    if ctx.project.input_mode == InputMode.voiceover:
        nar = narration_from_voiceover(Path(ctx.project.voiceover_path), script, ctx.settings, ctx.work, ctx.log)
    else:
        nar = narrate(script, ctx.brand, ctx.seats, ctx.settings, ctx.work, ctx.log)
    return [str(ctx.save("narration", nar)), nar.audio_path]


def stage_shotlist(ctx: StageContext) -> list[str]:
    from ..shots.director import build_shotlist
    research = ctx.maybe_load("research", Research)
    shotlist, plan = build_shotlist(ctx.project, ctx.load("script", Script), ctx.load("narration", Narration),
                                    research, ctx.profile, ctx.brand, ctx.seats, ctx.rng("shotlist"))
    ctx.save("plan", plan)
    record_fallbacks(ctx, "shotlist")
    return [str(ctx.save("shotlist", shotlist))]


def stage_sourcing(ctx: StageContext) -> list[str]:
    from ..source.registry import source_all
    from ..plan.live import is_clause, source_clause
    shotlist = ctx.load("shotlist", ShotList)
    if is_clause(ctx.profile):
        shotlist = source_clause(shotlist, ctx)
    else:
        shotlist = source_all(shotlist, ctx, force="sourcing" in ctx.extra.get("forced", ()))
    return [str(ctx.save("shotlist", shotlist))]


def stage_judge(ctx: StageContext) -> list[str]:
    from ..judge.director_loop import judge_and_select
    from ..plan.live import is_clause, pick_clause
    shotlist = ctx.load("shotlist", ShotList)
    if is_clause(ctx.profile):  # the picker: describe once, pick by comparison, the ladder (judge/pick_stage)
        return [str(ctx.save("shotlist", apply_trim(pick_clause(shotlist, ctx), ctx)))]
    if "judge" in ctx.extra.get("forced", ()) and not ctx.snapshot_used():
        clear_verdicts(shotlist)  # a project from before snapshots: its working copy holds old verdicts
    shotlist = apply_trim(judge_and_select(shotlist, ctx), ctx)
    return [str(ctx.save("shotlist", shotlist))]


def apply_trim(shotlist: ShotList, ctx: StageContext) -> ShotList:
    """Excerpt render for review: work/trim.json {"until": seconds} keeps the shots that start before until, so
    the video ends at the first shot end at or after it (a shot is never split), and shotlist.duration follows;
    the mix pads or cuts the narration to it and the sound gate checks speech up to it. {"from": seconds} must
    be 0: a later start would shift the shots to 0 and need the narration offset to follow, which the mixer and
    captions do not support. Read when judge runs, so a new trim.json needs judge rerun (resume --force judge)."""
    from ..render.assemble import TRIM_FILE
    path = ctx.work / TRIM_FILE
    if not path.exists():
        return shotlist
    trim = read_json(path)
    if float(trim.get("from") or 0) != 0:
        raise ValueError(f"{path}: only an excerpt from 0 is supported (from must be 0 or absent)")
    until = trim.get("until")
    if isinstance(until, bool) or not isinstance(until, (int, float)):
        raise ValueError(f"{path}: until must be a number of seconds")
    keep = [s for s in shotlist.shots if s.start < until]
    if not keep:
        raise ValueError(f"{path}: until {until} s is before the first shot")
    if len(keep) == len(shotlist.shots):
        return shotlist
    ctx.log.info("excerpt render: shot list trimmed", until=until, end=keep[-1].end, shots=len(keep),
                 dropped=len(shotlist.shots) - len(keep))
    return shotlist.model_copy(update={"shots": keep, "duration": keep[-1].end})


def stage_visuals(ctx: StageContext) -> list[str]:
    from ..visuals.prepare import prepare_visuals
    shotlist = prepare_visuals(ctx.load("shotlist", ShotList), ctx)
    return [str(ctx.save("shotlist", shotlist))]


def stage_motion(ctx: StageContext) -> list[str]:
    from ..mg.renderer import render_all_motion
    shotlist, outs = render_all_motion(ctx.load("shotlist", ShotList), ctx)
    return [str(ctx.save("shotlist", shotlist))] + [str(p) for p in outs]


def stage_mix(ctx: StageContext) -> list[str]:
    from ..audio.music import build_mix
    from ..render.captions import build_captions
    shotlist = ctx.load("shotlist", ShotList)
    nar = ctx.load("narration", Narration)
    script = ctx.load("script", Script)
    cap_paths = build_captions(nar, ctx.profile, ctx.root, ctx.settings)
    mix_path = build_mix(nar, script, shotlist, ctx.brand, ctx.profile, ctx.settings, ctx.work, ctx.log,
                         ctx.rng("music"))
    return [str(p) for p in cap_paths] + [str(mix_path)]


def stage_assembly(ctx: StageContext) -> list[str]:
    from ..render.assemble import assemble
    shotlist = ctx.load("shotlist", ShotList)
    final, shotlist = assemble(shotlist, ctx)
    ctx.save("shotlist", shotlist)
    shutil.copy2(ctx.work_path("shotlist"), ctx.out_path("shotlist.json"))
    return [str(final), str(ctx.out_path("shotlist.json"))]


def stage_packaging(ctx: StageContext) -> list[str]:
    from ..judge.qa import build_qa, refresh_mock_used
    from ..render.package import package
    shotlist = ctx.load("shotlist", ShotList)
    outs = build_qa(shotlist, ctx)  # before the slow thumbnails, so qa_report always describes this final.mp4
    outs += package(shotlist, ctx)
    refresh_mock_used(ctx)  # titles and thumbnails ran after build_qa
    from ..plan.live import is_clause
    if is_clause(ctx.profile):  # the tile audit of this final.mp4, so its six numbers open qa_report.md
        outs += audit_at_end(shotlist, ctx)
    minutes = ctx.load("narration", Narration).duration / 60.0
    ctx.tracker.finalize(minutes)
    return [str(p) for p in outs] + [str(ctx.out_path("cost_report.json"))]


def audit_at_end(shotlist: ShotList, ctx: StageContext) -> list[str]:
    """Run the tile audit on the finished video (its cost goes to audit/audit_cost_report.json, as the audit
    command's does) and rebuild the QA report so it opens with the six numbers. A failed audit is logged and
    the QA report keeps its line saying how to run it."""
    import argparse
    from ..audit.cli import cmd_audit
    from ..judge.qa import build_qa
    try:
        code = cmd_audit(ctx.settings, argparse.Namespace(target=str(ctx.root), script=None, out=None, every=2.0,
                                                          limit=0, workers=8))
    except Exception as e:  # noqa: BLE001 - the video is finished; an audit failure must not fail packaging
        ctx.log.warn("tile audit failed", error=str(e)[:300])
        return []
    if code != 0:
        ctx.log.warn("tile audit wrote no summary", code=code)
        return []
    return [str(p) for p in build_qa(shotlist, ctx)] + [str(ctx.root / "audit" / "audit.md")]


def clear_verdicts(shotlist: ShotList) -> None:
    """Forget judge verdicts so a forced judge stage scores every real candidate again (unchanged prompts
    come back from the reply cache at no cost). Generated candidates are dropped: judge makes them anew."""
    for shot in shotlist.shots:
        shot.candidates = [c for c in shot.candidates if c.source.value not in ("ai_image", "ai_video", "motion")]
        for c in shot.candidates:
            c.score, c.judge_reason, c.reject_reason = None, "", None
        shot.selected, shot.judge_score, shot.judge_notes, shot.attempts = None, None, "", 0


STAGE_FUNCS = {
    "intake": stage_intake,
    "research": stage_research,
    "outline": stage_outline,
    "script": stage_script,
    "narration": stage_narration,
    "shotlist": stage_shotlist,
    "sourcing": stage_sourcing,
    "judge": stage_judge,
    "visuals": stage_visuals,
    "motion": stage_motion,
    "mix": stage_mix,
    "assembly": stage_assembly,
    "packaging": stage_packaging,
}
