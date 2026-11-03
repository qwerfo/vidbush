"""Stage functions: thin glue from disk inputs to module calls to disk outputs.

Each returns the list of output paths it wrote. Module functions called here
are the integration contracts (see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..models import (DirectorPlan, InputMode, Narration, Research, Script, ShotList, VideoFormat)
from ..util import write_json
from .context import StageContext


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
    return [str(ctx.save("shotlist", shotlist))]


def stage_sourcing(ctx: StageContext) -> list[str]:
    from ..source.registry import source_all
    shotlist = ctx.load("shotlist", ShotList)
    shotlist = source_all(shotlist, ctx)
    return [str(ctx.save("shotlist", shotlist))]


def stage_judge(ctx: StageContext) -> list[str]:
    from ..judge.director_loop import judge_and_select
    shotlist = judge_and_select(ctx.load("shotlist", ShotList), ctx)
    return [str(ctx.save("shotlist", shotlist))]


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
    from ..judge.qa import build_qa
    from ..render.package import package
    shotlist = ctx.load("shotlist", ShotList)
    outs = package(shotlist, ctx)
    outs += build_qa(shotlist, ctx)
    minutes = ctx.load("narration", Narration).duration / 60.0
    ctx.tracker.finalize(minutes)
    return [str(p) for p in outs] + [str(ctx.out_path("cost_report.json"))]


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
