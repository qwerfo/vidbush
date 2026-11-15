"""apply_art(shotlist, ctx): the art layer inside the pipeline (visuals stage, after the judge).

Order and budgets come from the style profile's art block (models.ArtProfile; every share is off unless the
profile sets it):
1. Illustrated maps: of the map slots (full screen maps and map insets), those whose narration establishes a
   place (paint.paint_reason is None) may get a painted base, up to illustrated_maps times the number of maps,
   full screen maps first. The same view is never illustrated twice within map_repeat_s; outside that window
   the earlier painting is reused at no cost.
2. Illustrations: up to illustrations times the number of shots, chosen by illustrate.choose, judged against
   the shot window like any still; only a passing one replaces the pick.
3. Clips: one per clip_every_s of runtime, the opening first (clips.plan_clips), from illustrations, painted
   full screen maps and (when allowed) photo-style AI stills; never from a real photograph. A painted map's
   checked clip becomes its moving base (payload.art_clip); any other clip replaces the still. A failed clip
   keeps the still with its Ken Burns move.
Art is never mocked: with no Elevate image seat, offline mode or every share at 0 nothing changes. Any error
leaves the shot it happened on as it was. Every decision is written to work/art/art_report.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..models import MapPayload, MotionKind, MotionSpec, Shot, ShotList, ShotType, SourceKind
from . import clips as C
from . import common, illustrate as IL, paint as P
from .mapref import view_key


def _map_spec(shot: Shot) -> tuple[Optional[MotionSpec], str]:
    if shot.motion is not None and shot.motion.kind == MotionKind.map:
        return shot.motion, "full"
    if shot.overlay is not None and shot.overlay.kind == MotionKind.map:
        return shot.overlay, "inset"
    return None, ""


def enabled(ctx: Any) -> bool:
    a = ctx.profile.art
    return (a.illustrated_maps > 0 or a.illustrations > 0 or a.clip_every_s > 0) and common.elevate_spec(ctx) is not None


def paint_maps(shotlist: ShotList, ctx: Any, narr: Any, report: dict[str, Any]) -> int:
    a = ctx.profile.art
    maps = [(s, *_map_spec(s)) for s in shotlist.shots]
    maps = [(s, spec, where) for s, spec, where in maps if spec is not None]
    if a.illustrated_maps <= 0 or not maps:
        return 0
    budget = max(1, round(a.illustrated_maps * len(maps)))
    spec_e = common.elevate_spec(ctx)
    painted: dict[str, tuple[str, float]] = {}  # view -> (base path, last time on screen)
    done = 0
    out = common.art_dir(ctx) / "maps"
    for s, spec, where in sorted(maps, key=lambda m: (m[2] != "full", m[0].start)):
        row: dict[str, Any] = {"shot": s.id, "where": where}
        report["maps"].append(row)
        window = common.shot_text(narr, s)
        reason = P.paint_reason(spec, window)
        if reason:
            row["plain"] = reason
            continue
        p = spec.typed_payload()
        vk = json.dumps(view_key(p), sort_keys=True)
        if vk in painted:
            base, last = painted[vk]
            if abs(s.start - last) < a.map_repeat_s:
                row["plain"] = f"same view illustrated {abs(s.start - last):.0f} s away"
                continue
            p.art_base = base
            spec.payload = p.model_dump()
            painted[vk] = (base, s.start)
            row["base"], row["reused"] = base, True
            continue
        if done >= budget:
            row["plain"] = "illustrated map share used"
            continue
        if spec_e is not None and common.over_hour_cap(ctx, spec_e):
            row["plain"] = "hour credit cap reached"
            continue
        try:
            rep = P.paint_map(p, window, ctx, out, tag=s.id)
        except Exception as e:  # noqa: BLE001  the map stays plain
            row["plain"] = f"paint failed: {type(e).__name__}: {str(e)[:200]}"
            continue
        row["report"] = rep
        done += 1
        if rep.get("base"):
            p.art_base = rep["base"]
            spec.payload = p.model_dump()
            painted[vk] = (rep["base"], s.start)
            row["base"] = rep["base"]
        else:
            row["plain"] = "no variant passed registration and the text check"
    return sum(1 for r in report["maps"] if r.get("base"))


def add_illustrations(shotlist: ShotList, ctx: Any, narr: Any, report: dict[str, Any]) -> int:
    a = ctx.profile.art
    if a.illustrations <= 0:
        return 0
    budget = max(1, round(a.illustrations * len(shotlist.shots)))
    spec_e = common.elevate_spec(ctx)
    out = common.art_dir(ctx) / "illustrations"
    n = 0
    for pick in IL.choose(shotlist, ctx, budget):
        try:
            shot = shotlist.by_id(pick["shot_id"])
        except KeyError:
            continue
        if spec_e is not None and common.over_hour_cap(ctx, spec_e):
            report["illustrations"].append({"shot": shot.id, "skipped": "hour credit cap reached"})
            break
        try:
            rep = IL.illustrate(shot, pick, ctx, out, common.shot_text(narr, shot))
        except Exception as e:  # noqa: BLE001  the shot keeps its pick
            report["illustrations"].append({"shot": shot.id, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        cand = rep.pop("candidate", None)
        report["illustrations"].append(rep)
        if cand is not None:
            IL.apply_to_shot(shot, cand)
            n += 1
    return n


def _clip_kind(shot: Shot) -> Optional[str]:
    spec, where = _map_spec(shot)
    if spec is not None:
        return "illustrated_map" if where == "full" and spec.payload.get("art_base") else None
    c = shot.selected_candidate()
    if c is None or c.media != "image" or not c.local_path or not Path(c.local_path).exists():
        return None
    if c.source == SourceKind.illustration:
        return "illustration"
    if c.source == SourceKind.ai_image:
        return "ai_image"
    return None  # a real photograph never comes alive


def add_clips(shotlist: ShotList, ctx: Any, narr: Any, report: dict[str, Any]) -> int:
    a = ctx.profile.art
    if a.clip_every_s <= 0 or not ctx.seats.is_real("video"):
        return 0
    eligible = {s.id: k for s in shotlist.shots if (k := _clip_kind(s))}
    out = common.art_dir(ctx) / "clips"
    n = 0
    for sid in C.plan_clips(shotlist, a, eligible):
        shot, kind = shotlist.by_id(sid), eligible[sid]
        try:
            if kind == "illustrated_map":
                p = MapPayload(**shot.motion.payload)
                res = C.make_clip(shot, Path(p.art_base), kind, ctx, out, payload=p, window=common.shot_text(narr, shot))
                if res.get("ok") and res.get("raw"):
                    p.art_clip = res["raw"]  # the renderer plays it as the moving base, markers and labels on top
                    shot.motion.payload = p.model_dump()
                    n += 1
            else:
                still = shot.selected_candidate().local_path
                res = C.make_clip(shot, Path(still), kind, ctx, out, window=common.shot_text(narr, shot))
                if res.get("candidate") is not None:
                    C.apply_to_shot(shot, res["candidate"])
                    n += 1
        except Exception as e:  # noqa: BLE001  the shot keeps its still
            res = {"shot_id": sid, "kind": kind, "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        res.pop("candidate", None)
        report["clips"].append(res)
    return n


def apply_art(shotlist: ShotList, ctx: Any) -> dict[str, int]:
    """Run the art layer on a judged shot list in place. Returns counts; nothing happens when disabled."""
    if not enabled(ctx):
        return {}
    narr = common.narration(ctx)
    report: dict[str, Any] = {"maps": [], "illustrations": [], "clips": []}
    counts = {"illustrated_maps": 0, "illustrations": 0, "clips": 0}
    for key, step in (("illustrated_maps", paint_maps), ("illustrations", add_illustrations), ("clips", add_clips)):
        try:
            counts[key] = step(shotlist, ctx, narr, report)
        except Exception as e:  # noqa: BLE001  art is decoration: a broken step never stops the video
            ctx.log.warn("art step failed, continuing without it", step=key, error=f"{type(e).__name__}: {str(e)[:300]}")
    report["counts"] = counts
    common.write_report(ctx, "art_report.json", report)
    ctx.log.info("art layer", **counts)
    return counts


def art_counts(shotlist: ShotList) -> dict[str, int]:
    """Designed animation in a shot list, for reports: illustrated maps, illustrations, art clips; photo-style AI
    stills and clips are counted apart."""
    out = {"illustrated_maps": 0, "illustrations": 0, "art_clips": 0, "ai_stills": 0, "ai_clips": 0}
    for s in shotlist.shots:
        spec, _ = _map_spec(s)
        if spec is not None and spec.payload.get("art_base"):
            out["illustrated_maps"] += 1
        c = s.selected_candidate()
        src = c.source if c is not None else None
        out["illustrations"] += src == SourceKind.illustration
        out["art_clips"] += src == SourceKind.art_video
        out["ai_stills"] += src == SourceKind.ai_image
        out["ai_clips"] += src == SourceKind.ai_video
    return out
