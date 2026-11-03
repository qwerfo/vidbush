"""Visuals stage: prepare_visuals(shotlist, ctx) -> ShotList.

Runs after the judge. Derived crops for reused stills, varied in points for
reused clips, ai_video upgrades when the profile asks for them and the video
seat is real, and DepthFlow parallax when enabled and installed. Ken Burns
parameters themselves are computed at render time by kenburns.kenburns_params.
"""
from __future__ import annotations

from typing import Any

from ..models import ShotList, ShotType
from . import depthflow
from .ai_video import animate_still
from .derive import STILL_SHOT_TYPES, derive_crops, vary_repeated_video


def prepare_visuals(shotlist: ShotList, ctx: Any) -> ShotList:
    """Apply derived framings, clip variation, ai_video and optional DepthFlow. Returns the shot list."""
    crops = derive_crops(shotlist, ctx)
    clips = vary_repeated_video(shotlist, ctx)
    videos = upgrade_ai_video(shotlist, ctx)
    parallax = apply_depthflow(shotlist, ctx)
    ctx.log.info("visuals prepared", derived_crops=crops, varied_clips=clips, ai_videos=videos,
                 depthflow=parallax)
    return shotlist


def upgrade_ai_video(shotlist: ShotList, ctx: Any) -> int:
    """ai_video shots still holding a still: animate it, else fall back to an ai_image shot."""
    done = 0
    for shot in shotlist.shots:
        still = shot.selected_candidate()
        if shot.type != ShotType.ai_video or still is None or still.media != "image":
            continue
        clip = animate_still(shot, still, ctx)
        if clip is None:
            shot.type = ShotType.ai_image
            continue
        shot.candidates.append(clip)
        shot.selected = clip.id
        done += 1
    return done


def apply_depthflow(shotlist: ShotList, ctx: Any) -> int:
    """Swap still candidates to DepthFlow parallax clips when enabled and installed."""
    if not ctx.profile.motion.depthflow:
        return 0
    if not depthflow.available():
        ctx.log.info("depthflow enabled but the depthflow CLI is not installed, keeping Ken Burns")
        return 0
    done = 0
    for shot in shotlist.shots:
        cand = shot.selected_candidate()
        if shot.type not in STILL_SHOT_TYPES or cand is None:
            continue
        clip = depthflow.parallax_candidate(shot, cand, ctx)
        if clip is not None:
            shot.candidates = [clip if c.id == cand.id else c for c in shot.candidates]
            done += 1
    return done
