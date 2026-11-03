"""Final assembly: cached per-shot segments, concat with stream copy, intro and outro, audio mux.

assemble(shotlist, ctx) -> (final_path, shotlist). Frames come from cumulative
rounding of shot times, so the concatenated picture has exactly
round(ShotList.duration * fps) frames and matches work/mix.wav. Intro and
outro clips are normalised with the same encoder settings (the picture is held
on its last frame when the audio runs longer) and keep their own audio, which is
normalised on its own to the render target with linear gain and a limiter. The
already mastered body mix is never re-normalised: final audio is intro + mix +
outro concatenated and trimmed. captions.srt and captions.ass are rewritten on
the final timeline (shifted by the intro). Writes work/render_stats.json,
work/assembly.json and ctx.root/final.mp4; verify() raises when picture and
sound disagree by more than MAX_AV_DRIFT_FRAMES or the frame count is off.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..models import Shot, ShotList
from ..audio.loudness import normalize
from ..models import ShotType
from ..util import ToolError, ffprobe, media_duration, read_json, run, stable_hash, write_json
from . import ffmpeg_cmds as fc
from .cache import content_fingerprint, frame_range, lines_in_window, resolve_media, segment_file, segment_hash
from .segment import HOLD_WARN_S, footage_plan, render_segment

MAX_AV_DRIFT_FRAMES = 3
BUMPER_VERSION = "bumper-2"


@dataclass
class SegmentJob:
    shot: Shot
    frames: tuple[int, int]
    fade_out: bool
    render_hash: str
    path: Path


def assemble(shotlist: ShotList, ctx: Any) -> tuple[Path, ShotList]:
    """Render or reuse every segment, then build final.mp4 (see module docstring)."""
    settings: Settings = ctx.settings
    seg_dir = ctx.work / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    lines = load_lines(ctx.work / "captions.json")
    jobs = plan_segments(shotlist, ctx.profile, lines, settings, seg_dir, ctx.log)
    if not jobs:
        raise ValueError("shot list has no shot of at least one frame to render")
    stats = render_jobs(jobs, ctx.profile, lines, settings, seg_dir, ctx.log)
    write_json(ctx.work / "render_stats.json", stats)
    for job in jobs:
        job.shot.render_hash = job.render_hash
        job.shot.render_segment_path = str(job.path)
        flag_held_frames(job, settings)
    intro = _bumper(ctx.brand.intro_path, "intro", settings, seg_dir, ctx.log)
    outro = _bumper(ctx.brand.outro_path, "outro", settings, seg_dir, ctx.log)
    fps = settings.render.fps
    body_frames = sum(j.frames[1] - j.frames[0] for j in jobs)
    video_dur = (body_frames + (intro[2] if intro else 0) + (outro[2] if outro else 0)) / fps
    audio = _final_audio(ctx.work, intro, outro, video_dur, settings)
    videos = ([intro[0]] if intro else []) + [j.path for j in jobs] + ([outro[0]] if outro else [])
    final = ctx.root / "final.mp4"
    _mux(videos, audio, final, ctx.work)
    total_frames = body_frames + (intro[2] if intro else 0) + (outro[2] if outro else 0)
    report = verify(final, fps, ctx.log, total_frames)
    intro_s = intro[2] / fps if intro else 0.0
    report.update({"intro_duration": intro_s, "outro_duration": outro[2] / fps if outro else 0.0})
    write_json(ctx.work / "assembly.json", report)
    _write_caption_sidecars(lines, ctx, intro_s)
    return final, shotlist


def _write_caption_sidecars(lines: list[Any], ctx: Any, offset: float) -> None:
    """Rewrite captions.srt/.ass on the final timeline (work/captions.json stays in body time)."""
    if not lines:
        return
    from .captions import CaptionLine, write_sidecars
    objs = [ln if isinstance(ln, CaptionLine) else CaptionLine(**ln) for ln in lines]
    write_sidecars(objs, ctx.profile, ctx.root, ctx.settings, offset)


def flag_held_frames(job: "SegmentJob", settings: Settings) -> None:
    """Set or clear the held_frame flag on a footage shot whose clip is too short even when extended."""
    shot = job.shot
    shot.flags = [f for f in shot.flags if f != "held_frame"]
    c = shot.selected_candidate()
    if c is None or c.media != "video" or shot.type == ShotType.motion_graphic:
        return
    path = resolve_media(c.local_path, settings)
    if not path.is_file():
        return
    try:
        plan = footage_plan(c, path, (job.frames[1] - job.frames[0]) / settings.render.fps, settings.render.fps)
    except (ToolError, OSError, ValueError):
        return
    if plan.hold > HOLD_WARN_S:
        shot.flags.append("held_frame")


def load_lines(path: Path) -> list[Any]:
    """Caption lines from work/captions.json (captions module, or plain dicts as a fallback)."""
    if not path.exists():
        return []
    try:
        from .captions import load_caption_lines
        return list(load_caption_lines(path))
    except ImportError:
        data = read_json(path)
        items = data.get("lines", []) if isinstance(data, dict) else data
        return [d for d in items if isinstance(d, dict)]


def plan_segments(shotlist: ShotList, profile: Any, lines: list[Any], settings: Settings, seg_dir: Path,
                  log: Any) -> list[SegmentJob]:
    """Contiguous frame ranges, fades and render hashes for every shot with at least one frame."""
    fps = settings.render.fps
    total = int(round(shotlist.duration * fps))
    jobs: list[SegmentJob] = []
    cursor = 0
    for i, shot in enumerate(shotlist.shots):
        end = frame_range(shot, fps)[1]
        if i == len(shotlist.shots) - 1:
            end = max(end, total)
        if end <= cursor:
            log.warn("shot shorter than one frame, skipped in render", shot=shot.id)
            continue
        nxt = shotlist.shots[i + 1] if i + 1 < len(shotlist.shots) else None
        fade_out = nxt is not None and nxt.transition_in == "fade"
        frames = (cursor, end)
        window = lines_in_window(lines, cursor / fps, end / fps)
        h = segment_hash(shot, profile, window, settings, frames, fade_out)
        jobs.append(SegmentJob(shot, frames, fade_out, h, segment_file(seg_dir, h)))
        cursor = end
    return jobs


def render_jobs(jobs: list[SegmentJob], profile: Any, lines: list[Any], settings: Settings, seg_dir: Path,
                log: Any) -> dict[str, Any]:
    """Render missing segments in a thread pool; return render stats."""
    t0 = time.time()
    todo = list({j.render_hash: j for j in jobs if not j.path.exists()}.values())

    def one(job: SegmentJob) -> Path:
        ts = time.time()
        out = render_segment(job.shot, profile, lines, settings, seg_dir, job.frames, job.fade_out, log)
        log.info("segment rendered", shot=job.shot.id, hash=job.render_hash,
                 frames=job.frames[1] - job.frames[0], seconds=round(time.time() - ts, 2))
        return out

    with ThreadPoolExecutor(max_workers=max(1, settings.render.workers)) as pool:
        list(pool.map(one, todo))
    stats = {"rendered": len(todo), "cached": len(jobs) - len(todo), "segments": len(jobs),
             "seconds": round(time.time() - t0, 2)}
    log.info("segments ready", **stats)
    return stats


def _bumper(rel: Optional[str], kind: str, settings: Settings, seg_dir: Path,
            log: Any) -> Optional[tuple[Path, Path, int]]:
    """Normalised (video, wav, frames) for an intro or outro, or None when not configured."""
    if not rel:
        return None
    src = resolve_media(rel, settings)
    if not src.is_file():
        log.warn(f"{kind} file not found, skipped", path=str(src))
        return None
    r = settings.render
    frames = int(round(media_duration(src) * r.fps))
    if frames <= 0:
        log.warn(f"{kind} has no duration, skipped", path=str(src))
        return None
    key = stable_hash(kind, BUMPER_VERSION, content_fingerprint(src), r.width, r.height, r.fps, r.crf, r.preset,
                      fc.encoder_args(settings), r.target_lufs, r.true_peak)
    video, wav = seg_dir / f"{kind}_{key}.mp4", seg_dir / f"{kind}_{key}.wav"
    if not video.exists():
        tmp = video.with_suffix(".part.mp4")
        # unbounded clone pad: -frames:v caps the length, so audio longer than video holds the last frame
        vf = fc.chain(fc.fit_pad(r.width, r.height), f"fps={r.fps}", "format=yuv420p",
                      "tpad=stop_mode=clone:stop=-1")
        run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(src), "-vf", vf, "-frames:v", str(frames),
             *fc.encoder_args(settings), str(tmp)], timeout=1800)
        tmp.replace(video)
    if not wav.exists():
        _bumper_audio(src, wav, frames / r.fps, settings)
    return video, wav, frames


def _bumper_audio(src: Path, wav: Path, duration: float, settings: Settings) -> None:
    """Bumper audio: 48 kHz stereo wav of exactly duration seconds at the render loudness target
    (linear gain plus limiter, never loudnorm dynamic mode), or silence when it has none."""
    has_audio = any(s.get("codec_type") == "audio" for s in ffprobe(src).get("streams", []))
    inputs = ["-i", str(src)] if has_audio else ["-f", "lavfi", "-i", f"anullsrc=r={fc.AUDIO_RATE}:cl=stereo"]
    af = fc.chain(f"aresample={fc.AUDIO_RATE}", fc.pad_trim(duration))
    raw = wav.with_name(wav.stem + "_raw.wav")
    run(["ffmpeg", "-y", "-v", "error", "-nostdin", *inputs, "-vn", "-af", af, "-ac", "2",
         "-ar", str(fc.AUDIO_RATE), "-c:a", "pcm_s16le", str(raw)], timeout=600)
    r = settings.render
    tmp = wav.with_name(wav.stem + "_part.wav")
    normalize(raw, tmp, r.target_lufs, r.true_peak, fc.AUDIO_RATE, 2,
              exact_samples=int(round(duration * fc.AUDIO_RATE)))
    raw.unlink(missing_ok=True)
    tmp.replace(wav)


def _final_audio(work: Path, intro: Optional[tuple[Path, Path, int]], outro: Optional[tuple[Path, Path, int]],
                 video_dur: float, settings: Settings) -> Path:
    """Audio of exactly video_dur seconds: mix.wav alone, or intro + mix + outro joined (each part is
    already at the target, so nothing is re-normalised)."""
    mix = work / "mix.wav"
    if not mix.exists():
        raise FileNotFoundError(f"missing {mix}: run the mix stage first")
    out = work / "final_audio.wav"
    if not intro and not outro:
        run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(mix), "-af",
             fc.chain(f"aresample={fc.AUDIO_RATE}", fc.pad_trim(video_dur)), "-ac", "2", "-ar", str(fc.AUDIO_RATE),
             "-c:a", "pcm_s16le", str(out)], timeout=1800)
        return out
    parts = ([intro[1]] if intro else []) + [mix] + ([outro[1]] if outro else [])
    inputs = [a for p in parts for a in ("-i", str(p))]
    pre = "".join(f"[{i}:a]aresample={fc.AUDIO_RATE},aformat=channel_layouts=stereo[a{i}];"
                  for i in range(len(parts)))
    cat = ("".join(f"[a{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1,"
           f"{fc.pad_trim(video_dur)}[out]")
    run(["ffmpeg", "-y", "-v", "error", "-nostdin", *inputs, "-filter_complex", pre + cat, "-map", "[out]",
         "-ac", "2", "-ar", str(fc.AUDIO_RATE), "-c:a", "pcm_s16le", str(out)], timeout=1800)
    return out


def _mux(videos: list[Path], audio: Path, final: Path, work: Path) -> None:
    """Concat video with stream copy and mux AAC 192 kbps 48 kHz stereo with +faststart."""
    lst = fc.write_concat_list(videos, work / "concat.txt")
    tmp = final.with_suffix(".part.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-ar", str(fc.AUDIO_RATE), "-ac", "2", "-movflags", "+faststart", str(tmp)], timeout=3600)
    tmp.replace(final)


def verify(final: Path, fps: int, log: Any, expected_frames: Optional[int] = None) -> dict[str, Any]:
    """Probe final.mp4. durations_match means within one frame; raise ToolError beyond
    MAX_AV_DRIFT_FRAMES of drift or when the frame count differs from expected_frames."""
    info = ffprobe(final)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), {})
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), {})
    vd, ad = float(v.get("duration", 0.0)), float(a.get("duration", 0.0))
    ok = abs(vd - ad) <= 1.0 / fps + 1e-6
    frames = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None
    fields = {"video_duration": round(vd, 4), "audio_duration": round(ad, 4), "durations_match": ok,
              "frames": frames, "expected_frames": expected_frames}
    (log.info if ok else log.warn)("final durations checked", path=str(final), **fields)
    if abs(vd - ad) > MAX_AV_DRIFT_FRAMES / fps + 1e-6:
        raise ToolError(f"final.mp4 picture ({vd:.3f} s) and sound ({ad:.3f} s) are out of sync")
    if expected_frames is not None and frames is not None and frames != expected_frames:
        raise ToolError(f"final.mp4 has {frames} frames, expected {expected_frames}")
    return fields
