"""Learn a StyleProfile from a reference video.

learn_style(source, name, settings) -> Path downloads a URL with yt-dlp (720p
max, into settings.source_cache_folder) or uses a local file, cuts it into
shots with PySceneDetect, samples two keyframes per shot, classifies shots
with the cheap_classifier vision seat when a real provider is configured,
measures pacing, letterbox and burned captions, and writes
config/style_profiles/<name>.yaml (a full StyleProfile with a provenance
block). In mock mode shots are not classified: the visual mix stays the
default and provenance.mix is "unknown".
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cv2

from ..config import Settings, load_style_profile, save_yaml
from ..models import CARDS_KEY, OVERLAYS_KEY, SHOT_TYPES, PacingProfile, StyleProfile, TypePacing
from ..providers.base import CostTracker
from ..providers.seats import Seats
from ..util import slug
from . import analyze
from .analyze import CaptionResult, DurationStats, LetterboxResult

BASE_PROFILE = "documentary_default"
CLASS_LABELS = SHOT_TYPES + ["talking_head", "unknown"]
MIN_SHOT_S = 1.5
MIN_MOTION_S = 4.0
MIN_SAMPLES = 16
CLASSIFY_WORKERS = 4
CLASSIFY_SYSTEM = ("You classify one shot of a faceless YouTube documentary from two keyframes. Reply with strict "
                   "JSON only, no prose: {\"type\": \"<label>\", \"score\": <0-100 confidence>, \"reason\": "
                   "\"<five words>\"}.")
CLASSIFY_PROMPT = ("Labels: footage (live action video), footage_overlay (video with a graphic or text overlay "
                   "that is not a caption), image (a still photo or painting, often with slow zoom), image_overlay "
                   "(a still with a graphic overlay), ai_image (an AI generated still), ai_video (AI generated "
                   "motion), motion_graphic (full frame map, chart, number or title card), talking_head (a person "
                   "speaking to camera), unknown. Ignore burned captions and black letterbox bars. Pick one label.")


class LearnError(RuntimeError):
    """The reference could not be fetched or analysed."""


# ---------------------------------------------------------------- source

def fetch_source(source: str, settings: Settings) -> Path:
    """Local path as is; URLs are downloaded once with yt-dlp at 720p max into the source cache."""
    if not source.lower().startswith(("http://", "https://")):
        path = Path(source).expanduser()
        if not path.exists():
            raise LearnError(f"reference video not found: {path}")
        return path
    if settings.offline:
        raise LearnError("offline mode is on, so the reference cannot be downloaded; pass a local video file")
    return _download(source, settings.path(settings.source_cache_folder) / "learn")


def _download(url: str, folder: Path) -> Path:
    import yt_dlp
    folder.mkdir(parents=True, exist_ok=True)
    opts = {"format": "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]/b",
            "outtmpl": str(folder / "%(extractor)s_%(id)s.%(ext)s"), "noplaylist": True, "quiet": True,
            "no_warnings": True, "noprogress": True, "retries": 2, "socket_timeout": 30}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as e:  # yt-dlp raises DownloadError and plain network errors
        raise LearnError(f"could not download the reference video ({type(e).__name__}): {str(e)[:300]}. "
                         "Check the network or pass a local video file instead.") from e
    if not path.exists():
        raise LearnError(f"yt-dlp finished but no file was written for {url}")
    return path


# ---------------------------------------------------------------- keyframes and classification

def shot_keyframes(video: Path, shots: list[tuple[float, float]], out_dir: Path) -> list[list[str]]:
    """Two jpg keyframes per shot (at one third and two thirds of the shot)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    times = [s + (e - s) * f for s, e in shots for f in (1 / 3, 2 / 3)]
    frames = analyze.sample_frames(video, times, width=640)
    paths: list[list[str]] = [[] for _ in shots]
    for k, frame in enumerate(frames[: 2 * len(shots)]):
        p = out_dir / f"shot_{k // 2 + 1:03d}_k{k % 2 + 1}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths[k // 2].append(str(p))
    return paths


def classify_shots(keyframes: list[list[str]], seats: Seats) -> list[str]:
    """One label from CLASS_LABELS per shot through the cheap_classifier seat."""
    def one(images: list[str]) -> str:
        if not images:
            return "unknown"
        res = seats.vision_json("cheap_classifier", "classify", CLASSIFY_SYSTEM, CLASSIFY_PROMPT, images,
                                stage="learn", intent="classify shot type")
        label = str(res.get("type") or res.get("label") or "").strip().lower().replace(" ", "_")
        return label if label in CLASS_LABELS else "unknown"

    with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as pool:
        return list(pool.map(one, keyframes))


# ---------------------------------------------------------------- profile building

def _scaled(tp: TypePacing, k: float, floor: float) -> TypePacing:
    lo = max(floor, round(tp.min * k, 2))
    hi = max(lo + 0.5, round(tp.max * k, 2))
    return TypePacing(min=lo, max=hi, mean=round(min(max(tp.mean * k, lo), hi), 2))


def _measured(st: DurationStats, floor: float) -> TypePacing:
    lo = max(floor, st.min)
    hi = max(lo + 0.5, st.max)
    return TypePacing(min=lo, max=hi, mean=round(min(max(st.mean, lo), hi), 2))


def _default_mean(profile: StyleProfile) -> float:
    mix = profile.shot_mix()
    total = sum(mix.values()) or 1.0
    return sum(profile.pacing.for_type(t).mean * w for t, w in mix.items() if t in SHOT_TYPES) / total


def build_pacing(base: StyleProfile, durations: list[float], labels: Optional[list[str]],
                 starts: list[float]) -> PacingProfile:
    """Measured per-type pacing where classified; other types scale the base by the overall tempo ratio."""
    overall = analyze.duration_stats(durations)
    k = min(2.5, max(0.4, overall.mean / max(0.1, _default_mean(base))))
    data: dict[str, Any] = {}
    for t in SHOT_TYPES:
        floor = MIN_MOTION_S if t == "motion_graphic" else MIN_SHOT_S
        mine = [d for d, lab in zip(durations, labels or []) if lab == t]
        if len(mine) >= 2:
            st = analyze.duration_stats(mine)
            data[t] = _measured(st, floor)
        else:
            data[t] = _scaled(base.pacing.for_type(t), k, floor)
    hook = [d for d, s in zip(durations, starts) if s < base.pacing.hook_seconds]
    hook_mult = base.pacing.hook_multiplier
    if starts and starts[-1] + durations[-1] > 2 * base.pacing.hook_seconds and len(hook) >= 3:
        hook_mult = round(min(1.0, max(0.5, (sum(hook) / len(hook)) / max(0.1, overall.mean))), 2)
    jitter = round(min(0.4, max(0.1, overall.cv)), 2)
    return base.pacing.model_copy(update={**data, "hook_multiplier": hook_mult, "jitter": jitter})


def visual_mix(durations: list[float], labels: list[str], default: dict[str, float]) -> dict[str, float]:
    """Share of shot count per shot type in percent (the planner's unit), with motion graphics reported as
    full_screen_cards and animated_overlays 0. Talking heads count as footage.

    durations is kept for the call signature; screen time is reported in provenance, not in the mix.
    """
    acc = {t: 0 for t in SHOT_TYPES}
    for lab in labels:
        lab = "footage" if lab == "talking_head" else lab
        if lab in acc:
            acc[lab] += 1
    total = sum(acc.values())
    if total <= 0:
        return dict(default)
    out = {t: round(100.0 * v / total, 1) for t, v in acc.items()}
    # A classifier sees whole frames: a motion graphic shot is a full screen card, and overlays it saw are
    # already counted in the overlay types, so no extra animated overlays are added on top.
    out[CARDS_KEY] = out.pop("motion_graphic", 0.0)
    out[OVERLAYS_KEY] = 0.0
    return out


def screen_time_mix(durations: list[float], labels: list[str]) -> dict[str, float]:
    """Share of screen time per shot type in percent (secondary info for provenance)."""
    acc = {t: 0.0 for t in SHOT_TYPES}
    for d, lab in zip(durations, labels):
        lab = "footage" if lab == "talking_head" else lab
        if lab in acc:
            acc[lab] += d
    total = sum(acc.values())
    return {t: round(100.0 * v / total, 1) for t, v in acc.items()} if total > 0 else {}


def caption_colour(hex_colour: str) -> str:
    """Measured text colour; near white (compression tinted) snaps to pure white."""
    rgb = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#FFFFFF" if min(rgb) >= 228 else hex_colour.upper()


def build_profile(name: str, base: StyleProfile, durations: list[float], starts: list[float],
                  labels: Optional[list[str]], letterbox: LetterboxResult, captions: CaptionResult,
                  provenance: dict[str, Any]) -> StyleProfile:
    """A full StyleProfile from the measurements, starting from the base profile."""
    caps = base.captions.model_copy(update={"enabled": captions.detected})
    if captions.detected:
        caps = caps.model_copy(update={"size": captions.font_px_1080, "band_height": captions.band_height,
                                       "color": caption_colour(captions.color)})
    lb = base.letterbox.model_copy(update={"enabled": letterbox.detected,
                                           "ratio": letterbox.ratio or base.letterbox.ratio})
    mix = visual_mix(durations, labels, base.visual_mix) if labels else dict(base.visual_mix)
    pacing = build_pacing(base, durations, labels, starts)
    return base.model_copy(update={"id": slug(name), "name": name, "pacing": pacing, "captions": caps,
                                   "letterbox": lb, "visual_mix": mix, "provenance": provenance})


# ---------------------------------------------------------------- entry point

def _measure(video: Path, shots: list[tuple[float, float]]) -> tuple[LetterboxResult, CaptionResult, int]:
    """Letterbox and captions from two frames per shot (topped up with uniform samples)."""
    times = [s + (e - s) * f for s, e in shots for f in (1 / 3, 2 / 3)]
    if len(times) < MIN_SAMPLES:
        end = shots[-1][1]
        times += [end * (i + 0.5) / MIN_SAMPLES for i in range(MIN_SAMPLES)]
    frames = analyze.sample_frames(video, sorted(times))
    lb = analyze.detect_letterbox(frames)
    return lb, analyze.detect_captions(frames, lb), len(frames)


def _provenance(source: str, shots: list[tuple[float, float]], stats: DurationStats, labels: Optional[list[str]],
                lb: LetterboxResult, cap: CaptionResult, samples: int) -> dict[str, Any]:
    counts = {lab: labels.count(lab) for lab in sorted(set(labels))} if labels else {}
    return {"source": source, "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "shot_count": len(shots), "duration_s": round(shots[-1][1] - shots[0][0], 2) if shots else 0.0,
            "mix": "classified" if labels else "unknown", "letterbox_detected": lb.detected,
            "captions_detected": cap.detected, "letterbox_ratio": lb.ratio, "caption_font_px_1080": cap.font_px_1080,
            "caption_band_height": cap.band_height, "caption_frame_share": cap.frame_share,
            "caption_color": cap.color if cap.detected else None, "sampled_frames": samples,
            "durations": stats.model_dump(), "classified_counts": counts, "mix_unit": "shot_count",
            "screen_time_mix": screen_time_mix([e - s for s, e in shots], labels) if labels else {}}


def learn_style(source: str, name: str, settings: Settings) -> Path:
    """Analyse a reference video and write config/style_profiles/<name>.yaml; returns the path."""
    video = fetch_source(source, settings)
    shots = analyze.detect_shots(video)
    durations = [round(e - s, 3) for s, e in shots]
    work = settings.path(settings.source_cache_folder) / "learn" / slug(name)
    keyframes = shot_keyframes(video, shots, work / "keyframes")
    tracker = CostTracker(f"learn_{slug(name)}", work / "cost_report.json")
    seats = Seats(settings, tracker)
    labels = classify_shots(keyframes, seats) if seats.is_real("cheap_classifier") else None
    lb, cap, samples = _measure(video, shots)
    stats = analyze.duration_stats(durations)
    prov = _provenance(source, shots, stats, labels, lb, cap, samples)
    base = load_style_profile(settings, BASE_PROFILE)
    profile = build_profile(name, base, durations, [s for s, _ in shots], labels, lb, cap, prov)
    StyleProfile.model_validate(json.loads(profile.model_dump_json()))
    out = Path(settings.config_folder) / "style_profiles" / f"{slug(name)}.yaml"
    save_yaml(out, profile.model_dump(mode="json"))
    return out
