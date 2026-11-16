"""Stills that come alive (Part D): image to video clips of three to five seconds.

Principles, each for a reason:
- The clip's first frame is the chosen still (an illustration, a painted map or a photo-style AI
  still), so a shot can hold the still and then move without a visible jump. compose() holds the
  still first and plays the start of the clip last.
- Every clip is checked before use: the first frame must match the still, and a vision call answers
  FIXED yes or no questions about the last frame against the first (same subject, same layout, no new
  objects, no text, no morphing). Fixed questions make the check repeatable and its failures
  countable; a free-form score would drift with the judge. A painted map clip must also keep its
  registration on the last frame, so moving water never moves the coast.
- A failed clip falls back to the still with the normal Ken Burns move. A clip is never forced in.
- Real photographs and archival stills never come alive: animating them would fabricate footage of a
  real event. Only designed art and photo-style AI stills (already declared generated) are eligible.
- Budget per runtime (profile art.clip_every_s), with the opening served first: the opening needs
  motion most, the rest is spread evenly so clips do not bunch.
"""
from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..models import Candidate, Shot, ShotList, ShotType, SourceKind
from ..util import media_duration
from .common import strip_names

FIRST_FRAME_MIN = 0.80  # grey correlation between the still and the clip's first frame
KIND_RANK = {"illustration": 0, "illustrated_map": 0, "ai_image": 1}
W, H, FPS = 1920, 1080, 30
QUESTIONS = ("same_subject", "same_layout", "new_objects", "text", "morphing")
CHECK_SYSTEM = ("You check short generated video clips for a documentary. You compare frames of one clip and "
                "answer fixed yes or no questions honestly. Reply with strict JSON only.")
CHECK_PROMPT = (
    "The images are the first, middle and last frame of one generated clip, in that order. Answer each "
    "question with true or false.\n"
    "same_subject: does the last frame show the same main subject as the first frame?\n"
    "same_layout: are the positions of the main elements, the horizon and the framing unchanged from the "
    "first frame to the last?\n"
    "new_objects: does anything appear in the middle or last frame that is not in the first frame (a person, "
    "animal, vehicle, building or object)?\n"
    "text: is there any text, letters, numbers, signs or logos in any frame?\n"
    "morphing: is any face, body, animal, building or object distorted, melted or changing shape?\n"
    'Reply as {"same_subject": bool, "same_layout": bool, "new_objects": bool, "text": bool, '
    '"morphing": bool}.')
MOTION = {
    # no clouds or fog: they cover the coast (registration then fails) and veo drifts the camera with them
    "illustrated_map": ("Only the painted sea moves: slow gentle ripples and soft glints of light. Locked-off "
                        "camera, the frame does not move at all. No clouds, no fog, no weather. The land, "
                        "coastlines, rivers and mountains stay perfectly still."),
    "illustration": ("Gentle painterly motion of what is already in the picture only: water, smoke, snow, "
                     "grass or light move softly. Keep the painted style."),
    "ai_image": "Gentle natural motion of what is already in the picture, a slow subtle camera drift at most.",
}
RULES = ("Keep the composition, the framing and every object exactly where it is. No new people, animals or "
         "objects. Nothing appears or disappears. No morphing. No text, letters or logos. No cut, no zoom, no camera move.")


# ---------------------------------------------------------------- planning

def plan_clips(shotlist: ShotList, profile_art: Any, eligible: dict[str, str]) -> list[str]:
    """Shot ids to animate: the budget per runtime, the opening first, the rest one per even bucket."""
    every = float(getattr(profile_art, "clip_every_s", 0) or 0)
    if every <= 0:
        return []
    min_s = float(getattr(profile_art, "clip_min_s", 3.0))
    ai_ok = bool(getattr(profile_art, "clip_from_ai_stills", False))
    shots = [s for s in shotlist.shots if s.id in eligible and s.duration >= min_s
             and eligible[s.id] in KIND_RANK and (ai_ok or eligible[s.id] != "ai_image")]
    if not shots:
        return []
    duration = float(shotlist.duration or max(s.end for s in shotlist.shots))
    n = max(1, int(math.floor(duration / every)))

    def best(pool: list[Shot]) -> Optional[Shot]:
        return min(pool, key=lambda s: (KIND_RANK[eligible[s.id]], -s.duration, s.start), default=None)

    opening = float(getattr(profile_art, "clip_opening_s", 60.0))
    picks: list[str] = []
    first = best([s for s in shots if s.start < opening])
    if first is not None:
        picks.append(first.id)
    left = n - len(picks)
    if left > 0:
        lo = opening if first is not None else 0.0
        step = max(duration - lo, 0.0) / left
        for i in range(left):
            a, b = lo + i * step, lo + (i + 1) * step
            pick = best([s for s in shots if a <= s.start < b and s.id not in picks])
            if pick is not None:
                picks.append(pick.id)
    return picks


def motion_prompt(kind: str, subject: str, window: str = "") -> str:
    """Deterministic image to video prompt: motion of this kind, the subject without names, fixed rules."""
    from ..visuals.ai_images import guess_names
    text = subject or window
    clean = strip_names(text, guess_names(text))
    words = clean.split()
    clean = " ".join(words[:30]).rstrip(",;:") if words else ""
    parts = ["Animate this exact picture with subtle natural motion.", MOTION.get(kind, MOTION["ai_image"])]
    if clean:
        parts.append(f"The picture shows: {clean.rstrip('.')}.")
    parts.append(RULES)
    return " ".join(parts)


# ---------------------------------------------------------------- generate and check

def generate(shot: Shot, still: Path, kind: str, ctx: Any, out_dir: Path, window: str = "",
             subject: str = "") -> Optional[Path]:
    """Raw clip from the video seat (kept on disk: a rerun reuses it). None on any failure."""
    # the raw clip is tied to its still: a different still never reuses a clip of the old one
    tag = hashlib.sha256(Path(still).read_bytes()).hexdigest()[:8] if Path(still).exists() else "nostill"
    out = Path(out_dir) / f"{shot.id}_{tag}_clip_raw.mp4"
    if out.exists() and out.stat().st_size > 2048:
        return out
    try:
        path = ctx.seats.video(motion_prompt(kind, subject or shot.intent, window), Path(still), out, shot.duration,
                               ctx.stage or "visuals")
    except Exception as e:  # noqa: BLE001  any provider failure keeps the still
        ctx.log.warn("art clip failed, keeping still", shot=shot.id, error=str(e)[:300])
        return None
    return Path(path) if path and Path(path).exists() else None


def frames(video: Path, times: list[float]) -> list[np.ndarray]:
    """RGB frames at the given seconds (clamped into the clip)."""
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out = []
        for t in times:
            idx = int(round(t * fps))
            idx = max(0, min(idx, count - 1)) if count else max(0, idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if ok:
                out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return out
    finally:
        cap.release()


def _cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale to cover w x h, then center crop (the same fit compose() gives the clip)."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))), interpolation=cv2.INTER_AREA)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(_cover(a, 160, 90), cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    gb = cv2.cvtColor(_cover(b, 160, 90), cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    ga, gb = ga - ga.mean(), gb - gb.mean()
    den = float(np.linalg.norm(ga) * np.linalg.norm(gb))
    return float(ga @ gb / den) if den > 0 else 0.0


def _read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


CLIP_MAX_SCALE = 0.01  # a map clip frame may differ from the true view by 1 percent of scale
CLIP_MAX_SHIFT = 0.005  # and half a percent of the frame in position: it plays under the true labels unwarped


def map_frames_hold(frames_rgb: list[np.ndarray], payload: Any, reasons: list[str]) -> dict[str, Any]:
    """Register the clip's middle and last frames: each must pass, and with a near identity refit, because the
    renderer plays the clip as it is under the true borders, labels and markers (a still is warped onto the
    view once; a clip frame is not), so a camera drift that registration would correct is wrong on screen."""
    from . import mapref
    out: dict[str, Any] = {}
    for name, f in zip(("mid", "last"), frames_rgb):
        reg = mapref.register(f, payload)
        out[name] = {k: v for k, v in reg.items() if k != "warp"}
        if not reg.get("ok"):
            reasons.append(f"{name} frame map registration: {reg.get('reason') or 'failed'}")
            continue
        w = np.asarray(reg["warp"], np.float64)
        rw, rh = mapref.REG_SIZE
        drift = max(abs(w[0, 0] - 1), abs(w[1, 1] - 1))
        shift = max(abs(w[0, 2]) / rw, abs(w[1, 2]) / rh)
        out[name]["drift"], out[name]["shift"] = round(float(drift), 4), round(float(shift), 4)
        if drift > CLIP_MAX_SCALE or shift > CLIP_MAX_SHIFT:
            reasons.append(f"{name} frame moved off the view (scale {drift:.3f}, shift {shift:.3f})")
    return out


def played_length(shot_s: float, raw_s: float, clip_min_s: float = 3.0, clip_max_s: float = 5.0) -> float:
    """Seconds of the raw clip that reach the screen (compose and hooks.moving_base play its start)."""
    return min(max(shot_s, clip_min_s), clip_max_s, raw_s) if raw_s > 0 else 0.0


def check(raw: Path, still: Path, kind: str, ctx: Any, payload: Any = None, play_s: float = 0.0) -> dict[str, Any]:
    """First-frame match, fixed yes or no vision questions, and map registration on the last frame."""
    raw = Path(raw)
    out: dict[str, Any] = {"ok": False, "reasons": [], "answers": {}, "first_corr": 0.0, "map": None,
                           "frames": []}
    try:
        dur = media_duration(raw)
    except Exception:  # noqa: BLE001
        dur = 0.0
    span = min(play_s, dur) if play_s > 0 else dur  # the frames that play, not the unused tail of the clip
    fr = frames(raw, [0.0, span / 2, max(0.0, span - 0.1)])
    if len(fr) < 3:
        out["reasons"].append("clip frames unreadable")
        return out
    paths = []
    for name, f in zip(("first", "mid", "last"), fr):
        p = raw.with_name(f"{raw.stem}_{name}.jpg")
        cv2.imwrite(str(p), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(str(p))
    out["frames"] = paths
    corr = _corr(_read_rgb(Path(still)), fr[0])
    out["first_corr"] = round(corr, 4)
    if corr < FIRST_FRAME_MIN:
        out["reasons"].append(f"first frame does not match the still (correlation {corr:.2f})")
    try:
        ans = ctx.seats.vision_json("vision", "art_clip_check", CHECK_SYSTEM, CHECK_PROMPT, paths,
                                    ctx.stage or "visuals")
    except Exception as e:  # noqa: BLE001  no verdict, no clip
        ans = {}
        out["reasons"].append(f"vision check failed: {str(e)[:200]}")
    if ans.get("_mock"):
        out["reasons"].append("vision check is a mock (image statistics, not content)")
    answers = {q: ans.get(q) for q in QUESTIONS}
    out["answers"] = answers
    want = {"same_subject": True, "same_layout": True, "new_objects": False, "text": False, "morphing": False}
    for q, good in want.items():
        if answers.get(q) is not good:
            out["reasons"].append(f"{q}: {answers.get(q)}")
    if kind == "illustrated_map" and payload is not None:
        out["map"] = map_frames_hold(fr[1:], payload, out["reasons"])
    out["ok"] = not out["reasons"]
    return out


# ---------------------------------------------------------------- compose

def compose(raw: Path, still: Path, duration: float, out: Path, clip_min_s: float = 3.0,
            clip_max_s: float = 5.0, fps: int = FPS) -> Path:
    """Final clip of exactly duration seconds at 1920x1080: the still held first, then the clip's start."""
    raw_len = media_duration(raw)
    length = min(max(duration, clip_min_s), clip_max_s, raw_len)
    hold = max(0.0, duration - length)
    play = min(duration, length)
    fit = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={fps},format=yuv420p"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if hold > 0.01:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-loop", "1", "-t", f"{hold:.3f}", "-i", str(still), "-i", str(raw),
               "-filter_complex",
               f"[0:v]{fit},trim=duration={hold:.3f},setpts=PTS-STARTPTS[a];"
               f"[1:v]trim=duration={play:.3f},setpts=PTS-STARTPTS,{fit}[b];[a][b]concat=n=2:v=1:a=0[v]",
               "-map", "[v]"]
    else:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
               "-vf", f"trim=duration={play:.3f},setpts=PTS-STARTPTS,{fit}"]
    cmd += ["-an", "-t", f"{duration:.3f}", "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    got = media_duration(out)
    if abs(got - duration) > 0.1:
        raise RuntimeError(f"composed clip is {got:.2f} s, wanted {duration:.2f} s")
    return out


# ---------------------------------------------------------------- one clip

def make_clip(shot: Shot, still: Path, kind: str, ctx: Any, out_dir: Path, payload: Any = None,
              window: str = "", subject: str = "") -> dict[str, Any]:
    """Generate, check and compose one clip. A candidate only when every check passed."""
    res: dict[str, Any] = {"shot_id": shot.id, "kind": kind, "raw": None, "final": None, "check": None,
                           "ok": False, "candidate": None}
    raw = generate(shot, Path(still), kind, ctx, Path(out_dir), window, subject)
    if raw is None:
        return res
    res["raw"] = str(raw)
    art = ctx.profile.art if hasattr(ctx, "profile") else None
    try:
        raw_s = media_duration(raw)
    except Exception:  # noqa: BLE001  unreadable length: check the whole clip
        raw_s = 0.0
    play = played_length(shot.duration, raw_s, float(getattr(art, "clip_min_s", 3.0)),
                         float(getattr(art, "clip_max_s", 5.0)))
    chk = check(raw, Path(still), kind, ctx, payload, play_s=play)
    res["check"] = chk
    if not chk["ok"]:
        ctx.log.info("art clip rejected, keeping still", shot=shot.id, reasons=chk["reasons"][:5])
        return res
    art = ctx.profile.art if hasattr(ctx, "profile") else None
    final = Path(out_dir) / f"{shot.id}_clip.mp4"
    try:
        compose(raw, Path(still), shot.duration, final, float(getattr(art, "clip_min_s", 3.0)),
                float(getattr(art, "clip_max_s", 5.0)))
    except Exception as e:  # noqa: BLE001  a broken encode keeps the still
        ctx.log.warn("art clip compose failed, keeping still", shot=shot.id, error=str(e)[:300])
        return res
    designed = kind in ("illustration", "illustrated_map")
    res["final"] = str(final)
    res["ok"] = True
    res["candidate"] = Candidate(
        id=f"{shot.id}_art_clip", source=SourceKind.art_video if designed else SourceKind.ai_video,
        local_path=str(final), media="video", in_point=0.0, out_point=round(shot.duration, 3), width=W, height=H,
        licence="AI generated", attribution="AI generated animation" if designed else "AI generated video",
        title=shot.intent[:120], keyframes=list(chk["frames"]), score=shot.judge_score,
        judge_reason="clip check passed: " + ", ".join(f"{k}={v}" for k, v in chk["answers"].items()))
    return res


def apply_to_shot(shot: Shot, cand: Candidate) -> None:
    """Select the clip; the shot becomes a moving clip so the renderer plays it instead of Ken Burns."""
    shot.candidates = [c for c in shot.candidates if c.id != cand.id] + [cand]
    shot.selected = cand.id
    shot.type = ShotType.ai_video
