"""Render a sound plan (sound_plan.py) under the narration into work/mix.wav, with stems and work/sound_plan.json.

Every layer is built in numpy at 48 kHz stereo from a gain curve whose ramps are raised cosines of at least
1 s, so nothing can cut, pump or step: there is no compressor in the chain. Music sections meet in equal
power crossfades; beds loop by crossfading between irregular stretches of the file (no audible repeat
point); effects are short one-shots. The sum is normalised once (loudness.normalize: linear gain, limiter
only on peaks) to the render target. Measurement of the result is sound_qa.py.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import Settings
from ..models import Brand, Narration, Script, ShotList, StyleProfile
from ..util import JsonLogger, write_json
from . import sound_plan as sp
from .loudness import SR, db_to_amp, measure, normalize, read_audio, rms_db, write_audio

CTRL_RATE = 100  # gain curves are computed at 100 Hz, then interpolated per sample
FADE_FLOOR_DB = -48.0  # fades to and from silence run at a constant dB rate down to here, then stop
LOOP_SEAM_S = 0.05  # a track marked loopable wraps with this short crossfade
RESTART_XFADE_S = 3.0  # a track that does not loop restarts with this crossfade
DELOOP_XFADE_S = 3.0
DELOOP_MIN_S, DELOOP_MAX_S = 8.0, 18.0
EFFECT_WORD_MARGIN_DB = 12.0  # an effect plays at least this far under the words it overlaps
MIN_BED_LUFS = -62.0  # quieter beds are mostly noise once lifted to level
MAX_BED_GAIN_DB = 30.0
MAX_BED_SPREAD_DB = 12.0
MIN_BED_S = 10.0  # shorter files loop too audibly to make a bed
MAX_TAKES = 4  # a place with this many takes in the library (all unusable) gets no new generation
LEVEL_WINDOW_S = 1.0  # bed leveller: 1 s power windows
LEVEL_OVER_DB = 3.0  # moments louder than the bed's median by more than this are ridden down
LEVEL_MAX_CUT_DB = 12.0
_SPREAD: dict[str, float] = {}
_DECODED: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------- curves

def curve(points: list[tuple[float, float]], n: int, sr: int = SR, fade: bool = False) -> np.ndarray:
    """Per sample values from (time, value) breakpoints joined by raised cosine ramps (flat between equal
    values). Points must be sorted by time.

    fade=True reads values as 0..1 fade positions and ramps them at a constant dB rate (0 dB at 1 down to
    FADE_FLOOR_DB just above 0, silence at 0): a fade to silence then sounds even all the way down instead of
    dropping fast at its tail, and no 0.2 s of it moves more than about 7 dB."""
    if not points:
        return np.ones(n, dtype=np.float32)
    m = int(np.ceil(n / sr * CTRL_RATE)) + 2
    t = np.arange(m) / CTRL_RATE
    ts = np.array([p[0] for p in points], dtype=np.float64)
    vs = np.array([p[1] for p in points], dtype=np.float64)
    idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    nxt = np.clip(idx + 1, 0, len(ts) - 1)
    span = np.maximum(ts[nxt] - ts[idx], 1e-9)
    frac = np.clip((t - ts[idx]) / span, 0.0, 1.0)
    frac = np.where(nxt == idx, 0.0, frac)
    shape = frac if fade else 0.5 - 0.5 * np.cos(np.pi * frac)
    ctrl = vs[idx] + (vs[nxt] - vs[idx]) * shape
    if fade:
        ctrl = np.where(ctrl > 1e-6, 10 ** (FADE_FLOOR_DB * (1.0 - np.clip(ctrl, 0.0, 1.0)) / 20.0), 0.0)
    return np.interp(np.arange(n) / sr, t, ctrl).astype(np.float32)


def fade_points(start: float, end: float, ease_in: float, ease_out: float) -> list[tuple[float, float]]:
    """0..1 envelope of a bed easing in centred on start and out centred on end."""
    a0, a1 = max(0.0, start - ease_in / 2), start + ease_in / 2
    b0, b1 = end - ease_out / 2, end + ease_out / 2
    if a1 > b0:  # too short for both ramps: meet in the middle
        a1 = b0 = (a0 + b1) / 2
    return [(0.0, 0.0), (a0, 0.0), (a1, 1.0), (b0, 1.0), (b1, 0.0)]


# ---------------------------------------------------------------- audio helpers

def decoded(path: str) -> np.ndarray:
    if path not in _DECODED:
        _DECODED[path] = read_audio(path, SR, 2)
    return _DECODED[path]


def _chain(pieces: Any, n: int, k: int, channels: int = 2) -> np.ndarray:
    """n samples from consecutive pieces overlapped k samples in equal power crossfades, written into one
    buffer (linear time). Every piece must be at least 2k long, so each one advances the output."""
    out = np.zeros((n + 2 * k + 1, channels), dtype=np.float32)
    up = np.sin(np.linspace(0.0, 1.0, k, dtype=np.float32) * np.pi / 2)[:, None] if k else None
    pos, first = 0, True
    for piece in pieces:
        seg = np.array(piece, dtype=np.float32, copy=True)
        if k:
            if not first:
                seg[:k] *= up
            seg[-k:] *= up[::-1]
        end = min(len(out), pos + len(seg))
        out[pos:end] += seg[: end - pos]
        pos += len(seg) - k
        first = False
        if pos >= n:
            break
    return out[:n]


def looped(audio: np.ndarray, start_s: float, length_s: float, loopable: bool) -> np.ndarray:
    """length_s of a track from start_s, wrapping to the start (short seam if it loops, else a long crossfade)."""
    n = int(round(length_s * SR))
    if len(audio) < 4 or n <= 0:
        return np.zeros((max(n, 0), 2), dtype=np.float32)
    k = min(int((LOOP_SEAM_S if loopable else RESTART_XFADE_S) * SR), len(audio) // 4)
    pos = int(start_s * SR) % len(audio)
    if len(audio) - pos < 2 * k:
        pos = 0

    def pieces() -> Any:
        yield audio[pos:]
        while True:
            yield audio

    return _chain(pieces(), n, k)


def delooped(audio: np.ndarray, length_s: float, seed: str) -> np.ndarray:
    """A bed of length_s from irregular stretches of the file at seeded offsets, crossfaded so no repeat
    point recurs at a fixed interval."""
    n = int(round(length_s * SR))
    if len(audio) < 4 or n <= 0:
        return np.zeros((max(n, 0), 2), dtype=np.float32)
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16))
    k = min(int(DELOOP_XFADE_S * SR), len(audio) // 4)
    dur = len(audio) / SR

    def pieces() -> Any:
        while True:
            seg = max(min(rng.uniform(DELOOP_MIN_S, DELOOP_MAX_S), dur), 2 * k / SR + 0.01)
            off = rng.uniform(0.0, max(0.0, dur - seg))
            a = int(off * SR)
            yield audio[a: max(a + 2 * k, a + int(seg * SR))]

    return _chain(pieces(), n, k)


def levelled(sig: np.ndarray) -> np.ndarray:
    """A bed held steady: 1 s windows louder than its median by more than LEVEL_OVER_DB are lowered toward it
    (at most LEVEL_MAX_CUT_DB), with the gain itself smoothed over 1 s so the riding is never heard as a move.
    A wave or a gust stays in the bed; it no longer stands out of it."""
    hop = int(0.1 * SR)
    k = len(sig) // hop
    if k < 20:
        return sig
    p = np.mean(sig[: k * hop].astype(np.float64) ** 2, axis=1).reshape(k, hop).mean(axis=1) if sig.ndim == 2 \
        else np.mean(sig[: k * hop].astype(np.float64).reshape(k, hop) ** 2, axis=1)
    w = int(LEVEL_WINDOW_S / 0.1)
    lvl = 10 * np.log10(np.convolve(p, np.ones(w) / w, mode="same") + 1e-12)
    ref = float(np.median(lvl))
    cut = np.clip(lvl - (ref + LEVEL_OVER_DB), 0.0, LEVEL_MAX_CUT_DB)
    cut = np.convolve(cut, np.ones(w) / w, mode="same")
    g = 10 ** (-np.interp(np.arange(len(sig)) / hop, np.arange(k) + 0.5, cut) / 20.0)
    return (sig * (g[:, None] if sig.ndim == 2 else g)).astype(np.float32)


def _place(buf: np.ndarray, sig: np.ndarray, t0: float) -> None:
    a = int(round(t0 * SR))
    lo, hi = max(0, a), min(len(buf), a + len(sig))
    if hi > lo:
        buf[lo:hi] += sig[lo - a: hi - a]


def word_margin_trim(sig: np.ndarray, nar: np.ndarray, words: list[Any], t: float) -> float:
    """dB to lower an effect so it sits EFFECT_WORD_MARGIN_DB under the words it overlaps (0 when it already
    does, or lands in a pause). RMS of the effect against RMS of only the overlapping words."""
    end = t + len(sig) / SR
    over = [(max(w.start, t), min(w.end, end)) for w in words if w.start < end and w.end > t]
    if not over:
        return 0.0
    speech = np.concatenate([nar[int(a * SR): int(b * SR)] for a, b in over])
    if not speech.size:
        return 0.0
    margin = rms_db(speech) - rms_db(sig)
    return max(0.0, EFFECT_WORD_MARGIN_DB - margin)


def level_lufs(path: Path) -> float:
    """Integrated loudness, or RMS of the audible part for sounds too short to gate (under 0.4 s)."""
    got = float(measure(path)["lufs"])
    if got > -69.0:
        return got
    x = read_audio(path, SR, 1)[:, 0]
    act = x[np.abs(x) > np.abs(x).max() * 0.05] if x.size else x
    return rms_db(act) if act.size else -70.0


def bed_ok(root: Path, snd: Any) -> bool:
    """Long enough to loop, loud enough to lift without raising hiss, steady enough to sit under a voice."""
    if snd is None or snd.loudness_lufs is None or float(snd.loudness_lufs) < MIN_BED_LUFS:
        return False
    path = Path(snd.file) if Path(snd.file).is_absolute() else root / snd.file
    return path.exists() and (snd.seconds or MIN_BED_S) >= MIN_BED_S and bed_spread_db(path) <= MAX_BED_SPREAD_DB


def usable_bed(settings: Settings, place: str, log: JsonLogger) -> Any:
    """The first usable bed for a place (reviewed first, then unreviewed, see bed_ok); one new take is
    generated when none is, and used only if it passes the same checks."""
    from . import library
    root = library_root_of(settings)
    found = library.candidates(root, "bed", place)
    for snd in found:
        if bed_ok(root, snd):
            return snd
    if len(found) >= MAX_TAKES:
        log.warn("no steady bed for this place in the library; leaving it without one", place=place, takes=len(found))
        return None
    snd = library.ensure(settings, "bed", place, log, variant=len(found))
    return snd if bed_ok(root, snd) else None


def bed_spread_db(path: Path) -> float:
    """How far a bed's loud moments rise over its body: 99th percentile minus median of 50 ms levels. A steady
    bed reads 3 to 8 dB; one with events in it (clicks, calls, a door) reads 12 or more."""
    key = str(path)
    if key not in _SPREAD:
        x = read_audio(path, SR, 1)[:, 0]
        k = len(x) // 2400
        db = 10 * np.log10(np.mean(x[: k * 2400].reshape(k, 2400).astype(np.float64) ** 2, axis=1) + 1e-12)
        db = db[db > -100]
        _SPREAD[key] = float(np.percentile(db, 99) - np.median(db)) if db.size else 99.0
    return _SPREAD[key]


def synth_room_tone(work_dir: Path) -> Any:
    """A soft room tone made in code (dark noise, 60 Hz to 1.5 kHz, 30 s) for when the library has none."""
    from .library import Sound
    path = Path(work_dir) / "stems" / "room_tone_synth.wav"
    if not path.exists():
        rng = np.random.default_rng(11)
        n = 30 * SR
        spec = np.fft.rfft(rng.standard_normal((n, 2)), axis=0)
        f = np.fft.rfftfreq(n, 1.0 / SR)[:, None]
        spec *= ((f > 60) & (f < 1500)) / np.sqrt(np.maximum(f, 1.0))
        y = np.fft.irfft(spec, n, axis=0)
        write_audio(path, (y / (np.abs(y).max() + 1e-9) * 0.25).astype(np.float32))
    return Sound(id="synth/room_tone", file=str(path), category="bed", place="room_tone", name="Room tone (made in code)",
                 source="vidbush", loudness_lufs=float(measure(path)["lufs"]), review_status="made in code, unheard")


def library_root_of(settings: Settings) -> Path:
    from . import library
    return library.library_root(settings)


# ---------------------------------------------------------------- music inputs

def track_table(folder: Path, log: JsonLogger) -> list[dict[str, Any]]:
    """Every track with its measured tempo, phrase, energy, brightness and mood (tracks.json, built and
    written automatically when missing or stale)."""
    from . import tracks as tr
    try:
        meta = tr.build_tracks_json(folder, write=True) if tr.needs_refresh(folder) else \
            json.loads((folder / "tracks.json").read_text(encoding="utf-8"))
    except OSError:
        meta = tr.build_tracks_json(folder, write=False)
    out = []
    for name, m in meta.items():
        if name.startswith("_") or not (folder / name).exists():
            continue
        out.append({**m, "file": name, "path": str(folder / name)})
    log.info("music tracks measured", folder=str(folder), tracks=len(out))
    return out


# ---------------------------------------------------------------- build

def build_sound(narration: Narration, script: Script, shotlist: ShotList, brand: Brand, profile: StyleProfile,
                settings: Settings, work_dir: Path, log: JsonLogger) -> Path:
    """Plan and render the mix: work/mix.wav, work/mix.json, work/sound_plan.json, work/stems/*.wav."""
    from . import library
    from .music import music_folder, read_override
    from .sound_qa import qa_markdown, measure_mix
    t_start = time.time()
    work_dir = Path(work_dir)
    stems = work_dir / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    cfg = profile.sound
    duration = float(shotlist.duration)
    n = int(round(duration * SR))
    plan_v2 = sp.load_plan_v2(work_dir)
    scenes = sp.scene_infos(script, narration, plan_v2)
    quiets = sp.quiet_spans(shotlist, plan_v2)
    speech = sp.speech_spans(narration)
    nar = read_audio(narration.audio_path, SR, 2)
    nar = np.pad(nar, ((0, max(0, n - len(nar))), (0, 0)))[:n]
    nar_lufs = float(measure(narration.audio_path)["lufs"])
    rel = lambda db: nar_lufs + db  # noqa: E731  absolute LUFS of a level relative to the narration

    # music
    folder = music_folder(brand, settings)
    tracks = track_table(folder, log) if folder else []
    override = read_override(work_dir)
    if override:
        tracks = [t for t in tracks if t["file"] == override] or tracks
    sections, changes = sp.plan_music(scenes, quiets, tracks, duration)
    by_file = {t["file"]: t for t in tracks}
    music = np.zeros((n, 2), dtype=np.float32)
    music_rows = []
    for i, s in enumerate(sections):
        tr = by_file[s.track]
        xf_out = sections[i + 1].xfade_in if i + 1 < len(sections) else 0.0
        t0 = max(0.0, s.start - s.xfade_in / 2)
        t1 = min(duration, s.end + xf_out / 2)
        sig = looped(decoded(tr["path"]), s.offset, t1 - t0, bool(tr.get("loop") or tr.get("loopable")))
        lufs = float(tr.get("lufs") or -14.0)
        sig = sig * db_to_amp(nar_lufs - lufs)
        env = np.ones(len(sig), dtype=np.float32)
        kin, kout = int(s.xfade_in * SR), int(xf_out * SR)
        if kin:
            env[:kin] = np.sin(np.linspace(0, 1, kin) * np.pi / 2)
        elif i == 0:
            k0 = min(int(0.6 * SR), len(env))
            env[:k0] = np.sin(np.linspace(0, 1, k0) * np.pi / 2)
        if kout:
            env[-kout:] = np.cos(np.linspace(0, 1, kout) * np.pi / 2)
        _place(music, sig * env[:, None], t0)
        music_rows.append({"start": s.start, "end": s.end, "track": s.track, "file": tr["path"], "offset_s": s.offset,
                           "mood": s.mood, "track_mood": tr.get("mood"), "track_energy": tr.get("energy"),
                           "crossfade_in_s": s.xfade_in, "after_quiet": s.after_quiet, "reason": s.reason,
                           "gain_db": round(nar_lufs - lufs, 2)})
    level_pts = sp.music_gain_points(speech, duration, cfg.music_under_speech_db, cfg.music_open_db)
    quiet_pts = sp.quiet_gain_points(quiets, duration)
    if sections:
        music *= (curve([(t, db_to_amp(v)) for t, v in level_pts], n) * curve(quiet_pts, n, fade=True))[:, None]

    # ambience and room tone
    amb = np.zeros((n, 2), dtype=np.float32)
    amb_rows, amb_skipped = [], []
    chosen: dict[str, Any] = {}  # one bed choice (and at most one generation) per place per render
    beds = sp.plan_ambience(shotlist, quiets) if cfg.ambience else []
    tone_spans = sp.room_tone_spans(quiets, beds, duration)
    layers = [(b.place, b.start, b.end, b.ease_in, b.ease_out, cfg.ambience_db, b.reason, b.shots) for b in beds]
    layers += [("room_tone", a, b, 1.5, 1.5, cfg.room_tone_db, "room tone under a quiet: the floor while the music is away", [])
               for a, b in tone_spans]
    for place, a, b, ein, eout, db, reason, shots in layers:
        if b >= duration - 0.05:
            eout = 0.0  # a layer that reaches the last frame holds to it: a fade there would end in dead air
        if place not in chosen:
            chosen[place] = usable_bed(settings, place, log)
        snd = chosen[place]
        if snd is None and place == "room_tone":
            snd = synth_room_tone(work_dir)  # the floor under a quiet never depends on the library or network
        if snd is None:
            amb_skipped.append({"place": place, "start": a, "end": b, "why": "no bed in the library and none generated"})
            continue
        path = Path(snd.file) if Path(snd.file).is_absolute() else library.library_root(settings) / snd.file
        lo, hi = max(0.0, a - ein / 2), min(duration, b + eout / 2)
        sig = levelled(delooped(decoded(str(path)), hi - lo, f"{snd.id}:{a:.2f}"))
        gain = db_to_amp(min(rel(db) - float(snd.loudness_lufs), MAX_BED_GAIN_DB))
        env = curve(fade_points(a - lo, b - lo, ein, eout), len(sig), fade=True)
        _place(amb, sig * (gain * env)[:, None], lo)
        amb_rows.append({"place": place, "start": round(a, 3), "end": round(b, 3), "file": str(path), "id": snd.id,
                         "level_db": db, "ease_in_s": ein, "ease_out_s": eout, "review_status": snd.review_status,
                         "shots": shots, "reason": reason})

    # effects
    fx = np.zeros((n, 2), dtype=np.float32)
    fx_rows: list[dict[str, Any]] = []
    events, fx_skipped = sp.plan_effects(shotlist, narration, quiets) if cfg.effects else ([], [])
    variant: dict[str, int] = {}
    for ev in events:
        v = variant.get(ev.kind, 0)
        snd = library.ensure(settings, "effect", ev.kind, log, variant=v % 2) or \
            library.ensure(settings, "effect", ev.kind, log)
        if snd is None:
            fx_skipped.append({"t": ev.t, "kind": ev.kind, "shot_id": ev.shot_id, "why": "no sound in the library"})
            continue
        variant[ev.kind] = v + 1
        path = library.library_root(settings) / snd.file
        sig = decoded(str(path)) * db_to_amp(rel(cfg.effects_db) - level_lufs(path))
        trim = word_margin_trim(sig, nar, narration.words, ev.t)
        _place(fx, sig * db_to_amp(-trim), ev.t)
        fx_rows.append({"t": ev.t, "kind": ev.kind, "shot_id": ev.shot_id, "file": str(path), "id": snd.id,
                        "level_db": round(cfg.effects_db - trim, 1), "review_status": snd.review_status,
                        "reason": ev.reason + (f"; {trim:.1f} dB lower to stay {EFFECT_WORD_MARGIN_DB:.0f} dB under "
                                               "the word it lands on" if trim > 0 else "")})

    # sum, normalise, stems
    for name, arr in (("narration", nar), ("music", music), ("ambience", amb), ("effects", fx)):
        write_audio(stems / f"{name}.wav", arr)  # 16 bit: these are for measuring and the levels page
    pre = work_dir / "mix_pre.wav"
    write_audio(pre, nar + music + amb + fx, codec="pcm_f32le")
    del music, amb, fx
    out = work_dir / "mix.wav"
    loud = normalize(pre, out, settings.render.target_lufs, settings.render.true_peak, SR, 2, exact_samples=n)
    pre_lufs = float(measure(pre)["lufs"])
    pre.unlink(missing_ok=True)
    render_s = round(time.time() - t_start, 1)

    plan = {
        "duration": duration, "narration_lufs": round(nar_lufs, 2), "music_folder": str(folder or ""),
        "music_source": "bundled tracks (fixtures/music)" if folder and folder.name == "music"
        and folder.parent.name == "fixtures" else "operator folder",
        "levels_db_rel_narration": cfg.model_dump(),
        "music": music_rows, "changes": sp.as_dicts(changes),
        "music_level_points": level_pts, "music_quiet_points": quiet_pts,
        "scenes": [{"id": s.id, "start": s.start, "end": s.end, "purpose": s.purpose, "mood": s.mood,
                    "mood_words": s.words} for s in scenes],
        "quiets": [{**sp.asdict(q), "fade_out_start": round(q.start - sp.QUIET_FADE_S, 3),
                    "return_start": round(q.end + sp.LAND_S, 3),
                    "return_end": round(q.end + sp.LAND_S + sp.RETURN_FADE_S, 3)} for q in quiets],
        "ambience": amb_rows, "ambience_skipped": amb_skipped,
        "effects": fx_rows, "effects_skipped": fx_skipped,
        "loudness": {**loud, "target_lufs": settings.render.target_lufs, "target_tp": settings.render.true_peak,
                     "final_gain_db": round(float(loud["lufs"]) - pre_lufs, 2)},
        "render_s": render_s,
    }
    plan["qa"] = measure_mix(work_dir, plan, speech, quiets, [(w.start, w.end) for w in narration.words])
    write_json(work_dir / "sound_plan.json", plan)
    (work_dir / "sound_qa.md").write_text(qa_markdown(plan), encoding="utf-8")
    write_json(work_dir / "mix.json", {
        "duration": duration, "samples": n, "sample_rate": SR, "channels": 2, "selection": "sound_plan",
        "music_folder": str(folder or ""), "narration_only": not sections,
        "tracks": [{"track": r["track"], "start": r["start"], "end": r["end"], "purpose": r["mood"],
                    "gain_db": r["gain_db"]} for r in music_rows],
        "ducking": {}, "sound_plan": "sound_plan.json",
        "sfx": {"enabled": cfg.effects, "events": [{"t": r["t"], "kind": r["kind"]} for r in fx_rows]},
        "loudness": plan["loudness"]})
    _DECODED.clear()  # a long-lived worker must not keep every track, nor read a changed file from memory
    _SPREAD.clear()
    log.info("mix ready", duration=duration, sections=len(sections), beds=len(amb_rows), effects=len(fx_rows),
             quiets=len(quiets), lufs=loud["lufs"], true_peak=loud["true_peak"], seconds=render_s)
    return out
