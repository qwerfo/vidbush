"""Music bed and final mix: track choice by scene purpose, crossfades, ducking, loudness.

Levels: each track is measured and gained so the bed sits at REF_LUFS +
bed_gain_db under narration (about -32 LUFS with the defaults) and rises by
ducking_depth_db in the open (intro, genuinely long pauses, the tail). Ducking
is FFmpeg sidechaincompress keyed by a smoothed speech envelope: speech regions
with gaps under KEY_MERGE_GAP_S (2.5 s) are merged into one constant level and
held KEY_HOLD_S after the last word, so the bed holds one ducked level through
sentences, scene pauses and paragraph breaks instead of pumping. The key ramps in
dB (so the gain reduction ramps linearly): down over KEY_RAMP_IN_S before speech,
up over KEY_RAMP_OUT_S after it, with a compressor release of at least
MIN_RELEASE_MS. The final mix is two pass normalized (linear gain plus limiter)
to settings.render.target_lufs and true_peak.

Optional sparse SFX (off by default): a soft synthesized whoosh (title cards,
maps) or low hit (number cards) at the start of some motion graphic shots, at
most max_per_minute, mixed at a fixed gain under the narration. Enable it with
an sfx.json ({"enabled": true, "max_per_minute": 3, "gain_db": -24, "kinds":
[...]}) in the brand music folder, or per project in work/sfx.json (which wins).
Events are recorded in mix.json under "sfx".
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from ..config import Settings
from ..models import Brand, Narration, Script, ShotList, StyleProfile
from ..util import JsonLogger, file_fingerprint, media_duration, write_json
from . import align
from .loudness import SR, measure, normalize_graph, write_audio

AUDIO_EXT = {".mp3", ".ogg", ".wav", ".m4a", ".flac", ".aac", ".opus"}
REF_LUFS = -14.0
KEY_MERGE_GAP_S = 2.5  # pauses shorter than this keep the bed ducked
KEY_HOLD_S = 0.4  # the key stays up this long after the last word
KEY_RAMP_IN_S = 0.35  # duck ramp before speech
KEY_RAMP_OUT_S = 1.5  # slow rise after speech
MIN_RELEASE_MS = 1000.0
KEY_RATE = 2000
KEY_LEVEL_DB = -6.0
RATIO = 20.0
MIN_SECTION_S = 20.0
MAX_SECTION_S = 300.0
CHANGE_PURPOSES = {"setup", "turn", "climax", "resolution"}
# purpose -> (preferred moods in order, target energy)
PURPOSE_MOOD: dict[str, tuple[list[str], float]] = {
    "hook": (["uplifting", "tense"], 0.75),
    "setup": (["calm", "mysterious"], 0.3),
    "act": (["mysterious", "calm"], 0.4),
    "item": (["uplifting", "mysterious"], 0.55),
    "turn": (["tense", "mysterious"], 0.6),
    "climax": (["uplifting", "tense"], 0.8),
    "resolution": (["calm", "uplifting"], 0.3),
    "outro": (["calm", "uplifting"], 0.3),
}
_LUFS_CACHE: dict[str, float] = {}


class Track(BaseModel):
    file: str
    path: str
    mood: str = ""
    energy: float = 0.5
    loop: bool = False
    duration: float = 0.0
    lufs: float = REF_LUFS


class Section(BaseModel):
    """One stretch of the bed played by one track. start/end are the nominal boundaries."""

    track: str
    start: float
    end: float
    purpose: str
    gain_db: float = 0.0


# ---------------------------------------------------------------- library

def music_folder(brand: Brand, settings: Settings) -> Optional[Path]:
    """Brand folder (relative to the repo root) when it has audio, else settings.music_folder."""
    for rel in (brand.music_folder, settings.music_folder):
        if not rel:
            continue
        p = settings.path(rel)
        if p.is_dir() and any(f.suffix.lower() in AUDIO_EXT for f in p.iterdir()):
            return p
    return None


def track_lufs(path: Path) -> float:
    key = file_fingerprint(path)
    if key not in _LUFS_CACHE:
        _LUFS_CACHE[key] = float(measure(path)["lufs"])
    return _LUFS_CACHE[key]


def load_library(folder: Path) -> tuple[list[Track], bool]:
    """Tracks in folder with tracks.json metadata. Returns (tracks, has_metadata)."""
    meta_path = folder / "tracks.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tracks = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in AUDIO_EXT:
            continue
        m = meta.get(f.name, {})
        tracks.append(Track(file=f.name, path=str(f), mood=str(m.get("mood", "")),
                            energy=float(m.get("energy", 0.5)), loop=bool(m.get("loop", False)),
                            duration=media_duration(f)))
    return tracks, bool(meta)


# ---------------------------------------------------------------- planning

def scene_starts(script: Script, narration: Narration) -> list[tuple[float, str]]:
    """(start time, purpose) for each scene, from the narration beat spans."""
    out = []
    for sc in script.scenes:
        starts = [narration.beat_spans[b.id][0] for b in sc.beats if b.id in narration.beat_spans]
        if starts:
            out.append((min(starts), sc.purpose.value))
    out.sort()
    return out or [(0.0, "setup")]


def _family(purpose: str) -> str:
    return PURPOSE_MOOD.get(purpose, (["calm"], 0.4))[0][0]


def plan_boundaries(scenes: list[tuple[float, str]], duration: float) -> list[tuple[float, str]]:
    """Section starts: change track only at some scene boundaries, not every scene."""
    out = [(0.0, scenes[0][1])]
    for t, purpose in scenes[1:]:
        since, left = t - out[-1][0], duration - t
        if left < MIN_SECTION_S * 0.7 or since < MIN_SECTION_S:
            continue
        mood_change = purpose in CHANGE_PURPOSES and _family(purpose) != _family(out[-1][1])
        if mood_change or since >= MAX_SECTION_S:
            out.append((t, purpose))
    return out


def pick_track(purpose: str, tracks: list[Track], prev: Optional[str], used: dict[str, int],
               rng: random.Random, use_meta: bool) -> Track:
    """Best track for a scene purpose: mood match, energy closeness, variety."""
    if not use_meta:
        pool = [t for t in tracks if t.file != prev] or tracks
        return rng.choice(pool)
    moods, energy = PURPOSE_MOOD.get(purpose, (["calm"], 0.4))

    def score(t: Track) -> float:
        mood = 2.0 if t.mood == moods[0] else 1.0 if t.mood in moods else 0.0
        return (mood - 2.0 * abs(t.energy - energy) - (3.0 if t.file == prev else 0.0)
                - 0.6 * used.get(t.file, 0) + rng.uniform(0, 0.05))

    return max(tracks, key=score)


def split_for_loops(sections: list[Section], tracks: dict[str, Track], xfade: float) -> list[Section]:
    """Tracks that do not loop seamlessly restart with a crossfade instead of a hard loop."""
    out: list[Section] = []
    for s in sections:
        tr = tracks[s.track]
        limit = tr.duration - xfade - 0.5
        if tr.loop or limit <= MIN_SECTION_S / 2 or s.end - s.start <= limit:
            out.append(s)
            continue
        t = s.start
        while t < s.end - 1e-3:
            out.append(s.model_copy(update={"start": t, "end": min(s.end, t + limit)}))
            t += limit
    return out


def plan_sections(script: Script, narration: Narration, duration: float, tracks: list[Track], use_meta: bool,
                  override: Optional[str], rng: random.Random, xfade: float) -> list[Section]:
    """Sections covering [0, duration] with one track each."""
    by_file = {t.file: t for t in tracks}
    if override and override in by_file:
        return split_for_loops([Section(track=override, start=0.0, end=duration, purpose="override")], by_file, xfade)
    bounds = plan_boundaries(scene_starts(script, narration), duration)
    sections: list[Section] = []
    used: dict[str, int] = {}
    for i, (t, purpose) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else duration
        tr = pick_track(purpose, tracks, sections[-1].track if sections else None, used, rng, use_meta)
        used[tr.file] = used.get(tr.file, 0) + 1
        sections.append(Section(track=tr.file, start=round(t, 3), end=round(end, 3), purpose=purpose))
    return split_for_loops(sections, by_file, xfade)


# ---------------------------------------------------------------- ducking key

def speech_envelope(regions: list[align.Region], duration: float, merge_gap: float = KEY_MERGE_GAP_S,
                    lead_s: float = 0.0, hold_s: float = 0.0) -> list[align.Region]:
    """Speech regions with gaps under merge_gap merged, each started lead_s early (so the dip lands on
    the word) and held hold_s after its end."""
    merged: list[list[float]] = []
    for s, e in sorted(regions):
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(round(max(0.0, s - lead_s), 4), round(min(duration, e + hold_s), 4)) for s, e in merged]


def key_db_curve(envelope: list[align.Region], duration: float, depth_db: float,
                 ramp_in: float = KEY_RAMP_IN_S, ramp_out: float = KEY_RAMP_OUT_S,
                 rate: int = KEY_RATE) -> np.ndarray:
    """Key level in dB per sample: KEY_LEVEL_DB inside the envelope, linear dB ramps from the compressor
    threshold before and after each region, -inf (silence) elsewhere."""
    n = int(np.ceil(duration * rate)) + 1
    t = np.arange(n, dtype=np.float64) / rate
    thr = 20 * np.log10(key_threshold(depth_db))
    span = KEY_LEVEL_DB - thr
    frac = np.zeros(n, dtype=np.float64)
    for s, e in envelope:
        up = np.clip((t - (s - ramp_in)) / max(ramp_in, 1e-3), 0.0, 1.0)
        down = np.clip(1.0 - (t - e) / max(ramp_out, 1e-3), 0.0, 1.0)
        frac = np.maximum(frac, np.minimum(up, down))
    return np.where(frac > 0, thr + span * frac, -np.inf)


def write_key(envelope: list[align.Region], duration: float, path: Path, depth_db: float = 12.0,
              ramp_in: float = KEY_RAMP_IN_S, ramp_out: float = KEY_RAMP_OUT_S) -> Path:
    """Stereo sidechain key at KEY_RATE from key_db_curve (gain reduction ramps linearly in dB)."""
    db = key_db_curve(envelope, duration, depth_db, ramp_in, ramp_out)
    key = np.where(np.isfinite(db), 10 ** (np.nan_to_num(db, neginf=-200.0) / 20.0), 0.0).astype(np.float32)
    return write_audio(path, np.stack([key, key], 1), sr=KEY_RATE)  # stereo: a mono upmix would cost 3 dB


def key_threshold(depth_db: float) -> float:
    """Linear threshold so a key at KEY_LEVEL_DB gives exactly depth_db of gain reduction at RATIO."""
    thr_db = KEY_LEVEL_DB - depth_db * RATIO / (RATIO - 1.0)
    return float(max(10 ** (thr_db / 20.0), 0.000977))


# ---------------------------------------------------------------- graph

def _fit(label_in: str, n: int) -> str:
    return (f"{label_in}aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"apad=whole_len={n},atrim=end_sample={n}")


def bed_graph(sections: list[Section], tracks: dict[str, Track], duration: float,
              xfade: float) -> tuple[list[str], str]:
    """Inputs and filter graph producing [bed]: sections crossfaded (equal power) and summed."""
    inputs: list[str] = []
    chains: list[str] = []
    last = len(sections) - 1
    for i, s in enumerate(sections):
        tr = tracks[s.track]
        a = max(0.0, s.start - xfade / 2) if i > 0 else 0.0
        b = min(duration, s.end + xfade / 2) if i < last else duration
        length = max(b - a, 0.1)
        if tr.loop or tr.duration < length:
            inputs += ["-stream_loop", "-1"]
        inputs += ["-i", tr.path]
        fade_in = min(xfade if i > 0 else 0.6, length / 2)
        fade_out = min(xfade if i < last else 3.0, length / 2)
        chains.append(
            f"[{i}:a]aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo,atrim=0:{length:.3f},"
            f"asetpts=PTS-STARTPTS,volume={s.gain_db:.2f}dB,afade=t=in:d={fade_in:.3f}:curve=qsin,"
            f"afade=t=out:st={length - fade_out:.3f}:d={fade_out:.3f}:curve=qsin,"
            f"adelay=delays={int(round(a * 1000))}:all=1[m{i}]")
    n = int(round(duration * SR))
    mix_in = "".join(f"[m{i}]" for i in range(len(sections)))
    chains.append(f"{mix_in}amix=inputs={len(sections)}:normalize=0:duration=longest,"
                  f"apad=whole_len={n},atrim=end_sample={n}[bed]")
    return inputs, ";".join(chains)


def duck_graph(bed: tuple[list[str], str], key: Path, profile: StyleProfile, n: int) -> tuple[list[str], str]:
    """Extend the bed graph with the sidechain key; the result ends in [duck]."""
    inputs, graph = bed
    k = inputs.count("-i")
    m = profile.music
    duck = (f"sidechaincompress=threshold={key_threshold(m.ducking_depth_db):.6f}:ratio={RATIO}:"
            f"attack={max(m.attack_ms, 0.01)}:release={max(m.release_ms, MIN_RELEASE_MS)}:knee=1:detection=rms:"
            f"link=maximum")
    graph += f";{_fit(f'[{k}:a]', n)}[key];[bed][key]{duck}[duck]"
    return inputs + ["-i", str(key)], graph


def mix_graph(narration_path: str, n: int, bed: Optional[tuple[list[str], str]], key: Optional[Path],
              profile: StyleProfile, sfx: Optional[Path] = None) -> tuple[list[str], str]:
    """Inputs and graph producing [pre]: narration plus the ducked bed and the optional SFX track."""
    if bed is None or key is None:
        inputs, graph, parts = ["-i", narration_path], _fit("[0:a]", n) + "[nar]", ["[nar]"]
    else:
        inputs, graph = duck_graph(bed, key, profile, n)
        k = inputs.count("-i")
        graph += f";{_fit(f'[{k}:a]', n)}[nar]"
        inputs, parts = inputs + ["-i", narration_path], ["[nar]", "[duck]"]
    if sfx is not None:
        k = inputs.count("-i")
        graph += f";{_fit(f'[{k}:a]', n)}[sfx]"
        inputs, parts = inputs + ["-i", str(sfx)], parts + ["[sfx]"]
    if len(parts) == 1:
        return inputs, graph + ";[nar]anull[pre]"
    return inputs, graph + f";{''.join(parts)}amix=inputs={len(parts)}:normalize=0:duration=first[pre]"


# ---------------------------------------------------------------- sparse sfx

class SfxConfig(BaseModel):
    """Optional sparse SFX (sfx.json in the music folder, or work/sfx.json which wins)."""

    enabled: bool = False
    max_per_minute: float = 3.0  # hard cap: events at least 60 / max_per_minute s apart
    gain_db: float = -24.0  # peak level of each effect in the pre-normalization mix
    kinds: list[str] = ["title_card", "map", "number_card"]


SFX_SOUND = {"number_card": "hit"}  # every other kind gets a whoosh
SFX_MIN_START_S = 1.0


def sfx_config(folder: Optional[Path], work_dir: Path) -> SfxConfig:
    """Project work/sfx.json, else <music folder>/sfx.json, else defaults (disabled)."""
    for p in (Path(work_dir) / "sfx.json", (folder / "sfx.json") if folder else None):
        if p is not None and p.exists():
            try:
                return SfxConfig(**json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return SfxConfig()


def sfx_events(shotlist: ShotList, cfg: SfxConfig) -> list[dict[str, Any]]:
    """[{t, shot_id, kind, sound}] at motion graphic starts of the configured kinds, capped in rate."""
    if not cfg.enabled or cfg.max_per_minute <= 0:
        return []
    spacing = 60.0 / cfg.max_per_minute
    out: list[dict[str, Any]] = []
    for shot in shotlist.shots:
        spec = shot.motion
        if shot.type.value != "motion_graphic" or spec is None or spec.kind.value not in cfg.kinds:
            continue
        if shot.start < SFX_MIN_START_S or (out and shot.start - out[-1]["t"] < spacing):
            continue
        out.append({"t": round(shot.start, 3), "shot_id": shot.id, "kind": spec.kind.value,
                    "sound": SFX_SOUND.get(spec.kind.value, "whoosh")})
    return out


def synth_whoosh(sr: int = SR, seconds: float = 0.9, seed: int = 7) -> np.ndarray:
    """Soft stereo whoosh: band-passed noise swept upward, swelling to a peak at 70 percent of its length."""
    n = int(seconds * sr)
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal((n, 2)), axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)[:, None]
    spec *= np.exp(-0.5 * ((np.log(np.maximum(f, 1.0)) - np.log(1400.0)) / 0.7) ** 2)
    y = np.fft.irfft(spec, n, axis=0)
    t = np.linspace(0.0, 1.0, n)[:, None]
    env = np.where(t < 0.7, (t / 0.7) ** 2, np.exp(-(t - 0.7) * 14.0))
    y = y * env
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def synth_hit(sr: int = SR, seconds: float = 0.7) -> np.ndarray:
    """Soft low hit: a 58 Hz sine dropping in pitch with a fast exponential decay."""
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * (58.0 * t + 12.0 * (1 - np.exp(-t * 18.0)) / 18.0)
    y = np.sin(phase) * np.exp(-t * 7.0) * np.minimum(1.0, t / 0.004)
    y = y / (np.abs(y).max() + 1e-9)
    return np.stack([y, y], 1).astype(np.float32)


def write_sfx_track(events: list[dict[str, Any]], duration: float, path: Path, gain_db: float) -> Path:
    """Stereo wav of exactly duration s holding every event; whooshes peak on the cut, hits start on it."""
    n = int(round(duration * SR))
    out = np.zeros((n, 2), dtype=np.float32)
    sounds = {"whoosh": synth_whoosh(), "hit": synth_hit()}
    gain = 10 ** (gain_db / 20.0)
    for ev in events:
        snd = sounds[ev["sound"]]
        start = int(round(ev["t"] * SR)) - (int(len(snd) * 0.7) if ev["sound"] == "whoosh" else 0)
        a, b = max(0, start), min(n, start + len(snd))
        if b > a:
            out[a:b] += snd[a - start: b - start] * gain
    return write_audio(path, out)


# ---------------------------------------------------------------- entry point

def read_override(work_dir: Path) -> Optional[str]:
    p = Path(work_dir) / "music_override.json"
    if not p.exists():
        return None
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("track") or "") or None
    except (json.JSONDecodeError, AttributeError):
        return None


def _choose(script: Script, narration: Narration, duration: float, brand: Brand, profile: StyleProfile,
            settings: Settings, work_dir: Path, log: JsonLogger,
            rng: random.Random) -> tuple[list[Section], dict[str, Track], str, Optional[Path]]:
    """Plan the bed. Returns (sections, tracks by file, selection mode, folder)."""
    folder = music_folder(brand, settings)
    if folder is None:
        log.warn("music folder is empty, mixing narration alone", folder=brand.music_folder)
        return [], {}, "none", None
    tracks, use_meta = load_library(folder)
    if not use_meta:
        log.info("no tracks.json in music folder, picking tracks at random", folder=str(folder))
    override = read_override(work_dir)
    if override and override not in {t.file for t in tracks}:
        log.warn("music override track not found, choosing automatically", track=override)
        override = None
    sections = plan_sections(script, narration, duration, tracks, use_meta, override, rng,
                             profile.music.scene_crossfade_s)
    by_file = {t.file: t for t in tracks}
    m = profile.music
    for s in sections:
        tr = by_file[s.track]
        tr.lufs = track_lufs(Path(tr.path))
        s.gain_db = round(REF_LUFS + m.bed_gain_db + m.ducking_depth_db - tr.lufs, 2)
    mode = "override" if override else "tracks.json" if use_meta else "random"
    return sections, by_file, mode, folder


def build_mix(narration: Narration, script: Script, shotlist: ShotList, brand: Brand, profile: StyleProfile,
              settings: Settings, work_dir: Path, log: JsonLogger, rng: random.Random) -> Path:
    """Write work_dir/mix.wav (48 kHz stereo, exactly shotlist.duration s) and work_dir/mix.json."""
    work_dir = Path(work_dir)
    duration = float(shotlist.duration)
    n = int(round(duration * SR))
    sections, tracks, mode, folder = _choose(script, narration, duration, brand, profile, settings, work_dir,
                                             log, rng)
    m = profile.music
    bed, key, env = None, None, []
    if sections:
        regions, _ = align.speech_regions(narration.audio_path)
        env = speech_envelope(regions, duration, KEY_MERGE_GAP_S, m.attack_ms / 1000.0, KEY_HOLD_S)
        key = write_key(env, duration, work_dir / "mix_key.wav", m.ducking_depth_db)
        bed = bed_graph(sections, tracks, duration, m.scene_crossfade_s)
    sfx_cfg = sfx_config(folder or (settings.path(brand.music_folder) if brand.music_folder else None), work_dir)
    events = sfx_events(shotlist, sfx_cfg)
    sfx = write_sfx_track(events, duration, work_dir / "mix_sfx.wav", sfx_cfg.gain_db) if events else None
    inputs, graph = mix_graph(narration.audio_path, n, bed, key, profile, sfx)
    out = work_dir / "mix.wav"
    loud = normalize_graph(inputs, graph, out, settings.render.target_lufs, settings.render.true_peak, SR, 2,
                           exact_samples=n)
    info = {
        "duration": duration, "samples": n, "sample_rate": SR, "channels": 2, "selection": mode,
        "music_folder": str(folder) if folder else "", "narration_only": not sections,
        "tracks": [{**s.model_dump(), "mood": tracks[s.track].mood, "energy": tracks[s.track].energy}
                   for s in sections],
        "ducking": {"depth_db": m.ducking_depth_db, "attack_ms": m.attack_ms, "release_ms": m.release_ms,
                    "ratio": RATIO, "threshold": key_threshold(m.ducking_depth_db), "key_merge_gap_s": KEY_MERGE_GAP_S,
                    "hold_s": KEY_HOLD_S, "ramp_in_s": KEY_RAMP_IN_S, "ramp_out_s": KEY_RAMP_OUT_S,
                    "compressor_release_ms": max(m.release_ms, MIN_RELEASE_MS), "speech_regions": len(env)},
        "sfx": {"enabled": sfx_cfg.enabled, "gain_db": sfx_cfg.gain_db, "max_per_minute": sfx_cfg.max_per_minute,
                "events": events},
        "bed": {"bed_gain_db": m.bed_gain_db, "under_speech_lufs": REF_LUFS + m.bed_gain_db,
                "open_lufs": REF_LUFS + m.bed_gain_db + m.ducking_depth_db, "crossfade_s": m.scene_crossfade_s},
        "loudness": {"lufs": loud["lufs"], "true_peak": loud["true_peak"], "lra": loud["lra"],
                     "target_lufs": settings.render.target_lufs, "target_tp": settings.render.true_peak},
    }
    write_json(work_dir / "mix.json", info)
    for tmp in (key, sfx):
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    log.info("mix ready", duration=duration, sections=len(sections), selection=mode, lufs=loud["lufs"],
             true_peak=loud["true_peak"])
    return out
