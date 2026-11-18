"""Labelled sound library: ambience beds and soft graphic effects, reused before anything is generated.

Layout under settings.sound_library:
  catalogue.json                     {"version": 1, "note": ..., "sounds": [Sound, ...]}
  beds/<place>/<place>_<n>.wav        48 kHz stereo loops, each with a .json sidecar
  effects/<family>/<effect>_<n>.wav   48 kHz mono one-shots, each with a .json sidecar

Adam's rulings: a sound must match what is on screen; reviewed sounds are used first, refused ones
never; every generated sound is saved here with a label so it is reused.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from pydantic import BaseModel

from ..util import run
from . import elevenlabs_sfx
from .loudness import SR, measure, write_audio

APPROVED = "approved by Adam"
UNREVIEWED = "not yet reviewed by Adam"
REFUSED = "refused by Adam"
NEVER = 2  # review_rank of a refused sound: never offered

BED_LOOP = "All at a low, even level, one seamless loop."
PLACES: dict[str, dict[str, str]] = {
    "open_ground": {"name": "Open ground, light wind", "depicts": "light wind over open land, grass or tundra",
                    "prompt": "Open grassland under a wide sky, heard outdoors. A light wind moving over the grass, "
                              "soft and steady, open air all around."},
    "snow_wind": {"name": "Snow and ice, cold wind", "depicts": "cold steady wind over snow and ice",
                  "prompt": "A wide snowfield in winter, heard outdoors. A cold steady wind moving over snow and ice, "
                            "a soft dry hiss of drifting snow, open air all around."},
    "coast": {"name": "Calm harbour shore", "depicts": "a calm harbour or shore: water lapping",
              "prompt": "A calm harbour shore, heard outdoors. Small waves lapping gently and evenly against stones "
                        "close by, soft sea air moving over the water."},
    "river": {"name": "River flowing", "depicts": "a river flowing steadily",
              "prompt": "A river flowing steadily, heard from its bank. Smooth running water close by, "
                        "a soft wide rush further off, open air above."},
    "forest": {"name": "Forest air", "depicts": "forest air, faint birds and leaves",
               "prompt": "Forest air in daytime, heard outdoors. Leaves moving softly overhead, "
                         "a few faint birds calling far off among the trees."},
    "city_street": {"name": "Modern city street", "depicts": "a modern city street: traffic and footsteps, distant",
                    "prompt": "A modern city street by day, heard from the pavement. Steady traffic passing a little "
                              "way off, footsteps on paving, a soft city hum."},
    "town_street": {"name": "Small-town street", "depicts": "a small-town street, light activity",
                    "prompt": "A small-town main street by day, heard from the pavement. Light footsteps, a faint "
                              "murmur of a few people, a car rolling past far off."},
    "road": {"name": "Distant highway", "depicts": "distant highway traffic, tyre hiss",
             "prompt": "A highway heard from a distance. Cars passing steadily far off, a soft hiss of tyres on "
                       "dry tarmac, open air beside the road."},
    "interior_quiet": {"name": "Quiet room", "depicts": "a quiet room, its own air",
                       "prompt": "A quiet room, heard from inside. Its own still air and the soft hush of its walls, "
                                 "every moment the same."},
    "interior_public": {"name": "Large public room", "depicts": "a large public room with a faint murmur of people",
                        "prompt": "A large public hall, heard from inside. A faint far-off murmur of many people, "
                                  "soft footsteps across the floor, the wide air of the room."},
    "rain": {"name": "Steady rain outdoors", "depicts": "steady rain outdoors",
             "prompt": "Steady rain outdoors, heard in the open. Rain falling evenly on the ground and leaves "
                       "all around, a soft wide hiss."},
    "room_tone": {"name": "Room tone, near silence",
                  "depicts": "near silence: the soft air of an empty space, used under quiet moments",
                  "prompt": "An empty space in near silence, heard from inside. Only its own soft air and the "
                            "faint hush of the room."},
}

PAPER_IDENTITY = "Every sound is paper, card and pencil on a wooden desk."
EFFECTS: dict[str, dict[str, Any]] = {
    "line_draw": {"family": "paper", "name": "Pencil line on paper", "seconds": 1.2,
                  "depicts": "a soft pencil line drawn across heavy paper",
                  "take": "A soft pencil line drawn slowly across heavy paper", "source": "pencil"},
    "number_land": {"family": "paper", "name": "Soft wooden tap", "seconds": 0.5,
                    "depicts": "a soft muted wooden tap with a short paper touch",
                    "take": "A soft muted tap of wood on a desk with a short touch of paper", "source": "wooden tap"},
    "card_settle": {"family": "paper", "name": "Card laid on paper", "seconds": 0.6,
                    "depicts": "a thick card laid softly down on paper",
                    "take": "A thick card laid softly down flat on a sheet of paper", "source": "card"},
    "mark_sweep": {"family": "paper", "name": "Marker sweep on a page", "seconds": 0.8,
                   "depicts": "a soft felt marker sweep across a printed page",
                   "take": "A soft felt marker swept once across a printed page", "source": "felt marker"},
}

TREAT = "highpass=f=60,highshelf=f=7000:g=-6"  # rumble out; gentle top cut, the audience is older
BED_GEN_SECONDS = 30.0
CROSSFADE_SECONDS = 0.2


class Sound(BaseModel):
    id: str
    file: str  # relative to the library root, forward slashes
    category: str  # "bed" | "effect"
    place: str = ""
    effect: str = ""
    family: str = ""
    name: str = ""
    depicts: str = ""
    suits: str = ""
    seconds: float = 0.0
    loudness_lufs: Optional[float] = None
    loopable: bool = False
    source: str = "elevenlabs"  # or "radioplay:<their id>"
    prompt: str = ""
    generated_date: str = ""  # dd/mm/yyyy
    review_status: str = UNREVIEWED
    review_words: Optional[str] = None
    sha256: str = ""

    @property
    def key(self) -> str:
        return self.place if self.category == "bed" else self.effect


def bed_prompt(place: str) -> str:
    return f"{PLACES[place]['prompt']} {BED_LOOP}"


def effect_prompt(effect: str) -> str:
    e = EFFECTS[effect]
    return (f"{e['take']}, heard close, just above a wooden desk. {PAPER_IDENTITY} "
            f"A single {e['source']} foley sound, close and dry, the whole recording.")


def library_root(settings: Any) -> Path:
    return settings.path(settings.sound_library)


def load_catalogue(root: Path) -> list[Sound]:
    p = Path(root) / "catalogue.json"
    if not p.is_file():
        return []
    return [Sound(**s) for s in json.loads(p.read_text(encoding="utf-8")).get("sounds", [])]


def save_catalogue(root: Path, sounds: list[Sound]) -> None:
    p = Path(root) / "catalogue.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"version": 1, "note": "Vidbush sound library: ambience beds and soft effects. Written by "
           "vidbush/audio/library.py; each file has a .json sidecar with the same record.",
           "sounds": [s.model_dump() for s in sounds]}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def review_rank(status: str) -> int:
    """0 for sounds Adam approved or kept, 1 for unreviewed, NEVER (2) for refused."""
    s = (status or "").lower()
    if s.startswith("refused"):
        return NEVER
    if s.startswith("approved") or s.startswith("kept"):
        return 0
    return 1


def candidates(root: Path, category: str, key: str, exclude: Iterable[str] = ()) -> list[Sound]:
    """Usable sounds for a key: reviewed first, then unreviewed, catalogue order; never refused or missing."""
    skip = set(exclude)
    found = [s for s in load_catalogue(root)
             if s.category == category and s.key == key and s.id not in skip
             and review_rank(s.review_status) < NEVER and (Path(root) / s.file).is_file()]
    return sorted(found, key=lambda s: review_rank(s.review_status))  # stable: keeps catalogue order


def pick(root: Path, category: str, key: str, exclude: Iterable[str] = ()) -> Optional[Sound]:
    found = candidates(root, category, key, exclude)
    return found[0] if found else None


def store(root: Path, sound: Sound, data: np.ndarray, credits: float = 0.0) -> Sound:
    """Write data as the next numbered wav for the sound's key, measure it, label it and add it to the catalogue.
    sound.id, file, seconds, loudness_lufs and sha256 are filled here."""
    root = Path(root)
    folder = root / ("beds" / Path(sound.place) if sound.category == "bed" else "effects" / Path(sound.family))
    n = 1
    while (folder / f"{sound.key}_{n}.wav").exists():
        n += 1
    wav = write_audio(folder / f"{sound.key}_{n}.wav", data, SR)
    sha = hashlib.sha256(wav.read_bytes()).hexdigest()
    try:
        lufs = measure(wav)["lufs"]
        lufs = round(float(lufs), 2) if lufs is not None and np.isfinite(lufs) and lufs > -69.9 else None
    except Exception:  # a clip too short or silent for the meter
        lufs = None
    sound = sound.model_copy(update={
        "id": f"{sound.key}-{sha[:8]}", "file": wav.relative_to(root).as_posix(), "sha256": sha,
        "seconds": round(len(data) / SR, 3), "loudness_lufs": lufs,
        "generated_date": sound.generated_date or date.today().strftime("%d/%m/%Y")})
    wav.with_suffix(".json").write_text(json.dumps({**sound.model_dump(), "credits": credits}, indent=2,
                                                   ensure_ascii=False), encoding="utf-8")
    save_catalogue(root, load_catalogue(root) + [sound])
    return sound


# ---------------------------------------------------------------- signal helpers

def decode(audio: bytes, channels: int, af: str = TREAT) -> np.ndarray:
    """Decode audio bytes (mp3 or anything FFmpeg reads) to float32 (n, channels) at 48 kHz through a filter."""
    cp = run(["ffmpeg", "-v", "error", "-i", "-", "-vn", "-af", af, "-ac", str(channels), "-ar", str(SR),
              "-f", "f32le", "-"], timeout=600, input_bytes=audio)
    return np.frombuffer(cp.stdout, dtype=np.float32).reshape(-1, channels).copy()


def loop_tile(x: np.ndarray, copies: int = 2, xf: int = int(CROSSFADE_SECONDS * SR)) -> np.ndarray:
    """Make x a seamless loop (its tail equal-power crossfaded into its head), then repeat it.
    The result ends just before x[len - xf], which is where its own start begins, so it wraps cleanly."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    t = (np.arange(xf, dtype=np.float32) + 0.5) / xf
    head = x[:xf] * np.sin(t * np.pi / 2)[:, None] + x[len(x) - xf:] * np.cos(t * np.pi / 2)[:, None]
    body = np.concatenate([head, x[xf:len(x) - xf]])
    return np.concatenate([body] * copies)


def trim_effect(x: np.ndarray, floor_db: float = -45.0) -> np.ndarray:
    """Trim leading and trailing silence (below floor_db of the peak), then 10 ms fade in, 30 ms fade out."""
    mono = np.abs(x).max(axis=1)
    peak = float(mono.max()) if len(mono) else 0.0
    if peak <= 0:
        return x
    loud = np.flatnonzero(mono > peak * 10 ** (floor_db / 20))
    pad = int(0.005 * SR)
    y = x[max(0, loud[0] - pad):loud[-1] + pad + 1].copy()
    fi, fo = min(int(0.01 * SR), len(y)), min(int(0.03 * SR), len(y))
    y[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
    y[len(y) - fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
    return y


# ---------------------------------------------------------------- ensure

def ensure(settings: Any, category: str, key: str, log: Any = None, variant: int = 0) -> Optional[Sound]:
    """The variant-th usable sound for a key (reviewed first); generated, treated and labelled when the
    library holds fewer than variant + 1. None when offline or without a key, so the mix still runs."""
    if category not in ("bed", "effect") or key not in (PLACES if category == "bed" else EFFECTS):
        raise ValueError(f"unknown {category} key {key!r}")
    root = library_root(settings)
    found = candidates(root, category, key)
    if len(found) > variant:
        return found[variant]
    prompt = bed_prompt(key) if category == "bed" else effect_prompt(key)
    try:
        if category == "bed":
            mp3, credits = elevenlabs_sfx.generate(prompt, BED_GEN_SECONDS, 0.3, settings)
            data = loop_tile(decode(mp3, 2))
        else:
            seconds = max(1.0, EFFECTS[key]["seconds"] * 1.5)  # room to trim; very short renders come out thin
            mp3, credits = elevenlabs_sfx.generate(prompt, seconds, 0.7, settings)
            data = trim_effect(decode(mp3, 1))
    except elevenlabs_sfx.SoundGenerationError as e:
        if log is not None:
            log.warn(f"sound library: no {category} for {key}: {e}")
        return None
    if category == "bed":
        p = PLACES[key]
        sound = Sound(id="", file="", category="bed", place=key, name=p["name"], depicts=p["depicts"],
                      suits=f"{key.replace('_', ' ')} scenes", loopable=True, prompt=prompt)
    else:
        e = EFFECTS[key]
        sound = Sound(id="", file="", category="effect", effect=key, family=e["family"], name=e["name"],
                      depicts=e["depicts"], suits="map and graphic moments", prompt=prompt)
    sound = store(root, sound, data, credits)
    if log is not None:
        log.info(f"sound library: generated {sound.id} ({credits:.0f} credits)")
    return sound
