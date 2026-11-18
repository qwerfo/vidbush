"""Prove the planned mix on a finished project without a pipeline run.

Rebuilds the mix of a stored project (shot list, script, narration on disk) with the planner replay's scene
purposes and quiet labels, then cuts excerpts and muxes old and new sound onto the existing picture.

  python scripts/sound_proof.py build    # new mix (and the old one, for timing) into <out>/work_new, work_old
  python scripts/sound_proof.py cut      # <out>/{opening,story}_{old,new}.mp4 and numbers.json

Defaults point at Alaska run 3 and the plan v2 replay; see the constants below.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vidbush.audio import music  # noqa: E402
from vidbush.audio.loudness import SR, measure, read_audio  # noqa: E402
from vidbush.audio.sound_qa import gaps, window_db  # noqa: E402
from vidbush.config import load_settings  # noqa: E402
from vidbush.models import Brand, Narration, Script, ShotList, StyleProfile  # noqa: E402
from vidbush.util import NullLogger, run  # noqa: E402

PROJECT = Path("E:/coding projects/Vidbush/output/alaska_run3")
REPLAY = Path("E:/coding projects/Vidbush/plan_v2/run/work")
OUT = Path("E:/coding projects/Vidbush/sound")
EXCERPTS = {"opening": (0.0, 180.0), "story": (300.0, 390.0)}  # story holds the quiet clauses c100 and c108


def _inputs() -> tuple[Narration, Script, ShotList]:
    w = PROJECT / "work"
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    return Narration(**load(w / "narration.json")), Script(**load(w / "script.json")), ShotList(**load(w / "shotlist.json"))


def build() -> dict:
    settings = load_settings(Path(__file__).resolve().parents[1], use_dotenv=False)
    from vidbush.config import load_dotenv
    main_env = load_dotenv(Path("C:/projects/vidbush_tmp/.env"))  # a worktree has no .env; sound keys only
    settings.env.update({k: v for k, v in main_env.items() if k.startswith("ELEVENLABS_")})
    nar, script, sl = _inputs()
    brand = Brand(id="default")
    times = {}
    for name, planned in (("new", True), ("old", False)):
        work = OUT / f"work_{name}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        if planned:
            shutil.copy2(REPLAY / "plan_v2.json", work / "plan_v2.json")
        profile = StyleProfile(id="p", sound={"planned": planned})
        t = time.time()
        music.build_mix(nar, script, sl, brand, profile, settings, work, NullLogger(), random.Random(3))
        times[name] = round(time.time() - t, 1)
        print(name, "mix built in", times[name], "s")
    (OUT / "build_times.json").write_text(json.dumps(times, indent=2), encoding="utf-8")
    return times


def _excerpt_numbers(wav: Path, a: float, b: float) -> dict:
    x = read_audio(wav, SR, 1)[int(a * SR): int(b * SR), 0]
    g = gaps(x, SR)
    db = window_db(x, int(0.05 * SR))
    tmp = OUT / "_x.wav"
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a}", "-t", f"{b - a}", "-i", str(wav), str(tmp)])
    m = measure(tmp)
    tmp.unlink(missing_ok=True)
    return {"lufs": m["lufs"], "true_peak": m["true_peak"], "lra": m["lra"],
            "gaps_over_0_2_s": g, "quietest_50ms_dbfs": round(float(db.min()), 1)}


def cut() -> dict:
    final = PROJECT / "final.mp4"
    numbers = {}
    for name, (a, b) in EXCERPTS.items():
        for which, wav in (("old", PROJECT / "work" / "final_audio.wav"), ("new", OUT / "work_new" / "mix.wav")):
            dst = OUT / f"{name}_{which}.mp4"
            run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a}", "-t", f"{b - a}", "-i", str(final), "-ss", f"{a}",
                 "-t", f"{b - a}", "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset",
                 "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                 "-movflags", "+faststart", str(dst)], timeout=1800)
            numbers[f"{name}_{which}"] = _excerpt_numbers(wav, a, b)
            print(dst.name, numbers[f"{name}_{which}"])
    numbers["old_bed_estimate"] = old_bed_estimate()
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    return numbers


def old_bed_estimate() -> dict:
    """The old mix has no stems: estimate its bed by removing the best scaled copy of the narration."""
    mix = read_audio(PROJECT / "work" / "mix.wav", SR, 1)[:, 0]
    nar = read_audio(PROJECT / "work" / "narration" / "narration.wav", SR, 1)[:, 0]
    n = min(len(mix), len(nar))
    mix, nar = mix[:n], nar[:n]
    g = float(np.dot(mix, nar) / max(np.dot(nar, nar), 1e-9))
    bed = mix - g * nar
    words = json.loads((PROJECT / "work" / "narration.json").read_text(encoding="utf-8"))["words"]
    mask = np.zeros(n, dtype=bool)
    for w in words:
        mask[int(w["start"] * SR): int(w["end"] * SR)] = True

    def lvl(x: np.ndarray) -> float:
        return round(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12), 1)

    return {"narration_gain": round(g, 3), "speech_rms_db": lvl(g * nar[mask]), "bed_under_speech_rms_db": lvl(bed[mask]),
            "bed_rel_speech_db": round(lvl(bed[mask]) - lvl(g * nar[mask]), 1),
            "note": "RMS, unweighted; the new mix's LUFS numbers are in work_new/sound_plan.json qa"}


if __name__ == "__main__":
    {"build": build, "cut": cut}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
