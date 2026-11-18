"""Listen by measurement: levels of every layer against the narration, and anything that would read as a fault.

Faults flagged (each with its time):
  gap        the delivered mix drops below GAP_DB for longer than MAX_GAP_S (dead air)
  cut        the music or ambience bus level (400 ms windows) steps by more than CUT_DB within 0.2 s while
             audible: a bed dropping out or cutting in. Sounds inside a bed (a wave, a gust) are not steps.
  fast ramp  a planned gain change faster than the easing rule (a ramp shorter than MIN_RAMP_S)
  loud bed   music or ambience under speech more than LIMIT_SLACK_DB above its set level
  on a word  a graphic effect less than WORD_MARGIN_DB under the narration it plays over
  loudness   integrated loudness more than 1 LU from target, or true peak over the ceiling
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .loudness import SR, measure, read_audio, write_audio

GAP_DB = -65.0  # RMS dBFS of a 50 ms window counted as dead air
MAX_GAP_S = 0.2
CUT_DB = 15.0  # an eased change (1 s or longer through the 48 dB fade range) moves under 10 dB in 0.2 s
CUT_AUDIBLE_DBFS = -55.0
MIN_RAMP_S = 1.0
LIMIT_SLACK_DB = 3.0
WORD_MARGIN_DB = 10.0
QA_SR = 48000


def _mask(spans: list[tuple[float, float]], n: int, sr: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in spans:
        m[max(0, int(a * sr)): min(n, int(b * sr))] = True
    return m


def masked_lufs(x: np.ndarray, mask: np.ndarray, tmp: Path) -> float:
    """Integrated loudness of the samples under mask (joined), via the ebur128 meter."""
    sel = x[mask]
    if sel.size < QA_SR * 0.5 or not np.any(np.abs(sel) > 1e-7):
        return -70.0
    write_audio(tmp, sel, sr=QA_SR, codec="pcm_f32le")
    try:
        return round(float(measure(tmp)["lufs"]), 2)
    finally:
        tmp.unlink(missing_ok=True)


def window_db(x: np.ndarray, win: int) -> np.ndarray:
    k = len(x) // win
    if k == 0:
        return np.array([-120.0])
    blocks = x[: k * win].reshape(k, win).astype(np.float64)
    return 10 * np.log10(np.mean(blocks ** 2, axis=1) + 1e-12)


def gaps(x: np.ndarray, sr: int = QA_SR) -> list[tuple[float, float]]:
    """(start, length) of every run of 50 ms windows under GAP_DB longer than MAX_GAP_S."""
    w = int(0.05 * sr)
    quiet = window_db(x, w) < GAP_DB
    out, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * 0.05 > MAX_GAP_S:
                out.append((round(start * 0.05, 2), round((i - start) * 0.05, 2)))
            start = None
    return out


def cuts(x: np.ndarray, sr: int = QA_SR) -> list[dict[str, float]]:
    """Times where a bus's 400 ms level changes by more than CUT_DB within 0.2 s while audible (one entry per
    step, the largest)."""
    hop = int(0.05 * sr)
    db = window_db(x, hop)  # 50 ms blocks, averaged in power over 8 blocks = 400 ms
    p = 10 ** (db / 10)
    k = 8
    if len(p) <= k + 4:
        return []
    lvl = 10 * np.log10(np.convolve(p, np.ones(k) / k, mode="valid") + 1e-12)
    step = np.abs(lvl[4:] - lvl[:-4])  # 0.2 s apart
    hot = (step > CUT_DB) & (np.maximum(lvl[4:], lvl[:-4]) > CUT_AUDIBLE_DBFS)
    out: list[dict[str, float]] = []
    for i in np.where(hot)[0]:
        t = round((i + 4 + k / 2) * 0.05, 2)
        if out and t - out[-1]["t"] < 1.0:
            out[-1]["step_db"] = max(out[-1]["step_db"], round(float(step[i]), 1))
            continue
        out.append({"t": t, "step_db": round(float(step[i]), 1)})
    return out[:50]


def shortest_ramp(points: list[tuple[float, float]]) -> float:
    best = float("inf")
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if abs(v1 - v0) > 1e-6:
            best = min(best, t1 - t0)
    return round(best, 3) if best != float("inf") else None  # None: the level never changes


def measure_mix(work_dir: Path, plan: dict[str, Any], speech: list[tuple[float, float]],
                quiets: list[Any], words: Any = ()) -> dict[str, Any]:
    """Every number in the sound section of qa_report.md, from the stems and the delivered mix."""
    work_dir = Path(work_dir)
    st = work_dir / "stems"
    nar, mus, amb, fx = (read_audio(st / f"{k}.wav", QA_SR, 1)[:, 0]
                         for k in ("narration", "music", "ambience", "effects"))
    n = len(nar)
    g = float(plan["loudness"].get("final_gain_db", 0.0))
    tmp = work_dir / "_qa_tmp.wav"
    words_mask = _mask(speech, n, QA_SR)
    quiet_mask = _mask([(q.start - 1.5, q.end + 3.0) for q in quiets], n, QA_SR)
    lv = plan["levels_db_rel_narration"]
    speech_lufs = masked_lufs(nar, words_mask, tmp)
    music_under = masked_lufs(mus, words_mask & ~quiet_mask, tmp)
    music_open = masked_lufs(mus, ~words_mask & ~quiet_mask, tmp)
    amb_mask = _mask([(r["start"], r["end"]) for r in plan["ambience"] if r["place"] != "room_tone"], n, QA_SR)
    tone_mask = _mask([(r["start"], r["end"]) for r in plan["ambience"] if r["place"] == "room_tone"], n, QA_SR)
    amb_under = masked_lufs(amb, words_mask & amb_mask, tmp)
    beds = []
    for r in plan["ambience"]:
        m = _mask([(r["start"], r["end"])], n, QA_SR)
        got = masked_lufs(amb, m & words_mask, tmp) if r["place"] != "room_tone" else masked_lufs(amb, m, tmp)
        beds.append({"place": r["place"], "start": r["start"], "end": r["end"], "rel_db": round(got - speech_lufs, 1),
                     "set_db": r["level_db"]})
    effects = []
    for r in plan["effects"]:
        a, b = int(r["t"] * QA_SR), int((r["t"] + 0.8) * QA_SR)
        e_db = float(window_db(fx[a:b], max(b - a, 1))[0])
        # the voice it plays over: only the words that overlap it, so a pause does not count as quiet speech
        over = [(max(s, r["t"]), min(e, r["t"] + 0.8)) for s, e in words if s < r["t"] + 0.8 and e > r["t"]]
        sel = np.concatenate([nar[int(s * QA_SR): int(e * QA_SR)] for s, e in over]) if over else np.zeros(0)
        s_db = float(window_db(sel, max(len(sel), 1))[0]) if sel.size else -120.0
        effects.append({"t": r["t"], "kind": r["kind"], "effect_rms_db": round(e_db + g, 1),
                        "word_rms_db": round(s_db + g, 1) if sel.size else None,
                        "margin_db": round(s_db - e_db, 1) if sel.size else None, "over_speech": bool(sel.size)})
    mix = read_audio(work_dir / "mix.wav", QA_SR, 1)[:, 0]
    gap_list = gaps(mix)
    bed_cuts = cuts(mus) + [{**c, "bus": "ambience"} for c in cuts(amb)]
    ramps = {"music_level": shortest_ramp(plan["music_level_points"]),
             "quiet": shortest_ramp(plan["music_quiet_points"]),
             "music_crossfade": min([c["xfade_s"] for c in plan["changes"] if c["kind"] == "crossfade"] or [0.0]),
             "ambience_ease": min([e for r in plan["ambience"] for e in (r["ease_in_s"], r["ease_out_s"]) if e > 0] or [0.0])}
    loud = plan["loudness"]
    faults: list[str] = []
    for t, length in gap_list:
        faults.append(f"gap: {length:.2f} s of dead air at {t:.2f} s")
    for c in bed_cuts[:10]:
        faults.append(f"cut: bed level steps {c['step_db']} dB at {c['t']:.2f} s")
    for k, v in ramps.items():
        if v is not None and 0 < v < MIN_RAMP_S:
            faults.append(f"fast ramp: {k} changes over {v:.2f} s (rule: at least {MIN_RAMP_S} s)")
    rel_music = round(music_under - speech_lufs, 1)
    if music_under > -69 and rel_music > lv["music_under_speech_db"] + LIMIT_SLACK_DB:
        faults.append(f"loud bed: music under speech at {rel_music} dB (set {lv['music_under_speech_db']} dB)")
    for b in beds:
        if b["place"] != "room_tone" and b["rel_db"] > b["set_db"] + LIMIT_SLACK_DB and b["rel_db"] > -60:
            faults.append(f"loud bed: {b['place']} at {b['rel_db']} dB (set {b['set_db']} dB) at {b['start']:.1f} s")
    for e in effects:
        if e["over_speech"] and e["margin_db"] < WORD_MARGIN_DB:
            faults.append(f"on a word: {e['kind']} at {e['t']:.2f} s only {e['margin_db']} dB under the voice")
    if abs(float(loud["lufs"]) - float(loud["target_lufs"])) > 1.0:
        faults.append(f"loudness: {loud['lufs']} LUFS against {loud['target_lufs']}")
    if float(loud["true_peak"]) > float(loud["target_tp"]) + 0.1:
        faults.append(f"loudness: true peak {loud['true_peak']} dBTP over {loud['target_tp']}")
    return {
        "speech_lufs": round(speech_lufs + g, 2),
        "music_under_speech_lufs": round(music_under + g, 2), "music_under_speech_rel_db": rel_music,
        "music_open_lufs": round(music_open + g, 2) if music_open > -69 else None,
        "music_open_rel_db": round(music_open - speech_lufs, 1) if music_open > -69 else None,
        "ambience_under_speech_lufs": round(amb_under + g, 2),
        "ambience_under_speech_rel_db": round(amb_under - speech_lufs, 1),
        "room_tone_lufs": round(masked_lufs(amb, tone_mask, tmp) + g, 2) if tone_mask.any() else None,
        "beds": beds, "effects": effects,
        "longest_gap_s": max([x[1] for x in gap_list] or [0.0]), "gaps": gap_list,
        "longest_dead_air_any_s": _longest_quiet(mix),
        "bed_cuts": bed_cuts, "shortest_ramps_s": ramps,
        "lufs": loud["lufs"], "true_peak": loud["true_peak"], "lra": loud.get("lra"),
        "faults": faults,
    }


def _longest_quiet(x: np.ndarray) -> float:
    """Longest run of 50 ms windows under GAP_DB, however short (0 when none)."""
    quiet = window_db(x, int(0.05 * QA_SR)) < GAP_DB
    best = run = 0
    for q in quiet:
        run = run + 1 if q else 0
        best = max(best, run)
    return round(best * 0.05, 2)


def qa_markdown(plan: dict[str, Any]) -> str:
    """The short sound section of qa_report.md."""
    q = plan["qa"]
    lv = plan["levels_db_rel_narration"]
    lines = ["## Sound", "",
             f"- Music: {len(plan['music'])} sections from {plan['music_source']} ({plan['music_folder']}), "
             f"{len(plan['changes'])} changes, {len(plan['quiets'])} quiet moments.",
             f"- Ambience: {sum(1 for r in plan['ambience'] if r['place'] != 'room_tone')} place beds, "
             f"{sum(1 for r in plan['ambience'] if r['place'] == 'room_tone')} room tone spans; "
             f"effects: {len(plan['effects'])} ({len(plan['effects_skipped'])} skipped).",
             f"- Speech {q['speech_lufs']} LUFS. Music under speech {q['music_under_speech_rel_db']} dB "
             f"(set {lv['music_under_speech_db']}), in the open {q['music_open_rel_db']} dB. Ambience under speech "
             f"{q['ambience_under_speech_rel_db']} dB (set {lv['ambience_db']}).",
             f"- Mix {q['lufs']} LUFS, true peak {q['true_peak']} dBTP, longest dead air {q['longest_dead_air_any_s']} s.",
             f"- Shortest ramps (s): " + ", ".join(f"{k} {v}" for k, v in q["shortest_ramps_s"].items()) + ".",
             ""]
    if q["faults"]:
        lines += ["Faults:", ""] + [f"- {f}" for f in q["faults"]] + [""]
    else:
        lines += ["No sound faults: no gap, no cut, no fast ramp, no bed over its limit, no effect on a word.", ""]
    lines.append("Every music section, change, quiet, bed and effect with its time, file, level and reason: "
                 "work/sound_plan.json.")
    return "\n".join(lines) + "\n"


def qa_section(work_dir: Path) -> str:
    """Sound section for qa_report.md, or "" when the mix was not planned (older projects)."""
    p = Path(work_dir) / "sound_qa.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_plan(work_dir: Path) -> dict[str, Any]:
    p = Path(work_dir) / "sound_plan.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
