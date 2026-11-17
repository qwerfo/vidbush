"""Pure tile audit logic: word timings, tile windows, shot lookup, the six numbers, worst stretches, comparison.

Nothing here calls a model or ffmpeg, so every function is tested offline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

KINDS = ["real_footage", "real_still", "document", "ai_image", "ai_video", "map", "chart", "text_card",
         "designed_card", "other"]
MATCHES = ["on_words", "early", "late", "related", "generic", "wrong"]
REAL_KINDS = {"real_footage", "real_still", "document"}
AI_KINDS = {"ai_image", "ai_video"}  # FABLE_REVIEW 3: "AI images, and video that looks AI-made"
AI_SOURCES = {"ai_image", "ai_video"}  # candidate sources of generated pictures in a shot list
DESIGNED_KINDS = {"map", "chart", "text_card", "designed_card"}
SUMMARY_KEYS = ["real_share", "ai_share", "designed_share", "on_words", "noticeable", "overlay_noticeable"]
# Vidrush render of the Alaska script, FABLE_REVIEW section 3, and the section 6.2 targets
BENCHMARK = {"real_share": 55, "ai_share": 2, "designed_share": 42, "on_words": 64, "noticeable": 19,
             "overlay_noticeable": 21}
TARGETS = {"ai_share": "<= 5", "on_words": ">= 64", "noticeable": "<= 19", "overlay_noticeable": "< 15"}
LABELS = {"real_share": "Real share (footage, stills, documents)", "ai_share": "AI share",
          "designed_share": "Designed share (maps, charts, cards)", "on_words": "Picture on_words",
          "noticeable": "Noticeable", "overlay_noticeable": "Overlay tiles noticeable"}
CONTEXT_S = 10.0  # "line before / line after": speech this long either side of the tile window


Word = tuple[str, float, float]  # text, start s, end s


def load_words(path: Path | str) -> list[Word]:
    """Word timings in seconds from a project dir, narration.json (words[text,start,end] in seconds) or a
    transcript JSON (words[word,startMs,endMs] in milliseconds)."""
    p = Path(path)
    if p.is_dir():
        p = p / "work" / "narration.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("words", []) if isinstance(data, dict) else data
    out: list[Word] = []
    for w in raw:
        if "startMs" in w:
            out.append((str(w.get("word", w.get("text", ""))).strip(), w["startMs"] / 1000.0, w["endMs"] / 1000.0))
        else:
            out.append((str(w.get("text", w.get("word", ""))).strip(), float(w["start"]), float(w["end"])))
    return sorted((w for w in out if w[0]), key=lambda w: w[1])


def tile_times(duration: float, every: float = 2.0) -> list[float]:
    """Tile starts 0, every, 2*every ... whose frame (t + every/2) is inside the video."""
    out, t = [], 0.0
    while t + every / 2 < duration:
        out.append(round(t, 3))
        t += every
    return out


def _join(ws: Iterable[Word]) -> str:
    return " ".join(w[0] for w in ws)


def tile_words(words: list[Word], t: float, every: float = 2.0) -> dict[str, str]:
    """Words overlapping [t - every/2, t + every] (now), and the speech CONTEXT_S before and after it."""
    a, b = t - every / 2, t + every
    now = [w for w in words if w[1] < b and w[2] > a]
    before = [w for w in words if w[2] <= a and w[1] >= a - CONTEXT_S]
    after = [w for w in words if w[1] >= b and w[1] < b + CONTEXT_S]
    return {"words": _join(now), "before": _join(before), "after": _join(after)}


def load_shots(root: Optional[Path]) -> list[dict[str, Any]]:
    """Shots (id, type, start, end, source of the selected candidate) from <root>/work/shotlist.json or
    <root>/shotlist.json; [] when absent."""
    if root is None:
        return []
    for p in (root / "work" / "shotlist.json", root / "shotlist.json"):
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            shots = d.get("shots", []) if isinstance(d, dict) else d
            out = []
            for s in shots:
                sel = next((c for c in s.get("candidates") or [] if c.get("id") == s.get("selected")), {})
                out.append({"id": s.get("id"), "type": s.get("type"), "start": float(s.get("start", 0)),
                            "end": float(s.get("end", 0)), "source": sel.get("source", "")})
            return out
    return []


def shift(items: list[Any], offset: float) -> list[Any]:
    """Word tuples or shot dicts moved later by offset seconds (an intro before the narration in final.mp4)."""
    if not offset:
        return items
    return [(w[0], w[1] + offset, w[2] + offset) if isinstance(w, tuple) else
            {**w, "start": w["start"] + offset, "end": w["end"] + offset} for w in items]


def intro_offset(root: Optional[Path]) -> float:
    """Seconds of intro before narration time 0 in final.mp4, from work/assembly.json (0 when absent)."""
    try:
        return float(json.loads((root / "work" / "assembly.json").read_text(encoding="utf-8"))
                     .get("intro_duration", 0.0) or 0.0) if root else 0.0
    except (OSError, ValueError, AttributeError):
        return 0.0


def frame_name(t: float) -> str:
    """Frame file of tile t, in milliseconds so tiles under a second apart never collide."""
    return f"t{round(t * 1000):08d}.jpg"


def shot_at(shots: list[dict[str, Any]], t: float) -> Optional[dict[str, Any]]:
    for s in shots:
        if s["start"] <= t < s["end"]:
            return s
    return None


def spread(items: list[Any], n: int) -> list[Any]:
    """n items evenly spaced through the list (all of them when n >= len)."""
    if n <= 0 or n >= len(items):
        return list(items)
    return [items[int(i * len(items) / n)] for i in range(n)]


def _valid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r.get("error") and r.get("match") in MATCHES]


def _pct(n: int, d: int) -> Optional[float]:
    """Share in percent; None for an empty denominator (0.0 would read as meeting a target)."""
    return round(100.0 * n / d, 1) if d else None


def has_overlay(r: dict[str, Any]) -> bool:
    return bool(str(r.get("overlay") or "").strip())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The six headline numbers as percentages 0 to 100 (1 decimal), over tiles with a valid verdict."""
    v = _valid(rows)
    n = len(v)
    ov = [r for r in v if has_overlay(r)]
    return {"real_share": _pct(sum(r.get("kind") in REAL_KINDS for r in v), n),
            "ai_share": _pct(sum(r.get("kind") in AI_KINDS for r in v), n),
            "designed_share": _pct(sum(r.get("kind") in DESIGNED_KINDS for r in v), n),
            "on_words": _pct(sum(r.get("match") == "on_words" for r in v), n),
            "noticeable": _pct(sum(bool(r.get("noticeable")) for r in v), n),
            "overlay_noticeable": _pct(sum(bool(r.get("noticeable")) for r in ov), len(ov)),
            "tiles": n, "overlay_tiles": len(ov)}


def mmss(t: float) -> str:
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


def worst_stretches(rows: list[dict[str, Any]], every: float = 2.0, top: int = 10) -> list[dict[str, Any]]:
    """Runs of consecutive noticeable tiles, longest first (earlier first on ties)."""
    runs: list[list[dict[str, Any]]] = []
    for r in sorted(rows, key=lambda r: r["t"]):
        if not r.get("noticeable"):
            continue
        if runs and abs(r["t"] - runs[-1][-1]["t"] - every) < 1e-6:
            runs[-1].append(r)
        else:
            runs.append([r])
    runs.sort(key=lambda run: (-len(run), run[0]["t"]))
    out = []
    for run in runs[:top]:
        defects = list(dict.fromkeys(str(r.get("defect") or "").strip() for r in run if r.get("defect")))
        out.append({"start": run[0]["t"], "end": run[-1]["t"] + every, "tiles": len(run),
                    "range": f"{mmss(run[0]['t'])}-{mmss(run[-1]['t'] + every)}", "defects": defects})
    return out


def latest_rows(path: Path) -> list[dict[str, Any]]:
    """The last row per tile of an append-only audit.jsonl, in time order ([] when absent)."""
    if not path.exists():
        return []
    rows: dict[Any, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue  # a line torn by a crash mid-write
        rows[r["t"]] = r
    return sorted(rows.values(), key=lambda r: r["t"])


def read_jsonl(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def compare(audit: list[dict[str, Any]], ref: list[dict[str, Any]]) -> dict[str, Any]:
    """Six numbers of both sets on the tiles they share (joined on t), differences, per-tile agreement."""
    by_t = {r["t"]: r for r in _valid(ref)}
    pairs = [(a, by_t[a["t"]]) for a in _valid(audit) if a["t"] in by_t]
    mine = summarize([a for a, _ in pairs])
    theirs = summarize([b for _, b in pairs])
    n = len(pairs)

    def group(k: str) -> str:
        return "real" if k in REAL_KINDS else "ai" if k in AI_KINDS else "designed" if k in DESIGNED_KINDS else "other"

    agree = {"tiles": n,
             "match": _pct(sum(a["match"] == b["match"] for a, b in pairs), n),
             "on_words": _pct(sum((a["match"] == "on_words") == (b["match"] == "on_words") for a, b in pairs), n),
             "noticeable": _pct(sum(bool(a.get("noticeable")) == bool(b.get("noticeable")) for a, b in pairs), n),
             "kind_group": _pct(sum(group(a.get("kind", "")) == group(b.get("kind", "")) for a, b in pairs), n),
             "noticeable_both": sum(bool(a.get("noticeable")) and bool(b.get("noticeable")) for a, b in pairs),
             "noticeable_audit_only": sum(bool(a.get("noticeable")) and not b.get("noticeable") for a, b in pairs),
             "noticeable_ref_only": sum(not a.get("noticeable") and bool(b.get("noticeable")) for a, b in pairs)}
    return {"audit": mine, "reference": theirs,
            "diff": {k: None if mine[k] is None or theirs[k] is None else round(mine[k] - theirs[k], 1)
                     for k in SUMMARY_KEYS}, "agreement": agree}
