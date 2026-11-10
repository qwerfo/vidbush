"""Acceptance checks for a finished project (spec section 10, A1 to A3).

python -m vidbush check <project_id> prints every check and exits non-zero on a failure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .models import PacingProfile, ShotList
from .util import EM_DASH, ffprobe, read_model

REQUIRED_FILES = ["final.mp4", "captions.srt", "captions.ass", "chapters.txt", "description.md",
                  "shotlist.json", "qa_report.json", "qa_report.md", "cost_report.json"]


def _video_checks(final: Path) -> dict[str, Any]:
    info = ffprobe(final)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})
    rate = v.get("r_frame_rate", "0/1").split("/")
    return {
        "duration_s": round(float(info["format"]["duration"]), 2),
        "size_1920x1080": (v.get("width"), v.get("height")) == (1920, 1080),
        "fps_30": abs(float(rate[0]) / float(rate[1] or 1) - 30) < 0.01,
        "h264_high": v.get("codec_name") == "h264" and v.get("profile") == "High",
        "aac": a.get("codec_name") == "aac",
        "aac_192k": abs(int(a.get("bit_rate", 0) or 0) - 192000) < 20000,
    }


def _loudness(final: Path) -> dict[str, float]:
    from .audio.loudness import measure
    m = measure(final)
    return {"lufs": round(float(m.get("lufs", 0)), 2), "true_peak": round(float(m.get("true_peak", 0)), 2)}


def check_project(settings: Settings, project_id: str, min_minutes: float = 3.0) -> dict[str, Any]:
    """Return {"ok": bool, "checks": {name: bool}, "facts": {...}}."""
    root = settings.path(settings.output_folder) / project_id
    checks: dict[str, bool] = {f"file:{f}": (root / f).exists() for f in REQUIRED_FILES}
    thumbs = sorted((root / "thumbnails").glob("*.png"))
    checks["thumbnails_3"] = len(thumbs) >= 3
    facts: dict[str, Any] = {"thumbnails": [p.name for p in thumbs]}
    if not checks["file:final.mp4"] or not checks["file:shotlist.json"]:
        return {"ok": False, "checks": checks, "facts": facts}
    vid = _video_checks(root / "final.mp4")
    facts.update(vid)
    checks["min_duration"] = vid["duration_s"] >= min_minutes * 60
    for k in ("size_1920x1080", "fps_30", "h264_high", "aac", "aac_192k"):
        checks[k] = bool(vid[k])
    loud = _loudness(root / "final.mp4")
    facts.update(loud)
    checks["lufs_-14"] = abs(loud["lufs"] + 14) <= 1.0
    checks["true_peak_-1"] = loud["true_peak"] <= -0.5
    sl = read_model(root / "shotlist.json", ShotList)
    types = {s.type.value for s in sl.shots}
    kinds = {s.motion.kind.value for s in sl.shots if s.motion} | {s.overlay.kind.value for s in sl.shots if s.overlay}
    facts.update({"shots": len(sl.shots), "types": sorted(types), "motion_kinds": sorted(kinds)})
    checks["shots_20"] = len(sl.shots) >= 20
    checks["types_5_of_7"] = len(types) >= 5
    checks["map"] = "map" in kinds
    checks["chart"] = bool({"bar_chart", "line_chart"} & kinds)
    checks["title_card"] = "title_card" in kinds
    checks["every_shot_selected"] = all(s.selected for s in sl.shots)
    checks.update(_animation_checks(sl, facts, _top10(root)))
    checks.update(_text_checks(root))
    return {"ok": all(checks.values()), "checks": checks, "facts": facts}


CARD_SHARE_MAX = 25.0  # percent of shots; above this the video reads as a slideshow of cards
CARD_TIME_MAX = 10.0  # percent of runtime on full screen cards; long cards weigh more than their count
CARD_HOLD_SLACK_S = 0.5  # a static card may hold this much past the profile's card_hold_max (rounding, splits)


def _top10(root: Path) -> bool:
    """A top 10 video (project format or profile) opens every item on a full screen number card by design."""
    def load(p: Path) -> dict[str, Any]:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    snap = load(root / "work" / "profile_snapshot.json").get("profile") or {}
    return load(root / "project.json").get("format") == "top10" or bool(snap.get("top10"))


def _animation_checks(sl: ShotList, facts: dict[str, Any], top10: bool = False) -> dict[str, bool]:
    """Animation moves over the picture: full screen cards stay a minority by count and by runtime (top 10
    number cards are structural, so a top 10 video skips the runtime check), a text or number card holds only
    briefly, and some map or chart rides on footage or a still as an inset."""
    from .judge.qa import animation_share
    anim = animation_share(sl)
    facts["animation"] = anim
    return {"cards_sparing": anim["full_screen_cards"] <= CARD_SHARE_MAX,
            "card_time_sparing": top10 or anim["full_screen_card_time"] <= CARD_TIME_MAX,
            "card_hold_short": anim["max_card_hold_s"] <= PacingProfile().card_hold_max + CARD_HOLD_SLACK_S,
            "overlays_on_picture": anim["overlay_shots"] >= anim["full_screen_cards"],
            "inset_map_or_chart": anim["inset_overlays"] > 0}


def _text_checks(root: Path) -> dict[str, bool]:
    desc = (root / "description.md").read_text(encoding="utf-8") if (root / "description.md").exists() else ""
    chapters = (root / "chapters.txt").read_text(encoding="utf-8").strip().splitlines() if (root / "chapters.txt").exists() else []
    mix = root / "work" / "mix.json"
    mix_data = json.loads(mix.read_text(encoding="utf-8")) if mix.exists() else {}
    srt = (root / "captions.srt").read_text(encoding="utf-8") if (root / "captions.srt").exists() else ""
    return {
        "chapters_3": len(chapters) >= 3 and chapters[0].startswith("00:00"),
        "description_attribution": "credit" in desc.lower() or "attribution" in desc.lower(),
        "music_ducking": bool(mix_data.get("ducking") or mix_data.get("tracks")),
        "captions_present": srt.count("-->") >= 20,
        "no_em_dash": EM_DASH not in desc + srt + "\n".join(chapters),
    }
