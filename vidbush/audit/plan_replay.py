"""Plan replay: measure a stored shot plan against its narration, without rendering anything.

    python -m vidbush plan-replay <project id or dir> [--shotlist PATH] [--json OUT]

Reads work/narration.json, work/director_plan.json and a shot list (default work/shotlist.shotlist.json
when it exists, else work/shotlist.json) and reports:

- anchor: picture shots (every shot but motion graphics) whose anchor words (shots.ideas.idea_anchor_time
  of the shot's own idea) fall inside the shot window, and how many show their picture early (the words
  come after the shot) or late (the words came before it);
- cuts per minute and the median shot length;
- screen time by kind: real footage, real still, AI, designed or motion graphic (judge.qa.screen_time);
- the narration units no picture shot overlaps: clauses when work/clauses.json exists (a list of
  {id, start, end}, or {"clauses": [...]}), else beats.

Nothing is written into the project folder; --json writes the report where it is told.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional

from ..judge.qa import screen_time
from ..models import ShotList, ShotType, VisualIdea, Word
from ..shots.ideas import idea_anchor_time

PICTURE_FAMILIES = ("footage", "still")
KINDS = {"real_footage": ("real_footage",), "real_still": ("real_still",),
         "ai": ("generated_still", "generated_video"), "designed_or_motion": ("designed", "graphic")}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def work_dir(path: Path) -> Path:
    """The work folder of a project folder, or the folder itself when it already holds narration.json."""
    return path if (path / "narration.json").exists() else path / "work"


def default_shotlist(work: Path) -> Path:
    snap = work / "shotlist.shotlist.json"
    return snap if snap.exists() else work / "shotlist.json"


def _units(work: Path, spans: dict[str, tuple[float, float]]) -> tuple[str, dict[str, tuple[float, float]]]:
    path = work / "clauses.json"
    if path.exists():
        data = _read(path)
        rows = data.get("clauses", []) if isinstance(data, dict) else data
        return "clause", {str(r.get("id", i)): (float(r["start"]), float(r["end"])) for i, r in enumerate(rows)}
    return "beat", spans


def _anchor_offset(shot: Any, ideas: list[VisualIdea], words: list[Word],
                   spans: dict[str, tuple[float, float]]) -> Optional[float]:
    """Seconds the shot's anchor words lie outside its window (0 inside, > 0 later, < 0 earlier); None
    when the shot has no idea or the idea's words are never spoken."""
    idea = next((i for i in ideas if i.intent == shot.intent and i.family in PICTURE_FAMILIES), None)
    t = idea_anchor_time(idea, words, spans.get(idea.beat_id, (0.0, 0.0))) if idea else None
    if t is None:
        return None
    return 0.0 if shot.start <= t <= shot.end else (t - shot.end if t > shot.end else t - shot.start)


def plan_report(work: Path, shotlist_path: Optional[Path] = None) -> dict[str, Any]:
    """The plan measures as a dict (see the module docstring); work may be the project folder."""
    work = work_dir(Path(work))
    shotlist_path = Path(shotlist_path) if shotlist_path else default_shotlist(work)
    nar = _read(work / "narration.json")
    words = [Word(**w) for w in nar.get("words") or []]
    spans = {k: (float(v[0]), float(v[1])) for k, v in (nar.get("beat_spans") or {}).items()}
    ideas = [VisualIdea.model_validate(i) for i in _read(work / "director_plan.json").get("ideas") or []]
    sl = ShotList.model_validate(_read(shotlist_path))
    pictures = [s for s in sl.shots if s.type != ShotType.motion_graphic]
    offsets = [_anchor_offset(s, ideas, words, spans) for s in pictures]
    timed = [o for o in offsets if o is not None]
    inside = sum(1 for o in timed if o == 0)
    durations = [s.end - s.start for s in sl.shots]
    total = max([s.end for s in sl.shots] + [float(nar.get("duration") or 0.0), 0.01])
    pct = screen_time(sl)[0]
    unit, units = _units(work, spans)
    uncovered = [uid for uid, (a, b) in units.items()
                 if not any(min(b, s.end) - max(a, s.start) > 1e-6 for s in pictures)]
    return {
        "shotlist": str(shotlist_path),
        "picture_shots": len(pictures),
        "anchor": {"inside": inside, "early": sum(1 for o in timed if o > 0), "late": sum(1 for o in timed if o < 0),
                   "no_anchor": len(offsets) - len(timed),
                   "inside_pct": round(100.0 * inside / len(pictures), 1) if pictures else 0.0},
        "shots": len(sl.shots),
        "cuts_per_minute": round(max(len(sl.shots) - 1, 0) / (total / 60.0), 1),
        "median_shot_s": round(statistics.median(durations), 2) if durations else 0.0,
        "screen_time_pct": {k: round(sum(pct.get(x, 0.0) for x in xs), 1) for k, xs in KINDS.items()},
        "unit": unit, "units": len(units), "uncovered": uncovered,
    }


def report_text(r: dict[str, Any]) -> str:
    a, st = r["anchor"], r["screen_time_pct"]
    return "\n".join([
        f"shot list: {r['shotlist']}",
        f"anchor words inside their own shot window: {a['inside']} of {r['picture_shots']} picture shots "
        f"({a['inside_pct']:g}%); picture early {a['early']}, late {a['late']}, no anchor {a['no_anchor']}",
        f"shots {r['shots']}, cuts per minute {r['cuts_per_minute']:g}, median shot {r['median_shot_s']:g} s",
        f"screen time: real footage {st['real_footage']:g}%, real still {st['real_still']:g}%, AI {st['ai']:g}%, "
        f"designed or motion graphic {st['designed_or_motion']:g}%",
        f"{r['unit']}s with no picture shot: {len(r['uncovered'])} of {r['units']}"
        + (f" ({', '.join(r['uncovered'])})" if r["uncovered"] else ""),
    ])


def run(path: Path, shotlist: Optional[Path] = None, json_out: Optional[Path] = None) -> int:
    r = plan_report(path, shotlist)
    print(report_text(r))
    if json_out:
        Path(json_out).write_text(json.dumps(r, indent=2), encoding="utf-8")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vidbush.audit.plan_replay", description=__doc__.split("\n\n")[0])
    p.add_argument("project", type=Path, help="project folder (or its work folder)")
    p.add_argument("--shotlist", type=Path, help="shot list to measure (default: the latest stored one)")
    p.add_argument("--json", type=Path, help="write the report here")
    a = p.parse_args(argv)
    return run(a.project, a.shotlist, a.json)


if __name__ == "__main__":
    raise SystemExit(main())
