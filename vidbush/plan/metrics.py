"""Plan replay numbers, old planner against new, on the same stored narration (no render needed)."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional

from ..models import Narration, ShotList, ShotType, VisualIdea
from . import anchors as A
from .clauses import Clause


def _longest_ai_run(sl: ShotList, lo: float, hi: float) -> int:
    best = run = 0
    for s in sl.shots:
        if s.end <= lo or s.start >= hi:
            continue
        run = run + 1 if s.type == ShotType.ai_image else 0
        best = max(best, run)
    return best


def _anchor_stats(pairs: list[tuple[float, float, Optional[float]]]) -> dict:
    """pairs of (shot start, shot end, anchor time)."""
    n = len(pairs)
    inside = [(s, e, t) for s, e, t in pairs if t is not None and s - 1e-6 <= t <= e + 1e-6]
    early = sum(1 for s, e, t in pairs if t is not None and t > e)
    late = sum(1 for s, e, t in pairs if t is not None and t < s)
    leads = [t - s for s, e, t in inside]
    return {"shots": n, "anchor_inside": len(inside), "anchor_inside_pct": round(100.0 * len(inside) / n, 1) if n else 0.0,
            "picture_early": early, "picture_late": late, "no_anchor_time": sum(1 for p in pairs if p[2] is None),
            "median_lead_s": round(statistics.median(leads), 2) if leads else None,
            "lead_over_1_5s": sum(1 for x in leads if x > 1.5)}


def new_anchors(plan: dict, sl: ShotList, narration: Narration, clauses: list[Clause]) -> list:
    by_id = {c.id: c for c in clauses}
    pairs = []
    for sc in plan["scenes"]:
        for row in sc["shots"]:
            if row.get("note"):  # held into the previous shot: that shot is measured by its own head
                continue
            gc = [by_id[c] for c in row["clauses"]]
            shot = next(s for s in sl.shots if s.start <= gc[0].start + 1e-6 < s.end)
            words = narration.words[gc[0].words[0]:gc[-1].words[1] + 1]
            t = A.time_of(words, str(row.get("anchor", ""))) if row.get("treatment") != "hold" else None
            pairs.append((shot.start, shot.end, t))
    return pairs


def old_anchors(project: Path, old: ShotList, narration: Narration, lo: float, hi: float) -> list:
    """The fable_review anchor_offset measure: each stored shot's idea anchor against the shot window."""
    from ..shots import ideas as I

    path = project / "work" / "director_plan.json"
    if not path.exists():
        return []
    ideas = [VisualIdea.model_validate(i) for i in json.loads(path.read_text(encoding="utf-8"))["ideas"]]
    spans = {k: tuple(v) for k, v in narration.beat_spans.items()}
    pairs = []
    for s in old.shots:
        if s.type == ShotType.motion_graphic or s.end <= lo or s.start >= hi:
            continue
        fam = "footage" if s.type in (ShotType.footage, ShotType.footage_overlay) else "still"
        m = [i for i in ideas if i.intent == s.intent and i.family == fam] or [i for i in ideas if i.intent == s.intent]
        t = I.idea_anchor_time(m[0], narration.words, spans.get(m[0].beat_id, (0, 0))) if m else None
        pairs.append((s.start, s.end, t))
    return pairs


def _span_shots(sl: ShotList, lo: float, hi: float) -> list:
    return [s for s in sl.shots if s.start < hi and s.end > lo]


def report(clauses: list[Clause], plan: dict, sl: ShotList, narration: Narration, old: ShotList,
           only: Optional[set[str]] = None, project: Optional[Path] = None) -> dict:
    cl = [c for c in clauses if not only or c.scene in only]
    lo, hi = (cl[0].start, cl[-1].end) if cl else (0.0, 0.0)
    minutes = sum(c.duration for c in cl) / 60.0
    rows = [r for sc in plan["scenes"] for r in sc["shots"]]
    treat = Counter()
    for r in rows:
        treat[r["treatment"]] += 1
        treat["hold"] += len(r["clauses"]) - 1
    pres = Counter(r.get("presentation") for r in rows if r.get("presentation"))
    grammar = Counter(g for r in rows for g in r.get("grammar", []))
    nothing = [{"clause": r["clauses"][0], "subject": r.get("subject", ""), "reason": r["edit"].get("reason", ""),
                "fallback": r["edit"].get("fallback", ""), "resolved": r["edit"].get("resolved", "")}
               for r in rows if r.get("edit") and r["edit"]["pick"] is None
               and (r["treatment"] == "picture" or r["treatment"] in ("number_over_footage", "date_title", "document_focus")
                    or r.get("presentation") in ("panel", "full_then_dock", "panel_then_expand"))]
    pictures = [r for r in rows if r["treatment"] == "picture"]
    unused = [r for r in pictures if r.get("edit") and r["edit"]["pick"] is None]
    new_shots = _span_shots(sl, lo, hi)
    old_shots = _span_shots(old, lo, hi)
    ai_picks = [r for r in rows if r.get("edit", {}).get("ai")]
    out = {
        "scope": sorted(only) if only else "all",
        "minutes": round(minutes, 2),
        "clauses": len(cl),
        "new": {
            "shots": len(new_shots), "cuts_per_minute": round((len(new_shots) - 1) / minutes, 2) if minutes else 0,
            "median_shot_s": round(statistics.median(s.duration for s in new_shots), 2) if new_shots else 0,
            "anchors": _anchor_stats(new_anchors(plan, sl, narration, cl)),
            "treatments": dict(treat), "presentations": dict(pres), "grammar": dict(grammar),
            "types": dict(Counter(s.type.value for s in new_shots)),
            "ai_picks": len(ai_picks), "longest_ai_run": _longest_ai_run(sl, lo, hi),
            "nothing_fits": nothing, "director_pictures": len(pictures), "director_pictures_unused": len(unused),
        },
        "old": {
            "shots": len(old_shots), "cuts_per_minute": round((len(old_shots) - 1) / minutes, 2) if minutes else 0,
            "median_shot_s": round(statistics.median(s.duration for s in old_shots), 2) if old_shots else 0,
            "anchors": _anchor_stats(old_anchors(project, old, narration, lo, hi)) if project else {},
            "types": dict(Counter(s.type.value for s in old_shots)),
            "longest_ai_run": _longest_ai_run(old, lo, hi),
        },
    }
    return out
