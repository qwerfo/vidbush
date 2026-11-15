"""Calibrate the rendered-frame QA against a human or agent frame review.

python scripts/qa_calibration.py --project <output dir> --review <review.json> --out <scratch dir> [--tag t]

Runs vidbush.judge.qa's frame check over <project>/final.mp4 (read only: frames, verdicts, costs and logs go
to --out) and compares it shot by shot with the review: recall is the share of review-defective shots QA
fails, the false alarm rate is the share of review-passed shots QA fails. The review JSON is the run 2 frame
review shape: {"reviews": [{"area", "defects": [{"time", "shot", "problem", "cause_guess"}]}], "synth"}.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's vidbush, not an installed one

SHOT_NUM = re.compile(r"(?<![\d.:])(\d{3})(?![\d.:])")
# entries the review marks as fine, and style notes that are not a defect of one shot
FINE = re.compile(r"\bOK\b|^\s*FINE\b")
STYLE_NOTE = re.compile(r"^\s*every callout is cent", re.I)
# the synthesizer's own corrections: shots it re-checked and found a loose fit, not a defect
CORRECTION = re.compile(r"shot_(\d{3})\b[^\n]{0,400}?\bloose fit, not (?:a )?clear mismatch", re.I)
# Shots the run 2 review names as frame defects only in its root-cause evidence (AI area: "Heads or faces cut
# in 048, 064, 066, 071, 076, 096 and 104", "Diptychs or triptychs in 001, 088, 096, 099, 107, 110, 115 (092
# intended triptych)", "104 and 030 trees on tundra coast"), not in its defect list. Evidence of causes about
# sourcing or about QA itself names shots that are not frame defects and is left out.
EVIDENCE_DEFECTS = {"shot_064": "heads cut by the letterbox", "shot_076": "heads cut by the letterbox",
                    "shot_096": "heads cut, split panel", "shot_099": "split panel", "shot_107": "split panel",
                    "shot_030": "trees on a tundra coast"}
# Risk notes the review says it did not count (an AI face standing in for a named elder, identifiable children):
# neither a defect nor a pass, so they leave the comparison.
NEUTRAL = {"shot_018", "shot_028", "shot_029"}  # 1:52 to 1:58 spans shots 028 and 029


def review_labels(review: dict[str, Any]) -> tuple[dict[str, list[str]], set[str]]:
    """(shot id -> defect areas, shot ids the review notes as style only). A defect entry names its shots in
    its "shot" field (shot_034, 'shots 034, 035', 'callouts 064, 083'); entries marked OK or FINE are passes,
    and a shot the synthesizer corrected to a loose fit is not a defect."""
    defects: dict[str, list[str]] = {}
    style: set[str] = set()
    synth = review.get("synth", {})
    notes = synth.get("notes", "") if isinstance(synth, dict) else ""
    corrected = {f"shot_{m}" for m in CORRECTION.findall(notes if isinstance(notes, str) else json.dumps(notes))}
    for area in review.get("reviews", []):
        name = str(area.get("area", "")).split("(")[0].strip()[:40]
        for d in area.get("defects", []):
            field, problem = str(d.get("shot", "")), str(d.get("problem", ""))
            if FINE.search(field) or FINE.search(problem):
                continue
            ids = {f"shot_{n}" for n in SHOT_NUM.findall(field)}
            if STYLE_NOTE.search(problem):
                style |= ids
                continue
            for sid in ids - corrected:
                defects.setdefault(sid, [])
                if name not in defects[sid]:
                    defects[sid].append(name)
    if review.get("reviews"):
        for sid, why in EVIDENCE_DEFECTS.items():
            defects.setdefault(sid, []).append(f"root cause evidence: {why}")
    return defects, style


def compare(fails: set[str], defects: set[str], universe: set[str]) -> dict[str, Any]:
    """Recall over review defects and false alarms over review passes, both within universe."""
    d, p = defects & universe, universe - defects
    hit, false = fails & d, fails & p
    return {"defects": len(d), "passes": len(p), "found": len(hit), "false_alarms": len(false),
            "recall_pct": round(100.0 * len(hit) / len(d), 1) if d else 0.0,
            "false_alarm_pct": round(100.0 * len(false) / len(p), 1) if p else 0.0,
            "missed": sorted(d - fails), "false": sorted(false)}


def _ctx(project: Path, out: Path) -> Any:
    """A read-only stand-in for a stage context: loads work/*.json of the project, writes nothing there."""
    from vidbush.config import REPO_ROOT, load_settings
    from vidbush.providers.base import CostTracker
    from vidbush.providers.seats import Seats
    from vidbush.util import JsonLogger, read_model
    from vidbush import models as M
    main = Path(r"C:\projects\vidbush_tmp")  # the operator checkout holds .env and .env.local
    settings = load_settings(main if (main / ".env").exists() else REPO_ROOT)
    log = JsonLogger("qa_calibration", out / "logs", echo=False)
    tracker = CostTracker("qa_calibration", out / "cost_report.json")
    seats = Seats(settings, tracker, log)
    names = {"narration": "narration.json", "script": "script.json"}

    def maybe_load(key: str, model: Any) -> Optional[Any]:
        path = project / "work" / names.get(key, f"{key}.json")
        return read_model(path, model) if path.exists() else None

    return SimpleNamespace(settings=settings, seats=seats, log=log, tracker=tracker, stage="packaging",
                           maybe_load=maybe_load, M=M)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--reuse-frames", action="store_true")
    ap.add_argument("--reuse-verdicts", action="store_true", help="recompute fails from saved verdicts only")
    ap.add_argument("--min-score", type=float, default=None, help="score below this fails (default threshold)")
    ap.add_argument("--ignore", default="", help="comma separated checks that do not fail a shot")
    a = ap.parse_args(argv)
    import vidbush
    from vidbush.judge import qa
    from vidbush.models import Narration, ShotList
    print(f"vidbush from {vidbush.__file__}", file=sys.stderr)
    from vidbush.util import read_model
    project, out = Path(a.project), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sl = read_model(project / "shotlist.json", ShotList)
    verdict_path = out / f"verdicts_{a.tag}.json"
    if a.reuse_verdicts and verdict_path.exists():
        verdicts = json.loads(verdict_path.read_text(encoding="utf-8"))
    else:
        ctx = _ctx(project, out)
        frames_dir = out / "frames"
        if a.reuse_frames and frames_dir.exists():
            frames: dict[str, list[str]] = {}
            for f in sorted(frames_dir.glob("f_*.jpg")):
                sid = "_".join(f.stem.split("_")[1:3])
                frames.setdefault(sid, []).append(str(f))
        else:
            frames = qa.sample_frames(project / "final.mp4", sl, frames_dir)
        words = read_model(project / "work" / "narration.json", Narration).words
        verdicts = qa.judge_rendered(sl, frames, ctx, words)
        verdict_path.write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
        print(f"cost so far: {ctx.tracker.report.total_usd:.4f} USD", file=sys.stderr)
    threshold = a.min_score if a.min_score is not None else qa.SCORE_FLOOR
    ignore = {x for x in a.ignore.split(",") if x}
    fails = {sid for sid, v in verdicts.items()
             if set(v["failed"]) - ignore or v["rejects"] or v["score"] < threshold}
    review = json.loads(Path(a.review).read_text(encoding="utf-8"))
    defects, style = review_labels(review)
    universe = {s.id for s in sl.shots if s.id in verdicts} - NEUTRAL
    res = compare(fails, set(defects), universe)
    by_check = Counter(c for sid in fails for c in verdicts[sid]["failed"])
    res.update({"tag": a.tag, "qa_fails": len(fails & universe), "shots": len(universe), "style_only": sorted(style),
                "fail_by_check": dict(by_check),
                "missed_detail": {sid: {"areas": defects[sid], "score": verdicts[sid]["score"],
                                        "qa": verdicts[sid]["reason"][:160]} for sid in res["missed"]},
                "false_detail": {sid: {"score": verdicts[sid]["score"], "qa": verdicts[sid]["reason"][:160]}
                                 for sid in res["false"]}})
    (out / f"calibration_{a.tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("tag", "shots", "defects", "passes", "found", "recall_pct",
                                          "false_alarms", "false_alarm_pct", "qa_fails", "fail_by_check")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
