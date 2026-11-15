"""Regenerate chosen AI shots of a finished project into a scratch copy, with the current AI still code.

python scripts/regen_ai_shots.py --project <output dir> --out <scratch dir> --shots shot_068,shot_087 [--review r.json]

The project is read only: script, narration, project and shot list are copied to <out>, a stage context is
built on the copy (logs, costs and images go to <out>), and each shot runs director_loop.add_ai (scene
brief, prompt, image, judge, fixed-question check, regeneration with the failing reasons). With --review,
the shots are the AI-area defects of a run 2 style frame review. Writes <out>/regen.json: per shot every
take with its prompt, brief, judge score and reason, check answers, and whether the picked take passes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's vidbush, not an installed one


def review_ai_defects(review: dict[str, Any]) -> list[str]:
    """Shot ids the review's AI still area lists as defective."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qa_calibration import FINE, SHOT_NUM, STYLE_NOTE
    out: list[str] = []
    for area in review.get("reviews", []):
        if not str(area.get("area", "")).lower().startswith("ai"):
            continue
        for d in area.get("defects", []):
            field, problem = str(d.get("shot", "")), str(d.get("problem", ""))
            if FINE.search(field) or FINE.search(problem) or STYLE_NOTE.search(problem):
                continue
            out += [f"shot_{n}" for n in SHOT_NUM.findall(field)]
    return sorted(dict.fromkeys(out))


def scratch_context(project: Path, out: Path, settings_root: str = "") -> Any:
    from vidbush.config import REPO_ROOT, load_settings
    from vidbush.jobs.context import build_context
    from vidbush.models import Project
    from vidbush.util import read_model, write_json
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("script.json", "narration.json", "script_summary.json"):
        if (project / "work" / name).exists():
            shutil.copy2(project / "work" / name, work / name)
    proj = read_model(project / "project.json", Project)
    proj.id = f"{proj.id}_regen"
    proj.paths.root, proj.paths.work = str(out), str(work)
    proj.paths.sources, proj.paths.segments = str(out / "sources"), str(work / "segments")
    proj.paths.thumbnails = str(out / "thumbnails")
    write_json(out / "project.json", proj)
    settings = load_settings(Path(settings_root) if settings_root else REPO_ROOT,
                             overrides={"data_folder": str(out / "data")})
    ctx = build_context(settings, proj, echo=False)
    ctx.stage = "judge"
    return ctx


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", default="")
    ap.add_argument("--review", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--settings-root", default="", help="checkout with .env and config (default: this one)")
    a = ap.parse_args(argv)
    import vidbush
    from concurrent.futures import ThreadPoolExecutor
    from vidbush.judge.director_loop import add_ai
    from vidbush.models import ShotList, SourceKind
    from vidbush.util import read_model
    print(f"vidbush from {vidbush.__file__}", file=sys.stderr)
    project, out = Path(a.project), Path(a.out)
    ids = [x for x in a.shots.split(",") if x]
    if a.review:
        ids += review_ai_defects(json.loads(Path(a.review).read_text(encoding="utf-8")))
    sl = read_model(project / "shotlist.json", ShotList)
    shots = [s.model_copy(deep=True) for s in sl.shots if s.id in set(ids)]
    ctx = scratch_context(project, out, a.settings_root)
    threshold = float(ctx.settings.judge.threshold)

    def one(shot: Any) -> dict[str, Any]:
        old = shot.selected_candidate()
        shot.candidates = []
        try:
            best = add_ai(shot, ctx)
        except Exception as e:  # noqa: BLE001  report and go on
            return {"shot": shot.id, "error": str(e)[:300]}
        takes = [c for c in shot.candidates if c.source == SourceKind.ai_image]
        return {"shot": shot.id, "start": shot.start, "end": shot.end, "intent": shot.intent,
                "old": old.local_path if old else "", "flags": shot.flags, "best": best.id,
                "best_path": best.local_path,
                "passes": best.reject_reason is None and (best.score or 0) >= threshold,
                "takes": [{"id": c.id, "path": c.local_path, "score": c.score, "judge": c.judge_reason,
                           "reject": c.reject_reason, "prompt": c.prompt, "brief": c.brief, "checks": c.checks}
                          for c in takes]}

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        results = list(pool.map(one, shots))
    (out / "regen.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    passed = sum(1 for r in results if r.get("passes"))
    print(json.dumps({"shots": len(results), "pass_checker": passed,
                      "errors": [r["shot"] for r in results if "error" in r],
                      "cost_usd": round(ctx.tracker.report.total_usd, 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
