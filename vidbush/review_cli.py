"""Review subcommands (vidbush/review/): every action of the Review page, plus export and calibration.

review-set <id> <shot> --verdict V --comment TEXT --reject R ...   one shot's review
review-comment <id> TEXT                                           the project comment
review-summary <id>                                                progress, verdict counts, hard rejects
review-export <id>                                                 write output/<id>/review.md
review-fixes <id> [--apply]                                        planned EditOps from the verdicts
review-calibrate <id>                                              judge scores against the verdicts
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import Settings
from .review.store import REJECTS, VERDICTS


def add_review_parsers(sub: Any) -> None:
    st = sub.add_parser("review-set", help="set one shot's review: verdict, comment, hard rejects")
    st.add_argument("project")
    st.add_argument("shot", help="shot id or 1-based shot number")
    st.add_argument("--verdict", choices=[*VERDICTS, "none"], help="none clears the verdict")
    st.add_argument("--comment", help="free text, no length limit; use - to read it from stdin")
    st.add_argument("--reject", action="append", choices=REJECTS, help="hard reject (repeat for several)")
    st.add_argument("--clear-rejects", action="store_true", help="untick every hard reject")
    pc = sub.add_parser("review-comment", help="set the project-level review comment")
    pc.add_argument("project")
    pc.add_argument("text", help="free text, no length limit; use - to read it from stdin")
    for name, help_text in (("review-summary", "print review progress, verdict counts and hard rejects"),
                            ("review-export", "write output/<id>/review.md for the next tuning session"),
                            ("review-calibrate", "compare judge scores with the review verdicts")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("project")
        sp.add_argument("--json", action="store_true", help="print JSON instead of text")
    fx = sub.add_parser("review-fixes", help="show (or apply) the EditOps the review verdicts map to")
    fx.add_argument("project")
    fx.add_argument("--apply", action="store_true", help="apply the planned ops and re-render")
    fx.add_argument("--json", action="store_true")
    fx.add_argument("--quiet", action="store_true")


def _text(value: str) -> str:
    return sys.stdin.read() if value == "-" else value


def _guard(fn: Any) -> Any:
    def wrapped(settings: Settings, args: argparse.Namespace) -> int:
        from .jobs.pipeline import project_dir
        from .review.store import ReviewError
        if not (project_dir(settings, args.project) / "project.json").exists():
            print(f"project not found: {args.project}", file=sys.stderr)
            return 2
        try:
            return fn(settings, args)
        except ReviewError as e:
            print(str(e), file=sys.stderr)
            return 2
    return wrapped


@_guard
def cmd_review_set(settings: Settings, args: argparse.Namespace) -> int:
    from .review import store
    kw: dict[str, Any] = {}
    if args.verdict is not None:
        kw["verdict"] = None if args.verdict == "none" else args.verdict
    if args.comment is not None:
        kw["comment"] = _text(args.comment)
    if args.clear_rejects:
        kw["rejects"] = []
    if args.reject:
        kw["rejects"] = args.reject
    if not kw:
        print("nothing to change: pass --verdict, --comment, --reject or --clear-rejects", file=sys.stderr)
        return 2
    review, e = store.update_shot(settings, args.project, args.shot, **kw)
    print(f"shot {e.index + 1} ({e.shot_id}): {e.verdict or 'not reviewed'}"
          + (f", rejects {', '.join(e.rejects)}" if e.rejects else "")
          + (f", comment {len(e.comment)} chars" if e.comment else ""))
    print(f"reviewed {review.reviewed} of {review.total}")
    return 0


@_guard
def cmd_review_comment(settings: Settings, args: argparse.Namespace) -> int:
    from .review import store
    review = store.set_project_comment(settings, args.project, _text(args.text))
    print(f"project comment saved ({len(review.project_comment)} chars)")
    return 0


@_guard
def cmd_review_summary(settings: Settings, args: argparse.Namespace) -> int:
    from .review import export, store
    review = store.load(settings, args.project)
    print(json.dumps(export.summary(review), indent=2) if args.json else export.summary_text(review))
    return 0


@_guard
def cmd_review_export(settings: Settings, args: argparse.Namespace) -> int:
    from .review import export, store
    path = export.write_export(settings, args.project)
    if args.json:
        print(json.dumps({"path": str(path), **export.summary(store.load(settings, args.project))}, indent=2))
    else:
        print(export.summary_text(store.load(settings, args.project)))
        print(f"wrote {path}")
    return 0


@_guard
def cmd_review_calibrate(settings: Settings, args: argparse.Namespace) -> int:
    from .review import calibrate
    c = calibrate.calibrate_project(settings, args.project)
    print(json.dumps(c, indent=2) if args.json else calibrate.calibration_text(c))
    return 0


@_guard
def cmd_review_fixes(settings: Settings, args: argparse.Namespace) -> int:
    from .editor.ops import apply_and_render
    from .models import EditOp
    from .review import fixes, store
    p = fixes.plan(settings, args.project)
    print(json.dumps(p, indent=2) if args.json else fixes.plan_text(p))
    if not args.apply or not p["ops"]:
        return 0
    ops = [EditOp.model_validate(o) for o in p["ops"]]
    store.mark_fixed(settings, args.project, p["ops"], None)
    result = apply_and_render(settings, args.project, ops, echo=not args.quiet)
    for line in result.applied:
        print(f"applied {line}")
    for line in result.errors:
        print(f"error {line}", file=sys.stderr)
    return 1 if result.errors else 0


REVIEW_COMMANDS = {"review-set": cmd_review_set, "review-comment": cmd_review_comment,
                   "review-summary": cmd_review_summary, "review-export": cmd_review_export,
                   "review-calibrate": cmd_review_calibrate, "review-fixes": cmd_review_fixes}
