"""Batch subcommands: batch, batch-status, batch-pause, batch-resume (see vidbush/jobs/batch.py)."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import Settings


def add_batch_parsers(sub: Any) -> None:
    """Register the batch subcommands on the main parser's subparsers."""
    b = sub.add_parser("batch", help="queue every brief, script and voiceover of a folder or list file and run them")
    b.add_argument("input", help="folder (non recursive) or a list file with one path per line")
    b.add_argument("--brand", default="default")
    b.add_argument("--profile")
    b.add_argument("--format", choices=["documentary", "top10"])
    b.add_argument("--minutes", type=float)
    b.add_argument("--cost-cap", type=float, default=None,
                   help="USD per project run (default batch.cost_cap_usd in config/vidbush.yaml; 0 = no cap)")
    b.add_argument("--id-prefix", default="", help="prefix for the project ids derived from file stems")
    b.add_argument("--batch-id", help="batch id (default: a timestamp)")
    b.add_argument("--plan", action="store_true", help="only queue and print the plan, do not run")
    b.add_argument("--quiet", action="store_true")
    st = sub.add_parser("batch-status", help="list batches, or print one batch's report")
    st.add_argument("batch_id", nargs="?")
    st.add_argument("--json", action="store_true", help="print batch_report.json instead of the table")
    pa = sub.add_parser("batch-pause", help="pause a batch after the project that is running now")
    pa.add_argument("batch_id")
    rs = sub.add_parser("batch-resume", help="continue a paused or interrupted batch from its first unfinished item")
    rs.add_argument("batch_id")
    rs.add_argument("--retry-failed", action="store_true", help="queue failed and capped items again")
    rs.add_argument("--cost-cap", type=float, default=None, help="new USD cap per project run (0 = no cap)")
    rs.add_argument("--quiet", action="store_true")


def _finish(settings: Settings, state: Any) -> int:
    """Print the report table and path; exit 1 when any item failed."""
    from .jobs.batch import batch_dir
    from .jobs.batch_report import REPORT_MD, report_markdown
    print(report_markdown(state))
    print(f"report: {batch_dir(settings, state.id) / REPORT_MD}")
    if state.status == "paused":
        print(f"batch paused: python -m vidbush batch-resume {state.id}", file=sys.stderr)
    return 1 if any(i.status == "failed" for i in state.items) else 0


def _progress(state: Any, item: Any) -> None:
    done = sum(1 for i in state.items if i.status not in ("queued", "running"))
    print(f"[batch {state.id}] {done}/{len(state.items)} {item.project_id}: {item.status}"
          f"{' (' + item.note + ')' if item.note else ''}", file=sys.stderr)


def cmd_batch(settings: Settings, args: argparse.Namespace) -> int:
    from .config import list_ids
    from .jobs.batch import create_batch, run_batch
    if args.brand not in list_ids(settings, "brands"):
        print(f"unknown brand: {args.brand}", file=sys.stderr)
        return 2
    try:
        state = create_batch(settings, args.input, brand=args.brand, profile=args.profile, fmt=args.format,
                             minutes=args.minutes, cost_cap=args.cost_cap, id_prefix=args.id_prefix,
                             batch_id=args.batch_id)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"batch {state.id}: {len(state.items)} items queued", file=sys.stderr)
    if args.plan:
        return _finish(settings, state)
    return _finish(settings, run_batch(settings, state.id, echo=not args.quiet, on_item=_progress))


def cmd_batch_status(settings: Settings, args: argparse.Namespace) -> int:
    from .jobs.batch import effective_status, list_batches, load_batch
    from .jobs.batch_report import report_json, report_markdown, summary
    if not args.batch_id:
        rows = [{"id": b.id, "status": effective_status(settings, b), "paused": b.paused, "brand": b.brand,
                 "source": b.source, "items": len(b.items), **summary(b)} for b in list_batches(settings)]
        print(json.dumps(rows, indent=2))
        return 0
    try:
        state = load_batch(settings, args.batch_id)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2
    state.status = effective_status(settings, state)  # type: ignore[assignment]
    print(json.dumps(report_json(state), indent=2) if args.json else report_markdown(state))
    return 0


def cmd_batch_pause(settings: Settings, args: argparse.Namespace) -> int:
    from .jobs.batch import is_running, pause_batch
    try:
        state = pause_batch(settings, args.batch_id)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2
    when = "after the running project finishes" if is_running(settings, state.id) else "now"
    print(json.dumps({"batch": state.id, "paused": True, "takes_effect": when}, indent=2))
    return 0


def cmd_batch_resume(settings: Settings, args: argparse.Namespace) -> int:
    from .jobs.batch import resume_batch
    try:
        state = resume_batch(settings, args.batch_id, echo=not args.quiet, retry_failed=args.retry_failed,
                             cost_cap=args.cost_cap)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2
    return _finish(settings, state)


BATCH_COMMANDS = {"batch": cmd_batch, "batch-status": cmd_batch_status, "batch-pause": cmd_batch_pause,
                  "batch-resume": cmd_batch_resume}
