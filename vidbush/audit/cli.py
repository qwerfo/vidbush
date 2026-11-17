"""Audit subcommands.

audit <project id | project dir | video.mp4> [--script ...] [--out DIR] [--every 2] [--limit N] [--workers 8]
audit-compare <audit.jsonl> <reference rows jsonl ...>
audit-sheet <audit.jsonl> [--html ...] [--key ...]          the 40 tile calibration page and its key
audit-calibrate <marks.json> [--key ...]                    agreement of a person's marks with the audit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from . import core


def _audit_root(settings: Settings) -> Path:
    """Audits of bare videos and the calibration files: <output folder's parent>/audit."""
    return settings.path(settings.output_folder).parent / "audit"


def add_audit_parsers(sub: Any) -> None:
    a = sub.add_parser("audit", help="tile audit of a finished video: one frame every 2 s against its words")
    a.add_argument("target", help="project id, project dir, or a video file")
    a.add_argument("--script", help="narration.json, transcript.json (startMs/endMs) or a project dir")
    a.add_argument("--out", help="output dir (default <project>/audit, or <audit root>/<video stem>)")
    a.add_argument("--every", type=float, default=2.0, help="seconds between tiles")
    a.add_argument("--limit", type=int, default=0, help="audit only N tiles spread evenly over the video")
    a.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("audit-compare", help="compare an audit.jsonl with reference tile rows, joined on t")
    c.add_argument("audit")
    c.add_argument("reference", nargs="+")
    c.add_argument("--json", action="store_true")
    s = sub.add_parser("audit-sheet", help="build the 40 tile calibration page and its key from an audit")
    s.add_argument("audit", help="audit.jsonl (its tiles/ folder must sit beside it)")
    s.add_argument("--html", help="default <audit root>/calibration_<name>.html")
    s.add_argument("--key", help="default <audit root>/calibration_<name>_key.json")
    s.add_argument("--name", default="run3")
    k = sub.add_parser("audit-calibrate", help="agreement of exported calibration marks with the audit")
    k.add_argument("marks")
    k.add_argument("--key", help="default <audit root>/calibration_run3_key.json")


def _resolve(settings: Settings, target: str) -> tuple[Path, Optional[Path]]:
    """(video, project root or None)."""
    p = Path(target)
    if p.is_file():
        return p, p.parent if (p.parent / "project.json").exists() else None
    if not p.is_dir():
        from ..jobs.pipeline import project_dir
        p = project_dir(settings, target)
    return p / "final.mp4", p


def cmd_audit(settings: Settings, args: argparse.Namespace) -> int:
    from ..providers.base import CostTracker
    from ..providers.seats import Seats
    from ..util import media_duration
    from . import auditor

    video, root = _resolve(settings, args.target)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    script = Path(args.script) if args.script else root
    if script is None:
        print("a bare video needs --script (narration.json or transcript.json)", file=sys.stderr)
        return 2
    words = core.load_words(script)
    out = Path(args.out) if args.out else (root / "audit" if root else _audit_root(settings) / video.stem)
    out.mkdir(parents=True, exist_ok=True)
    # the audit's own cost file, never the project's cost_report.json
    tracker = CostTracker(f"audit:{video.stem}", out / "cost_report.json")
    seats = Seats(settings, tracker)
    if not seats.is_real(auditor.ROLE):
        print("warning: the auditor seat has no configured provider; replies are mock and every tile fails",
              file=sys.stderr)
    t0, cost0 = time.time(), tracker.total_usd
    rows = auditor.run_audit(video, words, out, auditor.seats_vision(seats), media_duration(video),
                             shots=core.load_shots(root), every=args.every, limit=args.limit, workers=args.workers)
    summary = auditor.write_reports(rows, out, video, auditor.model_used(tracker.report.entries, t0),
                                    tracker.total_usd - cost0, args.every)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out / 'audit.md'} in {time.time() - t0:.0f} s")
    return 0


def cmd_audit_compare(settings: Settings, args: argparse.Namespace) -> int:
    res = core.compare(core.read_jsonl([args.audit]), core.read_jsonl(args.reference))
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    ag = res["agreement"]
    print(f"{ag['tiles']} tiles joined on t")
    print(f"{'measure':<22}{'audit':>8}{'reference':>11}{'diff':>8}")
    for k in core.SUMMARY_KEYS:
        print(f"{k:<22}{res['audit'][k]:>8.1f}{res['reference'][k]:>11.1f}{res['diff'][k]:>+8.1f}")
    print(f"overlay tiles: audit {res['audit']['overlay_tiles']}, reference {res['reference']['overlay_tiles']}")
    print(f"per-tile agreement: match {ag['match']}%, on_words yes/no {ag['on_words']}%, "
          f"noticeable {ag['noticeable']}%, kind group {ag['kind_group']}%")
    print(f"noticeable: both {ag['noticeable_both']}, audit only {ag['noticeable_audit_only']}, "
          f"reference only {ag['noticeable_ref_only']}")
    return 0


def cmd_audit_sheet(settings: Settings, args: argparse.Namespace) -> int:
    from . import calibrate
    audit = Path(args.audit)
    rows = core.read_jsonl([audit])
    summary = audit.parent / "audit_summary.json"
    video = json.loads(summary.read_text(encoding="utf-8"))["video"] if summary.exists() else str(audit.parent)
    root = _audit_root(settings)
    html_path = Path(args.html) if args.html else root / f"calibration_{args.name}.html"
    key_path = Path(args.key) if args.key else root / f"calibration_{args.name}_key.json"
    chosen = calibrate.build_sheet(rows, audit.parent / "tiles", video, html_path, key_path)
    print(f"{len(chosen)} tiles ({sum(r['noticeable'] for r in chosen)} flagged): {html_path}, key {key_path}")
    return 0


def cmd_audit_calibrate(settings: Settings, args: argparse.Namespace) -> int:
    from . import calibrate
    key_path = Path(args.key) if args.key else _audit_root(settings) / "calibration_run3_key.json"
    res = calibrate.agreement(json.loads(Path(args.marks).read_text(encoding="utf-8")),
                              json.loads(key_path.read_text(encoding="utf-8")))
    print(f"agreement {res['agree']} of {res['total']} ({res['marked']} marked)")
    if res["unmarked"]:
        print(f"unmarked: {', '.join(res['unmarked'])}")
    print("TRUSTED" if res["trusted"] else f"NOT TRUSTED (needs {calibrate.TRUSTED_AT} of 40, every tile marked)")
    for d in res["disagreements"]:
        print(f"- {d['id']} at {core.mmss(d['t'] or 0)}: you {d['person']}, audit {d['audit']}"
              + (f"; audit defect: {d['audit_defect']}" if d["audit_defect"] else "")
              + (f"; your comment: {d['comment']}" if d["comment"] else ""))
    return 0 if res["trusted"] else 1


AUDIT_COMMANDS = {"audit": cmd_audit, "audit-compare": cmd_audit_compare, "audit-sheet": cmd_audit_sheet,
                  "audit-calibrate": cmd_audit_calibrate}
