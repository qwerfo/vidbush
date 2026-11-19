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


COST_FILE = "audit_cost_report.json"


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
    k.add_argument("--key", help="default: the calibration_*_key.json under the audit root for the marks' video")


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
    from ..providers.base import CheapSeatDown, CostTracker
    from ..providers.seats import Seats
    from ..util import media_duration, write_json
    from . import auditor

    video, root = _resolve(settings, args.target)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    script = Path(args.script) if args.script else root
    if script is None:
        print("a bare video needs --script (narration.json or transcript.json)", file=sys.stderr)
        return 2
    # narration and shot times are on the body timeline; final.mp4 may open with an intro
    intro = core.intro_offset(root)
    words = core.load_words(script)
    if root and script.resolve().is_relative_to(root.resolve()):
        words = core.shift(words, intro)
    out = Path(args.out) if args.out else (root / "audit" if root else _audit_root(settings) / video.stem)
    out.mkdir(parents=True, exist_ok=True)
    # the audit's own cost file under its own name, never a project's cost_report.json, whatever --out is
    tracker = CostTracker(f"audit:{video.stem}", out / COST_FILE)
    seats = Seats(settings, tracker)
    configured = [p for p in seats.providers_for(auditor.ROLE) if p.configured]
    if not configured:
        print("warning: the auditor seat has no configured provider; replies are mock and every tile fails",
              file=sys.stderr)
    model = f"{configured[0].name}/{configured[0].model}" if configured else "mock"
    t0 = time.time()
    try:
        auditor.run_audit(video, words, out, auditor.seats_vision(seats), media_duration(video),
                          shots=core.shift(core.load_shots(root), intro), every=args.every, limit=args.limit,
                          workers=args.workers, model=model)
    except CheapSeatDown as e:  # never a dearer auditor: stop, keep the rows so far, resume later
        write_json(out / COST_FILE, tracker.report)
        print(f"audit stopped: {e}", file=sys.stderr)
        return 1
    write_json(out / COST_FILE, tracker.report)  # written even when this run made no call
    # the reports cover every row of the audit, never only this run's slice
    summary = auditor.write_reports(core.latest_rows(out / "audit.jsonl"), out, video,
                                    auditor.model_used(tracker.report.entries, 0) or model, tracker.total_usd,
                                    args.every)
    if summary is None:
        print(f"no tile has a valid verdict; no summary written (see {out / 'audit.jsonl'})", file=sys.stderr)
        return 1
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
    def cell(v: Any, width: int, sign: str = "") -> str:
        return f"{'n/a':>{width}}" if v is None else f"{v:>{sign}{width}.1f}"

    for k in core.SUMMARY_KEYS:
        print(f"{k:<22}{cell(res['audit'][k], 8)}{cell(res['reference'][k], 11)}{cell(res['diff'][k], 8, '+')}")
    print(f"overlay tiles: audit {res['audit']['overlay_tiles']}, reference {res['reference']['overlay_tiles']}")
    print(f"per-tile agreement: match {ag['match']}%, on_words yes/no {ag['on_words']}%, "
          f"noticeable {ag['noticeable']}%, kind group {ag['kind_group']}%")
    print(f"noticeable: both {ag['noticeable_both']}, audit only {ag['noticeable_audit_only']}, "
          f"reference only {ag['noticeable_ref_only']}")
    return 0


def cmd_audit_sheet(settings: Settings, args: argparse.Namespace) -> int:
    from . import calibrate
    audit = Path(args.audit)
    from .auditor import video_hash
    rows = core.latest_rows(audit)
    summary = audit.parent / "audit_summary.json"
    video = json.loads(summary.read_text(encoding="utf-8"))["video"] if summary.exists() else str(audit.parent)
    side = audit.parent / "tiles" / "video.json"  # identity of the video the frames came from
    if not side.exists():
        print(f"missing {side}: rerun the audit so frames and video identity match", file=sys.stderr)
        return 2
    vhash = video_hash(json.loads(side.read_text(encoding="utf-8")))
    root = _audit_root(settings)
    html_path = Path(args.html) if args.html else root / f"calibration_{args.name}.html"
    key_path = Path(args.key) if args.key else root / f"calibration_{args.name}_key.json"
    chosen = calibrate.build_sheet(rows, audit.parent / "tiles", video, vhash, html_path, key_path)
    print(f"{len(chosen)} tiles ({sum(r['noticeable'] for r in chosen)} flagged): {html_path}, key {key_path}")
    return 0


def cmd_audit_calibrate(settings: Settings, args: argparse.Namespace) -> int:
    from . import calibrate
    marks = json.loads(Path(args.marks).read_text(encoding="utf-8"))
    if args.key:
        key_path = Path(args.key)
    else:  # the key file under the audit root made for the same video as the marks
        found = [p for p in sorted(_audit_root(settings).glob("calibration_*_key.json"))
                 if json.loads(p.read_text(encoding="utf-8")).get("video") == marks.get("video")]
        if len(found) != 1:
            print(f"{len(found)} key files match video {marks.get('video')!r}; pass --key", file=sys.stderr)
            return 2
        key_path = found[0]
    try:
        res = calibrate.agreement(marks, json.loads(key_path.read_text(encoding="utf-8")))
    except calibrate.CalibrationError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
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
