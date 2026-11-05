"""Command line entry: python -m vidbush <command>."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from .batch_cli import BATCH_COMMANDS, add_batch_parsers
from .config import Settings, load_settings
from .models import Brief, InputMode, VideoFormat
from .review_cli import REVIEW_COMMANDS, add_review_parsers

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def parse_brief_file(path: Path) -> Optional[Brief]:
    """A brief is a markdown file with YAML front matter that has a topic key."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not m:
        return None
    meta = yaml.safe_load(m.group(1)) or {}
    if "topic" not in meta:
        return None
    body = m.group(2).strip()
    return Brief(topic=meta["topic"], target_minutes=float(meta.get("target_minutes", 8)),
                 format=VideoFormat(meta.get("format", "documentary")),
                 audience=meta.get("audience", "") or body, angle=meta.get("angle", ""))


def detect_input(path: Path, script: Optional[str]) -> tuple[InputMode, dict]:
    if path.suffix.lower() in AUDIO_EXT:
        sidecar = script
        if sidecar is None:
            for ext in (".md", ".txt"):
                cand = path.with_suffix(ext)
                if cand.exists():
                    sidecar = str(cand)
                    break
        return InputMode.voiceover, {"voiceover_path": str(path), "script_path": sidecar}
    brief = parse_brief_file(path)
    if brief:
        return InputMode.brief, {"brief": brief}
    return InputMode.script, {"script_path": str(path)}


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    from .jobs.pipeline import create_project, run_project
    path = Path(args.input)
    if not path.exists():
        print(f"input not found: {path}", file=sys.stderr)
        return 2
    mode, kw = detect_input(path, args.script)
    pid = args.id or re.sub(r"[^a-zA-Z0-9_\-]", "_", path.stem)
    fmt = VideoFormat(args.format) if args.format else None
    create_project(settings, pid, mode, brand_id=args.brand, fmt=fmt, target_minutes=args.minutes,
                   style_profile_id=args.profile, seed=args.seed, **kw)
    proj = run_project(settings, pid, echo=not args.quiet)
    return _report(settings, proj)


def cmd_demo(settings: Settings, args: argparse.Namespace) -> int:
    args.input = str(settings.path("fixtures/demo_brief.md"))
    args.id = args.id or "demo"
    args.script = None
    return cmd_run(settings, args)


def cmd_resume(settings: Settings, args: argparse.Namespace) -> int:
    from .jobs.pipeline import run_project
    proj = run_project(settings, args.project, force=args.force or [], echo=not args.quiet,
                       cost_cap=getattr(args, "cost_cap", None))
    return _report(settings, proj)


def _report(settings: Settings, proj) -> int:
    out = Path(proj.paths.root)
    print(json.dumps({"project": proj.id, "status": proj.status, "final": str(out / "final.mp4"),
                      "exists": (out / "final.mp4").exists()}, indent=2))
    if proj.status == "needs_review":
        print("some seats used mock output while real providers are configured: check qa_report.md",
              file=sys.stderr)
    return 0 if proj.status in ("done", "needs_review") else 1


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn
    from .api.app import create_app
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_learn(settings: Settings, args: argparse.Namespace) -> int:
    from .learn.style_learner import learn_style
    path = learn_style(args.source, args.name, settings)
    print(str(path))
    return 0


def cmd_swap(settings: Settings, args: argparse.Namespace) -> int:
    from .editor.ops import apply_and_render
    from .models import EditOp, EditOpName
    target = args.candidate
    if Path(target).exists():
        op = EditOp(op=EditOpName.upload_shot, args={"shot_id": args.shot, "path": target})
    else:
        op = EditOp(op=EditOpName.swap_shot, args={"shot_id": args.shot, "candidate_id": target})
    res = apply_and_render(settings, args.project, [op], echo=not args.quiet)
    print(res.model_dump_json(indent=2))
    return 0 if not res.errors else 1


def _apply(settings: Settings, project: str, ops: list, quiet: bool, dry_run: bool = False) -> int:
    """Print the ops, then apply and re-render them (the same path the UI job takes)."""
    from .editor.ops import apply_and_render
    print(json.dumps([o.model_dump(mode="json") for o in ops], indent=2))
    if dry_run:
        return 0
    res = apply_and_render(settings, project, ops, echo=not quiet)
    print(res.model_dump_json(indent=2))
    return 0 if not res.errors else 1


def parse_op_args(pairs: list[str]) -> dict:
    """key=value pairs; values are JSON when they parse (numbers, lists, objects), else plain strings."""
    out: dict = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"expected key=value, got {pair!r}")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def cmd_op(settings: Settings, args: argparse.Namespace) -> int:
    from .models import EditOp, EditOpName
    try:
        op = EditOp(op=EditOpName(args.name), args=parse_op_args(args.args), reason="cli")
    except ValueError as e:
        names = ", ".join(n.value for n in EditOpName)
        print(f"{e} (ops: {names})", file=sys.stderr)
        return 2
    return _apply(settings, args.project, [op], args.quiet, args.dry_run)


def cmd_ops(settings: Settings, args: argparse.Namespace) -> int:
    from .models import EditOp
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    items = data.get("ops", []) if isinstance(data, dict) else data
    return _apply(settings, args.project, [EditOp.model_validate(o) for o in items], args.quiet, args.dry_run)


def _shot_cmd(op_name: str, reason: str, **extra: Any) -> Any:
    """A subcommand mirroring one shot action of the UI (routes_shots)."""
    def run(settings: Settings, args: argparse.Namespace) -> int:
        from .models import EditOp, EditOpName
        op_args = {"shot_id": args.shot}
        for key, attr in extra.items():
            op_args[key] = getattr(args, attr)
        return _apply(settings, args.project, [EditOp(op=EditOpName(op_name), args=op_args, reason=reason)],
                      args.quiet)
    return run


def cmd_projects(settings: Settings, args: argparse.Namespace) -> int:
    from .jobs.pipeline import load_project
    out_dir = settings.path(settings.output_folder)
    rows = []
    for pj in sorted(out_dir.glob("*/project.json")) if out_dir.exists() else []:
        try:
            p = load_project(settings, pj.parent.name)
        except ValueError:
            continue
        rows.append({"id": p.id, "title": p.title, "status": p.status, "stage": p.current_stage or "",
                     "mode": p.input_mode.value, "format": p.format.value,
                     "final": (pj.parent / "final.mp4").exists()})
    print(json.dumps(rows, indent=2))
    return 0


def cmd_brands(settings: Settings, args: argparse.Namespace) -> int:
    from .config import list_ids, load_brand
    rows = []
    for bid in list_ids(settings, "brands"):
        b = load_brand(settings, bid)
        rows.append({"id": bid, "name": b.name or bid, "style_profile_id": b.style_profile_id})
    print(json.dumps({"brands": rows, "style_profiles": list_ids(settings, "style_profiles")}, indent=2))
    return 0


def cmd_seats(settings: Settings, args: argparse.Namespace) -> int:
    from .providers.base import HEALTH, CostTracker
    from .providers.seats import Seats
    status = Seats(settings, CostTracker("cli")).status()
    print(json.dumps({"offline": settings.offline, "allow_mock_fallback": settings.allow_mock_fallback,
                      "seats": status, "mock_roles": sorted(r for r, v in status.items() if not v["real"]),
                      "down": HEALTH.snapshot()}, indent=2))
    return 0


def cmd_revise(settings: Settings, args: argparse.Namespace) -> int:
    from .editor.revise import map_message
    return _apply(settings, args.project, map_message(settings, args.project, args.message), args.quiet,
                  args.dry_run)


def cmd_doctor(settings: Settings, args: argparse.Namespace) -> int:
    from .doctor import doctor
    report = doctor(settings, probe=getattr(args, "probe", False))
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


def cmd_check(settings: Settings, args: argparse.Namespace) -> int:
    from .acceptance import check_project
    res = check_project(settings, args.project, min_minutes=args.min_minutes)
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vidbush", description="Long-form faceless video production")
    p.add_argument("--offline", action="store_true", help="never use the network; mock every seat")
    sub = p.add_subparsers(dest="cmd", required=True)

    def run_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--id")
        sp.add_argument("--brand", default="default")
        sp.add_argument("--format", choices=["documentary", "top10"])
        sp.add_argument("--minutes", type=float)
        sp.add_argument("--profile")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--quiet", action="store_true")

    d = sub.add_parser("demo", help="run fixtures/demo_brief.md end to end")
    run_opts(d)
    r = sub.add_parser("run", help="run a brief (.md with front matter), a script, or a voiceover")
    r.add_argument("input")
    r.add_argument("--script", help="script file to pair with a voiceover")
    run_opts(r)
    rs = sub.add_parser("resume", help="resume a project from its first unfinished stage")
    rs.add_argument("project")
    rs.add_argument("--force", nargs="*", help="stages to rerun even if done")
    rs.add_argument("--quiet", action="store_true")
    rs.add_argument("--cost-cap", type=float, default=None, help="stop (needs_review) once the project spent this many USD")
    sv = sub.add_parser("serve", help="start the API and UI")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8080)
    ls = sub.add_parser("learn-style", help="learn a style profile from a reference video")
    ls.add_argument("source")
    ls.add_argument("name")
    sw = sub.add_parser("swap", help="swap a shot for a candidate id or a local file")
    sw.add_argument("project")
    sw.add_argument("shot", help="shot id or 1-based shot number")
    sw.add_argument("candidate", help="candidate id or path to your own file")
    sw.add_argument("--quiet", action="store_true")
    rv = sub.add_parser("revise", help="apply a natural language revision")
    rv.add_argument("project")
    rv.add_argument("message")
    rv.add_argument("--dry-run", action="store_true")
    rv.add_argument("--quiet", action="store_true")
    _edit_parsers(sub)
    sub.add_parser("projects", help="list projects with status")
    sub.add_parser("brands", help="list brands and style profiles")
    sub.add_parser("seats", help="seat status per role (real or mock) and providers marked down")
    dp = sub.add_parser("doctor", help="check ffmpeg, yt-dlp, fonts, seats")
    dp.add_argument("--probe", action="store_true", help="send one tiny live request per configured role")
    ck = sub.add_parser("check", help="verify a finished project against the acceptance checks")
    ck.add_argument("project")
    ck.add_argument("--min-minutes", type=float, default=3.0)
    add_batch_parsers(sub)
    add_review_parsers(sub)
    return p


def _edit_parsers(sub: Any) -> None:
    """Subcommands for every shot action of the UI, plus raw ops."""
    def shot_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("project")
        sp.add_argument("shot", help="shot id or 1-based shot number")
        sp.add_argument("--quiet", action="store_true")
        return sp

    up = shot_parser("upload", "use your own image or video for a shot")
    up.add_argument("path")
    rg = shot_parser("regenerate", "regenerate a shot as an AI image")
    rg.add_argument("--prompt", default="", help="extra words for the image prompt")
    du = shot_parser("duration", "set a shot's duration (the next shot gives or takes the time)")
    du.add_argument("seconds", type=float)
    shot_parser("remove-overlay", "remove a shot's overlay")
    op = sub.add_parser("op", help="apply one edit op: op <project> <op_name> key=value ...")
    op.add_argument("project")
    op.add_argument("name")
    op.add_argument("args", nargs="*", help="key=value, values parsed as JSON when possible")
    ops = sub.add_parser("ops", help="apply a JSON file of edit ops (a list, or {\"ops\": [...]})")
    ops.add_argument("project")
    ops.add_argument("file")
    for sp in (op, ops):
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--quiet", action="store_true")


COMMANDS = {"demo": cmd_demo, "run": cmd_run, "resume": cmd_resume, "serve": cmd_serve,
            "learn-style": cmd_learn, "swap": cmd_swap, "revise": cmd_revise, "doctor": cmd_doctor, "check": cmd_check,
            "op": cmd_op, "ops": cmd_ops, "projects": cmd_projects, "brands": cmd_brands, "seats": cmd_seats,
            "upload": _shot_cmd("upload_shot", "operator uploaded a file", path="path"),
            "regenerate": _shot_cmd("regenerate_ai", "operator asked for an AI image", prompt_delta="prompt"),
            "duration": _shot_cmd("set_duration", "operator set the duration", seconds="seconds"),
            "remove-overlay": _shot_cmd("remove_overlay", "operator removed the overlay")}
COMMANDS.update(BATCH_COMMANDS)
COMMANDS.update(REVIEW_COMMANDS)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    if args.offline:
        settings.offline = True
    from .jobs.lock import ProjectBusy
    try:
        return COMMANDS[args.cmd](settings, args)
    except ProjectBusy as e:
        print(str(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
