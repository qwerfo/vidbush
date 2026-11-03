"""Command line entry: python -m vidbush <command>."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

from .config import Settings, load_settings
from .models import Brief, InputMode, VideoFormat

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
    proj = run_project(settings, args.project, force=args.force or [], echo=not args.quiet)
    return _report(settings, proj)


def _report(settings: Settings, proj) -> int:
    out = Path(proj.paths.root)
    print(json.dumps({"project": proj.id, "status": proj.status, "final": str(out / "final.mp4"),
                      "exists": (out / "final.mp4").exists()}, indent=2))
    return 0 if proj.status == "done" else 1


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


def cmd_revise(settings: Settings, args: argparse.Namespace) -> int:
    from .editor.ops import apply_and_render
    from .editor.revise import map_message
    ops = map_message(settings, args.project, args.message)
    print(json.dumps([o.model_dump(mode="json") for o in ops], indent=2))
    if args.dry_run:
        return 0
    res = apply_and_render(settings, args.project, ops, echo=not args.quiet)
    print(res.model_dump_json(indent=2))
    return 0 if not res.errors else 1


def cmd_doctor(settings: Settings, args: argparse.Namespace) -> int:
    from .doctor import doctor
    report = doctor(settings)
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
    sub.add_parser("doctor", help="check ffmpeg, yt-dlp, fonts, seats")
    ck = sub.add_parser("check", help="verify a finished project against the acceptance checks")
    ck.add_argument("project")
    ck.add_argument("--min-minutes", type=float, default=3.0)
    return p


COMMANDS = {"demo": cmd_demo, "run": cmd_run, "resume": cmd_resume, "serve": cmd_serve,
            "learn-style": cmd_learn, "swap": cmd_swap, "revise": cmd_revise, "doctor": cmd_doctor, "check": cmd_check}


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    if args.offline:
        settings.offline = True
    return COMMANDS[args.cmd](settings, args)


if __name__ == "__main__":
    sys.exit(main())
