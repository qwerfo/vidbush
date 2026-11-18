"""Replay describe-once and pick-by-comparison on a stored project, with no render.

    python -m vidbush.judge.pick_replay <project_dir> <out_dir> [--scenes s01,s08] [--env <dir with .env>]
        [--direct-from <replay dir with work/direct.json>] [--extra <results.json> <input.md>]...
        [--salt <text>] [--budget 4.0] [--resource] [--sheets s01,s08] [--previous <replay dir>]

Pools: the candidates stored in the project's work/shotlist.json, plus --extra pairs of a v2 sources
results file and its input sentences: each item's candidates join the pool of the stretch of narration
where its sentence is spoken. Director plans come from --direct-from (the same plans as an earlier replay,
so only the edit step differs) or a fresh director call. Description records are cached by content hash
(DESCRIBE_CACHE_FOLDER); picker and director replies are cached in out_dir/cache by a hash of the call,
so a replay with unchanged prompts costs nothing; --salt changes the picker's cache key (a second
independent run). Writes work/{clauses,direct,plan_v2,shotlist}.json, picks.json, metrics.json, and with
--sheets one review sheet per scene: clause words, the earlier pick and tonight's pick at the frame that
plays.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import textwrap
from pathlib import Path
from typing import Any, Optional

from ..models import Candidate, Narration, SourceKind
from ..plan import replay as R

_W = re.compile(r"[a-z0-9]+")


class SaltedSeats(R.CachedSeats):
    """CachedSeats whose picker key carries a salt: a second run asks the model again."""

    def __init__(self, seats, cache_dir: Path, salt: str = "", budget: float = 0.0, tracker=None) -> None:
        super().__init__(seats, cache_dir)
        self.salt, self.budget, self.tracker = salt, budget, tracker
        self.per_task: dict[str, dict[str, float]] = {}

    def _guard(self) -> None:
        if self.budget and self.tracker is not None and self.tracker.total_usd > self.budget:
            raise RuntimeError(f"spend {self.tracker.total_usd:.2f} USD is over the replay budget {self.budget}")

    def _count(self, task: str, fn):
        def run():
            self._guard()
            t0 = time.time()
            before = self.tracker.total_usd if self.tracker is not None else 0.0
            v = fn()
            d = self.per_task.setdefault(task, {"calls": 0, "usd": 0.0, "seconds": 0.0})
            d["calls"] += 1
            d["usd"] += (self.tracker.total_usd if self.tracker is not None else 0.0) - before
            d["seconds"] += time.time() - t0
            return v
        return run

    def llm_json(self, role, task, system, user, stage, **kw):
        return self._cached(self._key(role, task, system, user, stage, sorted(kw.items())),
                            self._count(task, lambda: self.seats.llm_json(role, task, system, user, stage, **kw)))

    def vision_json(self, role, task, system, prompt, images, stage, **kw):
        if task == "describe_asset":  # describe has its own content-hash cache
            return self._count(task, lambda: self.seats.vision_json(role, task, system, prompt, images, stage, **kw))()
        salt = self.salt if task == "plan_pick" else ""
        return self._cached(self._key(role, task, system, prompt, stage, sorted(kw.items()), salt, images=list(images)),
                            self._count(task, lambda: self.seats.vision_json(role, task, system, prompt, images,
                                                                              stage, **kw)))

    def providers_for(self, role):
        return self.seats.providers_for(role)


# ---------------------------------------------------------------- extra pool from v2 sources results


def _words(s: str) -> list[str]:
    return _W.findall(s.lower())


def sentence_span(sentence: str, narration: Narration) -> Optional[tuple[float, float]]:
    """Time span where the sentence is spoken: the window of narration words that best matches it."""
    want = _words(sentence)
    if not want:
        return None
    nw = [(_words(w.text), w) for w in narration.words]
    flat = [(t, w) for ts, w in nw for t in ts]
    n = len(want)
    best, at = 0, -1
    for i in range(0, max(1, len(flat) - n + 1)):
        win = [t for t, _ in flat[i:i + n]]
        score = sum(1 for a, b in zip(win, want) if a == b) + len(set(win) & set(want)) * 0.5
        if score > best:
            best, at = score, i
    if at < 0 or best < n * 0.6:
        return None
    return flat[at][1].start, flat[min(len(flat) - 1, at + n - 1)][1].end


def extra_from_results(results: Path, sentences: Path, narration: Narration) -> list[tuple[Candidate, float, float]]:
    """(candidate, start, end) for every candidate of a v2 sources proof whose sentence is in the narration."""
    data = json.loads(results.read_text(encoding="utf-8"))
    lines = [ln.strip() for ln in sentences.read_text(encoding="utf-8").splitlines() if ln.strip()]
    proj = results.parent / "projects"
    side: dict[tuple[str, str], Path] = {}
    for js in proj.glob("*/sources/*/*.json"):
        try:
            d = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        side[(js.parent.name, d.get("page_url", "") + "|" + d.get("title", ""))] = js.with_suffix("")
    out = []
    for key, item in data.items():
        m = re.match(r"[A-Z](\d+)_", key)
        if not m or int(m.group(1)) > len(lines):
            continue
        span = sentence_span(lines[int(m.group(1)) - 1], narration)
        if span is None:
            continue
        for c in item.get("candidates", []):
            folder = Path(c.get("keyframe", "")).parent.parent.name
            f = side.get((folder, c.get("page_url", "") + "|" + c.get("title", "")))
            if f is None or not f.exists():
                continue
            try:
                src = SourceKind(c["source"])
            except ValueError:
                continue
            w, h = (int(x) for x in c["size"].split("x")) if c.get("size") else (0, 0)
            sc = json.loads((f.parent / (f.name + ".json")).read_text(encoding="utf-8"))
            cand = Candidate(id=c["id"], source=src, url=sc.get("source_url", ""), local_path=str(f),
                             media="video" if item.get("media") == "video" else "image",
                             in_point=float(c.get("in", 0.0)), out_point=float(c.get("out", 0.0)), width=w, height=h,
                             licence=c.get("licence", "unknown"), author=sc.get("author", ""),
                             attribution=sc.get("attribution", ""), title=c.get("title", ""),
                             keyframes=[c["keyframe"]] if c.get("keyframe") else [], flags=list(c.get("flags", [])),
                             query=c.get("query", ""), rank=int(c.get("rank", 0) or 0),
                             description=sc.get("description", ""), date=sc.get("date", ""),
                             page_url=c.get("page_url", ""), found_by=c.get("found_by", ""))
            out.append((cand, span[0], span[1]))
    return out


# ---------------------------------------------------------------- review sheets

def _frame_at(c: Candidate, t: Any, out: Path) -> Optional[str]:
    from . import describe as DS
    if c.media != "video":
        return c.local_path if Path(c.local_path).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") else \
            DS.still_frame(c.local_path, out / f"{c.id}_still.jpg")
    tt = float(t) if isinstance(t, (int, float)) else c.in_point + 0.1
    return DS.extract_frame(c.local_path, tt + 0.3, out / f"{re.sub(r'[^A-Za-z0-9_]', '_', c.id)}_{tt:.1f}.jpg", 480)


def review_sheet(scene: str, clauses, picks_by_clause: dict, prev_by_clause: dict, out_dir: Path) -> list[str]:
    """One row per clause: its words, the earlier pick, tonight's pick (frame that plays), how it was filled."""
    from PIL import Image, ImageDraw

    from ..plan.sheets import _fit, _font
    TW, TH, TXT, ROWH, PAGE = 240, 135, 560, 145, 14
    f_txt, f_small, f_head = _font(15), _font(13), _font(17)
    frames = out_dir / "frames"
    cl = [c for c in clauses if c.scene == scene]
    paths = []
    for page in range(0, len(cl), PAGE):
        chunk = cl[page:page + PAGE]
        W = TXT + 2 * (TW + 10) + 10
        img = Image.new("RGB", (W, 52 + ROWH * len(chunk)), (250, 250, 247))
        d = ImageDraw.Draw(img)
        d.text((8, 6), f"{scene}", fill=(0, 0, 0), font=f_head)
        d.text((TXT + 10, 36), "EARLIER (last night's edit)", fill=(140, 0, 0), font=f_small)
        d.text((TXT + TW + 20, 36), "TONIGHT (describe + picker)", fill=(0, 90, 0), font=f_small)
        for r, c in enumerate(chunk):
            y = 52 + r * ROWH
            cur = picks_by_clause.get(c.id, {})
            d.line([(0, y), (W, y)], fill=(200, 200, 200))
            d.text((8, y + 4), f"{c.id}  {c.start:6.1f}s  {c.duration:.1f}s  [{cur.get('tag', '')}]",
                   fill=(20, 20, 120), font=f_small)
            for k, line in enumerate(textwrap.wrap(c.text, 62)[:4]):
                d.text((8, y + 22 + k * 18), line, fill=(0, 0, 0), font=f_txt)
            note = cur.get("note", "")
            for k, line in enumerate(textwrap.wrap(note, 76)[:2]):
                d.text((8, y + 98 + k * 16), line, fill=(90, 90, 90), font=f_small)
            for col, info in enumerate((prev_by_clause.get(c.id, {}), cur)):
                x = TXT + 10 + col * (TW + 10)
                path, tag = info.get("path"), info.get("label", "")
                if path and Path(path).exists():
                    img.paste(_fit(path, TW, TH), (x, y + 4))
                else:
                    d.rectangle([x, y + 4, x + TW, y + 4 + TH], fill=(60, 60, 70))
                    d.multiline_text((x + 8, y + 12), "\n".join(textwrap.wrap(info.get("text", tag), 30)[:5]),
                                     fill=(255, 255, 255), font=f_small)
                d.rectangle([x, y + 4 + TH - 18, x + TW, y + 4 + TH], fill=(0, 0, 0))
                d.text((x + 4, y + 4 + TH - 17), tag[:34], fill=(255, 255, 0), font=f_small)
        p = out_dir / f"review_{scene}_{page // PAGE + 1}.jpg"
        img.save(p, quality=90)
        paths.append(str(p))
    return paths


def _held_map(plan: dict) -> dict[str, dict]:
    out = {}
    for sc in plan["scenes"]:
        for r in sc["shots"]:
            for k, cid in enumerate(r["clauses"]):
                out[cid] = dict(r, _held=k > 0)
    return out


def sheet_rows(plan: dict, sl: dict, clauses, frames_dir: Path) -> dict:
    """Per clause: the shot on screen at mid-speech, as {path, label, tag, text, note}. sl is the shot list as
    plain JSON, so an earlier replay whose layout schema has since changed still reads."""
    rows = _held_map(plan)
    out = {}
    for c in clauses:
        mid = (c.speech_start + c.speech_end) / 2
        shot = next((s for s in sl["shots"] if s["start"] <= mid < s["end"]), None)
        r = rows.get(c.id, {})
        ed = r.get("edit") or {}
        info: dict[str, Any] = {"tag": ("hold" if r.get("_held") else r.get("treatment", "?"))}
        if shot is None:
            info["text"] = "no shot"
        else:
            cand = next((x for x in shot.get("candidates", []) if x["id"] == shot.get("selected")), None)
            lay = shot.get("layout") or None
            if cand is not None:
                cc = Candidate.model_validate(cand)
                info["path"] = _frame_at(cc, cc.in_point, frames_dir)
                info["label"] = ("AI " if cc.source.value.startswith("ai") else "") + cc.source.value
            else:
                info["label"] = str(shot.get("type", ""))
            if lay:
                kind = str(lay.get("kind", ""))
                info["text"] = kind + "\n" + " / ".join(str(e.get("text", "")) for e in lay.get("elements", [])[:3])
                info["label"] = f"{kind} " + info.get("label", "")
            ov = shot.get("overlay") or None
            if ov:
                info["label"] = "text: " + str((ov.get("payload") or {}).get("title", ""))[:24]
            info["note"] = " ".join(f for f in shot.get("flags", []) if f.startswith(("label:", "fallback:", "nothing_fits")))
        if ed:
            fb = ed.get("filled_by") or ""
            lab = f" label: {ed['label']}" if ed.get("label") else ""
            info["note"] = (f"[{fb}] " if fb else "") + (ed.get("reason") or "")[:130] + lab
        out[c.id] = info
    return out


# ---------------------------------------------------------------- main

def main(argv: Optional[list[str]] = None) -> int:
    from ..plan import clauses as C
    from ..plan import direct as D
    from ..plan import edit as E
    from ..plan import emit as M
    from ..plan import metrics as X
    from . import ladder as L

    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--env", default="")
    ap.add_argument("--direct-from", default="")
    ap.add_argument("--extra", nargs=2, action="append", default=[])
    ap.add_argument("--salt", default="")
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--resource", action="store_true")
    ap.add_argument("--no-ladder", action="store_true", help="picks only (run-to-run variation)")
    ap.add_argument("--sheets", default="")
    ap.add_argument("--previous", default="")
    a = ap.parse_args(argv)
    project, out = Path(a.project), Path(a.out)
    (out / "work").mkdir(parents=True, exist_ok=True)
    script, narration, old = R.load_project(project)
    only = set(filter(None, a.scenes.split(","))) or None
    clauses = C.segment(script, narration)
    C.save(clauses, out / "work" / "clauses.json")
    base_seats, tracker = R.build_seats(out, Path(a.env) if a.env else None)
    seats = SaltedSeats(base_seats.seats, out / "cache", a.salt, a.budget, tracker)
    settings = base_seats.seats.settings
    t0 = time.time()

    def log(m: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {m} (spent {tracker.total_usd:.3f} USD)", flush=True)

    if a.direct_from:
        scenes = json.loads((Path(a.direct_from) / "work" / "direct.json").read_text(encoding="utf-8"))
        scenes = [s for s in scenes if not only or s["scene"] in only]
    else:
        scenes = D.direct_all(seats, script, clauses, narration, log=log, only=only)
    (out / "work" / "direct.json").write_text(json.dumps(scenes, indent=1, ensure_ascii=False), encoding="utf-8")
    extra = []
    for res, sent in a.extra:
        got = extra_from_results(Path(res), Path(sent), narration)
        log(f"extra pool from {Path(res).name}: {len(got)} candidates")
        extra += got
    resource = None
    if a.resource:
        from . import resource as RS
        resource = RS.Resourcer(project, out, settings, log).search
    ladder = None if a.no_ladder else L.Ladder(seats, settings, out / "sheets", log, resource)
    t_edit = time.time()
    edits = E.edit_all(seats, scenes, clauses, narration, old, out / "sheets", log=log, settings=settings,
                       extra=extra, ladder=ladder)
    edit_minutes = (time.time() - t_edit) / 60
    plan, sl = M.emit(project.name, script, narration, clauses, scenes, edits)
    for sc, e in zip(plan["scenes"], edits):  # what filled each clause travels with the plan rows
        by = {p["clause"]: p for p in e["picks"]}
        for r in sc["shots"]:
            p = by.get(r["clauses"][0])
            if p is not None and "edit" in r:
                r["edit"].update(filled_by=p.get("filled_by", ""), identity=p.get("identity", ""),
                                 identity_field=p.get("identity_field", ""), start=p.get("start"),
                                 restated=p.get("restated"), search_again=p.get("search_again", []),
                                 passages_found=p.get("passages_found"))
    (out / "work" / "plan_v2.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "work" / "shotlist.json").write_text(sl.model_dump_json(indent=1), encoding="utf-8")
    m = X.report(clauses, plan, sl, narration, old, only, project=project)
    m["picker"] = picker_metrics(edits, seats, ladder, edit_minutes)
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    picks_dump(edits, out / "picks.json")
    if a.sheets:
        prev = None
        if a.previous:
            pp = Path(a.previous)
            pv = pp / "work" / "plan_v2.json"  # absent for an old slot-planner run: its shot list alone
            pplan = json.loads(pv.read_text(encoding="utf-8")) if pv.exists() else {"scenes": []}
            psl = json.loads((pp / "work" / "shotlist.json").read_text(encoding="utf-8"))
            prev = sheet_rows(pplan, psl, clauses, out / "review" / "frames")
        cur = sheet_rows(plan, json.loads(sl.model_dump_json()), clauses, out / "review" / "frames")
        for sc in a.sheets.split(","):
            print(review_sheet(sc, clauses, cur, prev or {}, out / "review"))
    print(json.dumps(m["picker"], indent=1, default=str))
    print("spent", round(tracker.total_usd, 4), "cache hits", seats.hits)
    return 0


def picker_metrics(edits: list[dict], seats: SaltedSeats, ladder, edit_minutes: float) -> dict:
    """Calls, cost and seconds per task, what filled each clause, and what code removed."""
    filled: dict[str, int] = {}
    nothing = []
    for e in edits:
        for p, s in zip(e["picks"], e["shots"]):
            from ..plan.edit import needs_pick
            if not needs_pick(s["plan"]) and p.get("filled_by") in ("", "pick", None):
                continue
            fb = p.get("filled_by") or "pick"
            filled[fb] = filled.get(fb, 0) + 1
            if fb != "pick":
                nothing.append({"clause": p["clause"], "subject": s["plan"].get("subject", "")[:120],
                                "filled_by": fb, "why": (p.get("reason") or "")[:160],
                                "resolved": p.get("resolved", ""), "search_again": p.get("search_again", []),
                                "restated": p.get("restated")})
    return {"edit_minutes": round(edit_minutes, 2), "tasks": seats.per_task, "filled_by": filled,
            "pool_assets": sum(e["pool_all"] for e in edits), "shown": sum(len(e["pool"]) for e in edits),
            "blocked_by_record": sum(len(e["blocked"]) for e in edits),
            "shortlist_dropped": sum(len(e["shortlist_dropped"]) for e in edits),
            "described_new": sum(1 for e in edits for r in e["records"].values() if not r.get("cached")),
            "attempt_errors": {e["scene"]: [len(x["errors"]) for x in e["attempts"]] for e in edits},
            "resolved": [{"clause": p["clause"], "resolved": p["resolved"]} for e in edits for p in e["picks"]
                         if p.get("resolved")], "not_picked": nothing}


def picks_dump(edits: list[dict], path: Path) -> None:
    rows = []
    for e in edits:
        for p in e["picks"]:
            c = p.get("_cand")
            rec = e["records"].get(c.id, {}) if c is not None else {}
            rows.append({"scene": e["scene"], "clause": p["clause"], "pick": c.id if c is not None else None,
                         "title": c.title if c is not None else "", "start": p.get("start"),
                         "shows": rec.get("visual", {}).get("shows", ""), "identity": p.get("identity", ""),
                         "label": p.get("label", ""), "filled_by": p.get("filled_by", ""),
                         "reason": p.get("reason", ""), "resolved": p.get("resolved", "")})
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
