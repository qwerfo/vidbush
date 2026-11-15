"""Prove the art layer on a finished project's stored narration and shot list (read only).

    python -m vidbush.art.prove --source <project output dir> --review <review dir> [--maps 8 --illustrations 10
                                --clips 14 --image-cap 300 --stop-clips-at 2026-11-15T08:30]

The source project is never written: its project.json, shotlist, narration and script are copied into
<review>/art_project, and every generated file lives there. Three workers run at once, so the slow video lane
(one clip at a time per key, about six minutes each) starts at once:
- maps: the director names moments where the narration establishes a place or describes a region; each gets
  a map from the bundled gazetteer and the normal map builder, painted (paint.paint_map) when paint_reason
  allows it;
- illustrations: illustrate.choose and illustrate.illustrate on the stored shot list;
- clips: a queue fed first with the opening's AI stills (the opening needs motion most), then with every
  illustration and painted map that passed; clips.make_clip checks each.
Image submits wait while the key's image credits this UTC hour reach --image-cap (the key is shared).
Outputs: art_report.json (every item, check and score), yield numbers, the contact sheet of the best 12 and
the numbered clips.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..config import load_settings
from ..models import MapPayload, MotionKind, MotionSpec, Narration, Project, ShotList
from . import clips as C
from . import common, illustrate as IL, paint as P
from .elevate import hour_credits

MOMENTS_SYSTEM = (
    "You plan illustrated maps for a geography documentary. From the timed narration, pick moments where the "
    "narration establishes a place or describes what a region is like (its land, climate, wildlife or way of "
    "life), not moments about a route, a distance or a border. Each moment names one to three real places that "
    "a gazetteer knows (towns, cities, mountains, rivers, states or countries) spelled as in an atlas, and the "
    "start and end second of the narration it serves. Prefer different regions. Reply with strict JSON only: "
    '{"moments": [{"start": s, "end": s, "places": ["..."], "why": "short"}]}'
)


def sentences(narr: Narration) -> list[dict[str, Any]]:
    out, cur = [], []
    for w in narr.words:
        cur.append(w)
        if re.search(r"[.!?][\"')]*$", w.text):
            out.append({"start": round(cur[0].start, 1), "end": round(cur[-1].end, 1),
                        "text": " ".join(x.text for x in cur)})
            cur = []
    if cur:
        out.append({"start": round(cur[0].start, 1), "end": round(cur[-1].end, 1), "text": " ".join(x.text for x in cur)})
    return out


def _wait_cap(spec: Any, cap: float, log: Any) -> None:
    while cap > 0:
        used, _ = hour_credits(spec)
        if used < cap:
            return
        now = datetime.now(timezone.utc)
        wait = 3600 - (now.minute * 60 + now.second) + 20
        log.info("art prove: image hour cap reached, waiting for the hour", used=used, cap=cap, wait_s=wait)
        time.sleep(min(wait, 600))


def build_map(places: list[str], start: float, end: float) -> Optional[MotionSpec]:
    from ..mg import builders, geo
    hits = []
    for name in places:
        # a bare name is looked up in the gazetteer (find_places scans sentences and drops lone words)
        hit = geo.lookup(name, us_bias=True)
        if hit is not None and hit.kind != "continent":
            hits.append((hit.name, hit.lat, hit.lon))
    if not hits:
        return None
    title = ", ".join(h[0] for h in hits[:2])
    spec = builders.map_spec(hits[:3], duration=max(6.0, min(9.0, end - start)), title=title, push_in=True)
    # an establishing map shows the place in its region: at regional zoom the coast is distinctive enough for a
    # painter to trace, where a zoom 6 delta or fjord coast defeated every model in the proving run
    p = spec.typed_payload()
    if p.zoom > ESTABLISH_ZOOM:
        p.zoom = ESTABLISH_ZOOM
        p.zoom_end = min(p.zoom_end or ESTABLISH_ZOOM + 0.4, ESTABLISH_ZOOM + 0.4)
        spec.payload = p.model_dump()
    return spec


ESTABLISH_ZOOM = 5.2


def widen_until_checkable(spec: MotionSpec, steps: int = 3) -> MotionSpec:
    """A view without a coastline cannot be registered; widen it (lower zoom, same center) until it can."""
    from .mapref import land_mask, paintable
    p = spec.typed_payload()
    for _ in range(steps):
        if paintable(land_mask(p, 384, 216)):
            break
        p.zoom = max(1.5, p.zoom - 0.8)
        if p.zoom_end is not None:
            p.zoom_end = max(p.zoom, p.zoom_end - 0.8)
    spec.payload = p.model_dump()
    return spec


class Prove:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.review = Path(args.review)
        self.root = self.review / "art_project"
        self.items: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.clipq: "queue.Queue[Optional[dict]]" = queue.Queue()
        self.ctx = self._context()
        self.narr = self.ctx.load("narration", Narration)
        # read without rebasing: the stored media paths point into the source project, which stays where it is
        from ..util import read_model
        self.shotlist = read_model(self.root / "work" / "shotlist.json", ShotList)
        self.spec = common.elevate_spec(self.ctx)
        stop = datetime.fromisoformat(args.stop_clips_at) if args.stop_clips_at else None
        # an offset given is kept; a time without one is UTC (the key's budgets reset on the UTC hour)
        self.stop_clips = (stop if stop.tzinfo else stop.replace(tzinfo=timezone.utc)) if stop else None

    def _context(self) -> Any:
        from ..jobs.context import build_context
        src = Path(self.args.source)
        work = self.root / "work"
        work.mkdir(parents=True, exist_ok=True)
        for name in ("shotlist.json", "narration.json", "script.json"):
            if not (work / name).exists():
                shutil.copy2(src / "work" / name, work / name)
        data = json.loads((src / "project.json").read_text(encoding="utf-8"))
        project = Project(**data)
        project.paths.root = str(self.root)
        project.paths.work = str(work)
        (self.root / "project.json").write_text(project.model_dump_json(indent=1), encoding="utf-8")
        settings = load_settings()
        ctx = build_context(settings, project, echo=False)
        ctx.stage = "art_prove"
        a = ctx.profile.art
        a.map_variants, a.illustration_variants = 3, 2
        return ctx

    def add(self, item: dict[str, Any]) -> None:
        with self.lock:
            self.items.append(item)
            name = {"ai-clips": "art_report_ai_clips.json", "maps": "art_report_maps.json",
                    "extra-illustrations": "art_report_extra_ills.json"}.get(self.args.only, "art_report.json")
            common.write_report(self.ctx, name, self.items)

    # ------------------------------------------------------------ maps

    def _cached(self, name: str, make: Any) -> Any:
        """The director's choices are kept on disk, so a restarted proving run reuses them and its paid files."""
        path = common.art_dir(self.ctx) / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        value = make()
        path.write_text(json.dumps(value, indent=1), encoding="utf-8")
        return value

    def moments(self, n: int) -> list[dict[str, Any]]:
        return self._cached("prove_moments.json", lambda: self._moments(n))

    def _moments(self, n: int, exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        sents = sentences(self.narr)
        text = "\n".join(f"[{s['start']:.0f}-{s['end']:.0f}] {s['text']}" for s in sents)
        avoid = f"Do not use these places again: {', '.join(exclude)}. Prefer towns and coasts.\n" if exclude else ""
        raw = self.ctx.seats.llm_json("director", "art_map_moments", MOMENTS_SYSTEM,
                                      f"Pick {n + 4} moments.\n{avoid}{text}", "art_prove", temperature=0.4)
        rows = (raw or {}).get("moments") if isinstance(raw, dict) else None
        return [r for r in rows or [] if isinstance(r, dict) and r.get("places")]

    def run_maps(self) -> None:
        out = self.root / "work" / "art" / "maps"
        done, seen = 0, set()
        first = self.moments(self.args.maps)
        used = tuple(sorted({str(x) for m in first for x in m.get("places", [])}))
        more = self._cached("prove_moments2.json", lambda: self._moments(self.args.maps, used)) \
            if self.args.only == "maps" else []
        for m in first + more:
            if done >= self.args.maps:
                break
            try:
                start, end = float(m.get("start", 0)), float(m.get("end", 0))
                if set(map(str, m["places"])) & set(self.args.skip_places):
                    self.add({"type": "map_skipped", "start": start, "places": m["places"], "reason": "skipped place"})
                    continue
                spec = build_map([str(x) for x in m["places"]][:3], start, end)
                if spec is None:
                    self.add({"type": "map_skipped", "start": start, "places": m["places"], "reason": "no place found"})
                    continue
                spec = widen_until_checkable(spec)
                p = spec.typed_payload()
                from .mapref import view_key
                vk = json.dumps(view_key(p), sort_keys=True)
                window = common.window_text(self.narr, start, end, 0.0)
                reason = P.paint_reason(spec, window)
                if vk in seen or reason:
                    self.add({"type": "map_skipped", "start": start, "places": m["places"], "reason": reason or "same view"})
                    continue
                seen.add(vk)
                _wait_cap(self.spec, self.args.image_cap, self.ctx.log)
                rep = P.paint_map(p, window, self.ctx, out, tag=f"m{done}")
                if rep["variants"] and all(v.get("error") for v in rep["variants"]):
                    # the image lane was failing (every variant errored, nothing was judged): wait for it once
                    self.ctx.log.info("art prove: every variant errored, waiting 5 minutes and retrying once")
                    time.sleep(300)
                    from ..providers.base import HEALTH
                    for model in {self.ctx.profile.art.map_model, self.ctx.profile.art.map_fallback_model}:
                        HEALTH.mark_up(f"{self.spec.name}:{model}")
                    rep = P.paint_map(p, window, self.ctx, out, tag=f"m{done}")
                item = {"type": "illustrated_map", "start": start, "end": end, "window": window, "places": m["places"],
                        "why": m.get("why", ""), "spec": spec.model_dump(mode="json"), "report": rep,
                        "ok": bool(rep.get("base"))}
                if item["ok"]:
                    item["frame"] = self.render_map(spec, rep["base"], out, f"m{done}")
                    self.clipq.put({"kind": "illustrated_map", "still": rep["base"], "item": item, "spec": spec})
                self.add(item)
                done += 1
            except Exception as e:  # noqa: BLE001  one moment never stops the run
                self.ctx.log.warn("art prove map failed", error=f"{type(e).__name__}: {str(e)[:300]}")

    def render_map(self, spec: MotionSpec, base: str, out: Path, tag: str, clip: str = "") -> str:
        """Final look of an illustrated map (painted base, renderer's markers, title and credit): a frame, or the
        whole map clip when a checked art clip is given."""
        from ..mg.renderer import render_spec
        p = spec.typed_payload()
        p.art_base = base
        p.art_clip = clip
        s2 = spec.model_copy(update={"payload": p.model_dump(), "source_note": "Map data: Natural Earth"})
        mp4 = render_spec(s2, out / f"{tag}_{'clip' if clip else 'render'}.mp4", False, self.ctx.settings)
        if clip:
            return str(mp4)
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 15))
        ok, f = cap.read()
        cap.release()
        jpg = out / f"{tag}_frame.jpg"
        if not ok:
            return ""
        cv2.imwrite(str(jpg), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return str(jpg)

    # ------------------------------------------------------------ illustrations

    def run_illustrations(self) -> None:
        out = self.root / "work" / "art" / "illustrations"
        if self.args.only == "extra-illustrations":  # a second round of picks, never a shot already tried
            tried = {p["shot_id"] for p in self._cached("prove_picks.json", lambda: [])}
            rest = self.shotlist.model_copy(update={"shots": [s for s in self.shotlist.shots if s.id not in tried]})
            picks = self._cached("prove_picks_extra.json", lambda: IL.choose(rest, self.ctx, self.args.illustrations))
        else:
            picks = self._cached("prove_picks.json", lambda: IL.choose(self.shotlist, self.ctx, self.args.illustrations))
        for pick in picks:
            try:
                shot = self.shotlist.by_id(pick["shot_id"])
                window = common.shot_text(self.narr, shot)
                _wait_cap(self.spec, self.args.image_cap, self.ctx.log)
                rep = IL.illustrate(shot, pick, self.ctx, out, window)
                cand = rep.pop("candidate", None)
                item = {"type": "illustration", "shot_id": shot.id, "start": shot.start, "end": shot.end,
                        "window": window, "report": rep, "ok": cand is not None}
                if cand is not None:
                    item["still"] = cand.local_path
                    item["score"] = cand.score
                    self.clipq.put({"kind": "illustration", "still": cand.local_path, "item": item, "shot": shot,
                                    "subject": pick.get("brief", "")})
                self.add(item)
            except Exception as e:  # noqa: BLE001
                self.ctx.log.warn("art prove illustration failed", error=f"{type(e).__name__}: {str(e)[:300]}")

    # ------------------------------------------------------------ clips

    def seed_ai_clips(self, n: int) -> None:
        """The opening first: AI stills of the stored shot list, picked by the clip planner."""
        eligible = {}
        for s in self.shotlist.shots:
            c = s.selected_candidate()
            if c is not None and c.source.value == "ai_image" and c.local_path and Path(c.local_path).exists():
                eligible[s.id] = "ai_image"
        art = self.ctx.profile.art.model_copy(update={"clip_every_s": max(60.0, self.shotlist.duration / max(1, n))})
        for sid in C.plan_clips(self.shotlist, art, eligible)[:n]:
            s = self.shotlist.by_id(sid)
            self.clipq.put({"kind": "ai_image", "still": s.selected_candidate().local_path, "shot": s,
                            "item": {"type": "ai_still", "shot_id": s.id, "start": s.start}})

    def run_clips(self) -> None:
        out = self.root / "work" / "art" / "clips"
        made = 0
        while made < self.args.clips:
            job = self.clipq.get()
            if job is None:
                break
            if self.stop_clips and datetime.now(timezone.utc) >= self.stop_clips:
                self.ctx.log.info("art prove: clip window over, no new clips")
                break
            try:
                made += 1
                self.one_clip(job, out)
            except Exception as e:  # noqa: BLE001
                self.ctx.log.warn("art prove clip failed", error=f"{type(e).__name__}: {str(e)[:300]}")

    def one_clip(self, job: dict[str, Any], out: Path) -> None:
        from ..models import Shot, ShotType
        kind, item = job["kind"], job["item"]
        if kind == "illustrated_map":
            spec = job["spec"]
            dur = 6.0
            shot = Shot(id=f"map_{item['report']['tag']}", beat_id="", start=item["start"], end=item["start"] + dur,
                        duration=dur, type=ShotType.motion_graphic, intent="illustrated map")
            payload = spec.typed_payload()
            res = C.make_clip(shot, Path(job["still"]), kind, self.ctx, out, payload=payload,
                              window=item.get("window", ""))
            if res.get("ok") and res.get("raw"):
                res["final"] = self.render_map(spec, job["still"], out, shot.id, clip=res["raw"])
        else:
            shot = job["shot"]
            res = C.make_clip(shot, Path(job["still"]), kind, self.ctx, out,
                              window=common.shot_text(self.narr, shot), subject=job.get("subject", ""))
        res.pop("candidate", None)
        self.add({"type": "clip", "kind": kind, "source_item": item.get("shot_id") or item.get("report", {}).get("tag"),
                  "start": item.get("start"), "window": item.get("window") or common.shot_text(self.narr, shot),
                  "still": job["still"], **res})

    # ------------------------------------------------------------ run

    def run(self) -> None:
        if self.spec is None:
            raise SystemExit("no Elevate image seat configured")
        self.seed_ai_clips(self.args.ai_clips)
        if self.args.only == "maps":
            self.run_maps()
            self.clipq.put(None)
            self.run_clips()  # painted maps that pass come alive too
            return
        if self.args.only == "extra-illustrations":
            self.run_illustrations()
            self.clipq.put(None)
            self.run_clips()  # its passing illustrations come alive too, one clip lane shared by every process
            return
        if self.args.only == "ai-clips":
            self.clipq.put(None)
            self.run_clips()
            common.write_report(self.ctx, "art_report_ai_clips.json", self.items)
            return
        t_clip = threading.Thread(target=self.run_clips, name="clips")
        t_clip.start()
        workers = [threading.Thread(target=self.run_maps, name="maps"),
                   threading.Thread(target=self.run_illustrations, name="ills")]
        for t in workers:
            t.start()
        for t in workers:
            t.join()
        self.clipq.put(None)
        t_clip.join()
        common.write_report(self.ctx, "art_report.json", self.items)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--maps", type=int, default=8)
    ap.add_argument("--illustrations", type=int, default=10)
    ap.add_argument("--clips", type=int, default=14)
    ap.add_argument("--ai-clips", type=int, default=4)
    ap.add_argument("--image-cap", type=float, default=300.0)
    ap.add_argument("--stop-clips-at", default="")
    ap.add_argument("--skip-places", nargs="*", default=[], help="place names the gazetteer resolves wrongly")
    ap.add_argument("--only", default="", choices=["", "ai-clips", "extra-illustrations", "maps"])
    args = ap.parse_args(argv)
    Prove(args).run()


if __name__ == "__main__":
    main()
