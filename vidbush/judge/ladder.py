"""The ladder for clauses where the specific picture was not found, in Adam's order:

0. (off by default) re-source once with the picker's queries into the same pool and ask the picker again;
1. the picker's fitting picture: a real picture of the same situation (period, season, setting,
   circumstance), shown plain with no text;
2. the director restates the clause as a layout, kept only when the layout is complete (a number or a date
   over a real picture, two real pictures side by side, a map whose places all resolve);
3. hold or extend a neighbouring real picture the picker said still fits, up to HOLD_FILL_MAX_S in total;
4. an AI still where Adam's rulings allow it (never a named real person or a document, never in the opening
   15 s, at most two in a row, at most 4 s each);
5. the picker's loosely related real picture, plain.

The narration's words are never typed over a picture in place of the missing one (Adam). Every clause
records which step filled it in pick["filled_by"]: pick, resource, fitting, restate, hold, ai_still, loose,
or none (the shot is flagged nothing_fits and the live fill holds a neighbour).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from ..models import Candidate, Narration
from ..plan import anchors as A
from ..plan import edit as E
from ..plan import vocab as V
from . import describe as DS
from . import pick_prompts as PP

RESTATE_SEAT, RESTATE_TASK, STAGE = "director", "plan_restate", "judge"
RESTATE_RETRIES = 1
MAX_TEXT_WORDS = 6


def place_resolves(name: str) -> bool:
    """The map renderer can place it (the same gazetteer lookup the route map uses)."""
    from ..mg.layouts import routemap as RM
    try:
        return RM.resolve(*RM.split_place(name)) is not None
    except Exception:  # noqa: BLE001 - an unknown place is a place we cannot draw
        return False


def missing_document(p: dict, plan: dict) -> bool:
    """A document quoted as evidence or subject whose page is not in the pool: its words go over footage."""
    return (E.needs_document(plan) and p.get("_doc") is None
            and (plan.get("layout") or {}).get("purpose") in V.MARKED_PURPOSES)


def is_open(p: dict, plan: dict) -> bool:
    """Still needs the ladder: a picture is needed and none was picked, or a quoted document was not found;
    and no ladder step has filled it yet (resolve marks a clause needing no picture as filled by pick)."""
    unfilled = p.get("filled_by") in ("", None) or (p.get("filled_by") == "pick" and not p.get("fallback")
                                                    and missing_document(p, plan))
    needs = (E.needs_pick(plan) and p.get("_cand") is None) or missing_document(p, plan)
    return unfilled and needs


def open_clauses(e: dict) -> list[int]:
    return [i for i, (p, s) in enumerate(zip(e["picks"], e["shots"])) if is_open(p, s["plan"])]


def _ai_run_around(e: dict, i: int, ai_tail: int) -> int:
    """AI stills in a row if shot i became one (counting the previous scene's tail)."""
    def ai(k: int) -> bool:
        p = e["picks"][k]
        c = p.get("_cand")
        return (c is not None and E.is_ai(c)) or p.get("fallback") == "ai_still"
    before = 0
    k = i - 1
    while k >= 0 and ai(k):
        before, k = before + 1, k - 1
    if k < 0:
        before += ai_tail
    after = 0
    k = i + 1
    while k < len(e["picks"]) and ai(k):
        after, k = after + 1, k + 1
    return before + 1 + after


def _names(rec: dict, rt: dict) -> bool:
    """The record's facts or visible text contain every word of the real thing's name."""
    name = set(E.content_words(rt.get("name", "")))
    return bool(name) and name <= set(E.content_words(E.record_text(rec)))


def _used(e: dict, used_before: dict[str, str]) -> dict[str, str]:
    used = dict(used_before)
    for p in e["picks"]:
        for c in [p.get("_cand"), p.get("_doc")] + [x["_cand"] for x in p.get("element_picks", [])]:
            if c is not None:
                used.setdefault(E.asset_key(c), p["clause"])
    return used


class Ladder:
    """Callable handed to edit_all; resource({clause: {queries, medium, subject, real_thing}}, scene_id)
    returns new candidates (vidbush/judge/resource.py), or is None to skip step 1."""

    def __init__(self, seats, settings: Any, sheet_dir, log: Optional[Callable] = None,
                 resource: Optional[Callable[[dict[str, dict], str], list[Candidate]]] = None) -> None:
        self.seats, self.settings, self.sheet_dir, self.log = seats, settings, sheet_dir, log
        self.resource = resource
        self.stats: dict[str, int] = {}

    def __call__(self, e: dict, scene_plan: dict, narration: Narration, used_before: dict[str, str],
                 ai_tail: int, previous: list[dict]) -> None:
        if open_clauses(e) and self.resource is not None:
            self.step_resource(e, scene_plan, narration, used_before, ai_tail, previous)
        for i in open_clauses(e):
            self.step_alternate(e, i, "_fit", "fitting", used_before, ai_tail)
        if open_clauses(e):
            self.step_restate(e, scene_plan, narration, used_before)
        for i in open_clauses(e):
            self.step_hold(e, i)
        for i in open_clauses(e):
            self.step_ai(e, i, ai_tail)
        for i in open_clauses(e):
            self.step_alternate(e, i, "_loose", "loose", used_before, ai_tail)
        for p, s in zip(e["picks"], e["shots"]):
            if is_open(p, s["plan"]):
                p["filled_by"] = "none"
            elif not p.get("filled_by"):
                p["filled_by"] = "pick"
            self.stats[p["filled_by"]] = self.stats.get(p["filled_by"], 0) + 1

    # ------------------------------------------------------------ 1. re-source once
    def step_resource(self, e, scene_plan, narration, used_before, ai_tail, previous) -> None:
        idx = open_clauses(e)
        queries = {}
        for i in idx:
            qs = [str(q) for q in (e["picks"][i].get("search_again") or []) if str(q).strip()][:3]
            plan = e["shots"][i]["plan"]
            if qs:
                queries[e["picks"][i]["clause"]] = {"queries": qs, "medium": plan.get("medium", "either"),
                                                    "subject": plan.get("subject", ""),
                                                    "real_thing": plan.get("real_thing")}
        if not queries:
            return
        try:
            new = self.resource(queries, scene_plan["scene"]) or []
        except Exception as ex:  # noqa: BLE001 - a failed search leaves the next steps of the ladder
            if self.log:
                self.log(f"{scene_plan['scene']} re-source failed: {str(ex)[:200]}")
            new = []
        have = {E.asset_key(c) for c in e["pool"]}
        new = [c for c in new if E.asset_key(c) not in have and E._usable(c)]
        e["resourced"] = {"queries": queries, "new_assets": len(new)}
        if not new:
            return
        recs = DS.describe_pool(self.seats, new, self.settings, self.log)
        e["records"].update(recs)
        new = [c for c in new if c.id in recs and not DS.blocked(recs[c.id]) and E.seeable(recs[c.id])]
        if not new:
            return
        # the re-pick sees the new assets first, then the old pool, within the same image cap
        merged = new + [c for c in e["pool"] if c not in new]
        e["pool"] = merged[:E.MAX_ASSETS] if len(merged) > E.MAX_ASSETS else merged
        if len(merged) > E.MAX_ASSETS and self.log:
            self.log(f"{scene_plan['scene']} re-pick pool capped at {E.MAX_ASSETS}, {len(merged) - E.MAX_ASSETS} left out")
        sub = [e["shots"][i] for i in idx]
        used = _used(e, used_before)
        picks, attempts, _ = E.pick_call(self.seats, scene_plan.get("scene_purpose", ""), sub, narration, e["pool"],
                                         e["records"], self.sheet_dir, scene_plan["scene"], used, ai_tail, previous,
                                         self.log, tag="_r")
        e["attempts"] += [dict(a, step="resource") for a in attempts]
        for i, p in zip(idx, picks):
            if p.get("_cand") is None:
                e["picks"][i]["search_again_reason"] = p.get("reason", "")
                continue
            old = e["picks"][i]
            e["picks"][i] = p
            # the re-pick saw these shots alone: check AI runs and holds against the real neighbours
            why = self._neighbour_problem(e, i, ai_tail)
            if why:
                old["resolved"] = (old.get("resolved", "") + f"; re-sourced pick {why}").strip("; ")
                e["picks"][i] = old
            else:
                p["filled_by"] = "resource"

    def _neighbour_problem(self, e: dict, i: int, ai_tail: int) -> str:
        p = e["picks"][i]
        c = p["_cand"]
        if E.is_ai(c) and _ai_run_around(e, i, ai_tail) > V.AI_RUN_MAX:
            return "removed: too many AI stills in a row"
        dur = E.shot_seconds(e["shots"][i])
        for j in (i - 1, i + 1):
            if 0 <= j < len(e["picks"]):
                n = e["picks"][j].get("_cand")
                if n is not None and E.asset_key(n) == E.asset_key(c):
                    if dur + E.shot_seconds(e["shots"][j]) > E._limit(c) + 1e-6:
                        return "removed: with the same asset next to it, it stays on screen over the limit"
        return ""

    # ------------------------------------------------------------ 2. restate as a layout
    def step_restate(self, e, scene_plan, narration, used_before) -> None:
        # a missing document is never restated: no layout stands in for a page (the shot keeps a picture)
        idx = [i for i in open_clauses(e) if not E.needs_document(e["shots"][i]["plan"])]
        if not idx:
            return
        rows = []
        for i in idx:
            r = E.clause_row(e["shots"][i], narration)
            r["why_nothing"] = e["picks"][i].get("reason", "")
            rows.append(r)
        real = [(n, c) for n, c in enumerate(e["pool"], 1) if not E.is_ai(c)]
        pool = [{"n": n, "media": c.media, "shows": e["records"][c.id]["visual"].get("shows", ""),
                 "title": c.title[:100], "used": E.asset_key(c) in _used(e, used_before)} for n, c in real]
        base = PP.restate_prompt(scene_plan.get("scene_purpose", ""), rows, pool)
        prompt = base
        for attempt in range(RESTATE_RETRIES + 1):
            raw = self.seats.llm_json(RESTATE_SEAT, RESTATE_TASK, PP.RESTATE_SYSTEM, prompt, STAGE)
            done, errors = self._apply_restate(e, raw, idx, narration, used_before, commit=False)
            e.setdefault("restate_attempts", []).append({"attempt": attempt, "errors": errors})
            if not errors:
                break
            prompt = base + "\n\nYour previous reply and the problems found in it:\n" + str(raw)[:8000] + \
                "\n\nProblems:\n" + "\n".join(f"- {x}" for x in errors)
        self._apply_restate(e, raw, idx, narration, used_before, commit=True)

    def _apply_restate(self, e, raw, idx, narration, used_before, commit: bool) -> tuple[int, list[str]]:
        items = raw.get("restated") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return 0, ["The reply needs a restated list."]
        by = {str(i.get("clause")): i for i in items if isinstance(i, dict)}
        errors: list[str] = []
        used = _used(e, used_before)
        done = 0
        for i in idx:
            p, s = e["picks"][i], e["shots"][i]
            cid = p["clause"]
            r = by.get(cid)
            if not r or not r.get("kind"):
                continue
            kind = str(r["kind"])
            dur = E.shot_seconds(s)
            cw = s["clauses"]
            words = narration.words[cw[0].words[0]:cw[-1].words[1] + 1]
            text = " ".join(str(r.get("text", "")).split())
            anchor = str(r.get("anchor", ""))
            errs = []
            if kind not in PP.RESTATE_KINDS:
                errs.append(f"{cid}: kind must be one of {', '.join(PP.RESTATE_KINDS)}.")
            at = A.time_of(words, anchor) if anchor else None
            if kind != "route_map" and at is None:
                errs.append(f"{cid}: anchor '{anchor}' is not spoken in this clause.")
            if kind not in ("route_map", "split_comparison") and not (0 < len(text.split()) <= MAX_TEXT_WORDS):
                errs.append(f"{cid}: text needs 1 to {MAX_TEXT_WORDS} words.")
            if kind == "number_over_footage" and not re.search(r"\d", text):
                errs.append(f"{cid}: number_over_footage text needs the number (in digits) with its unit.")
            beneath = None
            rt = s["plan"].get("real_thing") or {}
            if kind in ("number_over_footage", "date_title"):
                beneath = E._cand_at(e["pool"], r.get("beneath"))
                if beneath is None or E.is_ai(beneath):
                    errs.append(f"{cid}: beneath must be a real asset number from the pool.")
                elif E.asset_key(beneath) in used:
                    errs.append(f"{cid}: asset {r.get('beneath')} is already used at {used[E.asset_key(beneath)]}.")
                else:
                    _, twhy = E.time_problem(beneath, e["records"].get(beneath.id, {}), None, dur)
                    if twhy:
                        errs.append(f"{cid}: asset {r.get('beneath')}: {twhy}.")
                    if beneath.media != "video" and dur > V.HOLD_MAX_S + 1e-6:
                        errs.append(f"{cid}: a still under text for {dur:.1f} s is over the {V.HOLD_MAX_S:g} s "
                                    f"limit; choose footage.")
                    if rt.get("name") and not _names(e["records"].get(beneath.id, {}), rt):
                        errs.append(f"{cid}: asset {r.get('beneath')} would be read as {rt['name']} but its record "
                                    f"does not name it; a look-alike is refused. Choose footage whose record names "
                                    f"it, or a route_map or date_title over something else.")
            pictures: list = []
            if kind == "split_comparison":
                pictures = [E._cand_at(e["pool"], n) for n in (r.get("pictures") or [])][:2]
                if len(pictures) != 2 or any(c is None or E.is_ai(c) or E.asset_key(c) in used for c in pictures):
                    errs.append(f"{cid}: split_comparison needs pictures, two real unused asset numbers.")
                    pictures = []
                if "/" not in text:
                    errs.append(f"{cid}: split_comparison text is 'left words / right words'.")
            places = [str(x) for x in (r.get("places") or []) if str(x).strip()]
            if kind == "route_map" and not places:
                errs.append(f"{cid}: route_map needs places.")
            unknown = [pl for pl in places if not place_resolves(pl)] if kind == "route_map" else []
            if unknown:
                errs.append(f"{cid}: route_map places {', '.join(unknown)} are not on our map; a map with a "
                            f"missing place is not drawn. Choose another kind or set kind null.")
            errors += [x for x in errs if x]
            if errs or not commit:
                continue
            done += 1
            self._commit(e, i, kind, text, anchor, at, beneath, places, str(r.get("reason", "")), pictures)
            for c in [beneath] + pictures:
                if c is not None:
                    used.setdefault(E.asset_key(c), cid)
        return done, errors

    def _commit(self, e, i, kind, text, anchor, at, beneath, places, reason, pictures=()) -> None:
        """Apply a restated clause. The scene plan's clause dict is updated in place, because emit reads it."""
        p, s = e["picks"][i], e["shots"][i]
        plan = s["plan"]
        plan["planned_treatment"] = plan["treatment"]
        plan["restate_reason"] = reason
        dur = E.shot_seconds(s)
        start_at = at if at is not None else s["clauses"][0].speech_start

        def play_from(c):
            t = E.time_problem(c, e["records"].get(c.id, {}), None, dur)[0]
            return t if t is not None else "a"

        if kind == "route_map":
            plan.update(treatment="route_map", presentation="full", layout={"title": " to ".join(places), "elements": [
                {"role": "place", "text": pl, "anchor": "", "at": start_at} for pl in places]})
        elif kind == "split_comparison":
            plan.update(treatment="split_comparison", layout={"title": "", "elements": [
                {"role": side, "text": t, "anchor": anchor, "at": start_at, "picture_subject": "", "stage": 1}
                for side, t in zip(("left", "right"), [x.strip() for x in text.split("/")] + [""])]})
            p["element_picks"] = [{"element": k, "pick": e["pool"].index(c) + 1, "frame": play_from(c), "_cand": c}
                                  for k, c in enumerate(pictures)]
        else:
            role = "value" if kind == "number_over_footage" else "date"
            plan.update(treatment=kind, layout={"title": "", "elements": [
                {"role": role, "text": text, "anchor": anchor, "at": start_at, "picture_subject": "", "stage": 1}]})
            p["_cand"] = beneath
            p["pick"] = e["pool"].index(beneath) + 1
            p["start"] = p["frame"] = play_from(beneath)
        p["filled_by"] = "restate"
        p["restated"] = {"kind": kind, "text": text, "anchor": anchor, "places": places, "reason": reason}

    # ------------------------------------------------------------ 1 and 5. the picker's alternates
    def step_alternate(self, e: dict, i: int, key: str, step: str, used_before: dict[str, str],
                       ai_tail: int) -> None:
        """The picker's fitting (step 1) or loose (step 5) real picture, plain, when no other clause of the
        film shows that asset yet and it breaks no neighbour rule."""
        p = e["picks"][i]
        if p.get("_cand") is not None:  # a missing document over its own picture: the picture stays
            p["filled_by"] = "pick_no_document"
            return
        alt = p.get(key)
        if not alt:
            return
        c, start = alt
        if E.asset_key(c) in _used(e, used_before):
            return
        p.update(_cand=c, start=start, frame=start, label="", filled_by=step)
        if self._neighbour_problem(e, i, ai_tail):
            p.update(_cand=None, start=None, frame=None, filled_by="")

    # ------------------------------------------------------------ 3. hold a neighbour
    def step_hold(self, e: dict, i: int) -> None:
        """Hold the previous real picture over this clause, or start the next one early, when the picker said
        that picture still fits and the two together stay within HOLD_FILL_MAX_S (and inside the clip)."""
        p, s = e["picks"][i], e["shots"][i]
        dur = E.shot_seconds(s)
        side = p.get("hold")
        j = i - 1 if side == "previous" else i + 1 if side == "next" else None
        if j is None or not 0 <= j < len(e["picks"]):
            return
        n, ns = e["picks"][j], e["shots"][j]
        c = n.get("_cand")
        if c is None or E.is_ai(c) or n.get("fallback") or n.get("filled_by") == "hold":
            return
        if side == "next" and ns["plan"]["treatment"] != "picture":
            return  # a layout's cut stays on its words
        total = E.shot_seconds(ns) + dur
        if total > V.HOLD_FILL_MAX_S + 1e-6:
            return
        if c.media == "video":
            start = n.get("start") if isinstance(n.get("start"), (int, float)) else c.in_point
            if side == "next":
                start = max(c.in_point, start - dur)
            if start + total > c.out_point + 0.05:
                return
            if side == "next":
                n["start"] = n["frame"] = round(start, 3)
        p["fallback"] = "hold_previous" if side == "previous" else "extend_next"
        p["filled_by"] = "hold"

    # ------------------------------------------------------------ 4. AI still
    def step_ai(self, e: dict, i: int, ai_tail: int) -> None:
        p, s = e["picks"][i], e["shots"][i]
        if p.get("_cand") is not None:
            return
        rt = s["plan"].get("real_thing") or {}
        if (rt.get("kind") not in E.NO_AI_KINDS and not E.needs_document(s["plan"])
                and E.shot_seconds(s) <= V.AI_HOLD_MAX_S + 1e-6 and E.ai_allowed_at(s["clauses"][0].start)
                and _ai_run_around(e, i, ai_tail) <= V.AI_RUN_MAX):
            p["fallback"] = "ai_still"
            p["filled_by"] = "ai_still"
