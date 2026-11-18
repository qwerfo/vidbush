"""The ladder for clauses where nothing in the pool fits, in this order:

1. re-source once, with the queries the picker wrote from what the pool contained and what was missing,
   into the same pool; new assets are described and the picker is asked again for those clauses only;
2. ask the director to restate the clause as a layout that needs no missing picture (a number over real
   footage, a comparison of things we do have, a date title, a route map, the quoted words set as text
   over related real footage);
3. hold the previous real picture when the clause is short and the hold stays inside the limit;
4. an AI still, only where Adam's rulings allow it (no named real person, real event or document; at most
   two in a row; at most 4 s).

Every clause records which step filled it in pick["filled_by"]: pick, resource, restate, hold, ai_still, or
none (the shot is flagged nothing_fits and left for the operator).
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

SHORT_CLAUSE_S = 3.0  # a hold of the previous picture reads as continuous only over a short clause
RESTATE_SEAT, RESTATE_TASK, STAGE = "director", "plan_restate", "judge"
RESTATE_RETRIES = 1
MAX_TEXT_WORDS = 6


def open_clauses(e: dict) -> list[int]:
    return [i for i, (p, s) in enumerate(zip(e["picks"], e["shots"]))
            if E.needs_pick(s["plan"]) and p.get("_cand") is None and not p.get("filled_by")]


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


def _used(e: dict, used_before: dict[str, str]) -> dict[str, str]:
    used = dict(used_before)
    for p in e["picks"]:
        for c in [p.get("_cand"), p.get("_fallback_cand"), p.get("_doc")] + [x["_cand"] for x in p.get("element_picks", [])]:
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
        if open_clauses(e):
            self.step_restate(e, scene_plan, narration, used_before)
        for i in open_clauses(e):
            self.step_hold_or_ai(e, i, ai_tail)
        for p, s in zip(e["picks"], e["shots"]):
            if not p.get("filled_by"):
                p["filled_by"] = "none" if E.needs_pick(s["plan"]) else "pick"
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
        new = [c for c in new if c.id in recs and not DS.blocked(recs[c.id])]
        e["pool"] = e["pool"] + new
        if not new:
            return
        sub = [e["shots"][i] for i in idx]
        used = _used(e, used_before)
        picks, attempts, _ = E.pick_call(self.seats, scene_plan.get("scene_purpose", ""), sub, narration, e["pool"],
                                         e["records"], self.sheet_dir, scene_plan["scene"], used, ai_tail, previous,
                                         self.log, tag="_r")
        e["attempts"] += [dict(a, step="resource") for a in attempts]
        for i, p in zip(idx, picks):
            if p.get("_cand") is not None:
                p["filled_by"] = "resource"
                e["picks"][i] = p
            else:
                e["picks"][i]["search_again_reason"] = p.get("reason", "")

    # ------------------------------------------------------------ 2. restate as a layout
    def step_restate(self, e, scene_plan, narration, used_before) -> None:
        idx = open_clauses(e)
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
            if kind in ("number_over_footage", "date_title", "text_over_footage"):
                beneath = E._cand_at(e["pool"], r.get("beneath"))
                rt = s["plan"].get("real_thing") or {}
                if beneath is None or E.is_ai(beneath):
                    errs.append(f"{cid}: beneath must be a real asset number from the pool.")
                elif E.asset_key(beneath) in used:
                    errs.append(f"{cid}: asset {r.get('beneath')} is already used at {used[E.asset_key(beneath)]}.")
                else:
                    _, twhy = E.time_problem(beneath, e["records"].get(beneath.id, {}), None, dur)
                    if twhy:
                        errs.append(f"{cid}: asset {r.get('beneath')}: {twhy}.")
                    if rt.get("kind") == "document" and kind != "text_over_footage":
                        errs.append(f"{cid}: a missing document is stated as its quoted words over footage "
                                    f"(text_over_footage).")
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

        if kind == "text_over_footage":
            p["fallback"] = "text_over_footage"
            p["_fallback_cand"] = beneath
            p["overlay_text"] = text
            p["frame"] = play_from(beneath)
        elif kind == "route_map":
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

    # ------------------------------------------------------------ 3. hold, 4. AI still
    def step_hold_or_ai(self, e: dict, i: int, ai_tail: int) -> None:
        p, s = e["picks"][i], e["shots"][i]
        dur = E.shot_seconds(s)
        prev = e["picks"][i - 1] if i > 0 else None
        pc = prev.get("_cand") if prev else None
        if pc is not None and not E.is_ai(pc) and dur <= SHORT_CLAUSE_S + 1e-6:
            prev_dur = E.shot_seconds(e["shots"][i - 1])
            if prev_dur + dur <= V.HOLD_MAX_S + 1e-6:
                p["fallback"] = "hold_previous"
                p["filled_by"] = "hold"
                return
        rt = s["plan"].get("real_thing") or {}
        if (rt.get("kind") not in E.NO_AI_KINDS and dur <= V.AI_HOLD_MAX_S + 1e-6
                and _ai_run_around(e, i, ai_tail) <= V.AI_RUN_MAX):
            p["fallback"] = "ai_still"
            p["filled_by"] = "ai_still"
