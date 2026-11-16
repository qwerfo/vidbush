"""Part 2 of the clause-first planner: one director call per scene, validated, errors fed back on retry.

Code checks only what is physical or legal (every clause planned once, known treatment names, anchors that are
actually spoken in the shot, reading time, hold limits). The model decides content.
"""
from __future__ import annotations

from typing import Any, Optional

from ..models import Narration, Script
from . import anchors as A
from . import prompts as P
from . import vocab as V
from .clauses import Clause, clause_words

RETRIES = 2
SEAT, TASK, STAGE = "director", "plan_direct", "shotlist"


def _shot_groups(plans: list[dict], clauses: list[Clause]) -> list[list[int]]:
    """Indices of clauses grouped into shots: a clause plus the holds that follow it."""
    groups: list[list[int]] = []
    for i, p in enumerate(plans):
        if p.get("treatment") == "hold" and groups:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _words_of(group: list[int], clauses: list[Clause], narration: Narration):
    lo = clauses[group[0]].words[0]
    hi = clauses[group[-1]].words[1]
    return narration.words[lo:hi + 1]


def _has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


def _is_year(s: str) -> bool:
    t = s.strip().rstrip("s").replace("'", "")
    return t.isdigit() and 1000 <= int(t) <= 2100


def validate(raw: Any, clauses: list[Clause], narration: Narration) -> tuple[list[dict], list[str]]:
    """Normalised clause plans in clause order plus a list of plain-language problems."""
    errors: list[str] = []
    if not isinstance(raw, dict) or not isinstance(raw.get("clauses"), list):
        errors.append("The reply must be an object with scene_purpose and a clauses list.")
        raw = {"scene_purpose": "x", "clauses": []}  # every clause is then planned as missing
    if not str(raw.get("scene_purpose", "")).strip():
        errors.append("scene_purpose is missing.")
    by_id: dict[str, dict] = {}
    for item in raw["clauses"]:
        if isinstance(item, dict) and item.get("id"):
            if item["id"] in by_id:
                errors.append(f"{item['id']} appears twice.")
            by_id[str(item["id"])] = item
    ids = [c.id for c in clauses]
    missing = [i for i in ids if i not in by_id]
    extra = [i for i in by_id if i not in ids]
    if missing:
        errors.append(f"These clauses are missing: {', '.join(missing)}. Plan every clause.")
    if extra:
        errors.append(f"These ids are not clauses of this scene: {', '.join(extra)}.")
    plans: list[dict] = []
    for n, c in enumerate(clauses):
        p = dict(by_id.get(c.id) or {"id": c.id, "treatment": "picture", "subject": "", "_missing": True})
        t = str(p.get("treatment", "")).strip()
        if t not in V.TREATMENTS:
            errors.append(f"{c.id}: treatment '{t}' is not one of {', '.join(V.TREATMENTS)}.")
            t = "picture"
        if t == "hold" and n == 0:
            errors.append(f"{c.id}: the first clause of a scene cannot be a hold.")
            t = "picture"
        p["treatment"] = t
        if t == "hold":
            if not str(p.get("reason", "")).strip():
                errors.append(f"{c.id}: a hold needs a reason.")
        else:
            if not str(p.get("subject", "")).strip():
                errors.append(f"{c.id}: subject is empty.")
            anchor = str(p.get("anchor", "")).strip()
            if not anchor or A.find(clause_words(c, narration), anchor) is None:
                errors.append(f"{c.id}: anchor '{anchor}' is not words of this clause; copy words from: "
                              f"\"{c.text}\".")
            if p.get("medium") not in V.MEDIA:
                p["medium"] = "either"
        if t in V.PRESENTED_KINDS:
            if p.get("presentation") not in V.PRESENTATIONS:
                errors.append(f"{c.id}: {t} needs presentation, one of {', '.join(V.PRESENTATIONS)}.")
        else:
            p.pop("presentation", None)
        rt = p.get("real_thing")
        if isinstance(rt, dict) and rt.get("name"):
            if rt.get("kind") not in V.REAL_KINDS:
                errors.append(f"{c.id}: real_thing.kind must be one of {', '.join(V.REAL_KINDS)}.")
        else:
            p["real_thing"] = None
        q = p.get("queries")
        p["queries"] = {k: [str(x) for x in v if str(x).strip()][:3] for k, v in q.items()
                        if k in V.SOURCE_CLASSES and isinstance(v, list)} if isinstance(q, dict) else {}
        if t == "picture" and not any(p["queries"].values()):
            errors.append(f"{c.id}: a picture needs queries for at least one source class.")
        g = [x for x in (p.get("grammar") or []) if x in V.GRAMMAR]
        p["grammar"] = g
        cb = str(p.get("callback_to") or "")
        if "callback" in g and not (len(cb) == 4 and cb[0] == "c" and cb[1:].isdigit() and cb < c.id):
            errors.append(f"{c.id}: callback needs callback_to, the id of the earlier clause it returns to.")
        plans.append(p)
    # shot level: hold limits, layout elements spoken inside the shot, reading time
    for g in _shot_groups(plans, clauses):
        head = plans[g[0]]
        t = head["treatment"]
        c0, c1 = clauses[g[0]], clauses[g[-1]]
        dur = c1.end - c0.start
        limit = V.HOLD_MAX_S if t == "picture" else V.LAYOUT_HOLD_MAX_S
        if len(g) > 1 and dur > limit:
            errors.append(f"{c0.id} to {c1.id}: holding one {t} for {dur:.1f} s is over the {limit:g} s limit; "
                          f"end the hold earlier or start a new picture.")
        if t in V.LAYOUT_KINDS:
            lay = head.get("layout") if isinstance(head.get("layout"), dict) else {}
            els = [e for e in (lay.get("elements") or []) if isinstance(e, dict)]
            head["layout"] = {"title": str(lay.get("title", "")), "timeline_id": str(lay.get("timeline_id", "")),
                              "elements": els}
            if t == "timeline" and not head["layout"]["timeline_id"]:
                errors.append(f"{c0.id}: a timeline needs a timeline_id.")
            if not els:
                errors.append(f"{c0.id}: layout {t} has no elements.")
            words = _words_of(g, clauses, narration)
            last = -1
            for k, e in enumerate(els):
                text = str(e.get("text", "")).strip()
                if not text:
                    errors.append(f"{c0.id} element {k}: text is empty.")
                if len(text.split()) > 6:
                    errors.append(f"{c0.id} element {k}: '{text}' is over 6 words.")
                if _has_digit(text) and not _is_year(text) and not any(ch.isalpha() or ch in "%$" for ch in text):
                    errors.append(f"{c0.id} element {k}: the number '{text}' needs its unit.")
                try:
                    e["stage"] = int(e.get("stage", 1))
                except (TypeError, ValueError):
                    e["stage"] = 1
                if e["stage"] not in (1, 2) or (e["stage"] == 2 and head.get("presentation") != "panel_then_expand"):
                    errors.append(f"{c0.id} element {k}: stage 2 is only for panel_then_expand.")
                idx = A.find(words, str(e.get("anchor", "")))
                if idx is None:
                    errors.append(f"{c0.id} element {k}: anchor '{e.get('anchor', '')}' is not spoken in this "
                                  f"shot (\"{' '.join(w.text for w in words)}\").")
                    continue
                if idx < last:
                    errors.append(f"{c0.id} element {k}: elements must be in spoken order.")
                last = idx
                on = words[idx].start
                read = max(V.READ_MIN_S, V.READ_S_PER_WORD * len(text.split()))
                if c1.end - on < read:
                    errors.append(f"{c0.id} element {k}: '{text}' appears {c1.end - on:.1f} s before the cut, "
                                  f"under the {read:.1f} s needed to read it; anchor it earlier or hold longer.")
                e["at"] = round(on, 3)
    return plans, errors


def repair(plans: list[dict], clauses: list[Clause], narration: Narration) -> list[dict]:
    """Last resort after the retries: make the plan legal without inventing content."""
    out = []
    for n, (p, c) in enumerate(zip(plans, clauses)):
        p = dict(p)
        notes = []
        if p.get("_missing"):
            prev = next((q for q in reversed(out) if q["treatment"] != "hold"), None)
            p.update(treatment="hold" if prev else "picture", reason="not planned by the director",
                     subject=(prev or {}).get("subject", c.text))
            notes.append("missing")
        if p["treatment"] not in ("picture", "hold"):
            lay = p.get("layout") or {}
            words = clause_words(c, narration)
            good = [e for e in lay.get("elements", []) if "at" in e]
            if not good and not any(A.find(words, str(e.get("anchor", ""))) is not None
                                    for e in lay.get("elements", [])):
                notes.append(f"layout {p['treatment']} had no valid element; shown as a picture")
                p["treatment"] = "picture"
        if p["treatment"] in V.PRESENTED_KINDS and p.get("presentation") not in V.PRESENTATIONS:
            p["presentation"] = "full"
            notes.append("presentation defaulted to full")
        if p["treatment"] != "hold" and A.find(clause_words(c, narration), str(p.get("anchor", ""))) is None:
            p["anchor"] = c.text
            notes.append("anchor set to the clause")
        out.append(p)
        if notes:
            p["repairs"] = notes
    # break holds that run over the limit
    start = None
    for i, (p, c) in enumerate(zip(out, clauses)):
        if p["treatment"] != "hold" or start is None:
            start = i
            continue
        head = out[start]
        limit = V.HOLD_MAX_S if head["treatment"] == "picture" else V.LAYOUT_HOLD_MAX_S
        if c.end - clauses[start].start > limit:
            p.update(treatment="picture", subject=head.get("subject", ""), anchor=c.text,
                     queries=head.get("queries", {}), real_thing=head.get("real_thing"),
                     medium=head.get("medium", "either"), grammar=["sequence"],
                     grammar_note="hold split at the hold limit: closer framing of the same subject")
            p.setdefault("repairs", []).append("hold over the limit split into a new picture")
            start = i
    return out


def direct_scene(seats, script: Script, scene, clauses: list[Clause], narration: Narration,
                 earlier: list[dict], heroes: int, timelines: list[str], log=None) -> dict:
    user = P.director_user(script, scene, clauses, earlier, heroes, timelines)
    prompt = user
    attempts: list[dict] = []
    plans: list[dict] = []
    raw: Any = None
    errors: list[str] = []
    for attempt in range(RETRIES + 1):
        raw = seats.llm_json(SEAT, TASK, P.DIRECTOR_SYSTEM, prompt, STAGE)
        plans, errors = validate(raw, clauses, narration)
        attempts.append({"attempt": attempt, "errors": errors})
        if log:
            log(f"{scene.id} director attempt {attempt}: {len(errors)} problems")
        if not errors:
            break
        prompt = P.director_retry(user, raw, errors)
    if errors:
        plans = repair(plans, clauses, narration)
    purpose = str(raw.get("scene_purpose", "")) if isinstance(raw, dict) else ""
    return {"scene": scene.id, "scene_purpose": purpose, "clauses": plans, "attempts": attempts,
            "unresolved": errors}


def earlier_summary(scene_plans: list[dict]) -> list[dict]:
    out = []
    for sp in scene_plans:
        out.append({"scene": sp["scene"], "purpose": sp["scene_purpose"],
                    "showed": [{"id": p["id"], "treatment": p["treatment"], "subject": p.get("subject", "")[:90]}
                               for p in sp["clauses"] if p["treatment"] != "hold"]})
    return out


def direct_all(seats, script: Script, clauses: list[Clause], narration: Narration, log=None,
               only: Optional[set[str]] = None) -> list[dict]:
    out: list[dict] = []
    for scene in script.scenes:
        cl = [c for c in clauses if c.scene == scene.id]
        if not cl or (only and scene.id not in only):
            continue
        heroes = sum("hero" in p.get("grammar", []) for sp in out for p in sp["clauses"])
        tls = sorted({(p.get("layout") or {}).get("timeline_id", "") for sp in out for p in sp["clauses"]} - {""})
        out.append(direct_scene(seats, script, scene, cl, narration, earlier_summary(out), heroes, tls, log))
    return out
