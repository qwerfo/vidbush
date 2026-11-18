"""Part 3 of the clause-first planner: describe the pool once, then pick by comparison, one call per scene.

Every asset in a scene's pool has a description record (vidbush/judge/describe.py): what is visible, and
apart from it what sourcing knows. The picker seat sees the scene's clauses, the records and one numbered
image per asset, and for each clause picks an asset and the time to play from, or says nothing fits and
what to search for. Code then checks only what can be checked: Adam's AI rulings (no AI still for a named
real person, a real event or a document; at most two in a row; each at most 4 s), no accidental repeat of
an asset, that the picked time range exists, that a named real thing is supported by the record, that a
document's quoted words were found on the page, and that a label comes from the record. Problems go back
to the model once, in plain words; what is still wrong is removed and the clause goes down the ladder
(vidbush/judge/ladder.py).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

from ..judge import describe as DS
from ..judge import pick_prompts as PP
from ..models import Candidate, Narration, ShotList, SourceKind
from . import sheets as SH
from . import vocab as V
from .clauses import Clause

RETRIES = 1
SEAT, FALLBACK_SEAT, TASK, STAGE = "picker", "vision", "plan_pick", "judge"
FRAME_POS = {"a": 0.2, "b": 0.5, "c": 0.8}  # keyframe letters of source/keyframes.py, kept for old replays
NO_AI_KINDS = ("person", "event", "document")
MAX_ASSETS = 40  # assets per picker call (one image each): a larger pool is shortlisted before description
# by how many clause words its sourcing text shares, and the rest are logged; a cost and image-count cap
OPEN_LICENCE = re.compile(r"public domain|\bcc0\b|\bcc[- ]by\b(?!.*\bn[cd]\b)|creative commons|no known copyright",
                          re.I)
IDENTITY_FIELDS = ("title", "description", "page_url", "date", "text_in_frame", "ocr_text")
LABEL_FIELDS = IDENTITY_FIELDS
DOC_WORDS_FOUND = 0.7  # share of a passage's words that must be on the page
_STOP = set("the a an and or of to in on at for by with from as is was were be been his her their its this that "
            "these those it he she they we you i not no but so than then there here into over under".split())
_WORD = re.compile(r"[a-z0-9]+")


def is_ai(c: Candidate) -> bool:
    return c.source in (SourceKind.ai_image, SourceKind.ai_video)


def _usable(c: Candidate) -> bool:
    """Drop what the local prefilter rejected for physical or legal reasons, and mocks."""
    rr = (c.reject_reason or "").lower()
    if "mock" in c.flags or not c.local_path or not Path(c.local_path).exists():
        return False
    return not (rr.startswith("prefilter") or rr.startswith("resolution") or "burned_in" in rr)


def asset_key(c: Candidate) -> str:
    return (c.local_path or c.url or c.id).lower()


def scene_pool(old: ShotList, lo: float, hi: float,
               extra: Optional[list[tuple[Candidate, float, float]]] = None) -> list[Candidate]:
    """Candidates found for any stored shot overlapping [lo, hi), plus extra (candidate, start, end) found
    for that stretch elsewhere; one per asset."""
    seen: set[str] = set()
    pool = []
    found = [(c, s.start, s.end) for s in old.shots for c in s.candidates] + list(extra or [])
    for c, a, b in found:
        if b <= lo or a >= hi:
            continue
        k = asset_key(c)
        if k in seen or not _usable(c):
            continue
        seen.add(k)
        pool.append(c)
    return pool


def shots_of(plans: list[dict], clauses: list[Clause]) -> list[dict]:
    """One entry per shot: head clause plan plus the clauses its holds cover."""
    out: list[dict] = []
    for p, c in zip(plans, clauses):
        if p["treatment"] == "hold" and out:
            out[-1]["clauses"].append(c)
        else:
            out.append({"plan": p, "clauses": [c]})
    return out


def needs_pick(plan: dict) -> bool:
    """A picture on screen beneath (or as) this shot is needed."""
    t = plan["treatment"]
    if t == "picture" or t in V.OVER_PICTURE_KINDS:
        return True
    return t in V.PRESENTED_KINDS and plan.get("presentation") in ("panel", "full_then_dock", "panel_then_expand")


def needs_document(plan: dict) -> bool:
    return plan["treatment"] == "document_focus"


def shot_seconds(shot: dict) -> float:
    return shot["clauses"][-1].end - shot["clauses"][0].start


def clause_row(shot: dict, narration: Narration) -> dict:
    p, cl = shot["plan"], shot["clauses"]
    lo, hi = cl[0].words[0], cl[-1].words[1]
    row = {"clause": cl[0].id, "seconds": round(shot_seconds(shot), 2),
           "words": " ".join(w.text for w in narration.words[lo:hi + 1]),
           "subject": p.get("subject", ""), "treatment": p["treatment"], "needs_pick": needs_pick(p)}
    if needs_document(p):
        lay = p.get("layout") or {}
        row["needs_document_pick"] = True
        row["document_purpose"] = lay.get("purpose", "")
        row["passages"] = [e.get("text", "") for e in lay.get("elements", [])]
    for k in ("presentation", "real_thing", "medium", "grammar"):
        if p.get(k):
            row[k] = p[k]
    els = [{"element": k, "picture_subject": e.get("picture_subject", ""), "text": e.get("text", "")}
           for k, e in enumerate((p.get("layout") or {}).get("elements", [])) if e.get("picture_subject")]
    if els:
        row["element_pictures"] = els
    return row


# ---------------------------------------------------------------- record text checks


def words(s: str) -> list[str]:
    return _WORD.findall((s or "").lower())


def content_words(s: str) -> list[str]:
    return [w for w in words(s) if w not in _STOP and (len(w) > 2 or w.isdigit())]


def record_field(rec: dict, field: str) -> str:
    f = rec.get("source_facts", {})
    if field in ("text_in_frame",):
        return rec.get("visual", {}).get("text_in_frame", "")
    if field == "ocr_text":
        return rec.get("ocr_text", "")
    return str(f.get(field, ""))


def record_text(rec: dict, fields=IDENTITY_FIELDS) -> str:
    return " ".join(record_field(rec, f) for f in fields)


def _norm(s: str) -> str:
    return " ".join(words(s))


def identity_supported(rec: dict, real_thing: dict, quote: str, field: str) -> Optional[str]:
    """None when the quoted words are in the named record field and name the real thing; else the problem."""
    if field not in IDENTITY_FIELDS:
        return f"identity_field must be one of {', '.join(IDENTITY_FIELDS)}"
    q = _norm(quote)
    if not q or q not in _norm(record_field(rec, field)):
        return f"the identity words are not in the record's {field}"
    name = set(content_words(real_thing.get("name", ""))) | set(_WORD.findall(str(real_thing.get("date", ""))))
    if name and not name & set(content_words(quote)):
        return f"the identity words do not name {real_thing.get('name', 'the real thing')}"
    return None


def passages_found(rec: dict, passages: list[str]) -> list[bool]:
    """Per passage: are DOC_WORDS_FOUND of its words on the page (local text reader or visible text)?"""
    page = set(content_words(rec.get("ocr_text", "") + " " + rec.get("visual", {}).get("text_in_frame", "")))
    out = []
    for p in passages:
        w = content_words(p)
        out.append(bool(w) and sum(x in page for x in w) / len(w) >= DOC_WORDS_FOUND)
    return out


def label_supported(rec: dict, label: str) -> bool:
    """Every content word of the label is in the record's facts or visible text."""
    lw = content_words(label)
    have = set(content_words(record_text(rec, LABEL_FIELDS)))
    return bool(lw) and all(w in have for w in lw)


def open_licence(c: Candidate) -> bool:
    return bool(OPEN_LICENCE.search(c.licence or "")) and "nc_nd" not in c.flags


# ---------------------------------------------------------------- pool for the prompt


def shortlist(pool: list[Candidate], recs: dict[str, dict], rows: list[dict],
              limit: int = MAX_ASSETS) -> tuple[list[Candidate], list[Candidate]]:
    """At most limit assets: ranked by how many clause words their record shares (a cap the image count
    of one call needs, not a judgement of quality); returns (kept, dropped) keeping pool order."""
    if len(pool) <= limit:
        return pool, []
    want = set()
    for r in rows:
        want |= set(content_words(r.get("subject", "") + " " + r.get("words", "") + " "
                                  + str((r.get("real_thing") or {}).get("name", ""))))

    def score(c: Candidate) -> int:
        rec = recs.get(c.id)
        if rec is None:  # before description: what sourcing knows
            f = DS.source_facts(c)
            txt = " ".join(str(f.get(k, "")) for k in ("title", "description", "query", "page_url"))
        else:
            txt = record_text(rec) + " " + rec.get("visual", {}).get("shows", "")
        return len(want & set(content_words(txt)))

    ranked = sorted(pool, key=score, reverse=True)
    keep = {c.id for c in ranked[:limit]}
    return [c for c in pool if c.id in keep], [c for c in pool if c.id not in keep]


def prompt_record(n: int, c: Candidate, rec: dict, used_before: dict[str, str]) -> dict:
    vis = rec.get("visual", {})
    e: dict[str, Any] = {"n": n, "media": c.media, "ai": is_ai(c), "open_licence": open_licence(c),
                         "shows": vis.get("shows", ""), "people": vis.get("people", {}),
                         "place": vis.get("place", ""), "period": vis.get("period", ""), "look": vis.get("look", "")}
    if vis.get("text_in_frame"):
        e["text_in_frame"] = vis["text_in_frame"][:400]
    if rec.get("ocr_text"):
        e["ocr_text"] = rec["ocr_text"][:600]
    if vis.get("quality"):
        e["quality"] = vis["quality"]
    if c.media == "video":
        e["clip"] = [round(c.in_point, 2), round(c.out_point, 2)]
        e["frame_times"] = [f["t"] for f in rec.get("frames", [])]
        e["usable"] = vis.get("usable", [])
    sf = rec.get("source_facts", {})
    e["facts"] = {k: (v[:160] if isinstance(v, str) else v) for k, v in sf.items()
                  if k in ("title", "source", "date", "description", "page_url", "licence", "size")}
    if asset_key(c) in used_before:
        e["used_earlier_at"] = used_before[asset_key(c)]
    return e


# ---------------------------------------------------------------- code checks


def _cand_at(pool: list[Candidate], n: Any) -> Optional[Candidate]:
    try:
        i = int(n)
    except (TypeError, ValueError):
        return None
    return pool[i - 1] if 1 <= i <= len(pool) else None


def _limit(c: Candidate) -> float:
    return V.AI_HOLD_MAX_S if is_ai(c) else V.HOLD_MAX_S


def time_problem(c: Candidate, rec: dict, start: Any, seconds: float) -> tuple[Optional[float], Optional[str]]:
    """(start, problem): a video needs start inside the clip with start plus the shot inside it too."""
    if c.media != "video":
        return None, None
    span = c.out_point - c.in_point
    if span + 0.05 < seconds:
        return None, f"the clip is {span:.1f} s and the shot needs {seconds:.1f} s"
    try:
        t = float(start)
    except (TypeError, ValueError):
        ranges = rec.get("visual", {}).get("usable") or []
        t = ranges[0]["from"] if ranges else c.in_point
    if t < c.in_point - 0.05 or t + seconds > c.out_point + 0.05:
        return None, (f"start {t:.1f} s plus {seconds:.1f} s is outside the clip "
                      f"({c.in_point:.1f} to {c.out_point:.1f} s)")
    return round(max(c.in_point, t), 3), None


def check(raw: Any, shots: list[dict], pool: list[Candidate], recs: dict[str, dict],
          used_before: dict[str, str], ai_tail: int) -> tuple[list[dict], list[str]]:
    """Normalised picks, one per shot in order, and plain-language problems."""
    errors: list[str] = []
    items = raw.get("picks") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return ([{"clause": s["clauses"][0].id, "pick": None, "_cand": None, "element_picks": [], "_doc": None}
                 for s in shots], ["The reply needs a picks list."])
    by = {str(i.get("clause")): i for i in items if isinstance(i, dict)}
    picks: list[dict] = []
    used: dict[str, str] = {}
    run = ai_tail
    prev_key, prev_secs = None, 0.0

    def seen(key: str) -> Optional[str]:
        return used.get(key) or used_before.get(key)

    for s in shots:
        cid = s["clauses"][0].id
        p = dict(by.get(cid) or {"clause": cid, "pick": None, "reason": "no answer for this clause"})
        p["clause"] = cid
        dur = shot_seconds(s)
        rt = s["plan"].get("real_thing") or {}
        n = p.get("pick")
        cand = _cand_at(pool, n) if n is not None else None
        if n is not None and cand is None:
            errors.append(f"{cid}: pick {n} is not an asset number.")
            p["pick"] = None
        ai_here = False
        if cand is not None:
            rec = recs.get(cand.id, {})
            t, why = time_problem(cand, rec, p.get("start"), dur)
            if why:
                errors.append(f"{cid}: asset {n}: {why}; give a start inside the clip or pick another asset.")
                p["_time_bad"] = why  # kept as given, so resolve removes the pick if the retry repeats it
            else:
                p["start"] = t
                p["frame"] = t if t is not None else "a"  # emit reads the play-from point as frame
            if DS.blocked(rec):
                errors.append(f"{cid}: asset {n} has {', '.join(DS.blocked(rec))} in its record; never pick it.")
            key = asset_key(cand)
            adjacent = key == prev_key
            if seen(key) and not adjacent and not p.get("callback"):
                errors.append(f"{cid}: asset {n} is already used at {seen(key)}; pick another, or mark it a "
                              f"callback and say what changed.")
            on_screen = dur + (prev_secs if adjacent else 0.0)
            if on_screen > _limit(cand) + 1e-6:
                errors.append(f"{cid}: asset {n} would stay on screen {on_screen:.1f} s; the limit is "
                              f"{_limit(cand):g} s{' for an AI still' if is_ai(cand) else ''}. Use another picture.")
            if is_ai(cand):
                ai_here = True
                if rt.get("kind") in NO_AI_KINDS:
                    errors.append(f"{cid}: asset {n} is an AI still, never allowed for a real {rt['kind']} "
                                  f"({rt.get('name', '')}).")
            elif rt.get("name"):
                why = identity_supported(rec, rt, str(p.get("identity") or ""), str(p.get("identity_field") or ""))
                if why:
                    errors.append(f"{cid}: asset {n} is not shown to be {rt['name']}: {why}. Copy the record "
                                  f"words that show it, or answer nothing fits.")
                    p["_unsupported"] = True
            if p.get("label") and (is_ai(cand) or not label_supported(rec, str(p["label"]))):
                errors.append(f"{cid}: label '{p['label']}' is not supported by asset {n}'s record; a label comes "
                              f"only from its title, description, date, page or visible text. Leave it empty.")
                p["_bad_label"] = True
            used.setdefault(key, cid)
            prev_key, prev_secs = key, on_screen
        else:
            prev_key, prev_secs = None, 0.0
            p["label"] = ""
        if ai_here:
            run += 1
            if run > V.AI_RUN_MAX:
                errors.append(f"{cid}: this would be AI still number {run} in a row; at most {V.AI_RUN_MAX}. "
                              f"Answer nothing fits instead.")
        else:
            run = 0
        n_els = len((s["plan"].get("layout") or {}).get("elements", []))
        eps = []
        for ep in p.get("element_picks") or []:
            ec = _cand_at(pool, ep.get("pick")) if isinstance(ep, dict) else None
            try:
                k = int(ep.get("element", -1))
            except (TypeError, ValueError, AttributeError):
                k = -1
            if ec is None or not 0 <= k < n_els:
                errors.append(f"{cid}: element pick {ep} names no element of this layout or no asset.")
                continue
            if is_ai(ec) and (rt.get("kind") in NO_AI_KINDS or s["plan"]["treatment"] == "name_card"):
                errors.append(f"{cid}: element {k} cannot be an AI still: this layout shows a real "
                              f"{rt.get('kind') or 'person'}.")
            if DS.blocked(recs.get(ec.id, {})):
                errors.append(f"{cid}: element {k} asset {ep.get('pick')} has a mark or subtitles; never pick it.")
            ek = asset_key(ec)
            if seen(ek) and seen(ek) != cid and not p.get("callback"):
                errors.append(f"{cid}: element {k} asset {ep.get('pick')} is already used at {seen(ek)}.")
            used.setdefault(ek, cid)
            et, _ = time_problem(ec, recs.get(ec.id, {}), ep.get("start"), dur)
            eps.append({"element": k, "pick": int(ep["pick"]), "frame": et if et is not None else "a", "_cand": ec})
        p["element_picks"] = eps
        p["_doc"] = None
        if needs_document(s["plan"]) and p.get("document_pick") is not None:
            dc = _cand_at(pool, p.get("document_pick"))
            if dc is None or is_ai(dc):
                errors.append(f"{cid}: document_pick {p.get('document_pick')} must be a real asset, never AI.")
            elif seen(asset_key(dc)) and seen(asset_key(dc)) != cid and not p.get("callback"):
                errors.append(f"{cid}: document_pick {p.get('document_pick')} is already used at {seen(asset_key(dc))}.")
            else:
                lay = s["plan"].get("layout") or {}
                passages = [e.get("text", "") for e in lay.get("elements", [])] \
                    if lay.get("purpose") in V.MARKED_PURPOSES else []
                found = passages_found(recs.get(dc.id, {}), passages)
                p["passages_found"] = found
                if passages and not all(found):
                    missing = [t for t, f in zip(passages, found) if not f]
                    errors.append(f"{cid}: document_pick {p.get('document_pick')}: the text reader did not find "
                                  f"{'; '.join(repr(m) for m in missing)} on that page, so it is not the document. "
                                  f"Leave document_pick null unless another page has those words.")
                    p["_doc_unverified"] = True
                used.setdefault(asset_key(dc), cid)
                p["_doc"] = dc
        p["_cand"] = cand
        picks.append(p)
    return picks, errors


def resolve(picks: list[dict], shots: list[dict], ai_tail: int, recs: dict[str, dict],
            used_before: Optional[dict[str, str]] = None) -> list[dict]:
    """Remove what is still illegal after the retry; the clause then goes down the ladder. Never adds content."""
    run = ai_tail
    used = dict(used_before or {})
    prev_key, prev_secs = None, 0.0

    def drop(p: dict, why: str) -> None:
        p["resolved"] = (p.get("resolved", "") + "; " + why).strip("; ")
        p["pick"] = p["_cand"] = None
        p["label"] = ""

    for p, s in zip(picks, shots):
        dur = shot_seconds(s)
        rt = s["plan"].get("real_thing") or {}
        cand = p.get("_cand")
        if cand is not None:
            rec = recs.get(cand.id, {})
            key = asset_key(cand)
            adjacent = key == prev_key
            on_screen = dur + (prev_secs if adjacent else 0.0)
            _, twhy = time_problem(cand, rec, p.get("start"), dur)
            twhy = twhy or p.get("_time_bad")
            if key in used and not adjacent and not p.get("callback"):
                drop(p, f"pick {p.get('pick')} removed: asset already used at {used[key]}")
            elif twhy:
                drop(p, f"pick {p.get('pick')} removed: {twhy}")
            elif DS.blocked(rec):
                drop(p, f"pick {p.get('pick')} removed: {', '.join(DS.blocked(rec))}")
            elif on_screen > _limit(cand) + 1e-6:
                drop(p, f"pick {p.get('pick')} removed: {on_screen:.1f} s on screen is over the limit")
            elif is_ai(cand) and (rt.get("kind") in NO_AI_KINDS or run + 1 > V.AI_RUN_MAX):
                drop(p, f"AI pick {p.get('pick')} removed by the AI rulings")
            elif not is_ai(cand) and rt.get("name") and identity_supported(
                    rec, rt, str(p.get("identity") or ""), str(p.get("identity_field") or "")):
                drop(p, f"pick {p.get('pick')} removed: the record does not show it is {rt['name']}")
            cand = p.get("_cand")
            if cand is not None:
                if p.get("label") and (is_ai(cand) or not label_supported(rec, str(p["label"]))):
                    p["resolved"] = (p.get("resolved", "") + f"; label '{p['label']}' removed: not in the record").strip("; ")
                    p["label"] = ""
                used.setdefault(key, p.get("clause", ""))
                prev_key, prev_secs = key, on_screen
            else:
                prev_key, prev_secs = None, 0.0
        else:
            prev_key, prev_secs = None, 0.0
        keep = []
        for ep in p.get("element_picks", []):
            ec = ep["_cand"]
            ek = asset_key(ec)
            if is_ai(ec) and (rt.get("kind") in NO_AI_KINDS or s["plan"]["treatment"] == "name_card"):
                continue
            if DS.blocked(recs.get(ec.id, {})):
                continue
            if ek in used and used[ek] != p.get("clause") and not p.get("callback"):
                continue
            used.setdefault(ek, p.get("clause", ""))
            keep.append(ep)
        p["element_picks"] = keep
        dc = p.get("_doc")
        if dc is not None and p.get("_doc_unverified"):
            p["_doc"] = dc = None
            p["resolved"] = (p.get("resolved", "") + "; document pick removed: quoted words not found on it").strip("; ")
        if dc is not None:
            if is_ai(dc) or (asset_key(dc) in used and used[asset_key(dc)] != p.get("clause") and not p.get("callback")):
                p["_doc"] = None
            else:
                used.setdefault(asset_key(dc), p.get("clause", ""))
        p["filled_by"] = "pick" if p.get("_cand") is not None or not needs_pick(s["plan"]) else ""
        run = run + 1 if (cand is not None and is_ai(cand)) else 0
    return picks


# ---------------------------------------------------------------- the call


def picker_role(seats) -> str:
    try:
        return SEAT if any(p.configured for p in seats.providers_for(SEAT)) else FALLBACK_SEAT
    except AttributeError:
        return SEAT


def _previous_note(previous: list[dict]) -> list[dict]:
    return [{"words": p.get("words", "")[:160], "picture": p.get("shows", "")[:160]} for p in previous[-2:]]


def pick_call(seats, scene_purpose: str, shots: list[dict], narration: Narration, pool: list[Candidate],
              recs: dict[str, dict], sheet_dir: Path, scene_id: str, used_before: dict[str, str],
              ai_tail: int, previous: list[dict], log: Optional[Callable] = None,
              tag: str = "") -> tuple[list[dict], list[dict], Any]:
    """One picker call (and one retry) for these shots over this pool; returns (picks, attempts, raw)."""
    images = []
    for n, c in enumerate(pool, 1):
        p = SH.asset_card(n, recs[c.id], sheet_dir, f"{scene_id}{tag}")
        images.append(p)
    note = (f"{len(images)} images follow in order, one per asset, labelled #n (n is the asset number in "
            "assets). A video image shows its first frame large and its other frames below, each with its "
            "time in the asset.")
    rows = [clause_row(s, narration) for s in shots]
    records = [prompt_record(n, c, recs[c.id], used_before) for n, c in enumerate(pool, 1)]
    base = PP.pick_prompt(scene_purpose, rows, records, _previous_note(previous), note)
    prompt = base
    attempts: list[dict] = []
    raw: Any = None
    picks: list[dict] = []
    role = picker_role(seats)
    imgs = [i for i in images if i]
    for attempt in range(RETRIES + 1):
        raw = seats.vision_json(role, TASK, PP.PICK_SYSTEM, prompt, imgs, STAGE, details=["high"] * len(imgs))
        picks, errors = check(raw, shots, pool, recs, used_before, ai_tail)
        attempts.append({"attempt": attempt, "errors": errors})
        if log:
            log(f"{scene_id}{tag} pick attempt {attempt}: {len(errors)} problems, pool {len(pool)}")
        if not errors:
            break
        prompt = PP.pick_retry(base, {k: v for k, v in (raw or {}).items() if k != "_mock"}, errors)
    return resolve(picks, shots, ai_tail, recs, used_before), attempts, raw


def edit_scene(seats, scene_plan: dict, clauses: list[Clause], narration: Narration, old: ShotList,
               sheet_dir: Path, used_before: dict[str, str], ai_tail: int, log=None, *,
               settings: Any = None, extra: Optional[list] = None, previous: Optional[list[dict]] = None,
               ladder: Optional[Callable] = None) -> dict:
    shots = shots_of(scene_plan["clauses"], clauses)
    lo, hi = clauses[0].start, clauses[-1].end
    pool_all = scene_pool(old, lo, hi, extra)
    rows = [clause_row(s, narration) for s in shots]
    pool, dropped = shortlist(pool_all, {}, rows)
    recs = DS.describe_pool(seats, pool, settings, log)
    blocked = [c for c in pool if c.id in recs and DS.blocked(recs[c.id])]
    pool = [c for c in pool if c.id in recs and not DS.blocked(recs[c.id])]
    if log and dropped:
        log(f"{scene_plan['scene']}: pool shortlisted to {len(pool)} assets, {len(dropped)} left out")
    picks, attempts, raw = pick_call(seats, scene_plan.get("scene_purpose", ""), shots, narration, pool, recs,
                                     sheet_dir, scene_plan["scene"], used_before, ai_tail, previous or [], log)
    out = {"scene": scene_plan["scene"], "pool": pool, "picks": picks, "attempts": attempts, "records": recs,
           "pool_all": len(pool_all), "blocked": [c.id for c in blocked], "shortlist_dropped": [c.id for c in dropped],
           "mock": bool(isinstance(raw, dict) and raw.get("_mock")), "shots": shots}
    if ladder is not None:
        ladder(out, scene_plan, narration, used_before, ai_tail, previous or [])
    return out


def _used_after(e: dict) -> list[tuple[Candidate, str]]:
    out = []
    for p in e["picks"]:
        for c in [p.get("_cand"), p.get("_fallback_cand"), p.get("_doc")] + [ep["_cand"] for ep in p.get("element_picks", [])]:
            if c is not None:
                out.append((c, p["clause"]))
    return out


def last_picks(e: dict, narration: Narration) -> list[dict]:
    out = []
    for p, s in zip(e["picks"], e["shots"]):
        c = p.get("_cand") or p.get("_fallback_cand")
        if c is None:
            continue
        rec = e["records"].get(c.id, {})
        lo, hi = s["clauses"][0].words[0], s["clauses"][-1].words[1]
        out.append({"clause": p["clause"], "words": " ".join(w.text for w in narration.words[lo:hi + 1]),
                    "shows": rec.get("visual", {}).get("shows", c.title)})
    return out[-2:]


def edit_all(seats, scene_plans: list[dict], clauses: list[Clause], narration: Narration, old: ShotList,
             sheet_dir: Path, log=None, *, settings: Any = None, extra: Optional[list] = None,
             ladder: Optional[Callable] = None) -> list[dict]:
    out = []
    used: dict[str, str] = {}
    ai_tail = 0
    previous: list[dict] = []
    for sp in scene_plans:
        cl = [c for c in clauses if c.scene == sp["scene"]]
        e = edit_scene(seats, sp, cl, narration, old, sheet_dir, used, ai_tail, log, settings=settings,
                       extra=extra, previous=previous, ladder=ladder)
        for c, cid in _used_after(e):
            used.setdefault(asset_key(c), cid)
        ai_tail = 0
        for p in reversed(e["picks"]):
            c = p.get("_cand")
            if (c is not None and is_ai(c)) or p.get("fallback") == "ai_still":
                ai_tail += 1
            else:
                break
        previous = last_picks(e, narration)
        out.append(e)
    return out


def in_point(c: Candidate, frame: Any, seconds: float) -> tuple[float, float]:
    """In and out points for a shot of `seconds`: frame is the picked start time in the asset (seconds), or
    for older plans a keyframe letter (a, b, c at 20, 50, 80 percent); the shot stays inside the clip."""
    if c.media != "video":
        return c.in_point, c.out_point
    span = max(0.0, c.out_point - c.in_point)
    if isinstance(frame, (int, float)) and not isinstance(frame, bool):
        t = float(frame)
    else:
        t = c.in_point + span * FRAME_POS.get(str(frame), 0.2)
    t = max(c.in_point, min(t, c.out_point - seconds)) if span >= seconds else c.in_point
    return round(t, 3), round(min(c.out_point, t + seconds) if span else c.out_point, 3)
