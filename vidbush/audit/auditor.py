"""Run a tile audit: extract frames, one auditor vision call per tile, write audit.jsonl, audit.md, audit_summary.json."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from ..util import run, write_json
from . import core

ROLE = "auditor"
STAGE = "audit"

SYSTEM = """You audit single frames of a finished narrated video the way a viewer experiences them at normal playback speed: \
they hear the words and look at the picture, and they notice when the two do not go together or when something on \
screen looks broken, false or confusing. For one frame you get the picture, the narration spoken while it is on \
screen (NOW), and the narration before (BEFORE) and after (AFTER) it. Your answers are counted over the whole \
video to measure how closely the pictures follow the narration and how often a viewer notices a problem.

Ignore the format of the video. Letterbox bars are not a defect. A running subtitle line that transcribes the \
narration word for word (usually centred at the bottom of the frame or in the bottom bar) is never the overlay \
and never a defect; look past it at the picture for any text the editor placed.

Reply with one JSON object and nothing else, keys in this order:
{"kind": "...", "shows": "...", "overlay": "...", "overlay_true": null, "match": "...", "defect": "", "noticeable": false}

kind: what fills the frame. One of:
- real_footage: filmed real world video: live action, aerial, stock, archival or news film.
- real_still: a real photograph.
- document: a real printed or written item photographed or scanned: newspaper, page, letter, form, sign, \
a map page from a printed atlas or old map sheet, a screenshot of a web page.
- ai_image: a still made by an image generator. These are common in such videos and are often given slow \
camera motion, so they can pass for footage. Tells: smooth, waxy or glossy surfaces, an even staged light, \
garbled or fake lettering, odd hands, faces or animals, impossible details, past or historic scenes shown in \
clean modern colour and detail, and black and white or sepia scenes that look like old photographs but are too \
clean, sharp and well composed. Judge by the look, not by whether the subject fits.
- ai_video: moving footage that looks computer generated in the same way.
- map: a map graphic made or animated for the video, or a digital map or satellite map view shown as a map.
- chart: a graph, chart or diagram of numbers.
- text_card: mostly text on a plain or simple background.
- designed_card: a composed layout made for the video: split comparison, a number or name set over a darkened \
picture, framed pictures with captions, a title over footage that dominates the frame.
- other: black or blank frame, transition, end screen, anything else.

shows: one short line naming what is visible, specific enough to check against the words.

overlay: the text or small graphic drawn over the picture by the editor (label, title, callout box, inset map, \
lower third), quoted as read; "" if none. Text that is part of the photographed scene or document is not an \
overlay.
overlay_true: true when the overlay is correct for the words and for the picture under it; false when it \
contradicts the narration, names something the picture does not show, or arrives at the wrong moment; null \
when there is no overlay.

match: how the picture relates to the NOW words. Use BEFORE and AFTER only to recognise a picture that belongs \
to another sentence.
- on_words: it shows a subject of the NOW words: a person, place, object, action, number or idea named or \
described in NOW, or a plain illustration a viewer takes as that thing, even when details are imperfect. A map, \
card or label that names or marks it counts.
- early: it shows the subject of a sentence in AFTER that NOW does not mention.
- late: it shows the subject of a sentence in BEFORE that NOW does not mention.
- related: same topic, region or story as the narration, but not a subject of NOW.
- generic: filler that would fit almost any sentence of this video (a street, a car, hands, a skyline) with no \
link to what is said.
- wrong: it contradicts the words or shows a different specific thing than the one named (another place, \
person, period, season, quantity).

defect: one line naming the main problem a viewer could see in this moment, "" if none.

noticeable: decide after writing the defect. True when the problem would register with a viewer at normal speed:
- the picture plainly shows something else than what is being said: match wrong always, early or late when \
the picture clearly shows another sentence's subject, related or generic when the gap between words and picture \
is obvious;
- a picture that contradicts the words in place, person, period, season or quantity;
- editor text that is false for the words or the picture, belongs to another line, is cut off, too small or \
faint to read, or a box or layout that is empty or half built;
- a generated picture standing in for a specific real person, event or record the words name;
- visible faults: black, blank or broken frames, garbled lettering, deformed people or objects, watermarks, \
another channel's logos or subscribe prompts, cluttered stacked graphics, a graphic whose marks are unlabelled \
or illegible so it means nothing, a narrow picture that leaves most of the frame as blurred bars.
False when the picture fits the moment well enough that a viewer would follow along without a second thought, \
including a believable stand-in or a generated picture of an ordinary scene with no visible tells. When your \
defect says the picture shows another subject, place, period or moment than the words, or a generated stand-in \
for something real the words name, noticeable is true."""

USER = """Time in video: {mmss}
NOW (spoken while this frame is on screen): "{words}"
BEFORE: "{before}"
AFTER: "{after}"
"""

GENERATED_NOTE = "SOURCE: the project's shot list says this picture was made by an image generator.\n"

PROMPT_HASH = hashlib.sha256((SYSTEM + USER + GENERATED_NOTE).encode("utf-8")).hexdigest()[:12]

VisionFn = Callable[[str, str, str], dict[str, Any]]  # (system, user, image path) -> reply


def extract_frame(video: Path, t: float, out: Path, width: int = 640) -> Path:
    """One JPEG frame at t seconds (cached: an existing file is kept)."""
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
             "-vf", f"scale={width}:-2", "-q:v", "3", str(out)], timeout=120)
    return out


def normalize(reply: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The reply keys in the audit vocabulary, or None when the reply is not a usable verdict."""
    match = str(reply.get("match", "")).strip().lower()
    if match not in core.MATCHES or "noticeable" not in reply:
        return None
    kind = str(reply.get("kind", "")).strip().lower()
    overlay = str(reply.get("overlay") or "").strip()
    ot = reply.get("overlay_true")
    noticeable = reply.get("noticeable")
    if isinstance(noticeable, str):
        noticeable = noticeable.strip().lower() == "true"
    return {"kind": kind if kind in core.KINDS else "other", "match": match, "noticeable": bool(noticeable),
            "shows": str(reply.get("shows") or "").strip(), "overlay": overlay,
            "overlay_true": (bool(ot) if isinstance(ot, bool) else None) if overlay else None,
            "defect": str(reply.get("defect") or "").strip()}


def seats_vision(seats: Any) -> VisionFn:
    def fn(system: str, user: str, image: str) -> dict[str, Any]:
        return seats.vision_json(ROLE, "audit_tile", system, user, [image], STAGE, details=["high"])
    return fn


_URL_QUERY = re.compile(r"(https?://[^\s?#'\"]+)[?#][^\s'\"]*")
_SECRET = re.compile(r"(?i)\b(api[_-]?key|key|token|secret|authorization|bearer)\b(\s*[=:]\s*|\s+)\S+")


def scrub(e: BaseException) -> str:
    """Exception class and a short message with URL query strings and credential values removed."""
    msg = _SECRET.sub(r"\1=<redacted>", _URL_QUERY.sub(r"\1", str(e)))
    return f"{type(e).__name__}: {' '.join(msg.split())[:160]}"


def audit_tile(vision: VisionFn, tile: dict[str, Any], frame: Path) -> dict[str, Any]:
    user = USER.format(mmss=core.mmss(tile["frame_t"]), words=tile["words"], before=tile["before"],
                       after=tile["after"]) + source_line(tile)
    last = ""
    for _ in range(2):  # one retry: a relay sometimes returns prose or a cut reply
        try:
            reply = vision(SYSTEM, user, str(frame))
        except Exception as e:  # noqa: BLE001  a failed tile is recorded, the audit goes on
            last = scrub(e)
            continue
        if reply.get("_mock"):
            last = "mock reply (no real auditor seat)"
            break
        got = normalize(reply)
        if got:
            if source_line(tile) and got["kind"] not in core.AI_KINDS | core.DESIGNED_KINDS:
                # provenance beats a visual guess of real or other; a designed layout stays designed (it may hold
                # a generated picture). The guess is kept for checking the auditor.
                got["kind_seen"] = got["kind"]
                got["kind"] = "ai_image"
            return {**tile, **got, "prompt": PROMPT_HASH}
        last = "unusable reply: " + _URL_QUERY.sub(r"\1", json.dumps(reply))[:160]
    return {**tile, "error": last, "prompt": PROMPT_HASH}


def source_line(tile: dict[str, Any]) -> str:
    return GENERATED_NOTE if tile.get("source") in core.AI_SOURCES else ""


def video_identity(video: Path) -> dict[str, Any]:
    """Name, size and whole-second mtime; a re-render of the same path gets a new identity."""
    try:
        st = video.stat()
        return {"name": video.name, "size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {"name": video.name, "size": -1, "mtime": 0}


def video_hash(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def row_key(tile: dict[str, Any], model: str, vhash: str) -> str:
    """What a verdict depends on: prompt, auditor model, the words shown to it, the source line, the video."""
    parts = [PROMPT_HASH, model, tile["words"], tile["before"], tile["after"], source_line(tile), vhash]
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()[:16]


def build_tiles(video: Path, words: list[core.Word], shots: list[dict[str, Any]], duration: float,
                every: float = 2.0, limit: int = 0) -> list[dict[str, Any]]:
    tiles = []
    for t in core.spread(core.tile_times(duration, every), limit):
        ft = round(t + every / 2, 3)
        tile = {"t": int(t) if float(t).is_integer() else t, "frame_t": int(ft) if float(ft).is_integer() else ft,
                **core.tile_words(words, t, every)}
        s = core.shot_at(shots, ft)
        if s:
            tile["shot_id"], tile["shot_type"] = s["id"], s["type"]
            if s.get("source"):
                tile["source"] = s["source"]
        tiles.append(tile)
    return tiles


def _fresh_tiles_dir(tiles_dir: Path, identity: dict[str, Any]) -> None:
    """Drop cached frames extracted from another version of the video."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    side = tiles_dir / "video.json"
    try:
        same = json.loads(side.read_text(encoding="utf-8")) == identity
    except (OSError, ValueError):
        same = False
    if not same:
        for f in tiles_dir.glob("*.jpg"):
            f.unlink()
        side.write_text(json.dumps(identity), encoding="utf-8")


def run_audit(video: Path, words: list[core.Word], out: Path, vision: VisionFn, duration: float,
              shots: Optional[list[dict[str, Any]]] = None, every: float = 2.0, limit: int = 0, workers: int = 8,
              extract: Optional[Callable[[Path, float, Path], Path]] = None, model: str = "",
              log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Audit every tile (or `limit` evenly spread ones) and return this run's rows.

    A stored row is reused only when its key (prompt, model, words, source line, video identity) matches, so a
    rerun after a crash or a slice pays only for new tiles. Each finished row is appended to audit.jsonl at
    once, so a crash loses no paid verdict; the file is compacted (latest row per tile) at the end."""
    out.mkdir(parents=True, exist_ok=True)
    identity = video_identity(video)
    vhash = video_hash(identity)
    tiles_dir = out / "tiles"
    _fresh_tiles_dir(tiles_dir, identity)
    path = out / "audit.jsonl"
    old = {r["t"]: r for r in core.latest_rows(path) if not r.get("error")}
    done: dict[Any, dict[str, Any]] = {}
    todo = []
    for tile in build_tiles(video, words, shots or [], duration, every, limit):
        tile["key"] = row_key(tile, model, vhash)
        prev = old.get(tile["t"])
        if prev and prev.get("key") == tile["key"]:
            done[tile["t"]] = prev
        else:
            todo.append(tile)
    log(f"tiles {len(done) + len(todo)}: {len(done)} kept from the last audit, {len(todo)} to audit")

    def one(tile: dict[str, Any]) -> dict[str, Any]:
        try:
            frame = (extract or extract_frame)(video, tile["frame_t"], tiles_dir / core.frame_name(tile["t"]))
            return audit_tile(vision, tile, frame)
        except Exception as e:  # noqa: BLE001  ffmpeg failure or timeout: an error row, the audit goes on
            return {**tile, "error": scrub(e), "prompt": PROMPT_HASH}

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, path.open("a", encoding="utf-8") as fh:
            for i, fut in enumerate(as_completed([pool.submit(one, t) for t in todo]), 1):
                row = fut.result()
                done[row["t"]] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 25 == 0:
                    log(f"  {i}/{len(todo)} audited")
    finally:
        rows = core.latest_rows(path)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return sorted(done.values(), key=lambda r: r["t"])


def _cell(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def write_reports(rows: list[dict[str, Any]], out: Path, video: Path, model: str, cost_usd: float,
                  every: float = 2.0) -> Optional[dict[str, Any]]:
    """audit.md and audit_summary.json from every row of the audit; None (and nothing written) when no tile has a
    valid verdict."""
    s = core.summarize(rows)
    if not s["tiles"]:
        return None
    summary = {**{k: s[k] for k in core.SUMMARY_KEYS}, "tiles": s["tiles"], "overlay_tiles": s["overlay_tiles"],
               "video": str(video), "model": model, "cost_usd": round(cost_usd, 4)}
    write_json(out / "audit_summary.json", summary)
    errors = [r for r in rows if r.get("error")]
    prompts = {r.get("prompt") for r in rows if not r.get("error")}
    lines = [f"# Tile audit: {video.name}", "",
             f"{s['tiles']} tiles every {every:g} s, {s['overlay_tiles']} with an overlay. Model {model}, "
             f"audit cost so far {cost_usd:.4f} USD."
             + (f" {len(errors)} tiles had no usable verdict and are left out." if errors else "")
             + (f" Rows come from {len(prompts)} prompt versions." if len(prompts) > 1 else ""),
             "", "| Measure (share of tiles) | This video | Vidrush benchmark | Target |", "|---|---|---|---|"]
    for k in core.SUMMARY_KEYS:
        lines.append(f"| {core.LABELS[k]} | {_cell(s[k])} | {core.BENCHMARK[k]}% | {core.TARGETS.get(k, '')} |")
    lines += ["", "## Ten worst stretches (runs of noticeable tiles)", ""]
    worst = core.worst_stretches(rows, every)
    if not worst:
        lines.append("No noticeable tiles.")
    for w in worst:
        lines.append(f"- **{w['range']}** ({w['tiles']} tiles): " + ("; ".join(w["defects"]) or "no defect text"))
    (out / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def model_used(entries: list[Any], since: float) -> str:
    c = Counter(f"{e.provider}/{e.model}" for e in entries if e.role == ROLE and e.ok and not e.mock and e.ts >= since)
    return c.most_common(1)[0][0] if c else ""
