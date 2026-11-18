"""Build the Vidbush sound library: copy fitting sounds from Adam's RadioPlay library, then generate the rest.

Files are COPIED (converted to 48 kHz wav), never referenced. RadioPlay review status and words are kept;
refused sounds are skipped. Each mapping was checked by hand against the sound's depicts and suits text,
and a required word guards it here. Re-running is safe: copies already present and keys already filled
are skipped.

Run: PYTHONPATH=. python scripts/sound_library_seed.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidbush.audio import library as lib  # noqa: E402
from vidbush.audio.elevenlabs_sfx import USD_PER_CREDIT  # noqa: E402
from vidbush.audio.loudness import SR, read_audio  # noqa: E402
from vidbush.config import load_dotenv, load_settings  # noqa: E402

RP = Path("C:/projects/RadioPlay")
ROOTS = {"data": RP / "productions", "seed": RP}
OPERATOR_ENV = Path("C:/projects/vidbush_tmp/.env")

# (RadioPlay root, audio path, our key, a word its depicts or suits must contain)
COPIES = [
    ("seed", "library/beds/road_traffic/road_traffic_loop_v1.mp3", "road", "highway"),
    ("seed", "library/beds/street/street_loop_v1.mp3", "town_street", "small-town"),
    ("seed", "library/beds/cafe/cafe_loop_v4a.mp3", "interior_public", "murmur"),
    ("seed", "library/beds/forest_day/forest_day_loop_v3.mp3", "forest", "forest"),
    ("seed", "library/beds/forest/forest_loop_v1.mp3", "forest", "birdsong"),
    ("seed", "library/beds/barn/barn_loop_v2.mp3", "room_tone", "near silence"),
    ("seed", "library/beds/office/office_loop_v2.mp3", "interior_quiet", "still air"),
    ("seed", "library/beds/living_room/living_room_loop_v2.mp3", "interior_quiet", "absorbed air"),
    ("seed", "library/beds/wind/wind_loop_v1.mp3", "open_ground", "open ground"),
    ("seed", "library/beds/exterior_open/exterior_open_loop_v1.mp3", "open_ground", "open country"),
    ("seed", "library/beds/exterior_open/exterior_open_loop_v2.mp3", "open_ground", "open high ground"),
    ("seed", "library/beds/riverbank/riverbank_bed.mp3", "river", "flowing water"),
    ("seed", "library/beds/street_day/street_day_loop_v1.mp3", "city_street", "city"),
    ("data", "library/sfx/6f521308.wav", "card_settle", "card"),
]
# Not copied, on purpose: rain_on_window (indoors, not rain outdoors); beach_bed (no water in it, a dry
# frontier room tone); data cafe-5cfe1c1e (the same take as cafe_loop_v4a); wind_loop_v2 (a gale by a wall, not snow); the courtroom beds (one refused for chatter).


class Log:
    def info(self, msg: str, **_: object) -> None:
        print(msg)

    def warn(self, msg: str, **_: object) -> None:
        print("WARNING", msg)


def review_of(sidecar: dict) -> dict:
    """RadioPlay stages/sound.py review_of, copied: refused_by_operator, else heard_date + heard_note."""
    refusal = sidecar.get("refused_by_operator")
    if isinstance(refusal, dict) and refusal:
        return {"status": lib.REFUSED, "words": refusal.get("words")}
    when, note = str(sidecar.get("heard_date") or "").strip(), str(sidecar.get("heard_note") or "").strip()
    if when and note and sidecar.get("unheard") is not True:
        after = re.search(r"kept \(\d\d/\d\d/\d{4}\):\s*(.+)$", note)
        quoted = re.findall(r'"([^"]+)"', note)
        words = after.group(1).strip() if after else (quoted[-1] if quoted else None)
        return {"status": lib.APPROVED, "words": words}
    return {}


def sidecar_of(audio: Path) -> dict:
    """The audio's own .json, else the bed sidecar whose files.loop or files.generation names it."""
    own = audio.with_suffix(".json")
    if own.is_file():
        return json.loads(own.read_text(encoding="utf-8"))
    for p in audio.parent.glob("*.json"):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if audio.name in ((doc.get("files") or {}).get("loop"), (doc.get("files") or {}).get("generation")):
            return doc
    return {}


def prompt_of(audio: Path, sidecar: dict) -> str:
    text = str(sidecar.get("prompt") or "").strip()
    named = (sidecar.get("files") or {}).get("prompt")
    if not text and named and (audio.parent / named).is_file():
        text = (audio.parent / named).read_text(encoding="utf-8").strip()
    return text


def copy_from_radioplay(root: Path, rows: list[list[str]]) -> None:
    catalogue = json.loads((RP / "productions/library/catalogue.json").read_text(encoding="utf-8"))["sounds"]
    have = {s.source for s in lib.load_catalogue(root)}
    for rp_root, audio, key, word in COPIES:
        entry = next((s for s in catalogue if s["root"] == rp_root and s["audio"] == audio), None)
        if entry is None:
            rows.append(["skip", key, audio, "not in RadioPlay catalogue", ""])
            continue
        if f"radioplay:{entry['id']}" in have:
            rows.append(["have", key, audio, "already copied", ""])
            continue
        if word not in f"{entry['depicts']} {entry['suits']}".lower():
            rows.append(["skip", key, audio, f"depicts lacks {word!r}", ""])
            continue
        path = ROOTS[rp_root] / audio
        side = sidecar_of(path)
        review = review_of(side)
        status, words = entry["review_status"], entry.get("review_words")
        if review.get("status") == lib.REFUSED or (lib.review_rank(status) == 1 and review):
            status, words = review["status"], review.get("words")  # the sidecar is newer than the catalogue
        if lib.review_rank(status) >= lib.NEVER:
            rows.append(["skip", key, audio, "refused by Adam", ""])
            continue
        bed = key in lib.PLACES
        fields = {"category": "bed", "place": key, "loopable": True} if bed else \
            {"category": "effect", "effect": key, "family": lib.EFFECTS[key]["family"]}
        sound = lib.Sound(id="", file="", name=entry["name"], depicts=entry["depicts"], suits=entry["suits"],
                          source=f"radioplay:{entry['id']}", prompt=prompt_of(path, side),
                          generated_date=entry.get("generated_date") or "", review_status=status,
                          review_words=words, **fields)
        sound = lib.store(root, sound, read_audio(path, SR, 2 if bed else 1))
        rows.append(["copied", key, sound.id, status, ""])


def main() -> None:
    settings = load_settings()
    if not settings.key("ELEVENLABS_API_KEY"):
        extra = load_dotenv(OPERATOR_ENV)
        settings.env.update({k: v for k, v in extra.items() if k.startswith("ELEVENLABS_")})
    root = lib.library_root(settings)
    print(f"library: {root}")
    rows: list[list[str]] = []
    copy_from_radioplay(root, rows)
    log = Log()
    before = {s.id for s in lib.load_catalogue(root)}
    wanted = [("bed", k, 0) for k in lib.PLACES] + [("effect", k, v) for k in lib.EFFECTS for v in (0, 1)]
    for category, key, variant in wanted:
        sound = lib.ensure(settings, category, key, log=log, variant=variant)
        if sound is None:
            rows.append(["FAILED", key, f"variant {variant}", "", ""])
        elif sound.id not in before:
            side = json.loads((root / sound.file).with_suffix(".json").read_text(encoding="utf-8"))
            rows.append(["generated", key, sound.id, sound.review_status, f"{side.get('credits', 0):.0f}"])
            before.add(sound.id)
    credits = sum(float(r[4]) for r in rows if r[4])
    print(f"\n{'action':10} {'key':16} {'sound':40} {'status':45} credits")
    for r in rows:
        print(f"{r[0]:10} {r[1]:16} {r[2][:40]:40} {r[3][:45]:45} {r[4]}")
    print(f"\ncredits used: {credits:.0f}  (about {credits * USD_PER_CREDIT:.3f} USD)")
    print(f"catalogue: {len(lib.load_catalogue(root))} sounds")


if __name__ == "__main__":
    main()
