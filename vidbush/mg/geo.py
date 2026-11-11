"""Offline geocoding over bundled Natural Earth data and a curated landmark list.

Sources, in preference order: assets/geo/landmarks.json (historically relevant
sites), country label points from ne_50m_admin_0_countries.geojson, and the
Natural Earth populated places gazetteer. States and provinces (shading and
framing only) come from ne_50m_admin_1_states_provinces.geojson. Everything
loads lazily and is cached.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Sequence

from pydantic import BaseModel

from ..config import REPO_ROOT

GEO_DIR = Path(REPO_ROOT) / "assets" / "geo"
US = "United States of America"

Kind = Literal["landmark", "country", "place"]


class GeoHit(BaseModel):
    """One geocoding result."""

    name: str
    lat: float
    lon: float
    kind: Kind
    country: str = ""
    pop: int = 0
    iso_a3: str = ""


# ---------------------------------------------------------------- loading

def norm(text: str) -> str:
    """Lowercase, strip accents, collapse punctuation and spaces."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace("'s ", " ").replace("’", "'")
    t = re.sub(r"[^a-z0-9']+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def landmarks() -> list[dict]:
    """Curated landmark records: {name, lat, lon, kind, country, aliases}."""
    return json.loads((GEO_DIR / "landmarks.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def gazetteer() -> list[list]:
    """Populated places: [name, lat, lon, pop_max, country, admin1]."""
    return json.loads((GEO_DIR / "gazetteer_places.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _country_features() -> list[dict]:
    data = json.loads((GEO_DIR / "ne_50m_admin_0_countries.geojson").read_text(encoding="utf-8"))
    return data["features"]


def _iso(props: dict) -> str:
    iso = props.get("ISO_A3") or "-99"
    return props.get("ADM0_A3", "") if iso == "-99" else iso


@lru_cache(maxsize=1)
def countries() -> dict[str, GeoHit]:
    """Country label points keyed by normalised NAME, NAME_LONG, ADMIN and ISO_A3."""
    out: dict[str, GeoHit] = {}
    for f in _country_features():
        p = f["properties"]
        hit = GeoHit(name=p["NAME"], lat=float(p["LABEL_Y"]), lon=float(p["LABEL_X"]), kind="country",
                     country=p["ADMIN"], pop=int(p.get("POP_EST") or 0), iso_a3=_iso(p))
        for key in (p.get("NAME"), p.get("NAME_LONG"), p.get("ADMIN"), hit.iso_a3):
            if key and key != "-99":
                out.setdefault(norm(key), hit)
    for alias, target in (("usa", "united states of america"), ("united states", "united states of america"),
                          ("america", "united states of america"), ("us", "united states of america"),
                          ("britain", "united kingdom"), ("great britain", "united kingdom"),
                          ("england", "united kingdom"), ("holland", "netherlands")):
        if target in out:
            out.setdefault(alias, out[target])
    return out


@lru_cache(maxsize=1)
def _index() -> dict[str, list[GeoHit]]:
    """Normalised name or alias to candidate hits (landmarks first)."""
    idx: dict[str, list[GeoHit]] = {}
    for lm in landmarks():
        hit = GeoHit(name=lm["name"], lat=lm["lat"], lon=lm["lon"], kind="landmark", country=lm.get("country", ""))
        for key in [lm["name"], *lm.get("aliases", [])]:
            for k in {norm(key), norm(re.sub(r"^the\s+", "", key, flags=re.I))}:
                idx.setdefault(k, []).append(hit)
    for key, hit in countries().items():
        if len(key) > 3 or key in ("usa", "uk"):
            idx.setdefault(key, []).append(hit)
    for name, lat, lon, pop, country, _admin in gazetteer():
        idx.setdefault(norm(name), []).append(GeoHit(name=name, lat=lat, lon=lon, kind="place", country=country,
                                                     pop=int(pop or 0)))
    return idx


# ---------------------------------------------------------------- geocoding

_KIND_RANK = {"landmark": 3, "country": 2, "place": 1}


def _best(hits: list[GeoHit], us_bias: bool) -> GeoHit:
    def key(h: GeoHit) -> tuple:
        bias = 1 if (us_bias and h.country == US) else 0
        return (_KIND_RANK[h.kind], bias, h.pop)
    return max(hits, key=key)


def candidates(name: str) -> list[GeoHit]:
    """Every hit for a name under its normalised forms (homonyms in other countries included)."""
    out: list[GeoHit] = []
    for variant in _variants(name.strip()) if name.strip() else []:
        out += [h for h in _index().get(variant, []) if h not in out]
    return out


def lookup(name: str, us_bias: bool = False) -> Optional[GeoHit]:
    """Best hit for a name: exact, then case-insensitive, then alias forms."""
    name = name.strip()
    if not name:
        return None
    exact = [h for h in _index().get(norm(name), []) if h.name == name]
    if exact:
        return _best(exact, us_bias)
    for variant in _variants(name):
        hits = _index().get(variant)
        if hits:
            return _best(hits, us_bias)
    return None


def _variants(name: str) -> list[str]:
    n = norm(name)
    out = [n]
    stripped = re.sub(r"^(the|city of|port of|town of)\s+", "", n)
    out.append(stripped)
    out.append(re.sub(r"'s?\b", "", stripped).strip())
    out.append(re.sub(r"\s+(city|town|harbour|harbor)$", "", stripped))
    if "," in name:
        out.append(norm(name.split(",")[0]))
    return [v for i, v in enumerate(out) if v and v not in out[:i]]


def geocode(name: str, us_bias: bool = False) -> Optional[tuple[float, float]]:
    """(lat, lon) for a place name or None."""
    hit = lookup(name, us_bias)
    return (hit.lat, hit.lon) if hit else None


AREA_KINDS = {"country", "region", "continent", "ocean", "sea", "gulf", "strait", "route", "isthmus"}
WATER_KINDS = {"ocean", "sea", "gulf", "strait", "bay", "channel"}


def place_kind(name: str) -> str:
    """'country', a landmark kind ('region' for states, 'continent', 'ocean', 'city', 'dam', ...), 'city' for a
    gazetteer place, or '' when unknown."""
    hit = lookup(name)
    if hit is None:
        return ""
    if hit.kind == "country":
        return "country"
    if hit.kind == "place":
        return "city"
    return next((lm.get("kind", "landmark") for lm in landmarks() if lm["name"] == hit.name), "landmark")


def is_area(name: str) -> bool:
    """True for countries, states, continents and seas: places a single marker or lower third cannot show."""
    return place_kind(name) in AREA_KINDS


def is_country(name: str) -> bool:
    """True when the name resolves to a country (name, long name, admin or ISO A3)."""
    return norm(name) in countries()


def country_name(name: str) -> Optional[str]:
    """Canonical Natural Earth NAME for a country reference, else None."""
    hit = countries().get(norm(name))
    return hit.name if hit else None


@lru_cache(maxsize=512)
def country_bbox(name: str) -> Optional[tuple[float, float, float, float]]:
    """(min_lat, min_lon, max_lat, max_lon) of the largest polygon of a country."""
    hit = countries().get(norm(name))
    if hit is None:
        return None
    for f in _country_features():
        if f["properties"]["NAME"] != hit.name:
            continue
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        ring = max((p[0] for p in polys), key=len)
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        return min(lats), min(lons), max(lats), max(lons)
    return None


@lru_cache(maxsize=1)
def admin1() -> dict[str, dict]:
    """State and province features keyed by normalised name."""
    data = json.loads((GEO_DIR / "ne_50m_admin_1_states_provinces.geojson").read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for f in data["features"]:
        p = f["properties"]
        if p.get("name"):
            out[norm(p["name"])] = out[norm(admin1_key(p))] = f
    return out


def admin1_key(props: dict) -> str:
    """Name a state or province is shaded by: 'Georgia, United States of America' when a country shares its
    name (the renderer then fills that feature only), else the plain name."""
    return f"{props['name']}, {props.get('admin', '')}" if is_country(props["name"]) else props["name"]


def region_name(name: str, near: Sequence[tuple[float, float]] = ()) -> Optional[str]:
    """Canonical name of a country or a state or province the bundled layers can shade, else None. When a
    state and a country share the name, the state wins if its box holds a point of near (the map's markers or
    view center), else the country."""
    f = admin1().get(norm(name))
    if f is None:
        return country_name(name)
    key = admin1_key(f["properties"])
    if not is_country(name) or norm(name) == norm(key):
        return key
    box = region_bbox(key)
    return key if any(box[0] <= la <= box[2] and box[1] <= lo <= box[3] for la, lo in near) else country_name(name)


@lru_cache(maxsize=512)
def region_bbox(name: str) -> Optional[tuple[float, float, float, float]]:
    """(min_lat, min_lon, max_lat, max_lon) of the largest polygon of a country, state or province (a name
    region_name gives)."""
    box = country_bbox(name)
    f = admin1().get(norm(name)) if box is None else None
    if f is None:
        return box
    geom = f["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    ring = max((p[0] for p in polys), key=len)
    return min(pt[1] for pt in ring), min(pt[0] for pt in ring), max(pt[1] for pt in ring), max(pt[0] for pt in ring)


def distance_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Approximate angular distance in degrees (longitude scaled by latitude)."""
    dlat = a[0] - b[0]
    dlon = (a[1] - b[1] + 180.0) % 360.0 - 180.0
    return math.hypot(dlat, dlon * math.cos(math.radians((a[0] + b[0]) / 2)))


# ---------------------------------------------------------------- text extraction

STOPWORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "he", "her", "his", "i", "if", "in", "into", "it",
    "its", "of", "on", "or", "our", "she", "so", "that", "the", "their", "then", "there", "these", "they",
    "this", "those", "to", "we", "what", "when", "where", "which", "who", "why", "with", "you", "yet", "after",
    "before", "during", "over", "under", "today", "now", "once", "every", "each", "all", "some", "most", "many",
    "one", "two", "three", "few", "how", "no", "not", "nor", "only", "even", "still", "within",
    "chapter", "part", "number", "act", "episode", "inside", "outside", "between", "through", "while", "until",
    "president", "king", "queen", "general", "captain", "doctor", "dr", "mr", "mrs", "sir", "lord", "colonel",
    "emperor", "pope", "saint", "lake", "river", "mount", "cape", "fort", "bay", "gulf", "north", "south",
    "east", "west", "new", "old", "great", "little", "big", "canal", "dam", "bridge", "city", "island", "first",
    "last", "second", "third", "world", "war", "empire", "republic", "union", "state", "states", "national",
    "army", "navy", "company", "american", "french", "british", "english", "spanish", "german", "chinese",
    "japanese", "russian", "european", "african", "asian", "mexican", "roman", "greek", "egyptian", "panamanian",
    "colombian", "americans", "europeans", "god", "ocean", "sea", "here", "yes", "maybe", "never", "always",
}
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
          "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
FIRST_NAMES = {
    "john", "james", "william", "george", "thomas", "theodore", "franklin", "abraham", "andrew", "charles",
    "henry", "robert", "richard", "joseph", "edward", "david", "michael", "peter", "paul", "mark", "louis",
    "ferdinand", "philippe", "napoleon", "ulysses", "woodrow", "herbert", "harry", "dwight", "lyndon", "ronald",
    "jimmy", "gerald", "calvin", "warren", "grover", "benjamin", "samuel", "sam", "walter", "frank", "albert",
    "alexander", "elizabeth", "mary", "anne", "victoria", "catherine", "margaret", "martin", "neil", "buzz",
    "orville", "wilbur", "amelia", "howard", "chester", "rutherford", "millard", "zachary", "jesse",
    "daniel", "francis", "hernan", "vasco", "christopher", "ernest", "leonard", "arthur", "frederick", "otto",
    "jean", "pierre", "philip", "isaac", "nikola", "alfred", "lewis", "meriwether",
}
AMBIGUOUS = {"lincoln", "jackson", "washington", "victoria", "franklin", "madison", "jefferson", "hamilton",
             "wilson", "taylor", "grant", "lee", "cleveland", "houston", "monroe", "austin", "eugene", "florence",
             "marion", "columbus", "raleigh", "charlotte", "mobile", "independence", "concord", "union", "reading",
             "bath", "nice", "sale", "split", "georgia", "jordan", "chad", "turkey", "darwin", "hope", "liberty",
             "troy", "phoenix", "orange"}
LOCATIVE = {"in", "at", "to", "from", "near", "toward", "towards", "across", "via", "into", "outside", "reached",
            "through", "past", "around", "off", "of", "between", "and"}
CONNECTORS = {"of", "de", "del", "la", "da", "du", "von", "van", "the"}
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'’\.\-]*|[.!?;:,]")


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _clean(tok: str) -> str:
    tok = re.sub(r"['’]s$", "", tok)
    return tok.rstrip(".-'’")


def cap_runs(toks: list[str]) -> list[tuple[int, list[str], bool]]:
    """(start index, words, sentence_initial) for each run of capitalised words."""
    runs: list[tuple[int, list[str], bool]] = []
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        if not t[0].isupper():
            i += 1
            continue
        j = i
        words: list[str] = []
        while j < n and (toks[j][0].isupper() or (toks[j].lower() in CONNECTORS and j + 1 < n
                                                  and toks[j + 1][0].isupper() and words)):
            words.append(_clean(toks[j]))
            j += 1
        initial = i == 0 or toks[i - 1] in ".!?;:"
        runs.append((i, words, initial))
        i = j
    return runs


def _single_ok(word: str, hit: GeoHit, prev: str, initial: bool) -> bool:
    """Decide whether a one-word match is really a place in context."""
    w = word.lower()
    if w in STOPWORDS or w in MONTHS or len(w) < 3:
        return False
    locative = prev.lower() in LOCATIVE
    if w in AMBIGUOUS and hit.kind != "country" and not locative:
        return False
    if hit.kind == "place" and hit.pop < 150_000 and not locative:
        return False
    return not (initial and hit.kind == "place" and hit.pop < 1_000_000 and not locative)


def _match_run(words: list[str], prev: str, initial: bool) -> list[tuple[str, GeoHit]]:
    """Greedy longest n-gram matching (up to 4 words) inside one capitalised run."""
    out: list[tuple[str, GeoHit]] = []
    if words and words[0].lower() in FIRST_NAMES and len(words) >= 2:
        whole = lookup(" ".join(words))
        return [(" ".join(words), whole)] if whole and whole.kind == "landmark" else []
    i = 0
    while i < len(words):
        for size in range(min(4, len(words) - i), 0, -1):
            gram = words[i:i + size]
            if gram[0].lower() in CONNECTORS or gram[-1].lower() in CONNECTORS:
                continue
            name = " ".join(gram)
            hit = lookup(name)
            if hit is None:
                continue
            p = prev if i == 0 else words[i - 1]
            if size == 1 and not _single_ok(name, hit, p, initial and i == 0):
                continue
            out.append((name, hit))
            i += size
            break
        else:
            i += 1
    return out


# A name the text gives to a person or an animal ("his lead dog, Togo", "a horse named Comanche"). The noun
# must come first: places are named the same way ("a town called Nome"), so 'named X' alone is not enough.
_CREATURE_RE = re.compile(
    r"\b(?:dog|dogs|horse|horses|mule|pony|cat|husky|puppy|mare|stallion|wolf|bear|son|daughter|wife|husband|"
    r"brother|sister|friend|partner|boy|girl|man|woman)\s*,?\s+(?:(?:named|called|nicknamed|dubbed)\s+)?"
    r"([A-Z][\w'’-]+)")


def creature_names(text: str) -> set[str]:
    """Normalised names the text gives to people or animals: such a name is never a place in that text."""
    return {norm(_clean(name)) for name in _CREATURE_RE.findall(text)}


def find_place_hits(text: str, not_places: frozenset[str] | set[str] = frozenset()) -> list[GeoHit]:
    """Places mentioned in text as GeoHit records, in order of appearance, deduplicated. Names in not_places
    (normalised, see creature_names) are not places wherever they appear."""
    toks = tokens(text)
    out: list[GeoHit] = []
    seen: set[str] = set()
    for start, words, initial in cap_runs(toks):
        prev = toks[start - 1] if start > 0 else ""
        for name, hit in _match_run(words, prev, initial):
            if norm(name) in not_places:
                continue
            key = norm(hit.name)
            if key in seen or any(distance_deg((hit.lat, hit.lon), (o.lat, o.lon)) < 0.02 for o in out):
                continue
            seen.add(key)
            out.append(hit)
    return out


def find_places(text: str, not_places: frozenset[str] | set[str] = frozenset()) -> list[tuple[str, float, float]]:
    """Places mentioned in text, in order of appearance, deduplicated: [(name, lat, lon)]."""
    return [(h.name, h.lat, h.lon) for h in find_place_hits(text, not_places)]
