"""Spoken numbers in narration words, as the digits a label or anchor uses: narration timings spell numbers
out ("nineteen fifty-nine", "seven point six") while labels write them ("1959", "7.6")."""
from __future__ import annotations

import re

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
_TEENS = {w: i + 10 for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_DIGITS_RE = re.compile(r"\$?(\d[\d,]*(?:\.\d+)?)%?")
_EDGE = "\"'()[]{}“”‘’"


def normalise_number(text: str) -> str:
    """Digit form of the first number in a label: '1,959' -> '1959', '7.6' -> '7.6', '$77' -> '77'; '' when
    there is none."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return m.group(0).replace(",", "") if m else ""


def _token(raw: str) -> tuple[list[str], str, bool]:
    """(word parts split on hyphens, the bare token, ends a sentence)."""
    t = raw.strip().strip(_EDGE)
    end = bool(re.search(r"[.!?]$", t))
    bare = re.sub(r"['’]s$", "", t.rstrip(".,;:!?").strip(_EDGE)).lower()
    return [p for p in bare.split("-") if p], bare, end


def _pair(toks: list, i: int) -> tuple[int, int] | None:
    """Second half of a spoken year at i: (value 0-99, tokens used). 'oh five', 'fifty-nine', 'fifty nine',
    'eleven', 'hundred'."""
    parts = toks[i][0]
    if parts == ["oh"] and i + 1 < len(toks) and not toks[i][2] and len(toks[i + 1][0]) == 1 \
            and toks[i + 1][0][0] in _UNITS and toks[i + 1][0][0] != "zero":
        return _UNITS[toks[i + 1][0][0]], 2
    if parts == ["hundred"]:
        return 0, 1
    if len(parts) == 1 and parts[0] in _TEENS:
        return _TEENS[parts[0]], 1
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _UNITS and parts[1] != "zero":
        return _TENS[parts[0]] + _UNITS[parts[1]], 1
    if len(parts) == 1 and parts[0] in _TENS:
        nxt = toks[i + 1][0] if i + 1 < len(toks) and not toks[i][2] else []
        if len(nxt) == 1 and nxt[0] in _UNITS and nxt[0] != "zero":
            return _TENS[parts[0]] + _UNITS[nxt[0]], 2
        return _TENS[parts[0]], 1
    return None


def _year(toks: list, i: int) -> tuple[int, str] | None:
    """A year said as two pairs from i: (last index, digits). The first pair is 10-19 or twenty."""
    parts, _bare, end = toks[i]
    if end or len(parts) != 1 or not (parts[0] in _TEENS or parts[0] == "twenty") or i + 1 >= len(toks):
        return None
    second = _pair(toks, i + 1)
    if second is None:
        return None
    last = i + second[1]
    if last + 1 < len(toks) and not toks[last][2] and toks[last + 1][0][:1] and \
            toks[last + 1][0][0] in _SCALES | {"hundred": 0}:
        return None  # 'nineteen fifty thousand' is a count, not a year
    head = _TEENS.get(parts[0], 20)
    return last, str(head * 100 + second[0])


def _cardinal(toks: list, i: int) -> tuple[int, str] | None:
    """A spelled out count (with hundred, thousand, million, billion, 'and', 'point') from i: (last index,
    digits)."""
    total = current = 0
    last_kind, last_scale, last_i = None, 10 ** 12, None
    j = i
    while j < len(toks):
        parts, _bare, end = toks[j]
        if parts == ["and"] and last_kind in ("h", "s") and not end and j + 1 < len(toks) \
                and toks[j + 1][0][:1] and toks[j + 1][0][0] in _UNITS | _TEENS | _TENS:
            j += 1
            continue
        ok = bool(parts)
        for p in parts:
            if p in _UNITS and last_kind in (None, "t", "h", "s"):
                current += _UNITS[p]
                kind = "u"
            elif p in _TEENS and last_kind in (None, "h", "s"):
                current += _TEENS[p]
                kind = "u"
            elif p in _TENS and last_kind in (None, "h", "s"):
                current += _TENS[p]
                kind = "t"
            elif p == "hundred" and last_kind in ("u", "t") and current % 100 == current:
                current *= 100
                kind = "h"
            elif p in _SCALES and last_kind in ("u", "t", "h") and _SCALES[p] < last_scale:
                total += current * _SCALES[p]
                current, last_scale = 0, _SCALES[p]
                kind = "s"
            else:
                ok = False
                break
            last_kind = kind
        if not ok:
            break
        last_i = j
        if end:
            break
        j += 1
    if last_i is None:
        return None
    value = str(total + current)
    k = last_i + 1
    if k + 1 < len(toks) and not toks[last_i][2] and toks[k][0] == ["point"] and not toks[k][2]:
        frac = ""
        for m in range(k + 1, len(toks)):
            parts = toks[m][0]
            if len(parts) != 1 or parts[0] not in _UNITS | {"oh": 0}:
                break
            frac += str(_UNITS.get(parts[0], 0))
            last_i = m
            if toks[m][2]:
                break
        if frac:
            value += "." + frac
    return last_i, value


def spoken_numbers(tokens: list[str]) -> list[tuple[int, int, str]]:
    """Runs of number words (and digit tokens) in narration tokens: (first index, last index inclusive, digits).
    Years read as two pairs come first ('nineteen fifty-nine' is 1959, 'twenty eleven' is 2011), then counts
    ('seven hundred thirty-three thousand, three hundred ninety-one' is 733391, 'seven point six' is 7.6).
    A run never crosses the end of a sentence. Every number is returned, 'one' too: callers decide."""
    toks = [_token(t) for t in tokens]
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(toks):
        m = _DIGITS_RE.fullmatch(toks[i][1])
        hit = (i, m.group(1).replace(",", "")) if m else _year(toks, i) or _cardinal(toks, i)
        if hit:
            out.append((i, hit[0], hit[1]))
            i = hit[0] + 1
        else:
            i += 1
    return out
