"""Licence normalisation and NC/ND detection.

normalise_licence(raw) turns the many spellings sources use (Commons short
names, Creative Commons URLs, LoC rights statements, Archive licenseurl)
into one readable string. Nothing is ever blocked because of its licence:
CC NC and ND assets only get the "nc_nd" flag when settings ask for it.
"""
from __future__ import annotations

import re

from ..config import Settings

PUBLIC_DOMAIN = "Public domain"
CC0 = "CC0 1.0"
UNKNOWN = "unknown"

_CC_URL = re.compile(r"creativecommons\.org/licenses/([a-z\-]+)/?([0-9.]+)?", re.I)
_CC_TEXT = re.compile(r"\bcc[\s\-_]*((?:by)(?:[\s\-_]*(?:sa|nc|nd))*)(?:[\s\-_]*([0-9](?:\.[0-9])?))?", re.I)
_ANNOTATED = re.compile(r"^\s*(cc0|cc by[a-z\- ]*|public domain)( [0-9.]+)? \([^)]*\)\s*$")
_PD_WORDS = ("public domain", "publicdomain", "pd-", "no known copyright", "no known restrictions",
             "not in copyright", "no copyright", "pdm")


def _cc_name(parts: str, version: str | None) -> str:
    tokens = [t for t in re.split(r"[\s\-_]+", parts.lower()) if t]
    name = "CC " + "-".join(t.upper() for t in tokens)
    return f"{name} {version}" if version else name


def normalise_licence(raw: str) -> str:
    """Canonical licence string, e.g. 'CC BY-SA 4.0', 'CC0 1.0', 'Public domain'."""
    text = (raw or "").strip()
    if not text:
        return UNKNOWN
    low = text.lower()
    if _ANNOTATED.match(low):
        return re.sub(r"\s+", " ", text)  # already canonical with a note, e.g. "CC0 (Vidbush generated fixture)"
    if "publicdomain/zero" in low or re.search(r"\bcc[\s\-_]*0\b|\bcc[\s\-_]*zero\b", low):
        return CC0
    m = _CC_URL.search(text)
    if m:
        return _cc_name(m.group(1), m.group(2))
    m = _CC_TEXT.search(text)
    if m:
        return _cc_name(m.group(1), m.group(2))
    if "creative commons attribution" in low:
        return _creative_commons_words(low)
    if any(w in low for w in _PD_WORDS) or low in {"pd", "cc-pd"}:
        return f"{PUBLIC_DOMAIN} ({_pd_owner(low)})" if _pd_owner(low) else PUBLIC_DOMAIN
    return re.sub(r"\s+", " ", text)[:120]


def _creative_commons_words(low: str) -> str:
    """'Creative Commons Attribution-NonCommercial 4.0' style long names."""
    parts = ["BY"]
    if "sharealike" in low or "share alike" in low:
        parts.append("SA")
    if "noncommercial" in low or "non-commercial" in low:
        parts.insert(1, "NC")
    if "noderiv" in low or "no deriv" in low:
        parts.append("ND")
    ver = re.search(r"([0-9]\.[0-9])", low)
    if len(parts) == 1 and not ver:
        return "Creative Commons Attribution"  # YouTube's wording, kept as reported
    name = "CC " + "-".join(parts)
    return f"{name} {ver.group(1)}" if ver else name


def _pd_owner(low: str) -> str:
    if "nasa" in low:
        return "NASA"
    if "library of congress" in low:
        return "Library of Congress"
    return ""


def is_nc_nd(licence: str) -> bool:
    """True for Creative Commons NonCommercial or NoDerivatives licences."""
    low = (licence or "").lower()
    if re.search(r"\bcc\b[^,;]*\b(nc|nd)\b", low.replace("-", " ")):
        return True
    return "noncommercial" in low or "non-commercial" in low or "noderiv" in low or "no deriv" in low


def is_public_domain(licence: str) -> bool:
    low = (licence or "").lower()
    return low.startswith("public domain") or low.startswith("cc0")


def licence_flags(licence: str, settings: Settings) -> list[str]:
    """Flags to attach to a candidate for this licence ('nc_nd' when enabled)."""
    if settings.sources.flag_nc_nd and is_nc_nd(licence):
        return ["nc_nd"]
    return []
