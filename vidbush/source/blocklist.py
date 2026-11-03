"""Domain blocklist: settings.sources.blocklist plus the brand blocklist.

A domain blocks itself and all its subdomains. The page URL, the download
URL and the candidate URL are all checked. This is the only filter that
removes results outright.
"""
from __future__ import annotations

from typing import Iterable, Optional
from urllib.parse import urlparse


def normalise_domain(entry: str) -> str:
    """'https://www.Example.com/path' -> 'example.com'."""
    text = (entry or "").strip().lower()
    if not text:
        return ""
    host = urlparse(text if "://" in text else f"//{text}").hostname or ""
    host = host.strip(".")
    return host[4:] if host.startswith("www.") else host


def host_of(url: str) -> str:
    if not url or "://" not in url:
        return ""
    return (urlparse(url).hostname or "").lower().strip(".")


class Blocklist:
    """Set of blocked domains with subdomain matching."""

    def __init__(self, domains: Iterable[str]) -> None:
        self.domains = sorted({d for d in (normalise_domain(x) for x in domains) if d})

    @classmethod
    def from_settings(cls, settings_blocklist: Iterable[str], brand_blocklist: Optional[Iterable[str]] = None) -> "Blocklist":
        return cls(list(settings_blocklist or []) + list(brand_blocklist or []))

    def host_blocked(self, host: str) -> bool:
        host = (host or "").lower().strip(".")
        return bool(host) and any(host == d or host.endswith("." + d) for d in self.domains)

    def blocked(self, *urls: str) -> bool:
        """True when any of the URLs is on a blocked domain."""
        return any(self.host_blocked(host_of(u)) for u in urls if u)

    def __bool__(self) -> bool:
        return bool(self.domains)
