"""Web search adapters (Tavily, Serper, Brave). Any one key is enough."""
from __future__ import annotations

from typing import Any

from .base import CallResult, ProviderSpec, TransportError
from .http import get_json, post_json


def search(spec: ProviderSpec, query: str, n: int = 8) -> CallResult:
    """Return value=[{url, title, snippet}]."""
    kind = spec.type
    if kind == "tavily":
        data = post_json("https://api.tavily.com/search",
                         {"api_key": spec.api_key, "query": query, "max_results": n}, timeout=30)
        items = [{"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")}
                 for r in data.get("results", [])]
    elif kind == "serper":
        data = post_json("https://google.serper.dev/search", {"q": query, "num": n},
                         {"X-API-KEY": spec.api_key}, timeout=30)
        items = [{"url": r.get("link", ""), "title": r.get("title", ""), "snippet": r.get("snippet", "")}
                 for r in data.get("organic", [])]
    elif kind == "brave":
        data = get_json("https://api.search.brave.com/res/v1/web/search", {"q": query, "count": n},
                        {"X-Subscription-Token": spec.api_key, "Accept": "application/json"})
        items = [{"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("description", "")}
                 for r in (data.get("web") or {}).get("results", [])]
    else:
        raise TransportError(f"unknown search provider {kind}")
    return CallResult(value=items[:n], provider=spec, mock=False, cost_usd=0.005, units=1, unit_kind="queries")


def wikipedia_search(query: str, n: int = 5, lang: str = "en") -> list[dict[str, Any]]:
    """Wikipedia API search, no key needed."""
    data = get_json(f"https://{lang}.wikipedia.org/w/api.php",
                    {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": n},
                    headers={"User-Agent": "Vidbush/0.1 (self-hosted video tool)"})
    out = []
    for r in data.get("query", {}).get("search", []):
        title = r.get("title", "")
        out.append({"url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}", "title": title,
                    "snippet": r.get("snippet", "")})
    return out


def wikipedia_extract(title: str, lang: str = "en") -> str:
    data = get_json(f"https://{lang}.wikipedia.org/w/api.php",
                    {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json"},
                    headers={"User-Agent": "Vidbush/0.1 (self-hosted video tool)"})
    pages = data.get("query", {}).get("pages", {})
    return next(iter(pages.values()), {}).get("extract", "") if pages else ""
