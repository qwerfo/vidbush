"""Fact check pass: mark each card verified, unverified or contradicted; drop contradicted.

The researcher seat (task "factcheck") returns {"verdicts": [{"id", "status"}]}.
Without a usable answer, a card is verified when its years and numbers all
appear in the fetched excerpt of its own source, otherwise it keeps its status.
A real seat that fails on every provider raises ProviderError (stage fails,
resume retries), per the mock fallback policy.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..models import FactCard, Research
from ..providers.base import ProviderError
from ..script import prompts
from ..util import JsonLogger, NullLogger

STATUSES = {"verified", "unverified", "contradicted"}


def fact_check(research: Research, seats: Any, log: Optional[JsonLogger] = None,
               excerpts: Optional[dict[str, str]] = None) -> Research:
    """Return a copy of research with statuses set and contradicted cards removed.

    excerpts maps source url to the fetched page text (used by the prompt and the heuristic).
    """
    log = log or NullLogger()
    excerpts = excerpts or {}
    verdicts = _ask(research, seats, excerpts)
    out = research.model_copy(deep=True)
    for card in out.facts:
        status = verdicts.get(card.id)
        if status is None and not verdicts:
            status = heuristic_status(card, excerpts.get(card.source_url, ""))
        if status:
            card.status = status  # type: ignore[assignment]
    dropped = [c.id for c in out.facts if c.status == "contradicted"]
    out.facts = [c for c in out.facts if c.status != "contradicted"]
    for a in out.anchors:
        a.fact_ids = [f for f in a.fact_ids if f not in dropped]
    counts = {s: sum(1 for c in out.facts if c.status == s) for s in ("verified", "unverified")}
    log.info("fact check done", verified=counts["verified"], unverified=counts["unverified"],
             dropped=len(dropped), dropped_ids=dropped, used_seat=bool(verdicts))
    return out


def _ask(research: Research, seats: Any, excerpts: dict[str, str]) -> dict[str, str]:
    """Seat verdicts as {fact_id: status}; empty when the seat gives nothing usable."""
    cards = "\n".join(f"{c.id}: {c.claim} (source: {c.source_title or c.source_url})" for c in research.facts)
    text = "\n\n".join(f"{url}\n{body[:4000]}" for url, body in list(excerpts.items())[:6])
    user = prompts.FACTCHECK_USER.format(cards=cards or "none", excerpts=text or "none provided")
    try:
        data = seats.llm_json("researcher", "factcheck", prompts.FACTCHECK_SYSTEM, user, stage="research",
                              temperature=0.1)
    except ProviderError:
        raise
    except Exception:  # a malformed answer keeps the cards as they are
        return {}
    rows = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}
    out = {}
    for r in rows:
        if isinstance(r, dict) and str(r.get("status", "")).lower() in STATUSES and r.get("id"):
            out[str(r["id"])] = str(r["status"]).lower()
    return out


def heuristic_status(card: FactCard, source_text: str) -> Optional[str]:
    """Verified when every year and number in the claim appears in its source text; else None."""
    if not source_text:
        return None
    tokens = re.findall(r"\d[\d,]*(?:\.\d+)?", card.claim)
    if not tokens:
        return None
    flat = source_text.replace(",", "")
    if all(t.replace(",", "") in flat for t in tokens):
        return "verified"
    return None
