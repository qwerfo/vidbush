"""Batch report: data/batches/<id>/batch_report.md and batch_report.json, one row per project.

Figures per project come from its outputs: duration from final.mp4 (ffprobe), else shotlist.json;
total cost from cost_report.json; QA percent is pct_above_threshold from qa_report.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..util import media_duration, read_json, write_json
from .batch import BatchItem, BatchState, batch_dir

REPORT_MD = "batch_report.md"
REPORT_JSON = "batch_report.json"
COLUMNS = ("project", "input", "mode", "status", "duration", "cost", "cost/min", "QA %", "note")


def _json_value(path: Path, key: str) -> Optional[float]:
    try:
        value = read_json(path).get(key) if path.exists() else None
        return None if value is None else float(value)
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def project_duration(root: Path) -> Optional[float]:
    """Seconds of final.mp4, or the shot list duration when there is no final video yet."""
    final = root / "final.mp4"
    if final.exists():
        try:
            return round(media_duration(final), 2)
        except Exception:  # noqa: BLE001 - unreadable video: fall back to the shot list
            pass
    return _json_value(root / "shotlist.json", "duration") or _json_value(root / "work" / "shotlist.json",
                                                                          "duration")


def project_metrics(settings: Settings, project_id: str) -> dict[str, Optional[float]]:
    """duration_s, cost_usd, cost_per_min_usd and qa_pct of one project (None where unknown)."""
    from .pipeline import project_dir
    root = project_dir(settings, project_id)
    duration = project_duration(root)
    cost = _json_value(root / "cost_report.json", "total_usd")
    per_min = round(cost / (duration / 60.0), 4) if cost is not None and duration else None
    return {"duration_s": duration, "cost_usd": cost, "cost_per_min_usd": per_min,
            "qa_pct": _json_value(root / "qa_report.json", "pct_above_threshold")}


def summary(state: BatchState) -> dict[str, Any]:
    """Counts per status and totals over the finished items."""
    counts: dict[str, int] = {}
    for item in state.items:
        counts[item.status] = counts.get(item.status, 0) + 1
    cost = sum(i.cost_usd or 0.0 for i in state.items)
    minutes = sum((i.duration_s or 0.0) for i in state.items) / 60.0
    return {"counts": counts, "total_cost_usd": round(cost, 4), "total_minutes": round(minutes, 2),
            "cost_per_min_usd": round(cost / minutes, 4) if minutes else None}


def report_json(state: BatchState) -> dict[str, Any]:
    """The machine readable report: batch settings, summary and one row per item."""
    return {"batch_id": state.id, "source": state.source, "brand": state.brand, "status": state.status,
            "paused": state.paused, "cost_cap_usd": state.cost_cap_usd, "updated_at": state.updated_at,
            "summary": summary(state), "rows": [row(i) for i in state.items]}


def row(item: BatchItem) -> dict[str, Any]:
    return {"project_id": item.project_id, "input": item.input, "mode": item.mode, "status": item.status,
            "duration_s": item.duration_s, "cost_usd": item.cost_usd, "cost_per_min_usd": item.cost_per_min_usd,
            "qa_pct": item.qa_pct, "note": item.note}


def _fmt(value: Optional[float], kind: str) -> str:
    if value is None:
        return "-"
    if kind == "duration":
        return f"{int(value // 60)}:{int(value % 60):02d}"
    if kind == "usd":
        return f"${value:.2f}"
    return f"{value:g}%"


def report_markdown(state: BatchState) -> str:
    """batch_report.md text: header lines, then a table with one row per project."""
    s = summary(state)
    cap = f"${state.cost_cap_usd:.2f}" if state.cost_cap_usd else "none"
    counts = ", ".join(f"{k} {v}" for k, v in sorted(s["counts"].items())) or "empty"
    lines = [f"# Batch {state.id}", "", f"- Source: `{state.source}`", f"- Brand: {state.brand}",
             f"- Status: {state.status}{' (paused)' if state.paused and state.status != 'paused' else ''}",
             f"- Cost cap per project run: {cap}", f"- Items: {counts}",
             f"- Total: {_fmt(s['total_cost_usd'], 'usd')} for {s['total_minutes']} min"
             f" ({_fmt(s['cost_per_min_usd'], 'usd')} per minute)", "",
             "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for i in state.items:
        note = (i.note or "").replace("|", "/").replace("\n", " ")
        cells = [i.project_id, Path(i.input).name, i.mode or "-", i.status, _fmt(i.duration_s, "duration"),
                 _fmt(i.cost_usd, "usd"), _fmt(i.cost_per_min_usd, "usd"), _fmt(i.qa_pct, "pct"), note or ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(settings: Settings, state: BatchState) -> Path:
    """Write batch_report.md and batch_report.json into the batch folder. Returns the .md path."""
    folder = batch_dir(settings, state.id)
    write_json(folder / REPORT_JSON, report_json(state))
    path = folder / REPORT_MD
    path.write_text(report_markdown(state), encoding="utf-8")
    return path
