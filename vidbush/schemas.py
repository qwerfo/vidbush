"""Export JSON schemas for every contract: python -m vidbush.schemas"""
from __future__ import annotations

import json
from pathlib import Path

from .models import ALL_MODELS


def export(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in ALL_MODELS.items():
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(model.model_json_schema(), indent=2), encoding="utf-8")
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in export(Path(__file__).resolve().parents[1] / "docs" / "schemas"):
        print(p)
