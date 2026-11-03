"""Health, settings and seat status, brands and style profiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import ValidationError

from ..config import Settings, list_ids, load_brand, load_style_profile, save_yaml
from ..models import Brand
from .app import get_settings
from .media import check_id

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    pool = getattr(request.app.state, "pool", None)
    return {"ok": True, "workers": bool(pool), "offline": get_settings(request).offline}


@router.get("/settings")
def settings_status(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Seat resolution (real or mock per role), sources and render config, doctor checks."""
    from ..doctor import doctor
    from ..providers.base import CostTracker
    from ..providers.seats import Seats
    seats = Seats(settings, CostTracker("api")).status()
    return {
        "offline": settings.offline,
        "seats": seats,
        "mock_roles": sorted(r for r, v in seats.items() if not v["real"]),
        "sources": settings.sources.model_dump(mode="json"),
        "render": settings.render.model_dump(mode="json"),
        "judge": settings.judge.model_dump(mode="json"),
        "music_folder": settings.music_folder,
        "output_folder": settings.output_folder,
        "doctor": doctor(settings),
    }


def _brand_file(settings: Settings, brand_id: str) -> Path:
    return Path(settings.config_folder) / "brands" / f"{check_id(brand_id, 'brand id')}.yaml"


@router.get("/brands")
def brands(settings: Settings = Depends(get_settings)) -> list[dict[str, Any]]:
    out = []
    for bid in list_ids(settings, "brands"):
        b = load_brand(settings, bid)
        out.append({"id": bid, "name": b.name or bid, "style_profile_id": b.style_profile_id})
    return out


@router.get("/brands/{brand_id}")
def get_brand(brand_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    if not _brand_file(settings, brand_id).exists():
        raise HTTPException(404, f"brand not found: {brand_id}")
    return load_brand(settings, brand_id).model_dump(mode="json")


@router.put("/brands/{brand_id}")
def put_brand(brand_id: str, body: dict[str, Any] = Body(...),
              settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Validate with models.Brand and write config/brands/<id>.yaml. The URL id wins over the body."""
    path = _brand_file(settings, brand_id)
    try:
        brand = Brand.model_validate({**body, "id": brand_id})
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False, include_context=False)) from e
    if brand.style_profile_id not in list_ids(settings, "style_profiles"):
        raise HTTPException(422, f"unknown style profile: {brand.style_profile_id}")
    save_yaml(path, brand.model_dump(mode="json"))
    return brand.model_dump(mode="json")


@router.get("/style_profiles")
def style_profiles(settings: Settings = Depends(get_settings)) -> list[dict[str, Any]]:
    out = []
    for pid in list_ids(settings, "style_profiles"):
        prof = load_style_profile(settings, pid)
        out.append({"id": pid, "name": prof.name or pid, "palette": prof.palette, "top10": prof.top10,
                    "visual_mix": prof.visual_mix, "learned": bool(prof.provenance)})
    return out
