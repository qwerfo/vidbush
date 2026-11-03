"""Image-to-video adapter (fal.ai queue style). Optional seat, off by default."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from .base import CallResult, ProviderSpec, RefusalError
from .http import get_bytes, post_json


def generate_video(spec: ProviderSpec, prompt: str, still: Optional[Path], out_path: Path, seconds: float,
                   per_second_usd: float) -> CallResult:
    """Animate a still into a short clip. Returns value=out_path."""
    base = (spec.base_url or "https://fal.run").rstrip("/")
    body: dict = {"prompt": prompt, "duration": str(int(round(min(max(seconds, 3), 10))))}
    if still is not None and still.exists():
        body["image_url"] = "data:image/png;base64," + base64.b64encode(still.read_bytes()).decode()
    headers = {"Authorization": f"Key {spec.api_key}"} if spec.type == "fal" else {"Authorization": f"Bearer {spec.api_key}"}
    data = post_json(f"{base}/{spec.model}", body, headers, timeout=600)
    url = (data.get("video") or {}).get("url") or data.get("url")
    if not url:
        raise RefusalError(f"video provider returned no clip: {str(data)[:200]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(get_bytes(url))
    return CallResult(value=out_path, provider=spec, mock=False, cost_usd=seconds * per_second_usd,
                      units=seconds, unit_kind="seconds")
