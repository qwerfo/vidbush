"""FastAPI application: JSON API under /api, the built React UI at /."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..config import Settings
from .jobs_handler import handle_job
from .media import safe_join

MISSING_UI = """<!doctype html><html><head><meta charset="utf-8"><title>Vidbush</title>
<style>body{background:#111418;color:#e6e8eb;font:16px/1.6 system-ui,sans-serif;max-width:640px;margin:60px auto;
padding:0 16px}code{background:#1d2228;padding:2px 6px;border-radius:4px}</style></head><body>
<h1>Vidbush API is running</h1><p>The web UI has not been built yet. Build it once with Node 18 or newer:</p>
<p><code>cd ui &amp;&amp; npm install &amp;&amp; npm run build</code></p>
<p>Then reload this page. The JSON API is available under <code>/api</code>,
for example <a href="/api/health" style="color:#7cc4ff">/api/health</a>.</p></body></html>"""


def get_settings(request: Request) -> Settings:
    """Dependency: the Settings object the app was created with."""
    return request.app.state.settings


def create_app(settings: Settings, start_workers: bool = True, workers: Optional[int] = None) -> FastAPI:
    """Build the app. start_workers=False skips the background WorkerPool (tests call the handler)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = None
        if start_workers:
            from ..jobs.runner import WorkerPool
            pool = WorkerPool(settings, handle_job, workers=workers or 2)
            pool.start()
        app.state.pool = pool
        yield
        if pool:
            pool.stop()

    app = FastAPI(title="Vidbush", version="1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.pool = None
    from . import routes_batches, routes_misc, routes_projects, routes_shots
    app.include_router(routes_batches.router, prefix="/api")
    app.include_router(routes_misc.router, prefix="/api")
    app.include_router(routes_projects.router, prefix="/api")
    app.include_router(routes_shots.router, prefix="/api")
    _mount_ui(app, Path(settings.repo_root) / "ui" / "dist")
    return app


def _mount_ui(app: FastAPI, dist: Path) -> None:
    """Serve ui/dist with SPA fallback to index.html; a help page when the UI is not built."""

    @app.get("/api/{rest:path}", include_in_schema=False)
    def api_not_found(rest: str) -> JSONResponse:
        return JSONResponse({"detail": f"no such endpoint: /api/{rest}"}, status_code=404)

    @app.get("/{full_path:path}", include_in_schema=False)
    def ui(full_path: str):
        index = dist / "index.html"
        if not index.exists():
            return HTMLResponse(MISSING_UI)
        if full_path:
            try:
                path = safe_join(dist, full_path)
            except HTTPException:
                path = None
            if path:
                cache = "public, max-age=31536000, immutable" if full_path.startswith("assets/") else "no-cache"
                return FileResponse(path, headers={"Cache-Control": cache})
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
