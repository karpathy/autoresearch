from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from component_system.services.workflow import default_workflow_service
from component_system.task import ensure_queue_layout
from component_system.web.routes import router

WEB_ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = WEB_ROOT / "templates"
STATIC_ROOT = WEB_ROOT / "static"


def _static_version() -> str:
    """Cache-busting version from app.js mtime so browsers load fresh static assets after changes."""
    app_js = STATIC_ROOT / "app.js"
    if app_js.exists():
        return str(int(app_js.stat().st_mtime))
    return str(int(time.time()))


def create_app() -> FastAPI:
    ensure_queue_layout()
    app = FastAPI(title="Component System", version="0.1.0")
    app.state.workflow = default_workflow_service()
    app.state.static_version = _static_version()
    app.state.templates = Jinja2Templates(directory=str(TEMPLATE_ROOT))
    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
    app.include_router(router, prefix="/component-system")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/component-system", status_code=307)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
    def chrome_devtools_probe() -> Response:
        # Chrome DevTools probes this endpoint; return 204 to avoid log spam.
        return Response(status_code=204)

    return app


app = create_app()
