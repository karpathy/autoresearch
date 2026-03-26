from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pdca_system.services.workflow import default_workflow_service
from pdca_system.task import ensure_queue_layout
from pdca_system.web.routes import router

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
    app = FastAPI(title="PDCA System", version="0.1.0")
    app.state.workflow = default_workflow_service()
    app.state.static_version = _static_version()
    app.state.templates = Jinja2Templates(directory=str(TEMPLATE_ROOT))

    def _format_ts(ts: float | None) -> str:
        if ts is None:
            return ""
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except (TypeError, OSError):
            return ""

    def _format_ts_timeline(ts: float | None) -> str:
        """Format timestamp like Timeline (local time, %Y-%m-%d %H:%M:%S)."""
        if ts is None:
            return ""
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, OSError):
            return ""

    app.state.templates.env.filters["format_ts"] = _format_ts
    app.state.templates.env.filters["format_ts_timeline"] = _format_ts_timeline
    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
    app.include_router(router, prefix="/pdca-system")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/pdca-system", status_code=307)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
    def chrome_devtools_probe() -> Response:
        # Chrome DevTools probes this endpoint; return 204 to avoid log spam.
        return Response(status_code=204)

    return app


app = create_app()
