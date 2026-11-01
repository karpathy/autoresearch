"""The read-only dashboard page and its assets."""

from __future__ import annotations

import json
from pathlib import Path


ASSETS_DIR = Path(__file__).resolve().parent / "dashboard_assets"
ASSET_TYPES = {
    "styles.css": "text/css; charset=utf-8",
    "live_app.js": "text/javascript; charset=utf-8",
}


def read_asset(name: str) -> bytes | None:
    if name not in ASSET_TYPES or any(part in name for part in ("/", "\\", "..")):
        return None
    path = (ASSETS_DIR / name).resolve()
    root = ASSETS_DIR.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path.read_bytes()


def render_unified_dashboard(*, root_id: str) -> str:
    """Fill the one dashboard page. The page reads its root id from the payload; all lab data comes from /api/v1."""
    payload = json.dumps({"root_id": root_id}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    page = (ASSETS_DIR / "index.html").read_text(encoding="utf-8")
    return page.replace("__PAYLOAD__", payload)
