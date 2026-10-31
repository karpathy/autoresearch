"""Loopback-only live dashboard. Mutations are delete, archive, and restore of one chosen lab."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen
from typing import Any

from .catalog import _human_goal, lab_lifecycle, lab_status, refresh_catalog, resolve_lab_path, save_catalog
from .dashboard import (
    ASSET_TYPES,
    STATUS_LABELS,
    assign_short_labels,
    build_problems,
    build_verdict,
    chart_series,
    metric_as_percent,
    metric_label,
    presentation_status,
    read_asset,
    reason_label,
    render_unified_dashboard,
)




def _read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback

def _merged_usage(run: dict[str, Any], usage_path: Path) -> dict[str, Any]:
    result_usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
    file_usage = _read_json(usage_path, {}) or {}
    merged = dict(file_usage) if isinstance(file_usage, dict) else {}
    for key, value in result_usage.items():
        if (key in {"requests", "cost_usd", "confirmed_cost_usd", "unknown_requests", "unknown_cost_usd"}
                and isinstance(value, (int, float)) and not isinstance(value, bool)):
            existing = merged.get(key)
            merged[key] = max(value, existing) if isinstance(existing, (int, float)) and not isinstance(existing, bool) else value
        else:
            merged[key] = value
    return merged
