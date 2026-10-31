"""Split runtime facade. Behavior lives in the sibling modules."""
from __future__ import annotations

import webbrowser

from .dash_io import _merged_usage, _read_json
from .dash_model import _historical_rechecks
from .dash_model import _STATUS_CACHE, _STATUS_CACHE_LOCK, _STATUS_NAMES, _STATUS_SKIP, _compute_dashboard_status, _lab_replay_rechecks, _lab_summary, _status_signature, _usage_totals, dashboard_status
from .dash_api import _directory_running, _is_deletable_lab, _open_lab, _path_segment, _safe_archive_dir, _safe_diff, _safe_lab_dir, archive_lab, dashboard_details, delete_lab, render_live_page, restore_lab
from .dash_http import DashboardHTTPServer, DashboardHandler, create_server, notify_dashboard
from .dash_process import _SERVER_PROCESSES, _START_LOCKS, _START_LOCKS_GUARD, _dashboard_start_lock, _server_info, _start_dashboard_locked, _write_server_info, dashboard_server_status, ensure_dashboard_server, main, serve, start_dashboard, stop_dashboard



if __name__ == '__main__':
    raise SystemExit(main())
