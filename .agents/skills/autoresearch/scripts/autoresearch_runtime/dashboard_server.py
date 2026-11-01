"""Split runtime facade. Behavior lives in the sibling modules."""
from __future__ import annotations

from .dash_http import DashboardHTTPServer, create_server
from .dash_process import (
    _server_info,
    dashboard_server_status,
    main,
    start_dashboard,
    stop_dashboard,
)



if __name__ == '__main__':
    raise SystemExit(main())
