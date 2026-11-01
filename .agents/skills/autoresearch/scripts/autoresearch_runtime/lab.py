"""Split runtime facade. Behavior lives in the sibling modules."""
from __future__ import annotations

from .errors import LabError
from .store import _write_json
from .config import load_config
from .snapshot import init_lab, setup_project
from .state import _refresh_dashboard, load_state
from .evaluator import _remaining, _spent
from .baseline import run_baseline
from .trial import assess_candidate, recover_lab, run_trial, status_lab
from .vault import restore_vault
from .replay import replay_plan, run_replay
from .promote import export_best, promote_in_place, promote_to_branch
from .cli import main


if __name__ == '__main__':
    raise SystemExit(main())
